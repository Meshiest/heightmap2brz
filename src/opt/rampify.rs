//! Wrapperup's rampifier, over a heightmap's column field.
//!
//! The slope-selection algorithm is Wrapperup/rampifier's, by way of the
//! BRDB-native port in `obj2brz` (`crates/obj2brz/src/rampify.rs`): scan the
//! surface, fit the longest ramp that meets the terrain rising in front of it,
//! prefer corner ramps at convex and concave turns, then fill whatever is left
//! with plain bricks. The fitting rules, the run/rise limits and the brick
//! geometry are all carried over unchanged.
//!
//! What is NOT carried over is the dense voxel grid. `obj2brz` rampifies an
//! arbitrary mesh, so it materialises one cell per voxel of the bounding box
//! and caps the model at 64M of them. A heightmap is a HEIGHT FIELD -- a column
//! is solid from the ground to its top and empty above -- so occupancy is a
//! comparison rather than a lookup, and a 4096x4096 map at 255 plates costs a
//! `Vec<i32>` per column instead of the four terabytes its bounding box would
//! ask for. Two consequences fall out of that and are relied on below:
//!
//! * **Only the top cell of a column can anchor a slope.** Both `fit_ramp` and
//!   `fit_corner` refuse a cell with anything above it, so the whole
//!   ascending-Z scan reduces to visiting each column once, in height order.
//! * **There is no enclosed air and no underside.** The mesh rampifier floods
//!   air to find the inside of a watertight model and fits upside-down ramps
//!   under overhangs; a height field has neither, so only the floor pass runs.
//!
//! One voxel is one pixel wide (`GenOptions::size` half extents) and one plate
//! tall (4 units). `--vertical` is read as units and rounded to a whole number
//! of plates, because a `PB_DefaultRamp`'s rise is quantized to plates.

use crate::map::*;
use crate::util::*;
use brdb::{
    Brick, BrickSize, BrickType, Color, Direction, Position, Rotation,
    assets::{
        bricks::{
            PB_DEFAULT_BRICK, PB_DEFAULT_RAMP, PB_DEFAULT_RAMP_CORNER,
            PB_DEFAULT_RAMP_INNER_CORNER, PB_DEFAULT_WEDGE,
        },
        materials::{GLOW, PLASTIC},
    },
};
use log::info;
use std::collections::HashMap;

/// Longest run a ramp may span, in cells. Wrapperup's value.
const RAMP_MAX_RUN: i32 = 4;
/// Tallest rise a ramp may span, in cells. Wrapperup's value.
const RAMP_MAX_RISE: i32 = 12;
/// Vertical units in one cell: a plate.
const CELL_UNITS: i32 = 4;
/// The largest half extent a procedural brick may carry.
const MAX_HALF_EXTENT: i32 = 250;

/// A cell coordinate. X and Y index the pixel grid; Z counts plates up from the
/// ground.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Cell(i32, i32, i32);

impl std::ops::Add for Cell {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl std::ops::Sub for Cell {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

impl std::ops::Mul<i32> for Cell {
    type Output = Self;
    fn mul(self, rhs: i32) -> Self {
        Self(self.0 * rhs, self.1 * rhs, self.2 * rhs)
    }
}

impl Cell {
    /// Which way a ramp of this rotation runs UPHILL. Rotation zero steps down
    /// toward `+X`, so its uphill direction is `-X`; the rest are quarter turns
    /// counter-clockwise around `+Z`.
    fn forward(rotation: Rotation) -> Self {
        match rotation {
            Rotation::Deg0 => Self(-1, 0, 0),
            Rotation::Deg90 => Self(0, -1, 0),
            Rotation::Deg180 => Self(1, 0, 0),
            Rotation::Deg270 => Self(0, 1, 0),
        }
    }
}

fn next_rotation(rotation: Rotation) -> Rotation {
    match rotation {
        Rotation::Deg0 => Rotation::Deg90,
        Rotation::Deg90 => Rotation::Deg180,
        Rotation::Deg180 => Rotation::Deg270,
        Rotation::Deg270 => Rotation::Deg0,
    }
}

/// The height field the rampifier reads, plus the cells already claimed by a
/// slope.
struct Field {
    width: i32,
    height: i32,
    /// Solid cells in each column: the column occupies `0..cells`.
    cells: Vec<i32>,
    colors: Vec<[u8; 4]>,
    /// Inclusive Z ranges consumed by an emitted slope, per column. A column
    /// carries at most a handful, so a small `Vec` beats a per-cell bitmap by
    /// orders of magnitude on a real map.
    claimed: Vec<Vec<(i32, i32)>>,
}

impl Field {
    fn index(&self, x: i32, y: i32) -> Option<usize> {
        (x >= 0 && y >= 0 && x < self.width && y < self.height)
            .then(|| y as usize * self.width as usize + x as usize)
    }

    fn column(&self, cell: Cell) -> i32 {
        self.index(cell.0, cell.1).map_or(0, |i| self.cells[i])
    }

    fn exists(&self, cell: Cell) -> bool {
        cell.2 >= 0 && cell.2 < self.column(cell)
    }

    fn is_ramp(&self, cell: Cell) -> bool {
        self.index(cell.0, cell.1).is_some_and(|i| {
            self.claimed[i]
                .iter()
                .any(|(low, high)| cell.2 >= *low && cell.2 <= *high)
        })
    }

    /// How many cells of solid material sit at and above `cell`, capped at 31.
    /// `i32::MIN` when the cell itself is air, so an empty neighbour can never
    /// win the `max_by_key` in [`Self::best_rotation`].
    fn slope_height(&self, cell: Cell) -> i32 {
        if !self.exists(cell) {
            return i32::MIN;
        }
        (self.column(cell) - cell.2).min(31)
    }

    /// The direction whose terrain rises highest in front of `cell` while the
    /// cell BEHIND it is open, i.e. the way a ramp here should climb.
    fn best_rotation(&self, cell: Cell) -> Option<Rotation> {
        if self.exists(cell + Cell(0, 0, 1)) {
            return None;
        }
        [
            (Cell(-1, 0, 0), Cell(1, 0, 0), Rotation::Deg0),
            (Cell(1, 0, 0), Cell(-1, 0, 0), Rotation::Deg180),
            (Cell(0, -1, 0), Cell(0, 1, 0), Rotation::Deg90),
            (Cell(0, 1, 0), Cell(0, -1, 0), Rotation::Deg270),
        ]
        .into_iter()
        .filter(|(_, back, _)| !self.exists(cell + *back))
        .map(|(forward, _, rotation)| (self.slope_height(cell + forward), rotation))
        .max_by_key(|(height, _)| *height)
        .and_then(|(height, rotation)| (height > 0).then_some(rotation))
    }

    /// The run and rise of the straight ramp anchored at `cell` climbing
    /// `rotation`, or `None` where no ramp fits.
    fn fit_ramp(&self, cell: Cell, rotation: Rotation) -> Option<(i32, i32)> {
        let forward = Cell::forward(rotation);
        let up = Cell(0, 0, 1);

        let mut run = 0;
        for _ in 0..RAMP_MAX_RUN - 1 {
            if !self.exists(cell + up + forward * run)
                && self.exists(cell + forward * (run + 1))
                && !self.is_ramp(cell + forward * (run + 1))
            {
                run += 1;
            } else {
                break;
            }
        }
        if run == 0 {
            return None;
        }

        let mut rise = 0;
        for _ in 1..RAMP_MAX_RISE {
            let tip = cell + up * rise + forward * run;
            if self.exists(tip) && !self.is_ramp(tip) {
                rise += 1;
            } else {
                break;
            }
        }
        // A summit the ramp runs off the far side of gets one more cell of
        // rise, so a ridge is capped rather than left a plate short.
        let mut add_one = 0;
        for step in 1..RAMP_MAX_RUN {
            let beyond = cell + up * rise + forward * (run + step);
            if !self.exists(beyond) && !self.is_ramp(beyond) {
                add_one = 1;
            } else {
                add_one = 0;
                break;
            }
        }
        rise += add_one;
        (rise > 1).then_some((run + 1, rise - 1))
    }

    /// The runs along each wall axis and the rise of a corner ramp whose two
    /// high walls face `rotation` and `rotation + 90`, with `cell` as the low
    /// outer corner.
    ///
    /// An OUTER corner (a convex contour turn) has open cells behind both wall
    /// axes and rising terrain only past the far diagonal. An INNER corner (a
    /// concave turn) sits where the edge wraps the other way: the cells behind
    /// it are still edge, only the diagonal between them is open, and the
    /// terrain rises along the whole far row and column.
    fn fit_corner(&self, cell: Cell, rotation: Rotation, inner: bool) -> Option<(i32, i32, i32)> {
        let up = Cell(0, 0, 1);
        let forward_a = Cell::forward(rotation);
        let forward_b = Cell::forward(next_rotation(rotation));
        if self.exists(cell + up) {
            return None;
        }
        let corner_shaped = if inner {
            self.exists(cell - forward_a)
                && self.exists(cell - forward_b)
                && !self.exists(cell - forward_a - forward_b)
        } else {
            !self.exists(cell - forward_a) && !self.exists(cell - forward_b)
        };
        if !corner_shaped {
            return None;
        }
        let (run_a, rise_a) = self.fit_ramp(cell, rotation)?;
        let (run_b, rise_b) = self.fit_ramp(cell, next_rotation(rotation))?;
        // The two edge fits rarely agree exactly on rough terrain; the lower
        // rise still meets both neighbouring slopes.
        let rise = rise_a.min(rise_b);

        // An outer corner's surface is the INTERSECTION of the two straight
        // ramps, so it only reaches full height at the far diagonal cell; an
        // inner corner's is their UNION and is full height along the whole far
        // row and column. Those cells may hold the rising terrain; everywhere
        // else the footprint must be flat with air above it.
        let clear = |cells_a: i32, cells_b: i32| {
            for i in 0..cells_a {
                for j in 0..cells_b {
                    let footprint = cell + forward_a * i + forward_b * j;
                    if !self.exists(footprint) || self.is_ramp(footprint) {
                        return false;
                    }
                    let on_far_a = i == cells_a - 1;
                    let on_far_b = j == cells_b - 1;
                    let full_height = if inner {
                        on_far_a || on_far_b
                    } else {
                        on_far_a && on_far_b
                    };
                    if !full_height && self.exists(footprint + up) {
                        return false;
                    }
                }
            }
            true
        };
        // The full-run footprint is often obstructed on rough terrain, so fall
        // back to the largest clear rectangle that still makes a corner (2x2).
        let mut best: Option<(i32, i32)> = None;
        for cells_a in 2..=run_a {
            for cells_b in 2..=run_b {
                if best.is_none_or(|(a, b)| cells_a * cells_b > a * b) && clear(cells_a, cells_b) {
                    best = Some((cells_a, cells_b));
                }
            }
        }
        best.map(|(cells_a, cells_b)| (cells_a, cells_b, rise))
    }

    /// Claim every cell in a box and report the colour that dominates the solid
    /// ones. A slope covers several columns, and taking the anchor's colour
    /// instead would tint every ramp with the colour of its lowest pixel.
    fn claim(&mut self, origin: Cell, axes: [(Cell, i32); 2], rise: i32) -> Color {
        let mut counts = HashMap::<[u8; 4], usize>::new();
        for i in 0..axes[0].1 {
            for j in 0..axes[1].1 {
                let base = origin + axes[0].0 * i + axes[1].0 * j;
                let Some(index) = self.index(base.0, base.1) else {
                    continue;
                };
                let low = base.2;
                let high = base.2 + rise - 1;
                self.claimed[index].push((low, high));
                let solid = self.cells[index].min(high + 1) - low;
                if solid > 0 {
                    *counts.entry(self.colors[index]).or_default() += solid as usize;
                }
            }
        }
        let rgba = counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(color, _)| color)
            .unwrap_or([0, 0, 0, 255]);
        Color {
            r: rgba[0],
            g: rgba[1],
            b: rgba[2],
        }
    }
}

/// Where the rampifier's cell grid sits in world units, and how bricks are
/// styled. Split out so the geometry can be exercised without a `GenOptions`.
struct Layout {
    /// Half extent of one cell in X and Y, in units.
    half: i32,
    offset_x: i32,
    offset_y: i32,
    /// World Z of the bottom of cell layer zero.
    z_floor: i32,
    glow: bool,
    collision: brdb::Collision,
}

impl Layout {
    fn brick(
        &self,
        asset: brdb::BString,
        size: BrickSize,
        position: Position,
        color: Color,
        rotation: Rotation,
    ) -> Brick {
        Brick {
            asset: BrickType::Procedural { asset, size },
            position,
            collision: self.collision,
            color,
            owner_index: None,
            direction: Direction::ZPositive,
            rotation,
            material_intensity: if self.glow { 0 } else { 5 },
            material: if self.glow { GLOW } else { PLASTIC },
            ..Default::default()
        }
    }
}

/// Emit a straight ramp (or a one-cell wedge) anchored at `cell`.
///
/// The anchor is the ramp's LOW end and the piece extends `run` cells along the
/// rotation's forward axis, which for two of the four rotations runs toward
/// NEGATIVE X or Y. The per-rotation offsets below place the brick's centre
/// accordingly; they are the ported `create_ramp` offsets with `obj2brz`'s
/// hard-coded stud (5 units half, 10 full) generalised to `Layout::half`.
fn ramp_brick(
    layout: &Layout,
    cell: Cell,
    run: i32,
    rise: i32,
    rotation: Rotation,
    color: Color,
) -> Brick {
    let size = BrickSize::new(
        (run * layout.half) as u16,
        layout.half as u16,
        (rise * 2) as u16,
    );
    let full = layout.half * 2;
    let mut position = Position::new(
        cell.0 * full + layout.offset_x,
        cell.1 * full + layout.offset_y,
        layout.z_floor + cell.2 * CELL_UNITS,
    );
    match rotation {
        Rotation::Deg0 => {
            position.x += full - size.x as i32;
            position.y += size.y as i32;
        }
        Rotation::Deg90 => {
            position.x += size.y as i32;
            position.y += full - size.x as i32;
        }
        Rotation::Deg180 => {
            position.x += size.x as i32;
            position.y += size.y as i32;
        }
        Rotation::Deg270 => {
            position.x += size.y as i32;
            position.y += size.x as i32;
        }
    }
    position.z += size.z as i32;
    layout.brick(
        // A single-cell run has no ramp asset; the wedge is its one-cell form.
        if run < 2 {
            PB_DEFAULT_WEDGE
        } else {
            PB_DEFAULT_RAMP
        },
        size,
        position,
        color,
        rotation,
    )
}

/// Emit a corner ramp anchored at its low outer corner.
fn corner_brick(
    layout: &Layout,
    cell: Cell,
    (run_a, run_b, rise): (i32, i32, i32),
    inner: bool,
    rotation: Rotation,
    color: Color,
) -> Brick {
    let forward_a = Cell::forward(rotation);
    let forward_b = Cell::forward(next_rotation(rotation));
    let far = cell + forward_a * (run_a - 1) + forward_b * (run_b - 1);
    // Local X follows the first wall axis, so a quarter turn swaps which run
    // spans world X.
    let (x_cells, y_cells) = if forward_a.0 != 0 {
        (run_a, run_b)
    } else {
        (run_b, run_a)
    };
    let size = BrickSize::new(
        (run_a * layout.half) as u16,
        (run_b * layout.half) as u16,
        (rise * 2) as u16,
    );
    let full = layout.half * 2;
    let position = Position::new(
        cell.0.min(far.0) * full + x_cells * layout.half + layout.offset_x,
        cell.1.min(far.1) * full + y_cells * layout.half + layout.offset_y,
        layout.z_floor + cell.2 * CELL_UNITS + size.z as i32,
    );
    layout.brick(
        if inner {
            PB_DEFAULT_RAMP_INNER_CORNER
        } else {
            PB_DEFAULT_RAMP_CORNER
        },
        size,
        position,
        color,
        rotation,
    )
}

/// Generate a rampified heightmap: full-size ramps, wedges and ramp corners
/// fitted onto the column field's surface, with plain bricks under and beside
/// them.
pub fn gen_rampify_heightmap<F: Fn(f32) -> bool>(
    heightmap: &dyn Heightmap,
    colormap: &dyn Colormap,
    options: GenOptions,
    progress_f: F,
) -> Result<Vec<Brick>, String> {
    macro_rules! progress {
        ($e:expr) => {
            if !progress_f($e) {
                return Err("Stopped by user".to_string());
            }
        };
    }
    progress!(0.0);

    let (width, height) = heightmap.size();
    if colormap.size() != (width, height) {
        return Err("Heightmap and colormap must have same dimensions".to_string());
    }
    if width == 0 || height == 0 {
        return Err("Heightmap is empty".to_string());
    }

    // A ramp's rise is quantized to plates, so the vertical scale is too.
    let plates_per_shade = ((options.scale as i32 + CELL_UNITS / 2) / CELL_UNITS).max(1);
    let effective = plates_per_shade * CELL_UNITS;
    if effective != options.scale as i32 {
        info!(
            "Rampified terrain rises {effective} unit(s) per shade (--vertical {}, rounded to \
             {plates_per_shade} plate(s): a ramp's rise is quantized to plates)",
            options.scale
        );
    }

    info!("Reading height columns");
    let count = width as usize * height as usize;
    let mut field = Field {
        width: width as i32,
        height: height as i32,
        cells: vec![0; count],
        colors: vec![[0, 0, 0, 255]; count],
        claimed: vec![Vec::new(); count],
    };
    for y in 0..height {
        for x in 0..width {
            let index = y as usize * width as usize + x as usize;
            let color = colormap.at(x, y);
            let shade = heightmap.at(x, y).min(i32::MAX as u32) as i32;
            field.colors[index] = color;
            field.cells[index] = if options.cull && (shade == 0 || color[3] == 0) {
                0
            } else {
                // `+ 1` so a black pixel is still one plate of ground rather
                // than a hole, matching the minimum slab the blocky modes emit.
                (shade.saturating_mul(plates_per_shade)).saturating_add(1)
            };
        }
    }
    progress!(0.15);

    let layout = Layout {
        half: options.size as i32,
        offset_x: -(width as i32 * options.size as i32),
        offset_y: -(height as i32 * options.size as i32),
        // A one-plate column's TOP lands where the blocky modes put the top of
        // a zero-height cell, so the two modes share a ground plane.
        z_floor: options.base_height() - 5 - CELL_UNITS,
        glow: options.glow,
        collision: options.collision(),
    };

    // The full ascending-Z scan of the mesh rampifier, reduced to its only
    // productive cells: `fit_ramp` and `fit_corner` both refuse a cell with
    // anything above it, so on a height field the sole candidate per column is
    // its top. Sorting by (height, y, x) visits them in exactly the order the
    // z/y/x loop nest would have.
    let mut anchors: Vec<u32> = (0..count as u32)
        .filter(|i| field.cells[*i as usize] > 0)
        .collect();
    anchors.sort_unstable_by_key(|i| {
        let i = *i as usize;
        (field.cells[i], i)
    });

    let mut bricks = Vec::new();
    let mut ramps = 0usize;
    let mut corners = 0usize;

    // Corners run first: a straight ramp along either edge next to a convex
    // corner would otherwise consume the corner's own footprint.
    info!("Fitting corner ramps");
    for (n, index) in anchors.iter().enumerate() {
        let index = *index as usize;
        let cell = Cell(
            (index % width as usize) as i32,
            (index / width as usize) as i32,
            field.cells[index] - 1,
        );
        if field.is_ramp(cell) {
            continue;
        }
        'placed: for rotation in [
            Rotation::Deg0,
            Rotation::Deg90,
            Rotation::Deg180,
            Rotation::Deg270,
        ] {
            for inner in [false, true] {
                if let Some(fit) = field.fit_corner(cell, rotation, inner) {
                    let (run_a, run_b, rise) = fit;
                    let color = field.claim(
                        cell,
                        [
                            (Cell::forward(rotation), run_a),
                            (Cell::forward(next_rotation(rotation)), run_b),
                        ],
                        rise,
                    );
                    bricks.push(corner_brick(&layout, cell, fit, inner, rotation, color));
                    corners += 1;
                    break 'placed;
                }
            }
        }
        if n % 4096 == 0 {
            progress!(0.15 + 0.3 * (n as f32 / anchors.len().max(1) as f32));
        }
    }
    progress!(0.45);

    info!("Fitting ramps");
    for (n, index) in anchors.iter().enumerate() {
        let index = *index as usize;
        let cell = Cell(
            (index % width as usize) as i32,
            (index / width as usize) as i32,
            field.cells[index] - 1,
        );
        if field.is_ramp(cell) {
            continue;
        }
        let Some(rotation) = field.best_rotation(cell) else {
            continue;
        };
        let Some((run, rise)) = field.fit_ramp(cell, rotation) else {
            continue;
        };
        let color = field.claim(
            cell,
            [(Cell::forward(rotation), run), (Cell(0, 0, 0), 1)],
            rise,
        );
        bricks.push(ramp_brick(&layout, cell, run, rise, rotation, color));
        ramps += 1;
        if n % 4096 == 0 {
            progress!(0.45 + 0.3 * (n as f32 / anchors.len().max(1) as f32));
        }
    }
    progress!(0.75);

    info!("Filling gaps");
    let fills = fill_gaps(&field, &layout, &mut bricks, &progress_f)?;

    info!(
        "Rampified {} pixel(s) into {} brick(s): {corners} corner ramp(s), {ramps} ramp(s), \
         {fills} block(s)",
        count,
        bricks.len(),
    );
    progress!(1.0);
    Ok(bricks)
}

/// Fill everything the slopes did not claim with plain bricks, merging equal
/// spans across neighbouring columns.
///
/// The mesh rampifier grows a box cell by cell because its colours vary in Z.
/// A height field's colour is a property of the COLUMN, so what is left of a
/// column after the slopes is a handful of Z spans, and two neighbouring
/// columns merge exactly when a span and a colour both match. That turns the
/// per-cell growth into one pass over spans.
fn fill_gaps<F: Fn(f32) -> bool>(
    field: &Field,
    layout: &Layout,
    out: &mut Vec<Brick>,
    progress_f: &F,
) -> Result<usize, String> {
    let count = field.cells.len();
    let mut spans: Vec<Vec<(i32, i32)>> = Vec::with_capacity(count);
    for index in 0..count {
        spans.push(unclaimed_spans(field, index));
    }
    let mut taken: Vec<Vec<bool>> = spans.iter().map(|s| vec![false; s.len()]).collect();

    // A brick may not exceed 500 units on any axis.
    let max_run = (MAX_HALF_EXTENT / layout.half).max(1);
    let mut emitted = 0usize;

    // A neighbouring column joins this box only when it offers the SAME span,
    // still unclaimed, under the same colour. Spans within one column are
    // disjoint, so at most one can match.
    let matching =
        |index: usize, span: (i32, i32), color: [u8; 4], taken: &[Vec<bool>]| -> Option<usize> {
            if field.colors[index] != color {
                return None;
            }
            spans[index]
                .iter()
                .position(|s| *s == span)
                .filter(|k| !taken[index][*k])
        };

    for y in 0..field.height {
        for x in 0..field.width {
            let index = y as usize * field.width as usize + x as usize;
            for k in 0..spans[index].len() {
                if taken[index][k] {
                    continue;
                }
                let span = spans[index][k];
                let color = field.colors[index];
                let mut claimed = vec![(index, k)];

                let mut run_x = 1;
                while run_x < max_run && x + run_x < field.width {
                    let neighbour = index + run_x as usize;
                    match matching(neighbour, span, color, &taken) {
                        Some(j) => {
                            claimed.push((neighbour, j));
                            run_x += 1;
                        }
                        None => break,
                    }
                }

                let mut run_y = 1;
                'rows: while run_y < max_run && y + run_y < field.height {
                    let mut row = Vec::with_capacity(run_x as usize);
                    for dx in 0..run_x {
                        let neighbour =
                            index + (run_y as usize * field.width as usize) + dx as usize;
                        match matching(neighbour, span, color, &taken) {
                            Some(j) => row.push((neighbour, j)),
                            // A partial row is discarded whole: the box has to
                            // stay rectangular.
                            None => break 'rows,
                        }
                    }
                    claimed.extend(row);
                    run_y += 1;
                }

                for (i, j) in claimed {
                    taken[i][j] = true;
                }

                let plates = span.1 - span.0 + 1;
                let size = BrickSize::new(
                    (run_x * layout.half) as u16,
                    (run_y * layout.half) as u16,
                    (plates * 2) as u16,
                );
                let full = layout.half * 2;
                out.push(layout.brick(
                    PB_DEFAULT_BRICK,
                    size,
                    Position::new(
                        x * full + size.x as i32 + layout.offset_x,
                        y * full + size.y as i32 + layout.offset_y,
                        layout.z_floor + span.0 * CELL_UNITS + size.z as i32,
                    ),
                    Color {
                        r: color[0],
                        g: color[1],
                        b: color[2],
                    },
                    Rotation::Deg0,
                ));
                emitted += 1;
            }
        }
        if y % 64 == 0 && !progress_f(0.75 + 0.2 * (y as f32 / field.height.max(1) as f32)) {
            return Err("Stopped by user".to_string());
        }
    }
    Ok(emitted)
}

/// One column's solid Z spans minus everything a slope claimed, split so no
/// span exceeds the tallest brick the format can carry.
fn unclaimed_spans(field: &Field, index: usize) -> Vec<(i32, i32)> {
    let top = field.cells[index];
    if top <= 0 {
        return Vec::new();
    }
    let mut claimed = field.claimed[index].clone();
    claimed.sort_unstable();

    let mut spans = Vec::new();
    let mut z = 0;
    for (low, high) in claimed {
        let low = low.max(0);
        let high = high.min(top - 1);
        if high < low {
            continue;
        }
        if low > z {
            spans.push((z, low - 1));
        }
        z = z.max(high + 1);
    }
    if z < top {
        spans.push((z, top - 1));
    }

    let max_plates = MAX_HALF_EXTENT / 2;
    let mut split = Vec::with_capacity(spans.len());
    for (low, high) in spans {
        let mut low = low;
        while low <= high {
            let end = (low + max_plates - 1).min(high);
            split.push((low, end));
            low = end + 1;
        }
    }
    split
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A field built from a closure over `(x, y)` giving each column's plate
    /// count, all one colour.
    fn field(width: i32, height: i32, column: impl Fn(i32, i32) -> i32) -> Field {
        let count = (width * height) as usize;
        let mut cells = vec![0; count];
        for y in 0..height {
            for x in 0..width {
                cells[(y * width + x) as usize] = column(x, y);
            }
        }
        Field {
            width,
            height,
            cells,
            colors: vec![[9, 8, 7, 255]; count],
            claimed: vec![Vec::new(); count],
        }
    }

    fn layout() -> Layout {
        Layout {
            half: 5,
            offset_x: 0,
            offset_y: 0,
            z_floor: 0,
            glow: false,
            collision: brdb::Collision::default(),
        }
    }

    /// `brdb::Rotation` carries no `PartialEq`, so assertions compare quarter
    /// turns instead.
    fn turn(rotation: Option<Rotation>) -> Option<u8> {
        rotation.map(|r| match r {
            Rotation::Deg0 => 0,
            Rotation::Deg90 => 1,
            Rotation::Deg180 => 2,
            Rotation::Deg270 => 3,
        })
    }

    fn top(field: &Field, x: i32, y: i32) -> Cell {
        Cell(x, y, field.cells[(y * field.width + x) as usize] - 1)
    }

    /// The rampifier's whole premise: a staircase becomes a slope. A ramp must
    /// fit at the foot of a rising run, and it must climb toward the rise.
    #[test]
    fn a_rising_staircase_fits_a_ramp_climbing_it() {
        // Columns get taller with +X, so the uphill direction is +X: rotation
        // 180, whose forward is (1, 0, 0).
        let field = field(8, 3, |x, _| 1 + x);
        let anchor = top(&field, 0, 1);
        assert_eq!(turn(field.best_rotation(anchor)), Some(2));
        let (run, rise) = field
            .fit_ramp(anchor, Rotation::Deg180)
            .expect("a ramp fits at the foot of a staircase");
        assert!(run >= 2 && rise >= 1, "got run {run}, rise {rise}");
    }

    /// A cell with material above it can never anchor a slope. This is what
    /// licenses visiting one anchor per column instead of scanning every Z.
    #[test]
    fn only_the_top_cell_of_a_column_can_anchor_a_slope() {
        let field = field(8, 3, |x, _| 1 + x);
        let buried = Cell(3, 1, 0);
        assert!(
            field.exists(buried + Cell(0, 0, 1)),
            "the test cell must be buried"
        );
        assert_eq!(turn(field.best_rotation(buried)), None);
        assert_eq!(field.fit_corner(buried, Rotation::Deg0, false), None);
        assert_eq!(field.fit_corner(buried, Rotation::Deg0, true), None);
    }

    /// Inside a flat plain there is nothing to climb, so no cell may slope.
    ///
    /// The RIM is deliberately excluded: a column at the map's edge has open
    /// air behind it and material in front, which is a one-plate step, and the
    /// rampifier bevels it exactly as the mesh version bevels the edge of a
    /// slab. That is a property of the edge, not of the plain.
    #[test]
    fn the_interior_of_a_flat_plain_fits_no_ramp_at_all() {
        let field = field(8, 8, |_, _| 3);
        for y in 1..7 {
            for x in 1..7 {
                assert_eq!(
                    turn(field.best_rotation(top(&field, x, y))),
                    None,
                    "({x}, {y}) is surrounded by its own height and must not slope"
                );
            }
        }
        // ...and the rim really does bevel, so the exclusion above is a
        // statement about this renderer rather than a hole in the test.
        assert!(turn(field.best_rotation(top(&field, 0, 4))).is_some());
    }

    /// A convex plateau turn is what the corner asset exists for; a straight
    /// edge must never match one, or corners would eat the whole ridge.
    #[test]
    fn a_convex_plateau_turn_fits_a_corner_and_a_straight_edge_does_not() {
        // A 12x12 plain with a raised quadrant, so the plateau's outer corner
        // is a convex turn and its edges are straight. The anchor is the
        // plateau's own corner cell -- a corner ramp BEVELS the turn it sits
        // on, it does not sit on the ground beside it.
        let field = field(12, 12, |x, y| if x >= 6 && y >= 6 { 4 } else { 1 });
        let corner = top(&field, 6, 6);
        let fits = [
            Rotation::Deg0,
            Rotation::Deg90,
            Rotation::Deg180,
            Rotation::Deg270,
        ]
        .into_iter()
        .any(|r| field.fit_corner(corner, r, false).is_some());
        assert!(
            fits,
            "the convex plateau corner must fit an outer corner ramp"
        );

        let straight = top(&field, 6, 9);
        let straight_fits = [
            Rotation::Deg0,
            Rotation::Deg90,
            Rotation::Deg180,
            Rotation::Deg270,
        ]
        .into_iter()
        .any(|r| field.fit_corner(straight, r, false).is_some());
        assert!(
            !straight_fits,
            "a straight plateau edge must be a plain ramp, not a corner"
        );
    }

    /// Claiming a slope's cells must remove them from the fill pass, or every
    /// ramp would be buried inside the blocks meant to sit under it.
    #[test]
    fn claimed_cells_do_not_come_back_as_fill_blocks() {
        let mut field = field(8, 3, |x, _| 1 + x);
        let anchor = top(&field, 0, 1);
        let (run, rise) = field.fit_ramp(anchor, Rotation::Deg180).unwrap();
        field.claim(
            anchor,
            [(Cell::forward(Rotation::Deg180), run), (Cell(0, 0, 0), 1)],
            rise,
        );

        let index = (1 * field.width + anchor.0) as usize;
        assert!(field.is_ramp(anchor));
        let spans = unclaimed_spans(&field, index);
        assert!(
            spans
                .iter()
                .all(|(low, high)| anchor.2 < *low || anchor.2 > *high),
            "the anchor cell is still offered to the fill pass: {spans:?}"
        );
    }

    #[test]
    fn a_tall_column_is_split_into_legal_bricks() {
        let field = field(1, 1, |_, _| 1000);
        let spans = unclaimed_spans(&field, 0);
        assert!(spans.len() > 1, "a 1000-plate column must be split");
        for (low, high) in &spans {
            assert!(
                (high - low + 1) * 2 <= MAX_HALF_EXTENT,
                "span {low}..={high} exceeds the largest legal brick"
            );
        }
        // No gaps and no overlap: the splits must still tile the column.
        assert_eq!(spans[0].0, 0);
        assert_eq!(spans.last().unwrap().1, 999);
        for pair in spans.windows(2) {
            assert_eq!(pair[0].1 + 1, pair[1].0);
        }
    }

    /// A flat plain must merge into a few big blocks rather than one per pixel.
    #[test]
    fn the_fill_pass_merges_equal_columns_into_boxes() {
        let field = field(16, 16, |_, _| 2);
        let mut bricks = Vec::new();
        let filled = fill_gaps(&field, &layout(), &mut bricks, &|_| true).unwrap();
        assert_eq!(filled, bricks.len());
        assert!(
            bricks.len() < 32,
            "256 identical columns collapsed to only {} brick(s)",
            bricks.len()
        );
    }

    /// One-cell runs have no ramp asset; the wedge is the one-cell ramp.
    #[test]
    fn a_single_cell_run_uses_the_wedge_asset() {
        let wedge = ramp_brick(
            &layout(),
            Cell(0, 0, 0),
            1,
            2,
            Rotation::Deg0,
            Color::new(1, 2, 3),
        );
        let ramp = ramp_brick(
            &layout(),
            Cell(0, 0, 0),
            3,
            2,
            Rotation::Deg0,
            Color::new(1, 2, 3),
        );
        let BrickType::Procedural {
            asset: wedge_asset, ..
        } = &wedge.asset
        else {
            panic!("rampify emits procedural bricks");
        };
        let BrickType::Procedural {
            asset: ramp_asset, ..
        } = &ramp.asset
        else {
            panic!("rampify emits procedural bricks");
        };
        assert_eq!(wedge_asset.as_ref(), "PB_DefaultWedge");
        assert_eq!(ramp_asset.as_ref(), "PB_DefaultRamp");
    }

    /// A ramp's footprint must land exactly on the cells it consumed, whichever
    /// way it runs. Two of the four rotations run toward negative axes, which
    /// is exactly where a centring mistake hides.
    #[test]
    fn a_ramp_is_centred_over_the_cells_it_covers() {
        let layout = layout();
        let full = layout.half * 2;
        for (rotation, forward) in [
            (Rotation::Deg0, Cell(-1, 0, 0)),
            (Rotation::Deg90, Cell(0, -1, 0)),
            (Rotation::Deg180, Cell(1, 0, 0)),
            (Rotation::Deg270, Cell(0, 1, 0)),
        ] {
            let anchor = Cell(4, 4, 0);
            let run = 3;
            let brick = ramp_brick(&layout, anchor, run, 2, rotation, Color::new(0, 0, 0));
            let far = anchor + forward * (run - 1);
            // Centre of the covered span, in units, cell centres included.
            let expect_x = (anchor.0.min(far.0) * full + anchor.0.max(far.0) * full + full) / 2;
            let expect_y = (anchor.1.min(far.1) * full + anchor.1.max(far.1) * full + full) / 2;
            assert_eq!(
                (brick.position.x, brick.position.y),
                (expect_x, expect_y),
                "{rotation:?} ramp is off its footprint"
            );
        }
    }
}
