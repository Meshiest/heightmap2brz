//! The Wrapperup rampifier, over the height columns of a heightmap.
//!
//! The algorithm that selects the slopes is the Wrapperup/rampifier
//! algorithm. It comes here from the BRDB version in `obj2brz`
//! (`crates/obj2brz/src/rampify.rs`): examine the surface, fit the longest
//! ramp that touches the ground that goes up in front of it, use corner ramps
//! at convex turns and at concave turns, then fill the remainder with usual
//! bricks. The fit rules, the limits on the run and the rise, and the brick
//! geometry are all the same as in that version.
//!
//! The dense grid of voxels is NOT the same. `obj2brz` makes ramps from any
//! mesh, so it makes one cell for each voxel of the bounding box and refuses a
//! model of more than 64M cells. A heightmap is a HEIGHT FIELD: a column is
//! solid from the ground to its top and is empty above the top. To find the
//! material is thus a comparison and not a lookup, and a map of 4096x4096 at
//! 255 plates costs one `Vec<i32>` for each column. The bounding box would
//! ask for four terabytes. Two results of this come from that fact and the
//! code below uses them:
//!
//! * **Only the top cell of a column can hold a slope.** `fit_ramp` and
//!   `fit_corner` both refuse a cell that has material above it. The full scan
//!   up the Z axis thus becomes one visit to each column, in the sequence of
//!   the heights.
//! * **There is no closed air and no lower face.** The mesh rampifier fills
//!   air to find the inside of a closed model, and it puts ramps upside down
//!   below an overhang. A height field has neither, so only the floor pass
//!   runs.
//!
//! One voxel is one pixel wide (`GenOptions::size` half extents) and one plate
//! high (4 units). The code reads `--vertical` as units and changes it to a
//! whole number of plates, because the rise of a `PB_DefaultRamp` must be a
//! whole number of plates.

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

/// The longest run of a ramp, in cells. This is the Wrapperup value.
const RAMP_MAX_RUN: i32 = 4;
/// The largest rise of a ramp, in cells. This is the Wrapperup value.
const RAMP_MAX_RISE: i32 = 12;
/// The number of vertical units in one cell. One cell is one plate.
const CELL_UNITS: i32 = 4;
/// The largest half extent that a procedural brick can hold.
const MAX_HALF_EXTENT: i32 = 250;

/// One cell position. X and Y give the pixel, and Z counts the plates above
/// the ground.
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
    /// The direction in which a ramp with this rotation goes UP. Rotation zero
    /// goes down at `+X`, so it goes up at `-X`. The other rotations are
    /// quarter turns counterclockwise around `+Z`.
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

/// The height field that the rampifier reads, with the cells that a slope
/// already uses.
struct Field {
    width: i32,
    height: i32,
    /// The number of solid cells in each column. The column fills `0..cells`.
    cells: Vec<i32>,
    colors: Vec<[u8; 4]>,
    /// The Z ranges of each column that a slope uses, with both ends included.
    /// A column holds only a small number of these, so a short `Vec` uses much
    /// less memory than one bit for each cell on a real map.
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

    /// The number of solid cells at `cell` and above it, to a maximum of 31.
    /// The result is `i32::MIN` if the cell is empty. An empty adjacent cell
    /// can thus never win the `max_by_key` in [`Self::best_rotation`].
    fn slope_height(&self, cell: Cell) -> i32 {
        if !self.exists(cell) {
            return i32::MIN;
        }
        (self.column(cell) - cell.2).min(31)
    }

    /// The direction in which the ground in front of `cell` goes up the most,
    /// while the cell BEHIND it is empty. A ramp at this cell must go up in
    /// that direction.
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

    /// The run and the rise of the straight ramp that starts at `cell` and
    /// goes up at `rotation`. The result is `None` if no ramp fits.
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
        // A top that the ramp goes over gets one more cell of rise. A ridge
        // thus gets a full top and does not stay one plate too low.
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

    /// The runs along each wall axis and the rise of a corner ramp. The two
    /// high walls point at `rotation` and at `rotation + 90`, and `cell` is
    /// the low outer corner.
    ///
    /// An OUTER corner is a convex turn. It has empty cells behind both wall
    /// axes, and the ground goes up only after the far diagonal cell. An INNER
    /// corner is a concave turn. The edge turns the other way there: the cells
    /// behind it are still part of the edge, only the diagonal between them is
    /// empty, and the ground goes up along the full far row and the full far
    /// column.
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
        // On rough ground the two edge fits are usually different. The lower
        // rise still touches both adjacent slopes.
        let rise = rise_a.min(rise_b);

        // The surface of an outer corner is where the two straight ramps
        // AGREE, so it is at the full height only at the far diagonal cell.
        // The surface of an inner corner is where EITHER ramp is high, so it
        // is at the full height along the full far row and the full far
        // column. Those cells can hold the ground that goes up. At each other
        // cell the area must be flat, with air above it.
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
        // On rough ground something usually blocks the area of the full run.
        // The code then uses the largest empty rectangle that is still a
        // corner, which is 2x2.
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

    /// Use each cell in a box, and give the color that occurs most in the
    /// solid cells. A slope covers more than one column. To use the color of
    /// the first cell would give each ramp the color of its lowest pixel.
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

/// The position of the cell grid in world units, and the appearance of the
/// bricks. It is a separate type, so a test can use the geometry without a
/// `GenOptions`.
struct Layout {
    /// The half extent of one cell in X and Y, in units.
    half: i32,
    offset_x: i32,
    offset_y: i32,
    /// The world Z of the bottom of cell layer zero.
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

/// Make a straight ramp at `cell`, or a wedge if the run is one cell.
///
/// `cell` is the LOW end of the ramp. The brick continues for `run` cells
/// along the forward axis of the rotation. For two of the four rotations that
/// axis points at NEGATIVE X or NEGATIVE Y. The values below put the center of
/// the brick at the correct position for each rotation. They come from
/// `create_ramp`, but the fixed stud of `obj2brz` (5 units for a half, 10 for
/// the full width) becomes `Layout::half`.
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
        // There is no ramp asset for a run of one cell. The wedge is the ramp
        // of one cell.
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

/// Make a corner ramp. Its position is its low outer corner.
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
    // The local X axis follows the first wall axis. A quarter turn thus
    // changes which run is along the world X axis.
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

/// Make a heightmap with ramps: usual ramps, wedges and corner ramps on the
/// surface of the column field, with plain bricks below them and beside
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
    // A zero cell size would make `layout.half` zero, which divides by zero in
    // `fill_gaps` (`MAX_HALF_EXTENT / layout.half`) and would emit zero-extent
    // bricks besides. Refuse it like the other invalid inputs above.
    if options.size == 0 {
        return Err("Brick size must be at least 1".to_string());
    }

    // The rise of a ramp is a whole number of plates, so the vertical scale
    // must also be a whole number of plates. Round in i64 so an absurdly large
    // `--vertical` can't overflow i32; the result fits i32 (at most u32::MAX/4)
    // and the per-shade height product below is a saturating_mul.
    let plates_per_shade =
        (((options.scale as i64) + (CELL_UNITS / 2) as i64) / CELL_UNITS as i64).max(1) as i32;
    let effective = plates_per_shade as i64 * CELL_UNITS as i64;
    if effective != options.scale as i64 {
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
                // The `+ 1` gives a black pixel one plate of ground and not a
                // hole. This agrees with the smallest brick that the other
                // modes make.
                (shade.saturating_mul(plates_per_shade)).saturating_add(1)
            };
        }
    }
    progress!(0.15);

    let layout = Layout {
        half: options.size as i32,
        offset_x: -(width as i32 * options.size as i32),
        offset_y: -(height as i32 * options.size as i32),
        // The TOP of a column of one plate is where the other modes put the
        // top of a cell of height zero. The modes thus share a ground level.
        z_floor: options.base_height() - 5 - CELL_UNITS,
        glow: options.glow,
        collision: options.collision(),
    };

    // The mesh rampifier scans up the Z axis. Only some cells can give a
    // result: `fit_ramp` and `fit_corner` both refuse a cell that has material
    // above it, so on a height field the only candidate in a column is its
    // top. To sort by height and then by position visits the cells in the same
    // sequence as the loops over z, y and x.
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

    // The corners come first. If they did not, a straight ramp along an edge
    // beside a convex corner would use the cells of the corner.
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

/// Fill each cell that the slopes did not use with plain bricks, and join
/// equal Z ranges in adjacent columns.
///
/// The mesh rampifier makes a box one cell at a time, because its colors
/// change along Z. But in a height field the color belongs to the COLUMN. What
/// stays after the slopes is thus a small number of Z ranges in each column,
/// and two adjacent columns join if a range and a color both agree. One pass
/// over the ranges then replaces the growth cell by cell.
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

    // An adjacent column joins this box only if it has the SAME range, if no
    // brick uses that range, and if the color agrees. The ranges in one column
    // do not touch, so one range at most can agree.
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
                            // The code rejects the full row, because the box
                            // must stay rectangular.
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

/// The solid Z ranges of one column, less each range that a slope uses. The
/// code divides a range that is higher than the largest brick that the format
/// holds.
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

    /// A field where a closure gives the number of plates in each column. Each
    /// column has the same color.
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

    /// `brdb::Rotation` has no `PartialEq`, so the tests compare quarter turns.
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

    fn rotations() -> [Rotation; 4] {
        [
            Rotation::Deg0,
            Rotation::Deg90,
            Rotation::Deg180,
            Rotation::Deg270,
        ]
    }

    fn asset(brick: &Brick) -> &str {
        let BrickType::Procedural { asset, .. } = &brick.asset else {
            panic!("rampify makes procedural bricks only")
        };
        asset.as_ref()
    }

    /// The purpose of the rampifier: steps become a slope. A ramp must fit at
    /// the bottom of a run that goes up, and it must go up at the high side.
    #[test]
    fn a_rising_staircase_fits_a_ramp_climbing_it() {
        // The columns get higher at +X, so the ramp goes up at +X. That is
        // rotation 180, whose forward direction is (1, 0, 0).
        let field = field(8, 3, |x, _| 1 + x);
        let anchor = top(&field, 0, 1);
        assert_eq!(turn(field.best_rotation(anchor)), Some(2));
        let (run, rise) = field
            .fit_ramp(anchor, Rotation::Deg180)
            .expect("a ramp fits at the bottom of the steps");
        assert!(run >= 2 && rise >= 1, "the fit gave run {run} and rise {rise}");
    }

    /// A cell that has material above it can never hold a slope. This rule
    /// permits one visit to each column in place of a scan of each Z value.
    #[test]
    fn only_the_top_cell_of_a_column_can_anchor_a_slope() {
        let field = field(8, 3, |x, _| 1 + x);
        let buried = Cell(3, 1, 0);
        assert!(
            field.exists(buried + Cell(0, 0, 1)),
            "the test cell must have material above it"
        );
        assert_eq!(turn(field.best_rotation(buried)), None);
        assert_eq!(field.fit_corner(buried, Rotation::Deg0, false), None);
        assert_eq!(field.fit_corner(buried, Rotation::Deg0, true), None);
    }

    /// Inside a flat plain there is nothing to go up, so no cell can slope.
    ///
    /// The edge of the map is not part of this test. A column at the edge has
    /// air behind it and material in front of it, which is a step of one
    /// plate. The rampifier cuts that edge, in the same way as the mesh
    /// version cuts the edge of a slab. That is a property of the edge and not
    /// of the plain, and the last line holds it.
    #[test]
    fn the_interior_of_a_flat_plain_fits_no_ramp_at_all() {
        let field = field(8, 8, |_, _| 3);
        for y in 1..7 {
            for x in 1..7 {
                assert_eq!(
                    turn(field.best_rotation(top(&field, x, y))),
                    None,
                    "the cell ({x}, {y}) has the same height as each adjacent cell"
                );
            }
        }
        assert!(turn(field.best_rotation(top(&field, 0, 4))).is_some());
    }

    /// A corner asset is for a convex turn. A straight edge must never fit
    /// one, or the corners would use the full ridge.
    #[test]
    fn a_convex_plateau_turn_fits_a_corner_and_a_straight_edge_does_not() {
        // A plain of 12x12 with one high quarter. The outer corner of the high
        // part is a convex turn, and its edges are straight. The test starts
        // at the corner cell of the high part, because a corner ramp cuts the
        // turn that it is on. It does not sit on the ground beside it.
        let field = field(12, 12, |x, y| if x >= 6 && y >= 6 { 4 } else { 1 });
        let corner = top(&field, 6, 6);
        assert!(
            rotations()
                .into_iter()
                .any(|r| field.fit_corner(corner, r, false).is_some()),
            "the convex corner of the high part must fit an outer corner ramp"
        );

        let straight = top(&field, 6, 9);
        assert!(
            !rotations()
                .into_iter()
                .any(|r| field.fit_corner(straight, r, false).is_some()),
            "a straight edge must be a usual ramp and not a corner"
        );
    }

    /// A slope must remove its cells from the fill pass. If it did not, each
    /// ramp would be inside the blocks that must go below it.
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

        assert!(field.is_ramp(anchor));
        let spans = unclaimed_spans(&field, (field.width + anchor.0) as usize);
        assert!(
            spans
                .iter()
                .all(|(low, high)| anchor.2 < *low || anchor.2 > *high),
            "the fill pass can still use the cell of the ramp: {spans:?}"
        );
    }

    /// A column that is higher than one brick must become more than one brick.
    /// Those bricks must fill the column, with no gap and no overlap.
    #[test]
    fn a_tall_column_is_split_into_legal_bricks() {
        let field = field(1, 1, |_, _| 1000);
        let spans = unclaimed_spans(&field, 0);
        assert!(spans.len() > 1, "a column of 1000 plates must be divided");
        for (low, high) in &spans {
            assert!(
                (high - low + 1) * 2 <= MAX_HALF_EXTENT,
                "the range {low}..={high} is larger than the largest legal brick"
            );
        }
        assert_eq!(spans[0].0, 0);
        assert_eq!(spans.last().unwrap().1, 999);
        for pair in spans.windows(2) {
            assert_eq!(pair[0].1 + 1, pair[1].0);
        }
    }

    /// A flat plain must become a small number of large blocks and not one
    /// block for each pixel.
    #[test]
    fn the_fill_pass_merges_equal_columns_into_boxes() {
        let field = field(16, 16, |_, _| 2);
        let mut bricks = Vec::new();
        let filled = fill_gaps(&field, &layout(), &mut bricks, &|_| true).unwrap();
        assert_eq!(filled, bricks.len());
        assert!(
            bricks.len() < 32,
            "256 equal columns gave {} brick(s)",
            bricks.len()
        );
    }

    /// The geometry of an emitted ramp: the correct asset for its length, and
    /// a center over the cells that it covers.
    ///
    /// Two of the four rotations go at a NEGATIVE axis, which is where an
    /// error in the center is easy to make.
    #[test]
    fn a_ramp_uses_the_right_asset_and_covers_its_own_cells() {
        let layout = layout();
        let full = layout.half * 2;
        let color = Color::new(1, 2, 3);

        assert_eq!(
            asset(&ramp_brick(&layout, Cell(0, 0, 0), 1, 2, Rotation::Deg0, color)),
            "PB_DefaultWedge"
        );
        assert_eq!(
            asset(&ramp_brick(&layout, Cell(0, 0, 0), 3, 2, Rotation::Deg0, color)),
            "PB_DefaultRamp"
        );

        for (rotation, forward) in [
            (Rotation::Deg0, Cell(-1, 0, 0)),
            (Rotation::Deg90, Cell(0, -1, 0)),
            (Rotation::Deg180, Cell(1, 0, 0)),
            (Rotation::Deg270, Cell(0, 1, 0)),
        ] {
            let (anchor, run) = (Cell(4, 4, 0), 3);
            let brick = ramp_brick(&layout, anchor, run, 2, rotation, color);
            let far = anchor + forward * (run - 1);
            // The center of the cells that the ramp covers, in units.
            let expect_x = (anchor.0.min(far.0) * full + anchor.0.max(far.0) * full + full) / 2;
            let expect_y = (anchor.1.min(far.1) * full + anchor.1.max(far.1) * full + full) / 2;
            assert_eq!(
                (brick.position.x, brick.position.y),
                (expect_x, expect_y),
                "the ramp at {rotation:?} is not over its own cells"
            );
        }
    }

    /// A zero brick size must be refused with an error, not a divide-by-zero
    /// panic in `fill_gaps` (`MAX_HALF_EXTENT / layout.half`).
    #[test]
    fn a_zero_brick_size_is_refused_not_a_panic() {
        struct Flat;
        impl Heightmap for Flat {
            fn at(&self, _x: u32, _y: u32) -> u32 {
                2
            }
            fn size(&self) -> (u32, u32) {
                (4, 4)
            }
        }
        struct Grey;
        impl Colormap for Grey {
            fn at(&self, _x: u32, _y: u32) -> [u8; 4] {
                [128, 128, 128, 255]
            }
            fn size(&self) -> (u32, u32) {
                (4, 4)
            }
        }
        let opts = GenOptions {
            size: 0,
            scale: 4,
            asset: PB_DEFAULT_BRICK,
            cull: false,
            micro: false,
            stud: false,
            snap: false,
            img: false,
            glow: false,
            hdmap: false,
            lrgb: false,
            nocollide: false,
            quadtree: true,
            greedy: false,
            surface: SurfaceMode::Rampify,
        };
        match gen_rampify_heightmap(&Flat, &Grey, opts, |_| true) {
            Err(e) => assert!(e.contains("size"), "unexpected error: {e}"),
            Ok(_) => panic!("a zero brick size must be refused"),
        }
    }
}
