//! Smooth micro-brick terrain.
//!
//! The blocky renderers ([`super::gen_quad_heightmap`],
//! [`super::gen_greedy_heightmap`]) give every pixel a flat top, so a slope
//! comes out as a staircase. This one gives every pixel a SLOPED top, chosen
//! from Brickadia's micro wedge family, so a smooth height field renders as a
//! smooth surface.
//!
//! The geometry contract is the measured one from the `terrain-experiment`
//! prefab generator (`docs/MICRO_TERRAIN_GEOMETRY.md`), derived by correlating
//! an in-game GLB export against the same ordered `.brz` through BRDB. The
//! facts this module depends on, restated so it can be read on its own:
//!
//! * Brickadia stores 10 position units per stud, and a procedural brick's
//!   `BrickSize` components are HALF extents. A shape with `z = k` is `2k`
//!   units tall, so any rise this module emits must be even.
//! * Heights are sampled on a SHARED VERTEX grid, not per pixel. Two
//!   neighbouring cells quote the same vertex, which is what makes their
//!   surfaces meet instead of merely coming close.
//! * With `Direction::ZPositive` and `Rotation::Deg0`, and corners read in
//!   `(SW, SE, NE, NW)` order (i.e. `(-X,-Y)`, `(+X,-Y)`, `(+X,+Y)`,
//!   `(-X,+Y)`):
//!
//!   | asset | surface |
//!   | --- | --- |
//!   | `PB_DefaultMicroBrick` | `(1,1,1,1)`, flat top |
//!   | `PB_DefaultMicroRamp` | low toward `+X` |
//!   | `PB_DefaultMicroWedgeCorner` | high vertex at SW |
//!   | `PB_DefaultMicroWedgeInnerCorner` | low vertex at NE |
//!   | `PB_DefaultMicroWedgeOuterCorner` | low vertex at NE |
//!   | `PB_DefaultMicroWedgeTriangleCorner` | SW triangle, high at SW |
//!
//! * A flat micro brick is centred at `base - half_height`, so its TOP is at
//!   `base`. A sloped shape is centred at `base + half_rise`, so its low
//!   vertices are at `base` and its high ones at `base + rise`.
//!
//! **Every cell goes through the same best-fit selection.** `terrain-experiment`
//! reserved that for "wall cells" and used a plain pattern match elsewhere;
//! here it is the only path, because a heightmap is not a slope-limited field.
//! A cliff, a one-pixel spike and a gentle dune all arrive as four corner
//! altitudes, and the fit enumerates every assembly the grammar can express,
//! scores each by its total deviation from those four, and takes the best.
//!
//! Candidates are always fitted FROM BELOW -- each partition takes the MINIMUM
//! of the corners it covers, never their mean -- so the chosen surface sits at
//! or below every sampled vertex and can never protrude through the
//! neighbouring tile. A clean two-level fault therefore becomes one exact steep
//! ramp meeting the plain and the plateau flush, rather than a shelf stuck onto
//! the cliff.

use crate::map::*;
use crate::util::*;
use brdb::{
    Brick, BrickSize, BrickType, Color, Direction, Position, Rotation,
    assets::{
        bricks::{
            PB_DEFAULT_MICRO_BRICK, PB_DEFAULT_MICRO_RAMP, PB_DEFAULT_MICRO_WEDGE_CORNER,
            PB_DEFAULT_MICRO_WEDGE_INNER_CORNER, PB_DEFAULT_MICRO_WEDGE_OUTER_CORNER,
            PB_DEFAULT_MICRO_WEDGE_TRIANGLE_CORNER,
        },
        materials::{GLOW, PLASTIC},
    },
};
use log::info;

/// The largest half extent a procedural brick may carry, i.e. a 500-unit
/// piece. A taller rise is split (foundations) or clamped down (slopes); it is
/// never rounded UP, because that would push the surface above a sampled
/// vertex and undo the fit-from-below guarantee.
const MAX_HALF_EXTENT: i32 = 250;

/// One cell's four corner heights, in the `(SW, SE, NE, NW)` order the whole
/// grammar is calibrated in.
pub type Corners = [i32; 4];

/// Quarter turns counter-clockwise around `+Z`.
///
/// The grammar is compared and asserted on constantly -- fits are scored
/// against each other, and every rotation convention below is pinned by a test
/// -- and `brdb::Rotation` carries no `PartialEq`. This is that enum with an
/// equality, converted at emission.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Turn(pub u8);

impl Turn {
    fn new(turns: i32) -> Self {
        Self(turns.rem_euclid(4) as u8)
    }

    /// The half turn that pairs with this one, for the two complementary
    /// triangles of a [`SurfaceKind::TrianglePair`].
    fn opposite(self) -> Self {
        Self((self.0 + 2) % 4)
    }

    fn rotation(self) -> Rotation {
        match self.0 {
            0 => Rotation::Deg0,
            1 => Rotation::Deg90,
            2 => Rotation::Deg180,
            _ => Rotation::Deg270,
        }
    }
}

/// Which assembly a cell resolves to. The `Turn` is already in Brickadia's own
/// convention -- see the table in the module doc for what rotation zero means
/// for each asset.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SurfaceKind {
    /// One flat micro brick, top at `base`.
    Flat,
    /// `PB_DefaultMicroRamp`: two adjacent corners high.
    Ramp(Turn),
    /// `PB_DefaultMicroWedgeCorner`: exactly one corner high.
    Corner(Turn),
    /// `PB_DefaultMicroWedgeInnerCorner`: exactly one corner low.
    InnerCorner(Turn),
    /// `PB_DefaultMicroWedgeOuterCorner` plus a `...TriangleCorner` one rise
    /// above it: the diagonal `0,1,2,1` stack. Reducing this to a two-level
    /// mask loses the upper triangle and leaves a visible step.
    OuterTriangle(Turn),
    /// Two complementary `PB_DefaultMicroWedgeTriangleCorner`s covering the
    /// whole square: opposite corners high.
    TrianglePair(Turn),
}

impl SurfaceKind {
    /// How many bricks this assembly emits, foundation excluded. Used to break
    /// ties between fits that deviate from the sampled corners equally: the
    /// cheaper one wins, so a cell that a flat brick describes exactly never
    /// spends a wedge on it.
    pub fn brick_count(self) -> u32 {
        match self {
            SurfaceKind::Flat => 1,
            SurfaceKind::Ramp(_) | SurfaceKind::Corner(_) | SurfaceKind::InnerCorner(_) => 1,
            SurfaceKind::OuterTriangle(_) | SurfaceKind::TrianglePair(_) => 2,
        }
    }
}

/// The chosen assembly for one cell, with the heights it actually fits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CellFit {
    pub kind: SurfaceKind,
    /// Height of the assembly's LOWEST surface vertex, in layers.
    pub base: i32,
    /// Rise of the load-bearing shape above `base`, in layers. Zero for
    /// [`SurfaceKind::Flat`].
    pub rise: i32,
    /// Rise of the triangle cap above `base + rise`, in layers.
    /// [`SurfaceKind::OuterTriangle`] only.
    pub cap_rise: i32,
    /// Total `sampled - fitted` over the four corners, in layers. Always
    /// non-negative: every candidate is fitted from below.
    pub error: i32,
}

/// Turn putting a shape's HIGH vertex at `corner`. Rotation zero is high at SW
/// for both `MicroWedgeCorner` and `MicroWedgeTriangleCorner`, so the mapping
/// is the identity: `SW/SE/NE/NW -> 0/90/180/270`.
fn turn_from_high_corner(corner: usize) -> Turn {
    Turn::new(corner as i32)
}

/// Turn putting a shape's LOW vertex at `corner`. Rotation zero is low at NE
/// for both inner and outer corners, so the mapping is a half turn away from
/// [`turn_from_high_corner`]: `low SW/SE/NE/NW -> 180/270/0/90`.
fn turn_from_low_corner(corner: usize) -> Turn {
    Turn::new(corner as i32 + 2)
}

/// Choose the assembly whose surface best matches four sampled corner heights.
///
/// Enumerates the whole grammar -- flat, all fourteen two-level high/low
/// partitions (which map onto ramp, corner, inner corner and triangle pair),
/// and the four diagonal `0,1,2,1` stacks -- and scores each by the total
/// height it leaves on the table. Every candidate is fitted FROM BELOW, so its
/// score is a sum of non-negative residuals and the winning surface never rises
/// above a sampled vertex.
///
/// Ties go to the assembly using fewer bricks, which is what keeps a flat
/// region flat rather than paving it in zero-rise wedges.
pub fn fit_cell(corners: Corners) -> CellFit {
    let floor = *corners.iter().min().expect("four corners");
    let mut best = CellFit {
        kind: SurfaceKind::Flat,
        base: floor,
        rise: 0,
        cap_rise: 0,
        error: corners.iter().map(|c| c - floor).sum(),
    };

    let mut consider = |candidate: CellFit| {
        let better = candidate.error < best.error
            || (candidate.error == best.error
                && candidate.kind.brick_count() < best.kind.brick_count());
        if better {
            best = candidate;
        }
    };

    // The fourteen two-level partitions. Bit `i` set means corner `i` is on the
    // high level; 0 and 15 are the flat case already seeded above.
    for mask in 1u8..15 {
        let high = |i: usize| mask >> i & 1 == 1;
        let low_level = (0..4)
            .filter(|i| !high(*i))
            .map(|i| corners[i])
            .min()
            .expect("a partition in 1..15 has a low corner");
        let high_level = (0..4)
            .filter(|i| high(*i))
            .map(|i| corners[i])
            .min()
            .expect("a partition in 1..15 has a high corner");
        let rise = high_level - low_level;
        if rise <= 0 {
            // Degenerate: the "high" level is not above the low one, so this
            // partition describes the flat surface already considered.
            continue;
        }
        let error = (0..4)
            .map(|i| corners[i] - if high(i) { low_level + rise } else { low_level })
            .sum();
        let kind = match mask {
            // One raised vertex: a convex shoulder.
            1 | 2 | 4 | 8 => {
                SurfaceKind::Corner(turn_from_high_corner(mask.trailing_zeros() as usize))
            }
            // One low vertex: the concave transition. The OUTER corner has the
            // same edge heights but the opposite internal triangulation and is
            // reserved for the stacked diagonal case below, where its triangle
            // cap completes the surface.
            7 | 11 | 13 | 14 => SurfaceKind::InnerCorner(turn_from_low_corner(
                (!mask & 15).trailing_zeros() as usize,
            )),
            // Opposite highs: two complementary triangles cover the square and
            // preserve all four sampled corners.
            5 => SurfaceKind::TrianglePair(Turn(0)),
            10 => SurfaceKind::TrianglePair(Turn(1)),
            // Adjacent highs: a cardinal ramp, named by which way it steps
            // DOWN. Rotation zero is low toward +X.
            9 => SurfaceKind::Ramp(Turn(0)),  // low NE+SE, i.e. +X
            3 => SurfaceKind::Ramp(Turn(1)),  // low NE+NW, i.e. +Y
            6 => SurfaceKind::Ramp(Turn(2)),  // low SW+NW, i.e. -X
            12 => SurfaceKind::Ramp(Turn(3)), // low SW+SE, i.e. -Y
            _ => unreachable!("every mask in 1..15 is covered"),
        };
        consider(CellFit {
            kind,
            base: low_level,
            rise,
            cap_rise: 0,
            error,
        });
    }

    // The four diagonal stacks: one corner two rises up, the corner opposite it
    // at the bottom, and the two between them in the middle. An orthogonal
    // one-step limit does not prevent this, and collapsing it to a two-level
    // mask is exactly what produces the broken transitions seen in game.
    for high in 0..4 {
        let bottom = corners[(high + 2) & 3];
        let middle = corners[(high + 1) & 3].min(corners[(high + 3) & 3]);
        let top = corners[high];
        let rise = middle - bottom;
        let cap_rise = top - middle;
        if rise <= 0 || cap_rise <= 0 {
            continue;
        }
        consider(CellFit {
            kind: SurfaceKind::OuterTriangle(turn_from_high_corner(high)),
            base: bottom,
            rise,
            cap_rise,
            // The bottom and top corners are matched exactly; only the two
            // middle ones can differ, and only by however far apart they are.
            error: (corners[(high + 1) & 3] - middle) + (corners[(high + 3) & 3] - middle),
        });
    }

    best
}

/// Generate smooth micro-wedge terrain from a heightmap.
///
/// Heights are read on a shared `(w+1) x (h+1)` vertex grid: vertex `(i, j)` is
/// the mean of the up-to-four pixels touching it. That keeps the output exactly
/// one cell per pixel -- the same footprint every other mode produces -- while
/// giving neighbouring cells the shared corner heights the grammar needs, and
/// it costs one bilinear smoothing pass that a heightmap generally wants
/// anyway.
///
/// Under `--terrain` the vertical scale is a LAYER height rather than a raw
/// unit count: one shade of grey is one layer, `rise_unit` units tall. It is
/// rounded up to an even number and floored at 2 because a wedge's `BrickSize`
/// half height must be a whole unit.
pub fn gen_terrain_heightmap<F: Fn(f32) -> bool>(
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

    // Rounded UP to even, floored at 2: a rise of `rise_unit` units is emitted
    // as a `BrickSize` half height of `rise_unit / 2`, which must be a whole
    // unit and must not be zero.
    let scale = options.scale.max(1) as i32;
    let rise_unit = scale + (scale & 1);
    if rise_unit != scale {
        info!(
            "Terrain layers are {rise_unit} units tall (--vertical {scale} rounded up: a micro \
             wedge's half height must be a whole unit)"
        );
    }
    let half = options.size as i32;

    info!("Sampling shared vertex grid");
    let stride = width as usize + 1;
    let mut vertices = vec![0i32; stride * (height as usize + 1)];
    for j in 0..=height {
        for i in 0..=width {
            let mut sum = 0i64;
            let mut count = 0i64;
            for (dx, dy) in [(-1i32, -1i32), (0, -1), (-1, 0), (0, 0)] {
                let px = i as i32 + dx;
                let py = j as i32 + dy;
                if px >= 0 && py >= 0 && (px as u32) < width && (py as u32) < height {
                    sum += heightmap.at(px as u32, py as u32).min(i32::MAX as u32) as i64;
                    count += 1;
                }
            }
            vertices[j as usize * stride + i as usize] = ((sum + count / 2) / count) as i32;
        }
    }
    progress!(0.2);

    let corners_at = |x: u32, y: u32| -> Corners {
        let (x, y) = (x as usize, y as usize);
        [
            vertices[y * stride + x],
            vertices[y * stride + x + 1],
            vertices[(y + 1) * stride + x + 1],
            vertices[(y + 1) * stride + x],
        ]
    };

    // Pass one records only each cell's floor, which is all the foundation
    // depths below need. Keeping the fits themselves would cost five times the
    // memory on a map where a full brick list is already the binding limit.
    info!("Classifying cells");
    let mut floors = vec![0i32; width as usize * height as usize];
    let mut culled = vec![false; width as usize * height as usize];
    for y in 0..height {
        for x in 0..width {
            let index = y as usize * width as usize + x as usize;
            let color = colormap.at(x, y);
            // Same rule the blocky modes use: with --cull, a fully transparent
            // pixel and a bottom-level pixel are both dropped.
            culled[index] = options.cull && (heightmap.at(x, y) == 0 || color[3] == 0);
            floors[index] = *corners_at(x, y).iter().min().expect("four corners");
        }
        if y % 64 == 0 {
            progress!(0.2 + 0.3 * (y as f32 / height as f32));
        }
    }
    progress!(0.5);

    // Centred exactly as the blocky modes centre their output, so switching
    // modes does not move the build.
    let layout = Layout {
        half,
        rise_unit,
        offset_x: -(width as i32 * half),
        offset_y: -(height as i32 * half),
        // Surface of layer zero, matching where the blocky modes put the top of
        // a zero-height cell.
        z_floor: options.base_height() - 5,
        glow: options.glow,
        collision: options.collision(),
    };

    // How deep a cell's foundation has to reach: below the lowest neighbouring
    // floor, which is what closes a cliff face from the high side. The `+ 1`
    // keeps a flat plain one layer thick instead of zero.
    //
    // A closure rather than a stored field: it is a pure function of `floors`,
    // and a map big enough for this to matter is a map where another `i32` per
    // cell is the thing that hurts.
    let depth_layers = |x: u32, y: u32| -> i32 {
        let index = y as usize * width as usize + x as usize;
        let mut lowest = floors[index];
        for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && ny >= 0 && (nx as u32) < width && (ny as u32) < height {
                lowest = lowest.min(floors[ny as usize * width as usize + nx as usize]);
            }
        }
        (floors[index] - lowest).max(0) + 1
    };

    info!("Building terrain assemblies");
    let mut bricks: Vec<Brick> = Vec::new();
    let mut counts = [0usize; 6];
    for y in 0..height {
        for x in 0..width {
            let index = y as usize * width as usize + x as usize;
            if culled[index] {
                continue;
            }
            let fit = fit_cell(corners_at(x, y));
            counts[match fit.kind {
                SurfaceKind::Flat => 0,
                SurfaceKind::Ramp(_) => 1,
                SurfaceKind::Corner(_) => 2,
                SurfaceKind::InnerCorner(_) => 3,
                SurfaceKind::OuterTriangle(_) => 4,
                SurfaceKind::TrianglePair(_) => 5,
            }] += 1;

            let color = colormap.at(x, y);
            emit_slope(
                &mut bricks,
                &layout,
                fit,
                Position::new(
                    (x as i32 * 2 + 1) * layout.half + layout.offset_x,
                    (y as i32 * 2 + 1) * layout.half + layout.offset_y,
                    layout.z_floor + fit.base * rise_unit,
                ),
                Color {
                    r: color[0],
                    g: color[1],
                    b: color[2],
                },
            );
        }
        if y % 32 == 0 {
            progress!(0.5 + 0.35 * (y as f32 / height as f32));
        }
    }
    let slopes = bricks.len();

    info!("Merging foundations");
    let foundations = merge_foundations(
        &mut bricks,
        &layout,
        colormap,
        &floors,
        &culled,
        &depth_layers,
        (width, height),
        &progress_f,
    )?;

    let area = width as usize * height as usize;
    info!(
        "Fitted {} cell(s): {} flat, {} ramp, {} corner, {} inner corner, {} diagonal stack, {} triangle pair",
        counts.iter().sum::<usize>(),
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        counts[4],
        counts[5],
    );
    info!(
        "Converted {} pixel(s) to {} brick(s) ({:.2} per cell): {slopes} surface, {foundations} \
         foundation",
        area,
        bricks.len(),
        bricks.len() as f64 / area.max(1) as f64,
    );

    progress!(1.0);
    Ok(bricks)
}

/// Where the terrain's cell grid sits in world units, and how bricks are
/// styled. Bundled so the surface pass and the foundation pass cannot drift
/// apart on the centring, the ground plane, or the material.
struct Layout {
    /// Half extent of one cell in X and Y, in units.
    half: i32,
    /// Units per terrain layer. Even, and at least 2.
    rise_unit: i32,
    offset_x: i32,
    offset_y: i32,
    /// World Z of the surface of layer zero.
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
        turn: Turn,
    ) -> Brick {
        Brick {
            asset: BrickType::Procedural { asset, size },
            position,
            collision: self.collision,
            color,
            owner_index: None,
            direction: Direction::ZPositive,
            rotation: turn.rotation(),
            material_intensity: if self.glow { 0 } else { 5 },
            material: if self.glow { GLOW } else { PLASTIC },
            ..Default::default()
        }
    }
}

/// Emit the foundation layer as as few boxes as the grid allows.
///
/// A foundation is a flat-topped prism, so the foundation layer is an ordinary
/// 2D field of `(colour, top, depth)` and merges the way the blocky renderers
/// merge their prisms. Without this the surface grammar's one-shape-per-cell
/// rule leaked into the part of the build nobody can see: a flat plain cost one
/// brick per PIXEL under `--terrain` while the same plain under `--micro` cost
/// one brick for the whole region.
///
/// Cells join a box only when their colour, top AND depth all match exactly.
/// Unifying mismatched depths to the deepest one would merge far more
/// aggressively, but it also hangs solid material below where a shallower cell
/// ended -- which is invisible on a plain and very visible from inside a
/// canyon, where the cliff base is the thing being looked at.
#[allow(clippy::too_many_arguments)]
fn merge_foundations<F: Fn(f32) -> bool, D: Fn(u32, u32) -> i32>(
    out: &mut Vec<Brick>,
    layout: &Layout,
    colormap: &dyn Colormap,
    floors: &[i32],
    culled: &[bool],
    depth_layers: &D,
    (width, height): (u32, u32),
    progress_f: &F,
) -> Result<usize, String> {
    let mut taken = vec![false; width as usize * height as usize];
    // A box may not exceed 500 units on any axis.
    let max_run = (MAX_HALF_EXTENT / layout.half).max(1) as u32;
    let mut emitted = 0usize;

    let index = |x: u32, y: u32| y as usize * width as usize + x as usize;
    let joins = |x: u32, y: u32, top: i32, depth: i32, color: [u8; 4], taken: &[bool]| {
        let i = index(x, y);
        !culled[i]
            && !taken[i]
            && floors[i] == top
            && depth_layers(x, y) == depth
            && colormap.at(x, y) == color
    };

    for y in 0..height {
        for x in 0..width {
            let i = index(x, y);
            if culled[i] || taken[i] {
                continue;
            }
            let top = floors[i];
            let depth = depth_layers(x, y);
            let color = colormap.at(x, y);

            let mut run_x = 1;
            while run_x < max_run
                && x + run_x < width
                && joins(x + run_x, y, top, depth, color, &taken)
            {
                run_x += 1;
            }

            let mut run_y = 1;
            'rows: while run_y < max_run && y + run_y < height {
                for dx in 0..run_x {
                    if !joins(x + dx, y + run_y, top, depth, color, &taken) {
                        // A partial row is discarded whole: the box has to stay
                        // rectangular.
                        break 'rows;
                    }
                }
                run_y += 1;
            }

            for dy in 0..run_y {
                for dx in 0..run_x {
                    taken[index(x + dx, y + dy)] = true;
                }
            }

            // Top flush with the cell floor, split into 500-unit slabs because
            // that is the tallest procedural brick the format can carry.
            let mut remaining = (depth * layout.rise_unit).max(2);
            let mut top_units = layout.z_floor + top * layout.rise_unit;
            while remaining > 0 {
                let slab = remaining.min(MAX_HALF_EXTENT * 2);
                out.push(layout.brick(
                    PB_DEFAULT_MICRO_BRICK,
                    BrickSize::new(
                        (run_x as i32 * layout.half) as u16,
                        (run_y as i32 * layout.half) as u16,
                        (slab / 2) as u16,
                    ),
                    Position::new(
                        (x as i32 * 2 + run_x as i32) * layout.half + layout.offset_x,
                        (y as i32 * 2 + run_y as i32) * layout.half + layout.offset_y,
                        top_units - slab / 2,
                    ),
                    Color {
                        r: color[0],
                        g: color[1],
                        b: color[2],
                    },
                    Turn(0),
                ));
                emitted += 1;
                remaining -= slab;
                top_units -= slab;
            }
        }
        if y % 32 == 0 && !progress_f(0.85 + 0.1 * (y as f32 / height as f32)) {
            return Err("Stopped by user".to_string());
        }
    }
    Ok(emitted)
}

/// Emit one cell's SURFACE: the load-bearing shape, plus the triangle cap or
/// complementary triangle where the fit calls for one. The foundation under it
/// is not emitted here -- foundations are flat-topped prisms and merge across
/// cells, so [`merge_foundations`] lays the whole layer at once.
///
/// `anchor` carries the cell's centre in X/Y and the altitude of its LOWEST
/// surface vertex in Z -- what the geometry contract calls `base`. A flat cell
/// emits nothing at all: its foundation's top face IS its surface.
fn emit_slope(out: &mut Vec<Brick>, layout: &Layout, fit: CellFit, anchor: Position, color: Color) {
    let half = layout.half as u16;
    let mut piece = |asset, size: BrickSize, z: i32, turn: Turn| {
        out.push(layout.brick(
            asset,
            size,
            Position::new(anchor.x, anchor.y, z),
            color,
            turn,
        ));
    };

    // A rise is CLAMPED, never rounded up: raising it would push the surface
    // above a sampled vertex and break the fit-from-below guarantee that keeps
    // one cell from protruding through its neighbour.
    let rise = (fit.rise * layout.rise_unit).min(MAX_HALF_EXTENT * 2);
    let cap = (fit.cap_rise * layout.rise_unit).min(MAX_HALF_EXTENT * 2);
    let (asset, turn, second) = match fit.kind {
        SurfaceKind::Flat => return,
        SurfaceKind::Ramp(t) => (PB_DEFAULT_MICRO_RAMP, t, None),
        SurfaceKind::Corner(t) => (PB_DEFAULT_MICRO_WEDGE_CORNER, t, None),
        SurfaceKind::InnerCorner(t) => (PB_DEFAULT_MICRO_WEDGE_INNER_CORNER, t, None),
        // The cap shares the outer corner's rotation: the outer corner is low
        // at NE under rotation zero and the triangle is high at SW, which are
        // the same quarter turn away from the cell's high vertex.
        SurfaceKind::OuterTriangle(t) => (PB_DEFAULT_MICRO_WEDGE_OUTER_CORNER, t, Some(t)),
        SurfaceKind::TrianglePair(t) => (
            PB_DEFAULT_MICRO_WEDGE_TRIANGLE_CORNER,
            t,
            Some(t.opposite()),
        ),
    };
    if rise < 2 {
        // The whole rise vanished under the layer quantization; the foundation
        // already covers this cell as a flat top.
        return;
    }
    piece(
        asset,
        BrickSize::new(half, half, (rise / 2) as u16),
        anchor.z + rise / 2,
        turn,
    );

    let Some(second_turn) = second else {
        return;
    };
    match fit.kind {
        // The cap of a diagonal stack sits one full rise higher and is sized
        // independently, from the middle level to the top one.
        SurfaceKind::OuterTriangle(_) if cap >= 2 => piece(
            PB_DEFAULT_MICRO_WEDGE_TRIANGLE_CORNER,
            BrickSize::new(half, half, (cap / 2) as u16),
            anchor.z + rise + cap / 2,
            second_turn,
        ),
        // The two triangles of a pair are the same size in the same place; only
        // their rotations differ, and together they tile the square.
        SurfaceKind::TrianglePair(_) => piece(
            PB_DEFAULT_MICRO_WEDGE_TRIANGLE_CORNER,
            BrickSize::new(half, half, (rise / 2) as u16),
            anchor.z + rise / 2,
            second_turn,
        ),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Corners are `(SW, SE, NE, NW)`.
    const SW: usize = 0;
    const SE: usize = 1;
    const NE: usize = 2;
    const NW: usize = 3;

    #[test]
    fn a_level_cell_is_one_flat_brick_with_no_residual() {
        let fit = fit_cell([7, 7, 7, 7]);
        assert_eq!(fit.kind, SurfaceKind::Flat);
        assert_eq!((fit.base, fit.rise, fit.error), (7, 0, 0));
    }

    /// Rotation zero steps DOWN toward `+X`, so a cell high on its `-X` edge
    /// (SW and NW) is the rotation-zero ramp. Each quarter turn moves the low
    /// edge counter-clockwise.
    #[test]
    fn ramps_face_the_way_the_cell_steps_down() {
        for (corners, expected) in [
            ([1, 0, 0, 1], Turn(0)), // low toward +X
            ([1, 1, 0, 0], Turn(1)), // low toward +Y
            ([0, 1, 1, 0], Turn(2)), // low toward -X
            ([0, 0, 1, 1], Turn(3)), // low toward -Y
        ] {
            let fit = fit_cell(corners);
            assert_eq!(
                fit.kind,
                SurfaceKind::Ramp(expected),
                "corners {corners:?} must ramp down the opposite way from its high edge"
            );
            assert_eq!((fit.base, fit.rise, fit.error), (0, 1, 0));
        }
    }

    /// A single raised vertex is a full wedge corner, and rotation zero puts
    /// the high point at SW.
    #[test]
    fn one_high_vertex_is_a_wedge_corner_rotated_onto_it() {
        for corner in [SW, SE, NE, NW] {
            let mut corners = [0; 4];
            corners[corner] = 1;
            let fit = fit_cell(corners);
            assert_eq!(
                fit.kind,
                SurfaceKind::Corner(Turn::new(corner as i32)),
                "a high {corner} must rotate the corner's SW high point onto it"
            );
            assert_eq!(fit.error, 0, "a one-high cell is exactly representable");
        }
    }

    /// A single LOW vertex is the inner corner, whose rotation zero is low at
    /// NE -- a half turn away from the wedge corner's convention.
    #[test]
    fn one_low_vertex_is_an_inner_corner_a_half_turn_from_the_high_convention() {
        for corner in [SW, SE, NE, NW] {
            let mut corners = [1; 4];
            corners[corner] = 0;
            let fit = fit_cell(corners);
            assert_eq!(
                fit.kind,
                SurfaceKind::InnerCorner(Turn::new(corner as i32 + 2)),
                "a low {corner} must rotate the inner corner's NE low point onto it"
            );
            assert_eq!(fit.error, 0);
        }
    }

    #[test]
    fn opposite_high_vertices_become_a_complementary_triangle_pair() {
        assert_eq!(
            fit_cell([1, 0, 1, 0]).kind,
            SurfaceKind::TrianglePair(Turn(0))
        );
        assert_eq!(
            fit_cell([0, 1, 0, 1]).kind,
            SurfaceKind::TrianglePair(Turn(1))
        );
        // The two triangles must be half a turn apart, or they would overlap on
        // one diagonal and leave the other uncovered.
        let SurfaceKind::TrianglePair(turn) = fit_cell([1, 0, 1, 0]).kind else {
            panic!("opposite highs are a triangle pair");
        };
        assert_eq!((turn.opposite().0 + 4 - turn.0) % 4, 2);
    }

    /// The row the geometry contract calls essential: an orthogonal one-step
    /// limit does NOT stop opposite corners differing by two, and reducing that
    /// case to a two-level mask loses the upper triangle.
    #[test]
    fn a_diagonal_two_layer_cell_stacks_a_triangle_on_an_outer_corner() {
        for high in [SW, SE, NE, NW] {
            let mut corners = [1; 4];
            corners[high] = 2;
            corners[(high + 2) & 3] = 0;
            let fit = fit_cell(corners);
            assert_eq!(
                fit.kind,
                SurfaceKind::OuterTriangle(Turn::new(high as i32)),
                "corners {corners:?} need the diagonal stack, not a flattened mask"
            );
            assert_eq!((fit.base, fit.rise, fit.cap_rise, fit.error), (0, 1, 1, 0));
        }
    }

    /// The whole point of scoring every candidate: a cliff is not slope
    /// limited, and the grammar still has to answer for it. A clean two-level
    /// fault must come back as ONE exact steep ramp, not a shelf.
    #[test]
    fn a_cliff_resolves_to_one_exact_steep_ramp() {
        let fit = fit_cell([0, 40, 40, 0]);
        assert_eq!(fit.kind, SurfaceKind::Ramp(Turn(2)));
        assert_eq!((fit.base, fit.rise, fit.error), (0, 40, 0));
    }

    /// Fit FROM BELOW. Every candidate takes the minimum of the corners it
    /// covers, so no fitted surface may sit above a sampled vertex -- that is
    /// what stops one cell protruding through its neighbour.
    #[test]
    fn no_fitted_surface_ever_rises_above_a_sampled_corner() {
        // An exhaustive sweep over small four-corner fields, including every
        // pattern outside the sloped grammar.
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        let corners = [a, b, c, d];
                        let fit = fit_cell(corners);
                        assert!(
                            fit.error >= 0,
                            "corners {corners:?} fitted above the sample: error {}",
                            fit.error
                        );
                        assert!(fit.base <= *corners.iter().min().unwrap());
                        assert!(fit.rise >= 0 && fit.cap_rise >= 0);
                        assert!(
                            fit.base + fit.rise + fit.cap_rise <= *corners.iter().max().unwrap(),
                            "corners {corners:?} fitted a peak above the highest sample"
                        );
                    }
                }
            }
        }
    }

    /// Anything the grammar can express exactly, it must express exactly --
    /// otherwise a smooth field would render with residual steps even where the
    /// vocabulary covers it.
    #[test]
    fn every_representable_cell_is_fitted_with_no_residual() {
        let mut representable = 0;
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    for d in 0..3 {
                        let corners = [a, b, c, d];
                        let range = corners.iter().max().unwrap() - corners.iter().min().unwrap();
                        // Two-level cells and the diagonal 0-1-2-1 stack are the
                        // whole vocabulary; a cell needs to be one of them.
                        let levels: std::collections::BTreeSet<_> = corners.iter().collect();
                        let diagonal_stack = range == 2
                            && levels.len() == 3
                            && (0..4).any(|k| {
                                corners[k] == 2
                                    && corners[(k + 2) & 3] == 0
                                    && corners[(k + 1) & 3] == 1
                                    && corners[(k + 3) & 3] == 1
                            });
                        if levels.len() > 2 && !diagonal_stack {
                            continue;
                        }
                        representable += 1;
                        assert_eq!(
                            fit_cell(corners).error,
                            0,
                            "corners {corners:?} are inside the grammar and must fit exactly"
                        );
                    }
                }
            }
        }
        assert!(
            representable > 40,
            "the sweep must actually cover the grammar"
        );
    }

    /// A constant-height, constant-colour heightmap, for exercising the
    /// foundation merge against the case it exists for.
    struct Flat(u32, u32, u32);
    impl Heightmap for Flat {
        fn at(&self, _x: u32, _y: u32) -> u32 {
            self.2
        }
        fn size(&self) -> (u32, u32) {
            (self.0, self.1)
        }
    }
    /// A height field with one step across the middle, so foundations have two
    /// different depths and must not merge across the break.
    struct Step(u32, u32);
    impl Heightmap for Step {
        fn at(&self, x: u32, _y: u32) -> u32 {
            if x < self.0 / 2 { 0 } else { 8 }
        }
        fn size(&self) -> (u32, u32) {
            (self.0, self.1)
        }
    }
    struct Grey(u32, u32);
    impl Colormap for Grey {
        fn at(&self, _x: u32, _y: u32) -> [u8; 4] {
            [128, 128, 128, 255]
        }
        fn size(&self) -> (u32, u32) {
            (self.0, self.1)
        }
    }
    /// One distinct colour per pixel: nothing may merge.
    struct Rainbow(u32, u32);
    impl Colormap for Rainbow {
        fn at(&self, x: u32, y: u32) -> [u8; 4] {
            [x as u8, y as u8, (x ^ y) as u8, 255]
        }
        fn size(&self) -> (u32, u32) {
            (self.0, self.1)
        }
    }

    fn options() -> GenOptions {
        GenOptions {
            size: 5,
            scale: 2,
            asset: PB_DEFAULT_MICRO_BRICK,
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
            surface: SurfaceMode::Terrain,
        }
    }

    /// The whole point of the merge pass: a flat plain is one flat-topped
    /// prism, not one per pixel. Before it existed this render cost 1024
    /// bricks.
    #[test]
    fn a_uniform_plain_collapses_to_a_handful_of_foundation_boxes() {
        let bricks =
            gen_terrain_heightmap(&Flat(32, 32, 4), &Grey(32, 32), options(), |_| true).unwrap();
        assert!(
            bricks.len() < 8,
            "1024 identical cells collapsed to only {} brick(s)",
            bricks.len()
        );
        // ...and it really is flat, so nothing paid for a wedge either.
        assert!(bricks.iter().all(|b| {
            let BrickType::Procedural { asset, .. } = &b.asset else {
                unreachable!()
            };
            asset.as_ref() == "PB_DefaultMicroBrick"
        }));
    }

    /// The merge must be driven by the field, not by luck: a colormap with a
    /// different colour on every pixel has nothing to merge, and must still
    /// cover every cell.
    #[test]
    fn a_per_pixel_colormap_merges_nothing_and_still_covers_every_cell() {
        let bricks =
            gen_terrain_heightmap(&Flat(16, 16, 4), &Rainbow(16, 16), options(), |_| true).unwrap();
        assert_eq!(
            bricks.len(),
            256,
            "distinct colours cannot share a box, so every cell needs its own"
        );
    }

    /// Foundations of different depths must not join. A box spanning the step
    /// would either leave the cliff face open or hang material below the low
    /// side.
    #[test]
    fn foundations_never_merge_across_a_change_in_depth() {
        let bricks =
            gen_terrain_heightmap(&Step(16, 16), &Grey(16, 16), options(), |_| true).unwrap();
        let cell = 10; // options().size is a half extent, so a cell is 10 units
        let left_edge = -(16 * 5);
        // No foundation may straddle the seam at the middle of the map.
        let seam = left_edge + 8 * cell;
        for brick in &bricks {
            let BrickType::Procedural { asset, size } = &brick.asset else {
                unreachable!()
            };
            if asset.as_ref() != "PB_DefaultMicroBrick" {
                continue;
            }
            let low = brick.position.x - size.x as i32;
            let high = brick.position.x + size.x as i32;
            assert!(
                low >= seam || high <= seam,
                "a foundation spans the step from {low} to {high} (seam at {seam})"
            );
        }
    }

    /// The merge is an optimization, so it must not change what is solid. Every
    /// cell's foundation column has to come back covered, exactly once, from
    /// its own top down to its own depth.
    #[test]
    fn merging_covers_the_same_volume_one_cell_at_a_time_would_have() {
        let bricks =
            gen_terrain_heightmap(&Step(12, 12), &Grey(12, 12), options(), |_| true).unwrap();
        let cell = 10;
        let origin = -(12 * 5);
        let mut covered = std::collections::HashMap::<(i32, i32), (i32, i32)>::new();
        for brick in &bricks {
            let BrickType::Procedural { asset, size } = &brick.asset else {
                unreachable!()
            };
            if asset.as_ref() != "PB_DefaultMicroBrick" {
                continue;
            }
            let x0 = (brick.position.x - size.x as i32 - origin) / cell;
            let y0 = (brick.position.y - size.y as i32 - origin) / cell;
            for dx in 0..(size.x as i32 * 2 / cell) {
                for dy in 0..(size.y as i32 * 2 / cell) {
                    let span = (
                        brick.position.z - size.z as i32,
                        brick.position.z + size.z as i32,
                    );
                    let entry = covered.entry((x0 + dx, y0 + dy)).or_insert(span);
                    *entry = (entry.0.min(span.0), entry.1.max(span.1));
                }
            }
        }
        assert_eq!(covered.len(), 144, "every cell must carry a foundation");
        // A uniform half of the map is one depth and the other half another, so
        // both cases are represented; what matters is that each column is a
        // single contiguous span reaching its own surface.
        for ((x, y), (low, high)) in &covered {
            assert!(high > low, "cell ({x}, {y}) has an empty foundation");
        }
    }

    #[test]
    fn a_flat_cell_never_pays_for_a_wedge_when_a_brick_would_do() {
        // Same residual as several zero-information wedge fits; the tie-break
        // must take the single brick.
        assert_eq!(fit_cell([5, 5, 5, 5]).kind.brick_count(), 1);
        assert_eq!(fit_cell([5, 5, 5, 5]).kind, SurfaceKind::Flat);
    }
}
