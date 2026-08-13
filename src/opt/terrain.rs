//! Smooth terrain from micro bricks.
//!
//! The other renderers give each pixel a flat top, so a slope becomes a
//! staircase. This renderer gives each pixel a SLOPED top. It selects that top
//! from the Brickadia micro wedge family. A smooth height field then becomes a
//! smooth surface.
//!
//! The geometry rules come from the `terrain-experiment` prefab generator
//! (`docs/MICRO_TERRAIN_GEOMETRY.md`). They are measured values: a GLB export
//! from the game was compared with the same `.brz` file read through BRDB.
//! This module uses these rules:
//!
//! * Brickadia uses 10 position units for one stud. The `BrickSize` values of
//!   a procedural brick are HALF extents. A shape with `z = k` is `2k` units
//!   high, so each rise must be an even number of units.
//! * The heights come from a grid of SHARED vertices, not from each pixel. Two
//!   adjacent cells use the same vertex, so their surfaces touch exactly.
//! * The table below applies at `Direction::ZPositive` and `Rotation::Deg0`.
//!   Read the corners in the sequence `(SW, SE, NE, NW)`. That sequence is
//!   `(-X,-Y)`, `(+X,-Y)`, `(+X,+Y)`, `(-X,+Y)`.
//!
//!   | asset | surface |
//!   | --- | --- |
//!   | `PB_DefaultMicroBrick` | `(1,1,1,1)`, flat top |
//!   | `PB_DefaultMicroRamp` | low at `+X` |
//!   | `PB_DefaultMicroWedgeCorner` | high vertex at SW |
//!   | `PB_DefaultMicroWedgeInnerCorner` | low vertex at NE |
//!   | `PB_DefaultMicroWedgeOuterCorner` | low vertex at NE |
//!   | `PB_DefaultMicroWedgeTriangleCorner` | SW triangle, high at SW |
//!
//! * The center of a flat micro brick is at `base - half_height`, so its top
//!   is at `base`. The center of a sloped shape is at `base + half_rise`. Its
//!   low vertices are then at `base` and its high vertices at `base + rise`.
//!
//! EACH cell uses the same best-fit selection. `terrain-experiment` uses that
//! selection for "wall cells" only and uses a simple match for the other
//! cells. This module has one path, because a heightmap has no slope limit. A
//! cliff, a spike of one pixel and a smooth dune all supply four corner
//! heights. The fit examines each shape that the rules permit. It gives each
//! shape a score from the difference to those four heights, then keeps the
//! best shape.
//!
//! The fit always stays BELOW the sample. Each candidate uses the MINIMUM of
//! the corners that it covers, not their mean. The selected surface is thus at
//! or below each sampled vertex, and one cell can never go through the
//! adjacent cell. A clean cliff of two levels becomes one correct steep ramp.
//! That ramp touches the low ground and the high ground exactly.

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

/// The largest half extent that a procedural brick can hold. It is a piece of
/// 500 units. The code divides a higher foundation into more than one piece.
/// It decreases a higher slope. It never increases a rise, because a higher
/// surface would go above a sampled vertex.
const MAX_HALF_EXTENT: i32 = 250;

/// The four corner heights of one cell. The sequence is `(SW, SE, NE, NW)`,
/// which is the sequence used for all the calibrated shapes.
pub type Corners = [i32; 4];

/// The number of quarter turns counterclockwise around `+Z`.
///
/// The code compares shapes frequently. It gives each candidate a score, and a
/// test holds each rotation rule below. But `brdb::Rotation` has no
/// `PartialEq`. This type adds that comparison. The code changes it into a
/// `Rotation` when it makes the brick.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Turn(pub u8);

impl Turn {
    fn new(turns: i32) -> Self {
        Self(turns.rem_euclid(4) as u8)
    }

    /// The half turn that goes with this turn. A
    /// [`SurfaceKind::TrianglePair`] uses the two turns for its two
    /// triangles.
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

/// The set of bricks that a cell becomes. The `Turn` already agrees with the
/// Brickadia rules. The table in the module description shows what rotation
/// zero means for each asset.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SurfaceKind {
    /// One flat micro brick. Its top is at `base`.
    Flat,
    /// `PB_DefaultMicroRamp`. Two adjacent corners are high.
    Ramp(Turn),
    /// `PB_DefaultMicroWedgeCorner`. One corner only is high.
    Corner(Turn),
    /// `PB_DefaultMicroWedgeInnerCorner`. One corner only is low.
    InnerCorner(Turn),
    /// `PB_DefaultMicroWedgeOuterCorner` with a `...TriangleCorner` one rise
    /// above it. This is the diagonal `0,1,2,1` stack. A mask of two levels
    /// loses the upper triangle and leaves a step that you can see.
    OuterTriangle(Turn),
    /// Two `PB_DefaultMicroWedgeTriangleCorner` bricks that fill the square.
    /// The two opposite corners are high.
    TrianglePair(Turn),
}

impl SurfaceKind {
    /// The number of bricks in this set. It does not count the foundation.
    /// Two candidates can have the same score. The code then keeps the
    /// candidate with fewer bricks. A cell that one flat brick fits exactly
    /// thus never gets a wedge.
    pub fn brick_count(self) -> u32 {
        match self {
            SurfaceKind::Flat => 1,
            SurfaceKind::Ramp(_) | SurfaceKind::Corner(_) | SurfaceKind::InnerCorner(_) => 1,
            SurfaceKind::OuterTriangle(_) | SurfaceKind::TrianglePair(_) => 2,
        }
    }
}

/// The selected set of bricks for one cell, with the heights that it fits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CellFit {
    pub kind: SurfaceKind,
    /// The height of the LOWEST surface vertex, in layers.
    pub base: i32,
    /// The rise of the main shape above `base`, in layers. It is zero for
    /// [`SurfaceKind::Flat`].
    pub rise: i32,
    /// The rise of the top triangle above `base + rise`, in layers. Only
    /// [`SurfaceKind::OuterTriangle`] uses it.
    pub cap_rise: i32,
    /// The sum of `sampled - fitted` for the four corners, in layers. It is
    /// never negative, because each candidate stays below the sample.
    pub error: i32,
}

/// The turn that puts the HIGH vertex of a shape at `corner`. Rotation zero is
/// high at SW for `MicroWedgeCorner` and for `MicroWedgeTriangleCorner`. The
/// result is therefore direct: `SW/SE/NE/NW` gives `0/90/180/270`.
fn turn_from_high_corner(corner: usize) -> Turn {
    Turn::new(corner as i32)
}

/// The turn that puts the LOW vertex of a shape at `corner`. Rotation zero is
/// low at NE for the inner corner and for the outer corner. The result is thus
/// a half turn from [`turn_from_high_corner`]: a low `SW/SE/NE/NW` gives
/// `180/270/0/90`.
fn turn_from_low_corner(corner: usize) -> Turn {
    Turn::new(corner as i32 + 2)
}

/// Select the set of bricks whose surface agrees best with four sampled corner
/// heights.
///
/// The code examines each shape that the rules permit: the flat shape, the
/// fourteen divisions into a high group and a low group (these give the ramp,
/// the corner, the inner corner and the triangle pair), and the four diagonal
/// `0,1,2,1` stacks. It gives each candidate a score from the height that the
/// candidate does not fill. Each candidate stays BELOW the sample, so each
/// score is a sum of values that are not negative, and the selected surface
/// never goes above a sampled vertex.
///
/// If two candidates have the same score, the code keeps the candidate with
/// fewer bricks. A flat area thus stays flat and does not get wedges with no
/// rise.
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

    // The fourteen divisions into two levels. If bit `i` is set, corner `i` is
    // on the high level. The masks 0 and 15 are the flat shape, which the code
    // above already put in `best`.
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
            // The "high" level is not above the low level. This division thus
            // gives the flat surface, which the code already examined.
            continue;
        }
        let error = (0..4)
            .map(|i| corners[i] - if high(i) { low_level + rise } else { low_level })
            .sum();
        let kind = match mask {
            // One high vertex. This is a convex shoulder.
            1 | 2 | 4 | 8 => {
                SurfaceKind::Corner(turn_from_high_corner(mask.trailing_zeros() as usize))
            }
            // One low vertex. This is the concave change. The OUTER corner has
            // the same edge heights but the opposite internal triangles. The
            // code keeps it for the diagonal stack below, where its top
            // triangle completes the surface.
            7 | 11 | 13 | 14 => SurfaceKind::InnerCorner(turn_from_low_corner(
                (!mask & 15).trailing_zeros() as usize,
            )),
            // Two opposite corners are high. Two triangles then fill the
            // square and keep all four sampled corners.
            5 => SurfaceKind::TrianglePair(Turn(0)),
            10 => SurfaceKind::TrianglePair(Turn(1)),
            // Two adjacent corners are high. This is a ramp along an axis.
            // Its name shows the direction in which it goes DOWN. Rotation
            // zero is low at +X.
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

    // The four diagonal stacks. One corner is two rises high, the opposite
    // corner is at the bottom, and the other two corners are between them. A
    // limit of one step between adjacent vertices does not prevent this shape.
    // A mask of two levels gives the incorrect changes that occur in the
    // game.
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
            // The bottom corner and the top corner agree exactly. Only the
            // two middle corners can differ, by the distance between them.
            error: (corners[(high + 1) & 3] - middle) + (corners[(high + 3) & 3] - middle),
        });
    }

    best
}

/// Make smooth micro wedge terrain from a heightmap.
///
/// The code reads the heights from a shared grid of `(w+1) x (h+1)` vertices.
/// Vertex `(i, j)` is the mean of the four pixels that touch it, or of fewer
/// pixels at an edge. The output keeps one cell for each pixel, which is the
/// same area that the other modes give. But adjacent cells then get the shared
/// corner heights that the shapes need. The cost is one smoothing pass, which
/// is usually good for a heightmap.
///
/// With `--terrain`, the vertical scale gives the height of one LAYER, not a
/// count of units. One level of grey is one layer of `rise_unit` units. The
/// code increases `rise_unit` to an even number and to a minimum of 2, because
/// the half height in a `BrickSize` must be a whole unit.
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
    // A zero cell size would make `layout.half` zero, which divides by zero in
    // `merge_foundations` (`MAX_HALF_EXTENT / layout.half`) and would emit
    // zero-extent bricks besides. Refuse it like the other invalid inputs above.
    if options.size == 0 {
        return Err("Brick size must be at least 1".to_string());
    }

    // Increased to an even number, with a minimum of 2. A rise of `rise_unit`
    // units becomes a `BrickSize` half height of `rise_unit / 2`. That value
    // must be a whole unit and must not be zero.
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

    // This pass keeps the floor of each cell only, because the foundation
    // depths below need no more. To keep the selected shapes would use five
    // times more memory, and on a large map the list of bricks is already the
    // limit.
    info!("Classifying cells");
    let mut floors = vec![0i32; width as usize * height as usize];
    let mut culled = vec![false; width as usize * height as usize];
    for y in 0..height {
        for x in 0..width {
            let index = y as usize * width as usize + x as usize;
            let color = colormap.at(x, y);
            // This is the rule that the other modes use. With --cull, the
            // code removes a fully transparent pixel and a pixel at the lowest
            // level.
            culled[index] = options.cull && (heightmap.at(x, y) == 0 || color[3] == 0);
            floors[index] = *corners_at(x, y).iter().min().expect("four corners");
        }
        if y % 64 == 0 {
            progress!(0.2 + 0.3 * (y as f32 / height as f32));
        }
    }
    progress!(0.5);

    // The center agrees with the center that the other modes give, so a
    // change of mode does not move the build.
    let layout = Layout {
        half,
        rise_unit,
        offset_x: -(width as i32 * half),
        offset_y: -(height as i32 * half),
        // The surface of layer zero. The other modes put the top of a cell of
        // height zero at the same position.
        z_floor: options.base_height() - 5,
        glow: options.glow,
        collision: options.collision(),
    };

    // The depth that the foundation of a cell must reach. It goes below the
    // lowest floor of the adjacent cells, which closes the face of a cliff
    // from the high side. The `+ 1` gives a flat plain a thickness of one
    // layer and not of zero.
    //
    // This is a closure and not a stored array, because the result comes from
    // `floors` only. On a map that is large enough for the speed to be
    // important, one more `i32` for each cell is the larger problem.
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

/// The position of the cell grid in world units, and the appearance of the
/// bricks. These values stay together, so the surface pass and the foundation
/// pass always agree about the center, the ground level and the material.
struct Layout {
    /// The half extent of one cell in X and Y, in units.
    half: i32,
    /// The number of units in one terrain layer. It is even and is 2 or more.
    rise_unit: i32,
    offset_x: i32,
    offset_y: i32,
    /// The world Z of the surface of layer zero.
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

/// Make the foundation layer from as few boxes as the grid permits.
///
/// A foundation is a box with a flat top. The foundation layer is thus a
/// usual 2D field of `(color, top, depth)`, and it joins in the same way as
/// the boxes of the other modes. Before this pass, the rule of one shape for
/// each cell also applied to the part of the build that you cannot see: with
/// `--terrain` a flat plain used one brick for each PIXEL, but the same plain
/// with `--micro` used one brick for the full area.
///
/// Cells go into the same box only if their color, top AND depth all agree
/// exactly. To give a group the largest depth in that group would join many
/// more cells. But it would also put material below the bottom of a cell that
/// is less deep. You cannot see that material on a plain, but you see it
/// clearly in a canyon, where the bottom of the cliff is what you look at.
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
    // A box must not be more than 500 units on an axis.
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
                        // The code rejects the full row, because the box must
                        // stay rectangular.
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

            // The top is level with the cell floor. The code divides the box
            // into pieces of 500 units, because that is the highest procedural
            // brick that the format holds.
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

/// Make the SURFACE of one cell: the main shape, with the top triangle or the
/// second triangle if the selected shape needs one. This function does not
/// make the foundation below. Foundations are boxes with flat tops that join
/// across cells, so [`merge_foundations`] makes the full layer together.
///
/// `anchor` gives the center of the cell in X and Y, and the height of its
/// LOWEST surface vertex in Z. The geometry rules call that height `base`. A
/// flat cell makes no brick here, because the top face of its foundation IS
/// its surface.
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

    // The code DECREASES a rise that is too large. It never increases a rise,
    // because a higher surface would go above a sampled vertex. One cell would
    // then go through the adjacent cell.
    let rise = (fit.rise * layout.rise_unit).min(MAX_HALF_EXTENT * 2);
    let cap = (fit.cap_rise * layout.rise_unit).min(MAX_HALF_EXTENT * 2);
    let (asset, turn, second) = match fit.kind {
        SurfaceKind::Flat => return,
        SurfaceKind::Ramp(t) => (PB_DEFAULT_MICRO_RAMP, t, None),
        SurfaceKind::Corner(t) => (PB_DEFAULT_MICRO_WEDGE_CORNER, t, None),
        SurfaceKind::InnerCorner(t) => (PB_DEFAULT_MICRO_WEDGE_INNER_CORNER, t, None),
        // The top triangle uses the rotation of the outer corner. At rotation
        // zero the outer corner is low at NE and the triangle is high at SW.
        // Both are thus the same quarter turn from the high vertex of the
        // cell.
        SurfaceKind::OuterTriangle(t) => (PB_DEFAULT_MICRO_WEDGE_OUTER_CORNER, t, Some(t)),
        SurfaceKind::TrianglePair(t) => (
            PB_DEFAULT_MICRO_WEDGE_TRIANGLE_CORNER,
            t,
            Some(t.opposite()),
        ),
    };
    if rise < 2 {
        // The layer height removed the full rise. The foundation already gives
        // this cell a flat top.
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
        // The top triangle of a diagonal stack is one full rise higher. Its
        // height comes from the middle level to the top level.
        SurfaceKind::OuterTriangle(_) if cap >= 2 => piece(
            PB_DEFAULT_MICRO_WEDGE_TRIANGLE_CORNER,
            BrickSize::new(half, half, (cap / 2) as u16),
            anchor.z + rise + cap / 2,
            second_turn,
        ),
        // The two triangles of a pair have the same size and the same
        // position. Only their rotations differ, and together they fill the
        // square.
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
    use std::collections::HashMap;

    /// The corner sequence is `(SW, SE, NE, NW)`.
    const SW: usize = 0;
    const SE: usize = 1;
    const NE: usize = 2;
    const NW: usize = 3;

    /// A flat cell must become one flat brick and must lose no height. A wedge
    /// with no rise gives the same score, so this also holds the rule that
    /// selects the candidate with fewer bricks.
    #[test]
    fn a_level_cell_becomes_one_flat_brick() {
        let fit = fit_cell([7, 7, 7, 7]);
        assert_eq!(fit.kind, SurfaceKind::Flat);
        assert_eq!((fit.base, fit.rise, fit.error), (7, 0, 0));
        assert_eq!(fit.kind.brick_count(), 1);
    }

    /// Rotation zero goes DOWN at `+X`. A cell that is high on its `-X` edge
    /// (at SW and NW) is thus the ramp at rotation zero. Each quarter turn
    /// moves the low edge counterclockwise.
    #[test]
    fn ramps_face_the_way_the_cell_steps_down() {
        for (corners, expected) in [
            ([1, 0, 0, 1], Turn(0)), // low at +X
            ([1, 1, 0, 0], Turn(1)), // low at +Y
            ([0, 1, 1, 0], Turn(2)), // low at -X
            ([0, 0, 1, 1], Turn(3)), // low at -Y
        ] {
            let fit = fit_cell(corners);
            assert_eq!(
                fit.kind,
                SurfaceKind::Ramp(expected),
                "the corners {corners:?} must go down opposite to the high edge"
            );
            assert_eq!((fit.base, fit.rise, fit.error), (0, 1, 0));
        }
    }

    /// One high vertex must give a full wedge corner. Rotation zero puts the
    /// high point at SW.
    #[test]
    fn one_high_vertex_is_a_wedge_corner_rotated_onto_it() {
        for corner in [SW, SE, NE, NW] {
            let mut corners = [0; 4];
            corners[corner] = 1;
            let fit = fit_cell(corners);
            assert_eq!(
                fit.kind,
                SurfaceKind::Corner(Turn::new(corner as i32)),
                "a high corner {corner} must move the SW high point of the wedge onto it"
            );
            assert_eq!(fit.error, 0, "a cell with one high corner fits exactly");
        }
    }

    /// One LOW vertex must give the inner corner. Rotation zero is low at NE,
    /// which is a half turn from the rule for the wedge corner.
    #[test]
    fn one_low_vertex_is_an_inner_corner_a_half_turn_from_the_high_convention() {
        for corner in [SW, SE, NE, NW] {
            let mut corners = [1; 4];
            corners[corner] = 0;
            let fit = fit_cell(corners);
            assert_eq!(
                fit.kind,
                SurfaceKind::InnerCorner(Turn::new(corner as i32 + 2)),
                "a low corner {corner} must move the NE low point of the inner corner onto it"
            );
            assert_eq!(fit.error, 0);
        }
    }

    /// Two opposite high vertices must give two triangles. The two turns must
    /// be a half turn apart. If they were not, the triangles would cover one
    /// diagonal two times and leave the other diagonal open.
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
        let SurfaceKind::TrianglePair(turn) = fit_cell([1, 0, 1, 0]).kind else {
            panic!("two opposite high corners give a triangle pair");
        };
        assert_eq!((turn.opposite().0 + 4 - turn.0) % 4, 2);
    }

    /// The geometry rules call this shape necessary. A limit of one step
    /// between adjacent vertices does not prevent two opposite corners from
    /// differing by two levels, and a mask of two levels loses the upper
    /// triangle.
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
                "the corners {corners:?} need the diagonal stack, not a mask of two levels"
            );
            assert_eq!((fit.base, fit.rise, fit.cap_rise, fit.error), (0, 1, 1, 0));
        }
    }

    /// This is the reason to give each candidate a score. A heightmap has no
    /// slope limit, and the shapes must still answer for a cliff. A clean
    /// cliff of two levels must give ONE correct steep ramp and not a shelf.
    #[test]
    fn a_cliff_resolves_to_one_exact_steep_ramp() {
        let fit = fit_cell([0, 40, 40, 0]);
        assert_eq!(fit.kind, SurfaceKind::Ramp(Turn(2)));
        assert_eq!((fit.base, fit.rise, fit.error), (0, 40, 0));
    }

    /// The two rules that make cells touch, over each small four-corner field.
    ///
    /// The fit must stay at or below each sampled corner, because a higher
    /// surface would go through the adjacent cell. And where the shapes can
    /// give an exact answer, the fit must be exact. If it were not, a smooth
    /// field would show steps where the shapes cover it.
    #[test]
    fn the_fit_stays_below_the_sample_and_is_exact_where_the_shapes_allow() {
        let mut exact = 0;
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        let corners = [a, b, c, d];
                        let fit = fit_cell(corners);
                        let low = *corners.iter().min().unwrap();
                        let high = *corners.iter().max().unwrap();
                        assert!(
                            fit.error >= 0 && fit.base <= low,
                            "the corners {corners:?} fitted above the sample"
                        );
                        assert!(fit.rise >= 0 && fit.cap_rise >= 0);
                        assert!(
                            fit.base + fit.rise + fit.cap_rise <= high,
                            "the corners {corners:?} put a peak above the highest sample"
                        );

                        // A cell is inside the shapes if it has two levels, or
                        // if it is the diagonal 0-1-2-1 stack.
                        let levels: std::collections::BTreeSet<_> = corners.iter().collect();
                        let stack = high - low == 2
                            && levels.len() == 3
                            && (0..4).any(|k| {
                                corners[k] == high
                                    && corners[(k + 2) & 3] == low
                                    && corners[(k + 1) & 3] == low + 1
                                    && corners[(k + 3) & 3] == low + 1
                            });
                        if levels.len() <= 2 || stack {
                            exact += 1;
                            assert_eq!(
                                fit.error, 0,
                                "the corners {corners:?} are inside the shapes and must fit \
                                 exactly"
                            );
                        }
                    }
                }
            }
        }
        assert!(exact > 40, "the loop must cover the shapes");
    }

    /// A heightmap of one height and one color, to exercise the pass that
    /// joins the foundations.
    struct Flat(u32, u32, u32);
    impl Heightmap for Flat {
        fn at(&self, _x: u32, _y: u32) -> u32 {
            self.2
        }
        fn size(&self) -> (u32, u32) {
            (self.0, self.1)
        }
    }
    /// A height field with one step at the middle, so the foundations get two
    /// different depths.
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
    /// A different color for each pixel, so no two cells can join.
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
            nocollide: false,
            quadtree: true,
            greedy: false,
            surface: SurfaceMode::Terrain,
        }
    }

    fn foundations(bricks: &[Brick]) -> impl Iterator<Item = (&Brick, BrickSize)> {
        bricks.iter().filter_map(|brick| {
            let BrickType::Procedural { asset, size } = &brick.asset else {
                unreachable!("terrain makes procedural bricks only")
            };
            (asset.as_ref() == "PB_DefaultMicroBrick").then_some((brick, *size))
        })
    }

    /// The purpose of the pass that joins foundations. A flat plain must be a
    /// small number of boxes. Before that pass this render used 1024 bricks.
    #[test]
    fn a_uniform_plain_collapses_to_a_few_foundation_boxes() {
        let bricks =
            gen_terrain_heightmap(&Flat(32, 32, 4), &Grey(32, 32), options(), |_| true).unwrap();
        assert!(
            bricks.len() < 8,
            "1024 equal cells gave {} brick(s)",
            bricks.len()
        );
        assert_eq!(
            foundations(&bricks).count(),
            bricks.len(),
            "a flat plain must have no wedges"
        );
    }

    /// The field and not chance must control the joins. Cells join only if the
    /// color AND the depth agree.
    ///
    /// A box across a change of depth would open the face of the cliff or
    /// would put material below the bottom of the low cell.
    #[test]
    fn foundations_only_merge_where_color_and_depth_agree() {
        let bricks =
            gen_terrain_heightmap(&Flat(16, 16, 4), &Rainbow(16, 16), options(), |_| true).unwrap();
        assert_eq!(
            bricks.len(),
            256,
            "cells with different colors cannot share a box"
        );

        let bricks =
            gen_terrain_heightmap(&Step(16, 16), &Grey(16, 16), options(), |_| true).unwrap();
        // `options().size` is a half extent, so one cell is 10 units.
        let seam = -(16 * 5) + 8 * 10;
        for (brick, size) in foundations(&bricks) {
            let low = brick.position.x - size.x as i32;
            let high = brick.position.x + size.x as i32;
            assert!(
                low >= seam || high <= seam,
                "a foundation goes from {low} to {high} across the step at {seam}"
            );
        }
    }

    /// The joins are an optimization, so they must not change the material.
    /// Each cell must get one foundation column, one time only, and that
    /// column must reach the surface of its own cell.
    #[test]
    fn merging_covers_every_cell_exactly_once() {
        let (side, cell) = (12i32, 10i32);
        let bricks =
            gen_terrain_heightmap(&Step(12, 12), &Grey(12, 12), options(), |_| true).unwrap();
        let origin = -(side * 5);

        let mut columns = HashMap::<(i32, i32), Vec<(i32, i32)>>::new();
        for (brick, size) in foundations(&bricks) {
            let x0 = (brick.position.x - size.x as i32 - origin) / cell;
            let y0 = (brick.position.y - size.y as i32 - origin) / cell;
            let span = (
                brick.position.z - size.z as i32,
                brick.position.z + size.z as i32,
            );
            for dx in 0..(size.x as i32 * 2 / cell) {
                for dy in 0..(size.y as i32 * 2 / cell) {
                    columns.entry((x0 + dx, y0 + dy)).or_default().push(span);
                }
            }
        }
        assert_eq!(
            columns.len() as i32,
            side * side,
            "each cell must get a foundation"
        );

        for ((x, y), spans) in &columns {
            let mut spans = spans.clone();
            spans.sort_unstable();
            // The tops of a divided column must touch, with no gap and no
            // overlap. `Step` is not high enough to divide a column, but the
            // check must hold if it were.
            for pair in spans.windows(2) {
                assert_eq!(
                    pair[0].1, pair[1].0,
                    "the foundation of the cell ({x}, {y}) has a gap or an overlap"
                );
            }
            let (low, high) = (spans[0].0, spans.last().unwrap().1);
            assert!(high > low, "the cell ({x}, {y}) has an empty foundation");
        }
    }

    /// A zero brick size must be refused with an error, not a divide-by-zero
    /// panic in `merge_foundations` (`MAX_HALF_EXTENT / layout.half`).
    #[test]
    fn a_zero_brick_size_is_refused_not_a_panic() {
        let opts = GenOptions { size: 0, ..options() };
        match gen_terrain_heightmap(&Flat(4, 4, 2), &Grey(4, 4), opts, |_| true) {
            Err(e) => assert!(e.contains("size"), "unexpected error: {e}"),
            Ok(_) => panic!("a zero brick size must be refused"),
        }
    }
}
