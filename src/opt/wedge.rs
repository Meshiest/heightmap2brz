//! Brickadia wedge terrain: terraced flat tops with 45-degree chamfered
//! outlines, from full-size bricks and side wedges.
//!
//! The other sloped renderers approximate the height field. This one commits
//! to TERRACES: every height is a whole number of terrace steps, tops stay
//! flat everywhere, and the outlines of the terraces are what gets shaped.
//! Where a plateau has a convex corner -- two adjacent sides dropping to
//! lower ground -- the top of the column is cut by a vertical triangular
//! prism (`PB_DefaultSideWedge`) whose hypotenuse is a vertical face, so the
//! corner turns at 45 degrees. Concave corners get the mirror treatment: a
//! filler wedge tucked in so inner outlines turn at 45 degrees too. The
//! result is the "brick terrain" look of hand-built Brickadia maps rather
//! than a smooth surface.
//!
//! The rules are a port of the BRWorldSculptor sculpting tool
//! (`f1shar/br-terrain-gen`), with the sculpting left behind and the
//! heightmap pipeline kept. Stage for stage:
//!
//! 1. **Bake** -- one shade of grey is one terrace step of `--vertical`
//!    units, rounded to whole plates (a side wedge's height is in plates).
//! 2. **Erosion** -- remove the two configurations that cannot be chamfered:
//!    diagonal crossings (two cells touching only corner-to-corner, which
//!    force hard right angles) and spikes (cells with 3+ lower orthogonal
//!    sides). Every rule only LOWERS a cell to the height of a real
//!    neighbour, so the process is monotone and always reaches a fixpoint.
//! 3. **Classification** -- per cell: chamfer (exactly two ADJACENT lower
//!    sides and the diagonal between them lower), filler (the concave
//!    mirror), or flat.
//! 4. **Planning** -- passes that each claim cells the next cannot see:
//!    collinear 45-degree chamfers merge with the triangle of same-height
//!    cells behind them into one NxN wedge; lone corners get a seeded chance
//!    to stretch into a shallow 1xN wedge cutting into a wall; leftover 1x1
//!    chamfers and fillers; then a greedy box merge over the flat tops.
//!    (The sculpt tool has one more pass for maps whose minimum brick spans
//!    several cells; here one pixel is one cell, so it has nothing to do.)
//! 5. **Output** -- boxes become `PB_DefaultBrick`, cuts and fillers become
//!    `PB_DefaultSideWedge`. Box bottoms follow the surface as a shell one
//!    plate past the lowest ground they abut, except at the map border and
//!    against culled cells, where they drop to the floor so the outside of
//!    the build stays closed.
//!
//! Where the sculpt tool gates every merge on its rock/dirt material, this
//! port gates on the COLORMAP: a brick is one colour, so a piece that would
//! span two colours is declined, not recoloured, and falls through to a
//! finer pass. That is the same rule the flat-top optimizers already follow.
//!
//! The wedge rotation convention (`ROT_HYP`: rotation `r` points the
//! vertical hypotenuse at diagonal corner `r`, counterclockwise from +X/+Y)
//! was calibrated in-game by the sculpt tool over several rounds of feedback.
//! Grid axes and world axes map here exactly as they do there, so the
//! numbers carry over unchanged.

use crate::map::*;
use crate::util::*;
use brdb::{
    Brick, BrickSize, BrickType, Color, Direction, Position, Rotation,
    assets::{
        bricks::{PB_DEFAULT_BRICK, PB_DEFAULT_SIDE_WEDGE},
        materials::{GLOW, PLASTIC},
    },
};
use log::info;

/// The number of vertical units in one plate.
const CELL_UNITS: i32 = 4;
/// The largest half extent that a procedural brick can hold (500 units).
const MAX_HALF_EXTENT: i32 = 250;
/// The tallest piece one brick can be: half height `2 * plates` must stay
/// inside [`MAX_HALF_EXTENT`].
const MAX_PLATE_SPAN: i32 = MAX_HALF_EXTENT / 2;
/// The longest staircase run worth trying. The footprint cap below is what
/// usually binds; this only keeps the search bounded on huge maps.
const MERGE_RUN: i32 = 64;
/// How many box bottoms reach past the lowest ground they abut. Flush faces
/// make the bevel on a brick's bottom edge meet the bevel on its neighbour's
/// top edge, and the seam reads as a groove in what should be a flat wall.
const SKIRT: i32 = 1;

/// The chance a lone corner takes a shallow stretched cut instead of a 1x1.
/// The sculpt tool exposes this as a slider; its default is what ships here.
/// Randomising WHICH corners stretch (seeded, per absolute cell, so the same
/// map always builds the same) is what keeps long walls from looking punched
/// out by one repeated stamp.
const VARIETY: f32 = 0.5;
/// How far outlines are smoothed, 0..=1. Raises the minimum staircase run a
/// 45-degree face needs before it beats a shallow bevel, and lengthens the
/// shallow stretches. The sculpt tool's default.
const SMOOTHNESS: f32 = 0.3;
/// Seed for the variety hash. Fixed: the same input must give the same save.
const SEED: u32 = 0;

/// Side index i steps to SIDES[i]: 0=+X, 1=+Y, 2=-X, 3=-Y.
const SIDES: [(i32, i32); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];
/// Diagonal d is the corner between sides d and (d+1)%4.
const DIAGONALS: [(i32, i32); 4] = [(1, 1), (-1, 1), (-1, -1), (1, -1)];

const FLAT: u8 = 0;
const CHAMFER: u8 = 1;
const FILLER: u8 = 2;

/// Deterministic per-cell hash in `[0, 1)`, keyed on absolute cell
/// coordinates. The same xxHash-style mix the sculpt tool uses, so a given
/// seed picks the same corners there and here.
fn cell_hash(seed: u32, x: i32, y: i32) -> f32 {
    let mut h = seed ^ (x as u32).wrapping_mul(374761393) ^ (y as u32).wrapping_mul(668265263);
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h ^= h >> 16;
    h as f32 / 4294967296.0
}

/// Shortest collinear 45-degree run that may still claim its corners. On a
/// genuinely diagonal edge the merged NxN wedge is the smoothest answer; on a
/// CURVE it is a straight chord across it, so smoothness raises the bar and
/// shorter runs fall through to the shallow stretch instead.
fn min_diagonal_run(smoothness: f32) -> i32 {
    2 + (smoothness.clamp(0.0, 1.0) * 3.0).round() as i32
}

/// Lengths to try for a shallow corner stretch, longest first -- so a corner
/// takes the shallowest turn that actually fits and drops to a tighter one
/// only where the terrain will not hold the long one.
fn stretch_lengths(smoothness: f32) -> Vec<i32> {
    let longest = 1 + (smoothness.clamp(0.0, 1.0) * 7.0).round() as i32;
    (2..=longest).rev().collect()
}

/// Which side a stretched corner's cliff runs along, given the wall side `w`
/// it cuts into.
fn cliff_for_wall(d: usize, w: usize) -> usize {
    if w == (d + 2) % 4 { (d + 1) % 4 } else { d }
}

/// The height field, in plates. Reads off the edge clamp per axis, so the
/// map behaves as if its border row continued outward forever -- the same
/// convention every stage of the sculpt tool uses.
struct Field {
    cols: i32,
    rows: i32,
    h: Vec<i32>,
}

impl Field {
    #[inline]
    fn at(&self, x: i32, y: i32) -> i32 {
        self.h[(y * self.cols + x) as usize]
    }

    #[inline]
    fn get_clamped(&self, x: i32, y: i32) -> i32 {
        let cx = x.clamp(0, self.cols - 1);
        let cy = y.clamp(0, self.rows - 1);
        self.h[(cy * self.cols + cx) as usize]
    }
}

/// Count orthogonal neighbours that hold this cell up (are >= its height).
fn support(f: &Field, x: i32, y: i32) -> i32 {
    let h = f.at(x, y);
    let mut n = 0;
    for (dx, dy) in SIDES {
        if f.get_clamped(x + dx, y + dy) >= h {
            n += 1;
        }
    }
    n
}

/// Apply both erosion rules until nothing changes. Both only ever LOWER a
/// cell, and always to the height of a real neighbour, so no floor needs to
/// be clamped in and the loop always terminates.
fn erode(f: &mut Field, max_iterations: usize) {
    let cols = f.cols;
    let rows = f.rows;
    for _ in 0..max_iterations {
        let mut changed = false;

        // rule 1: diagonal crossings -- two cells touching only
        // corner-to-corner. Chamfering either corner would sever the
        // connection, so the less-supported cell of the raised pair is
        // lowered onto the saddle instead.
        for y in 0..rows - 1 {
            for x in 0..cols - 1 {
                let p00 = f.at(x, y);
                let p10 = f.at(x + 1, y);
                let p01 = f.at(x, y + 1);
                let p11 = f.at(x + 1, y + 1);
                if p00.min(p11) > p10.max(p01) {
                    let (ax, ay) = if support(f, x, y) > support(f, x + 1, y + 1) {
                        (x + 1, y + 1)
                    } else {
                        (x, y)
                    };
                    f.h[(ay * cols + ax) as usize] = p10.max(p01);
                    changed = true;
                } else if p10.min(p01) > p00.max(p11) {
                    let (bx, by) = if support(f, x + 1, y) > support(f, x, y + 1) {
                        (x, y + 1)
                    } else {
                        (x + 1, y)
                    };
                    f.h[(by * cols + bx) as usize] = p00.max(p11);
                    changed = true;
                }
            }
        }

        // rule 2: spikes and 1-wide tips -- 3+ lower orthogonal sides.
        // Iteration eats whole 1-wide protrusions back to the terrain body.
        for y in 0..rows {
            for x in 0..cols {
                let h = f.at(x, y);
                let mut lower = 0;
                let mut highest_lower = i32::MIN;
                for (dx, dy) in SIDES {
                    let n = f.get_clamped(x + dx, y + dy);
                    if n < h {
                        lower += 1;
                        highest_lower = highest_lower.max(n);
                    }
                }
                if lower >= 3 {
                    f.h[(y * cols + x) as usize] = highest_lower;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }
}

/// Is this cell a convex corner? Only the yes/no matters here -- used for the
/// reentrant-corner check inside [`classify`], where a filler beside a
/// chamfered cell would add a second parallel row of wedges along a diagonal
/// staircase.
fn is_chamfer(f: &Field, x: i32, y: i32) -> bool {
    let h = f.get_clamped(x, y);
    let l: Vec<bool> = SIDES
        .iter()
        .map(|(dx, dy)| f.get_clamped(x + dx, y + dy) < h)
        .collect();
    if l.iter().filter(|&&b| b).count() != 2 {
        return false;
    }
    // with exactly two lower sides there is at most one adjacent pair
    for d in 0..4 {
        if l[d] && l[(d + 1) % 4] {
            let (dx, dy) = DIAGONALS[d];
            return f.get_clamped(x + dx, y + dy) < h;
        }
    }
    false // the two lower sides are opposite: no corner
}

/// Per-cell classification into four parallel arrays. `value` carries a
/// chamfer's base (the cut goes from there up to the cell height) or a
/// filler's top (the fill goes from the cell height up to there), and is 0
/// for a flat cell.
///
/// A chamfer needs exactly two lower sides and a filler needs none, so the
/// two cases are mutually exclusive and the lower-side count alone decides
/// which (if either) can apply.
fn classify(f: &Field) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<i32>) {
    let n = (f.cols * f.rows) as usize;
    let mut kind = vec![FLAT; n];
    let mut diag = vec![0u8; n];
    let mut rot = vec![0u8; n];
    let mut val = vec![0i32; n];

    let mut i = 0usize;
    for y in 0..f.rows {
        for x in 0..f.cols {
            let h = f.at(x, y);
            let s: [i32; 4] = [
                f.get_clamped(x + 1, y),
                f.get_clamped(x, y + 1),
                f.get_clamped(x - 1, y),
                f.get_clamped(x, y - 1),
            ];
            // Flat interior, by far the commonest cell on any map.
            if s == [h; 4] {
                i += 1;
                continue;
            }

            let lower = s.iter().filter(|&&v| v < h).count();
            if lower == 2 {
                for d in 0..4 {
                    let e = (d + 1) % 4;
                    if s[d] < h && s[e] < h {
                        let (dx, dy) = DIAGONALS[d];
                        let dg = f.get_clamped(x + dx, y + dy);
                        if dg < h {
                            kind[i] = CHAMFER;
                            diag[i] = d as u8;
                            // ROT_HYP is the identity: rotation r points the
                            // hypotenuse at diagonal corner r.
                            rot[i] = d as u8;
                            // cut only above everything the corner touches
                            val[i] = s[d].max(s[e]).max(dg);
                        }
                        break;
                    }
                }
            } else if lower == 0 {
                let higher = s.iter().filter(|&&v| v > h).count();
                if higher == 2 {
                    for d in 0..4 {
                        let e = (d + 1) % 4;
                        if s[d] > h && s[e] > h {
                            let (dx, dy) = DIAGONALS[d];
                            let dg = f.get_clamped(x + dx, y + dy);
                            // never fill across a diagonal gap, and only fill
                            // true reentrant corners: along a diagonal
                            // staircase the upper cells' chamfers already
                            // form one straight 45-degree wall
                            if dg > h {
                                let (ax, ay) = SIDES[d];
                                let (bx, by) = SIDES[e];
                                if !is_chamfer(f, x + ax, y + ay)
                                    && !is_chamfer(f, x + bx, y + by)
                                {
                                    kind[i] = FILLER;
                                    diag[i] = d as u8;
                                    rot[i] = ((d + 2) % 4) as u8;
                                    val[i] = s[d].min(s[e]).min(dg);
                                }
                            }
                            break;
                        }
                    }
                }
            }
            i += 1;
        }
    }
    (kind, diag, rot, val)
}

/// One planned piece, in cells and plates. This is the sculpt tool's piece
/// contract with its writer, minus the sculpt-only fields.
#[derive(Clone, Debug)]
struct Piece {
    wedge: bool,
    x: i32,
    y: i32,
    sx: i32,
    sy: i32,
    from: i32,
    to: i32,
    rotation: u8,
    color: [u8; 4],
}

/// Everything the planning passes read. One struct so the pass functions
/// below stay callable without threading nine arguments through each.
struct Plan<'a> {
    f: &'a Field,
    kind: &'a [u8],
    diag: &'a [u8],
    rot: &'a [u8],
    val: &'a [i32],
    colors: &'a [[u8; 4]],
    culled: &'a [bool],
    /// Cells a wedge has claimed. Culled cells start claimed, so no pass
    /// puts a piece over them.
    consumed: Vec<bool>,
    /// Where each cell's flat box now tops out: the terrain height, lowered
    /// to a wedge's base wherever a cut claimed the cell.
    box_to: Vec<i32>,
    pieces: Vec<Piece>,
}

impl Plan<'_> {
    #[inline]
    fn idx(&self, x: i32, y: i32) -> usize {
        (y * self.f.cols + x) as usize
    }

    fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && x < self.f.cols && y >= 0 && y < self.f.rows
    }

    /// Is a rectangle all one colour, with nothing culled? Every merge that
    /// spans more than one cell has to pass this, because a brick is one
    /// colour -- a wedge half in one colour and half in another has no colour
    /// that is right. Failing is not an error: the merge is declined and the
    /// cells fall through to a finer pass, which bottoms out at 1x1 pieces
    /// that cannot straddle anything.
    ///
    /// Tested over the whole BOUNDING BOX, not just the cells the piece
    /// consumes: a wedge's face sweeps the box even where the prism itself
    /// is hollow, so the box is what you see.
    fn color_uniform(&self, x: i32, y: i32, w: i32, h: i32, c: [u8; 4]) -> bool {
        for yy in y..y + h {
            for xx in x..x + w {
                let i = self.idx(xx, yy);
                if self.culled[i] || self.colors[i] != c {
                    return false;
                }
            }
        }
        true
    }

    /// Cells of the run triangle STRICTLY behind the staircase boundary, for
    /// a run of length `n` whose first boundary cell is `(x0, y0)`.
    fn interior_cells(x0: i32, y0: i32, d: usize, n: i32) -> Vec<(i32, i32)> {
        let ys = if d % 2 == 0 { y0 - (n - 1) } else { y0 };
        let mut cells = Vec::new();
        for y in ys..ys + n {
            for x in x0..x0 + n {
                let interior = match d {
                    0 => x + y < x0 + y0,
                    2 => x + y > x0 + y0,
                    1 => x - y > x0 - y0,
                    _ => x - y < x0 - y0,
                };
                if interior {
                    cells.push((x, y));
                }
            }
        }
        cells
    }

    /// Is `(x, y)` an unconsumed chamfer continuing the run at `h`/`base`?
    fn matches_run(&self, x: i32, y: i32, d: usize, h: i32, base: i32) -> bool {
        if !self.in_bounds(x, y) {
            return false;
        }
        let i = self.idx(x, y);
        !self.consumed[i]
            && self.kind[i] == CHAMFER
            && self.diag[i] == d as u8
            && self.f.at(x, y) == h
            && self.val[i] == base
    }

    /// Is the triangle behind a run of `count` free, flat and all at `h`?
    fn run_valid(&self, x0: i32, y0: i32, d: usize, h: i32, count: i32) -> bool {
        for (x, y) in Self::interior_cells(x0, y0, d, count) {
            if !self.in_bounds(x, y) {
                return false;
            }
            let i = self.idx(x, y);
            if self.consumed[i] || self.f.at(x, y) != h || self.kind[i] != FLAT {
                return false;
            }
        }
        true
    }

    fn consume(&mut self, x: i32, y: i32, base: i32) {
        let i = self.idx(x, y);
        self.consumed[i] = true;
        self.box_to[i] = base;
    }

    /// Merge collinear staircase chamfers -- plus the triangle of same-height
    /// cells behind them -- into one NxN wedge with a single 45-degree face.
    /// Corners 0/2 lie on x+y=const lines (step +1,-1), corners 1/3 on
    /// x-y=const lines (step +1,+1).
    fn merge_staircases(&mut self, max_run: i32) {
        let min_run = min_diagonal_run(SMOOTHNESS);
        for y in 0..self.f.rows {
            for x in 0..self.f.cols {
                let i = self.idx(x, y);
                if self.kind[i] != CHAMFER || self.consumed[i] {
                    continue;
                }
                let d = self.diag[i] as usize;
                let h = self.f.at(x, y);
                let base = self.val[i];
                let step_y = if d % 2 == 0 { -1 } else { 1 };

                let (mut x0, mut y0) = (x, y);
                while self.matches_run(x0 - 1, y0 - step_y, d, h, base) {
                    x0 -= 1;
                    y0 -= step_y;
                }

                let mut count = 1;
                while count < max_run
                    && self.matches_run(x0 + count, y0 + count * step_y, d, h, base)
                    && self.run_valid(x0, y0, d, h, count + 1)
                {
                    count += 1;
                }
                if count < min_run {
                    continue;
                }
                let ry0 = if d % 2 == 0 { y0 - (count - 1) } else { y0 };
                let color = self.colors[i];
                if !self.color_uniform(x0, ry0, count, count, color) {
                    continue; // a finer pass will cut this corner smaller
                }

                self.pieces.push(Piece {
                    wedge: true,
                    x: x0,
                    y: ry0,
                    sx: count,
                    sy: count,
                    from: base,
                    to: h,
                    rotation: self.rot[i],
                    color,
                });
                for k in 0..count {
                    self.consume(x0 + k, y0 + k * step_y, base);
                }
                for (cx, cy) in Self::interior_cells(x0, y0, d, count) {
                    self.consume(cx, cy, base);
                }
            }
        }
    }

    /// Stretch lone corners into shallow 1xN wedges cutting into a wall, as
    /// far as the terrain stays flat and the cliff holds below. Seeded per
    /// absolute cell, so the same map always stretches the same corners.
    fn stretch_corners(&mut self) {
        let lengths = stretch_lengths(SMOOTHNESS);
        for y in 0..self.f.rows {
            for x in 0..self.f.cols {
                let i = self.idx(x, y);
                if self.kind[i] != CHAMFER || self.consumed[i] {
                    continue;
                }
                if cell_hash(SEED ^ 0x51de, x, y) >= VARIETY {
                    continue;
                }
                let d = self.diag[i] as usize;
                let h = self.f.at(x, y);

                let mut walls = [(d + 2) % 4, (d + 3) % 4];
                if cell_hash(SEED ^ 0xba5e, x, y) < 0.5 {
                    walls.reverse();
                }

                // LENGTHS outer, walls inner: both walls have to be offered
                // the long length before either is offered a short one, or a
                // corner that could sweep far one way settles for two cells
                // the other way and smoothness silently does nothing there.
                'corner: for &length in &lengths {
                    for &w in &walls {
                        let wd = SIDES[w];
                        let nd = SIDES[cliff_for_wall(d, w)];
                        let mut base = self.val[i];
                        let mut ok = true;
                        for k in 1..length {
                            let px = x + k * wd.0;
                            let py = y + k * wd.1;
                            if !self.in_bounds(px, py) {
                                ok = false;
                                break;
                            }
                            let pi = self.idx(px, py);
                            if self.consumed[pi]
                                || self.kind[pi] != FLAT
                                || self.f.at(px, py) != h
                            {
                                ok = false;
                                break;
                            }
                            let cliff_h = self.f.get_clamped(px + nd.0, py + nd.1);
                            if cliff_h >= h {
                                ok = false;
                                break;
                            }
                            base = base.max(cliff_h);
                        }
                        if !ok || base >= h {
                            continue;
                        }
                        let px0 = x.min(x + (length - 1) * wd.0);
                        let py0 = y.min(y + (length - 1) * wd.1);
                        let psx = if wd.0 != 0 { length } else { 1 };
                        let psy = if wd.1 != 0 { length } else { 1 };
                        let color = self.colors[i];
                        // lengths are tried longest first, so declining here
                        // lets a shorter sweep -- or the other wall -- still
                        // be offered
                        if !self.color_uniform(px0, py0, psx, psy, color) {
                            continue;
                        }

                        self.pieces.push(Piece {
                            wedge: true,
                            x: px0,
                            y: py0,
                            sx: psx,
                            sy: psy,
                            from: base,
                            to: h,
                            rotation: self.rot[i],
                            color,
                        });
                        for k in 0..length {
                            self.consume(x + k * wd.0, y + k * wd.1, base);
                        }
                        break 'corner;
                    }
                }
            }
        }
    }

    /// Remaining 1x1 chamfers, and fillers. A small cut is still better than
    /// the hard 90-degree corner that declining would leave. Fillers round
    /// concave corners, but only where the wall they fill against has not
    /// already been cut away -- otherwise they sit beside that cut at the
    /// same orientation and height, the jagged "double wedge".
    fn leftovers_and_fillers(&mut self) {
        for y in 0..self.f.rows {
            for x in 0..self.f.cols {
                let i = self.idx(x, y);
                if self.consumed[i] {
                    continue;
                }
                let h = self.f.at(x, y);
                if self.kind[i] == CHAMFER {
                    self.box_to[i] = self.val[i];
                    self.pieces.push(Piece {
                        wedge: true,
                        x,
                        y,
                        sx: 1,
                        sy: 1,
                        from: self.val[i],
                        to: h,
                        rotation: self.rot[i],
                        color: self.colors[i],
                    });
                } else if self.kind[i] == FILLER {
                    let fd = self.diag[i] as usize;
                    let mut cut = false;
                    for (dx, dy) in [SIDES[fd], SIDES[(fd + 1) % 4], DIAGONALS[fd]] {
                        let wx = x + dx;
                        let wy = y + dy;
                        if self.in_bounds(wx, wy) && self.consumed[self.idx(wx, wy)] {
                            cut = true;
                            break;
                        }
                    }
                    if !cut {
                        self.pieces.push(Piece {
                            wedge: true,
                            x,
                            y,
                            sx: 1,
                            sy: 1,
                            from: h,
                            to: self.val[i],
                            rotation: self.rot[i],
                            color: self.colors[i],
                        });
                    }
                }
            }
        }
    }

    /// How far a box has to reach down to leave no gap: to the LOWEST solid
    /// top it abuts, then [`SKIRT`] plates further. The neighbour's `box_to`
    /// is the right bound, not its terrain height -- a chamfered cell is
    /// solid only up to the base of its wedge, and stopping at the height
    /// would leave a slot open under the wedge.
    ///
    /// Anything on the map border, or against a culled cell, goes to the
    /// floor: that is the outside of the build, with nothing beyond it to
    /// hide a hollow.
    fn shell_bottom(&self, x: i32, y: i32, w: i32, hgt: i32, top: i32, floor: i32) -> i32 {
        if x == 0 || y == 0 || x + w == self.f.cols || y + hgt == self.f.rows {
            return floor;
        }
        let bt = |xx: i32, yy: i32| -> i32 {
            let i = self.idx(xx, yy);
            if self.culled[i] { floor } else { self.box_to[i] }
        };
        let mut lowest = i32::MAX;
        for dy in 0..hgt {
            lowest = lowest.min(bt(x - 1, y + dy)).min(bt(x + w, y + dy));
        }
        for dx in 0..w {
            lowest = lowest.min(bt(x + dx, y - 1)).min(bt(x + dx, y + hgt));
        }
        // a box in a pit has every neighbour ABOVE it: its sides are buried,
        // so all it owes is a floor tile
        floor.max(lowest.min(top) - SKIRT)
    }

    /// Greedy-merge the flat boxes into big rectangles. Deliberately the
    /// simple greedy: the sculpt tool measured row-run stacking at 1-3%
    /// WORSE at every span tried; the span is the lever, not the algorithm.
    fn merge_boxes(&mut self, max_span: i32, floor: i32) {
        let mut used = self.culled.to_vec();
        for y in 0..self.f.rows {
            for x in 0..self.f.cols {
                let i = self.idx(x, y);
                if used[i] {
                    continue;
                }
                let v = self.box_to[i];
                let c = self.colors[i];
                let mut w = 1;
                while w < max_span
                    && x + w < self.f.cols
                    && !used[i + w as usize]
                    && self.box_to[i + w as usize] == v
                    && self.colors[i + w as usize] == c
                {
                    w += 1;
                }
                let mut hgt = 1;
                'rows: while hgt < max_span && y + hgt < self.f.rows {
                    let scan = self.idx(x, y + hgt);
                    for k in 0..w as usize {
                        if used[scan + k] || self.box_to[scan + k] != v || self.colors[scan + k] != c
                        {
                            break 'rows;
                        }
                    }
                    hgt += 1;
                }
                for dy in 0..hgt {
                    let fill = self.idx(x, y + dy);
                    for dx in 0..w as usize {
                        used[fill + dx] = true;
                    }
                }
                // The bottom is computed from the MERGED rectangle, not from
                // the cells it came from: two adjacent boxes with the same
                // top would get different bottoms from their own perimeters,
                // and that difference is enough to stop them merging at all.
                let bottom = self.shell_bottom(x, y, w, hgt, v, floor);
                if v > bottom {
                    self.pieces.push(Piece {
                        wedge: false,
                        x,
                        y,
                        sx: w,
                        sy: hgt,
                        from: bottom,
                        to: v,
                        rotation: 0,
                        color: c,
                    });
                }
            }
        }
    }
}

/// Split anything taller than the brick limit into stacked pieces. A side
/// wedge is a VERTICAL prism, so a vertical split leaves its shape exactly.
fn split_tall(pieces: Vec<Piece>) -> Vec<Piece> {
    let mut out = Vec::with_capacity(pieces.len());
    for p in pieces {
        if p.to - p.from <= MAX_PLATE_SPAN {
            out.push(p);
            continue;
        }
        let mut at = p.from;
        while at < p.to {
            let next = p.to.min(at + MAX_PLATE_SPAN);
            out.push(Piece {
                from: at,
                to: next,
                ..p.clone()
            });
            at = next;
        }
    }
    out
}

/// Make terraced wedge terrain from a heightmap.
///
/// With `--wedge`, `--vertical` is the height of one terrace step (one shade
/// of grey) in units, rounded to a whole number of plates -- a side wedge
/// spans whole plates. Every baked height is a multiple of that step, which
/// is what makes the terraces.
pub fn gen_wedge_heightmap<F: Fn(f32) -> bool>(
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
    // A zero cell size would emit zero-extent bricks and divide by zero in
    // the span cap below. Refuse it like the other invalid inputs.
    if options.size == 0 {
        return Err("Brick size must be at least 1".to_string());
    }

    let terrace = ((options.scale as i32 + CELL_UNITS / 2) / CELL_UNITS).max(1);
    if terrace * CELL_UNITS != options.scale as i32 {
        info!(
            "Wedge terraces rise {} unit(s) per shade (--vertical {}, rounded to {terrace} \
             plate(s): a side wedge spans whole plates)",
            terrace * CELL_UNITS,
            options.scale
        );
    }

    info!("Baking terraces");
    let cols = width as i32;
    let rows = height as i32;
    let count = (cols * rows) as usize;
    let mut field = Field {
        cols,
        rows,
        h: vec![0; count],
    };
    let mut colors = vec![[0u8; 4]; count];
    let mut culled = vec![false; count];
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let color = colormap.at(x, y);
            let shade = heightmap.at(x, y).min(1 << 24) as i32;
            colors[i] = color;
            // The same rule as the other modes: with --cull, a fully
            // transparent pixel and a pixel at the lowest level are removed.
            // A culled cell keeps height zero, so its neighbours grow a
            // proper chamfered cliff toward the hole.
            culled[i] = options.cull && (shade == 0 || color[3] == 0);
            field.h[i] = if culled[i] {
                0
            } else {
                shade.saturating_mul(terrace).min(1 << 24)
            };
        }
    }
    progress!(0.1);

    info!("Eroding unchamferable configurations");
    erode(&mut field, 200);
    progress!(0.25);

    info!("Classifying cells");
    let (kind, diag, rot, val) = classify(&field);
    progress!(0.35);

    // Spans are capped by the brick limit: a footprint's half extent
    // `span * half` must stay inside MAX_HALF_EXTENT. The cap goes into the
    // plan, not over it -- a piece too large to be one brick is never
    // planned, rather than planned and split.
    let half = options.size as i32;
    let max_span = (MAX_HALF_EXTENT / half).max(1);
    let max_run = MERGE_RUN.min(max_span);
    // Strictly below the lowest cell in the whole map, so every column --
    // a flat plain included -- keeps at least one plate of ground.
    let floor = field.h.iter().copied().min().unwrap_or(0) - 1;

    info!("Planning pieces");
    let mut plan = Plan {
        f: &field,
        kind: &kind,
        diag: &diag,
        rot: &rot,
        val: &val,
        colors: &colors,
        culled: &culled,
        consumed: culled.clone(),
        box_to: field.h.clone(),
        pieces: Vec::new(),
    };
    plan.merge_staircases(max_run);
    progress!(0.5);
    plan.stretch_corners();
    progress!(0.6);
    plan.leftovers_and_fillers();
    progress!(0.7);
    plan.merge_boxes(max_span, floor);
    progress!(0.85);

    let pieces = split_tall(plan.pieces);

    info!("Emitting bricks");
    let cell = half * 2;
    let offset_x = -(cols * half);
    let offset_y = -(rows * half);
    // The surface of plate zero, where the other modes put the top of a cell
    // of height zero -- so a change of mode does not move the build.
    let z_floor = options.base_height() - 5;
    let collision = options.collision();

    let mut bricks = Vec::with_capacity(pieces.len());
    let mut wedges = 0usize;
    let mut merged = 0usize;
    for p in &pieces {
        // Rotation turns the brick into the world, so 90/270 rotations swap
        // the footprint axes -- the size is pre-swapped so the world
        // footprint matches the planned cells. Getting this wrong drops
        // every odd-rotation wedge somewhere else entirely.
        let swap = p.wedge && p.rotation % 2 == 1;
        let half_x = if swap { p.sy } else { p.sx } * half;
        let half_y = if swap { p.sx } else { p.sy } * half;
        if p.wedge {
            wedges += 1;
        }
        if p.sx > 1 || p.sy > 1 {
            merged += 1;
        }
        bricks.push(Brick {
            asset: BrickType::Procedural {
                asset: if p.wedge {
                    PB_DEFAULT_SIDE_WEDGE
                } else {
                    PB_DEFAULT_BRICK
                },
                // Sizes are HALF extents and positions are centers, so a
                // piece spanning plates a..b has half height 2(b-a) and
                // center z 4a + 2(b-a).
                size: BrickSize::new(
                    half_x as u16,
                    half_y as u16,
                    (2 * (p.to - p.from)) as u16,
                ),
            },
            position: Position::new(
                offset_x + p.x * cell + p.sx * half,
                offset_y + p.y * cell + p.sy * half,
                z_floor + CELL_UNITS * p.from + 2 * (p.to - p.from),
            ),
            collision,
            color: Color {
                r: p.color[0],
                g: p.color[1],
                b: p.color[2],
            },
            owner_index: None,
            direction: Direction::ZPositive,
            rotation: match p.rotation {
                0 => Rotation::Deg0,
                1 => Rotation::Deg90,
                2 => Rotation::Deg180,
                _ => Rotation::Deg270,
            },
            material_intensity: if options.glow { 0 } else { 5 },
            material: if options.glow { GLOW } else { PLASTIC },
            ..Default::default()
        });
    }

    info!(
        "Converted {} pixel(s) to {} brick(s) ({:.2} per cell): {wedges} wedge, {} box, \
         {merged} merged",
        count,
        bricks.len(),
        bricks.len() as f64 / count.max(1) as f64,
        bricks.len() - wedges,
    );
    progress!(1.0);
    Ok(bricks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use brdb::assets::bricks::{PB_DEFAULT_MICRO_BRICK, PB_DEFAULT_SIDE_WEDGE};

    /// A heightmap from a closure, for small hand-built fields.
    struct Fn2D(u32, u32, fn(u32, u32) -> u32);
    impl Heightmap for Fn2D {
        fn at(&self, x: u32, y: u32) -> u32 {
            (self.2)(x, y)
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
    /// Transparent on the left half, opaque on the right.
    struct HalfClear(u32, u32);
    impl Colormap for HalfClear {
        fn at(&self, x: u32, _y: u32) -> [u8; 4] {
            [128, 128, 128, if x < self.0 / 2 { 0 } else { 255 }]
        }
        fn size(&self) -> (u32, u32) {
            (self.0, self.1)
        }
    }

    fn options(scale: u32) -> GenOptions {
        GenOptions {
            size: 5,
            scale,
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
            quadtree: false,
            greedy: false,
            surface: SurfaceMode::Wedge,
        }
    }

    fn build(hm: &dyn Heightmap, cm: &dyn Colormap, opts: GenOptions) -> Vec<Brick> {
        gen_wedge_heightmap(hm, cm, opts, |_| true).unwrap()
    }

    fn side_wedges(bricks: &[Brick]) -> impl Iterator<Item = (&Brick, BrickSize)> {
        bricks.iter().filter_map(|b| {
            let BrickType::Procedural { asset, size } = &b.asset else {
                unreachable!("wedge terrain makes procedural bricks only")
            };
            (*asset == PB_DEFAULT_SIDE_WEDGE).then_some((b, *size))
        })
    }

    /// The point of the greedy box pass: a uniform plain must be a handful
    /// of boxes with no wedges anywhere, one plate thick (the floor sits one
    /// plate under the lowest cell).
    #[test]
    fn a_flat_plain_collapses_to_one_thin_box() {
        let bricks = build(&Fn2D(16, 16, |_, _| 0), &Grey(16, 16), options(4));
        assert_eq!(bricks.len(), 1, "16x16 equal cells must merge to one box");
        assert_eq!(side_wedges(&bricks).count(), 0);
        let BrickType::Procedural { size, .. } = &bricks[0].asset else {
            unreachable!()
        };
        assert_eq!(size.z, 2, "one plate of ground, as half extents");
    }

    /// The core of the style: a plateau's four convex corners are cut at 45
    /// degrees by vertical side wedges spanning exactly the terrace rise.
    #[test]
    fn a_plateau_gets_side_wedge_chamfers_on_its_corners() {
        let hm = Fn2D(12, 12, |x, y| ((4..8).contains(&x) && (4..8).contains(&y)) as u32);
        let bricks = build(&hm, &Grey(12, 12), options(4));
        let wedges: Vec<_> = side_wedges(&bricks).collect();
        assert!(
            wedges.len() >= 4,
            "each of the plateau's four corners needs a cut, got {}",
            wedges.len()
        );
        for (brick, size) in &wedges {
            assert_eq!(size.z, 2, "a cut spans the whole one-plate terrace step");
            assert_eq!(
                brick.position.z, 2 - 5 + 2,
                "the cut sits on top of the column, under the plateau top"
            );
        }
    }

    /// A raised region bounded by x+y<=k has a straight 45-degree edge: a
    /// staircase of collinear chamfers. The planner must merge it into NxN
    /// wedges rather than emit one cut per step -- and at the merged corner
    /// diagonal 0 (+X/+Y lower), the calibrated rotation is Deg0.
    #[test]
    fn a_diagonal_staircase_merges_into_one_large_wedge() {
        let hm = Fn2D(12, 12, |x, y| (x + y <= 5) as u32);
        let bricks = build(&hm, &Grey(12, 12), options(4));
        let big: Vec<_> = side_wedges(&bricks)
            .filter(|(_, s)| s.x == s.y && s.x >= 3 * 5)
            .collect();
        assert!(
            !big.is_empty(),
            "a 6-step staircase must produce at least one merged NxN wedge"
        );
        assert!(
            big.iter()
                .all(|(b, _)| matches!(b.rotation, Rotation::Deg0)),
            "the hypotenuse of a +X/+Y corner faces diagonal 0, which is rotation 0"
        );
    }

    /// Erosion rule 2: a single-cell spike has four lower sides and cannot
    /// be chamfered. It must be eaten back to the plain, not built.
    #[test]
    fn a_spike_is_eroded_flat() {
        let hm = Fn2D(8, 8, |x, y| if (x, y) == (4, 4) { 5 } else { 0 });
        let bricks = build(&hm, &Grey(8, 8), options(4));
        assert_eq!(bricks.len(), 1, "the spike must erode away entirely");
        assert_eq!(side_wedges(&bricks).count(), 0);
    }

    /// Erosion rule 1: two cells touching only corner-to-corner force hard
    /// right angles (chamfering either would sever the connection), so one
    /// of them is lowered. No wedge may end up cutting either corner of the
    /// crossing while it stands.
    #[test]
    fn a_diagonal_crossing_is_eroded() {
        let hm = Fn2D(8, 8, |x, y| ((x, y) == (3, 3) || (x, y) == (4, 4)) as u32);
        let bricks = build(&hm, &Grey(8, 8), options(4));
        // Both raised cells sit corner-to-corner; erosion lowers one, and
        // the survivor is a 1x1 spike, which rule 2 then removes.
        assert_eq!(bricks.len(), 1, "the crossing must not survive to be built");
    }

    /// A cliff taller than the brick limit must be stacked, not emitted as
    /// one illegal brick.
    #[test]
    fn tall_columns_split_into_legal_bricks() {
        let hm = Fn2D(8, 8, |x, _| if x < 4 { 0 } else { 200 });
        let bricks = build(&hm, &Grey(8, 8), options(4));
        assert!(bricks.len() > 2, "a 200-plate cliff cannot be two bricks");
        for b in &bricks {
            let BrickType::Procedural { size, .. } = &b.asset else {
                unreachable!()
            };
            assert!(
                (size.z as i32) <= MAX_HALF_EXTENT,
                "a brick is {} half-units tall, over the {MAX_HALF_EXTENT} limit",
                size.z
            );
        }
    }

    /// `--vertical` rounds to whole plates: at 8 units per shade every
    /// terrace step is two plates, so a one-step cut is 8 units tall.
    #[test]
    fn the_terrace_step_follows_the_vertical_scale_in_plates() {
        let hm = Fn2D(12, 12, |x, y| ((4..8).contains(&x) && (4..8).contains(&y)) as u32);
        let bricks = build(&hm, &Grey(12, 12), options(8));
        assert!(
            side_wedges(&bricks).all(|(_, s)| s.z == 4),
            "a cut must span the whole two-plate terrace step"
        );
    }

    /// With --cull, transparent cells are holes: nothing may be built over
    /// them, and the terrain beside them still gets its ground.
    #[test]
    fn culled_cells_emit_no_bricks() {
        let hm = Fn2D(16, 16, |_, _| 2);
        let mut opts = options(4);
        opts.cull = true;
        let bricks = build(&hm, &HalfClear(16, 16), opts);
        assert!(!bricks.is_empty(), "the opaque half must still be built");
        // Cells 0..8 are culled; the world seam between cell 7 and 8 is at
        // -(16*5) + 8*10 = 0.
        for b in &bricks {
            let BrickType::Procedural { size, .. } = &b.asset else {
                unreachable!()
            };
            assert!(
                b.position.x - size.x as i32 >= 0,
                "a brick reaches into the culled half: {} - {}",
                b.position.x,
                size.x
            );
        }
    }

    /// Merges must not cross a colour edge: a brick is one colour, so a
    /// two-colour map can never collapse to one box.
    #[test]
    fn boxes_stop_at_a_color_edge() {
        struct TwoTone(u32, u32);
        impl Colormap for TwoTone {
            fn at(&self, x: u32, _y: u32) -> [u8; 4] {
                if x < self.0 / 2 { [200, 40, 40, 255] } else { [40, 200, 40, 255] }
            }
            fn size(&self) -> (u32, u32) {
                (self.0, self.1)
            }
        }
        let bricks = build(&Fn2D(16, 16, |_, _| 0), &TwoTone(16, 16), options(4));
        assert_eq!(bricks.len(), 2, "one box per colour, and no third piece");
        let seam = 0; // between cells 7 and 8 on a 16-wide map of 10-unit cells
        for b in &bricks {
            let BrickType::Procedural { size, .. } = &b.asset else {
                unreachable!()
            };
            let (low, high) = (b.position.x - size.x as i32, b.position.x + size.x as i32);
            assert!(low >= seam || high <= seam, "a box crosses the colour seam");
        }
    }

    /// Every cell a wedge cuts must expose the box below it down to the
    /// wedge's base: the cut cell's column tops out at the base, not at the
    /// terrain height, or the wedge floats on a full-height box. The
    /// coverage may be split across several boxes -- what matters is that
    /// under every cell of the wedge some box top meets the wedge bottom.
    #[test]
    fn the_box_under_a_cut_stops_at_the_wedge_base() {
        let hm = Fn2D(12, 12, |x, y| ((4..8).contains(&x) && (4..8).contains(&y)) as u32);
        let bricks = build(&hm, &Grey(12, 12), options(4));
        assert!(side_wedges(&bricks).count() >= 4, "the plateau must be cut");
        for (wedge, size) in side_wedges(&bricks) {
            let wedge_bottom = wedge.position.z - size.z as i32;
            // `size` is in the brick's LOCAL axes; a 90/270 rotation swaps
            // them into the world, so un-swap to get the world footprint
            let odd = matches!(wedge.rotation, Rotation::Deg90 | Rotation::Deg270);
            let (wx, wy) = if odd { (size.y, size.x) } else { (size.x, size.y) };
            // every 10-unit cell under the wedge footprint, by its center
            for cy in 0..(wy as i32 / 5) {
                for cx in 0..(wx as i32 / 5) {
                    let px = wedge.position.x - wx as i32 + cx * 10 + 5;
                    let py = wedge.position.y - wy as i32 + cy * 10 + 5;
                    let covered = bricks.iter().any(|b| {
                        let BrickType::Procedural { asset, size: bs } = &b.asset else {
                            unreachable!()
                        };
                        *asset == PB_DEFAULT_BRICK
                            && b.position.z + bs.z as i32 == wedge_bottom
                            && (b.position.x - bs.x as i32) < px
                            && (b.position.x + bs.x as i32) > px
                            && (b.position.y - bs.y as i32) < py
                            && (b.position.y + bs.y as i32) > py
                    });
                    assert!(
                        covered,
                        "no box top meets the wedge bottom z {wedge_bottom} under ({px}, {py})"
                    );
                }
            }
        }
    }
}
