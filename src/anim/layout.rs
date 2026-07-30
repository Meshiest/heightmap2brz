//! Placement of gates inside a microchip's inner grid.
//!
//! The chip is a 3D mirror of the screen: a pixel's gates sit at that pixel's
//! own row/column, with the gate stages stacked along depth. Wires stay short
//! and a misplaced pixel is visible by eye when the chip is opened.
use brdb::{Direction, IntVector, Position, Rotation};

/// Gate lattice pitch, in game units. Every gate this crate emits is
/// half-size 5x5x2, i.e. 10x10x4 units, so a uniform 10-unit pitch clears
/// on all three axes (the minimum non-overlapping depth pitch is 4).
pub const CELL: i32 = 10;

/// Half-size shared by every gate brick used by the animation renderer.
pub const GATE_HALF: IntVector = IntVector { x: 5, y: 5, z: 2 };

/// The half-extent a brick authored with `half` actually OCCUPIES once
/// `direction` and `rotation` are applied.
///
/// This exists because `brdb::Brick::local_bounds` deliberately does not apply
/// rotation ("the extent is taken as authored, which is exact for unrotated
/// bricks and an approximation otherwise"), so every collision and plane-extent
/// check in this crate measures whatever half-size the caller RECORDED. A
/// rotated brick recorded with its unrotated half leaves those checks measuring
/// a box the brick does not fill -- which is exactly how the pixel gates came
/// to carry a hand-written `{5, 2, 5}` that no orientation of a `{5, 5, 2}`
/// gate can produce.
///
/// # What is certain, and what is convention
///
/// `direction` is the game's `EBrickDirection` (`brdb::Direction`; the
/// discriminants `X_Positive = 0 ..= Z_Negative = 5` are in
/// `brdb/schemas/BRSavedComponentChunkSoA_max.schema`), and it names the WORLD
/// AXIS the brick's own local +Z points along -- `Z_Positive` is the default,
/// i.e. unrotated. `crate::text` documents the same enum the same way: its
/// `FACE_X_POSITIVE` is the face an upright wall stands on, `FACE_Z_POSITIVE`
/// the upward face of something lying flat.
///
/// So the CERTAIN part is which world axis carries the authored `z` extent:
/// the one `direction` names. The two remaining authored extents (`x` and `y`)
/// fill the two remaining world axes, and `rotation` -- a quarter-turn roll
/// ABOUT the direction axis -- swaps them at 90 and 270 degrees.
///
/// Which of the two remaining authored extents lands on which world axis at
/// `Deg0` is a CONVENTION this function fixes rather than a fact it derives,
/// and it is only observable when `half.x != half.y`. Every rotated brick this
/// crate emits is a logic gate, whose authored half is `{5, 5, 2}` -- square in
/// x/y -- so the convention is unobservable for every use here, which is what
/// makes deriving the extent safe without an in-game measurement.
/// `a_rotated_gate_is_square_in_the_two_non_direction_axes` pins that.
pub const fn rotated_half(half: IntVector, direction: Direction, rotation: Rotation) -> IntVector {
    // The two authored extents that are NOT on the brick's own z axis, in the
    // order they fill the two world axes the direction did not claim. A
    // quarter turn about the direction axis swaps them.
    let (a, b) = match rotation {
        Rotation::Deg0 | Rotation::Deg180 => (half.x, half.y),
        Rotation::Deg90 | Rotation::Deg270 => (half.y, half.x),
    };
    match direction {
        // The brick's own z lands on world x; a and b fill y and z.
        Direction::XPositive | Direction::XNegative => IntVector { x: half.z, y: a, z: b },
        // ... on world y; a and b fill x and z.
        Direction::YPositive | Direction::YNegative => IntVector { x: a, y: half.z, z: b },
        // Unrotated (and `MAX`, which is not a real facing): as authored,
        // except for the roll's x/y swap.
        Direction::ZPositive | Direction::ZNegative | Direction::MAX => {
            IntVector { x: a, y: b, z: half.z }
        }
    }
}

/// Depth (z) pitch between pipeline stages.
///
/// [`CELL`] would be right while the gates lie flat: a flat gate is 4 units
/// through, so a 10-unit pitch would leave 6 units of air between stages. The
/// per-pixel gates are NOT flat -- `bricks::PIXEL_GATE_FACING` stands them on
/// end, which makes them 10 units through on z ([`rotated_half`] derives
/// `half.z = 5`), and a `CELL` pitch would then put every stage face exactly
/// flush against the next.
///
/// 15 is that 10 plus a 5-unit air gap. `stages_clear_each_other_at_the_stage_pitch`
/// pins it against the derived extent rather than against a remembered number,
/// so a change to the facing cannot silently leave this too tight again.
pub const STAGE_PITCH: i32 = 15;

/// Height offset applied to the whole staged lattice, lifting every brick in
/// the chip off the interaction plane instead of starting flush against it.
///
/// Local **Z is up** inside the chip -- the plane lies flat, so its normal is
/// world up. (X and Y are the two in-plane horizontal axes; shifting either
/// one moves gates sideways, which is what made two earlier attempts to
/// "raise" the gates go astray.) Every brick in the chip picks this up: the
/// pixel gates and the service gates call [`lattice_pos_staged`] directly,
/// and the clock and its I/O pins derive their positions from the service
/// origin, so a single constant moves the whole graph.
pub const STAGE_BASE_Z: i32 = 6;

/// [`lattice_pos`], with an explicit depth pitch between stages.
///
/// `lattice_pos` spaces stages by [`CELL`], which is right for flat gates but
/// too tight for upright ones -- see [`STAGE_PITCH`].
pub fn lattice_pos_staged(
    col: i32,
    row: i32,
    stage: i32,
    height: i32,
    half: IntVector,
    stage_pitch: i32,
) -> Position {
    Position {
        x: (height - 1 - row) * CELL + half.x,
        y: col * CELL + half.y,
        z: stage * stage_pitch + half.z + STAGE_BASE_Z,
    }
}

/// Position for the gate serving pixel (`col`, `row`) at pipeline `stage`.
///
/// `half` is the brick's own half-size: `Brick.position` is the center, so
/// the min corner lands on the lattice line only after adding it.
pub fn lattice_pos(col: i32, row: i32, stage: i32, height: i32, half: IntVector) -> Position {
    Position {
        x: (height - 1 - row) * CELL + half.x,
        y: col * CELL + half.y,
        z: stage * CELL + half.z,
    }
}

/// Scan `boxes` (as `(min, max)` corners) for an intersecting pair.
///
/// Returns the first pair found, plus **how many pairs were actually tested** --
/// which is what the tests below assert on, because the whole point of this
/// function is that the second number does not grow quadratically.
///
/// # Why this is not the obvious all-pairs loop
///
/// It used to be, in both callers. A 192x108 render -- the size both
/// `text_bricks`'s and `color_bricks`'s module docs use as their reference --
/// puts 41,472 pixel gates on the chip's inner grid, so the inner check alone
/// performed ~8.6e8 pair comparisons and the main-grid one ~2.1e8. Both run
/// from [`super::chip::finish`], i.e. after the last cancellation poll and
/// after `progress.finish()`, so a user who cancelled at 99% still sat through
/// the whole quadratic phase with no progress bar -- flatly contradicting
/// `build_brick_world`'s promise that a cancel "lands just as promptly as one
/// pressed during the decode".
///
/// # The method
///
/// A uniform spatial hash. The bucket size is the largest box in the set on
/// each axis (never smaller than 1), so no box can span more than two buckets
/// per axis and each therefore touches at most eight. A box is compared only
/// against boxes already filed in a bucket it touches.
///
/// That is exhaustive, not a heuristic: if two boxes intersect, the
/// intersection is a non-empty region, every point of it lies inside both
/// boxes, and the bucket containing any such point is filed under both -- so
/// the pair is tested. `the_spatial_hash_agrees_with_brute_force` checks that
/// against an all-pairs reference over a few thousand awkward arrangements
/// rather than taking the argument's word for it.
fn overlap_scan(boxes: &[(Position, Position)]) -> (Option<(usize, usize)>, usize) {
    let mut tested = 0usize;
    if boxes.len() < 2 {
        return (None, tested);
    }

    // Bucket size: the largest extent present on each axis, so a box spans at
    // most two buckets per axis. `.max(1)` keeps a set of degenerate
    // (zero-size) boxes from dividing by zero.
    let mut cell = (1i32, 1i32, 1i32);
    for (min, max) in boxes {
        cell.0 = cell.0.max(max.x - min.x);
        cell.1 = cell.1.max(max.y - min.y);
        cell.2 = cell.2.max(max.z - min.z);
    }

    let key = |p: &Position| {
        (
            p.x.div_euclid(cell.0),
            p.y.div_euclid(cell.1),
            p.z.div_euclid(cell.2),
        )
    };

    let mut buckets: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
        std::collections::HashMap::with_capacity(boxes.len());
    // Reused across iterations so the scan does not allocate per box.
    let mut candidates: Vec<usize> = Vec::new();

    for (i, (amin, amax)) in boxes.iter().enumerate() {
        let (lo, hi) = (key(amin), key(amax));
        candidates.clear();
        for cx in lo.0..=hi.0 {
            for cy in lo.1..=hi.1 {
                for cz in lo.2..=hi.2 {
                    if let Some(bucket) = buckets.get(&(cx, cy, cz)) {
                        candidates.extend_from_slice(bucket);
                    }
                }
            }
        }
        // A box straddling a bucket boundary is filed in several buckets, so
        // the same candidate can turn up more than once.
        candidates.sort_unstable();
        candidates.dedup();

        for &j in &candidates {
            let (bmin, bmax) = &boxes[j];
            tested += 1;
            // Strict inequality: flush neighbours share a face and are legal.
            let overlaps = amin.x < bmax.x
                && bmin.x < amax.x
                && amin.y < bmax.y
                && bmin.y < amax.y
                && amin.z < bmax.z
                && bmin.z < amax.z;
            if overlaps {
                return (Some((j, i)), tested);
            }
        }

        for cx in lo.0..=hi.0 {
            for cy in lo.1..=hi.1 {
                for cz in lo.2..=hi.2 {
                    buckets.entry((cx, cy, cz)).or_default().push(i);
                }
            }
        }
    }
    (None, tested)
}

/// Errors if any two placed bricks intersect.
///
/// The game silently DROPS overlapping bricks at load, which orphans their
/// components and dangles every wire into them -- producing a save that opens
/// fine and does nothing. This is the cheapest place to catch that.
///
/// `half` is the caller's own record of the brick's half-size, because
/// `Brick::local_bounds` cannot describe a rotated brick (see
/// [`rotated_half`]). Detection is exactly what it always was -- an AABB
/// intersection with strict inequality, so flush neighbours are legal -- only
/// the search for the pair is now proportional rather than all-pairs; see
/// [`overlap_scan`].
pub fn assert_no_overlap(placed: &[(Position, IntVector)]) -> Result<(), String> {
    let boxes: Vec<(Position, Position)> = placed
        .iter()
        .map(|(p, h)| {
            (
                Position { x: p.x - h.x, y: p.y - h.y, z: p.z - h.z },
                Position { x: p.x + h.x, y: p.y + h.y, z: p.z + h.z },
            )
        })
        .collect();
    let (hit, _) = overlap_scan(&boxes);
    let Some((i, j)) = hit else { return Ok(()) };
    let (pa, ha) = placed[i];
    let (pb, hb) = placed[j];
    Err(format!(
        "bricks overlap: ({},{},{}) half ({},{},{}) vs ({},{},{}) half ({},{},{})",
        pa.x, pa.y, pa.z, ha.x, ha.y, ha.z, pb.x, pb.y, pb.z, hb.x, hb.y, hb.z
    ))
}

/// Errors if any two bricks on a grid intersect, using each brick's own
/// asset-aware bounds rather than a caller-supplied half-size.
///
/// This exists because [`assert_no_overlap`] only ever sees a microchip's
/// INNER grid, so it cannot catch a collision on the main grid -- which is
/// exactly where the first real one happened: the `B_1x1_Microchip` shell
/// (half-extent 5,5,2) sat at z=6 on top of a display brick at z=2 whose
/// default half-extent is 5,5,6, so their z spans [4,8] and [-4,8] overlapped
/// on an identical x/y footprint. The game dropped one and a 2x2 screen came
/// back as a 3-brick L with its wires unconnectable.
///
/// Sizes differ per asset -- a default procedural brick is 5,5,6, gates are
/// 5,5,2, a reroute node is 1,1,1 -- so this reads `local_bounds()` per brick
/// instead of assuming a uniform size. The pair search is [`overlap_scan`]'s,
/// so this is proportional rather than all-pairs; what counts as an overlap is
/// unchanged.
pub fn assert_bricks_dont_overlap(bricks: &[brdb::Brick]) -> Result<(), String> {
    let boxes: Vec<(Position, Position)> = bricks.iter().map(|b| b.local_bounds()).collect();
    let (hit, _) = overlap_scan(&boxes);
    let Some((i, j)) = hit else { return Ok(()) };
    let (a, (amin, amax)) = (&bricks[i], boxes[i]);
    let (b, (bmin, bmax)) = (&bricks[j], boxes[j]);
    Err(format!(
        "bricks overlap on the main grid: brick at ({},{},{}) spanning \
         ({},{},{})..({},{},{}) vs brick at ({},{},{}) spanning \
         ({},{},{})..({},{},{})",
        a.position.x,
        a.position.y,
        a.position.z,
        amin.x,
        amin.y,
        amin.z,
        amax.x,
        amax.y,
        amax.z,
        b.position.x,
        b.position.y,
        b.position.z,
        bmin.x,
        bmin.y,
        bmin.z,
        bmax.x,
        bmax.y,
        bmax.z,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact collision that shipped in the first in-game spike: a default
    /// display brick at the origin and a microchip shell four units above it.
    #[test]
    fn microchip_shell_stacked_on_a_display_brick_is_caught() {
        let display = brdb::Brick {
            position: Position { x: 0, y: 0, z: 2 },
            ..Default::default()
        };
        let shell = brdb::Brick {
            asset: brdb::assets::bricks::B_MICROCHIP,
            position: Position { x: 0, y: 0, z: 6 },
            ..Default::default()
        };
        assert!(
            assert_bricks_dont_overlap(&[display, shell]).is_err(),
            "a chip shell inside a display brick's z span must be rejected"
        );
    }

    #[test]
    fn a_flat_screen_of_default_bricks_is_legal() {
        let bricks: Vec<_> = (0..3)
            .flat_map(|row| {
                (0..3).map(move |col| brdb::Brick {
                    position: Position { x: col * 10, y: row * 10, z: 2 },
                    ..Default::default()
                })
            })
            .collect();
        assert_bricks_dont_overlap(&bricks).expect("a 10-unit pitch screen must be legal");
    }

    #[test]
    fn a_chip_shell_clear_of_the_screen_is_legal() {
        let display = brdb::Brick {
            position: Position { x: 0, y: 0, z: 2 },
            ..Default::default()
        };
        let shell = brdb::Brick {
            asset: brdb::assets::bricks::B_MICROCHIP,
            position: Position { x: -20, y: 0, z: 2 },
            ..Default::default()
        };
        assert_bricks_dont_overlap(&[display, shell]).expect("clear of the screen is legal");
    }

    #[test]
    fn row_zero_sits_highest() {
        let top = lattice_pos(0, 0, 0, 4, GATE_HALF);
        let bottom = lattice_pos(0, 3, 0, 4, GATE_HALF);
        assert!(top.x > bottom.x, "row 0 must be above the last row");
    }

    #[test]
    fn columns_advance_along_y_and_stages_along_z() {
        let a = lattice_pos(0, 0, 0, 1, GATE_HALF);
        let b = lattice_pos(1, 0, 0, 1, GATE_HALF);
        let c = lattice_pos(0, 0, 1, 1, GATE_HALF);
        assert_eq!(b.y - a.y, CELL);
        assert_eq!(c.z - a.z, CELL);
        assert_eq!(a.y, CELL * 0 + GATE_HALF.y, "min corner lands on the lattice");
    }

    #[test]
    fn a_full_pixel_lattice_never_overlaps() {
        let (w, h, stages) = (8, 6, 2);
        let mut placed = Vec::new();
        for row in 0..h {
            for col in 0..w {
                for stage in 0..stages {
                    placed.push((lattice_pos(col, row, stage, h, GATE_HALF), GATE_HALF));
                }
            }
        }
        assert_eq!(placed.len(), (w * h * stages) as usize);
        assert_no_overlap(&placed).expect("lattice must be collision-free");
    }

    #[test]
    fn overlap_is_detected() {
        let p = Position { x: 0, y: 0, z: 0 };
        let placed = vec![(p, GATE_HALF), (p, GATE_HALF)];
        assert!(assert_no_overlap(&placed).is_err(), "identical positions must collide");
    }

    #[test]
    fn touching_faces_do_not_count_as_overlap() {
        let a = (Position { x: 0, y: 0, z: 0 }, GATE_HALF);
        let b = (Position { x: 10, y: 0, z: 0 }, GATE_HALF);
        assert_no_overlap(&[a, b]).expect("flush neighbours are legal");
    }

    // --- overlap scan: proportional, and still exhaustive --------------------

    /// The all-pairs loop [`overlap_scan`] replaced, kept as the reference the
    /// spatial hash is checked against. Returns whether ANY pair intersects --
    /// which pair is found first legitimately differs between the two, since
    /// the hash does not visit pairs in index order.
    fn brute_force_any_overlap(boxes: &[(Position, Position)]) -> bool {
        for (i, (amin, amax)) in boxes.iter().enumerate() {
            for (bmin, bmax) in boxes.iter().skip(i + 1) {
                if amin.x < bmax.x
                    && bmin.x < amax.x
                    && amin.y < bmax.y
                    && bmin.y < amax.y
                    && amin.z < bmax.z
                    && bmin.z < amax.z
                {
                    return true;
                }
            }
        }
        false
    }

    /// A cheap deterministic PRNG, so the arrangements below are awkward
    /// without being irreproducible. (`rand` is not a dependency of this
    /// crate, and a fixed sequence is what makes a failure debuggable.)
    fn lcg(state: &mut u64) -> i32 {
        *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (*state >> 33) as i32
    }

    /// **The check that the optimisation preserved exactly what it detects.**
    ///
    /// A spatial hash that missed a pair would turn this module's whole purpose
    /// -- catching bricks the game will silently drop -- into a no-op that
    /// always says "clear". So the hash is run against the all-pairs reference
    /// over a few hundred randomised arrangements, deliberately including
    /// sizes far larger than the lattice pitch (which forces boxes across many
    /// buckets) and sizes of zero (which forces the `.max(1)` bucket floor).
    #[test]
    fn the_spatial_hash_agrees_with_brute_force() {
        let mut state = 0x5EED_1234_ABCD_0001u64;
        let mut saw_overlap = 0;
        let mut saw_clear = 0;
        for case in 0..400 {
            // Vary the density: a tight spread makes overlaps near-certain, a
            // loose one makes them rare, so both answers are exercised.
            let spread = 8 + (case % 7) * 25;
            let max_half = 1 + (case % 5) * 4;
            let n = 3 + (case % 23);
            let boxes: Vec<(Position, Position)> = (0..n)
                .map(|_| {
                    let c = Position {
                        x: lcg(&mut state).rem_euclid(spread) - spread / 2,
                        y: lcg(&mut state).rem_euclid(spread) - spread / 2,
                        z: lcg(&mut state).rem_euclid(spread) - spread / 2,
                    };
                    let h = Position {
                        x: lcg(&mut state).rem_euclid(max_half),
                        y: lcg(&mut state).rem_euclid(max_half),
                        z: lcg(&mut state).rem_euclid(max_half),
                    };
                    (
                        Position { x: c.x - h.x, y: c.y - h.y, z: c.z - h.z },
                        Position { x: c.x + h.x, y: c.y + h.y, z: c.z + h.z },
                    )
                })
                .collect();
            let expected = brute_force_any_overlap(&boxes);
            let (hit, _) = overlap_scan(&boxes);
            assert_eq!(
                hit.is_some(),
                expected,
                "case {case}: spatial hash disagreed with all-pairs on {boxes:?}"
            );
            // And when it does report one, the pair it names really overlaps.
            if let Some((i, j)) = hit {
                assert_ne!(i, j, "case {case}: a box cannot overlap itself");
                let ((amin, amax), (bmin, bmax)) = (boxes[i], boxes[j]);
                assert!(
                    amin.x < bmax.x
                        && bmin.x < amax.x
                        && amin.y < bmax.y
                        && bmin.y < amax.y
                        && amin.z < bmax.z
                        && bmin.z < amax.z,
                    "case {case}: reported pair {i}/{j} does not actually overlap"
                );
                saw_overlap += 1;
            } else {
                saw_clear += 1;
            }
        }
        // The sweep is worthless if it only ever produced one answer.
        assert!(saw_overlap > 20, "only {saw_overlap} overlapping cases -- not a real sweep");
        assert!(saw_clear > 20, "only {saw_clear} clear cases -- not a real sweep");
    }

    /// **The complexity itself, as a number rather than as a stopwatch.**
    ///
    /// `overlap_scan` reports how many pairs it actually tested, so this pins
    /// the growth directly and deterministically: an all-pairs loop over the
    /// same lattice tests `n*(n-1)/2`, which for the 41,472 pixel gates of a
    /// 192x108 render is ~8.6e8. A bounded number of neighbours per box means
    /// the count grows LINEARLY, so doubling the lattice must roughly double
    /// the work rather than quadruple it.
    #[test]
    fn the_overlap_scan_is_proportional_to_the_brick_count() {
        // Two collision-free lattices at the crate's real pitch, one twice the
        // other, in the shape `chip::finish` actually hands over.
        let lattice = |rows: i32| -> Vec<(Position, Position)> {
            let mut v = Vec::new();
            for row in 0..rows {
                for col in 0..64 {
                    for stage in 0..2 {
                        let p = lattice_pos_staged(col, row, stage, rows, GATE_HALF, STAGE_PITCH);
                        v.push((
                            Position {
                                x: p.x - GATE_HALF.x,
                                y: p.y - GATE_HALF.y,
                                z: p.z - GATE_HALF.z,
                            },
                            Position {
                                x: p.x + GATE_HALF.x,
                                y: p.y + GATE_HALF.y,
                                z: p.z + GATE_HALF.z,
                            },
                        ));
                    }
                }
            }
            v
        };
        let small = lattice(30);
        let big = lattice(60);
        let (hit_s, tested_s) = overlap_scan(&small);
        let (hit_b, tested_b) = overlap_scan(&big);
        assert!(hit_s.is_none() && hit_b.is_none(), "the lattice must be collision-free");
        assert_eq!(big.len(), 2 * small.len());

        // A constant number of neighbours per box. The bound is deliberately
        // loose (all-pairs on the small lattice alone would be ~7.4e6, three
        // orders of magnitude past this) so it pins the SHAPE of the growth
        // without pinning the bucket arithmetic.
        for (n, tested) in [(small.len(), tested_s), (big.len(), tested_b)] {
            assert!(
                tested <= 40 * n,
                "{tested} pair tests for {n} bricks is not proportional -- all-pairs \
                 would be {}",
                n * (n - 1) / 2
            );
        }
        // ...and doubling the lattice must not more than roughly double it.
        assert!(
            tested_b < 3 * tested_s,
            "doubling the lattice took {tested_b} tests against {tested_s} -- that is \
             super-linear growth"
        );
    }

    /// The scan must stay proportional on the ONE input shape a spatial hash
    /// can degrade on: many bricks piled into the same region. Nothing this
    /// crate builds looks like that, but a future caller's might, and a check
    /// that silently becomes quadratic again is exactly what this fix was for.
    /// The pile is genuinely overlapping, so the scan also has to return early.
    #[test]
    fn a_dense_pile_still_returns_promptly() {
        let boxes: Vec<(Position, Position)> = (0..5000)
            .map(|i| {
                (
                    Position { x: 0, y: 0, z: i % 3 },
                    Position { x: 10, y: 10, z: 4 + i % 3 },
                )
            })
            .collect();
        let (hit, tested) = overlap_scan(&boxes);
        assert!(hit.is_some(), "a pile of identical boxes must be caught");
        assert!(tested < 100, "must bail on the first overlapping pair, not scan the pile");
    }

    // --- rotated extents ----------------------------------------------------

    /// The one fact [`rotated_half`] is derived from: the world axis
    /// `direction` names is the one that carries the brick's own z extent.
    /// Everything else follows from "the other two authored extents fill the
    /// other two world axes".
    ///
    /// Written against a brick whose three authored extents are all DIFFERENT,
    /// so a permutation error cannot pass on symmetry -- unlike a gate, which
    /// is square in x/y and hides the roll entirely.
    #[test]
    fn the_direction_axis_carries_the_bricks_own_z_extent() {
        let authored = IntVector { x: 3, y: 7, z: 2 };
        for (dir, axis) in [
            (Direction::XPositive, 0),
            (Direction::XNegative, 0),
            (Direction::YPositive, 1),
            (Direction::YNegative, 1),
            (Direction::ZPositive, 2),
            (Direction::ZNegative, 2),
        ] {
            for rot in [Rotation::Deg0, Rotation::Deg90, Rotation::Deg180, Rotation::Deg270] {
                let got = rotated_half(authored, dir, rot);
                let on_axis = [got.x, got.y, got.z][axis];
                assert_eq!(
                    on_axis, authored.z,
                    "{dir:?}/{rot:?}: the direction axis must carry the authored z extent"
                );
                // Nothing is created or destroyed: the three extents are the
                // same multiset however the brick is turned.
                let mut before = [authored.x, authored.y, authored.z];
                let mut after = [got.x, got.y, got.z];
                before.sort_unstable();
                after.sort_unstable();
                assert_eq!(before, after, "{dir:?}/{rot:?}: a rotation permutes, it does not resize");
            }
        }
    }

    /// The default orientation must be the identity, or every unrotated brick
    /// in the crate would silently change size.
    #[test]
    fn the_default_orientation_leaves_the_authored_extent_alone() {
        assert_eq!(
            rotated_half(GATE_HALF, Direction::default(), Rotation::default()),
            GATE_HALF
        );
    }

    /// A quarter turn about the direction axis swaps the two extents that are
    /// NOT on it, and leaves the one that is.
    #[test]
    fn a_quarter_turn_swaps_the_two_non_direction_extents() {
        let authored = IntVector { x: 3, y: 7, z: 2 };
        assert_eq!(
            rotated_half(authored, Direction::XPositive, Rotation::Deg0),
            IntVector { x: 2, y: 3, z: 7 }
        );
        assert_eq!(
            rotated_half(authored, Direction::XPositive, Rotation::Deg90),
            IntVector { x: 2, y: 7, z: 3 },
            "the roll swaps y and z, and leaves x (the direction axis) alone"
        );
    }

    /// Why deriving the pixel gates' extent is safe without an in-game
    /// measurement: a gate is SQUARE in the two axes the roll could permute, so
    /// the one part of [`rotated_half`] that is convention rather than fact is
    /// unobservable for every brick this crate rotates.
    #[test]
    fn a_rotated_gate_is_square_in_the_two_non_direction_axes() {
        assert_eq!(
            GATE_HALF.x, GATE_HALF.y,
            "a gate's authored footprint is square, so no roll can change its extent"
        );
        let a = rotated_half(GATE_HALF, Direction::XPositive, Rotation::Deg0);
        let b = rotated_half(GATE_HALF, Direction::XPositive, Rotation::Deg90);
        assert_eq!(a, b, "every roll of a gate gives the same box");
        assert_eq!(
            a,
            IntVector { x: 2, y: 5, z: 5 },
            "an X_Positive gate is 4 units through on world x and 10 on y and z"
        );
    }

    /// [`STAGE_PITCH`] against the REAL depth of the brick it has to clear,
    /// rather than against a remembered 15. An upright gate is
    /// `2 * rotated_half.z` units through; the pitch has to exceed that or two
    /// stages sit face-flush (or worse, interpenetrate).
    #[test]
    fn stages_clear_each_other_at_the_stage_pitch() {
        let half = rotated_half(GATE_HALF, Direction::XPositive, Rotation::Deg90);
        let depth = 2 * half.z;
        assert_eq!(depth, 10, "an upright gate is 10 units through on z");
        assert!(
            STAGE_PITCH > depth,
            "STAGE_PITCH {STAGE_PITCH} must leave air between stages that are {depth} deep"
        );
        // And the flat pitch really would NOT have been enough -- which is why
        // this is a separate constant from CELL.
        assert!(CELL <= depth, "CELL {CELL} would put upright stages flush or overlapping");
    }
}
