//! Placement of gates inside a microchip's inner grid.
//!
//! The chip is a 3D mirror of the screen: a pixel's gates sit at that pixel's
//! own row/column, with the gate stages stacked along depth. Wires stay short
//! and a misplaced pixel is visible by eye when the chip is opened.
use brdb::{IntVector, Position};

/// Gate lattice pitch, in game units. Every gate this crate emits is
/// half-size 5x5x2, i.e. 10x10x4 units, so a uniform 10-unit pitch clears
/// on all three axes (the minimum non-overlapping depth pitch is 4).
pub const CELL: i32 = 10;

/// Half-size shared by every gate brick used by the animation renderer.
pub const GATE_HALF: IntVector = IntVector { x: 5, y: 5, z: 2 };

/// Depth pitch between pipeline stages.
///
/// [`CELL`] while the gates lie flat: a flat gate is 4 units through, so a
/// 10-unit pitch leaves 6 units of air between stages.
///
/// This is a separate constant because it stops being right the moment gates
/// are rotated. Standing them on end makes them 10 units through, and a
/// `CELL` pitch then puts every stage face flush against the next. Rotation
/// was attempted and reverted (see `bricks::PIXEL_GATE_FACING`) precisely
/// because the rotated bounding box could not be established — so if gates
/// are ever rotated again, this must be widened to match the *measured*
/// rotated depth, not a guessed one.
pub const STAGE_PITCH: i32 = 15;

/// Height offset applied to the whole staged lattice, lifting every brick in
/// the chip off the interaction plane instead of starting flush against it.
///
/// Local **Z is up** inside the chip — the plane lies flat, so its normal is
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
/// too tight for upright ones — see [`STAGE_PITCH`].
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

/// Errors if any two placed bricks intersect.
///
/// The game silently DROPS overlapping bricks at load, which orphans their
/// components and dangles every wire into them — producing a save that opens
/// fine and does nothing. This is the cheapest place to catch that.
pub fn assert_no_overlap(placed: &[(Position, IntVector)]) -> Result<(), String> {
    for (i, (pa, ha)) in placed.iter().enumerate() {
        for (pb, hb) in placed.iter().skip(i + 1) {
            let overlaps = (pa.x - pb.x).abs() < ha.x + hb.x
                && (pa.y - pb.y).abs() < ha.y + hb.y
                && (pa.z - pb.z).abs() < ha.z + hb.z;
            if overlaps {
                return Err(format!(
                    "bricks overlap: ({},{},{}) half ({},{},{}) vs ({},{},{}) half ({},{},{})",
                    pa.x, pa.y, pa.z, ha.x, ha.y, ha.z, pb.x, pb.y, pb.z, hb.x, hb.y, hb.z
                ));
            }
        }
    }
    Ok(())
}

/// Errors if any two bricks on a grid intersect, using each brick's own
/// asset-aware bounds rather than a caller-supplied half-size.
///
/// This exists because [`assert_no_overlap`] only ever sees a microchip's
/// INNER grid, so it cannot catch a collision on the main grid — which is
/// exactly where the first real one happened: the `B_1x1_Microchip` shell
/// (half-extent 5,5,2) sat at z=6 on top of a display brick at z=2 whose
/// default half-extent is 5,5,6, so their z spans [4,8] and [-4,8] overlapped
/// on an identical x/y footprint. The game dropped one and a 2x2 screen came
/// back as a 3-brick L with its wires unconnectable.
///
/// Sizes differ per asset — a default procedural brick is 5,5,6, gates are
/// 5,5,2, a reroute node is 1,1,1 — so this reads `local_bounds()` per brick
/// instead of assuming a uniform size.
pub fn assert_bricks_dont_overlap(bricks: &[brdb::Brick]) -> Result<(), String> {
    let bounds: Vec<_> = bricks.iter().map(|b| (b.local_bounds(), b)).collect();
    for (i, ((amin, amax), a)) in bounds.iter().enumerate() {
        for ((bmin, bmax), b) in bounds.iter().skip(i + 1) {
            // Strict inequality: flush neighbours share a face and are legal.
            let overlaps = amin.x < bmax.x
                && bmin.x < amax.x
                && amin.y < bmax.y
                && bmin.y < amax.y
                && amin.z < bmax.z
                && bmin.z < amax.z;
            if overlaps {
                return Err(format!(
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
                ));
            }
        }
    }
    Ok(())
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
}
