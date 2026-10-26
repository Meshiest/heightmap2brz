//! Prefab metadata in the shape that the game writes.

use crate::anim::layout::rotated_half;
use brdb::{Brick, IntVector, Position, PrefabIntVec3, PrefabJson, World};

/// Mark a world as a prefab, with the bounds in the shape the game writes.
///
/// Each prefab the game writes has its bounds centered on zero, with whole
/// number half extents, and `addedGlobalGridOffset` moves the bricks into
/// that box. `World::make_prefab` puts the center at the brick positions
/// instead, and a span with an odd length gives a center such as 58.5. When
/// a prefab in that shape is placed on flat ground, the game deletes the
/// lowest bricks.
///
/// A span with an odd length gets one unit more at the top. The bottom of the
/// box thus stays level with the lowest bricks. The game lets the top brick
/// go one unit out of the box in this case, but every brick here stays in it.
pub fn make_prefab(world: &mut World) {
    world.make_prefab();
    let Some((min, max)) = oriented_bounds(&world.bricks) else {
        return;
    };
    let half = |lo: i32, hi: i32| (hi - lo + 1) / 2;
    let half = Position::new(half(min.x, max.x), half(min.y, max.y), half(min.z, max.z));
    let mut prefab = PrefabJson::from_bounds(Position::new(-half.x, -half.y, -half.z), half);
    prefab.added_global_grid_offset = PrefabIntVec3 {
        x: -half.x - min.x,
        y: -half.y - min.y,
        z: -half.z - min.z,
    };
    world.meta.prefab = Some(prefab);
}

/// The bounds of `bricks`, with the direction and rotation of each brick
/// applied. `Brick::local_bounds` does not apply them, so a ramp turned 90
/// degrees gets the wrong x and y extents.
fn oriented_bounds(bricks: &[Brick]) -> Option<(Position, Position)> {
    bricks
        .iter()
        .map(|b| {
            let (lo, hi) = b.local_bounds();
            let authored = IntVector {
                x: (hi.x - lo.x) / 2,
                y: (hi.y - lo.y) / 2,
                z: (hi.z - lo.z) / 2,
            };
            let h = rotated_half(authored, b.direction, b.rotation);
            let p = b.position;
            (
                Position::new(p.x - h.x, p.y - h.y, p.z - h.z),
                Position::new(p.x + h.x, p.y + h.y, p.z + h.z),
            )
        })
        .reduce(|(amin, amax), (bmin, bmax)| {
            (
                Position::new(amin.x.min(bmin.x), amin.y.min(bmin.y), amin.z.min(bmin.z)),
                Position::new(amax.x.max(bmax.x), amax.y.max(bmax.y), amax.z.max(bmax.z)),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use brdb::{
        BrickSize, BrickType, Rotation,
        assets::bricks::{PB_DEFAULT_BRICK, PB_DEFAULT_RAMP},
    };

    fn brick(asset: brdb::BString, size: (u16, u16, u16), at: (i32, i32, i32)) -> Brick {
        Brick {
            asset: BrickType::Procedural {
                asset,
                size: BrickSize::new(size.0, size.1, size.2),
            },
            position: Position::new(at.0, at.1, at.2),
            ..Default::default()
        }
    }

    /// Each brick with the offset applied, as the box the game reads.
    fn placed(world: &World) -> (Position, Position, PrefabJson) {
        let prefab = world.meta.prefab.clone().expect("a prefab");
        let o = prefab.added_global_grid_offset;
        let (min, max) = oriented_bounds(&world.bricks).expect("bricks");
        (
            Position::new(min.x + o.x, min.y + o.y, min.z + o.z),
            Position::new(max.x + o.x, max.y + o.y, max.z + o.z),
            prefab,
        )
    }

    /// The heightmap layout that lost its bottom row: the bricks span z 24 to
    /// 93, so the plain bounds had the center 58.5.
    #[test]
    fn the_box_is_centered_on_zero_and_the_bottom_is_level() {
        let mut world = World::new();
        world.add_bricks([
            brick(PB_DEFAULT_BRICK, (5, 5, 2), (-5, 0, 26)),
            brick(PB_DEFAULT_BRICK, (5, 5, 33), (5, 0, 60)),
        ]);
        make_prefab(&mut world);
        let (min, max, prefab) = placed(&world);
        let bounds = prefab.pivots.bounds_pivot;

        assert_eq!(world.meta.bundle.level_type, "Prefab");
        assert_eq!((bounds.center.x, bounds.center.y, bounds.center.z), (0.0, 0.0, 0.0));
        assert_eq!((bounds.half_extent.x, bounds.half_extent.y, bounds.half_extent.z), (10.0, 5.0, 35.0));
        assert_eq!(min.z, -35, "the lowest bricks must sit on the bottom of the box");
        assert_eq!(max.z, 34, "the odd span gets its extra unit at the top");
        assert_eq!((min.x, max.x, min.y, max.y), (-10, 10, -5, 5));
    }

    #[test]
    fn a_turned_ramp_uses_its_turned_extents() {
        let mut ramp = brick(PB_DEFAULT_RAMP, (15, 5, 2), (0, 0, 2));
        ramp.rotation = Rotation::Deg90;
        let mut world = World::new();
        world.add_bricks([ramp]);
        make_prefab(&mut world);
        let half = world.meta.prefab.expect("a prefab").pivots.bounds_pivot.half_extent;

        assert_eq!((half.x, half.y, half.z), (5.0, 15.0, 2.0));
    }
}
