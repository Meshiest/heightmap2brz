//! Microchip construction. A microchip is a 1x1 shell brick on the main grid
//! plus a separate entity owning its own brick grid, joined by a link table.
//!
//! The convention this module encodes, which `brdb` does not model: pins are
//! ordinary bricks placed *inside* the chip's own inner grid, each carrying a
//! `BrickComponentType_Internal_Microchip{Input,Output}` component with a
//! `PortLabel` string. Both pin kinds behave like a rerouter — they expose
//! exactly one input port `RER_Input` and one output port `RER_Output` on
//! themselves — so a wire from *outside* the chip must terminate on the
//! inner pin brick's own component, never on the chip shell brick. This was
//! cross-checked against:
//!   - the wirescript compiler's emitter
//!     (`bearilog/crates/wirescript/src/emit.rs`, `build_port_index` and the
//!     rerouter wiring around it), which remaps every chip port to
//!     `(inner_brick_id, Internal_Microchip{Input,Output}, RER_{Input,Output})`;
//!   - the live game inventory dump
//!     (`bearilog/crates/wirescript/data/logic_gate_inventory.simple.json`),
//!     which lists both `B_1x1_Gate_MicrochipInput` and
//!     `B_1x1_Gate_MicrochipOutput` with exactly one input (`RER_Input`) and
//!     one output (`RER_Output`).
//!
//! Because a pin's inner brick lives in a different grid (entity) than
//! whatever it's wired to outside the chip, `World::add_wire_connection`
//! emits a `RemoteWirePortSource` automatically — no special API call is
//! needed here.
use super::layout::{CELL, GATE_HALF, assert_bricks_dont_overlap, assert_no_overlap};
use brdb::{
    AsBrdbValue, Brick, Entity, IntVector, Position, Quat4f, Vector3f, WirePort, World,
    assets::{LiteralComponent, bricks, entities},
};

pub const MICROCHIP_INPUT: &str = "BrickComponentType_Internal_MicrochipInput";
pub const MICROCHIP_OUTPUT: &str = "BrickComponentType_Internal_MicrochipOutput";

/// A quarter turn clockwise about the world **up (Z)** axis, yawing the whole
/// interaction plane so it faces the other way round.
///
/// `World::add_microchip` leaves the grid entity's rotation at identity, so
/// there is nothing to compose with — this IS the plane's orientation.
///
/// A quaternion for angle θ about Z is `(0, 0, sin(θ/2), cos(θ/2))`; θ = −90°
/// (clockwise seen from above) gives `sin(−45°) = −1/√2`, `cos(−45°) = 1/√2`.
/// Negate `z` for counter-clockwise. Isolated behind one constant precisely
/// because a wrong quaternion misorients the plane *silently* — no error, and
/// nothing a test can catch.
const PLANE_SPIN: Quat4f = Quat4f {
    x: 0.0,
    y: 0.0,
    z: -std::f32::consts::FRAC_1_SQRT_2,
    w: std::f32::consts::FRAC_1_SQRT_2,
};

/// A microchip under construction: the outer shell brick (already pushed to
/// `world.bricks` by `World::add_microchip`) plus the inner grid's bricks,
/// accumulated here until `finish` publishes them.
pub struct Chip {
    pub brick_id: usize,
    pub entity_id: usize,
    /// The inner grid's entity. Private: `finish` owns publishing it, and a
    /// caller swapping it out would orphan every brick added here.
    entity: Entity,
    /// The `collapsed` flag baked into the entity by `add_microchip`.
    /// `new_chip` doesn't expose it as a parameter (every current caller
    /// wants `false`), but it's carried here rather than re-hardcoded in
    /// `recompute_plane_extent` so the two can never drift apart.
    collapsed: bool,
    /// The inner grid's bricks, and every one of their (center, half-size)
    /// bounds. These two MUST stay in lockstep — `finish`'s overlap check
    /// and `plane_extent_for` are both blind to a brick missing from
    /// `placed`, and a brick they can't see is one the game silently drops
    /// at load (taking its wires with it). They are private, and
    /// [`Chip::add_brick`] is the single place that writes to either, so
    /// the invariant is structural rather than remembered.
    bricks: Vec<Brick>,
    placed: Vec<(Position, IntVector)>,
}

impl Chip {
    /// Add a brick to the chip's inner grid, minting and returning its brick
    /// id (the id used to name a wire endpoint).
    ///
    /// This is the ONLY way to put a brick in a chip. `half` is the brick's
    /// half-size, which the caller must supply: it cannot be derived
    /// reliably, because `Brick::local_bounds` guesses `(5, 5, 6)` for any
    /// basic brick, and every gate this crate emits is actually
    /// [`GATE_HALF`] = `(5, 5, 2)`. The center is read from `brick.position`
    /// rather than passed separately, so there is no way for the recorded
    /// bounds to disagree with where the brick actually is.
    pub fn add_brick(&mut self, brick: Brick, half: IntVector) -> usize {
        let (brick, id) = brick.with_id_split();
        self.placed.push((brick.position, half));
        self.bricks.push(brick);
        id
    }

    /// Every inner brick's (center, half-size), in insertion order.
    ///
    /// Read-only on purpose — this is the input to [`plane_extent_for`] and
    /// mirrors exactly what `finish` will collision-check.
    pub fn placed(&self) -> &[(Position, IntVector)] {
        &self.placed
    }

    /// Recompute this chip's published `PlaneExtent` from every inner brick
    /// placed so far (via [`plane_extent_for`]). This is authoritative — it
    /// replaces whatever `plane_extent` [`new_chip`] was given, rather than
    /// merging with it — because by the time [`finish`] calls this, every
    /// gate and pin has been placed, so this is the only point that knows the
    /// true extent. A stale caller-supplied value (`new_chip` bakes one into
    /// the entity before any brick exists, so it can only ever be a guess)
    /// surviving into the published extent would silently reintroduce an
    /// oversized plane.
    fn recompute_plane_extent(&mut self) {
        // DO NOT translate the bricks. The gate lattice is origin-anchored
        // (`layout::lattice_pos` emits only non-negative coordinates) and it
        // must stay that way: **negative inner-grid coordinates break bricks
        // in-game**.
        //
        // Centring the content on the origin was tried twice, and both times
        // the *display* gates vanished while the clock survived. That split
        // is the tell: `lattice_pos` puts `x = (h-1-row)*CELL + half.x`, so
        // pixel gates occupy the LOW x end of the lattice and the service
        // gates the HIGH end — centring pushes exactly the pixel gates
        // negative and leaves the service gates positive.
        //
        // So the plane is moved onto the content instead, via `PlaneCenter`.
        let (center, extent) = plane_bounds_for(&self.placed);
        debug_assert!(
            self.placed
                .iter()
                .all(|(p, h)| p.x - h.x >= 0 && p.y - h.y >= 0 && p.z - h.z >= 0),
            "inner-grid bricks must stay non-negative; negative coordinates break them in-game"
        );
        self.entity.data = entities::microchip_grid_entity(self.collapsed, center, extent);

        // Centre the plane over the display.
        //
        // The lattice is origin-anchored (0..N), so its midpoint sits half a
        // plane-width away from the grid origin and the plane hangs off the
        // anchor instead of straddling it. The bricks themselves cannot be
        // moved to fix that — shifting them toward the origin drives the
        // low-x pixel gates negative, which deletes them in-game — so the
        // correction goes on the entity.
        //
        // It has to account for `PLANE_SPIN`, a -90 degree yaw about world Z,
        // which maps a grid-local (x, y) onto world (y, -x). So the x-extent
        // corrects world Y and the y-extent corrects world X, with the sign
        // flip on one of them. Applying them unswapped (as this did before)
        // pushes the plane along the wrong axes.
        self.entity.location.x -= extent.y as f32;
        self.entity.location.y += extent.x as f32;
    }
}

/// Start a microchip: spawns the outer shell brick on `world`'s main grid
/// and registers the microchip link. Returns a `Chip` the caller populates
/// with pins (and, in later tasks, gates) before calling `finish`.
///
/// `plane_extent` is only a placeholder, not the final published value:
/// `add_microchip` needs *something* to bake into the grid entity before any
/// brick exists to measure, but [`finish`] fully replaces it — via
/// [`Chip::recompute_plane_extent`], run after every gate/pin has been added —
/// with the true extent. Callers should pass a small, legal (non-degenerate)
/// value here, not a deliberately generous one: nothing about that generosity
/// survives to the published entity.
pub fn new_chip(
    world: &mut World,
    position: Position,
    entity_location: Vector3f,
    plane_extent: IntVector,
) -> Chip {
    // Start collapsed: a pasted animation is meant to be looked at, not
    // edited, and an expanded chip's interaction plane is large enough
    // (see `plane_extent_for`) to be visually in the way of the screen it
    // drives. `Chip` carries this through to `recompute_plane_extent`, which
    // rebuilds the entity data, so the two can never disagree.
    const COLLAPSED: bool = true;
    let (brick_id, entity_id, (entity, bricks)) =
        world.add_microchip(position, entity_location, plane_extent, COLLAPSED);
    let mut entity = entity;
    entity.rotation = PLANE_SPIN;
    let mut chip = Chip {
        brick_id,
        entity_id,
        entity,
        collapsed: COLLAPSED,
        bricks: Vec::new(),
        placed: Vec::new(),
    };
    // brdb 0.9.1 always hands back an empty vec here, but absorb it through
    // the same gate rather than trusting that: a brick that arrived without
    // its bounds registered is exactly the invisible-overlap bug this type
    // exists to prevent. `local_bounds` is the best available half-size for
    // a brick this module did not construct.
    for brick in bricks {
        let (min, max) = brick.local_bounds();
        let half = IntVector {
            x: (max.x - min.x) / 2,
            y: (max.y - min.y) / 2,
            z: (max.z - min.z) / 2,
        };
        chip.add_brick(brick, half);
    }
    chip
}

fn add_pin(chip: &mut Chip, label: &str, pos: Position, is_input: bool) -> usize {
    let (asset, class) = if is_input {
        (bricks::B_MICROCHIP_INPUT, MICROCHIP_INPUT)
    } else {
        (bricks::B_MICROCHIP_OUTPUT, MICROCHIP_OUTPUT)
    };
    let brick = Brick {
        asset,
        position: pos,
        ..Default::default()
    }
    .with_component(LiteralComponent::new(class).with_data([(
        "PortLabel",
        Box::new(label.to_string()) as Box<dyn AsBrdbValue>,
    )]));
    chip.add_brick(brick, GATE_HALF)
}

/// Add an input pin: fed a value from outside the chip on its `RER_Input`
/// port; that value appears on the same brick's `RER_Output` for wiring to
/// internal gates (see `pin_source`).
pub fn add_input_pin(chip: &mut Chip, label: &str, pos: Position) -> usize {
    add_pin(chip, label, pos, true)
}

/// Add an output pin: internal gates write a value to its `RER_Input` port
/// (see `pin_target`); that value appears on the same brick's `RER_Output`
/// for wiring to the outside world.
pub fn add_output_pin(chip: &mut Chip, label: &str, pos: Position) -> usize {
    add_pin(chip, label, pos, false)
}

/// The port that carries a pin's value onward: an output pin's value out to
/// the world, or an input pin's value into the chip's internal graph.
pub fn pin_source(brick_id: usize, is_input: bool) -> WirePort {
    let class = if is_input {
        MICROCHIP_INPUT
    } else {
        MICROCHIP_OUTPUT
    };
    WirePort::new(brick_id, class, "RER_Output")
}

/// The port a value is written to: an input pin's value from outside the
/// chip, or an output pin's value from the chip's internal graph.
pub fn pin_target(brick_id: usize, is_input: bool) -> WirePort {
    let class = if is_input {
        MICROCHIP_INPUT
    } else {
        MICROCHIP_OUTPUT
    };
    WirePort::new(brick_id, class, "RER_Input")
}

/// Publish the chip's inner grid. Uses `World::add_brick_grid`, which
/// applies the `-Position::CHUNK_HALF` shift; pushing to `world.grids`
/// directly would land every gate half a chunk away.
///
/// Recomputes and applies the entity's `PlaneExtent` (see
/// [`Chip::recompute_plane_extent`]) from every brick's position — deferred
/// to here, rather than trusted from whatever `new_chip` was given, because
/// no caller can know the true extent before every gate and pin exists.
///
/// A prior version of this function also translated every inner brick so its
/// bounding box straddled the origin before recomputing the extent, on the
/// theory that `layout::lattice_pos`'s positive-octant-only placement was
/// wasting half of `PlaneExtent`'s span (since `PlaneCenter` is always
/// `(0,0,0)`). That centering shipped, the plane got roughly 2x tighter as
/// predicted, and several display gates (`MakeColorHex`, `Substring`, the
/// array-index `ArrayVar_Get`) went missing in-game. A follow-up margin fix
/// to `plane_extent_for` (see its doc comment) didn't bring them back, which
/// rules out boundary clipping as the cause — the remaining difference from
/// the last known-working state is that centering put inner bricks at
/// negative grid coordinates, which `layout::lattice_pos` never produces on
/// its own. So the centering call was removed here and inner bricks are once
/// again left exactly where `lattice_pos` puts them: non-negative, origin-anchored,
/// with `PlaneExtent` consequently ~2x the content's half-span. That bloat is
/// cosmetic and accepted for now. Do not reintroduce centering (or otherwise
/// let inner bricks land at negative coordinates) without first verifying
/// in-game that negative inner-grid coordinates are actually safe — this is
/// exactly the kind of thing that looks like free tightening and quietly
/// breaks gates again.
///
/// Checks BOTH grids for overlap before publishing. The inner check alone is
/// not enough: the first in-game failure was on the main grid, where the
/// microchip shell was stacked inside a display brick's z span and the game
/// silently dropped one of them. Doing this here rather than leaving it to
/// callers means a renderer cannot forget it.
pub fn finish(world: &mut World, mut chip: Chip) -> Result<(), String> {
    chip.recompute_plane_extent();
    assert_no_overlap(&chip.placed)?;
    // The shell brick is already in world.bricks by this point (add_microchip
    // pushes it), so this sees the shell-versus-screen case.
    assert_bricks_dont_overlap(&world.bricks)?;
    world.add_brick_grid(chip.entity, chip.bricks);
    Ok(())
}

/// The half-size `PlaneExtent` needs to contain every placed brick, with a
/// full cell of clearance so no brick's outer face lands exactly on the
/// plane boundary.
///
/// `PlaneExtent` is a half-size measured from `PlaneCenter`, which
/// `World::add_microchip` always sets to `(0, 0, 0)` (in-game default is
/// `(14, 14, 2)`). So, per axis, the extent must reach past the *outer face*
/// of the farthest brick — `|center| + half-size` — not just its center.
///
/// An earlier draft of this function computed `max(|coord|) / 2 + 5`. That
/// halving is wrong for this crate's lattice: `layout::lattice_pos` places
/// every brick at a non-negative coordinate starting at `half` and growing
/// outward from the origin, so the lattice is never centered around
/// `PlaneCenter` — the extent has to cover the full span, not half of it.
/// Halving would under-size the plane for any chip with more than a couple
/// of pins/gates, silently clipping bricks past `extent - half-size` at
/// in-game placement (the save file is unaffected — clipping is purely a
/// property of how the game's `BrickGridMicrochipActor` bounds interaction
/// with the plane — but a clipped brick's wires still dangle).
///
/// A later regression made the outer-face reach *exactly* equal the
/// published extent, with no margin at all: an edge gate's outer face sat
/// precisely on the plane boundary, and the game clipped it in-game (the
/// save file itself looked fine — this is purely an in-game placement
/// interaction, same as the halving bug above). So every axis gets a flat
/// [`CELL`] (one gate cell, 10 units) of headroom added past the farthest
/// outer face, giving an edge gate a full cell of clearance rather than
/// landing on the line. The `.max(5)` floor is kept underneath that for a
/// degenerate (empty) chip, though `CELL` alone already exceeds it.
pub fn plane_extent_for(placed: &[(Position, IntVector)]) -> IntVector {
    plane_bounds_for(placed).1
}

/// The `(PlaneCenter, PlaneExtent)` pair that tightly wraps `placed`.
///
/// `PlaneExtent` is a half-size measured from `PlaneCenter`, so a plane that
/// contains content is `center ± extent`. The gate lattice
/// ([`super::layout::lattice_pos`]) is **origin-anchored**: every coordinate
/// is non-negative, running `0..N`. Pinning `PlaneCenter` to `(0,0,0)` and
/// solving for an extent that reaches `N` therefore describes a plane
/// spanning `-N..N` — twice as large as the content on every axis, with the
/// content crammed into a single quadrant. That is exactly what a user
/// reported seeing in-game.
///
/// Centering the *bricks* on the origin was tried first and reverted: it made
/// every inner coordinate negative and coincided with gates disappearing
/// in-game (see [`finish`]). This function fixes the same problem without
/// moving a single brick — it leaves the verified-working origin-anchored
/// placement alone and instead moves the *plane* onto the content, which is
/// what `PlaneCenter` is for.
///
/// The plane is measured in the inner grid's own unshifted coordinates, i.e.
/// the same space `placed` is recorded in — NOT the `-CHUNK_HALF`-shifted
/// space bricks are stored in by `World::add_brick_grid`. The working spike
/// establishes this: its content sat at `0..60` with `PlaneCenter (0,0,0)`
/// and an extent of 60, which contains the content only in unshifted space.
pub fn plane_bounds_for(placed: &[(Position, IntVector)]) -> (IntVector, IntVector) {
    let axis = |pf: fn(&Position) -> i32, hf: fn(&IntVector) -> i32| -> (i32, i32) {
        let lo = placed.iter().map(|(p, h)| pf(p) - hf(h)).min().unwrap_or(0);
        let hi = placed.iter().map(|(p, h)| pf(p) + hf(h)).max().unwrap_or(0);
        // Round the centre toward zero-safe integer math: the half-span is
        // computed from the same lo/hi so it always reaches both faces even
        // when the span is odd.
        let center = (lo + hi) / 2;
        let half = (hi - lo + 1) / 2;
        // One full gate cell of clearance so no brick's face lands on the
        // plane boundary.
        (center, (half + CELL).max(5))
    };
    let (cx, ex) = axis(|p| p.x, |h| h.x);
    let (cy, ey) = axis(|p| p.y, |h| h.y);
    // z is PINNED, not derived from the content.
    //
    // Local Z is up inside the chip, and the plane is a flat surface the
    // gates sit ON — not a volume they live inside. Deriving z from the
    // content did both of the wrong things at once: `extent.z` grew to
    // swallow every gate (so they rendered buried inside the chip), and
    // `center.z` tracked them, so raising the whole lattice raised the plane
    // with it and moved nothing on screen. Pinning z means
    // `layout::STAGE_BASE_Z` actually lifts the gates clear of the surface.
    (
        IntVector { x: cx, y: cy, z: 0 },
        IntVector { x: ex, y: ey, z: PLANE_HALF_THICKNESS },
    )
}

/// Half-thickness of the chip's interaction plane, in grid units.
///
/// The plane is a surface, not a box: this matches the `2` the game itself
/// uses for a hand-placed microchip's `PlaneExtent` z. Gates clear it by
/// sitting at `layout::STAGE_BASE_Z` and above.
pub const PLANE_HALF_THICKNESS: i32 = 2;

#[cfg(test)]
mod tests {
    use super::*;

    /// `plane_bounds_for` returns the content's plain midpoint as
    /// `PlaneCenter`, plus a half-span extent with a one-cell margin. Any
    /// pull-back needed to stop an origin-anchored lattice hanging off its
    /// anchor is applied to `entity.location` in `recompute_plane_extent`,
    /// not folded in here.
    #[test]
    fn the_plane_centre_is_the_content_midpoint() {
        let placed = vec![(Position { x: 0, y: 25, z: 0 }, GATE_HALF)];
        let (center, extent) = plane_bounds_for(&placed);
        assert_eq!(
            extent.y,
            5 + CELL,
            "extent is the half-span plus one cell of margin"
        );
        // y spans 20..30, midpoint 25.
        assert_eq!(center.y, 25, "the plane centre is the content midpoint");
        assert!(
            center.y - extent.y < 20 && 30 < center.y + extent.y,
            "the content clears the plane on both faces"
        );
    }

    #[test]
    fn plane_bounds_cover_every_axis_independently() {
        let placed = vec![
            (Position { x: 40, y: 0, z: 0 }, GATE_HALF),
            (Position { x: 0, y: 5, z: 0 }, GATE_HALF),
            (Position { x: 0, y: 0, z: 8 }, GATE_HALF),
        ];
        let (center, extent) = plane_bounds_for(&placed);
        // x spans -5..45 -> midpoint 20, half-span 25, extent 25 + CELL.
        assert_eq!(
            (center.x, extent.x),
            (20, 25 + CELL),
            "x driven by its own axis alone"
        );
        // z is pinned, NOT derived: the plane is a flat surface at z=0 that
        // the gates sit on top of, so it deliberately does not stretch to
        // contain them.
        assert_eq!(
            (center.z, extent.z),
            (0, PLANE_HALF_THICKNESS),
            "z is pinned to the plane surface, not grown to fit the content"
        );
    }

    #[test]
    fn the_extent_covers_the_whole_lattice_span() {
        // Mirrors the real renderer's shape: an origin-anchored lattice.
        let placed: Vec<_> = (0..6)
            .flat_map(|r| {
                (0..4).map(move |c| {
                    (
                        Position {
                            x: r * CELL + 5,
                            y: c * CELL + 5,
                            z: 2,
                        },
                        GATE_HALF,
                    )
                })
            })
            .collect();
        let (_anchor, extent) = plane_bounds_for(&placed);
        for (axis, e, lo, hi) in [("x", extent.x, 0, 6 * CELL), ("y", extent.y, 0, 4 * CELL)] {
            let half_span = (hi - lo + 1) / 2;
            assert!(
                e >= half_span,
                "{axis} extent {e} must cover the lattice half-span {half_span}"
            );
        }
    }

    #[test]
    fn plane_extent_for_an_empty_chip_is_exactly_one_cell_margin() {
        // Zero span on x and y, so those are just the margin itself. The
        // `.max(5)` floor is dead in practice there (CELL=10 already exceeds
        // it) but stays as a defensive backstop underneath it. z is pinned to
        // the plane's own half-thickness and never derived from content.
        assert_eq!(
            plane_extent_for(&[]),
            IntVector {
                x: CELL,
                y: CELL,
                z: PLANE_HALF_THICKNESS
            }
        );
    }

    // A prior version of this module also centered every inner brick's
    // bounding box on the origin before recomputing `PlaneExtent` (tightening
    // it roughly 2x, since `layout::lattice_pos` otherwise leaves the whole
    // lattice in the positive octant and `PlaneCenter` is always `(0,0,0)`).
    // That was reverted — see `finish`'s doc comment for the full story —
    // because it put inner bricks at negative grid coordinates and display
    // gates went missing in-game. The tests that exercised `center_on_origin`
    // and `Chip::translate` were removed along with the methods themselves;
    // `plane_extent_for` is still exercised directly above against
    // non-negative, origin-anchored coordinates, which is what `finish` now
    // always publishes against.

    // The full guarantee — that `finish` actually publishes an extent that
    // contains every brick for a real render, not just that `plane_extent_for`'s
    // math is right in isolation — needs to be checked against the entity
    // `finish` actually writes. That requires decoding a real save (the
    // entity's `PlaneExtent` lives behind an opaque `Arc<Box<dyn
    // BrdbComponent>>` this module has no reason to downcast), so it lives in
    // `tests/anim_world.rs` as `chip_plane_extent_contains_every_brick_in_a_real_render`,
    // built on a real `build_brick_world` output round-tripped through `.brz`.
}
