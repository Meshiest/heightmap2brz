//! Brick-mode renderer. A screen of display bricks on the main grid, driven
//! by frame-major string arrays inside a microchip.
//!
//! Service gates (clock, arrays, detector) live at lattice stage 2, behind
//! both pixel stages, so they can never collide with pixel gates.
use super::chip;
use super::clock::{self, gate};
use super::layout::{GATE_HALF, STAGE_PITCH, lattice_pos_staged};
use super::pack::{self, BANK_FRAMES, HEX_STRIDE, pack};
use crate::video::Clip;
use brdb::{
    Direction, Rotation,
    AsBrdbValue, Brick, BrickSize, BrickType, IntVector, Position, Vector3f, WirePort, World,
    assets::{
        LiteralComponent,
        bricks::{B_MICROCHIP, PB_DEFAULT_MICRO_BRICK, PB_DEFAULT_SMOOTH_TILE},
        materials::{GLOW, PLASTIC},
    },
    schema::{WireArrayVariant, WireVariant},
};
use std::collections::HashMap;

pub const SUBSTRING: &str = "BrickComponentType_WireGraph_Expr_String_Substring";
pub const MAKE_COLOR_HEX: &str = "BrickComponentType_WireGraph_Expr_MakeColorHex";
pub const ARRAY_VAR: &str = "BrickComponentType_WireGraphPseudo_ArrayVar";
pub const ARRAY_GET: &str = "BrickComponentType_WireGraph_Exec_ArrayVar_Get";
pub const CHANGE_DETECTOR: &str = "BrickComponentType_WireGraph_Expr_ChangeDetectorExec";
pub const PROP_CHANGER: &str = "Component_BrickPropertyChanger";
pub const COMPARE_GE: &str = "BrickComponentType_WireGraph_Expr_CompareGreaterOrEqual";
pub const BRANCH: &str = "BrickComponentType_WireGraph_Exec_Branch";
pub const SELECT: &str = "BrickComponentType_WireGraph_Expr_Select";
pub const SUBTRACT: &str = "BrickComponentType_WireGraph_Expr_MathSubtract";

/// Lattice stage holding clock, array and detector gates.
const SERVICE_STAGE: i32 = 2;

/// Facing for the per-pixel `Substring`/`MakeColorHex` gates.
///
/// Identity (the default). An earlier attempt stood these gates on end
/// (`XPositive` + `Deg90`) and it had to be reverted: `Brick::local_bounds`
/// does NOT apply rotation, so the overlap check could only ever measure the
/// *authored* box. Every attempt to describe the rotated box by hand was
/// wrong, and the symptoms were all downstream of that -- gates embedded in
/// the plane, stages flush, and finally gates overlapping in-game while the
/// build-time check reported everything clear.
///
/// To try again, first MEASURE the rotated extent in-game rather than
/// deriving it, then set `PIXEL_GATE_HALF` to that and widen
/// `layout::STAGE_PITCH` to suit.
const PIXEL_GATE_FACING: Direction = Direction::XPositive;

/// Roll for those same gates. Identity -- see `PIXEL_GATE_FACING`.
const PIXEL_GATE_ROLL: Rotation = Rotation::Deg90;

/// Half-extent of a pixel gate AFTER [`PIXEL_GATE_FACING`]/[`PIXEL_GATE_ROLL`]
/// are applied. Both are currently identity, so this is just [`GATE_HALF`].
///
/// It stays a distinct constant because it is the thing that must change --
/// and must be *measured*, not derived -- if those two ever become a real
/// rotation. `Brick::local_bounds` ignores rotation, so this value is the
/// only description of a rotated gate the overlap check will ever see.
const PIXEL_GATE_HALF: IntVector = IntVector { x: 5, y: 2, z: 5 };

/// How far above the screen the interaction plane floats, in world units.
/// Applied to the entity location, which — because `chip::finish` centres the
/// chip contents on the grid origin — is exactly where the plane's middle
/// lands, independent of the grid's internal axis orientation.
const PLANE_HEIGHT_ABOVE_SCREEN: i32 = 15;

/// Which brick asset a display pixel renders as.
///
/// Both variants use a `Procedural` brick whose `size` is a half-extent, not
/// a diameter (see `Brick::local_bounds`) -- so `pixel_extent` in
/// [`AnimOptions`] means exactly what it says on every axis it drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayBrickStyle {
    /// Micro bricks (the default): cube-shaped, half-extent `{e, e, e}`. At
    /// the smallest legal `pixel_extent` (1) that's a brick 2 units wide.
    Micro,
    /// Smooth tiles (the "normal" mode): half-extent `{5*e, 5*e, 2}` --
    /// always 4 units tall, regardless of `pixel_extent`. The 5x follows this
    /// crate's long-standing normal-vs-micro brick-size convention (see
    /// `src/main.rs`'s heightmap path, which scales normal-brick footprints
    /// by 5 and leaves micro bricks unscaled): a 1-unit-footprint smooth tile
    /// is not a legal normal brick and the game silently drops it.
    SmoothTile,
}

impl DisplayBrickStyle {
    /// The half-extent of a display pixel's footprint (x and y) at this
    /// style's own scale, for a given `pixel_extent`. This is the single
    /// source [`brick_type`](Self::brick_type) reads its `size.x`/`size.y`
    /// from, so callers that need the real on-ground footprint (pitch, chip
    /// shell clearance) can read it straight off the constructed `BrickType`
    /// instead of re-deriving it -- the two can then never disagree.
    fn footprint_half_extent(self, extent: u16) -> u16 {
        match self {
            DisplayBrickStyle::Micro => extent,
            DisplayBrickStyle::SmoothTile => 5 * extent,
        }
    }

    /// The `BrickType` for a display pixel of this style, sized to `extent`
    /// on x/y (and, for [`Micro`](Self::Micro), z as well).
    fn brick_type(self, extent: u16) -> BrickType {
        let footprint = self.footprint_half_extent(extent);
        match self {
            DisplayBrickStyle::Micro => BrickType::Procedural {
                asset: PB_DEFAULT_MICRO_BRICK,
                size: BrickSize { x: footprint, y: footprint, z: footprint },
            },
            DisplayBrickStyle::SmoothTile => BrickType::Procedural {
                asset: PB_DEFAULT_SMOOTH_TILE,
                size: BrickSize { x: footprint, y: footprint, z: 2 },
            },
        }
    }
}

pub struct AnimOptions {
    pub alpha_threshold: u8,
    /// Half-extent of a display pixel, in game units, BEFORE the style's own
    /// scale is applied -- `DisplayBrickStyle::footprint_half_extent` turns
    /// this into the brick's real on-ground half-extent (unscaled for
    /// [`Micro`](DisplayBrickStyle::Micro), 5x for
    /// [`SmoothTile`](DisplayBrickStyle::SmoothTile), matching this crate's
    /// normal-vs-micro brick-size convention). `1` is the smallest legal
    /// value, giving a 2-unit-wide micro brick or a 10-unit-wide smooth tile.
    ///
    /// This also drives the display-brick pitch, `2 * footprint_half_extent`:
    /// pixels always tile flush (touching, never gapped or overlapping) at
    /// that pitch, for any style and any extent -- see `build_brick_world`.
    pub pixel_extent: u16,
    pub brick_style: DisplayBrickStyle,
    pub external_clock: bool,
    /// Render display pixels as `GLOW` at intensity 0 instead of `PLASTIC` at
    /// the default intensity.
    ///
    /// Intensity 0 is the *lowest* glow setting, not "off": the brick still
    /// emits its own colour rather than being lit by the world, so the screen
    /// stays readable in the dark and its colours stop shifting with the time
    /// of day. That matters more here than for a static build, because every
    /// pixel's colour is driven by a wire each frame and any ambient tint sits
    /// on top of the colour the graph actually set.
    ///
    /// Mirrors the heightmap path's `--glow` (see `opt/generate.rs`), which
    /// pairs `GLOW` with `material_intensity: 0` the same way.
    pub glow: bool,
    /// Entries per wire array before frames spill into another bank.
    ///
    /// Defaults to `BANK_FRAMES`. Lowering it is how the multi-bank graph is
    /// tested without building a 65 536-frame clip.
    pub bank_size: usize,
}

impl Default for AnimOptions {
    fn default() -> Self {
        Self {
            alpha_threshold: 128,
            pixel_extent: 1,
            brick_style: DisplayBrickStyle::Micro,
            external_clock: false,
            glow: false,
            bank_size: BANK_FRAMES,
        }
    }
}

/// A pixel is emitted only if it is opaque enough in at least one frame.
fn visible(clip: &Clip, col: u32, row: u32, threshold: u8) -> bool {
    clip.frames.iter().any(|f| f.get_pixel(col, row).0[3] >= threshold)
}

pub fn build_brick_world(clip: &Clip, opts: &AnimOptions) -> Result<World, String> {
    // A zero-frame clip (reachable via `--start`/`--duration` past the
    // source's end, or the GUI's Start slider dragged past it -- see
    // `video::scale::resample_fps`, which returns `Ok` with an empty
    // `frames` rather than erroring) must not fall through to a "successful"
    // build: with no display bricks and `clock::build_clock` inlining
    // `Modulo.InputB = frame_count as f64 = 0.0`, the save would open fine
    // and silently divide by zero in-game on every tick. Caught once here so
    // both the CLI and GUI entry points, which both funnel through this
    // function, get the same clear error instead of a broken file.
    if clip.frames.is_empty() {
        return Err(
            "clip has 0 frames -- nothing to render (check --start/--duration, or the GUI's \
             Start/Duration, against the source's length)"
                .to_string(),
        );
    }

    let chunks = pack(clip, opts.alpha_threshold)?;
    let (w, h) = (clip.width as i32, clip.height as i32);

    let mut world = World::new();
    world.meta.bundle.description = "Animation generated from image frames".to_string();

    // --- 1. Display bricks on the main grid ---------------------------------
    let mut brick_of: HashMap<usize, usize> = HashMap::new();
    // `pixel_extent` is a pre-scale half-extent (see `AnimOptions::pixel_extent`'s
    // doc); `.max(1)` guards the degenerate case the CLI/GUI floors already
    // prevent -- a 0 extent would collapse every display brick to a
    // zero-size point. The real on-ground footprint depends on the style
    // (SmoothTile scales by 5, Micro does not), so it's read straight off the
    // `BrickType` this same call just built -- never re-derived separately --
    // so the pitch and the chip shell's clearance below can never disagree
    // with the brick that's actually placed.
    let extent = opts.pixel_extent.max(1);
    let brick_type = opts.brick_style.brick_type(extent);
    let BrickType::Procedural { size: brick_size, .. } = &brick_type else {
        unreachable!("DisplayBrickStyle::brick_type always builds a Procedural brick")
    };
    let footprint = brick_size.x as i32;
    // Two adjacent pixels' bricks only touch -- neither gapped nor
    // overlapping -- at twice the real footprint.
    let pitch = 2 * footprint;
    for row in 0..h {
        for col in 0..w {
            if !visible(clip, col as u32, row as u32, opts.alpha_threshold) {
                continue;
            }
            let (brick, id) = Brick {
                asset: brick_type.clone(),
                // z is the brick's own half-height, so every style rests its
                // underside on z=0. A hardcoded 2 grounded a smooth tile
                // (half-height 2, spanning 0..4) but left a micro brick
                // (half-height 1) floating a unit off the ground at 1..3.
                position: Position { x: col * pitch, y: row * pitch, z: brick_size.z as i32 },
                // Intensity 0 is the dimmest glow, not "off" -- see
                // `AnimOptions::glow`. The non-glow arm keeps 5, matching the
                // heightmap path's default rather than `Brick::default()`.
                material: if opts.glow { GLOW } else { PLASTIC },
                material_intensity: if opts.glow { 0 } else { 5 },
                ..Default::default()
            }
            .with_component(LiteralComponent::new(PROP_CHANGER))
            .with_id_split();
            world.add_brick(brick);
            brick_of.insert((row * w + col) as usize, id);
        }
    }

    // --- 2. The chip --------------------------------------------------------
    // The shell sits BESIDE the screen, never on it. The very first in-game
    // spike stacked it in Z instead (a default display brick's z span at z=2
    // overlapped the shell's), and the game silently DROPPED one of them: a
    // 3-brick L with unconnectable wires. Placing it beside the screen on X
    // sidesteps that class of bug entirely -- `assert_bricks_dont_overlap`
    // only needs ONE axis to be clear of overlap, and `chip::finish` still
    // asserts main-grid non-overlap as the safety net either way.
    //
    // The display style and its footprint are both options now (no longer a
    // hardcoded 5,5,6 default), so the clearance is computed from the real
    // half-extents in play rather than a fixed guess: column 0's own display
    // bricks span x in `[-footprint, footprint]` (their center is x=0), and
    // the shell's own half-extent on x is read from `local_bounds` -- not
    // re-hardcoded -- so this stays correct if that ever changes. Landing the
    // shell's outer face exactly on `-footprint` makes the two spans share at
    // most a face on x (flush, not overlapping -- see
    // `layout::assert_bricks_dont_overlap`'s strict-inequality doc comment),
    // for every row, since every other display brick sits at an x further
    // from the shell than column 0's. `footprint` (not the pre-scale
    // `extent`) is what actually matters here -- a SmoothTile's real x
    // half-extent is `5*extent`, and using the unscaled `extent` would leave
    // the shell overlapping the screen for that style.
    let shell_half_x = {
        let (min, max) = Brick { asset: B_MICROCHIP, ..Default::default() }.local_bounds();
        (max.x - min.x) / 2
    };
    let shell_x = -(shell_half_x + footprint);
    //
    // The plane_extent placeholder below is only a placeholder, not a real
    // bound: no caller can know the true `PlaneExtent` before every gate/pin
    // exists (see `chip::plane_extent_for`'s docstring on why halving the
    // lattice span -- what this used to do -- silently clips the clock's
    // control pins and most pixel gates on anything but a tiny screen).
    // `chip::finish` now recomputes and grows this to fit every placed brick,
    // via `Chip::recompute_plane_extent`, so this only needs to be a legal
    // (non-degenerate) starting value.
    // Park the interaction plane directly above the middle of the screen.
    //
    // `chip::finish` centres the chip's contents on the grid origin, so the
    // plane's middle coincides with `entity.location` — which means this
    // world point IS where the plane ends up, with no dependence on how the
    // grid's local axes map to world axes. That mapping has proven hard to
    // pin down (successive attempts to shift gates "up" moved them sideways
    // instead), so anything positional that can avoid relying on it should.
    let screen_span = |n: i32| (n - 1).max(0) * pitch;
    let plane_anchor = Vector3f {
        x: screen_span(w) as f32 / 2.0,
        y: screen_span(h) as f32 / 2.0,
        // Start at the screen's actual top face -- the display bricks rest
        // their underside on z=0, so that is twice their half-height, which
        // differs per style (a smooth tile is 4 units tall, a default micro
        // brick 2) -- then rise clear of it.
        z: (2 * brick_size.z as i32 + PLANE_HEIGHT_ABOVE_SCREEN) as f32,
    };
    let mut chip = chip::new_chip(
        &mut world,
        Position { x: shell_x, y: 0, z: 2 },
        plane_anchor,
        IntVector { x: 5, y: 5, z: 5 },
    );

    // Service gates sit behind BOTH pixel stages, so they must use the same
    // stage pitch the (upright, 10-deep) pixel gates do — with the flat
    // `CELL` pitch they landed inside stage 1's depth and collided.
    let service = |col: i32, row: i32| {
        lattice_pos_staged(col, row, SERVICE_STAGE, h, GATE_HALF, STAGE_PITCH)
    };

    // --- 3. Frame index source ---------------------------------------------
    let frame_index = if opts.external_clock {
        let pin = chip::add_input_pin(&mut chip, "Frame", service(0, -1));
        chip::pin_source(pin, true)
    } else {
        clock::build_clock(&mut world, &mut chip, clip.fps, clip.frames.len(), service(0, -2))
            .frame_index
    };

    // --- 4. Exec source -----------------------------------------------------
    let detector = gate(
        &mut chip,
        "B_1x1_Gate_Expr_ChangeDetectorExec",
        CHANGE_DETECTOR,
        service(0, -4),
        vec![],
    );
    world.add_wire_connection(
        frame_index.clone(),
        WirePort::new(detector, CHANGE_DETECTOR, "Input"),
    );

    // --- 5. Arrays and gets, one per (chunk, bank) --------------------------
    let bank_size = opts.bank_size.max(1);
    let n_banks = clip.frames.len().div_ceil(bank_size).max(1);

    // Per-bank index. Bank 0 reads the frame index directly; bank k subtracts
    // k*bank_size so its own array is addressed from zero.
    //
    // MathSubtract is typed float and Get.Index takes an int, but both
    // operands are integral so the difference is exact -- the same coercion
    // the clock already relies on for `BitwiseOR |0`.
    let mut index_of_bank = Vec::with_capacity(n_banks);
    index_of_bank.push(frame_index.clone());
    for k in 1..n_banks {
        let sub = gate(&mut chip, "B_1x1_Gate_Expr_MathSubtract", SUBTRACT,
            service(k as i32, -6), vec![(
                "InputB",
                Box::new(WireVariant::Number((k * bank_size) as f64)) as Box<dyn AsBrdbValue>,
            )]);
        world.add_wire_connection(frame_index.clone(), WirePort::new(sub, SUBTRACT, "InputA"));
        index_of_bank.push(WirePort::new(sub, SUBTRACT, "Output"));
    }

    // Boundary comparators. `ge[k-1]` is true once the frame index reaches
    // bank k, which is exactly `Select`'s `bSelectB` sense: true picks InputB,
    // the later bank.
    let mut ge = Vec::with_capacity(n_banks.saturating_sub(1));
    for k in 1..n_banks {
        let cmp = gate(&mut chip, "B_1x1_Gate_Expr_CompareGreaterOrEqual", COMPARE_GE,
            service(k as i32, -7), vec![(
                "InputB",
                Box::new(WireVariant::Int((k * bank_size) as i64)) as Box<dyn AsBrdbValue>,
            )]);
        world.add_wire_connection(frame_index.clone(), WirePort::new(cmp, COMPARE_GE, "InputA"));
        ge.push(WirePort::new(cmp, COMPARE_GE, "bOutput"));
    }

    // get_of[bank][chunk]
    let mut get_of: Vec<Vec<usize>> = vec![Vec::with_capacity(chunks.len()); n_banks];
    for (ci, chunk) in chunks.iter().enumerate() {
        for (bi, frames) in pack::bank_frames(&chunk.frames, bank_size).iter().enumerate() {
            let col = ((ci * n_banks + bi) * 2) as i32;
            let array = gate(&mut chip, "B_1x1_Gate_Variable_Array", ARRAY_VAR,
                service(col, -5), vec![(
                    "Value",
                    Box::new(WireArrayVariant::StringArray(frames.to_vec()))
                        as Box<dyn AsBrdbValue>,
                )]);
            let get = gate(&mut chip, "B_1x1_Gate_Exec_ArrayVar_Get", ARRAY_GET,
                service(col + 1, -5), vec![]);
            world.add_wire_connection(
                WirePort::new(array, ARRAY_VAR, "ArrayVarRef"),
                WirePort::new(get, ARRAY_GET, "ArrayVarRef"),
            );
            world.add_wire_connection(
                index_of_bank[bi].clone(),
                WirePort::new(get, ARRAY_GET, "Index"),
            );
            get_of[bi].push(get);
        }
    }

    // Exec: branches cascade at the FRONT, so exactly one bank's chain runs
    // and no exec input ever takes two sources. Branching per chunk and
    // rejoining would require exec fan-in, which is untested here.
    //
    // With n_banks == 1 this emits no branch at all and the chain is
    // byte-identical to the pre-spillover wiring.
    let mut exec_src = WirePort::new(detector, CHANGE_DETECTOR, "OnChanged");
    for bi in 0..n_banks {
        let entry = if bi + 1 < n_banks {
            let br = gate(&mut chip, "B_1x1_Gate_Exec_Branch", BRANCH,
                service(bi as i32, -8), vec![]);
            world.add_wire_connection(ge[bi].clone(), WirePort::new(br, BRANCH, "bCond"));
            world.add_wire_connection(exec_src, WirePort::new(br, BRANCH, "Exec"));
            // true -> keep descending; false -> this bank
            exec_src = WirePort::new(br, BRANCH, "ExecOutA");
            WirePort::new(br, BRANCH, "ExecOutB")
        } else {
            exec_src.clone()
        };
        let mut prev = entry;
        for &get in &get_of[bi] {
            world.add_wire_connection(prev, WirePort::new(get, ARRAY_GET, "Exec"));
            prev = WirePort::new(get, ARRAY_GET, "ExecOut");
        }
    }

    // Value: one select per chunk per boundary, cascading. For a frame in
    // bank j, ge[0..j] are true so select j picks bank j, and every later
    // select passes it through unchanged.
    let mut value_of_chunk = Vec::with_capacity(chunks.len());
    for ci in 0..chunks.len() {
        let mut value = WirePort::new(get_of[0][ci], ARRAY_GET, "Value");
        for bi in 1..n_banks {
            let sel = gate(&mut chip, "B_1x1_Gate_Expr_Select", SELECT,
                service((ci * n_banks + bi) as i32, -9), vec![]);
            world.add_wire_connection(ge[bi - 1].clone(), WirePort::new(sel, SELECT, "bSelectB"));
            world.add_wire_connection(value, WirePort::new(sel, SELECT, "InputA"));
            world.add_wire_connection(
                WirePort::new(get_of[bi][ci], ARRAY_GET, "Value"),
                WirePort::new(sel, SELECT, "InputB"),
            );
            value = WirePort::new(sel, SELECT, "Output");
        }
        value_of_chunk.push(value);
    }

    // --- 6. Two gates per surviving pixel -----------------------------------
    for (ci, chunk) in chunks.iter().enumerate() {
        let value = value_of_chunk[ci].clone();
        for local in 0..chunk.pixel_count {
            let idx = chunk.first_pixel + local;
            let Some(&brick_id) = brick_of.get(&idx) else {
                continue; // culled: slot reserved in the string, no gates
            };
            let (col, row) = ((idx as i32) % w, (idx as i32) / w);

            // Stage 1 (above) = Substring, stage 0 (bottom) = MakeColorHex.
            // This layering was verified in-game on the phase-1 spike; keep it.
            //
            // The per-pixel gates face +X and are rolled a quarter turn so
            // they point downward, rather than facing +Z at the viewer like
            // the clock's gates. Purely cosmetic: orientation moves no
            // centre, and `GATE_HALF` remains the collision half-size, so
            // the overlap check and plane sizing are untouched.
            let sub = clock::gate_oriented(
                &mut chip,
                "B_1x1_Gate_Expr_String_Substring",
                SUBSTRING,
                lattice_pos_staged(col, row, 1, h, PIXEL_GATE_HALF, STAGE_PITCH),
                vec![
                    // bare i64, not WireVariant::Int — this gate's schema
                    // fields are plain scalars; a WireVariant fails at encode
                    // time with UnimplementedCast.
                    (
                        "Start",
                        Box::new((local * HEX_STRIDE) as i64) as Box<dyn AsBrdbValue>,
                    ),
                    ("Length", Box::new(HEX_STRIDE as i64) as Box<dyn AsBrdbValue>),
                ],
                PIXEL_GATE_FACING,
                PIXEL_GATE_ROLL,
                PIXEL_GATE_HALF,
            );
            let mkcolor = clock::gate_oriented(
                &mut chip,
                "B_1x1_Gate_Expr_MakeColorHex",
                MAKE_COLOR_HEX,
                lattice_pos_staged(col, row, 0, h, PIXEL_GATE_HALF, STAGE_PITCH),
                vec![],
                PIXEL_GATE_FACING,
                PIXEL_GATE_ROLL,
                PIXEL_GATE_HALF,
            );

            world.add_wire_connection(value.clone(), WirePort::new(sub, SUBSTRING, "Input"));
            world.add_wire_connection(
                WirePort::new(sub, SUBSTRING, "Output"),
                WirePort::new(mkcolor, MAKE_COLOR_HEX, "Hex"),
            );
            world.add_wire_connection(
                WirePort::new(mkcolor, MAKE_COLOR_HEX, "Output"),
                WirePort::new(brick_id, PROP_CHANGER, "Color"),
            );
        }
    }

    // --- 7. Publish -------------------------------------------------------
    chip::finish(&mut world, chip)?;
    // Must be last: it registers every component type and port name actually
    // used, and it must see all bricks, grids and wires first.
    world.register_used_components();
    Ok(world)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The user-facing default: micro bricks at the smallest legal extent, a
    /// 2-unit-wide brick.
    #[test]
    fn default_style_is_micro_at_the_smallest_extent() {
        let opts = AnimOptions::default();
        assert_eq!(opts.brick_style, DisplayBrickStyle::Micro);
        assert_eq!(opts.pixel_extent, 1);
        assert_eq!(
            opts.brick_style.brick_type(opts.pixel_extent),
            BrickType::Procedural {
                asset: PB_DEFAULT_MICRO_BRICK,
                size: BrickSize { x: 1, y: 1, z: 1 },
            },
            "the default must be a 1-half-extent (2-unit-wide) micro brick"
        );
    }

    /// The "normal" mode: smooth tiles are always 4 units tall (half-extent
    /// z=2), regardless of the requested extent, while x/y follow `5*extent`
    /// -- normal bricks are 5-unit-footprint, unlike micro bricks (see
    /// `src/main.rs`'s `if micro { 1 } else { 5 }` convention this mirrors).
    #[test]
    fn smooth_tile_footprint_is_5x_the_extent_and_always_4_units_tall() {
        for extent in [1u16, 3, 12] {
            match DisplayBrickStyle::SmoothTile.brick_type(extent) {
                BrickType::Procedural { asset, size } => {
                    assert_eq!(asset, PB_DEFAULT_SMOOTH_TILE);
                    assert_eq!(size.x, 5 * extent, "x must follow 5x the extent");
                    assert_eq!(size.y, 5 * extent, "y must follow 5x the extent");
                    assert_eq!(size.z, 2, "z half-extent must stay 2 (4 units tall)");
                }
                other => panic!("expected a Procedural brick type, got {other:?}"),
            }
        }
    }

    /// Micro bricks are cubes: x/y/z all follow the extent together, unlike
    /// smooth tiles where z is pinned.
    #[test]
    fn micro_is_a_cube_that_follows_the_extent_on_every_axis() {
        for extent in [1u16, 4, 9] {
            match DisplayBrickStyle::Micro.brick_type(extent) {
                BrickType::Procedural { asset, size } => {
                    assert_eq!(asset, PB_DEFAULT_MICRO_BRICK);
                    assert_eq!(size, BrickSize { x: extent, y: extent, z: extent });
                }
                other => panic!("expected a Procedural brick type, got {other:?}"),
            }
        }
    }
}
