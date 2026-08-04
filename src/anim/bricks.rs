//! Brick-mode renderer. A screen of display bricks on the main grid, driven
//! by frame-major string arrays inside a microchip.
//!
//! Service gates (clock, arrays, detector) live at lattice stage 2, behind
//! both pixel stages, so they can never collide with pixel gates.
use super::cascade;
use super::chip;
use super::clock::{self, gate};
use super::controls;
use super::layout::{GATE_HALF, STAGE_PITCH, lattice_pos_staged, rotated_half};
use super::pack::{self, BANK_FRAMES, HEX_STRIDE};
use crate::progress::{FrameTotal, Progress};
use crate::video::stream::FrameSource;
use brdb::{
    Direction, Rotation,
    AsBrdbValue, Brick, BrickSize, BrickType, IntVector, Position, Vector3f, WirePort, World,
    assets::{
        LiteralComponent,
        bricks::{B_MICROCHIP, PB_DEFAULT_MICRO_BRICK, PB_DEFAULT_SMOOTH_TILE},
        materials::{GLOW, PLASTIC},
    },
    schema::WireArrayVariant,
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

/// Facing for the per-pixel `Substring`/`MakeColorHex` gates: stood on end,
/// point down, rather than facing +Z at the viewer like the clock's gates.
/// Cosmetic; [`layout::STAGE_PITCH`] already allows the 15 units an upright
/// (10-deep) gate needs.
const PIXEL_GATE_FACING: Direction = Direction::XPositive;

/// Roll for those same gates: a quarter turn about [`PIXEL_GATE_FACING`]'s
/// axis. A gate is square in x/y, so this changes no extent.
const PIXEL_GATE_ROLL: Rotation = Rotation::Deg90;

/// Half-extent of a pixel gate after facing/roll are applied. Derived via
/// [`super::layout::rotated_half`], not hand-written: a hand value can
/// disagree with the facing (the authored thin axis lands on world x here,
/// not y), which `layout::assert_no_overlap`, `chip`'s non-negativity checks,
/// and [`super::layout::lattice_pos_staged`]'s position offset all rely on.
const PIXEL_GATE_HALF: IntVector = rotated_half(GATE_HALF, PIXEL_GATE_FACING, PIXEL_GATE_ROLL);

/// World units the interaction plane's start floats above the screen top face.
/// Only a starting anchor: `chip::recompute_plane_extent` moves
/// `entity.location` again, so this is not where the plane's middle ends up.
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
    /// always 4 units tall, regardless of `pixel_extent`. The 5x matches this
    /// crate's normal-brick convention; a 1-unit-footprint smooth tile is not
    /// a legal normal brick and the game silently drops it.
    SmoothTile,
}

impl DisplayBrickStyle {
    /// Half-extent of a display pixel's footprint (x and y) at this style's
    /// own scale. Single source [`brick_type`](Self::brick_type) reads its
    /// `size.x`/`size.y` from, so the two can never disagree.
    fn footprint_half_extent(self, extent: u16) -> u16 {
        match self {
            DisplayBrickStyle::Micro => extent,
            DisplayBrickStyle::SmoothTile => 5 * extent,
        }
    }

    /// The `BrickType` for a display pixel of this style, sized to `extent`
    /// on x/y (and, for [`Micro`](Self::Micro), z as well).
    pub fn brick_type(self, extent: u16) -> BrickType {
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
    /// Half-extent of a display pixel, in game units, before the style's own
    /// scale is applied (unscaled for Micro, 5x for SmoothTile). `1` is the
    /// smallest legal value: a 2-unit-wide micro brick or 10-unit smooth tile.
    /// Also drives the display-brick pitch, `2 * footprint_half_extent`, so
    /// pixels always tile flush at any style and extent.
    pub pixel_extent: u16,
    pub brick_style: DisplayBrickStyle,
    pub external_clock: bool,
    /// Repeat the clip forever (`true`, the default) or stop on its last
    /// frame (`false`). Sets `Timer.Limit` (see
    /// [`crate::anim::clock::stop_limit`]); changes no gate, wire or brick
    /// count. Inert under [`Self::external_clock`], which builds no timer.
    pub loop_playback: bool,
    /// Pre-generate three physical labelled button bricks on the main grid,
    /// wired into the clock's `Pause`/`Restart`/`Resume` pins, so a fresh
    /// render is controllable out of the box (see [`super::controls`]).
    ///
    /// `true` by default. Adds [`controls::CONTROL_BRICKS`] main-grid bricks
    /// and [`controls::CONTROL_WIRES`] wires; adds no inner-grid gate. Inert
    /// under [`Self::external_clock`], which exposes no pins to drive.
    pub control_buttons: bool,
    /// Render display pixels as `GLOW` at intensity 0 instead of `PLASTIC` at
    /// the default intensity. Intensity 0 is the lowest glow setting, not
    /// "off": the brick still emits its own colour rather than being lit by
    /// the world, so it stays readable in the dark and its colour does not
    /// shift with time of day.
    pub glow: bool,
    /// Convert sRGB frame pixels to linear before hex-encoding (see
    /// `pack::Packer::linearize`). Off by default.
    pub srgb_to_linear: bool,
    /// Entries per wire array before frames spill into another bank.
    ///
    /// Defaults to `BANK_FRAMES`. Lowering it is how the multi-bank graph is
    /// tested without building a 65 536-frame clip.
    pub bank_size: usize,
    /// Text mode only: quantize every frame to at most this many colours with
    /// a median-cut palette ([`super::palette::Palette`]) before encoding.
    /// `0` (default) means no quantization; both brick encodings ignore this
    /// field, since they spend a fixed cost per pixel no palette can change.
    pub colors: usize,
    /// Text mode only: the font, glyph and component geometry a text render
    /// uses. Lives on `AnimOptions` rather than as an extra parameter because
    /// all three renderers share one signature. Both brick encodings ignore
    /// it; defaults to `FontPreset::MonaspaceArgon.options(1.0)`.
    pub text: crate::text::TextOptions,
    /// A subtitle track to render across the bottom of the screen, or `None`
    /// for no subtitles at all. `None` (default) is a hard gate: all three
    /// renderers skip [`super::subtitle_display`] entirely, so a render
    /// without subtitles produces exactly the graph it produces today.
    ///
    /// An `Arc` because the GUI hands the same parsed track to a render
    /// thread while keeping it for the cost readout, avoiding a re-parse or
    /// deep-copy of a few thousand cues.
    pub subtitles: Option<std::sync::Arc<crate::subs::Subtitles>>,
    /// How much bigger a subtitle line is than one row of the screen.
    /// Defaults to [`super::subtitle_display::DEFAULT_SUBTITLE_SCALE`]. Inert
    /// while [`Self::subtitles`] is `None`.
    pub subtitle_scale: f32,
    /// How many world units the subtitle anchor is lifted "up the picture"
    /// from its bare bottom-centre baseline. Defaults to
    /// [`super::subtitle_display::DEFAULT_SUBTITLE_LIFT`], calibrated by eye
    /// in text mode only. Applied along +z for text mode's upright wall
    /// and -y for both brick encodings' flat, ground-facing screen (see
    /// `subtitle_extent`'s doc); the brick-mode axis is unverified by eye.
    /// Inert while [`Self::subtitles`] is `None`.
    ///
    /// Both brick encodings accept any lift, including one that carries the
    /// anchor to a negative `y` -- brdb round-trips negative main-grid
    /// coordinates fine. Text mode rejects a lift pushing its anchor below
    /// `z = 0`, since that mode keeps its whole main grid non-negative by
    /// construction and a subtitle is the one brick outside that translation.
    pub subtitle_lift: f32,
    /// The source timestamp of output frame 0, in seconds -- i.e. the
    /// `--start` the frame source was opened with. Only the subtitle timing
    /// reads this: frame `i` is at source time `source_start_s + i / fps`
    /// (see `scale::FpsStream`), and a subtitle file is in source time, so
    /// `Subtitles::per_frame` needs the same offset or every cue lands
    /// `--start` seconds early. `0.0` (default) is correct for any source
    /// rendered from its beginning.
    pub source_start_s: f64,
}

impl Default for AnimOptions {
    fn default() -> Self {
        Self {
            alpha_threshold: 128,
            pixel_extent: 1,
            brick_style: DisplayBrickStyle::Micro,
            external_clock: false,
            // Looping, which is what every render did before the flag
            // existed. The default has to stay this way or an unchanged
            // command line would quietly produce a different save.
            loop_playback: true,
            // ON: a fresh render ships pausable/restartable/resumable. Turned
            // off with `--no-control-buttons` (a default-on off-switch).
            control_buttons: true,
            glow: false,
            srgb_to_linear: false,
            bank_size: BANK_FRAMES,
            colors: 0,
            text: crate::text::FontPreset::MonaspaceArgon.options(1.0),
            subtitles: None,
            subtitle_scale: super::subtitle_display::DEFAULT_SUBTITLE_SCALE,
            subtitle_lift: super::subtitle_display::DEFAULT_SUBTITLE_LIFT,
            source_start_s: 0.0,
        }
    }
}

/// On-ground geometry of one display pixel, derived once from the options so
/// the pitch, chip shell clearance, and interaction plane anchor can never
/// disagree with the brick actually placed. Shared by both brick encodings
/// ([`build_brick_world`] and [`super::color_bricks::build_color_array_world`]).
pub struct ScreenGeometry {
    /// A display brick's real on-ground half-extent on x and y, after the
    /// style's own scale (unscaled for `Micro`, 5x for `SmoothTile`). Read off
    /// the constructed `BrickType`, never re-derived.
    pub footprint: i32,
    /// A display brick's own half-height. Every style rests its underside on
    /// z=0 by sitting at this z, so it is also the brick's centre height.
    pub half_height: i32,
    /// Distance between neighbouring pixels' centres: `2 * footprint`, the
    /// only pitch at which they touch without gapping or overlapping.
    pub pitch: i32,
}

/// Where a subtitle sits relative to a screen of display bricks: laid across
/// the picture's bottom edge, standing clear of its surface. Shared by both
/// brick encodings, derived from the geometry [`add_display_bricks`] already
/// returned.
///
/// A display-brick screen lies flat: `x` is the image column, `y` the image
/// row, `z` a constant. The anchor is centred on `x`, at the bottom of `y`
/// (rows increase toward the picture's bottom, so `--subtitle-lift`'s axis
/// here is `-y`, the opposite of text mode's `+z`), and one anchor-cube above
/// the screen's top face on `z` so glyphs draw a full cube clear of the
/// picture rather than z-fighting it or overlapping a display brick.
///
/// `face` is `FACE_Z_POSITIVE` because the screen presents its top (`+Z`) to
/// the viewer; the default `X_Positive` face would stand the subtitle
/// edge-on and unreadable. Which way "along the line" points within that
/// plane is unverified by eye for this flat orientation -- only text mode's
/// vertical wall has been checked.
///
/// A negative `y` here (any picture 4 rows or shorter, with default options)
/// is legal: brdb round-trips negative main-grid coordinates exactly, and the
/// chip shell itself already sits at a negative main-grid `x` in every
/// render (see [`new_screen_chip`]). Text mode still rejects its equivalent,
/// because that mode keeps its whole main grid non-negative by construction.
pub fn subtitle_extent(
    geometry: &ScreenGeometry,
    cols: i32,
    rows: i32,
    lift: f32,
) -> Result<super::subtitle_display::ScreenExtent, String> {
    let baseline_y = (rows.max(1) - 1) * geometry.pitch + geometry.footprint;
    let lift_units = super::subtitle_display::lift_units(lift);
    let y = baseline_y - lift_units;
    Ok(super::subtitle_display::ScreenExtent {
        anchor: Position {
            x: (cols.max(1) - 1) * geometry.pitch / 2,
            y,
            z: 2 * geometry.half_height + crate::text::ANCHOR_CUBE_HALF,
        },
        row_height: geometry.pitch as f32,
        face: crate::text::FACE_Z_POSITIVE,
    })
}

/// Place one display brick per visible pixel on `world`'s main grid, returning
/// the pixel-index-to-brick-id map and the geometry every later placement
/// depends on.
///
/// `visible` is row-major (`index = row * width + col`), the order both
/// packers produce and this loop walks. A pixel that was never opaque enough
/// in any frame is skipped entirely -- no brick, and (in either encoding) no
/// gates.
pub fn add_display_bricks(
    world: &mut World,
    opts: &AnimOptions,
    w: i32,
    h: i32,
    visible: &[bool],
) -> (HashMap<usize, usize>, ScreenGeometry) {
    // .max(1): a 0 extent collapses the brick to a point.
    let extent = opts.pixel_extent.max(1);
    let brick_type = opts.brick_style.brick_type(extent);
    let BrickType::Procedural { size: brick_size, .. } = &brick_type else {
        unreachable!("DisplayBrickStyle::brick_type always builds a Procedural brick")
    };
    let geometry = ScreenGeometry {
        footprint: brick_size.x as i32,
        half_height: brick_size.z as i32,
        // Two adjacent pixels' bricks only touch -- neither gapped nor
        // overlapping -- at twice the real footprint.
        pitch: 2 * brick_size.x as i32,
    };

    let mut brick_of: HashMap<usize, usize> = HashMap::new();
    for row in 0..h {
        for col in 0..w {
            if !visible[(row * w + col) as usize] {
                continue;
            }
            let (brick, id) = Brick {
                asset: brick_type.clone(),
                // z is the brick's own half-height, so every style rests its
                // underside on z=0 (a hardcoded 2 would float a micro brick).
                position: Position {
                    x: col * geometry.pitch,
                    y: row * geometry.pitch,
                    z: geometry.half_height,
                },
                // Intensity 0 is the dimmest glow, not "off". Non-glow arm
                // keeps 5, matching the heightmap path's default.
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
    (brick_of, geometry)
}

/// Start the microchip that drives a screen of the given size and geometry.
///
/// The shell sits beside the screen, on x, never stacked on top of it -- a
/// stacked shell can overlap a display brick's z span, and the game silently
/// drops one of the two bricks. `assert_bricks_dont_overlap` only needs one
/// axis clear, and `chip::finish` still asserts main-grid non-overlap as the
/// safety net either way.
///
/// Clearance is computed from the real half-extents in play, not a fixed
/// guess: the shell's x half-extent comes from `local_bounds`, and the real
/// `footprint` (not the pre-scale `pixel_extent`) is what matters -- a
/// SmoothTile's real x half-extent is `5*extent`, and using the unscaled
/// value would leave the shell overlapping the screen.
///
/// The entity location computed here aims the interaction plane at the
/// screen's middle but does not land there: `chip::finish` does not centre
/// the chip's contents on the grid origin, and `chip::recompute_plane_extent`
/// shifts `entity.location` again afterward. This is only the plane's
/// starting point.
///
/// The `plane_extent` passed to `chip::new_chip` is a placeholder, not a real
/// bound -- no caller can know the true extent before every gate/pin exists.
/// `chip::finish` recomputes and grows it to fit every placed brick.
pub fn new_screen_chip(
    world: &mut World,
    w: i32,
    h: i32,
    geometry: &ScreenGeometry,
) -> chip::Chip {
    let shell_half_x = {
        let (min, max) = Brick { asset: B_MICROCHIP, ..Default::default() }.local_bounds();
        (max.x - min.x) / 2
    };
    let shell_x = -(shell_half_x + geometry.footprint);
    let screen_span = |n: i32| (n - 1).max(0) * geometry.pitch;
    let plane_anchor = Vector3f {
        x: screen_span(w) as f32 / 2.0,
        y: screen_span(h) as f32 / 2.0,
        // Screen's top face is at 2*half_height (bricks rest their underside
        // on z=0); rise clear of it from there.
        z: (2 * geometry.half_height + PLANE_HEIGHT_ABOVE_SCREEN) as f32,
    };
    chip::new_chip(
        world,
        Position { x: shell_x, y: 0, z: 2 },
        plane_anchor,
        IntVector { x: 5, y: 5, z: 5 },
    )
}

/// Streams `source` into a wired, animated display-brick [`World`].
///
/// A cancelled `progress` (checked once per frame during decode, once per
/// pixel during the build) returns an empty `World`, never a partial one --
/// deliberately not an error, since cancellation is a normal outcome. It is
/// on the caller to re-check `progress.is_cancelled()` and discard the
/// result: an empty `World` still encodes to a valid save, so nothing
/// downstream can otherwise tell a cancelled render from a real one.
/// `gui::util::deliver_world_unless_cancelled` is that check. Callers that
/// cannot cancel use `NoProgress`, whose `is_cancelled` is always `false`.
pub fn build_brick_world(
    source: &dyn FrameSource,
    opts: &AnimOptions,
    progress: &mut dyn Progress,
) -> Result<World, String> {
    let info = source.info();
    let (w, h) = (info.width as i32, info.height as i32);

    // One fused streaming pass builds both the per-chunk frame strings and
    // the per-pixel visibility bitmap; no frame is ever retained (see
    // `pack::Packer`'s doc). A `stream.next()` error is fatal and propagates
    // immediately -- treating it as end-of-clip would silently drop the tail.
    // `FrameTotal` falls back to the source's estimate when no exact count is
    // available, and says which it is.
    FrameTotal::new(info.frame_count_hint, source.frame_count_estimate())
        .begin(progress, "packing frames");
    let mut packer = pack::Packer::new(info.width, info.height, opts.alpha_threshold, HEX_STRIDE)
        .linearize(opts.srgb_to_linear);
    // Wrapped in a closure so `progress.finish()` below runs on every exit,
    // including a mid-stream error -- otherwise the bar is left "stuck".
    // Cancellation breaks this same loop rather than erroring, so `stream`
    // (and, for `FfmpegSource`, its child process) still gets dropped when
    // the closure returns. Checked once per frame, not per pixel.
    let seen: Result<u64, String> = (|| {
        let mut stream = source.open()?;
        let mut seen: u64 = 0;
        while let Some(frame) = stream.next()? {
            packer.push_frame(&frame)?;
            // Borrows the frame's buffer -- no copy. `Progress::frame`'s
            // default body is a no-op.
            progress.frame(frame.width(), frame.height(), frame.as_raw());
            seen += 1;
            progress.tick(seen);
            if progress.is_cancelled() {
                break;
            }
        }
        Ok(seen)
    })();
    progress.finish();
    let seen = seen?;

    // Cancelled: stop here, before a single brick is placed. Deliberately
    // ahead of the zero-frame guard below, so a cancel can never surface as
    // that error.
    if progress.is_cancelled() {
        return Ok(World::new());
    }

    let (chunks, visible) = packer.finish();
    let frame_count = seen as usize;

    // A zero-frame source (`--start`/`--duration` past the source's end)
    // must not fall through: `clock::build_clock` would inline
    // `Modulo.InputB = 0.0`, and the save would divide by zero in-game on
    // every tick.
    if frame_count == 0 {
        return Err(
            "clip has 0 frames -- nothing to render (check --start/--duration, or the GUI's \
             Start/Duration, against the source's length)"
                .to_string(),
        );
    }

    let mut world = World::new();
    world.meta.bundle.description = "Animation generated from image frames".to_string();

    // --- 1. Display bricks on the main grid ---------------------------------
    let (brick_of, geometry) = add_display_bricks(&mut world, opts, w, h, &visible);

    // --- 2. The chip --------------------------------------------------------
    let mut chip = new_screen_chip(&mut world, w, h, &geometry);

    // Service gates sit behind both pixel stages, so they must use the same
    // stage pitch the (upright, 10-deep) pixel gates do -- with the flat
    // `CELL` pitch they landed inside stage 1's depth and collided.
    let service = |col: i32, row: i32| {
        lattice_pos_staged(col, row, SERVICE_STAGE, h, GATE_HALF, STAGE_PITCH)
    };

    // --- 3. Frame index source ---------------------------------------------
    // `control_pins` carries the clock's Pause/Restart/Resume pin ids for the
    // control buttons below; `None` under `--external-clock`, which builds no
    // timer and so has no pins to drive.
    let (frame_index, control_pins) = if opts.external_clock {
        let pin = chip::add_input_pin(&mut chip, "Frame", service(0, -1));
        (chip::pin_source(pin, true), None)
    } else {
        let clock = clock::build_clock(
            &mut world,
            &mut chip,
            info.fps,
            frame_count,
            opts.loop_playback,
            service(0, -2),
        );
        let pins = (clock.pause_pin, clock.restart_pin, clock.resume_pin);
        (clock.frame_index, Some(pins))
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
    let n_banks = frame_count.div_ceil(bank_size).max(1);

    // The shared per-bank spine: per-bank index, boundary comparators, and the
    // front-cascade of branches. `entry_of_bank` is kept so the subtitle
    // display can fan out from it rather than being spliced into a chunk chain
    // (see `cascade`). At n_banks == 1 this emits no gate at all and the chain
    // is byte-identical to the pre-spillover wiring.
    let cascade::BankCascade { index_of_bank, ge, entry_of_bank } = cascade::bank_cascade(
        &mut world,
        &mut chip,
        &frame_index,
        WirePort::new(detector, CHANGE_DETECTOR, "OnChanged"),
        n_banks,
        bank_size,
        &service,
    );

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

    // Exec: each bank's gets chain off that bank's entry. Per-chunk here rather
    // than a fan-out, because at one Get per chunk the chain is a handful of
    // gates deep, not thousands (see `color_bricks` for why the pixel-major
    // encoding must fan out instead). Every Get's Exec still takes exactly one
    // source, so no exec input ever gains a second.
    for bi in 0..n_banks {
        let mut prev = entry_of_bank[bi].clone();
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
            // Polled per pixel, not per chunk: this is where nearly the whole
            // gate count of a render is spent, and a chunk-granularity poll
            // would still leave a cancel waiting through the rest of a chunk.
            // The poll itself is far below the cost of a pixel's own gate
            // allocations, so striding it buys nothing.
            if progress.is_cancelled() {
                return Ok(World::new());
            }
            let idx = chunk.first_pixel + local;
            let Some(&brick_id) = brick_of.get(&idx) else {
                continue; // culled: slot reserved in the string, no gates
            };
            let (col, row) = ((idx as i32) % w, (idx as i32) / w);

            // Stage 1 (above) = Substring, stage 0 (bottom) = MakeColorHex.
            // Facing/roll are cosmetic; PIXEL_GATE_HALF is the real collision box.
            let sub = clock::gate_oriented(
                &mut chip,
                "B_1x1_Gate_Expr_String_Substring",
                SUBSTRING,
                lattice_pos_staged(col, row, 1, h, PIXEL_GATE_HALF, STAGE_PITCH),
                vec![
                    // bare i64, not WireVariant::Int -- this gate's schema
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

    // --- 6b. Subtitles, if any ----------------------------------------------
    //
    // Last, after every pixel gate exists -- see `add_subtitle_display`'s doc
    // for why the ordering matters -- and gated on `opts.subtitles`, so a
    // render without a track is exactly the graph it is today.
    if let Some(subs) = &opts.subtitles {
        // `opts.source_start_s`, not 0.0: a subtitle file is in source time,
        // and frame 0 of what this renderer receives is at source time
        // `--start` (see `AnimOptions::source_start_s`). Timing the cues from
        // zero puts the whole track `--start` seconds early.
        let per_frame = subs.per_frame(opts.source_start_s, info.fps as f64, frame_count)?;
        super::subtitle_display::add_subtitle_display(
            &mut world,
            &mut chip,
            super::subtitle_display::FrameIndex {
                index_of_bank: &index_of_bank,
                entry_of_bank: &entry_of_bank,
                ge: &ge,
            },
            &per_frame,
            info.fps,
            opts,
            subtitle_extent(&geometry, w, h, opts.subtitle_lift)?,
        )?;
    }

    // --- 6c. Control buttons ------------------------------------------------
    //
    // Default-on: three physical labelled buttons on the main grid, each wired
    // by its `bHeld` output straight into one of the clock's control pins, so a
    // fresh render is pausable/restartable/resumable with no hand-wiring.
    // Skipped under `--external-clock` (no timer, no pins) and when the toggle
    // is off. Built before `chip::finish` so its main-grid overlap check sees
    // these bricks, and before `register_used_components`.
    if let (true, Some((pause, restart, resume))) = (opts.control_buttons, control_pins) {
        let anchor = controls::control_anchor(&world);
        controls::add_control_buttons(&mut world, pause, restart, resume, anchor);
    }

    // --- 7. Publish -------------------------------------------------------
    //
    // One last poll before publish: the per-pixel loop only polls at the top
    // of each iteration, and the subtitle step above does not poll at all, so
    // without this a cancel could land only after `chip::finish`'s grid
    // collision checks -- still the largest remaining piece of work.
    if progress.is_cancelled() {
        return Ok(World::new());
    }
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

    /// The extent the pixel gates actually occupy, measured rather than
    /// asserted from the number the renderer recorded. Re-derives the box
    /// from the authored [`GATE_HALF`] and the facing/roll the bricks are
    /// written with: `chip::recompute_plane_extent`'s non-negativity check
    /// and `layout::assert_no_overlap` both read the recorded half, so
    /// neither could catch `PIXEL_GATE_HALF` being the wrong permutation.
    #[test]
    fn a_column_zero_pixel_gate_occupies_no_negative_coordinate() {
        // The extent a `GATE_HALF` brick really fills once stood on end,
        // derived from the facing/roll the renderer writes -- not read back
        // off `PIXEL_GATE_HALF`, which is the value under test.
        let real = rotated_half(GATE_HALF, PIXEL_GATE_FACING, PIXEL_GATE_ROLL);
        assert_eq!(
            PIXEL_GATE_HALF, real,
            "the recorded half must be the box the brick actually fills"
        );

        // The low corner of a real lattice: bottom row (largest `row`, which
        // `lattice_pos_staged` maps to the lowest x), column 0, stage 0.
        for h in [1i32, 3, 108] {
            for stage in [0i32, 1] {
                let pos = lattice_pos_staged(0, h - 1, stage, h, PIXEL_GATE_HALF, STAGE_PITCH);
                let min = (pos.x - real.x, pos.y - real.y, pos.z - real.z);
                assert!(
                    min.0 >= 0 && min.1 >= 0 && min.2 >= 0,
                    "h={h} stage={stage}: a column-0 pixel gate centred at \
                     ({}, {}, {}) with real half ({}, {}, {}) reaches {min:?} -- negative \
                     inner-grid coordinates are what this crate says delete bricks in game",
                    pos.x,
                    pos.y,
                    pos.z,
                    real.x,
                    real.y,
                    real.z
                );
            }
        }
    }

    /// The pixel lattice must still be collision-free when measured with the
    /// real rotated extent, not just the recorded half. Both pixel stages,
    /// several rows and columns, at the real pitches.
    #[test]
    fn the_pixel_lattice_is_collision_free_at_the_real_rotated_extent() {
        let real = rotated_half(GATE_HALF, PIXEL_GATE_FACING, PIXEL_GATE_ROLL);
        let h = 6;
        let mut placed = Vec::new();
        for row in 0..h {
            for col in 0..5 {
                for stage in 0..2 {
                    placed.push((
                        lattice_pos_staged(col, row, stage, h, PIXEL_GATE_HALF, STAGE_PITCH),
                        real,
                    ));
                }
            }
        }
        // Plus the service stage, which is not rotated and keeps GATE_HALF.
        for i in -9..0 {
            placed.push((
                lattice_pos_staged(0, i, SERVICE_STAGE, h, GATE_HALF, STAGE_PITCH),
                GATE_HALF,
            ));
        }
        super::super::layout::assert_no_overlap(&placed)
            .expect("the lattice must clear itself at the extent the bricks really fill");
    }

    /// A `Progress` that reports cancelled once `tick` reaches `cancel_after`
    /// frames streamed, recording the tick count so a test can assert exactly
    /// how far the loop got.
    struct CancelAfter {
        cancel_after: u64,
        ticks: u64,
    }

    impl Progress for CancelAfter {
        fn begin(&mut self, _label: &str, _total: Option<u64>) {}
        fn tick(&mut self, n: u64) {
            self.ticks = n;
        }
        fn finish(&mut self) {}
        fn is_cancelled(&self) -> bool {
            self.ticks >= self.cancel_after
        }
    }

    /// A solid-colour `frames`-frame clip; every pixel opaque so nothing gets
    /// culled and every frame does real packer work.
    fn solid_clip(frames: usize) -> crate::video::Clip {
        solid_clip_sized(frames, 2, 2)
    }

    /// [`solid_clip`] at a chosen size, for the per-pixel cancellation test:
    /// it counts loop iterations, and 2x2's four pixels can't distinguish
    /// "bailed at the first pixel" from "walked them all".
    fn solid_clip_sized(frames: usize, w: u32, h: u32) -> crate::video::Clip {
        crate::video::Clip {
            width: w,
            height: h,
            fps: 10.0,
            frames: (0..frames)
                .map(|i| {
                    image::RgbaImage::from_pixel(w, h, image::Rgba([(i % 255) as u8, 0, 0, 255]))
                })
                .collect(),
        }
    }

    /// A `Progress` that reports cancelled once `is_cancelled` has been asked
    /// `after_polls` times. Unlike [`CancelAfter`], which can only flip during
    /// decode (nothing ticks once the build starts), a poll count set past
    /// the clip's frame count lands the cancellation inside the build -- the
    /// only way a test can reach the per-pixel poll. `Cell` because
    /// `is_cancelled` takes `&self`.
    struct CancelAtPoll {
        polls: std::cell::Cell<usize>,
        after_polls: usize,
    }

    impl Progress for CancelAtPoll {
        fn begin(&mut self, _label: &str, _total: Option<u64>) {}
        fn tick(&mut self, _n: u64) {}
        fn finish(&mut self) {}
        fn is_cancelled(&self) -> bool {
            let n = self.polls.get() + 1;
            self.polls.set(n);
            n > self.after_polls
        }
    }

    /// The regression test for cancellation: a `Progress` that flags
    /// cancelled after a handful of frames must stop the loop near that
    /// frame, not drain a much longer clip -- and the result must still be
    /// `Ok`, never an `Err`, since cancellation is a normal outcome, not a
    /// failure (see `build_brick_world`'s doc).
    #[test]
    fn is_cancelled_stops_the_loop_near_frame_n_not_the_full_clip() {
        const CANCEL_AFTER: u64 = 5;
        const TOTAL_FRAMES: usize = 500;

        let clip = solid_clip(TOTAL_FRAMES);
        let mut progress = CancelAfter { cancel_after: CANCEL_AFTER, ticks: 0 };

        let world = build_brick_world(&clip, &AnimOptions::default(), &mut progress)
            .expect("a cancelled render must be Ok, not an error");

        assert!(
            progress.is_cancelled(),
            "the reporter must have actually flagged cancellation for this to be a real test"
        );
        assert_eq!(
            progress.ticks, CANCEL_AFTER,
            "the loop must stop at frame {CANCEL_AFTER}, not drain the whole \
             {TOTAL_FRAMES}-frame clip"
        );

        // A cancelled render's `World` is real but partial; the caller is
        // responsible for discarding it (see `build_brick_world`'s doc).
        let _ = world;
    }

    /// The complementary case: a `Progress` whose `is_cancelled` never fires
    /// must still process every frame, pinned by asserting `tick` reached
    /// the clip's full length.
    #[test]
    fn without_cancellation_the_render_completes_the_full_clip() {
        const TOTAL_FRAMES: usize = 37;
        let clip = solid_clip(TOTAL_FRAMES);

        // Past the clip's length: `is_cancelled` never fires, mirroring
        // `NoProgress`'s permanent `false`.
        let mut progress = CancelAfter { cancel_after: TOTAL_FRAMES as u64 + 1, ticks: 0 };

        build_brick_world(&clip, &AnimOptions::default(), &mut progress).expect("build");

        assert!(!progress.is_cancelled());
        assert_eq!(
            progress.ticks, TOTAL_FRAMES as u64,
            "an uncancelled render must process every frame of the clip"
        );
    }

    /// A cancel mid-decode must build no graph at all. Checked against an
    /// uncancelled control built from the same clip, since `Ok` alone can't
    /// distinguish "stopped early" from "built anyway".
    #[test]
    fn a_cancel_while_packing_builds_no_graph_at_all() {
        const TOTAL_FRAMES: usize = 200;
        let clip = solid_clip(TOTAL_FRAMES);

        let mut cancelled = CancelAfter { cancel_after: 5, ticks: 0 };
        let stopped = build_brick_world(&clip, &AnimOptions::default(), &mut cancelled)
            .expect("a cancelled render must be Ok, not an error");

        // The control: the same clip, the same options, a reporter that can
        // never flip. Without this the assertions below would also pass on a
        // renderer that had simply stopped building anything at all.
        let mut ran = CancelAfter { cancel_after: TOTAL_FRAMES as u64 + 1, ticks: 0 };
        let full = build_brick_world(&clip, &AnimOptions::default(), &mut ran).expect("build");
        assert!(
            !full.bricks.is_empty() && !full.wires.is_empty() && !full.grids.is_empty(),
            "the uncancelled control must actually build a graph"
        );

        assert!(cancelled.is_cancelled(), "the reporter must really have flagged cancellation");
        assert!(stopped.bricks.is_empty(), "a cancelled render must place no display bricks");
        assert!(stopped.grids.is_empty(), "a cancelled render must build no chip grid");
        assert!(stopped.wires.is_empty(), "a cancelled render must wire nothing");
        assert!(stopped.entities.is_empty(), "a cancelled render must publish no entities");
    }

    /// A cancel that arrives once the decode is already done must still be
    /// honoured, at the next pixel rather than at the end of the build -- the
    /// per-pixel poll in section 6.
    ///
    /// Asserted by counting polls, because that is the only observable that
    /// distinguishes "bailed immediately" from "finished the loop": the world
    /// comes back empty either way.
    #[test]
    fn a_cancel_during_the_gate_build_stops_at_the_next_pixel() {
        const FRAMES: usize = 4;
        // 256 opaque pixels, so the per-pixel loop has 256 iterations to run
        // if it is not stopped -- a count nothing could reach by accident.
        const SIDE: u32 = 16;
        let clip = solid_clip_sized(FRAMES, SIDE, SIDE);

        // The polls a render makes before the first pixel: one per frame in
        // the decode loop, then the post-decode check. Budgeting exactly that
        // many lets both of those read "not cancelled" and flips the very next
        // poll, which is the first pixel's.
        let before_the_pixels = FRAMES + 1;
        let mut progress =
            CancelAtPoll { polls: std::cell::Cell::new(0), after_polls: before_the_pixels };

        let world = build_brick_world(&clip, &AnimOptions::default(), &mut progress)
            .expect("a cancelled render must be Ok, not an error");

        assert!(world.bricks.is_empty(), "a cancelled render must place no display bricks");
        assert!(world.grids.is_empty(), "a cancelled render must build no chip grid");
        assert_eq!(
            progress.polls.get(),
            before_the_pixels + 1,
            "the pixel loop must bail on its first poll -- {} polls means it walked all \
             {} pixels instead",
            before_the_pixels + (SIDE * SIDE) as usize,
            SIDE * SIDE
        );
    }

    /// The last gap in cancel promptness: the per-pixel loop only polls at
    /// the top of each iteration, and the subtitle step doesn't poll at all,
    /// so this budgets exactly the polls a whole render makes and asserts the
    /// pre-publish poll is the first to see the cancellation, with the render
    /// coming back empty.
    #[test]
    fn a_cancel_arriving_after_the_last_pixel_still_skips_the_publish_phase() {
        const FRAMES: usize = 3;
        const SIDE: u32 = 2;
        let clip = solid_clip_sized(FRAMES, SIDE, SIDE);
        let pixels = (SIDE * SIDE) as usize;
        // frames (decode loop) + the post-decode guard + one per pixel.
        let whole_render = FRAMES + 1 + pixels;
        let mut progress =
            CancelAtPoll { polls: std::cell::Cell::new(0), after_polls: whole_render };

        let world = build_brick_world(&clip, &AnimOptions::default(), &mut progress)
            .expect("a cancelled render must be Ok, not an error");

        assert_eq!(
            progress.polls.get(),
            whole_render + 1,
            "the budget must have been spent exactly, with the pre-publish poll the first \
             to see the cancellation"
        );
        assert!(
            world.bricks.is_empty() && world.grids.is_empty() && world.wires.is_empty(),
            "a cancel landing after the last pixel must still return an empty world rather \
             than paying for chip::finish's two grid collision checks"
        );

        // The control: one more poll of budget and the same render completes.
        // Buttons off so the brick count is exactly the screen plus the shell.
        let mut ran =
            CancelAtPoll { polls: std::cell::Cell::new(0), after_polls: whole_render + 1 };
        let opts = AnimOptions { control_buttons: false, ..AnimOptions::default() };
        let full = build_brick_world(&clip, &opts, &mut ran).expect("build");
        assert_eq!(full.bricks.len(), pixels + 1, "the uncancelled control must build the screen");
    }

    /// `NoProgress`, the library's actual default, must never cancel a real
    /// render through `build_brick_world`.
    #[test]
    fn no_progress_never_cancels_a_real_render() {
        let clip = solid_clip(10);
        let world = build_brick_world(&clip, &AnimOptions::default(), &mut crate::progress::NoProgress)
            .expect("NoProgress must never turn a normal render into an error");
        let _ = world;
    }
}
