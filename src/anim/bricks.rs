//! Brick-mode renderer. A screen of display bricks on the main grid, driven
//! by frame-major string arrays inside a microchip.
//!
//! Service gates (clock, arrays, detector) live at lattice stage 2, behind
//! both pixel stages, so they can never collide with pixel gates.
use super::chip;
use super::clock::{self, gate};
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
/// These gates ARE rotated: stood on end so they point downward rather than
/// facing +Z at the viewer like the clock's gates. Purely cosmetic --
/// orientation moves no centre, and [`layout::STAGE_PITCH`] is already the
/// 15 units an upright (10-deep) gate needs.
///
/// This constant's doc used to claim the rotation had been "attempted and
/// reverted" and that both this and [`PIXEL_GATE_ROLL`] were identity. Neither
/// was ever true -- all three constants landed rotated in the branch's first
/// commit -- and the stale text was actively dangerous, because it told a
/// reader the extent below was `GATE_HALF` when it was a hand-written triple
/// no orientation of a gate can produce. See [`PIXEL_GATE_HALF`].
const PIXEL_GATE_FACING: Direction = Direction::XPositive;

/// Roll for those same gates: a quarter turn about [`PIXEL_GATE_FACING`]'s
/// axis. A gate is square in x/y, so this changes no extent -- see
/// [`PIXEL_GATE_HALF`].
const PIXEL_GATE_ROLL: Rotation = Rotation::Deg90;

/// Half-extent of a pixel gate AFTER [`PIXEL_GATE_FACING`]/[`PIXEL_GATE_ROLL`]
/// are applied, DERIVED from [`GATE_HALF`] rather than written by hand.
///
/// `Brick::local_bounds` ignores rotation, so this value is the only
/// description of a rotated gate that `layout::assert_no_overlap`,
/// `chip::plane_bounds_for` and `chip::recompute_plane_extent`'s
/// non-negativity assert will ever see -- and it is also what
/// [`super::layout::lattice_pos_staged`] offsets the position by, so getting it
/// wrong moves the gate as well as mismeasuring it.
///
/// It WAS wrong. This was a hand-written `{5, 2, 5}`, which is the permutation
/// a `Y_Positive` facing would give: it puts the authored thin axis (2) on
/// world y. The code's facing is `X_Positive`, which puts it on world x, so the
/// true extent is `{2, 5, 5}` -- the same three numbers, two of them swapped.
/// Under the old value every column-0 pixel gate really reached `y = -3` while
/// the recorded box said `y = 0`, so both the overlap check and the
/// non-negativity assert were reading a box the brick did not fill. (The
/// lattice happened to stay collision-free under either value -- 5s against a
/// 10-unit pitch are flush, 2s are gapped -- so nothing broke in game; it was
/// wrong rather than harmful.)
///
/// Deriving it through [`super::layout::rotated_half`] is what stops that
/// recurring: the extent can no longer disagree with the facing it is supposed
/// to describe. See that function for exactly which part of the mapping is a
/// fact about `EBrickDirection` and which is an unobservable convention.
const PIXEL_GATE_HALF: IntVector = rotated_half(GATE_HALF, PIXEL_GATE_FACING, PIXEL_GATE_ROLL);

/// How far above the screen the interaction plane floats, in world units.
///
/// Applied to the entity location [`new_screen_chip`] computes. That is NOT
/// where the plane's middle finally lands: this doc used to say
/// "`chip::finish` centres the chip contents on the grid origin, so the plane's
/// middle coincides with `entity.location`", and `chip::finish` does the
/// opposite on purpose -- the centering was tried, shipped, broke gates in
/// game, and was removed (see `chip::finish`'s own doc, which says not to
/// reintroduce it). `chip::recompute_plane_extent` then moves
/// `entity.location` again, by `(-extent.y, +extent.x)`, so on a large screen
/// the plane ends up a long way from directly above the middle of the picture.
/// That displacement is invisible today only because the chip is published
/// COLLAPSED (`chip::new_chip`), and it is called out here rather than
/// silently left as a claim the code does not honour.
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
    /// Repeat the clip forever (`true`, the default) or stop on its last
    /// frame (`false`).
    ///
    /// Reaches the render through `clock::build_clock`, which turns it into
    /// the value inlined on `Timer.Limit` and nothing else -- see
    /// [`crate::anim::clock::stop_limit`] for why the limit is
    /// `(frame_count - 0.5) / fps` rather than the clip's exact duration, and
    /// for what about it is still unverified in game.
    ///
    /// The graph is identical either way, so this changes no gate, wire or
    /// brick count and every cost estimate stays true for both settings.
    /// Inert under [`Self::external_clock`], which builds no timer at all.
    pub loop_playback: bool,
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
    /// Convert sRGB frame pixels to linear before hex-encoding (see
    /// `pack::Packer::linearize`). Off by default.
    pub srgb_to_linear: bool,
    /// Entries per wire array before frames spill into another bank.
    ///
    /// Defaults to `BANK_FRAMES`. Lowering it is how the multi-bank graph is
    /// tested without building a 65 536-frame clip.
    pub bank_size: usize,
    /// TEXT MODE ONLY (see [`super::text_bricks`]): quantize every frame to at
    /// most this many colours with a median-cut palette
    /// ([`super::palette::Palette`]) before encoding. `0` -- the default --
    /// means no quantization at all, which
    /// [`Palette::map`](super::palette::Palette::map) implements as a
    /// pass-through, so the "off" path costs nothing.
    ///
    /// Text mode writes a 16-character `<color="RRGGBB">` tag at the start of
    /// every colour RUN, so its size is governed by run length rather than tag
    /// width; collapsing the palette lengthens runs. Both brick encodings
    /// ignore this field entirely -- they spend a fixed cost per pixel that no
    /// palette can change.
    pub colors: usize,
    /// TEXT MODE ONLY: the font, glyph and component geometry a text render
    /// uses. `char_repeat` also drives the band layout
    /// ([`super::text_layout::plan_bands`]), which is why this has to live on
    /// `AnimOptions` rather than being an extra parameter -- all three
    /// renderers share one signature so `AnimMode` can dispatch without
    /// branching on anything else.
    ///
    /// Defaults to `FontPreset::MonaspaceArgon.options(1.0)` so
    /// [`AnimOptions::default`] keeps working and every brick-mode call site
    /// compiles unchanged; both brick encodings ignore it.
    pub text: crate::text::TextOptions,
    /// A subtitle track to render across the bottom of the screen, or `None`
    /// for no subtitles at all.
    ///
    /// `None` -- the default -- is a hard gate: all three renderers skip
    /// [`super::subtitle_display`] entirely, so a render without subtitles
    /// produces exactly the graph it produces today, down to the component
    /// count. Honoured by every mode, because the display is wired only to
    /// the frame index and knows nothing about how the picture is drawn.
    ///
    /// An `Arc` because the GUI hands the same parsed track to a render
    /// thread while keeping it for the cost readout, and re-parsing (or
    /// deep-copying a few thousand cues) for that is pure waste. It also
    /// keeps [`AnimOptions`] cheap to pass around, which matters because
    /// `crate::subs` is pure `std` and reaches the wasm build.
    pub subtitles: Option<std::sync::Arc<crate::subs::Subtitles>>,
    /// How much bigger a subtitle line is than one row of the screen.
    ///
    /// Defaults to [`super::subtitle_display::DEFAULT_SUBTITLE_SCALE`], which
    /// carries the full reasoning -- and the fact that nobody has yet checked
    /// it by eye in game. Inert while [`Self::subtitles`] is `None`.
    pub subtitle_scale: f32,
    /// How many world units the subtitle anchor is lifted "up the picture"
    /// from its bare bottom-centre baseline.
    ///
    /// Defaults to [`super::subtitle_display::DEFAULT_SUBTITLE_LIFT`], whose
    /// doc carries the calibration note: it was measured by eye against ONE
    /// real render, in TEXT mode, at 192x108 with `--subtitle-scale 6`. Each
    /// renderer applies it along whichever axis means "toward the top of the
    /// picture" for that renderer's own screen orientation -- **+z** for text
    /// mode's upright wall (the mode this was actually checked against), and
    /// **-y** for both brick encodings' flat, ground-facing screen (see
    /// `subtitle_extent`'s doc; unverified by eye there). Inert while
    /// [`Self::subtitles`] is `None`.
    ///
    /// Both brick encodings accept any lift, including one that carries the
    /// anchor past the top of the picture to a negative `y` -- their own chip
    /// shell already sits at a negative main-grid `x` in every render (see
    /// [`new_screen_chip`] and `subtitle_extent`). TEXT mode still rejects a
    /// lift that would push its anchor below `z = 0`, because that mode keeps
    /// its entire main grid non-negative by construction (`crate::text::add_text_tiles`
    /// translates the whole grid for it) and a subtitle is the one brick placed
    /// outside that translation.
    pub subtitle_lift: f32,
    /// The SOURCE timestamp of output frame 0, in seconds -- i.e. the
    /// `--start` the frame source was opened with.
    ///
    /// **Only the subtitle timing reads this, and it is not optional there.**
    /// Nothing in a renderer seeks: the [`crate::video::stream::AdaptedSource`]
    /// wrapped around the source has already done the seeking, and frame `i`
    /// of what a renderer receives is at source time
    /// `source_start_s + i / fps` (see `scale::FpsStream`). A subtitle file is
    /// in SOURCE time, so `Subtitles::per_frame` has to be handed that same
    /// offset or every cue lands `--start` seconds early -- with `--start 120`
    /// the whole track is two minutes out, which is not subtle and is not
    /// visible from anywhere but a real render.
    ///
    /// `0.0` -- the default -- is correct for any source rendered from its
    /// beginning, which is every call site that predates this field.
    pub source_start_s: f64,
}

impl Default for AnimOptions {
    fn default() -> Self {
        Self {
            alpha_threshold: 128,
            pixel_extent: 1,
            brick_style: DisplayBrickStyle::Micro,
            external_clock: false,
            // LOOPING, which is what every render did before the flag
            // existed. The default has to stay this way or an unchanged
            // command line would quietly produce a different save.
            loop_playback: true,
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

/// The on-ground geometry of one display pixel, derived ONCE from the options
/// so the pitch, the chip shell's clearance and the interaction plane's anchor
/// can never disagree with the brick that is actually placed.
///
/// Shared by both encodings ([`build_brick_world`] and
/// [`super::color_bricks::build_color_array_world`]) -- the screen is the part
/// of the build the two agree on completely, so it is built in one place
/// rather than described twice.
pub struct ScreenGeometry {
    /// A display brick's real on-ground half-extent on x and y, AFTER the
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
/// the picture's bottom edge, standing clear of its surface.
///
/// Shared by both brick encodings for the same reason [`ScreenGeometry`] is:
/// the screen is the part of the build they agree on completely, so where its
/// bottom edge lands is described once. Derived from the geometry
/// [`add_display_bricks`] already returned -- the real pitch and half-height of
/// the bricks that were actually placed -- never re-guessed.
///
/// A display-brick screen lies FLAT: `x` is the image column, `y` the image
/// row, `z` a constant. So the anchor is
///
/// * centred on `x`: the columns' centres span `0 ..= (cols - 1) * pitch`, so
///   their midpoint is half of that (`pitch` is even, so the halving is exact);
/// * at the bottom of `y`: the last row's outer face, `(rows - 1) * pitch +
///   footprint`, which is where the picture actually ends;
/// * ONE ANCHOR-CUBE ABOVE the screen's top face on `z`. The display bricks
///   rest their underside on `z = 0`, so their top face is at
///   `2 * half_height`; a 1-half-extent cube centred one half past that rests
///   flush on the picture and draws its glyphs a full cube clear of it. That is
///   what keeps the subtitle IN FRONT of the picture rather than z-fighting
///   inside it -- and, incidentally, what keeps its cube from overlapping a
///   display brick, which `chip::finish` would reject.
///
/// **The plane the subtitle is drawn in is why `face` is set here.** A
/// `TextDisplay` draws in the plane of the anchor face it is given, and this
/// screen lies FLAT -- it presents its top (`+Z`) to the viewer. The default
/// face, `X_Positive`, would stand the subtitle edge-on to the picture: a
/// razor-thin line of glyphs seen from directly above, unreadable at any
/// `--subtitle-scale`. `FACE_Z_POSITIVE` is the game's own
/// `EBrickDirection::Z_Positive` (the save schema types `Face` as that enum),
/// so the glyphs lie in the screen's own plane and read from where the picture
/// is read from.
///
/// **Still unverified by eye**: which way "along the line" and "the next line
/// up" point within that plane is a property of the component the schema does
/// not describe, so the subtitle may run along either horizontal axis of the
/// flat screen -- and this places it at the bottom-centre of the picture as the
/// IMAGE sees it, which is only the bottom-centre a viewer sees if the two
/// agree. It is legible either way, which the edge-on default was not. The CLI
/// says so out loud when the two are combined; text mode, whose screen is a
/// vertical wall in the same plane the glyphs already draw in, is the mode this
/// feature was designed for and needs no such caveat.
///
/// **`--subtitle-lift`'s axis here is `-y`, NOT `+z`.** `z` is height above
/// the ground here, not a row of the picture -- the image's rows run along
/// `y`, increasing toward the picture's BOTTOM (see above), so moving the
/// anchor toward the picture's TOP means DECREASING `y`. This is the opposite
/// of text mode's `+z` (`text_bricks::build_text_world`'s anchor, a vertical
/// wall where `z` genuinely is the image row) and is itself unverified by eye
/// for the same reason the in-plane line direction is: `--subtitle-lift`'s
/// default was calibrated against text mode only.
///
/// # A lift bigger than the picture is ALLOWED, and used to be an error
///
/// This used to return `Err` whenever `lift` carried the anchor past the top
/// of the picture to a negative `y`, on the stated grounds that "main-grid
/// brick coordinates cannot be negative -- `brdb`'s chunk encoding mishandles
/// them". **That reason was false, and this renderer contradicted it in the
/// same breath.** Two things settle it, both readable in the source:
///
/// * `brdb`'s encoder is exact for negatives. `Position::to_relative` splits a
///   coordinate with `div_euclid`/`rem_euclid`, and `from_relative` is its
///   exact inverse -- `brdb_round_trips_negative_brick_coordinates_exactly`
///   checks the whole neighbourhood of the origin and of a chunk boundary.
/// * The crate already depends on that. `World::add_brick_grid` shifts every
///   inner-grid brick by `-Position::CHUNK_HALF` (1024 units) before storing
///   it, so EVERY gate in EVERY microchip this crate has ever written lives at
///   an absolute coordinate around `-1024`, in chunk `(-1, -1, -1)` -- and
///   those renders work in game. On the main grid, [`new_screen_chip`] puts the
///   chip shell at `x = -(shell_half + footprint)`, i.e. negative, in every
///   brick-mode render, subtitled or not.
///
/// So a brick-mode render that refused a negative subtitle `y` while
/// unconditionally writing a negative shell `x` was rejecting legal renders --
/// with default options, any picture 4 rows or shorter -- to guard against
/// something that does not happen. The lift is now applied as asked.
///
/// TEXT mode is a different case and still rejects its equivalent: that mode
/// keeps its whole main grid non-negative by construction, because
/// `crate::text::add_text_tiles` translates the entire glyph grid to make it
/// so, and its subtitle is the one brick placed outside that translation. That
/// is a self-consistency rule of that renderer, not a claim about `brdb`.
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

/// Place one display brick per VISIBLE pixel on `world`'s main grid, returning
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
    // `pixel_extent` is a pre-scale half-extent (see `AnimOptions::pixel_extent`'s
    // doc); `.max(1)` guards the degenerate case the CLI/GUI floors already
    // prevent -- a 0 extent would collapse every display brick to a
    // zero-size point. The real on-ground footprint depends on the style
    // (SmoothTile scales by 5, Micro does not), so it's read straight off the
    // `BrickType` this same call just built -- never re-derived separately --
    // so the pitch and the chip shell's clearance can never disagree with the
    // brick that's actually placed.
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
                // underside on z=0. A hardcoded 2 grounded a smooth tile
                // (half-height 2, spanning 0..4) but left a micro brick
                // (half-height 1) floating a unit off the ground at 1..3.
                position: Position {
                    x: col * geometry.pitch,
                    y: row * geometry.pitch,
                    z: geometry.half_height,
                },
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
    (brick_of, geometry)
}

/// Start the microchip that drives a screen of the given size and geometry.
///
/// The shell sits BESIDE the screen, never on it. The very first in-game
/// spike stacked it in Z instead (a default display brick's z span at z=2
/// overlapped the shell's), and the game silently DROPPED one of them: a
/// 3-brick L with unconnectable wires. Placing it beside the screen on X
/// sidesteps that class of bug entirely -- `assert_bricks_dont_overlap`
/// only needs ONE axis to be clear of overlap, and `chip::finish` still
/// asserts main-grid non-overlap as the safety net either way.
///
/// The display style and its footprint are both options (no longer a
/// hardcoded 5,5,6 default), so the clearance is computed from the real
/// half-extents in play rather than a fixed guess: column 0's own display
/// bricks span x in `[-footprint, footprint]` (their center is x=0), and
/// the shell's own half-extent on x is read from `local_bounds` -- not
/// re-hardcoded -- so this stays correct if that ever changes. Landing the
/// shell's outer face exactly on `-footprint` makes the two spans share at
/// most a face on x (flush, not overlapping -- see
/// `layout::assert_bricks_dont_overlap`'s strict-inequality doc comment),
/// for every row, since every other display brick sits at an x further
/// from the shell than column 0's. The real `footprint` (not the pre-scale
/// `pixel_extent`) is what matters here -- a SmoothTile's real x half-extent
/// is `5*extent`, and using the unscaled one would leave the shell
/// overlapping the screen for that style.
///
/// The entity location this computes AIMS the interaction plane at the middle
/// of the screen. It does not land there. `chip::finish` does NOT centre the
/// chip's contents on the grid origin (that was tried, shipped, broke gates in
/// game and was removed -- see its doc), and
/// `chip::recompute_plane_extent` then shifts `entity.location` again by
/// `(-extent.y, +extent.x)`, which on a 192x108 screen is several hundred
/// units. So this is the plane's STARTING point, not its final one. The
/// difference is invisible today because `chip::new_chip` publishes the chip
/// collapsed. Stated plainly here because the previous wording asserted the
/// opposite as a fact, and nothing tests `entity.location`.
///
/// The `plane_extent` passed to `chip::new_chip` is only a placeholder, not a
/// real bound: no caller can know the true `PlaneExtent` before every gate/pin
/// exists (see `chip::plane_extent_for`'s docstring on why halving the lattice
/// span -- what this used to do -- silently clips the clock's control pins and
/// most pixel gates on anything but a tiny screen). `chip::finish` recomputes
/// and grows it to fit every placed brick, so this only needs to be a legal
/// (non-degenerate) starting value.
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
        // Start at the screen's actual top face -- the display bricks rest
        // their underside on z=0, so that is twice their half-height, which
        // differs per style (a smooth tile is 4 units tall, a default micro
        // brick 2) -- then rise clear of it.
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
/// A render `progress` reports [`Progress::is_cancelled`] for stops early --
/// as soon as the current in-flight frame finishes, or at the next pixel if
/// the cancel lands after decoding -- and returns an EMPTY [`World`], never a
/// partial one. This is deliberately NOT an error: cancellation is a normal
/// outcome, not a failure.
///
/// Returning early, rather than merely stopping the decoder, is the whole
/// point and is what this function used to get wrong. Breaking out of the
/// pull loop only ends the decode; every phase after it -- the display
/// bricks, the chip, the two gates per surviving pixel, the wiring,
/// `chip::finish` -- then ran anyway over the frames already packed. On a
/// 192x108 clip that is tens of thousands of gates and wires built *after*
/// the user asked for the render to stop, so a cancel looked like it had done
/// nothing for a long time and then discarded the result regardless. Both
/// this function and the two other renderers now return the moment they
/// notice, and the long per-pixel loop polls as it goes so a cancel pressed
/// during the build lands just as promptly as one pressed during the decode.
///
/// It is still on the CALLER to re-check `progress.is_cancelled()` once this
/// returns and write nothing when it is true: an empty `World` encodes to a
/// perfectly valid (and perfectly empty) save, so nothing downstream can tell
/// a cancelled render from a real one on its own.
/// `gui::util::deliver_world_unless_cancelled` is that check, and it logs the
/// cancellation as INFO rather than an error. Every caller that cannot cancel
/// uses a `Progress` whose `is_cancelled` defaults to `false` (see that trait
/// method's doc), so none of them need to change.
pub fn build_brick_world(
    source: &dyn FrameSource,
    opts: &AnimOptions,
    progress: &mut dyn Progress,
) -> Result<World, String> {
    let info = source.info();
    let (w, h) = (info.width as i32, info.height as i32);

    // One fused streaming pass builds both the per-chunk frame strings and
    // the per-pixel visibility bitmap, so no frame is ever retained (see
    // `pack::Packer`'s doc comment) -- this replaces the old two whole-clip
    // passes (`pack(clip, ...)` plus a `visible()` scan per pixel). A
    // `stream.next()` error is FATAL and propagates immediately: treating a
    // mid-stream failure as end-of-clip would silently write a save missing
    // its tail.
    // The denominator falls back to the source's ESTIMATE when it cannot give
    // an exact count, and says which it is -- see `FrameTotal`. An `.mkv`
    // reports no frame count at all, so this used to be an unlabelled
    // indeterminate spinner for the whole render, which reads as a hang.
    FrameTotal::new(info.frame_count_hint, source.frame_count_estimate())
        .begin(progress, "packing frames");
    let mut packer = pack::Packer::new(info.width, info.height, opts.alpha_threshold, HEX_STRIDE)
        .linearize(opts.srgb_to_linear);
    // The whole pull loop is wrapped in an immediately-invoked closure so
    // `progress.finish()` below runs on every exit, not only the success
    // path: `source.open()?`, `stream.next()?` and `packer.push_frame(&frame)?`
    // can all propagate an error, and without this a mid-stream failure would
    // print under a bar that never called `finish()` -- left visually
    // "stuck" rather than closed out, even though the render itself did stop.
    //
    // A cancellation (`progress.is_cancelled()`) breaks this SAME loop rather
    // than propagating an error: `stream` (and, for `FfmpegSource`, the child
    // process it owns) is dropped the moment this closure returns, whichever
    // way it exits, so a cancelled render still tears down its source exactly
    // as a failed or completed one would. Checked once per frame, never per
    // pixel -- polling any more often than the source can produce frames
    // would just be wasted work.
    let seen: Result<u64, String> = (|| {
        let mut stream = source.open()?;
        let mut seen: u64 = 0;
        while let Some(frame) = stream.next()? {
            packer.push_frame(&frame)?;
            // Borrows the frame's own buffer -- no copy here. `Progress::frame`'s
            // default body is a no-op, so a reporter that doesn't override it
            // (every one except `ChannelProgress`) pays for one virtual call and
            // nothing else.
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

    // CANCELLED: stop here, before a single brick is placed. Everything below
    // -- display bricks, chip, clock, arrays, two gates per pixel, wiring,
    // publish -- is skipped, which is the entire difference between a cancel
    // that lands and one that merely stops the decoder (see this function's
    // doc). Deliberately AHEAD of the zero-frame guard below, so a cancel can
    // never come back as that error: the caller must see a cancellation as a
    // normal outcome, not as a failed render.
    if progress.is_cancelled() {
        return Ok(World::new());
    }

    let (chunks, visible) = packer.finish();
    let frame_count = seen as usize;

    // A zero-frame source (reachable via `--start`/`--duration` past the
    // source's end, or the GUI's Start slider dragged past it -- see
    // `video::scale::FpsStream`, which ends with no frames emitted rather
    // than erroring) must not fall through to a "successful"
    // build: with no display bricks and `clock::build_clock` inlining
    // `Modulo.InputB = frame_count as f64 = 0.0`, the save would open fine
    // and silently divide by zero in-game on every tick. Caught once here so
    // both the CLI and GUI entry points, which both funnel through this
    // function, get the same clear error instead of a broken file.
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

    // Service gates sit behind BOTH pixel stages, so they must use the same
    // stage pitch the (upright, 10-deep) pixel gates do -- with the flat
    // `CELL` pitch they landed inside stage 1's depth and collided.
    let service = |col: i32, row: i32| {
        lattice_pos_staged(col, row, SERVICE_STAGE, h, GATE_HALF, STAGE_PITCH)
    };

    // --- 3. Frame index source ---------------------------------------------
    let frame_index = if opts.external_clock {
        let pin = chip::add_input_pin(&mut chip, "Frame", service(0, -1));
        chip::pin_source(pin, true)
    } else {
        clock::build_clock(
            &mut world,
            &mut chip,
            info.fps,
            frame_count,
            opts.loop_playback,
            service(0, -2),
        )
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
    let n_banks = frame_count.div_ceil(bank_size).max(1);

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
    // Each bank's entry is also kept, so the subtitle display can FAN OUT from
    // it rather than being spliced into the chunk chain. Fan-out is free; a
    // splice would put a second source on an exec input somewhere, which is
    // the one thing nothing here may require.
    let mut entry_of_bank = Vec::with_capacity(n_banks);
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
        entry_of_bank.push(entry.clone());
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
            // The one loop long enough to need its own poll: this is where
            // essentially the whole gate count of a render is spent, and a
            // cancel pressed here would otherwise wait for every remaining
            // pixel. Polled per PIXEL rather than per chunk -- a chunk is
            // `HEX_STRIDE`-strided over 1666 pixels, so a per-chunk poll on a
            // 13-chunk screen would still leave a cancel waiting on 8% of the
            // build. The poll itself is one virtual call (an atomic load, for
            // the GUI's reporter) against the several heap allocations this
            // pixel's two gates are about to cost, so it is far below the
            // noise floor and is not worth striding.
            if progress.is_cancelled() {
                return Ok(World::new());
            }
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
    // LAST, after every pixel gate exists -- see `add_subtitle_display`'s doc
    // for why the ordering matters -- and gated on `opts.subtitles`, so a
    // render without a track is exactly the graph it is today.
    if let Some(subs) = &opts.subtitles {
        // `opts.source_start_s`, NOT 0.0: a subtitle file is in SOURCE time,
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

    // --- 7. Publish -------------------------------------------------------
    //
    // One last poll before the publish phase. The per-pixel loop polls at the
    // TOP of each iteration, so a cancel arriving during the final pixel (or
    // during the subtitle step above, which does not poll at all) would
    // otherwise be noticed only by the caller, after `chip::finish` had
    // already collision-checked both grids. That check is proportional now
    // (`layout::overlap_scan`) rather than quadratic, but it is still the
    // largest single piece of work left, and a cancel is meant to land
    // promptly -- see this function's doc.
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

    /// **The extent the pixel gates actually occupy, measured rather than
    /// asserted from the number the renderer recorded.**
    ///
    /// This is the check that was missing. `chip::recompute_plane_extent`'s
    /// non-negativity `debug_assert` and `layout::assert_no_overlap` both read
    /// the RECORDED half, and `lattice_pos_staged` offsets the position by that
    /// same recorded half -- so `position - recorded_half >= 0` is true by
    /// construction whatever the recorded half says, and neither of them could
    /// ever have caught `PIXEL_GATE_HALF` being the wrong permutation.
    ///
    /// So this re-derives the box from the AUTHORED [`GATE_HALF`] and the
    /// facing/roll the bricks are actually written with, and asserts the
    /// derived box lands where the recorded one claims. Under the old
    /// hand-written `{5, 2, 5}` a column-0 gate really reached `y = -3`.
    #[test]
    fn a_column_zero_pixel_gate_occupies_no_negative_coordinate() {
        // The extent a `GATE_HALF` brick really fills once it is stood on end,
        // derived from the facing/roll the renderer writes -- NOT read back
        // off `PIXEL_GATE_HALF`, which is the value under test.
        let real = rotated_half(GATE_HALF, PIXEL_GATE_FACING, PIXEL_GATE_ROLL);
        assert_eq!(
            PIXEL_GATE_HALF, real,
            "the recorded half must be the box the brick actually fills"
        );

        // The whole low corner of a real lattice: bottom row (largest `row`,
        // which `lattice_pos_staged` maps to the LOWEST x), column 0, stage 0.
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
    /// REAL rotated extent, not just with whatever half was recorded. Both
    /// pixel stages, several rows and columns, at the real pitches.
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
        // Plus the service stage, which is NOT rotated and keeps GATE_HALF.
        for i in -9..0 {
            placed.push((
                lattice_pos_staged(0, i, SERVICE_STAGE, h, GATE_HALF, STAGE_PITCH),
                GATE_HALF,
            ));
        }
        super::super::layout::assert_no_overlap(&placed)
            .expect("the lattice must clear itself at the extent the bricks really fill");
    }

    /// A `Progress` that reports cancelled once its own `tick` has reached
    /// `cancel_after` -- i.e. after that many frames have actually been
    /// streamed through `build_brick_world`'s loop. Recording the tick count
    /// (not just a bool) lets a test assert on exactly how far the loop got,
    /// which is the most direct evidence available that it stopped early
    /// rather than draining its whole source.
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

    /// A solid-color `frames`-frame clip, big enough to matter for the
    /// cancellation tests below but with no visual content that matters --
    /// only its length and the fact that every pixel is opaque (so nothing
    /// gets culled and every frame does real packer work).
    fn solid_clip(frames: usize) -> crate::video::Clip {
        solid_clip_sized(frames, 2, 2)
    }

    /// [`solid_clip`] at a chosen size. Only the per-pixel cancellation test
    /// needs this: it has to count loop iterations, and 2x2's four pixels
    /// cannot tell "bailed at the first pixel" from "walked them all".
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
    /// `after_polls` times, counting the polls itself.
    ///
    /// [`CancelAfter`] can only flip during the DECODE, because `tick` is the
    /// only thing that moves it and nothing ticks once the build starts. This
    /// one flips on a poll count instead, so a budget set past the clip's
    /// frame count lands the cancellation inside the build -- the only way a
    /// test can reach the per-pixel poll, since no other thread exists to set
    /// a flag while the render is running.
    ///
    /// `Cell`, because `is_cancelled` takes `&self` by design (see
    /// `Progress::is_cancelled`), and the count is also what the test asserts
    /// on: how far the pixel loop got is visible in nothing else.
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

        // A cancelled render's `World` is real but partial -- it is on the
        // CALLER (see `build_brick_world`'s doc) to notice
        // `progress.is_cancelled()` and discard it; `build_brick_world`
        // itself must not error just because it stopped early. Nothing more
        // to assert on `world` here -- that hand-off is exercised at the
        // GUI layer (`gui::video`'s own tests).
        let _ = world;
    }

    /// The complementary case: a `Progress` whose `is_cancelled` never fires
    /// (mirroring `NoProgress` and every pre-existing caller) must still
    /// process every frame, not just "not crash" -- pinned by asserting
    /// `tick` reached the clip's full length, which every OTHER test in this
    /// module already relies on implicitly but none states directly.
    #[test]
    fn without_cancellation_the_render_completes_the_full_clip() {
        const TOTAL_FRAMES: usize = 37;
        let clip = solid_clip(TOTAL_FRAMES);

        // `cancel_after` set past the clip's length -- `is_cancelled` can
        // never fire, exactly mirroring `NoProgress`'s permanent `false`,
        // but still lets this test observe the tick count `NoProgress`
        // itself has no field to report.
        let mut progress = CancelAfter { cancel_after: TOTAL_FRAMES as u64 + 1, ticks: 0 };

        build_brick_world(&clip, &AnimOptions::default(), &mut progress).expect("build");

        assert!(!progress.is_cancelled());
        assert_eq!(
            progress.ticks, TOTAL_FRAMES as u64,
            "an uncancelled render must process every frame of the clip"
        );
    }

    /// THE REGRESSION TEST for the cancel that did not cancel.
    ///
    /// Stopping the decode loop was never enough: every phase after it --
    /// display bricks, chip, clock, arrays, two gates per pixel, wiring,
    /// publish -- used to run anyway over the frames already packed, so a
    /// cancel bought nothing but a shorter clip to build from. Asserting the
    /// call returns `Ok` cannot see that; only the CONTENTS of the returned
    /// `World` can, which is why this checks them against an uncancelled
    /// control built from the very same clip.
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
            "the uncancelled control must actually build a graph, or this test proves nothing"
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
            "the pixel loop must bail on its FIRST poll -- {} polls means it walked all \
             {} pixels instead",
            before_the_pixels + (SIDE * SIDE) as usize,
            SIDE * SIDE
        );
    }

    /// **The last gap in cancel promptness.** The per-pixel loop polls at the
    /// TOP of each iteration, so a cancel arriving during the final pixel (or
    /// during the subtitle step, which polls not at all) was noticed by
    /// nothing until the caller's own check -- by which time `chip::finish`
    /// had already run both grid collision checks over every brick in the
    /// render.
    ///
    /// This budgets exactly the polls a whole render makes -- one per frame,
    /// the post-decode guard, one per pixel -- so every one of them reads "not
    /// cancelled" and the FIRST poll to flip is the new pre-publish one. The
    /// render must then come back empty, exactly as a cancel anywhere earlier
    /// does.
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
             to see the cancellation -- otherwise this test is measuring something else"
        );
        assert!(
            world.bricks.is_empty() && world.grids.is_empty() && world.wires.is_empty(),
            "a cancel landing after the last pixel must still return an EMPTY world rather \
             than paying for chip::finish's two grid collision checks"
        );

        // The control: one more poll of budget and the same render completes.
        let mut ran =
            CancelAtPoll { polls: std::cell::Cell::new(0), after_polls: whole_render + 1 };
        let full = build_brick_world(&clip, &AnimOptions::default(), &mut ran).expect("build");
        assert_eq!(
            full.bricks.len(),
            pixels + 1,
            "the uncancelled control must really build the screen, or this proves nothing"
        );
    }

    /// `NoProgress` itself -- the library's actual default, used by the CLI
    /// and every pre-existing test -- must never cancel a real render
    /// through `build_brick_world`. `no_progress_swallows_everything_
    /// without_panicking` in `progress.rs` already checks `NoProgress` in
    /// isolation; this checks it wired to the real render path instead.
    #[test]
    fn no_progress_never_cancels_a_real_render() {
        let clip = solid_clip(10);
        let world = build_brick_world(&clip, &AnimOptions::default(), &mut crate::progress::NoProgress)
            .expect("NoProgress must never turn a normal render into an error");
        let _ = world;
    }
}
