//! The subtitle display: one `Component_TextDisplay` laid across the bottom of
//! the picture, fed a per-frame string out of its own `ArrayVar` by its own
//! `ArrayVar_Get`.
//!
//! Two gates for a whole subtitle track. This is the same shape
//! [`super::text_bricks`] gives a band of image rows -- an `ArrayVar` holding
//! one string per frame, an `ArrayVar_Get` reading it at the frame index, and
//! the value wired into a `TextDisplay`'s `Text` port -- with one component
//! instead of fifty-four, so it is built from the same constants
//! ([`TEXT_DISPLAY`], [`TEXT_PORT`]) rather than a second description of the
//! same thing.
//!
//! # Why a component and not burned pixels
//!
//! A `TextDisplay` has no resolution: it draws vector glyphs at whatever size
//! its geometry asks for. Burning subtitles into a 192x108 frame makes a glyph
//! two pixels tall, which no quality setting rescues. Because the component is
//! wired only to the frame index, it works identically over brick mode,
//! colour-array mode and text mode.
//!
//! # Exec: fan-out, never fan-in
//!
//! The subtitle's `Get.Exec` hangs off the same per-bank exec entry the screen
//! already uses -- one more consumer of an output that already drives many.
//! Exec fan-out is supported and free; exec fan-in (two sources into one exec
//! input) is unverified, and nothing here produces it: the subtitle `Get` is a
//! fresh exec input with exactly one source.
use super::bricks::{ARRAY_GET, ARRAY_VAR, AnimOptions, SELECT};
use super::chip::Chip;
use super::clock::gate;
use super::layout::{CELL, GATE_HALF, STAGE_BASE_Z};
use super::pack;
use crate::text::{
    ANCHOR_BOTTOM_CENTRE, MAX_COMPONENT_CHARS, OUTLINE_OUTLINED, TextOptions, Vector2f,
    add_text_block_styled,
};
use brdb::{AsBrdbValue, Position, Vector3f, WirePort, World, schema::WireArrayVariant};

/// The text component and its input port, reused verbatim from
/// [`super::text_bricks`] -- the subtitle is the same component the text
/// renderer drives, so a rename can only ever happen in one place.
pub use super::text_bricks::{TEXT_DISPLAY, TEXT_PORT};

/// How much bigger a subtitle line is than one row of the screen, by default.
///
/// Unverified by eye in game; an arithmetic guess, and this constant is
/// deliberately the single place to change it. At 192 px wide with
/// `char_repeat` 2 the screen is 384 glyph cells across, while a subtitle
/// line is 40-60 characters: at equal glyph size the text would occupy about
/// a seventh of the width and be useless, while at 6x it covers roughly
/// half, which is what a subtitle normally looks like. `--subtitle-scale`
/// exposes it, since the right value depends on the screen's width and the
/// track's typical line length.
pub const DEFAULT_SUBTITLE_SCALE: f32 = 6.0;

/// How many world units the subtitle anchor is lifted "up the picture" -- off
/// its bare bottom-centre baseline -- by default.
///
/// Calibrated by eye in one mode at one configuration only: `--anim-mode
/// text` at 192x108 with `--subtitle-scale 6`. A different resolution,
/// scale, or `--anim-mode brick`/`color-array` render may want a different
/// value, which is why `--subtitle-lift` exposes it rather than leaving it a
/// buried constant. See [`ScreenExtent`] and `bricks::subtitle_extent` for
/// how each renderer applies it along whichever axis is "up the picture" for
/// its own screen orientation.
pub const DEFAULT_SUBTITLE_LIFT: f32 = 8.0;

/// Rounds `--subtitle-lift`'s world-unit value to the nearest whole brick
/// coordinate, the same way every other float-to-position conversion in this
/// module does.
///
/// A non-finite input (NaN/infinite -- unreachable from the CLI/GUI's own
/// parsers, but not necessarily from a future caller) is treated as no lift at
/// all rather than producing a garbage position: the same guard
/// [`glyph_scale`] applies to `subtitle_scale`.
pub fn lift_units(lift: f32) -> i32 {
    if lift.is_finite() { lift.round() as i32 } else { 0 }
}

/// The subtitle's `Component_TextDisplay` `Anchor`: horizontally centred
/// (X 0.5) and anchored at the text's bottom edge (Y 1.0).
///
/// This is what makes [`ScreenExtent::anchor`] a bottom-centre position rather
/// than a bottom-left one: the cube names the block's bottom-centre point, so a
/// line of any length stays centred on the picture and a two-line cue grows
/// upward into the picture instead of downward out of it. Re-exported from
/// [`crate::text`] so the value and its explanation live next to each other.
pub const SUBTITLE_ANCHOR: Vector2f = ANCHOR_BOTTOM_CENTRE;

/// The subtitle's `Outline`: `EBRTextOutline::Outlined`, a solid outline drawn
/// around each glyph in the component's `OutlineColor` (opaque black), at
/// [`SUBTITLE_OUTLINE_WIDTH`].
///
/// A subtitle is the one text this crate draws over content rather than over
/// empty world: white glyphs vanish against a white frame, and the outline is
/// what makes a cue readable regardless of what the picture is doing behind it.
pub const SUBTITLE_OUTLINE: u8 = OUTLINE_OUTLINED;

/// The subtitle's `OutlineWidth` -- the crate-wide
/// [`crate::text::DEFAULT_OUTLINE_WIDTH`]. Kept as a named re-export rather
/// than inlined so the subtitle's styling still reads as one block next to
/// [`SUBTITLE_ANCHOR`] and [`SUBTITLE_OUTLINE`].
pub const SUBTITLE_OUTLINE_WIDTH: f32 = crate::text::DEFAULT_OUTLINE_WIDTH;

/// Where a renderer's screen ended, so the subtitle can be laid across it.
///
/// Each renderer fills this in from geometry it has already computed (the
/// display-brick pitch, or the text layout's world line height) rather than
/// a second, independent derivation that could disagree with where the
/// bricks actually went.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenExtent {
    /// Main-grid position of the subtitle's anchor cube: the bottom centre
    /// of the picture, one anchor-cube proud of the picture's own surface.
    ///
    /// Bottom centre because the subtitle carries [`SUBTITLE_ANCHOR`]: the
    /// cube names the text block's bottom-centre point, so the line spreads
    /// either side of this position and grows upward from it. Proud of the
    /// surface because a block drawn in the picture's own plane would
    /// z-fight with it -- the cube rests flush on the picture's front face,
    /// one [`crate::text::ANCHOR_CUBE_HALF`] past it. Which axis is "in
    /// front" differs per renderer (world +X for text mode's upright wall,
    /// +Z for a brick mode's ground screen), which is why this is a whole
    /// position and not an offset.
    ///
    /// May be negative: brdb round-trips negative main-grid coordinates
    /// exactly (see `bricks::subtitle_extent`'s doc for the concrete case
    /// and the encoder-level proof). Text mode nonetheless keeps its own
    /// main grid non-negative by construction
    /// (`crate::text::add_text_tiles` translates the whole glyph grid),
    /// moving the picture rather than anchoring a subtitle outside that
    /// translation -- see `text_bricks::subtitle_centre_shift`.
    pub anchor: Position,
    /// World units one image row of the screen occupies.
    ///
    /// This is what `subtitle_scale` multiplies, which is why it is a
    /// per-renderer input rather than a constant: a text-mode row is one line
    /// of glyphs (under a world unit), while a brick-mode row is a whole
    /// display-brick pitch. Scaling against the row makes one
    /// `subtitle_scale` mean the same thing in every mode.
    pub row_height: f32,
    /// Which face of the anchor cube the glyphs are drawn on
    /// ([`crate::text::FACE_X_POSITIVE`] / [`crate::text::FACE_Z_POSITIVE`]),
    /// i.e. which plane the subtitle lies in.
    ///
    /// A per-renderer input because the three renderers do not build the
    /// same kind of screen: text mode's is a vertical wall facing world +X,
    /// so its subtitle goes on the same upright face every glyph band uses.
    /// Both brick encodings lay their screen flat and present its top to the
    /// viewer, so a subtitle on the +X face would stand edge-on and
    /// unreadable; those two pass [`crate::text::FACE_Z_POSITIVE`] instead.
    pub face: u8,
}

/// The frame-index plumbing a renderer has already built, borrowed so the
/// subtitle can hang off it instead of duplicating it.
///
/// All three slices are the renderers' own locals, and all three are indexed
/// by bank: `index_of_bank[k]` is the frame index rebased so bank `k`'s array
/// is addressed from zero, `entry_of_bank[k]` is the exec source that fires
/// while bank `k` is live, and `ge[k - 1]` is true once the frame index has
/// reached bank `k` (which is `Select`'s `bSelectB` sense). Reusing them is
/// what keeps a subtitle track at two gates: it adds no comparator, no
/// subtract and no branch of its own, at any bank count.
#[derive(Clone, Copy)]
pub struct FrameIndex<'a> {
    /// Per-bank frame index. One entry per bank.
    pub index_of_bank: &'a [WirePort],
    /// Per-bank exec entry. One entry per bank. The subtitle's `Get.Exec`
    /// fans out from this; it never joins an existing exec input.
    pub entry_of_bank: &'a [WirePort],
    /// Boundary comparators, `n_banks - 1` of them: `ge[k - 1]` is true from
    /// the first frame of bank `k` onward.
    pub ge: &'a [WirePort],
}

/// Errors if any frame's subtitle exceeds a `TextDisplay`'s character limit,
/// naming the offending frame's timestamp.
///
/// The game truncates past [`MAX_COMPONENT_CHARS`] silently: the save
/// writes, the render plays, and the only symptom is a line that stops
/// mid-word -- invisible until a human happens to look at that one frame. So
/// this is a hard error and never a truncation. `fps` is used for nothing but
/// turning the frame index into a timestamp a human can seek to.
pub fn check_char_limit(per_frame: &[String], fps: f32) -> Result<(), String> {
    for (i, line) in per_frame.iter().enumerate() {
        // Unicode chars, not bytes -- the limit is a char count, and subtitle
        // text is exactly the place non-ASCII shows up.
        let chars = line.chars().count();
        if chars > MAX_COMPONENT_CHARS {
            let at = if fps.is_finite() && fps > 0.0 {
                format!("{:.3}s", i as f64 / fps as f64)
            } else {
                "an unknown time".to_string()
            };
            return Err(format!(
                "the subtitle showing at {at} (frame {i}) is {chars} characters, over the \
                 {MAX_COMPONENT_CHARS}-character TextDisplay limit -- the game truncates past \
                 that silently, so this is rejected rather than cut short; shorten the cue or \
                 split it"
            ));
        }
    }
    Ok(())
}

/// Place the subtitle `TextDisplay` and wire it to `frame_index`.
///
/// Adds exactly one `ArrayVar` and one `ArrayVar_Get` per bank, plus one
/// `Select` per bank boundary -- the same cost as a single text band, and two
/// gates flat for any clip short of [`pack::BANK_FRAMES`] frames.
///
/// Call this last, after every other gate is in the chip and immediately
/// before [`super::chip::finish`]. The subtitle's gates are placed a clear
/// cell beyond the chip's current x extent, which is only collision-free
/// against bricks that already exist -- a gate added afterwards at a deeper
/// service row would land on top of them. All three renderers call it from
/// exactly that position.
///
/// `fps` is carried only so [`check_char_limit`] can name a timestamp.
pub fn add_subtitle_display(
    world: &mut World,
    chip: &mut Chip,
    frame_index: FrameIndex<'_>,
    per_frame: &[String],
    fps: f32,
    opts: &AnimOptions,
    screen: ScreenExtent,
) -> Result<(), String> {
    check_char_limit(per_frame, fps)?;

    let bank_size = opts.bank_size.max(1);
    let n_banks = per_frame.len().div_ceil(bank_size).max(1);
    if frame_index.index_of_bank.len() != n_banks
        || frame_index.entry_of_bank.len() != n_banks
        || frame_index.ge.len() + 1 != n_banks
    {
        return Err(format!(
            "{} subtitle frames at bank size {bank_size} bank {n_banks} ways, but the renderer \
             supplied {} per-bank indices, {} exec entries and {} boundary comparators -- the \
             subtitle array has to bank exactly as the screen's arrays do or it would read the \
             wrong frame past a seam",
            per_frame.len(),
            frame_index.index_of_bank.len(),
            frame_index.entry_of_bank.len(),
            frame_index.ge.len(),
        ));
    }

    // --- The component, on the main grid ------------------------------------
    //
    // Frame 0's line is baked in as the authored `Text`, exactly as
    // `text_bricks` bakes frame 0 into every band, so the save shows the right
    // subtitle before the clock has ticked once.
    let scale = glyph_scale(opts, screen.row_height);
    let text_opts = TextOptions {
        // LineOffset and the glyph-fit Offsets are world-unit nudges sized for
        // the screen's glyphs; a subtitle glyph `scale` times bigger needs
        // them `scale` times bigger too, or the nudge that centres a screen
        // pixel becomes a rounding error on a subtitle line.
        line_offset: opts.text.line_offset * scale,
        ..opts.text.clone()
    };
    let text_id = add_text_block_styled(
        world,
        per_frame.first().cloned().unwrap_or_default(),
        screen.anchor,
        opts.text.line_height * scale,
        opts.text.kerning * scale,
        Vector3f {
            x: opts.text.offset_x * scale,
            y: opts.text.offset_y * scale,
            // Out-of-plane, not in-plane: this pushes the glyphs off the
            // anchor's face rather than sizing them, so it does not scale.
            //
            // Not what puts the subtitle in front of the picture --
            // `ScreenExtent::anchor` already places the whole cube proud of
            // the picture's surface. This stays the font preset's own
            // glyph-fit nudge, exactly as it is for every other block.
            z: opts.text.offset_z,
        },
        false,
        // The plane the glyphs are drawn in -- upright over text mode's wall,
        // flat over a brick mode's ground screen. See `ScreenExtent::face`.
        screen.face,
        // Centred on the anchor and growing upward from it, with an outline so
        // the cue reads over whatever the picture is showing.
        SUBTITLE_ANCHOR,
        SUBTITLE_OUTLINE,
        SUBTITLE_OUTLINE_WIDTH,
        &text_opts,
    );

    // --- The gates, in the chip ---------------------------------------------
    //
    // A clear cell beyond everything already placed. Deriving the position
    // from the chip's own contents rather than from a lattice row/stage keeps
    // this working across three renderers whose lattices are laid out
    // differently, and it cannot collide: every existing brick's outer x face
    // is at most `base_x - CELL`.
    let base_x = chip
        .placed()
        .iter()
        .map(|(p, h)| p.x + h.x)
        .max()
        .unwrap_or(0)
        + CELL;
    let at = |slot: i32| Position {
        x: base_x + GATE_HALF.x,
        y: slot * CELL + GATE_HALF.y,
        z: STAGE_BASE_Z + GATE_HALF.z,
    };
    let mut slot = 0;

    let banks = pack::bank_frames(per_frame, bank_size);
    debug_assert_eq!(banks.len(), n_banks, "bank_frames must agree with n_banks");
    let mut get_of_bank = Vec::with_capacity(n_banks);
    for (k, frames) in banks.iter().enumerate() {
        let array = gate(
            chip,
            "B_1x1_Gate_Variable_Array",
            ARRAY_VAR,
            at(slot),
            vec![(
                "Value",
                Box::new(WireArrayVariant::StringArray(frames.to_vec())) as Box<dyn AsBrdbValue>,
            )],
        );
        slot += 1;
        let get = gate(
            chip,
            "B_1x1_Gate_Exec_ArrayVar_Get",
            ARRAY_GET,
            at(slot),
            vec![],
        );
        slot += 1;
        world.add_wire_connection(
            WirePort::new(array, ARRAY_VAR, "ArrayVarRef"),
            WirePort::new(get, ARRAY_GET, "ArrayVarRef"),
        );
        world.add_wire_connection(
            frame_index.index_of_bank[k].clone(),
            WirePort::new(get, ARRAY_GET, "Index"),
        );
        // The fan-out: one more consumer of an exec output the screen already
        // drives. This `Get.Exec` gains exactly one source, so no fan-in.
        world.add_wire_connection(
            frame_index.entry_of_bank[k].clone(),
            WirePort::new(get, ARRAY_GET, "Exec"),
        );
        get_of_bank.push(get);
    }

    // One select per boundary, cascading -- identical to a text band's. For a
    // frame in bank j, ge[0..j] are true so select j picks bank j and every
    // later select passes it through unchanged.
    let mut value = WirePort::new(get_of_bank[0], ARRAY_GET, "Value");
    for k in 1..n_banks {
        let sel = gate(
            chip,
            "B_1x1_Gate_Expr_Select",
            SELECT,
            at(slot),
            vec![],
        );
        slot += 1;
        world.add_wire_connection(
            frame_index.ge[k - 1].clone(),
            WirePort::new(sel, SELECT, "bSelectB"),
        );
        world.add_wire_connection(value, WirePort::new(sel, SELECT, "InputA"));
        world.add_wire_connection(
            WirePort::new(get_of_bank[k], ARRAY_GET, "Value"),
            WirePort::new(sel, SELECT, "InputB"),
        );
        value = WirePort::new(sel, SELECT, "Output");
    }

    // Chip -> main grid. The endpoints live in different grids and
    // `World::add_wire_connection` emits the `RemoteWirePortSource` itself.
    world.add_wire_connection(value, WirePort::new(text_id, TEXT_DISPLAY, TEXT_PORT));
    Ok(())
}

/// How much bigger the subtitle's glyphs are than the screen's own, given the
/// world height of one screen row.
///
/// `subtitle_scale` is defined against one screen row, not against the text
/// options' line height, so a mode whose row is not a line of text -- both
/// brick encodings, where a row is a display-brick pitch -- picks up that
/// ratio too and one `subtitle_scale` means the same thing everywhere.
fn glyph_scale(opts: &AnimOptions, row_height: f32) -> f32 {
    let screen_line = opts.text.line_world_height * opts.text.pitch_y;
    if screen_line.is_finite() && screen_line > 0.0 && row_height.is_finite() && row_height > 0.0 {
        opts.subtitle_scale * (row_height / screen_line)
    } else {
        // Degenerate geometry: fall back to the bare scale rather than
        // producing an infinite or NaN LineHeight, which writes a save the
        // game cannot lay out.
        opts.subtitle_scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::bricks::{
        BRANCH, CHANGE_DETECTOR, COMPARE_GE, SUBTRACT, ScreenGeometry, build_brick_world,
    };
    use crate::anim::chip;
    use crate::anim::color_bricks::build_color_array_world;
    use crate::anim::text_bricks::build_text_world;
    use crate::progress::NoProgress;
    use crate::subs::{Cue, Subtitles};
    use crate::video::Clip;
    use brdb::{
        IntVector, IntoReader,
        schema::{BrdbValue, WireVariant},
    };
    use image::{Rgba, RgbaImage};
    use std::sync::Arc;

    /// Main-grid bricks live on grid 1 in a written save -- `brdb`'s
    /// `World::to_unsaved` says so where it packs them ("Main grid bricks are
    /// on grid 1"), and the microchip's inner grid is a later id.
    const MAIN_GRID: usize = 1;

    /// Bricks carrying `class`, across both grids.
    ///
    /// The subtitle spans them: its `TextDisplay` is a main-grid brick and its
    /// gates are inner-grid bricks, so a counter that looked at one grid only
    /// would silently pass whichever assertion it could not see.
    fn count_components(world: &World, class: &str) -> usize {
        world
            .bricks
            .iter()
            .chain(world.grids.iter().flat_map(|(_, bricks)| bricks.iter()))
            .filter(|b| {
                b.components
                    .iter()
                    .any(|c| c.component_type().is_some_and(|t| t.to_string() == class))
            })
            .count()
    }

    /// Every inner-grid brick centre in the lattice's own coordinate space.
    /// `World::add_brick_grid` stores inner bricks shifted by `-CHUNK_HALF`;
    /// this undoes that, exactly as `tests/anim_color.rs` does.
    fn inner_grid_positions(world: &World) -> Vec<Position> {
        world.grids[0]
            .1
            .iter()
            .map(|b| b.position + Position::CHUNK_HALF)
            .collect()
    }

    /// A bare harness: a chip holding the frame-index plumbing a renderer
    /// would build, and nothing else, so every component this counts belongs
    /// to the subtitle. A real render would drown the subtitle's one
    /// `ArrayVar` in the screen's own.
    ///
    /// The per-bank subtract/comparator/branch chain is copied from the
    /// renderers deliberately: `add_subtitle_display` must work against the
    /// wiring they actually produce, and none of those three gate classes is
    /// one the assertions below count.
    fn try_render_with_subs_banked<S: AsRef<str>>(
        per_frame: &[S],
        bank_size: usize,
    ) -> Result<World, String> {
        let opts = AnimOptions {
            bank_size,
            ..AnimOptions::default()
        };
        let frames: Vec<String> = per_frame.iter().map(|s| s.as_ref().to_string()).collect();
        let bank_size = bank_size.max(1);
        let n_banks = frames.len().div_ceil(bank_size).max(1);

        let mut world = World::new();
        let mut chip = chip::new_chip(
            &mut world,
            Position { x: 20, y: 20, z: 2 },
            Vector3f { x: 20.0, y: 20.0, z: 20.0 },
            IntVector { x: 5, y: 5, z: 5 },
        );
        let at = |slot: i32| Position {
            x: GATE_HALF.x,
            y: slot * CELL + GATE_HALF.y,
            z: STAGE_BASE_Z + GATE_HALF.z,
        };
        let mut slot = 0;

        let pin = chip::add_input_pin(&mut chip, "Frame", at(slot));
        slot += 1;
        let frame_index = chip::pin_source(pin, true);
        let detector = gate(
            &mut chip,
            "B_1x1_Gate_Expr_ChangeDetectorExec",
            CHANGE_DETECTOR,
            at(slot),
            vec![],
        );
        slot += 1;
        world.add_wire_connection(
            frame_index.clone(),
            WirePort::new(detector, CHANGE_DETECTOR, "Input"),
        );

        let mut index_of_bank = vec![frame_index.clone()];
        let mut ge = Vec::new();
        for k in 1..n_banks {
            let sub = gate(
                &mut chip,
                "B_1x1_Gate_Expr_MathSubtract",
                SUBTRACT,
                at(slot),
                vec![(
                    "InputB",
                    Box::new(WireVariant::Number((k * bank_size) as f64)) as Box<dyn AsBrdbValue>,
                )],
            );
            slot += 1;
            world.add_wire_connection(frame_index.clone(), WirePort::new(sub, SUBTRACT, "InputA"));
            index_of_bank.push(WirePort::new(sub, SUBTRACT, "Output"));

            let cmp = gate(
                &mut chip,
                "B_1x1_Gate_Expr_CompareGreaterOrEqual",
                COMPARE_GE,
                at(slot),
                vec![(
                    "InputB",
                    Box::new(WireVariant::Int((k * bank_size) as i64)) as Box<dyn AsBrdbValue>,
                )],
            );
            slot += 1;
            world.add_wire_connection(frame_index.clone(), WirePort::new(cmp, COMPARE_GE, "InputA"));
            ge.push(WirePort::new(cmp, COMPARE_GE, "bOutput"));
        }

        let mut entry_of_bank = Vec::with_capacity(n_banks);
        let mut exec_src = WirePort::new(detector, CHANGE_DETECTOR, "OnChanged");
        for bi in 0..n_banks {
            if bi + 1 < n_banks {
                let br = gate(&mut chip, "B_1x1_Gate_Exec_Branch", BRANCH, at(slot), vec![]);
                slot += 1;
                world.add_wire_connection(ge[bi].clone(), WirePort::new(br, BRANCH, "bCond"));
                world.add_wire_connection(exec_src, WirePort::new(br, BRANCH, "Exec"));
                exec_src = WirePort::new(br, BRANCH, "ExecOutA");
                entry_of_bank.push(WirePort::new(br, BRANCH, "ExecOutB"));
            } else {
                entry_of_bank.push(exec_src.clone());
            }
        }

        add_subtitle_display(
            &mut world,
            &mut chip,
            FrameIndex {
                index_of_bank: &index_of_bank,
                entry_of_bank: &entry_of_bank,
                ge: &ge,
            },
            &frames,
            10.0,
            &opts,
            ScreenExtent {
                anchor: Position { x: 2, y: 2, z: 4 },
                row_height: 1.0,
                face: crate::text::FACE_X_POSITIVE,
            },
        )?;
        chip::finish(&mut world, chip)?;
        world.register_used_components();
        Ok(world)
    }

    /// Discards the `World` so the result is `Debug` -- `brdb::World` is not,
    /// and `Result::expect_err` needs the Ok side to be.
    fn try_render_with_subs<S: AsRef<str>>(per_frame: &[S]) -> Result<(), String> {
        try_render_with_subs_banked(per_frame, usize::MAX).map(|_| ())
    }

    fn render_with_subs<S: AsRef<str>>(per_frame: &[S]) -> World {
        try_render_with_subs_banked(per_frame, usize::MAX)
            .expect("the subtitle display must build")
    }

    fn render_with_subs_banked<S: AsRef<str>>(per_frame: &[S], bank_size: usize) -> World {
        try_render_with_subs_banked(per_frame, bank_size)
            .expect("the banked subtitle display must build")
    }

    /// A real render with `subtitles: None`. Unlike the harness above, this
    /// exercises the gate the renderers put the whole feature behind, so the
    /// "adds no components at all" assertion has something to be wrong about.
    fn render_without_subs(frames: usize) -> World {
        // `control_buttons: false`: this test asserts a subtitle-free render adds
        // no `Component_TextDisplay`, and the default-on control-button labels
        // are TextDisplays that have nothing to do with subtitles.
        let opts = AnimOptions { control_buttons: false, ..AnimOptions::default() };
        build_brick_world(&clip(2, 2, frames), &opts, &mut NoProgress)
            .expect("a subtitle-free render must build")
    }

    /// A tiny opaque clip. Nothing about its content matters here -- only its
    /// length and that no pixel is culled.
    fn clip(w: u32, h: u32, frames: usize) -> Clip {
        Clip {
            width: w,
            height: h,
            fps: 10.0,
            frames: (0..frames)
                .map(|i| RgbaImage::from_pixel(w, h, Rgba([(i % 255) as u8, 40, 90, 255])))
                .collect(),
        }
    }

    /// Every string in every `ArrayVar` the save actually persisted.
    ///
    /// Component data is only reachable through a written file --
    /// `BrdbComponent` exposes `component_type()` and nothing else -- so this
    /// round-trips through a `.brz`, the same way `text_bricks`' tests read
    /// their band strings. The bytes are not reproducible run to run, so the
    /// round-trip is for reading structure and never for comparing files.
    fn subtitle_strings(world: &World) -> Vec<String> {
        let path = std::env::temp_dir().join(format!(
            "h2b_subs_{}_{:?}.brz",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
        let db = brdb::Brz::open(&path).expect("reopen").into_reader();
        let mut grid = None;
        for index in db.entity_chunk_index().expect("entity chunk index") {
            for e in db.entity_chunk(index).expect("entity chunk") {
                if e.is_microchip_grid() {
                    grid = e.id;
                }
            }
        }
        let gid = grid.expect("exactly one microchip grid must be published");
        let mut out = Vec::new();
        for chunk in &db.brick_chunk_index(gid).expect("chunk index") {
            let (_soa, structs) = db.component_chunk(gid, chunk.index).expect("components");
            for s in &structs {
                if s.get_name() == "BrickComponentData_WireGraphPseudo_ArrayVar"
                    && let Some(value) = s.get("Value")
                {
                    let variant: WireArrayVariant =
                        value.try_into().expect("ArrayVar Value must decode");
                    if let WireArrayVariant::StringArray(v) = variant {
                        out.extend(v);
                    }
                }
            }
        }
        let _ = std::fs::remove_file(&path);
        out
    }

    /// What one persisted `Component_TextDisplay` says about how it is drawn.
    ///
    /// `outline` alone is not enough to know a subtitle will actually show an
    /// outline: the enum only selects a style, and the colour, width and the
    /// two "yes, really use them" booleans have to reach the save with it.
    /// They are written unconditionally by `text_display_component`, which is
    /// exactly the kind of thing that is true until it isn't -- so they are
    /// read back rather than assumed.
    #[derive(Clone, Debug, PartialEq)]
    struct TextStyle {
        text: String,
        anchor: (f32, f32),
        outline: u64,
        /// `(R, G, B, A)`.
        outline_color: (u8, u8, u8, u8),
        outline_width: f32,
        override_outline_color: bool,
        sharp_outlines: bool,
    }

    /// Every main-grid `Component_TextDisplay` the save actually persisted.
    ///
    /// Round-trips through a written `.brz` for the same reason
    /// [`subtitle_strings`] does: component property values are unreachable
    /// from an in-memory `World`, which exposes only `component_type()`. So an
    /// assertion about `Anchor` or `Outline` that did not write a file would be
    /// asserting about the constant it was written from, not about the save.
    /// The bytes are never compared -- only this structure is read back.
    fn text_display_styles(world: &World) -> Vec<TextStyle> {
        let path = std::env::temp_dir().join(format!(
            "h2b_subs_style_{}_{:?}.brz",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
        let db = brdb::Brz::open(&path).expect("reopen").into_reader();
        let mut out = Vec::new();
        for chunk in &db.brick_chunk_index(MAIN_GRID).expect("chunk index") {
            let (_soa, structs) = db
                .component_chunk(MAIN_GRID, chunk.index)
                .expect("components");
            for s in &structs {
                if s.get_name() != "BrickComponentData_TextDisplay" {
                    continue;
                }
                let text = match s.get("Text") {
                    Some(BrdbValue::String(v)) => v.clone(),
                    other => panic!("Text is not a string: {other:?}"),
                };
                let anchor = match s.get("Anchor") {
                    Some(BrdbValue::Struct(v)) => v.clone(),
                    other => panic!("Anchor is not a struct: {other:?}"),
                };
                let axis = |n: &str| match anchor.get(n) {
                    Some(BrdbValue::F32(v)) => *v,
                    other => panic!("Anchor.{n} is not an f32: {other:?}"),
                };
                // `Outline` is typed as `EBRTextOutline` in the save schema, so
                // it comes back as an enum; the `u8` fallback is there so a
                // schema that ever stops calling it one fails loudly on the
                // value rather than on the shape.
                let outline = match s.get("Outline") {
                    Some(BrdbValue::Enum(e)) => e.value,
                    Some(BrdbValue::U8(v)) => *v as u64,
                    other => panic!("Outline is neither an enum nor a u8: {other:?}"),
                };
                let colour = match s.get("OutlineColor") {
                    Some(BrdbValue::Struct(v)) => v.clone(),
                    other => panic!("OutlineColor is not a struct: {other:?}"),
                };
                let channel = |n: &str| match colour.get(n) {
                    Some(BrdbValue::U8(v)) => *v,
                    other => panic!("OutlineColor.{n} is not a u8: {other:?}"),
                };
                let flag = |n: &str| match s.get(n) {
                    Some(BrdbValue::Bool(v)) => *v,
                    other => panic!("{n} is not a bool: {other:?}"),
                };
                out.push(TextStyle {
                    text,
                    anchor: (axis("X"), axis("Y")),
                    outline,
                    outline_color: (
                        channel("R"),
                        channel("G"),
                        channel("B"),
                        channel("A"),
                    ),
                    outline_width: match s.get("OutlineWidth") {
                        Some(BrdbValue::F32(v)) => *v,
                        other => panic!("OutlineWidth is not an f32: {other:?}"),
                    },
                    override_outline_color: flag("bOverrideOutlineColor"),
                    sharp_outlines: flag("bSharpOutlines"),
                });
            }
        }
        let _ = std::fs::remove_file(&path);
        out
    }

    #[test]
    fn a_subtitle_track_costs_one_array_and_one_get() {
        let world = render_with_subs(&["", "hi", "hi"]);
        assert_eq!(count_components(&world, ARRAY_VAR), 1);
        assert_eq!(count_components(&world, ARRAY_GET), 1);
    }

    #[test]
    fn the_array_has_one_entry_per_frame_including_the_empty_ones() {
        let world = render_with_subs(&["", "hi", ""]);
        assert_eq!(subtitle_strings(&world), vec!["", "hi", ""]);
    }

    #[test]
    fn no_subtitles_option_adds_no_components_at_all() {
        let world = render_without_subs(3);
        assert_eq!(count_components(&world, TEXT_DISPLAY), 0);
    }

    #[test]
    fn a_cue_over_the_component_limit_errors_naming_its_time() {
        let huge = "x".repeat(crate::text::MAX_COMPONENT_CHARS + 1);
        let err = try_render_with_subs(&[huge]).expect_err("must reject");
        assert!(err.contains("10000") || err.contains("limit"), "{err}");
    }

    #[test]
    fn the_display_never_sits_at_a_negative_coordinate() {
        let world = render_with_subs(&["hi"]);
        for pos in inner_grid_positions(&world) {
            assert!(pos.x >= 0 && pos.y >= 0 && pos.z >= 0, "negative coord {pos:?}");
        }
    }

    #[test]
    fn spilling_past_one_bank_adds_a_select_per_boundary() {
        let world = render_with_subs_banked(&["a", "b", "c", "d", "e"], 2);
        assert_eq!(count_components(&world, ARRAY_VAR), 3, "one array per bank");
        assert_eq!(count_components(&world, SELECT), 2, "one select per boundary");
    }

    // --- Beyond the plan's six: the renderer wiring -------------------------
    //
    // The six above pin the piece in isolation. These pin the thing the piece
    // exists for -- that all three renderers gain it, gain it only when asked,
    // and gain exactly two gates for it.

    /// A track whose cue covers every frame of a 10 fps clip.
    fn track() -> Arc<Subtitles> {
        Arc::new(Subtitles::new(vec![Cue {
            start_s: 0.0,
            end_s: 60.0,
            text: "a subtitle".to_string(),
        }]))
    }

    /// Inner-grid bricks: every gate plus the chip's I/O pins. The delta
    /// between two renders of the same clip is what matters, and the pin count
    /// is identical on both sides of it.
    fn inner_bricks(world: &World) -> usize {
        world.grids[0].1.len()
    }

    #[test]
    fn every_renderer_spends_exactly_two_gates_on_a_subtitle_track() {
        type Build = fn(&Clip, &AnimOptions, &mut NoProgress) -> Result<World, String>;
        let renderers: [(&str, Build); 3] = [
            ("hex", |c, o, p| build_brick_world(c, o, p)),
            ("colour-array", |c, o, p| build_color_array_world(c, o, p)),
            ("text", |c, o, p| build_text_world(c, o, p)),
        ];
        for (name, build) in renderers {
            let c = clip(16, 8, 4);
            let without = build(&c, &AnimOptions::default(), &mut NoProgress)
                .unwrap_or_else(|e| panic!("{name} without subtitles: {e}"));
            let with = build(
                &c,
                &AnimOptions {
                    subtitles: Some(track()),
                    ..AnimOptions::default()
                },
                &mut NoProgress,
            )
            .unwrap_or_else(|e| panic!("{name} with subtitles: {e}"));

            assert_eq!(
                inner_bricks(&with) - inner_bricks(&without),
                2,
                "{name}: a subtitle track is one ArrayVar and one Get, nothing else"
            );
            assert_eq!(
                count_components(&with, TEXT_DISPLAY) - count_components(&without, TEXT_DISPLAY),
                1,
                "{name}: exactly one subtitle component"
            );
            assert_eq!(
                with.bricks.len() - without.bricks.len(),
                1,
                "{name}: exactly one extra main-grid brick, the subtitle's anchor cube"
            );
            // A `Component_TextDisplay` on a brick-mode world is genuinely new
            // -- neither brick encoding has ever placed one -- so the
            // component type, its `Text` port and the font asset reference all
            // have to survive `register_used_components`. Encoding is where
            // that shows up; a world that only exists in memory proves
            // nothing.
            with.to_brz_vec()
                .unwrap_or_else(|e| panic!("{name} with subtitles must encode: {e}"));
        }
    }

    /// The subtitle's `Text` port must be driven exactly once, and its `Get`'s
    /// exec input must have exactly one source at every bank count -- fan-out
    /// is free, fan-in is unverified and nothing here may need it.
    #[test]
    fn no_exec_input_ever_has_two_sources_and_the_display_is_driven_once() {
        for bank_size in [usize::MAX, 3, 2, 1] {
            let opts = AnimOptions {
                bank_size,
                subtitles: Some(track()),
                ..AnimOptions::default()
            };
            let world = build_text_world(&clip(32, 8, 5), &opts, &mut NoProgress).expect("build");
            let mut seen = std::collections::HashSet::new();
            for wire in &world.wires {
                let t = &wire.target;
                let port = t.port_name.to_string();
                if port.contains("Exec") && !port.contains("Out") {
                    assert!(
                        seen.insert((t.brick_id, port.clone())),
                        "bank_size {bank_size}: exec input {}.{port} has more than one source",
                        t.brick_id
                    );
                }
            }
            let driven = world
                .wires
                .iter()
                .filter(|w| w.target.component_type.to_string() == TEXT_DISPLAY)
                .count();
            // One per band, plus the subtitle's.
            let bands = crate::anim::text_layout::plan_bands(32, 8, 2).unwrap().len();
            assert_eq!(driven, bands + 1, "bank_size {bank_size}");
        }
    }

    /// Every main-grid brick a subtitled text render places must stay
    /// non-negative -- text mode keeps its whole main grid non-negative by
    /// construction, and the subtitle is placed below everything else, which
    /// is the one thing that could push the stack under zero.
    #[test]
    fn a_subtitled_text_render_keeps_every_main_grid_brick_non_negative() {
        let opts = AnimOptions {
            subtitles: Some(track()),
            ..AnimOptions::default()
        };
        let world = build_text_world(&clip(64, 16, 3), &opts, &mut NoProgress).expect("build");
        for b in &world.bricks {
            assert!(
                b.position.x >= 0 && b.position.y >= 0 && b.position.z >= 0,
                "negative main-grid coord {:?}",
                b.position
            );
        }
    }

    /// `subtitle_scale` is defined against one screen row, so the same scale
    /// yields a bigger `LineHeight` on a screen whose rows are bigger.
    #[test]
    fn the_glyph_scale_tracks_the_screen_row_height() {
        let opts = AnimOptions::default();
        let one_row = opts.text.line_world_height * opts.text.pitch_y;
        assert!(
            (glyph_scale(&opts, one_row) - opts.subtitle_scale).abs() < 1e-6,
            "a screen whose row is one text line scales by exactly subtitle_scale"
        );
        assert!(
            glyph_scale(&opts, one_row * 4.0) > glyph_scale(&opts, one_row),
            "a screen with taller rows needs proportionally bigger subtitle glyphs"
        );
        // Degenerate geometry must not produce a NaN or infinite LineHeight,
        // which writes a save the game cannot lay out.
        assert!(glyph_scale(&opts, 0.0).is_finite());
        assert!(glyph_scale(&opts, f32::NAN).is_finite());
    }

    /// `lift_units` rounds to the nearest whole coordinate and treats a
    /// non-finite input as no lift, the same guard `glyph_scale` applies.
    #[test]
    fn lift_units_rounds_and_guards_non_finite_input() {
        assert_eq!(lift_units(8.0), 8);
        assert_eq!(lift_units(8.4), 8);
        assert_eq!(lift_units(8.6), 9);
        assert_eq!(lift_units(-1.5), -2, "rounds away from zero, like every other cast here");
        assert_eq!(lift_units(f32::NAN), 0);
        assert_eq!(lift_units(f32::INFINITY), 0);
        assert_eq!(lift_units(f32::NEG_INFINITY), 0);
    }

    /// Read back out of a written save, because component property values
    /// are unreachable from an in-memory `World`.
    ///
    /// The subtitle must be centred (Anchor.X 0.5) and anchored at its own
    /// bottom edge (Anchor.Y 1.0) with `EBRTextOutline::Outlined` (2) -- and,
    /// just as importantly, the glyph bands sharing the same world must still
    /// be top-left anchored with no outline, or the styled variant leaked into
    /// the pixel art and every frame of the picture is now outlined.
    #[test]
    fn the_subtitle_is_centred_bottom_anchored_and_outlined_and_the_bands_are_not() {
        let opts = AnimOptions {
            subtitles: Some(track()),
            // Buttons off: their labels are TextDisplays too, and this test
            // partitions the world's TextDisplays into the subtitle and the
            // glyph bands.
            control_buttons: false,
            ..AnimOptions::default()
        };
        let world = build_text_world(&clip(32, 8, 3), &opts, &mut NoProgress).expect("build");
        let styles = text_display_styles(&world);

        let bands = crate::anim::text_layout::plan_bands(32, 8, 2).unwrap().len();
        assert_eq!(styles.len(), bands + 1, "every band plus the subtitle");

        let (subtitle, rest): (Vec<_>, Vec<_>) =
            styles.into_iter().partition(|s| s.text == "a subtitle");
        assert_eq!(subtitle.len(), 1, "exactly one component carries the cue");
        let cue = &subtitle[0];
        assert_eq!(cue.anchor, (0.5, 1.0), "Anchor.X 0.5, Anchor.Y 1.0");
        assert_eq!(cue.outline, u64::from(SUBTITLE_OUTLINE), "EBRTextOutline::Outlined");
        // The enum only picks a style; these are what make it visible, and
        // they have to survive the write for the outline to appear in game.
        assert_eq!(cue.outline_color, (0, 0, 0, 255), "opaque black outline");
        assert_eq!(cue.outline_width, SUBTITLE_OUTLINE_WIDTH, "4.0, not the 2.0 every other block uses");
        assert!(cue.override_outline_color, "or the colour is ignored");
        assert!(cue.sharp_outlines);

        assert_eq!(rest.len(), bands, "the rest are the picture's glyph bands");
        for band in rest {
            assert_eq!(
                band.anchor,
                (0.0, 0.0),
                "a glyph band stays top-left anchored"
            );
            assert_eq!(band.outline, 0, "a glyph band stays unoutlined");
        }
    }

    /// A brick-mode subtitle carries the same style -- the anchor and outline
    /// belong to the subtitle, not to text mode.
    #[test]
    fn both_brick_encodings_style_their_subtitle_the_same_way() {
        type Build = fn(&Clip, &AnimOptions, &mut NoProgress) -> Result<World, String>;
        let renderers: [(&str, Build); 2] = [
            ("hex", |c, o, p| build_brick_world(c, o, p)),
            ("colour-array", |c, o, p| build_color_array_world(c, o, p)),
        ];
        for (name, build) in renderers {
            // 1x1 as well as a real screen: the subtitle sits over the
            // picture rather than a row past it, so a screen with nothing to
            // one side of it is where an off-by-one in the centring would put
            // the anchor cube inside a display brick -- which `chip::finish`
            // rejects, so the build itself is the assertion.
            for (w, h) in [(16u32, 8u32), (1, 1)] {
                let opts = AnimOptions {
                    subtitles: Some(track()),
                    // The default lift is calibrated against a real text mode
                    // render (see `DEFAULT_SUBTITLE_LIFT`'s doc) and, applied
                    // to brick mode's `-y`, overshoots a 1-row picture.
                    // Zeroed here because this test is about styling and
                    // centring, not the lift, at every size including that
                    // one-row edge case.
                    subtitle_lift: 0.0,
                    // Buttons off: their labels are TextDisplays, and this test
                    // asserts a brick screen's only TextDisplay is the subtitle.
                    control_buttons: false,
                    ..AnimOptions::default()
                };
                let world = build(&clip(w, h, 3), &opts, &mut NoProgress)
                    .unwrap_or_else(|e| panic!("{name} at {w}x{h}: {e}"));
                let styles = text_display_styles(&world);
                assert_eq!(styles.len(), 1, "{name}: a brick screen has no other text");
                let cue = &styles[0];
                assert_eq!(cue.text, "a subtitle", "{name}: frame 0's cue is baked in");
                assert_eq!(cue.anchor, (0.5, 1.0), "{name}: Anchor.X 0.5, Anchor.Y 1.0");
                assert_eq!(cue.outline, u64::from(SUBTITLE_OUTLINE), "{name}: EBRTextOutline::Outlined");
                assert_eq!(cue.outline_color, (0, 0, 0, 255), "{name}: black outline");
                assert_eq!(cue.outline_width, SUBTITLE_OUTLINE_WIDTH, "{name}");
                assert!(cue.override_outline_color, "{name}");
                assert!(cue.sharp_outlines, "{name}");
            }
        }
    }

    /// The subtitle overlays the picture and stands clear of it: its anchor
    /// cube sits at the screen's horizontal centre, at the bottom row, and one
    /// cube above the display bricks' top face -- not level with them, which
    /// would z-fight and the overlap check would reject.
    #[test]
    fn a_brick_screens_subtitle_anchors_bottom_centre_in_front_of_the_screen() {
        let geometry = ScreenGeometry {
            footprint: 2,
            half_height: 2,
            pitch: 4,
        };
        // No lift here -- this pins the bare baseline anchor; the lift's own
        // effect on this same geometry is `subtitle_lift_moves_the_brick_
        // anchor_up_the_picture_along_minus_y`, right below.
        let extent = crate::anim::bricks::subtitle_extent(&geometry, 9, 5, 0.0)
            .expect("a zero lift must always be legal");
        // Columns' centres run 0..=32, so the middle is 16; the last row's
        // outer face is 4*4 + 2 = 18; the screen's top face is 2*2 = 4 and the
        // 1-half-extent cube rests on it at 5.
        assert_eq!(extent.anchor, Position { x: 16, y: 18, z: 5 });
        assert!(
            extent.anchor.z > 2 * geometry.half_height,
            "the cube must clear the screen's top face, not sit inside it"
        );
    }

    /// Brick mode's up-the-picture axis is -y, not +z. A display-brick screen
    /// lies flat: the image's rows run along y, increasing toward the
    /// picture's bottom (see `subtitle_extent`'s doc), so lifting the anchor
    /// toward the picture's top moves it toward smaller y -- the opposite of
    /// text mode's +z. Unverified by eye (see `subtitle_extent`'s doc on the
    /// still-unknown in-plane orientation); only asserted to move along the
    /// axis the geometry says is "up".
    #[test]
    fn subtitle_lift_moves_the_brick_anchor_up_the_picture_along_minus_y() {
        let geometry = ScreenGeometry {
            footprint: 2,
            half_height: 2,
            pitch: 4,
        };
        let base = crate::anim::bricks::subtitle_extent(&geometry, 9, 5, 0.0)
            .expect("a zero lift must always be legal")
            .anchor;
        let lifted = crate::anim::bricks::subtitle_extent(&geometry, 9, 5, 8.0)
            .expect("a lift this small for a 5-row picture must be legal")
            .anchor;
        assert_eq!(lifted.y, base.y - 8, "the lift must move the anchor along -y by exactly itself");
        assert_eq!(lifted.x, base.x, "the lift must not move the anchor sideways");
        assert_eq!(lifted.z, base.z, "the lift must not move the anchor off the screen's surface");
    }

    /// `brdb` encodes negative brick coordinates exactly, by exercising the
    /// encoder itself: `Position::to_relative` splits a coordinate into a
    /// chunk index and an in-chunk offset with `div_euclid`/`rem_euclid`, and
    /// `from_relative` is its exact inverse -- for negatives as much as
    /// positives.
    ///
    /// Swept across the origin, across a chunk boundary, and across the
    /// `-CHUNK_HALF` shift `World::add_brick_grid` applies to every inner-grid
    /// brick -- which is the strongest evidence available without loading a
    /// save, because it means every microchip gate this crate has ever written
    /// is already stored at an absolute coordinate near -1024, in chunk
    /// `(-1, -1, -1)`, and those renders work in game.
    #[test]
    fn brdb_round_trips_negative_brick_coordinates_exactly() {
        let chunk = brdb::CHUNK_SIZE;
        let mut cases: Vec<i32> = Vec::new();
        for base in [0, -chunk, chunk, -brdb::CHUNK_HALF, -2 * chunk] {
            for d in -3..=3 {
                cases.push(base + d);
            }
        }
        for &x in &cases {
            for &y in &cases {
                let p = Position { x, y, z: -6 };
                let (index, relative) = p.to_relative();
                assert_eq!(
                    Position::from_relative(index, relative),
                    p,
                    "{p:?} did not survive the chunk split -- chunk {index:?}, relative \
                     {relative:?}"
                );
            }
        }

        // The exact shift `World::add_brick_grid` applies to every inner-grid
        // brick, on a coordinate the gate lattice really produces.
        let inner = Position { x: 5, y: 5, z: 8 } - Position::CHUNK_HALF;
        assert!(inner.x < 0 && inner.y < 0 && inner.z < 0, "the shift really does go negative");
        let (index, relative) = inner.to_relative();
        assert_eq!(Position::from_relative(index, relative), inner);
        assert_eq!(
            (relative.x, relative.y, relative.z),
            (5, 5, 8),
            "the shift exists precisely so an inner coordinate becomes the in-chunk offset"
        );
    }

    /// A `--subtitle-lift` bigger than a brick-mode picture is allowed, and
    /// produces the negative `y` it implies, because that renderer already
    /// writes a negative main-grid `x` for its chip shell in every render.
    #[test]
    fn a_brick_mode_lift_bigger_than_the_picture_is_applied_not_refused() {
        let geometry = ScreenGeometry {
            footprint: 2,
            half_height: 2,
            pitch: 4,
        };
        // A single row's baseline y is just the footprint, 2; a lift of 3
        // lands at y = -1.
        let extent = crate::anim::bricks::subtitle_extent(&geometry, 9, 1, 3.0)
            .expect("a lift bigger than the picture must be applied, not refused");
        assert_eq!(extent.anchor.y, -1, "the lift must be applied exactly as asked");

        // And a real render of a picture too short for the default lift must
        // build.
        let opts = AnimOptions {
            subtitles: Some(track()),
            ..AnimOptions::default()
        };
        let world = crate::anim::bricks::build_brick_world(
            &clip(4, 3, 2),
            &opts,
            &mut NoProgress,
        )
        .expect("a 3-row subtitled brick render at the default lift must build");
        assert!(
            world.bricks.iter().any(|b| b.position.y < 0),
            "the subtitle anchor really must have landed at a negative y here"
        );
        assert!(
            world.bricks.iter().any(|b| b.position.x < 0),
            "...alongside the chip shell, which has always been at a negative x"
        );
    }

    /// Text mode's picture moves sideways only for a subtitle, by half its own
    /// width, so the centre the subtitle anchors at is a legal coordinate.
    #[test]
    fn the_picture_only_moves_sideways_when_there_are_subtitles() {
        use crate::anim::text_bricks::subtitle_centre_shift;
        let off = AnimOptions::default();
        assert_eq!(subtitle_centre_shift(&off, 192), 0, "no track, no shift");
        let on = AnimOptions {
            subtitles: Some(track()),
            ..AnimOptions::default()
        };
        let half = 192.0 * on.text.line_world_height * on.text.pitch_x / 2.0;
        assert_eq!(subtitle_centre_shift(&on, 192), half.ceil() as i32);
        // Degenerate geometry must not produce a coordinate at all rather than
        // a NaN cast, which in Rust is 0 but by accident rather than by intent.
        let broken = AnimOptions {
            subtitles: Some(track()),
            text: TextOptions {
                line_world_height: f32::NAN,
                ..on.text.clone()
            },
            ..AnimOptions::default()
        };
        assert_eq!(subtitle_centre_shift(&broken, 192), 0);
    }

    /// The subtitle's anchor cube must land inside the picture's horizontal
    /// span (that is what "centred on it" means) and still be non-negative,
    /// across widths whose half-width rounds either way.
    #[test]
    fn a_text_renders_subtitle_lands_inside_the_picture_and_stays_non_negative() {
        for w in [16u32, 32, 33, 64, 65] {
            let opts = AnimOptions {
                subtitles: Some(track()),
                // Buttons off: their labels are TextDisplays placed at the
                // greatest main-grid x, which would be picked as "the frontmost
                // text cube" instead of the subtitle by the isolation below.
                control_buttons: false,
                ..AnimOptions::default()
            };
            let world = build_text_world(&clip(w, 8, 2), &opts, &mut NoProgress)
                .unwrap_or_else(|e| panic!("{w}px: {e}"));
            // Among the bricks carrying text, the subtitle's cube is the
            // frontmost: it rests on the glyph wall's face and every band cube
            // is in that wall. (The chip shell is further out on x still and
            // carries no text, which is why this filters first.)
            let mut text_cubes: Vec<_> = world
                .bricks
                .iter()
                .filter(|b| {
                    b.components
                        .iter()
                        .any(|c| {
                            c.component_type()
                                .is_some_and(|t| t.to_string() == TEXT_DISPLAY)
                        })
                })
                .collect();
            text_cubes.sort_by_key(|b| b.position.x);
            let sub = text_cubes.pop().expect("the subtitle's anchor cube");
            let bands = text_cubes;
            assert!(!bands.is_empty(), "{w}px: the picture must have bands");
            assert!(
                bands.iter().all(|b| b.position.x < sub.position.x),
                "{w}px: the subtitle must sit in front of every glyph band, not in their plane"
            );
            let left = bands.iter().map(|b| b.position.y).max().unwrap();
            let width =
                (w as f32 * opts.text.line_world_height * opts.text.pitch_x).round() as i32;
            assert!(
                sub.position.y <= left && sub.position.y >= left - width,
                "{w}px: the subtitle at y={} is outside the picture's {}..={left}",
                sub.position.y,
                left - width
            );
            for b in &world.bricks {
                assert!(
                    b.position.x >= 0 && b.position.y >= 0 && b.position.z >= 0,
                    "{w}px: negative main-grid coord {:?}",
                    b.position
                );
            }
        }
    }

    /// The subtitle's anchor cube, isolated from the picture's own glyph
    /// bands by the same "frontmost on x" rule
    /// `a_text_renders_subtitle_lands_inside_the_picture_and_stays_non_negative`
    /// uses.
    fn text_subtitle_anchor(opts: &AnimOptions, w: u32, h: u32, frames: usize) -> Position {
        let world = build_text_world(&clip(w, h, frames), opts, &mut NoProgress)
            .expect("a subtitled text render must build");
        let mut text_cubes: Vec<_> = world
            .bricks
            .iter()
            .filter(|b| {
                b.components
                    .iter()
                    .any(|c| c.component_type().is_some_and(|t| t.to_string() == TEXT_DISPLAY))
            })
            .collect();
        text_cubes.sort_by_key(|b| b.position.x);
        text_cubes
            .pop()
            .expect("the subtitle's anchor cube")
            .position
    }

    /// Text mode's screen is the vertical wall `DEFAULT_SUBTITLE_LIFT`'s doc
    /// says was actually checked by eye, and there `z` genuinely is the
    /// image row -- so "up the picture" is `+z` (see
    /// `text_bricks::build_text_world`'s `subtitle_z`). A real end-to-end
    /// render, driven twice with only the lift changed, must move the
    /// anchor by exactly the difference along that one axis.
    #[test]
    fn a_text_mode_lift_moves_the_subtitle_anchor_up_by_exactly_itself_along_plus_z() {
        // Buttons off: `text_subtitle_anchor` isolates the subtitle as the
        // frontmost text cube, which the control-button labels (also
        // TextDisplays, placed further out on x) would otherwise displace.
        let base = text_subtitle_anchor(
            &AnimOptions {
                subtitles: Some(track()),
                subtitle_lift: 0.0,
                control_buttons: false,
                ..AnimOptions::default()
            },
            32,
            8,
            2,
        );
        let lifted = text_subtitle_anchor(
            &AnimOptions {
                subtitles: Some(track()),
                subtitle_lift: 20.0,
                control_buttons: false,
                ..AnimOptions::default()
            },
            32,
            8,
            2,
        );
        assert_eq!(
            lifted.z,
            base.z + 20,
            "the anchor must move along +z by exactly the lift"
        );
        assert_eq!(lifted.x, base.x, "the lift must not move the anchor off the wall's surface");
        assert_eq!(lifted.y, base.y, "the lift must not move the anchor sideways");
    }

    /// The default lift must apply whenever `AnimOptions::subtitle_lift` is
    /// never touched -- i.e. the module's own constant, not a second copy of
    /// the number a caller forgot to update.
    #[test]
    fn the_default_lift_applies_when_the_flag_is_absent() {
        let opts = AnimOptions {
            subtitles: Some(track()),
            // Buttons off so `text_subtitle_anchor`'s frontmost-x isolation
            // finds the subtitle, not a control-button label.
            control_buttons: false,
            ..AnimOptions::default()
        };
        assert_eq!(
            opts.subtitle_lift, DEFAULT_SUBTITLE_LIFT,
            "a bare AnimOptions must carry the module's own default, not a second copy of it"
        );
        // And a real render must actually apply it -- not just default the
        // field and then ignore it -- which is exactly the delta the lift
        // test right above measures between 0.0 and a non-default value.
        let default_anchor = text_subtitle_anchor(&opts, 32, 8, 2);
        let zero_anchor = text_subtitle_anchor(
            &AnimOptions { subtitle_lift: 0.0, ..opts },
            32,
            8,
            2,
        );
        assert_eq!(
            default_anchor.z,
            zero_anchor.z + DEFAULT_SUBTITLE_LIFT.round() as i32,
            "an untouched AnimOptions must render with the default lift actually applied"
        );
    }
}

