//! Text mode: the whole clip rendered as a stack of animated
//! `Component_TextDisplay` bricks, at ~2 gates per BAND of image rows instead
//! of 2 gates per PIXEL.
//!
//! This is the third sibling of [`super::bricks::build_brick_world`] and
//! [`super::color_bricks::build_color_array_world`], and it shares their
//! signature exactly so a mode dispatcher never has to branch on anything but
//! the mode. It shares almost nothing else: there is no screen of display
//! bricks, no `BrickPropertyChanger`, and no per-pixel anything. One
//! `TextDisplay` draws a whole band of rows from a single pre-rendered string,
//! swapped every frame out of that band's own `ArrayVar`.
//!
//! Brick mode spends 2 gates per pixel and the game drops frames near 20 000
//! gates -- about 10 000 pixels. Text mode spends 2 gates per band: a 192x108
//! screen is 54 bands, so 113 gates including the clock. The cost does not
//! vanish so much as MOVE, from the wire graph into the game's text layout,
//! which is why `examples/text_probe.rs` exists and why the human-verification
//! list at the end of the plan is not optional.
//!
//! # Per-band anchors, not `start_row` newlines
//!
//! [`crate::text::encode_bands`] pads a band with leading newlines and
//! [`crate::text::add_text_bricks`] stacks all of a tile's bands behind ONE
//! anchor cube, compensating with a per-band component `Offset.Z` equal to its
//! depth. At 54 bands that needs an `Offset.Z` of up to 106, and
//! [`crate::text::add_text_tiles`]' own doc states the game does not honour
//! large component `Offset` values. So every band gets its OWN anchor cube at
//! its own image row -- one [`TextTile`] per band with `tile_override` set to
//! the band height -- which is how an ordinary tiled `--text` export already
//! anchors. The probe established this in game; it is not a preference.
//!
//! # Exec wiring: fan-out, never fan-in
//!
//! Each bank's exec entry drives every one of that bank's `Get.Exec` inputs
//! directly, the pattern [`super::color_bricks`] established. Exec fan-out (one
//! output driving many exec inputs) is supported and costs the same as a chain;
//! exec FAN-IN (two sources into one exec input) is unverified, so nothing here
//! ever produces it -- the bank branches cascade at the FRONT of the chain
//! rather than branching and rejoining.
//!
//! # Determinism
//!
//! Bands are walked in index order, which is image-row order. Nothing here
//! iterates a `HashMap`: doing so would mint brick ids in a different order on
//! every run and make two renders of the same clip differ for no reason.
use super::bricks::{
    ARRAY_GET, ARRAY_VAR, AnimOptions, BRANCH, CHANGE_DETECTOR, COMPARE_GE, SELECT, SUBTRACT,
};
use super::chip;
use super::clock::{self, gate};
use super::layout::{GATE_HALF, STAGE_PITCH, lattice_pos_staged};
use super::pack;
use super::palette::Palette;
use super::subtitle_display;
use super::text_layout::plan_bands;
use super::text_pack::TextPacker;
use crate::progress::{FrameTotal, Progress};
use crate::text::{TextBand, TextOptions, TextTile, add_text_tiles};
use crate::video::stream::FrameSource;
use brdb::{
    AsBrdbValue, IntVector, Position, Vector3f, WirePort, World,
    schema::{WireArrayVariant, WireVariant},
};
use image::RgbaImage;

/// The text component this mode drives, and the input port carrying its
/// string. Both taken from the generated catalog in the `brdb` crate
/// (`src/assets/component_catalog.rs`), which lists `Component_TextDisplay`
/// with `Text` among its inputs -- not guessed, and the same component
/// [`crate::text`] already builds for a static export.
pub const TEXT_DISPLAY: &str = "Component_TextDisplay";
/// The `Component_TextDisplay` input port a frame's string is written to.
pub const TEXT_PORT: &str = "Text";

/// Frames sampled to build the median-cut palette when `opts.colors > 0`.
///
/// Spread evenly across the whole clip rather than taken from the front: a
/// source with scene cuts must not have its palette decided by its opening
/// shot. 120 is the same cap `examples/text_sizing.rs` samples at.
pub const PALETTE_SAMPLE_FRAMES: usize = 120;

/// Lattice stages one band occupies per bank: its `ArrayVar` and its
/// `ArrayVar_Get`. Mirrors [`super::color_bricks`]'s constant of the same name
/// -- the two modes stage identically, they just index by band instead of by
/// pixel.
const STAGES_PER_BANK: i32 = 2;

/// Where the chip shell sits on the main grid, RELATIVE to the anchor column.
///
/// Beside the anchor column on +X/+Y, never stacked on it: the anchor cubes
/// span x and y in `[-1, 1]` about their own column (they are 1-half-extent
/// micro bricks in a single column) and the shell's own half-extent is 5, so 20
/// is clear with room to spare. Every coordinate stays non-negative, which is
/// what the game requires.
///
/// The `y` is an OFFSET from the anchor column, not an absolute: a subtitled
/// render slides the whole picture sideways ([`subtitle_centre_shift`]) so the
/// subtitle can anchor at the picture's centre, and a shell pinned to an
/// absolute `y = 20` would end up inside the anchor column for any clip whose
/// shift lands near it. Moving with the column keeps the 20 units of clearance
/// this constant is chosen for at every width.
const CHIP_SHELL_POS: Position = Position { x: 20, y: 20, z: 2 };

/// Streams `source` into a wired, animated `TextDisplay` [`World`].
///
/// Signature, streaming contract and cancellation semantics are identical to
/// [`super::bricks::build_brick_world`] -- including that a cancelled render
/// returns an EMPTY `Ok(World)` rather than a partial one, and that it is still
/// on the CALLER to re-check `progress.is_cancelled()` and write nothing when
/// it is true. This mode has one extra place a cancel can land -- the palette
/// pass below, which is a whole additional traversal of the source -- and
/// honours it there too.
///
/// When `opts.colors > 0` the source is traversed TWICE: once to sample frames
/// for the palette (see [`sample_frames`]) and once to encode. `FrameSource` is
/// a cheap re-openable handle precisely so a consumer can do that. At
/// `opts.colors == 0` -- the default -- the sampling pass is skipped entirely
/// and the render is single-pass like the other two modes.
pub fn build_text_world(
    source: &dyn FrameSource,
    opts: &AnimOptions,
    progress: &mut dyn Progress,
) -> Result<World, String> {
    let info = source.info();

    // --- 1. The band layout, decided before a single frame is looked at -----
    //
    // Closed-form from the width alone (see `text_layout`), which is exactly
    // what lets the encode pass stream: the row range each component draws is
    // fixed for the whole clip, so frames can be encoded and dropped one at a
    // time.
    let char_repeat = opts.text.char_repeat.max(1);
    let plan = plan_bands(info.width as usize, info.height as usize, char_repeat)?;
    if plan.is_empty() {
        return Err(format!(
            "source is {}x{} -- a zero-height clip has no rows to band",
            info.width, info.height
        ));
    }

    // --- 2. The palette (optional, and a second traversal) ------------------
    let palette = if opts.colors > 0 {
        // The total is the number of frames that will actually be SAMPLED, not
        // the cap: a bar that stops at 30/120 on a 30-frame clip reads as a
        // stall. When the source cannot give an exact length, its estimate is
        // capped the same way and the bar says it is an estimate -- see
        // `FrameTotal`.
        FrameTotal::new(
            info.frame_count_hint.map(|n| n.min(PALETTE_SAMPLE_FRAMES)),
            source.frame_count_estimate().map(|n| n.min(PALETTE_SAMPLE_FRAMES)),
        )
        .begin(progress, "sampling colours");
        let sampled = sample_frames(source, PALETTE_SAMPLE_FRAMES, progress);
        progress.finish();
        // The encoder's OWN alpha threshold, not `opts.alpha_threshold`:
        // `crate::text::encode_row` skips pixels below `opts.text.alpha_threshold`
        // entirely, so those pixels are never drawn and must not be allowed to
        // spend palette entries on colours nothing displays.
        Palette::build(&sampled?, opts.colors, opts.text.alpha_threshold)
    } else {
        // The no-quantization path. An empty palette maps every colour through
        // unchanged, so there is no branch on it anywhere downstream.
        Palette::default()
    };

    // CANCELLED during the palette pass: return before the encode pass reopens
    // the source. Sampling is a whole extra traversal (`sample_frames` may
    // decode most of the clip to reach its last sample), so this is a real
    // place for a cancel to land and the encode pass behind it is the entire
    // rest of the render.
    if progress.is_cancelled() {
        return Ok(World::new());
    }

    // --- 3. Stream and encode ----------------------------------------------
    //
    // Structurally identical to `build_brick_world`'s pull loop -- see its
    // comments for why the closure exists (so `progress.finish()` runs on every
    // exit, not just the success path) and why cancellation breaks rather than
    // errors. Only the accumulator differs.
    // Same fallback-to-an-estimate as the other two renderers -- see
    // `build_brick_world`'s call and `FrameTotal`'s doc.
    FrameTotal::new(info.frame_count_hint, source.frame_count_estimate())
        .begin(progress, "packing frames");
    let mut packer = TextPacker::new(
        info.width,
        info.height,
        plan.clone(),
        opts.text.clone(),
        palette,
    );
    let seen: Result<u64, String> = (|| {
        let mut stream = source.open()?;
        let mut seen: u64 = 0;
        while let Some(frame) = stream.next()? {
            packer.push_frame(&frame)?;
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

    // CANCELLED: return before anything is built, and ahead of the zero-frame
    // guard so a cancel is never reported as that error. Same reasoning, in
    // full, on `build_brick_world`.
    if progress.is_cancelled() {
        return Ok(World::new());
    }

    let band_texts = packer.finish();
    let frame_count = seen as usize;

    // Same guard, same reason as both brick paths: with no frames,
    // `clock::build_clock` would inline `Modulo.InputB = 0.0` and the save
    // would divide by zero in-game on every tick. There would also be no
    // frame-0 string to bake into each component.
    if frame_count == 0 {
        return Err(
            "clip has 0 frames -- nothing to render (check --start/--duration, or the GUI's \
             Start/Duration, against the source's length)"
                .to_string(),
        );
    }

    let mut world = World::new();
    world.meta.bundle.description = "Animation generated from image frames".to_string();

    // --- 4. Text bricks on the main grid ------------------------------------
    //
    // One TILE per band, each carrying a single band, so `add_text_tiles`
    // anchors every component at its own image row -- no depth stack and no
    // large `Offset` values (see the module doc). The bands are a uniform
    // `rows` tall (only the last may be short), so their `start_row`s are exact
    // multiples of that, which is what makes the tile grid's
    // `start_row / tile_px` indexing land one band per row slot.
    let rows_per_band = plan[0].rows;
    debug_assert!(
        plan.iter().all(|b| b.start_row % rows_per_band == 0),
        "band starts must be multiples of the band height for the tile grid to index them"
    );
    // `tile_override` is the documented lever for exactly this: it tells
    // `add_text_tiles` how many image rows one anchor covers, which sets the
    // world pitch between consecutive anchors. Only the row (z) axis matters
    // here -- every band spans the full width, so there is a single column and
    // the horizontal pitch never comes into play.
    let place_opts = TextOptions {
        tile_override: Some(rows_per_band as u32),
        ..opts.text.clone()
    };

    // Frame 0's strings are baked in as each component's authored Text, so the
    // save shows the first frame before the clock has ticked once.
    let tiles: Vec<TextTile> = plan
        .iter()
        .zip(&band_texts)
        .map(|(p, texts)| TextTile {
            start_col: 0,
            start_row: p.start_row,
            bands: vec![TextBand {
                start_row: 0,
                rows: p.rows,
                text: texts[0].clone(),
                chars: texts[0].chars().count(),
            }],
        })
        .collect();
    let text_ids = add_text_tiles(&mut world, tiles, &place_opts);
    if text_ids.len() != plan.len() {
        return Err(format!(
            "add_text_tiles returned {} anchors for {} bands -- every band must get exactly one \
             TextDisplay, or the wiring below would drive the wrong rows",
            text_ids.len(),
            plan.len()
        ));
    }

    // Each band renders DOWNWARD from its own anchor, so the bottom band hangs
    // one band-height below the lowest cube (z=1) and would sit underground.
    // Positions are plain data until the world is encoded (`World::add_brick`
    // only pushes), so shifting them here is safe and carries each band's text
    // with its own cube.
    //
    // The subtitle needs no strip of its own below the picture -- it is
    // anchored at its own BOTTOM edge and grows upward over the picture (see
    // `subtitle_display::SUBTITLE_ANCHOR`) -- but it does need the picture's
    // horizontal CENTRE to be a legal brick coordinate, which is a sideways
    // shift rather than a lift. Both are 0 without subtitles, which is what
    // keeps a subtitle-free render's brick positions exactly what they are
    // today.
    let band_lift =
        (rows_per_band as f32 * opts.text.line_world_height * opts.text.pitch_y).ceil() as i32;
    let centre_shift = subtitle_centre_shift(opts, info.width as usize);
    for brick in world.bricks.iter_mut() {
        brick.position.z += band_lift;
        brick.position.y += centre_shift;
    }
    // The picture's own surface and bottom edge, read off the bricks that were
    // actually placed rather than re-derived from the layout arithmetic a
    // second time.
    //
    // World +X faces the viewer, so the frontmost brick is the picture's front:
    // normally the front plane of anchor cubes, and for a graffiti render the
    // invisible collision canvas `add_text_tiles` puts one cube ahead of them,
    // which is the surface the glyphs actually appear on. The BOTTOM, though,
    // is a property of the cubes alone -- each band draws `band_lift` downward
    // from its own cube, and the canvas is a slab whose centre says nothing
    // about where the picture ends -- so that one looks only at the cubes,
    // which are the collisionless bricks (`crate::text::anchor_cube` clears
    // every collision flag; the canvas deliberately keeps its own).
    let picture_front_x = world.bricks.iter().map(|b| b.position.x).max().unwrap_or(0);
    let is_cube = |b: &&brdb::Brick| !b.collision.player;
    let picture_bottom_z = world
        .bricks
        .iter()
        .filter(is_cube)
        .map(|b| b.position.z)
        .min()
        .unwrap_or(band_lift)
        - band_lift;
    // The picture's LEFT edge: every band anchors there and its glyphs run
    // from it toward world -Y (`add_text_tiles`), and the leftmost anchor is
    // the one at the greatest y.
    let picture_left_y = world
        .bricks
        .iter()
        .filter(is_cube)
        .map(|b| b.position.y)
        .max()
        .unwrap_or(centre_shift);

    // --- 5. The chip --------------------------------------------------------
    let n_bands = plan.len() as i32;
    // Beside the anchor column wherever the column ended up -- see
    // `CHIP_SHELL_POS` on why its y travels with `centre_shift`.
    let shell_pos = Position {
        y: CHIP_SHELL_POS.y + centre_shift,
        ..CHIP_SHELL_POS
    };
    let mut chip = chip::new_chip(
        &mut world,
        shell_pos,
        Vector3f {
            x: shell_pos.x as f32,
            y: shell_pos.y as f32,
            z: (band_lift + 20) as f32,
        },
        IntVector { x: 5, y: 5, z: 5 },
    );

    let bank_size = opts.bank_size.max(1);
    let n_banks = frame_count.div_ceil(bank_size).max(1);

    // Service gates sit BEHIND every band stage, so the service stage depends
    // on how many stages a band uses -- the same arithmetic
    // `color_bricks` uses, with bands where it has pixels. A band occupies
    // `STAGES_PER_BANK` stages per bank (its ArrayVar and its Get) plus one
    // stage per boundary for its own Select; at a single bank that is stages
    // 0..=1 and a service stage of 2, exactly the probe's layout.
    //
    // `lattice_pos_staged`'s `height` is the BAND count here, and service rows
    // are always negative, so `x = (n_bands - 1 - row) * CELL + half.x` stays
    // positive for both -- no inner-grid coordinate can go negative.
    let boundaries = (n_banks - 1) as i32;
    let service_stage = STAGES_PER_BANK * n_banks as i32 + boundaries;
    let service = |col: i32, row: i32| {
        lattice_pos_staged(col, row, service_stage, n_bands, GATE_HALF, STAGE_PITCH)
    };

    // --- 6. Frame index source ---------------------------------------------
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

    // --- 7. Exec source -----------------------------------------------------
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

    // --- 8. Per-bank index and boundary comparators -------------------------
    // Byte-for-byte the same construction as both brick paths': bank 0 reads
    // the frame index directly, bank k subtracts `k * bank_size` so its own
    // array is addressed from zero, and `ge[k-1]` is true once the frame index
    // reaches bank k -- which is `Select`'s `bSelectB` sense (true picks
    // InputB, the later bank).
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

    // --- 9. Exec entry per bank ---------------------------------------------
    // Branches cascade at the FRONT so exactly one bank's gets run and no exec
    // input ever takes two sources (ExecOutA = keep descending, ExecOutB = this
    // bank). With `n_banks == 1` this emits no branch at all and
    // `entry_of_bank[0]` is simply the detector's `OnChanged`, so a clip that
    // never spills is wired exactly as the probe wired it.
    let mut entry_of_bank = Vec::with_capacity(n_banks);
    let mut exec_src = WirePort::new(detector, CHANGE_DETECTOR, "OnChanged");
    for bi in 0..n_banks {
        if bi + 1 < n_banks {
            let br = gate(&mut chip, "B_1x1_Gate_Exec_Branch", BRANCH,
                service(bi as i32, -8), vec![]);
            world.add_wire_connection(ge[bi].clone(), WirePort::new(br, BRANCH, "bCond"));
            world.add_wire_connection(exec_src, WirePort::new(br, BRANCH, "Exec"));
            // true -> keep descending; false -> this bank
            exec_src = WirePort::new(br, BRANCH, "ExecOutA");
            entry_of_bank.push(WirePort::new(br, BRANCH, "ExecOutB"));
        } else {
            entry_of_bank.push(exec_src.clone());
        }
    }

    // --- 10. Two gates per band, per bank -----------------------------------
    //
    // Walked in band order, which is image-row order -- never by iterating a
    // map, so two runs of the same clip mint the same brick ids.
    //
    // A bank boundary costs one extra ArrayVar + Get + Select per BAND, which
    // at 54 bands is cheap in absolute terms (three orders of magnitude under
    // colour-array mode's per-PIXEL seam) -- and it only bites past
    // `BANK_FRAMES` (65 535) frames, about 90 minutes at 12 fps.
    for (bi, texts) in band_texts.into_iter().enumerate() {
        // Polled per band, the same way both brick modes poll per pixel. There
        // are only tens of bands rather than thousands of pixels, but a band
        // carries one string PER FRAME and copies the lot into its `ArrayVar`,
        // so an iteration here is far from free on a long clip.
        if progress.is_cancelled() {
            return Ok(World::new());
        }
        debug_assert_eq!(
            texts.len(),
            frame_count,
            "every band must hold one string per frame"
        );
        let row = bi as i32;
        let banks = pack::bank_frames(&texts, bank_size);
        debug_assert_eq!(
            banks.len(),
            n_banks,
            "every band's string list must bank identically -- they are all frame_count long"
        );

        // One (ArrayVar, Get) pair per bank, each pair on its own two stages.
        let mut get_of_bank = Vec::with_capacity(n_banks);
        for (k, frames) in banks.iter().enumerate() {
            let array_stage = STAGES_PER_BANK * k as i32;
            let array = gate(
                &mut chip,
                "B_1x1_Gate_Variable_Array",
                ARRAY_VAR,
                lattice_pos_staged(0, row, array_stage, n_bands, GATE_HALF, STAGE_PITCH),
                vec![(
                    "Value",
                    Box::new(WireArrayVariant::StringArray(frames.to_vec()))
                        as Box<dyn AsBrdbValue>,
                )],
            );
            let get = gate(
                &mut chip,
                "B_1x1_Gate_Exec_ArrayVar_Get",
                ARRAY_GET,
                lattice_pos_staged(0, row, array_stage + 1, n_bands, GATE_HALF, STAGE_PITCH),
                vec![],
            );
            world.add_wire_connection(
                WirePort::new(array, ARRAY_VAR, "ArrayVarRef"),
                WirePort::new(get, ARRAY_GET, "ArrayVarRef"),
            );
            world.add_wire_connection(
                index_of_bank[k].clone(),
                WirePort::new(get, ARRAY_GET, "Index"),
            );
            // THE FAN-OUT. One exec source drives every band's Get for this
            // bank; no chaining, and no exec input ever gains a second source.
            world.add_wire_connection(
                entry_of_bank[k].clone(),
                WirePort::new(get, ARRAY_GET, "Exec"),
            );
            get_of_bank.push(get);
        }

        // Value: one select per boundary, cascading. For a frame in bank j,
        // ge[0..j] are true so select j picks bank j and every later select
        // passes it through unchanged.
        let mut value = WirePort::new(get_of_bank[0], ARRAY_GET, "Value");
        for k in 1..n_banks {
            let sel = gate(
                &mut chip,
                "B_1x1_Gate_Expr_Select",
                SELECT,
                lattice_pos_staged(
                    0,
                    row,
                    STAGES_PER_BANK * n_banks as i32 + k as i32 - 1,
                    n_bands,
                    GATE_HALF,
                    STAGE_PITCH,
                ),
                vec![],
            );
            world.add_wire_connection(ge[k - 1].clone(), WirePort::new(sel, SELECT, "bSelectB"));
            world.add_wire_connection(value, WirePort::new(sel, SELECT, "InputA"));
            world.add_wire_connection(
                WirePort::new(get_of_bank[k], ARRAY_GET, "Value"),
                WirePort::new(sel, SELECT, "InputB"),
            );
            value = WirePort::new(sel, SELECT, "Output");
        }

        // Chip -> main grid. The two endpoints live in different grids and
        // `World::add_wire_connection` emits the `RemoteWirePortSource` itself;
        // nothing extra is needed for a TextDisplay -- it is an ordinary
        // component with an ordinary input port.
        world.add_wire_connection(
            value,
            WirePort::new(text_ids[bi], TEXT_DISPLAY, TEXT_PORT),
        );
    }

    // --- 10b. Subtitles, if any ---------------------------------------------
    //
    // LAST, after every band gate exists: `add_subtitle_display` places its
    // own two gates a clear cell beyond the chip's current x extent, which is
    // only collision-free against bricks that are already there.
    //
    // Gated on `opts.subtitles` so a render without a track is untouched.
    // Anchored at the picture's bottom CENTRE, one cube in front of the glyph
    // wall, so the cue lies across the bottom of the picture instead of
    // hanging under it -- see the `ScreenExtent` built below.
    if let Some(subs) = &opts.subtitles {
        // `opts.source_start_s`, NOT 0.0: a subtitle file is in SOURCE time,
        // and frame 0 of what this renderer receives is at source time
        // `--start` (see `AnimOptions::source_start_s`). Timing the cues from
        // zero puts the whole track `--start` seconds early.
        let per_frame = subs.per_frame(opts.source_start_s, info.fps as f64, frame_count)?;
        // `--subtitle-lift`, along `+z`. Text mode's screen is an upright wall
        // where `z` genuinely IS the image row, so moving the anchor toward
        // the picture's TOP means INCREASING z -- the opposite of both brick
        // encodings' flat, ground-facing screen (`-y`; see
        // `bricks::subtitle_extent`). **This is the mode
        // `DEFAULT_SUBTITLE_LIFT` was measured by eye against** -- see its
        // doc for the exact configuration (192x108, `--subtitle-scale 6`).
        //
        // Rejected, not clamped, if it would push the anchor below z=0.
        //
        // The reason is TEXT MODE'S OWN, not a claim about `brdb`. This used to
        // say "main-grid brick coordinates cannot be negative (`brdb`'s chunk
        // encoding mishandles them)", which is false -- `Position::to_relative`
        // is exact for negatives and `World::add_brick_grid` already stores
        // every microchip gate around -1024; see
        // `bricks::subtitle_extent`, where the same false claim was rejecting
        // legal BRICK-mode renders and has been dropped.
        //
        // What is true here is narrower: this mode keeps its entire main grid
        // non-negative by construction, because `crate::text::add_text_tiles`
        // translates the whole glyph grid to make it so
        // (`a_subtitled_text_render_keeps_every_main_grid_brick_non_negative`
        // pins it). The subtitle's anchor cube is the one brick placed outside
        // that translation, so a lift below the picture's own bottom edge would
        // be the single thing to break the invariant -- which is worth an error
        // rather than a silent exception to it.
        let lift = subtitle_display::lift_units(opts.subtitle_lift);
        let subtitle_z = picture_bottom_z + lift;
        if subtitle_z < 0 {
            return Err(format!(
                "--subtitle-lift {} pushes the subtitle anchor to z={subtitle_z}, below the \
                 picture's own bottom edge (z={picture_bottom_z}). Text mode keeps every \
                 main-grid brick non-negative (see `crate::text::add_text_tiles`, which \
                 translates the whole glyph grid for it), so this is refused rather than \
                 made the one exception. Raise --subtitle-lift",
                opts.subtitle_lift
            ));
        }
        subtitle_display::add_subtitle_display(
            &mut world,
            &mut chip,
            subtitle_display::FrameIndex {
                index_of_bank: &index_of_bank,
                entry_of_bank: &entry_of_bank,
                ge: &ge,
            },
            &per_frame,
            info.fps,
            opts,
            subtitle_display::ScreenExtent {
                anchor: Position {
                    // ONE CUBE IN FRONT of the picture's own surface: the
                    // subtitle now overlays the picture, and a block drawn in
                    // the same plane as the glyph wall would z-fight with it.
                    // Resting the cube flush on that surface puts the
                    // subtitle's glyph plane a full cube (2 units) proud of it.
                    x: picture_front_x + 2 * crate::text::ANCHOR_CUBE_HALF,
                    // The picture's horizontal centre. The band cubes anchor at
                    // the picture's LEFT edge and the glyphs run toward world
                    // -Y from there (see `add_text_tiles`), so the centre is
                    // half a picture-width in that direction -- which is why
                    // `centre_shift` slid the whole stack up-y first: without
                    // it this lands at a negative coordinate.
                    y: picture_left_y
                        - (picture_width_world(info.width as usize, &opts.text) / 2.0).round()
                            as i32,
                    // The picture's bottom edge, so the cue's own bottom
                    // (`SUBTITLE_ANCHOR`'s Y of 1) sits there and the line
                    // grows upward INTO the picture -- PLUS `--subtitle-lift`,
                    // which lifts that whole baseline further up-picture still
                    // (see the comment above `lift`).
                    z: subtitle_z,
                },
                row_height: opts.text.line_world_height * opts.text.pitch_y,
                // The screen here IS a vertical wall facing world +X, in the
                // very plane a TextDisplay draws in, so the subtitle uses the
                // same upright face every glyph band does. (Both brick
                // encodings lay their screen flat and need `FACE_Z_POSITIVE`
                // instead -- see `bricks::subtitle_extent`.)
                face: crate::text::FACE_X_POSITIVE,
            },
        )?;
    }

    // --- 11. Publish --------------------------------------------------------
    //
    // One last poll before the publish phase, for the same reason
    // `bricks::build_brick_world` has one: the per-band loop polls at the top
    // of each iteration and the subtitle step does not poll at all, so without
    // this a late cancel would still pay for `chip::finish`'s two grid
    // collision checks.
    if progress.is_cancelled() {
        return Ok(World::new());
    }
    // Asserts non-overlap on BOTH grids before publishing.
    chip::finish(&mut world, chip)?;
    // Must be last, and must come AFTER `chip::finish`: it registers every
    // component type and port name actually used, and has to see all bricks,
    // grids and wires first. `add_text_tiles` already called it once, before
    // the chip existed -- this re-registration is the one that counts.
    world.register_used_components();
    Ok(world)
}

/// World width of a rendered text-mode picture.
///
/// [`add_text_tiles`] advances one TILE -- `tile_px` image pixels -- by
/// `tile_px * line_world_height * pitch_x`, so one pixel is
/// `line_world_height * pitch_x` wide and `width_px` of them are this. It is
/// the same two fields the placement itself uses, on purpose: an independently
/// derived width could disagree with where the glyphs actually went, and the
/// subtitle would then be centred on a picture edge that does not exist.
///
/// `char_repeat` deliberately does not appear. The crate's placement model does
/// not use it either: `pitch_x` is CALIBRATED for the chosen repeat (two block
/// characters per pixel is what makes a Monaspace/Iosevka pixel square), which
/// is exactly what [`TextOptions::pitch_x`] documents.
fn picture_width_world(width_px: usize, opts: &TextOptions) -> f32 {
    width_px as f32 * opts.line_world_height * opts.pitch_x
}

/// World units a SUBTITLED text render slides its whole picture along +Y, and
/// `0` when there are no subtitles.
///
/// Text mode's band cubes anchor at the picture's LEFT edge and their glyphs
/// run from there toward world -Y, so an unshifted picture occupies
/// `-width ..= 0` and its horizontal CENTRE -- where the subtitle's anchor cube
/// has to go, since the cue is centred on its anchor -- is at a negative
/// coordinate. Negative main-grid brick coordinates are not an option HERE --
/// not because `brdb` cannot encode them (it can; see
/// `crate::anim::bricks::subtitle_extent`, where that claim was false and has
/// been dropped) but because this mode keeps its whole main grid non-negative
/// by construction, via [`crate::text::add_text_tiles`]'s own translation. So
/// the picture moves instead of the anchor, which keeps the rule whole rather
/// than carving one brick out of it.
///
/// Half a picture-width, rounded UP, is the least that does it: the centre then
/// lands at `ceil(w/2) - round(w/2)`, which is 0 or 1 and never negative. The
/// glyphs still spill to negative y on the picture's right-hand half, which is
/// harmless -- they are drawn by a component, not placed as bricks, and that is
/// already true of every text render today.
///
/// This replaced the old `subtitle_headroom`, which lifted the picture UP to
/// make room for a cue hanging below it. Nothing hangs below it any more (see
/// [`subtitle_display::SUBTITLE_ANCHOR`]), so that lift is gone; this is a
/// different axis and a different reason, not a rename of it.
pub fn subtitle_centre_shift(opts: &AnimOptions, width_px: usize) -> i32 {
    if opts.subtitles.is_none() {
        return 0;
    }
    let half = picture_width_world(width_px, &opts.text) / 2.0;
    // A NaN or negative width would `ceil` into something unusable as a
    // coordinate, so the floor is applied to the RESULT rather than trusting
    // the arithmetic -- the same guard the old `subtitle_headroom` applied.
    if half.is_finite() && half > 0.0 {
        half.ceil() as i32
    } else {
        0
    }
}

/// Up to `n` frames spread evenly across `source`, for palette construction.
///
/// Uses the same `i * len / n` indexing `examples/text_sizing.rs` samples with,
/// so the palette is decided from the whole clip rather than its opening shot.
/// The frames are pulled through [`crate::video::stream::FrameStream::advance`]
/// rather than `next`, which lets a stream skip the work of producing frames
/// nobody will look at (`FpsStream`/`ResizeStream` both override it) -- so a
/// long clip is not fully decoded just to sample 120 frames from it.
///
/// A source that cannot report its own length has no index to spread against,
/// so that case falls back to a decimating traversal: retain every frame until
/// `n` are held, then halve the retained set and double the stride. That still
/// yields `n` evenly-spaced frames with bounded memory, at the cost of reading
/// the whole stream -- the only way to sample evenly from an unknown length.
///
/// Both branches stop early on [`Progress::is_cancelled`], for the same reason
/// the encode loop does: this pass can decode most of a clip, and a cancel
/// pressed under the "sampling colours" bar has to be honoured while that bar
/// is still up rather than one whole traversal later. A cancelled sample
/// returns the frames it did gather -- `build_text_world` throws them away at
/// its own check immediately afterwards, so they are never built from.
fn sample_frames(
    source: &dyn FrameSource,
    n: usize,
    progress: &mut dyn Progress,
) -> Result<Vec<RgbaImage>, String> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut stream = source.open()?;
    match source.info().frame_count_hint {
        Some(len) if len > 0 => {
            let n = n.min(len);
            let mut out = Vec::with_capacity(n);
            // How many frames have been pulled off the stream so far. Targets
            // are strictly increasing (`len >= n` makes `len / n >= 1`), so
            // every `skip` below is at least 1 and the cursor only advances.
            let mut cursor = 0usize;
            for i in 0..n {
                let target = i * len / n;
                let skip = target + 1 - cursor;
                let (got, frame) = stream.advance(skip)?;
                cursor += got;
                match frame {
                    Some(f) => out.push(f),
                    // The source drained early -- its real length was shorter
                    // than the hint. Sample what there was.
                    None => break,
                }
                progress.tick(out.len() as u64);
                if got < skip || progress.is_cancelled() {
                    break;
                }
            }
            Ok(out)
        }
        _ => {
            let mut out: Vec<RgbaImage> = Vec::new();
            let mut stride = 1usize;
            let mut index = 0usize;
            while let Some(f) = stream.next()? {
                if index.is_multiple_of(stride) {
                    out.push(f);
                }
                index += 1;
                // Frames SCANNED, not retained: the retained count halves
                // periodically, and a tick that went backwards would read as a
                // bug. There is no total to compare it against here anyway --
                // this branch runs precisely when the length is unknown.
                progress.tick(index as u64);
                if out.len() > n {
                    // Halve: keep every other retained frame and double the
                    // stride. The kept frames were at multiples of the old
                    // stride, so every second one is a multiple of the new --
                    // the retained set stays exactly "every `stride`th frame".
                    let mut i = 0;
                    out.retain(|_| {
                        let keep = i % 2 == 0;
                        i += 1;
                        keep
                    });
                    stride *= 2;
                }
                if progress.is_cancelled() {
                    break;
                }
            }
            Ok(out)
        }
    }
}

#[cfg(test)]
#[path = "../../tests/wire_integrity.rs"]
mod wire_integrity;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::bricks::{ARRAY_GET, ARRAY_VAR, AnimOptions, SELECT};
    use crate::anim::text_layout::plan_bands;
    use crate::progress::NoProgress;
    use crate::video::Clip;
    use brdb::{IntoReader, Position, schema::WireArrayVariant};
    use image::{Rgba, RgbaImage};

    /// An in-memory [`crate::video::stream::FrameSource`] whose every
    /// (pixel, frame) triple is a distinct colour, so a transposition or an
    /// off-by-one between the source and a saved band string shows up rather
    /// than passing on symmetry. Modelled on `tests/anim_color.rs`'s
    /// `distinct_clip`.
    fn clip(w: u32, h: u32, frames: usize) -> Clip {
        let frames = (0..frames as u32)
            .map(|f| {
                RgbaImage::from_fn(w, h, |x, y| {
                    Rgba([
                        (x * 17 + f) as u8,
                        (y * 53 + f * 7) as u8,
                        (x * 31 + y * 11 + f * 3) as u8,
                        255,
                    ])
                })
            })
            .collect();
        Clip { width: w, height: h, fps: 10.0, frames }
    }

    /// A smooth horizontal gradient: 64 distinct colours across every row, so
    /// unquantized every pixel opens its own `<color>` tag and a small palette
    /// has something real to collapse.
    fn gradient_clip() -> Clip {
        let (w, h) = (64u32, 8u32);
        let frames = (0..3u32)
            .map(|f| {
                RgbaImage::from_fn(w, h, |x, y| {
                    Rgba([
                        (x * 4) as u8,
                        (y * 8 + f * 3) as u8,
                        ((x + y * 2 + f) * 2) as u8,
                        255,
                    ])
                })
            })
            .collect();
        Clip { width: w, height: h, fps: 10.0, frames }
    }

    /// Inner-grid bricks carrying `class`. Same shape as
    /// `tests/anim_color.rs`'s `count_component`.
    fn count_components(world: &brdb::World, class: &str) -> usize {
        world.grids[0]
            .1
            .iter()
            .filter(|b| {
                b.components
                    .iter()
                    .any(|c| c.component_type().is_some_and(|t| t.to_string() == class))
            })
            .count()
    }

    /// Every inner-grid brick centre, in the lattice's own coordinate space.
    /// `World::add_brick_grid` stores inner bricks shifted by `-CHUNK_HALF`;
    /// this undoes that, exactly as `tests/anim_color.rs` does.
    fn inner_grid_positions(world: &brdb::World) -> Vec<Position> {
        world.grids[0]
            .1
            .iter()
            .map(|b| b.position + Position::CHUNK_HALF)
            .collect()
    }

    /// A `Progress` that flags cancelled once `tick` has reached
    /// `cancel_after`, and records the name of every phase it was told to
    /// begin.
    ///
    /// The phase list is what makes the palette test below meaningful: text
    /// mode has two traversals, and only the label distinguishes "cancelled
    /// while sampling colours and stopped there" from "cancelled while
    /// sampling and then packed frames anyway until the cancel was noticed a
    /// second time". Both leave the same tick count behind.
    #[derive(Default)]
    struct CancelAfter {
        cancel_after: u64,
        ticks: u64,
        phases: Vec<String>,
    }

    impl crate::progress::Progress for CancelAfter {
        fn begin(&mut self, label: &str, _total: Option<u64>) {
            self.phases.push(label.to_string());
        }
        fn tick(&mut self, n: u64) {
            self.ticks = n;
        }
        fn finish(&mut self) {}
        fn is_cancelled(&self) -> bool {
            self.ticks >= self.cancel_after
        }
    }

    /// The same regression both brick modes carry (see
    /// `bricks::tests::a_cancel_while_packing_builds_no_graph_at_all`):
    /// stopping the decode loop used to leave every later phase to run anyway.
    #[test]
    fn a_cancel_while_packing_builds_no_graph_at_all() {
        const TOTAL_FRAMES: usize = 120;
        let c = clip(32, 16, TOTAL_FRAMES);

        let mut cancelled = CancelAfter { cancel_after: 4, ..Default::default() };
        let stopped = build_text_world(&c, &AnimOptions::default(), &mut cancelled)
            .expect("a cancelled render must be Ok, not an error");

        let mut ran =
            CancelAfter { cancel_after: TOTAL_FRAMES as u64 + 1, ..Default::default() };
        let full = build_text_world(&c, &AnimOptions::default(), &mut ran).expect("build");
        assert!(
            !full.bricks.is_empty() && !full.grids.is_empty(),
            "the uncancelled control must actually build a graph, or this test proves nothing"
        );

        assert!(stopped.bricks.is_empty(), "a cancelled render must place no text bricks");
        assert!(stopped.grids.is_empty(), "a cancelled render must build no chip grid");
        assert!(stopped.wires.is_empty(), "a cancelled render must wire nothing");
    }

    /// Text mode's extra cancellation point: the palette pass, a whole second
    /// traversal of the source that runs BEFORE the encode loop and so is
    /// reached by no other cancellation test in the crate.
    ///
    /// `cancel_after` is set inside the sample budget, so the flag is already
    /// up when sampling ends -- the encode pass and the entire build behind it
    /// must then be skipped rather than run.
    ///
    /// Asserted on the PHASE list rather than on the returned world: the
    /// post-decode check would produce an empty world here too, having decoded
    /// the clip a second time first. Only the absence of a "packing frames"
    /// phase shows the encode pass was never entered.
    #[test]
    fn a_cancel_while_sampling_the_palette_skips_the_encode_pass_entirely() {
        let c = clip(32, 16, 60);
        let opts = AnimOptions { colors: 16, ..AnimOptions::default() };

        let mut cancelled = CancelAfter { cancel_after: 3, ..Default::default() };
        let stopped = build_text_world(&c, &opts, &mut cancelled)
            .expect("a cancelled render must be Ok, not an error");

        assert!(cancelled.is_cancelled(), "the reporter must really have flagged cancellation");
        assert_eq!(
            cancelled.phases,
            ["sampling colours"],
            "the render must stop after the palette pass -- a 'packing frames' phase here \
             means the encode traversal ran anyway"
        );
        assert!(stopped.bricks.is_empty(), "a cancelled render must place no text bricks");
        assert!(stopped.grids.is_empty(), "a cancelled render must build no chip grid");

        // The control, same options: quantized text mode really does run both
        // phases and build something when it is allowed to finish.
        let mut ran = CancelAfter { cancel_after: u64::MAX, ..Default::default() };
        let full = build_text_world(&c, &opts, &mut ran).expect("build");
        assert_eq!(ran.phases, ["sampling colours", "packing frames"]);
        assert!(!full.bricks.is_empty() && !full.grids.is_empty());
    }

    /// Main-grid positions of the anchor cubes carrying a TextDisplay.
    fn text_anchor_positions(world: &brdb::World) -> Vec<Position> {
        world
            .bricks
            .iter()
            .filter(|b| {
                b.components
                    .iter()
                    .any(|c| c.component_type().is_some_and(|t| t.to_string() == TEXT_DISPLAY))
            })
            .map(|b| b.position)
            .collect()
    }

    /// Write a world out and reopen it, returning the reader plus the chip's
    /// persistent grid id (assigned at WRITE time, so it has to be discovered
    /// by reading entities back). Lifted from `tests/anim_color.rs`.
    fn write_and_open(
        world: &brdb::World,
        tag: &str,
    ) -> (
        std::path::PathBuf,
        brdb::BrReader<impl brdb::BrFsReader>,
        usize,
    ) {
        let path = std::env::temp_dir().join(format!(
            "h2b_text_{tag}_{}_{:?}.brz",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
        let db = brdb::Brz::open(&path).expect("reopen").into_reader();
        let mut chip_grid_id = None;
        for index in db.entity_chunk_index().expect("entity chunk index") {
            for e in db.entity_chunk(index).expect("entity chunk") {
                if e.is_microchip_grid() {
                    chip_grid_id = e.id;
                }
            }
        }
        let id = chip_grid_id.expect("the renderer must publish exactly one microchip grid");
        (path, db, id)
    }

    /// Every `<color="` tag across every band string actually persisted in the
    /// save. Component DATA is only reachable through a written file --
    /// `BrdbComponent` exposes the component type but not its values -- so this
    /// round-trips, the same way `tests/anim_color.rs` reads its colour arrays.
    fn total_color_tags(world: &brdb::World) -> usize {
        let (path, db, gid) = write_and_open(world, "tags");
        let mut tags = 0usize;
        for chunk in &db.brick_chunk_index(gid).expect("chunk index") {
            let (_soa, structs) = db.component_chunk(gid, chunk.index).expect("components");
            for s in &structs {
                if s.get_name() == "BrickComponentData_WireGraphPseudo_ArrayVar"
                    && let Some(value) = s.get("Value")
                {
                    let variant: WireArrayVariant =
                        value.try_into().expect("ArrayVar Value must decode");
                    if let WireArrayVariant::StringArray(v) = variant {
                        tags += v.iter().map(|s| s.matches("<color=\"").count()).sum::<usize>();
                    }
                }
            }
        }
        let _ = std::fs::remove_file(&path);
        tags
    }

    #[test]
    fn every_band_gets_one_array_and_one_get() {
        let opts = AnimOptions { colors: 0, ..AnimOptions::default() };
        let w = build_text_world(&clip(8, 8, 4), &opts, &mut NoProgress).unwrap();
        let bands = plan_bands(8, 8, 2).unwrap().len();
        assert_eq!(count_components(&w, ARRAY_VAR), bands);
        assert_eq!(count_components(&w, ARRAY_GET), bands);
    }

    #[test]
    fn no_inner_grid_coordinate_is_negative() {
        let opts = AnimOptions::default();
        let w = build_text_world(&clip(8, 8, 4), &opts, &mut NoProgress).unwrap();
        for pos in inner_grid_positions(&w) {
            assert!(
                pos.x >= 0 && pos.y >= 0 && pos.z >= 0,
                "negative coord {pos:?} deletes bricks"
            );
        }
    }

    #[test]
    fn each_band_anchor_sits_at_its_own_image_row_with_no_depth_stack() {
        let opts = AnimOptions::default();
        let w = build_text_world(&clip(192, 8, 2), &opts, &mut NoProgress).unwrap();
        let anchors = text_anchor_positions(&w);
        assert_eq!(anchors.len(), plan_bands(192, 8, 2).unwrap().len());
        // `add_text_tiles` stacks a tile's extra bands along world X, so DEPTH
        // is x -- one anchor per band means nothing may stack there.
        let mut depths: Vec<i32> = anchors.iter().map(|p| p.x).collect();
        depths.sort_unstable();
        depths.dedup();
        assert_eq!(depths.len(), 1, "anchors must not stack in depth: {depths:?}");
        // ... and each band must anchor at its OWN image row, which is z.
        let mut rows: Vec<i32> = anchors.iter().map(|p| p.z).collect();
        rows.sort_unstable();
        rows.dedup();
        assert_eq!(rows.len(), anchors.len(), "each band must sit at its own height");
    }

    /// Run at BOTH a small size and the 192x108 this mode is reported at: the
    /// latter puts 54 bands in the chip, whose lattice then spans far enough to
    /// land inner bricks in more than one brick chunk -- exactly where a wire's
    /// chunk-relative brick index can go wrong. The multi-bank case is included
    /// because its branch/select cascade is wiring nothing else here reaches.
    #[test]
    fn the_produced_world_passes_wire_integrity() {
        for (w, h, frames, bank_size, tag) in [
            (64u32, 16u32, 3usize, usize::MAX, "small"),
            (192, 108, 2, usize::MAX, "wide"),
            (64, 16, 5, 2, "banked"),
        ] {
            let opts = AnimOptions { bank_size, ..AnimOptions::default() };
            let world = build_text_world(&clip(w, h, frames), &opts, &mut NoProgress)
                .unwrap_or_else(|e| panic!("{w}x{h}x{frames} bank {bank_size} must build: {e}"));
            let path = std::env::temp_dir().join(format!(
                "h2b_text_wires_{tag}_{}_{:?}.brz",
                std::process::id(),
                std::thread::current().id()
            ));
            std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
            let result =
                std::panic::catch_unwind(|| super::wire_integrity::assert_wires_valid(&path));
            let _ = std::fs::remove_file(&path);
            if let Err(e) = result {
                std::panic::resume_unwind(e);
            }
        }
    }

    #[test]
    fn quantizing_reduces_the_tag_count_on_a_gradient() {
        // Same clip, same layout, palette on vs off: the palette must produce
        // strictly fewer colour tags, which is the entire reason it exists.
        let off = AnimOptions { colors: 0, ..AnimOptions::default() };
        let on = AnimOptions { colors: 8, ..AnimOptions::default() };
        let tags = |o: &AnimOptions| {
            total_color_tags(&build_text_world(&gradient_clip(), o, &mut NoProgress).unwrap())
        };
        assert!(tags(&on) < tags(&off), "a palette must lengthen runs");
    }

    #[test]
    fn a_clip_past_one_bank_spills_and_selects_per_band() {
        let opts = AnimOptions {
            // parameterised so the seam is testable cheaply
            bank_size: 2,
            ..AnimOptions::default()
        };
        let w = build_text_world(&clip(8, 8, 5), &opts, &mut NoProgress).unwrap();
        let bands = plan_bands(8, 8, 2).unwrap().len();
        // 5 frames at bank size 2 -> 3 banks, 2 boundaries.
        assert_eq!(
            count_components(&w, ARRAY_VAR),
            bands * 3,
            "one array per band per bank"
        );
        assert_eq!(
            count_components(&w, SELECT),
            bands * 2,
            "one select per band per boundary"
        );
        // The rest of the boundary hardware: one subtract, one comparator and
        // one branch per boundary, all SHARED across bands. Spelled out here so
        // the cost formula has a witness -- gates are every inner brick that is
        // not one of the chip's five I/O pins.
        let (banks, boundaries) = (3usize, 2usize);
        for shared in [SUBTRACT, COMPARE_GE, BRANCH] {
            assert_eq!(count_components(&w, shared), boundaries, "{shared} per boundary");
        }
        assert_eq!(
            w.grids[0].1.len() - 5,
            2 * bands * banks + 5 + boundaries * 3 + boundaries * bands,
            "2 per band per bank + the 5 service gates + 3 shared per boundary + 1 select per \
             band per boundary"
        );
    }

    /// The sampling pass must span the WHOLE clip, not its opening shot -- a
    /// source with scene cuts would otherwise get a palette from frame 0
    /// alone. Pinned on both `FrameSource` shapes: one that reports its own
    /// length (the seek path) and one that does not (the decimating fallback).
    #[test]
    fn palette_sampling_spreads_across_the_whole_clip() {
        /// A clip whose frame `i` is the solid colour `(i, 0, 0)`, so a
        /// sampled frame names its own index.
        fn numbered(n: usize) -> Clip {
            Clip {
                width: 1,
                height: 1,
                fps: 10.0,
                frames: (0..n)
                    .map(|i| RgbaImage::from_pixel(1, 1, Rgba([i as u8, 0, 0, 255])))
                    .collect(),
            }
        }

        let c = numbered(100);
        let got = sample_frames(&c, 10, &mut NoProgress).expect("sample");
        let indices: Vec<u8> = got.iter().map(|f| f.get_pixel(0, 0).0[0]).collect();
        assert_eq!(
            indices,
            vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "i * len / n indexing, exactly as examples/text_sizing.rs samples"
        );

        // Asking for more than exist yields every frame, once.
        let all = sample_frames(&numbered(5), 120, &mut NoProgress).expect("sample");
        assert_eq!(
            all.iter().map(|f| f.get_pixel(0, 0).0[0]).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );

        // A source with no frame-count hint: the decimating fallback must
        // still span the clip rather than stopping at its first `n` frames.
        struct NoHint(Clip);
        impl FrameSource for NoHint {
            fn info(&self) -> crate::video::stream::SourceInfo {
                crate::video::stream::SourceInfo {
                    frame_count_hint: None,
                    ..self.0.info()
                }
            }
            fn open(&self) -> Result<Box<dyn crate::video::stream::FrameStream + '_>, String> {
                self.0.open()
            }
        }
        let blind = sample_frames(&NoHint(numbered(100)), 10, &mut NoProgress).expect("sample");
        let indices: Vec<u8> = blind.iter().map(|f| f.get_pixel(0, 0).0[0]).collect();
        assert!(
            indices.len() <= 10,
            "the fallback must stay within the cap: {indices:?}"
        );
        assert!(
            *indices.last().expect("non-empty") >= 80,
            "the fallback must reach the END of the clip, not just its first frames: {indices:?}"
        );
    }

    /// A zero-frame source must be an error, not a save that divides by zero
    /// in-game on every tick -- the same guard both brick paths carry.
    #[test]
    fn a_zero_frame_clip_is_rejected() {
        let c = Clip { width: 8, height: 8, fps: 10.0, frames: Vec::new() };
        assert!(
            build_text_world(&c, &AnimOptions::default(), &mut NoProgress)
                .map(|_| ())
                .is_err(),
            "an empty clip must not produce a \"successful\" save"
        );
    }

    /// A single-bank render must emit no spillover hardware at all, so the
    /// common case is wired exactly as the in-game-verified probe wired it.
    #[test]
    fn a_single_bank_render_emits_no_spillover_gates() {
        let w = build_text_world(&clip(8, 8, 5), &AnimOptions::default(), &mut NoProgress).unwrap();
        for class in [COMPARE_GE, BRANCH, SELECT, SUBTRACT] {
            assert_eq!(
                count_components(&w, class),
                0,
                "a single-bank render must emit no {class}"
            );
        }
    }

    /// Exec **fan-in** is the unverified case, so this design must never
    /// produce it: no exec input may have two sources, at any bank count.
    #[test]
    fn no_exec_input_ever_has_two_sources() {
        for bank_size in [usize::MAX, 3, 2, 1] {
            let opts = AnimOptions { bank_size, ..AnimOptions::default() };
            let w = build_text_world(&clip(64, 24, 7), &opts, &mut NoProgress).expect("build");
            let mut seen = std::collections::HashSet::new();
            for wire in &w.wires {
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
        }
    }

    /// Every band's TextDisplay must be driven by exactly one wire, on the
    /// `Text` port -- one band, one string, no crossing over.
    #[test]
    fn every_text_display_is_driven_exactly_once() {
        let w = build_text_world(&clip(64, 16, 3), &AnimOptions::default(), &mut NoProgress)
            .expect("build");
        let bands = plan_bands(64, 16, 2).unwrap().len();
        let mut fed = std::collections::HashMap::new();
        for wire in &w.wires {
            if wire.target.component_type.to_string() == TEXT_DISPLAY {
                assert_eq!(wire.target.port_name.to_string(), TEXT_PORT);
                *fed.entry(wire.target.brick_id).or_insert(0usize) += 1;
            }
        }
        assert_eq!(fed.len(), bands, "every band's component must be driven");
        assert!(
            fed.values().all(|n| *n == 1),
            "no TextDisplay may be driven twice"
        );
    }

    /// **The headline number.** A 192x108 screen is 54 bands, so a
    /// single-bank render must cost exactly `2 * 54 + 5` gates -- two per band
    /// plus the shared clock chain and change detector -- against brick mode's
    /// 2 per PIXEL. Gates are every inner-grid brick that is not one of the
    /// chip's five I/O pins (Pause, Restart, Resume, Rate, Done), counted the
    /// same way `tests/anim_color.rs` counts them.
    #[test]
    fn a_192x108_render_costs_two_gates_per_band_plus_the_clock() {
        let w = build_text_world(&clip(192, 108, 2), &AnimOptions::default(), &mut NoProgress)
            .expect("build");
        let bands = plan_bands(192, 108, 2).unwrap().len();
        assert_eq!(bands, 54, "192 wide at char_repeat 2 bands 2 rows at a time");
        assert_eq!(
            w.grids[0].1.len() - 5,
            2 * bands + 5,
            "two gates per band, plus the 4-gate clock chain and the change detector"
        );
        assert_eq!(
            w.bricks.len(),
            bands + 1,
            "one anchor cube per band, plus the chip shell"
        );
    }

    /// The service stage must sit strictly BEHIND every band stage, or the
    /// clock can collide with the deepest band gates. Mirrors
    /// `color_bricks`'s equivalent, since the two share the arithmetic.
    #[test]
    fn the_service_stage_clears_every_band_stage() {
        for n_banks in 1..=6usize {
            let boundaries = (n_banks - 1) as i32;
            let service = STAGES_PER_BANK * n_banks as i32 + boundaries;
            let deepest_band_stage = if n_banks == 1 {
                STAGES_PER_BANK - 1
            } else {
                STAGES_PER_BANK * n_banks as i32 + boundaries - 1
            };
            assert!(
                service > deepest_band_stage,
                "{n_banks} banks: service stage {service} must clear the deepest band stage \
                 {deepest_band_stage}"
            );
        }
    }
}
