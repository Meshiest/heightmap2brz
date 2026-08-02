//! Colour-array mode: the same screen of display bricks [`super::bricks`]
//! builds, driven by PIXEL-major linear colour arrays instead of frame-major
//! hex strings.
//!
//! This is a sibling of [`super::bricks::build_brick_world`], not a
//! replacement, and it shares everything above the per-pixel layer with it:
//! the display bricks, the chip shell, the clock (or the external `Frame`
//! pin), the change detector, the per-bank index subtraction and the boundary
//! comparators are all built the same way, from the same helpers.
//!
//! # What differs, per pixel
//!
//! | | hex mode | colour-array mode |
//! |---|---|---|
//! | array | one per 1666-pixel chunk, one ~10 KB string per frame | one per PIXEL, one `(R,G,B,A)` per frame |
//! | `ArrayVar_Get` | one per chunk | one per pixel |
//! | pixel gates | `Substring` + `MakeColorHex` | none |
//! | components/pixel | 2 | 2 |
//! | gate evaluations/pixel/frame | 2 | 1 |
//! | string work/frame | one ~10 KB slice + parse per pixel | none |
//!
//! The component count per pixel is identical -- the per-chunk array/Get pair
//! simply moves onto the pixel and the two expression gates go away. The
//! hypothesis being tested in game is that halving the per-frame evaluations
//! and removing the string churn is worth the extra host memory
//! ([`super::color_pack`]).
//!
//! # Exec wiring: fan-out, not a chain
//!
//! The `ChangeDetectorExec`'s `OnChanged` output feeds EVERY pixel's
//! `Get.Exec` directly. It is deliberately not chained
//! `Get.ExecOut -> Get.Exec` the way [`super::bricks`] chains its handful of
//! per-chunk gets: at one Get per pixel that chain would be thousands of gates
//! deep. The repository owner has confirmed exec **fan-out** (one output
//! driving many exec inputs) is supported and costs the same as a chain.
//!
//! Exec **fan-in** (two sources into one exec input) is the thing that is NOT
//! verified, so this design never requires it -- every `Get.Exec` in the graph
//! has exactly one source, which
//! `tests/anim_color.rs::no_exec_input_ever_has_two_sources` pins.
use super::chip;
use super::clock::{self, gate};
use super::color_pack::{ColorPacker, LinearColor};
use super::layout::{GATE_HALF, STAGE_PITCH, lattice_pos_staged};
use super::pack;
use super::subtitle_display;
// Gate/component names are shared with the hex renderer -- reused, never
// redefined, so a rename can only ever happen in one place.
use super::bricks::{
    ARRAY_GET, ARRAY_VAR, AnimOptions, BRANCH, CHANGE_DETECTOR, COMPARE_GE, PROP_CHANGER, SELECT,
    SUBTRACT,
};
use crate::progress::{FrameTotal, Progress};
use crate::video::stream::FrameSource;
use brdb::{
    AsBrdbValue, WirePort, World,
    schema::{WireArrayVariant, WireVariant},
};

/// Lattice stages a single pixel occupies, per bank: its `ArrayVar` and its
/// `ArrayVar_Get`.
const STAGES_PER_BANK: i32 = 2;

/// Streams `source` into a wired, animated display-brick [`World`] using the
/// colour-array encoding.
///
/// Signature, streaming contract and cancellation semantics are identical to
/// [`super::bricks::build_brick_world`] -- including that a cancelled render
/// returns an EMPTY `Ok(World)` rather than a partial one, that it returns as
/// soon as it notices (after the decode loop, or at the next pixel once the
/// build is under way), and that it is still on the CALLER to re-check
/// `progress.is_cancelled()` and write nothing when it is true.
pub fn build_color_array_world(
    source: &dyn FrameSource,
    opts: &AnimOptions,
    progress: &mut dyn Progress,
) -> Result<World, String> {
    let info = source.info();
    let (w, h) = (info.width as i32, info.height as i32);

    // One fused streaming pass builds both the per-pixel colour arrays and the
    // per-pixel visibility bitmap; no frame is ever retained. Structurally
    // identical to `build_brick_world`'s pull loop (see its comments for why
    // the closure exists and why cancellation breaks rather than errors) --
    // only the accumulator differs.
    // Same fallback-to-an-estimate as the hex path -- see
    // `build_brick_world`'s call and `FrameTotal`'s doc.
    FrameTotal::new(info.frame_count_hint, source.frame_count_estimate())
        .begin(progress, "packing frames");
    // The hint is passed through so every pixel's array is reserved EXACTLY
    // once rather than doubled into place -- which is what makes
    // `accumulator_bytes` (and the memory figure the CLI prints from it) the
    // real peak instead of up to half of it. See `color_pack`'s Memory note.
    let mut packer = ColorPacker::new(
        info.width,
        info.height,
        opts.alpha_threshold,
        info.frame_count_hint.map(|n| n as usize),
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

    let (colors, visible) = packer.finish();
    let frame_count = seen as usize;

    // Same guard, same reason as the hex path: with no frames,
    // `clock::build_clock` would inline `Modulo.InputB = 0.0` and the save
    // would divide by zero in-game on every tick.
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
    // --- 2. The chip --------------------------------------------------------
    //
    // Both shared verbatim with the hex renderer rather than restated here:
    // the screen and the chip shell are the parts the two encodings agree on
    // completely, and every hard-won detail in them (the flush pitch, the
    // per-style half-height, the shell clearance that keeps the game from
    // silently dropping a brick) is one that must not be able to drift between
    // modes.
    let (brick_of, geometry) = super::bricks::add_display_bricks(&mut world, opts, w, h, &visible);
    let mut chip = super::bricks::new_screen_chip(&mut world, w, h, &geometry);

    let bank_size = opts.bank_size.max(1);
    let n_banks = frame_count.div_ceil(bank_size).max(1);

    // Service gates sit BEHIND every pixel stage, so the service stage depends
    // on how many stages a pixel uses -- unlike the hex path, where the pixel
    // stages are always 0 and 1 and the service stage is always 2.
    //
    // A pixel occupies `STAGES_PER_BANK` stages per bank (its ArrayVar and its
    // Get), plus one stage per boundary for its own Select. At a single bank
    // that is stages 0..=1 and a service stage of 2 -- exactly the hex layout.
    let boundaries = (n_banks - 1) as i32;
    let service_stage = STAGES_PER_BANK * n_banks as i32 + boundaries;
    let service =
        |col: i32, row: i32| lattice_pos_staged(col, row, service_stage, h, GATE_HALF, STAGE_PITCH);

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

    // --- 5. Per-bank index and boundary comparators -------------------------
    // Byte-for-byte the same construction as the hex path's: bank 0 reads the
    // frame index directly, bank k subtracts `k * bank_size`, and `ge[k-1]` is
    // true once the frame index reaches bank k (which is `Select`'s `bSelectB`
    // sense -- true picks InputB, the later bank).
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

    // --- 6. Exec entry per bank ---------------------------------------------
    // Branches cascade at the FRONT so exactly one bank's gets run and no exec
    // input ever takes two sources -- the same reasoning (and the same
    // ExecOutA = keep descending / ExecOutB = this bank polarity) as the hex
    // path. What differs is what happens at the entry: hex threads it through
    // a per-chunk chain, this fans it straight out to every pixel's Get.
    //
    // With n_banks == 1 this emits no branch at all and `entry_of_bank[0]` is
    // simply the detector's `OnChanged`.
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

    // --- 7. Two gates per surviving pixel, per bank -------------------------
    //
    // COST NOTE, deliberately recorded here: a bank boundary is far more
    // expensive in this mode than in hex mode. Hex mode pays one extra
    // ArrayVar + Get + Select per CHUNK per boundary (a 64x36 screen is 2
    // chunks); this pays one extra ArrayVar + Get + Select per PIXEL per
    // boundary (2304 of them for that same screen) -- roughly a thousandfold
    // more gates at every seam, because the arrays are pixel-major and each
    // one has to be split independently.
    //
    // It only bites past `BANK_FRAMES` (65 535) frames, which is about 90
    // minutes at 12 fps, so no ordinary render ever reaches it. Below that
    // there is exactly one bank, no comparators, no branches and no selects at
    // all, and the two modes cost the same 2 components per pixel.
    //
    // Walked in row-major pixel order, NOT by iterating `brick_of` -- a
    // `HashMap`'s iteration order is unspecified and would mint gate brick
    // ids in a different order on every run, making two renders of the same
    // clip differ byte for byte for no reason.
    for idx in 0..(w * h) as usize {
        // Polled per pixel, for the reason spelled out on `build_brick_world`'s
        // matching loop -- and this mode needs it more, not less: every pixel
        // here owns an `ArrayVar` holding its colour for the WHOLE clip, so a
        // pixel costs a `frames.to_vec()` of the entire track on top of its two
        // gates.
        if progress.is_cancelled() {
            return Ok(World::new());
        }
        let Some(&brick_id) = brick_of.get(&idx) else {
            continue; // culled in every frame: no display brick, so no gates
        };
        let (col, row) = ((idx as i32) % w, (idx as i32) / w);
        let banks = pack::bank_frames(&colors[idx], bank_size);
        debug_assert_eq!(
            banks.len(),
            n_banks,
            "every pixel's colour list must bank identically -- they are all frame_count long"
        );

        // One (ArrayVar, Get) pair per bank, each pair on its own two stages.
        let mut get_of_bank = Vec::with_capacity(n_banks);
        for (bi, frames) in banks.iter().enumerate() {
            let array_stage = STAGES_PER_BANK * bi as i32;
            let array = gate(
                &mut chip,
                "B_1x1_Gate_Variable_Array",
                ARRAY_VAR,
                lattice_pos_staged(col, row, array_stage, h, GATE_HALF, STAGE_PITCH),
                vec![(
                    "Value",
                    Box::new(WireArrayVariant::LinearColorArray(frames.to_vec()))
                        as Box<dyn AsBrdbValue>,
                )],
            );
            let get = gate(
                &mut chip,
                "B_1x1_Gate_Exec_ArrayVar_Get",
                ARRAY_GET,
                lattice_pos_staged(col, row, array_stage + 1, h, GATE_HALF, STAGE_PITCH),
                vec![],
            );
            world.add_wire_connection(
                WirePort::new(array, ARRAY_VAR, "ArrayVarRef"),
                WirePort::new(get, ARRAY_GET, "ArrayVarRef"),
            );
            world.add_wire_connection(
                index_of_bank[bi].clone(),
                WirePort::new(get, ARRAY_GET, "Index"),
            );
            // THE FAN-OUT. One exec source drives every pixel's Get for this
            // bank; no chaining, and no exec input ever gains a second source.
            world.add_wire_connection(
                entry_of_bank[bi].clone(),
                WirePort::new(get, ARRAY_GET, "Exec"),
            );
            get_of_bank.push(get);
        }

        // Value: one select per boundary, cascading, exactly as the hex path
        // does per chunk. For a frame in bank j, ge[0..j] are true so select j
        // picks bank j and every later select passes it through unchanged.
        let mut value = WirePort::new(get_of_bank[0], ARRAY_GET, "Value");
        for bi in 1..n_banks {
            let sel = gate(
                &mut chip,
                "B_1x1_Gate_Expr_Select",
                SELECT,
                lattice_pos_staged(
                    col,
                    row,
                    STAGES_PER_BANK * n_banks as i32 + bi as i32 - 1,
                    h,
                    GATE_HALF,
                    STAGE_PITCH,
                ),
                vec![],
            );
            world.add_wire_connection(ge[bi - 1].clone(), WirePort::new(sel, SELECT, "bSelectB"));
            world.add_wire_connection(value, WirePort::new(sel, SELECT, "InputA"));
            world.add_wire_connection(
                WirePort::new(get_of_bank[bi], ARRAY_GET, "Value"),
                WirePort::new(sel, SELECT, "InputB"),
            );
            value = WirePort::new(sel, SELECT, "Output");
        }

        // Straight into the display brick's colour. No Substring, no
        // MakeColorHex: the array element already IS a linear colour.
        world.add_wire_connection(value, WirePort::new(brick_id, PROP_CHANGER, "Color"));
    }

    // --- 7b. Subtitles, if any ----------------------------------------------
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
            super::bricks::subtitle_extent(&geometry, w, h, opts.subtitle_lift)?,
        )?;
    }

    // --- 8. Publish ---------------------------------------------------------
    //
    // One last poll before the publish phase, for the same reason
    // `bricks::build_brick_world` has one: the per-pixel loop polls at the top
    // of each iteration and the subtitle step does not poll at all, so without
    // this a late cancel would still pay for `chip::finish`'s two grid
    // collision checks.
    if progress.is_cancelled() {
        return Ok(World::new());
    }
    chip::finish(&mut world, chip)?;
    // Must be last: it registers every component type and port name actually
    // used, and it must see all bricks, grids and wires first.
    world.register_used_components();
    Ok(world)
}

/// The number of `(R, G, B, A)` array elements a render writes: one per pixel
/// per frame, whatever the banking (banking redistributes them between arrays,
/// it never changes the total). Exposed for the cost readout, which reports
/// this in place of hex mode's character count.
pub fn array_elements(pixels: usize, frames: usize) -> usize {
    pixels * frames
}

/// Host bytes the accumulator retains at peak: 16 per element (four `f32`).
/// See [`super::color_pack`]'s memory note -- this is ~2.7x hex mode's 6 bytes
/// per pixel per frame, and it is the figure that decides whether a long
/// render is feasible on a given machine.
///
/// **Exact, not a bound -- but only because [`ColorPacker::new`] reserves each
/// pixel's array exactly.** This same expression used to be printed by the CLI
/// against an accumulator built from unreserved `Vec`s, where Rust's growth
/// doubling meant the process actually held
/// [`unreserved_accumulator_bytes`] -- up to 2x more than the number the user
/// was shown before committing to the render. The reservation is what closed
/// that gap; if it is ever removed, this function is wrong again.
pub fn accumulator_bytes(pixels: usize, frames: usize) -> usize {
    array_elements(pixels, frames) * std::mem::size_of::<LinearColor>()
}

/// What [`accumulator_bytes`] would be if each pixel's array were grown one
/// `push` per frame from an unreserved `Vec` -- i.e. what a render whose
/// source cannot report its own length ahead of decode
/// (`SourceInfo::frame_count_hint` of `None`) really holds.
///
/// Rust's `RawVec` growth doubles from a minimum non-zero capacity of 4, so a
/// `Vec` that has had `frames` elements pushed into it sits at
/// `max(4, next_power_of_two(frames))` capacity. The worst case is one frame
/// past a power of two -- 32,769 frames reserve for 65,536 -- which is just
/// over 2x.
pub fn unreserved_accumulator_bytes(pixels: usize, frames: usize) -> usize {
    if frames == 0 {
        return 0;
    }
    let capacity = frames.next_power_of_two().max(4);
    pixels * capacity * std::mem::size_of::<LinearColor>()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A solid-colour clip whose every pixel is opaque, so nothing is culled
    /// and the per-pixel build loop really does run once per pixel.
    fn solid_clip(frames: usize, w: u32, h: u32) -> crate::video::Clip {
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

    /// A `Progress` that flags cancelled once `tick` has reached
    /// `cancel_after`. Same shape as `bricks`'s `CancelAfter`, restated here
    /// rather than shared because a `#[cfg(test)]` helper does not cross
    /// module boundaries.
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

    /// The same regression this mode's sibling carries (see
    /// `bricks::tests::a_cancel_while_packing_builds_no_graph_at_all`): a
    /// cancel that only stopped the decode left the ENTIRE build to run over
    /// the frames already packed. It costs more here than anywhere -- this
    /// mode gives every pixel its own whole-clip colour array -- so the
    /// no-build guarantee is pinned in this module too, not inherited by
    /// assumption from a shared comment.
    #[test]
    fn a_cancel_while_packing_builds_no_graph_at_all() {
        const TOTAL_FRAMES: usize = 120;
        let clip = solid_clip(TOTAL_FRAMES, 8, 8);

        let mut cancelled = CancelAfter { cancel_after: 4, ticks: 0 };
        let stopped = build_color_array_world(&clip, &AnimOptions::default(), &mut cancelled)
            .expect("a cancelled render must be Ok, not an error");

        let mut ran = CancelAfter { cancel_after: TOTAL_FRAMES as u64 + 1, ticks: 0 };
        let full =
            build_color_array_world(&clip, &AnimOptions::default(), &mut ran).expect("build");
        assert!(
            !full.bricks.is_empty() && !full.wires.is_empty() && !full.grids.is_empty(),
            "the uncancelled control must actually build a graph, or this test proves nothing"
        );

        assert!(cancelled.is_cancelled(), "the reporter must really have flagged cancellation");
        assert!(stopped.bricks.is_empty(), "a cancelled render must place no display bricks");
        assert!(stopped.grids.is_empty(), "a cancelled render must build no chip grid");
        assert!(stopped.wires.is_empty(), "a cancelled render must wire nothing");
    }

    /// The memory note in [`super::color_pack`] states 16 bytes per pixel per
    /// frame as a fact, and both the doc and the CLI readout are wrong if the
    /// element ever stops being four `f32`.
    #[test]
    fn a_linear_colour_is_the_sixteen_bytes_the_docs_claim() {
        assert_eq!(std::mem::size_of::<LinearColor>(), 16, "four f32");
        assert_eq!(array_elements(64 * 36, 300), 64 * 36 * 300);
        assert_eq!(accumulator_bytes(64 * 36, 300), 64 * 36 * 300 * 16);
        // The comparison the mode is judged on: hex mode's frame strings are
        // 6 ASCII bytes per pixel per frame.
        assert_eq!(
            accumulator_bytes(1, 1) / super::super::pack::HEX_STRIDE,
            2,
            "16 bytes against hex's 6 -- the ~2.7x the docs quote"
        );
    }

    /// **The figure the CLI prints must be the memory the process actually
    /// holds.**
    ///
    /// `accumulator_bytes` says `pixels * frames * 16`, and it was printed for
    /// a long time against an accumulator built from unreserved `Vec`s, where
    /// Rust's growth doubling means what is really retained is
    /// `pixels * next_power_of_two(frames) * 16`. At 128x72 over 40,000 frames
    /// that gap is 5.90 GB reported against 9.66 GB held -- the difference,
    /// on a 16 GB machine, between a render that fits and one that gets
    /// OOM-killed an hour in, after the user was shown a number saying it
    /// would fit.
    ///
    /// So this measures the real capacity of a packed pixel's array. It is the
    /// reservation in `ColorPacker::new` that closes the gap; without it this
    /// reads `next_power_of_two`.
    #[test]
    fn the_reservation_is_what_makes_the_memory_figure_true() {
        // A frame count deliberately just past a power of two -- the worst
        // case for doubling, and the one a round number would hide.
        for frames in [1usize, 3, 5, 9, 17, 100] {
            let clip = solid_clip(frames, 3, 2);
            let mut packer = ColorPacker::new(
                3,
                2,
                AnimOptions::default().alpha_threshold,
                Some(frames),
            );
            for frame in &clip.frames {
                packer.push_frame(frame).expect("push");
            }
            let (pixels, _visible) = packer.finish();
            for (i, p) in pixels.iter().enumerate() {
                assert_eq!(p.len(), frames, "pixel {i}: one entry per frame");
                assert_eq!(
                    p.capacity(),
                    frames,
                    "pixel {i}: {frames} frames must occupy EXACTLY {frames} slots -- \
                     a capacity of {} is the growth-doubling slack accumulator_bytes \
                     does not report",
                    p.capacity()
                );
            }
            // And the reported figure now matches what is really held.
            assert_eq!(
                accumulator_bytes(6, frames),
                pixels.iter().map(|p| p.capacity() * 16).sum::<usize>(),
                "{frames} frames: the printed figure must equal the retained bytes"
            );
        }
    }

    /// The no-hint case, stated rather than left implicit: a source that cannot
    /// report its length ahead of decode really does pay the doubling, and
    /// `unreserved_accumulator_bytes` is the honest number for it.
    #[test]
    fn without_a_frame_count_hint_the_accumulator_doubles_into_place() {
        const FRAMES: usize = 17;
        let clip = solid_clip(FRAMES, 2, 2);
        let mut packer = ColorPacker::new(2, 2, AnimOptions::default().alpha_threshold, None);
        for frame in &clip.frames {
            packer.push_frame(frame).expect("push");
        }
        let (pixels, _) = packer.finish();
        let held: usize = pixels.iter().map(|p| p.capacity() * 16).sum();
        assert_eq!(
            held,
            unreserved_accumulator_bytes(4, FRAMES),
            "the unreserved figure must be the one that matches an unreserved packer"
        );
        assert!(
            held > accumulator_bytes(4, FRAMES),
            "17 frames doubling to 32 must cost strictly more than 17 frames reserved -- \
             otherwise this test is not measuring the doubling at all"
        );
        // The exact worst case the doc quotes: one past a power of two.
        assert_eq!(unreserved_accumulator_bytes(1, 32_769), 65_536 * 16);
        assert_eq!(unreserved_accumulator_bytes(1, 0), 0, "no frames, no accumulator");
        assert_eq!(unreserved_accumulator_bytes(1, 1), 4 * 16, "MIN_NON_ZERO_CAP is 4");
    }

    /// The single-bank service stage must land exactly where the hex path puts
    /// it, since at one bank the two layouts are the same shape.
    #[test]
    fn a_single_bank_render_uses_the_same_stage_layout_as_hex_mode() {
        let n_banks = 1usize;
        let boundaries = (n_banks - 1) as i32;
        assert_eq!(STAGES_PER_BANK * n_banks as i32 + boundaries, 2);
    }

    /// Service gates must sit strictly BEHIND every pixel stage, or they can
    /// collide with the deepest pixel gates. The deepest pixel stage is the
    /// last Select's.
    #[test]
    fn the_service_stage_clears_every_pixel_stage() {
        for n_banks in 1..=6usize {
            let boundaries = (n_banks - 1) as i32;
            let service = STAGES_PER_BANK * n_banks as i32 + boundaries;
            let deepest_pixel_stage = if n_banks == 1 {
                STAGES_PER_BANK - 1
            } else {
                STAGES_PER_BANK * n_banks as i32 + boundaries - 1
            };
            assert!(
                service > deepest_pixel_stage,
                "{n_banks} banks: service stage {service} must clear the deepest pixel stage \
                 {deepest_pixel_stage}"
            );
        }
    }
}
