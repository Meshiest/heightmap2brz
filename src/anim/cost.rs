//! Build-cost estimation. Gate count is what actually limits a build, so the
//! UI shows this before the user commits to a render.
//!
//! Every estimator here takes the whole [`AnimOptions`] rather than the loose
//! numbers it happens to need -- the one property that keeps the readout and
//! the render describing the same graph. A parameter list that cannot
//! express a difference (e.g. `char_repeat`, or whether a subtitle track is
//! set) cannot get it wrong the way a hand-picked subset can.
use super::bricks::AnimOptions;
use super::pack::{HEX_STRIDE, PIXELS_PER_CHUNK};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    pub pixels: usize,
    pub gates: usize,
    pub wires: usize,
    pub bricks: usize,
    pub chunks: usize,
    pub banks: usize,
    pub frames: usize,
    pub chars: usize,
}

/// What a subtitle track adds to a render, in `(gates, wires, bricks)`.
///
/// Exactly what [`crate::anim::subtitle_display::add_subtitle_display`]
/// builds, which is the same shape as one more band of text mode: an
/// `ArrayVar` + an `ArrayVar_Get` per bank, a `Select` per bank boundary,
/// three wires per bank (`ArrayVarRef`, `Index`, `Exec`) plus three per
/// boundary (the select's inputs) plus one into the `TextDisplay`'s `Text`
/// port, and one main-grid anchor cube.
///
/// `(0, 0, 0)` when there are no subtitles -- the hard gate every renderer
/// applies, so a subtitle-free estimate is byte-for-byte the number it was
/// before this existed.
///
/// Shared by all three estimators rather than written out three times,
/// because the subtitle piece is genuinely mode-independent: it is wired only
/// to the frame index and knows nothing about how the picture is drawn. Three
/// copies of a formula that must agree is how a readout drifts from its
/// render.
fn subtitle_cost(opts: &AnimOptions, banks: usize) -> (usize, usize, usize) {
    if opts.subtitles.is_none() {
        return (0, 0, 0);
    }
    let boundaries = banks.saturating_sub(1);
    (
        2 * banks + boundaries,
        3 * banks + 3 * boundaries + 1,
        1,
    )
}

/// What the pre-wired control buttons add, in `(gates, wires, bricks)`.
///
/// Exactly what [`crate::anim::controls::add_control_buttons`] builds: three
/// animated-button bricks on the main grid, each carrying its own label
/// component (no separate cube), so [`crate::anim::controls::CONTROL_BRICKS`]
/// main-grid bricks and [`crate::anim::controls::CONTROL_WIRES`] wires (one
/// `bHeld` wire per button), and no gate: the buttons are main-grid bricks, and
/// `gates` here (as the estimate-vs-render tests measure it) counts inner-grid
/// gates only. The buttons are counted in `bricks`, the same slot the shell,
/// display bricks and subtitle anchor already sit in.
///
/// `(0, 0, 0)` when off, and also when [`AnimOptions::external_clock`] is set --
/// that render builds no timer, so there are no pins to drive and the renderer
/// skips the buttons. This matches the render exactly rather than following the
/// clock's own over-count, and no estimate-vs-render test pairs an external
/// clock with the buttons on.
fn control_button_cost(opts: &AnimOptions) -> (usize, usize, usize) {
    if opts.control_buttons && !opts.external_clock {
        (0, super::controls::CONTROL_WIRES, super::controls::CONTROL_BRICKS)
    } else {
        (0, 0, 0)
    }
}

/// Estimate the build cost of a `width * height` screen over `frames` frames,
/// spilling across arrays of at most `opts.bank_size` frames each.
///
/// This is an upper bound, and the bound is not uniform across fields.
/// `gates`, `wires` and `bricks` assume every pixel survives; a pixel that is
/// fully transparent across the whole clip is culled and emits no display
/// brick and no gates, so a real render of a mostly-transparent clip can come
/// in well under these numbers. That divergence is by design -- this feeds a UI
/// readout shown before a render is committed to, and it deliberately never
/// under-promises.
///
/// `chars` is exact, not a bound. A culled pixel still reserves its
/// [`HEX_STRIDE`] characters in every frame string, so that each surviving
/// pixel's offset stays a plain `pixel_in_chunk * HEX_STRIDE` with no remap
/// table -- see [`super::pack`].
///
/// `opts.subtitles` adds [`subtitle_cost`] on top; the whole `AnimOptions` is
/// taken rather than a `bank_size` so it cannot be read from a different
/// value than the render uses (see the module doc).
pub fn estimate(width: u32, height: u32, frames: usize, opts: &AnimOptions) -> Cost {
    let pixels = width as usize * height as usize;
    let chunks = pixels.div_ceil(PIXELS_PER_CHUNK).max(1);
    let banks = frames.div_ceil(opts.bank_size.max(1)).max(1);
    let boundaries = banks - 1;
    let (sub_gates, sub_wires, sub_bricks) = subtitle_cost(opts, banks);
    let (cb_gates, cb_wires, cb_bricks) = control_button_cost(opts);
    Cost {
        pixels,
        // 2 per pixel + 2 per chunk per bank (ArrayVar, Get) + 6 clock
        // + 1 detector, plus per boundary: comparator, branch, index subtract,
        // and one value select per chunk. The 6 clock gates are the 4 index
        // gates plus the length and progress status taps.
        gates: 2 * pixels + 2 * chunks * banks + 7 + boundaries * 3 + boundaries * chunks
            + sub_gates
            + cb_gates,
        // 3 per pixel + 2 per chunk per bank + exec chain + detector feed
        // + 11 clock (3 chain + 3 control pins + Rate + Done + the length and
        // progress taps: 2 pin writes + 1 shared frame-index read), plus per
        // boundary: comparator InputA (1) + subtract InputA (1) + branch
        // bCond/Exec (2) + one select's bSelectB/InputA/InputB (3) per chunk
        //
        // The exec chain is `chunks * banks`, not `chunks * banks - 1`:
        // `build_brick_world` writes one wire per `Get` including the first
        // (`detector.OnChanged -> get0.Exec`), and the separate `+ 1` here is
        // the detector's own feed (`frame_index -> detector.Input`), so
        // nothing absorbs that first link.
        wires: 3 * pixels
            + 2 * chunks * banks
            + chunks * banks
            + 1
            + 11
            + boundaries * (3 * chunks + 4)
            + sub_wires
            + cb_wires,
        // one display brick per pixel + the microchip shell + control buttons
        bricks: pixels + 1 + sub_bricks + cb_bricks,
        chunks,
        banks,
        frames,
        chars: pixels * frames * HEX_STRIDE,
    }
}

/// Estimate the build cost of the same screen rendered in colour-array mode
/// ([`crate::anim::color_bricks`]) instead of hex mode.
///
/// A separate function rather than a parameter on [`estimate`], because
/// almost every term differs: there are no pack chunks, no characters, one
/// array and one `Get` per pixel rather than per chunk, no `Substring` or
/// `MakeColorHex`, and a `Select` per pixel (not per chunk) at every bank
/// boundary. Handing [`estimate`]'s number to a colour-array render would be
/// wrong in both directions at once -- it over-counts the per-pixel expression
/// gates and wildly under-counts a boundary.
///
/// Like [`estimate`] this is an upper bound on `gates`, `wires` and `bricks`:
/// it assumes every pixel survives, and a pixel transparent across the whole
/// clip is culled and emits nothing at all.
///
/// The counts assume the built-in clock. `AnimOptions::external_clock`
/// replaces the 6 clock gates and their 11 wires with a single input pin, so
/// that render comes in slightly under -- the same simplification [`estimate`]
/// makes.
///
/// `chunks` and `chars` are reported as 0, and that is the true answer
/// rather than a placeholder: this encoding tiles nothing into chunks and
/// writes no strings whatsoever. What it does write is
/// `pixels * frames` `(R,G,B,A)` array elements, which
/// [`crate::anim::color_bricks::array_elements`] reports if a caller wants it.
pub fn estimate_color_array(width: u32, height: u32, frames: usize, opts: &AnimOptions) -> Cost {
    let pixels = width as usize * height as usize;
    let banks = frames.div_ceil(opts.bank_size.max(1)).max(1);
    let boundaries = banks - 1;
    let (sub_gates, sub_wires, sub_bricks) = subtitle_cost(opts, banks);
    let (cb_gates, cb_wires, cb_bricks) = control_button_cost(opts);
    Cost {
        pixels,
        // per pixel: one ArrayVar + one Get per bank, plus one Select per
        // boundary; plus 6 clock + 1 detector; plus per boundary a comparator,
        // a branch and an index subtract. The 6 clock gates are the 4 index
        // gates plus the length and progress status taps.
        gates: pixels * (2 * banks + boundaries) + 7 + boundaries * 3 + sub_gates + cb_gates,
        // per pixel: ArrayVarRef + Index + Exec per bank (3), plus
        // bSelectB/InputA/InputB per boundary (3), plus the one wire into the
        // display brick's Color; plus the detector feed (1), the clock's own
        // 11 (3 chain + 3 control pins + Rate + Done + length/progress taps);
        // plus per boundary the comparator's InputA, the subtract's InputA and
        // the branch's bCond/Exec.
        wires: pixels * (3 * banks + 3 * boundaries + 1) + 1 + 11 + boundaries * 4 + sub_wires
            + cb_wires,
        // one display brick per pixel + the microchip shell + control buttons
        bricks: pixels + 1 + sub_bricks + cb_bricks,
        // Genuinely zero, not unknown: see the doc comment.
        chunks: 0,
        banks,
        frames,
        chars: 0,
    }
}

/// Estimate the build cost of a `width * height` text-mode screen
/// ([`crate::anim::text_bricks`]) over `frames` frames, spilling across
/// arrays of at most `opts.bank_size` frames each.
///
/// Text mode's unit of cost is the band
/// ([`crate::anim::text_layout::plan_bands`]), not the pixel: each band gets
/// its own `ArrayVar` + `ArrayVar_Get` per bank, plus a `Select` per boundary
/// -- the same shape [`estimate`] and [`estimate_color_array`] use, just
/// counted per band instead of per pixel or per chunk. That is what makes
/// text mode roughly two orders of magnitude cheaper on gates than either
/// brick encoding at typical screen sizes: a 192x108 screen is 2304 pixels
/// but only 54 bands.
///
/// The band layout depends on `opts.text.char_repeat`
/// ([`crate::text::TextOptions`]); reading it off the same `AnimOptions` the
/// render is built from (rather than a separate parameter) is what keeps a
/// `--font` choosing a different-width glyph from silently disagreeing with
/// the render it describes -- see the module doc.
///
/// This is not the same tradeoff [`estimate`] and [`estimate_color_array`] make
/// by assuming the built-in clock. That assumption is off by a fixed handful of
/// gates; this one changes the unit the whole estimate is counted in.
///
/// Confirmed against a real render, not just arithmetic: a 192x108 clip
/// renders at exactly 115 gates and 55 bricks in
/// `text_bricks::tests::a_192x108_render_costs_two_gates_per_band_plus_the_clock`,
/// matching `2 * 54 + 7` and `54 + 1` here.
///
/// `chars` is reported as 0, and that is the honest answer rather than a
/// placeholder. Unlike hex mode's fixed per-pixel stride, text mode writes an
/// explicit `<color="RRGGBB">` tag at the start of every colour run, so
/// length depends on the clip's actual content -- something this function has
/// no way to know before a render. A number that looks like a measurement but
/// is a guess is worse than no number: the CLI instead prints
/// [`crate::anim::text_layout::worst_case_row_chars`]'s bound separately,
/// clearly labelled as a bound rather than an estimate.
///
/// `pixels` is still `width * height`, for display purposes (it is not the
/// unit gates/bricks are counted in here). `chunks` is genuinely 0: text
/// mode tiles nothing into [`super::pack`] chunks, the same true-zero
/// [`estimate_color_array`] reports for the same reason.
///
/// # Why this returns a `Result` and its two siblings do not
///
/// Text mode is the only encoding whose geometry can be impossible:
/// [`super::text_layout::plan_bands`] caps the width it can band (555 px at
/// `char_repeat` 2), and past that there is no layout at all. Swallowing
/// that into a band count of 0 reads out as "5 gate(s), 1 brick(s)" -- a
/// plausible, cheap-looking render for a configuration `build_text_world`
/// then refuses outright. An impossible geometry has to surface as
/// impossible, not as a small number.
pub fn estimate_text(
    width: u32,
    height: u32,
    frames: usize,
    opts: &AnimOptions,
) -> Result<Cost, String> {
    // The band count depends on `char_repeat`: a single-glyph font (Orbitron,
    // `char_repeat` 1) fits more rows per component than a double-glyph one,
    // so 192x108 bands 36 ways rather than 54. Reading the option off the
    // same struct the render is built from, rather than taking it as a
    // separate parameter, is what keeps this in sync with `--font`.
    //
    // Propagated, not swallowed: see this function's doc.
    let bands = super::text_layout::plan_bands(
        width as usize,
        height as usize,
        opts.text.char_repeat.max(1),
    )?
    .len();
    let banks = frames.div_ceil(opts.bank_size.max(1)).max(1);
    let boundaries = banks - 1;
    let (sub_gates, sub_wires, sub_bricks) = subtitle_cost(opts, banks);
    let (cb_gates, cb_wires, cb_bricks) = control_button_cost(opts);
    Ok(Cost {
        pixels: width as usize * height as usize,
        // 2 per band per bank (ArrayVar, Get) + 6 clock + 1 detector, plus
        // per boundary: comparator, branch, index subtract, and one select
        // per band. The 6 clock gates are the 4 index gates plus the length
        // and progress status taps.
        gates: 2 * bands * banks + 7 + boundaries * 3 + boundaries * bands + sub_gates + cb_gates,
        // 3 per band per bank (ArrayVarRef, Index, Exec) + 1 per band (the
        // wire into the TextDisplay's Text port) + detector feed (1) + 3
        // clock chain + 3 control pins + Rate + Done + length/progress taps
        // (11), plus per boundary: comparator InputA (1) + subtract InputA (1)
        // + branch bCond/Exec (2) + one select's bSelectB/InputA/InputB (3)
        // per band.
        wires: bands * (3 * banks + 3 * boundaries + 1) + 1 + 11 + boundaries * 4 + sub_wires
            + cb_wires,
        // one TextDisplay anchor per band + the microchip shell + control buttons
        bricks: bands + 1 + sub_bricks + cb_bricks,
        // Genuinely zero, not unknown: text mode tiles nothing into chunks.
        chunks: 0,
        banks,
        frames,
        // Content-dependent; see the doc comment.
        chars: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::pack::BANK_FRAMES;
    use crate::subs::{Cue, Subtitles};
    use std::sync::Arc;

    /// Default options at a given bank size -- what every pre-subtitle
    /// assertion in this module is written against, so the numbers below stay
    /// the ones a render without subtitles has always produced.
    fn opts(bank_size: usize) -> AnimOptions {
        // `control_buttons: false` so every absolute formula assertion below
        // stays the button-free number it has always been; the buttons add a
        // flat main-grid cost measured separately in
        // `the_control_buttons_add_a_flat_main_grid_cost`.
        AnimOptions { bank_size, control_buttons: false, ..AnimOptions::default() }
    }

    /// The same, with a subtitle track. The track's content is irrelevant to
    /// the cost -- one `ArrayVar` holds every frame's line however many cues
    /// there are -- so this is deliberately a single cue.
    fn subbed(bank_size: usize) -> AnimOptions {
        AnimOptions {
            bank_size,
            // Buttons off too, so the subtitle delta tests (`with` - `without`)
            // isolate the subtitle cost -- both sides carry the same (zero)
            // button cost.
            control_buttons: false,
            subtitles: Some(Arc::new(Subtitles::new(vec![Cue {
                start_s: 0.0,
                end_s: 1.0,
                text: "hi".to_string(),
            }]))),
            ..AnimOptions::default()
        }
    }

    /// The control buttons add a flat main-grid cost -- 3 bricks and 3 wires, no
    /// gate -- on top of any render, in every mode, exactly as
    /// `crate::anim::controls` builds it and each renderer wires it.
    #[test]
    fn the_control_buttons_add_a_flat_main_grid_cost() {
        use crate::anim::controls::{CONTROL_BRICKS, CONTROL_WIRES};
        let on = |o: AnimOptions| AnimOptions { control_buttons: true, ..o };
        for (without, with) in [
            (
                estimate(64, 36, 300, &opts(BANK_FRAMES)),
                estimate(64, 36, 300, &on(opts(BANK_FRAMES))),
            ),
            (
                estimate_color_array(64, 36, 300, &opts(BANK_FRAMES)),
                estimate_color_array(64, 36, 300, &on(opts(BANK_FRAMES))),
            ),
            (
                estimate_text(192, 108, 300, &opts(BANK_FRAMES)).unwrap(),
                estimate_text(192, 108, 300, &on(opts(BANK_FRAMES))).unwrap(),
            ),
        ] {
            assert_eq!(with.gates - without.gates, 0, "the detectors are main-grid, not gates");
            assert_eq!(with.wires - without.wires, CONTROL_WIRES);
            assert_eq!(with.bricks - without.bricks, CONTROL_BRICKS);
        }
        // And an external clock, which builds no timer, carries no buttons even
        // with the toggle on -- there are no pins to drive.
        let ext = AnimOptions { external_clock: true, control_buttons: true, ..opts(BANK_FRAMES) };
        assert_eq!(
            estimate(64, 36, 300, &ext).bricks,
            estimate(64, 36, 300, &opts(BANK_FRAMES)).bricks,
            "external clock exposes no control pins, so the estimate adds no buttons"
        );
    }

    #[test]
    fn matches_the_spec_formula() {
        // 64x36 = 2304 px; ceil(2304/1666) = 2 chunks
        // gates = 2*2304 + 2*2 + 7 = 4619
        let c = estimate(64, 36, 300, &opts(BANK_FRAMES));
        assert_eq!(c.pixels, 2304);
        assert_eq!(c.chunks, 2);
        assert_eq!(c.gates, 4619);
        // wires = 3/pixel + 2/chunk (ArrayVarRef, Index) + chunks exec chain
        //         + 1 detector feed + 11 clock
        //       = 6912 + 4 + 2 + 1 + 11 = 6930
        // Pinned against a real render in
        // `tests/anim_world.rs::the_hex_cost_estimate_matches_a_real_render`.
        assert_eq!(c.wires, 6930);
    }

    #[test]
    fn a_single_chunk_screen_needs_one_array() {
        assert_eq!(estimate(10, 10, 5, &opts(BANK_FRAMES)).chunks, 1);
    }

    #[test]
    fn char_total_is_pixels_times_frames_times_stride() {
        assert_eq!(estimate(10, 10, 5, &opts(BANK_FRAMES)).chars, 100 * 5 * 6);
    }

    #[test]
    fn the_chunk_boundary_is_exact() {
        assert_eq!(
            estimate(1666, 1, 2, &opts(BANK_FRAMES)).chunks,
            1,
            "exactly one chunk's worth"
        );
        assert_eq!(
            estimate(1667, 1, 2, &opts(BANK_FRAMES)).chunks,
            2,
            "one pixel over spills"
        );
    }

    #[test]
    fn a_sub_limit_clip_is_one_bank_and_costs_what_it_used_to() {
        let c = estimate(64, 36, 300, &opts(BANK_FRAMES));
        assert_eq!(c.banks, 1);
        assert_eq!(c.gates, 4619, "unchanged from the pre-spillover formula");
    }

    #[test]
    fn spilling_adds_three_gates_per_boundary_plus_one_select_per_chunk() {
        // 2304 px -> 2 chunks; 5 frames at bank size 2 -> 3 banks, 2 boundaries
        let one = estimate(64, 36, 5, &opts(65_535));
        let three = estimate(64, 36, 5, &opts(2));
        assert_eq!(three.banks, 3);
        // per boundary: comparator + branch + subtract, plus a select per chunk
        // plus the extra array/Get pair per chunk per extra bank
        let boundaries = 2;
        let chunks = 2;
        let extra = boundaries * 3 + boundaries * chunks + 2 * chunks * boundaries;
        assert_eq!(three.gates - one.gates, extra);
    }

    #[test]
    fn spilling_adds_the_boundary_wiring_to_the_wires_estimate() {
        // Same setup as the gates counterpart above: 2304 px -> 2 chunks;
        // 5 frames at bank size 2 -> 3 banks, 2 boundaries.
        let one = estimate(64, 36, 5, &opts(65_535));
        let three = estimate(64, 36, 5, &opts(2));
        assert_eq!(three.banks, 3);
        let boundaries = 2;
        let chunks = 2;
        // the extra array/Get pair per chunk per extra bank contributes 2
        // wires each (ArrayVarRef, Index), plus one more link in the exec
        // chain per extra bank per chunk; each boundary itself adds a
        // comparator InputA + subtract InputA + branch bCond/Exec (2) + one
        // select's bSelectB/InputA/InputB (3) per chunk
        let extra_bank_wires = 2 * chunks * boundaries + chunks * boundaries;
        let extra_boundary_wires = boundaries * (3 * chunks + 4);
        assert_eq!(three.wires - one.wires, extra_bank_wires + extra_boundary_wires);
        assert_eq!(three.wires - one.wires, 32, "12 bank wires + 20 boundary wires");
    }

    #[test]
    fn banking_moves_characters_between_arrays_without_changing_the_total() {
        assert_eq!(estimate(64, 36, 100, &opts(65_535)).chars, estimate(64, 36, 100, &opts(7)).chars);
    }

    // --- colour-array mode ---------------------------------------------------

    /// The headline comparison, at the reference 64x36 screen: colour-array
    /// mode keeps hex mode's 2 components per pixel and drops the per-chunk
    /// array/Get pair, so it comes in slightly cheaper on gates while doing
    /// half the per-frame work.
    #[test]
    fn a_single_bank_colour_array_render_is_two_gates_per_pixel_plus_overhead() {
        let c = estimate_color_array(64, 36, 300, &opts(BANK_FRAMES));
        assert_eq!(c.pixels, 2304);
        assert_eq!(c.banks, 1);
        assert_eq!(c.gates, 2 * 2304 + 7, "2 gates per pixel + 6 clock + 1 detector");
        assert_eq!(c.wires, 4 * 2304 + 12, "4 wires per pixel + detector feed + 11 clock");
        assert_eq!(c.bricks, 2304 + 1);

        // Against hex mode's own estimate for the same screen: the two
        // per-chunk array/Get pairs are the whole difference.
        let hex = estimate(64, 36, 300, &opts(BANK_FRAMES));
        assert_eq!(hex.gates - c.gates, 2 * hex.chunks, "hex pays 2 gates per chunk extra");
    }

    /// No chunking and no strings -- both genuinely zero, and both must stay
    /// zero however the frames bank.
    #[test]
    fn colour_array_mode_reports_no_chunks_and_no_characters() {
        for bank in [BANK_FRAMES, 7, 1] {
            let c = estimate_color_array(64, 36, 100, &opts(bank));
            assert_eq!(c.chunks, 0, "colour-array mode tiles nothing into chunks");
            assert_eq!(c.chars, 0, "colour-array mode writes no strings at all");
        }
    }

    /// The reason this mode gets its own estimate: a bank boundary costs a
    /// Select per pixel here, against a Select per chunk in hex mode -- on a
    /// 64x36 screen that is 2304 versus 2. Handing a colour-array render hex
    /// mode's boundary arithmetic would under-count it by three orders of
    /// magnitude.
    #[test]
    fn a_boundary_costs_a_select_per_pixel_not_per_chunk() {
        let one = estimate_color_array(64, 36, 5, &opts(BANK_FRAMES));
        let three = estimate_color_array(64, 36, 5, &opts(2));
        assert_eq!(three.banks, 3);
        let boundaries = 2;
        let pixels = 2304;
        // per boundary: an extra ArrayVar + Get per pixel, a Select per pixel,
        // and the shared comparator + branch + subtract.
        assert_eq!(three.gates - one.gates, boundaries * (3 * pixels + 3));

        let hex_one = estimate(64, 36, 5, &opts(BANK_FRAMES));
        let hex_three = estimate(64, 36, 5, &opts(2));
        assert!(
            three.gates - one.gates > 100 * (hex_three.gates - hex_one.gates),
            "a colour-array boundary must be enormously more expensive than a hex one -- \
             that asymmetry is exactly what this separate estimate exists to report"
        );
    }

    /// The per-boundary wire arithmetic, pinned separately from the gates:
    /// each extra bank adds ArrayVarRef/Index/Exec per pixel, each boundary
    /// adds a Select's three inputs per pixel plus the four shared ones.
    #[test]
    fn the_boundary_wiring_is_counted_too() {
        let one = estimate_color_array(64, 36, 5, &opts(BANK_FRAMES));
        let three = estimate_color_array(64, 36, 5, &opts(2));
        let (boundaries, pixels) = (2, 2304);
        assert_eq!(three.wires - one.wires, boundaries * (6 * pixels + 4));
    }

    /// Frame count must not move the gate count while it stays inside one
    /// bank: a longer clip in colour-array mode is a longer array, not more
    /// hardware. (It is more host memory -- see `color_pack`.)
    #[test]
    fn a_longer_clip_inside_one_bank_costs_no_extra_gates() {
        let short = estimate_color_array(32, 18, 10, &opts(BANK_FRAMES));
        let long = estimate_color_array(32, 18, BANK_FRAMES, &opts(BANK_FRAMES));
        assert_eq!(short.banks, long.banks);
        assert_eq!(short.gates, long.gates);
        assert_eq!(short.wires, long.wires);
    }

    // --- text mode -------------------------------------------------------

    #[test]
    fn text_mode_costs_two_gates_per_band_plus_the_clock() {
        // 192x108 at char_repeat 2 -> 54 bands.
        let c = estimate_text(192, 108, 300, &opts(BANK_FRAMES)).expect("a legal geometry must estimate");
        assert_eq!(c.banks, 1);
        assert_eq!(c.gates, 2 * 54 + 7);
        assert_eq!(c.bricks, 54 + 1);
    }

    #[test]
    fn text_mode_is_orders_of_magnitude_under_brick_mode() {
        let text = estimate_text(192, 108, 300, &opts(BANK_FRAMES)).expect("a legal geometry must estimate");
        let brick = estimate(192, 108, 300, &opts(BANK_FRAMES));
        assert!(brick.gates / text.gates > 300, "expected >300x, got {}x", brick.gates / text.gates);
    }

    #[test]
    fn spilling_adds_a_select_per_band_per_boundary() {
        let one = estimate_text(64, 32, 5, &opts(65_535)).expect("a legal geometry must estimate");
        let three = estimate_text(64, 32, 5, &opts(2)).expect("a legal geometry must estimate");
        assert_eq!(three.banks, 3);
        let bands = crate::anim::text_layout::plan_bands(64, 32, 2).unwrap().len();
        let boundaries = 2;
        assert_eq!(
            three.gates - one.gates,
            boundaries * 3 + boundaries * bands + 2 * bands * boundaries
        );
    }

    /// An impossible text geometry must read as impossible, not as a bargain:
    /// `plan_bands` caps the width it can lay out (555 px at `char_repeat` 2),
    /// and past that a band count of 0 reads out as 5 gates and 1 brick -- a
    /// plausible, unusually cheap render for a config `build_text_world`
    /// refuses outright.
    #[test]
    fn an_unlayoutable_text_geometry_is_an_error_not_a_five_gate_render() {
        let o = opts(BANK_FRAMES);
        let too_wide = 600u32;
        // The premise: this really is a geometry the renderer cannot build.
        assert!(
            super::super::text_layout::plan_bands(
                too_wide as usize,
                108,
                o.text.char_repeat.max(1)
            )
            .is_err(),
            "the test's premise is that {too_wide}px cannot be banded at all"
        );

        let err = estimate_text(too_wide, 108, 300, &o)
            .err()
            .expect("an unlayoutable geometry must not come back as a cost");
        assert_eq!(
            err,
            super::super::text_layout::plan_bands(
                too_wide as usize,
                108,
                o.text.char_repeat.max(1)
            )
            .unwrap_err(),
            "the estimate must surface the layout's own error, which is the one the \
             renderer would fail with"
        );

        // 0 bands is 5 gates and 1 brick, indistinguishable from a tiny legal
        // render -- exactly the number this Result now prevents.
        let zero_band_gates = 2 * 0 * 1 + 5;
        assert_eq!(zero_band_gates, 5, "0 bands reads as a 5-gate render");

        // The widest geometry that is layoutable must still estimate, so this
        // rejects only what the renderer rejects.
        assert!(
            estimate_text(555, 108, 300, &o).is_ok(),
            "555px is inside plan_bands' cap and must still produce a cost"
        );
    }

    #[test]
    fn text_mode_reports_no_character_estimate() {
        assert_eq!(estimate_text(192, 108, 300, &opts(BANK_FRAMES)).expect("a legal geometry must estimate").chars, 0,
            "content-dependent; the CLI prints a labelled bound instead");
    }

    // --- subtitles -------------------------------------------------------
    //
    // The readout must match the render: a subtitle track adds exactly two
    // gates, and these tests are what stop the estimate silently missing them.

    /// The headline: two gates, in every mode, at one bank.
    #[test]
    fn a_subtitle_track_costs_exactly_two_gates_in_every_mode() {
        for (name, without, with) in [
            (
                "hex",
                estimate(64, 36, 300, &opts(BANK_FRAMES)),
                estimate(64, 36, 300, &subbed(BANK_FRAMES)),
            ),
            (
                "color-array",
                estimate_color_array(64, 36, 300, &opts(BANK_FRAMES)),
                estimate_color_array(64, 36, 300, &subbed(BANK_FRAMES)),
            ),
            (
                "text",
                estimate_text(192, 108, 300, &opts(BANK_FRAMES)).expect("a legal geometry must estimate"),
                estimate_text(192, 108, 300, &subbed(BANK_FRAMES)).expect("a legal geometry must estimate"),
            ),
        ] {
            assert_eq!(with.gates - without.gates, 2, "{name}: one ArrayVar + one Get");
            assert_eq!(
                with.wires - without.wires,
                4,
                "{name}: ArrayVarRef + Index + Exec + the wire into the TextDisplay"
            );
            assert_eq!(
                with.bricks - without.bricks,
                1,
                "{name}: one main-grid anchor cube for the subtitle component"
            );
        }
    }

    /// No track means no change AT ALL -- the same hard gate the renderers
    /// apply, so every number above stays what it was before subtitles
    /// existed.
    #[test]
    fn no_subtitle_track_changes_nothing() {
        assert_eq!(subtitle_cost(&opts(BANK_FRAMES), 1), (0, 0, 0));
        assert_eq!(subtitle_cost(&opts(BANK_FRAMES), 9), (0, 0, 0));
    }

    /// A subtitle array banks exactly as the screen's arrays do, so past a
    /// bank boundary it costs one more array/Get pair and one `Select` -- the
    /// cost of a single text band, which is what it is.
    #[test]
    fn a_subtitle_track_banks_like_one_more_band() {
        // 5 frames at bank size 2 -> 3 banks, 2 boundaries.
        let without = estimate_text(64, 32, 5, &opts(2)).expect("a legal geometry must estimate");
        let with = estimate_text(64, 32, 5, &subbed(2)).expect("a legal geometry must estimate");
        assert_eq!(with.banks, 3);
        // per bank an ArrayVar + a Get (2 * 3), plus a Select per boundary (2).
        assert_eq!(with.gates - without.gates, 2 * 3 + 2);
        assert_eq!(with.wires - without.wires, 3 * 3 + 3 * 2 + 1);
        assert_eq!(with.bricks - without.bricks, 1, "still one anchor cube");
    }
}
