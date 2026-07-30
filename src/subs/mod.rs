//! Subtitle cues and per-frame timing lookup.
//!
//! This module is pure `std`: no `image`, no `brdb`, no `ffmpeg`. It must
//! build for `wasm32-unknown-unknown`. Parsers live in sibling modules
//! (`srt`, `ass` and `vtt`) and produce a [`Subtitles`] track; this module
//! only knows how to time one against a clock, not how to read a file
//! format.

pub mod ass;
// `ffmpeg-sidecar` spawns a subprocess, so this is native-only -- gated the
// same way `video::ffmpeg` and `audio::ffmpeg_src` are gated at their own
// `mod` declarations, so it never reaches the wasm build.
#[cfg(not(target_arch = "wasm32"))]
pub mod extract;
pub mod srt;
pub mod vtt;

/// Parse subtitle text with whichever parser the file is in, choosing by
/// `extension` first and sniffing the content when that does not settle it.
///
/// A DISPATCHER, not a parser: it knows only how to tell [`srt`], [`ass`] and
/// [`vtt`] apart. It lives here rather than in any one parser because both the
/// CLI's `--subtitles` and the GUI's subtitle picker need exactly this
/// decision, and two copies of it would drift -- a file the CLI read as ASS and
/// the GUI read as SubRip would render two different subtitle tracks from one
/// file.
///
/// The extension is authoritative when it is one this crate knows (`srt`,
/// `ass`, `ssa`, `vtt`, `webvtt`, matched case-insensitively). Anything else --
/// `.txt`, `.sub`, no extension at all, a file picked out of a browser sandbox
/// -- falls back to sniffing, in this order:
///
/// 1. a `WEBVTT` signature at the very front, which is WebVTT's only
///    unambiguous marker. It is tested FIRST because WebVTT also contains
///    `-->`, so testing the arrow first would hand every WebVTT file to the
///    SubRip parser, which cannot read one (different fraction separator,
///    optional hours field, a header block with no arrow at all);
/// 2. `[Events]`, ASS's own section header, which appears in no SubRip file;
/// 3. `-->`, SubRip's timing arrow, which appears in no ASS file.
///
/// A file matching none is an `Err` naming the markers, never a silently
/// empty track: an empty track renders as a video with no dialogue, which is
/// indistinguishable from a correct render of a scene where nobody speaks.
/// (An EMPTY `.srt` still parses to an empty track -- that is the parser's own
/// documented "no cues is valid" stance, and the extension said what it was.)
pub fn parse_auto(text: &str, extension: Option<&str>) -> Result<Subtitles, String> {
    match extension.map(|e| e.to_lowercase()).as_deref() {
        Some("srt") | Some("subrip") => return srt::parse(text),
        Some("vtt") | Some("webvtt") => return vtt::parse(text),
        Some("ass") | Some("ssa") => return ass::parse(text),
        _ => {}
    }
    // A BOM or leading whitespace does not affect the `[Events]`/`-->`
    // markers, and `vtt::is_webvtt` strips a BOM itself, so no normalisation
    // is needed before sniffing -- every parser below does its own.
    if vtt::is_webvtt(text) {
        vtt::parse(text)
    } else if text.contains("[Events]") {
        ass::parse(text)
    } else if text.contains("-->") {
        srt::parse(text)
    } else {
        Err(format!(
            "could not tell what subtitle format this is{}: it has no WebVTT 'WEBVTT' \
             signature, no ASS '[Events]' section and no SubRip '-->' timing line. \
             Supported formats are SubRip (.srt), WebVTT (.vtt) and Advanced SubStation \
             Alpha (.ass/.ssa)",
            match extension {
                Some(e) => format!(" from its '.{e}' extension or its contents"),
                None => " (it has no extension to go by)".to_string(),
            }
        ))
    }
}

/// A single subtitle line, active from `start_s` (inclusive) to `end_s`
/// (exclusive).
#[derive(Clone, Debug, PartialEq)]
pub struct Cue {
    pub start_s: f64,
    pub end_s: f64,
    pub text: String,
}

/// A parsed, timed subtitle track.
///
/// Cues are kept sorted by `start_s` so [`Subtitles::at`] can scan
/// deterministically. Overlaps and gaps are both expected: real files
/// overlap a sign translation with dialogue, and most of a file's runtime
/// has no subtitle at all.
#[derive(Clone, Debug, Default)]
pub struct Subtitles {
    cues: Vec<Cue>,
}

impl Subtitles {
    /// Builds a track from unordered cues, sorting by `start_s` (ties broken
    /// by `end_s`) so [`Subtitles::at`] can rely on ordering.
    pub fn new(mut cues: Vec<Cue>) -> Self {
        cues.sort_by(|a, b| {
            a.start_s
                .partial_cmp(&b.start_s)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.end_s.partial_cmp(&b.end_s).unwrap_or(std::cmp::Ordering::Equal))
        });
        Self { cues }
    }

    pub fn len(&self) -> usize {
        self.cues.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cues.is_empty()
    }

    /// The subtitle text active at time `t_s`, or `""` if none covers it.
    ///
    /// A cue covers `t_s` when `start_s <= t_s < end_s` -- the end is
    /// exclusive so that two adjacent cues sharing a boundary instant (one
    /// ending exactly where the next begins) do not both match at that
    /// instant.
    ///
    /// **Overlap resolution:** when more than one cue covers `t_s` (e.g. a
    /// sign translation running over dialogue), the cue with the greatest
    /// `start_s` -- the most recently begun -- wins. This is a deliberate
    /// simplification: merging overlapping lines would need layout rules
    /// (which line goes on top, how to fit both) that this module does not
    /// have. A linear scan is fine here because this is the RANDOM-ACCESS
    /// entry point, called once per query; [`Subtitles::per_frame`] walks
    /// time forwards and uses a sweep instead, precisely so that a large
    /// track does not cost `frames * cues`.
    pub fn at(&self, t_s: f64) -> &str {
        self.cues
            .iter()
            .filter(|c| c.start_s <= t_s && t_s < c.end_s)
            .max_by(|a, b| a.start_s.partial_cmp(&b.start_s).unwrap_or(std::cmp::Ordering::Equal))
            .map(|c| c.text.as_str())
            .unwrap_or("")
    }

    /// Renders one subtitle string per output frame, `frames` long, starting
    /// at clip time `start_s` and advancing by `1.0 / fps` per frame.
    ///
    /// `fps` must be finite and positive, checked with `!fps.is_finite() ||
    /// fps <= 0.0` rather than `fps <= 0.0` alone -- the latter is false for
    /// `NaN` and lets it slip through to produce a garbage array of
    /// timestamps. `audio::stft::hop_for` shipped exactly that bug once
    /// already in this project.
    ///
    /// Returns `Result` rather than panicking: this is reached from the GUI's
    /// render thread as well as the CLI, and an `Err` surfaces there as a
    /// message while a panic takes the window with it. Returning an all-empty
    /// track instead would be worse than either -- it renders as a video with
    /// no dialogue, which is indistinguishable from a correct render of a
    /// silent scene.
    ///
    /// **A SWEEP, not `frames` calls to [`Subtitles::at`].** Frame times only
    /// ever increase, so a cue whose `end_s` has passed can never match
    /// again: cues enter a max-heap keyed on `start_s` as their start is
    /// reached, and expire off the top once passed. That makes this
    /// `O((frames + cues) log cues)` instead of `O(frames * cues)`. The naive
    /// form was documented as safe because a track has "at most a few
    /// thousand cues", but word-level ASR caption exports routinely run 50k
    /// to 200k, and 100k cues against a 60k-frame render is ~6e9 iterations
    /// run synchronously before any progress is reported -- indistinguishable
    /// from a hang.
    ///
    /// The result is identical to calling `at` per frame, cue for cue; the
    /// `cues_agree_with_at_on_a_dense_overlapping_track` test below pins that
    /// against a generated track.
    pub fn per_frame(
        &self,
        start_s: f64,
        fps: f64,
        frames: usize,
    ) -> Result<Vec<String>, String> {
        if !fps.is_finite() || fps <= 0.0 {
            return Err(format!(
                "subtitle timing needs a positive finite fps, got {fps}"
            ));
        }

        // Only cues with FINITE bounds can take part in the sweep: its whole
        // argument rests on "expired at this t means expired at every later
        // t", which needs a real ordering on both bounds. Every parser in
        // this module now rejects a non-finite timestamp, so `odd` is empty
        // for any track read from a file -- it exists so a hand-built
        // `Subtitles` cannot quietly get a different answer here than from
        // `at`.
        let mut ordered: Vec<usize> = Vec::with_capacity(self.cues.len());
        let mut odd: Vec<usize> = Vec::new();
        for (i, c) in self.cues.iter().enumerate() {
            if c.start_s.is_finite() && c.end_s.is_finite() {
                ordered.push(i);
            } else {
                odd.push(i);
            }
        }
        // `Subtitles::new` already sorted, but its comparator falls back to
        // `Equal` on a NaN and is therefore not a total order when one is
        // present -- which can leave the vec genuinely unsorted. The sweep
        // needs a real ordering, so the finite subset is re-sorted rather
        // than trusted.
        ordered.sort_by(|&a, &b| self.cues[a].start_s.total_cmp(&self.cues[b].start_s));

        let mut out = Vec::with_capacity(frames);
        let mut next = 0usize;
        // Max-heap on (start_s, cue index): `at` resolves overlaps to the
        // greatest `start_s`, and `max_by` breaks a tie by taking the LAST
        // equal element in cue order, i.e. the greatest index.
        let mut live: std::collections::BinaryHeap<(u64, usize)> =
            std::collections::BinaryHeap::new();

        for i in 0..frames {
            let t = start_s + i as f64 / fps;

            while next < ordered.len() && self.cues[ordered[next]].start_s <= t {
                let idx = ordered[next];
                live.push((order_key(self.cues[idx].start_s), idx));
                next += 1;
            }
            // The heap's top is the greatest (start_s, index) in it, so
            // popping expired tops leaves a top that is both live and
            // maximal. Anything expired further down cannot outrank it.
            while let Some(&(_, idx)) = live.peek() {
                if self.cues[idx].end_s <= t {
                    live.pop();
                } else {
                    break;
                }
            }

            let mut best: Option<(f64, usize)> =
                live.peek().map(|&(_, idx)| (self.cues[idx].start_s, idx));
            for &idx in &odd {
                let c = &self.cues[idx];
                if c.start_s <= t && t < c.end_s {
                    let better = match best {
                        None => true,
                        Some((s, j)) => (c.start_s, idx) > (s, j),
                    };
                    if better {
                        best = Some((c.start_s, idx));
                    }
                }
            }

            out.push(best.map(|(_, idx)| self.cues[idx].text.clone()).unwrap_or_default());
        }
        Ok(out)
    }
}

/// Maps a non-NaN `f64` to a `u64` whose unsigned ordering matches the
/// float's numeric ordering, so a [`std::collections::BinaryHeap`] can be
/// keyed on a timestamp without a wrapper type. Only ever called with finite
/// values (see [`Subtitles::per_frame`]).
fn order_key(x: f64) -> u64 {
    let bits = x.to_bits();
    if bits & (1 << 63) != 0 {
        !bits // negative: flip every bit, so more-negative sorts lower
    } else {
        bits | (1 << 63) // positive: sits above every negative
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(start_s: f64, end_s: f64, text: &str) -> Cue {
        Cue { start_s, end_s, text: text.to_string() }
    }

    #[test]
    fn at_returns_the_covering_cue_and_empty_in_the_gaps() {
        let s = Subtitles::new(vec![cue(1.0, 2.0, "first"), cue(3.0, 4.0, "second")]);
        assert_eq!(s.at(0.5), "", "before the first cue");
        assert_eq!(s.at(1.0), "first", "inclusive at the start");
        assert_eq!(s.at(1.999), "first");
        assert_eq!(s.at(2.5), "", "in the gap");
        assert_eq!(s.at(3.5), "second");
        assert_eq!(s.at(9.0), "", "after the last cue");
    }

    #[test]
    fn the_end_of_a_cue_is_exclusive_so_adjacent_cues_do_not_double_show() {
        let s = Subtitles::new(vec![cue(1.0, 2.0, "a"), cue(2.0, 3.0, "b")]);
        assert_eq!(s.at(2.0), "b", "the later cue owns the boundary instant");
    }

    #[test]
    fn overlapping_cues_resolve_to_the_latest_started() {
        let s = Subtitles::new(vec![cue(1.0, 5.0, "long"), cue(2.0, 3.0, "short")]);
        assert_eq!(s.at(1.5), "long");
        assert_eq!(s.at(2.5), "short", "the more recently begun cue wins");
        assert_eq!(s.at(4.0), "long", "and reverts when it ends");
    }

    #[test]
    fn cues_are_sorted_on_construction_so_input_order_does_not_matter() {
        let a = Subtitles::new(vec![cue(3.0, 4.0, "second"), cue(1.0, 2.0, "first")]);
        assert_eq!(a.at(1.5), "first");
        assert_eq!(a.at(3.5), "second");
    }

    #[test]
    fn per_frame_maps_each_frame_to_its_timestamp() {
        // 4 fps: frames land at 0.0, 0.25, 0.5, 0.75.
        let s = Subtitles::new(vec![cue(0.5, 1.0, "hi")]);
        assert_eq!(s.per_frame(0.0, 4.0, 4).unwrap(), vec!["", "", "hi", "hi"]);
    }

    #[test]
    fn per_frame_honours_the_clip_start_offset() {
        // Starting 10 s in, frame 0 is at t=10.0.
        let s = Subtitles::new(vec![cue(10.0, 11.0, "hi")]);
        assert_eq!(s.per_frame(10.0, 1.0, 2).unwrap(), vec!["hi", ""]);
    }

    #[test]
    fn per_frame_of_an_empty_track_is_all_empty_not_short() {
        let s = Subtitles::new(vec![]);
        assert_eq!(s.per_frame(0.0, 12.0, 3).unwrap(), vec!["", "", ""]);
    }

    /// `per_frame` is a sweep rather than `frames` calls to `at`; this pins
    /// the two to the same answer over a track built to hit every case the
    /// sweep reasons about -- heavy overlap, nesting, shared boundaries,
    /// zero-length cues, gaps, and cues that expire while an earlier, longer
    /// one is still running (which the heap must fall back to).
    #[test]
    fn per_frame_agrees_with_at_on_a_dense_overlapping_track() {
        let mut cues = Vec::new();
        // A deterministic pseudo-random spread -- no dev-dependency needed,
        // and a fixed seed means a failure is reproducible.
        let mut seed = 0x5eed_1234u64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };
        for i in 0..400 {
            let start = rand() * 40.0;
            let len = rand() * 3.0; // sometimes 0.0-ish: zero-length cues
            cues.push(cue(start, start + len, &format!("cue{i}")));
        }
        // Deliberate exact-boundary and full-containment cases.
        cues.push(cue(10.0, 20.0, "long"));
        cues.push(cue(12.0, 13.0, "nested"));
        cues.push(cue(13.0, 13.0, "zero"));
        cues.push(cue(20.0, 21.0, "adjacent"));
        let s = Subtitles::new(cues);

        let fps = 30.0;
        let frames = 1500;
        let swept = s.per_frame(0.0, fps, frames).unwrap();
        let naive: Vec<String> =
            (0..frames).map(|i| s.at(i as f64 / fps).to_string()).collect();
        assert_eq!(swept, naive, "the sweep must agree with `at`, frame for frame");
    }

    #[test]
    fn per_frame_still_agrees_with_at_when_a_cue_has_infinite_bounds() {
        // Every parser now refuses these, but `Subtitles::new` is public, and
        // the sweep must not answer differently from `at` for a hand-built
        // track. A (-inf, +inf) cue genuinely covers every finite instant.
        let s = Subtitles::new(vec![
            cue(f64::NEG_INFINITY, f64::INFINITY, "always"),
            cue(1.0, 2.0, "sometimes"),
            cue(f64::NAN, 5.0, "never"),
        ]);
        let swept = s.per_frame(0.0, 2.0, 8).unwrap();
        let naive: Vec<String> = (0..8).map(|i| s.at(i as f64 / 2.0).to_string()).collect();
        assert_eq!(swept, naive);
        assert_eq!(swept[2], "sometimes", "t=1.0 is inside the finite cue");
        assert_eq!(swept[0], "always", "t=0.0 is only inside the infinite one");
    }

    // --- parse_auto ------------------------------------------------------

    const SRT: &str = "1\n00:00:01,000 --> 00:00:02,000\nHello\n";
    const ASS: &str = "[Events]\nFormat: Start, End, Text\n\
                       Dialogue: 0:00:01.00,0:00:02.00,Hello\n";
    const VTT: &str = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\nHello\n";

    #[test]
    fn a_known_extension_picks_the_parser() {
        assert_eq!(parse_auto(SRT, Some("srt")).unwrap().at(1.5), "Hello");
        assert_eq!(parse_auto(ASS, Some("ass")).unwrap().at(1.5), "Hello");
        assert_eq!(parse_auto(ASS, Some("ssa")).unwrap().at(1.5), "Hello");
    }

    #[test]
    fn the_extension_match_is_case_insensitive() {
        assert_eq!(parse_auto(SRT, Some("SRT")).unwrap().at(1.5), "Hello");
        assert_eq!(parse_auto(ASS, Some("ASS")).unwrap().at(1.5), "Hello");
    }

    #[test]
    fn a_vtt_extension_reaches_a_parser_that_can_actually_read_one() {
        // `.vtt` and `.webvtt` used to be routed at `srt::parse`, which
        // cannot read a single real WebVTT file: the `WEBVTT` header block
        // has no `-->` and the fraction separator is `.`, not `,`. The GUI's
        // own file filter offers `.vtt`, so every file a user picked through
        // it was rejected with a SubRip parse error.
        assert_eq!(parse_auto(VTT, Some("vtt")).unwrap().at(1.5), "Hello");
        assert_eq!(parse_auto(VTT, Some("webvtt")).unwrap().at(1.5), "Hello");
        assert_eq!(parse_auto(VTT, Some("VTT")).unwrap().at(1.5), "Hello");
    }

    #[test]
    fn webvtt_is_sniffed_by_its_signature_before_the_subrip_arrow() {
        // WebVTT contains `-->` too, so an arrow-first sniff would hand it to
        // the SubRip parser and fail on the header block.
        assert_eq!(parse_auto(VTT, Some("txt")).unwrap().at(1.5), "Hello");
        assert_eq!(parse_auto(VTT, None).unwrap().at(1.5), "Hello");
    }

    #[test]
    fn an_unknown_extension_falls_back_to_sniffing_the_content() {
        // The classic case: a track extracted to `.txt`, or a file with no
        // extension at all.
        assert_eq!(parse_auto(SRT, Some("txt")).unwrap().at(1.5), "Hello");
        assert_eq!(parse_auto(ASS, Some("txt")).unwrap().at(1.5), "Hello");
        assert_eq!(parse_auto(SRT, None).unwrap().at(1.5), "Hello");
        assert_eq!(parse_auto(ASS, None).unwrap().at(1.5), "Hello");
    }

    #[test]
    fn a_file_that_is_neither_format_errors_rather_than_parsing_to_nothing() {
        // An all-empty track is indistinguishable in game from a correct
        // render of a scene where nobody speaks, so this must never be one.
        let err = parse_auto("just some prose\nwith no timings at all\n", Some("txt"))
            .expect_err("must reject");
        assert!(err.contains("[Events]"), "names the ASS marker: {err}");
        assert!(err.contains("-->"), "names the SubRip marker: {err}");
    }

    #[test]
    fn an_empty_srt_is_still_an_empty_track_not_an_error() {
        // The extension settled the format; the parser's own "no cues is
        // valid" stance applies from there.
        assert!(parse_auto("", Some("srt")).unwrap().is_empty());
    }

    #[test]
    fn per_frame_rejects_a_nonsense_fps_rather_than_timing_against_it() {
        let s = Subtitles::new(vec![cue(1.0, 2.0, "hi")]);
        // NaN is the one that matters: `fps <= 0.0` alone is FALSE for NaN, so
        // a guard written that way lets it through and every timestamp becomes
        // NaN -- which matches no cue, yielding a silent all-empty track that
        // looks exactly like a correct render of a scene with no dialogue.
        // `audio::stft::hop_for` shipped this bug once already.
        for bad in [f64::NAN, 0.0, -12.0, f64::INFINITY] {
            let err = s
                .per_frame(0.0, bad, 3)
                .expect_err(&format!("fps {bad} must be rejected"));
            assert!(err.contains("fps"), "names the offending quantity: {err}");
        }
    }
}
