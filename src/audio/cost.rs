//! Build-cost estimation for an audio render.
//!
//! The audio analogue of [`crate::anim::cost`], and it exists for the same
//! reason: gate count is what actually limits a build, so a UI has to show it
//! before the user commits to a render that may take minutes.
//!
//! # Why this takes `AudioOptions` and nothing loose
//!
//! [`estimate`] derives the speaker count the same way the renderer does --
//! [`band_plan`] in bank mode, `opts.max_voices` in voice mode -- from the
//! **same** [`AudioOptions`] value the render will be handed. A readout that
//! disagrees with its render has shipped twice in this codebase already (a
//! hard-coded `char_repeat`, and an uncounted subtitle track), and both times
//! the shape of the bug was a parameter list that could express the
//! difference. This one cannot: there is no way to ask it about a band count
//! the render would not build.
//!
//! It returns `Result`, not a `Cost`, for the same reason. A `--subdiv` of 14
//! or a voice-mode `--max-voices` of 0 is a render that CANNOT happen, and the
//! honest readout for it is the renderer's own refusal -- not a plausible
//! number for a build nothing will produce. Both checks are the renderer's
//! ([`check_subdiv`], [`check_voice_count`]), called here rather than
//! reimplemented.
//!
//! # What is exact and what is not
//!
//! Everything derived from the options -- speakers, streams, gates, wires,
//! bricks per bank -- is **exact**, and
//! [`tests::the_estimate_matches_a_real_bank_build`] pins it against a real
//! [`build_speaker_world`](crate::audio::speakers::build_speaker_world) rather
//! than against arithmetic. Audio has no equivalent of brick mode's culled
//! transparent pixel: every band gets its speaker whether it ever sounds or
//! not.
//!
//! `frames` is the caller's, and only the caller knows how good it is. The
//! CLI knows it exactly (it has already analysed); a UI showing a live
//! estimate before the render has one from the source's duration hint, which
//! is a hint. [`AudioCost::frames`] is echoed back unchanged so a caller can
//! label it however it obtained it.
use super::track::{AudioOptions, band_plan, check_subdiv};
use super::voices::check_voice_count;
use super::AudioMode;

/// The clock's own gates: timer -> multiply -> truncate -> modulo. See
/// [`crate::anim::clock::build_clock`], which both render modes reuse
/// unchanged.
const CLOCK_GATES: usize = 4;

/// The clock's own wires: 3 down its chain, 3 control pins
/// (Pause/Restart/Resume), the Rate pin, and the Done output.
const CLOCK_WIRES: usize = 8;

/// The pause-mute detector's gates: a `BufferTicks`, a `CompareNotEqual` and a
/// `Select`, shared by the whole bank (see
/// [`crate::audio::speakers::scaffold`]). Flat -- +3 regardless of speaker
/// count -- because the Select gates ONE master volume that every speaker's
/// multiply then reads.
const PAUSE_MUTE_GATES: usize = 3;

/// The pause-mute detector's wires: `Time -> BufferTicks.Input`,
/// `Time -> CompareNotEqual.InputA`, `BufferTicks.Output ->
/// CompareNotEqual.InputB`, `CompareNotEqual.bOutput -> Select.bSelectB`, and
/// the `Volume` pin into `Select.InputB`. The Select's OUTPUT reuses the wire
/// each per-speaker multiply already spent on its `InputB` (it now sources from
/// the Select instead of straight from the pin), so those are not new.
const PAUSE_MUTE_WIRES: usize = 5;

/// Every chip input/output pin an audio build carries: the clock's five
/// (Pause, Restart, Resume, Rate, Done) plus the four `scaffold` adds
/// (Inner Radius, Max Distance, Directional, Volume).
///
/// Pins are inner-grid bricks but are NOT gates, and [`AudioCost::gates`]
/// excludes them -- the same convention `anim`'s own tests count by
/// (`world.grids[0].1.len() - pins`).
pub const CHIP_PINS: usize = 9;

/// What a render costs to build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioCost {
    /// Emitter bricks: one per band in bank mode, one per voice in voice mode.
    pub speakers: usize,
    /// Per-frame data streams. One per speaker in bank mode (its volume); TWO
    /// per speaker in voice mode (its pitch and its volume), which is why a
    /// voice-mode build of the same speaker count costs roughly twice the
    /// array machinery.
    pub streams: usize,
    /// The frame count this estimate was made for, echoed back unchanged.
    pub frames: usize,
    /// Wire arrays each stream spills across at `opts.bank_size`.
    pub banks: usize,
    /// Wire-graph gates, excluding the chip's [`CHIP_PINS`] I/O pins.
    pub gates: usize,
    pub wires: usize,
    /// Main-grid bricks: the speakers plus the microchip shell. The gates
    /// live on the chip's INNER grid and are not counted here.
    pub bricks: usize,
    /// Numbers written into wire arrays: `streams * frames`. Audio writes
    /// numeric arrays directly -- no hex packing, no strings -- so this is the
    /// whole of the render's data, and it is what makes a long audio render
    /// cheap where a long video render is not.
    pub elements: usize,
}

impl AudioCost {
    /// Gate count as a fraction of the ~20 000 where the owner has measured
    /// frame drops starting. Audio builds sit far below it (a full 79-band
    /// single-bank bank is well under 300 gates), which is worth saying: the
    /// number that limits a video render does not limit this one.
    pub fn gate_load(&self) -> f32 {
        self.gates as f32 / 20_000.0
    }
}

/// Estimate the build cost of `frames` frames of audio under `opts`.
///
/// `Err` is the renderer's own refusal, verbatim -- see the module doc for why
/// this reports one instead of a number.
///
/// # The counts
///
/// With `S` speakers, `T` streams and `N` banks (`boundaries = N - 1`):
///
/// * **gates** = clock (4) + pause-mute detector (3: BufferTicks, CompareNotEqual,
///   Select) + change detector (1) + one master-volume multiply per speaker + an
///   `ArrayVar` and an `ArrayVar_Get` per stream per bank + an index subtract, a
///   comparator and a branch per boundary + a `Select` per stream per boundary.
/// * **wires** = clock (8) + pause-mute detector (5: Time into the buffer and
///   the comparator, the buffer into the comparator, the comparator into the
///   Select, and the Volume pin into the Select) + three attenuation pins fanned
///   out to every speaker (3S) + two per volume multiply (the gated master in,
///   the emitter out) + the detector feed + `ArrayVarRef`/`Index` per stream per
///   bank + the per-bank exec chain (one per stream per bank) + the final wire
///   from each stream into its target + four per boundary (subtract and
///   comparator `InputA`, branch `bCond` and `Exec`) + three per `Select`.
/// * **bricks** = one emitter per speaker plus the chip shell.
pub fn estimate(
    mode: AudioMode,
    frames: usize,
    opts: &AudioOptions,
) -> Result<AudioCost, String> {
    // The renderer's checks, called rather than copied. `check_subdiv` runs
    // even in voice mode, where the grid is unused, ONLY through `band_plan`
    // below -- voice mode genuinely ignores `--subdiv`, and refusing a value
    // it never reads would be a readout stricter than its render.
    //
    // The attenuation pair is checked here as well as in `check` so the READOUT
    // refuses it too: both builders reject an inverted or non-positive pair, so
    // a cost line for one would describe a build that cannot happen, next to a
    // Generate button that (rightly) refuses it.
    super::speakers::check_attenuation(opts)?;
    let (speakers, streams_per_speaker) = match mode {
        AudioMode::Bank => (band_plan(opts)?.len(), 1),
        AudioMode::Voice => {
            check_voice_count(opts.max_voices)?;
            (opts.max_voices, 2)
        }
    };
    // Not reachable through `AudioOptions::default()`, but a caller can build
    // one by hand, and every count below divides by it.
    if opts.bank_size == 0 {
        return Err("--bank-size must be at least 1".to_string());
    }

    let streams = speakers * streams_per_speaker;
    let banks = frames.div_ceil(opts.bank_size).max(1);
    let boundaries = banks - 1;

    Ok(AudioCost {
        speakers,
        streams,
        frames,
        banks,
        gates: CLOCK_GATES
            + PAUSE_MUTE_GATES
            + 1
            + speakers
            + 2 * streams * banks
            + 3 * boundaries
            + streams * boundaries,
        wires: CLOCK_WIRES
            + PAUSE_MUTE_WIRES
            + 3 * speakers
            + 2 * speakers
            + 1
            + 2 * streams * banks
            + streams * banks
            + streams
            + 4 * boundaries
            + 3 * streams * boundaries,
        bricks: speakers + 1,
        elements: streams * frames,
    })
}

/// The frame count `frames` seconds of audio analyses to under `opts`, or
/// `None` if the caller has no duration to work from.
///
/// The SAME derivation [`crate::audio::track::analyze`] uses for its progress
/// denominator -- `hop_for` then `frame_count_for`, capped at `--max-frames` --
/// so a live readout and the render it describes count frames the same way.
/// A UI that divided a duration by fps itself would be off by most of a
/// window at every setting, and by a lot of one at `--window 16384`.
pub fn frames_for_duration(
    duration_s: f64,
    sample_rate: u32,
    opts: &AudioOptions,
) -> Option<usize> {
    let hop = super::stft::hop_for(sample_rate, opts.fps).ok()?;
    Some(super::stft::frame_count_for(duration_s, sample_rate, opts.window, hop).min(opts.max_frames))
}

/// Whether a set of options can be rendered at all, without costing anything.
///
/// [`estimate`]'s refusals **plus the attenuation pair**, which is the point: a
/// UI that disables its Generate button on one rule and shows a cost under
/// another would let a render start that the estimate had already refused.
/// `subdiv` is checked in bank mode only, matching what the renderer reads.
///
/// [`check_attenuation`](crate::audio::speakers::check_attenuation) is called
/// here, and not only from the builders, because it was the one refusal that
/// arrived AFTER the whole analysis: an Inner Radius above Max Distance showed a
/// full, plausible cost, offered Generate, ran for minutes and only then failed.
/// Both builders still check it themselves -- this is the same guard reaching
/// the front end, not a guard moved away from the renderer.
pub fn check(mode: AudioMode, opts: &AudioOptions) -> Result<(), String> {
    super::speakers::check_attenuation(opts)?;
    match mode {
        AudioMode::Bank => {
            check_subdiv(opts.subdiv)?;
            band_plan(opts).map(|_| ())
        }
        AudioMode::Voice => check_voice_count(opts.max_voices),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::bands::BandPlan;
    use crate::audio::speakers::{build_speaker_world, build_voice_world};
    use crate::audio::track::VoiceTrack;
    use crate::audio::voices::{VoiceStats, VoiceStreams};

    fn opts(bank_size: usize) -> AudioOptions {
        AudioOptions { bank_size, ..Default::default() }
    }

    /// A bank-mode track of `bands` bands and `frames` frames, with the plan
    /// `opts` asks for so the built world and the estimate are describing the
    /// same render.
    fn track(o: &AudioOptions, frames: usize) -> VoiceTrack {
        let plan = band_plan(o).expect("valid plan");
        let n = plan.len();
        VoiceTrack {
            plan,
            volumes: vec![vec![0.5; frames]; n],
            fps: o.fps,
            frame_count: frames,
        }
    }

    fn streams(voices: usize, frames: usize, fps: f32) -> VoiceStreams {
        VoiceStreams {
            pitches: vec![vec![1.0; frames]; voices],
            volumes: vec![vec![0.5; frames]; voices],
            fps,
            frame_count: frames,
            stats: VoiceStats::default(),
        }
    }

    /// Gates are every inner-grid brick that is not one of the chip's I/O
    /// pins, counted the way `anim`'s own tests count them.
    fn built_gates(w: &brdb::World) -> usize {
        w.grids[0].1.len() - CHIP_PINS
    }

    /// **The property this module exists for.** The estimate is not checked
    /// against arithmetic -- it is checked against a world that was actually
    /// built, in both bank layouts a render can have (one bank, and spilled
    /// across three).
    #[test]
    fn the_estimate_matches_a_real_bank_build() {
        for (bank_size, frames) in [(64, 40), (64, 130), (10, 10)] {
            // A narrow span so the build stays small; the count is what is
            // being checked, not the pitches.
            let o = AudioOptions { bands: Some(6), ..opts(bank_size) };
            let t = track(&o, frames);
            let world = build_speaker_world(&t, &o).expect("build");
            let c = estimate(AudioMode::Bank, frames, &o).expect("estimate");

            assert_eq!(c.speakers, 6, "{bank_size}/{frames}: speakers");
            assert_eq!(c.streams, 6, "{bank_size}/{frames}: one stream per band");
            assert_eq!(c.gates, built_gates(&world), "{bank_size}/{frames}: gates");
            assert_eq!(c.wires, world.wires.len(), "{bank_size}/{frames}: wires");
            assert_eq!(c.bricks, world.bricks.len(), "{bank_size}/{frames}: bricks");
        }
    }

    /// The same, for the mode whose stream count is DOUBLE its speaker count.
    /// A single formula parameterised only by speakers would pass the bank
    /// test above and be wrong by a factor of two here.
    #[test]
    fn the_estimate_matches_a_real_voice_build() {
        for (bank_size, frames) in [(64, 40), (64, 130), (10, 10)] {
            let o = AudioOptions { max_voices: 5, ..opts(bank_size) };
            let s = streams(5, frames, o.fps);
            let world = build_voice_world(&s, &o).expect("build");
            let c = estimate(AudioMode::Voice, frames, &o).expect("estimate");

            assert_eq!(c.speakers, 5, "{bank_size}/{frames}: speakers");
            assert_eq!(c.streams, 10, "{bank_size}/{frames}: pitch AND volume per voice");
            assert_eq!(c.gates, built_gates(&world), "{bank_size}/{frames}: gates");
            assert_eq!(c.wires, world.wires.len(), "{bank_size}/{frames}: wires");
            assert_eq!(c.bricks, world.bricks.len(), "{bank_size}/{frames}: bricks");
        }
    }

    /// Bank mode's speaker count follows `--bands`/`--noise-bands`/`--subdiv`
    /// through the renderer's own [`band_plan`], so a default (unset `--bands`)
    /// estimate must describe the full span rather than some second default.
    #[test]
    fn the_speaker_count_is_the_band_plan_the_renderer_builds() {
        let o = opts(65_535);
        let c = estimate(AudioMode::Bank, 100, &o).expect("estimate");
        assert_eq!(c.speakers, BandPlan::full(0, o.subdiv).unwrap().len());
        assert_eq!(c.speakers, 79, "79 tonal bands at --subdiv 12");

        let quarter = AudioOptions { subdiv: 24, ..o };
        assert_eq!(
            estimate(AudioMode::Bank, 100, &quarter).unwrap().speakers,
            159,
            "159 at quarter-tones"
        );
    }

    /// A render that cannot happen must be reported as the renderer's refusal,
    /// not costed. `--subdiv 14` is the value that was heard as sharp in game.
    #[test]
    fn a_subdiv_off_the_semitone_grid_is_refused_rather_than_costed() {
        let err = estimate(AudioMode::Bank, 100, &AudioOptions { subdiv: 14, ..opts(64) })
            .expect_err("14 is not a multiple of 12");
        assert!(err.contains("--subdiv must be a multiple of 12"), "{err}");
        // The same words the renderer itself would print, since it is the
        // renderer's own check being called.
        assert_eq!(err, check_subdiv(14).unwrap_err());
    }

    /// The flag whose meaning changes with the mode: 0 is "every band" in bank
    /// mode and a build with no speakers in voice mode.
    #[test]
    fn max_voices_zero_is_legal_in_bank_mode_and_refused_in_voice_mode() {
        let o = AudioOptions { max_voices: 0, ..opts(64) };
        assert!(
            estimate(AudioMode::Bank, 100, &o).is_ok(),
            "0 = every band is a legal bank render"
        );
        let err = estimate(AudioMode::Voice, 100, &o).expect_err("0 speakers is not a build");
        assert!(err.contains("--max-voices must be at least 1"), "{err}");
    }

    /// Voice mode reads no band grid, so an estimate for it must not refuse a
    /// `--subdiv` its render would ignore -- a readout stricter than its render
    /// is as wrong as one looser.
    #[test]
    fn voice_mode_ignores_the_band_grid_flags_the_way_its_renderer_does() {
        let o = AudioOptions { subdiv: 14, max_voices: 8, ..opts(64) };
        let c = estimate(AudioMode::Voice, 100, &o).expect("voice mode never reads --subdiv");
        assert_eq!(c.speakers, 8);
    }

    /// `check` and `estimate` must refuse exactly the same option sets, or a
    /// UI could enable its Generate button on a render the readout had already
    /// called impossible.
    #[test]
    fn check_refuses_exactly_what_estimate_refuses() {
        let cases = [
            AudioOptions { subdiv: 14, ..opts(64) },
            AudioOptions { subdiv: 12, max_voices: 0, ..opts(64) },
            AudioOptions { bands: Some(9999), ..opts(64) },
            AudioOptions { noise_bands: 9, ..opts(64) },
            opts(64),
        ];
        for o in cases {
            for mode in AudioMode::ALL {
                assert_eq!(
                    check(mode, &o).is_err(),
                    estimate(mode, 100, &o).is_err(),
                    "{mode:?} disagreed about {o:?}"
                );
            }
        }
    }

    /// Banking is what makes a long track buildable at all, so the boundary
    /// has to be exact: 65 535 frames is one array, 65 536 is two.
    #[test]
    fn the_bank_boundary_is_exact() {
        let o = opts(65_535);
        assert_eq!(estimate(AudioMode::Bank, 65_535, &o).unwrap().banks, 1);
        assert_eq!(estimate(AudioMode::Bank, 65_536, &o).unwrap().banks, 2);
        // A zero-frame render still reports one bank rather than none -- the
        // renderer refuses it separately, with a better message.
        assert_eq!(estimate(AudioMode::Bank, 0, &o).unwrap().banks, 1);
    }

    /// Array elements are the render's whole data payload, and a voice-mode
    /// build writes two streams per speaker.
    #[test]
    fn elements_are_every_number_the_arrays_carry() {
        let o = AudioOptions { bands: Some(10), max_voices: 10, ..opts(65_535) };
        assert_eq!(estimate(AudioMode::Bank, 300, &o).unwrap().elements, 10 * 300);
        assert_eq!(estimate(AudioMode::Voice, 300, &o).unwrap().elements, 2 * 10 * 300);
    }

    /// The live readout's frame count must be the analyser's, not a
    /// duration-times-fps guess: an STFT emits nothing until it holds a full
    /// window, which at `--window 16384` is a third of a second of audio.
    #[test]
    fn the_frame_count_is_the_analysers_and_respects_max_frames() {
        let o = AudioOptions { window: 16384, fps: 30.0, ..opts(65_535) };
        let hop = crate::audio::stft::hop_for(48_000, o.fps).unwrap();
        assert_eq!(
            frames_for_duration(10.0, 48_000, &o),
            Some(crate::audio::stft::frame_count_for(10.0, 48_000, o.window, hop)),
        );
        let capped = AudioOptions { max_frames: 7, ..o };
        assert_eq!(frames_for_duration(600.0, 48_000, &capped), Some(7));
    }

    /// An fps that cannot produce a hop has no frame count, and saying so is
    /// better than reporting a fabricated one.
    #[test]
    fn an_impossible_rate_has_no_frame_count() {
        let o = AudioOptions { fps: 0.0, ..opts(64) };
        assert_eq!(frames_for_duration(10.0, 48_000, &o), None);
    }
}
