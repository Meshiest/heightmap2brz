//! Per-content-type audio presets -- settings found by ear, not derived.
//!
//! Every number in this file came out of a listening session: dozens of
//! renders of real material, one flag varying at a time, judged in game.
//! None of it follows from the analysis, and none of it can be recovered by
//! reasoning about the DSP. See [`AudioPreset`]'s per-variant docs for the
//! specific reversals.
//!
//! They live here, in the library, rather than in the GUI pane that shows
//! them, for three reasons: they are audio knowledge and not UI state, they
//! are worth testing, and a scratch file is not a place to keep something
//! that took a session to find.
//!
//! A preset sets exactly the five fields the listening session varied --
//! [`AudioOptions::window`], [`AudioOptions::bands`],
//! [`AudioOptions::max_voices`], [`AudioOptions::peak_gate`] and
//! [`AudioOptions::release_ms`] -- plus the two settings that came out the
//! same on every source ([`AudioOptions::noise_bands`] `= 0` and
//! [`AudioOptions::leveling`] `= 1.0`). It touches nothing else: `--audio-fps`,
//! `--gain`, `--subdiv`, `--attack`, the attenuation radii and the frame caps
//! were not what the session was listening for.
//!
//! Applying a preset is therefore a seed, never a lock. Everything stays
//! editable afterwards, exactly like [`crate::text::FontPreset`].
//!
//! [`AudioOptions::bands`] and [`AudioOptions::peak_gate`] are bank-mode
//! only (`--audio-mode bank`); voice mode has no band grid and its own
//! prominence gate. `window`, `max_voices` and `release_ms` are meaningful in
//! both.
use super::track::AudioOptions;

/// The five settings a preset carries, plus the display name.
///
/// A plain struct rather than five methods on the enum: the whole point is
/// that these five move together -- `--window 2048` with a 150 ms release is
/// not "speech settings with one thing changed", it is a combination nobody
/// listened to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PresetValues {
    /// STFT window size. The single biggest lever: it trades pitch
    /// resolution against transient smearing, and the right answer follows
    /// the material rather than the genre.
    pub window: usize,
    /// Total speakers, or `None` for "every step the pitch range holds"
    /// (79 tonal at the default `--subdiv 12`). See [`AudioOptions::bands`].
    pub bands: Option<usize>,
    /// `--max-voices`.
    pub max_voices: usize,
    /// `--peak-gate`. Bank mode only.
    pub peak_gate: f32,
    /// `--release`, in milliseconds.
    pub release_ms: f32,
}

/// A content type someone actually sat and listened to.
///
/// The ordering is the quality gradient the session found, which tracks how
/// sinusoidal the source is rather than its genre: music box -> piano ->
/// guitar -> chiptune -> speech -> orchestral -> pop mix -> sound effects. A
/// preset exists for each point on it that behaved distinctly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AudioPreset {
    /// No preset: whatever [`AudioOptions::default`] gives, which is what a
    /// fresh session starts on. Selecting this again applies the defaults
    /// (including `--noise-bands 0` and `--leveling 1.0`), so it doubles as
    /// a reset.
    #[default]
    Default,
    /// Piano, music box, guitar -- long window (16384) for pitch resolution,
    /// 150 ms release so struck notes ring.
    Tonal,
    /// Chiptune -- same window and release as [`Self::Tonal`], but the full
    /// band span: narrowing the range crushes chiptune bass rather than
    /// cleaning it up.
    Chiptune,
    /// Speech -- short window (2048, formants move too fast for a long
    /// window) and a 45 ms release. The release is deliberately not
    /// optimised toward the longer run length used elsewhere: a phoneme is
    /// 50-100 ms, and a longer release smears across it.
    Speech,
    /// The catch-all -- middling window, the tightest peak gate of any
    /// preset (2.5), and a 60 ms release at the long end of the speech
    /// range.
    General,
}

impl AudioPreset {
    pub const ALL: [AudioPreset; 5] = [
        AudioPreset::Default,
        AudioPreset::Tonal,
        AudioPreset::Chiptune,
        AudioPreset::Speech,
        AudioPreset::General,
    ];

    /// Display name for UIs.
    pub fn name(&self) -> &'static str {
        match self {
            AudioPreset::Default => "Default",
            AudioPreset::Tonal => "Piano / music box / guitar",
            AudioPreset::Chiptune => "Chiptune",
            AudioPreset::Speech => "Speech",
            AudioPreset::General => "General",
        }
    }

    /// One line of "what this is for and what it does", for a tooltip.
    pub fn hint(&self) -> &'static str {
        match self {
            AudioPreset::Default => {
                "The module defaults: window 4096, every band the pitch range holds, 12 \
                 voices, peak gate 1.5, release 150 ms. Selecting this doubles as a reset."
            }
            AudioPreset::Tonal => {
                "Sustained, near-sinusoidal material -- the best case for this renderer. \
                 Long window for pitch resolution, long release so struck notes ring. \
                 Narrow Bands toward 48 for a music box or solo piano; 60 keeps a \
                 guitar's low E in range."
            }
            AudioPreset::Chiptune => {
                "Square/triangle chip music. Same long window and release as the tonal \
                 preset, but the FULL band span -- narrowing the range crushes chiptune \
                 bass, unlike piano where it helps."
            }
            AudioPreset::Speech => {
                "Dialogue with no score under it. Short window (formants move), and a \
                 30-60 ms release because a phoneme is 50-100 ms long -- a 150 ms release \
                 smears straight across it and speech comes back 'disembodied'. Low gate."
            }
            AudioPreset::General => {
                "The catch-all: start here when the material is neither purely tonal nor \
                 purely speech. Middling window, the tightest peak gate of any preset (far \
                 more competing energy per frame than speech or music alone), and a release \
                 at the long end of speech."
            }
        }
    }

    /// The settings themselves. See the variant docs for where each came from.
    pub fn values(&self) -> PresetValues {
        let d = AudioOptions::default();
        match self {
            AudioPreset::Default => PresetValues {
                window: d.window,
                bands: d.bands,
                max_voices: d.max_voices,
                peak_gate: d.peak_gate,
                release_ms: d.release_ms,
            },
            AudioPreset::Tonal => PresetValues {
                window: 16384,
                bands: Some(60),
                max_voices: 24,
                peak_gate: 1.5,
                release_ms: 150.0,
            },
            AudioPreset::Chiptune => PresetValues {
                window: 16384,
                bands: None,
                max_voices: 24,
                peak_gate: 1.5,
                release_ms: 150.0,
            },
            AudioPreset::Speech => PresetValues {
                window: 2048,
                bands: Some(72),
                max_voices: 16,
                peak_gate: 1.2,
                release_ms: 45.0,
            },
            AudioPreset::General => PresetValues {
                window: 8192,
                bands: Some(60),
                max_voices: 20,
                peak_gate: 2.5,
                release_ms: 60.0,
            },
        }
    }

    /// Seed `opts` with this preset.
    ///
    /// Writes the five listened-for fields plus the two that came out the
    /// same on every source, and touches nothing else -- see the module doc.
    /// A caller is expected to leave every one of them editable afterwards.
    pub fn apply(&self, opts: &mut AudioOptions) {
        let v = self.values();
        opts.window = v.window;
        opts.bands = v.bands;
        opts.max_voices = v.max_voices;
        opts.peak_gate = v.peak_gate;
        opts.release_ms = v.release_ms;
        // Right on every source tried, which is why they are also the
        // module defaults rather than a per-preset value -- see
        // `AudioOptions::noise_bands` and `AudioOptions::leveling`.
        opts.noise_bands = 0;
        opts.leveling = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These numbers are pinned; a change to any of them must be a deliberate
    /// edit to this test too, not a refactor.
    #[test]
    fn the_recorded_settings_are_exactly_what_was_heard() {
        assert_eq!(
            AudioPreset::Tonal.values(),
            PresetValues {
                window: 16384,
                bands: Some(60),
                max_voices: 24,
                peak_gate: 1.5,
                release_ms: 150.0
            }
        );
        assert_eq!(
            AudioPreset::Chiptune.values(),
            PresetValues {
                window: 16384,
                bands: None,
                max_voices: 24,
                peak_gate: 1.5,
                release_ms: 150.0
            }
        );
        assert_eq!(
            AudioPreset::Speech.values(),
            PresetValues {
                window: 2048,
                bands: Some(72),
                max_voices: 16,
                peak_gate: 1.2,
                release_ms: 45.0
            }
        );
        assert_eq!(
            AudioPreset::General.values(),
            PresetValues {
                window: 8192,
                bands: Some(60),
                max_voices: 20,
                peak_gate: 2.5,
                release_ms: 60.0
            }
        );
    }

    /// Every preset must zero `noise_bands` and set `leveling` to 1.0,
    /// regardless of what the caller had set before.
    #[test]
    fn every_preset_zeroes_the_noise_bands_and_levels_fully() {
        for p in AudioPreset::ALL {
            let mut opts = AudioOptions { noise_bands: 2, leveling: 0.0, ..Default::default() };
            p.apply(&mut opts);
            assert_eq!(opts.noise_bands, 0, "{}", p.name());
            assert_eq!(opts.leveling, 1.0, "{}", p.name());
        }
    }

    /// The speech release must stay inside the 30-60 ms range that was
    /// actually heard as correct, and below the default.
    #[test]
    fn the_speech_release_stays_in_the_range_that_was_heard() {
        let r = AudioPreset::Speech.values().release_ms;
        assert!(
            (30.0..=60.0).contains(&r),
            "speech release {r} ms is outside the 30-60 ms range that was heard as correct"
        );
        assert!(
            r < super::super::track::DEFAULT_RELEASE_MS,
            "the whole point is that speech wants a SHORTER release than the default"
        );
    }

    /// `Default` really is the module default, so the dropdown's first entry
    /// is a genuine reset rather than a sixth set of numbers that happens to
    /// look like the defaults.
    #[test]
    fn the_default_preset_is_the_module_default() {
        let mut opts = AudioOptions {
            window: 999,
            bands: Some(3),
            max_voices: 99,
            peak_gate: 9.0,
            release_ms: 900.0,
            ..Default::default()
        };
        AudioPreset::Default.apply(&mut opts);
        let d = AudioOptions::default();
        assert_eq!(opts.window, d.window);
        assert_eq!(opts.bands, d.bands);
        assert_eq!(opts.max_voices, d.max_voices);
        assert_eq!(opts.peak_gate, d.peak_gate);
        assert_eq!(opts.release_ms, d.release_ms);
    }

    /// `apply` must write every field of `values()`, for every variant, not
    /// just constants that happen to already be right.
    #[test]
    fn every_preset_applies_exactly_its_own_recorded_values() {
        for p in AudioPreset::ALL {
            let v = p.values();
            let name = p.name();
            // Seeded with values no preset uses, so "written" cannot be
            // mistaken for "happened to already be right".
            let mut opts = AudioOptions {
                window: 512,
                bands: Some(3),
                max_voices: 99,
                peak_gate: 9.0,
                release_ms: 900.0,
                ..Default::default()
            };
            p.apply(&mut opts);
            assert_eq!(opts.window, v.window, "{name}: window");
            assert_eq!(opts.bands, v.bands, "{name}: bands");
            assert_eq!(opts.max_voices, v.max_voices, "{name}: max voices");
            assert_eq!(opts.peak_gate, v.peak_gate, "{name}: peak gate");
            assert_eq!(opts.release_ms, v.release_ms, "{name}: release");
        }
    }

    /// A preset must not touch a field it never measured, across every
    /// variant, not just [`AudioPreset::General`].
    #[test]
    fn no_preset_touches_a_field_it_never_measured() {
        for p in AudioPreset::ALL {
            let before = AudioOptions {
                fps: 24.0,
                subdiv: 24,
                gain: 0.5,
                attack_ms: 3.0,
                voice_release_ms: 123.0,
                pitch_snap_cents: 20.0,
                floor_db: -42.0,
                inner_radius: 111.0,
                max_distance: 2222.0,
                max_frames: 4242,
                ..Default::default()
            };
            let mut after = before;
            p.apply(&mut after);
            let name = p.name();
            assert_eq!(after.fps, before.fps, "{name}: fps");
            assert_eq!(after.subdiv, before.subdiv, "{name}: subdiv");
            assert_eq!(after.gain, before.gain, "{name}: gain");
            assert_eq!(after.attack_ms, before.attack_ms, "{name}: attack");
            assert_eq!(after.voice_release_ms, before.voice_release_ms, "{name}: voice release");
            assert_eq!(after.pitch_snap_cents, before.pitch_snap_cents, "{name}: pitch snap");
            assert_eq!(after.floor_db, before.floor_db, "{name}: floor");
            assert_eq!(after.inner_radius, before.inner_radius, "{name}: inner radius");
            assert_eq!(after.max_distance, before.max_distance, "{name}: max distance");
            assert_eq!(after.max_frames, before.max_frames, "{name}: max frames");
        }
    }

    /// Every preset's window must be a power of two: the STFT is a radix FFT,
    /// and the pane offers powers of two only. A preset the pane's own
    /// dropdown could not represent would be unreachable by hand.
    #[test]
    fn every_preset_window_is_a_power_of_two() {
        for p in AudioPreset::ALL {
            let w = p.values().window;
            assert!(w.is_power_of_two(), "{}: window {w} is not a power of two", p.name());
        }
    }

    /// Names are what a user picks by, so they must be distinct and
    /// non-empty -- a dropdown with two identical entries is unusable.
    #[test]
    fn every_preset_has_a_distinct_name_and_hint() {
        let mut names: Vec<&str> = AudioPreset::ALL.iter().map(|p| p.name()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "preset names must be distinct");
        for p in AudioPreset::ALL {
            assert!(!p.name().is_empty());
            assert!(!p.hint().is_empty());
        }
    }
}
