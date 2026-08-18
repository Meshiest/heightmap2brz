pub mod backend;
pub mod bands;
pub mod cost;
#[cfg(not(target_arch = "wasm32"))]
pub mod ffmpeg_src;
pub mod percussion;
pub mod presets;
pub mod source;
pub mod speakers;
pub mod stft;
pub mod symphonia_src;
pub mod track;
pub mod voices;

/// Which renderer an audio build uses -- the `--audio-mode` flag, as a type.
///
/// Two renderers, one analysis front end.
/// [`Bank`](Self::Bank) puts every speaker on a fixed equal-tempered pitch and
/// writes only volumes; [`Voice`](Self::Voice) builds `--max-voices` speakers
/// that follow spectral peaks and writes both pitch and volume.
///
/// A shared type rather than a `bool` in each front end, because the two modes
/// disagree about what several flags mean -- `--max-voices` most sharply (an
/// upper bound on bands sounding vs the number of speakers built, with 0 legal
/// in one and an error in the other) -- and every place that has to branch on
/// that should be branching on the same thing the CLI parsed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AudioMode {
    /// A fixed bank of pitched speakers, volume-modulated. Best for speech
    /// and broadband material.
    #[default]
    Bank,
    /// `--max-voices` speakers that track spectral peaks, changing pitch and
    /// volume every frame -- no band grid, so no tuning error. Best for tonal
    /// material such as piano.
    Voice,
}

impl AudioMode {
    pub const ALL: [AudioMode; 2] = [AudioMode::Bank, AudioMode::Voice];

    /// The `--audio-mode` spelling. Round-trips with [`Self::parse`].
    pub const fn flag(&self) -> &'static str {
        match self {
            AudioMode::Bank => "bank",
            AudioMode::Voice => "voice",
        }
    }

    /// Display name for UIs.
    ///
    /// These describe what each mode does rather than how it is implemented:
    /// `Bank` gives every speaker one fixed pitch and moves only its volume,
    /// and `Voice` re-pitches a handful of speakers every frame to track
    /// spectral peaks. The `--audio-mode` spellings ([`Self::flag`]) are
    /// unaffected -- `bank` and `voice` remain the values the CLI parses.
    pub const fn name(&self) -> &'static str {
        match self {
            AudioMode::Bank => "Pitch-Per-Speaker",
            AudioMode::Voice => "Pitch Switching",
        }
    }

    /// Parse a `--audio-mode` value. The error is the CLI's own wording, so
    /// the flag reads the same however it is reached.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "bank" => Ok(AudioMode::Bank),
            "voice" => Ok(AudioMode::Voice),
            other => Err(format!("unsupported --audio-mode '{other}' (bank, voice)")),
        }
    }

    /// Whether this mode reads the equal-tempered band grid at all.
    ///
    /// `--bands`, `--subdiv`, `--noise-bands` and `--peak-gate` are grid
    /// settings: voice mode has no grid -- that is the entire reason it exists
    /// -- so it ignores every one of them, and the CLI says so flag by flag
    /// rather than dropping them silently.
    pub const fn uses_band_grid(&self) -> bool {
        matches!(self, AudioMode::Bank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flag spelling and the parser must be inverses, or the CLI and a UI
    /// that offers the same two choices can drift apart.
    #[test]
    fn every_mode_round_trips_through_its_flag() {
        for m in AudioMode::ALL {
            assert_eq!(AudioMode::parse(m.flag()), Ok(m));
        }
    }

    /// The error text is the one the CLI has always printed; it is asserted
    /// here so a refactor cannot quietly reword it.
    #[test]
    fn an_unknown_mode_names_both_valid_spellings() {
        assert_eq!(
            AudioMode::parse("banks"),
            Err("unsupported --audio-mode 'banks' (bank, voice)".to_string())
        );
    }

    #[test]
    fn only_bank_mode_reads_the_band_grid() {
        assert!(AudioMode::Bank.uses_band_grid());
        assert!(!AudioMode::Voice.uses_band_grid());
    }
}
