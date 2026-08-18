//! midi2brick: read a Standard MIDI File and play it back through a Brickadia
//! speaker world, one synth tone per track.
//!
//! A MIDI note maps to the audio emitter's `PitchMultiplier` exactly --
//! `2^((note - 69) / 12)`, note 69 = A4 = 440 Hz = multiplier 1.0 -- so unlike
//! audio2brick there is no FFT and no peak tracking: the pitches are the file's
//! own notes. The playback machinery (clock, per-frame speaker streams,
//! pause-mute, control buttons, spatialization pins) is reused wholesale from
//! [`crate::audio::speakers`]; this module only turns a `.mid` into the
//! per-speaker pitch/volume streams that builder consumes, and assigns each
//! track a tone.
//!
//! Front ends: the GUI offers a tone per discovered track plus an audible
//! preview ([`preview`]); the CLI offers a lite path (one tone for the whole
//! file, via [`ToneAssignment::Uniform`]) plus a MIDI info listing
//! ([`discover`]).
//!
//! # Playback model
//!
//! Playback is EVENT-BASED, not frame-based: each speaker stores its notes as
//! `(start, end, pitch, volume)` spans, and an in-chip circuit steps a runtime
//! "playhead" index through them by comparing the clock's `Time` against the
//! note ends (see `crate::audio::speakers::build_midi_event_world`). That is
//! tick-accurate and stores note count rather than a per-tick buffer, verified
//! in game via the playhead probe.
use crate::audio::track::SynthWave;

pub mod drums;
pub mod parse;
pub mod preview;
pub mod schedule;
pub mod timbre;

/// One note, with its start and end already resolved to absolute seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoteEvent {
    pub start_s: f64,
    pub end_s: f64,
    /// MIDI note number, 0..=127 (60 = middle C, 69 = A4).
    pub note: u8,
    /// MIDI velocity, 1..=127 (0 is a NoteOff and never reaches here).
    pub velocity: u8,
}

/// One percussion strike, from the percussion channel (10 / index 9).
///
/// A drum note has no pitch and no duration that matters: the note number
/// selects a SOUND, and a oneshot sample plays to its own end. Only the strike
/// time and velocity survive; the sound is resolved later through the drum
/// fold table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PercussionHit {
    pub start_s: f64,
    /// General MIDI percussion note (35..=81 for the standard kit).
    pub note: u8,
    /// MIDI velocity, 1..=127.
    pub velocity: u8,
}

/// A discovered instrument -- one `(track, channel)` pair that carries notes,
/// as shown in the GUI table and the CLI `--list`.
#[derive(Clone, Debug, PartialEq)]
pub struct Instrument {
    /// Track name meta-event, else the General MIDI program name, else
    /// `"Channel N"`.
    pub label: String,
    pub track: usize,
    pub channel: u8,
    /// The channel's General MIDI program, if it set one -- the timbre the
    /// automatic waveform pick reads.
    pub program: Option<u8>,
    /// Notes whose pitch lands inside the emitter's playable range.
    pub note_count: usize,
    /// The most notes sounding at once (over the in-range notes) -- the number
    /// of speakers this instrument wants before the cap.
    pub max_polyphony: usize,
    /// Notes dropped because their pitch is outside the playable range.
    pub dropped_notes: usize,
}

/// A few whole-file facts for the info listing.
#[derive(Clone, Debug, PartialEq)]
pub struct MidiSummary {
    /// SMF format: 0 (one track), 1 (parallel tracks), or 2 (sequential).
    pub format: u8,
    pub track_count: usize,
    /// End of the last note, in seconds.
    pub duration_s: f64,
    /// The file's initial tempo, in beats per minute.
    pub initial_bpm: f64,
    /// Total NoteOn events across every channel, percussion included.
    pub total_notes: usize,
    /// Whether the file has any note on the percussion channel (channel 10,
    /// index 9). Built as oneshot drum emitters unless `--no-percussion`.
    pub has_percussion: bool,
}

/// How a synth tone is chosen for a given instrument index.
///
/// `Uniform` is the CLI lite path -- one tone for the whole file, ignoring the
/// index. `PerInstrument` is the GUI, indexing the list; an index past the end
/// falls back to [`SynthWave::Sine`] so a lookup always has an answer.
#[derive(Clone, Debug, PartialEq)]
pub enum ToneAssignment {
    /// Pick a waveform per instrument from its GM program ([`timbre`]). Resolved
    /// into [`PerInstrument`](Self::PerInstrument) by [`Self::resolved`] in
    /// `analyze_midi`, where the programs are known. The CLI's default.
    Auto,
    Uniform(SynthWave),
    PerInstrument(Vec<SynthWave>),
}

impl Default for ToneAssignment {
    fn default() -> Self {
        ToneAssignment::Uniform(SynthWave::Sine)
    }
}

impl ToneAssignment {
    /// The synth for instrument `idx`.
    pub fn synth_for(&self, idx: usize) -> SynthWave {
        match self {
            // Resolved away before the build; the fallback keeps this total.
            ToneAssignment::Auto => SynthWave::Sine,
            ToneAssignment::Uniform(w) => *w,
            ToneAssignment::PerInstrument(v) => v.get(idx).copied().unwrap_or(SynthWave::Sine),
        }
    }

    /// Resolve [`Auto`](Self::Auto) into a concrete per-instrument list using
    /// each instrument's GM program (a program of `None` defaults to 0, GM's
    /// Acoustic Grand Piano). Every other variant passes through unchanged.
    pub fn resolved(&self, programs: &[Option<u8>]) -> ToneAssignment {
        match self {
            ToneAssignment::Auto => ToneAssignment::PerInstrument(
                programs
                    .iter()
                    .map(|p| timbre::synth_for_program(p.unwrap_or(0)))
                    .collect(),
            ),
            other => other.clone(),
        }
    }
}

/// Everything a MIDI build needs beyond the file itself.
///
/// Event-based playback has no fps, no envelope and no per-frame leveling: a
/// note is a constant-volume span, so only the spatialization/playback fields,
/// a per-note gain, the polyphony cap and the tone map remain.
#[derive(Clone, Debug)]
pub struct MidiOptions {
    /// Baked `InnerRadius` on every speaker (no-attenuation radius).
    pub inner_radius: f32,
    /// Baked `MaxDistance` on every speaker (where the sound stops).
    pub max_distance: f32,
    /// Pre-wire physical Pause/Restart/Resume buttons (default on).
    pub control_buttons: bool,
    /// Loop the piece forever (`true`) or play once and stop (`false`).
    pub loop_playback: bool,
    /// Place the speaker cluster inside the microchip's own inner grid.
    pub speakers_in_chip: bool,
    /// Multiplier on each note's velocity-derived volume, clamped to 1.0.
    pub gain: f32,
    /// Per-instrument volume multiplier, indexed like the discovered instrument
    /// list -- lets one part (say the bass) be turned down relative to the
    /// others. Empty means every instrument plays at 1.0 (the CLI's uniform
    /// case); the GUI fills one entry per instrument.
    pub instrument_volumes: Vec<f32>,
    /// Playback speed multiplier baked into the clock (1.0 = the file's own
    /// tempo, 2.0 = double speed). The generated `Rate` pin still overrides it
    /// at runtime.
    pub playback_rate: f32,
    /// Most speakers any one instrument gets, however many notes it plays at
    /// once. Overflow steals the oldest sounding note.
    pub polyphony_cap: usize,
    /// Seconds of the file the preview synthesizes; `0` = the whole file. Does
    /// not affect the generated build.
    pub preview_seconds: f32,
    /// Which tone each instrument sounds through.
    pub tones: ToneAssignment,
    /// Build the percussion channel (10) as oneshot drum emitters. Default
    /// true; the CLI `--no-percussion` clears it.
    pub build_percussion: bool,
    /// Per-role drum-sound override, indexed like
    /// [`crate::audio::percussion::PALETTE_ROLES`]. Empty (the default and the
    /// CLI) uses the baked fold table; the GUI's drum-kit dropdowns fill it.
    pub drum_kit: Vec<crate::audio::percussion::OneShotSound>,
}

impl Default for MidiOptions {
    fn default() -> Self {
        use crate::audio::speakers::{DEFAULT_INNER_RADIUS, DEFAULT_MAX_DISTANCE};
        Self {
            inner_radius: DEFAULT_INNER_RADIUS,
            max_distance: DEFAULT_MAX_DISTANCE,
            control_buttons: true,
            loop_playback: true,
            speakers_in_chip: false,
            gain: 1.0,
            instrument_volumes: Vec::new(),
            playback_rate: 1.0,
            polyphony_cap: 8,
            preview_seconds: 0.0,
            tones: ToneAssignment::default(),
            build_percussion: true,
            drum_kit: Vec::new(),
        }
    }
}

/// One note as the playback circuit reads it: a constant-pitch, constant-volume
/// span in seconds. `pitch` is the emitter `PitchMultiplier`, always in range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoteSpan {
    pub start_s: f64,
    pub end_s: f64,
    pub pitch: f64,
    pub volume: f64,
}

/// One built speaker: the notes it plays (non-overlapping, start-ordered) and
/// its tone. The playhead steps an index through `notes` at runtime.
#[derive(Clone, Debug)]
pub struct SpeakerVoice {
    pub notes: Vec<NoteSpan>,
    pub synth: SynthWave,
    /// Index into the discovered instrument list this speaker belongs to.
    pub instrument_idx: usize,
}

/// One percussion voice: a single oneshot sound and the times it is struck.
///
/// Every drum note that the fold table resolves to the same sound shares one
/// lane -- one emitter, one playhead pulsing its `Play` per strike.
#[derive(Clone, Debug, PartialEq)]
pub struct PercussionLane {
    pub sound: crate::audio::percussion::OneShotSound,
    /// Strike times in seconds, sorted ascending.
    pub hits: Vec<f64>,
}

/// The scheduled result: one voice per built speaker, one lane per distinct
/// drum sound, plus the piece's length (for the looping modulo and the preview).
#[derive(Clone, Debug)]
pub struct MidiScore {
    pub voices: Vec<SpeakerVoice>,
    pub percussion_lanes: Vec<PercussionLane>,
    pub duration_s: f64,
}

/// Parse `bytes` and report what is in the file, without scheduling or
/// building: the discovered instruments (percussion excluded) and a whole-file
/// summary. Backs the CLI `--list` and the GUI instrument table.
pub fn discover(bytes: &[u8]) -> Result<(Vec<Instrument>, MidiSummary), String> {
    let parsed = parse::parse(bytes)?;
    let instruments = parsed
        .instruments
        .iter()
        .map(schedule::instrument_of)
        .collect();
    Ok((instruments, parsed.summary))
}

/// Parse and schedule `bytes` into per-speaker note lists ready for the builder.
///
/// [`ToneAssignment::Auto`] is resolved here, where each instrument's GM program
/// is known, into a concrete per-instrument waveform list.
pub fn analyze_midi(bytes: &[u8], opts: &MidiOptions) -> Result<MidiScore, String> {
    let parsed = parse::parse(bytes)?;
    let programs: Vec<Option<u8>> = parsed.instruments.iter().map(|i| i.program).collect();
    let mut opts = opts.clone();
    opts.tones = opts.tones.resolved(&programs);
    schedule::schedule(&parsed.instruments, &parsed.percussion, &opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_uniform_tone_ignores_the_index() {
        let t = ToneAssignment::Uniform(SynthWave::Square);
        assert_eq!(t.synth_for(0), SynthWave::Square);
        assert_eq!(t.synth_for(9), SynthWave::Square);
    }

    #[test]
    fn a_per_instrument_tone_indexes_and_falls_back_to_sine() {
        let t = ToneAssignment::PerInstrument(vec![SynthWave::Square, SynthWave::Sawtooth]);
        assert_eq!(t.synth_for(0), SynthWave::Square);
        assert_eq!(t.synth_for(1), SynthWave::Sawtooth);
        // Past the end: a lookup always has an answer.
        assert_eq!(t.synth_for(2), SynthWave::Sine);
    }

    #[test]
    fn the_default_tone_is_uniform_sine() {
        assert_eq!(ToneAssignment::default(), ToneAssignment::Uniform(SynthWave::Sine));
    }

    #[test]
    fn auto_resolves_each_instrument_from_its_program() {
        // Violin (strings), Flute (pipe), and a channel with no program change.
        let programs = vec![Some(40u8), Some(73), None];
        assert_eq!(
            ToneAssignment::Auto.resolved(&programs),
            ToneAssignment::PerInstrument(vec![
                SynthWave::Sawtooth, // strings
                SynthWave::Sine,     // flute
                SynthWave::Sine,     // no program -> GM piano default -> sine
            ])
        );
    }

    #[test]
    fn resolving_a_non_auto_tone_leaves_it_unchanged() {
        let t = ToneAssignment::Uniform(SynthWave::Square);
        assert_eq!(t.resolved(&[Some(40), None]), t);
        let per = ToneAssignment::PerInstrument(vec![SynthWave::Triangle]);
        assert_eq!(per.resolved(&[Some(99)]), per);
    }

    #[test]
    fn defaults_match_the_spec() {
        let o = MidiOptions::default();
        assert_eq!(o.polyphony_cap, 8);
        assert_eq!(o.preview_seconds, 0.0, "0 = preview the whole file");
        assert_eq!(o.gain, 1.0);
        assert_eq!(o.playback_rate, 1.0, "1.0 = the file's own tempo");
        assert!(o.instrument_volumes.is_empty(), "empty = every instrument at 1.0");
        assert!(o.control_buttons, "buttons default on, like audio");
        assert!(o.loop_playback, "loops by default");
    }
}
