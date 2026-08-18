//! Offline preview synthesis: render a scheduled [`MidiScore`] to PCM (and WAV)
//! so a front end can play "roughly what this will sound like" without the game.
//!
//! Faithful to the notes, timing and tones, but not to the game's exact synth
//! timbre or the spatialization of many emitters. Pure and wasm-safe: no audio
//! device, no platform code. The players (desktop rodio, browser Web Audio) and
//! the CLI `--preview` all consume this.
use super::MidiScore;
use crate::audio::bands::{BASE_HZ, PITCH_MAX, PITCH_MIN};
use crate::audio::track::SynthWave;

/// Peak the normalized mix is scaled to, leaving headroom so summed voices do
/// not clip.
const PREVIEW_PEAK: f32 = 0.9;

/// One oscillator sample for a wave at `phase` in `[0, 1)`.
fn wave(synth: SynthWave, phase: f64) -> f64 {
    use std::f64::consts::TAU;
    match synth {
        SynthWave::Sine => (phase * TAU).sin(),
        SynthWave::Square => if phase < 0.5 { 1.0 } else { -1.0 },
        SynthWave::Triangle => if phase < 0.5 { 4.0 * phase - 1.0 } else { 3.0 - 4.0 * phase },
        SynthWave::Sawtooth => 2.0 * phase - 1.0,
    }
}

/// Render `score` to a mono PCM buffer at `sample_rate`, bounded to the first
/// `seconds` of the PIECE (`<= 0` renders the whole piece) and sped up by
/// `rate` (the same `playback_rate` the game clock applies, so 2.0 plays it in
/// half the wall-clock time). Each voice is an oscillator of its wave, sounding
/// its active note's pitch at that note's volume and silent in the gaps; phase
/// runs continuously (held pitch across gaps) so notes do not click. Notes whose
/// pitch is outside the emitter's `PITCH_MIN..=PITCH_MAX` range are DROPPED, as
/// the build drops them. The mix is normalized to [`PREVIEW_PEAK`].
pub fn synthesize(score: &MidiScore, sample_rate: u32, seconds: f32, rate: f32) -> Vec<f32> {
    if score.voices.is_empty() || sample_rate == 0 || score.duration_s <= 0.0 {
        return Vec::new();
    }
    let rate = (rate as f64).max(0.01);
    // `seconds` selects a span of the piece; the played (wall-clock) length is
    // that span divided by the playback rate.
    let span_s = if seconds <= 0.0 {
        score.duration_s
    } else {
        (seconds as f64).min(score.duration_s)
    };
    let n = ((span_s / rate) * sample_rate as f64) as usize;
    if n == 0 {
        return Vec::new();
    }
    let dt = 1.0 / sample_rate as f64;
    let mut out = vec![0.0f32; n];
    let (lo, hi) = (PITCH_MIN as f64, PITCH_MAX as f64);

    for voice in &score.voices {
        if voice.notes.is_empty() {
            continue;
        }
        let mut phase = 0.0f64;
        let mut cursor = 0usize; // first note not yet ended
        let mut held_pitch = voice.notes[0].pitch;
        for (i, sample) in out.iter_mut().enumerate() {
            // Score-time position: wall-clock time scaled up by the rate.
            let t = i as f64 * dt * rate;
            while cursor < voice.notes.len() && voice.notes[cursor].end_s <= t {
                cursor += 1;
            }
            let active = cursor < voice.notes.len() && voice.notes[cursor].start_s <= t;
            if cursor < voice.notes.len() {
                held_pitch = voice.notes[cursor].pitch;
            }
            // Advance phase in real time (pitch is unchanged by `rate`) so a
            // note resuming the waveform does not click.
            phase += held_pitch * BASE_HZ as f64 * dt;
            if phase >= 1.0 {
                phase -= phase.floor();
            }
            // Only sound in-range pitches -- the build drops the rest.
            if active && (lo..=hi).contains(&held_pitch) {
                *sample += (wave(voice.synth, phase) * voice.notes[cursor].volume) as f32;
            }
        }
    }

    let peak = out.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak > 0.0 {
        let g = PREVIEW_PEAK / peak;
        for s in &mut out {
            *s *= g;
        }
    }
    out
}

/// Encode mono f32 PCM as a 16-bit PCM WAV. Hand-written (no dependency).
pub fn to_wav(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    const BYTES_PER_SAMPLE: u32 = 2;
    let data_len = pcm.len() as u32 * BYTES_PER_SAMPLE;
    let mut w = Vec::with_capacity(44 + data_len as usize);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data_len).to_le_bytes());
    w.extend_from_slice(b"WAVE");
    w.extend_from_slice(b"fmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes()); // PCM
    w.extend_from_slice(&1u16.to_le_bytes()); // mono
    w.extend_from_slice(&sample_rate.to_le_bytes());
    w.extend_from_slice(&(sample_rate * BYTES_PER_SAMPLE).to_le_bytes());
    w.extend_from_slice(&(BYTES_PER_SAMPLE as u16).to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_len.to_le_bytes());
    for &s in pcm {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        w.extend_from_slice(&v.to_le_bytes());
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi::{MidiScore, NoteSpan, SpeakerVoice};

    fn one_note(pitch: f64, synth: SynthWave, dur: f64) -> MidiScore {
        MidiScore {
            voices: vec![SpeakerVoice {
                notes: vec![NoteSpan { start_s: 0.0, end_s: dur, pitch, volume: 1.0 }],
                synth,
                instrument_idx: 0,
            }],
            percussion_lanes: vec![],
            duration_s: dur,
        }
    }

    #[test]
    fn synthesis_is_deterministic() {
        let s = one_note(1.0, SynthWave::Sine, 1.0);
        assert_eq!(synthesize(&s, 44_100, 0.0, 1.0), synthesize(&s, 44_100, 0.0, 1.0));
    }

    #[test]
    fn a_held_a4_renders_near_440_hz() {
        let sr = 44_100u32;
        let s = one_note(1.0, SynthWave::Sine, 1.0); // pitch 1.0 -> 440 Hz
        let pcm = synthesize(&s, sr, 0.0, 1.0);
        let crossings = pcm.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
        let hz = crossings as f64 / (pcm.len() as f64 / sr as f64);
        assert!((hz - 440.0).abs() < 5.0, "dominant frequency {hz} should be ~440 Hz");
    }

    #[test]
    fn a_gap_between_notes_is_silent() {
        // Note 0.0..0.4, gap 0.4..0.6, note 0.6..1.0.
        let score = MidiScore {
            voices: vec![SpeakerVoice {
                notes: vec![
                    NoteSpan { start_s: 0.0, end_s: 0.4, pitch: 1.0, volume: 1.0 },
                    NoteSpan { start_s: 0.6, end_s: 1.0, pitch: 1.0, volume: 1.0 },
                ],
                synth: SynthWave::Sine,
                instrument_idx: 0,
            }],
            percussion_lanes: vec![],
            duration_s: 1.0,
        };
        let sr = 10_000u32;
        let pcm = synthesize(&score, sr, 0.0, 1.0);
        // Sample at t=0.5 (mid-gap) should be silent.
        let mid = (0.5 * sr as f64) as usize;
        assert!(pcm[mid].abs() < 1e-6, "the gap must be silent, got {}", pcm[mid]);
    }

    #[test]
    fn the_seconds_cap_bounds_the_render() {
        let s = one_note(1.0, SynthWave::Sine, 10.0);
        let one = synthesize(&s, 8_000, 1.0, 1.0);
        assert!((one.len() as i64 - 8_000).abs() < 8_000 / 10, "about one second");
        let all = synthesize(&s, 8_000, 0.0, 1.0);
        assert!(all.len() > one.len() * 5, "the whole piece is much longer");
    }

    #[test]
    fn the_wav_header_matches_the_pcm() {
        let pcm = vec![0.0f32, 0.5, -0.5, 1.0];
        let wav = to_wav(&pcm, 44_100);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        let data_len = u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]);
        assert_eq!(data_len as usize, pcm.len() * 2);
        assert_eq!(wav.len(), 44 + pcm.len() * 2);
    }
}
