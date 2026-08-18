//! Turn parsed notes into per-speaker note lists.
//!
//! Each instrument gets `min(max_polyphony, cap)` speakers. Notes are walked in
//! start order and slotted so no two notes share a slot at once; when every
//! slot is busy the oldest sounding note is stolen (cut short) so the newest
//! always plays. Each slot becomes one speaker's list of `(start, end, pitch,
//! volume)` spans -- no fps sampling, no envelope: the playback circuit reads
//! the spans directly against the clock's runtime.
use super::parse::InstrumentNotes;
use super::{
    Instrument, MidiOptions, MidiScore, NoteEvent, NoteSpan, PercussionHit, PercussionLane,
    SpeakerVoice, drums,
};
use crate::audio::bands::{PITCH_MAX, PITCH_MIN};

/// The `PitchMultiplier` a MIDI note maps to: `2^((note - 69) / 12)`, so note
/// 69 (A4) is 1.0 and each octave doubles it.
pub fn note_to_pitch(note: u8) -> f64 {
    2.0f64.powf((note as f64 - 69.0) / 12.0)
}

/// Whether a note's pitch lands inside the emitter's playable range. Notes
/// outside are dropped, never clamped (clamping plays a wrong note).
pub fn note_in_range(note: u8) -> bool {
    let p = note_to_pitch(note);
    p >= PITCH_MIN as f64 && p <= PITCH_MAX as f64
}

/// The most notes sounding at once, over a set of notes (seconds-based). A
/// sweep line: an end at the same instant as a start is processed first, since
/// a note ending exactly as another starts does not overlap it.
fn max_simultaneous(notes: &[&NoteEvent]) -> usize {
    let mut events: Vec<(f64, i32)> = Vec::with_capacity(notes.len() * 2);
    for n in notes {
        events.push((n.start_s, 1));
        events.push((n.end_s, -1));
    }
    events.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut cur = 0i32;
    let mut max = 0i32;
    for (_, d) in events {
        cur += d;
        max = max.max(cur);
    }
    max as usize
}

/// The discovered-instrument view of one parsed track/channel. Backs
/// [`super::discover`].
pub fn instrument_of(n: &InstrumentNotes) -> Instrument {
    let in_range: Vec<&NoteEvent> = n.notes.iter().filter(|e| note_in_range(e.note)).collect();
    Instrument {
        label: n.label.clone(),
        track: n.track,
        channel: n.channel,
        program: n.program,
        note_count: in_range.len(),
        max_polyphony: max_simultaneous(&in_range),
        dropped_notes: n.notes.len() - in_range.len(),
    }
}

/// One note as it is slotted: a mutable end (stealing cuts it short).
#[derive(Clone, Copy)]
struct Slotted {
    start_s: f64,
    end_s: f64,
    note: u8,
    velocity: u8,
}

fn validate(opts: &MidiOptions) -> Result<(), String> {
    if !opts.gain.is_finite() || opts.gain < 0.0 {
        return Err(format!("gain must be a non-negative finite number, got {}", opts.gain));
    }
    if opts.polyphony_cap == 0 {
        return Err("polyphony cap must be at least 1 (it is a speaker count per instrument)".to_string());
    }
    Ok(())
}

/// Schedule every instrument into per-speaker note lists.
pub fn schedule(
    instruments: &[InstrumentNotes],
    percussion: &[PercussionHit],
    opts: &MidiOptions,
) -> Result<MidiScore, String> {
    validate(opts)?;

    let mut voices: Vec<SpeakerVoice> = Vec::new();

    for (idx, inst) in instruments.iter().enumerate() {
        let mut in_range: Vec<&NoteEvent> = inst.notes.iter().filter(|e| note_in_range(e.note)).collect();
        if in_range.is_empty() {
            continue; // no speaker for an all-out-of-range instrument
        }
        in_range.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
        let slots_n = max_simultaneous(&in_range).min(opts.polyphony_cap).max(1);

        // Slot allocation with oldest-note stealing. A slot is free at `start`
        // when its last note has already ended.
        let mut slots: Vec<Vec<Slotted>> = vec![Vec::new(); slots_n];
        for n in in_range {
            let seg = Slotted { start_s: n.start_s, end_s: n.end_s, note: n.note, velocity: n.velocity };
            let free = (0..slots_n).find(|&k| slots[k].last().is_none_or(|s| s.end_s <= seg.start_s));
            let slot = if let Some(k) = free {
                k
            } else {
                // Steal the slot whose current note started earliest.
                let k = (0..slots_n)
                    .min_by(|&a, &b| slots[a].last().unwrap().start_s.total_cmp(&slots[b].last().unwrap().start_s))
                    .unwrap();
                let last = slots[k].last_mut().unwrap();
                if last.end_s > seg.start_s {
                    last.end_s = seg.start_s;
                }
                if last.end_s <= last.start_s {
                    slots[k].pop(); // stealing left nothing of it
                }
                k
            };
            slots[slot].push(seg);
        }

        let synth = opts.tones.synth_for(idx);
        // Per-instrument fader (empty list = 1.0 for every instrument, the CLI's
        // uniform case), on top of the global gain. Still clamped to 1.0.
        let inst_vol = opts.instrument_volumes.get(idx).copied().unwrap_or(1.0) as f64;
        for slot in slots {
            if slot.is_empty() {
                continue;
            }
            let notes: Vec<NoteSpan> = slot
                .iter()
                .map(|s| NoteSpan {
                    start_s: s.start_s,
                    end_s: s.end_s,
                    pitch: note_to_pitch(s.note),
                    volume: (s.velocity as f64 / 127.0 * opts.gain as f64 * inst_vol).min(1.0),
                })
                .collect();
            voices.push(SpeakerVoice { notes, synth, instrument_idx: idx });
        }
    }

    // Percussion: one lane per distinct sound the fold table resolves to, each
    // carrying that sound's strike times.
    let mut percussion_lanes: Vec<PercussionLane> = Vec::new();
    if opts.build_percussion {
        for hit in percussion {
            let sound = drums::drum_sound_with_kit(hit.note, &opts.drum_kit);
            match percussion_lanes.iter_mut().find(|l| l.sound == sound) {
                Some(lane) => lane.hits.push(hit.start_s),
                None => percussion_lanes.push(PercussionLane { sound, hits: vec![hit.start_s] }),
            }
        }
        for lane in &mut percussion_lanes {
            lane.hits.sort_by(f64::total_cmp);
        }
    }

    if voices.is_empty() && percussion_lanes.is_empty() {
        return Err("this MIDI has no in-range notes to play (every note is outside the emitter's pitch range)".to_string());
    }

    let duration_s = voices
        .iter()
        .flat_map(|v| v.notes.iter())
        .map(|n| n.end_s)
        .chain(percussion_lanes.iter().flat_map(|l| l.hits.iter().copied()))
        .fold(0.0f64, f64::max);

    Ok(MidiScore { voices, percussion_lanes, duration_s })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::track::SynthWave;
    use crate::midi::ToneAssignment;

    fn note(start_s: f64, end_s: f64, note: u8, velocity: u8) -> NoteEvent {
        NoteEvent { start_s, end_s, note, velocity }
    }
    fn inst(label: &str, channel: u8, notes: Vec<NoteEvent>) -> InstrumentNotes {
        InstrumentNotes { label: label.to_string(), track: 0, channel, program: None, notes }
    }
    fn opts(cap: usize) -> MidiOptions {
        MidiOptions { polyphony_cap: cap, ..MidiOptions::default() }
    }

    #[test]
    fn note_to_pitch_is_exact_at_a4_and_octaves() {
        assert!((note_to_pitch(69) - 1.0).abs() < 1e-12, "A4 is the identity");
        assert!((note_to_pitch(81) - 2.0).abs() < 1e-12, "an octave up doubles");
        assert!((note_to_pitch(57) - 0.5).abs() < 1e-12, "an octave down halves");
    }

    #[test]
    fn out_of_range_notes_are_dropped_and_counted() {
        let i = inst("x", 0, vec![note(0.0, 1.0, 0, 100), note(0.0, 1.0, 60, 100), note(0.0, 1.0, 127, 100)]);
        let d = instrument_of(&i);
        assert_eq!(d.note_count, 1, "only the middle note is playable");
        assert_eq!(d.dropped_notes, 2);
    }

    #[test]
    fn a_three_note_chord_gets_three_speakers() {
        let i = inst("chord", 0, vec![note(0.0, 1.0, 60, 100), note(0.0, 1.0, 64, 100), note(0.0, 1.0, 67, 100)]);
        let s = schedule(&[i], &[], &opts(8)).expect("schedule");
        assert_eq!(s.voices.len(), 3, "one speaker per simultaneous note");
    }

    #[test]
    fn a_monophonic_line_gets_one_speaker_with_all_its_notes() {
        let i = inst("mono", 0, vec![note(0.0, 0.5, 60, 100), note(0.5, 1.0, 62, 100), note(1.0, 1.5, 64, 100)]);
        let s = schedule(&[i], &[], &opts(8)).expect("schedule");
        assert_eq!(s.voices.len(), 1, "sequential notes share one speaker");
        assert_eq!(s.voices[0].notes.len(), 3, "and it holds all three");
        assert!((s.duration_s - 1.5).abs() < 1e-9);
    }

    #[test]
    fn a_chord_past_the_cap_steals_and_stays_within_the_cap() {
        let i = inst("big", 0, (0..6).map(|k| note(0.0, 1.0, 60 + k, 100)).collect());
        let s = schedule(&[i], &[], &opts(3)).expect("schedule");
        assert_eq!(s.voices.len(), 3, "capped at 3 speakers");
    }

    #[test]
    fn velocity_and_gain_set_the_note_volume() {
        let i = inst("v", 0, vec![note(0.0, 1.0, 69, 127)]);
        let s = schedule(&[i], &[], &MidiOptions { gain: 1.0, ..opts(8) }).expect("schedule");
        let span = s.voices[0].notes[0];
        assert!((span.volume - 1.0).abs() < 1e-9, "full velocity at gain 1 is 1.0");
        assert!((span.pitch - 1.0).abs() < 1e-9, "note 69 is pitch 1.0");
        // Gain above 1 clamps.
        let s2 = schedule(&[inst("v", 0, vec![note(0.0, 1.0, 69, 127)])], &[], &MidiOptions { gain: 4.0, ..opts(8) }).expect("schedule");
        assert_eq!(s2.voices[0].notes[0].volume, 1.0, "volume never exceeds 1.0");
    }

    #[test]
    fn a_per_instrument_volume_scales_only_its_own_notes() {
        let a = inst("a", 0, vec![note(0.0, 1.0, 69, 127)]);
        let b = inst("b", 1, vec![note(0.0, 1.0, 69, 127)]);
        let o = MidiOptions {
            gain: 1.0,
            instrument_volumes: vec![0.5, 1.0],
            ..opts(8)
        };
        let s = schedule(&[a, b], &[], &o).expect("schedule");
        assert!((s.voices[0].notes[0].volume - 0.5).abs() < 1e-9, "instrument 0 halved");
        assert!((s.voices[1].notes[0].volume - 1.0).abs() < 1e-9, "instrument 1 untouched");
    }

    #[test]
    fn a_missing_per_instrument_volume_defaults_to_full() {
        // Empty list (the CLI case): every instrument plays at 1.0.
        let i = inst("x", 0, vec![note(0.0, 1.0, 69, 127)]);
        let s = schedule(&[i], &[], &MidiOptions { gain: 1.0, ..opts(8) }).expect("schedule");
        assert!((s.voices[0].notes[0].volume - 1.0).abs() < 1e-9);
    }

    #[test]
    fn per_instrument_tones_reach_the_right_speakers() {
        let a = inst("a", 0, vec![note(0.0, 1.0, 60, 100)]);
        let b = inst("b", 1, vec![note(0.0, 1.0, 62, 100)]);
        let o = MidiOptions {
            tones: ToneAssignment::PerInstrument(vec![SynthWave::Square, SynthWave::Sawtooth]),
            ..opts(8)
        };
        let s = schedule(&[a, b], &[], &o).expect("schedule");
        assert_eq!(s.voices[0].synth, SynthWave::Square);
        assert_eq!(s.voices[1].synth, SynthWave::Sawtooth);
    }

    #[test]
    fn scheduling_is_deterministic() {
        let mk = || inst("d", 0, vec![note(0.0, 1.0, 60, 100), note(0.2, 0.8, 64, 90)]);
        let a = schedule(&[mk()], &[], &opts(8)).expect("a");
        let b = schedule(&[mk()], &[], &opts(8)).expect("b");
        assert_eq!(a.voices.len(), b.voices.len());
        for (x, y) in a.voices.iter().zip(&b.voices) {
            assert_eq!(x.notes, y.notes);
        }
    }

    #[test]
    fn a_bad_gain_is_refused_by_name() {
        let i = inst("x", 0, vec![note(0.0, 1.0, 60, 100)]);
        let err = schedule(&[i], &[], &MidiOptions { gain: -1.0, ..opts(8) }).unwrap_err();
        assert!(err.contains("gain"), "got: {err}");
    }

    #[test]
    fn percussion_groups_into_one_lane_per_folded_sound() {
        use crate::audio::percussion::KICK_1;
        use crate::midi::PercussionHit;
        let h = |start_s: f64, note: u8| PercussionHit { start_s, note, velocity: 100 };
        // Two kicks (36), one snare (38), one closed hat (42) -> three sounds.
        let perc = vec![h(0.0, 36), h(1.0, 38), h(0.5, 36), h(0.25, 42)];

        let s = schedule(&[], &perc, &opts(8)).expect("drum-only schedules");
        assert_eq!(s.percussion_lanes.len(), 3, "one lane per distinct sound");

        let kick = s.percussion_lanes.iter().find(|l| l.sound == KICK_1).unwrap();
        assert_eq!(kick.hits, vec![0.0, 0.5], "both kick strikes, sorted");
        assert!((s.duration_s - 1.0).abs() < 1e-9, "duration spans the last strike");
    }

    #[test]
    fn a_drum_only_score_has_no_voices_but_has_lanes() {
        use crate::midi::PercussionHit;
        let s = schedule(&[], &[PercussionHit { start_s: 0.0, note: 42, velocity: 100 }], &opts(8))
            .expect("drum-only must schedule");
        assert!(s.voices.is_empty());
        assert_eq!(s.percussion_lanes.len(), 1);
    }

    #[test]
    fn a_drum_kit_override_changes_the_built_lane_sound() {
        use crate::audio::percussion::{KICK_1, PALETTE_ROLES};
        use crate::midi::PercussionHit;
        // Remap the closed-hat role (note 42) to the kick sound, as the GUI would.
        let mut kit: Vec<_> = PALETTE_ROLES.iter().map(|r| r.sound).collect();
        kit[drums::role_index(42)] = KICK_1;
        let opts = MidiOptions { drum_kit: kit, ..opts(8) };

        let s = schedule(&[], &[PercussionHit { start_s: 0.0, note: 42, velocity: 100 }], &opts)
            .expect("schedule");
        assert_eq!(s.percussion_lanes[0].sound, KICK_1, "override reaches the lane");
    }
}
