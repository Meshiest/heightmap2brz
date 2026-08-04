//! Standard MIDI File parsing: bytes -> per-instrument note events in absolute
//! seconds, plus a whole-file summary.
//!
//! An "instrument" is a `(track, channel)` pair. Format 1 files put one channel
//! per track, so this is one instrument per track; Format 0 files put every
//! channel on one track, so this splits them by channel. Percussion (channel
//! 10, index 9) is recorded in the summary but excluded from the instrument
//! list: v1 does not build it (pitched noise is inaudible, see the design doc).
use super::{MidiSummary, NoteEvent};
use midly::{Format, MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};
use std::collections::HashMap;

/// One instrument's notes, before range-filtering or scheduling. Notes carry
/// both in-range and out-of-range pitches; the caller decides which to keep.
pub struct InstrumentNotes {
    pub label: String,
    pub track: usize,
    pub channel: u8,
    /// Sorted by start time.
    pub notes: Vec<NoteEvent>,
}

/// Everything [`parse`] pulls out of a file.
pub struct ParsedMidi {
    /// One per `(track, channel)` that carries notes, percussion excluded,
    /// in a stable order (track then channel). The index into this vector is
    /// the instrument index the tone assignment uses.
    pub instruments: Vec<InstrumentNotes>,
    pub summary: MidiSummary,
}

/// The percussion channel, zero-based (MIDI "channel 10").
const PERCUSSION_CHANNEL: u8 = 9;

/// Default tempo when a file states none: 120 BPM, i.e. 500000 microseconds per
/// quarter note. The MIDI spec's own default.
const DEFAULT_US_PER_QUARTER: u32 = 500_000;

/// Parse a Standard MIDI File.
pub fn parse(bytes: &[u8]) -> Result<ParsedMidi, String> {
    let smf = Smf::parse(bytes).map_err(|e| format!("not a valid MIDI file: {e}"))?;

    let format = match smf.header.format {
        Format::SingleTrack => 0u8,
        Format::Parallel => 1,
        Format::Sequential => 2,
    };

    // Absolute-tick -> seconds, honoring tempo changes (or SMPTE absolute time).
    let to_seconds = build_time_map(&smf);

    // Per (track, channel): the notes, and the channel's program (last one
    // seen). Per track: its name. Preserving first-seen order via a Vec of keys
    // so the output order is deterministic (track then channel).
    let mut by_key: HashMap<(usize, u8), Vec<NoteEvent>> = HashMap::new();
    let mut program: HashMap<(usize, u8), u8> = HashMap::new();
    let mut track_name: HashMap<usize, String> = HashMap::new();
    let mut order: Vec<(usize, u8)> = Vec::new();

    let mut total_notes = 0usize;
    let mut has_percussion = false;
    let mut last_tick: u64 = 0;

    for (ti, track) in smf.tracks.iter().enumerate() {
        let mut tick: u64 = 0;
        // (channel, key) -> stack of (start_tick, velocity), a stack so two
        // overlapping presses of the same key nest correctly.
        let mut active: HashMap<(u8, u8), Vec<(u64, u8)>> = HashMap::new();

        for ev in track {
            tick += ev.delta.as_int() as u64;
            last_tick = last_tick.max(tick);
            match ev.kind {
                TrackEventKind::Meta(MetaMessage::TrackName(name)) => {
                    let s = String::from_utf8_lossy(name).trim().to_string();
                    if !s.is_empty() {
                        track_name.entry(ti).or_insert(s);
                    }
                }
                TrackEventKind::Midi { channel, message } => {
                    let ch = channel.as_int();
                    match message {
                        MidiMessage::ProgramChange { program: p } => {
                            program.insert((ti, ch), p.as_int());
                        }
                        MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                            total_notes += 1;
                            if ch == PERCUSSION_CHANNEL {
                                has_percussion = true;
                                continue;
                            }
                            active
                                .entry((ch, key.as_int()))
                                .or_default()
                                .push((tick, vel.as_int()));
                        }
                        // A NoteOff, or a NoteOn with velocity 0 (the common
                        // "running status" note-off): close the most recent
                        // matching press.
                        MidiMessage::NoteOff { key, .. }
                        | MidiMessage::NoteOn { key, .. } => {
                            if ch == PERCUSSION_CHANNEL {
                                continue;
                            }
                            close_note(
                                &mut active,
                                &mut by_key,
                                &mut order,
                                ch,
                                key.as_int(),
                                ti,
                                tick,
                                &to_seconds,
                            );
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        // Any note still held at end of track ends there.
        let ends: Vec<(u8, u8)> = active.keys().copied().collect();
        for (ch, key) in ends {
            close_note(
                &mut active, &mut by_key, &mut order, ch, key, ti, tick, &to_seconds,
            );
        }
    }

    // Assemble instruments in the stable first-seen order, labeling each.
    let mut instruments = Vec::new();
    for key @ (ti, ch) in order {
        let mut notes = by_key.remove(&key).unwrap_or_default();
        if notes.is_empty() {
            continue;
        }
        notes.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
        let label = track_name
            .get(&ti)
            .cloned()
            .or_else(|| program.get(&key).map(|&p| gm_program_name(p).to_string()))
            .unwrap_or_else(|| format!("Channel {}", ch + 1));
        instruments.push(InstrumentNotes { label, track: ti, channel: ch, notes });
    }

    let duration_s = instruments
        .iter()
        .flat_map(|i| i.notes.iter())
        .map(|n| n.end_s)
        .fold(0.0f64, f64::max);

    let summary = MidiSummary {
        format,
        track_count: smf.tracks.len(),
        duration_s,
        initial_bpm: initial_bpm(&smf),
        total_notes,
        has_percussion,
    };

    if instruments.is_empty() {
        return Err(if has_percussion {
            "this MIDI has only percussion (channel 10), which is not built in this version"
                .to_string()
        } else {
            "this MIDI file has no notes to play".to_string()
        });
    }

    Ok(ParsedMidi { instruments, summary })
}

/// Pop the most recent unmatched press of `(ch, key)` and record the completed
/// note. A note-off with no matching press is ignored (some files emit them).
#[allow(clippy::too_many_arguments)]
fn close_note(
    active: &mut HashMap<(u8, u8), Vec<(u64, u8)>>,
    by_key: &mut HashMap<(usize, u8), Vec<NoteEvent>>,
    order: &mut Vec<(usize, u8)>,
    ch: u8,
    key: u8,
    track: usize,
    end_tick: u64,
    to_seconds: &dyn Fn(u64) -> f64,
) {
    let Some(stack) = active.get_mut(&(ch, key)) else {
        return;
    };
    let Some((start_tick, velocity)) = stack.pop() else {
        return;
    };
    let ikey = (track, ch);
    if !by_key.contains_key(&ikey) {
        order.push(ikey);
    }
    let start_s = to_seconds(start_tick);
    let mut end_s = to_seconds(end_tick);
    // A zero-length note (off at the same tick as on) would sound for no frames;
    // give it a hair so it survives sampling at any fps.
    if end_s <= start_s {
        end_s = start_s + 1e-4;
    }
    by_key.entry(ikey).or_default().push(NoteEvent {
        start_s,
        end_s,
        note: key,
        velocity,
    });
}

/// Build the tick -> seconds function for this file, capturing its timing and
/// tempo changes.
fn build_time_map(smf: &Smf) -> Box<dyn Fn(u64) -> f64> {
    match smf.header.timing {
        Timing::Metrical(tpq) => {
            let tpq = tpq.as_int().max(1) as f64;
            // (absolute_tick, microseconds_per_quarter) from every track,
            // merged and sorted -- tempo is global even in Format 1 where it
            // lives on track 0.
            let mut tempos: Vec<(u64, u32)> = Vec::new();
            for track in &smf.tracks {
                let mut tick = 0u64;
                for ev in track {
                    tick += ev.delta.as_int() as u64;
                    if let TrackEventKind::Meta(MetaMessage::Tempo(us)) = ev.kind {
                        tempos.push((tick, us.as_int()));
                    }
                }
            }
            tempos.sort_by_key(|&(t, _)| t);
            Box::new(move |tick: u64| {
                let mut secs = 0.0f64;
                let mut cur = DEFAULT_US_PER_QUARTER;
                let mut prev = 0u64;
                for &(et, us) in &tempos {
                    if et >= tick {
                        break;
                    }
                    secs += (et - prev) as f64 * cur as f64 / 1e6 / tpq;
                    prev = et;
                    cur = us;
                }
                secs + (tick - prev) as f64 * cur as f64 / 1e6 / tpq
            })
        }
        // SMPTE absolute time: ticks are a fixed fraction of a second, tempo
        // meta events do not apply.
        Timing::Timecode(fps, subframe) => {
            let ticks_per_second = (fps.as_f32() as f64) * (subframe as f64);
            let tps = if ticks_per_second > 0.0 { ticks_per_second } else { 1.0 };
            Box::new(move |tick: u64| tick as f64 / tps)
        }
    }
}

/// The file's tempo at tick 0, as BPM, for the summary.
fn initial_bpm(smf: &Smf) -> f64 {
    let mut us = DEFAULT_US_PER_QUARTER;
    for track in &smf.tracks {
        let mut tick = 0u64;
        for ev in track {
            tick += ev.delta.as_int() as u64;
            if tick > 0 {
                break;
            }
            if let TrackEventKind::Meta(MetaMessage::Tempo(v)) = ev.kind {
                us = v.as_int();
            }
        }
    }
    60_000_000.0 / us as f64
}

/// The General MIDI Level 1 instrument name for a program number.
fn gm_program_name(program: u8) -> &'static str {
    const GM: [&str; 128] = [
        "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano",
        "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord",
        "Clavinet", "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba",
        "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ", "Percussive Organ",
        "Rock Organ", "Church Organ", "Reed Organ", "Accordion", "Harmonica",
        "Tango Accordion", "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)",
        "Electric Guitar (jazz)", "Electric Guitar (clean)", "Electric Guitar (muted)",
        "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics", "Acoustic Bass",
        "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass", "Slap Bass 1",
        "Slap Bass 2", "Synth Bass 1", "Synth Bass 2", "Violin", "Viola", "Cello",
        "Contrabass", "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp",
        "Timpani", "String Ensemble 1", "String Ensemble 2", "Synth Strings 1",
        "Synth Strings 2", "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit",
        "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn", "Brass Section",
        "Synth Brass 1", "Synth Brass 2", "Soprano Sax", "Alto Sax", "Tenor Sax",
        "Baritone Sax", "Oboe", "English Horn", "Bassoon", "Clarinet", "Piccolo",
        "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle",
        "Ocarina", "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)",
        "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)",
        "Lead 8 (bass + lead)", "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)",
        "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)",
        "Pad 8 (sweep)", "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)",
        "FX 4 (atmosphere)", "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)",
        "FX 8 (sci-fi)", "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bagpipe",
        "Fiddle", "Shanai", "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
        "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal", "Guitar Fret Noise",
        "Breath Noise", "Seashore", "Bird Tweet", "Telephone Ring", "Helicopter",
        "Applause", "Gunshot",
    ];
    GM.get(program as usize).copied().unwrap_or("Instrument")
}

#[cfg(test)]
mod tests {
    use super::*;
    use midly::{
        Header, MetaMessage, MidiMessage, Smf, Track, TrackEvent, TrackEventKind,
        num::{u4, u7, u15, u24, u28},
    };

    /// Build an SMF byte buffer from tracks, at `tpq` ticks per quarter.
    fn write_smf(format: Format, tpq: u16, tracks: Vec<Track<'static>>) -> Vec<u8> {
        let smf = Smf {
            header: Header::new(format, Timing::Metrical(u15::new(tpq))),
            tracks,
        };
        let mut buf = Vec::new();
        smf.write(&mut buf).expect("write smf");
        buf
    }

    fn ev(delta: u32, kind: TrackEventKind<'static>) -> TrackEvent<'static> {
        TrackEvent { delta: u28::new(delta), kind }
    }

    fn note_on(ch: u8, key: u8, vel: u8) -> TrackEventKind<'static> {
        TrackEventKind::Midi {
            channel: u4::new(ch),
            message: MidiMessage::NoteOn { key: u7::new(key), vel: u7::new(vel) },
        }
    }
    fn note_off(ch: u8, key: u8) -> TrackEventKind<'static> {
        TrackEventKind::Midi {
            channel: u4::new(ch),
            message: MidiMessage::NoteOff { key: u7::new(key), vel: u7::new(0) },
        }
    }

    #[test]
    fn a_single_note_parses_with_the_right_pitch_and_timing() {
        // 480 tpq, default 120 BPM => 1 quarter = 0.5 s. A note from tick 0 to
        // tick 480 is one beat: 0.0 to 0.5 s.
        let track = vec![
            ev(0, note_on(0, 69, 100)),
            ev(480, note_off(0, 69)),
            ev(0, TrackEventKind::Meta(MetaMessage::EndOfTrack)),
        ];
        let bytes = write_smf(Format::SingleTrack, 480, vec![track]);
        let p = parse(&bytes).expect("parse");
        assert_eq!(p.instruments.len(), 1);
        let n = &p.instruments[0].notes;
        assert_eq!(n.len(), 1);
        assert_eq!(n[0].note, 69);
        assert!((n[0].start_s - 0.0).abs() < 1e-9);
        assert!((n[0].end_s - 0.5).abs() < 1e-6, "one beat at 120 BPM is 0.5 s");
    }

    #[test]
    fn a_tempo_change_shifts_later_notes() {
        // Beat 0 at 120 BPM (0.5 s/beat); at tick 480 tempo doubles to 240 BPM
        // (0.25 s/beat). A note at tick 960 starts at 0.5 + 0.25 = 0.75 s.
        let track = vec![
            ev(0, TrackEventKind::Meta(MetaMessage::Tempo(u24::new(500_000)))),
            ev(480, TrackEventKind::Meta(MetaMessage::Tempo(u24::new(250_000)))),
            ev(480, note_on(0, 60, 90)),
            ev(240, note_off(0, 60)),
            ev(0, TrackEventKind::Meta(MetaMessage::EndOfTrack)),
        ];
        let bytes = write_smf(Format::SingleTrack, 480, vec![track]);
        let p = parse(&bytes).expect("parse");
        let n = &p.instruments[0].notes[0];
        assert!((n.start_s - 0.75).abs() < 1e-6, "got {}", n.start_s);
    }

    #[test]
    fn a_note_on_velocity_zero_ends_a_note() {
        let track = vec![
            ev(0, note_on(0, 64, 80)),
            ev(240, note_on(0, 64, 0)), // running-status note off
            ev(0, TrackEventKind::Meta(MetaMessage::EndOfTrack)),
        ];
        let bytes = write_smf(Format::SingleTrack, 480, vec![track]);
        let p = parse(&bytes).expect("parse");
        assert_eq!(p.instruments[0].notes.len(), 1);
        assert!(p.instruments[0].notes[0].end_s > p.instruments[0].notes[0].start_s);
    }

    #[test]
    fn a_track_name_becomes_the_label() {
        let track = vec![
            ev(0, TrackEventKind::Meta(MetaMessage::TrackName(b"Lead Synth"))),
            ev(0, note_on(0, 60, 90)),
            ev(120, note_off(0, 60)),
            ev(0, TrackEventKind::Meta(MetaMessage::EndOfTrack)),
        ];
        let bytes = write_smf(Format::Parallel, 480, vec![track]);
        let p = parse(&bytes).expect("parse");
        assert_eq!(p.instruments[0].label, "Lead Synth");
    }

    #[test]
    fn a_program_change_labels_an_unnamed_track() {
        let track = vec![
            ev(
                0,
                TrackEventKind::Midi {
                    channel: u4::new(0),
                    message: MidiMessage::ProgramChange { program: u7::new(24) },
                },
            ),
            ev(0, note_on(0, 60, 90)),
            ev(120, note_off(0, 60)),
            ev(0, TrackEventKind::Meta(MetaMessage::EndOfTrack)),
        ];
        let bytes = write_smf(Format::SingleTrack, 480, vec![track]);
        let p = parse(&bytes).expect("parse");
        assert_eq!(p.instruments[0].label, "Acoustic Guitar (nylon)");
    }

    #[test]
    fn a_format_0_track_splits_into_one_instrument_per_channel() {
        // One track, two channels -> two instruments.
        let track = vec![
            ev(0, note_on(0, 60, 90)),
            ev(0, note_on(1, 48, 90)),
            ev(240, note_off(0, 60)),
            ev(0, note_off(1, 48)),
            ev(0, TrackEventKind::Meta(MetaMessage::EndOfTrack)),
        ];
        let bytes = write_smf(Format::SingleTrack, 480, vec![track]);
        let p = parse(&bytes).expect("parse");
        assert_eq!(p.instruments.len(), 2, "two channels are two instruments");
        let channels: Vec<u8> = p.instruments.iter().map(|i| i.channel).collect();
        assert!(channels.contains(&0) && channels.contains(&1));
    }

    #[test]
    fn percussion_is_excluded_but_recorded() {
        let track = vec![
            ev(0, note_on(9, 36, 100)), // kick on channel 10
            ev(240, note_off(9, 36)),
            ev(0, note_on(0, 60, 90)),
            ev(240, note_off(0, 60)),
            ev(0, TrackEventKind::Meta(MetaMessage::EndOfTrack)),
        ];
        let bytes = write_smf(Format::SingleTrack, 480, vec![track]);
        let p = parse(&bytes).expect("parse");
        assert_eq!(p.instruments.len(), 1, "only the pitched channel is an instrument");
        assert_eq!(p.instruments[0].channel, 0);
        assert!(p.summary.has_percussion, "percussion is recorded in the summary");
        assert_eq!(p.summary.total_notes, 2, "both notes counted");
    }

    #[test]
    fn an_empty_file_is_an_error() {
        let track = vec![ev(0, TrackEventKind::Meta(MetaMessage::EndOfTrack))];
        let bytes = write_smf(Format::SingleTrack, 480, vec![track]);
        assert!(parse(&bytes).is_err());
    }

    #[test]
    fn garbage_bytes_are_an_error_not_a_panic() {
        assert!(parse(b"not a midi file at all").is_err());
    }
}
