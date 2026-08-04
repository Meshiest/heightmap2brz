//! End-to-end midi2brick: a synthetic MIDI file through discover -> analyze ->
//! build, checking per-instrument tones reach the emitters and every wire in
//! the written `.brz` resolves.
#[path = "wire_integrity.rs"]
mod wire_integrity;

use heightmap::audio::speakers::{AUDIO_EMITTER, build_midi_event_world};
use heightmap::audio::track::SynthWave;
use heightmap::midi::{MidiOptions, ToneAssignment, analyze_midi, discover};
use midly::{
    Format, Header, MetaMessage, MidiMessage, Smf, Timing, Track, TrackEvent, TrackEventKind,
    num::{u4, u7, u15, u28},
};

/// A two-track file: a named "Lead" (channel 0) and a named "Bass" (channel 0
/// on its own track), each with one one-beat note. Two `(track, channel)`
/// instruments, in that order.
fn two_instrument_smf() -> Vec<u8> {
    fn ev(delta: u32, kind: TrackEventKind<'static>) -> TrackEvent<'static> {
        TrackEvent { delta: u28::new(delta), kind }
    }
    fn on(key: u8) -> TrackEventKind<'static> {
        TrackEventKind::Midi {
            channel: u4::new(0),
            message: MidiMessage::NoteOn { key: u7::new(key), vel: u7::new(100) },
        }
    }
    fn off(key: u8) -> TrackEventKind<'static> {
        TrackEventKind::Midi {
            channel: u4::new(0),
            message: MidiMessage::NoteOff { key: u7::new(key), vel: u7::new(0) },
        }
    }
    let lead: Track = vec![
        ev(0, TrackEventKind::Meta(MetaMessage::TrackName(b"Lead"))),
        ev(0, on(69)),
        ev(480, off(69)),
        ev(0, TrackEventKind::Meta(MetaMessage::EndOfTrack)),
    ];
    let bass: Track = vec![
        ev(0, TrackEventKind::Meta(MetaMessage::TrackName(b"Bass"))),
        ev(0, on(45)),
        ev(480, off(45)),
        ev(0, TrackEventKind::Meta(MetaMessage::EndOfTrack)),
    ];
    let smf = Smf {
        header: Header::new(Format::Parallel, Timing::Metrical(u15::new(480))),
        tracks: vec![lead, bass],
    };
    let mut buf = Vec::new();
    smf.write(&mut buf).expect("write smf");
    buf
}

#[test]
fn discover_lists_both_instruments_in_order() {
    let bytes = two_instrument_smf();
    let (instruments, summary) = discover(&bytes).expect("discover");
    assert_eq!(instruments.len(), 2, "two named tracks are two instruments");
    assert_eq!(instruments[0].label, "Lead");
    assert_eq!(instruments[1].label, "Bass");
    assert_eq!(instruments[0].note_count, 1);
    assert_eq!(instruments[0].max_polyphony, 1);
    assert_eq!(summary.format, 1, "SMF Format 1 (parallel)");
    assert_eq!(summary.total_notes, 2);
    assert!(!summary.has_percussion);
}

#[test]
fn a_midi_world_builds_with_per_instrument_tones_and_every_wire_resolves() {
    let bytes = two_instrument_smf();
    let opts = MidiOptions {
        tones: ToneAssignment::PerInstrument(vec![SynthWave::Square, SynthWave::Sawtooth]),
        ..MidiOptions::default()
    };
    let score = analyze_midi(&bytes, &opts).expect("analyze");
    assert_eq!(score.voices.len(), 2, "one speaker per monophonic instrument");
    assert_eq!(score.voices[0].synth, SynthWave::Square);
    assert_eq!(score.voices[1].synth, SynthWave::Sawtooth);

    let world = build_midi_event_world(&score, &opts).expect("build");

    // One AudioEmitter per scheduled speaker (default layout: main grid).
    let emitters = world
        .bricks
        .iter()
        .filter(|b| {
            b.components
                .iter()
                .any(|c| c.component_type().is_some_and(|t| t.to_string() == AUDIO_EMITTER))
        })
        .count();
    assert_eq!(emitters, score.voices.len());

    // Both chosen tones are referenced in the save's asset table.
    let refs: Vec<String> = world
        .global_data
        .external_asset_references
        .iter()
        .map(|(_, n)| n.clone())
        .collect();
    for wave in [SynthWave::Square, SynthWave::Sawtooth] {
        assert!(
            refs.iter().any(|n| n.as_str() == wave.asset().as_ref()),
            "{:?} tone must be referenced; refs = {refs:?}",
            wave
        );
    }

    // Every wire in the written save resolves.
    let path = std::env::temp_dir().join(format!("h2b_midi_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    let result = std::panic::catch_unwind(|| wire_integrity::assert_wires_valid(&path));
    let _ = std::fs::remove_file(&path);
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

#[test]
fn the_event_playhead_probe_is_structurally_valid() {
    // The event-based prototype (advancing-index playhead) must at least be a
    // valid save whose every wire resolves before it is worth loading in game.
    let world = heightmap::audio::speakers::build_playhead_probe_world().expect("build probe");
    let path = std::env::temp_dir().join(format!("h2b_playhead_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    let result = std::panic::catch_unwind(|| wire_integrity::assert_wires_valid(&path));
    let _ = std::fs::remove_file(&path);
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// A single melodic line of several sequential notes becomes one voice with
/// several notes, which exercises the count-ended index chain (not just the
/// single-note inline path). Its written save must still fully resolve.
#[test]
fn a_multi_note_melody_builds_a_valid_playhead_and_every_wire_resolves() {
    fn ev(delta: u32, kind: TrackEventKind<'static>) -> TrackEvent<'static> {
        TrackEvent { delta: u28::new(delta), kind }
    }
    // Four sequential quarter notes on one channel -> one voice, four notes.
    let mut track: Track = vec![ev(0, TrackEventKind::Meta(MetaMessage::TrackName(b"Melody")))];
    // Four back-to-back quarter notes: each note-on follows the prior note-off
    // with delta 0, and each note lasts 240 ticks.
    for key in [60u8, 62, 64, 65] {
        track.push(ev(0, TrackEventKind::Midi {
            channel: u4::new(0),
            message: MidiMessage::NoteOn { key: u7::new(key), vel: u7::new(100) },
        }));
        track.push(ev(240, TrackEventKind::Midi {
            channel: u4::new(0),
            message: MidiMessage::NoteOff { key: u7::new(key), vel: u7::new(0) },
        }));
    }
    track.push(ev(0, TrackEventKind::Meta(MetaMessage::EndOfTrack)));
    let smf = Smf {
        header: Header::new(Format::SingleTrack, Timing::Metrical(u15::new(480))),
        tracks: vec![track],
    };
    let mut bytes = Vec::new();
    smf.write(&mut bytes).expect("write");

    let opts = MidiOptions::default();
    let score = analyze_midi(&bytes, &opts).expect("analyze");
    assert_eq!(score.voices.len(), 1, "a monophonic line is one voice");
    assert_eq!(score.voices[0].notes.len(), 4, "with all four notes");

    let world = build_midi_event_world(&score, &opts).expect("build");
    let path = std::env::temp_dir().join(format!("h2b_midi_melody_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    let result = std::panic::catch_unwind(|| wire_integrity::assert_wires_valid(&path));
    let _ = std::fs::remove_file(&path);
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

#[test]
fn a_uniform_tone_gives_every_speaker_the_same_wave() {
    let bytes = two_instrument_smf();
    let opts = MidiOptions {
        tones: ToneAssignment::Uniform(SynthWave::Triangle),
        ..MidiOptions::default()
    };
    let score = analyze_midi(&bytes, &opts).expect("analyze");
    assert!(score.voices.iter().all(|s| s.synth == SynthWave::Triangle));
    build_midi_event_world(&score, &opts).expect("build");
}
