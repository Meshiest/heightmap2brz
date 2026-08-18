//! Write one reference `.mid` per MISSING drum role, so the real General MIDI
//! sound can be heard in any GM player and used as the tuning target for the
//! palette placeholder.
//!
//! ```sh
//! cargo run --example drum_reference_midis -- <out-dir>
//! ```
//!
//! Each file plays its role's GM percussion note (channel 10) ten times, one
//! second apart. Only the placeholder roles are written -- the five approved
//! sounds already have a target. Roles and notes come from
//! [`heightmap::audio::percussion::PALETTE_ROLES`].

use heightmap::audio::percussion::PALETTE_ROLES;
use midly::num::{u4, u7, u15, u24, u28};
use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind};

/// Ticks per quarter note.
const PPQ: u16 = 480;
/// 120 BPM: half a second per quarter, so one second is two quarters.
const US_PER_QUARTER: u32 = 500_000;
/// One second between successive hits (NoteOn to NoteOn).
const TICKS_PER_SECOND: u32 = PPQ as u32 * 2;
/// How long each struck note is held before its NoteOff. Percussion ignores
/// note length, so this is short; the rest of the second is silence.
const NOTE_LEN: u32 = 120;
/// The MIDI percussion channel, zero-based ("channel 10").
const DRUM_CHANNEL: u8 = 9;
const HITS: usize = 10;
const VELOCITY: u8 = 100;

fn note_on(delta: u32, note: u8) -> TrackEvent<'static> {
    TrackEvent {
        delta: u28::new(delta),
        kind: TrackEventKind::Midi {
            channel: u4::new(DRUM_CHANNEL),
            message: MidiMessage::NoteOn { key: u7::new(note), vel: u7::new(VELOCITY) },
        },
    }
}

fn note_off(delta: u32, note: u8) -> TrackEvent<'static> {
    TrackEvent {
        delta: u28::new(delta),
        kind: TrackEventKind::Midi {
            channel: u4::new(DRUM_CHANNEL),
            message: MidiMessage::NoteOff { key: u7::new(note), vel: u7::new(0) },
        },
    }
}

/// A single-track SMF: `note` struck `HITS` times, one second apart, on the
/// percussion channel, labelled `name`.
fn reference_midi(name: &'static str, note: u8) -> Smf<'static> {
    let mut track = vec![
        TrackEvent { delta: u28::new(0), kind: TrackEventKind::Meta(MetaMessage::TrackName(name.as_bytes())) },
        TrackEvent { delta: u28::new(0), kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(US_PER_QUARTER))) },
    ];
    for i in 0..HITS {
        // The first hit is at time 0; each later hit follows the previous
        // NoteOff by the rest of its one-second slot.
        let lead = if i == 0 { 0 } else { TICKS_PER_SECOND - NOTE_LEN };
        track.push(note_on(lead, note));
        track.push(note_off(NOTE_LEN, note));
    }
    track.push(TrackEvent { delta: u28::new(0), kind: TrackEventKind::Meta(MetaMessage::EndOfTrack) });

    let mut smf = Smf::new(Header::new(Format::SingleTrack, Timing::Metrical(u15::new(PPQ))));
    smf.tracks.push(track);
    smf
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "drum_reference_midis".to_string());
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("cannot create {dir}: {e}");
        std::process::exit(1);
    }

    let mut written = 0;
    for role in PALETTE_ROLES.iter().filter(|r| !r.approved) {
        let file = format!("{dir}/{:02}_{}.mid", role.gm_note, role.label.replace(' ', "_"));
        match reference_midi(role.label, role.gm_note).save(&file) {
            Ok(()) => {
                println!("{file}  (GM note {}, x{HITS} @ 1s)", role.gm_note);
                written += 1;
            }
            Err(e) => {
                eprintln!("failed to write {file}: {e}");
                std::process::exit(1);
            }
        }
    }
    println!("wrote {written} reference MIDIs to {dir}/");
}
