//! Event-based MIDI playback PROTOTYPE, for in-game verification.
//!
//! Writes `./midi_playhead_probe.brz`, a one-speaker circuit that stores eight
//! notes in arrays and steps a runtime "playhead" index through them as the
//! clock advances (see [`heightmap::audio::speakers::build_playhead_probe_world`]).
//!
//! Load it in game and press play. It should sound an ascending 8-note C-major
//! scale, one note per second, then fall silent. If it does, the advancing-index
//! architecture works and midi2brick can be rebuilt on it. Things to watch for:
//! notes advancing in time (the `BufferTicks` self-advance and the dynamic
//! `ArrayVar_Get` index both work), each note at the right pitch, and clean
//! silence in the gaps.
//!
//! Run: `cargo run --example midi_playhead_probe`
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let world = heightmap::audio::speakers::build_playhead_probe_world()?;
    std::fs::write("./midi_playhead_probe.brz", world.to_brz_vec()?)?;
    println!("wrote ./midi_playhead_probe.brz -- load it in game and press play.");
    println!("expect: an ascending 8-note scale, one note per second, then silence.");
    Ok(())
}
