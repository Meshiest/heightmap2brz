# MIDI Percussion

How the MIDI player builds drums, and how to change the sounds it uses.

A MIDI channel-10 note picks a drum *sound*, not a pitch, so percussion is not
built from the pitched synth speakers the melodic tracks use. Each drum sound
becomes one `Component_OneShotAudioEmitter` (`B_1x1_SoundEmitter`), pulsed on its
`Play` port by a playhead: a `Var` index steps through that sound's strike
times, and each increment fires the sample (`Play` is an exec input, so any
increment triggers it). The emitters sit in a block beside the speaker cluster;
the playhead gates share the melodic clock and scaffold, so one save plays melody
and drums together.

Percussion builds by default. `--no-percussion` (CLI) or the GUI **Percussion**
toggle skips it.

## The fold table

The user has a curated kit of sounds; a General MIDI drum kit uses ~47 note
slots. [`src/midi/drums.rs`](../src/midi/drums.rs) folds every percussion note
onto one of the kit's roles, then plays that role's sound:

- `role_index(note)` folds a GM note to the nearest role (electric snare to
  snare, crash 2 to crash, floor toms to low tom, ...).
- `drum_sound(note)` returns that role's tuned sound.
- `drum_sound_with_kit(note, kit)` is the same, but a per-role override kit
  (indexed like `PALETTE_ROLES`) supplies the sounds. The GUI drum-kit table
  fills this; an empty kit reproduces the defaults.

The roles, their canonical GM notes, and their tuned default sounds live in
`PALETTE_ROLES` in [`src/audio/percussion.rs`](../src/audio/percussion.rs). The
full catalog of oneshot sound names the GUI dropdown offers is
[`src/audio/oneshot_sounds.rs`](../src/audio/oneshot_sounds.rs) (generated from
the game asset catalog).

## The audition round-trip

The tool cannot play the game's oneshot samples, so the sounds are tuned in
game. The loop that updates the **built-in** defaults (`PALETTE_ROLES`):

1. **Generate the palette.** The GUI drum-kit accordion has a *Generate audition
   palette* button (builds the current kit, saves a `.brz` prefab, and copies it
   to the clipboard on desktop). The same prefab comes from
   `cargo run --example drum_palette -- <out.brz>`. Each role is a labelled
   button wired to its oneshot emitter.
2. **Tune in game.** Load the prefab, press each button to audition, and edit the
   emitter's sound / pitch / volume until it is right. Save the result.
3. **Send the save back** to be baked into the built-in defaults.
4. **Extract and bake.** `cargo run --example extract_drum_palette -- <save.brz>`
   reads each cell back (role label paired with its emitter's sound, via the
   brick the label and sound share). The recovered role -> sound table replaces
   `PALETTE_ROLES`, and the fold-table and palette tests are re-run.

`extract_drum_palette` uses the generic reader technique documented in the brdb
repo (`docs/extracting-components.md`): pair each component to its brick through
`component_brick_indices`, then read the `AudioDescriptor` (resolved through the
save's external asset references), `PitchMultiplier`, `VolumeMultiplier`, and the
co-located `TextDisplay` text.

## The GUI drum-kit table

The MIDI pane's **Drum kit** accordion is an editable, per-session kit (it does
not change the built-in defaults; only the round-trip above does):

- **Sound**: a filterable dropdown over every oneshot sound in the catalog.
- **Pitch / Vol**: number inputs.

Edits feed `MidiOptions::drum_kit`, so the built save uses them. The CLI has no
kit override and always uses the baked defaults.

## What builds where

| Piece | File |
|-------|------|
| Parse channel 10 into strikes | `src/midi/parse.rs` (`PercussionHit`) |
| Schedule strikes into per-sound lanes | `src/midi/schedule.rs` (`PercussionLane`) |
| Fold table | `src/midi/drums.rs` |
| Roles + default sounds | `src/audio/percussion.rs` (`PALETTE_ROLES`) |
| Oneshot sound catalog | `src/audio/oneshot_sounds.rs` |
| Emitter + playhead circuit | `src/audio/speakers.rs` (`build_midi_event_world`) |
| Palette generator / extractor | `examples/drum_palette.rs`, `examples/extract_drum_palette.rs` |
