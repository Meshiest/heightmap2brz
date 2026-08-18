//! The drum fold table: a General MIDI percussion note -> a playable sound.
//!
//! The user tuned a full kit of roles in game
//! ([`crate::audio::percussion::PALETTE_ROLES`]) -- kick, snare, clap, hats,
//! cymbals, toms and hand percussion. This table folds every GM percussion
//! note (0..=127, really 35..=81) onto the canonical note of the nearest role,
//! then reads that role's sound. A note that owns a role plays it exactly; the
//! rest fold to the closest one (electric snare -> snare, crash 2 -> crash,
//! floor toms -> low tom, and so on).
//!
//! Retuning is a one-file edit: change `PALETTE_ROLES` and every note that
//! folds to that role follows, because the sound is read through
//! [`role_sound`], never duplicated here.

use crate::audio::percussion::{OneShotSound, PALETTE_ROLES, SNARE};

/// The index into [`PALETTE_ROLES`] of the role that voices `gm_note`.
pub fn role_index(gm_note: u8) -> usize {
    let note = fold_to_role_note(gm_note);
    PALETTE_ROLES.iter().position(|r| r.gm_note == note).unwrap_or(0)
}

/// The sound a percussion note plays: the tuned sound of the role it folds to.
pub fn drum_sound(gm_note: u8) -> OneShotSound {
    PALETTE_ROLES.get(role_index(gm_note)).map(|r| r.sound).unwrap_or(SNARE)
}

/// Like [`drum_sound`], but a per-role override `kit` (indexed like
/// [`PALETTE_ROLES`]) supplies the sounds -- the GUI's editable drum-kit table
/// builds this. A role missing from `kit` (short or empty) keeps its default,
/// so an empty kit reproduces [`drum_sound`] exactly.
pub fn drum_sound_with_kit(gm_note: u8, kit: &[OneShotSound]) -> OneShotSound {
    let ri = role_index(gm_note);
    kit.get(ri)
        .copied()
        .unwrap_or_else(|| PALETTE_ROLES.get(ri).map(|r| r.sound).unwrap_or(SNARE))
}

/// Fold any GM percussion note onto the canonical note of the role that voices
/// it. Notes that own a role map to themselves; the rest map to the nearest.
fn fold_to_role_note(note: u8) -> u8 {
    match note {
        // The eighteen notes that own a role play themselves.
        35 | 36 | 37 | 38 | 39 | 42 | 44 | 45 | 46 | 47 | 49 | 50 | 51 | 53 | 54 | 56 | 70 | 76 => note,
        40 => 38,           // Electric Snare -> Snare
        41 | 43 => 45,      // Floor toms -> Low Tom
        48 => 47,           // Hi-Mid Tom -> Mid Tom
        52 | 55 | 57 => 49, // Chinese / Splash / Crash 2 -> Crash
        59 => 51,           // Ride 2 -> Ride
        60 | 61 => 50,      // Bongos -> High Tom
        62..=66 => 47,      // Congas / timbales -> Mid Tom
        67 | 68 => 56,      // Agogo -> Cowbell
        71 | 72 => 56,      // Whistles -> Cowbell
        69 | 73 | 74 | 82 => 70, // Cabasa / guiro / shaker -> Shaker
        75 | 77 => 76,      // Claves / low wood block -> Woodblock
        78 | 79 => 47,      // Cuica -> Mid Tom
        80 | 81 => 53,      // Triangle -> Ride Bell
        _ => 38,            // Anything else -> Snare
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::percussion::{BELL, CLAP, KICK_1, KICK_2, PALETTE_ROLES};

    /// The five core notes play their exact named sound.
    #[test]
    fn core_notes_play_their_exact_sound() {
        assert_eq!(drum_sound(36), KICK_1, "Bass Drum 1");
        assert_eq!(drum_sound(35), KICK_2, "Acoustic Bass Drum");
        assert_eq!(drum_sound(38), SNARE, "Acoustic Snare");
        assert_eq!(drum_sound(39), CLAP, "Hand Clap");
        assert_eq!(drum_sound(54), BELL, "Tambourine");
    }

    /// Every role's own note plays exactly that role's tuned sound.
    #[test]
    fn each_roled_note_plays_that_roles_tuned_sound() {
        for role in PALETTE_ROLES {
            assert_eq!(
                drum_sound(role.gm_note),
                role.sound,
                "note {} ({}) must play its own sound",
                role.gm_note,
                role.label
            );
        }
    }

    /// A note without a role of its own folds to the nearest role's sound.
    #[test]
    fn a_roleless_note_folds_to_the_nearest_role() {
        assert_eq!(drum_sound(40), drum_sound(38), "Electric Snare -> Snare");
        assert_eq!(drum_sound(57), drum_sound(49), "Crash 2 -> Crash");
        assert_eq!(drum_sound(41), drum_sound(45), "Low Floor Tom -> Low Tom");
    }

    /// A per-role override kit replaces exactly that role's sound; an empty kit
    /// reproduces the defaults; other roles are untouched.
    #[test]
    fn a_kit_override_replaces_only_the_overridden_role() {
        let hat = role_index(42); // Closed Hi-Hat
        let mut kit: Vec<OneShotSound> = PALETTE_ROLES.iter().map(|r| r.sound).collect();
        kit[hat] = KICK_1;

        assert_eq!(drum_sound_with_kit(42, &kit), KICK_1, "closed hat now plays the kick");
        assert_eq!(drum_sound_with_kit(38, &kit), drum_sound(38), "snare role untouched");
        assert_eq!(drum_sound_with_kit(42, &[]), drum_sound(42), "empty kit = defaults");
    }

    /// Every note a MIDI could carry resolves to a sound that is actually in
    /// the tuned kit -- never silence, never an unknown sound.
    #[test]
    fn every_note_plays_a_palette_role_sound() {
        let kit: Vec<OneShotSound> = PALETTE_ROLES.iter().map(|r| r.sound).collect();
        for note in 0u8..=127 {
            let s = drum_sound(note);
            assert!(kit.contains(&s), "note {note} -> {s:?}, not in the tuned kit");
        }
    }
}
