//! Pick a synth waveform from a General MIDI instrument.
//!
//! The MIDI player has four waves ([`SynthWave`]). This maps a GM program
//! number (0..=127) to one, following the user's mixing practice:
//!
//! - **Sine** -- piano, vocals/choir, flutes, bells: pure or breathy tones.
//! - **Sawtooth** -- bass, electric guitar, strings, brass, reeds, pads: rich,
//!   buzzy, sustained.
//! - **Square** -- lead/melody synths and organ: hollow and cutting.
//! - **Triangle** -- acoustic guitar and ethnic plucked: mellow, a softer
//!   melody option.
//!
//! GM programs come in families of eight (`program / 8`): 0 Piano, 1 Chromatic
//! Percussion, 2 Organ, 3 Guitar, 4 Bass, 5 Strings, 6 Ensemble, 7 Brass,
//! 8 Reed, 9 Pipe, 10 Synth Lead, 11 Synth Pad, 12 Synth Effects, 13 Ethnic,
//! 14 Percussive, 15 Sound Effects. A few programs override their family.

use crate::audio::track::SynthWave;

/// The waveform that best voices GM `program`.
pub fn synth_for_program(program: u8) -> SynthWave {
    use SynthWave::*;
    match program {
        // Voices and choirs (incl. Lead 6 "voice" and Pad 4 "choir") -> the
        // vocal tone, whichever family they sit in.
        52..=54 | 85 | 91 => Sine,
        71 => Square,        // Clarinet -- odd-harmonic, square-like
        80 => Square,        // Lead 1 (square)
        81 => Sawtooth,      // Lead 2 (sawtooth)
        26..=31 => Sawtooth, // Electric / overdriven / distortion guitars
        _ => match program / 8 {
            0 => Sine,      // Piano
            1 => Sine,      // Chromatic Percussion (bells, mallets)
            2 => Square,    // Organ
            3 => Triangle,  // Guitar (acoustic nylon/steel reach here)
            4 => Sawtooth,  // Bass
            5 => Sawtooth,  // Strings
            6 => Sawtooth,  // Ensemble
            7 => Sawtooth,  // Brass
            8 => Sawtooth,  // Reed
            9 => Sine,      // Pipe (flutes)
            10 => Square,   // Synth Lead (melody)
            11 => Sawtooth, // Synth Pad
            13 => Triangle, // Ethnic
            // Synth Effects (12), Percussive (14), Sound Effects (15).
            _ => Sine,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One representative program from each family maps to the intended wave.
    #[test]
    fn each_family_maps_to_its_wave() {
        assert_eq!(synth_for_program(0), SynthWave::Sine, "Acoustic Grand Piano");
        assert_eq!(synth_for_program(11), SynthWave::Sine, "Vibraphone (chromatic perc)");
        assert_eq!(synth_for_program(19), SynthWave::Square, "Church Organ");
        assert_eq!(synth_for_program(24), SynthWave::Triangle, "Nylon Guitar (acoustic)");
        assert_eq!(synth_for_program(33), SynthWave::Sawtooth, "Fingered Bass");
        assert_eq!(synth_for_program(40), SynthWave::Sawtooth, "Violin (strings)");
        assert_eq!(synth_for_program(48), SynthWave::Sawtooth, "String Ensemble");
        assert_eq!(synth_for_program(56), SynthWave::Sawtooth, "Trumpet (brass)");
        assert_eq!(synth_for_program(66), SynthWave::Sawtooth, "Tenor Sax (reed)");
        assert_eq!(synth_for_program(73), SynthWave::Sine, "Flute (pipe)");
        assert_eq!(synth_for_program(90), SynthWave::Sawtooth, "Warm Pad");
        assert_eq!(synth_for_program(104), SynthWave::Triangle, "Sitar (ethnic)");
    }

    /// The user's stated preferences, verbatim: bass and electric guitar are
    /// sawtooth, melody/lead is square, piano and vocals are sine.
    #[test]
    fn matches_the_users_mixing_practice() {
        assert_eq!(synth_for_program(33), SynthWave::Sawtooth, "bass -> sawtooth");
        assert_eq!(synth_for_program(27), SynthWave::Sawtooth, "clean electric guitar -> sawtooth");
        assert_eq!(synth_for_program(30), SynthWave::Sawtooth, "distortion guitar -> sawtooth");
        assert_eq!(synth_for_program(80), SynthWave::Square, "square-lead melody -> square");
        assert_eq!(synth_for_program(0), SynthWave::Sine, "piano -> sine");
        assert_eq!(synth_for_program(53), SynthWave::Sine, "Voice Oohs (vocals) -> sine");
    }

    /// Acoustic guitars stay mellow (triangle); electric guitars buzz (sawtooth).
    #[test]
    fn acoustic_and_electric_guitars_differ() {
        assert_eq!(synth_for_program(24), SynthWave::Triangle, "Nylon");
        assert_eq!(synth_for_program(25), SynthWave::Triangle, "Steel");
        assert_eq!(synth_for_program(26), SynthWave::Sawtooth, "Jazz (electric)");
        assert_eq!(synth_for_program(29), SynthWave::Sawtooth, "Overdriven");
    }

    /// The iconic per-program overrides win over their family default.
    #[test]
    fn iconic_instruments_override_their_family() {
        assert_eq!(synth_for_program(71), SynthWave::Square, "Clarinet -> square");
        assert_eq!(synth_for_program(80), SynthWave::Square, "Square Lead");
        assert_eq!(synth_for_program(81), SynthWave::Sawtooth, "Saw Lead");
    }
}
