//! Percussion for the MIDI player.
//!
//! A MIDI channel-10 note selects a drum SOUND, not a pitch, so percussion is
//! built from `Component_OneShotAudioEmitter` samples pulsed on their `Play`
//! port -- not the pitched synth emitters the melodic voices use
//! ([`super::speakers`]).
//!
//! This module currently provides the audition PALETTE: a labelled row of
//! candidate sounds, each a `B_Button` wired into a `B_1x1_SoundEmitter`, so
//! the sounds can be auditioned and tuned in game and the tuned save handed
//! back. The player's fold table and trigger circuit will build on the same
//! [`add_oneshot_emitter`] helper.
//!
//! # Asset registration is UNVALIDATED, on purpose
//!
//! [`super::speakers::audio_descriptor_value`] checks a name against brdb's
//! baked `ASSET_TYPES` catalog. That catalog is INCOMPLETE for
//! `BrickOneShotAudioDescriptor`: the `OSA_*` action sounds are missing from
//! it, including two the user's own reference prefab uses
//! (`OSA_SubmachineGun_Reload_BoltSlap`, `OSA_FragGrenade_Bounce`). The game
//! resolves the name at load regardless, so this module registers the raw
//! string -- the same mechanism the reference prefab loads through.
//!
//! # One thing this cannot verify -- confirm on a test render
//!
//! That `Button.bHeld -> emitter.Play` actually fires the oneshot. The whole
//! reason the palette ships first is to answer that in game before the
//! player's trigger circuit is built on it.

use crate::text::{
    ANCHOR_CENTRE, DEFAULT_OUTLINE_WIDTH, FACE_Z_POSITIVE, FontPreset, OUTLINE_OUTLINED,
    text_label_component,
};
use brdb::{
    AsBrdbValue, Brick, BrickType, Color, Position, WirePort, World, assets::LiteralComponent,
    schema::BrdbValue,
};
use std::collections::HashMap;

/// The oneshot emitter component and the brick that carries it (the user's
/// reference prefab uses this brick).
pub const ONESHOT_EMITTER: &str = "Component_OneShotAudioEmitter";
const SOUND_EMITTER_BRICK: &str = "B_1x1_SoundEmitter";
/// The external-asset type of a oneshot sound. See the module note on why the
/// name is not validated against brdb's catalog.
const ONESHOT_ASSET_TYPE: &str = "BrickOneShotAudioDescriptor";
/// The exec input that fires the sample: any non-zero value or increment plays
/// it once (so the percussion playhead's incrementing index drives it).
pub const PLAY_PORT: &str = "Play";

/// The animated push-button that visibly depresses when held -- the same one
/// [`crate::anim::controls`] uses. Its `bHeld` output drives the emitter.
const BUTTON: &str = "Component_Internal_AnimatedButton";
const BUTTON_BRICK: &str = "B_Button";
const BUTTON_HELD: &str = "bHeld";
const BUTTON_PROMPT_LABEL: &str = "PromptCustomLabel";

/// Emitter falloff, copied from the reference prefab: full volume within
/// `INNER_RADIUS` units, silent past `MAX_DISTANCE`.
const INNER_RADIUS: f32 = 100.0;
const MAX_DISTANCE: f32 = 400.0;

/// The height of a palette label, in units.
const LABEL_LINE_HEIGHT: f32 = 2.0;

/// A oneshot drum sound: the game asset plus its baked pitch and volume.
///
/// `pitch` is the emitter's `PitchMultiplier` (1.0 = the sample's own pitch);
/// `volume` is its `VolumeMultiplier`. Both are what the reference prefab
/// tuned per sound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OneShotSound {
    pub asset: &'static str,
    pub pitch: f32,
    pub volume: f32,
}

/// One candidate row in the audition palette: a human role label, the General
/// MIDI percussion note it stands for, whether the sound is one the user
/// already approved, and the sound itself.
///
/// `approved == false` marks a PLACEHOLDER guess -- a sound picked only so the
/// role has something to audition. No placeholder is ever used in a MIDI
/// mapping; the fold table draws only on approved sounds until the user
/// resubmits the tuned palette.
///
/// `gm_note` is the CANONICAL note for the role (e.g. 42 = Closed Hi-Hat). The
/// reference MIDIs play it, and the fold table anchors the role there; the fold
/// table additionally routes the notes with no role of their own.
pub struct PaletteRole {
    pub label: &'static str,
    pub gm_note: u8,
    pub approved: bool,
    pub sound: OneShotSound,
}

const fn sound(asset: &'static str, pitch: f32, volume: f32) -> OneShotSound {
    OneShotSound { asset, pitch, volume }
}

const fn role(label: &'static str, gm_note: u8, approved: bool, sound: OneShotSound) -> PaletteRole {
    PaletteRole { label, gm_note, approved, sound }
}

// The user's five original sounds. Named because the palette references them
// directly; the other thirteen roles are tuned inline in `PALETTE_ROLES`. The
// fold table reads every role's sound through `role_sound`. See
// `crate::midi::drums`.
pub const KICK_1: OneShotSound = sound("OBA_Door_Knock_Hard", 0.15, 2.0);
pub const KICK_2: OneShotSound = sound("OBA_Door_Knock_Soft", 0.2, 2.0);
pub const SNARE: OneShotSound = sound("OBA_Air_Brake", 1.3, 1.0);
pub const CLAP: OneShotSound = sound("OSA_SubmachineGun_Reload_BoltSlap", 0.6, 1.0);
pub const BELL: OneShotSound = sound("OSA_FragGrenade_Bounce", 1.8, 2.0);

/// The tuned sound for the role that owns `gm_note`, if any role does.
///
/// The drum fold table folds every percussion note onto a role's canonical
/// note, then reads the sound here, so retuning the palette (which updates
/// [`PALETTE_ROLES`]) is the only edit needed to change what a note plays.
pub fn role_sound(gm_note: u8) -> Option<OneShotSound> {
    PALETTE_ROLES.iter().find(|r| r.gm_note == gm_note).map(|r| r.sound)
}

/// A basic General MIDI drum kit's worth of roles for the user to audition.
///
/// The five `approved` rows are the user's reference-prefab sounds. Every
/// other row is a `guess`: a placeholder the user is expected to replace. The
/// pitch/volume on the guesses are starting points only.
pub const PALETTE_ROLES: &[PaletteRole] = &[
    // The five already dialled in, as a reference to tune the rest against.
    role("Kick 1", 36, true, KICK_1),
    role("Kick 2", 35, true, KICK_2),
    role("Snare", 38, true, SNARE),
    role("Clap", 39, true, CLAP),
    role("Bell", 54, true, BELL),
    // Tuned by the user in game and resubmitted -- now the drum kit proper.
    role("Closed HiHat", 42, true, sound("OBA_GlasShatter_3", 1.2, 0.5)),
    role("Pedal HiHat", 44, true, sound("OBA_GlasShatter_3", 1.2, 0.5)),
    role("Open HiHat", 46, true, sound("OBA_GlasShatter_1", 2.0, 1.0)),
    role("Crash", 49, true, sound("OBA_GlasShatter_1", 1.0, 1.0)),
    role("Ride", 51, true, sound("OBA_GlasShatter_1", 2.0, 1.0)),
    role("Ride Bell", 53, true, sound("OBA_GlasShatter_1", 2.0, 1.0)),
    role("Low Tom", 45, true, sound("BOSA_Buttons_Button_2_Press", 0.3, 0.9)),
    role("Mid Tom", 47, true, sound("BOSA_Buttons_Button_2_Press", 0.45, 0.9)),
    role("High Tom", 50, true, sound("BOSA_Buttons_Button_2_Press", 0.6, 0.9)),
    role("Side Stick", 37, true, sound("OSA_Derringer_Fire", 1.0, 0.3)),
    role("Cowbell", 56, true, sound("BOSA_Buttons_Button_1_Press", 0.9, 1.0)),
    role("Woodblock", 76, true, sound("OBA_Door_Knock_Soft", 1.9, 1.0)),
    role("Shaker", 70, true, sound("OSA_HeavyAssaultRifle_Reload_BoltRelease", 1.8, 0.8)),
];

/// Register a oneshot sound asset and return the component-field value that
/// points at it. See the module note: the name is NOT validated.
fn oneshot_descriptor_value(world: &mut World, asset: &str) -> Box<dyn AsBrdbValue> {
    let (idx, _) = world.global_data.external_asset_references.insert_full((
        ONESHOT_ASSET_TYPE.to_string(),
        asset.to_string(),
    ));
    Box::new(BrdbValue::Asset(Some(idx))) as Box<dyn AsBrdbValue>
}

/// Place a `B_1x1_SoundEmitter` carrying a `Component_OneShotAudioEmitter` for
/// `sound`, and return its brick id (for wiring its `Play` port).
///
/// `bEnableRepeat` is false and `RepeatCount` 1: one sample per `Play` pulse,
/// as the reference prefab sets.
pub fn add_oneshot_emitter(world: &mut World, sound: &OneShotSound, position: Position) -> usize {
    let mut data: HashMap<brdb::BString, Box<dyn AsBrdbValue>> = HashMap::new();
    data.insert("AudioDescriptor".into(), oneshot_descriptor_value(world, sound.asset));
    data.insert("PitchMultiplier".into(), Box::new(sound.pitch));
    data.insert("VolumeMultiplier".into(), Box::new(sound.volume));
    data.insert("InnerRadius".into(), Box::new(INNER_RADIUS));
    data.insert("MaxDistance".into(), Box::new(MAX_DISTANCE));
    data.insert("bSpatialization".into(), Box::new(true));
    data.insert("bEnableRepeat".into(), Box::new(false));
    data.insert("RepeatCount".into(), Box::new(1i32));

    let (brick, id) = Brick {
        asset: BrickType::from(SOUND_EMITTER_BRICK),
        position,
        ..Default::default()
    }
    .with_component(LiteralComponent::new_from_data(ONESHOT_EMITTER, std::sync::Arc::new(data)))
    .with_id_split();
    world.add_brick(brick);
    id
}

/// Green for a sound the user already approved, amber for a placeholder guess.
const APPROVED_COLOR: Color = Color { r: 60, g: 200, b: 90 };
const GUESS_COLOR: Color = Color { r: 235, g: 170, b: 60 };

/// Units between the button and its emitter within one cell, and between cells
/// down the row. Generous so no two bricks overlap whatever their footprints.
const CELL_GAP: i32 = 30;

/// Build the audition palette: one row cell per role, each a labelled `B_Button`
/// wired to its own oneshot emitter.
///
/// Cells run down +Y. In each, the button sits at the row's X and its emitter
/// `CELL_GAP` further along +X; the button's `bHeld` output feeds the emitter's
/// `Play`, so a press auditions the sound. The button is green for an approved
/// sound and amber for a placeholder, and its on-look prompt says which.
pub fn build_drum_palette(roles: &[PaletteRole]) -> World {
    let mut world = World::new();
    world.meta.bundle.description = format!(
        "Drum audition palette -- press each button to play its sound. Green = approved, \
         amber = placeholder to replace. {} sounds.",
        roles.len()
    );
    let label_opts = FontPreset::MonaspaceArgon.options(1.0);

    for (i, role) in roles.iter().enumerate() {
        let y = i as i32 * CELL_GAP;
        let button_pos = Position { x: 0, y, z: 6 };
        let emitter_pos = Position { x: CELL_GAP, y, z: 6 };

        let emitter_id = add_oneshot_emitter(&mut world, &role.sound, emitter_pos);

        // The visible role label on the button's top face.
        let label = text_label_component(
            &mut world,
            role.label.to_string(),
            LABEL_LINE_HEIGHT,
            FACE_Z_POSITIVE,
            ANCHOR_CENTRE,
            OUTLINE_OUTLINED,
            DEFAULT_OUTLINE_WIDTH,
            &label_opts,
        );

        let prompt = if role.approved {
            format!("Play {}", role.label)
        } else {
            format!("Play {} [PLACEHOLDER - replace]", role.label)
        };
        let (button, button_id) = Brick {
            asset: BrickType::from(BUTTON_BRICK),
            position: button_pos,
            color: if role.approved { APPROVED_COLOR } else { GUESS_COLOR },
            ..Default::default()
        }
        .with_component(LiteralComponent::new(BUTTON).with_data([(
            BUTTON_PROMPT_LABEL,
            Box::new(prompt) as Box<dyn AsBrdbValue>,
        )]))
        .with_component(label)
        .with_id_split();
        world.add_brick(button);

        // Press -> play. The single in-game unknown (see the module note).
        world.add_wire_connection(
            WirePort::new(button_id, BUTTON, BUTTON_HELD),
            WirePort::new(emitter_id, ONESHOT_EMITTER, PLAY_PORT),
        );
    }

    // Must be last: it records only the component types already placed above.
    world.register_used_components();
    world
}

#[cfg(test)]
mod tests {
    use super::*;
    use brdb::{Brz, IntoReader};

    /// Round-trip the world to a real `.brz` and return every
    /// `BrickOneShotAudioDescriptor` asset it carries -- what the game sees,
    /// not an in-memory peek (`.brz` bytes are not reproducible).
    fn written_oneshot_assets(w: &World) -> Vec<String> {
        let bytes = w.to_brz_vec().expect("the world must encode to brz");
        let global = Brz::read_slice(&bytes)
            .expect("the written brz must parse")
            .into_reader()
            .read_global_data()
            .expect("the written brz must carry global data");
        global
            .external_asset_references
            .iter()
            .filter(|(ty, _)| ty.as_str() == ONESHOT_ASSET_TYPE)
            .map(|(_, name)| name.clone())
            .collect()
    }

    /// Sum the wire count across the main grid's chunks in a written save.
    fn written_wire_count(w: &World) -> u32 {
        let bytes = w.to_brz_vec().expect("the world must encode to brz");
        let reader = Brz::read_slice(&bytes).expect("the written brz must parse").into_reader();
        reader
            .brick_chunk_index(1)
            .expect("the main grid must have a chunk index")
            .iter()
            .map(|c| c.num_wires)
            .sum()
    }

    /// Every role's sound must be baked into the save, or that palette cell is
    /// silent and the user cannot audition it.
    #[test]
    fn every_palette_role_bakes_its_candidate_sound_into_the_save() {
        let world = build_drum_palette(PALETTE_ROLES);
        let assets = written_oneshot_assets(&world);
        for role in PALETTE_ROLES {
            assert!(
                assets.iter().any(|a| a == role.sound.asset),
                "palette save is missing sound {:?} for role {:?}",
                role.sound.asset,
                role.label
            );
        }
    }

    /// Each cell must carry exactly one Button -> Play wire, or the button
    /// does nothing when pressed and the palette cannot be auditioned.
    #[test]
    fn each_cell_wires_its_button_to_its_emitter() {
        let world = build_drum_palette(PALETTE_ROLES);
        assert_eq!(
            written_wire_count(&world),
            PALETTE_ROLES.len() as u32,
            "expected one button->Play wire per role"
        );
    }
}
