//! Five minimal saves that isolate the four untested assumptions the audio
//! renderer rests on.
//!
//! The analysis chain is measured and correct (a synthetic 1053 Hz tone lands
//! 99.73% in the right band; the band-balance fix moved white noise from
//! leading 89.81% of frames to 2.57%), yet the render does not sound like the
//! source. So the fault is almost certainly in how the save DRIVES the game,
//! and these things have been load-bearing since the design without ever being
//! checked in-game:
//!
//!   1. a `Component_AudioEmitter` with a baked `AudioDescriptor` makes sound
//!      at all, with no wire graph anywhere;
//!   2. `PitchMultiplier` is a linear frequency-rate multiplier on a 440 Hz
//!      base, so a band's pitch is the note it plays;
//!   3. writing `VolumeMultiplier` from a wire every frame CHANGES the level of
//!      a running voice, rather than retriggering it;
//!   4. writing `PitchMultiplier` from a wire every frame RE-TUNES a running
//!      voice, rather than restarting it.
//!
//! Assumption 3 has been confirmed in game (diagnostic 3 fades smoothly).
//! **Assumption 4 has not**, and it is the one the `--audio-mode voice`
//! renderer rests on entirely: that mode parks a handful of speakers on
//! detected spectral peaks and re-pitches them every frame, so a pitch write
//! that retriggers would make it a 30 Hz click train. Diagnostic 5 is the test.
//!
//! # Why three of these five saves have no microchip in them at all
//!
//! Tests 1, 2 and 4 contain zero wires and zero brick grids -- nothing but
//! speaker bricks carrying baked component data. That is the entire point of
//! them: if a save with no wire graph whatsoever fails to make sound, the
//! failure is in the emitter, the asset reference or the pitch setup, and the
//! whole wire-graph layer (clock, banks, arrays, selects, the remote wires
//! across the chip boundary) is exonerated in one step. Adding a chip "just to
//! drive something" would destroy that property, so `assert_chipless` checks it
//! in code rather than leaving it to inspection.
//!
//! Only tests 3 and 5 need a chip, because the thing they test IS the chip
//! driving a speaker.
//!
//! Built entirely on the crate's public API; nothing under `src/` is touched.
use brdb::{
    AsBrdbValue, BString, Brick, BrickType, IntVector, IntoReader, Position, Vector3f, WirePort,
    World,
    assets::{
        LiteralComponent,
        external::{ASSET_TYPES, BA_SYNTH_BASIC_SINE, BA_SYNTH_NOISE_PINK, BA_SYNTH_NOISE_WHITE},
    },
    schema::{BrdbStruct, BrdbValue, WireArrayVariant},
};
use heightmap::anim::bricks::{ARRAY_GET, ARRAY_VAR, CHANGE_DETECTOR};
use heightmap::anim::chip::{finish, new_chip};
use heightmap::anim::clock::{build_clock, gate};
use heightmap::anim::layout::{
    GATE_HALF, STAGE_PITCH, assert_bricks_dont_overlap, lattice_pos_staged,
};
use heightmap::audio::bands::{BandKind, BandPlan, PITCH_MAX, PITCH_MIN};
use heightmap::audio::speakers::{
    AUDIO_ASSET_TYPE, AUDIO_EMITTER, DEFAULT_INNER_RADIUS, DEFAULT_MAX_DISTANCE, SPEAKER_BRICK,
    speaker_half, speaker_position,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Baked attenuation radii -- IMPORTED from `speakers.rs`, never restated.
///
/// They are NOT inert while `bSpatialization` is false: that flag disables
/// panning only, and distance attenuation applies regardless, which is how a
/// bank spread over 372 units came to be audible only in slices. A diagnostic
/// emitter must therefore be identical to a render emitter in every field
/// except the one under test -- a diagnostic that differs from the thing it is
/// diagnosing cannot exonerate it, and a hardcoded copy of these numbers here
/// would drift the moment the renderer's defaults were retuned.
const INNER_RADIUS: f32 = DEFAULT_INNER_RADIUS;
const MAX_DISTANCE: f32 = DEFAULT_MAX_DISTANCE;

/// The game's default is `true`. Writing it explicitly is not redundancy:
/// omitting the field turns 3D panning ON, and every save here would then be
/// measuring comb filtering instead of what it means to measure.
const SPATIALIZATION: bool = false;

/// Diagnostic 3's clock. 60 frames at 30 fps is a 2-second triangle, the same
/// per-frame write rate a real render uses.
const RAMP_FRAMES: usize = 60;
const RAMP_FPS: f32 = 30.0;

/// Lattice "height" fed to `lattice_pos_staged`, matching `speakers.rs`: this
/// chip has no pixel lattice, so 1 is the smallest value that puts row 0 at the
/// front and keeps every row index <= 0 mapped onto POSITIVE x.
///
/// **Negative inner-grid coordinates delete bricks in-game, silently.** This is
/// the reason every position below goes through `service` instead of being
/// written as a coordinate.
const LATTICE_HEIGHT: i32 = 1;

// ---------------------------------------------------------------------------
// Speaker construction
// ---------------------------------------------------------------------------

/// One emitter's audible settings.
///
/// `asset` is a [`BString`] rather than a `&'static str` because brdb's catalog
/// entries are `const BString`s: taking `.as_ref()` on one materialises a
/// temporary that cannot outlive the statement. Holding the value keeps the
/// asset names sourced from brdb's own catalog, which is what stops a typo
/// producing a save that encodes perfectly and is silent only in game.
struct SpeakerSpec {
    /// A `BrickAudioDescriptor` asset name.
    asset: BString,
    pitch: f32,
    volume: f32,
}

/// Resolve a synth asset name to the value `AudioDescriptor` expects, checking
/// it against brdb's catalog first.
///
/// Mirrors `speakers.rs`'s private `audio_descriptor_value`. Nothing on brdb's
/// write path compares the stored asset type against the schema's, so a typo in
/// either string encodes perfectly and produces a save that is silent only in
/// game -- which is exactly the failure mode these diagnostics exist to rule
/// out, so it must not be reintroduced by the diagnostics themselves.
///
/// The value is a [`BrdbValue::Asset`] carrying the reference's index. NOT a
/// `WireVariant::Object` -- that is the wire-graph value union and a different
/// encoding path entirely, and it also compiles.
fn descriptor_value(world: &mut World, asset_name: &str) -> Result<Box<dyn AsBrdbValue>, String> {
    let (_, known) = ASSET_TYPES
        .iter()
        .find(|(ty, _)| *ty == AUDIO_ASSET_TYPE)
        .ok_or_else(|| format!("brdb knows no external asset type {AUDIO_ASSET_TYPE:?}"))?;
    if !known.iter().any(|a| a.as_ref() == asset_name) {
        return Err(format!(
            "{asset_name:?} is not a {AUDIO_ASSET_TYPE} asset in brdb's catalog"
        ));
    }
    let (index, _) = world
        .global_data
        .external_asset_references
        .insert_full((AUDIO_ASSET_TYPE.to_string(), asset_name.to_string()));
    Ok(Box::new(BrdbValue::Asset(Some(index))) as Box<dyn AsBrdbValue>)
}

/// Add one speaker to the main grid at cluster slot `slot` of `total`,
/// returning its brick id.
///
/// The position comes from `speakers::speaker_position`, the renderer's OWN
/// layout function, so diagnostic 4 is the real bank's geometry and not a
/// lookalike. That matters more than it sounds: position is audible (see
/// `speakers::cluster_dims`), so a diagnostic laid out differently from the
/// renderer is measuring a different instrument.
///
/// `PitchMultiplier` and `VolumeMultiplier` are bare `f32` fields on
/// `BrickComponentData_AudioEmitter`, NOT the `WireVariant` tagged union that
/// wire-graph gate ports use. A `WireVariant::Number` here compiles and then
/// dies at encode with `UnimplementedCast("f32", WireVariant)`.
fn add_speaker(
    world: &mut World,
    spec: &SpeakerSpec,
    slot: usize,
    total: usize,
) -> Result<usize, String> {
    let descriptor = descriptor_value(world, spec.asset.as_ref())?;

    let mut data: HashMap<BString, Box<dyn AsBrdbValue>> = HashMap::new();
    data.insert(
        "PitchMultiplier".into(),
        Box::new(spec.pitch) as Box<dyn AsBrdbValue>,
    );
    data.insert(
        "VolumeMultiplier".into(),
        Box::new(spec.volume) as Box<dyn AsBrdbValue>,
    );
    data.insert(
        "bSpatialization".into(),
        Box::new(SPATIALIZATION) as Box<dyn AsBrdbValue>,
    );
    data.insert("bEnabled".into(), Box::new(true) as Box<dyn AsBrdbValue>);
    data.insert(
        "InnerRadius".into(),
        Box::new(INNER_RADIUS) as Box<dyn AsBrdbValue>,
    );
    data.insert(
        "MaxDistance".into(),
        Box::new(MAX_DISTANCE) as Box<dyn AsBrdbValue>,
    );
    data.insert("AudioDescriptor".into(), descriptor);

    let (brick, id) = Brick {
        asset: BrickType::from(SPEAKER_BRICK),
        position: speaker_position(slot, total),
        ..Default::default()
    }
    .with_component(LiteralComponent::new_from_data(
        AUDIO_EMITTER,
        Arc::new(data),
    ))
    .with_id_split();
    world.add_brick(brick);
    Ok(id)
}

/// A world holding nothing but `specs`' speakers: no chip, no wires.
fn chipless_world(description: &str, specs: &[SpeakerSpec]) -> Result<World, String> {
    let mut world = World::new();
    world.meta.bundle.description = description.to_string();
    for (slot, spec) in specs.iter().enumerate() {
        add_speaker(&mut world, spec, slot, specs.len())?;
    }
    // `chip::finish` normally runs this; with no chip there is nothing to call
    // it, and an overlap on the main grid makes the game silently DROP one of
    // the two bricks -- which would look exactly like "the emitter made no
    // sound".
    assert_bricks_dont_overlap(&world.bricks)?;
    // Must be LAST, after every brick exists: it walks `world.bricks` and
    // `world.grids` and registers only the component types it can see.
    world.register_used_components();
    Ok(world)
}

/// The property that makes tests 1, 2 and 4 diagnostic in the first place.
///
/// Checked in code, not by eye: a chip or a wire sneaking into one of these
/// saves would leave it looking identical from the outside while quietly
/// destroying its ability to exonerate the wire-graph layer.
fn assert_chipless(world: &World, name: &str) {
    assert!(
        world.grids.is_empty(),
        "{name} must contain NO brick grids (i.e. no microchip), found {} -- \
         a save with a chip in it cannot exonerate the wire-graph layer",
        world.grids.len()
    );
    assert!(
        world.wires.is_empty(),
        "{name} must contain NO wires, found {} -- the whole point of this save \
         is that nothing drives the speaker at runtime",
        world.wires.len()
    );
}

// ---------------------------------------------------------------------------
// The four diagnostics
// ---------------------------------------------------------------------------

/// 1. ONE speaker, sine, unity pitch, half volume, nothing driving it.
fn diag_1_static_tone() -> Result<World, String> {
    chipless_world(
        "Audio diagnostic 1: one speaker with a baked descriptor, no wire graph",
        &specs_1(),
    )
}

fn specs_1() -> Vec<SpeakerSpec> {
    vec![SpeakerSpec {
        asset: BA_SYNTH_BASIC_SINE,
        pitch: 1.0,
        volume: 0.5,
    }]
}

/// 2. THREE speakers at a major triad's pitch ratios.
///
/// `1.25992` is 2^(4/12) and `1.49831` is 2^(7/12) -- a major third and a
/// perfect fifth, IF `PitchMultiplier` is a linear rate multiplier. Those two
/// numbers are the test: they are only a major chord under that assumption.
fn diag_2_static_chord() -> Result<World, String> {
    chipless_world(
        "Audio diagnostic 2: three speakers at major-triad pitch ratios, no wire graph",
        &specs_2(),
    )
}

fn specs_2() -> Vec<SpeakerSpec> {
    [1.0f32, 1.25992, 1.49831]
        .into_iter()
        .map(|pitch| SpeakerSpec {
            asset: BA_SYNTH_BASIC_SINE,
            pitch,
            volume: 0.3,
        })
        .collect()
}

/// 3. ONE speaker whose volume is written every frame by a chip.
///
/// **The critical one for the band bank.** See [`one_driven_port`] for the
/// graph; the port under test is `VolumeMultiplier`.
fn diag_3_volume_ramp() -> Result<World, String> {
    one_driven_port(
        "Audio diagnostic 3: one speaker whose volume is written every frame by a chip",
        &specs_3()[0],
        "VolumeMultiplier",
        ramp_values(),
    )
}

/// ONE speaker with ONE of its emitter ports written every frame by a chip.
///
/// The graph is exactly the shape `build_speaker_world` uses, minus everything
/// that is not under test: clock -> change detector -> `ArrayVar_Get` (indexed
/// by the frame index, reading an `ArrayVar` of `f64`) -> straight into `port`.
/// No bank cascade, no master-volume multiply, no attenuation pins -- one
/// speaker, one array, one wire.
///
/// Shared by diagnostics 3 and 5 rather than copied, so the only difference
/// between "does a volume write retrigger" and "does a pitch write retrigger"
/// is the port name and the array's contents. A second hand-written copy could
/// differ somewhere else and the comparison would stop meaning anything.
fn one_driven_port(
    description: &str,
    spec: &SpeakerSpec,
    port: &'static str,
    values: Vec<f64>,
) -> Result<World, String> {
    let mut world = World::new();
    world.meta.bundle.description = description.to_string();

    let speaker = add_speaker(&mut world, spec, 0, 1)?;

    // Beside the speaker on X, never stacked on it: an overlap on the main
    // grid makes the game silently drop one of the two bricks. Matches
    // `build_speaker_world`'s own clearance -- the cluster's low corner is the
    // origin, so a shell at negative x clears it for any speaker count.
    let chip_x = -(speaker_half().x * 4);
    let mut chip = new_chip(
        &mut world,
        Position {
            x: chip_x,
            y: 0,
            z: 2,
        },
        Vector3f {
            x: chip_x as f32,
            y: 0.0,
            z: 40.0,
        },
        IntVector { x: 10, y: 10, z: 2 },
    );

    // Row indices are NEGATIVE going back through the lattice, but `service`
    // maps them onto POSITIVE x -- see LATTICE_HEIGHT. Never write an inner
    // coordinate directly.
    let service =
        |col: i32, row: i32| lattice_pos_staged(col, row, 0, LATTICE_HEIGHT, GATE_HALF, STAGE_PITCH);

    let clock = build_clock(
        &mut world,
        &mut chip,
        RAMP_FPS,
        RAMP_FRAMES,
        true,
        service(0, 0),
    );

    let detector = gate(
        &mut chip,
        "B_1x1_Gate_Expr_ChangeDetectorExec",
        CHANGE_DETECTOR,
        service(0, -4),
        vec![],
    );
    world.add_wire_connection(
        clock.frame_index.clone(),
        WirePort::new(detector, CHANGE_DETECTOR, "Input"),
    );

    let array = gate(
        &mut chip,
        "B_1x1_Gate_Variable_Array",
        ARRAY_VAR,
        service(0, -5),
        vec![(
            "Value",
            Box::new(WireArrayVariant::DoubleArray(values)) as Box<dyn AsBrdbValue>,
        )],
    );
    let get = gate(
        &mut chip,
        "B_1x1_Gate_Exec_ArrayVar_Get",
        ARRAY_GET,
        service(1, -5),
        vec![],
    );
    world.add_wire_connection(
        WirePort::new(array, ARRAY_VAR, "ArrayVarRef"),
        WirePort::new(get, ARRAY_GET, "ArrayVarRef"),
    );
    world.add_wire_connection(
        clock.frame_index.clone(),
        WirePort::new(get, ARRAY_GET, "Index"),
    );
    world.add_wire_connection(
        WirePort::new(detector, CHANGE_DETECTOR, "OnChanged"),
        WirePort::new(get, ARRAY_GET, "Exec"),
    );
    // Straight into the emitter -- no master-volume multiply in between. This
    // is the wire whose behaviour is in question.
    world.add_wire_connection(
        WirePort::new(get, ARRAY_GET, "Value"),
        WirePort::new(speaker, AUDIO_EMITTER, port),
    );

    finish(&mut world, chip)?;
    // Must be LAST, after `finish` has published the chip's inner grid.
    world.register_used_components();
    Ok(world)
}

/// Diagnostic 3's volume curve: a triangle, 0.0 at frame 0, 1.0 at frame 30 and
/// back down by frame 60. Two seconds at 30 fps, looping forever through the
/// clock's modulo.
///
/// A triangle rather than a sawtooth on purpose: a sawtooth's wrap is itself a
/// discontinuity, which would sound like a click every cycle and be
/// indistinguishable from the retriggering this save exists to detect.
fn ramp_values() -> Vec<f64> {
    let half = RAMP_FRAMES / 2;
    (0..RAMP_FRAMES)
        .map(|i| 1.0 - (i as f64 - half as f64).abs() / half as f64)
        .collect()
}

fn specs_3() -> Vec<SpeakerSpec> {
    vec![SpeakerSpec {
        asset: BA_SYNTH_BASIC_SINE,
        pitch: 1.0,
        // Baked silent; the wire drives it from frame 0.
        volume: 0.0,
    }]
}

/// 5. ONE speaker whose PITCH is written every frame by a chip.
///
/// **The gate on `--audio-mode voice`.** That mode's whole premise is that a
/// speaker can be parked on a detected spectral peak and re-tuned every frame,
/// which removes the band grid -- and with it the tuning error -- entirely. It
/// only works if a `PitchMultiplier` write behaves the way a `VolumeMultiplier`
/// write was measured to (diagnostic 3): it must RE-TUNE the running voice, not
/// restart it. If it restarts it, a voice re-pitched at 30 fps is a 30 Hz click
/// train, which is precisely the "random chime" failure the design exists to
/// remove.
///
/// The volume is BAKED at 0.5 and nothing writes it, so pitch is the only thing
/// changing anywhere in the save. Diagnostic 3 already exonerated volume
/// writes; driving both here would leave a click ambiguous between the two.
fn diag_5_pitch_ramp() -> Result<World, String> {
    one_driven_port(
        "Audio diagnostic 5: one speaker whose pitch is written every frame by a chip",
        &specs_5()[0],
        "PitchMultiplier",
        pitch_ramp_values(),
    )
}

/// Diagnostic 5's pitch curve: an octave up and back down over two seconds,
/// as a triangle in LOG pitch.
///
/// `2^t` with `t` a 0 -> 1 -> 0 triangle, so the multiplier runs 1.0 -> 2.0 ->
/// 1.0 and the *perceived* sweep is linear in semitones: a constant 0.4
/// semitones per frame, 12 semitones each way. A triangle in the multiplier
/// itself would sound like it sped up on the way down and slowed on the way up,
/// which is a second thing to explain away when the point is to hear one.
///
/// A triangle rather than a sawtooth, for the same reason as [`ramp_values`]:
/// a sawtooth's wrap from 2.0 back to 1.0 is an octave jump in one frame, i.e.
/// a discontinuity that would click once per cycle whatever the game does with
/// pitch -- indistinguishable from the retriggering this save exists to detect.
/// The triangle's own wrap (frame 59 -> frame 0) is a step of exactly one
/// frame's worth, the same as every other step in the sweep.
///
/// Every value is inside the emitter's legal `PitchMultiplier` range
/// (`bands::PITCH_MIN`..`PITCH_MAX`, 0.1..10.0); an out-of-range value would be
/// clamped in game and the sweep would flatten at the top with nothing to say
/// so.
fn pitch_ramp_values() -> Vec<f64> {
    let half = RAMP_FRAMES / 2;
    (0..RAMP_FRAMES)
        .map(|i| {
            let t = 1.0 - (i as f64 - half as f64).abs() / half as f64;
            2.0f64.powf(t)
        })
        .collect()
}

fn specs_5() -> Vec<SpeakerSpec> {
    vec![SpeakerSpec {
        asset: BA_SYNTH_BASIC_SINE,
        // The sweep's own frame-0 value, so a chip that is paused or unwired
        // holds the bottom of the sweep rather than some unrelated note.
        pitch: 1.0,
        // Constant, and BAKED -- no wire touches volume in this save.
        volume: 0.5,
    }]
}

/// 4. All 32 speakers at the real renderer's own band pitches.
///
/// The plan comes from `BandPlan::new(32, 2)` rather than being recomputed, so
/// these are byte-for-byte the pitches a real render uses -- including the two
/// noise bands, which carry pitch 1.0 and their own synth assets.
fn diag_4_full_bank() -> Result<(World, BandPlan), String> {
    let plan = BandPlan::new(32, 2)?;
    let world = chipless_world(
        "Audio diagnostic 4: all 32 band speakers at constant volume, no wire graph",
        &specs_4(&plan),
    )?;
    Ok((world, plan))
}

fn specs_4(plan: &BandPlan) -> Vec<SpeakerSpec> {
    plan.kinds
        .iter()
        .zip(plan.pitches.iter())
        .map(|(kind, &pitch)| SpeakerSpec {
            asset: asset_for(*kind),
            pitch,
            volume: 0.1,
        })
        .collect()
}

/// The synth asset each band kind plays.
fn asset_for(kind: BandKind) -> BString {
    match kind {
        BandKind::Tonal => BA_SYNTH_BASIC_SINE,
        BandKind::WhiteNoise => BA_SYNTH_NOISE_WHITE,
        BandKind::PinkNoise => BA_SYNTH_NOISE_PINK,
    }
}

// ---------------------------------------------------------------------------
// Read-back verification
//
// Every per-speaker property here is build-time component DATA, so nothing in
// the graph's shape reflects it and a `LiteralComponent`'s values are opaque in
// memory (`Box<dyn AsBrdbValue>`, with no way back out). The only place they
// can be checked is a written save, decoded again -- so that is what this does,
// for all four files, rather than assuming the write worked.
// ---------------------------------------------------------------------------

/// Confirm diagnostic 3's ramp actually reached the file.
///
/// The emitter checks in [`verify`] say nothing about it: the whole subject of
/// this save is the array the chip reads, and an `ArrayVar` that arrived empty
/// or truncated would leave a save that loads, looks right and holds one
/// constant volume forever -- which is exactly the "no click, must be fine"
/// reading that would send the investigation the wrong way.
fn verify_ramp(path: &Path, expected: &[f64]) -> Result<String, String> {
    let db = brdb::Brz::open(path)
        .map_err(|e| format!("reopening {}: {e}", path.display()))?
        .into_reader();
    let mut arrays: Vec<Vec<f64>> = Vec::new();
    for gid in 1..8usize {
        let Ok(chunks) = db.brick_chunk_index(gid) else {
            break;
        };
        for chunk in &chunks {
            if chunk.num_components == 0 {
                continue;
            }
            let (_, structs) = db
                .component_chunk_soa(gid, chunk.index)
                .map_err(|e| format!("component chunk soa: {e}"))?;
            for s in structs {
                if s.get_name() != "BrickComponentData_WireGraphPseudo_ArrayVar" {
                    continue;
                }
                let v = s.prop("Value").map_err(|e| format!("ArrayVar Value: {e}"))?;
                let WireArrayVariant::DoubleArray(d) = WireArrayVariant::try_from(v)
                    .map_err(|e| format!("ArrayVar Value is not a wire array variant: {e}"))?
                else {
                    return Err(format!("the ramp must be a DoubleArray, got {v:?}"));
                };
                arrays.push(d);
            }
        }
    }
    if arrays.len() != 1 {
        return Err(format!(
            "expected exactly one ArrayVar in the ramp save, found {}",
            arrays.len()
        ));
    }
    if arrays[0] != expected {
        return Err(format!(
            "the ramp read back with {} values (expected {}), or with different \
             contents -- first few: {:?}",
            arrays[0].len(),
            expected.len(),
            &arrays[0][..arrays[0].len().min(4)]
        ));
    }
    Ok(format!(
        "ramp read back: {} values, min {:.3} max {:.3}, peak at index {}",
        arrays[0].len(),
        arrays[0].iter().cloned().fold(f64::INFINITY, f64::min),
        arrays[0].iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        arrays[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("the ramp is never NaN"))
            .map(|(i, _)| i)
            .unwrap_or_default(),
    ))
}

/// What a written save actually contains.
struct SaveFacts {
    /// Every decoded `BrickComponentData_AudioEmitter`, in (grid, brick index)
    /// order -- which is band order, since the speakers are added in band order
    /// and are the only bricks on the main grid.
    emitters: Vec<BrdbStruct>,
    /// The save's own external asset reference table.
    assets: Vec<(String, String)>,
    /// How many brick grids the save carries. 1 = main grid only, i.e. no
    /// microchip.
    grids: usize,
}

fn read_back(path: &Path) -> Result<SaveFacts, String> {
    let db = brdb::Brz::open(path)
        .map_err(|e| format!("reopening {}: {e}", path.display()))?
        .into_reader();
    let assets: Vec<(String, String)> = db
        .global_data()
        .map_err(|e| format!("{}: global data: {e}", path.display()))?
        .external_asset_references
        .iter()
        .cloned()
        .collect();

    let mut found: Vec<(usize, u32, BrdbStruct)> = Vec::new();
    let mut grids = 0usize;
    for gid in 1..8usize {
        let Ok(chunks) = db.brick_chunk_index(gid) else {
            break;
        };
        grids += 1;
        for chunk in &chunks {
            if chunk.num_components == 0 {
                continue;
            }
            let (soa, structs) = db
                .component_chunk_soa(gid, chunk.index)
                .map_err(|e| format!("{}: component chunk soa: {e}", path.display()))?;
            for (i, s) in structs.into_iter().enumerate() {
                if s.get_name() == "BrickComponentData_AudioEmitter" {
                    found.push((gid, soa.component_brick_indices[i], s));
                }
            }
        }
    }
    found.sort_by_key(|(gid, bi, _)| (*gid, *bi));
    Ok(SaveFacts {
        emitters: found.into_iter().map(|(_, _, s)| s).collect(),
        assets,
        grids,
    })
}

fn f32_prop(s: &BrdbStruct, prop: &str) -> Result<f32, String> {
    match s.prop(prop).map_err(|e| format!("{prop}: {e}"))? {
        BrdbValue::F32(v) => Ok(*v),
        other => Err(format!("{prop} decoded as {other:?}, expected an f32")),
    }
}

fn bool_prop(s: &BrdbStruct, prop: &str) -> Result<bool, String> {
    match s.prop(prop).map_err(|e| format!("{prop}: {e}"))? {
        BrdbValue::Bool(v) => Ok(*v),
        other => Err(format!("{prop} decoded as {other:?}, expected a bool")),
    }
}

/// Confirm a written save's emitters carry exactly the descriptor, pitch,
/// volume and spatialization they were built with.
///
/// Returns a one-line summary of what was actually decoded, so the run's own
/// output is evidence rather than an assertion nobody can see.
fn verify(path: &Path, specs: &[SpeakerSpec], expect_chipless: bool) -> Result<String, String> {
    let facts = read_back(path)?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    if expect_chipless {
        // Grid 1 is the main grid; a microchip's inner grid would be grid 2.
        // This is the file-level counterpart of `assert_chipless`.
        if facts.grids != 1 {
            return Err(format!(
                "{name}: expected exactly 1 brick grid (main grid only, no chip), found {}",
                facts.grids
            ));
        }
    }

    if facts.emitters.len() != specs.len() {
        return Err(format!(
            "{name}: expected {} emitters in the written save, found {}",
            specs.len(),
            facts.emitters.len()
        ));
    }

    for (i, (s, spec)) in facts.emitters.iter().zip(specs).enumerate() {
        let pitch = f32_prop(s, "PitchMultiplier")?;
        if (pitch - spec.pitch).abs() > 1e-6 {
            return Err(format!(
                "{name}: speaker {i} PitchMultiplier is {pitch}, expected {}",
                spec.pitch
            ));
        }
        let volume = f32_prop(s, "VolumeMultiplier")?;
        if (volume - spec.volume).abs() > 1e-6 {
            return Err(format!(
                "{name}: speaker {i} VolumeMultiplier is {volume}, expected {}",
                spec.volume
            ));
        }
        if bool_prop(s, "bSpatialization")? {
            return Err(format!(
                "{name}: speaker {i} has bSpatialization TRUE -- the game's own default. \
                 Every diagnostic would be measuring 3D panning instead of what it means to."
            ));
        }
        if !bool_prop(s, "bEnabled")? {
            return Err(format!("{name}: speaker {i} is not enabled"));
        }

        let BrdbValue::Asset(Some(idx)) = s
            .prop("AudioDescriptor")
            .map_err(|e| format!("{name}: speaker {i} AudioDescriptor: {e}"))?
        else {
            return Err(format!(
                "{name}: speaker {i} AudioDescriptor did not decode as Asset(Some(..)) -- \
                 a WireVariant::Object encodes down a different path entirely"
            ));
        };
        let (ty, asset) = facts.assets.get(*idx).ok_or_else(|| {
            format!("{name}: speaker {i} asset index {idx} is outside the save's table")
        })?;
        if ty != AUDIO_ASSET_TYPE || asset.as_str() != spec.asset.as_ref() {
            return Err(format!(
                "{name}: speaker {i} references ({ty}, {asset}), expected \
                 ({AUDIO_ASSET_TYPE}, {})",
                spec.asset
            ));
        }
    }

    // Read back out of the FILE, not out of the builder's own inputs.
    let first = facts
        .emitters
        .first()
        .ok_or_else(|| format!("{name}: the save contains no emitters at all"))?;
    let BrdbValue::Asset(Some(idx)) = first.prop("AudioDescriptor").map_err(|e| e.to_string())?
    else {
        unreachable!("checked above");
    };
    Ok(format!(
        "read back: {} grid(s), {} emitter(s); speaker 0 = {} pitch {} volume {} \
         bSpatialization {}",
        facts.grids,
        facts.emitters.len(),
        facts.assets[*idx].1,
        f32_prop(first, "PitchMultiplier")?,
        f32_prop(first, "VolumeMultiplier")?,
        bool_prop(first, "bSpatialization")?,
    ))
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn write_save(dir: &Path, name: &str, world: &World) -> Result<(PathBuf, usize), String> {
    let bytes = world
        .to_brz_vec()
        .map_err(|e| format!("encoding {name}: {e}"))?;
    let path = dir.join(name);
    std::fs::write(&path, &bytes).map_err(|e| format!("writing {}: {e}", path.display()))?;
    Ok((path, bytes.len()))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| ".".to_string()));
    std::fs::create_dir_all(&dir)?;

    println!(
        "Five diagnostic saves. Tests 1, 2 and 4 contain NO microchip and NO wires at all --\n\
         only speaker bricks with baked component data. If one of those fails to make sound,\n\
         the fault is in the emitter/asset/pitch setup and the entire wire-graph layer is\n\
         exonerated. Tests 3 and 5 have a chip, because a chip driving a speaker IS what they\n\
         test -- 3 writes VolumeMultiplier every frame, 5 writes PitchMultiplier.\n"
    );

    // --- 1 -----------------------------------------------------------------
    let world = diag_1_static_tone()?;
    assert_chipless(&world, "diag_1_static_tone");
    let (p1, n1) = write_save(&dir, "diag_1_static_tone.brz", &world)?;
    let seen = verify(&p1, &specs_1(), true)?;
    println!("{}  ({n1} bytes)\n  {seen}", p1.display());
    println!(
        "  EXPECT: one steady, continuous tone (the sine synth's own authored pitch, \
         nominally 440 Hz), forever, at moderate volume.\n  \
         IF NOT (silence): a baked AudioDescriptor does not make an emitter sound at all -- \
         the asset reference, the component name or bEnabled is wrong, and no amount of \
         wiring will ever fix the render.\n"
    );

    // --- 2 -----------------------------------------------------------------
    let world = diag_2_static_chord()?;
    assert_chipless(&world, "diag_2_static_chord");
    let (p2, n2) = write_save(&dir, "diag_2_static_chord.brz", &world)?;
    let seen = verify(&p2, &specs_2(), true)?;
    println!("{}  ({n2} bytes)\n  {seen}", p2.display());
    println!(
        "  EXPECT: a clean, recognisable A-major triad (A 440 / C# 554 / E 659), held.\n  \
         IF NOT (out of tune, or not a major chord): PitchMultiplier is not a linear \
         frequency-rate multiplier on 440 Hz, so BandPlan's geometric pitches are the wrong \
         notes and every band in the real render is playing something other than its \
         analysed frequency.\n"
    );

    // --- 3 -----------------------------------------------------------------
    let world = diag_3_volume_ramp()?;
    assert!(
        world.grids.len() == 1,
        "diag_3 must carry exactly one microchip grid, found {}",
        world.grids.len()
    );
    assert!(
        !world.wires.is_empty(),
        "diag_3's whole subject is a wire driving VolumeMultiplier"
    );
    let (p3, n3) = write_save(&dir, "diag_3_volume_ramp.brz", &world)?;
    let seen = verify(&p3, &specs_3(), false)?;
    let ramp_seen = verify_ramp(&p3, &ramp_values())?;
    println!("{}  ({n3} bytes)\n  {seen}\n  {ramp_seen}", p3.display());
    println!(
        "  EXPECT: a smooth 2-second fade in and back out, repeating -- {RAMP_FRAMES} frames at \
         {RAMP_FPS} fps.\n  \
         IF NOT (a click, buzz or stutter at 30 Hz instead of a smooth fade): writing \
         VolumeMultiplier RETRIGGERS the voice instead of changing its level, and that single \
         fact explains the entire failure -- 32 speakers restarted 30 times a second is a \
         broadband click train, i.e. noise. The per-frame volume approach then cannot work as \
         designed.\n"
    );

    // --- 4 -----------------------------------------------------------------
    let (world, plan) = diag_4_full_bank()?;
    assert_chipless(&world, "diag_4_full_bank");
    let (p4, n4) = write_save(&dir, "diag_4_full_bank.brz", &world)?;
    let seen = verify(&p4, &specs_4(&plan), true)?;
    println!("{}  ({n4} bytes)\n  {seen}", p4.display());
    println!(
        "  EXPECT: a dense static drone -- {} sine bands from ~44 Hz to ~4.4 kHz plus a white \
         and a pink noise bed, all sounding at once.\n  \
         IF NOT (thinner than 32 voices, or bands audibly missing): the game caps concurrent \
         voices and is stealing some of the bank, so the real render is quieter and narrower \
         than designed.\n",
        plan.kinds
            .iter()
            .filter(|k| **k == BandKind::Tonal)
            .count(),
    );

    // --- 5 -----------------------------------------------------------------
    // Range-checked before it is written: an out-of-range PitchMultiplier is
    // clamped silently in game, which would flatten the top of the sweep and
    // look like the very stall this save exists to detect.
    let pitch_ramp = pitch_ramp_values();
    for (i, p) in pitch_ramp.iter().enumerate() {
        assert!(
            *p as f32 >= PITCH_MIN && *p as f32 <= PITCH_MAX,
            "pitch ramp frame {i} is {p}, outside the emitter's legal \
             {PITCH_MIN}..{PITCH_MAX} PitchMultiplier range -- the game would clamp it"
        );
    }
    let world = diag_5_pitch_ramp()?;
    assert!(
        world.grids.len() == 1,
        "diag_5 must carry exactly one microchip grid, found {}",
        world.grids.len()
    );
    assert!(
        !world.wires.is_empty(),
        "diag_5's whole subject is a wire driving PitchMultiplier"
    );
    let (p5, n5) = write_save(&dir, "diag_5_pitch_ramp.brz", &world)?;
    let seen = verify(&p5, &specs_5(), false)?;
    let ramp_seen = verify_ramp(&p5, &pitch_ramp)?;
    println!("{}  ({n5} bytes)\n  {seen}\n  {ramp_seen}", p5.display());
    println!(
        "  EXPECT: one smooth glissando -- an octave up and back down every two seconds, at a \
         constant volume, repeating. {RAMP_FRAMES} frames at {RAMP_FPS} fps, 0.4 semitones per \
         frame.\n  \
         IF NOT (a click, buzz or stutter at 30 Hz, or a machine-gun of separate notes instead \
         of one sliding tone): writing PitchMultiplier RETRIGGERS the voice instead of retuning \
         it. `--audio-mode voice` re-pitches its speakers every frame, so it would be a 30 Hz \
         click train; the mode would then have to hold each voice's pitch constant for the life \
         of a note and only re-pitch at note boundaries (where an attack is musically correct), \
         losing vibrato and glissando.\n"
    );

    println!(
        "\
Checklist -- work through these in order, in a quiet local world, standing still:

  1. Load diag_1_static_tone.brz. Do you hear a steady tone?
     No  -> STOP. Nothing downstream can work. The emitter setup itself is wrong.
     Yes -> assumption 1 holds: a baked descriptor sounds, with no wire graph at all.

  2. Load diag_2_static_chord.brz. Is it a recognisable major chord?
     No  -> the pitch mapping is wrong. BandPlan's pitches are not the frequencies
            the analysis assigned them, so every band plays the wrong note.
     Yes -> assumption 2 holds: PitchMultiplier is a linear rate multiplier on 440 Hz.

  3. Load diag_3_volume_ramp.brz. THIS IS THE ONE THAT MATTERS MOST.
     One full fade in and back out every two seconds, repeating. Is it SMOOTH?
       Yes -> assumption 3 holds. Per-frame volume writes are usable, and the fault
              is somewhere else entirely.
       No -- it clicks, buzzes or stutters at a steady rate instead of fading
            -> THIS IS THE ANSWER. Writing VolumeMultiplier retriggers the voice
               rather than changing its level. The per-frame volume approach cannot
               work as designed and the renderer needs a different mechanism; a
               32-speaker render is then a broadband click train, which is exactly
               what \"does not sound like the music\" would sound like.

  4. Load diag_4_full_bank.brz. Can you hear a full, dense drone?
     Thin or patchy -> a concurrent-voice cap is stealing part of the bank.
     Dense          -> 32 simultaneous emitters all sound; the bank width is real.

  5. Load diag_5_pitch_ramp.brz. THIS IS THE ONE THAT GATES --audio-mode voice.
     One octave up and back down every two seconds, at a constant volume,
     repeating. Is it a SMOOTH GLISSANDO -- one continuous tone sliding?
       Yes -> assumption 4 holds. A speaker can be re-pitched every frame, so a
              voice can be parked on a detected spectral peak and follow it. That
              is what removes the band grid, and with it the tuning error.
       No -- it clicks, buzzes, stutters, or sounds like a rapid run of separate
            notes rather than one sliding tone
            -> writing PitchMultiplier RETRIGGERS the voice. --audio-mode voice
               then cannot re-pitch per frame: it would have to freeze each
               voice's pitch for the life of a note and re-pitch only at note
               boundaries, where an attack is musically correct anyway. Vibrato
               and glissando are lost; the design itself still stands.

Only one speaker cluster exists per save, so load them one at a time and clear the
world in between. Every save above uses the renderer's own packing and its own
attenuation radii, so what you hear is the geometry a real render ships."
    );

    Ok(())
}
