#[path = "wire_integrity.rs"]
mod wire_integrity;

use brdb::assets::LiteralComponent;
use brdb::{AsBrdbValue, Brick, BrickType, Position, World, schema::WireArrayVariant};
use std::collections::HashMap;
use std::sync::Arc;

/// A full bank of doubles must survive serialisation. The whole design puts
/// one `ArrayVar` per band holding `BANK_FRAMES` volumes; if that does not
/// round-trip, the data layout has to change to packed strings and the gate
/// count doubles.
#[test]
fn a_full_bank_of_doubles_round_trips() {
    let n = heightmap::anim::pack::BANK_FRAMES;
    let values: Vec<f64> = (0..n).map(|i| (i % 256) as f64 / 255.0).collect();

    let mut world = World::new();
    let mut data: HashMap<brdb::BString, Box<dyn AsBrdbValue>> = HashMap::new();
    data.insert(
        "Value".into(),
        Box::new(WireArrayVariant::DoubleArray(values)) as Box<dyn AsBrdbValue>,
    );
    let brick = Brick {
        asset: BrickType::from("B_1x1_Gate_Variable_Array"),
        position: Position { x: 0, y: 0, z: 2 },
        ..Default::default()
    }
    .with_component(LiteralComponent::new_from_data(
        "BrickComponentType_WireGraphPseudo_ArrayVar",
        Arc::new(data),
    ));
    world.add_brick(brick);
    // Component types used by bricks must be registered in global_data before
    // packing, or `to_unsaved` rejects the world with UnregisteredComponentType.
    world.register_used_components();

    let bytes = world
        .to_brz_vec()
        .expect("a full bank of doubles must serialise");
    assert!(
        bytes.len() > 1000,
        "a 65535-element double array should produce a substantial file, got {} bytes",
        bytes.len()
    );
}

use heightmap::audio::bands::{BASE_HZ, BandKind};
use heightmap::audio::source::SampleClip;
use heightmap::audio::speakers::build_speaker_world;
use heightmap::audio::track::{AudioOptions, analyze};
use heightmap::progress::NoProgress;
use std::f32::consts::TAU;

fn tone_track(secs: f32, opts: &AudioOptions) -> heightmap::audio::track::VoiceTrack {
    let sr = 48_000u32;
    let n = (sr as f32 * secs) as usize;
    let clip = SampleClip::new(
        sr,
        (0..n)
            .map(|i| (TAU * 1000.0 * i as f32 / sr as f32).sin())
            .collect(),
    );
    analyze(&clip, opts, &mut NoProgress).expect("analyze")
}

/// Gates live on the chip's inner grid, never `world.bricks` (which is the
/// main grid only). A test that counts `world.bricks` would be vacuous.
///
/// `brdb` is a direct dependency of this crate, so import it directly --
/// `heightmap` does not re-export it.
fn inner_components(world: &brdb::World) -> Vec<String> {
    world.grids[0]
        .1
        .iter()
        .flat_map(|b| b.components.iter())
        .filter_map(|c| c.component_type().map(|t| t.to_string()))
        .collect()
}

#[test]
fn every_band_gets_one_speaker() {
    let opts = AudioOptions::default();
    let track = tone_track(1.0, &opts);
    let world = build_speaker_world(&track, &opts).expect("build");
    let speakers = world
        .bricks
        .iter()
        .flat_map(|b| b.components.iter())
        .filter(|c| {
            c.component_type()
                .map(|t| t.as_ref() == "Component_AudioEmitter")
                .unwrap_or(false)
        })
        .count();
    assert_eq!(speakers, track.plan.len(), "one speaker per band");
    // The derived default: every equal-tempered step the emitter's pitch
    // range holds at 12 per octave -- 79 semitones, F#1..C8 -- and NO noise
    // bands, which every source tried in game sounded worse with.
    assert_eq!(track.plan.len(), 79, "the default plan fills the scale");
}

#[test]
fn a_single_bank_track_emits_no_branch_and_only_the_pause_mute_select() {
    let opts = AudioOptions::default();
    let track = tone_track(1.0, &opts);
    let world = build_speaker_world(&track, &opts).expect("build");
    let comps = inner_components(&world);
    assert_eq!(
        comps.iter().filter(|c| c.contains("Exec_Branch")).count(),
        0,
        "a track inside one bank needs no branch cascade"
    );
    // The bank-VALUE cascade emits a Select per bank boundary, of which a
    // single bank has none. The ONE Select here is the always-present
    // pause-mute gate (see `scaffold`), which is shared by the whole bank and
    // does not scale with banks.
    assert_eq!(
        comps.iter().filter(|c| c.contains("Expr_Select")).count(),
        1,
        "single bank: no value-cascade selects, only the shared pause-mute Select"
    );
    assert_eq!(
        comps
            .iter()
            .filter(|c| c.contains("Variable_Array") || c.contains("ArrayVar") && !c.contains("Get"))
            .count(),
        track.plan.len(),
        "one array per band"
    );
}

/// The spillover shape, forced with a tiny bank size so the test stays fast.
#[test]
fn a_multi_bank_track_emits_the_branch_and_select_cascade() {
    let mut opts = AudioOptions::default();
    opts.bank_size = 10;
    let track = tone_track(2.0, &opts); // ~60 frames -> 6 banks
    let banks = track.frame_count.div_ceil(10);
    assert!(banks >= 2, "this test needs multiple banks, got {banks}");

    let world = build_speaker_world(&track, &opts).expect("build");
    let comps = inner_components(&world);
    assert_eq!(
        comps.iter().filter(|c| c.contains("Exec_Branch")).count(),
        banks - 1,
        "one branch per bank boundary"
    );
    assert_eq!(
        comps.iter().filter(|c| c.contains("Expr_Select")).count(),
        track.plan.len() * (banks - 1) + 1,
        "one select per band per boundary, plus the one shared pause-mute Select"
    );
}

#[test]
fn the_world_serialises() {
    let opts = AudioOptions::default();
    let track = tone_track(1.0, &opts);
    let world = build_speaker_world(&track, &opts).expect("build");
    let bytes = world.to_brz_vec().expect("a built world must serialise");
    assert!(bytes.len() > 500);
}

#[test]
fn noise_bands_carry_noise_assets_and_tonal_bands_do_not() {
    // Explicitly opted in: noise bands are off by default now (reported worse
    // on speech, piano and a pop mix alike), but the flag is kept for
    // percussion and the mechanism still has to be right when it is used.
    let opts = AudioOptions { noise_bands: 2, ..AudioOptions::default() };
    let track = tone_track(1.0, &opts);
    let tonal = track.plan.tonal_count();
    assert_eq!(track.plan.kinds[tonal], BandKind::WhiteNoise);
    assert_eq!(track.plan.kinds[tonal + 1], BandKind::PinkNoise);
    // Band 0 is the lowest legal semitone, 39 below A440: F#1, 46.25 Hz.
    assert_eq!(track.plan.step_of(0), Some(-39));
    assert!((track.plan.pitches[0] * BASE_HZ - 46.249).abs() < 0.01);
}

#[test]
fn an_empty_track_is_rejected_rather_than_producing_a_broken_save() {
    let opts = AudioOptions::default();
    let mut track = tone_track(1.0, &opts);
    let n = track.plan.len();
    track.frame_count = 0;
    track.volumes = vec![Vec::new(); n];
    assert!(build_speaker_world(&track, &opts).is_err());
}

use brdb::IntoReader;
use brdb::schema::{BrdbStruct, BrdbValue, WireVariant};
use heightmap::anim::bricks::{ARRAY_GET, ARRAY_VAR, BRANCH, SELECT};
use heightmap::anim::chip::MICROCHIP_INPUT;
use heightmap::anim::clock::{MULTIPLY, TIMER};
use heightmap::audio::speakers::{
    AUDIO_EMITTER, BUFFER_TICKS, COMPARE_NE, cluster_dims, speaker_half, speaker_inner_position,
    speaker_position,
};
use std::collections::HashSet;

/// The multi-bank fixture the wire-shape tests share: a small enough bank
/// size to force several banks out of a two-second clip.
fn multi_bank() -> (
    heightmap::audio::track::VoiceTrack,
    AudioOptions,
    brdb::World,
    usize,
) {
    let mut opts = AudioOptions::default();
    opts.bank_size = 10;
    let track = tone_track(2.0, &opts);
    let n_banks = track.frame_count.div_ceil(opts.bank_size);
    assert!(
        n_banks >= 3,
        "the cascade tests need several banks, got {n_banks}"
    );
    let world = build_speaker_world(&track, &opts).expect("build");
    (track, opts, world, n_banks)
}

/// brick id -> the component type it carries, for every brick in the world.
fn component_of(world: &brdb::World) -> HashMap<usize, String> {
    world
        .bricks
        .iter()
        .chain(world.grids.iter().flat_map(|(_, bs)| bs.iter()))
        .filter_map(|b| {
            let id = b.id?;
            let t = b.components.first()?.component_type()?;
            Some((id, t.to_string()))
        })
        .collect()
}

/// (target brick id, target port) -> every (source brick id, source port)
/// feeding it.
fn sources_by_target(world: &brdb::World) -> HashMap<(usize, String), Vec<(usize, String)>> {
    let mut map: HashMap<(usize, String), Vec<(usize, String)>> = HashMap::new();
    for w in &world.wires {
        map.entry((w.target.brick_id, w.target.port_name.to_string()))
            .or_default()
            .push((w.source.brick_id, w.source.port_name.to_string()));
    }
    map
}

/// (source brick id, source port) -> every (target brick id, target port) it
/// feeds.
fn targets_by_source(world: &brdb::World) -> HashMap<(usize, String), Vec<(usize, String)>> {
    let mut map: HashMap<(usize, String), Vec<(usize, String)>> = HashMap::new();
    for w in &world.wires {
        map.entry((w.source.brick_id, w.source.port_name.to_string()))
            .or_default()
            .push((w.target.brick_id, w.target.port_name.to_string()));
    }
    map
}

/// Inner-grid brick ids carrying `class`, in the order they were emitted.
fn inner_ids_of(world: &brdb::World, class: &str) -> Vec<usize> {
    world.grids[0]
        .1
        .iter()
        .filter(|b| {
            b.components
                .iter()
                .any(|c| c.component_type().map(|t| t.as_ref() == class).unwrap_or(false))
        })
        .filter_map(|b| b.id)
        .collect()
}

/// Main-grid speaker brick ids, in band order.
fn speaker_ids(world: &brdb::World) -> Vec<usize> {
    world
        .bricks
        .iter()
        .filter(|b| {
            b.components.iter().any(|c| {
                c.component_type()
                    .map(|t| t.as_ref() == AUDIO_EMITTER)
                    .unwrap_or(false)
            })
        })
        .filter_map(|b| b.id)
        .collect()
}

/// Write `world` to a temp `.brz` and hand every decoded component struct,
/// paired with its brick index in chunk, to `f` -- plus the save's own
/// external asset reference table.
fn with_decoded_components<R>(
    world: &brdb::World,
    tag: &str,
    f: impl FnOnce(&[(usize, u32, BrdbStruct)], &[(String, String)]) -> R,
) -> R {
    let path = std::env::temp_dir().join(format!("h2b_audio_{tag}_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    let out = {
        let db = brdb::Brz::open(&path).expect("reopen").into_reader();
        let assets: Vec<(String, String)> = db
            .global_data()
            .expect("global data")
            .external_asset_references
            .iter()
            .cloned()
            .collect();
        let mut all = Vec::new();
        for gid in 1..8usize {
            let Ok(chunks) = db.brick_chunk_index(gid) else {
                break;
            };
            for chunk in &chunks {
                if chunk.num_components == 0 {
                    continue;
                }
                let (soa, structs) = db
                    .component_chunk_soa(gid, chunk.index)
                    .expect("component chunk soa");
                for (i, s) in structs.into_iter().enumerate() {
                    all.push((gid, soa.component_brick_indices[i], s));
                }
            }
        }
        f(&all, &assets)
    };
    let _ = std::fs::remove_file(&path);
    out
}

fn f32_prop(s: &BrdbStruct, prop: &str) -> f32 {
    match s.prop(prop).unwrap_or_else(|e| panic!("{prop}: {e}")) {
        BrdbValue::F32(v) => *v,
        other => panic!("{prop} decoded as {other:?}, expected an f32"),
    }
}

fn bool_prop(s: &BrdbStruct, prop: &str) -> bool {
    match s.prop(prop).unwrap_or_else(|e| panic!("{prop}: {e}")) {
        BrdbValue::Bool(v) => *v,
        other => panic!("{prop} decoded as {other:?}, expected a bool"),
    }
}

/// The synth asset each band kind must play. An independent oracle: spelled
/// out here rather than imported, so a change in `speakers.rs` cannot agree
/// with itself.
fn expected_asset(kind: BandKind) -> &'static str {
    match kind {
        BandKind::Tonal => "BA_Synth_Basic_Sine",
        BandKind::WhiteNoise => "BA_Synth_Noise_White",
        BandKind::PinkNoise => "BA_Synth_Noise_Pink",
    }
}

/// Every per-speaker property is build-time component DATA, never a wire, so
/// nothing in the graph's shape reflects it and the only place it can be
/// checked is a written save. All four of these are silent in game if wrong:
/// a wrong pitch retunes a band, a missing `bSpatialization` makes the cluster
/// comb-filter against itself, a sine on the noise bands turns cymbals into a
/// whistle, and `bEnabled` false mutes the speaker outright.
#[test]
fn every_speakers_pitch_asset_and_spatialization_reach_the_save() {
    let opts = AudioOptions::default();
    let track = tone_track(1.0, &opts);
    let world = build_speaker_world(&track, &opts).expect("build");

    with_decoded_components(&world, "emitters", |all, assets| {
        let mut emitters: Vec<(u32, &BrdbStruct)> = all
            .iter()
            .filter(|(_, _, s)| s.get_name() == "BrickComponentData_AudioEmitter")
            .map(|(_, bi, s)| (*bi, s))
            .collect();
        emitters.sort_by_key(|(bi, _)| *bi);
        assert_eq!(emitters.len(), track.plan.len(), "one emitter per band");
        // The speakers are the only bricks in their chunk and were added in
        // band order, so brick index IS the band index. Stated as an
        // assertion rather than assumed.
        assert!(
            emitters
                .iter()
                .enumerate()
                .all(|(b, (bi, _))| *bi as usize == b),
            "speaker brick indices must run 0..n in band order, got {:?}",
            emitters.iter().map(|(bi, _)| *bi).collect::<Vec<_>>()
        );

        for (b, (_, s)) in emitters.iter().enumerate() {
            assert!(
                (f32_prop(s, "PitchMultiplier") - track.plan.pitches[b]).abs() < 1e-6,
                "band {b} must carry its own pitch {}, got {}",
                track.plan.pitches[b],
                f32_prop(s, "PitchMultiplier")
            );
            assert!(
                !bool_prop(s, "bSpatialization"),
                "band {b}: spatialization must be OFF -- a cluster of positioned \
                 emitters comb-filters against itself"
            );
            assert!(bool_prop(s, "bEnabled"), "band {b} must be enabled");
            assert_eq!(
                f32_prop(s, "VolumeMultiplier"),
                0.0,
                "band {b} must start silent; the wire drives it from frame 0"
            );

            let BrdbValue::Asset(Some(idx)) = s.prop("AudioDescriptor").expect("AudioDescriptor")
            else {
                panic!(
                    "band {b}: AudioDescriptor must decode as Asset(Some(..)), got {:?} -- a \
                     WireVariant::Object encodes down a different path entirely",
                    s.prop("AudioDescriptor")
                );
            };
            let (ty, name) = assets
                .get(*idx)
                .unwrap_or_else(|| panic!("band {b}: asset index {idx} is out of range"));
            assert_eq!(ty, "BrickAudioDescriptor", "band {b}: wrong asset type");
            assert_eq!(
                name,
                expected_asset(track.plan.kinds[b]),
                "band {b} ({:?}) is playing the wrong synth asset",
                track.plan.kinds[b]
            );
        }
    });
}

/// The bank slices must partition each band's volumes exactly: no frame
/// dropped, none duplicated. An off-by-one at either end of the slice shortens
/// every array by one frame and shifts the whole track against the clock, with
/// nothing in the graph's shape to show for it.
#[test]
fn the_bank_arrays_partition_every_bands_frames_exactly() {
    let (track, opts, world, n_banks) = multi_bank();

    let mut expected_lengths: Vec<usize> = Vec::new();
    let mut expected_values: Vec<f64> = Vec::new();
    for band in &track.volumes {
        for bi in 0..n_banks {
            let lo = bi * opts.bank_size;
            let hi = ((bi + 1) * opts.bank_size).min(band.len());
            expected_lengths.push(hi.saturating_sub(lo));
            expected_values.extend_from_slice(&band[lo..hi]);
        }
    }
    expected_lengths.sort_unstable();
    expected_values.sort_by(|a, b| a.partial_cmp(b).expect("volumes are never NaN"));
    assert_eq!(
        expected_values.len(),
        track.plan.len() * track.frame_count,
        "the oracle itself must cover every band's every frame"
    );

    with_decoded_components(&world, "arrays", |all, _| {
        let mut lengths = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for (_, _, s) in all {
            if s.get_name() != "BrickComponentData_WireGraphPseudo_ArrayVar" {
                continue;
            }
            let v = s.prop("Value").expect("an ArrayVar must carry a Value");
            let WireArrayVariant::DoubleArray(d) = WireArrayVariant::try_from(v)
                .expect("an audio bank must decode as a wire array variant")
            else {
                panic!("an audio bank must be a DoubleArray, not {v:?}");
            };
            lengths.push(d.len());
            values.extend_from_slice(&d);
        }
        lengths.sort_unstable();
        values.sort_by(|a, b| a.partial_cmp(b).expect("saved volumes are never NaN"));
        assert_eq!(
            lengths, expected_lengths,
            "every bank slice must be exactly as long as the frames it covers"
        );
        assert_eq!(
            values, expected_values,
            "the banks together must hold every band's every volume exactly once"
        );
    });
}

/// Each speaker must be driven by its own band's arrays. Pointing them all at
/// band 0 leaves every brick, gate and wire count identical, and the render
/// plays one band through 32 speakers.
#[test]
fn every_speaker_is_driven_by_its_own_bands_arrays() {
    let (track, _opts, world, n_banks) = multi_bank();
    let comp = component_of(&world);
    let by_target = sources_by_target(&world);
    let arrays = inner_ids_of(&world, ARRAY_VAR);
    let speakers = speaker_ids(&world);
    assert_eq!(speakers.len(), track.plan.len());
    assert_eq!(arrays.len(), track.plan.len() * n_banks);

    for (b, &speaker) in speakers.iter().enumerate() {
        let feeding = by_target
            .get(&(speaker, "VolumeMultiplier".to_string()))
            .unwrap_or_else(|| {
                panic!("band {b}'s speaker has NOTHING wired into VolumeMultiplier")
            });
        assert_eq!(
            feeding.len(),
            1,
            "band {b}: exactly one wire drives the volume"
        );

        // Walk back through the select cascade, collecting every array the
        // speaker can possibly read.
        let mut reached: HashSet<usize> = HashSet::new();
        let mut queue = feeding.clone();
        let mut guard = 0;
        while let Some((id, port)) = queue.pop() {
            guard += 1;
            assert!(guard < 10_000, "band {b}: cycle while tracing the value chain");
            match comp.get(&id).map(String::as_str) {
                // The master-volume multiply now sits between the band's value
                // and its speaker. Only `InputA` carries audio -- `InputB` is
                // the `Volume` pin, checked in its own test below.
                Some(MULTIPLY) => {
                    assert_eq!(port, "Output", "band {b}: a multiply feeds from its Output");
                    queue.extend(
                        by_target
                            .get(&(id, "InputA".to_string()))
                            .unwrap_or_else(|| {
                                panic!("band {b}: the volume multiply's InputA is unwired")
                            })
                            .clone(),
                    );
                }
                Some(SELECT) => {
                    assert_eq!(port, "Output", "band {b}: a select feeds from its Output");
                    for p in ["InputA", "InputB"] {
                        queue.extend(
                            by_target
                                .get(&(id, p.to_string()))
                                .unwrap_or_else(|| panic!("band {b}: select {p} is unwired"))
                                .clone(),
                        );
                    }
                }
                Some(ARRAY_GET) => {
                    assert_eq!(port, "Value", "band {b}: a get feeds from its Value");
                    let refs = by_target
                        .get(&(id, "ArrayVarRef".to_string()))
                        .unwrap_or_else(|| panic!("band {b}: a get has no array behind it"));
                    assert_eq!(refs.len(), 1);
                    reached.insert(refs[0].0);
                }
                other => panic!("band {b}: unexpected {other:?} in the value chain"),
            }
        }

        // Arrays are emitted band-major, so band b owns exactly its own slice
        // of them -- which pins the binding, not merely its distinctness.
        let expected: HashSet<usize> = arrays[b * n_banks..(b + 1) * n_banks]
            .iter()
            .copied()
            .collect();
        assert_eq!(
            reached, expected,
            "band {b}'s speaker must read band {b}'s own {n_banks} arrays"
        );
    }
}

/// `Exec_Branch` polarity: a TRUTHY `bCond` takes `ExecOutA`.
///
/// `ge[k]` is "the frame index has reached bank k+1", so true means keep
/// descending the cascade (`ExecOutA` -> the next branch) and false means run
/// this bank (`ExecOutB` -> its chain of gets). Swapping the two runs the
/// wrong bank at every frame -- and changes no count anywhere.
#[test]
fn a_truthy_branch_condition_descends_the_cascade_on_execouta() {
    let (_track, _opts, world, n_banks) = multi_bank();
    let comp = component_of(&world);
    let by_source = targets_by_source(&world);
    let branches = inner_ids_of(&world, BRANCH);
    assert_eq!(branches.len(), n_banks - 1, "one branch per bank boundary");

    for (i, &br) in branches.iter().enumerate() {
        let out_a = by_source
            .get(&(br, "ExecOutA".to_string()))
            .unwrap_or_else(|| panic!("branch {i}: ExecOutA is unwired"));
        let out_b = by_source
            .get(&(br, "ExecOutB".to_string()))
            .unwrap_or_else(|| panic!("branch {i}: ExecOutB is unwired"));
        assert_eq!(out_a.len(), 1, "branch {i}: ExecOutA takes one wire");
        assert_eq!(out_b.len(), 1, "branch {i}: ExecOutB takes one wire");

        // False -> run THIS bank: straight into a get's Exec.
        assert_eq!(
            (
                comp.get(&out_b[0].0).map(String::as_str),
                out_b[0].1.as_str()
            ),
            (Some(ARRAY_GET), "Exec"),
            "branch {i}: a FALSY condition must run this bank's chain from ExecOutB"
        );
        // True -> keep descending: into the next branch, or (at the last
        // boundary) into the final bank's chain.
        let want = if i + 1 < branches.len() {
            (Some(BRANCH), "Exec")
        } else {
            (Some(ARRAY_GET), "Exec")
        };
        assert_eq!(
            (
                comp.get(&out_a[0].0).map(String::as_str),
                out_a[0].1.as_str()
            ),
            want,
            "branch {i}: a TRUTHY condition must descend the cascade from ExecOutA"
        );
        if i + 1 < branches.len() {
            assert_eq!(
                out_a[0].0,
                branches[i + 1],
                "branch {i}'s ExecOutA must reach the NEXT branch"
            );
        }
    }
}

/// `Select`'s `bSelectB` is "the frame has reached the later bank", so the
/// LATER bank's value belongs on `InputB` and the accumulated cascade on
/// `InputA`. Swapping them inverts every boundary and, again, changes no count.
#[test]
fn the_select_cascade_puts_the_later_bank_on_inputb() {
    let (_track, _opts, world, _n_banks) = multi_bank();
    let comp = component_of(&world);
    let by_target = sources_by_target(&world);
    let selects = inner_ids_of(&world, SELECT);
    assert!(!selects.is_empty(), "this fixture must emit selects");

    let mut cascaded = 0usize;
    for &sel in &selects {
        // Skip the shared pause-mute Select: its InputA is a baked 0.0
        // (unwired) and its InputB is the Volume pin, so it is not part of the
        // bank-VALUE cascade this test is about. Every value-cascade Select has
        // InputA wired, so an unwired InputA identifies the pause-mute one
        // uniquely. It has its own test (`the_pause_mute_gate_...`).
        if by_target.get(&(sel, "InputA".to_string())).is_none() {
            continue;
        }
        for (port, want) in [("InputA", "cascade"), ("InputB", "later bank")] {
            let srcs = by_target
                .get(&(sel, port.to_string()))
                .unwrap_or_else(|| panic!("select {sel}: {port} ({want}) is unwired"));
            assert_eq!(srcs.len(), 1, "select {sel}: {port} takes one wire");
        }
        let a = &by_target[&(sel, "InputA".to_string())][0];
        let b = &by_target[&(sel, "InputB".to_string())][0];
        // InputB is ALWAYS a bank's own get -- never another select.
        assert_eq!(
            (comp.get(&b.0).map(String::as_str), b.1.as_str()),
            (Some(ARRAY_GET), "Value"),
            "select {sel}: InputB must be the later bank's own value, not the cascade"
        );
        // InputA is the running cascade: another select once past the first
        // boundary, the first bank's get at the first.
        match comp.get(&a.0).map(String::as_str) {
            Some(SELECT) => {
                assert_eq!(a.1, "Output");
                cascaded += 1;
            }
            Some(ARRAY_GET) => assert_eq!(a.1, "Value"),
            other => panic!("select {sel}: InputA came from {other:?}"),
        }
        let cond = by_target
            .get(&(sel, "bSelectB".to_string()))
            .unwrap_or_else(|| panic!("select {sel}: bSelectB is unwired"));
        assert_eq!(cond.len(), 1);
        assert_eq!(cond[0].1, "bOutput", "bSelectB must come from a comparator");
    }
    assert!(
        cascaded > 0,
        "this fixture must exercise a real cascade, not just single boundaries"
    );
}

/// Every `ArrayVar_Get` needs its own bank's index and a place in the exec
/// chain. Dropping the `Index` wire freezes every band on frame 0; feeding
/// every bank the raw frame index (instead of the per-bank subtraction) reads
/// past the end of every bank but the first; dropping the exec chain means no
/// get ever runs.
#[test]
fn every_get_is_clocked_and_indexed_by_its_own_bank() {
    let (track, _opts, world, n_banks) = multi_bank();
    let by_target = sources_by_target(&world);
    let gets = inner_ids_of(&world, ARRAY_GET);
    assert_eq!(
        gets.len(),
        track.plan.len() * n_banks,
        "one get per band per bank"
    );

    let mut index_sources: HashSet<(usize, String)> = HashSet::new();
    for &get in &gets {
        let idx = by_target.get(&(get, "Index".to_string())).unwrap_or_else(|| {
            panic!("get {get} has no Index wire -- it would read frame 0 forever")
        });
        assert_eq!(idx.len(), 1, "get {get}: exactly one index source");
        index_sources.insert(idx[0].clone());

        let exec = by_target
            .get(&(get, "Exec".to_string()))
            .unwrap_or_else(|| panic!("get {get} is not in any exec chain -- it never runs"));
        assert_eq!(exec.len(), 1, "get {get}: exactly one exec input");
    }
    assert_eq!(
        index_sources.len(),
        n_banks,
        "each of the {n_banks} banks needs its OWN index -- bank 0 the raw frame \
         index, bank k that index minus k*bank_size"
    );
}

/// The save must reference exactly the three synth assets the bank uses, under
/// the one asset type that exists for them. Nothing on brdb's write path
/// validates either string: a typo in `BrickAudioDescriptor` or in an asset
/// name encodes perfectly and fails only in game.
#[test]
fn the_save_references_exactly_the_three_synth_assets() {
    // `control_buttons: false`: the button labels register a `BrickFontDescriptor`
    // asset reference, which is real but has nothing to do with the synth assets
    // this test is pinning.
    let opts = AudioOptions { control_buttons: false, ..AudioOptions::default() };
    let track = tone_track(1.0, &opts);
    let world = build_speaker_world(&track, &opts).expect("build");

    let mut got: Vec<(String, String)> = world
        .global_data
        .external_asset_references
        .iter()
        .cloned()
        .collect();
    got.sort();
    let mut want: Vec<(String, String)> = track
        .plan
        .kinds
        .iter()
        .map(|k| {
            (
                "BrickAudioDescriptor".to_string(),
                expected_asset(*k).to_string(),
            )
        })
        .collect();
    want.sort();
    want.dedup();
    assert_eq!(
        got, want,
        "every tonal band shares one sine reference; white and pink add one each"
    );
}

// ---------------------------------------------------------------------------
// The four control pins: `Inner Radius`, `Max Distance`, `Directional` and
// `Volume`.
//
// The governing rule for all four, and the only failure here that is invisible
// until the save is loaded in game: AN UNWIRED INPUT PIN DRIVES NOTHING. It
// does not deliver 0 or false to whatever it points at -- the target keeps its
// baked value. So every pin must have a baked value standing behind it, and
// those baked values are what the tests below pin down. Get that backwards and
// a default render hands every speaker `MaxDistance = 0` and
// `bSpatialization = false`, which builds, serialises and inspects perfectly
// and may well be silent in game.
// ---------------------------------------------------------------------------

/// Grid id of the microchip's inner grid inside a written save. Grid 1 is the
/// main grid (the speaker cluster); the chip's own grid follows it. Mirrors
/// `world.grids[0]` in memory.
const CHIP_GRID: usize = 2;

/// Every decoded component on the chip's inner grid, keyed by the in-memory
/// brick id that wires name.
///
/// Component DATA (a pin's `PortLabel`, a multiply's baked `InputB`) is opaque
/// in memory -- `LiteralComponent` stores it as `Box<dyn AsBrdbValue>` with no
/// way back out -- so it can only be read from a written save. But wire
/// topology is only expressible in terms of brick ids, so checking that a pin
/// LABELLED "Inner Radius" is the one WIRED to `InnerRadius` needs both spaces
/// joined.
///
/// The join is the component chunk's `component_brick_indices`, which index
/// the grid's brick list in emission order. That is only usable while the chip
/// is a single chunk, so both halves of that assumption are asserted rather
/// than trusted: if the chip ever grows past one chunk this panics loudly
/// instead of silently mismatching ids. The default 79-band chip is two
/// chunks, so callers need an explicit smaller `--bands` (e.g. 32) to keep
/// this join valid.
fn inner_structs_by_brick_id(world: &brdb::World, tag: &str) -> HashMap<usize, BrdbStruct> {
    let path = std::env::temp_dir().join(format!("h2b_audio_{tag}_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    let out = {
        let db = brdb::Brz::open(&path).expect("reopen").into_reader();
        let chunks = db
            .brick_chunk_index(CHIP_GRID)
            .expect("the chip's inner grid must exist in the save");
        let inner = &world.grids[0].1;
        assert_eq!(
            chunks.len(),
            1,
            "this join assumes the chip is one brick chunk; it is now {} -- the \
             brick-index-to-brick-id mapping below is no longer valid and must be \
             rewritten before this test can mean anything",
            chunks.len()
        );
        assert_eq!(
            chunks[0].num_bricks as usize,
            inner.len(),
            "the chip's single chunk must hold every inner brick"
        );
        let (soa, structs) = db
            .component_chunk_soa(CHIP_GRID, chunks[0].index)
            .expect("component chunk soa");
        assert_eq!(
            structs.len(),
            soa.component_brick_indices.len(),
            "every decoded component must pair with a brick index"
        );
        let mut map = HashMap::new();
        for (i, s) in structs.into_iter().enumerate() {
            let bi = soa.component_brick_indices[i] as usize;
            let id = inner[bi].id.expect("every inner brick carries an id");
            map.insert(id, s);
        }
        map
    };
    let _ = std::fs::remove_file(&path);
    out
}

/// `PortLabel` -> brick id, for every microchip INPUT pin on the inner grid.
fn input_pins_by_label(structs: &HashMap<usize, BrdbStruct>) -> HashMap<String, usize> {
    let mut out: HashMap<String, usize> = HashMap::new();
    for (id, s) in structs {
        if s.get_name() != "BrickComponentData_Internal_MicrochipInput" {
            continue;
        }
        let BrdbValue::String(label) = s.prop("PortLabel").expect("a pin must carry a PortLabel")
        else {
            panic!("PortLabel must decode as a string");
        };
        assert!(
            out.insert(label.clone(), *id).is_none(),
            "two input pins are both labelled {label:?} -- a builder cannot tell them apart"
        );
    }
    out
}

/// Every wire leaving `(brick, port)`, as (source component type, target brick,
/// target port).
///
/// The source's declared COMPONENT TYPE is carried deliberately.
/// `chip::pin_source(id, false)` names the microchip-OUTPUT component on a
/// brick that actually carries the INPUT one; the brick id, the port name and
/// every count in the graph are identical either way, so the declared type is
/// the only thing that distinguishes them.
fn outgoing(world: &brdb::World, brick: usize, port: &str) -> Vec<(String, usize, String)> {
    world
        .wires
        .iter()
        .filter(|w| w.source.brick_id == brick && w.source.port_name.as_ref() == port)
        .map(|w| {
            (
                w.source.component_type.to_string(),
                w.target.brick_id,
                w.target.port_name.to_string(),
            )
        })
        .collect()
}

/// The three attenuation pins must each reach EVERY speaker, on the emitter
/// port their own label promises.
///
/// Three separate silent failures live here. A pin wired to only the first
/// speaker splits the bank in two the moment it is driven. `Inner Radius` and
/// `Max Distance` are both floats on the same component, so swapping them
/// leaves every count, every brick and every wire endpoint intact and simply
/// makes the two knobs lie. And `bSpatialization` sits next to `bEnabled` on
/// that component, both bools -- pointing `Directional` at `bEnabled` gives a
/// pin that mutes the bank instead of spatialising it.
#[test]
fn the_three_attenuation_pins_reach_every_speaker_on_the_port_they_name() {
    // `--bands 32` explicitly, not the derived default -- see
    // `inner_structs_by_brick_id` for why.
    let opts = AudioOptions { bands: Some(32), ..Default::default() };
    let track = tone_track(1.0, &opts);
    let world = build_speaker_world(&track, &opts).expect("build");
    let speakers: HashSet<usize> = speaker_ids(&world).into_iter().collect();
    assert_eq!(speakers.len(), track.plan.len(), "one speaker per band");

    let structs = inner_structs_by_brick_id(&world, "pins");
    let pins = input_pins_by_label(&structs);

    for (label, port) in [
        ("Inner Radius", "InnerRadius"),
        ("Max Distance", "MaxDistance"),
        ("Directional", "bSpatialization"),
    ] {
        let &pin = pins.get(label).unwrap_or_else(|| {
            let mut have: Vec<&String> = pins.keys().collect();
            have.sort();
            panic!(
                "no input pin is labelled {label:?} -- a builder cannot drive what \
                 the chip never exposes. Pins present: {have:?}"
            )
        });
        let out = outgoing(&world, pin, "RER_Output");
        assert_eq!(
            out.len(),
            speakers.len(),
            "the {label:?} pin must fan out to all {} speakers, not {} -- a partial \
             fan-out splits the bank into driven and undriven halves",
            speakers.len(),
            out.len()
        );
        let mut reached: HashSet<usize> = HashSet::new();
        for (class, target, target_port) in &out {
            assert_eq!(
                class, MICROCHIP_INPUT,
                "{label:?}: a wire out of an input pin is sourced from the INPUT \
                 component's RER_Output, never the output component's"
            );
            assert_eq!(
                target_port, port,
                "the {label:?} pin must drive {port:?}, not {target_port:?}"
            );
            assert!(
                speakers.contains(target),
                "{label:?} reaches brick {target}, which is not a speaker"
            );
            assert!(
                reached.insert(*target),
                "{label:?} wires into speaker {target} twice"
            );
        }
    }

    // The Volume pin deliberately does NOT touch an emitter, and no longer fans
    // out to every band: it now drives the SINGLE shared pause-mute Select
    // (`scaffold`), whose gated output is what each per-band multiply reads.
    let &vol = pins
        .get("Volume")
        .expect("no input pin is labelled \"Volume\"");
    let out = outgoing(&world, vol, "RER_Output");
    assert_eq!(
        out.len(),
        1,
        "the Volume pin drives exactly the one shared pause-mute Select, not every band"
    );
    let comp = component_of(&world);
    let (_, target, port) = &out[0];
    assert!(
        !speakers.contains(target),
        "the Volume pin must never wire straight into an emitter"
    );
    assert_eq!(port, "InputB", "the Volume pin drives the Select's InputB");
    assert_eq!(
        comp.get(target).map(String::as_str),
        Some(SELECT),
        "the Volume pin must feed the pause-mute Select"
    );
}

/// The master volume goes through one multiply per band, with the band's value
/// on `InputA` and the GATED master (the pause-mute Select's output) on
/// `InputB`; that Select's own `InputB` is the raw `Volume` pin.
///
/// Run against the multi-bank fixture so `InputA` is fed by the select cascade
/// rather than a bare get -- the shape that actually ships for a long track.
///
/// Swapping the two inputs is the subtle one: multiplication commutes, so the
/// arithmetic is unchanged, but the baked literal does not move with the wire.
/// `InputB` is where the inlined 1.0 lives, so a swap leaves the audio on
/// `InputB` (overriding the literal) and the unwired pin on `InputA`, whose
/// value is then whatever the schema defaults it to.
#[test]
fn the_volume_pin_scales_every_band_through_its_own_multiply() {
    let (track, _opts, world, _n_banks) = multi_bank();
    let comp = component_of(&world);
    let by_target = sources_by_target(&world);
    let speakers = speaker_ids(&world);
    assert_eq!(speakers.len(), track.plan.len());

    let mut pin_sources: HashSet<usize> = HashSet::new();
    let mut multiplies: HashSet<usize> = HashSet::new();
    for (b, &speaker) in speakers.iter().enumerate() {
        let feeding = by_target
            .get(&(speaker, "VolumeMultiplier".to_string()))
            .unwrap_or_else(|| panic!("band {b}: nothing drives VolumeMultiplier"));
        assert_eq!(
            feeding.len(),
            1,
            "band {b}: VolumeMultiplier takes exactly one source -- two sources on \
             one input is what the multiply exists to avoid"
        );
        let (mul, port) = &feeding[0];
        assert_eq!(
            comp.get(mul).map(String::as_str),
            Some(MULTIPLY),
            "band {b}: the band's value must reach the speaker THROUGH the \
             master-volume multiply, not straight from the value cascade"
        );
        assert_eq!(port, "Output", "band {b}: a multiply feeds from its Output");
        assert!(
            multiplies.insert(*mul),
            "band {b} shares its volume multiply with another band"
        );

        let a = by_target
            .get(&(*mul, "InputA".to_string()))
            .unwrap_or_else(|| panic!("band {b}: the volume multiply's InputA is unwired"));
        assert_eq!(a.len(), 1, "band {b}: InputA takes one wire");
        assert!(
            matches!(
                comp.get(&a[0].0).map(String::as_str),
                Some(SELECT) | Some(ARRAY_GET)
            ),
            "band {b}: InputA must carry the band's own value (a select or a get), \
             got {:?} -- InputA and InputB are not interchangeable, the baked \
             unity gain lives on InputB",
            comp.get(&a[0].0)
        );

        let bee = by_target
            .get(&(*mul, "InputB".to_string()))
            .unwrap_or_else(|| panic!("band {b}: the volume multiply's InputB is unwired"));
        assert_eq!(bee.len(), 1, "band {b}: InputB takes one wire");
        assert_eq!(
            comp.get(&bee[0].0).map(String::as_str),
            Some(SELECT),
            "band {b}: InputB must be driven by the gated master (the pause-mute \
             Select), not straight from the Volume pin"
        );
        assert_eq!(bee[0].1, "Output", "band {b}: from the Select's Output");
        pin_sources.insert(bee[0].0);
    }
    assert_eq!(
        pin_sources.len(),
        1,
        "every band must read the SAME gated master -- {} distinct sources \
         means the mute (or the volume) reaches only part of the bank",
        pin_sources.len()
    );
    assert_eq!(multiplies.len(), track.plan.len(), "one multiply per band");

    // ...and that single gated master is the pause-mute Select, whose own
    // InputB is the raw Volume pin: the scale still ultimately comes from the
    // pin, one hop further back now.
    let select = *pin_sources.iter().next().unwrap();
    let sel_inputb = by_target
        .get(&(select, "InputB".to_string()))
        .expect("the pause-mute Select's InputB is unwired");
    assert_eq!(sel_inputb.len(), 1, "the Select's InputB takes one wire");
    assert_eq!(
        comp.get(&sel_inputb[0].0).map(String::as_str),
        Some(MICROCHIP_INPUT),
        "the gated master's InputB must be driven by the Volume pin"
    );
    assert_eq!(sel_inputb[0].1, "RER_Output", "from the pin's RER_Output");
}

/// The baked defaults behind the unwired pins (bSpatialization false, real
/// InnerRadius/MaxDistance) must reach the save.
#[test]
fn the_baked_emitter_defaults_that_stand_behind_the_pins_reach_the_save() {
    let opts = AudioOptions::default();
    let track = tone_track(1.0, &opts);
    let world = build_speaker_world(&track, &opts).expect("build");

    with_decoded_components(&world, "attenuation", |all, _| {
        let emitters: Vec<&BrdbStruct> = all
            .iter()
            .filter(|(_, _, s)| s.get_name() == "BrickComponentData_AudioEmitter")
            .map(|(_, _, s)| s)
            .collect();
        assert_eq!(emitters.len(), track.plan.len(), "one emitter per band");
        for (b, s) in emitters.iter().enumerate() {
            assert!(
                !bool_prop(s, "bSpatialization"),
                "band {b}: the baked bSpatialization must be false -- it is what the \
                 Directional pin overrides, and the game's own default is TRUE"
            );
            assert_eq!(
                f32_prop(s, "InnerRadius"),
                15.0,
                "band {b}: InnerRadius must be baked to the default near-field radius"
            );
            assert_eq!(
                f32_prop(s, "MaxDistance"),
                400.0,
                "band {b}: MaxDistance must be baked to the default audible range"
            );
            assert!(
                f32_prop(s, "MaxDistance") > 0.0,
                "band {b}: a zero MaxDistance is a speaker nobody can ever hear"
            );
            assert!(
                f32_prop(s, "InnerRadius") < f32_prop(s, "MaxDistance"),
                "band {b}: the no-attenuation radius must sit inside the audible range"
            );
        }
    });
}

// ---------------------------------------------------------------------------
// Cluster geometry.
//
// Position is audible: `bSpatialization = false` turns off panning, not
// distance attenuation, so the cluster must stay compact. The two properties
// below check that: nothing may be far from anything else, and no two
// speakers may share a slot.
// ---------------------------------------------------------------------------

/// Every speaker's position on the main grid, in band order.
fn speaker_positions(world: &brdb::World) -> Vec<brdb::Position> {
    world
        .bricks
        .iter()
        .filter(|b| {
            b.components.iter().any(|c| {
                c.component_type()
                    .map(|t| t.as_ref() == AUDIO_EMITTER)
                    .unwrap_or(false)
            })
        })
        .map(|b| b.position)
        .collect()
}

fn separation(a: brdb::Position, b: brdb::Position) -> f64 {
    let d = |p: i32, q: i32| (p - q) as f64;
    (d(a.x, b.x).powi(2) + d(a.y, b.y).powi(2) + d(a.z, b.z).powi(2)).sqrt()
}

/// The greatest distance between any two speakers, and the pair it belongs to.
fn max_separation(ps: &[brdb::Position]) -> (f64, usize, usize) {
    let mut worst = (0.0f64, 0usize, 0usize);
    for i in 0..ps.len() {
        for j in i + 1..ps.len() {
            let d = separation(ps[i], ps[j]);
            if d > worst.0 {
                worst = (d, i, j);
            }
        }
    }
    worst
}

/// No two speakers may be far apart, so the cluster reads as one point source
/// instead of a spread of distance-attenuated ones.
#[test]
fn no_two_speakers_are_far_apart() {
    // A fixed geometric bound, NOT derived from the inner radius: the near-field
    // default (inner radius 15) is deliberately smaller than the cluster, so the
    // cluster's compactness and the no-attenuation radius are two independent
    // properties now. 100 units clears the real ~44-unit cluster diagonal with
    // room to spare while a column fails it by an order of magnitude.
    const BOUND: f64 = 100.0;

    let opts = AudioOptions::default();
    let track = tone_track(1.0, &opts);
    let world = build_speaker_world(&track, &opts).expect("build");
    let ps = speaker_positions(&world);
    assert_eq!(ps.len(), track.plan.len(), "one speaker per band");

    let (worst, i, j) = max_separation(&ps);
    assert!(
        worst < BOUND,
        "speakers {i} and {j} are {worst:.1} units apart, over the {BOUND}-unit bound \
         -- the cluster must stay compact so the bank reads as one point source"
    );
    // The shipped bug was a COLUMN, one speaker stacked on the next. Naming
    // it keeps this test discriminating rather than merely satisfiable: a
    // change that unpacked the cluster fails here by an order of magnitude,
    // at any band count.
    let column = (ps.len() - 1) as f64 * 12.0;
    assert!(
        column > worst * 5.0,
        "a column of {} speakers is {column:.0} units and the cluster is {worst:.1} -- \
         if those are comparable the cluster is not packing anything",
        ps.len()
    );
}

/// No two speakers may occupy the same position.
///
/// Two bricks in one slot OVERLAP, and an overlap makes the game silently
/// delete one of the two -- a band that is simply missing from the render,
/// with a save that builds, serialises, and inspects perfectly. `chip::finish`
/// runs the same check over the main grid, so a collision is a build error
/// today; this states the property directly so a change that stops calling
/// `finish` (or an arrangement that only collides at some band counts) cannot
/// slip through.
#[test]
fn no_two_speakers_share_a_position() {
    let opts = AudioOptions::default();
    let track = tone_track(1.0, &opts);
    let world = build_speaker_world(&track, &opts).expect("build");
    let ps = speaker_positions(&world);

    let mut seen: HashMap<(i32, i32, i32), usize> = HashMap::new();
    for (b, p) in ps.iter().enumerate() {
        if let Some(other) = seen.insert((p.x, p.y, p.z), b) {
            panic!(
                "bands {other} and {b} are both at ({}, {}, {}) -- two bricks in one \
                 slot overlap, and the game silently DELETES one of them",
                p.x, p.y, p.z
            );
        }
    }
    assert_eq!(seen.len(), ps.len());
}

/// The same property, over every band count a user might ask for.
///
/// `--bands` is tunable, so the arrangement has to generalise. A layout that
/// only happens to be collision-free at 32 is a trap for the first person who
/// types `--bands 40`.
#[test]
fn every_band_count_packs_without_collisions_or_spread() {
    for n in [1usize, 2, 3, 4, 5, 7, 8, 9, 16, 24, 31, 32, 33, 48, 64, 65, 100, 128] {
        let ps: Vec<brdb::Position> = (0..n).map(|k| speaker_position(k, n)).collect();
        let mut seen: HashMap<(i32, i32, i32), usize> = HashMap::new();
        for (k, p) in ps.iter().enumerate() {
            if let Some(other) = seen.insert((p.x, p.y, p.z), k) {
                panic!("n={n}: speakers {other} and {k} share ({}, {}, {})", p.x, p.y, p.z);
            }
        }
        // The cluster must stay near-cubic, so its diameter grows like the
        // CUBE root of n. 4 * n^(1/3) cells is generous room around the true
        // sqrt(3) * n^(1/3); a column would need n cells and blows straight
        // through it.
        let cell = 12.0; // the largest measured cell dimension, in units
        let allowed = 4.0 * (n as f64).cbrt() * cell;
        let (worst, i, j) = max_separation(&ps);
        assert!(
            worst <= allowed,
            "n={n}: speakers {i} and {j} are {worst:.1} units apart, over the \
             {allowed:.1} a near-cubic packing allows -- the arrangement is not \
             compact at this band count"
        );
        // Every coordinate stays in the positive octant, which is what lets
        // the chip sit at a fixed negative x and clear the cluster whatever
        // `--bands` is.
        for (k, p) in ps.iter().enumerate() {
            assert!(
                p.x >= 0 && p.y >= 0 && p.z >= 0,
                "n={n}: speaker {k} at ({}, {}, {}) leaves the positive octant, \
                 where the chip lives",
                p.x, p.y, p.z
            );
        }
    }
}

/// The arrangement must be a pure function of the band count.
///
/// Band `k` has to land in the same slot on every run of every build. If it
/// does not, two renders of the same file differ, and an owner comparing them
/// by ear is comparing geometry rather than audio -- with nothing in either
/// save to show which is which. Checked both on the layout function and on a
/// world actually built twice.
#[test]
fn the_cluster_layout_is_deterministic() {
    for n in [1usize, 7, 32, 33, 100] {
        let first: Vec<(i32, i32, i32)> = (0..n)
            .map(|k| {
                let p = speaker_position(k, n);
                (p.x, p.y, p.z)
            })
            .collect();
        for attempt in 0..8 {
            let again: Vec<(i32, i32, i32)> = (0..n)
                .map(|k| {
                    let p = speaker_position(k, n);
                    (p.x, p.y, p.z)
                })
                .collect();
            assert_eq!(
                first, again,
                "n={n}: the layout changed on attempt {attempt} -- band k must always \
                 take the same slot"
            );
        }
    }

    // A FIXED oracle for the default bank, written out rather than
    // recomputed. Repeating the call proves only that the layout is
    // self-consistent WITHIN one process, which a per-process shuffle (a
    // hash-order dependency, a seed taken once at startup) also is. Only a
    // constant catches that: the mapping is x fastest, then y, then z, over a
    // 4x3x3 box on the measured (10, 10, 12) cell.
    for (band, want) in [
        (0usize, (5, 5, 6)),
        (1, (15, 5, 6)),
        (3, (35, 5, 6)),
        (4, (5, 15, 6)),
        (11, (35, 25, 6)),
        (12, (5, 5, 18)),
        (31, (35, 15, 30)),
    ] {
        let p = speaker_position(band, 32);
        assert_eq!(
            (p.x, p.y, p.z),
            want,
            "band {band} of 32 must always sit at {want:?}"
        );
    }

    let opts = AudioOptions::default();
    let track = tone_track(1.0, &opts);
    let a = build_speaker_world(&track, &opts).expect("build");
    let b = build_speaker_world(&track, &opts).expect("build");
    let pa: Vec<(i32, i32, i32)> = speaker_positions(&a).iter().map(|p| (p.x, p.y, p.z)).collect();
    let pb: Vec<(i32, i32, i32)> = speaker_positions(&b).iter().map(|p| (p.x, p.y, p.z)).collect();
    assert_eq!(
        pa, pb,
        "two builds of the same track must place every band identically"
    );
}

/// The cluster's slot counts must be near-cubic and must cover every band.
///
/// Two independent failures: too few slots and some band has nowhere to go
/// (or doubles up on another's), and a lopsided box is the column bug again
/// in a milder form.
#[test]
fn the_cluster_dimensions_are_near_cubic_and_sufficient() {
    for n in 1usize..=200 {
        let (nx, ny, nz) = cluster_dims(n);
        assert!(
            nx * ny * nz >= n,
            "n={n}: {nx}x{ny}x{nz} is only {} slots",
            nx * ny * nz
        );
        assert!(
            nx >= ny && ny >= nz,
            "n={n}: {nx}x{ny}x{nz} must be sorted so the fewest layers land on the \
             tallest cell axis"
        );
        // Near-cubic: the longest side may exceed the shortest by at most one
        // slot. A column (n x 1 x 1) fails this at every n above 2.
        assert!(
            nx - nz <= 1,
            "n={n}: {nx}x{ny}x{nz} is not within one slot of cubic"
        );
        // ...and it must not be wastefully large either: one slot bigger on
        // the long axis would still have to hold every band.
        assert!(
            nx * ny * nz < n + nx * ny,
            "n={n}: {nx}x{ny}x{nz} has a whole empty layer"
        );
    }
    // The default bank, stated outright so the shape is on the record.
    assert_eq!(cluster_dims(32), (4, 3, 3));
    assert_eq!(cluster_dims(8), (2, 2, 2), "a perfect cube stays a cube");
}

/// The chip shell must clear the cluster, at any band count.
///
/// Same silent-deletion bug as two speakers in one slot, and the chip's x was
/// originally computed against the column's geometry. `build_speaker_world`
/// runs `assert_bricks_dont_overlap` through `chip::finish`, so this test's
/// real assertion is that the build SUCCEEDS -- but it repeats the check
/// explicitly so a failure names the cluster rather than a generic overlap.
#[test]
fn the_chip_clears_the_cluster_at_any_band_count() {
    for bands in [3usize, 8, 32, 64] {
        let mut opts = AudioOptions::default();
        opts.bands = Some(bands);
        opts.noise_bands = if bands >= 4 { 2 } else { 0 };
        let track = tone_track(1.0, &opts);
        let world = build_speaker_world(&track, &opts)
            .unwrap_or_else(|e| panic!("--bands {bands} must build: {e}"));
        heightmap::anim::layout::assert_bricks_dont_overlap(&world.bricks)
            .unwrap_or_else(|e| panic!("--bands {bands}: {e}"));
        assert_eq!(speaker_positions(&world).len(), bands);
    }
}

/// The two radii are options now, and they must reach the emitters they
/// configure -- on the field each one names.
///
/// Both are bare `f32`s sitting next to each other on the same component, so a
/// swap changes no count, no wire and no type. And a flag that parses but
/// never reaches the emitter leaves the render silently on its defaults, which
/// is precisely the failure the flags exist to let the owner escape.
#[test]
fn the_radius_options_reach_every_emitter_on_the_field_they_name() {
    let mut opts = AudioOptions::default();
    // Unequal, neither one a default, and deliberately not a round multiple
    // of the other.
    opts.inner_radius = 123.0;
    opts.max_distance = 4567.0;
    assert_ne!(opts.inner_radius, AudioOptions::default().inner_radius);
    assert_ne!(opts.max_distance, AudioOptions::default().max_distance);

    let track = tone_track(1.0, &opts);
    let world = build_speaker_world(&track, &opts).expect("build");

    with_decoded_components(&world, "radii", |all, _| {
        let emitters: Vec<&BrdbStruct> = all
            .iter()
            .filter(|(_, _, s)| s.get_name() == "BrickComponentData_AudioEmitter")
            .map(|(_, _, s)| s)
            .collect();
        assert_eq!(emitters.len(), track.plan.len(), "one emitter per band");
        for (b, s) in emitters.iter().enumerate() {
            assert_eq!(
                f32_prop(s, "InnerRadius"),
                123.0,
                "band {b}: --inner-radius must reach InnerRadius, not MaxDistance"
            );
            assert_eq!(
                f32_prop(s, "MaxDistance"),
                4567.0,
                "band {b}: --max-distance must reach MaxDistance, not InnerRadius"
            );
        }
    });
}

/// A radius that cannot make a sound must be an error, not a save.
///
/// Both fields are baked and never looked at again, so a zero, a NaN or an
/// inverted pair produces a file that builds, serialises and inspects
/// perfectly and is silent -- or audible nowhere -- only in game.
#[test]
fn an_impossible_radius_is_rejected_rather_than_baked() {
    let base = AudioOptions::default();
    let track = tone_track(1.0, &base);
    for (name, inner, max) in [
        ("zero max distance", 0.0f32, 0.0f32),
        ("zero inner radius", 0.0, 4000.0),
        ("negative max distance", 400.0, -1.0),
        ("nan inner radius", f32::NAN, 4000.0),
        ("infinite max distance", 400.0, f32::INFINITY),
        ("inner outside max", 4000.0, 400.0),
    ] {
        let mut opts = base;
        opts.inner_radius = inner;
        opts.max_distance = max;
        assert!(
            build_speaker_world(&track, &opts).is_err(),
            "{name} (inner {inner}, max {max}) must be rejected, not written into \
             every emitter"
        );
    }
}

/// The volume multiplies bake unity gain, so an unwired `Volume` pin leaves
/// the render bit-identical to one with no pin at all.
///
/// `build_clock` emits a `MathMultiply` of its own (the fps), so "every
/// multiply bakes 1.0" would be false and "there are exactly 32 multiplies"
/// falser still. The two are told apart by wiring, not by counting: the volume
/// multiplies are exactly those whose `Output` drives a speaker's
/// `VolumeMultiplier`. Filtering on the data struct name would not work either
/// -- `MathMultiply`, `MathSubtract` and `MathModuloFloored` all serialise as
/// `BrickComponentData_WireGraph_Expr_PrimMathVariantPrimMathVariant_PrimMathVariant`.
#[test]
fn the_volume_multiplies_bake_unity_gain_so_an_unwired_pin_changes_nothing() {
    // `--bands 32` explicitly, not the derived default -- see
    // `inner_structs_by_brick_id` for why.
    let opts = AudioOptions { bands: Some(32), ..Default::default() };
    let track = tone_track(1.0, &opts);
    assert!(
        (track.fps - 1.0).abs() > 0.5,
        "this test tells the volume multiplies from the clock's by their baked \
         value, so the fps must not itself be 1.0"
    );
    let world = build_speaker_world(&track, &opts).expect("build");

    // Every MathMultiply on the inner grid: one volume multiply per band, plus
    // the clock's three (fps multiply + length and progress status taps).
    let all_multiplies = inner_ids_of(&world, MULTIPLY);
    assert_eq!(
        all_multiplies.len(),
        track.plan.len() + 3,
        "one volume multiply per band, plus the clock's fps multiply and its \
         length + progress status taps"
    );

    let by_target = sources_by_target(&world);
    let volume_multiplies: HashSet<usize> = speaker_ids(&world)
        .iter()
        .map(|s| {
            by_target
                .get(&(*s, "VolumeMultiplier".to_string()))
                .expect("every speaker's volume is driven")[0]
                .0
        })
        .collect();
    assert_eq!(volume_multiplies.len(), track.plan.len());

    let structs = inner_structs_by_brick_id(&world, "gain");
    for &mul in &volume_multiplies {
        let s = structs
            .get(&mul)
            .unwrap_or_else(|| panic!("multiply {mul} has no component data in the save"));
        let v = s.prop("InputB").expect("a multiply must carry an InputB");
        let WireVariant::Number(n) =
            WireVariant::try_from(v).expect("a math port is a wire variant, not a bare f32")
        else {
            panic!("the baked gain must be a Number, got {v:?}");
        };
        assert_eq!(
            n, 1.0,
            "multiply {mul} bakes a gain of {n}, not 1.0 -- that is the value the \
             gate uses while the Volume pin is UNWIRED, so anything else rescales \
             (0.0 mutes) every render nobody has touched"
        );
    }

    // ...and among the ones left over -- the clock's fps multiply plus the
    // length and progress status taps -- one still carries the fps. (The length
    // tap bakes InputB 1.0 like a volume multiply, so the fps one is told apart
    // by its value, not by being the sole survivor.)
    let leftover: Vec<usize> = all_multiplies
        .iter()
        .copied()
        .filter(|m| !volume_multiplies.contains(m))
        .collect();
    assert_eq!(leftover.len(), 3, "the fps multiply plus the length and progress taps");
    let carries_fps = leftover.iter().any(|m| {
        structs[m]
            .prop("InputB")
            .ok()
            .and_then(|v| WireVariant::try_from(v).ok())
            .is_some_and(|wv| matches!(wv, WireVariant::Number(n) if n == track.fps as f64))
    });
    assert!(carries_fps, "the clock's own fps multiply must survive carrying the fps");
}

/// The emitter's pin-backed properties must be written explicitly by this
/// crate, not left for brdb's regenerated `STRUCT_DEFAULTS` to fill in.
#[test]
fn the_emitter_properties_behind_the_pins_are_written_explicitly() {
    let opts = AudioOptions::default();
    let track = tone_track(1.0, &opts);
    let world = build_speaker_world(&track, &opts).expect("build");

    let schema = brdb::schemas::bricks_components_schema_max();
    let emitter = schema
        .intern
        .get("BrickComponentData_AudioEmitter")
        .expect("the max component schema must describe the audio emitter");

    let mut checked = 0;
    for brick in &world.bricks {
        for c in &brick.components {
            let Some(ty) = c.component_type() else { continue };
            if ty.as_ref() != AUDIO_EMITTER {
                continue;
            }
            for prop in [
                "bSpatialization",
                "InnerRadius",
                "MaxDistance",
                "PitchMultiplier",
                "VolumeMultiplier",
                "bEnabled",
            ] {
                let interned = schema
                    .intern
                    .get(prop)
                    .unwrap_or_else(|| panic!("{prop} is not a name the schema knows"));
                assert!(
                    c.has_brdb_struct_prop(schema, emitter, interned),
                    "every speaker must carry its own {prop}; leaving it out hands the \
                     value to brdb's STRUCT_DEFAULTS, which is regenerated per game \
                     build and would move the render with it"
                );
            }
            checked += 1;
        }
    }
    assert_eq!(checked, track.plan.len(), "one emitter per band");
}

// ---------------------------------------------------------------------------
// --audio-mode voice
//
// The voice renderer shares the cluster, the chip, the clock, the four pins
// and the banking cascade with the band bank, so what needs testing here is
// exactly what DIFFERS: twice as many arrays, one of them wired into
// `PitchMultiplier` instead of a volume multiply, and a speaker count that is
// the voice count rather than the band count.
// ---------------------------------------------------------------------------

use heightmap::audio::speakers::build_voice_world;
use heightmap::audio::voices::{VoiceStreams, analyze_voices};

fn voice_streams(secs: f32, opts: &AudioOptions) -> VoiceStreams {
    let sr = 48_000u32;
    let n = (sr as f32 * secs) as usize;
    let clip = SampleClip::new(
        sr,
        (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.4 * (TAU * 440.0 * t).sin() + 0.3 * (TAU * 660.0 * t).sin()
            })
            .collect(),
    );
    analyze_voices(&clip, opts, &mut NoProgress).expect("analyze")
}

fn voice_opts(voices: usize) -> AudioOptions {
    AudioOptions {
        max_voices: voices,
        window: 4096,
        ..AudioOptions::default()
    }
}

/// `--max-voices` is the SPEAKER COUNT in this mode, not a cap on a bank.
#[test]
fn a_voice_render_builds_exactly_max_voices_speakers() {
    for n in [1usize, 4, 9] {
        let o = voice_opts(n);
        let streams = voice_streams(1.0, &o);
        assert_eq!(streams.voice_count(), n);
        let world = build_voice_world(&streams, &o).expect("build");
        let speakers = world
            .bricks
            .iter()
            .flat_map(|b| b.components.iter())
            .filter(|c| {
                c.component_type()
                    .map(|t| t.as_ref() == "Component_AudioEmitter")
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(speakers, n, "--max-voices {n} must build {n} speakers");
    }
}

/// **Two arrays per voice, not one.** This is the whole shape of the mode: a
/// pitch stream and a volume stream, both dense. A build that emitted one
/// array per voice would be the band bank with a wrong speaker count.
#[test]
fn a_voice_render_emits_two_arrays_and_two_gets_per_voice() {
    let o = voice_opts(5);
    let streams = voice_streams(1.0, &o);
    let world = build_voice_world(&streams, &o).expect("build");
    let types = inner_components(&world);
    assert_eq!(
        types.iter().filter(|t| t.as_str() == ARRAY_VAR).count(),
        10,
        "5 voices must carry 10 ArrayVars (pitch + volume each)"
    );
    assert_eq!(
        types.iter().filter(|t| t.as_str() == ARRAY_GET).count(),
        10,
        "one ArrayVar_Get per array"
    );
}

/// **The wire that makes this mode different from the bank.** Every speaker's
/// `PitchMultiplier` must be driven, and driven from an `ArrayVar_Get`
/// DIRECTLY -- never through the master-volume multiply, which would transpose
/// the whole render whenever a builder touched the `Volume` pin.
#[test]
fn every_speakers_pitch_is_wired_from_its_own_array_not_through_the_volume_multiply() {
    let o = voice_opts(4);
    let streams = voice_streams(1.0, &o);
    let world = build_voice_world(&streams, &o).expect("build");
    let component = component_of(&world);
    let sources = sources_by_target(&world);
    let speakers = speaker_ids(&world);
    assert_eq!(speakers.len(), 4);

    for (v, &speaker) in speakers.iter().enumerate() {
        let pitch = sources
            .get(&(speaker, "PitchMultiplier".to_string()))
            .unwrap_or_else(|| panic!("voice {v}'s PitchMultiplier is not driven at all"));
        assert_eq!(
            pitch.len(),
            1,
            "voice {v}'s PitchMultiplier must have exactly one source, got {pitch:?}"
        );
        assert_eq!(
            component.get(&pitch[0].0).map(String::as_str),
            Some(ARRAY_GET),
            "voice {v}'s pitch must come straight from an ArrayVar_Get, not through a gate"
        );

        // ...while the volume still goes through the multiply, so the `Volume`
        // pin scales levels and only levels.
        let vol = sources
            .get(&(speaker, "VolumeMultiplier".to_string()))
            .unwrap_or_else(|| panic!("voice {v}'s VolumeMultiplier is not driven"));
        assert_eq!(
            component.get(&vol[0].0).map(String::as_str),
            Some(MULTIPLY),
            "voice {v}'s volume must still pass through the master-volume multiply"
        );
    }
}

/// The four builder-facing pins are the same in both modes; a chip that lost
/// them in the new path would look identical until someone tried to use it.
#[test]
fn a_voice_render_keeps_the_four_input_pins_and_the_clock() {
    let o = voice_opts(3);
    let streams = voice_streams(1.0, &o);
    let world = build_voice_world(&streams, &o).expect("build");
    let types = inner_components(&world);
    // Five clock pins (Pause/Restart/Resume/Rate) plus the four here; `Done`
    // is an output pin.
    assert!(
        types.iter().filter(|t| t.as_str() == MICROCHIP_INPUT).count() >= 8,
        "the clock's four control pins and the four audio pins must all survive"
    );
    assert!(types.iter().any(|t| t.as_str() == MULTIPLY), "the clock must be built");
}

/// The baked `PitchMultiplier` is frame 0's value, so a paused or unwired chip
/// holds the first note rather than an arbitrary one -- and never 0, which the
/// game would clamp to 0.1 and play at 44 Hz.
#[test]
fn each_speakers_baked_pitch_is_its_own_first_frame() {
    let o = voice_opts(4);
    let streams = voice_streams(1.0, &o);
    let expected: Vec<f32> = (0..streams.voice_count())
        .map(|v| streams.pitches[v][0] as f32)
        .collect();
    let world = build_voice_world(&streams, &o).expect("build");
    with_decoded_components(&world, "voice_baked_pitch", |structs, _| {
        let mut seen = Vec::new();
        for (_, _, s) in structs {
            if s.get_name() == "BrickComponentData_AudioEmitter" {
                seen.push(f32_prop(s, "PitchMultiplier"));
            }
        }
        assert_eq!(seen.len(), expected.len());
        for (i, (got, want)) in seen.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "speaker {i} baked pitch {got}, expected frame 0's {want}"
            );
            assert!(*got >= 0.1 && *got <= 10.0, "speaker {i} baked pitch {got} is unplayable");
        }
    });
}

/// A stream carrying an unplayable pitch must be REFUSED, not clamped in game
/// into a wrong note. Nothing on the write path checks it, and nothing in a
/// written save would show it.
#[test]
fn an_unplayable_pitch_is_rejected_rather_than_clamped_in_game() {
    let o = voice_opts(2);
    for bad in [0.0f64, 0.05, 12.0, f64::NAN] {
        let mut streams = voice_streams(1.0, &o);
        streams.pitches[1][3] = bad;
        assert!(
            build_voice_world(&streams, &o).is_err(),
            "PitchMultiplier {bad} must be rejected"
        );
    }
}

/// Ragged streams must be refused: both arrays of every voice are read at the
/// same frame index, so a short one leaves that voice stuck on a stale value
/// partway through, silently.
#[test]
fn ragged_streams_are_rejected() {
    let o = voice_opts(3);
    let mut streams = voice_streams(1.0, &o);
    streams.volumes[2].pop();
    assert!(build_voice_world(&streams, &o).is_err());
}

#[test]
fn a_voice_world_serialises() {
    let o = voice_opts(6);
    let streams = voice_streams(1.0, &o);
    let world = build_voice_world(&streams, &o).expect("build");
    let bytes = world.to_brz_vec().expect("serialise");
    assert!(bytes.len() > 500);
}

/// Voice mode's dense per-voice arrays compress smaller than the bank's mostly-
/// zero ones, but only up to a small voice count; pinned at V=8, where it holds.
#[test]
fn a_voice_render_is_smaller_than_the_equivalent_bank_render() {
    let vo = voice_opts(8);
    let voice = build_voice_world(&voice_streams(2.0, &vo), &vo)
        .expect("voice build")
        .to_brz_vec()
        .expect("voice serialise");
    let bo = AudioOptions { window: 4096, ..AudioOptions::default() };
    let bank = build_speaker_world(&tone_track(2.0, &bo), &bo)
        .expect("bank build")
        .to_brz_vec()
        .expect("bank serialise");
    assert!(
        voice.len() < bank.len(),
        "8 voices ({} bytes) must be smaller than the 79-band bank ({} bytes)",
        voice.len(),
        bank.len()
    );
}

/// The audio path drives the SAME clock the video path does, so `loop_playback`
/// has to reach `Timer.Limit` here too -- and the value has to survive the
/// encode, which only reading the save back can show.
///
/// Split across both settings in one test because the interesting assertion is
/// the pair: looping writes the free-running 0, not-looping writes a real
/// limit derived from the track's own frame count and fps. A single-setting
/// test would pass against a build that ignored the flag entirely.
///
/// `.brz` bytes are not reproducible run to run, so this checks structure --
/// the decoded `Limit` -- and never a hash or a size.
#[test]
fn loop_playback_reaches_the_timer_limit_in_an_audio_render() {
    fn written_limit(loop_playback: bool, tag: &str) -> (f64, f32, usize) {
        let opts = AudioOptions { loop_playback, ..AudioOptions::default() };
        let track = tone_track(1.0, &opts);
        let (fps, frames) = (track.fps, track.frame_count);
        let world = build_speaker_world(&track, &opts).expect("build");
        let limit = with_decoded_components(&world, tag, |all, _| {
            let mut found = None;
            for (_, _, s) in all {
                if s.get_name() == "BrickComponentData_WireGraphPseudo_Timer" {
                    match s.get("Limit") {
                        Some(BrdbValue::F64(v)) => found = Some(*v),
                        other => panic!("Timer.Limit decoded as {other:?}, expected an f64"),
                    }
                }
            }
            found.expect("a speaker world must carry exactly one Timer")
        });
        (limit, fps, frames)
    }

    let (looping, _, _) = written_limit(true, "loop_on");
    assert_eq!(
        looping, 0.0,
        "a looping audio render must write the free-running Limit, as it always has"
    );

    let (stopping, fps, frames) = written_limit(false, "loop_off");
    assert!(frames > 1, "the fixture must have real frames for this to mean anything");
    assert_eq!(
        stopping,
        (frames as f64 - 0.5) / fps as f64,
        "a non-looping audio render must write (frames - 0.5) / fps"
    );

    // The same property the video path is pinned on: the limit leaves the
    // frame index on the LAST frame, not wrapped round to the first.
    let index = ((stopping * fps as f64).floor() as i64).rem_euclid(frames as i64);
    assert_eq!(index, frames as i64 - 1, "the track must stop on its last frame");
    assert_ne!(index, 0);
}

// ---------------------------------------------------------------------------
// Pause-mute: silence while the clock is frozen.
//
// The mechanism gates the master volume on whether `Timer.Time` is still
// ADVANCING (`BufferTicks` one tick back, `CompareNotEqual` against it, a
// `Select` on the result), so ANY stall -- the Pause exec, a stalled external
// clock, a no-loop track that ended -- drops the bank to 0 without depending on
// how the pause was wired. Three shared gates for the whole bank, +3 regardless
// of speaker count. All of it is topology and baked literals: nothing here
// shows up in a gate count alone.
// ---------------------------------------------------------------------------

/// The whole pause-mute chain, traced end to end from the Timer to every
/// speaker's multiply. The three gates and five wires all have to point the
/// right way; a single reversed or mis-named port dangles silently in game.
///
/// Run on the multi-bank fixture so there are also value-cascade `Select`s in
/// the graph, which proves the "bSelectB from a CompareNotEqual" filter really
/// isolates the ONE pause-mute Select rather than tripping over the others.
#[test]
fn the_pause_mute_gate_silences_every_speaker_when_the_clock_stops() {
    let (_track, _opts, world, _n_banks) = multi_bank();
    let comp = component_of(&world);
    let by_target = sources_by_target(&world);
    let by_source = targets_by_source(&world);

    // The pause-mute Select is the one whose bSelectB comes from a
    // CompareNotEqual; every value-cascade Select takes bSelectB from a
    // CompareGreaterOrEqual instead.
    let gated: Vec<usize> = inner_ids_of(&world, SELECT)
        .into_iter()
        .filter(|&s| {
            by_target
                .get(&(s, "bSelectB".to_string()))
                .and_then(|srcs| srcs.first())
                .map(|(id, _)| comp.get(id).map(String::as_str) == Some(COMPARE_NE))
                .unwrap_or(false)
        })
        .collect();
    assert_eq!(gated.len(), 1, "exactly one shared pause-mute Select");
    let select = gated[0];

    // bSelectB <- CompareNotEqual.bOutput ("playing")
    let cond = &by_target[&(select, "bSelectB".to_string())];
    assert_eq!(cond.len(), 1, "bSelectB takes one wire");
    let cmp = cond[0].0;
    assert_eq!(comp.get(&cmp).map(String::as_str), Some(COMPARE_NE));
    assert_eq!(cond[0].1, "bOutput", "playing = the != comparator's bOutput");

    // CompareNotEqual.InputA <- Timer.Time ; InputB <- BufferTicks.Output
    let a = &by_target[&(cmp, "InputA".to_string())];
    let b = &by_target[&(cmp, "InputB".to_string())];
    assert_eq!(comp.get(&a[0].0).map(String::as_str), Some(TIMER));
    assert_eq!(a[0].1, "Time", "InputA is the live clock time");
    let buffer = b[0].0;
    assert_eq!(comp.get(&buffer).map(String::as_str), Some(BUFFER_TICKS));
    assert_eq!(b[0].1, "Output", "InputB is last tick's time");

    // BufferTicks.Input <- the SAME Timer.Time (dataflow fan-out)
    let bi = &by_target[&(buffer, "Input".to_string())];
    assert_eq!(comp.get(&bi[0].0).map(String::as_str), Some(TIMER));
    assert_eq!(bi[0].1, "Time");
    assert_eq!(
        bi[0].0, a[0].0,
        "the buffer and the comparator must tap the same Timer.Time"
    );

    // Select.InputA is baked (unwired 0.0); Select.InputB is the Volume pin.
    assert!(
        by_target.get(&(select, "InputA".to_string())).is_none(),
        "InputA is the baked 0.0 -- it must be UNWIRED, or a source overrides the mute"
    );
    let sel_b = &by_target[&(select, "InputB".to_string())];
    assert_eq!(comp.get(&sel_b[0].0).map(String::as_str), Some(MICROCHIP_INPUT));
    assert_eq!(sel_b[0].1, "RER_Output", "the Volume pin feeds the Select's InputB");

    // Select.Output fans out to EVERY speaker's volume multiply, on InputB.
    let out = by_source
        .get(&(select, "Output".to_string()))
        .expect("the gated master drives nothing");
    let speakers = speaker_ids(&world);
    assert_eq!(
        out.len(),
        speakers.len(),
        "the gated master must reach every band's multiply"
    );
    for (mul, port) in out {
        assert_eq!(comp.get(mul).map(String::as_str), Some(MULTIPLY));
        assert_eq!(port, "InputB", "into each volume multiply's InputB");
    }
}

/// The Select's baked literals, read back from a real save: InputA is 0.0 (what
/// a FROZEN clock emits) and InputB is 1.0 (VOLUME_SCALE, what a PLAYING render
/// with the Volume pin unwired emits). Together these are the promise that the
/// feature is INAUDIBLE until the clock stops -- a no-pause render still
/// multiplies every band by 1.0 while playing, exactly as before the feature.
///
/// A single-bank render, so the only `Select` in the whole graph is the
/// pause-mute one and the read-back cannot pick the wrong gate.
#[test]
fn the_pause_mute_selects_baked_literals_keep_a_playing_render_unchanged() {
    let opts = AudioOptions { bands: Some(6), ..Default::default() };
    let track = tone_track(1.0, &opts);
    let world = build_speaker_world(&track, &opts).expect("build");

    with_decoded_components(&world, "pause_mute_select", |all, _| {
        let selects: Vec<&BrdbStruct> = all
            .iter()
            .filter(|(_, _, s)| s.get_name() == "BrickComponentData_WireGraph_Expr_Select")
            .map(|(_, _, s)| s)
            .collect();
        assert_eq!(
            selects.len(),
            1,
            "a single-bank render's only Select is the pause-mute gate"
        );
        let s = selects[0];

        let number = |prop: &str| -> f64 {
            let v = s.prop(prop).unwrap_or_else(|e| panic!("{prop}: {e}"));
            let WireVariant::Number(n) =
                WireVariant::try_from(v).expect("a Select input is a wire variant")
            else {
                panic!("{prop} did not decode as a Number");
            };
            n
        };
        assert_eq!(number("InputA"), 0.0, "frozen clock -> silence");
        assert_eq!(
            number("InputB"),
            1.0,
            "playing with Volume unwired -> unity gain, unchanged from before the feature"
        );
    });
}

/// Every wire in a BANK render must resolve to a brick that actually carries
/// the referenced component -- the three pause-mute gates included. A port-name
/// typo on the new gates (`BufferTicks.Input`, `CompareNotEqual.bOutput`,
/// `Select.bSelectB`/`InputB`) would encode fine, pass a range-only check, and
/// dangle silently in game. This is the loader's own port resolution.
#[test]
fn a_bank_render_passes_wire_integrity() {
    let opts = AudioOptions::default();
    let track = tone_track(1.0, &opts);
    let world = build_speaker_world(&track, &opts).expect("build");
    let path =
        std::env::temp_dir().join(format!("h2b_audio_wi_bank_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    wire_integrity::assert_wires_valid(&path);
    let _ = std::fs::remove_file(&path);
}

/// The same, for a VOICE render: it shares `scaffold` with the bank, so the
/// pause-mute gates are wired identically, but the surrounding graph differs
/// (two streams per voice, a pitch wire straight into the emitter) and the
/// harness must resolve every wire there too.
#[test]
fn a_voice_render_passes_wire_integrity() {
    let o = voice_opts(4);
    let streams = voice_streams(1.0, &o);
    let world = build_voice_world(&streams, &o).expect("build");
    let path =
        std::env::temp_dir().join(format!("h2b_audio_wi_voice_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    wire_integrity::assert_wires_valid(&path);
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// --speakers-in-chip: the whole device as one portable microchip.
//
// The one thing this option changes is WHICH GRID the speakers sit on: the
// chip's own inner grid instead of the world's main grid beside it. Everything
// else -- the emitter data, the gate graph, every wire endpoint -- is identical,
// and brdb resolves each wire to a local or remote source at write time from
// where the bricks land, so the cross-grid remote wires simply become same-grid
// internal ones. The two facts these tests CANNOT establish, because they are
// game-runtime behaviour, are that an in-chip speaker plays at all and that it
// plays from the chip's origin; the owner must confirm both on a test render.
// ---------------------------------------------------------------------------

/// Inner-grid emitter ids, in band/voice order (the in-chip analogue of the
/// main-grid `speaker_ids`).
fn inner_speaker_ids(world: &brdb::World) -> Vec<usize> {
    inner_ids_of(world, AUDIO_EMITTER)
}

/// ON: every speaker moves onto the chip's inner grid, none stay on the main
/// grid, and each one's `VolumeMultiplier` is still driven through its own
/// master-volume multiply. OFF: the default beside-the-chip cluster is
/// unchanged -- every speaker on the main grid, at exactly its
/// [`speaker_position`] slot, and nothing extra on the inner grid.
#[test]
fn speakers_in_chip_moves_the_cluster_onto_the_inner_grid_and_off_leaves_it() {
    let base = AudioOptions { bands: Some(16), ..Default::default() };
    let track = tone_track(1.0, &base);
    let n = track.plan.len();

    // OFF (default): speakers on the main grid at the beside-the-chip slots,
    // nothing on the inner grid. This is the "existing render is unchanged"
    // guard.
    let off = build_speaker_world(&track, &base).expect("build off");
    assert_eq!(speaker_ids(&off).len(), n, "OFF: every speaker on the main grid");
    assert_eq!(
        inner_speaker_ids(&off).len(),
        0,
        "OFF: no emitter on the inner grid"
    );
    let off_positions: Vec<(i32, i32, i32)> =
        speaker_positions(&off).iter().map(|p| (p.x, p.y, p.z)).collect();
    let want: Vec<(i32, i32, i32)> = (0..n)
        .map(|k| {
            let p = speaker_position(k, n);
            (p.x, p.y, p.z)
        })
        .collect();
    assert_eq!(
        off_positions, want,
        "OFF placement must be the beside-the-chip cluster, unchanged from before the flag"
    );

    // ON: every speaker on the inner grid, none on the main grid.
    let on_opts = AudioOptions { speakers_in_chip: true, ..base };
    let on = build_speaker_world(&track, &on_opts).expect("build on");
    assert_eq!(
        speaker_ids(&on).len(),
        0,
        "ON: no speaker may remain on the main grid"
    );
    let inner = inner_speaker_ids(&on);
    assert_eq!(inner.len(), n, "ON: every speaker on the chip's inner grid");

    // ...and their volume wires resolve, through the same per-band multiply as
    // the beside-the-chip layout -- a same-grid internal wire now, not a
    // cross-grid remote one, but the topology is identical.
    let comp = component_of(&on);
    let by_target = sources_by_target(&on);
    for (b, &speaker) in inner.iter().enumerate() {
        let feeding = by_target
            .get(&(speaker, "VolumeMultiplier".to_string()))
            .unwrap_or_else(|| panic!("band {b}: in-chip speaker VolumeMultiplier is unwired"));
        assert_eq!(feeding.len(), 1, "band {b}: exactly one wire drives the volume");
        assert_eq!(
            comp.get(&feeding[0].0).map(String::as_str),
            Some(MULTIPLY),
            "band {b}: the volume must reach the in-chip speaker through its master-volume multiply"
        );
    }
}

/// The written `.brz` proves the move: with the option ON every emitter decodes
/// on the chip's own grid ([`CHIP_GRID`]), never the main grid; with it OFF
/// every emitter decodes on the main grid, exactly as before the flag. Reading
/// the save back is the only thing that shows which grid a component landed in.
#[test]
fn the_saved_file_puts_in_chip_speakers_in_the_chip_grid_not_the_main_grid() {
    let emitter_grids = |world: &brdb::World, tag: &str| -> HashSet<usize> {
        with_decoded_components(world, tag, |all, _| {
            all.iter()
                .filter(|(_, _, s)| s.get_name() == "BrickComponentData_AudioEmitter")
                .map(|(gid, _, _)| *gid)
                .collect()
        })
    };

    let on = AudioOptions { speakers_in_chip: true, bands: Some(16), ..Default::default() };
    let track = tone_track(1.0, &on);
    let world = build_speaker_world(&track, &on).expect("build on");
    assert_eq!(
        emitter_grids(&world, "in_chip_grid"),
        HashSet::from([CHIP_GRID]),
        "every in-chip emitter must decode on the chip's inner grid ({CHIP_GRID})"
    );

    let off = AudioOptions { bands: Some(16), ..Default::default() };
    let track = tone_track(1.0, &off);
    let world = build_speaker_world(&track, &off).expect("build off");
    assert_eq!(
        emitter_grids(&world, "beside_chip_grid"),
        HashSet::from([1]),
        "the default layout keeps every emitter on the main grid (1)"
    );
}

/// The in-chip block must clear the gate lattice at every band count, and put
/// nothing on the main grid. `build_speaker_world` runs `assert_no_overlap`
/// over the whole inner grid through `chip::finish`, so a build that SUCCEEDS
/// is the proof the speaker block never collides with a gate -- and, in a debug
/// build, that no speaker went to a negative inner coordinate (the
/// `recompute_plane_extent` debug_assert).
#[test]
fn in_chip_speakers_clear_the_gate_lattice_at_any_band_count() {
    for bands in [3usize, 8, 32, 64, 79] {
        let mut opts = AudioOptions { speakers_in_chip: true, ..Default::default() };
        opts.bands = Some(bands);
        opts.noise_bands = if bands >= 4 { 2 } else { 0 };
        let track = tone_track(1.0, &opts);
        let world = build_speaker_world(&track, &opts)
            .unwrap_or_else(|e| panic!("--bands {bands} --speakers-in-chip must build: {e}"));
        assert_eq!(
            inner_speaker_ids(&world).len(),
            track.plan.len(),
            "--bands {bands}: every speaker must land on the inner grid"
        );
        assert_eq!(
            speaker_ids(&world).len(),
            0,
            "--bands {bands}: nothing may sit on the main grid but the chip shell"
        );
    }
}

/// The layout function itself: every coordinate non-negative (negative
/// inner-grid coordinates delete bricks in-game), and every speaker's low z
/// face at or above the gate layer. The block stacks ABOVE the gates (all at
/// lattice stage 0) rather than beside them, so it clears the playhead lattice
/// however far it spreads across x/y -- the MIDI playhead grows along both axes.
#[test]
fn in_chip_speaker_positions_are_nonnegative_and_clear_of_the_gate_layer() {
    // Every in-chip gate sits at lattice stage 0, whose top z-face is
    // STAGE_BASE_Z (6) + 2 * GATE_HALF.z (4) = 10; the speaker block starts at
    // or above it. Stated as a literal oracle, not imported.
    const GATE_LAYER_TOP_Z_FACE: i32 = 10;
    let hz = speaker_half().z;
    for n in [1usize, 2, 8, 32, 79, 128] {
        for k in 0..n {
            let p = speaker_inner_position(k, n);
            assert!(
                p.x >= 0 && p.y >= 0 && p.z >= 0,
                "n={n} speaker {k} at ({}, {}, {}) has a negative coordinate -- \
                 negative inner-grid coordinates delete bricks in-game",
                p.x, p.y, p.z
            );
            assert!(
                p.z - hz >= GATE_LAYER_TOP_Z_FACE,
                "n={n} speaker {k} low z-face {} does not clear the gate layer \
                 (which tops out at {GATE_LAYER_TOP_Z_FACE})",
                p.z - hz
            );
        }
    }
}

/// Wire integrity for an in-chip BANK render: every wire must still resolve to
/// a brick that carries the referenced component. The speaker's volume wires
/// are same-grid internal wires now rather than cross-grid remote ones, so this
/// exercises exactly the resolution path the move changed.
#[test]
fn an_in_chip_bank_render_passes_wire_integrity() {
    let opts = AudioOptions { speakers_in_chip: true, ..AudioOptions::default() };
    let track = tone_track(1.0, &opts);
    let world = build_speaker_world(&track, &opts).expect("build");
    let path = std::env::temp_dir()
        .join(format!("h2b_audio_wi_bank_inchip_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    wire_integrity::assert_wires_valid(&path);
    let _ = std::fs::remove_file(&path);
}

/// The voice path shares the emitter/scaffold code with the bank, so the option
/// must reach it too: every speaker on the inner grid, its pitch still wired
/// straight from an `ArrayVar_Get` (not through the volume multiply), and the
/// whole graph passing wire integrity as a same-grid render.
#[test]
fn an_in_chip_voice_render_moves_speakers_keeps_the_pitch_wire_and_passes_wire_integrity() {
    let o = AudioOptions { speakers_in_chip: true, ..voice_opts(4) };
    let streams = voice_streams(1.0, &o);
    let world = build_voice_world(&streams, &o).expect("build");

    assert_eq!(speaker_ids(&world).len(), 0, "voice ON: nothing on the main grid");
    let inner = inner_speaker_ids(&world);
    assert_eq!(inner.len(), 4, "voice ON: every voice speaker on the inner grid");

    let comp = component_of(&world);
    let sources = sources_by_target(&world);
    for (v, &speaker) in inner.iter().enumerate() {
        let pitch = sources
            .get(&(speaker, "PitchMultiplier".to_string()))
            .unwrap_or_else(|| panic!("voice {v}: in-chip PitchMultiplier is unwired"));
        assert_eq!(pitch.len(), 1, "voice {v}: one pitch source");
        assert_eq!(
            comp.get(&pitch[0].0).map(String::as_str),
            Some(ARRAY_GET),
            "voice {v}: pitch must still come straight from an ArrayVar_Get"
        );
    }

    let path = std::env::temp_dir()
        .join(format!("h2b_audio_wi_voice_inchip_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    wire_integrity::assert_wires_valid(&path);
    let _ = std::fs::remove_file(&path);
}

// --- Control buttons --------------------------------------------------------

/// **The pre-wired control buttons on an audio render, verified end to end.**
///
/// The audio twin of `anim_world.rs`'s control-button test: with the toggle ON
/// a speaker world carries three animated-button bricks on the MAIN grid
/// (beside the speaker cluster), each also carrying a `Component_TextDisplay`
/// label, each button's `bHeld` resolves to a distinct clock control pin, and
/// the written `.brz`'s wires all resolve. With it OFF the render is exactly
/// what it was before the feature.
///
/// Unverifiable in game and left to the owner: that a press actually
/// pauses/restarts/resumes, that `PromptCustomLabel` shows on look, and that the
/// label renders on the same brick as the animated button.
#[test]
fn the_control_buttons_appear_on_an_audio_render_and_vanish_when_disabled() {
    const BUTTON: &str = "Component_Internal_AnimatedButton";
    const TEXT_DISPLAY: &str = "Component_TextDisplay";

    let opts = AudioOptions::default();
    let track = tone_track(1.0, &opts);
    let on = build_speaker_world(&track, &opts).expect("build");
    let off = build_speaker_world(
        &track,
        &AudioOptions { control_buttons: false, ..AudioOptions::default() },
    )
    .expect("build");

    let count = |world: &brdb::World, needle: &str| {
        world
            .bricks
            .iter()
            .filter(|b| {
                b.components
                    .iter()
                    .any(|c| c.component_type().is_some_and(|t| t.as_ref() == needle))
            })
            .count()
    };

    assert_eq!(count(&on, BUTTON), 3, "three physical buttons");
    assert_eq!(count(&on, TEXT_DISPLAY), 3, "three labels");
    assert_eq!(count(&off, BUTTON), 0);
    assert_eq!(count(&off, TEXT_DISPLAY), 0);
    assert_eq!(on.bricks.len() - off.bricks.len(), 3, "3 buttons, each carrying a label");
    assert_eq!(on.wires.len() - off.wires.len(), 3, "1 wire per control");

    // Each button's bHeld drives a control pin the timer reads on
    // Pause/Restart/Resume.
    let mut button_pins: Vec<usize> = on
        .wires
        .iter()
        .filter(|w| {
            w.source.component_type.as_ref() == BUTTON
                && w.source.port_name.as_ref() == "bHeld"
                && w.target.component_type.as_ref() == MICROCHIP_INPUT
        })
        .map(|w| {
            assert_eq!(w.target.port_name.as_ref(), "RER_Input");
            w.target.brick_id
        })
        .collect();
    let mut timer_control_pins: Vec<usize> = on
        .wires
        .iter()
        .filter(|w| {
            w.target.component_type.as_ref() == TIMER
                && matches!(w.target.port_name.as_ref(), "Pause" | "Restart" | "Resume")
                && w.source.component_type.as_ref() == MICROCHIP_INPUT
        })
        .map(|w| w.source.brick_id)
        .collect();
    button_pins.sort_unstable();
    timer_control_pins.sort_unstable();
    assert_eq!(button_pins.len(), 3, "one button per control pin");
    assert_eq!(
        button_pins, timer_control_pins,
        "each button must resolve to a Pause/Restart/Resume pin the timer reads"
    );

    for (world, tag) in [(&on, "on"), (&off, "off")] {
        let path = std::env::temp_dir()
            .join(format!("h2b_audio_buttons_{tag}_{}.brz", std::process::id()));
        std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
        let result = std::panic::catch_unwind(|| wire_integrity::assert_wires_valid(&path));
        let _ = std::fs::remove_file(&path);
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }
}
