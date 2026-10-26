#[path = "wire_integrity.rs"]
mod wire_integrity;

use brdb::ChunkIndex;
use brdb::schema::{SavedBrickRef, SavedComponentRef};
use heightmap::anim::bricks::Driver;
use heightmap::audio::source::SampleClip;
use heightmap::audio::speakers::{build_speaker_world, build_voice_world};
use heightmap::audio::stream_refs::{StreamPorts, stream_loop_graph};
use heightmap::audio::track::{AudioOptions, analyze};
use heightmap::audio::voices::analyze_voices;
use heightmap::graph::sim::{Sim, Val};
use heightmap::graph::{End, ExtPort};
use heightmap::progress::NoProgress;
use std::f32::consts::TAU;

fn ext(name: &str) -> ExtPort {
    ExtPort::new(usize::MAX - name.len(), "Ext", name)
}

fn emitter(s: usize) -> SavedComponentRef {
    SavedComponentRef {
        brick: SavedBrickRef {
            grid_persistent_index: 1,
            chunk_index: ChunkIndex { x: 0, y: 0, z: 0 },
            brick_index_in_chunk: s as u32,
        },
        component_type_index: 3,
    }
}

#[test]
fn every_speaker_gets_its_frame_value_and_pausing_mutes() {
    let (speakers, frames) = (3usize, 8usize);
    let vol = |s: usize, f: usize| (s * 10 + f) as f64 / 100.0;
    let pitch = |s: usize, f: usize| 1.0 + (s + f) as f64 / 10.0;
    let g = stream_loop_graph(
        speakers,
        StreamPorts {
            frame_index: End::Ext(ext("Frame")),
            master: End::Ext(ext("Master")),
            volumes: End::Ext(ext("Volumes")),
            pitches: Some(End::Ext(ext("Pitches"))),
            emitters: End::Ext(ext("Emitters")),
        },
    );
    let mut sim = Sim::new(&g);
    let volumes: Vec<Val> = (0..speakers)
        .map(|s| sim.new_array((0..frames).map(|f| Val::Num(vol(s, f))).collect()))
        .collect();
    let pitches: Vec<Val> = (0..speakers)
        .map(|s| sim.new_array((0..frames).map(|f| Val::Num(pitch(s, f))).collect()))
        .collect();
    sim.bind_array(ext("Volumes"), volumes);
    sim.bind_array(ext("Pitches"), pitches);
    sim.bind_array(
        ext("Emitters"),
        (0..speakers).map(|s| Val::Component(emitter(s))).collect(),
    );
    let prop = |sim: &Sim, s: usize, p: &str| {
        sim.component_props
            .get(&(emitter(s), p.to_string()))
            .cloned()
    };

    // The load tick records the inputs without firing.
    sim.bind(ext("Frame"), Val::Int(0));
    sim.bind(ext("Master"), Val::Num(0.8));
    sim.tick();
    for f in [1, 2, 5, 3] {
        sim.bind(ext("Frame"), Val::Int(f as i64));
        sim.tick();
        for s in 0..speakers {
            assert_eq!(
                prop(&sim, s, "VolumeMultiplier"),
                Some(Val::Num(vol(s, f) * 0.8)),
                "speaker {s} frame {f}"
            );
            assert_eq!(
                prop(&sim, s, "PitchMultiplier"),
                Some(Val::Num(pitch(s, f))),
                "speaker {s} frame {f}"
            );
        }
    }
    // Pause: the frame stands still and the master drops to 0.
    sim.bind(ext("Master"), Val::Num(0.0));
    sim.tick();
    for s in 0..speakers {
        assert_eq!(
            prop(&sim, s, "VolumeMultiplier"),
            Some(Val::Num(0.0)),
            "speaker {s} muted"
        );
    }
}

#[test]
fn the_loop_is_the_same_size_for_any_speaker_count() {
    let size = |n| {
        stream_loop_graph(
            n,
            StreamPorts {
                frame_index: End::Ext(ext("Frame")),
                master: End::Ext(ext("Master")),
                volumes: End::Ext(ext("Volumes")),
                pitches: None,
                emitters: End::Ext(ext("Emitters")),
            },
        )
        .gate_count()
    };
    assert_eq!(size(2), size(200));
}

fn tone(secs: f32, opts: &AudioOptions) -> heightmap::audio::track::VoiceTrack {
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

fn write_and_check(world: &brdb::World, name: &str) {
    let path = std::env::temp_dir().join(format!("h2b_{name}_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    wire_integrity::assert_wires_valid(&path);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn bank_and_voice_refs_renders_write_and_every_wire_resolves() {
    let opts = AudioOptions {
        driver: Driver::Refs,
        ..AudioOptions::default()
    };
    let wired = build_speaker_world(&tone(1.0, &AudioOptions::wired()), &AudioOptions::wired()).unwrap();
    let refs = build_speaker_world(&tone(1.0, &opts), &opts).unwrap();
    write_and_check(&refs, "bank_refs");
    let gates = |w: &brdb::World| w.grids.iter().map(|(_, b)| b.len()).sum::<usize>();
    assert!(
        gates(&refs) < gates(&wired),
        "refs {} vs wires {}",
        gates(&refs),
        gates(&wired)
    );

    let voice = AudioOptions {
        max_voices: 4,
        window: 4096,
        ..opts
    };
    let sr = 48_000u32;
    let clip = SampleClip::new(
        sr,
        (0..sr as usize)
            .map(|i| (TAU * 440.0 * i as f32 / sr as f32).sin())
            .collect(),
    );
    let streams = analyze_voices(&clip, &voice, &mut NoProgress).expect("analyze");
    write_and_check(&build_voice_world(&streams, &voice).unwrap(), "voice_refs");
}

/// Speakers in the chip sit in the chip's own grid, which refs do not reach
/// yet, so a refs render with them is built wired: no speaker loop.
#[test]
fn refs_with_speakers_in_the_chip_falls_back_to_wires() {
    let opts = AudioOptions {
        driver: Driver::Refs,
        speakers_in_chip: true,
        ..AudioOptions::default()
    };
    assert!(!opts.uses_refs(10));
    let world = build_speaker_world(&tone(0.5, &opts), &opts).expect("builds wired");
    let loops = world
        .grids
        .iter()
        .flat_map(|(_, b)| b.iter())
        .filter(|b| b.components.iter().any(|c| c.component_type().is_some_and(|t| t.to_string() == "Component_Internal_Microchip_CountFor")))
        .count();
    assert_eq!(loops, 0, "a wired render has no loop shell");
    write_and_check(&world, "in_chip_fallback");
}
