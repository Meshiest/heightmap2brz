//! In-game probes for the loop / brick-ref features. Writes one prefab per
//! question into the directory given (default `./probes`); load each in game
//! and note what happens. What to look for is printed per file.
//!
//! cargo run --release --example gate_probes -- <out dir>

use brdb::{
    AsBrdbValue, Brick, BrickType, IntVector, Position, Vector3f, World, assets::LiteralComponent,
    schema::BrdbValue,
};
use heightmap::anim::bricks::{AnimOptions, Driver};
use heightmap::anim::chip;
use heightmap::anim::ref_player::RecordOptions;
use heightmap::anim::{AnimEncoding, AnimMode};
use heightmap::audio::percussion::{KICK_1, add_oneshot_emitter};
use heightmap::graph::{
    ARRAY_GET, ARRAY_VAR, BRICK_PROPERTY_COLOR, EmitPlacement, End, ExtPort, Graph, In, Lit, ROOT,
    SET_BRICK_PROPERTY, SET_COMPONENT_PROPERTY, VAR,
};
use heightmap::progress::NoProgress;
use heightmap::video::Clip;
use heightmap::wire_array::WireArray;
use image::{Rgba, RgbaImage};
use std::path::{Path, PathBuf};

/// Writes one probe, unless `PROBE_ONLY` is set to a name it does not contain
/// (the game locks a prefab it has open, so a rerun can skip those).
fn write(dir: &Path, name: &str, world: &World, look_for: &str) {
    if std::env::var("PROBE_ONLY").is_ok_and(|only| !name.contains(&only)) {
        return;
    }
    let path = dir.join(name);
    if let Err(e) = std::fs::write(&path, world.to_brz_vec().expect("encode")) {
        println!(
            "SKIPPED {} ({e}); close it in game and rerun with PROBE_ONLY\n",
            path.display()
        );
        return;
    }
    println!("{}\n    {look_for}\n", path.display());
}

fn hsv(h: f32) -> [u8; 3] {
    let h = h.rem_euclid(1.0) * 6.0;
    let x = (1.0 - (h % 2.0 - 1.0).abs()) * 255.0;
    let (r, g, b) = match h as u32 {
        0 => (255.0, x, 0.0),
        1 => (x, 255.0, 0.0),
        2 => (0.0, 255.0, x),
        3 => (0.0, x, 255.0),
        4 => (x, 0.0, 255.0),
        _ => (255.0, 0.0, x),
    };
    [r as u8, g as u8, b as u8]
}

/// Every pixel changes every frame: a rainbow sweeping across the screen.
fn sweep_clip(w: u32, h: u32, frames: usize) -> Clip {
    let frames = (0..frames)
        .map(|f| {
            RgbaImage::from_fn(w, h, |x, y| {
                let [r, g, b] = hsv((x + y) as f32 / (w + h) as f32 + f as f32 / 30.0);
                Rgba([r, g, b, 255])
            })
        })
        .collect();
    Clip {
        width: w,
        height: h,
        fps: 30.0,
        frames,
    }
}

/// A square bouncing over a still gradient: mostly small deltas.
fn sprite_clip(w: u32, h: u32, frames: usize) -> Clip {
    let frames = (0..frames)
        .map(|f| {
            let bounce = |n: i64, span: i64| {
                let p = n.rem_euclid(2 * span);
                if p < span { p } else { 2 * span - p }
            };
            let sx = bounce(f as i64 * 2, w as i64 - 12);
            let sy = bounce(f as i64, h as i64 - 12);
            RgbaImage::from_fn(w, h, |x, y| {
                let (x, y) = (x as i64, y as i64);
                if (sx..sx + 12).contains(&x) && (sy..sy + 12).contains(&y) {
                    Rgba([255, 230, 40, 255])
                } else {
                    Rgba([(x * 2) as u8, (y * 3) as u8, 90, 255])
                }
            })
        })
        .collect();
    Clip {
        width: w,
        height: h,
        fps: 30.0,
        frames,
    }
}

fn render(clip: &Clip, encoding: AnimEncoding) -> World {
    let opts = AnimOptions {
        driver: Driver::Refs,
        records: RecordOptions::default(),
        ..AnimOptions::default()
    };
    render_with(clip, encoding, &opts)
}

fn render_with(clip: &Clip, encoding: AnimEncoding, opts: &AnimOptions) -> World {
    AnimMode::Brick(encoding)
        .build(clip, opts, &mut NoProgress)
        .expect("render")
}

/// Every frame changes exactly half the pixels (the most a delta carries
/// before it becomes a keyframe), alternating halves, at `fps`. Far above
/// the wire graph's tick rate, each tick jumps many frames ahead, and the
/// player replays a keyframe plus every delta up to the target in one tick.
fn half_change_clip(w: u32, h: u32, frames: usize, fps: f32) -> Clip {
    let frames = (0..frames)
        .map(|f| {
            RgbaImage::from_fn(w, h, |x, y| {
                let i = (y * w + x) as usize;
                let step = (f + i % 2) / 2;
                let [r, g, b] = hsv(step as f32 * 0.07 + i as f32 / (w * h) as f32);
                Rgba([r, g, b, 255])
            })
        })
        .collect();
    Clip {
        width: w,
        height: h,
        fps,
        frames,
    }
}

/// A plate on the main grid, returning its index in `world.bricks`.
fn plate(world: &mut World, x: i32, rgb: (u8, u8, u8)) -> usize {
    world.add_brick(Brick {
        asset: (brdb::assets::bricks::PB_DEFAULT_BRICK, (10u16, 10u16, 2u16)).into(),
        position: Position { x, y: 60, z: 2 },
        color: rgb.into(),
        ..Default::default()
    });
    world.bricks.len() - 1
}

fn label(world: &mut World, x: i32, text: &str) {
    heightmap::text::add_text_block(
        world,
        text.to_string(),
        Position { x, y: 85, z: 2 },
        6.0,
        0.0,
        Vector3f {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        false,
        &heightmap::text::TextOptions::default(),
    );
}

/// A host chip on the main grid for a probe graph, plus a `brick[]` array of
/// refs to `targets`, returning the chip and the array's `ArrayVarRef`.
fn host(world: &mut World, targets: &[usize]) -> (chip::Chip, End) {
    let chip = chip::new_chip(
        world,
        Position { x: -40, y: 0, z: 2 },
        IntVector { x: 5, y: 5, z: 5 },
    );
    let at = Position {
        x: -40,
        y: 20,
        z: 2,
    };
    let refs = heightmap::graph::main_grid_brick_refs(world, targets);
    let (array, id) = Brick {
        asset: BrickType::from("B_1x1_Gate_Variable_Array"),
        position: at,
        ..Default::default()
    }
    .with_component(LiteralComponent::new(ARRAY_VAR).with_data([(
        "Value",
        Box::new(WireArray::Bricks(refs)) as Box<dyn AsBrdbValue>,
    )]))
    .with_id_split();
    world.add_brick(array);
    (chip, End::Ext(ExtPort::new(id, ARRAY_VAR, "ArrayVarRef")))
}

fn finish(mut world: World, mut chip: chip::Chip, g: &Graph) -> World {
    g.emit(
        &mut world,
        &mut chip,
        &EmitPlacement {
            root_origin: Position { x: 5, y: 5, z: 8 },
            root_columns: 8,
        },
    )
    .expect("emit");
    chip::finish(&mut world, chip).expect("finish");
    world.register_used_components();
    world
}

/// `pixels[i].Color = color` off `exec`; returns the set gate.
fn paint(g: &mut Graph, exec: End, pixels: &End, i: i64, color: In) -> usize {
    let get = g.exec(
        ROOT,
        exec,
        ARRAY_GET,
        vec![("ArrayVarRef", pixels.clone().into()), ("Index", i.into())],
    );
    g.exec(
        ROOT,
        End::Node(get, "ExecOut"),
        SET_BRICK_PROPERTY,
        vec![
            ("Brick", End::Node(get, "Value").into()),
            ("Property", Lit::Int(BRICK_PROPERTY_COLOR).into()),
            ("Value", color),
        ],
    )
}

/// Arrays longer than a u16 index: green = the element came back, red = a
/// wrong value, white = the chain never ran.
fn array_probe() -> World {
    let mut world = World::new();
    let a = plate(&mut world, 0, (255, 255, 255));
    let b = plate(&mut world, 25, (255, 255, 255));
    label(&mut world, 0, "70k");
    label(&mut world, 25, "200k");
    let (chip, pixels) = host(&mut world, &[a, b]);
    let mut g = Graph::new();
    for (k, len) in [70_000i64, 200_000].into_iter().enumerate() {
        let arr = g.gate(
            ROOT,
            ARRAY_VAR,
            vec![(
                "Value",
                Lit::Array(WireArray::I64((0..len).collect())).into(),
            )],
        );
        let load = g.gate(ROOT, "BrickComponentType_Internal_ReadBrickGrid", vec![]);
        let get = g.exec(
            ROOT,
            End::Node(load, "BrickGrid"),
            ARRAY_GET,
            vec![
                ("ArrayVarRef", End::Node(arr, "ArrayVarRef").into()),
                ("Index", (len - 1).into()),
            ],
        );
        let ok = g.gate(
            ROOT,
            "BrickComponentType_Expr_CompareGreater",
            vec![
                ("InputA", End::Node(get, "Value").into()),
                ("InputB", (len - 2).into()),
            ],
        );
        let branch = g.exec(
            ROOT,
            End::Node(get, "ExecOut"),
            "BrickComponentType_Exec_Branch",
            vec![("bCond", End::Node(ok, "bOutput").into())],
        );
        paint(
            &mut g,
            End::Node(branch, "ExecOutA"),
            &pixels,
            k as i64,
            Lit::Color(0.0, 1.0, 0.0, 1.0).into(),
        );
        paint(
            &mut g,
            End::Node(branch, "ExecOutB"),
            &pixels,
            k as i64,
            Lit::Color(1.0, 0.0, 0.0, 1.0).into(),
        );
    }
    finish(world, chip, &g)
}

/// Which colour space `SetBrickProperty` and `MakeColorHex` use. A is baked
/// sRGB 128 grey; B-D are painted on load. The one(s) matching A answer it.
fn gamma_probe() -> World {
    let mut world = World::new();
    let a = plate(&mut world, 0, (128, 128, 128));
    let b = plate(&mut world, 25, (255, 255, 255));
    let c = plate(&mut world, 50, (255, 255, 255));
    let d = plate(&mut world, 75, (255, 255, 255));
    for (x, t) in [
        (0, "A baked 128"),
        (25, "B linear .216"),
        (50, "C hex 808080"),
        (75, "D linear .502"),
    ] {
        label(&mut world, x, t);
    }
    let _ = a;
    let (chip, pixels) = host(&mut world, &[b, c, d]);
    let mut g = Graph::new();
    let load = g.gate(ROOT, "BrickComponentType_Internal_ReadBrickGrid", vec![]);
    let s1 = paint(
        &mut g,
        End::Node(load, "BrickGrid"),
        &pixels,
        0,
        Lit::Color(0.2158, 0.2158, 0.2158, 1.0).into(),
    );
    let hex = g.gate(
        ROOT,
        "BrickComponentType_Expr_MakeColorHex",
        vec![("Hex", Lit::Str("808080".into()).into())],
    );
    let s2 = paint(
        &mut g,
        End::Node(s1, "ExecOut"),
        &pixels,
        1,
        End::Node(hex, "Output").into(),
    );
    paint(
        &mut g,
        End::Node(s2, "ExecOut"),
        &pixels,
        2,
        Lit::Color(0.502, 0.502, 0.502, 1.0).into(),
    );
    finish(world, chip, &g)
}

/// `SetComponentProperty` through component refs, once a second: the text
/// flips TICK/TOCK, the sine's pitch flips low/high, and the knock should
/// sound once per second.
fn component_probe() -> World {
    let mut world = World::new();
    let text_id = heightmap::text::add_text_block(
        &mut world,
        "not set yet".into(),
        Position { x: 0, y: 60, z: 6 },
        6.0,
        0.0,
        Vector3f {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        true,
        &heightmap::text::TextOptions::default(),
    );
    let text_at = world
        .bricks
        .iter()
        .position(|b| b.id == Some(text_id))
        .expect("text brick");
    let (asset_index, _) = world
        .global_data
        .external_asset_references
        .insert_full(("BrickAudioDescriptor".into(), "BA_Synth_Basic_Sine".into()));
    world.add_brick(
        Brick {
            asset: BrickType::from("B_1x1F_Speaker"),
            position: Position { x: 30, y: 60, z: 2 },
            ..Default::default()
        }
        .with_component(LiteralComponent::new("Component_AudioEmitter").with_data([
            (
                "AudioDescriptor",
                Box::new(BrdbValue::Asset(Some(asset_index))) as Box<dyn AsBrdbValue>,
            ),
            ("PitchMultiplier", Box::new(1.0f32)),
            ("VolumeMultiplier", Box::new(0.5f32)),
            ("bEnabled", Box::new(true)),
            ("bSpatialization", Box::new(false)),
            ("InnerRadius", Box::new(200.0f32)),
            ("MaxDistance", Box::new(4000.0f32)),
        ])),
    );
    let emitter_at = world.bricks.len() - 1;
    add_oneshot_emitter(&mut world, &KICK_1, Position { x: 60, y: 60, z: 2 });
    let oneshot_at = world.bricks.len() - 1;

    let chip = chip::new_chip(
        &mut world,
        Position { x: -40, y: 0, z: 2 },
        IntVector { x: 5, y: 5, z: 5 },
    );
    let at = Position {
        x: -40,
        y: 20,
        z: 2,
    };
    let (array, array_id) = Brick {
        asset: BrickType::from("B_1x1_Gate_Variable_Array"),
        position: at,
        ..Default::default()
    }
    .with_component(LiteralComponent::new(ARRAY_VAR))
    .with_id_split();
    world.add_brick(array);
    let array_at = world.bricks.len() - 1;
    let comps = End::Ext(ExtPort::new(array_id, ARRAY_VAR, "ArrayVarRef"));

    let mut g = Graph::new();
    let clock = g.gate(
        ROOT,
        "BrickComponentType_Clock",
        vec![
            ("IntervalSeconds", Lit::Num(1.0).into()),
            ("bEnabled", Lit::Bool(true).into()),
        ],
    );
    let t = g.gate(ROOT, VAR, vec![("Value", Lit::Int(0).into())]);
    let inc = g.exec(
        ROOT,
        End::Node(clock, "Pulse"),
        "BrickComponentType_Exec_Var_Increment",
        vec![
            ("VarRef", End::Node(t, "VarRef").into()),
            ("Value", 1.into()),
        ],
    );
    let parity = g.gate(
        ROOT,
        "BrickComponentType_Expr_MathModuloFloored",
        vec![
            ("InputA", End::Node(t, "Value").into()),
            ("InputB", 2.into()),
        ],
    );
    let odd = g.gate(
        ROOT,
        "BrickComponentType_Expr_CompareGreater",
        vec![
            ("InputA", End::Node(parity, "Output").into()),
            ("InputB", 0.into()),
        ],
    );
    let word = g.gate(
        ROOT,
        "BrickComponentType_Expr_Select",
        vec![
            ("InputA", Lit::Str("TICK".into()).into()),
            ("InputB", Lit::Str("TOCK".into()).into()),
            ("bSelectB", End::Node(odd, "bOutput").into()),
        ],
    );
    let pitch = g.gate(
        ROOT,
        "BrickComponentType_Expr_Select",
        vec![
            ("InputA", Lit::Num(0.5).into()),
            ("InputB", Lit::Num(2.0).into()),
            ("bSelectB", End::Node(odd, "bOutput").into()),
        ],
    );
    let mut exec = End::Node(inc, "ExecOut");
    for (i, (class, prop, value)) in [
        ("Component_TextDisplay", "Text", End::Node(word, "Output")),
        (
            "Component_AudioEmitter",
            "PitchMultiplier",
            End::Node(pitch, "Output"),
        ),
        (
            "Component_OneShotAudioEmitter",
            "Play",
            End::Node(t, "Value"),
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let get = g.exec(
            ROOT,
            exec,
            ARRAY_GET,
            vec![
                ("ArrayVarRef", comps.clone().into()),
                ("Index", (i as i64).into()),
            ],
        );
        let set = g.exec(
            ROOT,
            End::Node(get, "ExecOut"),
            SET_COMPONENT_PROPERTY,
            vec![
                ("Component", End::Node(get, "Value").into()),
                ("ComponentType", Lit::ComponentClass(class).into()),
                ("PropertyName", Lit::Str(prop.into()).into()),
                ("Value", value.into()),
            ],
        );
        exec = End::Node(set, "ExecOut");
    }
    let mut world = finish(world, chip, &g);
    // The refs need the registered component type table; fill them in now.
    let refs = heightmap::graph::main_grid_component_refs(
        &world,
        &[
            (text_at, "Component_TextDisplay"),
            (emitter_at, "Component_AudioEmitter"),
            (oneshot_at, "Component_OneShotAudioEmitter"),
        ],
    )
    .expect("component refs");
    world.bricks[array_at].components =
        vec![Box::new(LiteralComponent::new(ARRAY_VAR).with_data([(
            "Value",
            Box::new(WireArray::Components(refs)) as Box<dyn AsBrdbValue>,
        )]))];
    world
}

/// Which form the game reads a brick ref in an array as, when the target sits
/// in a different chunk from the array. Five white plates in chunk (1,0,0);
/// arrays A-D in chunk (0,1,0) each hold one ref to their plate, written a
/// different way; E is a same-chunk control. A plate that turns green on load
/// is a form that resolves.
fn ref_forms_probe() -> World {
    use brdb::ChunkIndex;
    use brdb::schema::SavedBrickRef;
    let mut world = World::new();
    let names = [
        "A to-from",
        "B from-to",
        "C abs grid0",
        "D abs grid1",
        "E same chunk",
    ];
    for k in 0..5 {
        world.add_brick(Brick {
            asset: (brdb::assets::bricks::PB_DEFAULT_BRICK, (10u16, 10u16, 2u16)).into(),
            position: Position {
                x: 2100 + k * 25,
                y: 60,
                z: 2,
            },
            color: (255, 255, 255).into(),
            ..Default::default()
        });
    }
    for (k, name) in names.iter().enumerate() {
        label(&mut world, 2100 + k as i32 * 25, name);
    }
    let r = |grid: u32, x: i16, y: i16, i: u32| SavedBrickRef {
        grid_persistent_index: grid,
        chunk_index: ChunkIndex { x, y, z: 0 },
        brick_index_in_chunk: i,
    };
    // Plate k is brick k of chunk (1,0,0). Arrays A-D sit in chunk (0,1,0).
    let forms = [
        r(0, 1, -1, 0),
        r(0, -1, 1, 1),
        r(0, 1, 0, 2),
        r(1, 1, 0, 3),
        r(0, 0, 0, 4),
    ];
    let at = |k: usize| {
        if k < 4 {
            Position {
                x: 100 + k as i32 * 20,
                y: 2100,
                z: 2,
            }
        } else {
            Position {
                x: 2300,
                y: 20,
                z: 2,
            }
        }
    };
    let mut arrays = Vec::new();
    for (k, form) in forms.into_iter().enumerate() {
        let (b, id) = Brick {
            asset: BrickType::from("B_1x1_Gate_Variable_Array"),
            position: at(k),
            ..Default::default()
        }
        .with_component(LiteralComponent::new(ARRAY_VAR).with_data([(
            "Value",
            Box::new(WireArray::Bricks(vec![form])) as Box<dyn AsBrdbValue>,
        )]))
        .with_id_split();
        world.add_brick(b);
        arrays.push(End::Ext(ExtPort::new(id, ARRAY_VAR, "ArrayVarRef")));
    }
    let chip = chip::new_chip(
        &mut world,
        Position { x: -40, y: 0, z: 2 },
        IntVector { x: 5, y: 5, z: 5 },
    );
    let mut g = Graph::new();
    let load = g.gate(ROOT, "BrickComponentType_Internal_ReadBrickGrid", vec![]);
    for array in &arrays {
        paint(
            &mut g,
            End::Node(load, "BrickGrid"),
            array,
            0,
            Lit::Color(0.0, 1.0, 0.0, 1.0).into(),
        );
    }
    finish(world, chip, &g)
}

/// A fast arpeggio (16th notes at 150 bpm, 0.1 s each) over a kick on every
/// beat, as a single-track MIDI file.
fn arpeggio_smf() -> Vec<u8> {
    use midly::{
        Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind,
        num::{u4, u7, u15, u28},
    };
    let ev = |delta: u32, kind: TrackEventKind<'static>| TrackEvent {
        delta: u28::new(delta),
        kind,
    };
    let note = |on: bool, ch: u8, key: u8| TrackEventKind::Midi {
        channel: u4::new(ch),
        message: if on {
            MidiMessage::NoteOn {
                key: u7::new(key),
                vel: u7::new(100),
            }
        } else {
            MidiMessage::NoteOff {
                key: u7::new(key),
                vel: u7::new(0),
            }
        },
    };
    let mut track = vec![ev(
        0,
        TrackEventKind::Meta(MetaMessage::Tempo(midly::num::u24::new(400_000))),
    )];
    for bar in 0..8 {
        for (i, key) in [60u8, 64, 67, 72, 76, 72, 67, 64].iter().enumerate() {
            let pitch = key + if bar % 2 == 1 { 5 } else { 0 };
            if i % 4 == 0 {
                track.push(ev(0, note(true, 9, 36)));
            }
            track.push(ev(0, note(true, 0, pitch)));
            track.push(ev(120, note(false, 0, pitch)));
            if i % 4 == 0 {
                track.push(ev(0, note(false, 9, 36)));
            }
        }
    }
    track.push(ev(0, TrackEventKind::Meta(MetaMessage::EndOfTrack)));
    let smf = Smf {
        header: Header::new(Format::SingleTrack, Timing::Metrical(u15::new(480))),
        tracks: vec![track],
    };
    let mut bytes = Vec::new();
    smf.write(&mut bytes).expect("smf");
    bytes
}

fn midi(functions: bool) -> World {
    let opts = heightmap::midi::MidiOptions {
        functions,
        ..heightmap::midi::MidiOptions::default()
    };
    let score = heightmap::midi::analyze_midi(&arpeggio_smf(), &opts).expect("analyze");
    heightmap::audio::speakers::build_midi_event_world(&score, &opts).expect("midi")
}

fn text_refs_video() -> World {
    let opts = AnimOptions {
        driver: Driver::Refs,
        ..AnimOptions::default()
    };
    AnimMode::Text
        .build(&sweep_clip(96, 54, 60), &opts, &mut NoProgress)
        .expect("text render")
}

/// Two seconds rising through an octave and back, for the audio loops.
fn sweep_tone() -> heightmap::audio::source::SampleClip {
    let sr = 48_000u32;
    let n = sr as usize * 4;
    let mut phase = 0.0f32;
    let samples = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let tri = if t < 2.0 { t / 2.0 } else { (4.0 - t) / 2.0 };
            phase += std::f32::consts::TAU * 440.0 * 2f32.powf(tri) / sr as f32;
            0.5 * phase.sin()
        })
        .collect();
    heightmap::audio::source::SampleClip::new(sr, samples)
}

fn audio_refs(voice: bool) -> World {
    use heightmap::audio::track::AudioOptions;
    let opts = AudioOptions {
        driver: Driver::Refs,
        max_voices: 4,
        ..AudioOptions::default()
    };
    if voice {
        let streams =
            heightmap::audio::voices::analyze_voices(&sweep_tone(), &opts, &mut NoProgress)
                .expect("voices");
        heightmap::audio::speakers::build_voice_world(&streams, &opts).expect("voice world")
    } else {
        let track =
            heightmap::audio::track::analyze(&sweep_tone(), &opts, &mut NoProgress).expect("bank");
        heightmap::audio::speakers::build_speaker_world(&track, &opts).expect("bank world")
    }
}

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "probes".into()));
    std::fs::create_dir_all(&dir).expect("out dir");

    for side in [32u32, 72, 100, 142] {
        write(
            &dir,
            &format!("loop_throughput_{side}x{side}.brz"),
            &render(&sweep_clip(side, side, 60), AnimEncoding::Hex),
            &format!(
                "{} pixels repainted every frame at 30 fps. Smooth rainbow sweep = loops + function \
                 calls + brick refs work; note the frame rate. Black/static = the player failed.",
                side * side
            ),
        );
    }
    write(
        &dir,
        "loop_throughput_72x72_color_array.brz",
        &render(&sweep_clip(72, 72, 60), AnimEncoding::ColorArray),
        "Colour-array records spread over 5 banks (arrays of arrays). Same sweep as loop_throughput_72x72.",
    );
    write(
        &dir,
        "text_refs_video.brz",
        &text_refs_video(),
        "Text mode through component refs: the rainbow sweep drawn in text bands, all set by one loop.",
    );
    write(
        &dir,
        "audio_refs_bank.brz",
        &audio_refs(false),
        "Bank mode through component refs: a tone sliding up an octave and back. Press Resume/Restart; Pause must silence it.",
    );
    write(
        &dir,
        "audio_refs_voice.brz",
        &audio_refs(true),
        "Voice mode through component refs: the same slide, as tracked pitch. Pause must silence it.",
    );
    write(
        &dir,
        "loop_throughput_32x32_color_array.brz",
        &render(&sweep_clip(32, 32, 60), AnimEncoding::ColorArray),
        "Same as loop_throughput_32x32 through the colour-array encoding; compare its colours and frame rate.",
    );
    write(
        &dir,
        "delta_sprite_128x72.brz",
        &render(&sprite_clip(128, 72, 150), AnimEncoding::Hex),
        "A yellow square bouncing over a gradient, mostly delta frames. Look for trails or tearing; \
         press Restart mid-clip and check it recovers cleanly.",
    );
    write(
        &dir,
        "array_past_65535.brz",
        &array_probe(),
        "Green = arrays of 70k / 200k entries index fine; red = wrong value; white = the load chain never ran.",
    );
    write(
        &dir,
        "colour_space.brz",
        &gamma_probe(),
        "Which of B, C, D match A's grey? (B/D tell whether SetBrickProperty colours are linear or sRGB; C checks MakeColorHex.)",
    );
    write(
        &dir,
        "ref_forms.brz",
        &ref_forms_probe(),
        "Five plates far off along +x (about 2100 units). Which turn green on load? E is the same-chunk control and must turn green.",
    );
    write(
        &dir,
        "midi_wired.brz",
        &midi(false),
        "Reference: today's wired playheads. A fast arpeggio over a kick; press Resume/Restart to start.",
    );
    write(
        &dir,
        "midi_functions.brz",
        &midi(true),
        "The same arpeggio through function playheads. Should sound the same as midi_wired, notes as crisp; silence or stuck notes = function calls/port order wrong.",
    );
    write(
        &dir,
        "component_properties.brz",
        &component_probe(),
        "Once a second: text flips TICK/TOCK, the sine flips pitch, and a knock plays. Note which of the three happen.",
    );

    // Compression stage: dense keyframes, chunk spreading, catch-up cost.
    let dense = AnimOptions {
        driver: Driver::Refs,
        records: RecordOptions {
            dense_keyframes: true,
            ..RecordOptions::default()
        },
        ..AnimOptions::default()
    };
    write(
        &dir,
        "dense_sweep_96x96.brz",
        &render_with(&sweep_clip(96, 96, 60), AnimEncoding::Hex, &dense),
        "Dense keyframes: every frame is a keyframe of 9216 colours split over two strings. Look for \
         the same smooth rainbow sweep as loop_throughput_100x100; a band of wrong or frozen colours \
         from the middle of the screen down = the second string's pixel offset is wrong.",
    );
    write(
        &dir,
        "dense_sprite_128x72.brz",
        &render_with(&sprite_clip(128, 72, 150), AnimEncoding::Hex, &dense),
        "Dense keyframes with deltas between them: should look exactly like delta_sprite_128x72. Look \
         for trails, a scrambled gradient once every 2 seconds (a keyframe), and press Restart \
         mid-clip to check it recovers.",
    );
    let spread = AnimOptions {
        driver: Driver::Refs,
        bank_size: 4000,
        chunk_bytes: 1,
        ..AnimOptions::default()
    };
    write(
        &dir,
        "banks_across_chunks_72x72.brz",
        &render_with(&sweep_clip(72, 72, 60), AnimEncoding::ColorArray, &spread),
        "Colour-array record banks, each in its own chunk (about 2048 units apart along +x, so the \
         arrays trail off to one side). Look for the same sweep as loop_throughput_72x72_color_array; \
         black or frozen = refs to arrays in another chunk do not resolve.",
    );
    let text_spread = AnimOptions {
        driver: Driver::Refs,
        chunk_bytes: 1,
        ..AnimOptions::default()
    };
    write(
        &dir,
        "text_bands_across_chunks.brz",
        &AnimMode::Text
            .build(&sweep_clip(96, 54, 60), &text_spread, &mut NoProgress)
            .expect("text render"),
        "Text refs with every band's array in its own chunk. Look for the same picture as \
         text_refs_video; a blank or frozen band = refs to arrays in another chunk do not resolve.",
    );
    for fps in [600u32, 3000] {
        let clip = half_change_clip(96, 96, 600, fps as f32);
        let worst = worst_per_tick(&clip, fps as usize / 60);
        write(
            &dir,
            &format!("catch_up_{fps}fps.brz"),
            &render(&clip, AnimEncoding::Hex),
            &format!(
                "Throughput: at {fps} fps a 60 Hz tick jumps {} frames and replays up to {worst} \
                 records (a keyframe plus half-screen deltas; bee's worst seek is about 230k). Note \
                 the game's frame rate with it running against standing away from it; a frozen or \
                 black screen = the loop gave up. A locally hosted game ticks faster, so fewer \
                 records per tick there.",
                fps / 60
            ),
        );
    }
}

/// Most records one tick replays when each tick advances `per_tick` frames:
/// from the later of the previous target + 1 and the target's keyframe.
fn worst_per_tick(clip: &Clip, per_tick: usize) -> usize {
    use heightmap::anim::ref_player::{RecordEncoding, RecordPacker, pack_records};
    let mut p = RecordPacker::new(clip.width, clip.height, 128, RecordOptions::default());
    for f in &clip.frames {
        p.push_frame(f).expect("frame");
    }
    let (frames, visible) = p.finish();
    let packed = pack_records(
        &frames,
        &visible,
        RecordEncoding::Hex,
        32_767,
        65_535,
        false,
    )
    .expect("pack");
    (per_tick..frames.len())
        .map(|target| {
            let start = (target + 1 - per_tick).max(packed.seek_from[target] as usize);
            packed.records_per_frame[start..=target]
                .iter()
                .sum::<usize>()
        })
        .max()
        .unwrap_or(0)
}
