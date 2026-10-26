#[path = "wire_integrity.rs"]
mod wire_integrity;

use brdb::schema::{BrdbValue, ReadBrdbSchema, SavedBrickRef, SavedComponentRef};
use brdb::{AsBrdbValue, BrFsReader, ChunkIndex, IntoReader};
use heightmap::anim::AnimMode;
use heightmap::anim::bricks::{AnimOptions, Driver};
use heightmap::anim::text_refs::{TextLoopPorts, text_loop_graph};
use heightmap::graph::sim::{Sim, Val};
use heightmap::graph::{End, ExtPort};
use heightmap::progress::NoProgress;
use heightmap::video::Clip;
use heightmap::wire_array::WireArray;
use image::{Rgba, RgbaImage};

fn ext(name: &str) -> ExtPort {
    ExtPort::new(usize::MAX - name.len(), "Ext", name)
}

fn display(b: usize) -> SavedComponentRef {
    SavedComponentRef {
        brick: SavedBrickRef {
            grid_persistent_index: 1,
            chunk_index: ChunkIndex { x: 0, y: 0, z: 0 },
            brick_index_in_chunk: b as u32,
        },
        component_type_index: 7,
    }
}

#[test]
fn every_band_shows_the_frames_string() {
    let (bands, frames) = (5usize, 9usize);
    let text = |b: usize, f: usize| format!("band {b} frame {f}");
    let g = text_loop_graph(
        bands,
        TextLoopPorts {
            frame_index: End::Ext(ext("Frame")),
            on_change: End::Ext(ext("OnChanged")),
            texts: End::Ext(ext("Texts")),
            displays: End::Ext(ext("Displays")),
        },
    );
    let mut sim = Sim::new(&g);
    let members: Vec<Val> = (0..bands)
        .map(|b| sim.new_array((0..frames).map(|f| Val::Str(text(b, f))).collect()))
        .collect();
    sim.bind_array(ext("Texts"), members);
    sim.bind_array(
        ext("Displays"),
        (0..bands).map(|b| Val::Component(display(b))).collect(),
    );
    for target in [1, 2, 3, 8, 0, 5, 5, 4] {
        sim.bind(ext("Frame"), Val::Int(target as i64));
        sim.fire_ext(ext("OnChanged"));
        for b in 0..bands {
            assert_eq!(
                sim.component_props.get(&(display(b), "Text".to_string())),
                Some(&Val::Str(text(b, target))),
                "band {b} at frame {target}"
            );
        }
    }
}

#[test]
fn the_loop_is_the_same_size_for_any_band_count() {
    let size = |bands| {
        text_loop_graph(
            bands,
            TextLoopPorts {
                frame_index: End::Ext(ext("Frame")),
                on_change: End::Ext(ext("OnChanged")),
                texts: End::Ext(ext("Texts")),
                displays: End::Ext(ext("Displays")),
            },
        )
        .gate_count()
    };
    assert_eq!(size(3), size(300));
}

fn clip(w: u32, h: u32, frames: usize) -> Clip {
    let frames = (0..frames)
        .map(|f| {
            RgbaImage::from_fn(w, h, |x, y| {
                Rgba([(x * 30 + f as u32 * 50) as u8, (y * 40) as u8, 90, 255])
            })
        })
        .collect();
    Clip {
        width: w,
        height: h,
        fps: 10.0,
        frames,
    }
}

#[test]
fn a_text_refs_render_writes_resolves_and_matches_its_estimate() {
    // Tall enough for several bands.
    let c = clip(400, 60, 4);
    let opts = AnimOptions {
        driver: Driver::Refs,
        ..AnimOptions::default()
    };
    let world = AnimMode::Text.build(&c, &opts, &mut NoProgress).unwrap();
    let est = AnimMode::Text.estimate(400, 60, 4, &opts).unwrap();
    assert_eq!(est.bricks, world.bricks.len(), "bricks");
    assert_eq!(est.wires, world.wires.len(), "wires");

    let path = std::env::temp_dir().join(format!("h2b_text_refs_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    wire_integrity::assert_wires_valid(&path);

    // The display refs read back as main-grid refs to TextDisplay components.
    let db = brdb::Brz::open(&path).expect("open").into_reader();
    let global = db.global_data().expect("global data");
    let text_type = global
        .component_type_names
        .get_index_of("Component_TextDisplay")
        .expect("registered") as u16;
    let mut displays = None;
    for chunk in db.brick_chunk_index(1).expect("main grid") {
        if chunk.num_components == 0 {
            continue;
        }
        let file = format!("World/0/Bricks/Grids/1/Components/{}.mps", chunk.index);
        let found = db.find_file_by_path(&file).unwrap().unwrap();
        let bytes = db.find_blob(found.blob_id).unwrap().read().unwrap();
        let schema = db.components_schema_rev(found.created_at).unwrap();
        let mut buf = &bytes[..];
        let soa = buf.read_brdb(&schema, "BRSavedComponentChunkSoA").unwrap();
        for counter in soa
            .prop("ComponentTypeCounters")
            .unwrap()
            .as_array()
            .unwrap()
        {
            let ty = counter.prop("TypeIndex").unwrap().as_brdb_u32().unwrap() as usize;
            let name = global.component_data_struct_names[ty].clone();
            for _ in 0..counter.prop("NumInstances").unwrap().as_brdb_u32().unwrap() {
                let v: BrdbValue = buf.read_brdb(&schema, &name).unwrap();
                if name == "BrickComponentData_ArrayVar"
                    && let Ok(WireArray::Components(r)) =
                        WireArray::try_from(v.prop("Value").unwrap())
                    && r.first()
                        .is_some_and(|r| r.component_type_index == text_type)
                {
                    displays = Some(r);
                }
            }
        }
    }
    let displays = displays.expect("an array of TextDisplay refs");
    assert!(displays.len() > 1, "{} bands", displays.len());
    for r in &displays {
        assert_eq!(r.brick.grid_persistent_index, 1);
        assert_eq!(r.component_type_index, text_type);
    }
    let _ = std::fs::remove_file(&path);
}

/// Band arrays past `chunk_bytes` move to further chunks; the refs to them,
/// read back from the save, still land on `ArrayVar`s. Hold on, which changes
/// only the strings.
#[test]
fn band_arrays_past_the_chunk_limit_spread_over_chunks() {
    let c = clip(400, 60, 4);
    let opts = AnimOptions {
        driver: Driver::Refs,
        chunk_bytes: 1,
        hold: 8,
        ..AnimOptions::default()
    };
    let world = AnimMode::Text.build(&c, &opts, &mut NoProgress).unwrap();
    let est = AnimMode::Text.estimate(400, 60, 4, &opts).unwrap();
    assert_eq!(est.bricks, world.bricks.len(), "bricks");
    assert_eq!(est.wires, world.wires.len(), "wires");
    let path = std::env::temp_dir().join(format!("h2b_text_spread_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    wire_integrity::assert_wires_valid(&path);

    let db = brdb::Brz::open(&path).expect("open").into_reader();
    let global = db.global_data().expect("global data");
    let array_type = global
        .component_type_names
        .get_index_of("BrickComponentType_ArrayVar")
        .expect("registered") as u16;
    let mut band_refs = None;
    for chunk in db.brick_chunk_index(1).expect("main grid") {
        if chunk.num_components == 0 {
            continue;
        }
        let file = format!("World/0/Bricks/Grids/1/Components/{}.mps", chunk.index);
        let found = db.find_file_by_path(&file).unwrap().unwrap();
        let bytes = db.find_blob(found.blob_id).unwrap().read().unwrap();
        let schema = db.components_schema_rev(found.created_at).unwrap();
        let mut buf = &bytes[..];
        let soa = buf.read_brdb(&schema, "BRSavedComponentChunkSoA").unwrap();
        for counter in soa
            .prop("ComponentTypeCounters")
            .unwrap()
            .as_array()
            .unwrap()
        {
            let ty = counter.prop("TypeIndex").unwrap().as_brdb_u32().unwrap() as usize;
            let name = global.component_data_struct_names[ty].clone();
            for _ in 0..counter.prop("NumInstances").unwrap().as_brdb_u32().unwrap() {
                let v: BrdbValue = buf.read_brdb(&schema, &name).unwrap();
                if name == "BrickComponentData_ArrayVar"
                    && let Ok(WireArray::Components(r)) =
                        WireArray::try_from(v.prop("Value").unwrap())
                    && r.first()
                        .is_some_and(|r| r.component_type_index == array_type)
                {
                    band_refs = Some(r);
                }
            }
        }
    }
    let band_refs = band_refs.expect("the array of band arrays");
    assert!(band_refs.len() > 1, "{} bands", band_refs.len());
    let chunks: std::collections::HashSet<_> =
        band_refs.iter().map(|r| r.brick.chunk_index).collect();
    assert_eq!(
        chunks.len(),
        band_refs.len(),
        "one band per chunk at a 1-byte limit"
    );
    for r in &band_refs {
        assert_eq!(r.brick.grid_persistent_index, 1);
        let soa = db.brick_chunk_soa(1, r.brick.chunk_index).expect("chunk");
        assert!((r.brick.brick_index_in_chunk as usize) < soa.relative_positions.len());
    }
    let _ = std::fs::remove_file(&path);
}
