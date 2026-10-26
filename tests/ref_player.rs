#[path = "wire_integrity.rs"]
mod wire_integrity;

use brdb::schema::{BrdbValue, ReadBrdbSchema, SavedBrickRef};
use brdb::{AsBrdbValue, BrFsReader, ChunkIndex, IntoReader};
use heightmap::anim::bricks::{AnimOptions, Driver};
use heightmap::anim::ref_player::{
    Bank, FrameRecords, PackedRecords, PlayerPorts, RecordEncoding, RecordOptions, RecordPacker,
    pack_records, player_graph,
};
use heightmap::anim::{AnimEncoding, AnimMode};
use heightmap::graph::sim::{Sim, Val, array_vals};
use heightmap::graph::{End, ExtPort};
use heightmap::progress::NoProgress;
use heightmap::video::Clip;
use heightmap::wire_array::WireArray;
use image::{Rgba, RgbaImage};

/// A clip whose every pixel changes on its own schedule, with a few culled
/// (transparent) pixels that come and go and one never-visible column.
fn busy_clip(w: u32, h: u32, n: usize) -> Clip {
    let frames = (0..n)
        .map(|f| {
            RgbaImage::from_fn(w, h, |x, y| {
                if x == w - 1 {
                    return Rgba([9, 9, 9, 0]); // never visible
                }
                let i = (y * w + x) as usize;
                let changes = (f + i) % 3 != 0 || f % 5 == 0;
                let t = if changes { f } else { f.saturating_sub(1) };
                let culled = (i + t) % 7 == 0 && f % 2 == 1;
                Rgba([
                    (t * 37 + i * 11) as u8,
                    (t * 13) as u8,
                    (i * 29) as u8,
                    if culled { 0 } else { 255 },
                ])
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

fn records(clip: &Clip, opts: RecordOptions) -> (Vec<FrameRecords>, Vec<bool>) {
    let mut p = RecordPacker::new(clip.width, clip.height, 128, opts);
    for f in &clip.frames {
        p.push_frame(f).unwrap();
    }
    p.finish()
}

fn ext(port: &str) -> ExtPort {
    ExtPort::new(usize::MAX - port.len(), "Ext", port)
}

/// What the screen should show for `frame`: per visible pixel (in `pixels`
/// order), the expected colour value from the decoder.
fn expected(clip: &Clip, visible: &[bool], frame: usize, encoding: RecordEncoding) -> Vec<Val> {
    let lut = heightmap::util::srgb_to_linear_f32_table();
    clip.frames[frame]
        .as_raw()
        .chunks_exact(4)
        .zip(visible)
        .filter(|(_, v)| **v)
        .map(|(p, _)| {
            let culled = p[3] < 128;
            match encoding {
                RecordEncoding::Hex => {
                    let c = |b: u8| if culled { 0.0 } else { b as f32 / 255.0 };
                    Val::Color([c(p[0]), c(p[1]), c(p[2]), 1.0])
                }
                RecordEncoding::ColorArray => {
                    if culled {
                        Val::Color([0.0; 4])
                    } else {
                        Val::Color([
                            lut[p[0] as usize],
                            lut[p[1] as usize],
                            lut[p[2] as usize],
                            1.0,
                        ])
                    }
                }
            }
        })
        .collect()
}

fn brick(i: usize) -> SavedBrickRef {
    SavedBrickRef {
        grid_persistent_index: 0,
        chunk_index: ChunkIndex { x: 0, y: 0, z: 0 },
        brick_index_in_chunk: i as u32,
    }
}

/// The player for `packed`, with every port external.
fn build(packed: &PackedRecords) -> heightmap::anim::ref_player::Player {
    player_graph(
        packed,
        PlayerPorts {
            frame_index: End::Ext(ext("Frame")),
            on_change: End::Ext(ext("OnChanged")),
            pixels: End::Ext(ext("Pixels")),
            a_banks: End::Ext(ext("A Banks")),
            b_banks: End::Ext(ext("B Banks")),
            status: Some(End::Ext(ext("Status"))),
        },
    )
}

/// A simulator over `player` with `pixels` and the bank arrays bound, as
/// the main-grid arrays would hold them.
fn simulate<'g>(
    player: &'g heightmap::anim::ref_player::Player,
    packed: &PackedRecords,
) -> Sim<'g> {
    let mut sim = Sim::new(&player.graph);
    sim.bind_array(
        ext("Pixels"),
        (0..packed.pixel_count)
            .map(|i| Val::Brick(brick(i)))
            .collect(),
    );
    let (a, b): (Vec<Val>, Vec<Val>) = packed
        .banks
        .iter()
        .map(|bank| {
            let (a, b) = bank.arrays();
            (sim.new_array(array_vals(&a)), sim.new_array(array_vals(&b)))
        })
        .unzip();
    sim.bind_array(ext("A Banks"), a);
    sim.bind_array(ext("B Banks"), b);
    sim
}

/// Play `schedule` (target frame per tick) through the player and count
/// display bricks that differ from the source frame after each tick.
fn mismatches(
    clip: &Clip,
    opts: RecordOptions,
    encoding: RecordEncoding,
    schedule: &[usize],
    fire_first: bool,
    bank_size: usize,
) -> usize {
    let (frames, visible) = records(clip, opts);
    let packed = pack_records(
        &frames,
        &visible,
        encoding,
        64,
        bank_size,
        opts.dense_keyframes,
    )
    .unwrap();
    let player = build(&packed);
    let mut sim = simulate(&player, &packed);
    sim.fire_on_first_tick = fire_first;
    // Display bricks are baked with frame 0's colours.
    let start = expected(clip, &visible, 0, encoding);
    let mut bad = 0;
    let mut current = if fire_first { None } else { Some(0usize) };
    for &target in schedule {
        sim.bind(ext("Frame"), Val::Int(target as i64));
        if current != Some(target) {
            sim.fire_ext(ext("OnChanged"));
            current = Some(target);
        }
        let want = expected(clip, &visible, target, encoding);
        for (i, w) in want.iter().enumerate() {
            let shown = sim.brick_props.get(&(brick(i), 1)).unwrap_or(&start[i]);
            if shown != w {
                bad += 1;
            }
        }
    }
    bad
}

fn schedules(n: usize) -> Vec<(&'static str, Vec<usize>)> {
    let mut seed = 0x2545_f491u64;
    let random = (0..200)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        })
        .collect();
    vec![
        ("sequential", (0..n).collect()),
        ("every other frame", (0..n).step_by(2).collect()),
        ("wrap", (0..n).chain(0..n).chain(0..n / 2).collect()),
        ("random seek", random),
    ]
}

#[test]
fn every_schedule_shows_the_source_frame_exactly() {
    let clip = busy_clip(6, 4, 23);
    for encoding in [RecordEncoding::Hex, RecordEncoding::ColorArray] {
        for interval in [1, 4, 60] {
            // Dense keyframes span two 64-character strings here (10 records
            // each), so the per-string pixel offset is exercised.
            for dense_keyframes in [false, true] {
                let opts = RecordOptions {
                    keyframe_interval: interval,
                    delta_threshold: 0,
                    dense_keyframes,
                };
                for (name, schedule) in schedules(clip.frames.len()) {
                    for fire_first in [false, true] {
                        // 65,535: one bank. 30: records spill across many banks.
                        for bank_size in [65_535, 30] {
                            let bad =
                                mismatches(&clip, opts, encoding, &schedule, fire_first, bank_size);
                            assert_eq!(
                                bad, 0,
                                "{encoding:?} keyframe every {interval}, dense {dense_keyframes}, {name}, fire_first {fire_first}, bank {bank_size}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Dense keyframes drop the index: a keyframe string is `RRGGBB` per pixel,
/// its record count is negated, and the player stays a fixed size.
#[test]
fn dense_keyframes_store_colours_only() {
    let clip = busy_clip(6, 4, 23);
    let opts = RecordOptions {
        dense_keyframes: true,
        ..RecordOptions::default()
    };
    let (frames, visible) = records(&clip, opts);
    let packed = pack_records(&frames, &visible, RecordEncoding::Hex, 64, 65_535, true).unwrap();
    let plain = pack_records(&frames, &visible, RecordEncoding::Hex, 64, 65_535, false).unwrap();
    let mut dense_strings = 0;
    for bank in &packed.banks {
        let Bank::Hex { chunks, records } = bank else {
            panic!("hex banks")
        };
        for (s, n) in chunks.iter().zip(records) {
            if *n < 0 {
                dense_strings += 1;
                assert_eq!(s.len(), 6 * (-n) as usize);
            } else {
                assert_eq!(s.len(), (packed.index_width + 6) * *n as usize);
            }
        }
    }
    let keyframes = frames.iter().filter(|f| f.key).count();
    assert_eq!(dense_strings, keyframes * 2, "20 pixels, 10 per string");
    let chars = |p: &PackedRecords| -> usize {
        p.banks
            .iter()
            .map(|b| match b {
                Bank::Hex { chunks, .. } => chunks.iter().map(String::len).sum::<usize>(),
                _ => 0,
            })
            .sum()
    };
    assert!(chars(&packed) < chars(&plain));
    let size = |w, h| {
        let clip = busy_clip(w, h, 6);
        let (frames, visible) = records(&clip, opts);
        let packed =
            pack_records(&frames, &visible, RecordEncoding::Hex, 10_000, 65_535, true).unwrap();
        build(&packed).graph.gate_count()
    };
    let small = size(4, 3);
    assert_eq!(small, size(64, 36));
    // 27 of them are the cut-short reports and the step that clears them.
    assert!(small <= 98, "{small} gates");
}

#[test]
fn a_lossy_delta_never_drifts_past_its_threshold() {
    let clip = busy_clip(6, 4, 23);
    let threshold = 40u8;
    let opts = RecordOptions {
        keyframe_interval: 60,
        delta_threshold: threshold,
        dense_keyframes: false,
    };
    let (frames, visible) = records(&clip, opts);
    let packed = pack_records(&frames, &visible, RecordEncoding::Hex, 64, 65_535, false).unwrap();
    let player = build(&packed);
    let mut sim = simulate(&player, &packed);
    let start = expected(&clip, &visible, 0, RecordEncoding::Hex);
    for target in 1..clip.frames.len() {
        sim.bind(ext("Frame"), Val::Int(target as i64));
        sim.fire_ext(ext("OnChanged"));
        for (i, want) in expected(&clip, &visible, target, RecordEncoding::Hex)
            .iter()
            .enumerate()
        {
            let (Val::Color(a), Val::Color(b)) = (
                sim.brick_props.get(&(brick(i), 1)).unwrap_or(&start[i]),
                want,
            ) else {
                panic!("colours");
            };
            for c in 0..3 {
                let off = ((a[c] - b[c]).abs() * 255.0).round() as u8;
                assert!(
                    off <= threshold,
                    "frame {target} pixel {i} channel {c} is {off} off"
                );
            }
        }
    }
    assert!(
        packed.total_records() < frames.iter().map(|f| f.records.len()).sum::<usize>() + 1,
        "a threshold never adds records"
    );
}

#[test]
fn keyframes_follow_the_interval_and_large_deltas() {
    let clip = busy_clip(6, 4, 23);
    let (frames, _) = records(
        &clip,
        RecordOptions {
            keyframe_interval: 5,
            delta_threshold: 0,
            dense_keyframes: false,
        },
    );
    for (f, frame) in frames.iter().enumerate() {
        let n = frame.records.len();
        if frame.key {
            assert_eq!(n, 24, "a keyframe lists every pixel");
        } else {
            assert!(
                n * 2 <= 24,
                "frame {f}: a delta over half the screen becomes a keyframe"
            );
        }
    }
    assert!(frames[0].key);
    let gaps: Vec<usize> = frames
        .iter()
        .enumerate()
        .filter(|(_, f)| f.key)
        .map(|(i, _)| i)
        .collect();
    assert!(
        gaps.windows(2).all(|w| w[1] - w[0] <= 5),
        "keyframes at most 5 apart: {gaps:?}"
    );
}

#[test]
fn hex_chunks_hold_whole_records_under_the_limit() {
    let clip = busy_clip(6, 4, 23);
    let (frames, visible) = records(&clip, RecordOptions::default());
    let packed = pack_records(&frames, &visible, RecordEncoding::Hex, 64, 65_535, false).unwrap();
    let stride = packed.index_width + 6;
    for bank in &packed.banks {
        let Bank::Hex { chunks, records } = bank else {
            panic!("hex banks")
        };
        for (s, n) in chunks.iter().zip(records) {
            assert!(s.len() <= 64);
            assert_eq!(s.len(), stride * *n as usize);
        }
    }
    assert_eq!(packed.frame_first.len(), clip.frames.len());
    assert_eq!(
        packed.pixel_count,
        5 * 4,
        "the never-visible column has no brick"
    );
}

#[test]
fn the_player_is_the_same_size_at_every_resolution() {
    let size = |w, h| {
        let clip = busy_clip(w, h, 6);
        let (frames, visible) = records(&clip, RecordOptions::default());
        let packed = pack_records(
            &frames,
            &visible,
            RecordEncoding::Hex,
            10_000,
            65_535,
            false,
        )
        .unwrap();
        build(&packed).graph.gate_count()
    };
    let small = size(4, 3);
    assert_eq!(small, size(64, 36));
    // 27 of them are the cut-short reports and the step that clears them.
    assert!(small <= 86, "{small} gates");
}

fn ref_opts() -> AnimOptions {
    AnimOptions {
        driver: Driver::Refs,
        ..AnimOptions::default()
    }
}

#[test]
fn a_ref_render_writes_and_every_wire_resolves() {
    for (encoding, dense) in [
        (AnimEncoding::Hex, false),
        (AnimEncoding::Hex, true),
        (AnimEncoding::ColorArray, false),
    ] {
        let clip = busy_clip(6, 4, 9);
        let mut opts = ref_opts();
        opts.records.dense_keyframes = dense;
        let world = AnimMode::Brick(encoding)
            .build(&clip, &opts, &mut NoProgress)
            .unwrap();
        let path = std::env::temp_dir().join(format!(
            "h2b_refs_{encoding:?}_{dense}_{}.brz",
            std::process::id()
        ));
        std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
        wire_integrity::assert_wires_valid(&path);
        let _ = std::fs::remove_file(&path);
    }
}

/// The baked `pixels` refs, read back from the save, land on distinct display
/// bricks of the main grid.
#[test]
fn the_pixel_refs_resolve_to_the_display_bricks() {
    let clip = busy_clip(6, 4, 3);
    let world = AnimMode::Brick(AnimEncoding::Hex)
        .build(&clip, &ref_opts(), &mut NoProgress)
        .unwrap();
    let display_asset = world.bricks[0].asset.clone();
    let path = std::env::temp_dir().join(format!("h2b_refs_resolve_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    let db = brdb::Brz::open(&path).expect("open").into_reader();
    let global = db.global_data().expect("global data");

    let mut refs = None;
    let mut array_chunk = None;
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
                    && let Ok(WireArray::Bricks(r)) = WireArray::try_from(v.prop("Value").unwrap())
                {
                    refs = Some(r);
                    array_chunk = Some(chunk.index);
                }
            }
        }
    }
    let refs = refs.expect("a brick[] ArrayVar on the main grid");
    let array_chunk = array_chunk.unwrap();
    assert_eq!(refs.len(), 5 * 4);
    let mut seen = std::collections::HashSet::new();
    for r in &refs {
        assert_eq!(r.grid_persistent_index, 1, "absolute form: the main grid");
        assert_ne!(
            r.chunk_index, array_chunk,
            "the array sits in another chunk, so this checks the cross-chunk case"
        );
        let chunk = r.chunk_index;
        let soa = db
            .brick_chunk_soa(1, chunk)
            .expect("the ref's chunk exists");
        let i = r.brick_index_in_chunk as usize;
        assert!(i < soa.relative_positions.len(), "index {i} in range");
        assert!(seen.insert((chunk, i)), "refs are distinct");
    }
    // Every display brick is first in its chunk's order, so the refs are
    // 0..n in the one chunk this small screen fits in.
    assert!(
        refs.iter()
            .enumerate()
            .all(|(k, r)| r.brick_index_in_chunk as usize == k)
    );
    let _ = display_asset;
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_estimate_matches_the_render() {
    for encoding in [AnimEncoding::Hex, AnimEncoding::ColorArray] {
        let clip = busy_clip(6, 4, 9);
        let mut opaque = clip.clone();
        for f in &mut opaque.frames {
            for p in f.pixels_mut() {
                p.0[3] = 255;
            }
        }
        for dense_keyframes in [false, true] {
            let mut opts = ref_opts();
            opts.records.dense_keyframes = dense_keyframes;
            let world = AnimMode::Brick(encoding)
                .build(&opaque, &opts, &mut NoProgress)
                .unwrap();
            let est = AnimMode::Brick(encoding).estimate(6, 4, 9, &opts).unwrap();
            let what = format!("{encoding:?} dense {dense_keyframes}");
            assert_eq!(est.bricks, world.bricks.len(), "{what} bricks");
            assert_eq!(est.wires, world.wires.len(), "{what} wires");
            let in_grids: usize = world.grids.iter().map(|(_, b)| b.len()).sum();
            // Gates: every grid brick but the clock's 7 pins and the status
            // pin, plus the 5 main-grid arrays (`pixels`, one bank pair, the
            // two bank-ref arrays).
            assert_eq!(est.gates, in_grids - 8 + 5, "{what} gates");
        }
    }
}

#[test]
fn records_spill_into_banks_without_splitting_a_frame() {
    let clip = busy_clip(6, 4, 23);
    let (frames, visible) = records(&clip, RecordOptions::default());
    let packed =
        pack_records(&frames, &visible, RecordEncoding::ColorArray, 64, 30, false).unwrap();
    assert!(packed.banks.len() > 3, "{} banks", packed.banks.len());
    for (f, n) in packed.frame_entries.iter().enumerate() {
        let bank = &packed.banks[packed.frame_bank[f] as usize];
        assert!(
            packed.frame_first[f] + n <= bank.len() as i64,
            "frame {f} fits its bank"
        );
    }
    assert!(packed.banks.iter().all(|b| b.len() <= 30));
    // A frame larger than a bank is refused, not split.
    assert!(pack_records(&frames, &visible, RecordEncoding::ColorArray, 64, 10, false).is_err());
}

/// A render whose records need several banks writes the banks and the two
/// arrays of refs to them, and every wire resolves.
#[test]
fn a_many_bank_render_writes_and_every_wire_resolves() {
    let opts = AnimOptions {
        bank_size: 40,
        ..ref_opts()
    };
    let chunks = bank_ref_chunks(AnimEncoding::ColorArray, &opts);
    assert_eq!(
        chunks.len(),
        1,
        "under the chunk limit, the banks stay where they were"
    );
}

/// Past `chunk_bytes`, record banks move to further chunks, and the refs to
/// them, read back from the save, still land on `ArrayVar`s there.
#[test]
fn banks_past_the_chunk_limit_spread_over_chunks() {
    // Colour-array records spill into many banks at 40 entries; a hex clip
    // with more banks than one would be built wired. The spread is the same
    // code for both.
    let opts = AnimOptions {
        bank_size: 40,
        chunk_bytes: 1,
        ..ref_opts()
    };
    let chunks = bank_ref_chunks(AnimEncoding::ColorArray, &opts);
    assert!(chunks.len() > 2, "{} chunks", chunks.len());
}

/// Render a 30-frame busy clip, check every wire, read the two bank-ref
/// arrays back, and return the chunks the refs point into.
fn bank_ref_chunks(
    encoding: AnimEncoding,
    opts: &AnimOptions,
) -> std::collections::HashSet<ChunkIndex> {
    let clip = busy_clip(6, 4, 30);
    let world = AnimMode::Brick(encoding)
        .build(&clip, opts, &mut NoProgress)
        .unwrap();
    let outer: Vec<usize> = world
        .bricks
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            b.components.iter().any(|c| {
                c.component_type()
                    .is_some_and(|t| t.to_string() == "BrickComponentType_ArrayVar")
            })
        })
        .map(|(i, _)| i)
        .collect();
    assert!(
        outer.len() > 5,
        "pixels + several bank pairs + two ref arrays: {}",
        outer.len()
    );
    let path = std::env::temp_dir().join(format!(
        "h2b_refs_banks_{encoding:?}_{}_{}.brz",
        opts.chunk_bytes,
        std::process::id()
    ));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    wire_integrity::assert_wires_valid(&path);

    // Read the two bank-ref arrays back: every ref is an absolute main-grid
    // ref to an `ArrayVar` component that exists.
    let db = brdb::Brz::open(&path).expect("open").into_reader();
    let global = db.global_data().expect("global data");
    let array_type = global
        .component_type_names
        .get_index_of("BrickComponentType_ArrayVar")
        .expect("ArrayVar registered") as u16;
    let mut refs = Vec::new();
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
                {
                    refs.push(r);
                }
            }
        }
    }
    assert_eq!(refs.len(), 2, "the A and B bank-ref arrays");
    assert_eq!(refs[0].len(), refs[1].len());
    assert!(refs[0].len() > 1, "several banks");
    for r in refs.iter().flatten() {
        assert_eq!(r.brick.grid_persistent_index, 1);
        assert_eq!(r.component_type_index, array_type, "a ref to an ArrayVar");
        let soa = db.brick_chunk_soa(1, r.brick.chunk_index).expect("chunk");
        assert!((r.brick.brick_index_in_chunk as usize) < soa.relative_positions.len());
    }
    let _ = std::fs::remove_file(&path);
    refs.iter().flatten().map(|r| r.brick.chunk_index).collect()
}

/// The status text after playing frames `0..n` in order with the server's
/// per-tick gate budget set to `limit`.
fn status_after_playing(encoding: RecordEncoding, limit: Option<u64>) -> String {
    let clip = busy_clip(12, 9, 6);
    let opts = RecordOptions::default();
    let (frames, visible) = records(&clip, opts);
    let packed = pack_records(&frames, &visible, encoding, 64, 4096, false).unwrap();
    let player = build(&packed);
    let mut sim = simulate(&player, &packed);
    sim.gate_limit = limit;
    for target in 0..clip.frames.len() {
        sim.bind(ext("Frame"), Val::Int(target as i64));
        sim.fire_ext(ext("OnChanged"));
    }
    match sim.var(player.status.expect("a status var")) {
        Val::Str(s) => s.clone(),
        other => panic!("status is not a string: {other:?}"),
    }
}

#[test]
fn the_status_stays_blank_when_every_loop_finishes() {
    for encoding in [RecordEncoding::Hex, RecordEncoding::ColorArray] {
        assert_eq!(status_after_playing(encoding, None), "", "{encoding:?}");
    }
}

/// A frame change cut short by the gate budget leaves a warning naming the
/// frame and how far the loop got.
#[test]
fn a_loop_cut_short_by_the_gate_budget_leaves_a_warning() {
    for encoding in [RecordEncoding::Hex, RecordEncoding::ColorArray] {
        let status = status_after_playing(encoding, Some(150));
        assert!(status.contains("Loop cut short at frame"), "{encoding:?}: {status:?}");
        assert!(!status.contains('{'), "every placeholder filled: {status:?}");
        assert!(status.contains("Loop Gates off"), "{status:?}");
    }
}

/// Raising the server's limit clears the warning at the next keyframe, not
/// on the next delta, which would flash a cut-short keyframe's warning for a
/// single frame.
#[test]
fn raising_the_limit_clears_the_warning_at_the_next_keyframe() {
    // One pixel changes a frame, so every frame between keyframes is a delta.
    let (w, h) = (12u32, 9u32);
    let clip = Clip {
        width: w,
        height: h,
        fps: 10.0,
        frames: (0..8u32)
            .map(|f| {
                RgbaImage::from_fn(w, h, |x, y| {
                    let lit = y * w + x == f;
                    Rgba([if lit { 255 } else { 20 }, 40, 60, 255])
                })
            })
            .collect(),
    };
    let opts = RecordOptions {
        keyframe_interval: 4,
        ..RecordOptions::default()
    };
    let (frames, visible) = records(&clip, opts);
    let keys: Vec<bool> = frames.iter().map(|f| f.key).collect();
    assert_eq!(keys, [true, false, false, false, true, false, false, false]);
    let packed = pack_records(&frames, &visible, RecordEncoding::Hex, 64, 4096, false).unwrap();
    let player = build(&packed);
    let mut sim = simulate(&player, &packed);
    let status = |sim: &Sim| match sim.var(player.status.unwrap()) {
        Val::Str(s) => s.clone(),
        other => panic!("status is not a string: {other:?}"),
    };
    let mut play = |sim: &mut Sim, target: usize| {
        sim.bind(ext("Frame"), Val::Int(target as i64));
        sim.fire_ext(ext("OnChanged"));
    };
    sim.gate_limit = Some(150);
    play(&mut sim, 0);
    let warning = status(&sim);
    assert!(warning.contains("Loop cut short at frame 0"), "{warning:?}");
    assert_eq!(warning.lines().count(), 3, "{warning:?}");
    assert!(warning.contains("Max Work in Callable Chip Limits"), "{warning:?}");

    sim.gate_limit = None;
    play(&mut sim, 1);
    assert_eq!(status(&sim), warning, "a delta does not clear it");
    play(&mut sim, 4);
    assert_eq!(status(&sim), "", "a whole keyframe clears it");
}
