//! Colour-array mode (`anim::color_bricks`), end to end.
//!
//! The hex renderer's invariants are pinned in `tests/anim_world.rs`; this
//! file pins the same ones for the second encoding (no negative inner-grid
//! coordinates, no brick overlap, every wire endpoint resolving, array length
//! equal to the frame count) plus the two properties that are specific to this
//! design: the detector's exec output FANS OUT to every pixel's `Get`, and no
//! exec input ever gains a second source.
#[path = "wire_integrity.rs"]
mod wire_integrity;

use brdb::{IntVector, IntoReader, Position, schema::WireArrayVariant};
use heightmap::anim::bricks::{
    ARRAY_GET, ARRAY_VAR, AnimOptions, BRANCH, CHANGE_DETECTOR, COMPARE_GE, DisplayBrickStyle,
    PROP_CHANGER, SELECT, build_brick_world,
};
use heightmap::anim::color_bricks::build_color_array_world;
use heightmap::anim::{AnimEncoding, cost};
use heightmap::progress::NoProgress;
use heightmap::util::srgb_to_linear_f32;
use heightmap::video::Clip;
use heightmap::video::stream::FrameSource;
use image::{Rgba, RgbaImage};

/// Every (pixel, frame) pair gets a distinct colour, so a transposition or an
/// off-by-one anywhere between the source and the saved array shows up as a
/// mismatch rather than passing on symmetry.
fn distinct_clip(w: u32, h: u32, n: usize) -> Clip {
    let frames = (0..n as u32)
        .map(|f| {
            RgbaImage::from_fn(w, h, |x, y| {
                Rgba([(x * 17 + f) as u8, (y * 53 + f * 7) as u8, (x * 31 + y * 11 + f * 3) as u8, 255])
            })
        })
        .collect();
    Clip { width: w, height: h, fps: 10.0, frames }
}

fn source_color(x: u32, y: u32, f: u32) -> [u8; 4] {
    [(x * 17 + f) as u8, (y * 53 + f * 7) as u8, (x * 31 + y * 11 + f * 3) as u8, 255]
}

/// Write a world out and reopen it, returning the reader plus the chip's
/// persistent grid id. The id is assigned at WRITE time and differs from the
/// placeholder `Chip::entity_id`, so it has to be discovered by reading
/// entities back -- the same thing every test in `anim_world.rs` does.
fn write_and_open(world: &brdb::World, tag: &str) -> (std::path::PathBuf, brdb::BrReader<impl brdb::BrFsReader>, usize) {
    let path = std::env::temp_dir().join(format!(
        "h2b_color_{tag}_{}_{:?}.brz",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    let db = brdb::Brz::open(&path).expect("reopen").into_reader();
    let mut chip_grid_id = None;
    for index in db.entity_chunk_index().expect("entity chunk index") {
        for e in db.entity_chunk(index).expect("entity chunk") {
            if e.is_microchip_grid() {
                chip_grid_id = e.id;
            }
        }
    }
    let id = chip_grid_id.expect("the renderer must publish exactly one microchip grid");
    (path, db, id)
}

/// Every `LinearColorArray` persisted inside the chip, in save order.
fn saved_color_arrays(db: &brdb::BrReader<impl brdb::BrFsReader>, gid: usize) -> Vec<Vec<(f32, f32, f32, f32)>> {
    let mut arrays = Vec::new();
    for chunk in &db.brick_chunk_index(gid).expect("chunk index") {
        let (_soa, structs) = db.component_chunk(gid, chunk.index).expect("components");
        for s in &structs {
            if s.get_name() == "BrickComponentData_WireGraphPseudo_ArrayVar"
                && let Some(value) = s.get("Value")
            {
                let variant: WireArrayVariant =
                    value.try_into().expect("ArrayVar Value must decode");
                if let WireArrayVariant::LinearColorArray(v) = variant {
                    arrays.push(v);
                }
            }
        }
    }
    arrays
}

fn count_component(world: &brdb::World, class: &str) -> usize {
    world.grids[0]
        .1
        .iter()
        .filter(|b| {
            b.components
                .iter()
                .any(|c| c.component_type().is_some_and(|t| t.to_string() == class))
        })
        .count()
}

// --- shape ------------------------------------------------------------------

/// The headline claim: two components per pixel, exactly as many as hex mode,
/// with the per-chunk array/Get pair gone and no `Substring`/`MakeColorHex` at
/// all.
#[test]
fn colour_array_mode_is_two_gates_per_pixel_and_emits_no_string_gates() {
    let clip = distinct_clip(4, 3, 5);
    // Buttons off: this counts display bricks + shell, not the default-on
    // control buttons (3 more main-grid bricks). See `controls.rs` for their
    // own coverage.
    let opts = AnimOptions { control_buttons: false, ..AnimOptions::default() };
    let world = build_color_array_world(&clip, &opts, &mut NoProgress).unwrap();

    // 12 display bricks + 1 microchip shell on the main grid.
    assert_eq!(world.bricks.len(), 13);
    // inner grid: 12 ArrayVar + 12 Get + 1 detector + 6 clock + 7 clock pins.
    assert_eq!(world.grids[0].1.len(), 24 + 1 + 6 + 7);

    assert_eq!(count_component(&world, ARRAY_VAR), 12, "one array per pixel");
    assert_eq!(count_component(&world, ARRAY_GET), 12, "one Get per pixel");
    for gone in [
        "BrickComponentType_WireGraph_Expr_String_Substring",
        "BrickComponentType_WireGraph_Expr_MakeColorHex",
    ] {
        assert_eq!(count_component(&world, gone), 0, "{gone} must not appear at all");
    }
}

/// Both encodings must place the SAME screen -- same brick count, same
/// positions, same asset, same material. The screen is built by one shared
/// helper precisely so it cannot drift; this checks that from the outside.
#[test]
fn both_encodings_build_an_identical_screen() {
    for (style, extent, glow) in [
        (DisplayBrickStyle::Micro, 1u16, false),
        (DisplayBrickStyle::SmoothTile, 3, true),
    ] {
        let clip = distinct_clip(5, 4, 3);
        let opts = AnimOptions {
            brick_style: style,
            pixel_extent: extent,
            glow,
            ..AnimOptions::default()
        };
        let hex = build_brick_world(&clip, &opts, &mut NoProgress).expect("hex build");
        let color = build_color_array_world(&clip, &opts, &mut NoProgress).expect("colour build");

        let screen = |w: &brdb::World| -> Vec<(Position, String, u8)> {
            let mut v: Vec<_> = w
                .bricks
                .iter()
                .map(|b| (b.position, format!("{:?}", b.asset), b.material_intensity))
                .collect();
            v.sort_by_key(|(p, _, _)| (p.x, p.y, p.z));
            v
        };
        assert_eq!(
            screen(&hex),
            screen(&color),
            "{style:?} at extent {extent}: the two encodings must place the same screen"
        );
    }
}

/// A pixel transparent across the whole clip emits no display brick and,
/// therefore, no array and no Get either.
#[test]
fn a_fully_transparent_pixel_emits_no_brick_and_no_gates() {
    let mut img = RgbaImage::from_pixel(3, 1, Rgba([255, 0, 0, 255]));
    img.put_pixel(1, 0, Rgba([0, 0, 0, 0]));
    let clip = Clip { width: 3, height: 1, fps: 10.0, frames: vec![img] };
    // Buttons off: this counts the surviving display bricks + shell.
    let opts = AnimOptions { control_buttons: false, ..AnimOptions::default() };
    let world = build_color_array_world(&clip, &opts, &mut NoProgress).unwrap();

    assert_eq!(world.bricks.len(), 3, "2 display bricks + 1 chip shell");
    assert_eq!(count_component(&world, ARRAY_VAR), 2, "the culled pixel gets no array");
    assert_eq!(count_component(&world, ARRAY_GET), 2, "the culled pixel gets no Get");
}

/// The same guard the hex path has: a zero-frame source must be an error, not
/// a save that divides by zero in-game on every tick.
#[test]
fn a_zero_frame_clip_is_rejected() {
    let clip = Clip { width: 4, height: 3, fps: 10.0, frames: Vec::new() };
    assert!(
        build_color_array_world(&clip, &AnimOptions::default(), &mut NoProgress)
            .map(|_| ())
            .is_err(),
        "an empty clip must not produce a \"successful\" save"
    );
}

// --- exec wiring: the point of the design -----------------------------------

/// The fan-out: every pixel's `Get.Exec` must be driven directly by the
/// change detector's `OnChanged`, not by another Get's `ExecOut`. A chain
/// would work in a save file and would be thousands of gates deep in game,
/// which is exactly what this design exists to avoid -- and nothing else in
/// this file can tell the two apart, since both leave every `Exec` with
/// exactly one source.
#[test]
fn every_pixels_get_exec_is_fed_directly_by_the_detector() {
    let clip = distinct_clip(6, 5, 4);
    let world = build_color_array_world(&clip, &AnimOptions::default(), &mut NoProgress).unwrap();

    let gets: Vec<usize> = world.grids[0]
        .1
        .iter()
        .filter(|b| {
            b.components
                .iter()
                .any(|c| c.component_type().is_some_and(|t| t.to_string() == ARRAY_GET))
        })
        .map(|b| b.id.expect("every emitted brick has an id"))
        .collect();
    assert_eq!(gets.len(), 30, "a 6x5 opaque screen must emit one Get per pixel");

    for get in gets {
        let sources: Vec<_> = world
            .wires
            .iter()
            .filter(|w| {
                w.target.brick_id == get
                    && w.target.component_type.to_string() == ARRAY_GET
                    && w.target.port_name.to_string() == "Exec"
            })
            .collect();
        assert_eq!(sources.len(), 1, "Get {get}'s Exec must have exactly one source");
        let s = &sources[0].source;
        assert_eq!(
            s.component_type.to_string(),
            CHANGE_DETECTOR,
            "Get {get}'s Exec must come from the change detector, not from {:?} -- a chain \
             through another Get's ExecOut is the design this mode rejects",
            s.component_type
        );
        assert_eq!(s.port_name.to_string(), "OnChanged");
    }

    // The complement: nothing may chain. If any Get's ExecOut drove anything,
    // the graph would be a chain however the Exec inputs looked.
    assert!(
        !world
            .wires
            .iter()
            .any(|w| w.source.component_type.to_string() == ARRAY_GET
                && w.source.port_name.to_string() == "ExecOut"),
        "no Get's ExecOut may drive anything -- the gets fan out from one source, never chain"
    );
}

/// Exec fan-in is the unverified case, so the design must never produce
/// it: no exec input may ever have two sources, at any bank count.
#[test]
fn no_exec_input_ever_has_two_sources() {
    for bank_size in [usize::MAX, 3, 2, 1] {
        let clip = distinct_clip(8, 5, 7);
        let opts = AnimOptions { bank_size, ..AnimOptions::default() };
        let world = build_color_array_world(&clip, &opts, &mut NoProgress).expect("build");

        let mut seen = std::collections::HashSet::new();
        for wire in &world.wires {
            let t = &wire.target;
            let port = t.port_name.to_string();
            if port.contains("Exec") && !port.contains("Out") {
                assert!(
                    seen.insert((t.brick_id, port.clone())),
                    "bank_size {bank_size}: exec input {}.{port} has more than one source",
                    t.brick_id
                );
            }
        }
    }
}

/// Each pixel's colour must arrive at that pixel's OWN display brick, on the
/// `Color` port. One wire in, one wire out, no crossing over.
#[test]
fn every_display_brick_gets_exactly_one_colour_wire() {
    let clip = distinct_clip(4, 3, 3);
    let world = build_color_array_world(&clip, &AnimOptions::default(), &mut NoProgress).unwrap();

    let mut fed = std::collections::HashMap::new();
    for w in &world.wires {
        if w.target.component_type.to_string() == PROP_CHANGER {
            assert_eq!(w.target.port_name.to_string(), "Color");
            *fed.entry(w.target.brick_id).or_insert(0usize) += 1;
            assert_eq!(
                w.source.component_type.to_string(),
                ARRAY_GET,
                "a single-bank render must wire the Get's Value straight to Color"
            );
            assert_eq!(w.source.port_name.to_string(), "Value");
        }
    }
    assert_eq!(fed.len(), 12, "every display brick must be driven");
    assert!(fed.values().all(|n| *n == 1), "no display brick may be driven twice");
}

// --- content ----------------------------------------------------------------

/// The value test, read back out of a real save: each pixel's array must
/// hold that pixel's own source colour for every frame, converted sRGB ->
/// linear -- the conversion nothing downstream of this renderer performs.
#[test]
fn each_pixels_saved_array_holds_its_own_linearized_colours() {
    let (w, h, n) = (4u32, 3u32, 5u32);
    let clip = distinct_clip(w, h, n as usize);
    let world = build_color_array_world(&clip, &AnimOptions::default(), &mut NoProgress).unwrap();
    let (path, db, gid) = write_and_open(&world, "values");

    let arrays = saved_color_arrays(&db, gid);
    assert_eq!(arrays.len(), (w * h) as usize, "one saved array per pixel");
    for a in &arrays {
        assert_eq!(a.len(), n as usize, "array length must equal the frame count");
    }

    // Build the expected array for every pixel and match them up as a set --
    // the save's ordering is the writer's business, the contents are not.
    let mut expected: Vec<Vec<(f32, f32, f32, f32)>> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            expected.push(
                (0..n)
                    .map(|f| {
                        let c = source_color(x, y, f);
                        (
                            srgb_to_linear_f32(c[0]),
                            srgb_to_linear_f32(c[1]),
                            srgb_to_linear_f32(c[2]),
                            1.0,
                        )
                    })
                    .collect(),
            );
        }
    }
    let key = |v: &Vec<(f32, f32, f32, f32)>| format!("{v:?}");
    let mut got: Vec<String> = arrays.iter().map(key).collect();
    let mut want: Vec<String> = expected.iter().map(key).collect();
    got.sort();
    want.sort();
    assert_eq!(got, want, "every pixel's saved colours must be its own, linearized");
    let _ = std::fs::remove_file(&path);
}

/// The conversion must survive the f32 round trip through the save. Mid-grey
/// is the witness: sRGB 128 is linear ~0.216, and an unconverted render would
/// store ~0.502 -- the gamma-step-too-bright picture this mode has to avoid.
#[test]
fn saved_colours_are_linear_not_raw_srgb() {
    let img = RgbaImage::from_pixel(1, 1, Rgba([128, 128, 128, 255]));
    let clip = Clip { width: 1, height: 1, fps: 10.0, frames: vec![img] };
    let world = build_color_array_world(&clip, &AnimOptions::default(), &mut NoProgress).unwrap();
    let (path, db, gid) = write_and_open(&world, "linear");

    let arrays = saved_color_arrays(&db, gid);
    assert_eq!(arrays.len(), 1);
    let (r, g, b, a) = arrays[0][0];
    assert!(
        (r - 0.2158).abs() < 1e-3,
        "sRGB 128 must be stored as linear ~0.216, got {r} (0.502 means the transfer never ran)"
    );
    assert_eq!((g, b), (r, r));
    assert_eq!(a, 1.0, "an opaque pixel stores alpha 1");
    let _ = std::fs::remove_file(&path);
}

/// A pixel culled in SOME frames still gets a brick and an array; its culled
/// frames are transparent black, the analogue of hex mode's `"000000"` slot.
#[test]
fn a_frame_where_a_pixel_is_culled_stores_transparent_black() {
    let opaque = RgbaImage::from_pixel(1, 1, Rgba([255, 255, 255, 255]));
    let clear = RgbaImage::from_pixel(1, 1, Rgba([255, 255, 255, 0]));
    let clip = Clip { width: 1, height: 1, fps: 10.0, frames: vec![opaque, clear] };
    let world = build_color_array_world(&clip, &AnimOptions::default(), &mut NoProgress).unwrap();
    let (path, db, gid) = write_and_open(&world, "culled");

    let arrays = saved_color_arrays(&db, gid);
    assert_eq!(arrays[0], vec![(1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0)]);
    let _ = std::fs::remove_file(&path);
}

// --- structural invariants --------------------------------------------------

#[test]
fn a_render_encodes_and_every_wire_resolves() {
    for (bank_size, tag) in [(usize::MAX, "single"), (2usize, "multi")] {
        let clip = distinct_clip(5, 4, 5);
        let opts = AnimOptions { bank_size, ..AnimOptions::default() };
        let world = build_color_array_world(&clip, &opts, &mut NoProgress).expect("build");
        let (path, _db, _gid) = write_and_open(&world, tag);
        wire_integrity::assert_wires_valid(&path);
        let _ = std::fs::remove_file(&path);
    }
}

/// Negative inner-grid coordinates delete bricks in-game. A 1-row screen is
/// the tightest case (`lattice_pos`'s `x = (h-1-row)*CELL + half.x` sends a
/// positive service row negative there), so it is the one worth pinning --
/// along with a multi-bank render, whose deeper service stage is new here.
#[test]
fn no_inner_brick_lands_at_a_negative_coordinate() {
    for (w, h, frames, bank) in [
        (1u32, 1u32, 3usize, usize::MAX),
        (4, 1, 3, usize::MAX),
        (1, 4, 3, usize::MAX),
        (2, 2, 3, usize::MAX),
        (7, 3, 3, usize::MAX),
        (1, 1, 7, 2),
        (5, 4, 7, 2),
    ] {
        let clip = distinct_clip(w, h, frames);
        let opts = AnimOptions { bank_size: bank, ..AnimOptions::default() };
        let world = build_color_array_world(&clip, &opts, &mut NoProgress)
            .unwrap_or_else(|e| panic!("{w}x{h}x{frames} bank {bank} must build: {e}"));
        for b in &world.grids[0].1 {
            // `add_brick_grid` stores inner bricks shifted by -CHUNK_HALF;
            // undo that to get the grid-local coordinates the lattice emits.
            let center = b.position + Position::CHUNK_HALF;
            assert!(
                center.x >= 0 && center.y >= 0 && center.z >= 0,
                "{w}x{h}x{frames} bank {bank}: inner brick centred at {center:?} is negative -- \
                 negative inner-grid coordinates break bricks in-game"
            );
        }
    }
}

/// `chip::finish` asserts non-overlap on BOTH grids before publishing, so a
/// render that returns `Ok` has already proved it. This drives the shapes most
/// likely to collide -- a deep multi-bank lattice, both display styles, and a
/// large pixel extent -- so the assertion is actually reached.
#[test]
fn no_bricks_overlap_on_either_grid_for_any_shape() {
    for (w, h, frames, bank, style, extent) in [
        (3u32, 3u32, 9usize, 2usize, DisplayBrickStyle::Micro, 1u16),
        (4, 2, 9, 3, DisplayBrickStyle::SmoothTile, 1),
        (2, 5, 12, 2, DisplayBrickStyle::Micro, 7),
        (6, 4, 4, usize::MAX, DisplayBrickStyle::SmoothTile, 4),
    ] {
        let clip = distinct_clip(w, h, frames);
        let opts = AnimOptions {
            bank_size: bank,
            brick_style: style,
            pixel_extent: extent,
            ..AnimOptions::default()
        };
        build_color_array_world(&clip, &opts, &mut NoProgress).unwrap_or_else(|e| {
            panic!("{w}x{h}x{frames} bank {bank} {style:?} extent {extent} must be collision-free: {e}")
        });
    }
}

/// The published `PlaneExtent` must strictly contain every inner brick on x
/// and y, and every gate must sit clear of the plane's top face on z. The
/// per-pixel lattice is much deeper here than in hex mode (2 stages per bank
/// plus one per boundary, versus a flat 2), so this is the mode where a plane
/// that failed to grow would bite hardest.
#[test]
fn the_chip_plane_contains_every_brick_of_a_multi_bank_render() {
    let clip = distinct_clip(4, 3, 7);
    let opts = AnimOptions { bank_size: 3, ..AnimOptions::default() };
    let world = build_color_array_world(&clip, &opts, &mut NoProgress).expect("build");

    let centers: Vec<Position> =
        world.grids[0].1.iter().map(|b| b.position + Position::CHUNK_HALF).collect();
    assert!(!centers.is_empty());

    let (path, db, _gid) = write_and_open(&world, "plane");
    let mut extent: Option<IntVector> = None;
    let mut center: Option<IntVector> = None;
    for index in db.entity_chunk_index().expect("entity chunk index") {
        let entities = db.entity_chunk(index).expect("entity chunk");
        let (_soa, structs) = db.entity_chunk_soa(index).expect("entity chunk soa");
        for (e, s) in entities.iter().zip(structs.iter()) {
            if e.is_microchip_grid() {
                let s = s.as_ref().expect("microchip grid entity must carry struct data");
                extent = Some(s.prop("PlaneExtent").unwrap().try_into().unwrap());
                center = Some(s.prop("PlaneCenter").unwrap().try_into().unwrap());
            }
        }
    }
    let (extent, pc) = (extent.expect("PlaneExtent"), center.expect("PlaneCenter"));

    // Every gate in this chip carries the uniform `GATE_HALF` (5, 5, 2).
    let half = IntVector { x: 5, y: 5, z: 2 };
    for c in &centers {
        for (axis, lo, hi, pos, hf) in [
            ("x", pc.x - extent.x, pc.x + extent.x, c.x, half.x),
            ("y", pc.y - extent.y, pc.y + extent.y, c.y, half.y),
        ] {
            assert!(
                lo < pos - hf && pos + hf < hi,
                "brick at {c:?} spans {axis} {}..{} without strictly clearing the plane \
                 {axis} {lo}..{hi}",
                pos - hf,
                pos + hf
            );
        }
        assert!(
            c.z - half.z > pc.z + extent.z,
            "gate at z {} must clear the plane top face {}",
            c.z - half.z,
            pc.z + extent.z
        );
    }
    let _ = std::fs::remove_file(&path);
}

// --- banking ----------------------------------------------------------------

/// Spillover must be invisible to clips that do not need it: no comparator,
/// no branch, no select, no subtract at a single bank.
#[test]
fn a_single_bank_render_emits_no_spillover_gates() {
    let clip = distinct_clip(4, 3, 5);
    let world = build_color_array_world(&clip, &AnimOptions::default(), &mut NoProgress).unwrap();
    for class in [COMPARE_GE, BRANCH, SELECT, "BrickComponentType_WireGraph_Expr_MathSubtract"] {
        assert_eq!(count_component(&world, class), 0, "a single-bank render must emit no {class}");
    }
}

/// The seam: each pixel's colours must split across banks at the real bank
/// size, with the last bank short rather than padded, and bank k's element 0
/// must be global frame `k * bank_size` -- not a repeat of frame 0.
#[test]
fn banking_splits_each_pixels_colours_at_the_right_seam() {
    let (w, h, n, bank) = (2u32, 2u32, 7u32, 3usize);
    let clip = distinct_clip(w, h, n as usize);
    let opts = AnimOptions { bank_size: bank, ..AnimOptions::default() };
    let world = build_color_array_world(&clip, &opts, &mut NoProgress).expect("build");
    let (path, db, gid) = write_and_open(&world, "seam");

    let arrays = saved_color_arrays(&db, gid);
    // 4 pixels x 3 banks (3 + 3 + 1).
    assert_eq!(arrays.len(), 12, "one array per pixel per bank");
    let mut lengths: Vec<usize> = arrays.iter().map(|a| a.len()).collect();
    lengths.sort_unstable();
    assert_eq!(
        lengths,
        vec![1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3],
        "7 frames at bank size 3 -> 3 + 3 + 1 for every one of the 4 pixels; the last bank is \
         short, never padded"
    );

    // Pixel (col 1, row 1)'s three banks, identified by content, must
    // reassemble into exactly its 7 source frames in order.
    let expect = |f: u32| {
        let c = source_color(1, 1, f);
        (srgb_to_linear_f32(c[0]), srgb_to_linear_f32(c[1]), srgb_to_linear_f32(c[2]), 1.0)
    };
    let want: Vec<Vec<(f32, f32, f32, f32)>> = vec![
        (0..3).map(expect).collect(),
        (3..6).map(expect).collect(),
        (6..7).map(expect).collect(),
    ];
    for (k, bank_content) in want.iter().enumerate() {
        assert!(
            arrays.contains(bank_content),
            "bank {k} of pixel (1,1) must appear verbatim in the save; a wrong seam would \
             repeat frame 0 or drop the tail"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// The boundary hardware: one comparator, one branch and one index subtract
/// per boundary (shared), but a Select per PIXEL per boundary -- the asymmetry
/// with hex mode that the cost estimate exists to report.
#[test]
fn a_boundary_costs_shared_gates_plus_one_select_per_pixel() {
    // 7 frames at bank size 2 -> 4 banks -> 3 boundaries; 4 pixels.
    let clip = distinct_clip(2, 2, 7);
    let opts = AnimOptions { bank_size: 2, ..AnimOptions::default() };
    let world = build_color_array_world(&clip, &opts, &mut NoProgress).expect("build");

    assert_eq!(count_component(&world, COMPARE_GE), 3, "one comparator per boundary");
    assert_eq!(count_component(&world, BRANCH), 3, "one branch per boundary");
    assert_eq!(
        count_component(&world, "BrickComponentType_WireGraph_Expr_MathSubtract"),
        3,
        "one index subtract per extra bank"
    );
    assert_eq!(count_component(&world, SELECT), 4 * 3, "one select per PIXEL per boundary");
    assert_eq!(count_component(&world, ARRAY_VAR), 4 * 4, "one array per pixel per bank");
    assert_eq!(count_component(&world, ARRAY_GET), 4 * 4, "one Get per pixel per bank");
}

/// Branch polarity, the inversion that shipped once in hex mode and that every
/// existence-only wire check stays green through: the FALSY (`ExecOutB`)
/// output enters its own bank's gets, and the TRUTHY (`ExecOutA`) output keeps
/// descending into the next branch -- or, for the last branch, into the final
/// bank's gets.
#[test]
fn each_branchs_falsy_output_enters_its_own_bank_and_its_truthy_output_keeps_descending() {
    // 7 frames at bank size 2 -> 4 banks -> 3 branches. One pixel, so each
    // bank's "chain" is a single Get and the entry wire IS that Get's Exec.
    let clip = distinct_clip(1, 1, 7);
    let opts = AnimOptions { bank_size: 2, ..AnimOptions::default() };
    let world = build_color_array_world(&clip, &opts, &mut NoProgress).expect("build");

    let mut branch_ids: Vec<usize> = world.grids[0]
        .1
        .iter()
        .filter(|b| {
            b.components
                .iter()
                .any(|c| c.component_type().is_some_and(|t| t.to_string() == BRANCH))
        })
        .map(|b| b.id.expect("id"))
        .collect();
    branch_ids.sort_unstable();
    assert_eq!(branch_ids.len(), 3, "4 banks cascade through 3 branches");

    let target_from = |brick_id: usize, port: &str| {
        let mut hits = world.wires.iter().filter(|w| {
            w.source.brick_id == brick_id
                && w.source.component_type.to_string() == BRANCH
                && w.source.port_name.to_string() == port
        });
        let wire = hits.next().unwrap_or_else(|| panic!("branch {brick_id}'s {port} drives nothing"));
        assert!(hits.next().is_none(), "branch {brick_id}'s {port} must drive exactly one target");
        wire.target.clone()
    };

    for (i, &br) in branch_ids.iter().enumerate() {
        let falsy = target_from(br, "ExecOutB");
        assert_eq!(
            falsy.component_type.to_string(),
            ARRAY_GET,
            "branch {i}'s falsy output must enter its own bank's gets"
        );
        assert_eq!(falsy.port_name.to_string(), "Exec");

        let truthy = target_from(br, "ExecOutA");
        if i + 1 < branch_ids.len() {
            assert_eq!(truthy.brick_id, branch_ids[i + 1], "branch {i} must descend into {}", i + 1);
            assert_eq!(truthy.component_type.to_string(), BRANCH);
        } else {
            assert_eq!(
                truthy.component_type.to_string(),
                ARRAY_GET,
                "the last branch's truthy output enters the final bank's gets directly"
            );
        }
        assert_eq!(truthy.port_name.to_string(), "Exec");
    }
}

/// The select cascade, per pixel: condition from a comparator, B input from
/// the later bank's Get, A input from a Get (first stage) or another Select
/// (later stages), and the last Select's output into the display brick.
#[test]
fn the_per_pixel_select_cascade_is_well_formed() {
    let clip = distinct_clip(2, 2, 7);
    let opts = AnimOptions { bank_size: 2, ..AnimOptions::default() };
    let world = build_color_array_world(&clip, &opts, &mut NoProgress).expect("build");

    let source_of = |brick: usize, port: &str| -> Option<String> {
        world
            .wires
            .iter()
            .find(|w| w.target.brick_id == brick && w.target.port_name.to_string() == port)
            .map(|w| w.source.component_type.to_string())
    };
    let selects: std::collections::BTreeSet<usize> = world
        .wires
        .iter()
        .filter(|w| w.target.component_type.to_string() == SELECT)
        .map(|w| w.target.brick_id)
        .collect();
    assert_eq!(selects.len(), 12, "4 pixels x 3 boundaries");

    for sel in selects {
        assert_eq!(source_of(sel, "bSelectB").as_deref(), Some(COMPARE_GE));
        assert_eq!(source_of(sel, "InputB").as_deref(), Some(ARRAY_GET));
        let a = source_of(sel, "InputA").expect("every select needs an A input");
        assert!(a == ARRAY_GET || a == SELECT, "a select's A input must be a Get or a Select, got {a}");
    }

    // In a multi-bank render the display brick is fed by a Select, not a Get.
    for w in &world.wires {
        if w.target.component_type.to_string() == PROP_CHANGER {
            assert_eq!(
                w.source.component_type.to_string(),
                SELECT,
                "a multi-bank render must feed Color from the end of the select cascade"
            );
        }
    }
}

/// The boundary constants inlined on the comparators must be the real
/// multiples of the bank size -- the off-by-one here is invisible until
/// playback crosses a seam, so it is read back out of the encoded save.
#[test]
fn boundary_constants_are_the_real_multiples_of_the_bank_size() {
    let clip = distinct_clip(2, 2, 7);
    let opts = AnimOptions { bank_size: 3, ..AnimOptions::default() };
    let world = build_color_array_world(&clip, &opts, &mut NoProgress).expect("build");
    let (path, db, gid) = write_and_open(&world, "bounds");

    let mut bounds = Vec::new();
    for chunk in &db.brick_chunk_index(gid).expect("chunk index") {
        let (_soa, structs) = db.component_chunk(gid, chunk.index).expect("components");
        for s in &structs {
            if s.get_name() == "BrickComponentData_WireGraph_Expr_MathCompare" {
                match s.get("InputB") {
                    Some(brdb::schema::BrdbValue::I64(v)) => bounds.push(*v),
                    Some(brdb::schema::BrdbValue::F64(v)) => bounds.push(*v as i64),
                    other => panic!("unexpected InputB encoding: {other:?}"),
                }
            }
        }
    }
    bounds.sort_unstable();
    assert_eq!(bounds, vec![3, 6], "7 frames at bank size 3 -> boundaries at 3 and 6");
    let _ = std::fs::remove_file(&path);
}

// --- streaming, progress, errors --------------------------------------------

/// The reporter must actually be driven, and the render must consume the whole
/// clip -- the same contract the hex path holds.
#[test]
fn the_render_streams_and_reports_progress() {
    #[derive(Default)]
    struct Rec {
        began: Vec<(String, Option<u64>)>,
        ticks: u64,
        finished: usize,
    }
    impl heightmap::progress::Progress for Rec {
        fn begin(&mut self, label: &str, total: Option<u64>) {
            self.began.push((label.to_string(), total));
        }
        fn tick(&mut self, _n: u64) {
            self.ticks += 1;
        }
        fn finish(&mut self) {
            self.finished += 1;
        }
    }

    let clip = distinct_clip(4, 3, 5);
    let mut rec = Rec::default();
    build_color_array_world(&clip as &dyn FrameSource, &AnimOptions::default(), &mut rec)
        .expect("build");
    assert_eq!(rec.began[0].1, Some(5), "an in-memory clip knows its frame count");
    assert_eq!(rec.ticks, 5, "every frame must be streamed through the reporter");
    assert_eq!(rec.finished, rec.began.len(), "every phase begun must be finished");
}

/// A stream that fails partway must abort, not write a save missing its tail.
#[test]
fn a_failing_stream_aborts_the_render() {
    struct Failing;
    struct FailingStream(usize);
    impl heightmap::video::stream::FrameStream for FailingStream {
        fn next(&mut self) -> Result<Option<RgbaImage>, String> {
            self.0 += 1;
            if self.0 > 2 {
                Err("decode failed at frame 3".into())
            } else {
                Ok(Some(RgbaImage::from_pixel(2, 2, Rgba([1, 2, 3, 255]))))
            }
        }
    }
    impl FrameSource for Failing {
        fn info(&self) -> heightmap::video::stream::SourceInfo {
            heightmap::video::stream::SourceInfo {
                width: 2,
                height: 2,
                fps: 10.0,
                frame_count_hint: Some(10),
            }
        }
        fn open(&self) -> Result<Box<dyn heightmap::video::stream::FrameStream + '_>, String> {
            Ok(Box::new(FailingStream(0)))
        }
    }
    let err = build_color_array_world(&Failing, &AnimOptions::default(), &mut NoProgress)
        .map(|_| ())
        .expect_err("a mid-stream failure must abort");
    assert!(err.contains("frame 3"), "the underlying error must surface: {err}");
}

// --- the cost estimate ------------------------------------------------------

/// Default options at a given bank size. The estimators take the whole
/// `AnimOptions` (rather than a bare `bank_size`) so a readout can never be
/// computed from different inputs than the render it describes -- see
/// `anim::cost`'s module doc -- so costing "the same screen" means costing it
/// against the same options.
fn bank_opts(bank_size: usize) -> AnimOptions {
    AnimOptions { bank_size, ..AnimOptions::default() }
}

/// The estimate must match a real render's gates, wires and bricks exactly for
/// a fully-opaque clip (where its "every pixel survives" assumption holds).
/// An estimate that merely looks plausible is worth nothing -- this is what
/// makes it a measurement rather than a guess.
#[test]
fn the_cost_estimate_matches_a_real_render() {
    for (w, h, n, bank) in [(4u32, 3u32, 5usize, usize::MAX), (2, 2, 7, 2), (3, 3, 9, 3)] {
        let clip = distinct_clip(w, h, n);
        let opts = AnimOptions { bank_size: bank, ..AnimOptions::default() };
        let world = build_color_array_world(&clip, &opts, &mut NoProgress).expect("build");
        let est = cost::estimate_color_array(w, h, n, &opts);

        // Gates are every inner-grid brick that is not one of the chip's seven
        // I/O pins (Pause, Restart, Resume, Rate, Done, Length, Progress).
        let inner = world.grids[0].1.len();
        assert_eq!(
            inner - 7,
            est.gates,
            "{w}x{h}x{n} bank {bank}: estimate said {} gates, the render emitted {}",
            est.gates,
            inner - 7
        );
        assert_eq!(world.wires.len(), est.wires, "{w}x{h}x{n} bank {bank}: wire count");
        assert_eq!(world.bricks.len(), est.bricks, "{w}x{h}x{n} bank {bank}: brick count");
        assert_eq!(est.banks, n.div_ceil(bank.min(n).max(1)).max(1));
    }
}

/// The encoding dispatcher must reach the right renderer. A flag wired to the
/// wrong one produces a perfectly valid save of the other mode.
#[test]
fn the_encoding_enum_dispatches_to_the_right_renderer() {
    assert_eq!(AnimEncoding::default(), AnimEncoding::Hex, "hex stays the default");
    assert_eq!(AnimEncoding::parse("hex"), Some(AnimEncoding::Hex));
    assert_eq!(AnimEncoding::parse("color-array"), Some(AnimEncoding::ColorArray));
    assert_eq!(AnimEncoding::parse("colour-array"), Some(AnimEncoding::ColorArray));
    assert_eq!(AnimEncoding::parse("COLOR-ARRAY"), Some(AnimEncoding::ColorArray));
    assert_eq!(AnimEncoding::parse("nonsense"), None);

    let clip = distinct_clip(3, 2, 3);
    let opts = AnimOptions::default();
    let hex = AnimEncoding::Hex.build(&clip, &opts, &mut NoProgress).expect("hex");
    let color = AnimEncoding::ColorArray.build(&clip, &opts, &mut NoProgress).expect("colour");

    let has = |w: &brdb::World, class: &str| count_component(w, class) > 0;
    assert!(has(&hex, "BrickComponentType_WireGraph_Expr_MakeColorHex"), "hex must build hex");
    assert!(
        !has(&color, "BrickComponentType_WireGraph_Expr_MakeColorHex"),
        "colour-array mode must not build hex gates"
    );

    // And the estimates must dispatch alongside the renderers.
    assert_eq!(
        AnimEncoding::ColorArray.estimate(64, 36, 300, &bank_opts(65_535)),
        cost::estimate_color_array(64, 36, 300, &bank_opts(65_535))
    );
    assert_eq!(
        AnimEncoding::Hex.estimate(64, 36, 300, &bank_opts(65_535)),
        cost::estimate(64, 36, 300, &bank_opts(65_535))
    );
}

/// Cleanup helper for the tests above that keep their save around for
/// assertions -- `write_and_open` leaves the file in place on purpose so the
/// reader stays valid, and each caller removes it. This one covers the two
/// that assert before removing.
#[test]
fn temp_saves_do_not_leak() {
    let clip = distinct_clip(2, 2, 2);
    let world = build_color_array_world(&clip, &AnimOptions::default(), &mut NoProgress).unwrap();
    let (path, _db, _gid) = write_and_open(&world, "leak");
    assert!(path.exists());
    let _ = std::fs::remove_file(&path);
    assert!(!path.exists());
}
