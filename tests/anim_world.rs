#[path = "wire_integrity.rs"]
mod wire_integrity;

use brdb::{
    AsBrdbValue, Brick, BrickSize, BrickType, IntVector, IntoReader, Position, Vector3f, WirePort,
    World,
    assets::{
        LiteralComponent,
        bricks::{PB_DEFAULT_MICRO_BRICK, PB_DEFAULT_SMOOTH_TILE},
    },
    schema::WireArrayVariant,
};
use heightmap::anim::chip;
use heightmap::anim::clock::{build_clock, gate};
use heightmap::anim::layout::{CELL, GATE_HALF, lattice_pos};

/// Stand-in for the `gate` helper Task 5 adds: emit an arbitrary brick into
/// a chip. It exists here to prove that shape is writable against `Chip`'s
/// public API alone, with no access to its internals.
fn raw_gate(c: &mut chip::Chip, pos: Position) -> usize {
    c.add_brick(
        Brick {
            asset: brdb::assets::components::LogicGate::BoolNot.brick(),
            position: pos,
            ..Default::default()
        }
        .with_component(brdb::assets::components::LogicGate::BoolNot.component()),
        GATE_HALF,
    )
}

#[test]
fn a_chip_with_pins_writes_and_reads_back() {
    let mut world = World::new();
    world.meta.bundle.description = "chip smoke test".to_string();

    let mut c = chip::new_chip(
        &mut world,
        Position { x: 0, y: 0, z: 6 },
        Vector3f { x: 0.0, y: 0.0, z: 40.0 },
        IntVector { x: 14, y: 14, z: 5 },
    );
    let inp = chip::add_input_pin(&mut c, "Pause", Position { x: 5, y: 5, z: 2 });
    let out = chip::add_output_pin(&mut c, "Frame", Position { x: 5, y: 25, z: 2 });
    assert_ne!(inp, out, "pins must get distinct brick ids");
    chip::finish(&mut world, c).expect("chip must be collision-free");

    world.register_used_components();
    let path = std::env::temp_dir().join(format!("h2b_chip_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    assert!(std::fs::metadata(&path).unwrap().len() > 0);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn overlapping_pins_are_rejected() {
    let mut world = World::new();
    let mut c = chip::new_chip(
        &mut world,
        Position { x: 0, y: 0, z: 6 },
        Vector3f { x: 0.0, y: 0.0, z: 40.0 },
        IntVector { x: 14, y: 14, z: 5 },
    );
    let p = Position { x: 5, y: 5, z: 2 };
    chip::add_input_pin(&mut c, "A", p);
    chip::add_input_pin(&mut c, "B", p);
    assert!(chip::finish(&mut world, c).is_err(), "stacked pins must be caught");
}

/// A brick added through `Chip::add_brick` must register its bounds, so the
/// collision check sees it. If `bricks` and `placed` could drift apart, a
/// gate stacked on a pin would serialize happily and be dropped in game.
#[test]
fn bricks_added_via_add_brick_are_collision_checked() {
    let mut world = World::new();
    let mut c = chip::new_chip(
        &mut world,
        Position { x: 0, y: 0, z: 6 },
        Vector3f { x: 0.0, y: 0.0, z: 40.0 },
        IntVector { x: 14, y: 14, z: 5 },
    );
    let p = Position { x: 5, y: 5, z: 2 };
    let pin = chip::add_input_pin(&mut c, "In", p);
    let gate = raw_gate(&mut c, p);
    assert_ne!(pin, gate, "every brick gets a distinct id");
    assert_eq!(c.placed().len(), 2, "add_brick must record bounds, not just the brick");
    assert!(
        chip::finish(&mut world, c).is_err(),
        "a gate stacked on a pin must be caught"
    );
}

/// The whole point of tracking bounds: `plane_extent_for` sizes the chip's
/// plane from them, so a brick added via `add_brick` must widen the plane.
#[test]
fn add_brick_bounds_feed_the_plane_extent() {
    let mut world = World::new();
    let mut c = chip::new_chip(
        &mut world,
        Position { x: 0, y: 0, z: 6 },
        Vector3f { x: 0.0, y: 0.0, z: 40.0 },
        IntVector { x: 14, y: 14, z: 5 },
    );
    chip::add_input_pin(&mut c, "In", Position { x: 5, y: 5, z: 2 });
    let near = chip::plane_extent_for(c.placed());
    raw_gate(&mut c, Position { x: 5, y: 95, z: 2 });
    let far = chip::plane_extent_for(c.placed());
    // The extent is the content's HALF-span plus a one-cell margin (the
    // plane is centred on the content via `PlaneCenter`, not anchored at the
    // origin) -- see `chip::plane_bounds_for`'s doc comment.
    assert_eq!(
        near.y,
        GATE_HALF.y + CELL,
        "one pin spans y 0..10, a half-span of 5, plus a cell of margin"
    );
    assert_eq!(
        far.y,
        50 + CELL,
        "adding a gate at y=95 widens the span to y 0..100, a half-span of 50, plus a cell"
    );
    assert!(far.y > near.y, "a farther brick must widen the plane");
    chip::finish(&mut world, c).expect("spaced bricks are collision-free");
}

#[test]
fn clock_emits_four_gates_and_three_pins_without_collision() {
    let mut world = World::new();
    let mut c = chip::new_chip(
        &mut world,
        Position { x: 0, y: 0, z: 6 },
        Vector3f { x: 0.0, y: 0.0, z: 40.0 },
        IntVector { x: 60, y: 60, z: 20 },
    );
    // `Chip`'s brick list is private (see `chip.rs`); `placed()` is the
    // public read-only view kept in lockstep with it by `Chip::add_brick`,
    // so its length is an equivalent proxy for "how many bricks got added".
    let before = c.placed().len();
    let clock = heightmap::anim::clock::build_clock(
        &mut world,
        &mut c,
        15.0,
        90,
        Position { x: 5, y: 5, z: 2 },
    );
    assert_eq!(c.placed().len() - before, 9, "4 clock gates + 3 control pins + Rate in + Done out");
    assert_ne!(clock.pause_pin, clock.restart_pin);
    chip::finish(&mut world, c).expect("clock layout must be collision-free");

    world.register_used_components();
    let path = std::env::temp_dir().join(format!("h2b_clock_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    wire_integrity::assert_wires_valid(&path);

    // `Timer.Limit` must be free-running (0.0) in the actual encoded save,
    // not the schema's registered struct default (1.0) that a merely-unset
    // field falls back to (see `clock.rs`'s `build_clock`). Read the save
    // back and check the serialized value, the same way the game's loader
    // would, rather than trusting that omitting the field produces 0.
    //
    // The chip's inner grid gets its own *persistent* id assigned at write
    // time, distinct from the placeholder `Chip::entity_id` minted before
    // writing (`Brick::next_id()`) — so, like `wire_integrity`'s own grid
    // discovery, find it by reading entities back rather than reusing the
    // pre-write id.
    let db = brdb::Brz::open(&path).expect("reopen").into_reader();
    let mut chip_grid_id = None;
    for index in db.entity_chunk_index().expect("entity chunk index") {
        for e in db.entity_chunk(index).expect("entity chunk") {
            if e.is_microchip_grid() {
                chip_grid_id = e.id;
            }
        }
    }
    let chip_grid_id = chip_grid_id.expect("clock chip must publish exactly one microchip grid");

    let chunks = db.brick_chunk_index(chip_grid_id).expect("chip grid chunk index");
    let mut limit = None;
    for chunk in &chunks {
        let (_soa, structs) = db
            .component_chunk(chip_grid_id, chunk.index)
            .expect("component chunk");
        for s in &structs {
            if s.get_name() == "BrickComponentData_WireGraphPseudo_Timer" {
                if let Some(brdb::schema::BrdbValue::F64(v)) = s.get("Limit") {
                    limit = Some(*v);
                }
            }
        }
    }
    assert_eq!(
        limit,
        Some(0.0),
        "Timer.Limit must serialize as free-running (0.0), not the unset-field default"
    );

    let _ = std::fs::remove_file(&path);
}

// --- Task 6 spike: structural validation -----------------------------------
//
// `examples/spike_anim.rs` hand-builds a 2x2-pixel, 3-frame animation to
// de-risk four things source review cannot confirm: the chip I/O pin
// convention, 3D gate layering inside a chip, remote-wire fan-out across the
// chip boundary, and the 65535 array-length limit. Only a human pasting the
// result into the game can confirm those; this section covers what a test
// *can* confirm — that the graph the spike describes serializes into a save
// where every wire endpoint actually resolves.
//
// The graph is rebuilt here rather than shelling out to the example or
// reading its `main`, deliberately: a `#[test]` must not depend on another
// cargo target having already been run (that ordering isn't guaranteed by
// `cargo test`, and would break on a fresh checkout). The duplication is the
// price of that independence — keep this in sync with
// `examples/spike_anim.rs` if the spike's graph changes before Task 11
// replaces both with the real generator.

const SUBSTRING: &str = "BrickComponentType_WireGraph_Expr_String_Substring";
const MAKE_COLOR_HEX: &str = "BrickComponentType_WireGraph_Expr_MakeColorHex";
const ARRAY_VAR: &str = "BrickComponentType_WireGraphPseudo_ArrayVar";
const ARRAY_GET: &str = "BrickComponentType_WireGraph_Exec_ArrayVar_Get";
const CHANGE_DETECTOR: &str = "BrickComponentType_WireGraph_Expr_ChangeDetectorExec";
const PROP_CHANGER: &str = "Component_BrickPropertyChanger";

/// Build the exact graph `examples/spike_anim.rs` writes to
/// `./spike_anim.brz`, stopping just short of encoding it. Returns the
/// world plus how many display bricks it wired, so callers can check counts
/// without re-deriving them.
fn build_spike_world() -> (World, usize) {
    let frames = vec![
        "FF0000FF0000FF0000FF0000".to_string(), // all red
        "00FF0000FF0000FF0000FF00".to_string(), // all green
        "FF000000FF000000FFFFFFFF".to_string(), // red, green, blue, white
    ];
    let (w, h) = (2i32, 2i32);

    let mut world = World::new();
    world.meta.bundle.description = "Animation spike".to_string();

    let mut brick_ids = Vec::new();
    for row in 0..h {
        for col in 0..w {
            let (brick, id) = Brick {
                position: Position { x: col * 10, y: row * 10, z: 2 },
                ..Default::default()
            }
            .with_component(LiteralComponent::new(PROP_CHANGER))
            .with_id_split();
            world.add_brick(brick);
            brick_ids.push(id);
        }
    }

    let mut c = chip::new_chip(
        &mut world,
        Position { x: -20, y: 0, z: 2 }, // beside the screen, not stacked on it
        Vector3f { x: 0.0, y: 0.0, z: 40.0 },
        IntVector { x: 60, y: 60, z: 40 },
    );
    let service = |col: i32, row: i32| lattice_pos(col, row, 2, h, GATE_HALF);

    let clock = build_clock(&mut world, &mut c, 2.0, frames.len(), service(0, -2));
    let detector = gate(
        &mut c,
        "B_1x1_Gate_Expr_ChangeDetectorExec",
        CHANGE_DETECTOR,
        service(0, 0),
        vec![],
    );
    world.add_wire_connection(
        clock.frame_index.clone(),
        WirePort::new(detector, CHANGE_DETECTOR, "Input"),
    );

    let array = gate(
        &mut c,
        "B_1x1_Gate_Variable_Array",
        ARRAY_VAR,
        service(0, 1),
        vec![(
            "Value",
            Box::new(WireArrayVariant::StringArray(frames)) as Box<dyn AsBrdbValue>,
        )],
    );
    let get = gate(&mut c, "B_1x1_Gate_Exec_ArrayVar_Get", ARRAY_GET, service(1, 1), vec![]);
    world.add_wire_connection(
        WirePort::new(array, ARRAY_VAR, "ArrayVarRef"),
        WirePort::new(get, ARRAY_GET, "ArrayVarRef"),
    );
    world.add_wire_connection(clock.frame_index.clone(), WirePort::new(get, ARRAY_GET, "Index"));
    world.add_wire_connection(
        WirePort::new(detector, CHANGE_DETECTOR, "OnChanged"),
        WirePort::new(get, ARRAY_GET, "Exec"),
    );

    for (i, &brick_id) in brick_ids.iter().enumerate() {
        let (col, row) = (i as i32 % w, i as i32 / w);
        let sub = gate(
            &mut c,
            "B_1x1_Gate_Expr_String_Substring",
            SUBSTRING,
            lattice_pos(col, row, 1, h, GATE_HALF),
            vec![
                // Substring.Start/.Length are plain `i64` struct fields, not
                // the `WireVariant` tagged union `Multiply`/`ModuloFloored`
                // use (`BrickComponentData_WireGraph_Expr_String_Substring`
                // in brdb's `component_db.rs` registers bare `0i64`
                // defaults) — a `WireVariant::Int` here fails to encode with
                // `UnimplementedCast("i64", "WireVariant")`. This is the one
                // deviation from the task brief's inline listing, found by
                // actually running the spike.
                ("Start", Box::new(i as i64 * 6) as Box<dyn AsBrdbValue>),
                ("Length", Box::new(6i64) as Box<dyn AsBrdbValue>),
            ],
        );
        let mk = gate(
            &mut c,
            "B_1x1_Gate_Expr_MakeColorHex",
            MAKE_COLOR_HEX,
            lattice_pos(col, row, 0, h, GATE_HALF),
            vec![],
        );
        world.add_wire_connection(
            WirePort::new(get, ARRAY_GET, "Value"),
            WirePort::new(sub, SUBSTRING, "Input"),
        );
        world.add_wire_connection(
            WirePort::new(sub, SUBSTRING, "Output"),
            WirePort::new(mk, MAKE_COLOR_HEX, "Hex"),
        );
        world.add_wire_connection(
            WirePort::new(mk, MAKE_COLOR_HEX, "Output"),
            WirePort::new(brick_id, PROP_CHANGER, "Color"),
        );
    }

    chip::finish(&mut world, c).expect("spike layout must be collision-free");
    world.register_used_components();
    (world, brick_ids.len())
}

/// Sum of `num_bricks`/`num_wires` across every chunk of every grid
/// (main grid `1` plus every brick-grid/microchip-grid entity), the same
/// tally `brdb`'s own `count_wires` example produces. Used to pin the
/// spike's exact brick and wire counts so a human counting gates/wires
/// in-game (per the task brief's step 4) has a number to check against.
fn count_bricks_and_wires(path: &std::path::Path) -> (u64, u64) {
    let db = brdb::Brz::open(path).expect("reopen").into_reader();

    let mut grid_ids = vec![1usize];
    for index in db.entity_chunk_index().expect("entity chunk index") {
        for e in db.entity_chunk(index).expect("entity chunk") {
            if (e.is_brick_grid() || e.is_microchip_grid())
                && let Some(id) = e.id
            {
                grid_ids.push(id);
            }
        }
    }

    let (mut bricks, mut wires) = (0u64, 0u64);
    for gid in grid_ids {
        let Ok(chunks) = db.brick_chunk_index(gid) else {
            continue;
        };
        for chunk in &chunks {
            bricks += chunk.num_bricks as u64;
            wires += chunk.num_wires as u64;
        }
    }
    (bricks, wires)
}

/// Rebuilds the spike's graph and validates it structurally: every wire
/// endpoint resolves to a brick that actually carries the referenced
/// component (`wire_integrity::assert_wires_valid`), and the brick/wire
/// counts match what hand-tracing the graph predicts. This is the permanent
/// regression test for Task 6 — see the module-level comment above for why
/// it rebuilds the graph instead of reading the example's output file.
#[test]
fn spike_anim_graph_is_structurally_valid() {
    let (world, pixel_count) = build_spike_world();
    assert_eq!(pixel_count, 4, "2x2 screen must wire exactly 4 display bricks");

    let path = std::env::temp_dir().join(format!("h2b_spike_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");

    let (bricks, wires) = count_bricks_and_wires(&path);
    wire_integrity::assert_wires_valid(&path);

    // 4 display + 1 chip shell (main grid) + 20 inside the chip (4 clock gates +
    // 3 control pins + Rate in + Done out + detector + array + get + 4 pixels x
    // (Substring + MakeColorHex)).
    assert_eq!(bricks, 25, "brick count must match hand-traced total");
    // 3 clock-internal + 3 control-pin + 1 rate->mul + 1 timer->done + 1 detector +
    // 1 array->get + 1 clock->index + 1 detector->exec + 4 pixels x 3.
    assert_eq!(wires, 24, "wire count must match hand-traced total");

    let _ = std::fs::remove_file(&path);
}

/// Validates the literal artifact `cargo run --example spike_anim` writes —
/// the exact bytes a human pastes into the game — rather than a freshly
/// rebuilt copy. Ignored by default because it depends on that example
/// having already been run in this checkout (an ordering `cargo test` does
/// not guarantee); run explicitly after `cargo run --example spike_anim`:
///
/// ```text
/// cargo run --example spike_anim
/// cargo test --test anim_world -- --ignored spike_anim_output_file_is_valid
/// ```
#[test]
#[ignore = "depends on `cargo run --example spike_anim` having already written ./spike_anim.brz"]
fn spike_anim_output_file_is_valid() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("spike_anim.brz");
    assert!(
        path.exists(),
        "run `cargo run --example spike_anim` first to produce {}",
        path.display()
    );
    wire_integrity::assert_wires_valid(&path);
}

// --- Task 11: the real renderer --------------------------------------------

use brdb::assets::materials::{GLOW, PLASTIC};
use heightmap::anim::bricks::{
    AnimOptions, BRANCH, COMPARE_GE, DisplayBrickStyle, SELECT, build_brick_world,
};
use heightmap::anim::pack::PIXELS_PER_CHUNK;
use heightmap::progress::NoProgress;
use heightmap::video::Clip;
use heightmap::video::stream::FrameSource;
use image::{Rgba, RgbaImage};

fn tiny_clip(w: u32, h: u32, n: usize) -> Clip {
    let frames = (0..n)
        .map(|f| RgbaImage::from_pixel(w, h, Rgba([(f * 60) as u8, 10, 20, 255])))
        .collect();
    Clip { width: w, height: h, fps: 10.0, frames }
}

#[test]
fn brick_world_has_two_gates_per_pixel_plus_overhead() {
    let clip = tiny_clip(4, 3, 5);
    let world = build_brick_world(&clip, &AnimOptions::default(), &mut NoProgress).unwrap();
    // 12 display bricks + 1 microchip shell on the main grid
    assert_eq!(world.bricks.len(), 13);
    // inner grid: 2*12 pixel gates + 2 chunk gates + 1 detector + 4 clock
    //             + 5 clock pins (Pause, Restart, Resume, Rate, Done)
    let inner = &world.grids[0].1;
    assert_eq!(inner.len(), 24 + 2 + 1 + 4 + 5);
}

#[test]
fn brick_world_writes_and_every_wire_resolves() {
    let clip = tiny_clip(4, 3, 5);
    let world = build_brick_world(&clip, &AnimOptions::default(), &mut NoProgress).unwrap();
    let path = std::env::temp_dir().join(format!("h2b_anim_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    wire_integrity::assert_wires_valid(&path);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn fully_transparent_pixels_emit_no_brick() {
    let mut img = RgbaImage::from_pixel(2, 1, Rgba([255, 0, 0, 255]));
    img.put_pixel(1, 0, Rgba([0, 0, 0, 0]));
    let clip = Clip { width: 2, height: 1, fps: 10.0, frames: vec![img] };
    let world = build_brick_world(&clip, &AnimOptions::default(), &mut NoProgress).unwrap();
    assert_eq!(world.bricks.len(), 2, "1 display brick + 1 chip shell");
}

/// The user-facing default (per the brief's in-game feedback): a flush-tiled
/// screen of micro bricks at the smallest legal extent (1 -- a 2-unit-wide
/// brick) must build without `chip::finish`'s main-grid overlap error, and
/// every display brick's own asset/size must actually be the micro brick at
/// that extent, not just whatever `Brick::default()` used to hand back.
#[test]
fn micro_default_pixels_tile_flush_with_no_overlap_at_the_smallest_extent() {
    let clip = tiny_clip(4, 3, 2);
    let world = build_brick_world(&clip, &AnimOptions::default(), &mut NoProgress)
        .expect("flush-tiled micro pixels at the smallest extent must not overlap each other \
                 or the chip shell");

    let origin = world
        .bricks
        .iter()
        .find(|b| b.position == Position { x: 0, y: 0, z: 1 })
        .expect("column 0, row 0 must have a display brick (z = its own half-height, so it rests on z=0)");
    assert_eq!(
        origin.asset,
        BrickType::Procedural {
            asset: PB_DEFAULT_MICRO_BRICK,
            size: BrickSize { x: 1, y: 1, z: 1 },
        },
        "the default display brick must be a half-extent-1 micro brick"
    );

    // Pitch is `2 * extent = 2`, so the immediate neighbour on each axis
    // must sit exactly 2 units over -- proof the flush-tiling pitch actually
    // reached the renderer, not just that *some* legal layout was built.
    let right = world.bricks.iter().find(|b| b.position == Position { x: 0, y: 2, z: 1 });
    assert!(right.is_some(), "the next column over must sit at a 2-unit pitch");
}

/// `--glow` must reach the display bricks as GLOW at intensity 0 -- the
/// *dimmest* glow rather than "off" -- and must leave the chip's own gate
/// bricks alone.
///
/// Both halves matter. The flag existed on the CLI long before the animation
/// renderer did and only ever reached the heightmap path, so "the option
/// parses" proves nothing about the save; and intensity is easy to get
/// backwards, since 0 reads as off everywhere except here.
#[test]
fn glow_sets_the_display_bricks_to_glow_at_intensity_zero() {
    let clip = tiny_clip(3, 2, 2);

    let plain = build_brick_world(&clip, &AnimOptions::default(), &mut NoProgress)
        .expect("default must build");
    let lit = build_brick_world(
        &clip,
        &AnimOptions { glow: true, ..AnimOptions::default() },
        &mut NoProgress,
    )
    .expect("glow must build");

    // Display bricks rest at z = their own half-height; the chip shell and
    // its gates sit elsewhere, so this picks out the screen alone.
    let screen = |w: &brdb::World| -> Vec<(brdb::BString, u8)> {
        w.bricks
            .iter()
            .filter(|b| b.position.z == 1 && b.position.x >= 0)
            .map(|b| (b.material.clone(), b.material_intensity))
            .collect()
    };

    let lit_screen = screen(&lit);
    assert_eq!(lit_screen.len(), 6, "every pixel of a 3x2 clip must emit a display brick");
    for (material, intensity) in &lit_screen {
        assert_eq!(*material, GLOW, "--glow must set the GLOW material");
        assert_eq!(*intensity, 0, "glow intensity 0 is the dimmest setting, not 'off'");
    }

    for (material, intensity) in screen(&plain) {
        assert_eq!(material, PLASTIC, "without --glow the screen stays plastic");
        assert_eq!(intensity, 5, "the non-glow default intensity is 5, matching the heightmap path");
    }

    // The gates live on the chip's inner grid, so a screen-wide material
    // change must not have touched the main-grid brick count either way.
    assert_eq!(
        plain.bricks.len(),
        lit.bricks.len(),
        "glow is a material change only -- it must not add or drop bricks"
    );
}

/// The "normal" mode: smooth-tile display bricks must actually carry the
/// smooth-tile asset and be 4 units tall (half-extent z=2), while x/y follow
/// `5*pixel_extent` -- normal bricks are 5-unit-footprint, unlike micro
/// bricks (this crate's long-standing convention, see `src/main.rs`'s
/// `if micro { 1 } else { 5 }`). Read back from the real render, not just the
/// `DisplayBrickStyle::brick_type` unit tests in `anim::bricks`.
#[test]
fn smooth_tile_style_display_bricks_are_4_units_tall_and_follow_5x_the_extent() {
    let clip = tiny_clip(2, 2, 2);
    let opts = AnimOptions {
        brick_style: DisplayBrickStyle::SmoothTile,
        pixel_extent: 3,
        ..AnimOptions::default()
    };
    let world =
        build_brick_world(&clip, &opts, &mut NoProgress)
            .expect("a smooth-tile screen must build without overlap");

    let origin = world
        .bricks
        .iter()
        .find(|b| b.position == Position { x: 0, y: 0, z: 2 })
        .expect("column 0, row 0 must have a display brick");
    match &origin.asset {
        BrickType::Procedural { asset, size } => {
            assert_eq!(*asset, PB_DEFAULT_SMOOTH_TILE);
            assert_eq!(size.x, 15, "x must follow 5x pixel_extent (5*3=15)");
            assert_eq!(size.y, 15, "y must follow 5x pixel_extent (5*3=15)");
            assert_eq!(size.z, 2, "smooth tile must stay 4 units tall (half-extent 2)");
        }
        other => panic!("expected a Procedural brick, got {other:?}"),
    }
}

/// Flush tiling for the smooth-tile style specifically, mirroring
/// `micro_default_pixels_tile_flush_with_no_overlap_at_the_smallest_extent`
/// below but at the tile's own (5x-scaled) pitch: at `pixel_extent=1` a tile's
/// real footprint is half-extent 5 (10 units wide), so the next column over
/// must sit at a 10-unit pitch, not the 2-unit pitch a micro brick would use.
/// This is exactly the regression this crate shipped once -- treating the
/// pre-scale `pixel_extent` as the pitch for every style alike -- which
/// produced a 1-unit-footprint "smooth tile" the game silently dropped.
#[test]
fn smooth_tile_pixels_tile_flush_at_the_10_unit_pitch_at_the_smallest_extent() {
    let clip = tiny_clip(4, 3, 2);
    let opts = AnimOptions {
        brick_style: DisplayBrickStyle::SmoothTile,
        pixel_extent: 1,
        ..AnimOptions::default()
    };
    let world = build_brick_world(&clip, &opts, &mut NoProgress)
        .expect("flush-tiled smooth-tile pixels at the smallest extent must not overlap");

    let origin = world
        .bricks
        .iter()
        .find(|b| b.position == Position { x: 0, y: 0, z: 2 })
        .expect("column 0, row 0 must have a display brick");
    assert_eq!(
        origin.asset,
        BrickType::Procedural {
            asset: PB_DEFAULT_SMOOTH_TILE,
            size: BrickSize { x: 5, y: 5, z: 2 },
        },
        "the smooth-tile display brick must be a half-extent-5 (10-unit-wide) tile"
    );

    // Pitch is `2 * footprint = 10`, so the immediate neighbour on each axis
    // must sit exactly 10 units over -- proof the flush-tiling pitch actually
    // followed the tile's real (5x-scaled) footprint, not the raw extent.
    let right = world.bricks.iter().find(|b| b.position == Position { x: 0, y: 10, z: 2 });
    assert!(right.is_some(), "the next column over must sit at a 10-unit pitch");
}

/// The chip shell must clear the screen (never overlap a display brick on
/// the main grid) at the smallest extent, for both display styles --
/// `chip::finish`'s main-grid overlap assertion is the safety net, but this
/// pins the shell-clearance arithmetic in `build_brick_world` itself so a
/// wrong offset there is caught here rather than only in-game (see the F2
/// regression this crate already shipped once: a shell overlapping a
/// display brick made the game silently drop one, yielding a dangling-wire
/// 3-brick L).
#[test]
fn the_chip_shell_clears_the_screen_at_the_smallest_extent_for_every_style() {
    for style in [DisplayBrickStyle::Micro, DisplayBrickStyle::SmoothTile] {
        let clip = tiny_clip(3, 3, 2);
        let opts = AnimOptions { brick_style: style, pixel_extent: 1, ..AnimOptions::default() };
        build_brick_world(&clip, &opts, &mut NoProgress)
            .unwrap_or_else(|e| panic!("{style:?} at the smallest extent must clear the screen: {e}"));
    }
}

/// F3 regression: `video::scale::FpsStream` legitimately ends with zero
/// frames emitted when `start_s >= end_time` (e.g. `--start` past a short
/// clip's end, or the GUI's Start slider dragged past it). Before this
/// guard, `build_brick_world` would still "succeed" on that empty clip --
/// no display bricks, and `clock::build_clock` inlining
/// `Modulo.InputB = 0.0` -- producing a save that opens fine and silently
/// divides by zero in-game on every tick. The guard lives in
/// `build_brick_world` itself so both the CLI and the GUI (which both call
/// it) get the same rejection instead of a broken file.
#[test]
fn a_zero_frame_clip_is_rejected_instead_of_building_a_broken_save() {
    let clip = Clip { width: 4, height: 3, fps: 10.0, frames: Vec::new() };
    let result = build_brick_world(&clip, &AnimOptions::default(), &mut NoProgress);
    assert!(
        result.is_err(),
        "an empty clip must not produce a \"successful\" save with an in-game modulo-by-zero"
    );
}

/// F1 regression: the published microchip entity's `PlaneExtent` must
/// contain every brick actually placed inside it, on every axis. Before this
/// fix, `build_brick_world` computed the extent inline with a `* CELL / 2`
/// halving -- exactly the bug `chip::plane_extent_for` exists to prevent --
/// and `new_chip`'s caller-supplied extent was never revisited, so it shipped
/// unchanged. That clipped the clock's control pins and most pixel gates on
/// anything but a tiny screen: a save that opens fine and does nothing,
/// because the clipped bricks' wires dangle.
///
/// F2 regression (a since-reverted plane-centering fix): a version of this
/// renderer once translated every inner brick so its bounding box straddled
/// the origin before recomputing `PlaneExtent`, on the theory that
/// `layout::lattice_pos`'s positive-octant-only placement was wasting half of
/// `PlaneExtent`'s span (`PlaneCenter` is always `(0,0,0)`). That shipped,
/// several display gates went missing in-game, and a follow-up margin fix
/// (see F4 below) didn't bring them back -- which pointed at the centering
/// itself (specifically, the negative inner-grid coordinates it produces) as
/// the cause, so it was removed; see `chip::finish`'s doc comment for the
/// full account. This test therefore only checks CONTAINMENT with strict
/// clearance (see F4), not tightness -- an extent roughly 2x the content's
/// half-span is the current, intentional, cosmetic-only state.
///
/// F4 regression: the (now-reverted) centering fix made the recomputed extent
/// authoritative and exactly as large as the content's own span, with no
/// margin at all -- so every edge gate's outer face landed *exactly* on the
/// plane boundary and got clipped in-game (the save file itself is fine; this
/// is purely an in-game placement interaction, same as F1 above). The margin
/// this added to `plane_extent_for` survived the centering revert (it's
/// harmless and defensible regardless of centering). So this test asserts
/// STRICT clearance -- `|center| + half < extent`, not `<=` -- on every axis
/// for every placed brick: a brick sitting exactly on the boundary must fail
/// this test, not merely pass a "contains" check that a boundary-touching
/// brick would also satisfy.
///
/// This reads the *actual* extent `chip::finish` published into a real
/// `.brz` -- not `plane_extent_for` called a second time on the same inputs,
/// which would prove nothing about whether `build_brick_world` ever used the
/// recomputed value -- and checks it against every real inner brick's
/// reconstructed position.
#[test]
fn chip_plane_extent_contains_every_brick_in_a_real_render() {
    // Large enough to spread the clock's control pins and several columns of
    // pixel/service gates across multiple lattice stages -- exactly the
    // shape the old `(h+2)*CELL/2+5`-style halving underestimated.
    let clip = tiny_clip(6, 5, 3);
    let world = build_brick_world(&clip, &AnimOptions::default(), &mut NoProgress).unwrap();

    // Reconstruct every inner brick's original (pre-`add_brick_grid`-shift)
    // center: `World::add_brick_grid` subtracts `Position::CHUNK_HALF` from
    // every brick before storing it in `world.grids`, so adding it back
    // undoes exactly that shift. Every brick this renderer places inside a
    // chip -- gate or pin -- uses the uniform `GATE_HALF` (see
    // `chip::gate`/`chip::add_pin`), so no per-asset bounds lookup is needed.
    let inner = &world.grids[0].1;
    assert!(!inner.is_empty(), "test assumes at least one inner brick");
    let placed_centers: Vec<Position> =
        inner.iter().map(|b| b.position + Position::CHUNK_HALF).collect();

    let path = std::env::temp_dir().join(format!("h2b_extent_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");

    let db = brdb::Brz::open(&path).expect("reopen").into_reader();
    let mut extent: Option<IntVector> = None;
    let mut plane_center: Option<IntVector> = None;
    for index in db.entity_chunk_index().expect("entity chunk index") {
        let entities = db.entity_chunk(index).expect("entity chunk");
        let (_soa, structs) = db.entity_chunk_soa(index).expect("entity chunk soa");
        for (e, s) in entities.iter().zip(structs.iter()) {
            if e.is_microchip_grid() {
                let s = s
                    .as_ref()
                    .expect("microchip grid entity must carry PlaneExtent struct data");
                extent = Some(
                    s.prop("PlaneExtent")
                        .expect("microchip grid entity must have a PlaneExtent field")
                        .try_into()
                        .expect("PlaneExtent must decode as an IntVector"),
                );
                plane_center = Some(
                    s.prop("PlaneCenter")
                        .expect("microchip grid entity must have a PlaneCenter field")
                        .try_into()
                        .expect("PlaneCenter must decode as an IntVector"),
                );
            }
        }
    }
    let extent = extent.expect("renderer must publish exactly one microchip grid");
    let pc = plane_center.expect("renderer must publish a PlaneCenter");

    for center in &placed_centers {
        // Strict, not `<=`: a brick whose outer face lands exactly on the
        // plane boundary is the regression where an edge gate was clipped
        // in-game despite the save file "containing" it -- so touching the
        // boundary must fail this test, not pass it.
        //
        // Measured against the published `PlaneCenter`, which is the
        // content midpoint: the plane spans `centre +- extent` on every
        // axis. Any pull-back lives on `entity.location`, not in here.
        // x and y only. z is deliberately NOT contained: the plane is a flat
        // surface pinned at z=0 that the gates sit ON TOP of, so stretching
        // it to swallow them would be the bug, not the invariant (see
        // `chip::plane_bounds_for`). The z relationship worth holding is
        // checked separately, below.
        for (axis, lo, hi, pos, half) in [
            ("x", pc.x - extent.x, pc.x + extent.x, center.x, GATE_HALF.x),
            ("y", pc.y - extent.y, pc.y + extent.y, center.y, GATE_HALF.y),
        ] {
            assert!(
                lo < pos - half && pos + half < hi,
                "brick at {center:?} spans {axis} {}..{} without strictly clearing the \
                 published plane {axis} {lo}..{hi}",
                pos - half,
                pos + half,
            );
        }
    }

    // Every gate must sit ON TOP of the plane, not inside it: the plane is
    // pinned to a flat surface and `layout::STAGE_BASE_Z` lifts the lattice
    // clear of its top face. Before z was pinned, the extent grew to swallow
    // the gates (rendering them buried in the chip) and the centre tracked
    // them, so raising the lattice moved nothing on screen.
    for center in &placed_centers {
        assert!(
            center.z - GATE_HALF.z > pc.z + extent.z,
            "gate at z {} must clear the plane top face {}",
            center.z - GATE_HALF.z,
            pc.z + extent.z
        );
    }

    // The plane must HUG the content, not be anchored at the origin and
    // stretched to reach it. An origin-anchored plane over content spanning
    // 0..N needs extent N (twice the half-span) and leaves the content in
    // one quadrant -- exactly what was seen in-game. Allow one cell of
    // margin per side plus a rounding unit.
    for (axis, e, lo, hi) in [
        ("x", extent.x, placed_centers.iter().map(|p| p.x).min().unwrap(), placed_centers.iter().map(|p| p.x).max().unwrap()),
        ("y", extent.y, placed_centers.iter().map(|p| p.y).min().unwrap(), placed_centers.iter().map(|p| p.y).max().unwrap()),
    ] {
        let half_span = (hi - lo) / 2 + GATE_HALF.x;
        assert!(
            e <= half_span + CELL + 2,
            "plane {axis} extent {e} is far larger than the content half-span {half_span} \
             -- the plane is not centred on the content"
        );
    }

    // No tightness assertion here (deliberately): this renderer no longer
    // centers inner-grid content on the origin (see `chip::finish`'s doc
    // comment for why -- display gates went missing in-game and centering,
    // which lands inner bricks at negative grid coordinates, was the only
    // remaining difference from the last verified-working state). With
    // content left non-negative and origin-anchored, `PlaneExtent` is
    // naturally ~2x the content's half-span on every axis (`PlaneCenter` is
    // hardcoded `(0,0,0)`, so a plane containing a `0..N`-spanning lattice
    // must reach `N`, not `N/2`). That bloat is cosmetic and intentionally
    // accepted for now -- containment with strict clearance, asserted above,
    // is the invariant that actually matters.

    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_screen_wider_than_one_chunk_gets_more_arrays() {
    let n = PIXELS_PER_CHUNK + 10;
    let clip = tiny_clip(n as u32, 1, 2);
    let world = build_brick_world(&clip, &AnimOptions::default(), &mut NoProgress).unwrap();
    let inner = &world.grids[0].1;
    // 2 chunks => 2 ArrayVar + 2 Get; 4 clock gates + 1 detector + 5 clock pins
    assert_eq!(inner.len(), 2 * n + 4 + 1 + 4 + 5);
}

// --- Task 10 follow-up: cross-chunk `Start` correctness ---------------------
//
// Task 10's review flagged a gap: no test exercised `slice_of`/content
// correctness in a SECOND chunk (`first_pixel > 0`) with varying per-pixel
// colours — exactly where a `first_pixel` vs `pixel_in_chunk` (global vs
// chunk-local index) confusion in `build_brick_world`'s Substring `Start`
// math would hide. `a_screen_wider_than_one_chunk_gets_more_arrays` above
// only counts gates; it never reads a single `Start` value back.
//
// This test builds a real multi-chunk screen, writes it, reads the actual
// persisted `Substring.Start` values and `ArrayVar` string contents back out
// of the save (not from anything `build_brick_world` handed back in memory),
// and checks both:
//   1. that the chunk boundary resets `Start` back to 0 rather than
//      continuing to count the *global* pixel index, and
//   2. that slicing the real chunk-2 array data at that offset yields the
//      correct 6 hex characters.
use heightmap::anim::pack::{self, HEX_STRIDE};

/// Every pixel gets a color that encodes its own global column index across
/// R:G (16 bits — plenty of headroom for `PIXELS_PER_CHUNK + 10` columns) and
/// the frame number in B, so every (pixel, frame) pair decodes to a distinct,
/// independently-reconstructable hex string without relying on `pack` at all.
fn multi_chunk_clip(width: u32, frames: usize) -> Clip {
    let frames = (0..frames)
        .map(|f| {
            RgbaImage::from_fn(width, 1, |x, _y| {
                Rgba([(x >> 8) as u8, (x & 0xFF) as u8, f as u8, 255])
            })
        })
        .collect();
    Clip { width, height: 1, fps: 10.0, frames }
}

#[test]
fn a_pixel_in_the_second_pack_chunk_gets_a_chunk_relative_substring_start() {
    let n = (PIXELS_PER_CHUNK + 10) as u32;
    let clip = multi_chunk_clip(n, 2);
    let opts = AnimOptions::default();

    // Ground truth for what the chunks *should* contain, independent of
    // `build_brick_world` — `pack` already has its own Task 10 coverage.
    let chunks = pack::pack(&clip, opts.alpha_threshold).expect("pack");
    assert_eq!(chunks.len(), 2, "test assumes exactly two pack chunks");
    assert_eq!(
        chunks[1].first_pixel, PIXELS_PER_CHUNK,
        "chunk 2 must start exactly where chunk 1 ends"
    );

    let world = build_brick_world(&clip, &opts, &mut NoProgress).unwrap();
    let path = std::env::temp_dir().join(format!("h2b_anim_xchunk_{}.brz", std::process::id()));
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
    let chip_grid_id = chip_grid_id.expect("renderer must publish exactly one microchip grid");

    // Pull every Substring's `Start` and every ArrayVar's `Value` straight
    // out of the encoded save — no brick position or wire-graph decoding
    // needed for either.
    let mut starts: Vec<i64> = Vec::new();
    let mut arrays: Vec<Vec<String>> = Vec::new();
    for chunk in db.brick_chunk_index(chip_grid_id).expect("chip grid chunk index") {
        let (_soa, structs) = db
            .component_chunk(chip_grid_id, chunk.index)
            .expect("component chunk");
        for s in &structs {
            match s.get_name() {
                "BrickComponentData_WireGraph_Expr_String_Substring" => {
                    if let Some(brdb::schema::BrdbValue::I64(start)) = s.get("Start") {
                        starts.push(*start);
                    }
                    if let Some(brdb::schema::BrdbValue::I64(len)) = s.get("Length") {
                        assert_eq!(
                            *len, HEX_STRIDE as i64,
                            "every Substring must read exactly HEX_STRIDE chars"
                        );
                    }
                }
                "BrickComponentData_WireGraphPseudo_ArrayVar" => {
                    if let Some(value) = s.get("Value") {
                        let variant: WireArrayVariant =
                            value.try_into().expect("ArrayVar Value must decode");
                        if let WireArrayVariant::StringArray(v) = variant {
                            arrays.push(v);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // The chunk boundary must reset the per-gate `Start` back to 0, not keep
    // counting the *global* pixel index. Chunk 1 (1666 pixels) alone already
    // produces a Substring with `Start = 3 * HEX_STRIDE = 18` for its own
    // local pixel 3. If chunk 2's gates wrongly computed
    // `(chunk.first_pixel + local) * HEX_STRIDE` instead of
    // `local * HEX_STRIDE`, chunk 2's first ten pixels would start at
    // `1666 * 6 = 9996` and up — never producing 18 again — so `Start == 18`
    // would appear once across the whole save, not twice.
    let local3_start = (3 * HEX_STRIDE) as i64;
    let count = starts.iter().filter(|&&s| s == local3_start).count();
    assert_eq!(
        count, 2,
        "expected exactly two Substring gates with Start={local3_start} (local pixel 3 of \
         chunk 1 and local pixel 3 of chunk 2 each reading their own chunk's byte offset); \
         finding only one means chunk 2 is computing Start from the global pixel index \
         instead of resetting per chunk"
    );

    // Identify chunk 2's own ArrayVar by its actual persisted content — no
    // position or wire-graph decoding needed: it's the one array whose
    // per-frame strings equal `pack`'s own chunk-2 output.
    let chunk2_array = arrays
        .iter()
        .find(|a| **a == chunks[1].frames)
        .expect("chunk 2's ArrayVar content must appear in the saved world");

    // Slicing that array at the just-verified offset must yield exactly the
    // 6 hex characters `pack` assigned to local pixel 3 of chunk 2 — the same
    // offset the corresponding Substring gate is wired to read.
    let expected = pack::slice_of(&chunks[1], 0, 3);
    let start = local3_start as usize;
    let actual = &chunk2_array[0][start..start + HEX_STRIDE];
    assert_eq!(actual, expected, "chunk 2 pixel 3 must read back its own encoded color");

    let _ = std::fs::remove_file(&path);
}

/// Negative inner-grid coordinates break bricks in-game: a chip built that way
/// loses those gates entirely, while gates at positive coordinates survive.
/// It bit twice — once via a content-centring pass that pushed the low-x
/// pixel gates negative, and once latently for short screens, where
/// `lattice_pos`'s `x = (h-1-row)*CELL + half.x` sends a positive service row
/// negative (at `h = 1`, `row = 1` lands at `x = -5`).
///
/// A 1-row screen is the tightest case, so it is the one worth pinning.
#[test]
fn no_inner_brick_lands_at_a_negative_coordinate() {
    for (w, h) in [(1u32, 1u32), (4, 1), (1, 4), (2, 2), (7, 3)] {
        let clip = tiny_clip(w, h, 3);
        let world = build_brick_world(&clip, &AnimOptions::default(), &mut NoProgress)
            .unwrap_or_else(|e| panic!("{w}x{h} must build: {e}"));
        let inner = &world.grids[0].1;
        for b in inner {
            // `add_brick_grid` stores inner bricks shifted by -CHUNK_HALF;
            // the invariant is about the grid-local coordinates the lattice
            // actually emits, so undo that shift first.
            let center = b.position + Position::CHUNK_HALF;
            // Check the centre, not centre-minus-half: different gates carry
            // different halves (flat `GATE_HALF` vs the rotated pixel gates),
            // and `local_bounds` guesses `(5,5,6)` for every basic brick, so
            // neither is a usable per-brick half here. A non-negative centre
            // is the invariant that actually distinguishes the broken builds.
            let half = IntVector { x: 0, y: 0, z: 0 };
            assert!(
                center.x - half.x >= 0 && center.y - half.y >= 0 && center.z - half.z >= 0,
                "{w}x{h}: inner brick centred at {center:?} reaches ({},{},{}), which is \
                 negative — negative inner-grid coordinates break bricks in-game",
                center.x - half.x,
                center.y - half.y,
                center.z - half.z
            );
        }
    }
}

// --- Task 2: multi-bank wiring ----------------------------------------------

use heightmap::anim::pack::BANK_FRAMES;

/// Spillover must be invisible to clips that do not need it. Every render
/// made so far takes this path, so a regression here breaks everything that
/// currently works.
#[test]
fn a_single_bank_render_emits_no_spillover_gates() {
    let clip = tiny_clip(4, 3, 5);
    let world = build_brick_world(&clip, &AnimOptions::default(), &mut NoProgress).expect("build");
    // `BrdbComponent::component_type() -> Option<BString>` is the accessor;
    // there is no `name()`. Gates live inside the chip's own inner grid
    // (`world.grids[0].1`), not `world.bricks` -- that vec is the MAIN grid
    // (display bricks + the chip shell) only.
    let inner = &world.grids[0].1;
    for class in ["CompareGreaterOrEqual", "Exec_Branch", "Expr_Select", "MathSubtract"] {
        assert!(
            !inner.iter().any(|b| b.components.iter().any(|c| {
                c.component_type().is_some_and(|t| t.to_string().contains(class))
            })),
            "a single-bank render must emit no {class} gate"
        );
    }
    assert_eq!(
        AnimOptions::default().bank_size,
        BANK_FRAMES,
        "the default bank size is the real array limit"
    );
}

/// With a lowered bank size a small clip produces a genuine multi-bank graph,
/// so the cascade is testable without a 65536-frame clip.
#[test]
fn a_multi_bank_render_emits_one_array_per_chunk_per_bank() {
    // 5 frames at bank size 2 -> 3 banks; 4x3 px is a single chunk.
    let clip = tiny_clip(4, 3, 5);
    let opts = AnimOptions { bank_size: 2, ..AnimOptions::default() };
    let world = build_brick_world(&clip, &opts, &mut NoProgress).expect("build");

    // Gates live inside the chip's own inner grid (`world.grids[0].1`), not
    // `world.bricks` -- that vec is the MAIN grid (display bricks + the chip
    // shell) only.
    let inner = &world.grids[0].1;
    let count = |needle: &str| {
        inner.iter()
            .filter(|b| b.components.iter().any(|c| {
                c.component_type().is_some_and(|t| t.to_string().contains(needle))
            }))
            .count()
    };
    assert_eq!(count("ArrayVar") - count("ArrayVar_Get"), 3, "one array per bank");
    // N-1 of each, for N = 3 banks
    assert_eq!(count("CompareGreaterOrEqual"), 2, "one comparator per boundary");
    assert_eq!(count("Exec_Branch"), 2, "one branch per boundary");
    assert_eq!(count("MathSubtract"), 2, "one index subtract per extra bank");
    // one select per chunk per extra bank; 1 chunk x 2 boundaries
    assert_eq!(count("Expr_Select"), 2, "one select per chunk per boundary");
}

/// The property that makes the front-cascade design safe: branching per chunk
/// and rejoining would need two exec sources on one input, which nothing in
/// this codebase has tested.
#[test]
fn no_exec_input_ever_has_two_sources() {
    let clip = tiny_clip(40, 40, 7);
    let opts = AnimOptions { bank_size: 2, ..AnimOptions::default() };
    let world = build_brick_world(&clip, &opts, &mut NoProgress).expect("build");

    // `world.wires` is a `Vec<WireConnection>`; each has `.source`/`.target`
    // of type `WirePort { brick_id, component_type, port_name }`.
    let mut seen = std::collections::HashSet::new();
    for wire in &world.wires {
        let t = &wire.target;
        let port = t.port_name.to_string();
        if port.contains("Exec") && !port.contains("Out") {
            assert!(
                seen.insert((t.brick_id, port.clone())),
                "exec input {}.{port} has more than one source",
                t.brick_id
            );
        }
    }
}

/// Finding 1 (2026-07-26 spillover final review): the game's `Exec_Branch`
/// takes `ExecOutA` on a TRUTHY `bCond` and `ExecOutB` on a FALSY one -- the
/// opposite of what an earlier version of this renderer assumed. Because
/// `ArrayVar_Get` is an exec gate whose `Value` only refreshes when its own
/// `Exec` fires, wiring the branch backwards makes the bank that actually
/// EXECUTES the complement of the bank the Select cascade READS, so every
/// multi-bank render would show the component default for the whole clip --
/// yet every other test in this file, which only checks that a wire exists
/// and not which polarity it rides, stayed green either way. That is exactly
/// how the inversion survived five prior reviews.
///
/// This test pins the polarity directly against the wire graph: each
/// branch's falsy (`ExecOutB`) output must feed an `ArrayVar_Get.Exec` -- the
/// entry to its OWN bank's chain -- and its truthy (`ExecOutA`) output must
/// feed either the next branch's `Exec` (if one exists) or, for the last
/// branch, the final bank's `ArrayVar_Get.Exec` directly (there is no
/// further branch to descend into).
///
/// Both-directions check performed by hand while writing this test (per the
/// review brief): with the two `WirePort` labels in `bricks.rs`'s branch
/// cascade swapped back to `ExecOutB` = "keep descending" / `ExecOutA` =
/// "this bank", this test fails -- every non-last branch's `ExecOutB` then
/// targets the next branch's `Exec` (component type `BRANCH`, not
/// `ARRAY_GET`), tripping the first assertion in the loop. Re-applying the
/// fix makes it pass again.
#[test]
fn each_branchs_falsy_output_enters_its_own_bank_and_its_truthy_output_keeps_descending() {
    // 7 frames at bank size 2 -> 4 banks -> 3 boundaries -> 3 branches.
    // 4x3 px is a single chunk, so each bank's Get chain is exactly one gate
    // long and the branch's entry wire IS that gate's Exec wire.
    let clip = tiny_clip(4, 3, 7);
    let opts = AnimOptions { bank_size: 2, ..AnimOptions::default() };
    let world = build_brick_world(&clip, &opts, &mut NoProgress).expect("build");
    let inner = &world.grids[0].1;

    // Branches are emitted in `bi = 0, 1, ...` order and every brick id in
    // this chip is minted from one monotonic, whole-build counter
    // (`Brick::next_id`, see brdb's `wrapper/brick.rs`), so sorting the
    // branch bricks by id recovers that same bi order.
    let mut branch_ids: Vec<usize> = inner
        .iter()
        .filter(|b| {
            b.components
                .iter()
                .any(|c| c.component_type().is_some_and(|t| t.to_string() == BRANCH))
        })
        .map(|b| b.id.expect("every emitted brick has an id"))
        .collect();
    branch_ids.sort_unstable();
    assert_eq!(branch_ids.len(), 3, "4 banks must cascade through exactly 3 branches");

    // The one and only wire leaving `brick_id`'s `port` on its `Exec_Branch`
    // component -- there must be exactly one, since a branch output either
    // drives the next branch's `Exec` or a Get's `Exec`, never both.
    let target_from = |brick_id: usize, port: &str| -> WirePort {
        let mut hits = world.wires.iter().filter(|w| {
            w.source.brick_id == brick_id
                && w.source.component_type.to_string() == BRANCH
                && w.source.port_name.to_string() == port
        });
        let wire = hits
            .next()
            .unwrap_or_else(|| panic!("branch {brick_id}'s {port} must drive something"));
        assert!(hits.next().is_none(), "branch {brick_id}'s {port} must drive exactly one target");
        wire.target.clone()
    };

    for (i, &br) in branch_ids.iter().enumerate() {
        let falsy = target_from(br, "ExecOutB");
        assert_eq!(
            falsy.component_type.to_string(),
            ARRAY_GET,
            "branch {i}'s falsy (ExecOutB) output must enter its own bank's Get chain, not {:?}",
            falsy.component_type
        );
        assert_eq!(falsy.port_name.to_string(), "Exec");

        let truthy = target_from(br, "ExecOutA");
        if i + 1 < branch_ids.len() {
            assert_eq!(
                truthy.brick_id, branch_ids[i + 1],
                "branch {i}'s truthy (ExecOutA) output must keep descending into branch {}",
                i + 1
            );
            assert_eq!(truthy.component_type.to_string(), BRANCH);
            assert_eq!(truthy.port_name.to_string(), "Exec");
        } else {
            assert_eq!(
                truthy.component_type.to_string(),
                ARRAY_GET,
                "the last branch's truthy (ExecOutA) output must enter the final bank's Get \
                 chain directly, not {:?}",
                truthy.component_type
            );
            assert_eq!(truthy.port_name.to_string(), "Exec");
        }
    }
}

/// The select cascade must be well formed: each Select takes its condition
/// from a comparator, its B input from an array Get, and its A input from
/// either a Get (the first stage) or another Select (a later stage). A
/// cascade wired A-to-B backwards still builds and still produces a save --
/// it just shows the wrong bank.
#[test]
fn the_select_cascade_is_well_formed() {
    let clip = tiny_clip(4, 3, 7);
    let opts = AnimOptions { bank_size: 2, ..AnimOptions::default() };
    let world = build_brick_world(&clip, &opts, &mut NoProgress).expect("build");

    let source_of = |brick: usize, port: &str| -> Option<String> {
        world.wires.iter()
            .find(|w| w.target.brick_id == brick && w.target.port_name.to_string() == port)
            .map(|w| w.source.component_type.to_string())
    };
    let selects: Vec<usize> = world.wires.iter()
        .filter(|w| w.target.component_type.to_string() == SELECT)
        .map(|w| w.target.brick_id)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    assert!(!selects.is_empty(), "a 4-bank render must emit selects");

    for sel in selects {
        assert_eq!(
            source_of(sel, "bSelectB").as_deref(),
            Some(COMPARE_GE),
            "a select's condition must come from a boundary comparator"
        );
        assert_eq!(
            source_of(sel, "InputB").as_deref(),
            Some(ARRAY_GET),
            "a select's B input must be the later bank's array Get"
        );
        let a = source_of(sel, "InputA").expect("every select needs an A input");
        assert!(
            a == ARRAY_GET || a == SELECT,
            "a select's A input must be a Get or the previous select, got {a}"
        );
    }
}

/// None of the structural tests above actually encode a multi-bank world --
/// they only inspect the in-memory `World`. The inlined literals
/// `CompareGreaterOrEqual.InputB` and `MathSubtract.InputB` are exactly the
/// kind of thing that builds fine in memory and then fails at encode time
/// with `UnimplementedCast` if the wrong literal form was used (this
/// codebase has hit that twice before -- see `bricks.rs`'s Substring/
/// ModuloFloored comments). This is the multi-bank counterpart to
/// `brick_world_writes_and_every_wire_resolves`, which only ever exercised
/// the single-bank path.
#[test]
fn a_multi_bank_render_encodes_and_every_wire_resolves() {
    let clip = tiny_clip(4, 3, 5);
    let opts = AnimOptions { bank_size: 2, ..AnimOptions::default() };
    let world = build_brick_world(&clip, &opts, &mut NoProgress).expect("build");
    let path = std::env::temp_dir().join(format!("h2b_anim_multibank_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    wire_integrity::assert_wires_valid(&path);
    let _ = std::fs::remove_file(&path);
}

// --- Task 5: integration -- a real multi-bank save --------------------------
//
// The task brief for this step proposed `a_multi_bank_save_passes_wire_integrity`:
// build a multi-bank world, encode it, and run `wire_integrity::assert_wires_valid`
// on the result. That is exactly what `a_multi_bank_render_encodes_and_every_wire_resolves`
// above already does -- same clip shape (a `tiny_clip` at `bank_size: 2`, just
// different dimensions/frame count), same encode, same `assert_wires_valid`
// call, same cleanup. The only difference is a redundant explicit
// `world.register_used_components()` call, which is already the last step
// `build_brick_world` performs internally (see `bricks.rs`, "Must be last"),
// so it changes nothing observable. Adding the brief's version verbatim would
// be a near-duplicate test asserting the same property twice under different
// names, so it is intentionally not added here; the two tests below cover the
// parts of the brief that nothing else in this file exercises: that each
// bank's array holds only its own frame slice, and that the comparator
// boundaries encoded into the save are the real bank-size multiples.

/// Each bank's array must hold its own slice, and the slices must reassemble
/// into the clip. A bank that kept the whole frame list would play correctly
/// in bank 0 and repeat itself thereafter.
#[test]
fn each_banks_array_holds_only_its_own_frames() {
    let clip = tiny_clip(4, 3, 7);
    let opts = AnimOptions { bank_size: 3, ..AnimOptions::default() };
    let mut world = build_brick_world(&clip, &opts, &mut NoProgress).expect("build");
    world.register_used_components();
    let path = std::env::temp_dir().join(format!("h2b_bankarr_{}.brz", std::process::id()));
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
    let chip_grid_id = chip_grid_id.expect("a chip must publish a microchip grid");

    // `ArrayVar.Value` is a `WireGraphPrimMathVariant`-style tagged union, not
    // a bare `BrdbValue::Array` -- it decodes as `BrdbValue::Struct` wrapping a
    // `WireGraph*Array` struct (here `WireGraphStringArray`, holding the real
    // array under its own `Values` field). `WireArrayVariant`'s
    // `TryFrom<&BrdbValue>` already knows how to unwrap that; matching on the
    // exact data-struct name (rather than `Get`'s, which also contains
    // "ArrayVar" but carries no `Value` field) is what the sibling
    // multi-chunk test above (`a_pixel_in_the_second_pack_chunk_gets_a_chunk_relative_substring_start`)
    // already established as the working pattern.
    let mut lengths = Vec::new();
    for chunk in &db.brick_chunk_index(chip_grid_id).expect("chunk index") {
        let (_soa, structs) = db.component_chunk(chip_grid_id, chunk.index).expect("components");
        for s in &structs {
            if s.get_name() == "BrickComponentData_WireGraphPseudo_ArrayVar" {
                if let Some(value) = s.get("Value") {
                    let variant: WireArrayVariant =
                        value.try_into().expect("ArrayVar Value must decode");
                    if let WireArrayVariant::StringArray(v) = variant {
                        lengths.push(v.len());
                    }
                }
            }
        }
    }
    lengths.sort_unstable();
    assert_eq!(lengths, vec![1, 3, 3], "7 frames at bank size 3 -> 3 + 3 + 1");
    let _ = std::fs::remove_file(&path);
}

/// The boundary constants inlined on the comparators and subtracts must be
/// the actual multiples of the bank size. This is where an off-by-one lives,
/// and it is invisible until playback crosses a seam -- so read the values
/// back out of the encoded save rather than trusting the emitter.
#[test]
fn boundary_constants_are_the_real_multiples_of_the_bank_size() {
    let clip = tiny_clip(4, 3, 7);
    let opts = AnimOptions { bank_size: 3, ..AnimOptions::default() };
    let mut world = build_brick_world(&clip, &opts, &mut NoProgress).expect("build");
    world.register_used_components();
    let path = std::env::temp_dir().join(format!("h2b_bounds_{}.brz", std::process::id()));
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
    let chip_grid_id = chip_grid_id.expect("a chip must publish a microchip grid");

    // `CompareGreaterOrEqual` is the COMPONENT TYPE name, but `s.get_name()`
    // reports the underlying DATA STRUCT, and every `Compare*` comparator
    // (Greater, GreaterOrEqual, Less, LessOrEqual) shares one struct --
    // `BrickComponentData_WireGraph_Expr_MathCompare` -- per brdb's
    // `COMPONENT_TYPE_STRUCT_PAIRS`. This chip only ever emits
    // `CompareGreaterOrEqual` gates (see `bricks.rs`'s boundary cascade), so
    // matching the shared struct name is equivalent here and is what the
    // encoded save actually contains.
    let mut bounds = Vec::new();
    for chunk in &db.brick_chunk_index(chip_grid_id).expect("chunk index") {
        let (_soa, structs) = db.component_chunk(chip_grid_id, chunk.index).expect("components");
        for s in &structs {
            if s.get_name() == "BrickComponentData_WireGraph_Expr_MathCompare" {
                // The literal form depends on the field's declared type; accept
                // whichever the encoder produced and compare numerically.
                match s.get("InputB") {
                    Some(brdb::schema::BrdbValue::I64(v)) => bounds.push(*v as i64),
                    Some(brdb::schema::BrdbValue::F64(v)) => bounds.push(*v as i64),
                    other => panic!("unexpected InputB encoding: {other:?}"),
                }
            }
        }
    }
    bounds.sort_unstable();
    // 7 frames at bank size 3 -> 3 banks -> boundaries at 3 and 6
    assert_eq!(bounds, vec![3, 6], "comparators must sit on the real bank seams");
    let _ = std::fs::remove_file(&path);
}

// --- Task 5: render from a FrameSource --------------------------------------
//
// `build_brick_world` stops taking a `&Clip` and takes any `&dyn FrameSource`
// plus a `&mut dyn Progress`. Every call site above already proves a `Clip`
// (which implements `FrameSource`) still renders identically through the new
// signature -- these tests cover what's actually NEW: a non-`Clip` source, a
// driven progress reporter, and a mid-stream failure that must abort rather
// than truncate.

/// A `Clip` is a `FrameSource`, so every existing expectation must hold
/// through the new signature -- same bricks, same gates, same everything.
#[test]
fn rendering_from_a_source_matches_rendering_from_a_clip() {
    let clip = tiny_clip(4, 3, 5);
    let world =
        build_brick_world(&clip as &dyn FrameSource, &AnimOptions::default(), &mut NoProgress)
            .expect("build");
    // 12 pixels, all opaque -> 12 display bricks + the chip shell
    assert_eq!(world.bricks.len(), 13);
}

/// The reporter must actually be driven, with a real total when the source
/// knows its length. A progress bar that never ticks is worse than none.
#[test]
fn the_render_reports_progress() {
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

    let clip = tiny_clip(4, 3, 5);
    let mut rec = Rec::default();
    build_brick_world(&clip as &dyn FrameSource, &AnimOptions::default(), &mut rec)
        .expect("build");
    assert!(!rec.began.is_empty(), "the render must begin a phase");
    assert_eq!(rec.began[0].1, Some(5), "an in-memory clip knows its frame count");
    assert!(rec.ticks > 0, "the reporter must be ticked");
    assert_eq!(rec.finished, rec.began.len(), "every phase begun must be finished");
}

/// A stream that fails partway must abort the render, not write a save
/// silently missing its tail.
#[test]
fn a_failing_stream_aborts_the_render() {
    struct Failing;
    struct FailingStream(usize);
    impl heightmap::video::stream::FrameStream for FailingStream {
        fn next(&mut self) -> Result<Option<image::RgbaImage>, String> {
            self.0 += 1;
            if self.0 > 2 { Err("decode failed at frame 3".into()) }
            else { Ok(Some(image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 255])))) }
        }
    }
    impl FrameSource for Failing {
        fn info(&self) -> heightmap::video::stream::SourceInfo {
            heightmap::video::stream::SourceInfo {
                width: 2, height: 2, fps: 10.0, frame_count_hint: Some(10),
            }
        }
        fn open(&self) -> Result<Box<dyn heightmap::video::stream::FrameStream + '_>, String> {
            Ok(Box::new(FailingStream(0)))
        }
    }
    // `.expect_err`/`.unwrap_err` both require `T: Debug` on the `Ok` side --
    // `brdb::World` does not derive `Debug`, so the `Ok` variant is mapped
    // away first (deviation from the task brief, which called `.expect_err`
    // directly; that does not compile against this `World`).
    let err =
        build_brick_world(&Failing as &dyn FrameSource, &AnimOptions::default(), &mut NoProgress)
            .map(|_| ())
            .expect_err("a mid-stream failure must abort");
    assert!(err.contains("frame 3"), "the underlying error must surface: {err}");
}

/// Proof that the packer's row-major visibility vector (`idx = row * width +
/// col`) reaches the display-brick loop with the SAME index order `visible()`
/// used before it was deleted -- not transposed into column-major (`idx = col
/// * height + row`). A SQUARE clip cannot distinguish the two formulas (every
/// `(row, col)` pair has a symmetric partner with the same pair of numbers),
/// so this uses a 5-wide, 2-tall clip and marks exactly one pixel opaque, at
/// (col=2, row=1) -- chosen so the row-major index (7) and the naive
/// column-major index (5) actually differ, and so does the (col=3, row=1)
/// pixel a column-major reader would land on instead (index 7 under that
/// formula). A transposed read would either drop the display brick entirely
/// or place it at the wrong column; this asserts the exact position.
#[test]
fn visibility_indexing_follows_the_packers_row_major_layout_on_a_non_square_clip() {
    let (w, h) = (5u32, 2u32);
    let mut img = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0]));
    img.put_pixel(2, 1, Rgba([200, 100, 50, 255]));
    let clip = Clip { width: w, height: h, fps: 10.0, frames: vec![img] };

    let world = build_brick_world(&clip, &AnimOptions::default(), &mut NoProgress)
        .expect("a single opaque pixel must build");

    // 1 display brick (the lone opaque pixel) + 1 chip shell.
    assert_eq!(world.bricks.len(), 2, "exactly one pixel is opaque in any frame");

    // Micro style at the default pixel_extent=1 has footprint 1, pitch 2, so
    // (col=2, row=1) lands at world position (4, 2, 1). A column-major
    // indexing bug would instead light up (col=3, row=1) -> (6, 2, 1) (see
    // the doc comment above for the arithmetic), so this pins the exact
    // coordinates rather than merely the count. Display bricks rest at
    // z = their own half-height (1 for Micro); the chip shell sits at z=2,
    // so filtering on z=1 picks out the lone display brick unambiguously.
    let display_bricks: Vec<Position> =
        world.bricks.iter().filter(|b| b.position.z == 1).map(|b| b.position).collect();
    assert_eq!(
        display_bricks,
        vec![Position { x: 4, y: 2, z: 1 }],
        "the opaque pixel at (col=2, row=1) must render at its row-major position (4, 2), \
         not the column-major transposition (6, 2)"
    );
}
