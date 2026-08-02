//! Shared helper: re-read a written save and validate every wire endpoint,
//! mimicking the game loader's port resolution.
//!
//! For each wire endpoint (source and target, local and remote) this checks:
//!   1. the referenced brick index is in range for its chunk;
//!   2. the brick at that index actually carries the referenced component type;
//!   3. the referenced port index resolves to a registered wire-port name.
//!
//! Check 2 is the one that matters most: the writer resolves a `WirePort`'s
//! component type by *name lookup in global data alone* -- it never verifies the
//! brick carries that component. So wiring to the wrong brick (e.g. a
//! microchip's shell instead of its inner pin brick) produces a save that opens
//! fine, passes a range-only check, and silently does nothing in game.
//!
//! Declared as a module (not a crate) by consumers via
//! `#[path = "wire_integrity.rs"] mod wire_integrity;`, so it must not warn
//! when a consumer only uses part of its surface.
#![allow(dead_code)]

use brdb::{Brick, Brz, ChunkIndex, IntoReader, WireChunkSoA, World, schema::BrdbSchemaGlobalData};
use std::collections::HashMap;
use std::path::Path;

/// What a single (grid, chunk) pair contains: how many bricks, and which
/// component type indices each brick carries.
struct ChunkContents {
    brick_count: usize,
    /// brick index in chunk -> component type indices on that brick
    components: HashMap<u32, Vec<u16>>,
}

/// One end of one wire, already resolved to the grid and chunk it lives in.
struct Endpoint {
    /// Which of the four SoA arrays this came from, for the failure message.
    label: &'static str,
    grid_id: usize,
    chunk: ChunkIndex,
    brick_index: u32,
    component_type_index: u16,
    port_index: u16,
}

/// Panics with a descriptive message if any wire in the save is dangling.
pub fn assert_wires_valid(path: &Path) {
    let db = Brz::open(path).expect("open brz").into_reader();
    let data = db.global_data().expect("global data");

    // Main grid is 1; brick-grid entities carry their own ids.
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

    // Pass 1: map every (grid, chunk) to its brick count and per-brick
    // component types. Wires can name any grid, so this must be built up front.
    let mut contents: HashMap<(usize, ChunkIndex), ChunkContents> = HashMap::new();
    for &gid in &grid_ids {
        let Ok(chunks) = db.brick_chunk_index(gid) else {
            continue;
        };
        for chunk in &chunks {
            let soa = db.brick_chunk_soa(gid, chunk.index).expect("brick soa");
            let brick_count = soa.brick_type_indices.len();

            let mut components: HashMap<u32, Vec<u16>> = HashMap::new();
            if chunk.num_components > 0 {
                let (csoa, _) = db
                    .component_chunk_soa(gid, chunk.index)
                    .expect("component chunk soa");
                // component_type_counters is run-length encoded; expand it to
                // one type index per component instance, parallel to
                // component_brick_indices.
                let type_indices: Vec<u16> = csoa
                    .component_type_counters
                    .iter()
                    .flat_map(|c| {
                        let t = c.type_index as u16;
                        (0..c.num_instances).map(move |_| t)
                    })
                    .collect();
                assert_eq!(
                    type_indices.len(),
                    csoa.component_brick_indices.len(),
                    "grid {gid} chunk {:?}: {} component type counter instances but {} component \
                     brick indices -- component SoA is internally inconsistent",
                    chunk.index,
                    type_indices.len(),
                    csoa.component_brick_indices.len(),
                );
                for (i, brick_index) in csoa.component_brick_indices.iter().enumerate() {
                    components
                        .entry(*brick_index)
                        .or_default()
                        .push(type_indices[i]);
                }
            }

            contents.insert(
                (gid, chunk.index),
                ChunkContents {
                    brick_count,
                    components,
                },
            );
        }
    }

    // Pass 2: resolve every wire endpoint against pass 1.
    let mut checked = 0usize;
    for &gid in &grid_ids {
        let Ok(chunks) = db.brick_chunk_index(gid) else {
            continue;
        };
        for chunk in &chunks {
            if chunk.num_wires == 0 {
                continue;
            }
            let raw = db.wire_chunk_soa(gid, chunk.index).expect("wire chunk soa");
            let wires: WireChunkSoA = (&raw.to_value()).try_into().expect("wire chunk soa value");

            // Wires live in the *target's* grid and chunk. Local endpoints and
            // remote targets resolve here; only a remote source names another
            // grid and chunk.
            let mut endpoints: Vec<Endpoint> = Vec::new();
            for p in &wires.local_wire_sources {
                endpoints.push(Endpoint {
                    label: "local source",
                    grid_id: gid,
                    chunk: chunk.index,
                    brick_index: p.brick_index_in_chunk,
                    component_type_index: p.component_type_index,
                    port_index: p.port_index,
                });
            }
            for p in &wires.local_wire_targets {
                endpoints.push(Endpoint {
                    label: "local target",
                    grid_id: gid,
                    chunk: chunk.index,
                    brick_index: p.brick_index_in_chunk,
                    component_type_index: p.component_type_index,
                    port_index: p.port_index,
                });
            }
            for p in &wires.remote_wire_sources {
                endpoints.push(Endpoint {
                    label: "remote source",
                    grid_id: p.grid_persistent_index as usize,
                    chunk: p.chunk_index,
                    brick_index: p.brick_index_in_chunk,
                    component_type_index: p.component_type_index,
                    port_index: p.port_index,
                });
            }
            for p in &wires.remote_wire_targets {
                endpoints.push(Endpoint {
                    label: "remote target",
                    grid_id: gid,
                    chunk: chunk.index,
                    brick_index: p.brick_index_in_chunk,
                    component_type_index: p.component_type_index,
                    port_index: p.port_index,
                });
            }

            for endpoint in &endpoints {
                check_endpoint(endpoint, &contents, &data);
                checked += 1;
            }
        }
    }

    assert!(checked > 0, "save contains no wires -- nothing was validated");
}

/// Validate one wire endpoint against the chunk contents gathered in pass 1.
fn check_endpoint(
    endpoint: &Endpoint,
    contents: &HashMap<(usize, ChunkIndex), ChunkContents>,
    data: &BrdbSchemaGlobalData,
) {
    let &Endpoint {
        label,
        grid_id,
        chunk,
        brick_index,
        component_type_index,
        port_index,
    } = endpoint;

    let type_name = |t: u16| -> String {
        data.component_type_names
            .get_index(t as usize)
            .cloned()
            .unwrap_or_else(|| "<unregistered>".to_string())
    };
    let port_name = |p: u16| -> String {
        data.component_wire_port_names
            .get_index(p as usize)
            .cloned()
            .unwrap_or_else(|| "<unregistered>".to_string())
    };
    let wanted = format!(
        "component type {component_type_index} ({}) port {port_index} ({})",
        type_name(component_type_index),
        port_name(port_index)
    );
    let at = format!("{label} in grid {grid_id} chunk {chunk:?} brick {brick_index}");

    let Some(chunk_contents) = contents.get(&(grid_id, chunk)) else {
        panic!("{at} references {wanted}, but grid {grid_id} has no chunk {chunk:?}");
    };

    // 1. In-range check (cheap first pass).
    assert!(
        (brick_index as usize) < chunk_contents.brick_count,
        "wire target brick index {brick_index} out of range (chunk has {} bricks): {at} \
         references {wanted}",
        chunk_contents.brick_count
    );

    // 2. The brick must actually carry the referenced component type.
    let carried = chunk_contents
        .components
        .get(&brick_index)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    assert!(
        carried.contains(&component_type_index),
        "{at} references {wanted}, but that brick does not carry that component type -- it \
         carries {}",
        if carried.is_empty() {
            "no components at all".to_string()
        } else {
            carried
                .iter()
                .map(|t| format!("{t} ({})", type_name(*t)))
                .collect::<Vec<_>>()
                .join(", ")
        }
    );

    // 3. The port index must resolve to a registered wire-port name.
    assert!(
        data.component_wire_port_names
            .get_index(port_index as usize)
            .is_some(),
        "{at} references port index {port_index}, which is not in the registered wire-port \
         names ({} registered)",
        data.component_wire_port_names.len()
    );
}

/// Builds a two-brick world: a rerouter wired from a BoolNot gate's output.
/// `target_component_is_wrong` deliberately aims the target endpoint at a
/// component type the target brick does not carry.
fn write_fixture(path: &Path, target_component_is_wrong: bool) {
    let mut world = World::new();
    world.meta.bundle.description = "wire integrity fixture".to_string();

    let (a, a_id) = Brick {
        position: (0, 0, 1).into(),
        asset: brdb::assets::bricks::B_REROUTE,
        ..Default::default()
    }
    .with_component(brdb::assets::components::Rerouter)
    .with_id_split();

    let (b, b_id) = Brick {
        position: (15, 0, 1).into(),
        asset: brdb::assets::components::LogicGate::BoolNot.brick(),
        ..Default::default()
    }
    .with_component(brdb::assets::components::LogicGate::BoolNot.component())
    .with_id_split();

    world.add_bricks([a, b]);
    // Brick b carries the BoolNot gate component, not a Rerouter. Aiming the
    // rerouter input port at brick b is in range but references a component
    // that brick does not carry -- the microchip pin-vs-shell failure mode.
    let target_brick = if target_component_is_wrong { b_id } else { a_id };
    world.add_wire_connection(
        brdb::assets::components::LogicGate::BoolNot.output_of(b_id),
        brdb::assets::components::Rerouter::input_of(target_brick),
    );
    world.register_used_components();

    std::fs::write(path, world.to_brz_vec().expect("encode")).expect("write");
}

#[test]
fn helper_accepts_a_known_good_wired_save() {
    let path = std::env::temp_dir().join(format!("h2b_wire_ok_{}.brz", std::process::id()));
    write_fixture(&path, false);
    assert_wires_valid(&path);
    let _ = std::fs::remove_file(&path);
}

/// The check must reject a wire whose endpoint names a component the target
/// brick does not carry -- a range-only check reports this as green.
#[test]
#[should_panic(expected = "does not carry that component type")]
fn helper_rejects_a_wire_to_a_brick_lacking_the_component() {
    let path = std::env::temp_dir().join(format!("h2b_wire_bad_{}.brz", std::process::id()));
    write_fixture(&path, true);
    let result = std::panic::catch_unwind(|| assert_wires_valid(&path));
    let _ = std::fs::remove_file(&path);
    // Re-raise so #[should_panic] sees the message, after cleaning up the temp file.
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}
