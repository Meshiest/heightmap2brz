//! Read a tuned drum-palette `.brz` back and print, per cell, the role label
//! and the sound the user settled on -- the input for regenerating the fold
//! table after the user retunes the palette.
//!
//! ```sh
//! cargo run --example extract_drum_palette -- <palette.brz>
//! ```
//!
//! Components are paired to their brick through `component_brick_indices`, so a
//! sound and the text label sharing a brick are read off the same cell
//! regardless of brick order.

use brdb::schema::BrdbValue;
use brdb::{Brz, IntoReader};
use std::collections::BTreeMap;

#[derive(Default)]
struct Cell {
    label: Option<String>,
    prompt: Option<String>,
    asset: Option<String>,
    pitch: Option<f32>,
    volume: Option<f32>,
}

fn as_f32(v: Option<&BrdbValue>) -> Option<f32> {
    match v {
        Some(BrdbValue::F32(f)) => Some(*f),
        _ => None,
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: extract_drum_palette <file.brz>");
    let reader = Brz::open(&path).expect("open brz").into_reader();
    let global = reader.read_global_data().expect("global data");

    // brick (chunk-local) -> its components, across every chunk of the main grid.
    let mut cells: BTreeMap<(String, u32), Cell> = BTreeMap::new();
    for chunk in reader.brick_chunk_index(1).expect("brick chunk index") {
        if chunk.num_components == 0 {
            continue;
        }
        let (soa, comps) = reader.component_chunk_soa(1, chunk.index).expect("component soa");
        for (i, c) in comps.iter().enumerate() {
            let brick = soa.component_brick_indices[i];
            let cell = cells.entry((chunk.index.to_string(), brick)).or_default();
            if let Some(BrdbValue::Asset(Some(ai))) = c.get("AudioDescriptor") {
                if let Some((_, name)) = global.external_asset_references.get_index(*ai) {
                    cell.asset = Some(name.clone());
                }
                cell.pitch = as_f32(c.get("PitchMultiplier"));
                cell.volume = as_f32(c.get("VolumeMultiplier"));
            }
            if let Some(BrdbValue::String(t)) = c.get("Text") {
                cell.label = Some(t.clone());
            }
            if let Some(BrdbValue::String(p)) = c.get("PromptCustomLabel") {
                cell.prompt = Some(p.clone());
            }
        }
    }

    // A cell with a sound is an emitter brick; a cell with a prompt is a button.
    println!("--- emitter cells (sound + co-located label) ---");
    let mut sounds = 0;
    for cell in cells.values() {
        if let Some(asset) = &cell.asset {
            sounds += 1;
            println!(
                "label={:<14} sound={:<40} pitch={:<5} vol={}",
                format!("{:?}", cell.label.as_deref().unwrap_or("?")),
                asset,
                cell.pitch.unwrap_or(f32::NAN),
                cell.volume.unwrap_or(f32::NAN),
            );
        }
    }
    println!("\n--- button cells (prompt only, no sound) ---");
    for cell in cells.values() {
        if cell.asset.is_none() {
            if let Some(p) = &cell.prompt {
                println!("prompt={p:?}  label={:?}", cell.label.as_deref().unwrap_or("?"));
            }
        }
    }
    println!("\n{sounds} emitter cells");
}
