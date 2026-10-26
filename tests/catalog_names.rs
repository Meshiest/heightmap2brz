//! Every component type and gate brick named in `src/` must exist in the
//! brdb catalog of the game build it targets. A renamed or removed gate
//! otherwise only shows up when a save fails to load in game.

use brdb::assets::components::{COMPONENTS, component};
use std::path::Path;

fn source_files(dir: &Path, out: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).expect("read src dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            source_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(std::fs::read_to_string(&path).expect("read source"));
        }
    }
}

/// Quoted string literals starting with `prefix`, up to the closing quote.
fn literals<'a>(sources: &'a [String], prefix: &str) -> Vec<&'a str> {
    let mut found = Vec::new();
    let needle = format!("\"{prefix}");
    for src in sources {
        let mut rest = src.as_str();
        while let Some(at) = rest.find(&needle) {
            let start = at + 1;
            let len = rest[start..].find('"').expect("closing quote");
            let lit = &rest[start..start + len];
            if lit.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                found.push(lit);
            }
            rest = &rest[start + len..];
        }
    }
    found.sort_unstable();
    found.dedup();
    found
}

#[test]
fn every_emitted_component_and_gate_brick_is_in_the_catalog() {
    let mut sources = Vec::new();
    source_files(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut sources,
    );

    let mut missing = Vec::new();
    let types = literals(&sources, "BrickComponentType_");
    assert!(!types.is_empty(), "the scan found no component types");
    for name in types.iter().chain(&literals(&sources, "Component_")) {
        if component(name).is_none() {
            missing.push(format!("component {name}"));
        }
    }
    let host_bricks: Vec<&str> = COMPONENTS
        .iter()
        .flat_map(|c| c.bricks.iter().copied())
        .collect();
    for brick in literals(&sources, "B_1x1_Gate_") {
        if !host_bricks.contains(&brick) {
            missing.push(format!("gate brick {brick}"));
        }
    }
    assert!(missing.is_empty(), "not in the brdb catalog: {missing:#?}");
}
