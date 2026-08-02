//! Text mode (`anim::text_bricks`), end to end through `AnimMode::Text`.
//!
//! Every module text mode is built from carries its own unit tests; this file
//! exists for the properties that only appear once a REAL render has been
//! produced and written, and whose failure mode is invisible from Rust -- you
//! find out by loading the save in game and seeing a broken picture:
//!
//!   * a band's row range moving between frames scrambles the image, because a
//!     text component is a fixed brick and only its string changes;
//!   * a band string over `MAX_COMPONENT_CHARS` is truncated SILENTLY by the
//!     game, and the symptom is the right-hand end of a strip going blank;
//!   * an array shorter than the frame count plays a truncated clip;
//!   * a palette that changed the band layout would mean `--colors` altered
//!     geometry rather than only content.
//!
//! Band strings are read back out of a WRITTEN save, never off the in-memory
//! `World`: `BrdbComponent` exposes a component's type but not its property
//! values. `.brz` bytes are not reproducible run to run, so the round trip is
//! strictly for reading structure back -- never for comparing files or hashes.
#[path = "wire_integrity.rs"]
mod wire_integrity;

use heightmap::anim::{AnimMode, bricks::AnimOptions, text_layout::plan_bands};
use heightmap::progress::NoProgress;

/// In-memory `FrameSource`s and save round-tripping, modelled on the helpers
/// already at the top of `tests/anim_color.rs` and in `src/anim/text_bricks.rs`'s
/// own test module.
mod support {
    use brdb::{IntoReader, schema::WireArrayVariant};
    use heightmap::video::Clip;
    use image::{Rgba, RgbaImage};
    use std::path::Path;

    /// Every (pixel, frame) triple a distinct colour, so a transposition or an
    /// off-by-one between the source and a saved band string shows up rather
    /// than passing on symmetry. Same generator as `tests/anim_color.rs`'s
    /// `distinct_clip`.
    pub fn clip(w: u32, h: u32, frames: usize) -> Clip {
        let frames = (0..frames as u32)
            .map(|f| {
                RgbaImage::from_fn(w, h, |x, y| {
                    Rgba([
                        (x * 17 + f) as u8,
                        (y * 53 + f * 7) as u8,
                        (x * 31 + y * 11 + f * 3) as u8,
                        255,
                    ])
                })
            })
            .collect();
        Clip { width: w, height: h, fps: 10.0, frames }
    }

    /// The worst case the band layout is sized against: horizontally adjacent
    /// pixels never share a colour, so EVERY pixel opens its own 16-character
    /// `<color="RRGGBB">` tag and a row costs its full closed-form bound. A
    /// flat or smooth clip cannot trip the character limit at all, so it would
    /// make the limit test vacuous.
    pub fn noisy_clip(w: u32, h: u32, frames: usize) -> Clip {
        let frames = (0..frames as u32)
            .map(|f| {
                RgbaImage::from_fn(w, h, |x, y| {
                    // Consecutive x differ by 1 in `n`, hence in the low byte.
                    let n = y * w + x + f * 7919;
                    Rgba([
                        (n & 0xFF) as u8,
                        ((n >> 8) & 0xFF) as u8,
                        ((n >> 3) & 0xFF) as u8,
                        255,
                    ])
                })
            })
            .collect();
        Clip { width: w, height: h, fps: 10.0, frames }
    }

    /// A smooth two-axis gradient: unquantized, adjacent pixels differ in
    /// every row, so almost every pixel opens its own tag; a small palette has
    /// broad, genuinely-adjacent regions to collapse into long runs.
    pub fn gradient_clip(w: u32, h: u32, frames: usize) -> Clip {
        let frames = (0..frames as u32)
            .map(|f| {
                RgbaImage::from_fn(w, h, |x, y| {
                    Rgba([
                        (x * 255 / w.max(1)) as u8,
                        (y * 255 / h.max(1)) as u8,
                        ((x * 3 + y * 5 + f * 11) % 256) as u8,
                        255,
                    ])
                })
            })
            .collect();
        Clip { width: w, height: h, fps: 10.0, frames }
    }

    /// Write `world` to a temporary `.brz`, hand the path to `f`, and delete
    /// the file afterwards even if `f` panics -- an assertion failure must not
    /// leave saves behind in the temp directory.
    ///
    /// The bytes are NOT reproducible run to run, so a written save is only
    /// ever read back for its structure; never compare two of them.
    pub fn with_written_save<R>(world: &brdb::World, tag: &str, f: impl FnOnce(&Path) -> R) -> R {
        let path = std::env::temp_dir().join(format!(
            "h2b_anim_text_{tag}_{}_{:?}.brz",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&path)));
        let _ = std::fs::remove_file(&path);
        match out {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(e),
        }
    }

    /// Every band's per-frame strings, read back out of a written save.
    ///
    /// Component DATA is unreachable from an in-memory `World` -- `BrdbComponent`
    /// exposes `component_type()` but not its property values -- so this
    /// round-trips, exactly as `src/anim/text_bricks.rs`'s `total_color_tags`
    /// and `tests/anim_color.rs`'s `saved_color_arrays` do.
    ///
    /// One entry per persisted string `ArrayVar`, which text mode mints one of
    /// per band per BANK. Every test here uses the default bank size, so that
    /// is one per band; the inner `Vec` is that band's frames in play order,
    /// which is what makes each band's frames comparable without depending on
    /// the order bands come back in.
    pub fn band_strings(world: &brdb::World) -> Vec<Vec<String>> {
        with_written_save(world, "bands", |path| {
            let db = brdb::Brz::open(path).expect("reopen").into_reader();
            let mut chip_grid_id = None;
            for index in db.entity_chunk_index().expect("entity chunk index") {
                for e in db.entity_chunk(index).expect("entity chunk") {
                    if e.is_microchip_grid() {
                        chip_grid_id = e.id;
                    }
                }
            }
            let gid =
                chip_grid_id.expect("the renderer must publish exactly one microchip grid");

            let mut bands = Vec::new();
            for chunk in &db.brick_chunk_index(gid).expect("chunk index") {
                let (_soa, structs) = db.component_chunk(gid, chunk.index).expect("components");
                for s in &structs {
                    if s.get_name() == "BrickComponentData_WireGraphPseudo_ArrayVar"
                        && let Some(value) = s.get("Value")
                    {
                        let variant: WireArrayVariant =
                            value.try_into().expect("ArrayVar Value must decode");
                        if let WireArrayVariant::StringArray(v) = variant {
                            bands.push(v);
                        }
                    }
                }
            }
            bands
        })
    }

    /// Total characters across every band string -- the render's real text
    /// size, which is the only quantity a palette is meant to change.
    pub fn total_chars(bands: &[Vec<String>]) -> usize {
        bands
            .iter()
            .flat_map(|b| b.iter())
            .map(|s| s.chars().count())
            .sum()
    }
}

#[test]
fn a_real_render_passes_wire_integrity_at_two_sizes() {
    for (w, h) in [(64u32, 36u32), (192, 108)] {
        let world = AnimMode::Text
            .build(&support::clip(w, h, 8), &AnimOptions::default(), &mut NoProgress)
            .unwrap_or_else(|e| panic!("{w}x{h} must render: {e}"));
        support::with_written_save(&world, "wires", |path| {
            wire_integrity::assert_wires_valid(path);
        });
    }
}

/// A subtitled render must be structurally valid too. The subtitle's value
/// wire is the only one in a text render that crosses from the chip's inner
/// grid to a component the SUBTITLE placed rather than `add_text_tiles` -- a
/// remote wire whose chunk-relative brick index is exactly the thing that goes
/// wrong silently -- and the banked case adds the `Select` cascade on top.
///
/// The last band's strings are asserted alongside, because a subtitle
/// `ArrayVar` mixed in with the band arrays is the failure mode a plain
/// wire-integrity pass would not notice.
#[test]
fn a_subtitled_render_passes_wire_integrity_at_both_bank_counts() {
    use heightmap::subs::{Cue, Subtitles};
    use std::sync::Arc;

    let track = Arc::new(Subtitles::new(vec![
        Cue { start_s: 0.0, end_s: 0.3, text: "a line".to_string() },
        Cue { start_s: 0.3, end_s: 1.0, text: "another, longer line".to_string() },
    ]));
    for (bank_size, tag) in [(usize::MAX, "single"), (2usize, "banked")] {
        let opts = AnimOptions {
            bank_size,
            subtitles: Some(track.clone()),
            ..AnimOptions::default()
        };
        let world = AnimMode::Text
            .build(&support::clip(64, 36, 5), &opts, &mut NoProgress)
            .unwrap_or_else(|e| panic!("{tag} must render: {e}"));
        support::with_written_save(&world, &format!("sub_wires_{tag}"), |path| {
            wire_integrity::assert_wires_valid(path);
        });

        // The subtitle's own array is in there, holding one entry per frame
        // and carrying the cues at the frames their times land on. 10 fps, so
        // frames 0..2 are the first cue and 3..4 the second.
        let arrays = support::band_strings(&world);
        let subtitle: Vec<&Vec<String>> = arrays
            .iter()
            .filter(|a| a.iter().any(|s| s.contains("a line") || s.contains("another")))
            .collect();
        let expected_banks = if bank_size == usize::MAX { 1 } else { 3 };
        assert_eq!(subtitle.len(), expected_banks, "{tag}: one subtitle array per bank");
        let flat: Vec<String> = subtitle.iter().flat_map(|a| a.iter().cloned()).collect();
        assert_eq!(
            flat,
            vec![
                "a line",
                "a line",
                "a line",
                "another, longer line",
                "another, longer line",
            ],
            "{tag}: every frame gets the cue live at its own timestamp"
        );
    }
}

#[test]
fn band_row_ranges_are_identical_for_every_frame() {
    // The core invariant: text components are fixed bricks, so the rows a
    // given component draws must not move frame to frame.
    let plan = plan_bands(192, 108, 2).unwrap();
    let world = AnimMode::Text
        .build(&support::clip(192, 108, 12), &AnimOptions::default(), &mut NoProgress)
        .unwrap();
    let bands = support::band_strings(&world);
    for band in &bands {
        let line_counts: Vec<usize> = band.iter().map(|s| s.lines().count()).collect();
        assert!(
            line_counts.windows(2).all(|w| w[0] == w[1]),
            "a band's line count changed between frames: {line_counts:?}"
        );
    }
    assert_eq!(bands.len(), plan.len());

    // Stronger than "constant": the rows a band draws must be the rows the
    // layout planned for it. Compared as multisets because the order arrays
    // come back from the save in is the save's, not the plan's.
    let mut drawn: Vec<usize> = bands.iter().map(|b| b[0].lines().count()).collect();
    let mut planned: Vec<usize> = plan.iter().map(|b| b.rows).collect();
    drawn.sort_unstable();
    planned.sort_unstable();
    assert_eq!(drawn, planned, "each band must draw exactly its planned rows");
}

#[test]
fn no_frame_of_a_real_render_exceeds_the_component_limit() {
    // Every band of every frame, not a sample -- the failure mode is silent
    // truncation in game. The clip is NOISY on purpose: nearly every pixel
    // opens its own colour tag, which is the content the layout's closed-form
    // bound is sized against and the only content that can actually trip this.
    let world = AnimMode::Text
        .build(&support::noisy_clip(192, 108, 12), &AnimOptions::default(), &mut NoProgress)
        .unwrap();
    let mut worst = 0usize;
    for band in support::band_strings(&world) {
        for s in band {
            let n = s.chars().count();
            worst = worst.max(n);
            assert!(
                n <= heightmap::text::MAX_COMPONENT_CHARS,
                "{n} chars is over the {} limit; the game truncates silently",
                heightmap::text::MAX_COMPONENT_CHARS
            );
        }
    }
    println!(
        "worst band on a fully-noisy 192x108 frame: {worst} of {} chars",
        heightmap::text::MAX_COMPONENT_CHARS
    );
}

#[test]
fn array_length_equals_frame_count_per_band() {
    let world = AnimMode::Text
        .build(&support::clip(64, 36, 7), &AnimOptions::default(), &mut NoProgress)
        .unwrap();
    let bands = support::band_strings(&world);
    assert!(!bands.is_empty(), "a 64x36 render must produce bands");
    for band in &bands {
        assert_eq!(band.len(), 7, "a short array plays a truncated clip");
    }
}

#[test]
fn quantizing_shortens_the_render_without_changing_its_shape() {
    let q = AnimOptions { colors: 16, ..AnimOptions::default() };
    let plain = AnimMode::Text
        .build(&support::gradient_clip(96, 54, 6), &AnimOptions::default(), &mut NoProgress)
        .unwrap();
    let quant = AnimMode::Text
        .build(&support::gradient_clip(96, 54, 6), &q, &mut NoProgress)
        .unwrap();
    let (plain, quant) = (support::band_strings(&plain), support::band_strings(&quant));
    assert_eq!(
        plain.len(),
        quant.len(),
        "the palette must not change the band layout"
    );
    let (before, after) = (support::total_chars(&plain), support::total_chars(&quant));
    assert!(
        after < before,
        "a 16-colour palette must shorten the render: {before} -> {after}"
    );
    println!("96x54x6 gradient: {before} chars -> {after} at 16 colours");
}

// --- The estimate against a real render -------------------------------------

/// Text mode's twin of
/// `tests/anim_color.rs::the_cost_estimate_matches_a_real_render` and
/// `tests/anim_world.rs::the_hex_cost_estimate_matches_a_real_render`.
///
/// With this, all three estimators are pinned against a world that was
/// actually built rather than against arithmetic that only agrees with itself.
/// Hex mode's estimate was wrong by a constant 3 wires for its entire life
/// precisely because colour-array mode was the only one with this test.
#[test]
fn the_text_cost_estimate_matches_a_real_render() {
    use heightmap::anim::cost;
    for (w, h, n, bank) in [
        (64u32, 36u32, 5usize, usize::MAX),
        (32, 16, 7, 2),
        (48, 24, 9, 3),
    ] {
        let opts = AnimOptions { bank_size: bank, ..AnimOptions::default() };
        let world = AnimMode::Text
            .build(&support::clip(w, h, n), &opts, &mut NoProgress)
            .expect("build");
        let est = cost::estimate_text(w, h, n, &opts).expect("a legal geometry must estimate");

        // Gates are every inner-grid brick that is not one of the clock's five
        // I/O pins (Pause, Restart, Resume, Rate, Done).
        let inner = world.grids[0].1.len();
        assert_eq!(
            inner - 5,
            est.gates,
            "{w}x{h}x{n} bank {bank}: estimate said {} gates, the render emitted {}",
            est.gates,
            inner - 5
        );
        assert_eq!(
            world.wires.len(),
            est.wires,
            "{w}x{h}x{n} bank {bank}: estimate said {} wires, the render emitted {}",
            est.wires,
            world.wires.len()
        );
        assert_eq!(world.bricks.len(), est.bricks, "{w}x{h}x{n} bank {bank}: brick count");
    }
}

/// The same with a subtitle track, so `cost::subtitle_cost`'s contribution to
/// the text estimate is measured against a render too.
#[test]
fn the_text_cost_estimate_matches_a_subtitled_render() {
    use heightmap::anim::cost;
    use heightmap::subs::{Cue, Subtitles};
    use std::sync::Arc;
    for bank in [usize::MAX, 2] {
        let opts = AnimOptions {
            bank_size: bank,
            subtitles: Some(Arc::new(Subtitles::new(vec![Cue {
                start_s: 0.0,
                end_s: 1.0,
                text: "hi".to_string(),
            }]))),
            ..AnimOptions::default()
        };
        let world = AnimMode::Text
            .build(&support::clip(64, 36, 6), &opts, &mut NoProgress)
            .expect("build");
        let est = cost::estimate_text(64, 36, 6, &opts).expect("a legal geometry must estimate");
        assert_eq!(world.grids[0].1.len() - 5, est.gates, "bank {bank}: gate count");
        assert_eq!(world.wires.len(), est.wires, "bank {bank}: wire count");
        assert_eq!(world.bricks.len(), est.bricks, "bank {bank}: brick count");
    }
}
