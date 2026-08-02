//! **Throwaway spike, not the feature.** Renders a clip as animated
//! `Component_TextDisplay` bricks so a human can load the save in game and see
//! whether an animated text display holds frame rate.
//!
//! # The question
//!
//! Brick mode ([`heightmap::anim::bricks`]) spends 2 gates per PIXEL, and the
//! game drops frames near 20 000 gates -- about 10 000 pixels. Text mode
//! replaces the per-pixel gate pair with one `TextDisplay` per BAND of image
//! rows, fed a whole pre-rendered string per frame from a `StringArray`: 2
//! gates per band instead of 2 per pixel. On paper that is three orders of
//! magnitude fewer gates.
//!
//! The unmeasured risk is that the cost does not vanish, it MOVES: a
//! `TextDisplay` whose string changes every frame plausibly re-lays-out its
//! entire glyph run each time, and at ~20 000 glyphs that could cost more than
//! the gates it deletes. This probe exists only to put that in front of the
//! game's profiler. It deliberately does NOT decide anything the
//! implementation plan owns -- no quantisation, no CLI surface, no cost model,
//! no banking, no `src/main.rs` or GUI wiring.
//!
//! # Fixed band layout
//!
//! Text components are physical bricks at fixed positions with fixed wiring,
//! so the rows a given component draws MUST NOT move between frames. The
//! layout is therefore decided ONCE, from a worst-case bound, before a single
//! frame is looked at: a row of `width` pixels costs at most
//! `width * (COLOR_TAG_CHARS + char_repeat)` characters, because the worst
//! case is every pixel changing colour and so emitting its own `<color="…">`
//! tag. Rows are packed into bands until that bound would exceed
//! [`MAX_COMPONENT_CHARS`]. The bound cannot be exceeded by construction, so
//! no scan pass over the frames is needed -- which matters, because the
//! pipeline streams and never retains a frame.
//!
//! Every band's real text is still checked against [`MAX_COMPONENT_CHARS`] on
//! every frame. Silent truncation by the game is the failure this guards, and
//! it is invisible until someone looks at the render.
//!
//! # Invocation
//!
//! ```text
//! cargo run --release --example text_probe -- <input> [--out out.brz]
//!     [--width 192] [--height 108] [--fps 12] [--start S] [--duration S]
//!     [--pixel-size 1.0]
//! ```
use std::path::{Path, PathBuf};

use brdb::{
    AsBrdbValue, IntVector, Position, Vector3f, WirePort, World, schema::WireArrayVariant,
};
use image::RgbaImage;

use heightmap::anim::bricks::{ARRAY_GET, ARRAY_VAR, CHANGE_DETECTOR};
use heightmap::anim::chip;
use heightmap::anim::clock::{build_clock, gate};
use heightmap::anim::layout::{GATE_HALF, STAGE_PITCH, lattice_pos_staged};
use heightmap::anim::pack::BANK_FRAMES;
use heightmap::text::{
    FontPreset, MAX_COMPONENT_CHARS, TextBand, TextOptions, TextTile, add_text_tiles, encode_bands,
};
use heightmap::video::backend::{Backend, open_video};
use heightmap::video::scale::{Filter, FitMode};
use heightmap::video::source::{Source, decode, is_animated, is_video_path};
use heightmap::video::stream::{AdaptedSource, FrameSource};

/// The text component the probe drives, and the input port carrying its
/// string. Both taken from the generated catalog in the `brdb` crate
/// (`src/assets/component_catalog.rs`), which lists `Component_TextDisplay`
/// with `Text` among its inputs -- NOT guessed, and the same component
/// `heightmap::text` already builds.
const TEXT_DISPLAY: &str = "Component_TextDisplay";
const TEXT_PORT: &str = "Text";

/// Characters one `<color="RRGGBB">` tag costs: `<color="` (8) + 6 hex + `">`
/// (2). The worst-case row bound assumes EVERY pixel emits one, which is what
/// `heightmap::text::encode_row` does when no two neighbours share a colour.
const COLOR_TAG_CHARS: usize = 16;

/// One `TextDisplay`'s worth of image rows, fixed for the whole clip.
///
/// Each band gets its OWN anchor cube at its own image row (see
/// [`build_probe_world`]), which is how `heightmap::text::add_text_tiles`
/// places an ordinary tiled export. The alternative -- one anchor for all of
/// them, with leading newlines pushing each band down -- stacks 54 cubes in
/// depth and needs a component `Offset.Z` of up to 106 world units to bring
/// the far ones back to the image plane, and `add_text_tiles`' own doc warns
/// the game does not honour large `Offset` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BandPlan {
    /// First image row this band draws.
    start_row: usize,
    rows: usize,
}

/// Upper bound on the characters one image row can encode to.
fn worst_case_row_chars(width: usize, char_repeat: usize) -> usize {
    width * (COLOR_TAG_CHARS + char_repeat)
}

/// Decide the band layout for a `width` x `height` clip, once, from the
/// worst-case row bound alone -- no frame is consulted, so every frame is
/// guaranteed to produce exactly this layout, and no scan pass is needed.
///
/// Every row costs the same bound, so the packing is uniform: `rows` rows cost
/// `rows * bound + (rows - 1)` (the separating newlines), and the largest
/// `rows` satisfying that is the same for every band.
fn plan_bands(width: usize, height: usize, char_repeat: usize) -> Result<Vec<BandPlan>, String> {
    let row = worst_case_row_chars(width, char_repeat);
    if row > MAX_COMPONENT_CHARS {
        return Err(format!(
            "row 0: a {width}-pixel row encodes to at most {row} chars ({COLOR_TAG_CHARS} for a \
             colour tag + {char_repeat} glyph chars per pixel), over the \
             {MAX_COMPONENT_CHARS}-char TextDisplay limit -- no band layout can fit it, and \
             every other row of this image costs the same; render narrower or with a smaller \
             char_repeat"
        ));
    }
    // rows * row + (rows - 1) <= MAX  <=>  rows * (row + 1) <= MAX + 1
    let rows_per_band = ((MAX_COMPONENT_CHARS + 1) / (row + 1)).max(1);
    Ok((0..height)
        .step_by(rows_per_band)
        .map(|start_row| BandPlan {
            start_row,
            rows: rows_per_band.min(height - start_row),
        })
        .collect())
}

/// Encode one frame's rows for one band. Returns the component's string and
/// its char count.
///
/// The rows are cropped out and handed to `heightmap::text::encode_bands`
/// rather than re-implementing the encoder: a full-width row crop encodes
/// identically to those rows inside the whole image, and a band always starts
/// with fresh colour state, which is exactly what `encode_bands` does at a
/// band boundary.
fn encode_band(
    img: &RgbaImage,
    band: &BandPlan,
    opts: &TextOptions,
    frame: usize,
) -> Result<(String, usize), String> {
    let (w, _) = img.dimensions();
    let sub =
        image::imageops::crop_imm(img, 0, band.start_row as u32, w, band.rows as u32).to_image();
    let mut encoded = encode_bands(&sub, opts)?;
    if encoded.len() != 1 {
        return Err(format!(
            "frame {frame}, band at row {}: the fixed layout split into {} bands -- the \
             worst-case bound is wrong",
            band.start_row,
            encoded.len()
        ));
    }
    let inner = encoded.pop().expect("checked len == 1");
    // The guard the whole probe hangs on: the game truncates silently past
    // this, and a truncated band is invisible until someone looks at the
    // render.
    if inner.chars > MAX_COMPONENT_CHARS {
        return Err(format!(
            "frame {frame}, band at row {} ({} rows): {} chars exceeds the \
             {MAX_COMPONENT_CHARS}-char TextDisplay limit -- the game would truncate this \
             silently",
            band.start_row, band.rows, inner.chars
        ));
    }
    debug_assert_eq!(inner.chars, inner.text.chars().count());
    Ok((inner.text, inner.chars))
}

/// What the probe measured, for the comparison against brick mode.
#[derive(Debug, Clone, Copy)]
struct Report {
    width: u32,
    height: u32,
    frames: usize,
    bands: usize,
    max_rows_per_band: usize,
    /// Bricks on the main grid: one anchor cube per band, plus the chip shell.
    main_bricks: usize,
    /// Bricks inside the chip: gates AND microchip pins.
    chip_bricks: usize,
    /// Of those, the ones that are actually logic gates (pins excluded).
    chip_gates: usize,
    wires: usize,
    max_band_chars: usize,
    /// The most characters ONE frame costs across all bands together. This is
    /// the layout work the game is asked to redo on a single tick, and it is
    /// the number the whole spike is about.
    max_frame_chars: usize,
    total_chars: usize,
}

/// Gates the shared clock and the change detector contribute, independent of
/// the clip: `Timer -> Multiply -> BitwiseOR -> ModuloFloored`, plus the
/// detector. The rest of what `build_clock` adds are microchip pins, not gates.
const SERVICE_GATES: usize = 5;

/// Build the animated text world. Streams `source` exactly once; no frame is
/// ever retained (only the encoded strings are).
fn build_probe_world(source: &dyn FrameSource, opts: &TextOptions) -> Result<(World, Report), String> {
    let info = source.info();
    let plan = plan_bands(
        info.width as usize,
        info.height as usize,
        opts.char_repeat.max(1),
    )?;
    if plan.is_empty() {
        return Err("source has zero height -- nothing to band".to_string());
    }

    // --- 1. Stream and encode ----------------------------------------------
    let mut band_texts: Vec<Vec<String>> = vec![Vec::new(); plan.len()];
    let mut max_band_chars = 0usize;
    let mut max_frame_chars = 0usize;
    let mut total_chars = 0usize;
    let mut frames = 0usize;
    {
        let mut stream = source.open()?;
        while let Some(frame) = stream.next()? {
            let mut frame_chars = 0usize;
            if frame.dimensions() != (info.width, info.height) {
                return Err(format!(
                    "frame {frames} is {:?}, but the source promised {}x{}",
                    frame.dimensions(),
                    info.width,
                    info.height
                ));
            }
            for (bi, band) in plan.iter().enumerate() {
                let (text, chars) = encode_band(&frame, band, opts, frames)?;
                max_band_chars = max_band_chars.max(chars);
                frame_chars += chars;
                band_texts[bi].push(text);
            }
            total_chars += frame_chars;
            max_frame_chars = max_frame_chars.max(frame_chars);
            frames += 1;
            if frames % 25 == 0 {
                eprintln!("  encoded {frames} frames");
            }
        }
    }
    if frames == 0 {
        return Err("clip has 0 frames -- nothing to render (check --start/--duration)".to_string());
    }
    // No bank spillover here: that is brick mode's machinery and reproducing
    // it would be building the feature, not probing it. Refuse instead.
    if frames > BANK_FRAMES {
        return Err(format!(
            "{frames} frames exceeds the {BANK_FRAMES}-entry array limit; this probe does not \
             implement bank spillover -- shorten the clip with --duration"
        ));
    }

    // --- 2. Text bricks on the main grid -----------------------------------
    //
    // One TILE per band, each carrying a single band, so `add_text_tiles`
    // anchors every component at its own image row -- no depth stack and no
    // large `Offset` values. The bands are a uniform `rows` tall (only the
    // last may be short), so their `start_row`s are exact multiples of that,
    // which is what makes the tile grid's `start_row / tile_px` indexing land
    // one band per row slot.
    let rows_per_band = plan[0].rows;
    debug_assert!(
        plan.iter().all(|b| b.start_row % rows_per_band == 0),
        "band starts must be multiples of the band height for the tile grid to index them"
    );
    // `tile_override` is the documented lever for exactly this: it tells
    // `add_text_tiles` how many image rows one anchor covers, which sets the
    // world pitch between consecutive anchors. Only the row (z) axis matters
    // here -- every band spans the full width, so there is a single column and
    // the horizontal pitch never comes into play.
    let place_opts = TextOptions {
        tile_override: Some(rows_per_band as u32),
        ..opts.clone()
    };

    // Frame 0's strings are baked in as each component's authored Text, so the
    // save shows the first frame before the clock has ticked once.
    let tiles: Vec<TextTile> = plan
        .iter()
        .zip(&band_texts)
        .map(|(p, texts)| TextTile {
            start_col: 0,
            start_row: p.start_row,
            bands: vec![TextBand {
                start_row: 0,
                rows: p.rows,
                text: texts[0].clone(),
                chars: texts[0].chars().count(),
            }],
        })
        .collect();

    let mut world = World::new();
    world.meta.bundle.description = "Text-mode animation probe".to_string();
    let text_ids = add_text_tiles(&mut world, tiles, &place_opts);

    // Each band renders DOWNWARD from its own anchor, so the bottom band hangs
    // one band-height below the lowest cube (z=1) and would sit underground.
    // Positions are plain data until the world is encoded (`World::add_brick`
    // only pushes), so shifting them here is safe and carries each band's text
    // with its own cube.
    let lift = (rows_per_band as f32 * opts.line_world_height * opts.pitch_y).ceil() as i32;
    for brick in world.bricks.iter_mut() {
        brick.position.z += lift;
    }

    // --- 3. The chip -------------------------------------------------------
    // Beside the anchor column on +Y, never stacked on it: the anchors span
    // y in [-1, 1] and the shell's own half-extent is 5, so y=20 is clear with
    // room to spare. Every coordinate stays non-negative.
    let n_bands = plan.len() as i32;
    let mut chip = chip::new_chip(
        &mut world,
        Position { x: 20, y: 20, z: 2 },
        Vector3f {
            x: 20.0,
            y: 20.0,
            z: (lift + 20) as f32,
        },
        IntVector { x: 5, y: 5, z: 5 },
    );

    // Band `b` occupies lattice row `b`, stage 0 (its ArrayVar) and stage 1
    // (its Get); service gates sit behind both at stage 2 -- the same staging
    // brick mode uses, so the two cannot drift.
    let service = |col: i32, row: i32| {
        lattice_pos_staged(col, row, 2, n_bands, GATE_HALF, STAGE_PITCH)
    };

    let clock = build_clock(&mut world, &mut chip, info.fps, frames, true, service(0, -2));
    let detector = gate(
        &mut chip,
        "B_1x1_Gate_Expr_ChangeDetectorExec",
        CHANGE_DETECTOR,
        service(0, -4),
        vec![],
    );
    world.add_wire_connection(
        clock.frame_index.clone(),
        WirePort::new(detector, CHANGE_DETECTOR, "Input"),
    );

    // --- 4. Two gates per band ---------------------------------------------
    //
    // Walked in band order, which is row order -- never by iterating a map, so
    // two runs of the same clip mint the same brick ids.
    //
    // The detector's `OnChanged` FANS OUT to every band's `Get.Exec`, the
    // pattern `anim::color_bricks` established. No exec input in this graph
    // ever gains a second source, so nothing here needs exec fan-in.
    for (bi, texts) in band_texts.into_iter().enumerate() {
        debug_assert_eq!(texts.len(), frames, "every band must hold one string per frame");
        let row = bi as i32;
        let array = gate(
            &mut chip,
            "B_1x1_Gate_Variable_Array",
            ARRAY_VAR,
            lattice_pos_staged(0, row, 0, n_bands, GATE_HALF, STAGE_PITCH),
            vec![(
                "Value",
                Box::new(WireArrayVariant::StringArray(texts)) as Box<dyn AsBrdbValue>,
            )],
        );
        let get = gate(
            &mut chip,
            "B_1x1_Gate_Exec_ArrayVar_Get",
            ARRAY_GET,
            lattice_pos_staged(0, row, 1, n_bands, GATE_HALF, STAGE_PITCH),
            vec![],
        );
        world.add_wire_connection(
            WirePort::new(array, ARRAY_VAR, "ArrayVarRef"),
            WirePort::new(get, ARRAY_GET, "ArrayVarRef"),
        );
        world.add_wire_connection(
            clock.frame_index.clone(),
            WirePort::new(get, ARRAY_GET, "Index"),
        );
        world.add_wire_connection(
            WirePort::new(detector, CHANGE_DETECTOR, "OnChanged"),
            WirePort::new(get, ARRAY_GET, "Exec"),
        );
        // Chip -> main grid, exactly as brick mode wires a MakeColorHex to a
        // display brick's BrickPropertyChanger: the two endpoints live in
        // different grids, and `World::add_wire_connection` emits the
        // `RemoteWirePortSource` itself. Nothing extra is needed for a
        // TextDisplay -- it is an ordinary component with an ordinary input
        // port.
        world.add_wire_connection(
            WirePort::new(get, ARRAY_GET, "Value"),
            WirePort::new(text_ids[bi], TEXT_DISPLAY, TEXT_PORT),
        );
    }

    // --- 5. Publish --------------------------------------------------------
    let chip_bricks = chip.placed().len();
    // Asserts non-overlap on BOTH grids before publishing.
    chip::finish(&mut world, chip)?;
    // Must be last, and must come AFTER `chip::finish`: it registers every
    // component type and port name actually used, and has to see all bricks,
    // grids and wires first. `add_text_tiles` already called it once, before
    // the chip existed -- this re-registration is the one that counts.
    world.register_used_components();

    let report = Report {
        width: info.width,
        height: info.height,
        frames,
        bands: plan.len(),
        max_rows_per_band: plan.iter().map(|b| b.rows).max().unwrap_or(0),
        main_bricks: world.bricks.len(),
        chip_bricks,
        chip_gates: 2 * plan.len() + SERVICE_GATES,
        wires: world.wires.len(),
        max_band_chars,
        max_frame_chars,
        total_chars,
    };
    Ok((world, report))
}

/// Open whatever `path` is: a video through the decode backends, an animated
/// image through the image decoders, a still as a one-frame clip.
fn open_source(path: &Path, fallback_fps: f32) -> Result<Box<dyn FrameSource>, String> {
    if is_video_path(path) {
        return open_video(path, Backend::Auto, None, FitMode::Exact, Filter::Lanczos, None);
    }
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let source = if is_animated(&bytes) {
        Source::Animated(bytes)
    } else {
        Source::Still(
            image::load_from_memory(&bytes)
                .map_err(|e| format!("decoding {}: {e:?}", path.display()))?
                .to_rgba8(),
        )
    };
    Ok(Box::new(decode(source, fallback_fps)?))
}

struct Args {
    input: PathBuf,
    out: PathBuf,
    width: u32,
    height: u32,
    fps: f32,
    start: f32,
    duration: Option<f32>,
    pixel_size: f32,
}

const USAGE: &str = "usage: text_probe <input> [--out PATH] [--width N] [--height N] \
                     [--fps F] [--start SECONDS] [--duration SECONDS] [--pixel-size F]";

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        input: PathBuf::new(),
        out: PathBuf::from("text_probe.brz"),
        width: 192,
        height: 108,
        fps: 12.0,
        start: 0.0,
        duration: None,
        pixel_size: 1.0,
    };
    let mut input = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| -> Result<String, String> {
            it.next()
                .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))
        };
        match arg.as_str() {
            "--out" | "-o" => args.out = PathBuf::from(value("--out")?),
            "--width" => {
                args.width = value("--width")?.parse().map_err(|e| format!("--width: {e}"))?
            }
            "--height" => {
                args.height = value("--height")?
                    .parse()
                    .map_err(|e| format!("--height: {e}"))?
            }
            "--fps" => args.fps = value("--fps")?.parse().map_err(|e| format!("--fps: {e}"))?,
            "--start" => {
                args.start = value("--start")?
                    .parse()
                    .map_err(|e| format!("--start: {e}"))?
            }
            "--duration" => {
                args.duration = Some(
                    value("--duration")?
                        .parse()
                        .map_err(|e| format!("--duration: {e}"))?,
                )
            }
            "--pixel-size" => {
                args.pixel_size = value("--pixel-size")?
                    .parse()
                    .map_err(|e| format!("--pixel-size: {e}"))?
            }
            "--help" | "-h" => return Err(USAGE.to_string()),
            other if other.starts_with('-') => {
                return Err(format!("unknown flag {other}\n{USAGE}"));
            }
            other => {
                if input.is_some() {
                    return Err(format!("unexpected extra input {other}\n{USAGE}"));
                }
                input = Some(PathBuf::from(other));
            }
        }
    }
    args.input = input.ok_or_else(|| format!("no input given\n{USAGE}"))?;
    Ok(args)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let opts = FontPreset::MonaspaceArgon.options(args.pixel_size);

    let raw = open_source(&args.input, args.fps)?;
    let adapted = AdaptedSource {
        inner: raw.as_ref(),
        size: Some((args.width, args.height)),
        fit: FitMode::Exact,
        filter: Filter::Lanczos,
        target_fps: args.fps,
        start_s: args.start,
        duration_s: args.duration,
        max_frames: BANK_FRAMES,
    };

    eprintln!(
        "encoding {}x{} at {} fps",
        args.width, args.height, args.fps
    );
    let (world, report) = build_probe_world(&adapted, &opts)?;
    std::fs::write(&args.out, world.to_brz_vec()?)?;

    // Brick mode's cost for the SAME screen and clip, so the comparison the
    // spike exists to make is printed side by side. This is the existing
    // estimator, not a new cost model.
    // `AnimOptions::default()` is `BANK_FRAMES` and no subtitles -- exactly
    // what this probe builds. The estimators take the whole struct so a
    // readout can never be computed from different inputs than the render it
    // describes; see `anim::cost`'s module doc.
    let hex = heightmap::anim::AnimEncoding::Hex.estimate(
        report.width,
        report.height,
        report.frames,
        &heightmap::anim::bricks::AnimOptions::default(),
    );

    println!("wrote {}", args.out.display());
    println!(
        "text mode  {}x{}, {} frames, {} bands (<= {} rows each)",
        report.width, report.height, report.frames, report.bands, report.max_rows_per_band
    );
    println!(
        "  gates {:>8}   wires {:>8}   bricks {:>8}  (main {} + chip {} incl. pins)",
        report.chip_gates,
        report.wires,
        report.main_bricks + report.chip_bricks,
        report.main_bricks,
        report.chip_bricks
    );
    println!(
        "  chars  max band {} of {MAX_COMPONENT_CHARS}, {} per frame at worst ({} glyphs of \
         that are pixels), {} total across the clip",
        report.max_band_chars,
        report.max_frame_chars,
        report.width as usize * report.height as usize * opts.char_repeat,
        report.total_chars
    );
    println!(
        "brick mode (--anim-encoding hex, estimated)\n  gates {:>8}   wires {:>8}   bricks \
         {:>8}   chars {}",
        hex.gates, hex.wires, hex.bricks, hex.chars
    );
    println!(
        "  ratio  gates {:.0}x   wires {:.0}x   bricks {:.0}x",
        hex.gates as f64 / report.chip_gates as f64,
        hex.wires as f64 / report.wires as f64,
        hex.bricks as f64 / (report.main_bricks + report.chip_bricks) as f64,
    );
    Ok(())
}

#[cfg(test)]
#[path = "../tests/wire_integrity.rs"]
mod wire_integrity;

#[cfg(test)]
mod tests {
    use super::*;
    use heightmap::video::Clip;
    use image::Rgba;

    /// A clip whose every pixel is a different colour on every frame -- the
    /// worst case the band bound is derived from, so no test here can pass by
    /// accident on cheap content.
    fn worst_case_clip(w: u32, h: u32, frames: usize) -> Clip {
        let frames = (0..frames)
            .map(|f| {
                RgbaImage::from_fn(w, h, |x, y| {
                    let n = (x + y * w) as usize + f * 7;
                    Rgba([
                        (n % 251) as u8,
                        ((n / 251) % 241) as u8,
                        ((n / 60_491) % 239) as u8,
                        255,
                    ])
                })
            })
            .collect();
        Clip {
            width: w,
            height: h,
            fps: 12.0,
            frames,
        }
    }

    /// The property the whole design rests on: the layout is decided once and
    /// EVERY frame reproduces it. A band whose row range moved between frames
    /// would make a component draw different rows on different frames.
    #[test]
    fn the_band_layout_is_identical_for_every_frame() {
        let opts = TextOptions::default();
        let clip = worst_case_clip(192, 108, 4);
        let plan = plan_bands(192, 108, opts.char_repeat).expect("plan");

        // Rows are covered exactly once, in order, with no gap and no overlap.
        let mut next = 0usize;
        for band in &plan {
            assert_eq!(band.start_row, next, "bands must tile the image in order");
            assert!(band.rows > 0, "an empty band would drive nothing");
            next += band.rows;
        }
        assert_eq!(next, 108, "the bands must cover every row");

        // Frames with wildly different content all encode against that one
        // plan, and every band renders exactly its own rows on every frame --
        // a band whose line count moved would slide the rows below it.
        for (f, frame) in clip.frames.iter().enumerate() {
            for band in &plan {
                let (text, chars) = encode_band(frame, band, &opts, f).expect("encode");
                assert_eq!(chars, text.chars().count());
                // Counted from the separators, not `lines()`: a band whose
                // last row is fully transparent encodes to an empty final
                // line, which `lines()` drops.
                assert_eq!(
                    text.chars().filter(|c| *c == '\n').count() + 1,
                    band.rows,
                    "frame {f}, band at row {}: must render exactly its own rows",
                    band.start_row
                );
            }
        }
    }

    /// The bound is what makes a scan pass unnecessary, so it must actually
    /// hold against the worst content there is.
    #[test]
    fn no_band_of_a_worst_case_frame_exceeds_the_char_limit() {
        let opts = TextOptions::default();
        let clip = worst_case_clip(192, 108, 2);
        let plan = plan_bands(192, 108, opts.char_repeat).expect("plan");
        let mut worst = 0usize;
        for (f, frame) in clip.frames.iter().enumerate() {
            for band in &plan {
                let (_, chars) = encode_band(frame, band, &opts, f).expect("encode");
                assert!(chars <= MAX_COMPONENT_CHARS, "band at row {} overflowed", band.start_row);
                worst = worst.max(chars);
            }
        }
        // Every pixel of a worst-case frame really does emit its own tag, so
        // this must sit near the bound rather than far under it -- otherwise
        // the test is not exercising the worst case at all.
        let bound = worst_case_row_chars(192, opts.char_repeat);
        assert!(
            worst > bound,
            "a multi-row band must exceed a single row's bound ({worst} vs {bound})"
        );
    }

    /// A row too wide for one component must be a clear error NAMING the row,
    /// not a silently truncated render.
    #[test]
    fn a_row_too_wide_for_one_component_names_the_row() {
        // 600 px * (16 + 2) = 10 800 chars, over the 10 000 limit.
        let err = plan_bands(600, 4, 2).expect_err("a 600px row cannot fit one component");
        assert!(err.contains("row 0"), "unexpected error: {err}");
        assert!(err.contains("10800"), "the error must quote the cost: {err}");

        // The boundary itself: the widest row that still fits must plan, and
        // one pixel more must not.
        let widest = MAX_COMPONENT_CHARS / (COLOR_TAG_CHARS + 2);
        plan_bands(widest, 4, 2).expect("the widest fitting row must plan");
        assert!(plan_bands(widest + 1, 4, 2).is_err(), "one pixel over must not");
    }

    /// Each band's array must hold exactly one string per frame -- a short
    /// array would make the clock index off the end mid-loop.
    #[test]
    fn every_band_array_holds_one_entry_per_frame() {
        let opts = TextOptions::default();
        // Small and short: this is about array lengths, not throughput.
        let clip = worst_case_clip(48, 27, 5);
        let plan = plan_bands(48, 27, opts.char_repeat).expect("plan");
        let mut lengths = vec![0usize; plan.len()];
        let mut stream = clip.open().expect("open");
        let mut frames = 0usize;
        while let Some(frame) = stream.next().expect("next") {
            for (bi, band) in plan.iter().enumerate() {
                encode_band(&frame, band, &opts, frames).expect("encode");
                lengths[bi] += 1;
            }
            frames += 1;
        }
        assert_eq!(frames, 5);
        assert!(lengths.iter().all(|n| *n == frames), "{lengths:?}");
    }

    /// The whole build, end to end, through the real wire-integrity harness:
    /// every wire endpoint must resolve to a brick that actually carries the
    /// component it names -- the failure mode that produces a save which opens
    /// fine and does nothing.
    ///
    /// Run at BOTH a small size and the 192x108 the probe is reported at: the
    /// latter puts 54 bands in the chip, whose lattice then spans far enough
    /// to land inner bricks in more than one brick chunk, which is exactly
    /// where a wire's chunk-relative brick index can go wrong.
    #[test]
    fn the_probe_world_writes_and_every_wire_resolves() {
        build_and_validate(64, 36, 3);
        build_and_validate(192, 108, 2);
    }

    fn build_and_validate(w: u32, h: u32, frames: usize) {
        let opts = TextOptions::default();
        let clip = worst_case_clip(w, h, frames);
        let (world, report) = build_probe_world(&clip, &opts).expect("build");

        assert_eq!(report.frames, frames);
        assert!(report.bands >= 1);
        assert_eq!(
            report.main_bricks,
            report.bands + 1,
            "one anchor cube per band plus the chip shell"
        );
        // 3 wires into each Get (ArrayVarRef, Index, Exec) + 1 out to the
        // TextDisplay, plus the detector feed and the clock's own 8.
        assert_eq!(report.wires, 4 * report.bands + 1 + 8);

        // Each band anchors at its OWN image row: one distinct height per
        // band, and the whole column stays within one depth slot. A stack of
        // bands behind a single anchor would show up here as one shared z and
        // a depth span of 2*(bands-1) -- which is the layout that would need
        // a component Offset.Z the game may not honour.
        let anchors: Vec<_> = world.bricks.iter().filter(|b| !b.visible).collect();
        assert_eq!(anchors.len(), report.bands, "one anchor cube per band");
        let xs: Vec<i32> = anchors.iter().map(|b| b.position.x).collect();
        assert!(
            xs.iter().max().unwrap() - xs.iter().min().unwrap() <= 2,
            "anchors must not stack in depth: x spans {:?}..{:?}",
            xs.iter().min(),
            xs.iter().max()
        );
        let mut zs: Vec<i32> = anchors.iter().map(|b| b.position.z).collect();
        zs.sort_unstable();
        zs.dedup();
        assert_eq!(zs.len(), report.bands, "each band must sit at its own height");
        // No negative coordinates on the main grid. (The inner grid's own
        // non-negativity is asserted by `chip::finish` in the pre-publish
        // coordinate space -- `World::add_brick_grid` then shifts the
        // published bricks by `-CHUNK_HALF`, so it cannot be re-checked here.)
        assert!(
            world
                .bricks
                .iter()
                .all(|b| b.position.x >= 0 && b.position.y >= 0 && b.position.z >= 0),
            "main-grid bricks must stay non-negative"
        );

        let path = std::env::temp_dir().join(format!(
            "h2b_text_probe_{}_{w}x{h}.brz",
            std::process::id()
        ));
        std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
        let result = std::panic::catch_unwind(|| super::wire_integrity::assert_wires_valid(&path));
        let _ = std::fs::remove_file(&path);
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }
}
