pub mod map;
pub mod opt;
pub mod text;
pub mod util;

#[cfg(not(target_arch = "wasm32"))]
mod progress_cli;

#[cfg(not(target_arch = "wasm32"))]
use crate::{map::*, opt::*, text::*, util::*};
#[cfg(not(target_arch = "wasm32"))]
use brdb::World;
#[cfg(not(target_arch = "wasm32"))]
use brdb::assets::bricks::{
    PB_DEFAULT_BRICK, PB_DEFAULT_MICRO_BRICK, PB_DEFAULT_SMOOTH_TILE, PB_DEFAULT_STUDDED,
    PB_DEFAULT_TILE,
};
#[cfg(not(target_arch = "wasm32"))]
use clap::clap_app;
#[cfg(not(target_arch = "wasm32"))]
use env_logger::Builder;
#[cfg(not(target_arch = "wasm32"))]
use heightmap::{
    anim::{
        bricks::{AnimOptions, DisplayBrickStyle, build_brick_world},
        cost,
        pack::MAX_FRAMES,
    },
    video::{
        backend::{self, Backend},
        ffmpeg::{DownloadConsent, ensure_ffmpeg},
        scale::{Filter, FitMode, estimated_frame_count, max_frames_error},
        source::{Source, decode, is_animated, is_video_path},
        stream::{AdaptedSource, FrameSource},
    },
};
#[cfg(not(target_arch = "wasm32"))]
use log::{LevelFilter, error, info, warn};
#[cfg(not(target_arch = "wasm32"))]
use std::{boxed::Box, io::Write, path::PathBuf};

/// The CLI is native-only; the web build ships only the GUI bin.
#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    Builder::new()
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .filter(None, LevelFilter::Info)
        .init();

    let matches = clap_app!(heightmap =>
        (version: env!("CARGO_PKG_VERSION"))
        (author: "github.com/Meshiest")
        (about: "Converts heightmap images (PNG/JPG) to Brickadia save files")
        (@arg INPUT: +required +multiple "Input heightmap image files (PNG/JPG)")
        (@arg output: -o --output +takes_value "Output file (BRDB, BRZ)")
        (@arg colormap: -c --colormap +takes_value "Input colormap image (PNG/JPG)")
        (@arg vertical: -v --vertical +takes_value "Vertical scale multiplier (default 1)")
        (@arg size: -s --size +takes_value "Brick stud size (default 1)")
        (@arg cull: --cull "Automatically remove bottom level bricks and fully transparent bricks")
        (@arg tile: --tile "Render bricks as tiles")
        (@arg smooth: --smooth "Render bricks as smooth tiles")
        (@arg micro: --micro "Render bricks as micro bricks")
        (@arg stud: --stud "Render bricks as stud cubes")
        (@arg snap: --snap "Snap bricks to the brick grid")
        (@arg lrgb: --lrgb "Use linear rgb input color instead of sRGB")
        (@arg img: -i --img "Make the heightmap flat and render an image")
        (@arg glow: --glow "Make the heightmap (or animation display) glow at 0 intensity")
        (@arg hdmap: --hdmap "Using a high detail rgb color encoded heightmap")
        (@arg nocollide: --nocollide "Disable brick collision")
        (@arg greedy: --greedy "Use greedy optimization")
        (@arg text: --text "Render the input image as TextDisplay component bricks")
        (@arg fillchar: --("fill-char") +takes_value "Text mode: glyph for opaque pixels (default █)")
        (@arg emptychar: --("empty-char") +takes_value "Text mode: glyph for transparent pixels (default space)")
        (@arg charrepeat: --("char-repeat") +takes_value "Text mode: glyphs emitted per pixel (default 2)")
        (@arg alphathreshold: --("alpha-threshold") +takes_value "Text mode: alpha below this is transparent (default 128)")
        (@arg lineheight: --("line-height-world") +takes_value "Text mode: world units per pixel row / pixel size (default 1)")
        (@arg font: --font +takes_value "Text mode: font preset (monaspace, iosevka, orbitron; default monaspace)")
        (@arg braille: --braille "Text mode: monochrome braille glyphs (8 pixels per character)")
        (@arg blocks: --blocks "Text mode: monochrome quadrant-block glyphs (4 pixels per character)")
        (@arg lumathreshold: --("luma-threshold") +takes_value "Braille/blocks: pixels at least this bright are drawn (default 128)")
        (@arg invert: --invert "Braille/blocks: draw dark pixels instead of bright ones")
        (@arg material: --material +takes_value "Text mode: material (unlit, graffiti, plastic, metallic, glow, translucent, glass; default unlit)")
        (@arg animmode: --("anim-mode") +takes_value "Animation output mode (brick)")
        (@arg animfps: --fps +takes_value "Animation output frame rate (default 10)")
        (@arg animstart: --start +takes_value "Start offset into the source, seconds")
        (@arg animduration: --duration +takes_value "Duration taken from the source, seconds")
        (@arg animmaxframes: --("max-frames") +takes_value "Cap on emitted frames (default 1048560; frames past 65535 spill into extra arrays)")
        (@arg animwidth: --width +takes_value "Target width in pixels")
        (@arg animheight: --height +takes_value "Target height in pixels")
        (@arg animfit: --fit +takes_value "Fit mode (exact, contain, cover; default contain)")
        (@arg animfilter: --filter +takes_value "Resample filter (lanczos, nearest; default lanczos)")
        (@arg externalclock: --("external-clock") "Expose Frame as a chip input instead of running a timer")
        (@arg animbrickstyle: --("brick-style") +takes_value "Animation display-brick style (micro, tile; default micro)")
        (@arg animpixelextent: --("pixel-extent") +takes_value "Animation display-brick half-extent in units (default 1; 1 = smallest, 2 units wide; tile style is always 4 units tall)")
        (@arg yesdownload: --yes "Consent to downloading ffmpeg if it is missing and a video backend needs it")
        (@arg nodownload: --("no-download") "Never download ffmpeg; error instead if it is missing")
        (@arg backend: --backend +takes_value "Video decode backend (auto, rust, ffmpeg; default auto)")
    )
    .get_matches();

    // get files from matches
    let heightmap_files = matches
        .values_of("INPUT")
        .unwrap()
        .map(|s| PathBuf::from(s))
        .collect::<Vec<_>>();
    let colormap_file = matches
        .value_of("colormap")
        .map(PathBuf::from)
        .unwrap_or(heightmap_files[0].clone());
    let out_file = matches
        .value_of("output")
        .unwrap_or("./out.brz")
        .to_string();

    // Task 3 added these; this is where they finally do something.
    // `Ask` is the default because it downgrades to `Never` on a
    // non-terminal stdin, so a headless run errors rather than hanging.
    // Validated here regardless of render mode (cheap, and catches a
    // contradictory pair immediately); only a video render ever actually
    // consults `consent`, via `ensure_ffmpeg`.
    let consent = match (matches.is_present("yesdownload"), matches.is_present("nodownload")) {
        (true, true) => return error!("--yes and --no-download contradict each other"),
        (true, false) => DownloadConsent::Always,
        (false, true) => DownloadConsent::Never,
        (false, false) => DownloadConsent::Ask,
    };

    let backend_choice = match matches
        .value_of("backend")
        .map(|s| s.to_lowercase())
        .as_deref()
    {
        None | Some("auto") => Backend::Auto,
        Some("rust") => Backend::Rust,
        Some("ffmpeg") => Backend::Ffmpeg,
        Some(other) => {
            return error!("unknown --backend '{other}' (auto, rust, ffmpeg)");
        }
    };

    if matches.is_present("animmode") {
        // The BRZ/BRDB string-array encoding this renderer builds on tops
        // out at `MAX_FRAMES` frames; passing an unbounded sentinel here
        // would re-enable an unbounded resampling loop (a fat-fingered --fps
        // could OOM).
        let mode = matches.value_of("animmode").unwrap_or("brick");
        if mode != "brick" {
            return error!(
                "unsupported --anim-mode '{mode}' (only 'brick' is supported; text mode is a later phase)"
            );
        }

        if matches.is_present("colormap") {
            warn!("--anim-mode ignores --colormap");
        }

        let fps = matches
            .value_of("animfps")
            .map(|s| s.parse::<f32>().expect("fps must be a number"))
            .unwrap_or(10.0);
        // Both clamped at 0 here rather than left raw: `FpsStream` clamps
        // them internally the same way, so this changes no render, but it
        // lets the pre-flight frame-count check below mirror the stream's
        // arithmetic exactly instead of approximating it. The GUI clamps
        // these at the same point for the same reason.
        let start = matches
            .value_of("animstart")
            .map(|s| s.parse::<f32>().expect("start must be a number"))
            .unwrap_or(0.0)
            .max(0.0);
        let duration = matches
            .value_of("animduration")
            .map(|s| s.parse::<f32>().expect("duration must be a number").max(0.0));
        let max_frames = matches
            .value_of("animmaxframes")
            .map(|s| s.parse::<usize>().expect("max-frames must be an integer"))
            .unwrap_or(MAX_FRAMES)
            .min(MAX_FRAMES);

        let fit = match matches
            .value_of("animfit")
            .map(|s| s.to_lowercase())
            .as_deref()
        {
            None | Some("contain") => FitMode::Contain,
            Some("exact") => FitMode::Exact,
            Some("cover") => FitMode::Cover,
            Some(other) => {
                return error!("unknown fit mode '{other}' (exact, contain, cover)");
            }
        };
        let filter = match matches
            .value_of("animfilter")
            .map(|s| s.to_lowercase())
            .as_deref()
        {
            None | Some("lanczos") => Filter::Lanczos,
            Some("nearest") => Filter::Nearest,
            Some(other) => {
                return error!("unknown filter '{other}' (lanczos, nearest)");
            }
        };

        // A frame sequence outruns the OS argument limit long before it
        // outruns anything else this renderer cares about. At ~11 characters
        // per relative filename, Windows' 32767-character command line runs
        // dry around 3000 frames -- but a 90-minute clip at 12fps is 65391 of
        // them: inside one bank's 65535 entries, far inside the 1048560-frame
        // overall cap, and ~20x past argv. Naming the directory sidesteps it
        // entirely, so the real limit stays the one that means something.
        //
        // Scoped to anim mode on purpose: a directory of images is a frame
        // sequence, which has no meaning for the heightmap or text paths.
        let heightmap_files = if heightmap_files.len() == 1 && heightmap_files[0].is_dir() {
            let dir = &heightmap_files[0];
            let entries = match std::fs::read_dir(dir) {
                Ok(e) => e,
                Err(e) => return error!("Error reading directory {}: {e}", dir.display()),
            };
            let mut found = Vec::new();
            for entry in entries {
                let path = match entry {
                    Ok(e) => e.path(),
                    Err(e) => return error!("Error reading directory {}: {e}", dir.display()),
                };
                // Filter on extension rather than by attempting a decode: a
                // stray .txt or a nested directory should be skipped in
                // silence, while a genuinely corrupt .png must still be the
                // hard error it becomes below.
                let is_image = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_ascii_lowercase())
                    .is_some_and(|e| {
                        matches!(e.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "tga" | "webp")
                    });
                if is_image && path.is_file() {
                    found.push(path);
                }
            }
            if found.is_empty() {
                return error!("No image files found in directory {}", dir.display());
            }
            // `decode_sequence` re-sorts by natural key (so f_2 precedes
            // f_10); this sort only settles the order for the single-image
            // case below, which never reaches it.
            found.sort();
            info!("Found {} image(s) in {}", found.len(), dir.display());
            found
        } else {
            heightmap_files
        };

        // A video file goes to a decode backend, which pushes fps and
        // scaling into its own filters rather than materialising a Clip --
        // the whole point of this path. The directory and animated-image
        // branches below are unchanged.
        if heightmap_files.len() == 1 && is_video_path(&heightmap_files[0]) {
            // ffmpeg's availability is consulted only where it is actually
            // needed, and `open_video_ensuring` owns that decision so it can
            // be unit tested (see its doc, and this task's report):
            //
            // - a GIF/PNG/frame-sequence render never reaches this branch at
            //   all, so it is never prompted about ffmpeg;
            // - `--backend rust` never consults it -- that backend either
            //   decodes in pure Rust or refuses by name;
            // - `--backend ffmpeg` consults it eagerly, because the user
            //   named the backend and a missing binary should fail fast;
            // - `--backend auto` (the DEFAULT) TRIES the pure-Rust backend
            //   first and only consults ffmpeg once that has failed with
            //   something ffmpeg could actually help with. An earlier version
            //   checked eagerly here, which made a machine without ffmpeg
            //   refuse every video by default -- including CABAC H.264 files
            //   the pure-Rust backend handles perfectly well on its own.
            info!("Opening video {}", heightmap_files[0].display());
            let raw = match backend::open_video_ensuring(
                &heightmap_files[0],
                backend_choice,
                None,
                fit,
                filter,
                None,
                &mut || ensure_ffmpeg(consent),
            ) {
                Ok(s) => s,
                Err(e) => return error!("{e}"),
            };

            // `open_video` does not apply target size/rate to the pure-Rust
            // backend -- `RustVideoSource::open_path` takes no such
            // parameters at all -- so passing `--width`/`--height`/`--fps`
            // straight through to it here would make those flags work on the
            // ffmpeg backend and silently do nothing on the Rust one.
            // `AdaptedSource` is layered over whatever `open_video` returned
            // instead, exactly the way the image-sequence path below layers
            // it over a `Clip`, so every scaling/rate/window flag behaves
            // identically no matter which backend produced the raw frames.
            // `open_video` above is deliberately called with `None`/`None`
            // for size/fps so the raw source stays native and untouched on
            // EITHER backend -- there is exactly one place, this
            // `AdaptedSource`, that ever resizes or resamples.
            let native = raw.info();
            let size = if matches.is_present("animwidth") || matches.is_present("animheight") {
                // A raw `.expect` here (as the pre-existing image path a few
                // hundred lines down still has, deliberately left alone --
                // see this block's own comment) would turn a typo'd
                // `--width abc` into a Rust panic message instead of a clean
                // CLI error; a video render already got this far spawning a
                // decode backend, so this is also further into the run than
                // the image path's equivalent check.
                let target_w = match matches.value_of("animwidth") {
                    Some(s) => match s.parse::<u32>() {
                        Ok(v) => v,
                        Err(e) => return error!("--width must be an integer: {e}"),
                    },
                    None => native.width,
                };
                let target_h = match matches.value_of("animheight") {
                    Some(s) => match s.parse::<u32>() {
                        Ok(v) => v,
                        Err(e) => return error!("--height must be an integer: {e}"),
                    },
                    None => native.height,
                };
                Some((target_w, target_h))
            } else {
                None
            };

            let adapted = AdaptedSource {
                inner: raw.as_ref(),
                size,
                fit,
                filter,
                target_fps: fps,
                start_s: start,
                duration_s: duration,
                max_frames,
            };

            let brick_style = match matches
                .value_of("animbrickstyle")
                .map(|s| s.to_lowercase())
                .as_deref()
            {
                None | Some("micro") => DisplayBrickStyle::Micro,
                Some("tile") => DisplayBrickStyle::SmoothTile,
                Some(other) => {
                    return error!("unknown brick style '{other}' (micro, tile)");
                }
            };
            let pixel_extent = match matches.value_of("animpixelextent") {
                Some(s) => match s.parse::<u16>() {
                    Ok(v) => v,
                    Err(e) => return error!("--pixel-extent must be an integer: {e}"),
                },
                None => 1,
            }
            .max(1);

            let anim_opts = AnimOptions {
                external_clock: matches.is_present("externalclock"),
                brick_style,
                pixel_extent,
                glow: matches.is_present("glow"),
                ..AnimOptions::default()
            };

            // `AdaptedSource::info` folds the resample/window math into the
            // hint whenever the raw source can say its own frame count up
            // front (see its doc) -- a video probed from a real container
            // almost always can. When it genuinely can't, the hint is `None`:
            // there is no pre-flight number to refuse on, but `FpsStream`
            // itself still enforces `max_frames` mid-render (an `Err`, not a
            // silent truncation), so nothing is left unguarded -- only the
            // refusal-before-printing-a-cost-line optimization is unavailable.
            let info = adapted.info();
            if let Some(n) = info.frame_count_hint {
                if n > max_frames {
                    return error!("{}", max_frames_error(max_frames));
                }
            } else {
                warn!(
                    "source did not report a frame count ahead of decode; the cost estimate \
                     below counts 0 frames rather than guess, but max_frames is still enforced \
                     during the render"
                );
            }

            let cost = cost::estimate(
                info.width,
                info.height,
                info.frame_count_hint.unwrap_or(0),
                anim_opts.bank_size,
            );
            info!(
                "Estimated cost: {} pixel(s), {} gate(s), {} wire(s), {} brick(s), {} chunk(s), {} bank(s), {} frame(s)",
                cost.pixels, cost.gates, cost.wires, cost.bricks, cost.chunks, cost.banks, cost.frames
            );

            let world = match build_brick_world(
                &adapted,
                &anim_opts,
                &mut progress_cli::CliProgress::new(),
            ) {
                Ok(w) => w,
                Err(e) => return error!("{e}"),
            };

            info!("Writing Save to {}", out_file);
            if let Err(e) = write_world(&world, &out_file) {
                return error!("{e}");
            }
            return info!("Done!");
        }

        info!("Reading image file(s)");
        let source = if heightmap_files.len() == 1 {
            let input = &heightmap_files[0];
            let bytes = match std::fs::read(input) {
                Ok(b) => b,
                Err(e) => return error!("Error reading file {}: {e}", input.display()),
            };
            if is_animated(&bytes) {
                Source::Animated(bytes)
            } else {
                match image::load_from_memory(&bytes) {
                    Ok(i) => Source::Still(i.to_rgba8()),
                    Err(e) => return error!("Error reading image: {e:?}"),
                }
            }
        } else {
            let mut named = Vec::with_capacity(heightmap_files.len());
            for input in &heightmap_files {
                let img = match image::open(input) {
                    Ok(i) => i.to_rgba8(),
                    Err(e) => return error!("Error reading image {}: {e:?}", input.display()),
                };
                named.push((input.display().to_string(), img));
            }
            Source::Sequence(named)
        };

        let clip = match decode(source, fps) {
            Ok(c) => c,
            Err(e) => return error!("{e}"),
        };

        // Omitted width/height mean "use the clip's own dimensions" -- skip
        // the resize entirely rather than resampling to an identical size.
        let size = if matches.is_present("animwidth") || matches.is_present("animheight") {
            let target_w = matches
                .value_of("animwidth")
                .map(|s| s.parse::<u32>().expect("width must be an integer"))
                .unwrap_or(clip.width);
            let target_h = matches
                .value_of("animheight")
                .map(|s| s.parse::<u32>().expect("height must be an integer"))
                .unwrap_or(clip.height);
            Some((target_w, target_h))
        } else {
            None
        };

        // Resize (if requested) then resample, streamed rather than
        // materialized twice: `AdaptedSource` layers `ResizeStream` under
        // `FpsStream` so frames are scaled before selection, never the other
        // way around.
        let adapted = AdaptedSource {
            inner: &clip,
            size,
            fit,
            filter,
            target_fps: fps,
            start_s: start,
            duration_s: duration,
            max_frames,
        };

        let brick_style = match matches
            .value_of("animbrickstyle")
            .map(|s| s.to_lowercase())
            .as_deref()
        {
            None | Some("micro") => DisplayBrickStyle::Micro,
            Some("tile") => DisplayBrickStyle::SmoothTile,
            Some(other) => {
                return error!("unknown brick style '{other}' (micro, tile)");
            }
        };
        // No upper bound beyond `u16`'s own range: any half-extent tiles
        // flush at `2 * pixel_extent` (see `AnimOptions::pixel_extent`), so
        // there is no value that can trip the overlap check.
        let pixel_extent = matches
            .value_of("animpixelextent")
            .map(|s| s.parse::<u16>().expect("pixel-extent must be an integer"))
            .unwrap_or(1)
            .max(1);

        // Built before the estimate so the readout costs the options the
        // render will actually consume -- `bank_size` in particular decides
        // how many arrays the frames spill across.
        let anim_opts = AnimOptions {
            external_clock: matches.is_present("externalclock"),
            brick_style,
            pixel_extent,
            glow: matches.is_present("glow"),
            ..AnimOptions::default()
        };

        // The resampled frame count isn't knowable from the *stream* (see
        // `AdaptedSource::info`, whose hint is `None` once the rate changes),
        // but it is computable from the source's length plus the scalars --
        // counts and timings only, no frame data. `estimated_frame_count` is
        // pinned to `FpsStream`'s real output by a test sweep, so this is the
        // number the render will actually produce, not an approximation.
        let est_frames =
            estimated_frame_count(clip.frames.len(), clip.fps, fps, start, duration, max_frames);

        // Deliberately checked BEFORE the cost line, and against the
        // unclamped count. Clamping to `max_frames` and printing anyway
        // would report the cap as though it were the real answer and then
        // have the render immediately refuse the same request -- a cost
        // line the CLI itself does not believe. `resample_fps` used to
        // error before this block was ever reached; this restores that
        // ordering now that the refusal happens mid-stream instead. The
        // message comes from `scale::max_frames_error`, the same one
        // `FpsStream` would emit, so the two can never drift.
        if est_frames > max_frames {
            return error!("{}", max_frames_error(max_frames));
        }

        let info = adapted.info();

        let cost = cost::estimate(info.width, info.height, est_frames, anim_opts.bank_size);
        info!(
            "Estimated cost: {} pixel(s), {} gate(s), {} wire(s), {} brick(s), {} chunk(s), {} bank(s), {} frame(s)",
            cost.pixels, cost.gates, cost.wires, cost.bricks, cost.chunks, cost.banks, cost.frames
        );

        let world =
            match build_brick_world(&adapted, &anim_opts, &mut progress_cli::CliProgress::new()) {
                Ok(w) => w,
                Err(e) => return error!("{e}"),
            };

        info!("Writing Save to {}", out_file);
        if let Err(e) = write_world(&world, &out_file) {
            return error!("{e}");
        }
        return info!("Done!");
    }

    if matches.is_present("text") {
        if heightmap_files.len() > 1 {
            warn!(
                "--text uses only the first input image; ignoring {} extra input(s)",
                heightmap_files.len() - 1
            );
        }
        if matches.is_present("colormap") {
            warn!("--text ignores --colormap");
        }

        let preset = match matches
            .value_of("font")
            .map(|s| s.to_lowercase())
            .as_deref()
        {
            None | Some("monaspace") | Some("argon") => FontPreset::MonaspaceArgon,
            Some("iosevka") => FontPreset::IosevkaTerm,
            Some("orbitron") => FontPreset::Orbitron,
            Some(other) => {
                return error!("unknown font preset '{other}' (monaspace, iosevka, orbitron)");
            }
        };
        let material = match matches
            .value_of("material")
            .map(|s| s.to_lowercase())
            .as_deref()
        {
            None | Some("unlit") => TextMaterial::Unlit,
            Some("graffiti") => TextMaterial::Graffiti,
            Some("plastic") => TextMaterial::Plastic,
            Some("metallic") | Some("metal") => TextMaterial::Metallic,
            Some("glow") => TextMaterial::Glow,
            Some("translucent") => TextMaterial::TranslucentPlastic,
            Some("glass") => TextMaterial::Glass,
            Some(other) => {
                return error!(
                    "unknown material '{other}' (unlit, graffiti, plastic, metallic, glow, \
                     translucent, glass)"
                );
            }
        };
        let pixel_size = matches
            .value_of("lineheight")
            .map(|s| {
                s.parse::<f32>()
                    .expect("line-height-world must be a number")
            })
            .unwrap_or(1.0);
        let d = preset.options(pixel_size);
        let text_opts = TextOptions {
            fill_char: char_arg(&matches, "fillchar", d.fill_char),
            empty_char: char_arg(&matches, "emptychar", d.empty_char),
            char_repeat: matches
                .value_of("charrepeat")
                .map(|s| s.parse::<usize>().expect("char-repeat must be an integer"))
                .unwrap_or(d.char_repeat),
            alpha_threshold: matches
                .value_of("alphathreshold")
                .map(|s| s.parse::<u8>().expect("alpha-threshold must be 0-255"))
                .unwrap_or(d.alpha_threshold),
            mode: if matches.is_present("braille") {
                PixelMode::Braille
            } else if matches.is_present("blocks") {
                PixelMode::Blocks
            } else {
                PixelMode::Color
            },
            luma_threshold: matches
                .value_of("lumathreshold")
                .map(|s| s.parse::<u8>().expect("luma-threshold must be 0-255"))
                .unwrap_or(d.luma_threshold),
            invert: matches.is_present("invert"),
            material,
            ..d
        };
        let text_opts = if text_opts.mode == PixelMode::Color {
            text_opts
        } else {
            // mono modes use their own measured component geometry
            let (line_height, kerning, line_offset, pitch_x, pitch_y) =
                mono_geometry(text_opts.mode, pixel_size);
            TextOptions {
                line_height,
                kerning,
                line_offset,
                pitch_x: pitch_x.unwrap_or(text_opts.pitch_x),
                pitch_y,
                ..text_opts
            }
        };
        if text_opts.char_repeat == 0 {
            return error!("char-repeat must be at least 1");
        }

        let input = &heightmap_files[0];
        info!("Reading image file {}", input.display());
        let img = match image::open(input) {
            Ok(i) => i.to_rgba8(),
            Err(e) => return error!("Error reading image: {e:?}"),
        };
        let tiles = match encode_tiles(&img, &text_opts) {
            Ok(t) => t,
            Err(e) => return error!("{e}"),
        };
        info!(
            "Encoded {} tile(s), {} text band(s)",
            tiles.len(),
            tiles.iter().map(|t| t.bands.len()).sum::<usize>()
        );

        let (img_w, img_h) = img.dimensions();
        let mut world = World::new();
        add_text_tiles(&mut world, tiles, &text_opts);
        world.meta.bundle.description = "Text image generated from image file".to_string();
        // prefab pivots/bounds cover the full rendered image so ground
        // placement holds the anchor cubes up in the air
        make_text_prefab(&mut world, img_w, img_h, &text_opts);

        info!("Writing Save to {}", out_file);
        if let Err(e) = write_world(&world, &out_file) {
            return error!("{e}");
        }
        return info!("Done!");
    }

    // output options
    let options = GenOptions {
        size: matches
            .value_of("size")
            .unwrap_or("1")
            .parse::<u16>()
            .expect("Size must be integer")
            * if matches.is_present("micro") { 1 } else { 5 },
        scale: matches
            .value_of("vertical")
            .unwrap_or("1")
            .parse::<u32>()
            .expect("Scale must be integer"),
        cull: matches.is_present("cull"),
        asset: if matches.is_present("micro") {
            PB_DEFAULT_MICRO_BRICK
        } else if matches.is_present("tile") {
            PB_DEFAULT_TILE
        } else if matches.is_present("smooth") {
            PB_DEFAULT_SMOOTH_TILE
        } else if matches.is_present("stud") {
            PB_DEFAULT_STUDDED
        } else {
            PB_DEFAULT_BRICK
        },
        micro: matches.is_present("micro"),
        stud: matches.is_present("stud"),
        snap: matches.is_present("snap"),
        img: matches.is_present("img"),
        glow: matches.is_present("glow"),
        hdmap: matches.is_present("hdmap"),
        lrgb: matches.is_present("lrgb"),
        nocollide: matches.is_present("nocollide"),
        quadtree: true,
        greedy: matches.is_present("greedy"),
    };

    info!("Reading image files");

    // colormap file parsing
    let colormap = match file_ext(&colormap_file)
        .map(|s| s.to_lowercase())
        .as_deref()
    {
        Some("png") | Some("jpg") | Some("jpeg") => {
            match ColormapPNG::new(&colormap_file, options.lrgb) {
                Ok(map) => map,
                Err(err) => {
                    return error!("Error reading colormap: {:?}", err);
                }
            }
        }
        Some(ext) => {
            return error!("Unsupported colormap format '{}'", ext);
        }
        None => {
            return error!("Missing colormap format for '{}'", colormap_file.display());
        }
    };

    // heightmap file parsing
    let heightmap: Box<dyn Heightmap> = if heightmap_files.iter().all(|f| {
        matches!(
            file_ext(f).map(|s| s.to_lowercase()).as_deref(),
            Some("png") | Some("jpg") | Some("jpeg")
        )
    }) {
        if options.img {
            Box::new(HeightmapFlat::new(colormap.size()).unwrap())
        } else {
            match HeightmapPNG::new(heightmap_files.iter().collect(), options.hdmap) {
                Ok(map) => Box::new(map),
                Err(error) => {
                    return error!("Error reading heightmap: {:?}", error);
                }
            }
        }
    } else {
        return error!("Unsupported heightmap format");
    };

    let bricks = gen_opt_heightmap(&*heightmap, &colormap, options, |_| true)
        .expect("error during generation");

    info!("Writing Save to {}", out_file);
    let data = bricks_to_save(bricks);
    if let Err(e) = write_world(&data, &out_file) {
        return error!("{e}");
    }

    info!("Done!");
}

// first char of an arg's value, or the default when absent/empty
#[cfg(not(target_arch = "wasm32"))]
fn char_arg(matches: &clap::ArgMatches, name: &str, default: char) -> char {
    matches
        .value_of(name)
        .and_then(|s| s.chars().next())
        .unwrap_or(default)
}
