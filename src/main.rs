pub mod map;
pub mod opt;
pub mod util;
pub mod text;

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
        (@arg glow: --glow "Make the heightmap glow at 0 intensity")
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
        let pixel_size = matches
            .value_of("lineheight")
            .map(|s| s.parse::<f32>().expect("line-height-world must be a number"))
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
            ..d
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
