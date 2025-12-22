pub mod map;
pub mod opt;
pub mod util;

use crate::{map::*, opt::*, util::*};
use brdb::assets::bricks::{
    PB_DEFAULT_BRICK, PB_DEFAULT_MICRO_BRICK, PB_DEFAULT_STUDDED, PB_DEFAULT_TILE,
};
use clap::clap_app;
use env_logger::Builder;
use log::{LevelFilter, error, info};
use std::{boxed::Box, io::Write, path::PathBuf};

fn main() {
    Builder::new()
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .filter(None, LevelFilter::Info)
        .init();

    let matches = clap_app!(heightmap =>
        (version: env!("CARGO_PKG_VERSION"))
        (author: "github.com/Meshiest")
        (about: "Converts heightmap png files to Brickadia save files")
        (@arg INPUT: +required +multiple "Input heightmap PNG images")
        (@arg output: -o --output +takes_value "Output file (BRDB, BRZ)")
        (@arg colormap: -c --colormap +takes_value "Input colormap PNG image")
        (@arg vertical: -v --vertical +takes_value "Vertical scale multiplier (default 1)")
        (@arg size: -s --size +takes_value "Brick stud size (default 1)")
        (@arg cull: --cull "Automatically remove bottom level bricks and fully transparent bricks")
        (@arg tile: --tile "Render bricks as tiles")
        (@arg micro: --micro "Render bricks as micro bricks")
        (@arg stud: --stud "Render bricks as stud cubes")
        (@arg snap: --snap "Snap bricks to the brick grid")
        (@arg lrgb: --lrgb "Use linear rgb input color instead of sRGB")
        (@arg img: -i --img "Make the heightmap flat and render an image")
        (@arg glow: --glow "Make the heightmap glow at 0 intensity")
        (@arg hdmap: --hdmap "Using a high detail rgb color encoded heightmap")
        (@arg nocollide: --nocollide "Disable brick collision")
        (@arg greedy: --greedy "Use greedy optimization")
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

    // output options
    let mut options = GenOptions {
        size: matches
            .value_of("size")
            .unwrap_or("1")
            .parse::<u16>()
            .expect("Size must be integer")
            * 5,
        scale: matches
            .value_of("vertical")
            .unwrap_or("1")
            .parse::<u32>()
            .expect("Scale must be integer"),
        cull: matches.is_present("cull"),
        asset: PB_DEFAULT_BRICK,
        tile: matches.is_present("tile"),
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

    if options.tile {
        options.asset = PB_DEFAULT_TILE;
    } else if options.micro {
        options.size /= 5;
        options.asset = PB_DEFAULT_MICRO_BRICK;
    }
    if options.stud {
        options.asset = PB_DEFAULT_STUDDED;
    }

    info!("Reading image files");

    // colormap file parsing
    let colormap = match file_ext(&colormap_file) {
        Some("png") => match ColormapPNG::new(&colormap_file, options.lrgb) {
            Ok(map) => map,
            Err(err) => {
                return error!("Error reading colormap: {:?}", err);
            }
        },
        Some(ext) => {
            return error!("Unsupported colormap format '{}'", ext);
        }
        None => {
            return error!("Missing colormap format for '{}'", colormap_file.display());
        }
    };

    // heightmap file parsing
    let heightmap: Box<dyn Heightmap> =
        if heightmap_files.iter().all(|f| file_ext(f) == Some("png")) {
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
    if out_file.to_lowercase().ends_with(".brz") {
        let brz = match data.to_brz_vec() {
            Ok(b) => b,
            Err(e) => {
                error!("failed to encode brz: {e}");
                return;
            }
        };
        if let Err(e) = std::fs::write(&out_file, brz) {
            error!("failed to write file: {e}");
            return;
        }
    } else if out_file.to_lowercase().ends_with(".brdb") {
        if let Err(e) = data.write_brdb(&out_file) {
            error!("failed to write file: {e}");
            return;
        };
    } else {
        error!("output file must end with .brz or .brdb");
        return;
    }

    info!("Done!");
}
