use brdb::{BString, Brick, World};
use std::ffi::OsStr;
use std::path::PathBuf;

pub struct GenOptions {
    pub size: u16,
    pub scale: u32,
    pub asset: BString,
    pub cull: bool,
    pub micro: bool,
    pub stud: bool,
    pub snap: bool,
    pub img: bool,
    pub glow: bool,
    pub hdmap: bool,
    pub lrgb: bool,
    pub nocollide: bool,
    pub quadtree: bool,
    pub greedy: bool,
    /// Height threshold above which to generate full solid fill layers (0 = disabled)
    pub gen_full_layers_above_height: u32,
}

impl GenOptions {
    pub fn base_height(&self) -> i32 {
        if self.stud {
            5
        } else if self.micro {
            1
        } else {
            2
        }
    }
}

// convert gamma to linear gamma
pub fn to_linear_gamma(c: u8) -> u8 {
    let cf = (c as f64) / 255.0;
    (if cf > 0.04045 {
        (cf / 1.055 + 0.0521327).powf(2.4) * 255.0
    } else {
        cf / 12.192 * 255.0
    }) as u8
}

// convert sRGB to linear rgb
pub fn to_linear_rgb(rgb: [u8; 4]) -> [u8; 4] {
    [
        to_linear_gamma(rgb[0]),
        to_linear_gamma(rgb[1]),
        to_linear_gamma(rgb[2]),
        rgb[3],
    ]
}

// given an array of bricks, create a save
pub fn bricks_to_save(bricks: Vec<Brick>) -> World {
    let mut world = World::new();
    world.add_bricks(bricks);
    world.meta.bundle.description = "Save generated from heightmap file".to_string();
    world
}

// get extension from filename
#[allow(unused)]
pub fn file_ext(filename: &PathBuf) -> Option<&str> {
    filename.extension().and_then(OsStr::to_str)
}

// write a world to a .brz or .brdb file based on the extension
pub fn write_world(world: &World, out_file: &str) -> Result<(), String> {
    if out_file.to_lowercase().ends_with(".brz") {
        let brz = world
            .to_brz_vec()
            .map_err(|e| format!("failed to encode brz: {e}"))?;
        std::fs::write(out_file, brz).map_err(|e| format!("failed to write file: {e}"))?;
    } else if out_file.to_lowercase().ends_with(".brdb") {
        world
            .write_brdb(out_file)
            .map_err(|e| format!("failed to write file: {e}"))?;
    } else {
        return Err("output file must end with .brz or .brdb".to_string());
    }
    Ok(())
}
