use image::RgbaImage;
use std::{
    path::{Path, PathBuf},
    result::Result,
};

use crate::util::to_linear_rgb;

// generic heightmap trait returns scalar from X and Y
pub trait Heightmap {
    fn at(&self, x: u32, y: u32) -> u32;
    fn size(&self) -> (u32, u32);
}

// generic colormap trait returns color from X and Y
pub trait Colormap {
    fn at(&self, x: u32, y: u32) -> [u8; 4];
    fn size(&self) -> (u32, u32);
}

// PNG based heightmaps
pub struct HeightmapPNG {
    maps: Vec<RgbaImage>,
    rgba_encoded: bool,
}

// Heightmap lookup
impl Heightmap for HeightmapPNG {
    fn at(&self, x: u32, y: u32) -> u32 {
        if self.rgba_encoded {
            self.maps
                .iter()
                .fold(0, |sum, m| sum + u32::from_be_bytes(m.get_pixel(x, y).0))
        } else {
            self.maps
                .iter()
                .fold(0, |sum, m| sum + m.get_pixel(x, y).0[0] as u32)
        }
    }

    fn size(&self) -> (u32, u32) {
        (self.maps[0].width(), self.maps[0].height())
    }
}

// Heightmap image input
impl HeightmapPNG {
    pub fn new(images: Vec<&PathBuf>, rgba_encoded: bool) -> Result<Self, String> {
        if images.is_empty() {
            return Err("HeightmapPNG requires at least one image".to_string());
        }

        // read in the maps
        let mut maps: Vec<RgbaImage> = vec![];
        for file in images {
            if let Ok(img) = image::open(file) {
                maps.push(img.to_rgba8());
            } else {
                return Err(format!("Could not open image {}", file.display()));
            }
        }

        // check to ensure all images have the same dimensions
        let height = maps[0].height();
        let width = maps[0].width();
        for m in &maps {
            if m.height() != height || m.width() != width {
                return Err("Mismatched heightmap sizes".to_string());
            }
        }

        // return a reference to save on memory
        Ok(HeightmapPNG { maps, rgba_encoded })
    }
}

// A completely flat heightmap
pub struct HeightmapFlat {
    width: u32,
    height: u32,
}

// The heightmap always returns 1... because it's flat
impl Heightmap for HeightmapFlat {
    fn at(&self, _x: u32, _y: u32) -> u32 {
        1
    }

    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

// Flat heightmap just has dimensions
impl HeightmapFlat {
    pub fn new((width, height): (u32, u32)) -> Result<Self, String> {
        // return a reference to save on memory
        Ok(HeightmapFlat { width, height })
    }
}

// PNG based colormap
pub struct ColormapPNG {
    source: RgbaImage,
    lrgb: bool,
}

// Read in a color from X, Y
impl Colormap for ColormapPNG {
    fn at(&self, x: u32, y: u32) -> [u8; 4] {
        if self.lrgb {
            self.source.get_pixel(x, y).0
        } else {
            to_linear_rgb(self.source.get_pixel(x, y).0)
        }
    }

    fn size(&self) -> (u32, u32) {
        (self.source.width(), self.source.height())
    }
}

// Colormap image input
impl ColormapPNG {
    pub fn new(file: impl AsRef<Path>, lrgb: bool) -> Result<Self, String> {
        if let Ok(img) = image::open(&file) {
            Ok(ColormapPNG {
                source: img.to_rgba8(),
                lrgb,
            })
        } else {
            Err(format!("Could not open image {}", file.as_ref().display()))
        }
    }
}
