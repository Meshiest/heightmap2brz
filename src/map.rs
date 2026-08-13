use image::RgbaImage;
use std::{
    path::{Path, PathBuf},
    result::Result,
};

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
        let mut maps: Vec<RgbaImage> = vec![];
        for file in images {
            if let Ok(img) = image::open(file) {
                maps.push(img.to_rgba8());
            } else {
                return Err(format!("Could not open image {}", file.display()));
            }
        }
        Self::from_images(maps, rgba_encoded)
    }

    /// Construct from already-decoded images (web builds have no filesystem).
    pub fn from_images(maps: Vec<RgbaImage>, rgba_encoded: bool) -> Result<Self, String> {
        if maps.is_empty() {
            return Err("HeightmapPNG requires at least one image".to_string());
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
}

/// Read a color from X, Y.
///
/// The code uses the pixel EXACTLY as the image holds it. A save file stores
/// brick colours in the same encoding as an image, thus no conversion is
/// necessary. An sRGB to linear conversion ran here by default before, and
/// `--lrgb` disabled it. That conversion only made each render darker than
/// its colormap.
impl Colormap for ColormapPNG {
    fn at(&self, x: u32, y: u32) -> [u8; 4] {
        self.source.get_pixel(x, y).0
    }

    fn size(&self) -> (u32, u32) {
        (self.source.width(), self.source.height())
    }
}

// Colormap image input
impl ColormapPNG {
    pub fn new(file: impl AsRef<Path>) -> Result<Self, String> {
        if let Ok(img) = image::open(&file) {
            Ok(Self::from_image(img.to_rgba8()))
        } else {
            Err(format!("Could not open image {}", file.as_ref().display()))
        }
    }

    /// Construct from an already-decoded image (web builds have no filesystem).
    pub fn from_image(source: RgbaImage) -> Self {
        ColormapPNG { source }
    }

    /// The decoded image of the colormap.
    ///
    /// The save embeds this image as its preview (`util::save_screenshot`).
    /// The accessor prevents a second decode of the same file.
    pub fn image(&self) -> &RgbaImage {
        &self.source
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A brick has the colour of its colormap pixel.**
    ///
    /// This path applied an sRGB to linear transfer to each pixel by default,
    /// and `--lrgb` disabled it. The transfer made each render darker than its
    /// image. A save file stores brick colours in the same encoding as an
    /// image, thus no conversion is necessary. The transfer stays for the
    /// animation encoders. The last assertion shows that the colormap does not
    /// use it.
    #[test]
    fn colormap_pixels_reach_the_bricks_exactly_as_the_image_holds_them() {
        let mut img = RgbaImage::new(3, 1);
        img.put_pixel(0, 0, image::Rgba([12, 128, 250, 255]));
        img.put_pixel(1, 0, image::Rgba([255, 255, 255, 255]));
        // `--cull` reads the alpha channel, thus alpha must also stay.
        img.put_pixel(2, 0, image::Rgba([0, 0, 0, 128]));
        let map = ColormapPNG::from_image(img);

        assert_eq!(map.at(0, 0), [12, 128, 250, 255]);
        assert_eq!(map.at(1, 0), [255, 255, 255, 255]);
        assert_eq!(map.at(2, 0), [0, 0, 0, 128]);
        assert_eq!(map.size(), (3, 1));

        let converted = crate::util::to_linear_rgb([12, 128, 250, 255]);
        assert_ne!(
            map.at(0, 0),
            converted,
            "the colormap must not use the linear transfer, which gives {converted:?}"
        );
    }
}
