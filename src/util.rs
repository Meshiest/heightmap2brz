use brdb::{BString, Brick, CHUNK_SIZE, ChunkIndex, Collision, World};
use std::collections::HashSet;
use std::ffi::OsStr;
use std::path::PathBuf;

/// How the code makes the SURFACE of the heightmap. This is different from the
/// brick asset that the other modes use.
///
/// [`SurfaceMode::Blocks`] is each renderer that came before the sloped modes.
/// It makes one box for each area, the top face is flat, and
/// `GenOptions::asset` gives the asset. The other two modes replace that step
/// and select their own assets. To them `asset`, `micro`, `stud`, `snap`,
/// `quadtree` and `greedy` have no meaning. The CLI and the GUI both tell the
/// user this, because an option that appears to work but does nothing is
/// worse.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SurfaceMode {
    /// Boxes above boxes, with the quadtree method or the greedy method. This
    /// is the earlier behavior.
    #[default]
    Blocks,
    /// Smooth micro wedge terrain: one grid of shared vertices, and one
    /// calibrated shape for each cell. Refer to `opt::terrain`.
    Terrain,
    /// The Wrapperup rampifier over the height columns: usual ramps, wedges
    /// and corner ramps on the surface. Refer to `opt::rampify`.
    Rampify,
    /// Terraced Brickadia wedge terrain: flat tops everywhere, with convex
    /// and concave outline corners cut at 45 degrees by vertical side
    /// wedges. Refer to `opt::wedge`.
    Wedge,
}

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
    pub nocollide: bool,
    pub quadtree: bool,
    pub greedy: bool,
    /// The surface renderer that runs. The default is
    /// [`SurfaceMode::Blocks`], which is the earlier behavior. A `GenOptions`
    /// that you do not change thus gives the same result as before.
    pub surface: SurfaceMode,
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

    /// The collision values that each renderer makes from `--nocollide`. They
    /// are in one function, so the sloped modes always agree with the other
    /// modes.
    pub fn collision(&self) -> Collision {
        Collision {
            player: !self.nocollide,
            weapon: !self.nocollide,
            interact: !self.nocollide,
            tool: !self.nocollide,
            ..Default::default()
        }
    }
}

/// The units per stud on the brick grid. A 1x1 brick is 10 units on a side.
pub const UNITS_PER_STUD: f64 = 10.0;

/// The real-world inches per brick unit. One unit is one inch, thus one stud
/// is ten inches. The area readout uses this rule.
pub const INCHES_PER_UNIT: f64 = 1.0;

const INCHES_PER_MILE: f64 = 63360.0;
const INCHES_PER_FOOT: f64 = 12.0;

/// The largest half extent of a procedural brick, in units.
///
/// One pixel becomes one brick or more, and [`GenOptions::size`] is the half
/// extent of that brick. This value thus bounds the horizontal scale. The
/// quadtree optimizer stops merging at the same value.
pub const MAX_BRICK_HALF_EXTENT: u16 = 500;

/// The number of chunks the game loads from one save.
///
/// A chunk is a cube of [`CHUNK_SIZE`] units. A save with more chunks than
/// this does not complete its load. [`check_chunk_limit`] refuses to write
/// such a save.
pub const MAX_SAVE_CHUNKS: usize = 100_000;

/// The size of a render, in bricks and in real-world units.
///
/// All values come from the pixel count of the image and from
/// [`GenOptions::size`]. Thus the code can show the size before the render.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Footprint {
    /// The size of the source image, in pixels.
    pub pixels: (u32, u32),
    /// The half extent of one pixel's brick, in units ([`GenOptions::size`]).
    pub half_extent: u16,
    /// The horizontal size of the build, in units.
    pub units: (u64, u64),
    /// The same size in studs. The value is fractional in micro mode, where
    /// one pixel can be one fifth of a stud.
    pub studs: (f64, f64),
    /// The real-world area, at one inch per unit.
    pub sq_miles: f64,
    /// The height of the build at the brightest shade of the heightmap.
    pub max_height_units: u64,
}

const SQ_FEET_PER_SQ_MILE: f64 = 27_878_400.0;

/// Write `n` with a comma between each group of three digits.
///
/// The size readout shows values of six digits or more. Grouped digits are
/// easier to read.
pub fn commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.char_indices() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Write a stud count. Show one decimal only below 10 studs.
///
/// Micro mode can put one fifth of a stud in a pixel. Larger counts do not
/// need tenths.
fn studs_text(v: f64) -> String {
    if v >= 10.0 {
        commas(v.round() as u64)
    } else {
        format!("{v:.1}")
    }
}

/// Write a real-world length. Use miles, or feet below one tenth of a mile.
fn length_text(units: u64) -> String {
    let inches = units as f64 * INCHES_PER_UNIT;
    let miles = inches / INCHES_PER_MILE;
    if miles >= 0.1 {
        format!("{miles:.2} mi")
    } else {
        format!("{} ft", commas((inches / INCHES_PER_FOOT).round() as u64))
    }
}

impl Footprint {
    /// `1,024 x 1,024 px -> 5,120 x 5,120 studs`
    ///
    /// The line shows studs only. The unit count is ten times the stud count.
    /// The log shows it instead.
    pub fn size_text(&self) -> String {
        format!(
            "{} x {} px  ->  {} x {} studs",
            commas(self.pixels.0 as u64),
            commas(self.pixels.1 as u64),
            studs_text(self.studs.0),
            studs_text(self.studs.1),
        )
    }

    /// `0.81 mi x 0.81 mi, 0.66 sq mi`
    ///
    /// The values use one inch per unit. The row tooltip gives that rule. The
    /// line does not repeat it.
    pub fn real_text(&self) -> String {
        format!(
            "{} x {}, {}",
            length_text(self.units.0),
            length_text(self.units.1),
            self.area_text(),
        )
    }

    /// Write the real-world area. Use square miles, or square feet below one
    /// tenth of a square mile.
    ///
    /// Most heightmaps cover less than one mile. "0.03 sq mi" gives less
    /// information than the equivalent 728,000 sq ft.
    pub fn area_text(&self) -> String {
        if self.sq_miles >= 0.1 {
            format!("{:.2} sq mi", self.sq_miles)
        } else {
            format!(
                "{} sq ft",
                commas((self.sq_miles * SQ_FEET_PER_SQ_MILE).round() as u64)
            )
        }
    }

    /// `up to 2,550 studs tall (2,125 ft)`
    ///
    /// The line uses the same two units as the horizontal lines.
    pub fn height_text(&self) -> String {
        format!(
            "up to {} studs tall ({})",
            studs_text(self.max_height_units as f64 / UNITS_PER_STUD),
            length_text(self.max_height_units),
        )
    }

    /// The real-world width and length in miles, at one inch per unit.
    pub fn miles(&self) -> (f64, f64) {
        (
            self.units.0 as f64 * INCHES_PER_UNIT / INCHES_PER_MILE,
            self.units.1 as f64 * INCHES_PER_UNIT / INCHES_PER_MILE,
        )
    }

    /// The height of the highest point in feet, at one inch per unit.
    pub fn max_height_feet(&self) -> f64 {
        self.max_height_units as f64 * INCHES_PER_UNIT / INCHES_PER_FOOT
    }

    /// Show if one pixel is larger than one brick can be.
    ///
    /// The CLI and the GUI do not clamp the horizontal scale to the limit.
    /// Both report this condition instead.
    pub fn over_brick_limit(&self) -> bool {
        self.half_extent > MAX_BRICK_HALF_EXTENT
    }

    /// `each pixel is 1,200 units across, more than the 1,000 ...`
    pub fn brick_limit_text(&self) -> String {
        format!(
            "each pixel is {} units across, more than the {} of one brick. The game can \
             refuse bricks of this size",
            commas(self.half_extent as u64 * 2),
            commas(MAX_BRICK_HALF_EXTENT as u64 * 2),
        )
    }
}

/// Calculate the size that an image of `pixels` builds at this scale.
///
/// `size` is [`GenOptions::size`], the half extent of one pixel in units. One
/// pixel thus covers `2 * size` units in each brick mode. `scale` is
/// [`GenOptions::scale`], the height in units of one shade of grey.
/// `max_level` is the brightest shade of the heightmap, 255 for an 8-bit map.
pub fn footprint(pixels: (u32, u32), size: u16, scale: u32, max_level: u32) -> Footprint {
    let units = (
        pixels.0 as u64 * 2 * size as u64,
        pixels.1 as u64 * 2 * size as u64,
    );
    let max_height_units = scale as u64 * max_level as u64;

    Footprint {
        pixels,
        half_extent: size,
        units,
        studs: (
            units.0 as f64 / UNITS_PER_STUD,
            units.1 as f64 / UNITS_PER_STUD,
        ),
        sq_miles: (units.0 as f64 * INCHES_PER_UNIT / INCHES_PER_MILE)
            * (units.1 as f64 * INCHES_PER_UNIT / INCHES_PER_MILE),
        max_height_units,
    }
}

/// Count the save chunks that a set of bricks occupies.
///
/// The count uses `Position::to_relative`, the same function that `brdb` uses
/// to put a brick into a chunk. The two counts thus agree.
pub fn count_chunks(bricks: &[Brick]) -> usize {
    bricks
        .iter()
        .map(|b| b.position.to_relative().0)
        .collect::<HashSet<ChunkIndex>>()
        .len()
}

/// Refuse a save that the game cannot load. Return the chunk count.
///
/// The function returns the count because the caller must not read all the
/// bricks a second time.
///
/// **Only the completed bricks give this count.** The optimizer decides how
/// many bricks a render makes and where their centers are. A map can cover
/// 100 chunks with 12 merged bricks. Thus the code does not estimate the
/// count before the render. It does the check at save time.
///
/// The count increases with the AREA of the build, not with its detail. Thus
/// the message tells the user to decrease the horizontal scale or to use a
/// smaller image.
pub fn check_chunk_limit(bricks: &[Brick]) -> Result<usize, String> {
    let chunks = count_chunks(bricks);
    if chunks > MAX_SAVE_CHUNKS {
        return Err(format!(
            "this save has {} chunks, more than the {} that the game loads from one save. It \
             would not complete its load. Each chunk is a cube of {CHUNK_SIZE} units. Decrease \
             the horizontal scale (--size / Horizontal Scale) or the vertical scale \
             (--vertical / Vertical Size), or use a smaller heightmap",
            commas(chunks as u64),
            commas(MAX_SAVE_CHUNKS as u64),
        ));
    }
    Ok(chunks)
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

/// [`to_linear_gamma`] over an RGBA pixel, alpha untouched.
///
/// The colormap path does NOT use this: brick colours in a save file are
/// stored in the same encoding an image is, so converting only darkened every
/// render away from the colormap it was given. It stays for the animation
/// encoders' `--srgb-to-linear`, which converts frame pixels for a different
/// reason (a hex-encoded gate reads its colour as linear).
pub fn to_linear_rgb(rgb: [u8; 4]) -> [u8; 4] {
    [
        to_linear_gamma(rgb[0]),
        to_linear_gamma(rgb[1]),
        to_linear_gamma(rgb[2]),
        rgb[3],
    ]
}

/// The standard sRGB -> linear transfer for one 8-bit channel, as an `f32` in
/// `0.0..=1.0`, with NO `u8` round trip.
///
/// [`to_linear_gamma`] cannot be used where a float result is wanted, and the
/// difference is not cosmetic:
///
///   1. **It re-quantizes.** It multiplies back by 255 and truncates to `u8`,
///      so every linear value below `1/255` collapses to zero. sRGB 0..=12
///      (about 5% of the encoding range, and the whole of a dark scene's
///      shadow detail) all become linear 0 -- crushed to pure black with no
///      way to tell those inputs apart afterwards.
///   2. **Its constants are a different curve.** It computes
///      `(cf / 1.055 + 0.0521327)^2.4` and `cf / 12.192`, not the standard
///      `((cf + 0.055) / 1.055)^2.4` and `cf / 12.92`. Those agree to within
///      a `u8` step, which is why the older path never noticed, but they are
///      not the same function.
///
/// This is the standard piecewise sRGB EOTF: linear below the 0.04045 knee,
/// a 2.4 power law above it. Kept alongside the `u8` version rather than
/// replacing it -- the heightmap path's `--lrgb` and the hex animation path's
/// `--srgb-to-linear` both feed 8-bit sinks and their output must not shift.
pub fn srgb_to_linear_f32(c: u8) -> f32 {
    let cf = c as f32 / 255.0;
    if cf <= 0.04045 {
        cf / 12.92
    } else {
        ((cf + 0.055) / 1.055).powf(2.4)
    }
}

/// Every byte value as two uppercase ASCII hex digits, back to back:
/// `"000102...FDFEFF"`.
static HEX_PAIR_BYTES: [u8; 512] = {
    let mut t = [0u8; 512];
    let digits = *b"0123456789ABCDEF";
    let mut i = 0;
    while i < 256 {
        t[i * 2] = digits[i >> 4];
        t[i * 2 + 1] = digits[i & 0xF];
        i += 1;
    }
    t
};

/// [`HEX_PAIR_BYTES`] as a `&str`, so callers can `push_str` a pair straight
/// into a `String` with no UTF-8 validation pass and no `unsafe`.
static HEX_PAIRS: &str = match std::str::from_utf8(&HEX_PAIR_BYTES) {
    Ok(s) => s,
    Err(_) => panic!("the table is built from ASCII hex digits only"),
};

/// `b` as exactly two uppercase hex digits -- what `format!("{b:02X}")`
/// produces, without the formatting machinery.
///
/// This exists to keep `write!`/`format!` out of per-pixel paths. Formatting
/// six characters that are a pure function of three bytes spins up a
/// `Formatter`, three dynamically-dispatched `UpperHex` calls and all of
/// width/fill/sign handling; at 192x108 over 17,280 frames the hex animation
/// packer alone did that 358 million times, and it measured as that packer's
/// single dominant cost (removing it was worth ~4.4x on its own).
///
/// The output is byte-identical to the `format!` it replaces: uppercase,
/// always two digits, no separator, no prefix.
/// `the_hex_table_is_exactly_the_format_string_it_replaces` pins all 256
/// entries against `format!` itself, so the two cannot drift.
#[inline]
pub fn hex_pair(b: u8) -> &'static str {
    let i = b as usize * 2;
    &HEX_PAIRS[i..i + 2]
}

/// [`to_linear_gamma`] evaluated for every possible input, computed once.
///
/// The transfer is an `f64` `powf(2.4)` per channel -- tens of nanoseconds --
/// and the animation packers apply it per channel, per pixel, per frame. It is
/// a pure function of one `u8`, so 256 entries cover its whole domain exactly;
/// `the_u8_linear_table_is_exactly_the_function` pins every entry against the
/// function itself, so this is a speed change and never a colour change.
pub fn linear_gamma_table() -> &'static [u8; 256] {
    static TABLE: std::sync::OnceLock<[u8; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|i| to_linear_gamma(i as u8)))
}

/// [`srgb_to_linear_f32`] evaluated for every possible input, computed once.
///
/// Same reasoning as [`linear_gamma_table`], for the float transfer the
/// colour-array path uses: a `powf` per channel per pixel per frame, over a
/// domain of 256 values. Pinned entry for entry by
/// `the_f32_linear_table_is_exactly_the_function`.
pub fn srgb_to_linear_f32_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|i| srgb_to_linear_f32(i as u8)))
}

/// [`srgb_to_linear_f32`] over an RGBA pixel: R, G and B get the transfer,
/// A is only rescaled to `0.0..=1.0`.
///
/// Alpha is deliberately NOT transferred. It is a coverage fraction, not a
/// perceptual quantity, so it is already linear in every format that stores
/// it -- putting it through the EOTF would darken every partially transparent
/// pixel for no reason. This mirrors [`to_linear_rgb`], which passes `rgb[3]`
/// through untouched for the same reason.
pub fn to_linear_rgb_f32(rgba: [u8; 4]) -> [f32; 4] {
    [
        srgb_to_linear_f32(rgba[0]),
        srgb_to_linear_f32(rgba[1]),
        srgb_to_linear_f32(rgba[2]),
        rgba[3] as f32 / 255.0,
    ]
}

/// The longest edge of the `Meta/Screenshot.jpg` of a save.
///
/// The game writes screenshots of 1280x720 pixels. That size is its render
/// resolution, not a limit. This constant bounds the long edge only and keeps
/// the shape of the map.
const SCREENSHOT_EDGE: u32 = 1280;

/// The JPEG quality of the preview. The game shows the preview at a few
/// hundred pixels. A higher quality gives no visible improvement.
const SCREENSHOT_QUALITY: u8 = 85;

/// Encode the `Meta/Screenshot.jpg` of a generated save.
///
/// The game shows a grid of previews. A generated save has no render of
/// itself, but the source map is a view of the build from above. Use the
/// colormap, because it shows the colors of the build. Use the heightmap when
/// there is no colormap. The render makes the same selection.
///
/// The caller writes this preview for each heightmap save, world or prefab.
/// The preview was applied only with `--prefab` before. Thus a GUI save, which
/// is always a world bundle, had no preview.
///
/// The function keeps the shape of the map. It does not pad the map into the
/// 16:9 shape of a game screenshot, because a heightmap is usually square.
pub fn save_screenshot(source: &image::RgbaImage) -> Result<Vec<u8>, String> {
    use image::{DynamicImage, codecs::jpeg::JpegEncoder, imageops};

    let (w, h) = (source.width().max(1), source.height().max(1));
    let scale = (SCREENSHOT_EDGE as f32 / w as f32).min(SCREENSHOT_EDGE as f32 / h as f32);
    let fitted = if scale >= 1.0 {
        // Use Nearest to enlarge. A heightmap is pixel art, and a smooth
        // filter removes the pixel grid.
        DynamicImage::ImageRgba8(source.clone()).resize_exact(
            (w as f32 * scale).round().max(1.0) as u32,
            (h as f32 * scale).round().max(1.0) as u32,
            imageops::FilterType::Nearest,
        )
    } else {
        DynamicImage::ImageRgba8(source.clone()).resize(
            SCREENSHOT_EDGE,
            SCREENSHOT_EDGE,
            imageops::FilterType::Lanczos3,
        )
    };

    // JPEG has no alpha channel. An RGBA image causes an encode error, thus
    // remove the alpha channel here.
    let rgb = fitted.to_rgb8();
    let mut out = std::io::Cursor::new(Vec::new());
    JpegEncoder::new_with_quality(&mut out, SCREENSHOT_QUALITY)
        .encode_image(&rgb)
        .map_err(|e| format!("failed to encode the save preview: {e}"))?;
    Ok(out.into_inner())
}

/// Create a save from an array of bricks.
///
/// The description holds the brick count. The game shows the description next
/// to the preview. The count tells the user if the build is loadable, and only
/// the render knows it.
pub fn bricks_to_save(bricks: Vec<Brick>) -> World {
    let mut world = World::new();
    let count = bricks.len();
    world.add_bricks(bricks);
    world.meta.bundle.description = format!(
        "Save generated from heightmap file\n{} bricks",
        commas(count as u64)
    );
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

#[cfg(test)]
mod tests {
    use super::*;
    use brdb::{BrickSize, BrickType, Position, assets::bricks::PB_DEFAULT_BRICK};

    /// A 1024x1024 map at the default scale, in each unit of the readout.
    ///
    /// One pixel is `2 * size` units wide, because `size` is a HALF extent.
    /// The full readout depends on this rule. At `--size 1`, one stud,
    /// `GenOptions::size` is 5. One pixel is thus 10 units, or one stud.
    #[test]
    fn a_stud_per_pixel_builds_one_stud_per_pixel() {
        let plan = footprint((1024, 1024), 5, 1, 255);
        assert_eq!(plan.units, (10_240, 10_240));
        assert_eq!(plan.studs, (1024.0, 1024.0));
        // 10,240 inches is 853.3 ft. The area is that value squared.
        let (mi_x, _) = plan.miles();
        assert!((mi_x - 0.16162).abs() < 1e-4, "{mi_x} miles across");
        assert!((plan.sq_miles - 0.026121).abs() < 1e-5, "{} sq mi", plan.sq_miles);
    }

    /// Micro mode counts MICRO units, not studs. The same slider value thus
    /// builds one fifth of the map. For this reason the readout uses
    /// `GenOptions::size` and not the slider.
    #[test]
    fn micro_units_build_a_fifth_of_what_the_same_number_of_studs_does() {
        let micro = footprint((1024, 1024), 1, 1, 255);
        let studs = footprint((1024, 1024), 5, 1, 255);
        assert_eq!(micro.units, (2048, 2048));
        assert_eq!(micro.studs, (204.8, 204.8));
        assert_eq!(studs.units.0, micro.units.0 * 5);
    }

    /// The height is the vertical scale multiplied by the brightest shade.
    /// The readout gives it in feet, at one inch per unit.
    #[test]
    fn the_tallest_column_is_the_vertical_scale_times_the_brightest_shade() {
        let plan = footprint((16, 16), 5, 100, 255);
        assert_eq!(plan.max_height_units, 25_500);
        assert!((plan.max_height_feet() - 2125.0).abs() < 1e-6);
    }

    fn brick_at(x: i32, y: i32, z: i32) -> Brick {
        Brick {
            asset: BrickType::Procedural {
                asset: PB_DEFAULT_BRICK,
                size: BrickSize::new(5, 5, 6),
            },
            position: Position::new(x, y, z),
            ..Default::default()
        }
    }

    /// The count uses the same rule as the writer. A brick belongs to the
    /// CHUNK_SIZE cube that contains its position. The brick count is not
    /// related.
    #[test]
    fn bricks_sharing_a_chunk_are_counted_once() {
        let same = vec![
            brick_at(0, 0, 0),
            brick_at(10, 20, 30),
            brick_at(CHUNK_SIZE - 1, 0, 0),
        ];
        assert_eq!(count_chunks(&same), 1);

        let spread = vec![
            brick_at(0, 0, 0),
            brick_at(CHUNK_SIZE, 0, 0),
            brick_at(0, CHUNK_SIZE, 0),
            brick_at(0, 0, CHUNK_SIZE),
            // Negative coordinates give different chunks. The grid is
            // signed and has its center at the origin.
            brick_at(-1, 0, 0),
        ];
        assert_eq!(count_chunks(&spread), 5);
    }

    /// The function must refuse above the limit and accept at the limit.
    #[test]
    fn a_save_over_the_chunk_limit_is_refused_before_it_is_written() {
        // Use a 400x250 grid of chunks, not one row. `ChunkIndex` holds
        // each axis in an i16. 100,000 chunks on one axis thus overflow and
        // collide.
        let at_limit: Vec<Brick> = (0..MAX_SAVE_CHUNKS)
            .map(|i| brick_at((i % 400) as i32 * CHUNK_SIZE, (i / 400) as i32 * CHUNK_SIZE, 0))
            .collect();
        assert_eq!(count_chunks(&at_limit), MAX_SAVE_CHUNKS);
        assert_eq!(check_chunk_limit(&at_limit), Ok(MAX_SAVE_CHUNKS));

        let mut over = at_limit;
        over.push(brick_at(0, 0, CHUNK_SIZE));
        let err = check_chunk_limit(&over).expect_err("one chunk above the limit is refused");
        assert!(
            err.contains("100001") || err.contains("100,001"),
            "the message must give the count: {err}"
        );
        assert!(
            err.contains("--size"),
            "the message must name the option that corrects it: {err}"
        );
    }

    /// The CLI and the GUI permit a pixel that is wider than one brick. The
    /// code must report this condition.
    #[test]
    fn a_pixel_too_wide_to_be_one_brick_is_reported() {
        let ok = footprint((16, 16), MAX_BRICK_HALF_EXTENT, 1, 255);
        assert!(!ok.over_brick_limit());

        let over = footprint((16, 16), MAX_BRICK_HALF_EXTENT + 1, 1, 255);
        assert!(over.over_brick_limit());
        // Give the full width, which the user sees in the game. Do not give
        // the half extent, which is how the option is stored.
        assert!(
            over.brick_limit_text().contains("1,002 units"),
            "{}",
            over.brick_limit_text()
        );
    }

    /// The readout has three short lines. The log gives the unit count. The
    /// tooltip gives the one inch per unit rule. Repetition of these on each
    /// line hides the values.
    #[test]
    fn the_readout_lines_carry_studs_and_real_world_size_and_nothing_else() {
        let plan = footprint((1024, 1024), 5, 150, 255);

        assert_eq!(plan.size_text(), "1,024 x 1,024 px  ->  1,024 x 1,024 studs");
        assert_eq!(plan.real_text(), "0.16 mi x 0.16 mi, 728,178 sq ft");
        // Studs and a real-world length, as in the two lines above. Not the
        // 38,250 units of the stored height. This build is more than half a
        // mile high, thus the line changes to miles at the same limit as the
        // horizontal lines.
        assert_eq!(plan.height_text(), "up to 3,825 studs tall (0.60 mi)");
        // A lower build stays in feet.
        assert_eq!(
            footprint((16, 16), 5, 10, 255).height_text(),
            "up to 255 studs tall (213 ft)"
        );

        for line in [plan.size_text(), plan.real_text(), plan.height_text()] {
            assert!(!line.contains("units"), "{line}");
            assert!(!line.contains("inch"), "{line}");
        }
    }

    /// `Meta/Screenshot.jpg` must be a JPEG and must keep the shape of the
    /// map.
    #[test]
    fn a_save_preview_is_a_jpeg_that_keeps_the_maps_shape() {
        // The source is two times as wide as it is high. A change to 16:9,
        // or to a square, thus changes the size in the result.
        let source = image::RgbaImage::from_pixel(2048, 1024, image::Rgba([10, 20, 30, 255]));
        let jpg = save_screenshot(&source).expect("encodes");
        assert_eq!(&jpg[..2], &[0xFF, 0xD8], "Screenshot.jpg must be a JPEG");

        let shot = image::load_from_memory(&jpg).expect("the game must decode it");
        assert_eq!((shot.width(), shot.height()), (1280, 640), "2:1 stays 2:1");

        // A map that is smaller than the box is enlarged to fill it.
        let small = image::RgbaImage::from_pixel(64, 64, image::Rgba([9, 9, 9, 255]));
        let shot = image::load_from_memory(&save_screenshot(&small).expect("encodes")).unwrap();
        assert_eq!((shot.width(), shot.height()), (1280, 1280));
    }

    /// The game shows the bundle description next to the preview. The
    /// description must give the brick count with grouped digits, because the
    /// count can be more than one million.
    #[test]
    fn the_save_description_states_the_brick_count() {
        let world = bricks_to_save(vec![brick_at(0, 0, 0); 1_234_567]);
        assert!(
            world.meta.bundle.description.contains("1,234,567 bricks"),
            "{}",
            world.meta.bundle.description
        );
        assert_eq!(bricks_to_save(vec![]).meta.bundle.description.lines().count(), 2);
    }

    #[test]
    fn big_numbers_are_grouped_so_they_can_be_read() {
        assert_eq!(commas(0), "0");
        assert_eq!(commas(999), "999");
        assert_eq!(commas(1_000), "1,000");
        assert_eq!(commas(51_200), "51,200");
        assert_eq!(commas(1_234_567), "1,234,567");
    }

    /// Most heightmaps cover less than one mile. An area readout in square
    /// miles only would show "0.00 sq mi" for almost every render.
    #[test]
    fn a_small_build_reports_feet_rather_than_a_rounded_off_zero() {
        let small = footprint((64, 64), 5, 1, 255);
        assert!(small.sq_miles < 0.1);
        assert!(
            small.area_text().ends_with("sq ft"),
            "{}",
            small.area_text()
        );
        assert!(small.real_text().contains(" ft"), "{}", small.real_text());

        let big = footprint((8192, 8192), 5, 1, 255);
        assert_eq!(big.area_text(), "1.67 sq mi");
        assert!(big.real_text().contains("1.29 mi"), "{}", big.real_text());
    }

    /// The 0.04045 knee, in 8-bit terms: `0.04045 * 255 = 10.31`, so 10 is the
    /// last sample on the linear segment and 11 the first on the power
    /// segment. Both arms are exercised by name so a swapped comparison (`<`
    /// vs `>`, or the two arms transposed) cannot pass.
    const KNEE_LAST_LINEAR: u8 = 10;
    const KNEE_FIRST_POWER: u8 = 11;

    #[test]
    fn the_endpoints_are_exactly_zero_and_one() {
        assert_eq!(srgb_to_linear_f32(0), 0.0, "black must be exactly linear 0");
        // Exactly 1.0, not 0.9999: `(1.0 + 0.055) / 1.055` is 1.0 to the last
        // bit, and 1^2.4 is 1. A white pixel arriving as 0.999 would be a
        // visible, permanent dimming of the whole render.
        assert_eq!(srgb_to_linear_f32(255), 1.0, "white must be exactly linear 1");
    }

    #[test]
    fn the_linear_segment_below_the_knee_is_the_1_over_12_92_slope() {
        for c in [1u8, 5, KNEE_LAST_LINEAR] {
            let expected = (c as f32 / 255.0) / 12.92;
            assert!(
                (srgb_to_linear_f32(c) - expected).abs() < 1e-9,
                "sRGB {c} must take the linear arm: got {}, want {expected}",
                srgb_to_linear_f32(c)
            );
        }
    }

    #[test]
    fn the_power_segment_above_the_knee_is_the_2_4_power_law() {
        for c in [KNEE_FIRST_POWER, 128, 254] {
            let cf = c as f32 / 255.0;
            let expected = ((cf + 0.055) / 1.055).powf(2.4);
            assert!(
                (srgb_to_linear_f32(c) - expected).abs() < 1e-9,
                "sRGB {c} must take the power arm: got {}, want {expected}",
                srgb_to_linear_f32(c)
            );
        }
    }

    /// The curve must not step at the knee. The two arms of the sRGB transfer
    /// are chosen to meet there, so a wrong constant in either one shows up as
    /// a discontinuity even though both arms would still look individually
    /// plausible.
    #[test]
    fn the_two_arms_meet_at_the_knee_without_a_step() {
        let below = srgb_to_linear_f32(KNEE_LAST_LINEAR);
        let above = srgb_to_linear_f32(KNEE_FIRST_POWER);
        let one_step = 1.0 / 255.0 / 12.92; // the linear arm's own step size
        assert!(above > below, "the transfer must be monotonic across the knee");
        assert!(
            above - below < 2.0 * one_step,
            "the arms must meet at the knee: {below} -> {above} is a step of {}, far larger \
             than the {one_step} the linear arm advances per code",
            above - below
        );
    }

    /// The whole reason this exists rather than reusing [`to_linear_gamma`]:
    /// the `u8` version collapses every linear value below `1/255` to zero, so
    /// a sixth of the encoding range decodes to pure black and is
    /// indistinguishable afterwards. The float version must keep them apart.
    #[test]
    fn shadow_detail_survives_instead_of_being_crushed_to_zero() {
        let crushed: Vec<u8> = (1u8..=12).filter(|c| to_linear_gamma(*c) == 0).collect();
        assert!(
            crushed.len() >= 10,
            "this test is only meaningful if the u8 path really does crush shadows; it \
             crushed {crushed:?}"
        );
        let mut previous = srgb_to_linear_f32(0);
        for c in crushed {
            let v = srgb_to_linear_f32(c);
            assert!(v > previous, "sRGB {c} must stay distinct from {previous}, got {v}");
            previous = v;
        }
    }

    #[test]
    fn every_input_lands_inside_the_unit_range_and_never_decreases() {
        let mut previous = -1.0f32;
        for c in 0..=255u8 {
            let v = srgb_to_linear_f32(c);
            assert!((0.0..=1.0).contains(&v), "sRGB {c} decoded to {v}, outside 0..=1");
            assert!(v > previous, "the transfer must be strictly increasing at {c}");
            previous = v;
        }
    }

    /// Alpha is a coverage fraction, already linear -- it must be rescaled, not
    /// transferred. Putting it through the EOTF would darken every partially
    /// transparent pixel (128 would arrive as ~0.216 instead of ~0.502).
    #[test]
    fn alpha_is_rescaled_but_not_transferred() {
        let [r, g, b, a] = to_linear_rgb_f32([128, 128, 128, 128]);
        assert_eq!(r, srgb_to_linear_f32(128));
        assert_eq!(g, r);
        assert_eq!(b, r);
        assert!(
            (a - 128.0 / 255.0).abs() < 1e-9,
            "alpha must be a plain 0..=1 rescale, got {a}"
        );
        assert!(a > r, "a transferred alpha would be much darker than a rescaled one");
        assert_eq!(to_linear_rgb_f32([0, 0, 0, 255])[3], 1.0);
        assert_eq!(to_linear_rgb_f32([0, 0, 0, 0])[3], 0.0);
    }

    /// The channels must not be transposed. Three same-typed values in a row
    /// is exactly the shape a copy-paste swaps, and a swap renders a
    /// perfectly valid save with its colours rotated.
    #[test]
    fn each_channel_reaches_its_own_slot() {
        let out = to_linear_rgb_f32([255, 0, 0, 255]);
        assert_eq!(out[0], 1.0, "red must land in R");
        assert_eq!((out[1], out[2]), (0.0, 0.0));
        let out = to_linear_rgb_f32([0, 255, 0, 255]);
        assert_eq!(out[1], 1.0, "green must land in G");
        assert_eq!((out[0], out[2]), (0.0, 0.0));
        let out = to_linear_rgb_f32([0, 0, 255, 255]);
        assert_eq!(out[2], 1.0, "blue must land in B");
        assert_eq!((out[0], out[1]), (0.0, 0.0));
    }

    /// The hex table must be exactly what the `format!`/`write!` it replaced
    /// produced, for every one of the 256 byte values -- uppercase, always two
    /// digits, no separator. Checked against `format!` itself rather than
    /// against a hand-written expectation, so it cannot drift from the
    /// formatting it stands in for.
    #[test]
    fn the_hex_table_is_exactly_the_format_string_it_replaces() {
        for b in 0..=255u8 {
            assert_eq!(hex_pair(b), format!("{b:02X}"), "byte {b}");
        }
        assert_eq!(HEX_PAIRS.len(), 512);
        assert!(
            HEX_PAIRS.is_ascii(),
            "anim::pack::slice_of byte-slices these offsets directly, which is only sound \
             because everything written into a frame string is ASCII"
        );
    }

    /// The `u8` lookup table must BE the function, for all 256 inputs -- not
    /// close to it. It replaces the function in the animation hex packer's
    /// per-pixel path, and a single wrong entry is a permanently wrong colour
    /// in every render that uses `--srgb-to-linear`.
    #[test]
    fn the_u8_linear_table_is_exactly_the_function() {
        let t = linear_gamma_table();
        for i in 0..=255u8 {
            assert_eq!(t[i as usize], to_linear_gamma(i), "entry {i}");
        }
    }

    /// Same for the float table, and bit-exact rather than within a
    /// tolerance: the colour-array path writes these `f32`s straight into the
    /// save, so "close enough" would be a silent, unreviewable drift from
    /// what every previous render produced.
    #[test]
    fn the_f32_linear_table_is_exactly_the_function() {
        let t = srgb_to_linear_f32_table();
        for i in 0..=255u8 {
            assert_eq!(
                t[i as usize].to_bits(),
                srgb_to_linear_f32(i).to_bits(),
                "entry {i} must be bit-identical, not merely close"
            );
        }
    }
}
