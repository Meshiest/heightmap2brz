use brdb::{BString, Brick, Collision, World};
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
    pub lrgb: bool,
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

#[cfg(test)]
mod tests {
    use super::*;

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
