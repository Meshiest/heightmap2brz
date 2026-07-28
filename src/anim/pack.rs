//! Frame-major hex packing: turns a [`Clip`] into the string arrays a
//! Brickadia microchip reads.
//!
//! Each pixel contributes exactly [`HEX_STRIDE`] uppercase hex characters —
//! `RRGGBB`, no `#` (`MakeColorHex` accepts bare hex; a `#` would break the
//! fixed stride every downstream `Substring` offset depends on). Pixels are
//! indexed row-major (`index = row * width + col`) and tiled contiguously
//! into [`PIXELS_PER_CHUNK`]-pixel [`Chunk`]s, each holding one string per
//! frame sized to fit a single array component's
//! [`crate::text::MAX_COMPONENT_CHARS`] limit.
//!
//! A pixel below `alpha_threshold` in a given frame gets no brick and no
//! gates elsewhere in the pipeline. In production that decision is made by
//! [`Packer`], which folds visibility into the very same per-frame traversal
//! that builds these strings (see its own doc comment) — NOT independently by
//! whatever renderer calls [`pack`], which has no production caller left (see
//! `pack`'s own doc note). Either way, a culled pixel still contributes
//! exactly [`HEX_STRIDE`] characters to *that* frame's string, so every
//! surviving pixel's offset stays the plain `pixel_in_chunk * HEX_STRIDE` the
//! encoding relies on. No remap table, no gaps. Since nothing ever displays a
//! culled pixel's color, its slot is written as `"000000"` rather than its
//! real (and here, meaningless) source color.
use crate::text::MAX_COMPONENT_CHARS;
use crate::video::Clip;
use image::RgbaImage;
use std::fmt::Write as _;

/// Characters one pixel contributes to a frame string: `RRGGBB`, no `#`.
pub const HEX_STRIDE: usize = 6;

/// Pixels whose per-frame hex strings fit inside one
/// [`MAX_COMPONENT_CHARS`]-limited array entry.
pub const PIXELS_PER_CHUNK: usize = MAX_COMPONENT_CHARS / HEX_STRIDE;

/// Entries one wire array holds.
///
/// INFERRED, NOT MEASURED: the array index is a `u16`, which addresses
/// `0..=65535` — that is 65 536 entries, so this cap is one conservative. The
/// largest array actually loaded in-game to date is 63 340. If a multi-bank
/// render plays correctly and then breaks exactly at a bank seam, this
/// constant is the first thing to change.
pub const BANK_FRAMES: usize = 65_535;

/// Banks a single render may use. ~24 hours at 12fps — a guard against a
/// runaway `--fps`, not a meaningful ceiling.
pub const MAX_BANKS: usize = 16;

/// Largest frame count a render may carry, across all banks.
///
/// Kept under this name because `main.rs` (the `--max-frames` cap) and
/// `gui/video.rs` (the frame slider bound) both reference it. Its MEANING
/// widened with spillover: it used to be what one array holds, which is now
/// [`BANK_FRAMES`].
pub const MAX_FRAMES: usize = BANK_FRAMES * MAX_BANKS;

/// Split a frame list into per-array banks of at most `bank_size` entries.
///
/// The last bank is short rather than padded — padding would play phantom
/// frames at the end of the clip. An empty list still yields one (empty)
/// bank, because the renderer always needs an array to hang off.
///
/// `bank_size` is a parameter rather than a direct read of [`BANK_FRAMES`] so
/// the seam cases are testable at a size of 3: building a real 65 536-frame
/// clip costs minutes of CPU and gigabytes of RAM, which would leave the
/// boundary behaviour effectively untested.
pub fn bank_frames<'a>(frames: &'a [String], bank_size: usize) -> Vec<&'a [String]> {
    assert!(bank_size > 0, "bank size must be at least 1");
    if frames.is_empty() {
        return vec![&frames[..0]];
    }
    frames.chunks(bank_size).collect()
}

/// One horizontal span of the screen: `pixel_count` pixels starting at
/// row-major index `first_pixel`, one string per frame. Every frame string
/// is exactly `pixel_count * HEX_STRIDE` ASCII hex characters — a pixel
/// culled in a given frame (alpha below threshold there) still occupies its
/// slot in that frame's string, encoded as `"000000"`.
#[derive(Clone, Debug)]
pub struct Chunk {
    /// Row-major index (`row * width + col`) of this chunk's first pixel.
    pub first_pixel: usize,
    /// Number of pixels this chunk covers.
    pub pixel_count: usize,
    /// One entry per clip frame, each exactly `pixel_count * HEX_STRIDE`
    /// ASCII hex characters.
    pub frames: Vec<String>,
}

/// Encode `clip` into frame-major hex chunks.
///
/// Pixels are indexed row-major (`index = row * width + col`) and tiled
/// contiguously, first pixel ascending, into [`PIXELS_PER_CHUNK`]-pixel
/// chunks (the last chunk may be smaller; a zero-pixel screen produces zero
/// chunks). Within a frame's string, a pixel whose alpha there is strictly
/// below `alpha_threshold` encodes as `"000000"` — its real color carries no
/// meaning in that frame, since nothing displays it — while a pixel with
/// alpha `>= alpha_threshold` encodes its actual `RRGGBB`. A zero-frame clip
/// still reserves every pixel's slot; each chunk's `frames` is simply empty.
///
/// Errors (naming both limits) if `clip.frames.len()` exceeds [`MAX_FRAMES`]
/// — the overall cap across all [`MAX_BANKS`] banks of [`BANK_FRAMES`]
/// entries each; never truncates.
///
/// RETAINED DELIBERATELY, though it has no production caller any more (that
/// role now belongs to [`Packer`], which fuses this same encoding with
/// visibility in one pass instead of two): this whole-clip, two-pass
/// implementation is the byte-identity oracle every `Packer` test diffs
/// against (see `tests/anim_pack.rs`'s differential sweep, and
/// `the_fused_packer_matches_the_two_pass_result_exactly`). `pub` on purpose
/// so it stays reachable from the integration tests without a `#[cfg(test)]`
/// carve-out. Do not remove this function, and do not delete it as "dead
/// code" — doing so would delete the only independent check that `Packer`
/// still encodes frames exactly the way this crate always has.
pub fn pack(clip: &Clip, alpha_threshold: u8) -> Result<Vec<Chunk>, String> {
    if clip.frames.len() > MAX_FRAMES {
        return Err(format!(
            "clip has {} frames, over the {MAX_FRAMES}-frame limit \
             ({MAX_BANKS} banks of {BANK_FRAMES})",
            clip.frames.len()
        ));
    }

    let width = clip.width as usize;
    let total_pixels = width * clip.height as usize;

    let mut chunks = Vec::with_capacity(total_pixels.div_ceil(PIXELS_PER_CHUNK));
    let mut first_pixel = 0;
    while first_pixel < total_pixels {
        let pixel_count = PIXELS_PER_CHUNK.min(total_pixels - first_pixel);
        let mut frames = Vec::with_capacity(clip.frames.len());
        for frame in &clip.frames {
            let mut s = String::with_capacity(pixel_count * HEX_STRIDE);
            for local in 0..pixel_count {
                let idx = first_pixel + local;
                // row-major: index = row * width + col
                let col = (idx % width) as u32;
                let row = (idx / width) as u32;
                let p = frame.get_pixel(col, row).0;
                if p[3] < alpha_threshold {
                    s.push_str("000000");
                } else {
                    write!(s, "{:02X}{:02X}{:02X}", p[0], p[1], p[2])
                        .expect("writing to a String is infallible");
                }
            }
            frames.push(s);
        }
        chunks.push(Chunk { first_pixel, pixel_count, frames });
        first_pixel += pixel_count;
    }
    Ok(chunks)
}

/// The `HEX_STRIDE`-character slice for `pixel_in_chunk` within `frame`'s
/// string — the same `pixel_in_chunk * HEX_STRIDE` offset a `Substring`
/// gate's inlined `Start`/`Length` use.
///
/// Byte-slices directly rather than walking `chars()`. This is safe only
/// because [`pack`] writes exclusively ASCII bytes (hex digits or the
/// literal `"000000"`), so every `HEX_STRIDE`-aligned offset is guaranteed to
/// land on both a byte and a char boundary; the debug assertion below would
/// catch a future change to `pack` that broke that invariant (e.g. writing
/// anything non-ASCII) in a debug build, though not in release.
pub fn slice_of(chunk: &Chunk, frame: usize, pixel_in_chunk: usize) -> &str {
    let start = pixel_in_chunk * HEX_STRIDE;
    let end = start + HEX_STRIDE;
    let s = &chunk.frames[frame];
    debug_assert!(
        s.is_char_boundary(start) && s.is_char_boundary(end),
        "pack must only ever write ASCII hex; a non-ASCII byte would break this offset"
    );
    &s[start..end]
}

/// Builds the per-chunk frame strings and the per-pixel visibility bitmap in
/// ONE traversal of the frames, so no frame is ever retained.
///
/// This replaces two separate whole-clip scans: `bricks::visible` walked
/// every frame per pixel, and [`pack`] walked every frame again. Fusing them
/// costs one thing, deliberately accepted: `visible` short-circuits with
/// `.any()`, so a pixel opaque in frame 0 stopped its scan immediately, while
/// this must visit every frame because it is building the strings anyway. For
/// a clip opaque everywhere it therefore performs strictly more pixel visits.
/// It is still one traversal instead of two, and it is the only way to avoid
/// holding the frames.
pub struct Packer {
    width: usize,
    height: usize,
    alpha_threshold: u8,
    stride: usize,
    chunks: Vec<Chunk>,
    visible: Vec<bool>,
    frames_pushed: usize,
}

impl Packer {
    pub fn new(width: u32, height: u32, alpha_threshold: u8, stride: usize) -> Self {
        let width_usize = width as usize;
        let height_usize = height as usize;
        let total_pixels = width_usize * height_usize;
        let per_chunk = MAX_COMPONENT_CHARS / stride.max(1);
        let mut chunks = Vec::new();
        let mut first_pixel = 0;
        while first_pixel < total_pixels {
            let pixel_count = per_chunk.min(total_pixels - first_pixel);
            chunks.push(Chunk { first_pixel, pixel_count, frames: Vec::new() });
            first_pixel += pixel_count;
        }
        Self {
            width: width_usize,
            height: height_usize,
            alpha_threshold,
            stride,
            chunks,
            visible: vec![false; total_pixels],
            frames_pushed: 0,
        }
    }

    /// Push one frame's contribution to every chunk's per-frame string.
    ///
    /// `frame` MUST be exactly `width x height` (the dimensions `new` was
    /// built with -- the same ones a caller's `SourceInfo` reported, per the
    /// contract documented on [`crate::video::stream::FrameStream::next`]).
    /// Nothing upstream of this call enforces that: `build_brick_world` sizes
    /// this `Packer` from `source.info()` once and then indexes every
    /// subsequent frame with `get_pixel(col, row)` -- an undersized frame
    /// would panic there instead of here, and an oversized one would decode
    /// silently with its excess pixels ignored, no error at all. Checking
    /// here, before any pixel is read, turns both into one descriptive `Err`
    /// naming both the expected and the actual size.
    pub fn push_frame(&mut self, frame: &RgbaImage) -> Result<(), String> {
        if self.frames_pushed >= MAX_FRAMES {
            return Err(format!(
                "clip exceeds the {MAX_FRAMES}-frame limit ({MAX_BANKS} banks of {BANK_FRAMES})"
            ));
        }
        let (frame_w, frame_h) = (frame.width() as usize, frame.height() as usize);
        if frame_w != self.width || frame_h != self.height {
            return Err(format!(
                "frame {} is {frame_w}x{frame_h}, but the source's SourceInfo reported \
                 {}x{} -- every frame a FrameStream emits must match info()'s dimensions",
                self.frames_pushed, self.width, self.height
            ));
        }
        for chunk in &mut self.chunks {
            let mut s = String::with_capacity(chunk.pixel_count * self.stride);
            for local in 0..chunk.pixel_count {
                let idx = chunk.first_pixel + local;
                let col = (idx % self.width) as u32;
                let row = (idx / self.width) as u32;
                let p = frame.get_pixel(col, row).0;
                if p[3] < self.alpha_threshold {
                    for _ in 0..self.stride {
                        s.push('0');
                    }
                } else {
                    self.visible[idx] = true;
                    write!(s, "{:02X}{:02X}{:02X}", p[0], p[1], p[2])
                        .expect("writing to a String is infallible");
                }
            }
            chunk.frames.push(s);
        }
        self.frames_pushed += 1;
        Ok(())
    }

    /// The chunks, plus row-major per-pixel visibility (`true` where the
    /// pixel was opaque enough in at least one frame).
    pub fn finish(self) -> (Vec<Chunk>, Vec<bool>) {
        (self.chunks, self.visible)
    }
}
