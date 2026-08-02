//! Frame-major hex packing: turns a [`Clip`] into the string arrays a
//! Brickadia microchip reads.
//!
//! Each pixel contributes exactly [`HEX_STRIDE`] uppercase hex characters --
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
//! that builds these strings (see its own doc comment) -- NOT independently by
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
/// `0..=65535` -- that is 65 536 entries, so this cap is one conservative. The
/// largest array actually loaded in-game to date is 63 340. If a multi-bank
/// render plays correctly and then breaks exactly at a bank seam, this
/// constant is the first thing to change.
pub const BANK_FRAMES: usize = 65_535;

/// Banks a single render may use. ~24 hours at 12fps -- a guard against a
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
/// The last bank is short rather than padded -- padding would play phantom
/// frames at the end of the clip. An empty list still yields one (empty)
/// bank, because the renderer always needs an array to hang off.
///
/// `bank_size` is a parameter rather than a direct read of [`BANK_FRAMES`] so
/// the seam cases are testable at a size of 3: building a real 65 536-frame
/// clip costs minutes of CPU and gigabytes of RAM, which would leave the
/// boundary behaviour effectively untested.
///
/// Generic in the element type only so colour-array mode
/// ([`super::color_pack`]) banks its per-pixel colour lists on exactly this
/// code path rather than a second copy of the same edge cases. The hex path
/// instantiates it at `T = String` and is unchanged.
pub fn bank_frames<T>(frames: &[T], bank_size: usize) -> Vec<&[T]> {
    assert!(bank_size > 0, "bank size must be at least 1");
    if frames.is_empty() {
        return vec![&frames[..0]];
    }
    frames.chunks(bank_size).collect()
}

/// One horizontal span of the screen: `pixel_count` pixels starting at
/// row-major index `first_pixel`, one string per frame. Every frame string
/// is exactly `pixel_count * HEX_STRIDE` ASCII hex characters -- a pixel
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
/// below `alpha_threshold` encodes as `"000000"` -- its real color carries no
/// meaning in that frame, since nothing displays it -- while a pixel with
/// alpha `>= alpha_threshold` encodes its actual `RRGGBB`. A zero-frame clip
/// still reserves every pixel's slot; each chunk's `frames` is simply empty.
///
/// Errors (naming both limits) if `clip.frames.len()` exceeds [`MAX_FRAMES`]
/// -- the overall cap across all [`MAX_BANKS`] banks of [`BANK_FRAMES`]
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
/// code" -- doing so would delete the only independent check that `Packer`
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
/// string -- the same `pixel_in_chunk * HEX_STRIDE` offset a `Substring`
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
    /// Pixels per chunk, as `new` tiled them. Kept so `push_frame` can split
    /// [`Self::visible`] along exactly the same seams the chunks were cut on
    /// -- which is what lets each chunk own its slice of the bitmap outright
    /// instead of sharing one `Vec` across threads.
    per_chunk: usize,
    /// What one culled pixel contributes: `stride` `'0'`s, built once instead
    /// of pushed one character at a time per culled pixel per frame. Its
    /// length IS the stride this packer was built with -- it is all-ASCII, so
    /// `culled.len()` and the character stride are the same number.
    culled: String,
    chunks: Vec<Chunk>,
    visible: Vec<bool>,
    frames_pushed: usize,
    /// The sRGB -> linear table to run each pixel through before hex-encoding
    /// it, or `None` to encode the source bytes untouched.
    ///
    /// Decoded video frames are sRGB-encoded, which is what every image and
    /// video format stores. Whether that is also what the in-game
    /// `MakeColorHex` gate wants is a property of the GAME, not of this
    /// crate -- nothing here or in `brdb` transforms colour on the animation
    /// path, so the hex written below reaches that gate byte for byte. If
    /// the gate treats its hex as LINEAR, feeding it sRGB renders roughly one
    /// gamma step too bright, and this converts to compensate.
    ///
    /// `None` by default so existing renders are unchanged; `--srgb-to-linear`
    /// turns it on. Held as the precomputed table rather than a `bool` because
    /// [`crate::util::to_linear_gamma`] is an `f64` `powf` and this is a
    /// per-channel, per-pixel, per-frame path (see
    /// [`crate::util::linear_gamma_table`]); the table's entries are pinned
    /// against the function, so the colours are unchanged.
    linearize: Option<&'static [u8; 256]>,
}

impl Packer {
    pub fn new(width: u32, height: u32, alpha_threshold: u8, stride: usize) -> Self {
        let width_usize = width as usize;
        let height_usize = height as usize;
        let total_pixels = width_usize * height_usize;
        // `.max(1)`: a stride wider than a whole component would otherwise
        // give 0 pixels per chunk, and the loop below would never advance.
        let per_chunk = (MAX_COMPONENT_CHARS / stride.max(1)).max(1);
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
            per_chunk,
            culled: "0".repeat(stride),
            chunks,
            visible: vec![false; total_pixels],
            frames_pushed: 0,
            linearize: None,
        }
    }

    /// Enable the sRGB -> linear conversion described on [`Self::linearize`].
    ///
    /// A builder rather than a `new` parameter purely so the existing call
    /// sites and tests keep compiling unchanged.
    pub fn linearize(mut self, on: bool) -> Self {
        self.linearize = on.then(crate::util::linear_gamma_table);
        self
    }

    /// Push one frame's contribution to every chunk's per-frame string.
    ///
    /// `frame` MUST be exactly `width x height` (the dimensions `new` was
    /// built with -- the same ones a caller's `SourceInfo` reported, per the
    /// contract documented on [`crate::video::stream::FrameStream::next`]).
    /// Nothing upstream of this call enforces that: `build_brick_world` sizes
    /// this `Packer` from `source.info()` once and then reads every subsequent
    /// frame at those dimensions -- an undersized frame would panic deeper in
    /// (originally inside `get_pixel`; now on the raw-buffer slice below), and
    /// an oversized one would encode silently with its excess pixels ignored,
    /// no error at all. Checking here, before any pixel is read, turns both
    /// into one descriptive `Err` naming both the expected and the actual
    /// size -- and is also what makes the flat slicing below in-bounds by
    /// construction.
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
        // A chunk's pixels are a contiguous run of row-major indices, and
        // row-major RGBA is exactly the layout of `RgbaImage`'s raw buffer --
        // so a chunk IS one flat slice of it. Walking that slice replaces a
        // `%`, a `/` and a bounds-checked `get_pixel` per pixel with one
        // offset computed per chunk. The dimension check above is what makes
        // the slicing below in-bounds by construction.
        let raw = frame.as_raw();
        debug_assert_eq!(
            raw.len(),
            self.width * self.height * 4,
            "an RgbaImage's raw buffer is 4 bytes per pixel, row-major and gapless"
        );

        {
            let Self { chunks, visible, per_chunk, alpha_threshold, culled, linearize, .. } = self;
            let (alpha_threshold, linearize, culled) = (*alpha_threshold, *linearize, &**culled);
            encode_frame(chunks, visible, *per_chunk, |chunk, vis| {
                debug_assert_eq!(
                    vis.len(),
                    chunk.pixel_count,
                    "the visibility bitmap must split on the same seams the chunks were cut on"
                );
                let start = chunk.first_pixel * 4;
                let end = start + chunk.pixel_count * 4;
                let s = encode_chunk(&raw[start..end], culled, alpha_threshold, linearize, vis);
                chunk.frames.push(s);
            });
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

/// One chunk's contribution to one frame: its `pixel_count * stride`
/// characters, plus which of its pixels were opaque enough to keep.
///
/// `px` is that chunk's own `4 * pixel_count` RGBA bytes, already sliced out
/// of the frame, and `vis` is that chunk's own slice of the visibility
/// bitmap -- both indexed by pixel-within-chunk, so this function never needs
/// the screen width, a division, or a global pixel index. `vis` is only ever
/// set to `true` (a pixel visible in an earlier frame stays visible), which
/// is what lets the same slice accumulate across every frame.
fn encode_chunk(
    px: &[u8],
    culled: &str,
    alpha_threshold: u8,
    linearize: Option<&[u8; 256]>,
    vis: &mut [bool],
) -> String {
    let mut s = String::with_capacity(vis.len() * culled.len());
    for (local, p) in px.chunks_exact(4).enumerate() {
        if p[3] < alpha_threshold {
            s.push_str(culled);
        } else {
            vis[local] = true;
            // Alpha is untouched by the transfer and unused here.
            let rgb = match linearize {
                Some(t) => [t[p[0] as usize], t[p[1] as usize], t[p[2] as usize]],
                None => [p[0], p[1], p[2]],
            };
            for c in rgb {
                s.push_str(crate::util::hex_pair(c));
            }
        }
    }
    s
}

/// Fewest chunks rayon may put in one job.
///
/// MEASURED, not guessed. Without it, `par_iter_mut` splits down to a single
/// chunk per job, and one chunk is only ~1 666 pixels of work -- so on a
/// 192x108 screen (13 chunks) the parallel version came out **slower than the
/// serial one**: 101 Mpx/s against 209, because each frame paid a full
/// fan-out/join across the pool for jobs that finish in microseconds. Bounding
/// the split at 8 chunks per job makes the small case a wash (228 vs 209
/// Mpx/s) while keeping the large one, where the fan-out is amortized: at
/// 640x360 (139 chunks) it is 600 Mpx/s against the serial 227.
///
/// (400 frames, `--release`, 32 logical cores. Frames are pushed one at a
/// time by a streaming decoder, so this fan-out happens once PER FRAME and
/// its fixed cost is what the bound is protecting against.)
#[cfg(not(target_arch = "wasm32"))]
const MIN_CHUNKS_PER_JOB: usize = 8;

/// Run `encode` over every chunk paired with its own slice of the visibility
/// bitmap.
///
/// Native builds fan this out across rayon's pool. There is no shared mutable
/// state to race on: each chunk owns its `String`, and `chunks_mut(per_chunk)`
/// splits `visible` on exactly the seams `Packer::new` cut the chunks on, so
/// each closure invocation writes only its own disjoint slice. That is a
/// split borrow the compiler checks, not a lock and not an atomic -- there is
/// no way for two jobs to reach the same `visible[i]`.
///
/// `rayon` is a native-only dependency (see `Cargo.toml`'s target blocks --
/// it needs threads, which `wasm32-unknown-unknown` does not have without a
/// bundler-specific shim), so the wasm build gets the serial walk below. Both
/// produce identical output; only the order the chunks are visited in differs,
/// and nothing here depends on that order.
#[cfg(not(target_arch = "wasm32"))]
fn encode_frame(
    chunks: &mut [Chunk],
    visible: &mut [bool],
    per_chunk: usize,
    encode: impl Fn(&mut Chunk, &mut [bool]) + Send + Sync,
) {
    use rayon::prelude::*;
    chunks
        .par_iter_mut()
        .zip(visible.par_chunks_mut(per_chunk))
        .with_min_len(MIN_CHUNKS_PER_JOB)
        .for_each(|(chunk, vis)| encode(chunk, vis));
}

/// See the native sibling above.
#[cfg(target_arch = "wasm32")]
fn encode_frame(
    chunks: &mut [Chunk],
    visible: &mut [bool],
    per_chunk: usize,
    encode: impl Fn(&mut Chunk, &mut [bool]) + Send + Sync,
) {
    for (chunk, vis) in chunks.iter_mut().zip(visible.chunks_mut(per_chunk)) {
        encode(chunk, vis);
    }
}
