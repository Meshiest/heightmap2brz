//! Measures whether TextDisplay ("text mode", `src/text.rs`) is cheaper than
//! the hex animation encoding ("hex mode", `src/anim/pack.rs`) on real
//! content.
//!
//! Hex mode costs exactly [`HEX_STRIDE`] = 6 characters per pixel, always.
//! Text mode costs `16/L + W` characters per cell, where `L` is the mean
//! color-run length (a `<color="RRGGBB">` tag is exactly 16 chars and is
//! amortized over every pixel in the run it colors) and `W` is
//! `char_repeat` (each pixel in the run still emits its own `W` fill-glyph
//! copies). Text mode wins once `L` clears ~4 at `W=2` or ~3.2 at `W=1` --
//! but until this tool existed, nobody had measured `L` on real frames.
//!
//! This tool reuses [`heightmap::text::encode_bands`] -- the exact function
//! that produces the shipped TextDisplay strings -- for every character
//! count. Nothing here reimplements the run-length decision (whether a
//! pixel continues the previous color run or starts a new tag); that logic
//! lives in `encode_row`/`encode_bands` alone. All this tool adds on top:
//!
//! - counting how many `<color="` tags appear in the ALREADY-PRODUCED text
//!   (a substring count over the real output, not a parallel decision), to
//!   turn the real `chars` total into a mean run length;
//! - uniform per-channel bit-depth quantization of the input, since the
//!   question is how run length behaves at different palette sizes; and
//! - a median-cut perceptual palette, compared against that uniform
//!   quantization at every reduced palette size. Uniform bucketing collapses
//!   broad swathes of color space into one bucket -- exactly what produces
//!   long runs -- but it also bands visibly at low color counts. Text mode
//!   carries its own explicit `<color="RRGGBB">` tag per run and needs no
//!   in-game palette table, so any 256 (or fewer) colors are usable and can
//!   be chosen per clip. This tool measures whether a perceptually-placed
//!   palette shortens runs relative to uniform bucketing, and at what
//!   fidelity cost (see [`median_cut_palette`], [`nearest`], [`Fidelity`]).
//!
//! Deliberately calls [`encode_bands`] directly rather than
//! [`heightmap::text::encode_tiles`]: tiling exists to keep world-placement
//! offsets small for a *static-image* render (each 32x32 patch anchors its
//! own brick) and resets color state at every tile seam for that reason --
//! a concern specific to today's brick-placement scheme, not to the
//! character-cost question this tool answers. `encode_bands`'s row-major,
//! budget-only banding (splitting only at [`MAX_COMPONENT_CHARS`]) is the
//! part of the encoder a hypothetical per-frame animation encoding (the
//! thing being evaluated here, as a text-mode alternative to
//! `anim::pack`'s row-major hex chunks) would actually reuse.
use heightmap::anim::pack::MAX_FRAMES;
use heightmap::text::{FontPreset, PixelMode, TextOptions, encode_bands};
use heightmap::video::backend::{self, Backend};
use heightmap::video::ffmpeg::{DownloadConsent, ensure_ffmpeg};
use heightmap::video::scale::{FitMode, Filter};
use heightmap::video::source::{Source, decode, is_animated, is_video_path};
use heightmap::video::stream::{AdaptedSource, FrameSource};
use image::{Rgba, RgbaImage};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The literal opening of every color tag `encode_row` ever writes
/// (`<color="RRGGBB">` is 16 chars total). Counting its occurrences in the
/// REAL text `encode_bands` produced is how this tool learns the tag count
/// without re-deciding, anywhere, when a tag is needed.
const TAG_MARKER: &str = "<color=\"";

/// Hex mode's fixed per-pixel cost (`anim::pack::HEX_STRIDE`), repeated here
/// only as the comparison baseline printed alongside every measurement.
const HEX_STRIDE: f64 = 6.0;

// ---------------------------------------------------------------------------
// Quantization method: two ways of building a reduced palette, compared
// side by side at every reduced palette size below.
// ---------------------------------------------------------------------------

/// Which palette-construction method produced a quantized frame set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Method {
    /// Uniform per-channel bit-depth truncation (RGB332-style) -- see
    /// [`Palette::bits`]. Fast, deterministic from a bit count alone, but
    /// bands visibly at low color counts.
    Uniform,
    /// Median-cut: one palette built from every sampled frame COMBINED (not
    /// a fresh palette per frame), each pixel nearest-mapped to it by plain
    /// Euclidean RGB distance, no dithering. See [`median_cut_palette`] and
    /// [`nearest`].
    MedianCut,
}

impl Method {
    fn label(&self) -> &'static str {
        match self {
            Method::Uniform => "uniform",
            Method::MedianCut => "median-cut",
        }
    }
}

// ---------------------------------------------------------------------------
// Quantization: uniform per-channel bit-depth reduction. Alpha is untouched.
// ---------------------------------------------------------------------------

/// A palette size to sweep. Each maps to a fixed (R, G, B) bit split summing
/// to `log2(colors)`, biased R >= G >= B to match how the classic 8-bit
/// RGB332 scheme (3/3/2) allocates -- the eye is most sensitive to red and
/// green, least to blue. This is a predictable, fast, uniform reduction
/// (round each channel to the nearest of `2^bits` evenly spaced levels) --
/// NOT a perceptual or k-means palette, which would not reproduce
/// deterministically from a bit count alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Palette {
    /// No quantization: every channel keeps its full 8 bits.
    Full24,
    C256,
    C64,
    C32,
    C16,
}

impl Palette {
    const ALL: [Palette; 5] = [
        Palette::Full24,
        Palette::C256,
        Palette::C64,
        Palette::C32,
        Palette::C16,
    ];

    /// (R bits, G bits, B bits), summing to `log2(colors())`.
    fn bits(&self) -> (u32, u32, u32) {
        match self {
            Palette::Full24 => (8, 8, 8),
            Palette::C256 => (3, 3, 2),
            Palette::C64 => (2, 2, 2),
            Palette::C32 => (2, 2, 1),
            Palette::C16 => (2, 1, 1),
        }
    }

    /// Total distinct colors this split can represent.
    fn colors(&self) -> u32 {
        let (r, g, b) = self.bits();
        1u32 << (r + g + b)
    }

    fn label(&self) -> String {
        match self {
            Palette::Full24 => "full 24-bit".to_string(),
            _ => {
                let (r, g, b) = self.bits();
                format!("{} ({r}/{g}/{b})", self.colors())
            }
        }
    }
}

/// Round `v` to the nearest of `2^bits` evenly spaced levels across
/// `0..=255`, expanded back to `0..=255`. `bits >= 8` is a no-op (there are
/// already at most 256 representable 8-bit levels).
fn quantize_channel(v: u8, bits: u32) -> u8 {
    if bits >= 8 {
        return v;
    }
    let max_level = (1u32 << bits) - 1;
    let step = 255.0 / max_level as f32;
    let level = ((v as f32 / step).round() as u32).min(max_level);
    (level as f32 * step).round() as u8
}

/// Quantize RGB per `bits`; alpha passes through EXACTLY unchanged.
fn quantize_pixel(p: [u8; 4], bits: (u32, u32, u32)) -> [u8; 4] {
    [
        quantize_channel(p[0], bits.0),
        quantize_channel(p[1], bits.1),
        quantize_channel(p[2], bits.2),
        p[3],
    ]
}

fn quantize_image(img: &RgbaImage, bits: (u32, u32, u32)) -> RgbaImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        *p = Rgba(quantize_pixel(p.0, bits));
    }
    out
}

// ---------------------------------------------------------------------------
// Median-cut quantization: builds ONE palette from every sampled frame
// combined (not per frame -- a per-frame palette would be free here but is
// not what a real encoder would necessarily ship), then nearest-color
// (plain Euclidean RGB distance) maps each pixel to it. No dithering --
// dithering would improve appearance at the direct expense of the run
// lengths this tool measures. Alpha is never touched, matching the uniform
// quantizer.
// ---------------------------------------------------------------------------

/// One distinct color and how many sampled pixels carry it.
type ColorEntry = ([u8; 3], u64);

/// Build a distinct-color histogram (RGB -> pixel count) across every
/// sampled frame. Alpha is ignored: neither quantization method here ever
/// touches it, so it plays no part in palette construction.
fn color_histogram(frames: &[&RgbaImage]) -> HashMap<[u8; 3], u64> {
    let mut hist = HashMap::new();
    for &frame in frames {
        for p in frame.pixels() {
            *hist.entry([p.0[0], p.0[1], p.0[2]]).or_insert(0u64) += 1;
        }
    }
    hist
}

/// The channel (0=R, 1=G, 2=B) with the widest value spread across
/// `entries`, and that spread. `None` only for an empty bucket (never
/// produced by [`median_cut_palette`], but guarded rather than panicking).
fn widest_channel(entries: &[ColorEntry]) -> Option<(usize, u8)> {
    if entries.is_empty() {
        return None;
    }
    let mut lo = [255u8, 255, 255];
    let mut hi = [0u8, 0, 0];
    for &(c, _) in entries {
        for ch in 0..3 {
            lo[ch] = lo[ch].min(c[ch]);
            hi[ch] = hi[ch].max(c[ch]);
        }
    }
    let ranges = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let (ch, &range) = ranges.iter().enumerate().max_by_key(|&(_, r)| *r).unwrap();
    Some((ch, range))
}

/// Split `bucket` in two along `channel`, at the weighted median (the split
/// point where accumulated pixel-count first passes half the bucket's total
/// weight). Requires `bucket` to have a non-zero range along `channel`
/// (checked by the caller via [`widest_channel`]): a non-zero range means at
/// least two distinct colors are present, so both halves come out
/// non-empty.
fn split_bucket(mut bucket: Vec<ColorEntry>, channel: usize) -> (Vec<ColorEntry>, Vec<ColorEntry>) {
    bucket.sort_by_key(|&(c, _)| c[channel]);
    let total: u64 = bucket.iter().map(|&(_, w)| w).sum();
    let half = total / 2;
    let mut acc = 0u64;
    let mut split_at = bucket.len();
    for (i, &(_, w)) in bucket.iter().enumerate() {
        acc += w;
        if acc > half {
            split_at = i + 1;
            break;
        }
    }
    let split_at = split_at.clamp(1, bucket.len() - 1);
    let rest = bucket.split_off(split_at);
    (bucket, rest)
}

/// The weighted-average (pixel-count-weighted) RGB of a bucket, rounded to
/// `u8` -- the palette entry that bucket contributes.
fn bucket_centroid(bucket: &[ColorEntry]) -> [u8; 3] {
    let mut sum = [0f64; 3];
    let mut total = 0u64;
    for &(c, w) in bucket {
        for (ch, s) in sum.iter_mut().enumerate() {
            *s += c[ch] as f64 * w as f64;
        }
        total += w;
    }
    let total = (total.max(1)) as f64;
    [
        (sum[0] / total).round().clamp(0.0, 255.0) as u8,
        (sum[1] / total).round().clamp(0.0, 255.0) as u8,
        (sum[2] / total).round().clamp(0.0, 255.0) as u8,
    ]
}

/// Build a median-cut palette of at most `max_colors` entries from a
/// (distinct color -> pixel count) histogram. Repeatedly splits the bucket
/// with the widest per-channel spread at its weighted median until either
/// `max_colors` buckets exist, or every remaining bucket holds a single
/// distinct color -- at which point splitting stops even if `max_colors`
/// hasn't been reached: an image with fewer distinct colors than requested
/// yields a smaller palette rather than padding with duplicates.
fn median_cut_palette(histogram: &HashMap<[u8; 3], u64>, max_colors: usize) -> Vec<[u8; 3]> {
    if max_colors == 0 || histogram.is_empty() {
        return Vec::new();
    }
    let mut buckets: Vec<Vec<ColorEntry>> = vec![histogram.iter().map(|(&c, &w)| (c, w)).collect()];

    loop {
        if buckets.len() >= max_colors {
            break;
        }
        // Find the splittable bucket (range > 0 along its own widest
        // channel) with the largest such range across all buckets.
        let mut best: Option<(usize, usize, u8)> = None;
        for (i, b) in buckets.iter().enumerate() {
            if let Some((ch, range)) = widest_channel(b) {
                if range > 0 && best.is_none_or(|(_, _, r)| range > r) {
                    best = Some((i, ch, range));
                }
            }
        }
        let Some((idx, channel, _)) = best else {
            break; // every remaining bucket is a single distinct color
        };
        let bucket = buckets.swap_remove(idx);
        let (a, b) = split_bucket(bucket, channel);
        buckets.push(a);
        buckets.push(b);
    }

    buckets.iter().map(|b| bucket_centroid(b)).collect()
}

/// Squared Euclidean RGB distance.
fn dist2(a: [u8; 3], b: [u8; 3]) -> u32 {
    (0..3)
        .map(|ch| {
            let d = a[ch] as i32 - b[ch] as i32;
            (d * d) as u32
        })
        .sum()
}

/// The nearest entry of `palette` to `c` by plain Euclidean RGB distance
/// (ties keep the first minimal entry found).
fn nearest(c: [u8; 3], palette: &[[u8; 3]]) -> [u8; 3] {
    *palette
        .iter()
        .min_by_key(|&&p| dist2(c, p))
        .expect("palette must be non-empty")
}

/// Precompute the nearest palette entry for every distinct color in
/// `histogram`, so quantizing an image is a lookup per pixel rather than a
/// fresh nearest-neighbor search.
fn nearest_map(histogram: &HashMap<[u8; 3], u64>, palette: &[[u8; 3]]) -> HashMap<[u8; 3], [u8; 3]> {
    histogram
        .keys()
        .map(|&c| (c, nearest(c, palette)))
        .collect()
}

/// Map every pixel of `img` through `map` (built by [`nearest_map`] over a
/// histogram that must include every color `img` contains -- true whenever
/// `img` is drawn from the same sampled frame set the histogram was built
/// from). Alpha passes through unchanged, matching the uniform quantizer;
/// an unmapped color (should not occur) is left as-is rather than panicking.
fn apply_color_map(img: &RgbaImage, map: &HashMap<[u8; 3], [u8; 3]>) -> RgbaImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let rgb = [p.0[0], p.0[1], p.0[2]];
        let mapped = map.get(&rgb).copied().unwrap_or(rgb);
        *p = Rgba([mapped[0], mapped[1], mapped[2], p.0[3]]);
    }
    out
}

// ---------------------------------------------------------------------------
// Fidelity: mean squared error (and PSNR) of a quantized frame set against
// the original, unquantized sampled frames. This is the number that turns
// "median-cut costs N% more characters" into a decision -- what the
// character cost buys, or gives up.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Fidelity {
    /// Mean squared error across every sampled pixel and every RGB channel.
    /// Alpha is excluded: no quantizer in this file ever touches it, so its
    /// error is always exactly zero and would only dilute the number.
    mse: f64,
}

impl Fidelity {
    /// Peak signal-to-noise ratio in dB: `10 * log10(255^2 / mse)`. Exactly
    /// zero error reports `f64::INFINITY`, matching the convention this file
    /// already uses for undefined-by-zero ratios (see
    /// [`Stats::mean_run_length`]).
    fn psnr(&self) -> f64 {
        if self.mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 / self.mse).log10()
        }
    }
}

/// Mean squared error of `quantized` against `originals`, over every RGB
/// channel of every pixel in every (same-length, same-dimensions) frame
/// pair.
fn mse(originals: &[&RgbaImage], quantized: &[RgbaImage]) -> Fidelity {
    let mut sq_err: f64 = 0.0;
    let mut n: u64 = 0;
    for (orig, quant) in originals.iter().zip(quantized) {
        for (op, qp) in orig.pixels().zip(quant.pixels()) {
            for ch in 0..3 {
                let d = op.0[ch] as f64 - qp.0[ch] as f64;
                sq_err += d * d;
            }
            n += 3;
        }
    }
    Fidelity {
        mse: sq_err / n.max(1) as f64,
    }
}

// ---------------------------------------------------------------------------
// Measurement: reuses `encode_bands` for every character counted.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default, Debug)]
struct Stats {
    /// Total pixels ("cells") measured, across every sampled frame.
    cells: u64,
    /// Total characters `encode_bands` produced -- read straight off
    /// [`heightmap::text::TextBand::chars`], the SAME count the shipped
    /// encoder tracks as it writes each band's string. Never recomputed.
    chars: u64,
    /// Number of `<color="...">` tags found in the real output text.
    tags: u64,
}

impl Stats {
    fn chars_per_cell(&self) -> f64 {
        self.chars as f64 / self.cells.max(1) as f64
    }

    /// `cells / tags` -- the mean number of cells one tag was amortized
    /// over. `f64::INFINITY` when no tag was ever written (e.g. an
    /// entirely transparent input): there is no run to measure.
    fn mean_run_length(&self) -> f64 {
        if self.tags == 0 {
            f64::INFINITY
        } else {
            self.cells as f64 / self.tags as f64
        }
    }
}

impl std::ops::AddAssign for Stats {
    fn add_assign(&mut self, o: Stats) {
        self.cells += o.cells;
        self.chars += o.chars;
        self.tags += o.tags;
    }
}

/// Measure one (already quantized) frame with `opts`, via the real encoder.
///
/// The only things NOT sourced from `encode_bands`'s own output are the
/// frame's pixel count (trivial `width * height`, not run-length logic) and
/// the tag count (a substring count over the text `encode_bands` already
/// wrote -- see [`TAG_MARKER`]).
fn measure_frame(img: &RgbaImage, opts: &TextOptions) -> Result<Stats, String> {
    let (w, h) = img.dimensions();
    let bands = encode_bands(img, opts)?;
    let mut stats = Stats {
        cells: w as u64 * h as u64,
        ..Default::default()
    };
    for band in &bands {
        stats.chars += band.chars as u64;
        stats.tags += band.text.matches(TAG_MARKER).count() as u64;
    }
    Ok(stats)
}

/// One (palette, method) configuration's quantized sampled frames, plus its
/// fidelity against the originals -- computed ONCE and reused across the
/// `char_repeat` sweep, since char_repeat changes only the fill-glyph cost,
/// never the quantization.
struct QuantizedSet {
    frames: Vec<RgbaImage>,
    fidelity: Fidelity,
}

/// Quantize every sampled frame under one (palette, method) configuration.
/// `histogram` is the color histogram of `sampled` itself (built once by the
/// caller); only `Method::MedianCut` reads it -- `Method::Uniform` quantizes
/// each pixel independently, per [`quantize_channel`].
fn quantize_set(
    sampled: &[&RgbaImage],
    histogram: &HashMap<[u8; 3], u64>,
    palette: Palette,
    method: Method,
) -> QuantizedSet {
    let frames: Vec<RgbaImage> = match method {
        Method::Uniform => sampled
            .iter()
            .map(|&f| quantize_image(f, palette.bits()))
            .collect(),
        Method::MedianCut => {
            let mc_palette = median_cut_palette(histogram, palette.colors() as usize);
            let map = nearest_map(histogram, &mc_palette);
            sampled.iter().map(|&f| apply_color_map(f, &map)).collect()
        }
    };
    let fidelity = mse(sampled, &frames);
    QuantizedSet { frames, fidelity }
}

/// Measure a whole quantized frame set at one `char_repeat`.
fn measure_set(frames: &[RgbaImage], char_repeat: usize) -> Result<Stats, String> {
    let opts = TextOptions {
        char_repeat,
        mode: PixelMode::Color,
        ..FontPreset::MonaspaceArgon.options(1.0)
    };
    let mut total = Stats::default();
    for frame in frames {
        total += measure_frame(frame, &opts)?;
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Even sampling: at most `n` frames spread across the whole clip, not just
// its opening shot (a source with scene cuts must not be judged by frame 0).
// ---------------------------------------------------------------------------

fn sample_evenly(frames: &[RgbaImage], n: usize) -> Vec<&RgbaImage> {
    if frames.is_empty() || n == 0 {
        return Vec::new();
    }
    let n = n.min(frames.len());
    (0..n).map(|i| &frames[i * frames.len() / n]).collect()
}

// ---------------------------------------------------------------------------
// Input handling: reuses the crate's FrameSource abstraction so this works
// on still images (PNG/JPG), animated images (GIF/WebP/APNG), and video
// (mp4/mov/mkv/webm/avi/m4v) alike -- exactly the sources `main.rs`'s
// `--anim-mode` path opens.
// ---------------------------------------------------------------------------

fn drain_source(source: &dyn FrameSource, cap: usize) -> Result<Vec<RgbaImage>, String> {
    let mut stream = source.open()?;
    let mut frames = Vec::new();
    while frames.len() < cap {
        match stream.next()? {
            Some(f) => frames.push(f),
            None => break,
        }
    }
    Ok(frames)
}

fn open_and_drain(path: &Path, size: Option<(u32, u32)>) -> Result<Vec<RgbaImage>, String> {
    if is_video_path(path) {
        // `Backend::Auto` + `ensure_ffmpeg` mirrors main.rs's own default
        // video path: try the safe builtin decoder first, only ask about
        // ffmpeg if that refuses. `target`/`fps` stay `None` on the raw
        // open -- `AdaptedSource` below is the one place that resizes.
        let raw = backend::open_video_ensuring(
            path,
            Backend::Auto,
            None,
            FitMode::Contain,
            Filter::Lanczos,
            None,
            &mut || ensure_ffmpeg(DownloadConsent::Ask),
        )?;
        let native_fps = raw.info().fps;
        let adapted = AdaptedSource {
            inner: raw.as_ref(),
            size,
            fit: FitMode::Contain,
            filter: Filter::Lanczos,
            // The source's own rate: no resampling, every decoded frame is
            // kept (this tool samples evenly from the full set itself).
            target_fps: native_fps,
            start_s: 0.0,
            duration_s: None,
            max_frames: MAX_FRAMES,
        };
        drain_source(&adapted, MAX_FRAMES)
    } else {
        let bytes =
            std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        let source = if is_animated(&bytes) {
            Source::Animated(bytes)
        } else {
            let img = image::load_from_memory(&bytes)
                .map_err(|e| format!("decoding {}: {e:?}", path.display()))?
                .to_rgba8();
            Source::Still(img)
        };
        let clip = decode(source, 10.0)?;
        let adapted = AdaptedSource {
            inner: &clip,
            size,
            fit: FitMode::Contain,
            filter: Filter::Lanczos,
            target_fps: clip.fps,
            start_s: 0.0,
            duration_s: None,
            max_frames: MAX_FRAMES,
        };
        drain_source(&adapted, MAX_FRAMES)
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    input: PathBuf,
    width: Option<u32>,
    height: Option<u32>,
    frames: usize,
}

fn usage() -> String {
    "usage: text_sizing <input> [--width W --height H] [--frames N (default 120)]\n\
     \n\
     <input> is an image (png/jpg/gif/webp) or a video (mp4/mov/mkv/webm/avi/m4v).\n\
     --width/--height must be given together (or omitted, to keep the source's\n\
     native size). --frames caps how many frames are sampled, spread evenly\n\
     across the whole clip rather than taken from the start."
        .to_string()
}

fn parse_args() -> Result<Args, String> {
    let mut input = None;
    let mut width = None;
    let mut height = None;
    let mut frames = 120usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Err(usage()),
            "--width" => {
                let v = args.next().ok_or("--width needs a value")?;
                width = Some(
                    v.parse::<u32>()
                        .map_err(|e| format!("--width must be an integer: {e}"))?,
                );
            }
            "--height" => {
                let v = args.next().ok_or("--height needs a value")?;
                height = Some(
                    v.parse::<u32>()
                        .map_err(|e| format!("--height must be an integer: {e}"))?,
                );
            }
            "--frames" => {
                let v = args.next().ok_or("--frames needs a value")?;
                frames = v
                    .parse::<usize>()
                    .map_err(|e| format!("--frames must be an integer: {e}"))?;
            }
            other if input.is_none() => input = Some(PathBuf::from(other)),
            other => return Err(format!("unrecognized argument '{other}'\n\n{}", usage())),
        }
    }

    let input = input.ok_or_else(|| format!("missing input path\n\n{}", usage()))?;
    Ok(Args { input, width, height, frames })
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let size = match (args.width, args.height) {
        (Some(w), Some(h)) => Some((w, h)),
        (None, None) => None,
        _ => {
            return Err(
                "--width and --height must be given together (or neither, to keep the \
                 source's native size)"
                    .to_string(),
            );
        }
    };

    println!("Opening {}", args.input.display());
    let frames = open_and_drain(&args.input, size)?;
    if frames.is_empty() {
        return Err("source produced no frames".to_string());
    }
    let (w, h) = frames[0].dimensions();
    println!(
        "Decoded {} frame(s) at {w}x{h}; sampling up to {} evenly across the clip",
        frames.len(),
        args.frames,
    );
    let sampled = sample_evenly(&frames, args.frames);
    let n = sampled.len();
    println!(
        "Measuring on {n} sampled frame(s) ({} cells each, {} cells total)\n",
        w as u64 * h as u64,
        n as u64 * w as u64 * h as u64,
    );
    println!(
        "Two quantization methods are compared at every reduced palette size below full \
         24-bit:\n\
         - uniform: per-channel bit-depth reduction -- each channel rounds to the nearest of \
         2^bits evenly spaced levels (bits split R/G/B, biased toward red/green as RGB332 \
         does).\n\
         - median-cut: ONE palette built from all {n} sampled frames COMBINED (not a fresh \
         palette per frame -- a per-frame palette would be free here but is not what a real \
         encoder would necessarily ship across a whole clip); each pixel is then mapped to its \
         nearest palette entry by plain Euclidean RGB distance. No dithering (dithering would \
         improve appearance at the direct expense of the run lengths being measured here).\n\
         Neither method ever touches alpha.\n"
    );
    let histogram = color_histogram(&sampled);
    println!("Sampled frames contain {} distinct RGB colors.\n", histogram.len());

    println!(
        "{:<14} {:<11} {:>3} {:>10} {:>10} {:>11} {:>7} {:>11} {:>10} {:>9}",
        "palette", "method", "W", "cells", "chars", "chars/cell", "vs 6.0", "mean run L", "MSE", "PSNR(dB)"
    );
    for palette in Palette::ALL {
        let methods: &[Method] = if palette == Palette::Full24 {
            &[Method::Uniform]
        } else {
            &[Method::Uniform, Method::MedianCut]
        };
        for &method in methods {
            let set = quantize_set(&sampled, &histogram, palette, method);
            let method_label = if palette == Palette::Full24 { "n/a" } else { method.label() };
            let psnr = set.fidelity.psnr();
            let psnr_str = if psnr.is_finite() {
                format!("{psnr:.2}")
            } else {
                "inf".to_string()
            };
            for &char_repeat in &[1usize, 2] {
                let stats = measure_set(&set.frames, char_repeat)?;
                let run_l = stats.mean_run_length();
                let run_l_str = if run_l.is_finite() {
                    format!("{run_l:.2}")
                } else {
                    "inf".to_string()
                };
                println!(
                    "{:<14} {:<11} {:>3} {:>10} {:>10} {:>11.3} {:>7.3} {:>11} {:>10.3} {:>9}",
                    palette.label(),
                    method_label,
                    char_repeat,
                    stats.cells,
                    stats.chars,
                    stats.chars_per_cell(),
                    stats.chars_per_cell() / HEX_STRIDE,
                    run_l_str,
                    set.fidelity.mse,
                    psnr_str,
                );
            }
        }
    }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const RED: Rgba<u8> = Rgba([255, 0, 0, 255]);
    const GREEN: Rgba<u8> = Rgba([0, 255, 0, 255]);
    const BLUE: Rgba<u8> = Rgba([0, 0, 255, 255]);
    const WHITE: Rgba<u8> = Rgba([255, 255, 255, 255]);

    fn row(pixels: &[Rgba<u8>]) -> RgbaImage {
        let mut img = RgbaImage::new(pixels.len() as u32, 1);
        for (x, p) in pixels.iter().enumerate() {
            img.put_pixel(x as u32, 0, *p);
        }
        img
    }

    fn opts(char_repeat: usize) -> TextOptions {
        TextOptions {
            char_repeat,
            mode: PixelMode::Color,
            ..FontPreset::MonaspaceArgon.options(1.0)
        }
    }

    /// A uniform row is ONE run: one tag amortized over every cell.
    /// 5 cells, char_repeat 2: 16 (tag) + 5*2 (fill) = 26 chars exactly.
    #[test]
    fn uniform_row_is_one_run() {
        let img = row(&[RED; 5]);
        let stats = measure_frame(&img, &opts(2)).unwrap();
        assert_eq!(stats.cells, 5);
        assert_eq!(stats.tags, 1, "a uniform row must emit exactly one tag");
        assert_eq!(stats.chars, 16 + 5 * 2, "exact char count, not approximate");
        assert_eq!(stats.chars_per_cell(), 26.0 / 5.0);
        assert_eq!(stats.mean_run_length(), 5.0);
    }

    /// A row alternating between two colors is one run PER CELL: every
    /// pixel differs from its predecessor, so every pixel gets its own tag.
    /// 6 cells, char_repeat 2: 6 * (16 + 2) = 108 chars exactly.
    #[test]
    fn alternating_row_is_one_run_per_cell() {
        let img = row(&[RED, GREEN, RED, GREEN, RED, GREEN]);
        let stats = measure_frame(&img, &opts(2)).unwrap();
        assert_eq!(stats.cells, 6);
        assert_eq!(stats.tags, 6, "every cell must start a new run");
        assert_eq!(stats.chars, 6 * (16 + 2), "exact char count, not approximate");
        assert_eq!(stats.mean_run_length(), 1.0, "worst case: one cell per run");
    }

    /// char_repeat is `W` in the `16/L + W` formula: it changes the fill
    /// cost per cell but not the tag count, since the color sequence is
    /// unchanged.
    #[test]
    fn char_repeat_changes_fill_cost_not_tag_count() {
        let img = row(&[RED; 4]);
        let w1 = measure_frame(&img, &opts(1)).unwrap();
        let w2 = measure_frame(&img, &opts(2)).unwrap();
        assert_eq!(w1.tags, 1);
        assert_eq!(w2.tags, 1);
        assert_eq!(w1.chars, 16 + 4 * 1);
        assert_eq!(w2.chars, 16 + 4 * 2);
    }

    /// A run continues across row boundaries (matching `encode_bands`'s own
    /// documented behavior): two uniform rows of the same color is still
    /// exactly one tag, not two.
    #[test]
    fn a_run_spans_row_boundaries() {
        let mut img = RgbaImage::new(3, 2);
        for y in 0..2 {
            for x in 0..3 {
                img.put_pixel(x, y, RED);
            }
        }
        let stats = measure_frame(&img, &opts(2)).unwrap();
        assert_eq!(stats.cells, 6);
        assert_eq!(stats.tags, 1, "one color run must span the row boundary");
        // 16 (tag) + '\n' (1, the row separator) + 3*2 + 3*2 (fill on both rows)
        assert_eq!(stats.chars, 16 + 1 + 3 * 2 + 3 * 2);
    }

    /// An entirely transparent frame writes no tag at all: `mean_run_length`
    /// must report that as infinite rather than dividing by zero.
    #[test]
    fn an_all_transparent_frame_has_no_finite_run_length() {
        let img = RgbaImage::new(4, 1);
        let stats = measure_frame(&img, &opts(2)).unwrap();
        assert_eq!(stats.tags, 0);
        assert!(stats.mean_run_length().is_infinite());
    }

    // --- quantization -------------------------------------------------------

    #[test]
    fn quantize_channel_is_exact_at_1_bit() {
        // 1 bit: 2 levels, step = 255.0. 0 -> level 0 -> 0; 255 -> level 1
        // (255/255 = 1.0) -> 255; 128 -> 128/255 = 0.50196 rounds to level 1
        // -> 255.
        assert_eq!(quantize_channel(0, 1), 0);
        assert_eq!(quantize_channel(255, 1), 255);
        assert_eq!(quantize_channel(128, 1), 255);
    }

    #[test]
    fn quantize_channel_is_exact_at_2_bits() {
        // 2 bits: 4 levels, step = 85.0. Levels land exactly on 0/85/170/255.
        assert_eq!(quantize_channel(0, 2), 0);
        assert_eq!(quantize_channel(85, 2), 85);
        assert_eq!(quantize_channel(170, 2), 170);
        assert_eq!(quantize_channel(255, 2), 255);
        // 128 / 85 = 1.5058... rounds to level 2 -> 170.
        assert_eq!(quantize_channel(128, 2), 170);
    }

    #[test]
    fn full_24_bit_is_a_no_op() {
        for v in [0u8, 1, 17, 128, 254, 255] {
            assert_eq!(quantize_channel(v, 8), v);
        }
    }

    #[test]
    fn alpha_is_always_preserved_exactly() {
        for palette in Palette::ALL {
            let p = quantize_pixel([200, 100, 50, 137], palette.bits());
            assert_eq!(p[3], 137, "{palette:?} must not touch alpha");
        }
    }

    #[test]
    fn palette_bit_splits_match_their_advertised_color_count() {
        for (palette, expected) in [
            (Palette::Full24, 1u32 << 24),
            (Palette::C256, 256),
            (Palette::C64, 64),
            (Palette::C32, 32),
            (Palette::C16, 16),
        ] {
            assert_eq!(palette.colors(), expected, "{palette:?}");
        }
    }

    // --- median-cut -----------------------------------------------------------

    fn histogram_of(entries: &[([u8; 3], u64)]) -> HashMap<[u8; 3], u64> {
        entries.iter().copied().collect()
    }

    /// An image containing exactly N distinct colors, quantized to a
    /// palette of N entries, must reproduce it exactly: every color becomes
    /// its own singleton bucket, so nearest-mapping is the identity and MSE
    /// is zero. Weighted unevenly (5/3/7/1) so the split isn't accidentally
    /// balanced.
    #[test]
    fn median_cut_reproduces_exactly_when_palette_size_equals_distinct_colors() {
        let hist = histogram_of(&[
            ([255, 0, 0], 5),
            ([0, 255, 0], 3),
            ([0, 0, 255], 7),
            ([255, 255, 255], 1),
        ]);
        let palette = median_cut_palette(&hist, 4);
        assert_eq!(palette.len(), 4, "must use exactly one entry per distinct color");
        let palette_set: HashSet<[u8; 3]> = palette.iter().copied().collect();
        let original_set: HashSet<[u8; 3]> = hist.keys().copied().collect();
        assert_eq!(palette_set, original_set, "each color must map to itself exactly");

        // Nearest-mapping every original color must be the identity (each
        // is present verbatim in the palette, so distance-0 always wins),
        // and applying that map to an image must round-trip with zero MSE.
        let map = nearest_map(&hist, &palette);
        for &c in hist.keys() {
            assert_eq!(map[&c], c);
        }
        let img = row(&[RED, GREEN, BLUE, WHITE]);
        let quantized = apply_color_map(&img, &map);
        let fidelity = mse(&[&img], std::slice::from_ref(&quantized));
        assert_eq!(fidelity.mse, 0.0, "exact reproduction must have zero error");
    }

    /// Quantizing to a single palette entry must yield exactly one color:
    /// the ultimate case of "the palette never exceeds the requested size."
    #[test]
    fn median_cut_to_one_entry_yields_a_single_color() {
        let hist = histogram_of(&[([255, 0, 0], 10), ([0, 255, 0], 1), ([0, 0, 255], 1)]);
        let palette = median_cut_palette(&hist, 1);
        assert_eq!(palette.len(), 1);
    }

    /// The palette must never exceed the requested size, at any size,
    /// including when there are far more distinct colors available than
    /// requested.
    #[test]
    fn median_cut_palette_never_exceeds_requested_size() {
        // 50 distinct colors (only R varies, so distinctness is trivially
        // guaranteed): more than every requested size below.
        let hist: HashMap<[u8; 3], u64> = (0..50u32).map(|i| ([i as u8, 0, 0], 1u64)).collect();
        for k in [1usize, 5, 16, 32, 64] {
            let palette = median_cut_palette(&hist, k);
            assert!(
                palette.len() <= k,
                "k={k}: palette of {} entries exceeds the requested size",
                palette.len()
            );
        }
    }

    /// An image with FEWER distinct colors than the requested palette size
    /// must not crash, must not pad with duplicates, and must stop at the
    /// number of distinct colors actually present.
    #[test]
    fn median_cut_with_fewer_colors_than_requested_has_no_duplicates() {
        let hist = histogram_of(&[([10, 20, 30], 4), ([200, 100, 0], 9), ([5, 5, 5], 1)]);
        let palette = median_cut_palette(&hist, 10);
        assert_eq!(palette.len(), 3, "must stop at the number of distinct colors present");
        let unique: HashSet<[u8; 3]> = palette.iter().copied().collect();
        assert_eq!(unique.len(), palette.len(), "palette must not contain duplicate entries");
    }

    /// An empty histogram (a degenerate input, e.g. a zero-size image) must
    /// not crash and must simply produce an empty palette.
    #[test]
    fn median_cut_of_empty_histogram_is_empty() {
        let hist: HashMap<[u8; 3], u64> = HashMap::new();
        assert!(median_cut_palette(&hist, 16).is_empty());
    }

    /// Median-cut, like the uniform quantizer, must never touch alpha.
    #[test]
    fn median_cut_preserves_alpha_exactly() {
        let hist = histogram_of(&[([200, 100, 50], 3), ([10, 10, 10], 1)]);
        let palette = median_cut_palette(&hist, 2);
        let map = nearest_map(&hist, &palette);
        let mut img = RgbaImage::new(1, 1);
        img.put_pixel(0, 0, Rgba([200, 100, 50, 137]));
        let out = apply_color_map(&img, &map);
        assert_eq!(out.get_pixel(0, 0).0[3], 137, "must not touch alpha");
    }

    // --- fidelity ---------------------------------------------------------

    /// Hand-computed MSE: pixel 0 is unchanged (zero error); pixel 1 differs
    /// by (-10, +10, +10) per channel, contributing 3*100 = 300 to the
    /// squared-error sum. Total squared error 300 over 2 pixels * 3 channels
    /// = 6 samples -> MSE = 50.0 exactly. A denominator bug (e.g. dividing
    /// by pixel count instead of pixel*channel count) would give 150 here,
    /// not 50.
    #[test]
    fn mse_matches_hand_computed_value() {
        let mut orig = RgbaImage::new(2, 1);
        orig.put_pixel(0, 0, Rgba([10, 20, 30, 255]));
        orig.put_pixel(1, 0, Rgba([100, 150, 200, 255]));
        let mut quant = RgbaImage::new(2, 1);
        quant.put_pixel(0, 0, Rgba([10, 20, 30, 255]));
        quant.put_pixel(1, 0, Rgba([110, 140, 190, 255]));
        let fidelity = mse(&[&orig], std::slice::from_ref(&quant));
        assert_eq!(fidelity.mse, 50.0);
    }

    /// Alpha differences must not affect MSE: only RGB is quantized by
    /// either method, so only RGB error should be measured.
    #[test]
    fn mse_ignores_alpha_differences() {
        let mut orig = RgbaImage::new(1, 1);
        orig.put_pixel(0, 0, Rgba([50, 60, 70, 255]));
        let mut quant = RgbaImage::new(1, 1);
        quant.put_pixel(0, 0, Rgba([50, 60, 70, 0]));
        let fidelity = mse(&[&orig], std::slice::from_ref(&quant));
        assert_eq!(fidelity.mse, 0.0, "alpha-only difference must not count as error");
    }

    /// Hand-computed PSNR: MSE = 650.25 means RMSE = 25.5 exactly
    /// (25.5^2 = 650.25), and 255 / 25.5 = 10 exactly, so
    /// PSNR = 20*log10(10) = 20.0 dB exactly -- deliberately chosen so the
    /// expected value is derivable by hand, not just by re-running the
    /// formula under test.
    #[test]
    fn psnr_matches_hand_computed_value() {
        let fidelity = Fidelity { mse: 650.25 };
        assert!((fidelity.psnr() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn psnr_is_infinite_at_zero_error() {
        let fidelity = Fidelity { mse: 0.0 };
        assert!(fidelity.psnr().is_infinite());
    }

    // --- sampling ------------------------------------------------------------

    #[test]
    fn even_sampling_covers_the_whole_range_not_just_the_start() {
        let frames: Vec<RgbaImage> = (0..100)
            .map(|i| RgbaImage::from_pixel(1, 1, Rgba([i as u8, 0, 0, 255])))
            .collect();
        let sampled = sample_evenly(&frames, 10);
        assert_eq!(sampled.len(), 10);
        // Evenly spread indices i*100/10 = 0,10,20,...,90 -- not 0..10.
        let values: Vec<u8> = sampled.iter().map(|f| f.get_pixel(0, 0).0[0]).collect();
        assert_eq!(values, vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
    }

    #[test]
    fn sampling_more_than_available_returns_every_frame() {
        let frames: Vec<RgbaImage> = (0..5).map(|_| RgbaImage::new(1, 1)).collect();
        assert_eq!(sample_evenly(&frames, 120).len(), 5);
    }

    #[test]
    fn sampling_an_empty_clip_returns_nothing() {
        let frames: Vec<RgbaImage> = Vec::new();
        assert!(sample_evenly(&frames, 120).is_empty());
    }
}
