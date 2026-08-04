//! Streaming per-frame band encoder for text mode.
//!
//! This is the sibling of [`super::pack::Packer`] and [`super::color_pack::ColorPacker`]:
//! push one frame at a time, retain no images, and finish into the per-band
//! arrays a renderer wires up. Unlike those two, this one is band-major on
//! the output ([`TextPacker::finish`] returns `bands[band][frame]`) because
//! that is exactly the shape one band's `ArrayVar` needs -- one string per
//! frame, for that band alone.
//!
//! # Memory
//!
//! Unlike brick mode's fixed 6 bytes per pixel per frame (or colour-array
//! mode's fixed 16), text mode's retention is content-dependent and larger:
//! measured at ~25 MB for 60 frames at 192x108 unquantized, which
//! extrapolates to roughly 3 GB for ten minutes at 12 fps. A palette cuts
//! that by about a third (shorter strings from longer colour runs). This is
//! stated here plainly so nobody discovers it at 20 minutes into a render.
//!
//! # Frame size validation
//!
//! [`TextPacker::new`] takes `width` and `height` explicitly, exactly as
//! [`super::pack::Packer::new`] does, and every frame -- including the first --
//! is checked against them.
//!
//! A [`BandPlan`] list carries row ranges only, so the width it was computed
//! from is not recoverable from it (for a screen shorter than one band's row
//! capacity, every width up to the per-row character cap yields the identical
//! single-band plan). Inferring the expected size from the first frame instead
//! would therefore accept whatever arrived first -- which is precisely the case
//! worth rejecting. `build_text_world` sizes the plan from the source's
//! `SourceInfo`, and a `FrameStream` emitting frames that disagree with the
//! `info()` it reported is a contract violation
//! ([`crate::video::stream::FrameStream::next`]) that must fail loudly and
//! immediately, not silently redefine the layout the bands were built for.
//!
//! # Colour state
//!
//! [`crate::text::encode_bands`] carries its last emitted `<color>` tag
//! across pixels, gaps and rows within a band, starting fresh at each band.
//! Feeding it one band's rows as its own sub-image (below) gives exactly
//! that behaviour for free -- fresh colour state per band, per frame -- which
//! is also what [`super::text_layout::worst_case_row_chars`]'s bound assumes.
//! Colour state is never threaded between bands.
use crate::anim::palette::Palette;
use crate::anim::text_layout::BandPlan;
use crate::text::{MAX_COMPONENT_CHARS, TextOptions, encode_bands};
use image::RgbaImage;

/// Builds one string per frame per band, in one traversal of the frames, so
/// no frame is ever retained -- only the encoded strings are.
pub struct TextPacker {
    plan: Vec<BandPlan>,
    opts: TextOptions,
    palette: Palette,
    /// The size every frame must be, from the source's `SourceInfo`. See the
    /// module doc's "Frame size validation" section for why this is taken
    /// explicitly rather than inferred from the first frame.
    width: usize,
    height: usize,
    /// `bands[i]` holds band `i`'s strings, one per frame pushed so far.
    bands: Vec<Vec<String>>,
    frames_pushed: usize,
    /// Overrides [`super::pack::MAX_FRAMES`] in tests. Building a real
    /// 1,048,560-frame clip to exercise the limit would cost minutes of CPU
    /// and gigabytes of RAM -- the same reasoning that made
    /// [`super::pack::bank_frames`] take `bank_size` as a parameter rather
    /// than reading [`super::pack::BANK_FRAMES`] directly.
    #[cfg(test)]
    frame_limit: usize,
}

impl TextPacker {
    /// `width`/`height` are the source's reported dimensions; every pushed
    /// frame must match them exactly. `plan` is normally
    /// [`super::text_layout::plan_bands`]'s own output, unmodified, and must
    /// have been computed for these same dimensions. `palette` may be
    /// [`Palette::default()`] (empty), which leaves every pixel's colour
    /// untouched.
    pub fn new(
        width: u32,
        height: u32,
        plan: Vec<BandPlan>,
        opts: TextOptions,
        palette: Palette,
    ) -> Self {
        let bands = vec![Vec::new(); plan.len()];
        Self {
            plan,
            opts,
            palette,
            width: width as usize,
            height: height as usize,
            bands,
            frames_pushed: 0,
            #[cfg(test)]
            frame_limit: super::pack::MAX_FRAMES,
        }
    }

    /// `#[cfg(test)]`-only: see the field doc on `frame_limit`.
    #[cfg(test)]
    pub fn set_frame_limit_for_test(&mut self, limit: usize) {
        self.frame_limit = limit;
    }

    #[cfg(test)]
    fn frame_limit(&self) -> usize {
        self.frame_limit
    }

    #[cfg(not(test))]
    fn frame_limit(&self) -> usize {
        super::pack::MAX_FRAMES
    }

    /// Push one frame's contribution to every band's string array.
    ///
    /// `frame` must be exactly the `width` x `height` this packer was built
    /// with -- the dimensions the source's `SourceInfo` reported, and the ones
    /// the band plan was computed for. A mismatch is a `FrameStream` contract
    /// violation and is rejected here, on the first frame as much as the last;
    /// see the module doc's "Frame size validation" section.
    pub fn push_frame(&mut self, frame: &RgbaImage) -> Result<(), String> {
        let limit = self.frame_limit();
        if self.frames_pushed >= limit {
            return Err(format!(
                "clip exceeds the {}-frame limit ({} banks of {})",
                super::pack::MAX_FRAMES,
                super::pack::MAX_BANKS,
                super::pack::BANK_FRAMES
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

        // Quantize once, up front, so every band crops from the same
        // post-palette pixels. Skipped entirely for an empty palette so the
        // unquantized path allocates nothing extra.
        let quantized;
        let src: &RgbaImage = if self.palette.is_empty() {
            frame
        } else {
            let mut q = frame.clone();
            for p in q.pixels_mut() {
                let mapped = self.palette.map([p.0[0], p.0[1], p.0[2]]);
                p.0[0] = mapped[0];
                p.0[1] = mapped[1];
                p.0[2] = mapped[2];
            }
            quantized = q;
            &quantized
        };

        // Every band's string is produced up front and only then appended, so
        // an error in band 7 cannot leave bands 0..7 one frame longer than the
        // rest -- the ragged accumulator the previous push-as-you-go loop left
        // behind on its (bug-only) error paths.
        let encoded = encode_all_bands(&self.plan, src, frame_w, &self.opts)?;
        for (band, text) in self.bands.iter_mut().zip(encoded) {
            band.push(text);
        }

        self.frames_pushed += 1;
        Ok(())
    }

    /// The per-band string arrays: outer index is band, inner is frame.
    pub fn finish(self) -> Vec<Vec<String>> {
        self.bands
    }
}

/// One band's string for one frame: crop its rows out of `src` and encode them
/// as a single `TextDisplay` component's worth of text.
///
/// Takes the band's own index only to name it in the two "this is a bug in
/// this codebase" errors below.
fn encode_band(
    bi: usize,
    band: &BandPlan,
    src: &RgbaImage,
    frame_w: usize,
    opts: &TextOptions,
) -> Result<String, String> {
    let sub = image::imageops::crop_imm(
        src,
        0,
        band.start_row as u32,
        frame_w as u32,
        band.rows as u32,
    )
    .to_image();
    let band_opts = TextOptions {
        tile_override: Some(band.rows as u32),
        ..opts.clone()
    };
    let mut encoded = encode_bands(&sub, &band_opts)?;
    if encoded.len() != 1 {
        let chars: usize = encoded.iter().map(|b| b.chars).sum();
        return Err(format!(
            "band {bi} (row {}): the fixed layout produced {} TextBands (expected \
             exactly 1) totaling {chars} chars -- the worst-case bound in text_layout \
             and the encoder in text.rs disagree, which is a bug in this codebase, not \
             a user error",
            band.start_row,
            encoded.len()
        ));
    }
    let text_band = encoded.pop().expect("checked len == 1 above");
    debug_assert!(
        text_band.chars <= MAX_COMPONENT_CHARS,
        "band {bi}: {} chars exceeds the {MAX_COMPONENT_CHARS}-char TextDisplay limit \
         -- the layout bound should have made this impossible",
        text_band.chars
    );
    if text_band.chars > MAX_COMPONENT_CHARS {
        return Err(format!(
            "band {bi} (row {}): {} chars exceeds the {MAX_COMPONENT_CHARS}-char \
             TextDisplay limit -- the game would truncate this silently; the \
             worst-case layout bound and the real encoder disagree, which is a bug",
            band.start_row, text_band.chars
        ));
    }
    Ok(text_band.text)
}

/// Fewest bands rayon may put in one job. See `super::pack`'s
/// `MIN_CHUNKS_PER_JOB` for why the split is bounded at all: the fan-out
/// happens once per frame, so jobs that finish in microseconds are not worth
/// dispatching individually.
#[cfg(not(target_arch = "wasm32"))]
const MIN_BANDS_PER_JOB: usize = 4;

/// Every band's string for one frame, in band order.
///
/// The bands are independent by construction -- each crops its own disjoint
/// rows and starts with fresh colour state (see the module doc's "Colour
/// state" section, which is why nothing is threaded between them) -- so this
/// fans out across rayon's pool on native. `rayon` is a native-only dependency
/// (see `Cargo.toml`), so wasm gets the identical serial walk below.
///
/// Errors are collected per band and then reduced in index order, so the
/// error a caller sees is the lowest-numbered failing band whichever order the
/// jobs actually finished in. A parallel run must not report a different
/// failure than a serial one.
#[cfg(not(target_arch = "wasm32"))]
fn encode_all_bands(
    plan: &[BandPlan],
    src: &RgbaImage,
    frame_w: usize,
    opts: &TextOptions,
) -> Result<Vec<String>, String> {
    use rayon::prelude::*;
    let mut out: Vec<Result<String, String>> = Vec::with_capacity(plan.len());
    plan.par_iter()
        .enumerate()
        .with_min_len(MIN_BANDS_PER_JOB)
        .map(|(bi, band)| encode_band(bi, band, src, frame_w, opts))
        .collect_into_vec(&mut out);
    out.into_iter().collect()
}

/// See the native sibling above.
#[cfg(target_arch = "wasm32")]
fn encode_all_bands(
    plan: &[BandPlan],
    src: &RgbaImage,
    frame_w: usize,
    opts: &TextOptions,
) -> Result<Vec<String>, String> {
    plan.iter()
        .enumerate()
        .map(|(bi, band)| encode_band(bi, band, src, frame_w, opts))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::text_layout::plan_bands;
    use crate::text::{FontPreset, MAX_COMPONENT_CHARS};
    use image::{Rgba, RgbaImage};

    fn opts() -> TextOptions {
        FontPreset::MonaspaceArgon.options(1.0)
    }

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> RgbaImage {
        RgbaImage::from_pixel(w, h, Rgba([rgb[0], rgb[1], rgb[2], 0xFF]))
    }

    #[test]
    fn one_entry_per_frame_per_band() {
        let plan = plan_bands(8, 4, 2).unwrap();
        let bands = plan.len();
        let mut p = TextPacker::new(8, 4, plan, opts(), Palette::default());
        for _ in 0..5 {
            p.push_frame(&solid(8, 4, [0xFF, 0, 0])).unwrap();
        }
        let out = p.finish();
        assert_eq!(out.len(), bands);
        for band in &out {
            assert_eq!(band.len(), 5, "one string per frame");
        }
    }

    #[test]
    fn a_wrongly_sized_frame_is_rejected_naming_both_sizes() {
        let plan = plan_bands(8, 4, 2).unwrap();
        let mut p = TextPacker::new(8, 4, plan, opts(), Palette::default());
        // No priming push: the very first frame is checked, because the
        // expected size comes from the source's SourceInfo rather than from
        // whatever happened to arrive first. A stream whose first frame
        // already disagrees with info() is exactly the case worth catching.
        let err = p.push_frame(&solid(9, 4, [0, 0, 0])).expect_err("must reject");
        assert!(err.contains("9x4"), "names the actual size: {err}");
        assert!(err.contains("8x4"), "names the expected size: {err}");
    }

    #[test]
    fn no_band_string_ever_exceeds_the_component_limit() {
        // Worst case content: every pixel a different colour, so every pixel
        // opens its own tag.
        let (w, h) = (192u32, 8u32);
        let mut f = RgbaImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let n = (y * w + x) as u32;
                f.put_pixel(
                    x,
                    y,
                    Rgba([(n & 0xFF) as u8, ((n >> 8) & 0xFF) as u8, ((n >> 3) & 0xFF) as u8, 0xFF]),
                );
            }
        }
        let plan = plan_bands(w as usize, h as usize, 2).unwrap();
        let mut p = TextPacker::new(w, h, plan, opts(), Palette::default());
        p.push_frame(&f).unwrap();
        for band in p.finish() {
            for s in band {
                assert!(
                    s.chars().count() <= MAX_COMPONENT_CHARS,
                    "{} chars is over the limit",
                    s.chars().count()
                );
            }
        }
    }

    #[test]
    fn the_palette_is_applied_before_encoding() {
        // A 1-entry palette forces every pixel to one colour, so a frame of
        // two different colours must encode with exactly one tag.
        let mut f = RgbaImage::new(2, 1);
        f.put_pixel(0, 0, Rgba([0xFF, 0, 0, 0xFF]));
        f.put_pixel(1, 0, Rgba([0, 0, 0xFF, 0xFF]));
        let pal = Palette::build(&[f.clone()], 1, 1);
        let plan = plan_bands(2, 1, 2).unwrap();
        let mut p = TextPacker::new(2, 1, plan, opts(), pal);
        p.push_frame(&f).unwrap();
        let out = p.finish();
        assert_eq!(out[0][0].matches("<color=").count(), 1, "one colour, one tag");
    }

    #[test]
    fn an_empty_palette_leaves_colours_untouched() {
        let mut f = RgbaImage::new(1, 1);
        f.put_pixel(0, 0, Rgba([0x12, 0x34, 0x56, 0xFF]));
        let plan = plan_bands(1, 1, 2).unwrap();
        let mut p = TextPacker::new(1, 1, plan, opts(), Palette::default());
        p.push_frame(&f).unwrap();
        assert!(p.finish()[0][0].contains("123456"), "exact source colour");
    }

    #[test]
    fn pushing_past_the_frame_limit_errors_rather_than_truncating() {
        let plan = plan_bands(1, 1, 2).unwrap();
        let mut p = TextPacker::new(1, 1, plan, opts(), Palette::default());
        p.set_frame_limit_for_test(2);
        p.push_frame(&solid(1, 1, [0, 0, 0])).unwrap();
        p.push_frame(&solid(1, 1, [0, 0, 0])).unwrap();
        assert!(p.push_frame(&solid(1, 1, [0, 0, 0])).is_err(), "third must fail");
    }
}
