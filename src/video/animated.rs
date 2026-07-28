//! Decode animated GIF, APNG, and animated WebP into a [`Clip`].
//!
//! `image`'s per-format decoders already do the hard part: every [`Frame`]
//! handed back by `into_frames()` is a fully-composited, full-canvas RGBA
//! buffer with `left`/`top` always zero -- this holds across all four GIF
//! disposal methods, APNG `DisposeOp`/`BlendOp`, and animated WebP frame
//! stacking. This module does no compositing of its own; it only sniffs the
//! format, drives the right decoder, and turns per-frame delays into a
//! single `fps` for the [`Clip`].

use std::io::Cursor;

use image::codecs::gif::GifDecoder;
use image::codecs::png::PngDecoder;
use image::codecs::webp::WebPDecoder;
use image::{AnimationDecoder, Frame, ImageDecoder, ImageFormat, Limits};

use super::Clip;

/// `image` returns a GIF (or APNG) `delay == 0` verbatim -- some encoders
/// emit that to mean "as fast as possible". Browsers clamp it to 100ms
/// instead of treating it as instantaneous, and we do the same: left
/// unclamped, a zero-delay animation would produce an infinite `fps`, which
/// `video::scale::FpsStream` downstream rejects as a hard error rather than
/// an unbounded but working frame rate.
const ZERO_DELAY_CLAMP_MS: f64 = 100.0;

/// Decode an animated GIF, APNG, or animated WebP into a [`Clip`].
///
/// The format is sniffed from the leading bytes (`image::guess_format`).
/// GIF and WebP dispatch straight to their `AnimationDecoder`; PNG is
/// checked for an `acTL` chunk (`is_apng`) first, since a plain
/// (non-animated) PNG decodes fine as a still image but has no frame
/// sequence to build a `Clip` from.
///
/// `Clip::fps` is derived from the mean of the per-frame delays (see
/// [`mean_fps`]), so a source with variable per-frame timing is resampled to
/// a single constant rate -- `Clip` has no per-frame timestamps, only one
/// `fps` for the whole sequence.
pub fn decode_animated(bytes: &[u8]) -> Result<Clip, String> {
    let format =
        image::guess_format(bytes).map_err(|e| format!("could not identify image format: {e}"))?;

    let frames = match format {
        ImageFormat::Gif => decode_gif(bytes)?,
        ImageFormat::Png => decode_apng(bytes)?,
        ImageFormat::WebP => decode_webp(bytes)?,
        other => return Err(format!("{other:?} is not a supported animated format")),
    };

    frames_to_clip(frames)
}

fn decode_gif(bytes: &[u8]) -> Result<Vec<Frame>, String> {
    let mut decoder =
        GifDecoder::new(Cursor::new(bytes)).map_err(|e| format!("GIF decode error: {e}"))?;
    // Limits must be set before `into_frames()` consumes the decoder --
    // setting them after would have nothing left to configure.
    decoder
        .set_limits(Limits::default())
        .map_err(|e| format!("GIF decode error: {e}"))?;
    decoder
        .into_frames()
        .collect_frames()
        .map_err(|e| format!("GIF decode error: {e}"))
}

fn decode_apng(bytes: &[u8]) -> Result<Vec<Frame>, String> {
    let mut decoder =
        PngDecoder::new(Cursor::new(bytes)).map_err(|e| format!("PNG decode error: {e}"))?;
    decoder
        .set_limits(Limits::default())
        .map_err(|e| format!("PNG decode error: {e}"))?;
    if !decoder
        .is_apng()
        .map_err(|e| format!("PNG decode error: {e}"))?
    {
        return Err("PNG is not animated (no acTL chunk / not an APNG)".to_string());
    }
    // `apng()` wraps the still-image decoder in an `ApngDecoder`, which only
    // implements `AnimationDecoder` (not `ImageDecoder`) -- so limits must
    // be set on `decoder` above, before this conversion, not after.
    let apng = decoder
        .apng()
        .map_err(|e| format!("APNG decode error: {e}"))?;
    apng.into_frames()
        .collect_frames()
        .map_err(|e| format!("APNG decode error: {e}"))
}

fn decode_webp(bytes: &[u8]) -> Result<Vec<Frame>, String> {
    let mut decoder =
        WebPDecoder::new(Cursor::new(bytes)).map_err(|e| format!("WebP decode error: {e}"))?;
    decoder
        .set_limits(Limits::default())
        .map_err(|e| format!("WebP decode error: {e}"))?;
    if !decoder.has_animation() {
        return Err("WebP is not animated".to_string());
    }
    decoder
        .into_frames()
        .collect_frames()
        .map_err(|e| format!("WebP decode error: {e}"))
}

/// Build a [`Clip`] from the decoded frame sequence.
///
/// Frame dimensions (rather than a decoder's own `dimensions()`) are used
/// for `width`/`height` because `ApngDecoder` does not implement
/// `ImageDecoder` -- and because every frame is already full-canvas, reading
/// it off the first frame is equivalent and works uniformly for all three
/// formats.
fn frames_to_clip(frames: Vec<Frame>) -> Result<Clip, String> {
    let first = frames
        .first()
        .ok_or_else(|| "decoded animation has no frames".to_string())?;
    let (width, height) = first.buffer().dimensions();
    let fps = mean_fps(&frames);

    Ok(Clip {
        width,
        height,
        fps,
        frames: frames.into_iter().map(Frame::into_buffer).collect(),
    })
}

/// Derive a single frames-per-second value from the mean of the per-frame
/// delays. `Clip` carries one `fps` for the whole sequence, not per-frame
/// timestamps, so a source with variable delays (common in hand-tuned GIFs)
/// is deliberately averaged down to a constant rate rather than rejected or
/// arbitrarily taking the first frame's delay.
///
/// Always divides `numer_denom_ms()`'s numerator by its denominator instead
/// of assuming the denominator is 1: GIF delays are always whole
/// centiseconds so the denominator happens to be 1, but APNG delays are true
/// rationals (e.g. an NTSC-rate 1001/24 ms delay) and assuming denom == 1
/// would silently produce wrong frame timing for APNG only.
///
/// Zero (or sub-millisecond-rounds-to-zero) delays are clamped to
/// [`ZERO_DELAY_CLAMP_MS`] before averaging, matching browser behavior.
fn mean_fps(frames: &[Frame]) -> f32 {
    let total_ms: f64 = frames
        .iter()
        .map(|f| {
            let (numer, denom) = f.delay().numer_denom_ms();
            let ms = f64::from(numer) / f64::from(denom);
            if ms <= 0.0 { ZERO_DELAY_CLAMP_MS } else { ms }
        })
        .sum();
    let mean_ms = total_ms / frames.len() as f64;
    (1000.0 / mean_ms) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2-frame 2x1 GIF: frame 0 red+blue, frame 1 uses a partial-rect
    /// update touching only the left pixel. If compositing were skipped, the
    /// right pixel of frame 1 would come back transparent.
    fn two_frame_gif() -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = image::codecs::gif::GifEncoder::new(&mut out);
            enc.set_repeat(image::codecs::gif::Repeat::Infinite).unwrap();
            for c in [[255u8, 0, 0, 255], [0, 255, 0, 255]] {
                let mut img = image::RgbaImage::new(2, 1);
                img.put_pixel(0, 0, image::Rgba(c));
                img.put_pixel(1, 0, image::Rgba([0, 0, 255, 255]));
                enc.encode_frame(image::Frame::from_parts(
                    img,
                    0,
                    0,
                    image::Delay::from_numer_denom_ms(100, 1),
                ))
                .unwrap();
            }
        }
        out
    }

    #[test]
    fn decodes_every_frame_at_full_canvas_size() {
        let clip = decode_animated(&two_frame_gif()).unwrap();
        assert_eq!(clip.frames.len(), 2);
        assert_eq!((clip.width, clip.height), (2, 1));
        for f in &clip.frames {
            assert_eq!(f.dimensions(), (2, 1), "frames must be full-canvas");
        }
    }

    #[test]
    fn derives_fps_from_frame_delays() {
        let clip = decode_animated(&two_frame_gif()).unwrap();
        assert!((clip.fps - 10.0).abs() < 0.01, "100ms delay is 10fps, got {}", clip.fps);
    }

    #[test]
    fn rejects_bytes_that_are_not_an_animation() {
        assert!(decode_animated(b"not an image at all").is_err());
    }

    /// Regression test: a zero-delay GIF must not yield an infinite fps.
    /// `image` returns GIF `delay == 0` verbatim; without the 100ms clamp
    /// this divides to `fps == inf`, which `FpsStream` downstream treats
    /// as a hard error instead of a working 10fps animation.
    fn zero_delay_gif() -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = image::codecs::gif::GifEncoder::new(&mut out);
            enc.set_repeat(image::codecs::gif::Repeat::Infinite).unwrap();
            for c in [[255u8, 0, 0, 255], [0, 255, 0, 255]] {
                let mut img = image::RgbaImage::new(2, 1);
                img.put_pixel(0, 0, image::Rgba(c));
                img.put_pixel(1, 0, image::Rgba([0, 0, 255, 255]));
                enc.encode_frame(image::Frame::from_parts(
                    img,
                    0,
                    0,
                    image::Delay::from_numer_denom_ms(0, 1),
                ))
                .unwrap();
            }
        }
        out
    }

    #[test]
    fn zero_delay_frames_clamp_to_a_finite_fps() {
        let clip = decode_animated(&zero_delay_gif()).unwrap();
        assert!(clip.fps.is_finite(), "zero delay must not yield infinite fps");
        assert!(
            (clip.fps - 10.0).abs() < 0.01,
            "0ms delay clamps to 100ms (10fps), got {}",
            clip.fps
        );
    }
}
