//! Resizing and frame-rate resampling for [`Clip`]s.

use image::{Rgba, RgbaImage, imageops};

use super::Clip;

/// How a frame's aspect ratio is reconciled with a target box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitMode {
    /// Stretch to exactly `(w, h)`, ignoring the source aspect ratio.
    Exact,
    /// Scale to fit entirely inside `(w, h)`, preserving aspect ratio, and
    /// center on a fully transparent canvas (letterbox/pillarbox).
    Contain,
    /// Scale to fill `(w, h)` entirely, preserving aspect ratio, and
    /// center-crop whatever overhangs the box.
    Cover,
}

/// Resampling filter used when scaling frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// High quality, slower. Maps to [`imageops::FilterType::Lanczos3`].
    Lanczos,
    /// Blocky, fast, no new colors introduced. Maps to
    /// [`imageops::FilterType::Nearest`].
    Nearest,
}

impl Filter {
    fn to_filter_type(self) -> imageops::FilterType {
        match self {
            Filter::Lanczos => imageops::FilterType::Lanczos3,
            Filter::Nearest => imageops::FilterType::Nearest,
        }
    }
}

/// Resize every frame of `clip` to `(w, h)` under the given [`FitMode`] and
/// [`Filter`]. The returned clip's `width`/`height` are always `(w, h)`;
/// `fps` and frame count are unchanged.
pub fn resize_clip(clip: Clip, w: u32, h: u32, fit: FitMode, filter: Filter) -> Clip {
    let filter_type = filter.to_filter_type();
    let frames = clip
        .frames
        .into_iter()
        .map(|frame| resize_frame(&frame, w, h, fit, filter_type))
        .collect();
    Clip {
        width: w,
        height: h,
        fps: clip.fps,
        frames,
    }
}

fn resize_frame(
    frame: &RgbaImage,
    w: u32,
    h: u32,
    fit: FitMode,
    filter_type: imageops::FilterType,
) -> RgbaImage {
    match fit {
        FitMode::Exact => imageops::resize(frame, w, h, filter_type),
        FitMode::Contain => {
            let (sw, sh) = scale_to_fit(frame.width(), frame.height(), w, h, f64::min);
            let scaled = imageops::resize(frame, sw, sh, filter_type);
            // Fully transparent, not black: downstream, transparent pixels
            // are culled and emit no brick, so letterboxing is free. Black
            // would emit a wall of black bricks.
            let mut canvas = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0]));
            let x = (w.saturating_sub(sw)) / 2;
            let y = (h.saturating_sub(sh)) / 2;
            imageops::overlay(&mut canvas, &scaled, x as i64, y as i64);
            canvas
        }
        FitMode::Cover => {
            let (sw, sh) = scale_to_fit(frame.width(), frame.height(), w, h, f64::max);
            let scaled = imageops::resize(frame, sw, sh, filter_type);
            let x = (sw.saturating_sub(w)) / 2;
            let y = (sh.saturating_sub(h)) / 2;
            imageops::crop_imm(&scaled, x, y, w, h).to_image()
        }
    }
}

/// Scale `(sw, sh)` into box `(bw, bh)`, combining the two axis scale
/// factors with `combine` (`f64::min` for contain-inside, `f64::max` for
/// cover-and-crop). Never returns a zero dimension.
fn scale_to_fit(sw: u32, sh: u32, bw: u32, bh: u32, combine: fn(f64, f64) -> f64) -> (u32, u32) {
    let scale = combine(bw as f64 / sw as f64, bh as f64 / sh as f64);
    let w = ((sw as f64 * scale).round() as u32).max(1);
    let h = ((sh as f64 * scale).round() as u32).max(1);
    (w, h)
}

/// Resample `clip` to `target_fps`, taking the source frame nearest
/// `start_s + i / target_fps` (in source-clip time) for each output frame
/// `i`, starting from `start_s` seconds into the source and running for
/// `duration_s` seconds (or to the end of the source if `None`).
///
/// Errors, naming `max_frames`, if the requested output would exceed it —
/// this never silently truncates. A caller that asks for more frames than a
/// target format allows must be told, not handed a shorter animation.
pub fn resample_fps(
    clip: Clip,
    target_fps: f32,
    start_s: f32,
    duration_s: Option<f32>,
    max_frames: usize,
) -> Result<Clip, String> {
    let n = clip.frames.len();
    if n == 0 {
        return Ok(Clip {
            width: clip.width,
            height: clip.height,
            fps: target_fps,
            frames: Vec::new(),
        });
    }
    if !target_fps.is_finite() || target_fps <= 0.0 {
        return Err(format!(
            "resample_fps: target_fps must be positive and finite, got {target_fps}"
        ));
    }
    if !clip.fps.is_finite() || clip.fps <= 0.0 {
        return Err(format!(
            "resample_fps: source clip fps must be positive and finite, got {}",
            clip.fps
        ));
    }

    let source_duration = n as f32 / clip.fps;
    let end_time = duration_s.map_or(source_duration, |d| (start_s + d).min(source_duration));

    // First pass: compute source indices without cloning frame data, so an
    // over-limit request fails cheaply rather than after doing real work.
    //
    // The limit is enforced *inside* the loop, not after it. `target_fps` is
    // only checked for finiteness above, so a large-but-finite value (say
    // 1e9) would otherwise step `t` in 1e-9s increments across the whole
    // source and push billions of indices before anything rejected it —
    // an OOM instead of an error message. Bounded here, the work done on
    // the rejection path never exceeds the caller's own stated budget.
    let mut indices: Vec<usize> = Vec::new();
    let mut i: usize = 0;
    let over_limit = loop {
        let t = start_s + i as f32 / target_fps;
        if t >= end_time {
            break false;
        }
        if indices.len() >= max_frames {
            // Another output frame is due but the budget is already full,
            // so the true total is *more than* max_frames. We deliberately
            // stop without measuring exactly how much more.
            break true;
        }
        let src = ((t * clip.fps).round() as i64).clamp(0, n as i64 - 1) as usize;
        indices.push(src);
        i += 1;
    };

    if over_limit {
        // Phrased as "more than N" because breaking early means we never
        // counted the real total -- the message must not claim a number it
        // did not measure.
        return Err(format!(
            "resample_fps would produce more than the max_frames limit of {max_frames} frames; \
             lower the target fps or shorten the duration"
        ));
    }

    let frames = indices
        .into_iter()
        .map(|idx| clip.frames[idx].clone())
        .collect();

    Ok(Clip {
        width: clip.width,
        height: clip.height,
        fps: target_fps,
        frames,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    fn solid(w: u32, h: u32, c: [u8; 4]) -> RgbaImage {
        RgbaImage::from_pixel(w, h, Rgba(c))
    }

    fn clip(n: usize, fps: f32) -> Clip {
        Clip {
            width: 4,
            height: 4,
            fps,
            frames: (0..n).map(|i| solid(4, 4, [i as u8, 0, 0, 255])).collect(),
        }
    }

    #[test]
    fn halving_fps_takes_every_other_frame() {
        let out = resample_fps(clip(10, 20.0), 10.0, 0.0, None, 1000).unwrap();
        assert_eq!(out.frames.len(), 5);
        assert_eq!(out.fps, 10.0);
        // frame i of the output is source frame nearest i/10 seconds
        assert_eq!(out.frames[1].get_pixel(0, 0).0[0], 2);
    }

    #[test]
    fn start_offset_skips_leading_frames() {
        let out = resample_fps(clip(10, 10.0), 10.0, 0.5, None, 1000).unwrap();
        assert_eq!(out.frames[0].get_pixel(0, 0).0[0], 5);
    }

    #[test]
    fn duration_truncates() {
        let out = resample_fps(clip(10, 10.0), 10.0, 0.0, Some(0.3), 1000).unwrap();
        assert_eq!(out.frames.len(), 3);
    }

    #[test]
    fn exceeding_max_frames_is_an_error_not_a_silent_truncation() {
        let err = resample_fps(clip(10, 10.0), 10.0, 0.0, None, 4).unwrap_err();
        assert!(err.contains("65535") || err.contains("frame"), "got: {err}");
    }

    #[test]
    fn contain_preserves_aspect_and_pads_to_the_box() {
        let c = Clip {
            width: 8,
            height: 4,
            fps: 1.0,
            frames: vec![solid(8, 4, [255, 0, 0, 255])],
        };
        let out = resize_clip(c, 4, 4, FitMode::Contain, Filter::Nearest);
        assert_eq!((out.width, out.height), (4, 4));
        assert_eq!(out.frames[0].dimensions(), (4, 4));
        // letterbox rows are transparent, not black
        assert_eq!(out.frames[0].get_pixel(0, 0).0[3], 0);
    }

    #[test]
    fn exact_ignores_aspect() {
        let c = Clip {
            width: 8,
            height: 4,
            fps: 1.0,
            frames: vec![solid(8, 4, [255, 0, 0, 255])],
        };
        let out = resize_clip(c, 3, 7, FitMode::Exact, Filter::Nearest);
        assert_eq!(out.frames[0].dimensions(), (3, 7));
    }

    // --- behaviour the brief left unpinned, pinned down here ---

    #[test]
    fn max_frames_error_names_the_limit() {
        let err = resample_fps(clip(10, 10.0), 10.0, 0.0, None, 4).unwrap_err();
        assert!(
            err.contains('4'),
            "error should name the max_frames limit: {err}"
        );
        // The count is deliberately *not* measured (the loop breaks as soon
        // as the budget is blown), so the message must say "more than N"
        // rather than assert an exact total it never counted.
        assert!(
            err.contains("more than the max_frames limit of 4 frames"),
            "message must stay truthful about the unmeasured total: {err}"
        );
    }

    #[test]
    fn absurdly_high_target_fps_errors_promptly_instead_of_hanging() {
        // 2 frames @ 1fps = 2s of source. At 1e9 fps the loop steps t by
        // 1e-9s, so generating every index before checking the limit would
        // push ~2 billion usizes (~16 GB) before rejecting. target_fps is
        // finite, so the is_finite/positive guard does not catch this --
        // only the in-loop bound does.
        let start = std::time::Instant::now();
        let err = resample_fps(clip(2, 1.0), 1e9, 0.0, None, 10).unwrap_err();
        let elapsed = start.elapsed();
        assert!(err.contains("10"), "should name the limit: {err}");
        assert!(err.contains("frame"), "got: {err}");
        // Microseconds in practice; a full second means it is unbounded again.
        assert!(elapsed.as_secs() < 1, "took too long: {elapsed:?}");
    }

    #[test]
    fn huge_duration_is_bounded_by_the_source_length() {
        // end_time is min(start_s + duration_s, source_duration), so an
        // absurd duration cannot drive the loop past the end of the source.
        // 10 frames @ 10fps = 1.0s, so this yields the whole source.
        let start = std::time::Instant::now();
        let out = resample_fps(clip(10, 10.0), 10.0, 0.0, Some(1e9), 1000).unwrap();
        assert_eq!(out.frames.len(), 10);
        assert!(start.elapsed().as_secs() < 1, "duration_s is not bounded");
    }

    #[test]
    fn exactly_max_frames_is_accepted() {
        // Boundary: max_frames is inclusive. 10 frames @10fps -> 10 frames.
        let out = resample_fps(clip(10, 10.0), 10.0, 0.0, None, 10).unwrap();
        assert_eq!(out.frames.len(), 10);
        // ...and one fewer is the error case.
        assert!(resample_fps(clip(10, 10.0), 10.0, 0.0, None, 9).is_err());
    }

    #[test]
    fn resampling_up_repeats_source_frames_instead_of_erroring() {
        // target_fps > source fps: some source frames are sampled more than
        // once. This must not move-out and panic on a double index.
        let out = resample_fps(clip(4, 4.0), 8.0, 0.0, None, 1000).unwrap();
        assert_eq!(out.frames.len(), 8);
    }

    #[test]
    fn empty_clip_resamples_to_empty_without_error() {
        let out = resample_fps(clip(0, 10.0), 5.0, 0.0, None, 10).unwrap();
        assert_eq!(out.frames.len(), 0);
        assert_eq!(out.fps, 5.0);
    }

    #[test]
    fn non_positive_target_fps_is_an_error() {
        let err = resample_fps(clip(4, 10.0), 0.0, 0.0, None, 10).unwrap_err();
        assert!(err.contains("target_fps"), "got: {err}");
    }

    #[test]
    fn cover_fills_the_box_and_crops_the_overhang() {
        // 8x4 source into a 4x4 box: contain would be 4x2, cover fills the
        // box completely (no transparent border) by cropping the sides.
        let c = Clip {
            width: 8,
            height: 4,
            fps: 1.0,
            frames: vec![solid(8, 4, [255, 0, 0, 255])],
        };
        let out = resize_clip(c, 4, 4, FitMode::Cover, Filter::Nearest);
        assert_eq!(out.frames[0].dimensions(), (4, 4));
        // fully opaque everywhere -- no letterboxing under Cover
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out.frames[0].get_pixel(x, y).0[3], 255);
            }
        }
    }

    #[test]
    fn contain_preserves_source_pixels_at_the_center() {
        let c = Clip {
            width: 8,
            height: 4,
            fps: 1.0,
            frames: vec![solid(8, 4, [255, 0, 0, 255])],
        };
        let out = resize_clip(c, 4, 4, FitMode::Contain, Filter::Nearest);
        // the scaled 4x2 content is centered at rows 1..3
        assert_eq!(out.frames[0].get_pixel(0, 1).0, [255, 0, 0, 255]);
        assert_eq!(out.frames[0].get_pixel(3, 2).0, [255, 0, 0, 255]);
    }
}
