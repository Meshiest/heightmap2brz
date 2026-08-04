//! Resizing and frame-rate resampling for [`Clip`](super::Clip)s.
//!
//! The whole-clip `resize_clip`/`resample_fps` functions that used to live
//! here are gone: every call site now goes through the streaming adapters
//! below (`ResizeStream`/`FpsStream`, composed by
//! `crate::video::stream::AdaptedSource`) instead of materializing a second
//! full `Clip`.

use image::{Rgba, RgbaImage, imageops};

#[cfg(test)]
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

pub(crate) fn resize_frame(
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

use crate::video::stream::FrameStream;

/// Scales each frame as it passes. Frame count and order are unchanged.
pub struct ResizeStream<'a> {
    inner: Box<dyn FrameStream + 'a>,
    w: u32,
    h: u32,
    fit: FitMode,
    filter: imageops::FilterType,
}

impl<'a> ResizeStream<'a> {
    pub fn new(
        inner: Box<dyn FrameStream + 'a>,
        w: u32,
        h: u32,
        fit: FitMode,
        filter: Filter,
    ) -> Self {
        Self { inner, w, h, fit, filter: filter.to_filter_type() }
    }
}

impl ResizeStream<'_> {
    /// Resize `f`, unless it is already exactly the target size.
    ///
    /// This skip is what lets a backend do its own scaling without this
    /// stream redoing it. `FfmpegSource` scales inside its own filtergraph
    /// when handed a target, which is enormously cheaper: it then pipes
    /// target-sized frames instead of piping native-resolution ones for this
    /// to shrink. On a 1080p source that is ~8.3 MB per frame over the pipe
    /// versus ~9 KB at 64x36 -- for a 24-minute episode, ~280 GB of pipe
    /// traffic versus ~300 MB.
    ///
    /// `BuiltinVideoSource` takes no size parameter at all, so its frames
    /// arrive at native resolution and are resized here exactly as before.
    /// Both backends therefore still end at the same dimensions -- the
    /// property `main.rs` was protecting by passing `None` and resizing only
    /// here, so that `--width`/`--height` can never work on one backend and
    /// silently do nothing on the other.
    fn fit_frame(&self, f: RgbaImage) -> RgbaImage {
        if f.width() == self.w && f.height() == self.h {
            return f;
        }
        resize_frame(&f, self.w, self.h, self.fit, self.filter)
    }
}

impl FrameStream for ResizeStream<'_> {
    fn next(&mut self) -> Result<Option<RgbaImage>, String> {
        Ok(self.inner.next()?.map(|f| self.fit_frame(f)))
    }

    /// One resize per `advance`, not `n` of them.
    ///
    /// The default body would call `next` `n` times and throw away all but
    /// the last result -- i.e. run a full Lanczos resize on `n - 1` frames
    /// nobody ever looks at. `FpsStream` sits directly on top of this stream
    /// (see `AdaptedSource::open`) and passes over exactly those frames on
    /// every downsampling render, so this is the common case, not a corner
    /// one. Forwarding the `advance` down and resizing only the frame that
    /// comes back is identical in output -- selecting frames and resizing
    /// them are independent per-frame operations -- and does `1/n` the work.
    fn advance(&mut self, n: usize) -> Result<(usize, Option<RgbaImage>), String> {
        let (got, last) = self.inner.advance(n)?;
        Ok((got, last.map(|f| self.fit_frame(f))))
    }
}

/// The error [`FpsStream`] produces when a render would exceed `max_frames`.
///
/// Shared rather than inlined so the CLI's pre-flight budget check (which
/// must refuse before opening the stream) reports identical wording rather
/// than a second hand-written copy that could drift. Phrased as "more than
/// N": the true total is never counted (both stop as soon as the budget is
/// known to be blown), so the message must not claim a number it never
/// measured.
pub fn max_frames_error(max_frames: usize) -> String {
    format!(
        "resampling would produce more than the max_frames limit of {max_frames} frames; \
         lower the target fps or shorten the duration"
    )
}

/// Frame count an [`FpsStream`] would emit over a `source_frames`-long
/// source, computed by running the SAME per-iteration `t` recurrence
/// `FpsStream::next` does (a closed-form `ceil()` drifts +/-1 from it in
/// `f32` on a measurable fraction of inputs -- equal over the reals, not
/// once rounded). Bounded by `max_frames`, matching where `FpsStream` itself
/// would error rather than finish counting; returns 0 for a degenerate rate
/// rather than erroring (`FpsStream::new` rejects those; this is only ever a
/// sizing hint ahead of that).
///
/// Must agree with `FpsStream` exactly, not approximately: the CLI uses it
/// to refuse an over-budget render before printing anything, so a count one
/// too high refuses a render that would have worked, and one too low prints
/// a cost line the render then contradicts.
/// `estimated_frame_count_matches_fps_stream_over_thousands_of_configs` pins
/// the two together over a generated sweep.
pub fn estimated_frame_count(
    source_frames: usize,
    source_fps: f32,
    target_fps: f32,
    start_s: f32,
    duration_s: Option<f32>,
    max_frames: usize,
) -> usize {
    if !source_fps.is_finite() || source_fps <= 0.0 || !target_fps.is_finite() || target_fps <= 0.0
    {
        return 0;
    }
    let start_s = start_s.max(0.0);
    // Unlike `FpsStream` (which only learns this once the stream itself
    // drains), the true source duration is known immediately here: the
    // caller already knows `source_frames` up front.
    let source_end = source_frames as f32 / source_fps;
    let window_end = duration_s.map(|d| start_s + d.max(0.0));

    let mut emitted: usize = 0;
    loop {
        // Mirrors `FpsStream::next`'s `t` and its two `past_end` checks
        // (`end_time` from the caller's window, `source_end` from the
        // source's own length) exactly -- same expression, same order of
        // operations, so there is no separate rounding path left to drift.
        let t = start_s + emitted as f32 / target_fps;
        if window_end.is_some_and(|e| t >= e) || t >= source_end {
            break;
        }
        emitted += 1;
        if emitted > max_frames {
            // A frame is due, but the budget is already spent -- the exact
            // point where `FpsStream::next` would return `Err` instead of
            // counting further. `emitted` is already `> max_frames`, the
            // only property `est > max_frames` needs, so stopping here both
            // matches the stream's own refusal point and bounds this loop
            // against a runaway `target_fps`.
            break;
        }
    }
    emitted
}

/// Converts source fps to target fps by selecting source frames, windowed by
/// `start_s`/`duration_s` and bounded by `max_frames`.
///
/// Mirrors `resample_fps`'s index arithmetic exactly, including its refusal
/// to truncate: when another output frame is due but the budget is full, this
/// ERRORS. Silently stopping would hand back a short clip with no indication.
pub struct FpsStream<'a> {
    inner: Box<dyn FrameStream + 'a>,
    /// Source frames already pulled.
    pulled: usize,
    /// Most recent source frame, held so one source frame can be emitted
    /// more than once when upsampling.
    held: Option<RgbaImage>,
    emitted: usize,
    source_fps: f32,
    target_fps: f32,
    start_s: f32,
    end_time: Option<f32>,
    /// The source's true duration, learned only when the source drains.
    ///
    /// `resample_fps` has this up front as `n / clip.fps` and folds it into
    /// `end_time` via `min`. A stream cannot know it until it hits the end,
    /// so it is filled in at that moment and then bounds the output the same
    /// way. Without it, clamping to the held frame (below) would emit the
    /// last frame forever instead of ending the clip.
    source_end: Option<f32>,
    max_frames: usize,
    done: bool,
}

impl<'a> FpsStream<'a> {
    pub fn new(
        inner: Box<dyn FrameStream + 'a>,
        source_fps: f32,
        target_fps: f32,
        start_s: f32,
        duration_s: Option<f32>,
        max_frames: usize,
    ) -> Result<Self, String> {
        if !target_fps.is_finite() || target_fps <= 0.0 {
            return Err(format!(
                "FpsStream: target_fps must be positive and finite, got {target_fps}"
            ));
        }
        if !source_fps.is_finite() || source_fps <= 0.0 {
            return Err(format!(
                "FpsStream: source fps must be positive and finite, got {source_fps}"
            ));
        }
        Ok(Self {
            inner,
            pulled: 0,
            held: None,
            emitted: 0,
            source_fps,
            target_fps,
            start_s: start_s.max(0.0),
            end_time: duration_s.map(|d| start_s.max(0.0) + d.max(0.0)),
            source_end: None,
            max_frames,
            done: false,
        })
    }

    /// Whether output time `t` is past the end of what should be emitted:
    /// the caller's `duration_s` window, or the source's own length once
    /// that is known. Together these are `resample_fps`'s
    /// `min(start_s + duration_s, source_duration)`.
    fn past_end(&self, t: f32) -> bool {
        self.end_time.is_some_and(|e| t >= e) || self.source_end.is_some_and(|e| t >= e)
    }

    /// Pull source frames until `want` has been read, holding the last.
    ///
    /// Returns false only when the source yielded nothing at all -- a
    /// genuine end of stream. Running past the last index is not that:
    /// `resample_fps` clamps its index, so an upsample whose output time
    /// still falls inside the source repeats the final frame instead.
    ///
    /// One [`FrameStream::advance`] call, not a `next()` loop, so a stream
    /// that overrides `advance` (`ResizeStream` above, the builtin decoder's
    /// own) skips per-frame work for the frames strictly between `pulled`
    /// and `want`, which nothing ever looks at.
    fn advance_to(&mut self, want: usize) -> Result<bool, String> {
        if self.pulled > want {
            // The old `while` loop's condition, false on entry: already past
            // `want`, so nothing to pull and `held` is already the right frame.
            return Ok(true);
        }
        let need = want + 1 - self.pulled;
        let (got, last) = self.inner.advance(need)?;
        self.pulled += got;
        if last.is_some() {
            self.held = last;
        }
        if got < need {
            // Drained: the source's true length is now known, so record the
            // duration that bounds the output. `held` is the last frame,
            // which is what `want` clamps onto.
            self.source_end = Some(self.pulled as f32 / self.source_fps);
            return Ok(self.held.is_some());
        }
        Ok(true)
    }
}

impl FrameStream for FpsStream<'_> {
    fn next(&mut self) -> Result<Option<RgbaImage>, String> {
        if self.done {
            return Ok(None);
        }
        let t = self.start_s + self.emitted as f32 / self.target_fps;
        if self.past_end(t) {
            self.done = true;
            return Ok(None);
        }
        let want = (t * self.source_fps).round().max(0.0) as usize;
        if !self.advance_to(want)? {
            self.done = true;
            return Ok(None);
        }
        // `advance_to` may have just learned the source's length; re-test so
        // this doesn't emit one frame past the source's end.
        if self.past_end(t) {
            self.done = true;
            return Ok(None);
        }
        // A frame is genuinely due and the budget is already full, so the
        // true total is more than max_frames. Checked after the two bounds
        // above so a frame that was never due isn't charged against the
        // budget -- a request that exactly fills it must not error.
        if self.emitted >= self.max_frames {
            self.done = true;
            return Err(max_frames_error(self.max_frames));
        }
        self.emitted += 1;
        Ok(self.held.clone())
    }
}

#[cfg(test)]
mod advance_tests {
    use super::*;
    use crate::video::stream::FrameStream;
    use image::{Rgba, RgbaImage};
    use std::cell::Cell;
    use std::rc::Rc;

    /// A source stream that records which of the two pull APIs was used on
    /// it. It DOES override `advance`, so a caller that forwards reaches the
    /// override and a caller that falls back to the default body does not --
    /// which is exactly the distinction these tests need to see.
    struct Recording {
        frames: Vec<RgbaImage>,
        at: usize,
        nexts: Rc<Cell<usize>>,
        advances: Rc<Cell<usize>>,
    }

    impl FrameStream for Recording {
        fn next(&mut self) -> Result<Option<RgbaImage>, String> {
            self.nexts.set(self.nexts.get() + 1);
            let out = self.frames.get(self.at).cloned();
            if out.is_some() {
                self.at += 1;
            }
            Ok(out)
        }

        fn advance(&mut self, n: usize) -> Result<(usize, Option<RgbaImage>), String> {
            self.advances.set(self.advances.get() + 1);
            let take = n.min(self.frames.len() - self.at);
            self.at += take;
            Ok((take, take.gt(&0).then(|| self.frames[self.at - 1].clone())))
        }
    }

    fn frames(n: usize) -> Vec<RgbaImage> {
        (0..n).map(|i| RgbaImage::from_pixel(4, 4, Rgba([i as u8, 1, 2, 255]))).collect()
    }

    /// `ResizeStream::advance` is an optimization, so it must agree with the
    /// default body -- `n` calls to `next`, keeping the last -- frame for
    /// frame and count for count, including once the source has drained.
    #[test]
    fn resize_stream_advance_matches_repeated_next() {
        for step in 1..=4 {
            let (nexts, advances) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
            let src = |a: &Rc<Cell<usize>>, b: &Rc<Cell<usize>>| Recording {
                frames: frames(7),
                at: 0,
                nexts: a.clone(),
                advances: b.clone(),
            };
            let mut by_next = ResizeStream::new(
                Box::new(src(&nexts, &advances)),
                3,
                2,
                FitMode::Exact,
                Filter::Nearest,
            );
            let mut by_advance = ResizeStream::new(
                Box::new(src(&nexts, &advances)),
                3,
                2,
                FitMode::Exact,
                Filter::Nearest,
            );
            loop {
                let mut want_n = 0;
                let mut want_last = None;
                for _ in 0..step {
                    match by_next.next().expect("next") {
                        Some(f) => {
                            want_last = Some(f);
                            want_n += 1;
                        }
                        None => break,
                    }
                }
                let (got_n, got_last) = by_advance.advance(step).expect("advance");
                assert_eq!((got_n, got_last), (want_n, want_last), "step {step}");
                if want_n == 0 {
                    break;
                }
            }
        }
    }

    /// The whole point of the `advance` path, stated as a property rather
    /// than a timing: on a downsampling render the chain must ask the SOURCE
    /// to pass over the frames it is dropping, instead of pulling each one
    /// through `next` and resizing it first.
    ///
    /// A regression here -- someone reverting `ResizeStream::advance`, or
    /// rewriting `FpsStream::advance_to` back into a `next` loop -- is
    /// invisible in output and only shows up as the render getting ~2x
    /// slower, so it is worth pinning directly.
    #[test]
    fn a_downsampling_chain_passes_frames_over_instead_of_resizing_them() {
        let (nexts, advances) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
        let inner = Recording {
            frames: frames(30),
            at: 0,
            nexts: nexts.clone(),
            advances: advances.clone(),
        };
        // 30fps source, 10fps target: two frames in three are passed over.
        let resized = ResizeStream::new(Box::new(inner), 3, 2, FitMode::Exact, Filter::Nearest);
        let mut fps = FpsStream::new(Box::new(resized), 30.0, 10.0, 0.0, None, 1000).expect("fps");
        let mut emitted = 0;
        while fps.next().expect("next").is_some() {
            emitted += 1;
        }
        assert_eq!(emitted, 10, "30 frames at 30fps resampled to 10fps");
        assert_eq!(
            nexts.get(),
            0,
            "no frame may be pulled one at a time -- that is the path that resizes every \
             source frame, including the 20 this render never looks at"
        );
        assert!(
            advances.get() <= emitted + 1,
            "at most one advance per emitted frame (plus the one that finds the end), got {}",
            advances.get()
        );
    }
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

    // --- ported from `resample_fps`'s own tests, now exercised through
    // `FpsStream` since the whole-clip function is gone. Same setups and
    // assertions as before, just driven through `.open()` + `drain` instead
    // of a single call. ---

    #[test]
    fn fps_stream_halving_fps_takes_every_other_frame() {
        let c = clip(10, 20.0);
        let s = FpsStream::new(c.open().unwrap(), 20.0, 10.0, 0.0, None, 1000).expect("build");
        let out = drain(Box::new(s)).expect("drain");
        assert_eq!(out.len(), 5);
        // frame i of the output is source frame nearest i/10 seconds
        assert_eq!(out[1].get_pixel(0, 0).0[0], 2);
    }

    #[test]
    fn fps_stream_start_offset_skips_leading_frames() {
        let c = clip(10, 10.0);
        let s = FpsStream::new(c.open().unwrap(), 10.0, 10.0, 0.5, None, 1000).expect("build");
        let out = drain(Box::new(s)).expect("drain");
        assert_eq!(out[0].get_pixel(0, 0).0[0], 5);
    }

    #[test]
    fn fps_stream_duration_truncates() {
        let c = clip(10, 10.0);
        let s = FpsStream::new(c.open().unwrap(), 10.0, 10.0, 0.0, Some(0.3), 1000).expect("build");
        assert_eq!(drain(Box::new(s)).expect("drain").len(), 3);
    }

    #[test]
    fn fps_stream_exceeding_max_frames_is_an_error_not_a_silent_truncation() {
        let c = clip(10, 10.0);
        let s = FpsStream::new(c.open().unwrap(), 10.0, 10.0, 0.0, None, 4).expect("build");
        let err = drain(Box::new(s)).expect_err("must refuse");
        assert!(err.contains("frame"), "got: {err}");
    }

    #[test]
    fn resize_stream_contain_preserves_aspect_and_pads_to_the_box() {
        let c = Clip {
            width: 8,
            height: 4,
            fps: 1.0,
            frames: vec![solid(8, 4, [255, 0, 0, 255])],
        };
        let s = ResizeStream::new(c.open().unwrap(), 4, 4, FitMode::Contain, Filter::Nearest);
        let out = drain(Box::new(s)).expect("drain");
        assert_eq!(out[0].dimensions(), (4, 4));
        // letterbox rows are transparent, not black
        assert_eq!(out[0].get_pixel(0, 0).0[3], 0);
    }

    #[test]
    fn resize_stream_exact_ignores_aspect() {
        let c = Clip {
            width: 8,
            height: 4,
            fps: 1.0,
            frames: vec![solid(8, 4, [255, 0, 0, 255])],
        };
        let s = ResizeStream::new(c.open().unwrap(), 3, 7, FitMode::Exact, Filter::Nearest);
        let out = drain(Box::new(s)).expect("drain");
        assert_eq!(out[0].dimensions(), (3, 7));
    }

    // --- behaviour the brief left unpinned, pinned down here ---

    #[test]
    fn fps_stream_max_frames_error_names_the_limit() {
        let c = clip(10, 10.0);
        let s = FpsStream::new(c.open().unwrap(), 10.0, 10.0, 0.0, None, 4).expect("build");
        let err = drain(Box::new(s)).expect_err("must refuse");
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
    fn fps_stream_absurdly_high_target_fps_errors_promptly_instead_of_hanging() {
        // 2 frames @ 1fps = 2s of source. At 1e9 fps consecutive output
        // times are 1e-9s apart, so pulling a source frame per output frame
        // before checking the limit would need ~2 billion iterations before
        // rejecting. target_fps is finite, so the is_finite/positive guard
        // does not catch this -- only the emitted-count bound in `next` does.
        let c = clip(2, 1.0);
        let s = FpsStream::new(c.open().unwrap(), 1.0, 1e9, 0.0, None, 10).expect("build");
        let start = std::time::Instant::now();
        let err = drain(Box::new(s)).expect_err("must refuse");
        let elapsed = start.elapsed();
        assert!(err.contains("10"), "should name the limit: {err}");
        assert!(err.contains("frame"), "got: {err}");
        // Microseconds in practice; a full second means it is unbounded again.
        assert!(elapsed.as_secs() < 1, "took too long: {elapsed:?}");
    }

    #[test]
    fn fps_stream_huge_duration_is_bounded_by_the_source_length() {
        // end_time is min(start_s + duration_s, source_duration), so an
        // absurd duration cannot drive the stream past the end of the
        // source. 10 frames @ 10fps = 1.0s, so this yields the whole source.
        let c = clip(10, 10.0);
        let s = FpsStream::new(c.open().unwrap(), 10.0, 10.0, 0.0, Some(1e9), 1000).expect("build");
        let start = std::time::Instant::now();
        let out = drain(Box::new(s)).expect("drain");
        assert_eq!(out.len(), 10);
        assert!(start.elapsed().as_secs() < 1, "duration_s is not bounded");
    }

    #[test]
    fn fps_stream_resampling_up_repeats_source_frames_instead_of_erroring() {
        // target_fps > source fps: some source frames are sampled more than
        // once. This must not move-out and panic on a double index.
        let c = clip(4, 4.0);
        let s = FpsStream::new(c.open().unwrap(), 4.0, 8.0, 0.0, None, 1000).expect("build");
        assert_eq!(drain(Box::new(s)).expect("drain").len(), 8);
    }

    #[test]
    fn fps_stream_empty_clip_resamples_to_empty_without_error() {
        let c = clip(0, 10.0);
        let s = FpsStream::new(c.open().unwrap(), 10.0, 5.0, 0.0, None, 10).expect("build");
        assert_eq!(drain(Box::new(s)).expect("drain").len(), 0);
    }

    #[test]
    fn fps_stream_non_positive_target_fps_is_an_error() {
        let c = clip(4, 10.0);
        // `FpsStream` doesn't derive `Debug`, so `Result::unwrap_err` (which
        // requires the `Ok` side to be `Debug`) isn't available here --
        // match instead.
        match FpsStream::new(c.open().unwrap(), 10.0, 0.0, 0.0, None, 10) {
            Err(err) => assert!(err.contains("target_fps"), "got: {err}"),
            Ok(_) => panic!("a target_fps of 0 must be rejected"),
        }
    }

    #[test]
    fn resize_stream_cover_fills_the_box_and_crops_the_overhang() {
        // 8x4 source into a 4x4 box: contain would be 4x2, cover fills the
        // box completely (no transparent border) by cropping the sides.
        let c = Clip {
            width: 8,
            height: 4,
            fps: 1.0,
            frames: vec![solid(8, 4, [255, 0, 0, 255])],
        };
        let s = ResizeStream::new(c.open().unwrap(), 4, 4, FitMode::Cover, Filter::Nearest);
        let out = drain(Box::new(s)).expect("drain");
        assert_eq!(out[0].dimensions(), (4, 4));
        // fully opaque everywhere -- no letterboxing under Cover
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out[0].get_pixel(x, y).0[3], 255);
            }
        }
    }

    #[test]
    fn resize_stream_contain_preserves_source_pixels_at_the_center() {
        let c = Clip {
            width: 8,
            height: 4,
            fps: 1.0,
            frames: vec![solid(8, 4, [255, 0, 0, 255])],
        };
        let s = ResizeStream::new(c.open().unwrap(), 4, 4, FitMode::Contain, Filter::Nearest);
        let out = drain(Box::new(s)).expect("drain");
        // the scaled 4x2 content is centered at rows 1..3
        assert_eq!(out[0].get_pixel(0, 1).0, [255, 0, 0, 255]);
        assert_eq!(out[0].get_pixel(3, 2).0, [255, 0, 0, 255]);
    }

    // --- streaming adapters ---

    use crate::video::stream::{FrameSource, FrameStream};

    fn src(n: usize, fps: f32) -> Clip {
        let frames = (0..n)
            .map(|i| RgbaImage::from_pixel(4, 2, Rgba([i as u8, 0, 0, 255])))
            .collect();
        Clip { width: 4, height: 2, fps, frames }
    }

    fn drain(mut s: Box<dyn FrameStream + '_>) -> Result<Vec<RgbaImage>, String> {
        let mut out = Vec::new();
        while let Some(f) = s.next()? {
            out.push(f);
        }
        Ok(out)
    }

    #[test]
    fn resize_stream_scales_every_frame() {
        let c = src(3, 10.0);
        let s = ResizeStream::new(c.open().unwrap(), 8, 4, FitMode::Exact, Filter::Nearest);
        let got = drain(Box::new(s)).expect("drain");
        assert_eq!(got.len(), 3, "resizing must not change the frame count");
        for f in &got {
            assert_eq!(f.dimensions(), (8, 4));
        }
    }

    #[test]
    fn fps_stream_halving_the_rate_halves_the_frames() {
        let c = src(10, 10.0);
        let s = FpsStream::new(c.open().unwrap(), 10.0, 5.0, 0.0, None, 1000).expect("build");
        assert_eq!(drain(Box::new(s)).expect("drain").len(), 5);
    }

    /// The OOM guard, as a test. A large-but-finite target fps must be
    /// rejected rather than stepping through the source in nanosecond
    /// increments and pushing billions of frames.
    #[test]
    fn fps_stream_errors_rather_than_truncating_past_max_frames() {
        let c = src(100, 10.0);
        let s = FpsStream::new(c.open().unwrap(), 10.0, 1.0e9, 0.0, None, 50).expect("build");
        let err = drain(Box::new(s)).expect_err("must refuse");
        assert!(err.contains("50"), "error must name the limit: {err}");
        assert!(
            err.contains("more than"),
            "the message must not claim a total it never counted: {err}"
        );
    }

    #[test]
    fn fps_stream_rejects_a_non_positive_or_infinite_rate() {
        let c = src(4, 10.0);
        for bad in [0.0f32, -1.0, f32::INFINITY, f32::NAN] {
            assert!(
                FpsStream::new(c.open().unwrap(), 10.0, bad, 0.0, None, 100).is_err(),
                "target fps {bad} must be rejected"
            );
        }
        assert!(
            FpsStream::new(c.open().unwrap(), 0.0, 10.0, 0.0, None, 100).is_err(),
            "a source fps of 0 must be rejected"
        );
    }

    #[test]
    fn start_and_duration_window_the_stream() {
        let c = src(20, 10.0);
        // 2s in, 1s long, at the source rate -> 10 frames
        let s = FpsStream::new(c.open().unwrap(), 10.0, 10.0, 0.5, Some(0.5), 1000).expect("build");
        assert_eq!(drain(Box::new(s)).expect("drain").len(), 5);
    }

    /// The red channel of each frame from `src`, which is its source index --
    /// enough to identify *which* source frame each output frame came from.
    fn reds(frames: &[RgbaImage]) -> Vec<u8> {
        frames.iter().map(|f| f.get_pixel(0, 0).0[0]).collect()
    }

    #[test]
    fn fps_stream_repeats_the_only_frame_instead_of_ending_early() {
        // 1 frame at 1fps is 1 second of source, so upsampling to 2fps has a
        // second output frame due at t=0.5 -- still inside the source. It
        // must repeat frame 0, which is what `resample_fps`'s index clamp
        // does. Ending the stream here returns half the clip with no error.
        let c = src(1, 1.0);
        let s = FpsStream::new(c.open().unwrap(), 1.0, 2.0, 0.0, None, 1000).expect("build");
        let got = drain(Box::new(s)).expect("drain");
        assert_eq!(reds(&got), vec![0, 0], "the final frame must repeat, not vanish");
    }

    #[test]
    fn fps_stream_upsampling_repeats_the_true_final_frame() {
        // 3 frames at 2fps is 1.5s; at 3fps that is 5 output frames, the
        // last of which samples past the final source index and must clamp
        // back onto it. Integer-ratio rates, so no float-rounding is
        // involved -- the clamp is what this depends on.
        let c = src(3, 2.0);
        let s = FpsStream::new(c.open().unwrap(), 2.0, 3.0, 0.0, None, 1000).expect("build");
        let got = drain(Box::new(s)).expect("drain");
        assert_eq!(reds(&got), vec![0, 1, 1, 2, 2]);
    }

    /// The assertion that catches this whole class of defect rather than one
    /// instance of it. This used to compare `FpsStream`'s output against
    /// `resample_fps`'s directly, frame for frame, across the 16
    /// configurations below -- and that comparison is what caught a Critical
    /// bug where `FpsStream` silently dropped the final frame on upsampling,
    /// which every other test here passed straight through.
    ///
    /// `resample_fps` is deleted now, so it can no longer serve as the
    /// oracle. Its outputs for these exact cases were captured while it
    /// still existed (each frame's red channel is its source index, per the
    /// `src` helper below) and are hardcoded as `expected` so this test keeps
    /// pinning down the same behavior instead of losing its oracle along
    /// with the function.
    #[test]
    fn fps_stream_matches_resample_fps_frame_for_frame() {
        // (frames, source fps, target fps, start_s, duration_s, expected reds)
        let cases: [(usize, f32, f32, f32, Option<f32>, &[u8]); 16] = [
            (1, 1.0, 2.0, 0.0, None, &[0, 0]),
            (3, 2.0, 3.0, 0.0, None, &[0, 1, 1, 2, 2]),
            (2, 1.0, 3.0, 0.0, None, &[0, 0, 1, 1, 1, 1]),
            (4, 4.0, 8.0, 0.0, None, &[0, 1, 1, 2, 2, 3, 3, 3]),
            (5, 3.0, 7.0, 0.0, None, &[0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4]),
            (10, 10.0, 5.0, 0.0, None, &[0, 2, 4, 6, 8]),
            (10, 10.0, 10.0, 0.0, None, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (7, 30.0, 24.0, 0.0, None, &[0, 1, 3, 4, 5, 6]),
            (9, 12.0, 5.0, 0.0, None, &[0, 2, 5, 7]),
            (20, 10.0, 10.0, 0.5, Some(0.5), &[5, 6, 7, 8, 9]),
            (20, 10.0, 6.0, 0.3, Some(1.0), &[3, 5, 6, 8, 10, 11]),
            (6, 5.0, 5.0, 0.0, Some(10.0), &[0, 1, 2, 3, 4, 5]),
            // A start offset partway into the source, where the window ends
            // mid-frame rather than on a frame boundary.
            (1, 1.0, 2.0, 0.5, None, &[0]),
            (1, 1.0, 5.0, 0.1, None, &[0, 0, 0, 0, 0]),
            (
                3,
                2.0,
                60.0,
                0.1,
                Some(0.5),
                &[
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1,
                ],
            ),
            // Empty source: no frames, no error.
            (0, 10.0, 10.0, 0.0, None, &[]),
        ];
        for (n, source_fps, target_fps, start_s, duration_s, expected) in cases {
            let label = format!("n={n} {source_fps}fps -> {target_fps}fps start={start_s} dur={duration_s:?}");
            let c = src(n, source_fps);
            let s = FpsStream::new(c.open().unwrap(), source_fps, target_fps, start_s, duration_s, 1000)
                .expect("build");
            let got = drain(Box::new(s)).unwrap_or_else(|e| panic!("FpsStream failed for {label}: {e}"));
            assert_eq!(
                reds(&got),
                expected,
                "streamed output must match the frame sequence resample_fps used to produce for {label}"
            );
            // Reconstruct the expected frames themselves (not just their red
            // channel) from `expected`, the same way `src` builds them, so
            // this still checks full frame content -- not just the index
            // sequence -- like the original comparison against
            // `resample_fps` did.
            let want_frames: Vec<RgbaImage> = expected
                .iter()
                .map(|&r| RgbaImage::from_pixel(4, 2, Rgba([r, 0, 0, 255])))
                .collect();
            assert_eq!(got, want_frames, "frame contents must match for {label}");
        }
    }

    #[test]
    fn fps_stream_matches_resample_fps_at_the_max_frames_boundary() {
        // Mirrors `exactly_max_frames_is_accepted`: the limit is inclusive,
        // so a request that exactly fills the budget must not error.
        let c = src(10, 10.0);
        let s = FpsStream::new(c.open().unwrap(), 10.0, 10.0, 0.0, None, 10).expect("build");
        assert_eq!(drain(Box::new(s)).expect("exactly max_frames must be accepted").len(), 10);
        // ...and one fewer is the error case, as it is for resample_fps.
        let s = FpsStream::new(c.open().unwrap(), 10.0, 10.0, 0.0, None, 9).expect("build");
        assert!(drain(Box::new(s)).is_err(), "one over the budget must error");
    }

    /// `estimated_frame_count` must equal what the stream actually emits,
    /// not approximate it -- it decides whether the CLI refuses a render
    /// before printing anything about it. Every combination of frame count,
    /// source rate, target rate, start offset and duration below is swept
    /// (generated, not hand-picked, so it cannot share a blind spot with the
    /// implementation) rather than a small hand-picked set, which is what
    /// missed the closed-form `ceil()` implementation's +/-1 drift this test
    /// now pins against.
    #[test]
    fn estimated_frame_count_matches_fps_stream_over_thousands_of_configs() {
        let frame_counts = [0usize, 1, 2, 3, 5, 8, 13, 21];
        let source_fpses = [1.0f32, 4.0, 10.0, 24.0, 30.0];
        let target_fpses = [1.0f32, 5.0, 10.0, 24.0, 30.0, 60.0];
        // Includes a negative start (clamped to 0 by both `FpsStream` and
        // `estimated_frame_count`) and one far past any source's end.
        let starts = [0.0f32, 0.25, 0.5, 1.0, -1.0, 1000.0];
        // `None` (unbounded), a couple of ordinary windows, and a negative
        // duration (also clamped to 0).
        let durations = [None, Some(0.0f32), Some(0.5), Some(2.0), Some(-3.0)];
        // Generous enough that no combination below is actually over budget
        // -- this sweep is about the COUNT matching exactly, not the
        // boundary refusal, which `the_estimate_agrees_with_the_streams_budget_refusal_at_the_boundary`
        // already covers.
        const MAX_FRAMES_BUDGET: usize = 100_000;

        let mut compared = 0usize;
        let mut mismatches = Vec::new();
        for &n in &frame_counts {
            for &source_fps in &source_fpses {
                for &target_fps in &target_fpses {
                    for &start_s in &starts {
                        for &duration_s in &durations {
                            compared += 1;
                            let want = estimated_frame_count(
                                n,
                                source_fps,
                                target_fps,
                                start_s,
                                duration_s,
                                MAX_FRAMES_BUDGET,
                            );
                            let c = src(n, source_fps);
                            let s = FpsStream::new(
                                c.open().unwrap(),
                                source_fps,
                                target_fps,
                                start_s,
                                duration_s,
                                MAX_FRAMES_BUDGET,
                            )
                            .expect("build");
                            let label = format!(
                                "n={n} {source_fps}fps -> {target_fps}fps start={start_s} dur={duration_s:?}"
                            );
                            let got = drain(Box::new(s))
                                .unwrap_or_else(|e| panic!("FpsStream failed for {label}: {e}"))
                                .len();
                            if got != want {
                                mismatches.push(format!(
                                    "{label}: estimate {want} != emitted {got}"
                                ));
                            }
                        }
                    }
                }
            }
        }
        println!(
            "estimated_frame_count sweep: {compared} configurations compared, {} diverged",
            mismatches.len()
        );
        assert!(
            mismatches.is_empty(),
            "{} of {compared} configurations diverged:\n{}",
            mismatches.len(),
            mismatches.join("\n")
        );
    }

    /// The pre-flight budget check the CLI performs (`estimate > max_frames`)
    /// must agree with the stream's own refusal at the inclusive boundary --
    /// erroring one frame early would reject a render that works.
    #[test]
    fn the_estimate_agrees_with_the_streams_budget_refusal_at_the_boundary() {
        let (n, fps) = (10usize, 10.0f32);
        for (max_frames, should_error) in [(11usize, false), (10, false), (9, true)] {
            // `estimated_frame_count` takes `max_frames` as its own loop
            // bound now, so it must be recomputed per `max_frames` here --
            // it is exact when under budget (11, 10) and merely
            // "some number past the cap" when over it (9), which is all
            // `est > max_frames` below needs.
            let est = estimated_frame_count(n, fps, fps, 0.0, None, max_frames);
            let c = src(n, fps);
            let s = FpsStream::new(c.open().unwrap(), fps, fps, 0.0, None, max_frames)
                .expect("build");
            let stream_errored = drain(Box::new(s)).is_err();
            assert_eq!(
                stream_errored, should_error,
                "stream refusal at max_frames={max_frames}"
            );
            assert_eq!(
                est > max_frames,
                stream_errored,
                "the CLI's pre-flight check must match the stream at max_frames={max_frames}"
            );
        }
    }
}
