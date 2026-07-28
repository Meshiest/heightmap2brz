//! Streaming frames instead of holding them.
//!
//! Every frame of a clip used to be decoded up front into a `Vec<RgbaImage>`
//! and held for the whole render. Measured over 3 reps against a worktree
//! build of the pre-streaming baseline, a 128x72 clip of 65 392 frames fell
//! from 556 MB to 459 MB (~97 MB, ~17.5%) after this change — not the 2.41 GB
//! reduction an earlier draft of this comment claimed, which was never
//! measured and is not achievable yet: [`crate::video::Clip`] is still the
//! only [`FrameSource`], and it is still fully eager (`Clip::frames` is a
//! plain `Vec<RgbaImage>` populated up front by its decoder). The larger
//! figure needs a genuinely lazy `FrameSource` — one that decodes a frame
//! from disk/pipe on demand inside `FrameStream::next` rather than holding
//! them all — which does not exist yet. `ResizeStream`/`FpsStream` (below)
//! are already lazy adapters over whatever they're given, so the win lands
//! automatically the day a lazy source is added; nothing here needs to
//! change for that.
//!
//! The trait is split in two on purpose. Text mode needs TWO passes over the
//! frames (it measures each row's worst case across the whole clip before
//! packing anything), so something has to allow a second traversal. A
//! `rewind()` would put that burden on every backend, and a subtly wrong one
//! would corrupt the band layout silently. Instead a [`FrameSource`] is a
//! cheap re-openable handle and a [`FrameStream`] is a one-shot cursor, so
//! the two-pass property falls out of the shape rather than each backend's
//! discipline.
use image::RgbaImage;

/// What a source knows about itself before decoding anything.
///
/// `width`/`height` are a CONTRACT, not just advice: every [`RgbaImage`] a
/// [`FrameStream`] opened from a matching [`FrameSource`] emits must have
/// exactly these dimensions. A consumer that sizes itself from `info()` once
/// (e.g. `anim::pack::Packer`, which allocates its per-pixel visibility
/// vector up front) is entitled to treat a differently-sized frame as an
/// error rather than a `RgbaImage` it must re-check or reshape itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    /// Total frames, when the source can say without decoding them.
    ///
    /// `None` is a real answer, not a failure: a numbered sequence knows its
    /// length by counting files and a container usually knows from metadata,
    /// but a pipe may not. This drives whether progress shows a percentage or
    /// a spinner, so it must never be guessed — a bar that reaches 100% and
    /// keeps going reads as a bug.
    pub frame_count_hint: Option<usize>,
}

/// A re-openable handle to a sequence of frames.
pub trait FrameSource {
    fn info(&self) -> SourceInfo;
    fn open(&self) -> Result<Box<dyn FrameStream + '_>, String>;
}

/// A one-shot cursor over a [`FrameSource`].
pub trait FrameStream {
    /// The next frame, or `None` once drained. A drained stream keeps
    /// returning `None`.
    ///
    /// An error is FATAL to the render. A stream that fails halfway must not
    /// be treated as a short clip — that would write a save silently missing
    /// its tail.
    ///
    /// Every frame returned here must have exactly the `width`/`height` the
    /// originating [`FrameSource::info`] reported (see [`SourceInfo`]'s own
    /// doc) — a `FrameStream` implementation that cannot guarantee this for
    /// some input should treat the mismatch as its own fatal error instead of
    /// emitting a wrongly-sized frame and leaving a downstream consumer to
    /// discover it.
    fn next(&mut self) -> Result<Option<RgbaImage>, String>;
}

/// The in-memory source: a `Clip` that already holds its frames.
pub struct ClipStream<'a> {
    frames: &'a [RgbaImage],
    at: usize,
}

impl FrameStream for ClipStream<'_> {
    fn next(&mut self) -> Result<Option<RgbaImage>, String> {
        let out = self.frames.get(self.at).cloned();
        if out.is_some() {
            self.at += 1;
        }
        Ok(out)
    }
}

impl FrameSource for crate::video::Clip {
    fn info(&self) -> SourceInfo {
        SourceInfo {
            width: self.width,
            height: self.height,
            fps: self.fps,
            frame_count_hint: Some(self.frames.len()),
        }
    }

    fn open(&self) -> Result<Box<dyn FrameStream + '_>, String> {
        Ok(Box::new(ClipStream { frames: &self.frames, at: 0 }))
    }
}

use super::scale::{FitMode, Filter, FpsStream, ResizeStream, estimated_frame_count};

/// A source with resize and fps conversion layered over it.
pub struct AdaptedSource<'a> {
    pub inner: &'a dyn FrameSource,
    pub size: Option<(u32, u32)>,
    pub fit: FitMode,
    pub filter: Filter,
    pub target_fps: f32,
    pub start_s: f32,
    pub duration_s: Option<f32>,
    pub max_frames: usize,
}

impl FrameSource for AdaptedSource<'_> {
    fn info(&self) -> SourceInfo {
        let base = self.inner.info();
        let (width, height) = self.size.unwrap_or((base.width, base.height));
        SourceInfo {
            width,
            height,
            fps: self.target_fps,
            // A previous version only reported a hint when `target_fps ==
            // base.fps`, which reads as "the rate hasn't changed" but for a
            // frame sequence (`decode`, see `video::source`) bakes `--fps`
            // in as BOTH rates, so that condition was always true -- and the
            // check still ignored the `start_s`/`duration_s` window
            // entirely, so a `--duration 1.0` render reported the source's
            // full, unwindowed length as its hint. `estimated_frame_count`
            // already folds resampling AND the window together and is pinned
            // to `FpsStream`'s real output by its own test sweep, so this is
            // an honest count whenever the base source knows its own length
            // -- not a guess -- matching the design's requirement that a
            // hint be knowable up front rather than approximated. `None`
            // stays `None`: if the base source cannot say its own frame
            // count, nothing here can improve on that.
            frame_count_hint: base.frame_count_hint.map(|n| {
                estimated_frame_count(
                    n,
                    base.fps,
                    self.target_fps,
                    self.start_s,
                    self.duration_s,
                    self.max_frames,
                )
            }),
        }
    }

    fn open(&self) -> Result<Box<dyn FrameStream + '_>, String> {
        let base = self.inner.info();
        let mut s = self.inner.open()?;
        if let Some((w, h)) = self.size {
            s = Box::new(ResizeStream::new(s, w, h, self.fit, self.filter));
        }
        Ok(Box::new(FpsStream::new(
            s,
            base.fps,
            self.target_fps,
            self.start_s,
            self.duration_s,
            self.max_frames,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    fn clip(n: usize) -> crate::video::Clip {
        let frames = (0..n)
            .map(|i| RgbaImage::from_pixel(2, 2, Rgba([i as u8, 0, 0, 255])))
            .collect();
        crate::video::Clip { width: 2, height: 2, fps: 10.0, frames }
    }

    #[test]
    fn a_clip_reports_its_own_shape() {
        let info = clip(5).info();
        assert_eq!((info.width, info.height), (2, 2));
        assert_eq!(info.fps, 10.0);
        assert_eq!(info.frame_count_hint, Some(5), "an in-memory clip always knows its length");
    }

    /// The property text mode's two-pass banding depends on: opening the same
    /// source twice must yield identical frames. A source that consumed
    /// itself on first read would make the scan and the pack disagree, and
    /// the symptom would be a wrong band layout, not an error.
    #[test]
    fn two_streams_over_one_source_agree() {
        let c = clip(4);
        let drain = |s: &dyn FrameSource| -> Vec<RgbaImage> {
            let mut st = s.open().expect("open");
            let mut out = Vec::new();
            while let Some(f) = st.next().expect("next") {
                out.push(f);
            }
            out
        };
        let a = drain(&c);
        let b = drain(&c);
        assert_eq!(a.len(), 4);
        assert_eq!(a, b, "two streams over the same source must agree frame for frame");
    }

    #[test]
    fn a_stream_ends_with_none_and_stays_ended() {
        let c = clip(1);
        let mut s = c.open().expect("open");
        assert!(s.next().expect("first").is_some());
        assert!(s.next().expect("end").is_none());
        assert!(s.next().expect("still end").is_none(), "a drained stream stays drained");
    }

    #[test]
    fn an_empty_clip_yields_no_frames() {
        let c = clip(0);
        assert_eq!(c.info().frame_count_hint, Some(0));
        assert!(c.open().expect("open").next().expect("next").is_none());
    }

    #[test]
    fn an_adapted_source_resizes_and_resamples_together() {
        let base = clip(10); // 2x2, 10fps, 10 frames
        let adapted = AdaptedSource {
            inner: &base,
            size: Some((6, 4)),
            fit: crate::video::scale::FitMode::Exact,
            filter: crate::video::scale::Filter::Nearest,
            target_fps: 5.0,
            start_s: 0.0,
            duration_s: None,
            max_frames: 1000,
        };
        let info = adapted.info();
        assert_eq!((info.width, info.height), (6, 4), "info must report the ADAPTED size");
        assert_eq!(info.fps, 5.0, "info must report the target rate");
        assert_eq!(
            info.frame_count_hint,
            Some(5),
            "the base source knows its own length, so the hint must fold in the resample \
             (10 frames @ 10fps -> 5fps is 5) rather than reporting None or the unwindowed \
             source count"
        );

        let mut s = adapted.open().expect("open");
        let mut n = 0;
        while let Some(f) = s.next().expect("next") {
            assert_eq!(f.dimensions(), (6, 4), "every emitted frame must be resized");
            n += 1;
        }
        assert_eq!(n, 5, "10 frames at 10fps resampled to 5fps");
        assert_eq!(info.frame_count_hint, Some(n), "the hint must match what was actually emitted");
    }

    /// The bug this fixes, reproduced directly: a `--duration` window used to
    /// be ignored entirely because the old check only looked at whether
    /// `target_fps == base.fps`, which is true for every frame sequence
    /// (`decode` bakes `--fps` in as both rates) regardless of any window.
    #[test]
    fn an_adapted_source_hint_honors_the_duration_window_even_at_the_same_rate() {
        let base = clip(100); // 2x2, 10fps, 100 frames
        let adapted = AdaptedSource {
            inner: &base,
            size: None,
            fit: crate::video::scale::FitMode::Exact,
            filter: crate::video::scale::Filter::Nearest,
            target_fps: 10.0, // same rate as base -- the old bug's trigger
            start_s: 0.0,
            duration_s: Some(1.0), // 10 frames' worth at 10fps
            max_frames: 1000,
        };
        assert_eq!(
            adapted.info().frame_count_hint,
            Some(10),
            "a duration window must shrink the hint even when the rate is unchanged"
        );

        let start_adapted = AdaptedSource { start_s: 5.0, duration_s: None, ..adapted };
        assert_eq!(
            start_adapted.info().frame_count_hint,
            Some(50),
            "a start offset must shrink the hint the same way"
        );
    }
}
