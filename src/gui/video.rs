//! The Video pane: pick an animated file or a frame sequence, tune the
//! render options, and generate wired, animated display-brick bricks.
//!
//! Follows the same shape as [`crate::gui::text::TextApp`]: option state on a
//! struct, `poll_promise` for async picking, `deliver_save` for output, and
//! every failure routed through `log::error!` rather than a panic.
use std::sync::Arc;

use crate::{
    anim::{
        bricks::{AnimOptions, DisplayBrickStyle, build_brick_world},
        cost::{self, Cost},
        pack::{BANK_FRAMES, MAX_FRAMES},
    },
    gui::{
        SharedOptions,
        util::{PickedImage, deliver_save, pick_animated_bytes, pick_images, thumb},
    },
    progress::Progress,
    video::{
        Clip,
        scale::{Filter, FitMode, estimated_frame_count, max_frames_error},
        source::{Source, decode},
        stream::{AdaptedSource, FrameSource},
    },
};
use egui::{Button, Color32, Ui};
use image::RgbaImage;
use log::{error, info};
use poll_promise::Promise;

// A video file gets its own picker (returns a PATH, never bytes -- see
// `gui::util::pick_video_path`'s doc) and its own decode path
// (`video::backend`), neither of which exists on wasm: `pick_video_path` is
// itself `#[cfg(not(target_arch = "wasm32"))]` (a browser file handle has no
// real filesystem path to hand a streaming backend), and `video::ffmpeg`
// (subprocess spawning) does not compile there at all. So every piece of
// video support in this file -- the `Input::Video` variant, the picker
// button, the backend dropdown, the download-consent modal -- is gated the
// same way, and the Video pane simply keeps its pre-Task-8 two buttons
// (animated file, frame sequence) on wasm.
#[cfg(not(target_arch = "wasm32"))]
use crate::{
    gui::util::pick_video_path,
    video::{
        backend::{self, Backend},
        ffmpeg::{DownloadConsent, ensure_ffmpeg, ffmpeg_available},
    },
};

// `MAX_FRAMES` (imported above): the largest frame count a render may carry
// across all banks, now that a clip's frames can spill across multiple wire
// arrays. `BANK_FRAMES` (also imported) is the old single-array limit -- a
// wire array index is a `u16`, `0..=65535` -- and is what the `max_frames`
// slider *defaults* to, so a fresh session doesn't silently opt into a
// million-frame render; `MAX_FRAMES` is only the slider's upper *bound*.
// Passed to `AdaptedSource::max_frames` (`FpsStream`, underneath) and never
// `usize::MAX`, which would re-enable an unbounded resampling loop and let a
// fat-fingered fps OOM. `anim::pack` is
// the single source of truth for both limits; `main.rs` imports `MAX_FRAMES`
// alone for the CLI's `--max-frames` default/cap, which has no separate
// "fresh session" default to guard.

/// Smallest legal display-pixel half-extent. `AnimOptions::pixel_extent` is a
/// pre-scale value: `DisplayBrickStyle` turns it into the brick's real
/// on-ground footprint (5x for `SmoothTile`, unscaled for `Micro`), which
/// doubles into the display-brick pitch -- so pixels always tile flush
/// regardless of value or style -- unlike the old fixed-size-brick pitch this
/// replaces, no floor is needed to avoid
/// `layout::assert_bricks_dont_overlap`'s overlap error; this only rules out
/// the degenerate zero-extent brick.
const MIN_PIXEL_EXTENT: u16 = 1;

/// What the render thread tells the UI about its progress.
#[derive(Clone, Debug)]
enum ProgressMsg {
    Begin { label: String, total: Option<u64> },
    Tick(u64),
    Finish,
    /// A throttled snapshot of the frame just processed -- see
    /// [`ChannelProgress::frame`]. Never constructed on wasm (see there for
    /// why), which is the one variant kept here purely so `poll_generate`'s
    /// match stays identical on both targets -- hence the wasm-only
    /// `allow(dead_code)`.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    Preview { width: u32, height: u32, rgba: Vec<u8> },
}

/// Reports progress from the render thread to the UI over a channel.
///
/// Send failures are ignored on purpose: a closed channel means the UI went
/// away, which must never abort a render that is otherwise fine.
struct ChannelProgress {
    tx: std::sync::mpsc::Sender<ProgressMsg>,
    /// Wall-clock time of the last `Preview` actually sent, for throttling
    /// `frame`. Native only -- see `frame`'s wasm arm for why wasm needs
    /// none of this.
    #[cfg(not(target_arch = "wasm32"))]
    last_preview: Option<std::time::Instant>,
}

impl ChannelProgress {
    fn new(tx: std::sync::mpsc::Sender<ProgressMsg>) -> Self {
        Self {
            tx,
            #[cfg(not(target_arch = "wasm32"))]
            last_preview: None,
        }
    }
}

impl Progress for ChannelProgress {
    fn begin(&mut self, label: &str, total: Option<u64>) {
        let _ = self.tx.send(ProgressMsg::Begin { label: label.to_string(), total });
    }
    fn tick(&mut self, n: u64) {
        let _ = self.tx.send(ProgressMsg::Tick(n));
    }
    fn finish(&mut self) {
        let _ = self.tx.send(ProgressMsg::Finish);
    }

    /// Throttled to roughly 10 updates/sec (every ~100ms): `build_brick_world`
    /// calls this once per frame, and copying + sending tens of thousands of
    /// frames' worth of pixels would cost far more than the render itself.
    /// The pixel copy (`rgba.to_vec()`) only happens once the throttle
    /// actually allows a send -- everywhere else this returns immediately,
    /// untouched.
    ///
    /// Native only. `Instant::now()` panics on wasm32-unknown-unknown, and a
    /// preview would never be seen mid-render there anyway: `VideoApp::generate`
    /// runs `work` synchronously inline on the UI thread on wasm (no
    /// `std::thread::spawn`), so nothing repaints until the render -- and the
    /// save it produces -- is already done. Every `Preview` sent during that
    /// window would just sit unseen in the channel until `poll_generate`'s
    /// first drain, which happens after the promise has already resolved, so
    /// this skips previews entirely on wasm rather than pay for copies no one
    /// can look at.
    #[cfg(not(target_arch = "wasm32"))]
    fn frame(&mut self, width: u32, height: u32, rgba: &[u8]) {
        const PREVIEW_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);
        let now = std::time::Instant::now();
        if self
            .last_preview
            .is_some_and(|last| now.duration_since(last) < PREVIEW_INTERVAL)
        {
            return;
        }
        self.last_preview = Some(now);
        let _ = self.tx.send(ProgressMsg::Preview { width, height, rgba: rgba.to_vec() });
    }

    #[cfg(target_arch = "wasm32")]
    fn frame(&mut self, _width: u32, _height: u32, _rgba: &[u8]) {}
}

/// The picked source, in whichever shape lets the live cost estimate and the
/// render step both work without redundant decoding.
enum Input {
    None,
    /// A single animated file, decoded once right after picking. Caching the
    /// full decode (not just the raw bytes) means every render -- including
    /// a second click after the user tweaks fps/size -- reuses the same
    /// parse instead of re-decoding the GIF/APNG/WebP from scratch, and the
    /// live cost estimate has real numbers instead of a guess.
    Animated {
        name: String,
        clip: Clip,
        preview: PickedImage,
    },
    /// A frame sequence, left as picked images rather than decoded into a
    /// `Clip` up front: `decode_sequence` bakes the *current* fps into both
    /// the source and target rate, so decoding eagerly would freeze fps at
    /// pick time and go stale the moment the user moves the slider. Decoding
    /// stays cheap here regardless -- the images are already RGBA8, so it's
    /// just a sort and a clone.
    Sequence(Vec<PickedImage>),
    /// A video file, identified only by path -- opening it is deferred all
    /// the way to `generate`, unlike `Animated`'s eager decode. Two reasons:
    /// probing a video can spawn `ffprobe` (the ffmpeg backend) or need a
    /// download-consent decision this pane cannot make on its own, and doing
    /// either at pick time (before the user has even chosen a backend or
    /// pressed Generate) would be surprising. `draw_cost` shows a plain
    /// notice instead of a real estimate for this variant -- see there.
    #[cfg(not(target_arch = "wasm32"))]
    Video { path: std::path::PathBuf, name: String },
}

/// The picked input, captured just before a render is handed to the worker.
///
/// `Animated`'s `Clip` is already decoded (and cached in `Input` for reuse
/// across renders), so it crosses the thread boundary as-is -- `Clip::frames`
/// is a plain `Vec<RgbaImage>`, so that one really is a deep copy with no
/// cheaper option available from here.
///
/// A `Sequence` crosses as `Arc`s, NOT as copied pixel buffers. Capturing it
/// out of `&self` forces *a* clone, but `PickedImage::image` is already an
/// `Arc<RgbaImage>`, so that clone is an O(frames) refcount bump rather than
/// a full per-frame pixel copy. The deep copy `decode` does require (it needs
/// owned `RgbaImage`s) then happens inside the worker, right next to the
/// `decode` call -- keeping every byte-touching step off the UI thread, and
/// matching `HeightmapApp::run_converter`, which likewise clones only
/// `PickedImage`s (i.e. `Arc`s) up front and defers its `maps_from_images`
/// decode into its worker closure.
enum GenSource {
    Clip(Clip),
    Sequence(Vec<(String, Arc<RgbaImage>)>),
    /// Just the path -- opening happens inside the worker via
    /// `backend::open_video_ensuring`, same call `main.rs` makes for the
    /// CLI. See `generate`'s `work` closure.
    #[cfg(not(target_arch = "wasm32"))]
    Video(std::path::PathBuf),
}

/// The result of [`VideoApp::check_ffmpeg_consent`]: whether resolving a
/// video source needs a download the user hasn't consented to yet.
#[cfg(not(target_arch = "wasm32"))]
enum FfmpegCheck {
    /// ffmpeg either isn't needed for this file/backend, or is already
    /// installed -- `generate` proceeds straight to spawning the worker.
    Ready,
    /// The selected backend needs ffmpeg and it is not installed --
    /// `generate` shows the consent modal instead of starting a worker.
    NeedsConsent,
    /// A real failure unrelated to ffmpeg (an explicit `Backend::Rust` guard
    /// refusal, an unreadable file, or a container neither backend can
    /// parse). Already logged via `log::error!`.
    Failed,
}

pub struct VideoApp {
    input: Input,
    pending_pick_animated: Option<Promise<Option<(String, Vec<u8>)>>>,
    pending_pick_sequence: Option<Promise<Vec<PickedImage>>>,
    /// The in-flight "pick a video path" dialog. Native only -- see
    /// `Input::Video`'s doc.
    #[cfg(not(target_arch = "wasm32"))]
    pending_pick_video: Option<Promise<Option<std::path::PathBuf>>>,
    /// Which decode backend `generate` uses for an `Input::Video` source.
    #[cfg(not(target_arch = "wasm32"))]
    backend: Backend,
    /// Set by `generate` (via `check_ffmpeg_consent`) when the selected
    /// backend needs ffmpeg and ffmpeg is not installed. `draw_ffmpeg_modal`
    /// shows a Download/Cancel dialog for it instead of `DownloadConsent::
    /// Ask`'s stdin prompt, which the GUI has no terminal for. `None`
    /// whenever the modal should not be showing.
    #[cfg(not(target_arch = "wasm32"))]
    pending_ffmpeg_consent: Option<std::path::PathBuf>,
    /// The in-flight ffmpeg download, once the consent modal's "Download"
    /// button has been clicked. A real network fetch, so it runs on a
    /// background thread rather than blocking the UI.
    #[cfg(not(target_arch = "wasm32"))]
    pending_ffmpeg_download: Option<Promise<Result<(), String>>>,
    /// The in-flight render, if any. `Some` from the moment `generate` spawns
    /// the worker until `poll_generate` observes it's ready -- while `Some`,
    /// `draw_submit` hides the generate button so a second render can't start.
    pending_generate: Option<Promise<Result<(), String>>>,
    /// The receiving half of the channel handed to the worker's
    /// `ChannelProgress`, if a render is in flight. `Some` for exactly as
    /// long as `pending_generate` is -- both are set together in `generate`
    /// and cleared together in `poll_generate`, so the bar disappears the
    /// same frame the "Generating..." label would.
    progress_rx: Option<std::sync::mpsc::Receiver<ProgressMsg>>,
    progress_label: String,
    progress_pos: u64,
    progress_total: Option<u64>,
    /// The most recent live preview, if the worker has sent one for the
    /// render currently in flight. One handle only -- each `Preview`
    /// message replaces it rather than accumulating a new texture, which
    /// would otherwise leak GPU memory every ~100ms of a render.
    preview_texture: Option<egui::TextureHandle>,

    resize: bool,
    width: u32,
    height: u32,
    fit: FitMode,
    filter: Filter,

    fps: f32,
    start: f32,
    limit_duration: bool,
    duration: f32,
    max_frames: u32,

    alpha_threshold: u8,
    pixel_extent: u16,
    brick_style: DisplayBrickStyle,
    external_clock: bool,
    glow: bool,
}

impl Default for VideoApp {
    fn default() -> Self {
        let d = AnimOptions::default();
        Self {
            input: Input::None,
            pending_pick_animated: None,
            pending_pick_sequence: None,
            #[cfg(not(target_arch = "wasm32"))]
            pending_pick_video: None,
            #[cfg(not(target_arch = "wasm32"))]
            backend: Backend::Auto,
            #[cfg(not(target_arch = "wasm32"))]
            pending_ffmpeg_consent: None,
            #[cfg(not(target_arch = "wasm32"))]
            pending_ffmpeg_download: None,
            pending_generate: None,
            progress_rx: None,
            progress_label: String::new(),
            progress_pos: 0,
            progress_total: None,
            preview_texture: None,
            resize: false,
            width: 64,
            height: 64,
            fit: FitMode::Contain,
            filter: Filter::Lanczos,
            fps: 10.0,
            start: 0.0,
            limit_duration: false,
            duration: 5.0,
            // Bound rises with spillover, but the DEFAULT stays at one array's
            // worth: a fresh session should not silently opt into a
            // multi-gigabyte render.
            max_frames: BANK_FRAMES as u32,
            alpha_threshold: d.alpha_threshold,
            pixel_extent: d.pixel_extent,
            brick_style: d.brick_style,
            external_clock: d.external_clock,
            glow: d.glow,
        }
    }
}

fn fit_name(f: FitMode) -> &'static str {
    match f {
        FitMode::Exact => "Exact",
        FitMode::Contain => "Contain",
        FitMode::Cover => "Cover",
    }
}

fn brick_style_name(s: DisplayBrickStyle) -> &'static str {
    match s {
        DisplayBrickStyle::Micro => "Micro",
        DisplayBrickStyle::SmoothTile => "Smooth Tile",
    }
}

fn filter_name(f: Filter) -> &'static str {
    match f {
        Filter::Lanczos => "Lanczos",
        Filter::Nearest => "Nearest",
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn backend_name(b: Backend) -> &'static str {
    match b {
        Backend::Auto => "Auto",
        Backend::Rust => "Pure Rust",
        Backend::Ffmpeg => "ffmpeg",
    }
}

impl VideoApp {
    /// `(width, height, frame_count, source_fps)` of the picked input at its
    /// natural resolution/frame count, before any resize/resample. `None`
    /// until something is picked -- and, for `Input::Video`, ALWAYS (see
    /// `draw_cost`, which shows a plain notice instead of trying to fill
    /// this in for that variant without opening the file).
    fn source_info(&self) -> Option<(u32, u32, usize, f32)> {
        match &self.input {
            Input::None => None,
            Input::Animated { clip, .. } => {
                Some((clip.width, clip.height, clip.frames.len(), clip.fps))
            }
            Input::Sequence(frames) => {
                let first = frames.first()?;
                // sequences have no intrinsic rate -- decode_sequence bakes
                // the current fps in as both source and target, so this
                // mirrors that rather than inventing a separate number.
                Some((first.image.width(), first.image.height(), frames.len(), self.fps))
            }
            #[cfg(not(target_arch = "wasm32"))]
            Input::Video { .. } => None,
        }
    }

    /// The render options for the current UI state.
    ///
    /// The single source of these for both `generate` and `cost_preview`, so
    /// the estimate can never cost a different graph than the one that gets
    /// built -- adding an option here reaches the readout for free.
    fn anim_opts(&self) -> AnimOptions {
        AnimOptions {
            alpha_threshold: self.alpha_threshold,
            pixel_extent: self.pixel_extent,
            brick_style: self.brick_style,
            external_clock: self.external_clock,
            glow: self.glow,
            ..AnimOptions::default()
        }
    }

    /// A live approximation of `cost::estimate` for the current options,
    /// without actually resizing/resampling every frame on every UI tick.
    ///
    /// `None` while nothing has been picked yet -- the brief's "unknown until
    /// picked" case -- rather than showing a misleading zero. `Some(Err(_))`
    /// is the OTHER thing that isn't a `Cost`: a render that would exceed
    /// `max_frames`. A hand-rolled duplicate of `estimated_frame_count` used
    /// to live here, and it CLAMPED the result to `max_frames` before costing
    /// it -- so a 500-frame source with the slider at 100 showed a plausible
    /// "100 frame(s)" readout for a render that then errored after exactly
    /// 100 frames. Reusing the shared, stream-exact `estimated_frame_count`
    /// and refusing to clamp means this can no longer show a cost for a
    /// render that cannot run; it reports the refusal instead, with the same
    /// wording `FpsStream` itself would error with.
    fn cost_preview(&self) -> Option<Result<Cost, String>> {
        let (src_w, src_h, src_frames, src_fps) = self.source_info()?;
        let (w, h) = if self.resize {
            (self.width.max(1), self.height.max(1))
        } else {
            (src_w, src_h)
        };

        let src_fps = if src_fps.is_finite() && src_fps > 0.0 { src_fps } else { 1.0 };
        let target_fps = if self.fps.is_finite() && self.fps > 0.0 { self.fps } else { 1.0 };
        let start = self.start.max(0.0);
        let duration_s = if self.limit_duration { Some(self.duration.max(0.0)) } else { None };

        let max_frames = self.max_frames as usize;
        let frames =
            estimated_frame_count(src_frames, src_fps, target_fps, start, duration_s, max_frames);
        if frames > max_frames {
            return Some(Err(max_frames_error(max_frames)));
        }

        Some(Ok(cost::estimate(w, h, frames, self.anim_opts().bank_size)))
    }

    /// Poll the in-flight pickers and apply whichever resolves.
    fn poll_picks(&mut self) {
        if let Some(promise) = self.pending_pick_animated.take() {
            match promise.try_take() {
                Ok(result) => {
                    if let Some((name, bytes)) = result {
                        match decode(Source::Animated(bytes), self.fps) {
                            Ok(clip) => {
                                info!(
                                    "Selected animated file: {name} ({}x{}, {} frame(s), {:.2}fps)",
                                    clip.width,
                                    clip.height,
                                    clip.frames.len(),
                                    clip.fps
                                );
                                let preview_img = clip
                                    .frames
                                    .first()
                                    .cloned()
                                    .unwrap_or_else(|| RgbaImage::new(1, 1));
                                self.width = clip.width;
                                self.height = clip.height;
                                self.input = Input::Animated {
                                    preview: PickedImage {
                                        name: name.clone(),
                                        image: Arc::new(preview_img),
                                    },
                                    name,
                                    clip,
                                };
                            }
                            Err(e) => error!("could not decode {name}: {e}"),
                        }
                    }
                }
                Err(promise) => self.pending_pick_animated = Some(promise),
            }
        }

        if let Some(promise) = self.pending_pick_sequence.take() {
            match promise.try_take() {
                Ok(images) => {
                    if !images.is_empty() {
                        info!(
                            "Selected frame sequence: {:?}",
                            images.iter().map(|i| &i.name).collect::<Vec<_>>()
                        );
                        if let Some(first) = images.first() {
                            self.width = first.image.width();
                            self.height = first.image.height();
                        }
                        self.input = Input::Sequence(images);
                    }
                }
                Err(promise) => self.pending_pick_sequence = Some(promise),
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(promise) = self.pending_pick_video.take() {
            match promise.try_take() {
                Ok(result) => {
                    if let Some(path) = result {
                        let name = path
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| path.display().to_string());
                        info!("Selected video: {}", path.display());
                        self.input = Input::Video { path, name };
                    }
                }
                Err(promise) => self.pending_pick_video = Some(promise),
            }
        }
    }

    /// Poll the in-flight render, if any, same `try_take`/put-back idiom as
    /// `poll_picks` above. First drains every `ProgressMsg` the worker's
    /// `ChannelProgress` has sent so far -- on native there are usually a
    /// handful per frame; on wasm `work` already ran to completion inline by
    /// the time `generate` returned, so the very first drain here picks up
    /// everything at once (`ChannelProgress::frame` never sends a `Preview`
    /// on wasm at all -- see its doc comment). A `Preview` message is turned
    /// into a texture via `ctx.load_texture`, replacing `preview_texture`
    /// rather than accumulating a new one each time. The worker already
    /// logged any failure through `log::error!` before sending it, so
    /// there's nothing left to do once the promise resolves but drop it,
    /// freeing `draw_submit` to show the generate button again --
    /// `progress_rx` and `preview_texture` are both cleared in that same
    /// branch so the bar and preview disappear on that same frame.
    fn poll_generate(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.progress_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    ProgressMsg::Begin { label, total } => {
                        self.progress_label = label;
                        self.progress_total = total;
                        self.progress_pos = 0;
                    }
                    ProgressMsg::Tick(n) => self.progress_pos = n,
                    ProgressMsg::Finish => {}
                    ProgressMsg::Preview { width, height, rgba } => {
                        // Replaces the previous handle -- never accumulates
                        // a new named texture -- so only the latest frame's
                        // upload is ever resident on the GPU.
                        let image = egui::ColorImage::from_rgba_unmultiplied(
                            [width as usize, height as usize],
                            &rgba,
                        );
                        self.preview_texture = Some(ctx.load_texture(
                            "video_preview",
                            image,
                            egui::TextureOptions::default(),
                        ));
                    }
                }
            }
        }

        if let Some(promise) = self.pending_generate.take() {
            match promise.try_take() {
                Ok(_) => {
                    self.progress_rx = None;
                    // Drop the preview alongside the rest of the progress
                    // state so a stale frame from this run can't linger into
                    // the next idle view of the pane.
                    self.preview_texture = None;
                }
                Err(promise) => self.pending_generate = Some(promise),
            }
        }
    }

    /// Poll the in-flight ffmpeg download, if the consent modal's "Download"
    /// button was clicked. On resolution `pending_ffmpeg_consent` is cleared
    /// either way -- success means ffmpeg is now installed, failure means it
    /// still is not -- and either way the modal closes and the user presses
    /// Generate again to retry. This does not resume the render
    /// automatically: doing so would mean stashing a full copy of
    /// `SharedOptions` and every render option across the download, for a
    /// one-click saving that a `log::info!` plus a second Generate click
    /// covers just as well.
    #[cfg(not(target_arch = "wasm32"))]
    fn poll_ffmpeg_download(&mut self, ctx: &egui::Context) {
        let Some(promise) = self.pending_ffmpeg_download.take() else {
            return;
        };
        match promise.try_take() {
            Ok(result) => {
                self.pending_ffmpeg_consent = None;
                match result {
                    Ok(()) => info!("ffmpeg installed -- click Generate again to continue"),
                    Err(e) => error!("{e}"),
                }
            }
            Err(promise) => {
                self.pending_ffmpeg_download = Some(promise);
                // Same reasoning as `draw_submit`'s in-flight-render branch:
                // nothing else is waking the event loop while the download
                // runs on its own thread.
                ctx.request_repaint();
            }
        }
    }

    /// Whether resolving `path` with `self.backend` needs a download the
    /// user has not consented to yet.
    ///
    /// This performs the SAME trial `open_video_ensuring` makes internally
    /// (see that function's doc): try the backend, and only need ffmpeg if
    /// that attempt failed for a reason ffmpeg could fix AND ffmpeg is not
    /// currently installed. Running it here, synchronously on the UI thread,
    /// rather than inside the worker `generate` spawns, is what lets
    /// `draw_ffmpeg_modal` ask the question before any background work
    /// starts -- a modal needs the UI thread to answer, and the worker
    /// thread has none. Any `Box<dyn FrameSource>` this opens successfully
    /// is dropped immediately: it exists only to answer the question, and
    /// `generate`'s `work` closure opens its own via the identical call.
    ///
    /// This does briefly block the UI thread on a real probe (container
    /// parse for the pure-Rust backend, an `ffprobe` spawn for ffmpeg) --
    /// the same synchronous check `main.rs` performs for the CLI, rather
    /// than a second `Promise`-based polling path only for this. See this
    /// task's report for the tradeoff.
    #[cfg(not(target_arch = "wasm32"))]
    fn check_ffmpeg_consent(&self, path: &std::path::Path) -> FfmpegCheck {
        let mut needs_consent = false;
        let mut ensure = || {
            if ffmpeg_available() {
                Ok(())
            } else {
                needs_consent = true;
                Err("ffmpeg download consent required".to_string())
            }
        };
        match backend::open_video_ensuring(
            path,
            self.backend,
            None,
            self.fit,
            self.filter,
            None,
            &mut ensure,
        ) {
            Ok(_source) => FfmpegCheck::Ready,
            Err(_) if needs_consent => FfmpegCheck::NeedsConsent,
            Err(e) => {
                error!("{e}");
                FfmpegCheck::Failed
            }
        }
    }

    /// Shown while `pending_ffmpeg_consent` is `Some`: a real modal (egui's
    /// `Modal`, whose backdrop blocks input to the rest of the pane) rather
    /// than `DownloadConsent::Ask`'s stdin prompt, which the GUI has no
    /// terminal for. "Download" spawns `ensure_ffmpeg(DownloadConsent::
    /// Always)` on a background thread -- a real network fetch, so it must
    /// not block the UI -- and shows a spinner until `poll_ffmpeg_download`
    /// observes it's done. "Cancel" clears the pending state and leaves the
    /// render un-started. Neither path ever silently downloads or silently
    /// hangs.
    #[cfg(not(target_arch = "wasm32"))]
    fn draw_ffmpeg_modal(&mut self, ctx: &egui::Context) {
        if self.pending_ffmpeg_consent.is_none() {
            return;
        }

        if self.pending_ffmpeg_download.is_some() {
            egui::Modal::new(egui::Id::new("video_ffmpeg_download_modal")).show(ctx, |ui| {
                ui.set_max_width(360.0);
                ui.heading("Downloading ffmpeg...");
                ui.add(egui::ProgressBar::new(0.0).animate(true));
            });
            return;
        }

        let mut download = false;
        let mut cancel = false;
        egui::Modal::new(egui::Id::new("video_ffmpeg_consent_modal")).show(ctx, |ui| {
            ui.set_max_width(420.0);
            ui.heading("Download ffmpeg?");
            ui.label(format!(
                "This video needs the ffmpeg decode backend, and no ffmpeg install was found \
                 on this machine. Download it now from {}?",
                ffmpeg_sidecar::download::ffmpeg_download_url()
                    .unwrap_or("the official ffmpeg build server")
            ));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Download").clicked() {
                    download = true;
                }
                if ui.button("Cancel").clicked() {
                    cancel = true;
                }
            });
        });

        if download {
            self.pending_ffmpeg_download = Some(Promise::spawn_thread("ffmpeg_download", || {
                ensure_ffmpeg(DownloadConsent::Always)
            }));
        }
        if cancel {
            info!("ffmpeg download declined; video was not converted");
            self.pending_ffmpeg_consent = None;
        }
    }

    fn draw_settings(&mut self, ui: &mut Ui, shared: &mut SharedOptions) {
        ui.heading("Settings");
        ui.label(
            "Convert an animated image or frame sequence into wired, animated display bricks.",
        );

        egui::Grid::new("video_settings_grid")
            .striped(true)
            .spacing([40.0, 4.0])
            .show(ui, |ui| {
                ui.label("Save Destination")
                    .on_hover_text("The save will be created relative to the location of the exe.");
                ui.horizontal(|ui| {
                    // the clipboard flag is meaningless on web (saves are
                    // delivered as browser downloads)
                    #[cfg(not(target_arch = "wasm32"))]
                    ui.checkbox(&mut shared.out_clipboard, "Copy to clipboard")
                        .on_hover_text("Copy the save file path to clipboard after generation");
                    ui.add(egui::TextEdit::singleline(&mut shared.out_file).hint_text("File Name"));
                });
                ui.end_row();
                let out_file_lowercase = shared.out_file.to_lowercase();
                if !out_file_lowercase.ends_with(".brz") && !out_file_lowercase.ends_with(".brdb") {
                    ui.label("Warning:");
                    ui.colored_label(Color32::RED, "Output file must end with .brz or .brdb");
                    ui.end_row();
                }

                ui.label("Resize")
                    .on_hover_text("Override the source resolution; otherwise the clip's own size is used");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.resize, "Override size");
                    ui.add_enabled_ui(self.resize, |ui| {
                        ui.label("W");
                        ui.add(egui::Slider::new(&mut self.width, 1..=2048));
                        ui.label("H");
                        ui.add(egui::Slider::new(&mut self.height, 1..=2048));
                    });
                    if !self.resize {
                        ui.label("(using source size)");
                    }
                });
                ui.end_row();

                ui.add_enabled_ui(self.resize, |ui| {
                    ui.label("Fit Mode").on_hover_text(
                        "Exact: stretch to fit. Contain: letterbox, preserving aspect. Cover: fill and crop, preserving aspect.",
                    );
                    ui.horizontal(|ui| {
                        for f in [FitMode::Exact, FitMode::Contain, FitMode::Cover] {
                            ui.radio_value(&mut self.fit, f, fit_name(f));
                        }
                    });
                });
                ui.end_row();

                ui.add_enabled_ui(self.resize, |ui| {
                    ui.label("Filter")
                        .on_hover_text("Resample filter used when resizing frames");
                    ui.horizontal(|ui| {
                        for f in [Filter::Lanczos, Filter::Nearest] {
                            ui.radio_value(&mut self.filter, f, filter_name(f));
                        }
                    });
                });
                ui.end_row();

                ui.label("FPS")
                    .on_hover_text("Output frame rate; also the fallback rate for frame sequences");
                ui.add(
                    egui::DragValue::new(&mut self.fps)
                        .speed(0.1)
                        .range(0.01..=240.0),
                );
                ui.end_row();

                ui.label("Start")
                    .on_hover_text("Seconds into the source to start from");
                ui.add(
                    egui::DragValue::new(&mut self.start)
                        .speed(0.1)
                        .suffix("s")
                        .range(0.0..=f32::INFINITY),
                );
                ui.end_row();

                ui.label("Duration")
                    .on_hover_text("Limit how much of the source (from Start) is used; unlimited runs to the end");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.limit_duration, "Limit");
                    ui.add_enabled_ui(self.limit_duration, |ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.duration)
                                .speed(0.1)
                                .suffix("s")
                                .range(0.0..=f32::INFINITY),
                        );
                    });
                });
                ui.end_row();

                ui.label("Max Frames").on_hover_text(
                    "Hard cap on emitted frames -- frames past 65535 spill into extra wire arrays",
                );
                ui.add(
                    egui::Slider::new(&mut self.max_frames, 1..=MAX_FRAMES as u32)
                        .logarithmic(true),
                );
                ui.end_row();

                ui.label("Alpha Threshold")
                    .on_hover_text("Pixels below this alpha, in every frame, are culled entirely");
                ui.add(egui::Slider::new(&mut self.alpha_threshold, 0..=255));
                ui.end_row();

                ui.label("Brick Style").on_hover_text(
                    "Micro: small cube bricks (smallest at extent 1: 2 units wide). Smooth \
                     Tile: flat tiles, always 4 units tall regardless of extent.",
                );
                ui.horizontal(|ui| {
                    for s in [DisplayBrickStyle::Micro, DisplayBrickStyle::SmoothTile] {
                        ui.radio_value(&mut self.brick_style, s, brick_style_name(s));
                    }
                });
                ui.end_row();

                ui.label("Pixel Extent").on_hover_text(
                    "Half-extent of each display pixel, in units (1 = smallest: a 2-unit-wide \
                     brick). Adjacent pixels always tile flush at twice this value, so every \
                     value here is legal -- no size/style combination can overlap.",
                );
                ui.add(egui::Slider::new(&mut self.pixel_extent, MIN_PIXEL_EXTENT..=50).text("units"));
                ui.end_row();

                ui.label("Clock").on_hover_text(
                    "External: expose Frame as a chip input instead of running a built-in timer",
                );
                ui.checkbox(&mut self.external_clock, "External clock");
                ui.end_row();

                ui.label("Material").on_hover_text(
                    "Glow at intensity 0: the screen lights itself instead of being lit by \
                     the world, so its colours stay true at night",
                );
                ui.checkbox(&mut self.glow, "Glow");
                ui.end_row();
            });
    }

    fn draw_input(&mut self, ui: &mut Ui) {
        ui.add_space(8.0);
        ui.separator();
        ui.heading("Source");
        #[cfg(not(target_arch = "wasm32"))]
        ui.label(
            "Pick a single animated file (GIF/APNG/WebP), a numbered frame sequence (PNG/JPG), \
             or a video file (mp4/mov/mkv/webm/avi/m4v).",
        );
        #[cfg(target_arch = "wasm32")]
        ui.label(
            "Pick a single animated file (GIF/APNG/WebP) or a numbered frame sequence (PNG/JPG).",
        );

        let picking = self.pending_pick_animated.is_some() || self.pending_pick_sequence.is_some();
        #[cfg(not(target_arch = "wasm32"))]
        let picking = picking || self.pending_pick_video.is_some();
        ui.horizontal(|ui| {
            if ui
                .add(Button::new("Pick animated file").fill(Color32::from_rgb(60, 60, 120)))
                .clicked()
                && !picking
            {
                self.pending_pick_animated = Some(pick_animated_bytes());
            }
            if ui
                .add(Button::new("Pick frame sequence").fill(Color32::from_rgb(60, 60, 120)))
                .clicked()
                && !picking
            {
                self.pending_pick_sequence = Some(pick_images(true));
            }
            // No video picker on wasm: `pick_video_path` needs a real
            // filesystem path to stream from, which a browser file handle
            // does not have -- see the top-of-file doc comment.
            #[cfg(not(target_arch = "wasm32"))]
            if ui
                .add(Button::new("Pick video file").fill(Color32::from_rgb(60, 60, 120)))
                .clicked()
                && !picking
            {
                self.pending_pick_video = Some(pick_video_path());
            }
        });

        #[cfg(not(target_arch = "wasm32"))]
        ui.horizontal(|ui| {
            ui.label("Video Backend").on_hover_text(
                "Auto: pure-Rust decode when it is safe for the file, ffmpeg otherwise. Pure \
                 Rust: never uses ffmpeg (errors on formats it can't decode correctly). ffmpeg: \
                 always uses ffmpeg, downloading it first if it is missing and consented to. \
                 Only applies to a picked video file.",
            );
            for b in [Backend::Auto, Backend::Rust, Backend::Ffmpeg] {
                ui.radio_value(&mut self.backend, b, backend_name(b));
            }
        });

        let mut clear_input = false;
        match &self.input {
            Input::None => {
                ui.label("No source selected.");
            }
            Input::Animated { name, clip, preview } => {
                egui::Grid::new("video_animated_grid")
                    .striped(true)
                    .spacing([8.0, 4.0])
                    .min_col_width(4.0)
                    .show(ui, |ui| {
                        if ui.button("✖").clicked() {
                            clear_input = true;
                        }
                        thumb(ui, preview);
                        ui.label(format!(
                            "{name} — {}x{}, {} frame(s), {:.2} fps",
                            clip.width,
                            clip.height,
                            clip.frames.len(),
                            clip.fps
                        ));
                    });
            }
            Input::Sequence(frames) => {
                ui.label(format!("{} frame(s) in sequence", frames.len()));
                egui::Grid::new("video_sequence_grid")
                    .striped(true)
                    .spacing([8.0, 4.0])
                    .min_col_width(4.0)
                    .show(ui, |ui| {
                        for img in frames {
                            thumb(ui, img);
                            ui.label(&img.name);
                            ui.end_row();
                        }
                    });
                if ui.button("✖ Clear sequence").clicked() {
                    clear_input = true;
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            Input::Video { path, name } => {
                ui.horizontal(|ui| {
                    if ui.button("✖").clicked() {
                        clear_input = true;
                    }
                    ui.label(format!("{name} — video source ({})", path.display()));
                });
            }
        }
        if clear_input {
            self.input = Input::None;
        }
    }

    fn draw_cost(&self, ui: &mut Ui) {
        // `source_info` (and so `cost_preview`) always returns `None` for a
        // video source -- see `Input::Video`'s doc for why this is not
        // probed eagerly -- so this is called out separately rather than
        // folding into `cost_preview`'s own `None` case below, which reads
        // as "nothing picked yet" and would be misleading here.
        #[cfg(not(target_arch = "wasm32"))]
        if matches!(self.input, Input::Video { .. }) {
            ui.label(
                "Video source selected -- cost is measured once Generate opens the file.",
            );
            return;
        }

        match self.cost_preview() {
            None => {
                ui.label("Pick a source above to see a cost estimate.");
            }
            // The render would exceed Max Frames -- surface the refusal
            // itself rather than a plausible-looking number for a render
            // that cannot run (Generate would error partway through anyway).
            Some(Err(msg)) => {
                ui.colored_label(Color32::RED, msg);
            }
            Some(Ok(cost)) => {
                let text = format!(
                    "Estimated: {} pixel(s), {} frame(s) -> {} gate(s), {} wire(s), {} brick(s), {} chunk(s), {} bank(s)",
                    cost.pixels, cost.frames, cost.gates, cost.wires, cost.bricks, cost.chunks, cost.banks
                );
                if cost.gates > 6000 {
                    ui.colored_label(
                        Color32::from_rgb(255, 140, 60),
                        format!("{text} (large build -- may be slow to paste in-game)"),
                    );
                } else {
                    ui.label(text);
                }
            }
        }
    }

    fn draw_submit(&mut self, ui: &mut Ui, shared: &mut SharedOptions) {
        // A render already in flight: no button at all, so a second click
        // can't start a second one (the real guard -- `generate` also
        // refuses to start a second worker, belt-and-suspenders).
        if self.pending_generate.is_some() {
            // egui only repaints on input by default; without this the bar
            // (and the label it replaces) would only advance when the user
            // happened to move the mouse, since nothing else is waking the
            // event loop while the worker runs on its own thread.
            ui.ctx().request_repaint();
            if let Some(tex) = &self.preview_texture {
                // Capped so a large source frame can't blow out the pane's
                // layout; `Image`'s default fit keeps the aspect ratio and
                // only shrinks (never grows) toward this bound.
                const MAX_PREVIEW_SIDE: f32 = 200.0;
                ui.add(egui::Image::new(tex).max_size(egui::vec2(MAX_PREVIEW_SIDE, MAX_PREVIEW_SIDE)));
                ui.add_space(4.0);
            }
            match self.progress_total {
                Some(total) if total > 0 => {
                    let frac = self.progress_pos as f32 / total as f32;
                    ui.add(egui::ProgressBar::new(frac).text(format!(
                        "{} {}/{}",
                        self.progress_label, self.progress_pos, total
                    )));
                }
                // Unknown total (or a degenerate `Some(0)`): an animated
                // indeterminate bar, never a fabricated fraction that would
                // reach 100% and keep going.
                _ => {
                    ui.add(
                        egui::ProgressBar::new(0.0)
                            .animate(true)
                            .text(self.progress_label.clone()),
                    );
                }
            }
            return;
        }

        let has_input = match &self.input {
            Input::None => false,
            Input::Animated { .. } => true,
            Input::Sequence(frames) => !frames.is_empty(),
            #[cfg(not(target_arch = "wasm32"))]
            Input::Video { .. } => true,
        };

        // The ffmpeg consent modal (drawn elsewhere, see `draw_ffmpeg_modal`)
        // already blocks input to the rest of the pane via its backdrop
        // while it's open; this is belt-and-suspenders so a click that
        // somehow lands here anyway can't start a second `generate` call
        // while one is already waiting on consent or mid-download.
        #[cfg(not(target_arch = "wasm32"))]
        if self.pending_ffmpeg_consent.is_some() {
            ui.label("Waiting on the ffmpeg download prompt above...");
            return;
        }

        if has_input {
            if ui
                .add(Button::new("Generate video2brick save").fill(Color32::from_rgb(50, 90, 50)))
                .clicked()
            {
                self.generate(shared);
            }
        } else {
            #[cfg(not(target_arch = "wasm32"))]
            ui.label("Pick an animated file, frame sequence, or video file to continue...");
            #[cfg(target_arch = "wasm32")]
            ui.label("Pick an animated file or frame sequence to continue...");
        }
    }

    /// On click: decode (already cached for an animated file; deferred to
    /// the worker for a sequence) -> resize -> resample -> build the brick
    /// world -> encode -> deliver.
    ///
    /// The heavy work (decode/resize/resample/`build_brick_world`/encode)
    /// runs on a background thread on native, mirroring
    /// `HeightmapApp::run_converter`: capture the inputs a render needs out
    /// of `self`/`shared` up front, move them into a `work` closure, spawn
    /// it, and hand back a `Promise` for `poll_generate` to collect.
    /// `wasm32` has no usable `std::thread::spawn`, so there `work` just
    /// runs in place, same as before -- the tab still blocks for the
    /// render's duration on the web, which is unchanged from today.
    ///
    /// The captures below run in the click handler, so they are deliberately
    /// cheap: scalars, an `AnimOptions`, two `String`s, and (for a sequence)
    /// `Arc` refcount bumps -- see [`GenSource`]. The only unavoidably
    /// expensive one is `Animated`'s `clip.clone()`, since `Clip::frames` is
    /// a plain `Vec<RgbaImage>`.
    ///
    /// `deliver_save` runs *inside* `work`, i.e. on the worker on native:
    /// it does a direct `std::fs::write` plus an optional clipboard-path
    /// copy, neither of which needs the UI thread (no dialog is opened --
    /// the destination is the already-typed `out_file` text box), and
    /// `HeightmapApp::run_converter` already does the same. On `wasm32`
    /// `work` never leaves the calling (UI) thread anyway, so the
    /// browser-download path there is untouched.
    ///
    /// Every failure is logged through `log::error!`, never panicked.
    fn generate(&mut self, shared: &SharedOptions) {
        // Belt-and-suspenders: `draw_submit` already hides the button while
        // a render is in flight, but refuse here too rather than trust the
        // UI alone.
        if self.pending_generate.is_some() {
            return error!("a render is already in progress");
        }

        // A video source needs to know, before any worker starts, whether
        // the selected backend needs ffmpeg and ffmpeg is missing --
        // `draw_ffmpeg_modal` needs the UI thread to ask, and a worker
        // thread has none. `check_ffmpeg_consent` makes exactly the same
        // trial `open_video_ensuring` would make internally; see its doc.
        #[cfg(not(target_arch = "wasm32"))]
        if self.pending_ffmpeg_consent.is_some() || self.pending_ffmpeg_download.is_some() {
            return error!("waiting on the ffmpeg download prompt");
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Input::Video { path, .. } = &self.input {
            match self.check_ffmpeg_consent(path) {
                FfmpegCheck::Ready => {}
                FfmpegCheck::NeedsConsent => {
                    self.pending_ffmpeg_consent = Some(path.clone());
                    return;
                }
                // Already logged by `check_ffmpeg_consent`.
                FfmpegCheck::Failed => return,
            }
        }

        let source = match &self.input {
            Input::None => return error!("pick an animated file, frame sequence, or video first"),
            Input::Animated { clip, .. } => GenSource::Clip(clip.clone()),
            Input::Sequence(frames) => {
                if frames.is_empty() {
                    return error!("pick a frame sequence first");
                }
                // Arc clones only -- refcount bumps, no pixel data touched.
                // The deep copy `decode` needs happens on the worker below.
                GenSource::Sequence(
                    frames
                        .iter()
                        .map(|p| (p.name.clone(), Arc::clone(&p.image)))
                        .collect(),
                )
            }
            #[cfg(not(target_arch = "wasm32"))]
            Input::Video { path, .. } => GenSource::Video(path.clone()),
        };

        // Everything below is plain data or an owned clone -- nothing here
        // borrows `self` or `shared`, so it's all free to move into `work`.
        let resize = self.resize;
        let width = self.width.max(1);
        let height = self.height.max(1);
        let fit = self.fit;
        let filter = self.filter;
        let fps = self.fps;
        #[cfg(not(target_arch = "wasm32"))]
        let backend = self.backend;
        let start = self.start.max(0.0);
        let duration = if self.limit_duration { Some(self.duration.max(0.0)) } else { None };
        // Never pass an unbounded sentinel here -- `max_frames` is already
        // clamped by the slider's range, but clamp again defensively so a
        // future UI change can't silently re-open the OOM this guards.
        let max_frames = (self.max_frames as usize).min(MAX_FRAMES);
        let anim_opts = self.anim_opts();
        let out_file = shared.out_file.clone();
        let out_clipboard = shared.out_clipboard;

        // The reporter is built here, on the UI thread, and moved into
        // `work` below; `progress_tx` is a plain `std::sync::mpsc::Sender`,
        // which is `Send` regardless of what it carries (`ProgressMsg` holds
        // only a `String`/`Option<u64>`/`u64`, all `Send` too), so this
        // satisfies the worker closure's `Send` bound for `thread::spawn` on
        // native. The receiving half is stashed on `self` for `poll_generate`
        // to drain each frame; on wasm `work` runs to completion before
        // `generate` returns, so every message is already buffered in the
        // channel by the time the first drain happens -- unbounded `mpsc`
        // sends never block on an idle receiver, so that can't deadlock, and
        // nothing is dropped, just delivered in one batch instead of many.
        let (progress_tx, progress_rx) = std::sync::mpsc::channel::<ProgressMsg>();
        self.progress_rx = Some(progress_rx);
        self.progress_label.clear();
        self.progress_pos = 0;
        self.progress_total = None;
        // A previous run's texture, if the render that owned it never made
        // it back through `poll_generate` (e.g. it errored before the first
        // `Preview`), shouldn't linger into this one.
        self.preview_texture = None;

        let work = move || -> Result<(), String> {
            // A `Box<dyn FrameSource>` unifies all three input kinds here --
            // `Clip` already implements `FrameSource` (`video::stream`), and
            // so does whatever `open_video_ensuring` returns -- so exactly
            // one `AdaptedSource` below applies resize/fps/fit/filter
            // uniformly no matter which kind produced the raw frames. This
            // mirrors `main.rs`'s CLI video path exactly (open with
            // `None`/`None` for target/fps, then layer `AdaptedSource`),
            // which is what makes the pane's width/height/fit/filter/fps
            // controls behave identically for a picked video and for a
            // decoded `Clip`.
            let raw: Box<dyn FrameSource> = match source {
                GenSource::Clip(c) => Box::new(c),
                GenSource::Sequence(named) => {
                    // `decode` wants owned `RgbaImage`s, so the pixel copy
                    // has to happen somewhere -- here, on the worker, rather
                    // than in the click handler that captured these `Arc`s.
                    let named: Vec<(String, RgbaImage)> = named
                        .into_iter()
                        .map(|(name, image)| (name, (*image).clone()))
                        .collect();
                    Box::new(decode(Source::Sequence(named), fps)?)
                }
                #[cfg(not(target_arch = "wasm32"))]
                GenSource::Video(path) => {
                    // `DownloadConsent::Never` is safe here, not silent:
                    // `check_ffmpeg_consent` already ran this exact call on
                    // the UI thread before this worker was ever spawned, so
                    // either ffmpeg was not needed for this file/backend, or
                    // it is already installed (the consent modal's Download
                    // button ran `ensure_ffmpeg(DownloadConsent::Always)`
                    // and this worker only starts once that succeeded) --
                    // `ensure_ffmpeg`'s doc: an already-installed ffmpeg
                    // short-circuits every consent variant to `Ok`, so
                    // `Never` here can't newly refuse anything `Ready`
                    // didn't already clear.
                    backend::open_video_ensuring(
                        &path,
                        backend,
                        None,
                        fit,
                        filter,
                        None,
                        &mut || ensure_ffmpeg(DownloadConsent::Never),
                    )?
                }
            };

            // Omitted resize means "use the source's own dimensions" -- skip
            // the resize entirely rather than resampling to an identical
            // size. Resize (if requested) then resample, streamed rather
            // than materialized twice: `AdaptedSource` layers `ResizeStream`
            // under `FpsStream` so frames are scaled before selection, never
            // the other way around. The pure-Rust video backend has no
            // resize/resample parameters of its own (`RustVideoSource::
            // open_path` takes none), so this is the one place that ever
            // resizes or resamples regardless of which backend produced
            // `raw`.
            let size = if resize { Some((width, height)) } else { None };
            let adapted = AdaptedSource {
                inner: raw.as_ref(),
                size,
                fit,
                filter,
                target_fps: fps,
                start_s: start,
                duration_s: duration,
                max_frames,
            };

            // The resampled frame count isn't knowable up front without
            // running the stream (see `AdaptedSource::info`), so this logs
            // only what's known before the render starts.
            let info = adapted.info();
            info!("Building frames at {}x{} ({} fps)...", info.width, info.height, info.fps);
            let mut progress = ChannelProgress::new(progress_tx);
            let world = build_brick_world(&adapted, &anim_opts, &mut progress)?;

            info!("Writing Save to {out_file}");
            let data = world.to_brz_vec().map_err(|e| format!("failed to encode brz: {e}"))?;
            deliver_save(data, &out_file, out_clipboard)?;
            info!("Done!");
            Ok(())
        };

        let (sender, promise) = Promise::new();

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            let result = work();
            if let Err(e) = &result {
                error!("{e}");
            }
            sender.send(result);
        });

        #[cfg(target_arch = "wasm32")]
        {
            // no threads on the web: run synchronously (the tab blocks for
            // the duration of the generation, as it already did)
            let result = work();
            if let Err(e) = &result {
                error!("{e}");
            }
            sender.send(result);
        }

        self.pending_generate = Some(promise);
    }

    pub fn draw(&mut self, ui: &mut Ui, shared: &mut SharedOptions) {
        self.poll_picks();
        self.poll_generate(ui.ctx());
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_ffmpeg_download(ui.ctx());
        #[cfg(not(target_arch = "wasm32"))]
        self.draw_ffmpeg_modal(ui.ctx());
        self.draw_settings(ui, shared);
        self.draw_input(ui);
        ui.add_space(8.0);
        ui.separator();
        self.draw_cost(ui);
        ui.separator();
        self.draw_submit(ui, shared);
    }
}
