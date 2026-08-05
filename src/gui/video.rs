//! The Video pane: pick an animated file or a frame sequence, tune the
//! render options, and generate wired, animated display-brick bricks.
//!
//! Follows the same shape as [`crate::gui::text::TextApp`]: option state on a
//! struct, `poll_promise` for async picking, `deliver_save` for output, and
//! every failure routed through `log::error!` rather than a panic.
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::{
    anim::{
        AnimEncoding, AnimMode,
        bricks::{AnimOptions, DisplayBrickStyle},
        cost::Cost,
        pack::{BANK_FRAMES, MAX_FRAMES},
        text_layout,
    },
    gui::{
        SharedOptions,
        util::{
            self, PickedImage, RenderMsg, bound_pane_width, deliver_world_unless_cancelled,
            draw_progress_bar, note, out_file_warning_row, pick_animated_bytes, pick_images,
            pick_subtitle_bytes, refuse_bad_out_file, save_destination_row, section, thumb,
        },
    },
    progress::Progress,
    subs::{self, Subtitles},
    text::{FontPreset, TextOptions},
    video::{
        Clip,
        scale::{Filter, FitMode, estimated_frame_count, max_frames_error},
        source::{Source, decode},
        stream::{AdaptedSource, FrameSource},
    },
};
use crate::gui::theme::{icons, widgets};
use egui::{Color32, Ui};
use image::RgbaImage;
use log::{error, info};
use poll_promise::Promise;

// A video file is decoded very differently per target. NATIVE picks a PATH and
// streams it, and can reach the ffmpeg backend (subprocess) as well as the
// pure-Rust builtin one -- so the backend dropdown and the download-consent
// modal are native-only. The WEB build uploads the file's BYTES and decodes
// them with the builtin (H.264/CABAC-only) backend alone, since ffmpeg cannot
// spawn a subprocess in a browser. Both feed one `Input::Video`/`GenSource::
// Video` (its fields differ by target) and one `backend` entry point.
#[cfg(not(target_arch = "wasm32"))]
use crate::{
    gui::util::{FfmpegModal, draw_cancel_button, pick_video_path},
    video::{
        backend::{self, Backend},
        ffmpeg::{DownloadConsent, ensure_ffmpeg, ffmpeg_available},
    },
};
#[cfg(target_arch = "wasm32")]
use crate::{gui::util::pick_video_bytes, video::backend};

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

/// A throttled snapshot of the frame just processed -- see
/// [`ChannelProgress::frame`]. Never constructed on wasm (see there for why),
/// which is the one payload kept here purely so `poll_generate`'s match stays
/// identical on both targets -- hence the wasm-only `allow(dead_code)`.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
#[derive(Clone, Debug)]
struct Preview {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

/// What the render thread tells the UI about its progress: the shared
/// Begin/Tick/Finish core (see [`util::RenderMsg`]) plus this pane's one
/// extra payload, a frame [`Preview`] -- the audio pane, which has no
/// picture to show, carries no `Extra` at all.
type ProgressMsg = RenderMsg<Preview>;

/// Reports progress from the render thread to the UI over a channel, wrapping
/// the shared [`util::ChannelProgress`] to add the one thing only this pane
/// needs on top of its Begin/Tick/Finish/is_cancelled core: a throttled frame
/// preview.
struct ChannelProgress {
    inner: util::ChannelProgress<Preview>,
    /// Wall-clock time of the last `Preview` actually sent, for throttling
    /// `frame`. Native only -- see `frame`'s wasm arm for why wasm needs
    /// none of this.
    #[cfg(not(target_arch = "wasm32"))]
    last_preview: Option<std::time::Instant>,
}

impl ChannelProgress {
    fn new(tx: std::sync::mpsc::Sender<ProgressMsg>, cancel: Arc<AtomicBool>) -> Self {
        Self {
            inner: util::ChannelProgress::new(tx, cancel),
            #[cfg(not(target_arch = "wasm32"))]
            last_preview: None,
        }
    }
}

impl Progress for ChannelProgress {
    fn begin(&mut self, label: &str, total: Option<u64>) {
        self.inner.begin(label, total);
    }
    fn tick(&mut self, n: u64) {
        self.inner.tick(n);
    }
    fn finish(&mut self) {
        self.inner.finish();
    }

    /// Throttled to ~10 updates/sec; the pixel copy only happens when the
    /// throttle allows a send. Native only -- `Instant::now()` panics on
    /// wasm, and `generate` runs synchronously there anyway so no preview
    /// would be seen mid-render.
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
        self.inner.send_extra(Preview { width, height, rgba: rgba.to_vec() });
    }

    #[cfg(target_arch = "wasm32")]
    fn frame(&mut self, _width: u32, _height: u32, _rgba: &[u8]) {}

    /// Backed by the same `Arc<AtomicBool>` the UI thread's Cancel button
    /// sets. `Relaxed` is enough: a single flag with no other state to stay
    /// ordered with.
    fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

/// The picked source, in whichever shape lets the live cost estimate and the
/// render step both work without redundant decoding.
enum Input {
    None,
    /// Decoded once right after picking, so a repeated render or the live
    /// cost estimate reuses the same parse instead of re-decoding.
    Animated {
        name: String,
        clip: Clip,
        preview: PickedImage,
    },
    /// Left undecoded: `decode_sequence` bakes the *current* fps in, so
    /// decoding eagerly would go stale the moment the user moves the slider.
    Sequence(Vec<PickedImage>),
    /// Held undecoded (a path on native, the uploaded bytes on the web);
    /// opening is deferred to `generate`, unlike `Animated`'s eager decode --
    /// probing can spawn `ffprobe` or need a download-consent decision this pane
    /// can't make at pick time. `draw_cost` shows a plain notice rather than a
    /// real estimate here.
    Video {
        #[cfg(not(target_arch = "wasm32"))]
        path: std::path::PathBuf,
        #[cfg(target_arch = "wasm32")]
        bytes: std::sync::Arc<[u8]>,
        name: String,
    },
}

/// The picked input, captured just before a render is handed to the worker.
///
/// `Animated`'s `Clip` is already decoded, so it crosses as a real deep copy
/// (`Clip::frames` is a plain `Vec<RgbaImage>`, no cheaper option here). A
/// `Sequence` crosses as `Arc`s instead of copied pixel buffers --
/// `PickedImage::image` is already `Arc<RgbaImage>`, so capturing it is an
/// O(frames) refcount bump, and the real per-frame decode happens inside the
/// worker, off the UI thread.
enum GenSource {
    Clip(Clip),
    Sequence(Vec<(String, Arc<RgbaImage>)>),
    /// Opening happens inside the worker: `backend::open_video_ensuring` from a
    /// path on native (the same call `main.rs` makes for the CLI), or
    /// `backend::open_video_bytes` from the uploaded blob on the web.
    Video {
        #[cfg(not(target_arch = "wasm32"))]
        path: std::path::PathBuf,
        #[cfg(target_arch = "wasm32")]
        bytes: std::sync::Arc<[u8]>,
        #[cfg(target_arch = "wasm32")]
        name: String,
    },
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
    /// A real failure unrelated to ffmpeg (an explicit `Backend::Builtin` guard
    /// refusal, an unreadable file, or a container neither backend can
    /// parse). Already logged via `log::error!`.
    Failed,
}

pub struct VideoApp {
    input: Input,
    pending_pick_animated: Option<Promise<Option<(String, Vec<u8>)>>>,
    pending_pick_sequence: Option<Promise<Vec<PickedImage>>>,
    /// The in-flight video picker: a path dialog on native, a byte upload on
    /// the web.
    #[cfg(not(target_arch = "wasm32"))]
    pending_pick_video: Option<Promise<Option<std::path::PathBuf>>>,
    #[cfg(target_arch = "wasm32")]
    pending_pick_video: Option<Promise<Option<(String, Vec<u8>)>>>,
    /// Which decode backend `generate` uses for an `Input::Video` source.
    #[cfg(not(target_arch = "wasm32"))]
    backend: Backend,
    /// Shared ffmpeg download-consent + progress modal (see [`FfmpegModal`]),
    /// opened by `generate` when the selected backend needs an ffmpeg that
    /// isn't installed -- a dialog instead of `DownloadConsent::Ask`'s stdin
    /// prompt, which the GUI has no terminal for.
    #[cfg(not(target_arch = "wasm32"))]
    modal: FfmpegModal,
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
    /// Most recent live preview, if any. One handle only: each `Preview`
    /// message replaces it rather than accumulating textures, which would
    /// otherwise leak GPU memory every ~100ms of a render.
    preview_texture: Option<egui::TextureHandle>,
    /// The in-flight render's cancel flag: set by the Cancel button, read by
    /// the worker's `ChannelProgress::is_cancelled`. A fresh flag backs every
    /// new render, so returning to idle can't leave a stale `true` behind.
    ///
    /// Native only: on wasm `generate` runs synchronously to completion, so
    /// there is nothing a Cancel button could interrupt -- `draw_submit`
    /// hides it entirely.
    #[cfg(not(target_arch = "wasm32"))]
    cancel_flag: Option<Arc<AtomicBool>>,

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
    /// Repeat the clip forever, or stop on its last frame. See
    /// [`AnimOptions::loop_playback`] -- it becomes the clock's `Timer.Limit`
    /// and changes no gate, wire or brick count either way. Inert while
    /// [`Self::external_clock`] is set, which builds no timer.
    loop_playback: bool,
    /// Pre-generate the physical Pause/Restart/Resume buttons wired into the
    /// clock ([`AnimOptions::control_buttons`]). On by default; inert while
    /// [`Self::external_clock`] is set, which exposes no control pins.
    control_buttons: bool,
    glow: bool,

    /// Which medium the render uses -- one display brick per pixel, or a
    /// stack of animated `TextDisplay` bricks (see [`AnimMode`]). Brick-only
    /// fields above (`pixel_extent`, `brick_style`, `glow`) are simply
    /// ignored by [`AnimMode::Text`]'s renderer; text-only fields below are
    /// ignored by [`AnimMode::Brick`]'s.
    mode: AnimMode,
    /// TEXT MODE ONLY. Font preset for the text render -- same presets and
    /// the same reseed-on-change behaviour as [`crate::gui::text::TextApp`]'s
    /// "Font" control, which these fields and [`Self::load_text_preset`]
    /// mirror rather than duplicate from scratch.
    text_preset: FontPreset,
    /// TEXT MODE ONLY. First character is used, same as the Image2Text pane.
    text_fill_char: String,
    /// TEXT MODE ONLY. First character is used, same as the Image2Text pane.
    text_empty_char: String,
    /// TEXT MODE ONLY. Also drives the band layout
    /// ([`text_layout::plan_bands`]) -- see [`Self::anim_opts`].
    text_char_repeat: usize,
    /// TEXT MODE ONLY. The component `LineHeight` (font size); the Image2Text
    /// pane calls this control "Font Size".
    text_line_height: f32,
    /// TEXT MODE ONLY. Median-cut palette size passed through to
    /// [`AnimOptions::colors`]; `0` means full 24-bit colour. Shown only when
    /// [`Self::shows_colours_control`] is true.
    colors: usize,

    /// The picked subtitle track and its file name, or `None`. Parsed once
    /// at pick time, not per frame -- the cost readout redraws every frame
    /// and would waste re-parsing thousands of cues 60 times a second. The
    /// `Arc` makes sharing it with the render thread free.
    subtitles: Option<(String, Arc<Subtitles>)>,
    /// The in-flight subtitle picker. Bytes, not a path -- see
    /// [`crate::gui::util::pick_subtitle_bytes`], which is why this works on
    /// the web as well.
    pending_pick_subtitles: Option<Promise<Option<(String, Vec<u8>)>>>,
    /// How much bigger a subtitle line is than one row of the screen
    /// ([`AnimOptions::subtitle_scale`]). Inert while [`Self::subtitles`] is
    /// `None`, and the control is hidden then for the same reason the Colours
    /// slider is hidden under brick mode.
    subtitle_scale: f32,
    /// How many world units to lift the subtitle anchor toward the top of
    /// the picture ([`AnimOptions::subtitle_lift`]). Same gating as
    /// [`Self::subtitle_scale`] -- inert, and hidden, while
    /// [`Self::subtitles`] is `None`.
    subtitle_lift: f32,
}

impl Default for VideoApp {
    fn default() -> Self {
        let d = AnimOptions::default();
        let text_preset = FontPreset::MonaspaceArgon;
        // `d.text` is already `FontPreset::MonaspaceArgon.options(1.0)` (see
        // `AnimOptions::default`), so this reads the same values back rather
        // than recomputing them from `text_preset` a second time.
        let text_default = &d.text;
        Self {
            input: Input::None,
            pending_pick_animated: None,
            pending_pick_sequence: None,
            pending_pick_video: None,
            #[cfg(not(target_arch = "wasm32"))]
            backend: Backend::Auto,
            #[cfg(not(target_arch = "wasm32"))]
            modal: FfmpegModal::default(),
            pending_generate: None,
            progress_rx: None,
            progress_label: String::new(),
            progress_pos: 0,
            progress_total: None,
            preview_texture: None,
            #[cfg(not(target_arch = "wasm32"))]
            cancel_flag: None,
            // Default to a 64x64 Contain render rather than the source
            // resolution: a full-size video is an enormous brick build, and
            // 64x64 is a sensible starting point the user can raise.
            resize: true,
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
            loop_playback: d.loop_playback,
            control_buttons: d.control_buttons,
            glow: d.glow,
            // Brick mode, unchanged default behaviour -- this GUI never
            // exposed an `AnimEncoding` choice even before this field
            // existed, so the toggle this task adds is Bricks/Text only.
            mode: AnimMode::Brick(AnimEncoding::Hex),
            text_preset,
            text_fill_char: text_default.fill_char.to_string(),
            text_empty_char: text_default.empty_char.to_string(),
            text_char_repeat: text_default.char_repeat,
            text_line_height: text_default.line_height,
            colors: d.colors,
            subtitles: None,
            pending_pick_subtitles: None,
            subtitle_scale: d.subtitle_scale,
            subtitle_lift: d.subtitle_lift,
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
        Backend::Builtin => "Builtin",
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
            Input::Video { .. } => None,
        }
    }

    /// The width/height the render would actually produce: the Resize
    /// override when it's on, otherwise the picked source's own size. `None`
    /// under the same condition `source_info` is -- nothing picked, or a
    /// video source (see that method's doc). Shared by `live_cost` and
    /// `draw_text_char_bound` so the two can't quietly compute this
    /// differently from each other.
    fn output_dims(&self) -> Option<(u32, u32)> {
        let (src_w, src_h, ..) = self.source_info()?;
        Some(if self.resize {
            (self.width.max(1), self.height.max(1))
        } else {
            (src_w, src_h)
        })
    }

    /// The render options for the current UI state.
    ///
    /// The single source of these for both `generate` and `live_cost`, so
    /// the estimate can never cost a different graph than the one that gets
    /// built -- adding an option here reaches the readout for free. This is
    /// exactly the property `AnimMode::estimate` (called from `live_cost`)
    /// exists to protect: it takes the whole `AnimOptions` rather than loose
    /// numbers, because a `cost::estimate_text` that once took `char_repeat`
    /// as a hard-coded constant let a single-glyph `--font` band the image 36
    /// ways while the readout still claimed 54 bands' worth of gates. Every
    /// text field below (`text_char_repeat`, `text_fill_char`, ...) reaches
    /// both `live_cost` and `generate` through this one struct, so the two
    /// can never read a different value for the same setting.
    fn anim_opts(&self) -> AnimOptions {
        let text = TextOptions {
            fill_char: self.text_fill_char.chars().next().unwrap_or('█'),
            empty_char: self.text_empty_char.chars().next().unwrap_or(' '),
            char_repeat: self.text_char_repeat.max(1),
            // Same slider that culls brick-mode pixels, not a second control
            // (see `anim_options` in main.rs for why both fields must match).
            alpha_threshold: self.alpha_threshold,
            line_height: self.text_line_height,
            ..self.text_preset.options(1.0)
        };
        AnimOptions {
            alpha_threshold: self.alpha_threshold,
            pixel_extent: self.pixel_extent,
            brick_style: self.brick_style,
            external_clock: self.external_clock,
            loop_playback: self.loop_playback,
            control_buttons: self.control_buttons,
            glow: self.glow,
            colors: self.colors,
            text,
            // An `Arc` clone -- a refcount bump, not a copy of the cues --
            // which is what lets the readout below and the render thread hold
            // the same parsed track.
            subtitles: self.subtitles.as_ref().map(|(_, t)| Arc::clone(t)),
            subtitle_scale: self.subtitle_scale,
            subtitle_lift: self.subtitle_lift,
            // Clamped exactly as `generate` clamps it for the `AdaptedSource`,
            // and read off the same field, so the subtitle timing can never
            // disagree with where the render actually starts. A subtitle file
            // is in SOURCE time -- see `AnimOptions::source_start_s`.
            source_start_s: self.start.max(0.0) as f64,
            ..AnimOptions::default()
        }
    }

    /// Whether the Colours control is meaningful for the currently selected
    /// mode. Text-mode only: brick mode has no palette step at all and
    /// `AnimOptions::colors` is simply ignored there, so showing the slider
    /// under `AnimMode::Brick` would offer a control that does nothing.
    fn shows_colours_control(&self) -> bool {
        matches!(self.mode, AnimMode::Text)
    }

    /// A live approximation of `AnimMode::estimate` for the current options,
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
    ///
    /// Routed through `self.mode.estimate(..., &self.anim_opts())` rather
    /// than calling `cost::estimate`/`cost::estimate_text` directly -- the
    /// same reason `generate`'s worker calls `self.mode.build(...)` on the
    /// same `anim_opts()` value. Both this readout and the actual render
    /// dispatch off the identical `(mode, opts)` pair, so they can never
    /// describe two different graphs.
    fn live_cost(&self) -> Option<Result<Cost, String>> {
        let (_, _, src_frames, src_fps) = self.source_info()?;
        let (w, h) = self.output_dims()?;

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

        // `estimate` is fallible for text mode, and its error is the one
        // `build` would fail with -- so an unlayoutable geometry now shows up
        // in this readout as the refusal it is, in the same place the cost
        // would have gone, rather than as the plausible "5 gate(s), 1
        // brick(s)" a swallowed layout error used to read as.
        Some(self.mode.estimate(w, h, frames, &self.anim_opts()))
    }

    /// Reseed the text-mode glyph/geometry fields from `self.text_preset`,
    /// mirroring `crate::gui::text::TextApp::load_preset` (the pixel-size
    /// argument is fixed at `1.0` here -- unlike the Image2Text pane, this
    /// pane exposes no independent "world units per pixel row" control, so
    /// there is nothing else to reseed against).
    fn load_text_preset(&mut self) {
        let d = self.text_preset.options(1.0);
        self.text_fill_char = d.fill_char.to_string();
        self.text_empty_char = d.empty_char.to_string();
        self.text_char_repeat = d.char_repeat;
        self.text_line_height = d.line_height;
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

        if let Some(promise) = self.pending_pick_subtitles.take() {
            match promise.try_take() {
                Ok(result) => {
                    if let Some((name, bytes)) = result {
                        // Lossy UTF-8 for the same reason the CLI's
                        // `load_subtitles` uses it: subtitle files in the wild
                        // are often Latin-1, and one mangled accent is a far
                        // better trade than refusing the file. The timings are
                        // ASCII in both formats either way.
                        let text = String::from_utf8_lossy(&bytes);
                        // The SAME dispatcher the CLI uses, so one file can
                        // never be read as two different formats by the two
                        // front ends.
                        let ext = std::path::Path::new(&name)
                            .extension()
                            .and_then(|e| e.to_str())
                            .map(|e| e.to_string());
                        match subs::parse_auto(&text, ext.as_deref()) {
                            Ok(track) => {
                                if track.is_empty() {
                                    // Never swallowed: an empty track renders
                                    // as a video with no dialogue, which looks
                                    // exactly like a correct render of a
                                    // silent scene.
                                    error!("{name} parsed to 0 subtitle cues -- nothing will be shown");
                                }
                                info!("Selected subtitles: {name} ({} cue(s))", track.len());
                                self.subtitles = Some((name, Arc::new(track)));
                            }
                            Err(e) => error!("could not read {name}: {e}"),
                        }
                    }
                }
                Err(promise) => self.pending_pick_subtitles = Some(promise),
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

        if let Some(promise) = self.pending_pick_video.take() {
            match promise.try_take() {
                Ok(result) => {
                    if let Some(picked) = result {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            let path = picked;
                            let name = path
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_else(|| path.display().to_string());
                            info!("Selected video: {}", path.display());
                            self.input = Input::Video { path, name };
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            let (name, bytes) = picked;
                            info!("Selected video: {name}");
                            self.input = Input::Video { bytes: bytes.into(), name };
                        }
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
                // `apply_core` is the shared Begin/Tick/Finish bookkeeping
                // (see `util::RenderMsg`, including the "grow an
                // under-estimate rather than overflow the bar" rule); a
                // `Preview` payload comes back out here, since what it means
                // is pane-specific.
                let Some(Preview { width, height, rgba }) = msg.apply_core(
                    &mut self.progress_label,
                    &mut self.progress_pos,
                    &mut self.progress_total,
                ) else {
                    continue;
                };
                // Replaces the previous handle -- never accumulates a new
                // named texture -- so only the latest frame's upload is ever
                // resident on the GPU.
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

        if let Some(promise) = self.pending_generate.take() {
            match promise.try_take() {
                Ok(_) => {
                    self.progress_rx = None;
                    // Drop the preview alongside the rest of the progress
                    // state so a stale frame from this run can't linger into
                    // the next idle view of the pane.
                    self.preview_texture = None;
                    // Returning to idle clears the flag -- `generate` hands
                    // the NEXT render a brand new `Arc<AtomicBool>` regardless
                    // (see that field's doc), but clearing this one too means
                    // `draw_submit`'s Cancel button can't linger visible (or,
                    // worse, pre-armed) into the idle view between renders.
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        self.cancel_flag = None;
                    }
                }
                Err(promise) => self.pending_generate = Some(promise),
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
    /// rather than inside the worker `generate` spawns, is what lets the modal
    /// ask the question before any background work starts -- a modal needs the
    /// UI thread to answer, and the worker thread has none. Any
    /// `Box<dyn FrameSource>` this opens successfully
    /// is dropped immediately: it exists only to answer the question, and
    /// `generate`'s `work` closure opens its own via the identical call.
    ///
    /// This does briefly block the UI thread on a real probe (container
    /// parse for the builtin backend, an `ffprobe` spawn for ffmpeg) --
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

    /// The always-visible half of the pane: destination, render mode, screen
    /// size and frame rate.
    ///
    /// **Nothing on the path from "I picked a clip" to "I have a save" is
    /// allowed behind a collapsed section.** Size and FPS are on this side of
    /// the line with the mode because a render is meaningless without them --
    /// they are what the cost readout below is mostly made of. Everything that
    /// only tunes the picture lives in [`Self::draw_advanced_sections`].
    fn draw_settings(&mut self, ui: &mut Ui, shared: &mut SharedOptions) {
        ui.label(
            "Convert an animated image or frame sequence into wired, animated display bricks.",
        );

        widgets::settings_table(ui, |ui, t| {
            save_destination_row(t, ui, shared);
            out_file_warning_row(t, ui, &shared.out_file);

            t.row_hover(
                ui,
                "Render Mode",
                Some(
                    "Bricks: one display brick per pixel -- dense, and the most expensive at \
                     large screen sizes. Text: a stack of animated TextDisplay bricks, ~2 gates \
                     per BAND of image rows instead of per pixel -- two-plus orders of magnitude \
                     cheaper on gates at typical sizes. \"Brick Style\"/\"Pixel Extent\"/\"Glow\" \
                     in the sections below only apply to Bricks; the Text Options/Colours \
                     controls there only apply to Text.",
                ),
                |ui| {
                    ui.horizontal(|ui| {
                        widgets::radio(ui,&mut self.mode, AnimMode::Brick(AnimEncoding::Hex), "Bricks");
                        widgets::radio(ui,&mut self.mode, AnimMode::Text, "Text");
                    });
                },
            );

            t.row_hover(
                ui,
                "Resize",
                Some("Override the source resolution; otherwise the clip's own size is used"),
                |ui| {
                    widgets::toggle(ui,&mut self.resize, "Override size");
                },
            );

            // The two sliders get a ROW OF THEIR OWN rather than sharing one
            // with the checkbox: a checkbox plus two labelled sliders is wider
            // than the pane at the default 600px window, and a cell that
            // overflows widens the whole grid, which is what was dragging a
            // horizontal scrollbar across the pane.
            t.row(ui, "Size", |ui| {
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(self.resize, |ui| {
                        ui.label("W");
                        widgets::slider(ui, egui::Slider::new(&mut self.width, 1..=2048));
                        ui.label("H");
                        widgets::slider(ui, egui::Slider::new(&mut self.height, 1..=2048));
                    });
                });
            });
            // Also its own row: beside the sliders there is less width left
            // than this sentence's longest word, and that is the one case
            // where egui breaks text mid-word instead of at a space.
            if !self.resize {
                t.row(ui, "", |ui| {
                    note(ui, "(using source size)");
                });
            }

            t.row_hover(
                ui,
                "FPS",
                Some("Output frame rate; also the fallback rate for frame sequences"),
                |ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.fps)
                            .speed(0.1)
                            .range(0.01..=240.0),
                    );
                },
            );

            // Playback lives on the always-visible critical path, not in an
            // advanced section: whether a clip loops is a decision every render
            // makes, alongside the mode and the frame rate, not an advanced
            // tuning knob a user hunts for behind a header.
            t.row_hover(
                ui,
                "Playback",
                Some(
                    "Loop: repeat the clip forever (the default). Off: play through once and \
                     stop on the last frame -- the timer is given a limit of \
                     (frames - 0.5) / fps, which expires halfway through the final frame. \
                     Costs nothing either way, and does nothing with an external clock.",
                ),
                |ui| {
                    ui.add_enabled_ui(!self.external_clock, |ui| {
                        widgets::toggle(ui,&mut self.loop_playback, "Loop");
                    });
                },
            );
        });

        self.draw_advanced_sections(ui);
    }

    /// Everything that tunes a render, grouped behind collapsing headers.
    ///
    /// Each header carries the VALUES inside it as chips rather than a bare
    /// noun, so a collapsed section still explains why this render differs from
    /// the last one; the `*_is_tuned` predicates additionally open a section on
    /// its first draw when it holds a non-default value. See
    /// [`crate::gui::util::section`] for why the chips carry most of that
    /// weight.
    ///
    /// Every per-mode condition inside the rows is unchanged -- the text rows
    /// still annotate themselves under Bricks, and the subtitle rows still
    /// annotate themselves with no track picked. A section is a container, not
    /// a second place for a rule about when a control applies.
    fn draw_advanced_sections(&mut self, ui: &mut Ui) {
        let (chips, open) = (self.text_chips(), self.text_is_tuned());
        section(ui, "video_text_section", "Text options", &chips, open, |ui| {
            widgets::settings_table(ui, |ui, t| self.draw_text_rows(t, ui));
        });

        let (chips, open) = (self.subtitle_chips(), self.subtitles.is_some());
        section(ui, "video_subtitle_section", "Subtitles", &chips, open, |ui| {
            widgets::settings_table(ui, |ui, t| self.draw_subtitle_rows(t, ui));
        });

        let (chips, open) = (self.timing_chips(), self.timing_is_tuned());
        section(
            ui,
            "video_timing_section",
            "Scaling & timing",
            &chips,
            open,
            |ui| {
                widgets::settings_table(ui, |ui, t| self.draw_timing_rows(t, ui));
            },
        );

        let (chips, open) = (self.picture_chips(), self.picture_is_tuned());
        section(ui, "video_picture_section", "Picture", &chips, open, |ui| {
            widgets::settings_table(ui, |ui, t| self.draw_picture_rows(t, ui));
        });
    }

    fn text_chips(&self) -> Vec<String> {
        if !matches!(self.mode, AnimMode::Text) {
            // Named rather than summarised: under Bricks every control in
            // here is inert, and a row of values would imply otherwise.
            return vec!["Text mode only".to_string()];
        }
        vec![
            self.text_preset.name().to_string(),
            format!("x{} repeat", self.text_char_repeat),
            format!("line height {:.2}", self.text_line_height),
            match self.colors {
                0 => "full colour".to_string(),
                n => format!("{n} colours"),
            },
        ]
    }

    fn text_is_tuned(&self) -> bool {
        let d = AnimOptions::default();
        // Gated on the mode for the same reason the rows inside are: opening a
        // section whose every control is inert would be pointing at something
        // that cannot be affecting this render.
        matches!(self.mode, AnimMode::Text)
            && (self.text_preset != FontPreset::MonaspaceArgon
                || self.text_fill_char != d.text.fill_char.to_string()
                || self.text_empty_char != d.text.empty_char.to_string()
                || self.text_char_repeat != d.text.char_repeat
                || self.text_line_height != d.text.line_height
                || self.colors != d.colors)
    }

    fn subtitle_chips(&self) -> Vec<String> {
        match &self.subtitles {
            None => vec!["none".to_string()],
            Some((name, track)) => vec![
                name.clone(),
                format!("{} cues", track.len()),
                format!("scale {:.1}", self.subtitle_scale),
                format!("lift {:.0}", self.subtitle_lift),
            ],
        }
    }

    fn timing_chips(&self) -> Vec<String> {
        vec![
            fit_name(self.fit).to_string(),
            filter_name(self.filter).to_string(),
            format!("from {:.2}s", self.start),
            if self.limit_duration {
                format!("for {:.2}s", self.duration)
            } else {
                "to the end".to_string()
            },
            format!("max {} frames", self.max_frames),
        ]
    }

    fn timing_is_tuned(&self) -> bool {
        self.fit != FitMode::Contain
            || self.filter != Filter::Lanczos
            || self.start != 0.0
            || self.limit_duration
            || self.max_frames != BANK_FRAMES as u32
    }

    fn picture_chips(&self) -> Vec<String> {
        let mut chips = vec![
            format!("alpha {}", self.alpha_threshold),
            brick_style_name(self.brick_style).to_string(),
            format!("extent {}", self.pixel_extent),
        ];
        if self.glow {
            chips.push("glow".to_string());
        }
        if self.external_clock {
            chips.push("external clock".to_string());
        }
        // Buttons are ON by default, so only their ABSENCE is worth a chip --
        // the same "summarise the non-default" rule the glow/clock chips follow.
        if !self.control_buttons {
            chips.push("no buttons".to_string());
        }
        // No "no loop" chip: the Loop checkbox is on the always-visible grid
        // now, so its state is already in view and does not need a header
        // summary the way a collapsed section's contents do.
        chips
    }

    fn picture_is_tuned(&self) -> bool {
        let d = AnimOptions::default();
        self.alpha_threshold != d.alpha_threshold
            || self.brick_style != d.brick_style
            || self.pixel_extent != d.pixel_extent
            || self.external_clock != d.external_clock
            || self.control_buttons != d.control_buttons
            || self.glow != d.glow
    }

    /// Text mode's font, glyph and geometry controls.
    ///
    /// **Three grid rows, not one.** These five controls used to share a single
    /// cell, and five labelled widgets side by side are wider than the pane at
    /// the default 600px window -- a cell that overflows widens its whole
    /// column, which is what put a horizontal scrollbar across the pane. Each
    /// control, its range and its tooltip is unchanged; only which row it sits
    /// on moved.
    fn draw_text_rows(&mut self, t: &mut widgets::SettingsTable, ui: &mut Ui) {
        let text_mode = matches!(self.mode, AnimMode::Text);

        t.row_hover(
            ui,
            "Font",
            Some(
                "Text mode only. Reuses the Image2Text pane's font, glyph and geometry \
                 controls -- see the Image2Text tab for the full calibration set.",
            ),
            |ui| {
                if text_mode {
                    let mut preset_changed = false;
                    widgets::combo(
                        ui,
                        "video_text_font_preset",
                        self.text_preset.name(),
                        160.0,
                        |ui| {
                            for p in FontPreset::ALL {
                                preset_changed |= widgets::combo_item(ui, &mut self.text_preset, p, p.name())
                                    .changed();
                            }
                        },
                    );
                    if preset_changed {
                        self.load_text_preset();
                    }
                } else {
                    note(ui, "(switch Render Mode to Text to configure)");
                }
            },
        );

        if text_mode {
            t.row_hover(
                ui,
                "Glyphs",
                Some("Which characters stand for an opaque and a transparent pixel"),
                |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Fill");
                        ui.add(egui::TextEdit::singleline(&mut self.text_fill_char).desired_width(24.0))
                            .on_hover_text("Glyph for opaque pixels (first character is used)");
                        ui.label("Empty");
                        ui.add(egui::TextEdit::singleline(&mut self.text_empty_char).desired_width(24.0))
                            .on_hover_text("Glyph for transparent pixels (first character is used)");
                        ui.label("Repeat");
                        widgets::slider(ui, egui::Slider::new(&mut self.text_char_repeat, 1..=4))
                            .on_hover_text("Glyphs per pixel; also sets the band layout's row width bound");
                    });
                },
            );

            t.row_hover(
                ui,
                "Line Height",
                Some("Component LineHeight (font size)"),
                |ui| {
                    ui.add(egui::DragValue::new(&mut self.text_line_height).speed(0.01));
                },
            );
        }

        t.row_hover(
            ui,
            "Colours",
            Some(
                "Text mode only: quantize to at most N colours with a median-cut palette \
                 before encoding (0 = full 24-bit colour, no quantization). Text mode writes \
                 a 16-character <color=\"RRGGBB\"> tag at the start of every colour run, so \
                 collapsing the palette lengthens runs and shrinks the render; useful values \
                 are 16-64.",
            ),
            |ui| {
                if self.shows_colours_control() {
                    widgets::slider(ui, egui::Slider::new(&mut self.colors, 0..=256));
                } else {
                    note(ui, "(Text mode only)");
                }
            },
        );
    }

    fn draw_subtitle_rows(&mut self, t: &mut widgets::SettingsTable, ui: &mut Ui) {
        t.row_hover(
            ui,
            "Subtitles",
            Some(
                "Render an .srt/.ass subtitle file as ONE wired TextDisplay overlaying \
                 the bottom of the screen -- 2 gates for the whole track, centred and \
                 outlined, drawn as vector glyphs at their own size rather than on the \
                 screen's pixel grid. The file is read in SOURCE time, so the Start \
                 control is honoured. Works in every render mode; over Bricks the \
                 subtitle lies flat in the screen's plane (the screen is on the ground), \
                 which is unverified by eye -- Text mode is the one it was designed \
                 against.",
            ),
            |ui| {
                // Wrapped: the picked track's file name is arbitrarily long.
                ui.horizontal(|ui| {
                    if widgets::info(ui, format!("{}  Pick subtitle file", icons::FONT)).clicked()
                        && self.pending_pick_subtitles.is_none()
                    {
                        self.pending_pick_subtitles = Some(pick_subtitle_bytes());
                    }
                    // The "clear" decision is taken while the track is
                    // borrowed for its label, so it is applied afterwards
                    // rather than assigned through the borrow.
                    let mut clear = false;
                    match &self.subtitles {
                        Some((name, track)) => {
                            clear = widgets::danger_icon(ui, icons::XMARK).clicked();
                            ui.label(format!("{name} -- {} cue(s)", track.len()));
                        }
                        None => {
                            note(ui, "(none)");
                        }
                    }
                    if clear {
                        self.subtitles = None;
                    }
                });
            },
        );

        t.row_hover(
            ui,
            "Subtitle Scale",
            Some(
                "How much bigger a subtitle line is than one row of the screen. At 192 px \
                 wide the screen is hundreds of glyph cells across while a subtitle line \
                 is 40-60 characters, so at equal size the text would occupy a seventh of \
                 the width; the default 6 covers about half. Unverified by eye in game.",
            ),
            |ui| {
                // Hidden without a track for the same reason the Colours
                // slider is hidden under brick mode: a control that provably
                // does nothing is worse than no control.
                if self.subtitles.is_some() {
                    widgets::slider(ui, egui::Slider::new(&mut self.subtitle_scale, 1.0..=24.0));
                } else {
                    note(ui, "(pick a subtitle file to configure)");
                }
            },
        );

        t.row_hover(
            ui,
            "Subtitle Lift",
            Some(
                "World units to lift the subtitle anchor toward the top of the picture \
                 (default 8). Measured by eye against Text mode at 192x108 with Subtitle \
                 Scale 6 -- Bricks/Color Array lay their screen flat, so the lift there \
                 moves the opposite horizontal axis and is unverified by eye. A lift too \
                 big for the picture's own height is refused rather than silently clamped.",
            ),
            |ui| {
                // Same gating as Subtitle Scale, for the same reason.
                if self.subtitles.is_some() {
                    widgets::slider(ui, egui::Slider::new(&mut self.subtitle_lift, 0.0..=64.0));
                } else {
                    note(ui, "(pick a subtitle file to configure)");
                }
            },
        );
    }

    /// How the source is resampled and how much of it is used.
    ///
    /// Fit Mode and Filter stay gated on `self.resize` exactly as before --
    /// the Resize checkbox they follow is in the always-visible grid above, so
    /// the two are no longer adjacent, but nothing about when they are live
    /// has changed.
    fn draw_timing_rows(&mut self, t: &mut widgets::SettingsTable, ui: &mut Ui) {
        // Fit Mode and Filter stay greyed while Resize is off: the control is
        // wrapped in `add_enabled_ui(self.resize, ..)` inside the row so it
        // follows the Resize toggle exactly as before. (The table draws the
        // label itself, so only the control greys now.)
        t.row_hover(
            ui,
            "Fit Mode",
            Some(
                "Exact: stretch to fit. Contain: letterbox, preserving aspect. Cover: fill and crop, preserving aspect.",
            ),
            |ui| {
                ui.add_enabled_ui(self.resize, |ui| {
                    ui.horizontal(|ui| {
                        for f in [FitMode::Exact, FitMode::Contain, FitMode::Cover] {
                            widgets::radio(ui,&mut self.fit, f, fit_name(f));
                        }
                    });
                });
            },
        );

        t.row_hover(
            ui,
            "Filter",
            Some("Resample filter used when resizing frames"),
            |ui| {
                ui.add_enabled_ui(self.resize, |ui| {
                    ui.horizontal(|ui| {
                        for f in [Filter::Lanczos, Filter::Nearest] {
                            widgets::radio(ui,&mut self.filter, f, filter_name(f));
                        }
                    });
                });
            },
        );

        t.row_hover(
            ui,
            "Start",
            Some("Seconds into the source to start from"),
            |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.start)
                        .speed(0.1)
                        .suffix("s")
                        .range(0.0..=f32::INFINITY),
                );
            },
        );

        t.row_hover(
            ui,
            "Duration",
            Some("Limit how much of the source (from Start) is used; unlimited runs to the end"),
            |ui| {
                ui.horizontal(|ui| {
                    widgets::toggle(ui,&mut self.limit_duration, "Limit");
                    ui.add_enabled_ui(self.limit_duration, |ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.duration)
                                .speed(0.1)
                                .suffix("s")
                                .range(0.0..=f32::INFINITY),
                        );
                    });
                });
            },
        );

        t.row_hover(
            ui,
            "Max Frames",
            Some("Hard cap on emitted frames -- frames past 65535 spill into extra wire arrays"),
            |ui| {
                widgets::slider(ui, egui::Slider::new(&mut self.max_frames, 1..=MAX_FRAMES as u32).logarithmic(true));
            },
        );
    }

    /// What the screen itself is made of.
    ///
    /// Alpha Threshold and Clock apply in both render modes; Brick Style,
    /// Pixel Extent and Material are read by the brick renderer only, as the
    /// Render Mode tooltip says.
    fn draw_picture_rows(&mut self, t: &mut widgets::SettingsTable, ui: &mut Ui) {
        t.row_hover(
            ui,
            "Alpha Threshold",
            Some("Pixels below this alpha, in every frame, are culled entirely"),
            |ui| {
                widgets::slider(ui, egui::Slider::new(&mut self.alpha_threshold, 0..=255));
            },
        );

        t.row_hover(
            ui,
            "Brick Style",
            Some(
                "Micro: small cube bricks (smallest at extent 1: 2 units wide). Smooth \
                 Tile: flat tiles, always 4 units tall regardless of extent.",
            ),
            |ui| {
                ui.horizontal(|ui| {
                    for s in [DisplayBrickStyle::Micro, DisplayBrickStyle::SmoothTile] {
                        widgets::radio(ui,&mut self.brick_style, s, brick_style_name(s));
                    }
                });
            },
        );

        t.row_hover(
            ui,
            "Pixel Extent",
            Some(
                "Half-extent of each display pixel, in units (1 = smallest: a 2-unit-wide \
                 brick). Adjacent pixels always tile flush at twice this value, so every \
                 value here is legal -- no size/style combination can overlap.",
            ),
            |ui| {
                widgets::slider(ui, egui::Slider::new(&mut self.pixel_extent, MIN_PIXEL_EXTENT..=50).text("units"));
            },
        );

        t.row_hover(
            ui,
            "Clock",
            Some("External: expose Frame as a chip input instead of running a built-in timer"),
            |ui| {
                widgets::toggle(ui,&mut self.external_clock, "External clock");
            },
        );

        t.row_hover(
            ui,
            "Controls",
            Some(
                "Pre-generate three physical Pause/Restart/Resume buttons on the main grid, \
                 wired into the clock so the render is controllable out of the box. Off means \
                 you wire the clock's control pins yourself. Inert with an external clock, \
                 which exposes no control pins.",
            ),
            |ui| {
                ui.add_enabled_ui(!self.external_clock, |ui| {
                    widgets::toggle(ui,&mut self.control_buttons, "Control buttons");
                });
            },
        );

        t.row_hover(
            ui,
            "Material",
            Some(
                "Glow at intensity 0: the screen lights itself instead of being lit by \
                 the world, so its colours stay true at night",
            ),
            |ui| {
                widgets::toggle(ui,&mut self.glow, "Glow");
            },
        );
    }

    fn draw_input(&mut self, ui: &mut Ui) {
        ui.label(
            "Pick a single animated file (GIF/APNG/WebP), a numbered frame sequence (PNG/JPG), \
             or a video file (mp4/mov/mkv/webm/avi/m4v).",
        );
        #[cfg(target_arch = "wasm32")]
        ui.label(
            "In the browser a video file is decoded by the pure-Rust builtin backend, which \
             handles H.264/CABAC in MP4 or MKV only; anything else is refused with a clear \
             message (the desktop build's ffmpeg backend covers the rest).",
        );

        let picking = self.pending_pick_animated.is_some()
            || self.pending_pick_sequence.is_some()
            || self.pending_pick_video.is_some();
        ui.horizontal_wrapped(|ui| {
            if widgets::info(ui, format!("{}  Pick animated file", icons::FILM)).clicked()
                && !picking
            {
                self.pending_pick_animated = Some(pick_animated_bytes());
            }
            if widgets::info(ui, format!("{}  Pick frame sequence", icons::IMAGE)).clicked()
                && !picking
            {
                self.pending_pick_sequence = Some(pick_images(true));
            }
            if widgets::info(ui, format!("{}  Pick video file", icons::FILM)).clicked()
                && !picking
            {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.pending_pick_video = Some(pick_video_path());
                }
                #[cfg(target_arch = "wasm32")]
                {
                    self.pending_pick_video = Some(pick_video_bytes());
                }
            }
        });

        #[cfg(not(target_arch = "wasm32"))]
        ui.horizontal_wrapped(|ui| {
            ui.label("Video Backend").on_hover_text(
                "Auto: builtin decode when it is safe for the file, ffmpeg otherwise. Pure \
                 Builtin: never uses ffmpeg (errors on formats it can't decode correctly). ffmpeg: \
                 always uses ffmpeg, downloading it first if it is missing and consented to. \
                 Only applies to a picked video file.",
            );
            for b in [Backend::Auto, Backend::Builtin, Backend::Ffmpeg] {
                widgets::radio(ui,&mut self.backend, b, backend_name(b));
            }
        });

        let mut clear_input = false;
        match &self.input {
            Input::None => {
                ui.label("No source selected.");
            }
            Input::Animated { name, clip, preview } => {
                // `max_col_width` is what lets the (arbitrarily long) file
                // name wrap instead of widening the pane -- see
                // `gui::util::settings_grid` for why it is the switch that
                // turns wrapping on inside grid cells at all.
                let name_width = (ui.available_width() - 60.0).max(120.0);
                egui::Grid::new("video_animated_grid")
                    .striped(true)
                    .num_columns(3)
                    .spacing([8.0, 4.0])
                    .min_col_width(4.0)
                    .max_col_width(name_width)
                    .show(ui, |ui| {
                        if widgets::danger_icon(ui, icons::XMARK).clicked() {
                            clear_input = true;
                        }
                        thumb(ui, preview);
                        ui.label(format!(
                            "{name} -- {}x{}, {} frame(s), {:.2} fps",
                            clip.width,
                            clip.height,
                            clip.frames.len(),
                            clip.fps
                        ));
                    });
            }
            Input::Sequence(frames) => {
                ui.label(format!("{} frame(s) in sequence", frames.len()));
                let name_width = (ui.available_width() - 60.0).max(120.0);
                egui::Grid::new("video_sequence_grid")
                    .striped(true)
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .min_col_width(4.0)
                    .max_col_width(name_width)
                    .show(ui, |ui| {
                        for img in frames {
                            thumb(ui, img);
                            ui.label(&img.name);
                            ui.end_row();
                        }
                    });
                if widgets::danger(ui, format!("{}  Clear sequence", icons::XMARK)).clicked() {
                    clear_input = true;
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            Input::Video { path, name } => {
                // Wrapped: a full filesystem path is easily wider than the pane.
                ui.horizontal_wrapped(|ui| {
                    if widgets::danger_icon(ui, icons::XMARK).clicked() {
                        clear_input = true;
                    }
                    ui.label(format!("{name} -- video source ({})", path.display()));
                });
            }
            #[cfg(target_arch = "wasm32")]
            Input::Video { name, .. } => {
                ui.horizontal_wrapped(|ui| {
                    if widgets::danger_icon(ui, icons::XMARK).clicked() {
                        clear_input = true;
                    }
                    ui.label(format!("{name} -- video source"));
                });
            }
        }
        if clear_input {
            self.input = Input::None;
        }
    }

    fn draw_cost(&self, ui: &mut Ui) {
        // `source_info` (and so `live_cost`) always returns `None` for a
        // video source -- see `Input::Video`'s doc for why this is not
        // probed eagerly -- so this is called out separately rather than
        // folding into `live_cost`'s own `None` case below, which reads
        // as "nothing picked yet" and would be misleading here.
        if matches!(self.input, Input::Video { .. }) {
            ui.label(
                "Video source selected -- cost is measured once Generate opens the file.",
            );
            return;
        }

        match self.live_cost() {
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

                // Text mode's `Cost::chars` is 0 BY DESIGN (see
                // `cost::estimate_text`'s doc): unlike hex mode's fixed
                // per-pixel stride, a text render's length depends on how the
                // clip's colours actually run, which nothing can know before
                // the render. `cost.chars` is never printed above for any
                // mode, so there is no 0 here for a viewer to mistake for a
                // measurement -- but the reader still deserves SOME number to
                // judge the render by, so this prints the same closed-form
                // worst-case bound `main.rs`'s `log_cost` prints for the CLI,
                // labelled the same way: a ceiling, not an estimate.
                if matches!(self.mode, AnimMode::Text) {
                    self.draw_text_char_bound(ui);
                }
            }
        }
    }

    /// The text-mode worst-case character bound, shown alongside the gate
    /// estimate. Computed the same way `main.rs`'s `log_cost` computes it for
    /// `--anim-mode text`, from `text_layout::plan_bands` and
    /// `text_layout::worst_case_row_chars` over the SAME resized
    /// width/height `live_cost` used, so the two can't disagree either.
    fn draw_text_char_bound(&self, ui: &mut Ui) {
        let Some((w, h)) = self.output_dims() else { return };
        let repeat = self.text_char_repeat.max(1);
        let Ok(plan) = text_layout::plan_bands(w as usize, h as usize, repeat) else {
            // A layout this geometry cannot support -- Generate will fail
            // with `plan_bands`' own message, which names the width, the
            // limit and the fix; nothing useful to add here.
            return;
        };
        let Some(first) = plan.first() else { return };
        let row = text_layout::worst_case_row_chars(w as usize, repeat);
        let rows = first.rows;
        let per_band = rows * row + rows.saturating_sub(1);
        ui.label(format!(
            "{} band(s) of {rows} row(s); UPPER BOUND {per_band} character(s) per band per \
             frame ({row} per {w}-pixel row: 16 for a colour tag + {repeat} glyph char(s) per \
             pixel, worst case every pixel starting its own run). NOT an estimate -- real length \
             is content-dependent, which is why no character total is reported above; Colours is \
             what shortens it",
            plan.len(),
        ));
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
            draw_progress_bar(ui, &self.progress_label, self.progress_pos, self.progress_total);
            // Visible only while a render is actually in flight (this whole
            // branch), and only where a Cancel click can do anything -- on
            // wasm `work` already ran to completion before this button could
            // ever be drawn and clicked (see `cancel_flag`'s doc), so there
            // is no flag here to back it with.
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(flag) = &self.cancel_flag {
                draw_cancel_button(ui, flag);
            }
            return;
        }

        // A destination that cannot be written is refused BEFORE the button is
        // offered -- the red label in the settings grid above is the same
        // condition, and it used to be advisory: the pane warned and then wrote
        // to the bad name anyway. See `util::refuse_bad_out_file`.
        if refuse_bad_out_file(ui, &shared.out_file) {
            return;
        }

        let has_input = match &self.input {
            Input::None => false,
            Input::Animated { .. } => true,
            Input::Sequence(frames) => !frames.is_empty(),
            Input::Video { .. } => true,
        };

        // The ffmpeg consent modal (drawn elsewhere, see `FfmpegModal::draw`)
        // already blocks input to the rest of the pane via its backdrop
        // while it's open; this is belt-and-suspenders so a click that
        // somehow lands here anyway can't start a second `generate` call
        // while one is already waiting on consent or mid-download.
        #[cfg(not(target_arch = "wasm32"))]
        if self.modal.is_open() {
            ui.label("Waiting on the ffmpeg download prompt above...");
            return;
        }

        if has_input {
            if widgets::primary(ui, format!("{}  Generate video2brick save", icons::DOWNLOAD)).clicked() {
                self.generate(shared);
            }
        } else {
            ui.label("Pick an animated file, frame sequence, or video file to continue...");
        }
    }

    /// On click: decode (cached for an animated file, deferred to the worker
    /// for a sequence) -> resize -> resample -> build the brick world ->
    /// encode -> deliver. Runs on a background thread on native (mirroring
    /// `HeightmapApp::run_converter`); `wasm32` has no `std::thread::spawn`,
    /// so `work` runs in place and the tab blocks for the render. Captures
    /// are cheap (scalars, `Arc` bumps -- see [`GenSource`]) except
    /// `Animated`'s `clip.clone()`. `deliver_save` runs inside `work`, so it
    /// needs no UI thread. Every failure is logged via `log::error!`, never
    /// panicked.
    fn generate(&mut self, shared: &SharedOptions) {
        // Belt-and-suspenders: `draw_submit` already hides the button while
        // a render is in flight, but refuse here too rather than trust the
        // UI alone.
        if self.pending_generate.is_some() {
            return error!("a render is already in progress");
        }

        // A video source needs to know, before any worker starts, whether
        // the selected backend needs ffmpeg and ffmpeg is missing --
        // `FfmpegModal::draw` needs the UI thread to ask, and a worker
        // thread has none. `check_ffmpeg_consent` makes exactly the same
        // trial `open_video_ensuring` would make internally; see its doc.
        #[cfg(not(target_arch = "wasm32"))]
        if self.modal.is_open() {
            return error!("waiting on the ffmpeg download prompt");
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Input::Video { path, .. } = &self.input {
            match self.check_ffmpeg_consent(path) {
                FfmpegCheck::Ready => {}
                FfmpegCheck::NeedsConsent => {
                    self.modal.request(path.clone());
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
            Input::Video { path, .. } => GenSource::Video { path: path.clone() },
            #[cfg(target_arch = "wasm32")]
            Input::Video { bytes, name } => GenSource::Video {
                bytes: std::sync::Arc::clone(bytes),
                name: name.clone(),
            },
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
        // `Copy`, so this is a plain value capture -- no `Arc`/clone needed,
        // and (together with `anim_opts` below) this is the SAME `self.mode`
        // `live_cost` just read, so the worker below can never build a
        // different graph than the one the readout described.
        let mode = self.mode;
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

        // A FRESH flag every render -- never a reused one a previous render
        // might have left `true` -- is what makes "returning to idle clears
        // the flag" true without any separate reset step (see
        // `cancel_flag`'s doc). The clone stored on `self` is what
        // `draw_submit`'s Cancel button sets; the original moves into `work`
        // below, where `ChannelProgress::is_cancelled` reads it.
        let cancel_flag = Arc::new(AtomicBool::new(false));
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.cancel_flag = Some(Arc::clone(&cancel_flag));
        }

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
                GenSource::Video { path } => {
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
                // The web build decodes the uploaded bytes with the builtin
                // backend (no ffmpeg); the guards inside `open_video_bytes`
                // refuse anything it cannot decode correctly.
                #[cfg(target_arch = "wasm32")]
                GenSource::Video { bytes, name } => backend::open_video_bytes(&name, bytes)?,
            };

            // Omitted resize means "use the source's own dimensions" -- skip
            // the resize entirely rather than resampling to an identical
            // size. Resize (if requested) then resample, streamed rather
            // than materialized twice: `AdaptedSource` layers `ResizeStream`
            // under `FpsStream` so frames are scaled before selection, never
            // the other way around. The builtin video backend has no
            // resize/resample parameters of its own (`BuiltinVideoSource::
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
            let mut progress = ChannelProgress::new(progress_tx, cancel_flag);
            let world = mode.build(&adapted, &anim_opts, &mut progress)?;
            deliver_world_unless_cancelled(world, &progress, &out_file, out_clipboard)
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
        bound_pane_width(ui);
        self.poll_picks();
        self.poll_generate(ui.ctx());
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.modal.poll(ui.ctx());
            let prompt = format!(
                "This video needs the ffmpeg decode backend, and no ffmpeg install was found \
                 on this machine. Download it now from {}?",
                ffmpeg_sidecar::download::ffmpeg_download_url()
                    .unwrap_or("the official ffmpeg build server")
            );
            self.modal
                .draw(ui.ctx(), "video", &prompt, "video was not converted");
        }
        // File selection above the settings, each in its own card.
        widgets::section(ui, "Source", |ui| self.draw_input(ui));
        ui.add_space(10.0);
        widgets::section(ui, "Settings", |ui| self.draw_settings(ui, shared));
    }

    /// The fixed footer: the cost readout and the Generate button.
    pub fn draw_footer(&mut self, ui: &mut Ui, shared: &mut SharedOptions) {
        bound_pane_width(ui);
        self.draw_cost(ui);
        self.draw_submit(ui, shared);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use crate::anim::bricks::build_brick_world;
    use crate::progress::NoProgress;
    // Only the tests build a `World` directly now -- `generate` hands
    // whatever `mode.build` returns straight to
    // `gui::util::deliver_world_unless_cancelled`.
    use brdb::World;

    /// A `Progress` whose `is_cancelled` is fixed at construction, for
    /// testing `deliver_world_unless_cancelled` in isolation -- it needs no
    /// other `Progress` behaviour, so every other method is a no-op.
    struct FixedCancel(bool);
    impl Progress for FixedCancel {
        fn begin(&mut self, _label: &str, _total: Option<u64>) {}
        fn tick(&mut self, _n: u64) {}
        fn finish(&mut self) {}
        fn is_cancelled(&self) -> bool {
            self.0
        }
    }

    /// A tiny but real `World`, built the same way a render actually would
    /// (`build_brick_world` over a one-frame clip), so `to_brz_vec` is
    /// exercised on genuine content rather than gambling on whatever a bare
    /// `World::new()` happens to encode as.
    fn tiny_world() -> World {
        let clip = Clip {
            width: 2,
            height: 2,
            fps: 10.0,
            frames: vec![RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 0, 255]))],
        };
        build_brick_world(&clip, &AnimOptions::default(), &mut NoProgress).expect("build")
    }

    // Asserted on the file magic rather than on which function was called:
    // a BRZ archive starts `BRZ\0` and a brdb is a SQLite database, so
    // nothing about how the bytes were produced can fake the check.
    #[test]
    fn the_output_extension_decides_the_container() {
        let base = std::env::temp_dir().join(format!("h2b_container_{}", std::process::id()));
        let brz = base.with_extension("brz");
        let brdb = base.with_extension("brdb");
        for p in [&brz, &brdb] {
            let _ = std::fs::remove_file(p);
        }

        deliver_world_unless_cancelled(
            tiny_world(),
            &FixedCancel(false),
            &brz.to_string_lossy(),
            false,
        )
        .expect("a .brz destination must be written");
        deliver_world_unless_cancelled(
            tiny_world(),
            &FixedCancel(false),
            &brdb.to_string_lossy(),
            false,
        )
        .expect("a .brdb destination must be written");

        let brz_bytes = std::fs::read(&brz).expect("read the .brz");
        let brdb_bytes = std::fs::read(&brdb).expect("read the .brdb");
        assert!(
            brz_bytes.starts_with(b"BRZ"),
            "a .brz destination must hold a BRZ archive, got {:02x?}",
            &brz_bytes[..brz_bytes.len().min(16)]
        );
        assert!(
            brdb_bytes.starts_with(b"SQLite format 3"),
            "a .brdb destination must hold a brdb (SQLite) database, not BRZ bytes under a \
             .brdb name -- got {:02x?}",
            &brdb_bytes[..brdb_bytes.len().min(16)]
        );

        for p in [&brz, &brdb] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn a_destination_with_no_known_extension_is_refused() {
        let path = std::env::temp_dir().join(format!("h2b_container_{}_noext", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let result = deliver_world_unless_cancelled(
            tiny_world(),
            &FixedCancel(false),
            &path.to_string_lossy(),
            false,
        );

        assert!(result.is_err(), "an unknown extension must be refused");
        assert!(!path.exists(), "and nothing may be written to it");
    }

    // Must return `Ok(())`, not `Err`: `generate`'s worker logs any `Err`
    // via `log::error!`, which a cancellation must never trigger.
    #[test]
    fn a_cancelled_render_writes_no_output_file() {
        let path = std::env::temp_dir().join(format!(
            "h2b_cancel_test_{}_no_output.brz",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path); // in case a previous run left one
        let path_str = path.to_string_lossy().to_string();

        let result = deliver_world_unless_cancelled(tiny_world(), &FixedCancel(true), &path_str, false);

        assert!(result.is_ok(), "a cancelled render must not surface as an error: {result:?}");
        assert!(!path.exists(), "a cancelled render must write no output file");
    }

    #[test]
    fn an_uncancelled_render_still_writes_its_output_file() {
        let path = std::env::temp_dir().join(format!(
            "h2b_cancel_test_{}_output.brz",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let path_str = path.to_string_lossy().to_string();

        let result =
            deliver_world_unless_cancelled(tiny_world(), &FixedCancel(false), &path_str, false);

        assert!(result.is_ok(), "an uncancelled render must succeed: {result:?}");
        assert!(
            path.exists(),
            "an uncancelled render's default (non-cancelled) path must still write its output"
        );
        let written = std::fs::read(&path).expect("the written file must be readable");
        assert!(!written.is_empty(), "the written save must have real content");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn channel_progress_is_cancelled_reflects_the_shared_flag() {
        let (tx, _rx) = std::sync::mpsc::channel::<ProgressMsg>();
        let flag = Arc::new(AtomicBool::new(false));
        let progress = ChannelProgress::new(tx, Arc::clone(&flag));

        assert!(!progress.is_cancelled(), "a fresh flag must not read as cancelled");
        flag.store(true, Ordering::Relaxed);
        assert!(
            progress.is_cancelled(),
            "setting the shared flag (as the Cancel button does) must be visible immediately"
        );
    }

    /// A picked source with a real width/height/frame count, so `live_cost`
    /// has something to estimate without a real UI or a real decode. Frame
    /// dimensions are 1x1 -- `source_info` reads `clip.width`/`clip.height`
    /// (plain fields), never the pixel data itself, so the actual image size
    /// stored per frame is irrelevant to the estimate.
    fn picked_clip(width: u32, height: u32, fps: f32, frames: usize) -> Input {
        let clip = Clip { width, height, fps, frames: vec![RgbaImage::new(1, 1); frames] };
        Input::Animated {
            name: "clip".to_string(),
            preview: PickedImage { name: "clip".to_string(), image: Arc::new(RgbaImage::new(1, 1)) },
            clip,
        }
    }

    // Pins `live_cost` routing through `self.anim_opts()`, the same options
    // `generate` builds the world from -- switching `mode` alone must move
    // the gate estimate by the text-vs-brick order of magnitude.
    #[test]
    fn the_live_estimate_follows_the_selected_mode() {
        let mut app = VideoApp::default();
        // Matches `app.fps`'s own default (10.0) so `estimated_frame_count`
        // reports back exactly the 300 frames the clip carries, with no
        // resampling to account for.
        app.input = picked_clip(192, 108, 10.0, 300);

        app.mode = AnimMode::Brick(AnimEncoding::Hex);
        let brick = app
            .live_cost()
            .expect("a source is picked")
            .expect("300 frames is under the default max_frames")
            .gates;

        app.mode = AnimMode::Text;
        let text = app
            .live_cost()
            .expect("a source is picked")
            .expect("300 frames is under the default max_frames")
            .gates;

        assert!(
            text < brick / 100,
            "text mode ({text} gates) must be far under brick mode ({brick} gates) for the \
             identical source and settings"
        );
    }

    // `AnimOptions::colors` is read only by `text_bricks::build_text_world`.
    #[test]
    fn the_colours_control_is_only_meaningful_in_text_mode() {
        let mut app = VideoApp::default();
        app.mode = AnimMode::Brick(AnimEncoding::Hex);
        assert!(!app.shows_colours_control());
        app.mode = AnimMode::Text;
        assert!(app.shows_colours_control());
    }

    #[test]
    fn anim_opts_carries_the_colours_and_text_controls_through() {
        let mut app = VideoApp::default();
        app.colors = 24;
        app.text_char_repeat = 3;
        app.text_fill_char = "#".to_string();
        app.text_empty_char = ".".to_string();
        app.alpha_threshold = 77;

        let opts = app.anim_opts();
        assert_eq!(opts.colors, 24);
        assert_eq!(opts.text.char_repeat, 3);
        assert_eq!(opts.text.fill_char, '#');
        assert_eq!(opts.text.empty_char, '.');
        // Both fields, not just the top-level one -- see `anim_opts`'s doc
        // for why the palette needs the encoder's own visibility rule.
        assert_eq!(opts.alpha_threshold, 77);
        assert_eq!(opts.text.alpha_threshold, 77);
    }

    /// A three-cue track, built in memory -- the picker's parse is covered by
    /// `subs::parse_auto`'s own tests, and nothing here needs a file.
    fn picked_track() -> Arc<Subtitles> {
        Arc::new(Subtitles::new(vec![crate::subs::Cue {
            start_s: 0.0,
            end_s: 600.0,
            text: "a line".to_string(),
        }]))
    }

    // `subtitle_scale`/`subtitle_lift` are deliberately set off their
    // defaults: an assertion against the default would pass even unwired.
    #[test]
    fn anim_opts_carries_the_subtitle_controls_through() {
        let mut app = VideoApp::default();
        assert!(
            app.anim_opts().subtitles.is_none(),
            "no picked file means no track at all -- the hard gate every renderer applies"
        );
        assert_eq!(
            app.subtitle_scale,
            crate::anim::subtitle_display::DEFAULT_SUBTITLE_SCALE,
            "the pane must start at the module's own default, not a second copy of it"
        );
        assert_eq!(
            app.subtitle_lift,
            crate::anim::subtitle_display::DEFAULT_SUBTITLE_LIFT,
            "the pane must start at the module's own default, not a second copy of it"
        );

        app.subtitles = Some(("track.srt".to_string(), picked_track()));
        app.subtitle_scale = 9.5;
        app.subtitle_lift = 3.5;
        let opts = app.anim_opts();
        assert_eq!(opts.subtitles.as_ref().map(|t| t.len()), Some(1));
        assert_eq!(opts.subtitle_scale, 9.5);
        assert_eq!(opts.subtitle_lift, 3.5);
    }

    // A subtitle file is in source time, so Start must reach `AnimOptions`
    // too, or a track timed from 0 would be off by the render's own offset.
    #[test]
    fn anim_opts_carries_the_start_offset_the_subtitle_timing_needs() {
        let mut app = VideoApp::default();
        app.start = 120.0;
        assert_eq!(app.anim_opts().source_start_s, 120.0);
        // Clamped exactly as `generate` clamps it for the `AdaptedSource`, so
        // the two can't disagree about where frame 0 is.
        app.start = -5.0;
        assert_eq!(app.anim_opts().source_start_s, 0.0);
    }

    // The readout is routed through the same `(mode, anim_opts())` pair the
    // render dispatches on, so picking a subtitle file must move the estimate
    // by the two gates it actually builds.
    #[test]
    fn the_live_estimate_counts_a_picked_subtitle_track() {
        let mut app = VideoApp::default();
        app.input = picked_clip(192, 108, 10.0, 300);
        for mode in [AnimMode::Brick(AnimEncoding::Hex), AnimMode::Text] {
            app.mode = mode;
            app.subtitles = None;
            let without = app.live_cost().unwrap().unwrap().gates;
            app.subtitles = Some(("track.srt".to_string(), picked_track()));
            let with = app.live_cost().unwrap().unwrap().gates;
            assert_eq!(
                with - without,
                2,
                "{mode:?}: the readout must count the subtitle's ArrayVar and Get"
            );
        }
    }
}
