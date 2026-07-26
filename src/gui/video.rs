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
    video::{
        Clip,
        scale::{Filter, FitMode, resample_fps, resize_clip},
        source::{Source, decode},
    },
};
use egui::{Button, Color32, Ui};
use image::RgbaImage;
use log::{error, info};
use poll_promise::Promise;

// `MAX_FRAMES` (imported above): the largest frame count a render may carry
// across all banks, now that a clip's frames can spill across multiple wire
// arrays. `BANK_FRAMES` (also imported) is the old single-array limit -- a
// wire array index is a `u16`, `0..=65535` -- and is what the `max_frames`
// slider *defaults* to, so a fresh session doesn't silently opt into a
// million-frame render; `MAX_FRAMES` is only the slider's upper *bound*.
// Passed to `resample_fps` and never `usize::MAX`, which would re-enable an
// unbounded resampling loop and let a fat-fingered fps OOM. `anim::pack` is
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
}

pub struct VideoApp {
    input: Input,
    pending_pick_animated: Option<Promise<Option<(String, Vec<u8>)>>>,
    pending_pick_sequence: Option<Promise<Vec<PickedImage>>>,

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

impl VideoApp {
    /// `(width, height, frame_count, source_fps)` of the picked input at its
    /// natural resolution/frame count, before any resize/resample. `None`
    /// until something is picked.
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
    /// `None` while nothing has been picked yet -- the brief's "unknown
    /// until picked" case -- rather than showing a misleading zero.
    fn cost_preview(&self) -> Option<Cost> {
        let (src_w, src_h, src_frames, src_fps) = self.source_info()?;
        let (w, h) = if self.resize {
            (self.width.max(1), self.height.max(1))
        } else {
            (src_w, src_h)
        };

        let src_fps = if src_fps.is_finite() && src_fps > 0.0 { src_fps } else { 1.0 };
        let target_fps = if self.fps.is_finite() && self.fps > 0.0 { self.fps } else { 1.0 };
        let source_duration = src_frames as f32 / src_fps;
        let start = self.start.max(0.0);
        let end = if self.limit_duration {
            (start + self.duration.max(0.0)).min(source_duration)
        } else {
            source_duration
        };

        let frames = ((end - start).max(0.0) * target_fps).ceil() as usize;
        let frames = frames.min(self.max_frames as usize).min(MAX_FRAMES);

        Some(cost::estimate(w, h, frames, self.anim_opts().bank_size))
    }

    /// Poll the two in-flight pickers and apply whichever resolves.
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
        ui.label(
            "Pick a single animated file (GIF/APNG/WebP) or a numbered frame sequence (PNG/JPG).",
        );

        let picking = self.pending_pick_animated.is_some() || self.pending_pick_sequence.is_some();
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
        }
        if clear_input {
            self.input = Input::None;
        }
    }

    fn draw_cost(&self, ui: &mut Ui) {
        match self.cost_preview() {
            None => {
                ui.label("Pick a source above to see a cost estimate.");
            }
            Some(cost) => {
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
        let has_input = match &self.input {
            Input::None => false,
            Input::Animated { .. } => true,
            Input::Sequence(frames) => !frames.is_empty(),
        };

        if has_input {
            if ui
                .add(Button::new("Generate video2brick save").fill(Color32::from_rgb(50, 90, 50)))
                .clicked()
            {
                self.generate(shared);
            }
        } else {
            ui.label("Pick an animated file or frame sequence to continue...");
        }
    }

    /// On click: decode (already cached for an animated file; a cheap
    /// re-sort/clone for a sequence) -> resize -> resample -> build the
    /// brick world -> encode -> deliver. Every failure is logged, never
    /// panicked.
    fn generate(&self, shared: &SharedOptions) {
        let clip = match &self.input {
            Input::None => return error!("pick an animated file or frame sequence first"),
            Input::Animated { clip, .. } => clip.clone(),
            Input::Sequence(frames) => {
                if frames.is_empty() {
                    return error!("pick a frame sequence first");
                }
                let named: Vec<(String, RgbaImage)> = frames
                    .iter()
                    .map(|p| (p.name.clone(), (*p.image).clone()))
                    .collect();
                match decode(Source::Sequence(named), self.fps) {
                    Ok(c) => c,
                    Err(e) => return error!("{e}"),
                }
            }
        };

        // Omitted resize means "use the clip's own dimensions" -- skip the
        // resize entirely rather than resampling to an identical size.
        let clip = if self.resize {
            resize_clip(clip, self.width.max(1), self.height.max(1), self.fit, self.filter)
        } else {
            clip
        };

        let duration = if self.limit_duration { Some(self.duration.max(0.0)) } else { None };
        // Never pass an unbounded sentinel here -- `max_frames` is already
        // clamped by the slider's range, but clamp again defensively so a
        // future UI change can't silently re-open the OOM this guards.
        let max_frames = (self.max_frames as usize).min(MAX_FRAMES);
        let clip = match resample_fps(clip, self.fps, self.start.max(0.0), duration, max_frames) {
            Ok(c) => c,
            Err(e) => return error!("{e}"),
        };

        info!(
            "Building {} frame(s) at {}x{}...",
            clip.frames.len(),
            clip.width,
            clip.height
        );
        let anim_opts = self.anim_opts();
        let world = match build_brick_world(&clip, &anim_opts) {
            Ok(w) => w,
            Err(e) => return error!("{e}"),
        };

        info!("Writing Save to {}", shared.out_file);
        let data = match world.to_brz_vec() {
            Ok(d) => d,
            Err(e) => return error!("failed to encode brz: {e}"),
        };
        if let Err(e) = deliver_save(data, &shared.out_file, shared.out_clipboard) {
            return error!("{e}");
        }
        info!("Done!");
    }

    pub fn draw(&mut self, ui: &mut Ui, shared: &mut SharedOptions) {
        self.poll_picks();
        self.draw_settings(ui, shared);
        self.draw_input(ui);
        ui.add_space(8.0);
        ui.separator();
        self.draw_cost(ui);
        ui.separator();
        self.draw_submit(ui, shared);
    }
}
