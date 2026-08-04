use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{
    map::{Colormap, ColormapPNG, Heightmap, HeightmapFlat, HeightmapPNG},
    progress::Progress,
    util::GenOptions,
};
use brdb::World;
use image::RgbaImage;
use poll_promise::Promise;

type MapPair = (Box<dyn Heightmap>, Box<dyn Colormap>);

/// Two-column settings grid. `max_col_width` is load-bearing: it is the only
/// switch that turns on text wrapping inside grid cells, so without it a long
/// label forces a horizontal scrollbar. Don't use `horizontal_wrapped` in a
/// cell -- it under-reports its height and the next row draws on top; long
/// text should go through [`note`] instead.
pub fn settings_grid(ui: &egui::Ui, id: &str) -> egui::Grid {
    // Label column is short; the cap only stops column 1's long annotations
    // from pushing the pane wide.
    let max_col_width = (ui.available_width() - 60.0).max(160.0);
    egui::Grid::new(id)
        .striped(true)
        .num_columns(2)
        .spacing([40.0, 4.0])
        .min_col_width(LABEL_COLUMN_WIDTH)
        .max_col_width(max_col_width)
}

/// Floor so every section's own Grid sizes column 1 identically (a grid can't
/// span a collapsing body). `tests/gui_accordion.rs` asserts no label outgrows
/// it.
pub const LABEL_COLUMN_WIDTH: f32 = 112.0;

/// Pin a pane to the width it was handed and make text wrap by default.
/// Called once at the top of a pane's `draw`: `style.wrap_mode` is the first
/// thing `Ui::wrap_mode` consults, ahead of the fallbacks that otherwise
/// resolve to `TextWrapMode::Extend` in a `horizontal` or unbounded grid cell.
pub fn bound_pane_width(ui: &mut egui::Ui) {
    ui.set_max_width(ui.available_width());
    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
}

/// Wrap a long label so it folds instead of getting clipped. Unlike
/// `horizontal_wrapped`, a `Label` allocates its own already-wrapped rect, so
/// a `Grid` measures the row height correctly.
pub fn note(ui: &mut egui::Ui, text: impl Into<String>) -> egui::Response {
    ui.add(egui::Label::new(text.into()).wrap())
}

/// A small rounded badge for one value in an accordion header. Uses
/// `strong_text_color` (not the muted label colour) since it's read against a
/// filled background; zero vertical margin since `RichText::small` is already
/// shorter than a normal row.
pub fn chip(ui: &mut egui::Ui, text: &str) -> egui::Response {
    let (fill, stroke, text_color) = {
        let w = ui.visuals().widgets.inactive;
        (w.bg_fill, w.bg_stroke, ui.visuals().strong_text_color())
    };
    egui::Frame::new()
        .fill(fill)
        .stroke(stroke)
        .corner_radius(egui::CornerRadius::same(4))
        .inner_margin(egui::Margin {
            left: 5,
            right: 5,
            top: 0,
            bottom: 0,
        })
        .show(ui, |ui| {
            ui.add(
                egui::Label::new(egui::RichText::new(text).small().color(text_color))
                    .selectable(false),
            );
        })
        .response
}

/// One collapsible block of settings: a name plus a row of value [`chip`]s.
/// Custom header (not `CollapsingHeader`) so long chips wrap to a second line
/// instead of widening the pane. `id_salt` must be stable across chip
/// changes, or the header forgets whether it was open. `open_by_default`
/// only applies the first frame; egui remembers user toggles after that.
pub fn section(
    ui: &mut egui::Ui,
    id_salt: &str,
    title: &str,
    chips: &[String],
    open_by_default: bool,
    body: impl FnOnce(&mut egui::Ui),
) {
    let id = ui.make_persistent_id(id_salt);
    // Whole row toggles. selectable_labels defaults ON, so a plain label
    // wins the hit-test over the row; title+chips use .selectable(false).
    // The interaction rect is widened, never allocated: a zero-height
    // wrapped item still costs one item_spacing.y.
    let mut row_clicked = false;
    let mut header = egui::collapsing_header::CollapsingState::load_with_default_open(
        ui.ctx(),
        id,
        open_by_default,
    )
    .show_header(ui, |ui| {
        let row = ui.horizontal_wrapped(|ui| {
            ui.add(egui::Label::new(egui::RichText::new(title).strong()).selectable(false));
            for c in chips {
                chip(ui, c);
            }
        });
        let mut hit = row.response.rect;
        hit.max.x = hit.max.x.max(ui.max_rect().right());
        row_clicked = ui
            .interact(hit, row.response.id, egui::Sense::click())
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .clicked();
    });
    if row_clicked {
        header.toggle();
    }
    // The triangle is a separate widget from the row, so it needs saying
    // twice or the cursor changes over most of the header and not all of it.
    let (toggle_button, _, _) = header.body(body);
    toggle_button.on_hover_cursor(egui::CursorIcon::PointingHand);
}

/// An image the user picked, already decoded -- works identically on native
/// (file dialogs + filesystem) and web (async pickers + in-memory bytes).
#[derive(Clone)]
pub struct PickedImage {
    pub name: String,
    pub image: Arc<RgbaImage>,
}

/// Pick one or more images asynchronously; poll the returned promise each
/// frame. Files that fail to decode are dropped with a log message.
pub fn pick_images(multiple: bool) -> Promise<Vec<PickedImage>> {
    let dialog = rfd::AsyncFileDialog::new().add_filter("Image Files", &["png", "jpg", "jpeg"]);
    let pick = async move {
        let handles = if multiple {
            dialog.pick_files().await.unwrap_or_default()
        } else {
            dialog.pick_file().await.into_iter().collect()
        };
        let mut out = Vec::new();
        for h in handles {
            let name = h.file_name();
            let bytes = h.read().await;
            match image::load_from_memory(&bytes) {
                Ok(img) => out.push(PickedImage {
                    name,
                    image: Arc::new(img.to_rgba8()),
                }),
                Err(e) => log::error!("could not decode {name}: {e}"),
            }
        }
        out
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        Promise::spawn_thread("pick_images", move || pollster::block_on(pick))
    }
    #[cfg(target_arch = "wasm32")]
    {
        Promise::spawn_async(pick)
    }
}

/// Pick a single file's raw bytes, without any image decode -- unlike
/// `pick_images`, which decodes eagerly and keeps only the first frame of an
/// animated image. `None` if the user cancels.
pub fn pick_animated_bytes() -> Promise<Option<(String, Vec<u8>)>> {
    let dialog = rfd::AsyncFileDialog::new()
        .add_filter("Animated Images", &["gif", "png", "webp", "jpg", "jpeg"]);
    let pick = async move {
        let handle = dialog.pick_file().await?;
        let name = handle.file_name();
        let bytes = handle.read().await;
        Some((name, bytes))
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        Promise::spawn_thread("pick_animated_bytes", move || pollster::block_on(pick))
    }
    #[cfg(target_arch = "wasm32")]
    {
        Promise::spawn_async(pick)
    }
}

/// Pick a subtitle file's raw bytes and name -- bytes rather than a path so it
/// works on the web too; the name comes along since its extension picks the
/// parser (`subs::parse_auto`). `None` if the user cancels.
pub fn pick_subtitle_bytes() -> Promise<Option<(String, Vec<u8>)>> {
    let dialog =
        rfd::AsyncFileDialog::new().add_filter("Subtitle Files", &["srt", "ass", "ssa", "vtt"]);
    let pick = async move {
        let handle = dialog.pick_file().await?;
        let name = handle.file_name();
        let bytes = handle.read().await;
        Some((name, bytes))
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        Promise::spawn_thread("pick_subtitle_bytes", move || pollster::block_on(pick))
    }
    #[cfg(target_arch = "wasm32")]
    {
        Promise::spawn_async(pick)
    }
}

/// Pick a video file, returning its path rather than its bytes: the decode
/// backends stream from a path instead of reading a multi-hundred-MB file
/// into memory up front. Native only, since a browser file handle has no path.
#[cfg(not(target_arch = "wasm32"))]
pub fn pick_video_path() -> Promise<Option<std::path::PathBuf>> {
    let dialog = rfd::AsyncFileDialog::new()
        .add_filter("Video Files", &crate::video::source::VIDEO_EXTENSIONS);
    let pick = async move {
        let handle = dialog.pick_file().await?;
        Some(handle.path().to_path_buf())
    };
    Promise::spawn_thread("pick_video_path", move || pollster::block_on(pick))
}

/// Pick an audio file -- or a video container to pull an audio track out of
/// (`--audio-track` selects which one) -- returning its path. Path, not
/// bytes, so the re-openable `AudioSource` can decode a song twice (once for
/// the normalisation peak, once to emit) without holding it all in memory.
#[cfg(not(target_arch = "wasm32"))]
pub fn pick_audio_path() -> Promise<Option<std::path::PathBuf>> {
    let dialog = rfd::AsyncFileDialog::new()
        .add_filter("Audio Files", &crate::audio::source::AUDIO_EXTENSIONS)
        .add_filter("Video Files", &crate::video::source::VIDEO_EXTENSIONS);
    let pick = async move {
        let handle = dialog.pick_file().await?;
        Some(handle.path().to_path_buf())
    };
    Promise::spawn_thread("pick_audio_path", move || pollster::block_on(pick))
}

/// Small square thumbnail for a picked image, served via egui's bytes loader.
pub fn thumb(ui: &mut egui::Ui, img: &PickedImage) {
    let uri = format!("bytes://thumb/{}", img.name);
    if ui.ctx().try_load_bytes(&uri).is_err() {
        let mut buf = std::io::Cursor::new(Vec::new());
        let _ = image::DynamicImage::ImageRgba8((*img.image).clone())
            .thumbnail(32, 32)
            .write_to(&mut buf, image::ImageFormat::Png);
        ui.ctx().include_bytes(uri.clone(), buf.into_inner());
    }
    ui.add(
        egui::Image::new(egui::ImageSource::Uri(uri.into()))
            .fit_to_exact_size(egui::vec2(32.0, 32.0))
            .maintain_aspect_ratio(false),
    );
}

/// Build the generator inputs from picked images.
pub fn maps_from_images(
    options: &GenOptions,
    heightmaps: &[PickedImage],
    colormap: Option<&PickedImage>,
) -> Result<MapPair, String> {
    let colormap_img = colormap
        .or(heightmaps.first())
        .ok_or_else(|| "no images selected".to_string())?;
    let colormap = ColormapPNG::from_image((*colormap_img.image).clone(), options.lrgb);

    let heightmap: Box<dyn Heightmap> = if options.img {
        Box::new(HeightmapFlat::new(colormap.size()).unwrap())
    } else {
        Box::new(HeightmapPNG::from_images(
            heightmaps.iter().map(|p| (*p.image).clone()).collect(),
            options.hdmap,
        )?)
    };

    Ok((heightmap, Box::new(colormap)))
}

/// Deliver a generated save to the user: on native, write it next to the exe
/// and optionally copy the file path to the clipboard for in-game pasting;
/// on web, trigger a browser download.
pub fn deliver_save(data: Vec<u8>, out_file: &str, clipboard: bool) -> Result<(), String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::fs::write(out_file, data).map_err(|e| format!("failed to write file: {e}"))?;
        if clipboard {
            copy_path_to_clipboard(out_file)?;
        }
        Ok(())
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = clipboard;
        download_bytes(out_file, &data)
    }
}

/// Why the current Save Destination cannot be written, or `None` if it can.
/// Panes use this both for the warning label and to gate the Generate button,
/// so the two can't disagree.
pub fn out_file_problem(out_file: &str) -> Option<String> {
    let lower = out_file.trim().to_lowercase();
    if lower.is_empty() {
        return Some("Save Destination is empty".to_string());
    }
    if !lower.ends_with(".brz") && !lower.ends_with(".brdb") {
        return Some("Output file must end with .brz or .brdb".to_string());
    }
    None
}

/// Whether a save already exists at the destination, i.e. whether Generate
/// will overwrite something.
pub fn out_file_exists(out_file: &str) -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        out_file_problem(out_file).is_none() && std::path::Path::new(out_file.trim()).is_file()
    }
    // A browser download never overwrites in place -- it lands in the download
    // folder under whatever name the browser picks.
    #[cfg(target_arch = "wasm32")]
    {
        let _ = out_file;
        false
    }
}

/// Draw the Save Destination's warnings as the last rows of a settings grid.
pub fn draw_out_file_warnings(ui: &mut egui::Ui, out_file: &str) {
    if let Some(problem) = out_file_problem(out_file) {
        ui.label("Warning:");
        ui.colored_label(egui::Color32::RED, problem);
        ui.end_row();
    } else if out_file_exists(out_file) {
        ui.label("Note:");
        ui.colored_label(
            egui::Color32::from_rgb(255, 200, 100),
            "A save already exists at this name and will be overwritten",
        );
        ui.end_row();
    }
}

/// The Generate gate for a bad destination: `true` when the pane has drawn a
/// refusal instead and must not offer the button. Call before any other
/// draw_submit gate so a bad destination wins regardless of input state.
pub fn refuse_bad_out_file(ui: &mut egui::Ui, out_file: &str) -> bool {
    match out_file_problem(out_file) {
        Some(problem) => {
            ui.colored_label(egui::Color32::RED, format!("Cannot render: {problem}"));
            true
        }
        None => false,
    }
}

/// Write a finished `World` to `out_file` in the container its extension
/// names, and copy the path to the clipboard if asked. The one place any pane
/// picks the output format; on the web, `.brdb` is refused rather than
/// silently downloaded as BRZ (no filesystem to write it into).
pub fn deliver_world(world: &World, out_file: &str, clipboard: bool) -> Result<(), String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::util::write_world(world, out_file)?;
        if clipboard {
            copy_path_to_clipboard(out_file)?;
        }
        Ok(())
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = clipboard;
        let lower = out_file.to_lowercase();
        if lower.ends_with(".brz") {
            let data = world
                .to_brz_vec()
                .map_err(|e| format!("failed to encode brz: {e}"))?;
            download_bytes(out_file, &data)
        } else if lower.ends_with(".brdb") {
            Err("only .brz output is supported on the web".to_string())
        } else {
            Err("output file must end with .brz or .brdb".to_string())
        }
    }
}

/// Delivers a finished `World` as a save file unless `progress` reports the
/// render that produced it was cancelled, in which case nothing is written
/// and this still returns `Ok(())` -- a cancelled render must never surface
/// as a logged error.
pub fn deliver_world_unless_cancelled(
    world: World,
    progress: &dyn Progress,
    out_file: &str,
    out_clipboard: bool,
) -> Result<(), String> {
    if progress.is_cancelled() {
        log::info!("Render cancelled -- no save written");
        return Ok(());
    }
    log::info!("Writing Save to {out_file}");
    deliver_world(&world, out_file, out_clipboard)?;
    log::info!("Done!");
    Ok(())
}

/// Trigger a browser download of the given bytes.
#[cfg(target_arch = "wasm32")]
fn download_bytes(name: &str, data: &[u8]) -> Result<(), String> {
    use wasm_bindgen::JsCast;
    let err = |e: wasm_bindgen::JsValue| format!("download failed: {e:?}");

    let array = js_sys::Array::new();
    array.push(&js_sys::Uint8Array::from(data).buffer());
    let blob = web_sys::Blob::new_with_buffer_source_sequence(&array).map_err(err)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob).map_err(err)?;

    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or("no document")?;
    let a: web_sys::HtmlAnchorElement = document
        .create_element("a")
        .map_err(err)?
        .dyn_into()
        .map_err(|_| "anchor cast failed".to_string())?;
    a.set_href(&url);
    a.set_download(name);
    a.click();
    let _ = web_sys::Url::revoke_object_url(&url);
    Ok(())
}

/// Copy the output file's absolute path to the OS clipboard as a file list so
/// it can be pasted directly into Brickadia.
#[cfg(not(target_arch = "wasm32"))]
pub fn copy_path_to_clipboard(out_file: &str) -> Result<(), String> {
    let mut full_path = std::path::Path::new(out_file)
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from(out_file))
        .to_string_lossy()
        .to_string();

    // lowercase the first letter
    full_path.get_mut(0..1).map(|s| s.make_ascii_lowercase());

    #[cfg(target_os = "windows")]
    {
        clipboard_win::raw::open().map_err(|e| format!("failed to open clipboard: {e}"))?;
        let set = clipboard_win::raw::set_file_list(&[full_path.clone()])
            .map_err(|e| format!("failed to set clipboard: {e}"));
        let close =
            clipboard_win::raw::close().map_err(|e| format!("failed to close clipboard: {e}"));
        set?;
        close?;
        log::info!("Wrote path {full_path} to clipboard");
    }

    #[cfg(not(target_os = "windows"))]
    {
        log::info!("Clipboard file path support is only available on Windows");
        log::info!("File saved to: {}", full_path);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Shared render-progress channel plumbing.
//
// The Audio and Video panes both spawn a background render worker (a
// synchronous call on wasm, a real thread on native) and need to show its
// progress in the UI: a labelled bar (or an indeterminate one while the total
// is unknown), a Cancel button backed by a shared `Arc<AtomicBool>`, and a
// channel carrying `Progress` events from the worker back to the UI thread
// that owns egui. That was ~150 lines of near-identical code in each pane;
// the shared core -- the message shape, the `Progress` impl that forwards it
// over the channel, and the two draw calls -- lives here once instead.
// `Extra` is room for whatever a pane needs on top of that core (the video
// pane's throttled frame preview); a pane with nothing to add (the audio
// pane, which has no picture to show) uses `std::convert::Infallible`, so its
// own match on the result never needs an arm for it. Works on every target,
// including wasm -- unlike the ffmpeg-download plumbing below, both panes'
// render worker (and so this reporting) runs there too.
// ---------------------------------------------------------------------------

/// What a render worker reports to the UI over its progress channel: a named
/// phase with an optional total (`Begin`), cumulative position within it
/// (`Tick`), and a terminal `Finish` -- the shared core every pane's
/// [`ChannelProgress`] sends. `Extra` carries whatever a specific pane needs
/// on top of that; see [`ChannelProgress::send_extra`].
#[derive(Clone, Debug)]
pub enum RenderMsg<Extra> {
    Begin { label: String, total: Option<u64> },
    Tick(u64),
    Finish,
    Extra(Extra),
}

impl<Extra> RenderMsg<Extra> {
    /// Apply a `Begin`/`Tick`/`Finish` event to a pane's progress-bar state,
    /// growing an under-estimated total to meet an overrun tick rather than
    /// letting the bar draw past 100% (see
    /// [`crate::progress::reconcile_total`]). Returns the payload back out on
    /// `Extra` rather than handling it here -- what a pane's `Extra` means
    /// (the video pane's frame preview) is pane-specific, so the caller's
    /// `poll_generate` handles it after this returns.
    pub fn apply_core(
        self,
        label: &mut String,
        pos: &mut u64,
        total: &mut Option<u64>,
    ) -> Option<Extra> {
        match self {
            RenderMsg::Begin { label: l, total: t } => {
                *label = l;
                *total = t;
                *pos = 0;
                None
            }
            RenderMsg::Tick(n) => {
                let (p, t) = crate::progress::reconcile_total(*total, n);
                *pos = p;
                *total = t;
                None
            }
            RenderMsg::Finish => None,
            RenderMsg::Extra(e) => Some(e),
        }
    }
}

/// Reports progress from a render worker thread to the UI over a channel --
/// the shared core of the audio and video panes' [`Progress`] implementations.
/// Send failures are ignored on purpose: a closed channel means the UI went
/// away, which must never abort a render that is otherwise fine. A pane
/// needing more than this core (the video pane's throttled frame preview)
/// wraps this rather than reimplementing `begin`/`tick`/`finish`/
/// `is_cancelled` itself.
pub struct ChannelProgress<Extra> {
    tx: std::sync::mpsc::Sender<RenderMsg<Extra>>,
    /// Set by the UI thread's Cancel button, read once per frame by
    /// `is_cancelled`.
    cancel: Arc<AtomicBool>,
}

impl<Extra> ChannelProgress<Extra> {
    pub fn new(tx: std::sync::mpsc::Sender<RenderMsg<Extra>>, cancel: Arc<AtomicBool>) -> Self {
        Self { tx, cancel }
    }

    /// Send a pane-specific payload -- the video pane's throttled frame
    /// preview. Ignored on a closed channel, same as every other send here.
    pub fn send_extra(&self, extra: Extra) {
        let _ = self.tx.send(RenderMsg::Extra(extra));
    }
}

impl<Extra> Progress for ChannelProgress<Extra> {
    fn begin(&mut self, label: &str, total: Option<u64>) {
        let _ = self
            .tx
            .send(RenderMsg::Begin { label: label.to_string(), total });
    }
    fn tick(&mut self, n: u64) {
        let _ = self.tx.send(RenderMsg::Tick(n));
    }
    fn finish(&mut self) {
        let _ = self.tx.send(RenderMsg::Finish);
    }

    /// Backed by the same `Arc<AtomicBool>` the UI thread's Cancel button
    /// sets. `Relaxed` is enough: a single flag with no other state it must
    /// stay ordered with.
    fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// Draw an in-flight render's progress bar: a real bar once a total is
/// known, else an animated indeterminate one -- never a fabricated fraction
/// that would reach 100% and keep going. Requests a repaint every call,
/// since egui only repaints on input by default and a worker thread
/// advancing `pos`/`total` on its own has nothing else to wake the event
/// loop.
pub fn draw_progress_bar(ui: &mut egui::Ui, label: &str, pos: u64, total: Option<u64>) {
    ui.ctx().request_repaint();
    match total {
        Some(total) if total > 0 => {
            let frac = pos as f32 / total as f32;
            ui.add(egui::ProgressBar::new(frac).text(format!("{label} {pos}/{total}")));
        }
        // Unknown total (or a degenerate `Some(0)`): an animated
        // indeterminate bar, never a fabricated fraction.
        _ => {
            ui.add(egui::ProgressBar::new(0.0).animate(true).text(label.to_string()));
        }
    }
}

/// Draw the Cancel button for an in-flight render's cancel flag: disabled
/// and relabelled "Cancelling..." once the flag is set, so a second click
/// can't re-fire it. Native only -- on wasm a render runs synchronously to
/// completion before this could ever be drawn, so both panes' cancel flag
/// field is `#[cfg]`'d out there entirely.
#[cfg(not(target_arch = "wasm32"))]
pub fn draw_cancel_button(ui: &mut egui::Ui, flag: &Arc<AtomicBool>) {
    ui.add_space(4.0);
    let cancelling = flag.load(Ordering::Relaxed);
    ui.add_enabled_ui(!cancelling, |ui| {
        if ui
            .add(egui::Button::new(if cancelling { "Cancelling..." } else { "Cancel" }))
            .clicked()
        {
            flag.store(true, Ordering::Relaxed);
        }
    });
}

#[cfg(test)]
mod render_progress_tests {
    use super::*;

    #[test]
    fn begin_resets_position_and_records_label_and_total() {
        let mut label = "stale".to_string();
        let mut pos = 41;
        let mut total = Some(9);
        let msg: RenderMsg<std::convert::Infallible> =
            RenderMsg::Begin { label: "analyzing".to_string(), total: Some(100) };
        let extra = msg.apply_core(&mut label, &mut pos, &mut total);
        assert!(extra.is_none());
        assert_eq!(label, "analyzing");
        assert_eq!(pos, 0);
        assert_eq!(total, Some(100));
    }

    #[test]
    fn tick_past_an_estimated_total_grows_it_rather_than_overflowing() {
        let mut label = String::new();
        let mut pos = 0;
        let mut total = Some(100);
        let msg: RenderMsg<std::convert::Infallible> = RenderMsg::Tick(137);
        msg.apply_core(&mut label, &mut pos, &mut total);
        assert_eq!(pos, 137);
        assert_eq!(total, Some(137), "an under-estimate must grow to meet the tick");
    }

    #[test]
    fn extra_passes_through_untouched_and_does_not_move_the_bar() {
        let mut label = "phase".to_string();
        let mut pos = 3;
        let mut total = Some(10);
        let msg: RenderMsg<u32> = RenderMsg::Extra(7);
        let extra = msg.apply_core(&mut label, &mut pos, &mut total);
        assert_eq!(extra, Some(7));
        assert_eq!(label, "phase");
        assert_eq!(pos, 3);
        assert_eq!(total, Some(10));
    }

    #[test]
    fn channel_progress_is_cancelled_reflects_the_shared_flag() {
        let (tx, _rx) = std::sync::mpsc::channel::<RenderMsg<std::convert::Infallible>>();
        let flag = Arc::new(AtomicBool::new(false));
        let progress = ChannelProgress::new(tx, Arc::clone(&flag));

        assert!(!progress.is_cancelled(), "a fresh flag must not read as cancelled");
        flag.store(true, Ordering::Relaxed);
        assert!(
            progress.is_cancelled(),
            "setting the shared flag (as the Cancel button does) must be visible immediately"
        );
    }
}

// ---------------------------------------------------------------------------
// Shared ffmpeg-download modal + progress plumbing.
//
// The Video and Audio panes both need to fetch ffmpeg on demand, show a real
// progress bar while it downloads (the gyan.dev Windows mirror is slow enough
// that a spinner reads as a hang), let the user abandon the wait, and surface
// the OUTCOME in the modal rather than only in the log. That was ~150 lines of
// near-identical code in each pane; it lives here once instead, so a fix to one
// cannot drift from the other. Native only: `video::ffmpeg` (and so the whole
// download path) does not exist on wasm.
// ---------------------------------------------------------------------------

/// Which stage an in-flight ffmpeg download has reached.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum DownloadPhase {
    /// Connecting -- the fetch has begun but no bytes have arrived.
    Connecting,
    /// Bytes are arriving.
    Downloading,
    /// The archive is downloaded and being unpacked.
    Unpacking,
    /// ffmpeg is installed.
    Done,
}

/// A `Copy` snapshot of the download's progress for one render frame -- what
/// the modal body reads, taken under the state's lock so the UI thread never
/// holds it while drawing.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct DownloadView {
    phase: DownloadPhase,
    done: u64,
    total: u64,
    /// How long since the most recent byte arrived, or `None` before the first
    /// one. Drives the stall message and nothing else -- see
    /// [`crate::video::ffmpeg::download_is_stalled`].
    stalled_for: Option<std::time::Duration>,
}

/// Progress state shared between the background download thread (which writes
/// via [`FfmpegDownloadState::record`]) and the UI thread (which reads via
/// [`FfmpegDownloadState::view`] each frame and writes the terminal result via
/// [`FfmpegDownloadState::set_outcome`]).
///
/// All state is behind one `Mutex`: the writes are tiny and infrequent (once
/// per socket read at most), so lock contention is a non-issue and one mutex is
/// simpler to reason about than a fan of atomics. The download thread holds one
/// `Arc` and the UI another; when the UI abandons the wait it drops its `Arc`,
/// and the orphaned thread's clone keeps the state alive until that thread ends
/// -- nothing reads it by then, so its writes are harmless.
#[cfg(not(target_arch = "wasm32"))]
struct FfmpegDownloadState {
    inner: std::sync::Mutex<DownloadSnapshot>,
}

#[cfg(not(target_arch = "wasm32"))]
struct DownloadSnapshot {
    phase: DownloadPhase,
    done: u64,
    total: u64,
    last_byte: Option<std::time::Instant>,
    /// The terminal result, once the worker has finished. Written on the UI
    /// thread from the resolved `Promise`, never by the download thread.
    outcome: Option<Result<(), String>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl FfmpegDownloadState {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: std::sync::Mutex::new(DownloadSnapshot {
                phase: DownloadPhase::Connecting,
                done: 0,
                total: 0,
                last_byte: None,
                outcome: None,
            }),
        })
    }

    /// The download thread's sink. Records one progress event; on a
    /// `Downloading` event it also stamps the wall-clock time, which is what
    /// the stall check reads back.
    fn record(&self, p: crate::video::ffmpeg::FfmpegDownloadProgress) {
        use crate::video::ffmpeg::FfmpegDownloadProgress as P;
        let mut s = self.inner.lock().unwrap();
        match p {
            P::Starting => s.phase = DownloadPhase::Connecting,
            P::Downloading { done, total } => {
                s.phase = DownloadPhase::Downloading;
                s.done = done;
                s.total = total;
                s.last_byte = Some(std::time::Instant::now());
            }
            P::Unpacking => s.phase = DownloadPhase::Unpacking,
            P::Done => s.phase = DownloadPhase::Done,
        }
    }

    fn view(&self) -> DownloadView {
        let s = self.inner.lock().unwrap();
        DownloadView {
            phase: s.phase,
            done: s.done,
            total: s.total,
            stalled_for: s.last_byte.map(|t| t.elapsed()),
        }
    }

    fn set_outcome(&self, result: Result<(), String>) {
        self.inner.lock().unwrap().outcome = Some(result);
    }

    fn outcome(&self) -> Option<Result<(), String>> {
        self.inner.lock().unwrap().outcome.clone()
    }
}

/// Draw the body of the download modal for one frame from a progress snapshot:
/// a real progress bar with "X.X / Y.Y MB" and a percent, plus the stall notice
/// when the connection has gone quiet.
#[cfg(not(target_arch = "wasm32"))]
fn draw_ffmpeg_download_body(ui: &mut egui::Ui, view: DownloadView) {
    use crate::video::ffmpeg::{download_fraction, download_is_stalled};
    match view.phase {
        DownloadPhase::Connecting => {
            ui.add(egui::ProgressBar::new(0.0).animate(true).text("connecting..."));
        }
        DownloadPhase::Downloading => {
            let done_mb = view.done as f64 / 1_000_000.0;
            match download_fraction(view.done, view.total) {
                Some(frac) => {
                    let total_mb = view.total as f64 / 1_000_000.0;
                    ui.add(egui::ProgressBar::new(frac).text(format!(
                        "{done_mb:.1} / {total_mb:.1} MB ({:.0}%)",
                        frac * 100.0
                    )));
                }
                // No Content-Length: an honest indeterminate bar with the bytes
                // so far, never a fabricated percentage.
                None => {
                    ui.add(
                        egui::ProgressBar::new(0.0)
                            .animate(true)
                            .text(format!("{done_mb:.1} MB downloaded")),
                    );
                }
            }
        }
        DownloadPhase::Unpacking => {
            ui.add(egui::ProgressBar::new(1.0).text("unpacking..."));
        }
        DownloadPhase::Done => {
            ui.add(egui::ProgressBar::new(1.0).text("finishing..."));
        }
    }

    // Stall visibility, driven purely off the timestamp the callback records --
    // this aborts nothing, it only tells the user a dead gyan.dev connection
    // apart from a merely slow one.
    if let Some(since) = view.stalled_for {
        if download_is_stalled(since) {
            ui.add_space(4.0);
            ui.colored_label(
                egui::Color32::from_rgb(255, 180, 80),
                format!(
                    "Download appears stalled -- no data for {}s. Check your connection, or \
                     install ffmpeg manually and restart.",
                    since.as_secs()
                ),
            );
        }
    }
}

/// The shared ffmpeg download-consent + download modal, bundling the three
/// fields each pane used to carry separately (the pending consent request, the
/// in-flight download `Promise`, and the progress cell) behind one type with
/// the poll/draw logic attached.
///
/// A pane holds one of these, sets it going with [`FfmpegModal::request`] when
/// a source needs an ffmpeg it lacks, and calls [`FfmpegModal::poll`] and
/// [`FfmpegModal::draw`] once per frame. [`FfmpegModal::is_open`] is the gate
/// the pane's Generate button reads.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
pub struct FfmpegModal {
    /// The path awaiting a decision, and the marker that the modal is open at
    /// all -- `Some` from the moment a pane calls [`FfmpegModal::request`]
    /// until the user dismisses the modal (declining, abandoning, or
    /// acknowledging the outcome). The path itself is not read (the pane
    /// re-opens the file on the next Generate click rather than auto-resuming,
    /// unchanged from before), but keeping it documents which file asked.
    consent: Option<std::path::PathBuf>,
    /// The in-flight download worker, once "Download" is clicked. `Some` only
    /// while the network fetch is actually running; cleared when it resolves
    /// (its outcome moves into `progress`) or when the wait is abandoned.
    download: Option<Promise<Result<(), String>>>,
    /// The shared progress cell, `Some` from the moment a download starts. It
    /// outlives `download` so the resolved outcome can be shown in the modal.
    progress: Option<Arc<FfmpegDownloadState>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl FfmpegModal {
    /// Whether the modal is showing -- the gate a pane's Generate button and
    /// its "waiting on the ffmpeg prompt" notice both read.
    pub fn is_open(&self) -> bool {
        self.consent.is_some()
    }

    /// Ask for consent to download ffmpeg for `path`. Opens the modal on its
    /// consent prompt; the download does not start until the user clicks
    /// "Download". Resets any prior download/progress state.
    pub fn request(&mut self, path: std::path::PathBuf) {
        self.consent = Some(path);
        self.download = None;
        self.progress = None;
    }

    /// Poll the in-flight download, if any. On resolution the outcome is stored
    /// in the progress cell (so [`FfmpegModal::draw`] can show "installed" /
    /// "failed") and also logged, exactly as before; the modal stays open on
    /// its outcome view until the user dismisses it. While the download is
    /// still running this requests a repaint, since nothing else wakes the
    /// event loop while it runs on its own thread.
    pub fn poll(&mut self, ctx: &egui::Context) {
        let Some(promise) = self.download.take() else {
            return;
        };
        match promise.try_take() {
            Ok(result) => {
                match &result {
                    Ok(()) => log::info!("ffmpeg installed -- click Generate again to continue"),
                    Err(e) => log::error!("{e}"),
                }
                match &self.progress {
                    Some(state) => state.set_outcome(result),
                    // No progress cell should be impossible here (a download
                    // only starts alongside one), but if it ever were, close
                    // the modal rather than trap the user with no outcome view.
                    None => self.consent = None,
                }
            }
            Err(promise) => {
                self.download = Some(promise);
                ctx.request_repaint();
            }
        }
    }

    /// Draw the modal for one frame. `prompt` is the consent question body (the
    /// panes word it slightly differently and interpolate the download URL);
    /// `declined_log` is the info-level line logged when the user declines or
    /// abandons.
    pub fn draw(
        &mut self,
        ctx: &egui::Context,
        id_prefix: &str,
        prompt: &str,
        declined_log: &str,
    ) {
        if self.consent.is_none() {
            return;
        }

        // A download has been started (or has finished): show progress, or the
        // terminal outcome once one is in.
        if let Some(state) = &self.progress {
            let outcome = state.outcome();
            let mut dismiss = false;
            let mut abandon = false;
            egui::Modal::new(egui::Id::new(format!("{id_prefix}_ffmpeg_download_modal"))).show(
                ctx,
                |ui| {
                    ui.set_max_width(420.0);
                    match &outcome {
                        Some(Ok(())) => {
                            ui.heading("ffmpeg installed");
                            ui.label(
                                "ffmpeg was downloaded and installed. Click Generate again to \
                                 continue.",
                            );
                            ui.add_space(8.0);
                            if ui.button("Close").clicked() {
                                dismiss = true;
                            }
                        }
                        Some(Err(e)) => {
                            ui.heading("ffmpeg download failed");
                            ui.colored_label(egui::Color32::from_rgb(255, 120, 120), e);
                            ui.add_space(8.0);
                            ui.label(
                                "You can try again, or install ffmpeg manually and make sure it \
                                 is on PATH, then restart.",
                            );
                            ui.add_space(8.0);
                            if ui.button("Close").clicked() {
                                dismiss = true;
                            }
                        }
                        None => {
                            ui.heading("Downloading ffmpeg...");
                            draw_ffmpeg_download_body(ui, state.view());
                            ui.add_space(8.0);
                            // Cancel ABANDONS THE WAIT. The sidecar's progress
                            // callback returns `()` and cannot hard-abort the
                            // in-flight ureq read, so this does NOT interrupt
                            // the HTTP stream: it closes the modal and drops our
                            // Promise handle, and the orphaned download thread
                            // finishes or dies with the process. The value is
                            // that the user is no longer trapped watching a bar
                            // that may be crawling on a rate-limited mirror.
                            if ui.button("Cancel").clicked() {
                                abandon = true;
                            }
                        }
                    }
                },
            );
            if dismiss {
                self.consent = None;
                self.progress = None;
            }
            if abandon {
                // Drop both handles and close: the thread keeps running,
                // unwatched, on its own `Arc` clone of the state (see
                // `FfmpegDownloadState`'s doc).
                self.download = None;
                self.progress = None;
                self.consent = None;
                log::info!("ffmpeg download abandoned; {declined_log}");
            }
            return;
        }

        // No download yet: the consent prompt.
        let mut start = false;
        let mut decline = false;
        egui::Modal::new(egui::Id::new(format!("{id_prefix}_ffmpeg_consent_modal"))).show(
            ctx,
            |ui| {
                ui.set_max_width(420.0);
                ui.heading("Download ffmpeg?");
                ui.label(prompt);
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Download").clicked() {
                        start = true;
                    }
                    if ui.button("Cancel").clicked() {
                        decline = true;
                    }
                });
            },
        );
        if start {
            let state = FfmpegDownloadState::new();
            let sink_state = Arc::clone(&state);
            self.progress = Some(state);
            self.download = Some(Promise::spawn_thread("ffmpeg_download", move || {
                crate::video::ffmpeg::ensure_ffmpeg_with_progress(
                    crate::video::ffmpeg::DownloadConsent::Always,
                    move |p| sink_state.record(p),
                )
            }));
        }
        if decline {
            log::info!("ffmpeg download declined; {declined_log}");
            self.consent = None;
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::video::ffmpeg::FfmpegDownloadProgress;

    /// The modal's shared cell must reflect the events the download thread
    /// records: a `Downloading` event moves `done`/`total` and stamps a
    /// last-byte time (so the freshly-recorded byte is NOT yet stalled), and a
    /// later phase event advances the phase. This is the GUI half of the
    /// plumbing proof -- the sink writes, the view reads back what was written.
    #[test]
    fn the_progress_cell_reflects_recorded_events() {
        let state = FfmpegDownloadState::new();

        // Before any byte: no last-byte stamp, so nothing to call stalled.
        assert!(state.view().stalled_for.is_none());

        state.record(FfmpegDownloadProgress::Downloading { done: 40, total: 100 });
        let v = state.view();
        assert_eq!(v.done, 40);
        assert_eq!(v.total, 100);
        let since = v.stalled_for.expect("a recorded byte stamps a time");
        assert!(
            !crate::video::ffmpeg::download_is_stalled(since),
            "a byte recorded this instant must not read as stalled"
        );

        state.record(FfmpegDownloadProgress::Unpacking);
        assert!(
            matches!(state.view().phase, DownloadPhase::Unpacking),
            "a later event must advance the phase the modal shows"
        );
    }

    /// The outcome the UI thread writes on the resolved `Promise` must be the
    /// one the modal reads back, so it can show "installed" / "failed" rather
    /// than only logging it.
    #[test]
    fn the_outcome_written_on_the_ui_thread_is_read_back() {
        let ok = FfmpegDownloadState::new();
        assert!(ok.outcome().is_none(), "no outcome until the worker resolves");
        ok.set_outcome(Ok(()));
        assert!(matches!(ok.outcome(), Some(Ok(()))));

        let err = FfmpegDownloadState::new();
        err.set_outcome(Err("mirror timed out".to_string()));
        assert_eq!(err.outcome(), Some(Err("mirror timed out".to_string())));
    }

    /// A fresh modal is closed, and `request` opens it without starting a
    /// download (the network fetch only begins on the user's "Download"
    /// click) -- the consent step the task's constraints say not to change.
    #[test]
    fn request_opens_the_modal_on_the_consent_prompt() {
        let mut modal = FfmpegModal::default();
        assert!(!modal.is_open());
        modal.request(std::path::PathBuf::from("clip.mkv"));
        assert!(modal.is_open(), "request opens the modal");
        assert!(modal.download.is_none(), "but starts no download until Download is clicked");
        assert!(modal.progress.is_none());
    }
}
