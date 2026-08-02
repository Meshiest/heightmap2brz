use std::sync::Arc;

use crate::{
    map::{Colormap, ColormapPNG, Heightmap, HeightmapFlat, HeightmapPNG},
    progress::Progress,
    util::GenOptions,
};
use brdb::World;
use image::RgbaImage;
use poll_promise::Promise;

type MapPair = (Box<dyn Heightmap>, Box<dyn Colormap>);

/// A two-column settings grid whose cells WRAP instead of widening the pane.
///
/// `Grid::max_col_width` is doing two separate jobs here, and the second one is
/// the reason this helper exists at all:
///
/// 1. It caps how far one cell can push the pane out.
/// 2. It is the ONLY switch that turns text wrapping on inside grid cells.
///    `GridLayout::wrap_text()` is literally `max_cell_size.x.is_finite()`, and
///    `Ui::wrap_mode` consults it for every cell -- so a grid *without* a max
///    column width puts every label in `TextWrapMode::Extend`, and one long
///    annotation ("(Pitch-Per-Speaker only -- ...)") drags a horizontal
///    scrollbar across the entire window.
///
/// `num_columns(2)` is what keeps the cap from wasting space: it marks column 1
/// as the last one, which egui then sizes to the width actually left over
/// rather than to the same bound as column 0.
///
/// **Do not reach for `horizontal_wrapped` inside one of these cells.** A
/// wrapping horizontal layout reports a height short of what it drew, and the
/// grid then places the next row on top of it -- measured at every window
/// width, in both panes, on every row that tried it. Long text inside a cell
/// goes through [`note`], which wraps at the widget rather than at the layout
/// and so allocates the rect it actually used; text too long for what is left
/// beside a control gets a row of its own instead.
pub fn settings_grid(ui: &egui::Ui, id: &str) -> egui::Grid {
    // Generous, because the *label* column is always short: the cap exists to
    // stop the long annotations in column 1 from running off the edge, and
    // column 1 is sized from the real remaining width anyway (see above).
    let max_col_width = (ui.available_width() - 60.0).max(160.0);
    egui::Grid::new(id)
        .striped(true)
        .num_columns(2)
        .spacing([40.0, 4.0])
        .min_col_width(LABEL_COLUMN_WIDTH)
        .max_col_width(max_col_width)
}

/// Width reserved for the label column of every [`settings_grid`].
///
/// **This is what makes separate sections line up with each other.** Each
/// accordion section owns its own `egui::Grid` (it has to -- a grid cannot span
/// a collapsing body), and a grid sizes column 1 to ITS OWN widest label. Left
/// alone, "Analysis" and "Levels" put their controls at two different x
/// positions and the pane reads as several unrelated tables stacked up. Pinning
/// a floor that no single label reaches makes column 1 the same width in every
/// section of both panes, so column 2 starts at one x throughout.
///
/// Chosen to clear the longest label in either pane ("Save Destination" at
/// about 97 points, "Alpha Threshold") while still leaving the widest ROW --
/// the video pane's two size sliders -- room inside a 520 point window. Both
/// ends of that are tested: `tests/gui_accordion.rs` asserts no label outgrows
/// this, and separately that nothing overflows the pane at 520. It is in
/// POINTS, so it scales with the user's zoom exactly as the text does.
///
/// A label longer than this is not a disaster -- that one grid's column simply
/// grows, and only that section falls out of line -- but it will be a test
/// failure naming the label rather than something discovered by eye.
pub const LABEL_COLUMN_WIDTH: f32 = 112.0;

/// Pin a pane to the width it was handed, and make text wrap to it by default.
///
/// Called once at the top of a pane's `draw`. Two guarantees, both of which the
/// panes depend on and neither of which a pane can establish row by row:
///
/// - `set_max_width` fixes the pane's `max_rect` to the width actually
///   available, so nothing downstream can lay out against an unbounded region
///   and get clipped at the window edge instead of wrapping.
/// - `style.wrap_mode` is the FIRST thing `Ui::wrap_mode` consults, ahead of
///   the grid and layout fallbacks that otherwise resolve to
///   `TextWrapMode::Extend` inside a `horizontal` or an unbounded grid cell.
///   Setting it here means a label wraps unless a widget deliberately opts out.
pub fn bound_pane_width(ui: &mut egui::Ui) {
    ui.set_max_width(ui.available_width());
    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
}

/// Wrap a long label so it folds inside its parent instead of running past the
/// edge and being clipped.
///
/// `Ui::wrap_mode` resolves to `Extend` in a plain `ui.horizontal(..)`, which is
/// what every "(Pitch-Per-Speaker only -- ...)" annotation sits in. Asking for
/// wrapping on the widget itself is independent of the parent's layout, and --
/// unlike `horizontal_wrapped` -- a `Label` allocates its own exact (already
/// wrapped) rect, so an `egui::Grid` measures the row height correctly.
pub fn note(ui: &mut egui::Ui, text: impl Into<String>) -> egui::Response {
    ui.add(egui::Label::new(text.into()).wrap())
}

/// A small rounded badge for ONE value in an accordion header.
///
/// Every colour comes out of the current visuals rather than being hard-coded,
/// so a chip follows a light/dark theme switch like the rest of the widgets.
/// The foreground is `strong_text_color` rather than the weak/label colour: a
/// chip is read AT A GLANCE against a filled background, and the muted colour
/// that suits body text washes out on top of one.
///
/// The vertical inner margin is deliberately 0 -- with `RichText::small` the
/// glyphs are already shorter than a normal row, so any vertical padding at all
/// makes the header taller than a plain text header would be.
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

/// One collapsible block of advanced settings, headed by a name and a row of
/// value [`chip`]s.
///
/// **Built on `CollapsingState` rather than `CollapsingHeader` for two
/// reasons.** A `CollapsingHeader` takes only text, and it lays that text out
/// with `TextWrapMode::Extend` hard-coded -- so a summary long enough to be
/// worth reading pushes the pane wider than the window. A custom header can
/// hold real widgets and put them in a `horizontal_wrapped`, which folds the
/// chips onto a second line instead.
///
/// `id_salt` is NOT optional. The chips change as the user drags a slider, and
/// a header that derived its id from its own contents would mint a new id on
/// every change and forget whether it was open.
///
/// `open_by_default` only decides the FIRST frame the section is ever drawn;
/// after that egui remembers what the user did with it. It is set from "does
/// this section hold a non-default value", so a pane that starts on tuned
/// settings does not hide them. The chips are what carry that information
/// afterwards, which is why they are values and not a bare noun.
pub fn section(
    ui: &mut egui::Ui,
    id_salt: &str,
    title: &str,
    chips: &[String],
    open_by_default: bool,
    body: impl FnOnce(&mut egui::Ui),
) {
    let id = ui.make_persistent_id(id_salt);
    // The WHOLE header row toggles, not just the little triangle. Three things
    // have to be true at once for that, and each was verified by driving real
    // pointer events at the title, at a chip, at the empty space to their
    // right, and at the arrow -- opening AND closing on every one:
    //
    // 1. `toggle()`, not `set_open(true)`. Anything that only ever opens gives
    //    a header that will not collapse again.
    // 2. **The title and the chips must not sense clicks.** `egui`'s
    //    `Interaction::selectable_labels` defaults to ON, so a plain
    //    `ui.label` is a click-sensing widget (for text selection), and it wins
    //    the hit-test over the row underneath it -- the row reported
    //    `contains_pointer: true` with `hovered: false` and never fired. They
    //    are drawn with `.selectable(false)` so clicks fall through to the row,
    //    which is also why the click sense lives on the ROW and not on the
    //    title: one clickable widget, not two firing on the same press.
    // 3. The row has to REACH the empty space. A row is only as wide as its
    //    content (a stock `CollapsingHeader` has the same limitation), so a
    //    click to the right of the last chip would otherwise hit nothing. The
    //    fix widens the INTERACTION rect only -- see below for why it must not
    //    be an allocation.
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
        // Point 3. **Interaction only: do NOT allocate space to widen this.**
        // A zero-height `allocate_exact_size(available_width)` looks free and
        // is not -- this is a `main_wrap` layout, an exact fit is allowed to
        // spill onto the next line, and a wrapped zero-height item still costs
        // a full `item_spacing.y`. That put a blank line under every collapsed
        // header. Overriding the hit rect instead leaves the layout untouched
        // by construction: no widget is added, so there is no spacing to pay.
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

/// Pick a single file's raw bytes, without any image decode.
///
/// `pick_images` decodes eagerly via `image::load_from_memory(...).to_rgba8()`,
/// which keeps only the first frame of a GIF/APNG/animated-WebP and throws the
/// original bytes away. `video::source::decode_animated` needs every byte to
/// recover every frame, so this hands them over untouched instead.
///
/// `None` if the user cancels the dialog. Poll the returned promise each
/// frame, same as [`pick_images`].
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

/// Pick a subtitle file's raw bytes and name.
///
/// BYTES, not a path, and so available on the web too: a subtitle file is
/// kilobytes and is parsed in one pass, with nothing to stream -- the opposite
/// of [`pick_video_path`]'s case. The NAME comes back alongside them because
/// its extension is what picks the parser (`subs::parse_auto`), and a browser
/// file handle has no path to recover it from afterwards.
///
/// `None` if the user cancels the dialog.
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

/// Pick a video file, returning its PATH rather than its bytes.
///
/// Deliberately not `pick_animated_bytes`'s shape: that reads the whole file
/// into memory, and reading a 650 MB mp4 up front is exactly what the decode
/// backends exist to avoid. The backends open the path and stream from it.
///
/// The filter is only offered on targets where a backend can actually decode
/// these formats. A picker that accepts a file the tool then refuses is worse
/// than one that never offered it.
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

/// Pick an audio file -- or a video container to pull an audio track out of --
/// returning its PATH.
///
/// A path, not bytes, for [`pick_video_path`]'s reason and one more of its
/// own: `audio::symphonia_src::SymphoniaSource::open_path` and the ffmpeg
/// audio source both take a path and stream from it, and a song is decoded
/// twice (once to find the normalisation peak, once to emit), which is
/// exactly what the re-openable `AudioSource` handle exists for. Reading a
/// whole album track into a `Vec<u8>` first would defeat both.
///
/// Both filters are offered because the audio pipeline genuinely accepts
/// both: a video container carries audio tracks, and `--audio-track` selects
/// which one. Native only, same as [`pick_video_path`] -- a browser file
/// handle has no filesystem path for a decoder to open.
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
///
/// **One rule, one wording, one place -- and it is the same rule
/// [`deliver_world`] enforces.** Each of the five panes drew its own red label
/// from its own copy of the condition, and none of them stopped Generate, so the
/// surface that *showed* the warning was also the surface that wrote to the bad
/// name anyway (the CLI, which refuses it, was the strict one). Panes call this
/// for the label AND gate their Generate button on it, so the two cannot
/// disagree.
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

/// Whether a save already exists at the destination, i.e. whether Generate will
/// OVERWRITE something.
///
/// Neither surface said so before: every pane defaults to the same `out.brz`, so
/// a second render silently replaces the first, and nothing in the UI hints that
/// the file being replaced was a different build. A note rather than a
/// confirmation -- overwriting is usually what is wanted, and a modal on every
/// second render would be worse than the surprise.
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
///
/// Shared so the wording, the colours and the ORDER are identical on every pane;
/// each used to inline its own copy of the extension warning and none mentioned
/// an existing file at all.
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
/// refusal instead and must not offer the button.
///
/// Called FIRST in every pane's `draw_submit`, before "pick a file to
/// continue...": a destination that cannot be written is wrong whether or not an
/// input has been picked yet, and this is the only thing that makes the red
/// label above binding rather than advisory.
pub fn refuse_bad_out_file(ui: &mut egui::Ui, out_file: &str) -> bool {
    match out_file_problem(out_file) {
        Some(problem) => {
            ui.colored_label(egui::Color32::RED, format!("Cannot render: {problem}"));
            true
        }
        None => false,
    }
}

/// Write a finished `World` to `out_file` **in the container its extension
/// names**, and copy the path to the clipboard if asked.
///
/// **The one place any pane decides the output FORMAT**, and it exists because
/// the three newer panes did not decide it at all: `deliver_world_unless_
/// cancelled` and `TextApp::generate` called `world.to_brz_vec()`
/// unconditionally and then wrote the bytes to whatever name was in the box. All
/// five panes share one `SharedOptions::out_file`, and each of them draws its
/// "must end with .brz or .brdb" warning only when the name ends in NEITHER --
/// so `out.brdb` was explicitly endorsed by the UI and then handed BRZ content.
/// The two are genuinely different containers (`42 52 5a 00` against `53 51 4c
/// 69`, "SQLite format 3"), so the file was simply mislabelled.
///
/// On native this is [`crate::util::write_world`] -- the CLI's own writer, not a
/// second implementation of the same rule -- plus the clipboard step the GUI
/// adds. On the web there is no filesystem to write a `.brdb` into, so that name
/// is refused by name rather than silently downloaded as BRZ.
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

/// Delivers a finished `World` as a save file, UNLESS `progress` reports the
/// render that produced it was cancelled -- in which case nothing is written
/// and this still returns `Ok(())`, the same as a normal completion.
///
/// This is the one place that decides whether a cancelled render's
/// real-but-partial `World` ever reaches disk, and it must never turn a
/// cancellation into a logged error: a render worker only calls
/// `log::error!` on an `Err`, so returning `Ok(())` here is what keeps a
/// cancelled render from looking like a crash.
///
/// Shared by every pane that renders off-thread (Video, Audio) rather than
/// copied into each: this is a *policy* decision about what a cancel means,
/// and two copies of it is how one pane ends up writing a half-finished save
/// while the other does not. Split out from any one pane's `generate` closure
/// specifically so it is unit-testable on its own -- exercising the real
/// thing would mean driving an actual egui frame loop and a background
/// thread, neither of which a plain `#[test]` can do.
///
/// It decides ONLY that. The output format is [`deliver_world`]'s decision --
/// this used to hard-code BRZ, which is how a `.brdb` destination received BRZ
/// bytes.
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
