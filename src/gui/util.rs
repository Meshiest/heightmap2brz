use std::sync::Arc;

use crate::{
    map::{Colormap, ColormapPNG, Heightmap, HeightmapFlat, HeightmapPNG},
    util::GenOptions,
};
use image::RgbaImage;
use poll_promise::Promise;

type MapPair = (Box<dyn Heightmap>, Box<dyn Colormap>);

/// An image the user picked, already decoded — works identically on native
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
