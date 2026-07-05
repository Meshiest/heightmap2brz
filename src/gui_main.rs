#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use heightmap::gui::{BrzApp, logger};
use log::info;

// run the window with glium
#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), eframe::Error> {
    logger::init().unwrap();

    eframe::run_native(
        "brztools",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_decorations(true)
                .with_drag_and_drop(true)
                .with_inner_size([600.0, 600.0])
                .with_resizable(true),
            ..Default::default()
        },
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            info!("Select a tab to get started.");
            Ok(Box::<BrzApp>::default())
        }),
    )
}

// run inside the browser canvas (built with trunk, deployed to GitHub Pages)
#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast;

    logger::init().unwrap();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .and_then(|w| w.document())
            .expect("no document");
        let canvas = document
            .get_element_by_id("brz_canvas")
            .expect("no #brz_canvas element")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("#brz_canvas is not a canvas");

        eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|cc| {
                    egui_extras::install_image_loaders(&cc.egui_ctx);
                    info!("Select a tab to get started.");
                    Ok(Box::<BrzApp>::default())
                }),
            )
            .await
            .expect("failed to start eframe");
    });
}
