//! A live gallery of the Brickadia egui theme: the navy palette, the semantic
//! buttons and every bundled Font Awesome icon, all under `theme::install`.
//!
//! It's the visual acceptance check for `src/gui/theme/` — run it after
//! touching the palette, fonts or widgets to see the result without launching
//! the whole tool suite:
//!
//! ```sh
//! cargo run --example theme_sandbox --features gui   # or: just sandbox
//! ```
//!
//! The gallery UI itself lives in [`heightmap::gui::theme::Sandbox`] so it
//! travels with the theme (and can be dropped into a debug panel); this file is
//! only the eframe host window.
use heightmap::gui::theme::{self, Sandbox};

fn main() -> Result<(), eframe::Error> {
    eframe::run_native(
        "brz theme sandbox",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([560.0, 680.0]),
            ..Default::default()
        },
        Box::new(|cc| {
            theme::install(&cc.egui_ctx);
            Ok(Box::<SandboxApp>::default())
        }),
    )
}

#[derive(Default)]
struct SandboxApp {
    sandbox: Sandbox,
}

impl eframe::App for SandboxApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // No panel margin, full-width scroll: the scrollbar hugs the right edge
        // while the content is padded inside — same as the tool's GUI.
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::SURFACE_PAGE))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        egui::Frame::new()
                            .inner_margin(egui::Margin::same(12))
                            .show(ui, |ui| self.sandbox.ui(ui));
                    });
            });
    }
}
