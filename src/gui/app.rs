use super::logger;
use crate::gui::audio::AudioApp;
use crate::gui::heightmap::HeightmapApp;
use crate::gui::text::TextApp;
use crate::gui::video::VideoApp;
use eframe::App;
use egui::{CentralPanel, Color32, Context, Id, ScrollArea, TopBottomPanel, Ui};

pub struct BrzApp {
    always_on_top: bool,
    pane: Menu,

    shared: SharedOptions,
    heightmap: HeightmapApp,
    text: TextApp,
    video: VideoApp,
    audio: AudioApp,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Menu {
    Image,
    Text,
    Heightmap,
    Video,
    Audio,
}

impl AsRef<str> for Menu {
    fn as_ref(&self) -> &str {
        match self {
            Menu::Image => "Image2Brick",
            Menu::Text => "Image2Text",
            Menu::Heightmap => "Heightmap",
            Menu::Video => "Video2Brick",
            Menu::Audio => "Audio2Brick",
        }
    }
}

impl Menu {
    pub const fn tooltip(&self) -> &'static str {
        match self {
            Menu::Image => "Select an image to generate as bricks",
            Menu::Text => "Render an image as TextDisplay component bricks",
            Menu::Heightmap => {
                "Select a heightmap and colormap to generate optimized brick terrain"
            }
            Menu::Video => "Convert an animated image or frame sequence into wired, animated bricks",
            Menu::Audio => {
                "Turn a song into a cluster of wired, pitched speaker bricks that play it back"
            }
        }
    }
}

impl Default for BrzApp {
    fn default() -> Self {
        Self {
            pane: Menu::Image,
            always_on_top: false,
            shared: SharedOptions::default(),
            heightmap: HeightmapApp::default(),
            text: TextApp::default(),
            video: VideoApp::default(),
            audio: AudioApp::default(),
        }
    }
}

#[derive(Clone)]
pub struct SharedOptions {
    pub out_file: String,
    pub out_clipboard: bool,
}

impl Default for SharedOptions {
    fn default() -> Self {
        Self {
            out_file: "out.brz".to_string(),
            out_clipboard: true,
        }
    }
}

impl BrzApp {
    fn draw_header(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.heading("brz tools");
            ui.label(format!("v{}", env!("CARGO_PKG_VERSION")));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .checkbox(&mut self.always_on_top, "Always on top")
                    .changed()
                {
                    ui.ctx()
                        .send_viewport_cmd(egui::ViewportCommand::WindowLevel(
                            if self.always_on_top {
                                egui::WindowLevel::AlwaysOnTop
                            } else {
                                egui::WindowLevel::Normal
                            },
                        ));
                }
            });
        });
        ui.hyperlink("https://github.com/brickadia-community/heightmap2brz");
        ui.label(
            "Converts heightmap images (PNG/JPG) to Brickadia save files, also works as img2brick",
        );
        egui::warn_if_debug_build(ui);
    }

    fn menu(&mut self, ui: &mut Ui, pane: Menu) {
        if ui
            .selectable_label(self.pane == pane, pane.as_ref())
            .on_hover_text(pane.tooltip())
            .clicked()
        {
            self.pane = pane;
        }
    }
}

impl App for BrzApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        CentralPanel::default().show(ctx, |ui| {
            self.draw_header(ui);
            ui.horizontal(|ui| {
                ui.set_style(egui::Style {
                    override_text_style: Some(egui::TextStyle::Heading),
                    ..Default::default()
                });
                self.menu(ui, Menu::Image);
                self.menu(ui, Menu::Heightmap);
                self.menu(ui, Menu::Text);
                self.menu(ui, Menu::Video);
                self.menu(ui, Menu::Audio);
            });
            ui.separator();
            ScrollArea::vertical()
                .max_height((ui.available_height() - 50.0).max(50.0))
                .show(ui, |ui| match self.pane {
                    Menu::Image => self.heightmap.draw(ui, ctx, &mut self.shared, true),
                    Menu::Heightmap => self.heightmap.draw(ui, ctx, &mut self.shared, false),
                    Menu::Text => self.text.draw(ui, &mut self.shared),
                    Menu::Video => self.video.draw(ui, &mut self.shared),
                    Menu::Audio => self.audio.draw(ui, &mut self.shared),
                });

            TopBottomPanel::bottom(Id::new("logs"))
                .min_height(30.0)
                .resizable(true)
                .frame(egui::Frame {
                    fill: Color32::BLACK,
                    inner_margin: 4.0.into(),
                    outer_margin: 0.0.into(),
                    ..Default::default()
                })
                .show_inside(ui, |ui| {
                    logger::draw(ui);
                });
        });
    }
}
