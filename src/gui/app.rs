use super::logger;
use crate::gui::audio::AudioApp;
use crate::gui::heightmap::HeightmapApp;
use crate::gui::midi::MidiApp;
use crate::gui::text::TextApp;
use crate::gui::video::VideoApp;
use eframe::App;
use egui::{CentralPanel, Color32, Context, Id, ScrollArea, TopBottomPanel, Ui};

/// A homepage tool card is at least this wide; the grid picks a column count
/// from the view width so the cards fill it and reflow as the window resizes.
const MIN_CARD_W: f32 = 240.0;
/// Card height, tall enough for the longest description to wrap at the minimum
/// card width without clipping.
const CARD_H: f32 = 112.0;
/// Gap between cards, and the padding inside one.
const CARD_GAP: f32 = 10.0;
const CARD_PAD: f32 = 12.0;

pub struct BrzApp {
    always_on_top: bool,
    /// The open tool, or `None` for the homepage grid.
    pane: Option<Menu>,

    shared: SharedOptions,
    heightmap: HeightmapApp,
    text: TextApp,
    video: VideoApp,
    audio: AudioApp,
    midi: MidiApp,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Menu {
    Image,
    Text,
    Heightmap,
    Video,
    Audio,
    Midi,
}

impl AsRef<str> for Menu {
    fn as_ref(&self) -> &str {
        match self {
            Menu::Image => "Image2Brick",
            Menu::Text => "Image2Text",
            Menu::Heightmap => "Heightmap",
            Menu::Video => "Video2Brick",
            Menu::Audio => "Audio2Brick",
            Menu::Midi => "MIDI2Brick",
        }
    }
}

impl Menu {
    /// Every tool, in the order the homepage grid presents them.
    const ALL: [Menu; 6] = [
        Menu::Image,
        Menu::Heightmap,
        Menu::Text,
        Menu::Video,
        Menu::Audio,
        Menu::Midi,
    ];

    /// One-line description, shown under the card title and in each view's top
    /// bar.
    pub const fn description(&self) -> &'static str {
        match self {
            Menu::Image => "Select an image to generate as bricks.",
            Menu::Text => "Render an image as TextDisplay component bricks.",
            Menu::Heightmap => {
                "Select a heightmap and colormap to generate optimized brick terrain."
            }
            Menu::Video => {
                "Convert an animated image or frame sequence into wired, animated bricks."
            }
            Menu::Audio => {
                "Turn a song into a cluster of wired, pitched speaker bricks that play it back."
            }
            Menu::Midi => {
                "Turn a MIDI file into wired, pitched speaker bricks, one synth tone per instrument."
            }
        }
    }
}

impl Default for BrzApp {
    fn default() -> Self {
        Self {
            pane: None,
            always_on_top: false,
            shared: SharedOptions::default(),
            heightmap: HeightmapApp::default(),
            text: TextApp::default(),
            video: VideoApp::default(),
            audio: AudioApp::default(),
            midi: MidiApp::default(),
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
    /// The homepage: the app header (title, version, repo link, always-on-top)
    /// above a reflowing grid of tool cards.
    fn draw_home(&mut self, ui: &mut Ui) {
        self.draw_header(ui);
        ui.add_space(6.0);
        ui.separator();
        ui.add_space(10.0);

        ScrollArea::vertical().show(ui, |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(CARD_GAP, CARD_GAP);
            // Pick a column count from the view width so the cards fill it, then
            // give every card an equal share of the row (minus the gaps).
            let avail = ui.available_width();
            let cols = ((avail + CARD_GAP) / (MIN_CARD_W + CARD_GAP)).floor().max(1.0);
            let card_w = ((avail - CARD_GAP * (cols - 1.0)) / cols).floor();
            ui.horizontal_wrapped(|ui| {
                for menu in Menu::ALL {
                    if Self::menu_card(ui, menu, card_w) {
                        self.pane = Some(menu);
                    }
                }
            });
        });
    }

    /// The header shown on the homepage: the tool suite's name, version, repo
    /// link and the always-on-top toggle.
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
        ui.label("Convert images, heightmaps, video, audio, and MIDI into Brickadia save files.");
        egui::warn_if_debug_build(ui);
    }

    /// One clickable tool card of the given width: a framed panel whose title
    /// (heading) sits above its wrapped description. Returns whether it was
    /// clicked.
    ///
    /// The content is laid out top-down (the parent grid is horizontal, which
    /// would otherwise put the description beside the title), bounded to the
    /// card width so the description wraps instead of overrunning its
    /// neighbours. The background is painted in a second pass -- reserve a shape
    /// slot, lay out the content, then fill the slot from the card's own
    /// interaction visuals -- so it highlights on hover, which a `Frame` fill
    /// (drawn before the hover state is known) cannot do.
    fn menu_card(ui: &mut Ui, menu: Menu, width: f32) -> bool {
        let interior = width - 2.0 * CARD_PAD;
        let inner = ui.allocate_ui_with_layout(
            egui::vec2(width, CARD_H),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                let bg = ui.painter().add(egui::Shape::Noop);
                egui::Frame::new()
                    .inner_margin(egui::Margin::same(CARD_PAD as i8))
                    .show(ui, |ui| {
                        ui.set_min_size(egui::vec2(interior, CARD_H - 2.0 * CARD_PAD));
                        ui.set_max_width(interior);
                        ui.add(
                            egui::Label::new(egui::RichText::new(menu.as_ref()).heading().strong())
                                .selectable(false),
                        );
                        ui.add_space(4.0);
                        ui.add(egui::Label::new(menu.description()).selectable(false).wrap());
                    });
                bg
            },
        );
        let bg = inner.inner;
        let response = inner.response.interact(egui::Sense::click());
        // Release the style borrow before touching the painter again.
        let (fill, stroke) = {
            let v = ui.style().interact(&response);
            (v.bg_fill, v.bg_stroke)
        };
        ui.painter().set(
            bg,
            egui::epaint::RectShape::new(
                response.rect,
                egui::CornerRadius::same(6),
                fill,
                stroke,
                egui::StrokeKind::Inside,
            ),
        );
        response
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .clicked()
    }

    /// An open tool: a top bar (back button, then the tool's title and
    /// description) above the tool's own scrolling content.
    fn draw_view(&mut self, ui: &mut Ui, ctx: &Context, menu: Menu) {
        ui.horizontal(|ui| {
            // `\u{2B05}` (heavy leftwards arrow) is in the bundled emoji font
            // that already renders `✖` elsewhere, unlike the plain arrow
            // `\u{2190}`, which has no glyph and shows a tofu box.
            if ui
                .add_sized([88.0, 34.0], egui::Button::new("\u{2B05} Back"))
                .on_hover_text("Return to the tool list")
                .clicked()
            {
                self.pane = None;
            }
            ui.separator();
            ui.vertical(|ui| {
                ui.heading(menu.as_ref());
                ui.label(menu.description());
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                ui.label(egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION"))).weak());
            });
        });
        ui.separator();
        ui.add_space(4.0);

        ScrollArea::vertical().show(ui, |ui| match menu {
            Menu::Image => self.heightmap.draw(ui, ctx, &mut self.shared, true),
            Menu::Heightmap => self.heightmap.draw(ui, ctx, &mut self.shared, false),
            Menu::Text => self.text.draw(ui, &mut self.shared),
            Menu::Video => self.video.draw(ui, &mut self.shared),
            Menu::Audio => self.audio.draw(ui, &mut self.shared),
            Menu::Midi => self.midi.draw(ui, &mut self.shared),
        });
    }
}

impl App for BrzApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        // The log console is a real bottom panel now, so it shows under both the
        // homepage and any open tool. Added before the central panel so it
        // reserves its space first.
        TopBottomPanel::bottom(Id::new("logs"))
            .min_height(30.0)
            .resizable(true)
            .frame(egui::Frame {
                fill: Color32::BLACK,
                inner_margin: 4.0.into(),
                outer_margin: 0.0.into(),
                ..Default::default()
            })
            .show(ctx, |ui| {
                logger::draw(ui);
            });

        CentralPanel::default().show(ctx, |ui| match self.pane {
            None => self.draw_home(ui),
            Some(menu) => self.draw_view(ui, ctx, menu),
        });
    }
}
