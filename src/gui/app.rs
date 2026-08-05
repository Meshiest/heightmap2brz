use super::logger;
use crate::gui::audio::AudioApp;
use crate::gui::heightmap::HeightmapApp;
use crate::gui::midi::MidiApp;
use crate::gui::text::TextApp;
use crate::gui::theme::{self, icons, widgets};
use crate::gui::video::VideoApp;
use eframe::App;
use egui::{Align2, CentralPanel, Color32, Context, FontId, Id, ScrollArea, TopBottomPanel, Ui};

/// A homepage tool card is at least this wide; the grid picks a column count
/// from the view width so the cards fill it and reflow as the window resizes.
const MIN_CARD_W: f32 = 240.0;
/// Card height, tall enough for the longest description to wrap at the minimum
/// card width without clipping.
const CARD_H: f32 = 132.0;
/// Gap between cards, and the padding inside one.
const CARD_GAP: f32 = 10.0;
const CARD_PAD: f32 = 12.0;
/// Side length of a card's accent icon square.
const CARD_ICON: f32 = 56.0;
/// Horizontal padding around the scrolling content and the header bar.
const PAGE_PAD: i8 = 12;

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

    /// The Font Awesome glyph for this tool, shown on its homepage card and in
    /// its view header.
    pub const fn icon(&self) -> &'static str {
        match self {
            Menu::Image => icons::IMAGE,
            Menu::Text => icons::FONT,
            Menu::Heightmap => icons::MOUNTAIN,
            Menu::Video => icons::FILM,
            Menu::Audio => icons::WAVE_SQUARE,
            Menu::Midi => icons::MUSIC,
        }
    }

    /// Per-tool accent color for the homepage card's icon square.
    pub const fn accent(&self) -> Color32 {
        match self {
            Menu::Image => Color32::from_rgb(0x00, 0x9b, 0xee),     // blue
            Menu::Heightmap => Color32::from_rgb(0x5d, 0xa9, 0x3d), // green
            Menu::Text => Color32::from_rgb(0xff, 0xa1, 0x0b),      // orange
            Menu::Video => Color32::from_rgb(0xe0, 0x2d, 0x2d),     // red
            Menu::Audio => Color32::from_rgb(0x9b, 0x5d, 0xe5),     // purple
            Menu::Midi => Color32::from_rgb(0x00, 0xbf, 0xa6),      // teal
        }
    }

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
    /// The homepage tool grid, rendered inside the scrolling central panel.
    /// Laid out as explicit rows of `cols` cards (rather than
    /// `horizontal_wrapped`, whose exactly-fits wrapping is fragile) so the tiles
    /// stay a uniform grid with even gaps.
    fn draw_home_grid(&mut self, ui: &mut Ui) {
        // Equal horizontal and vertical gaps: item_spacing governs BOTH the gap
        // between cards in a row and between the rows themselves.
        ui.spacing_mut().item_spacing = egui::vec2(CARD_GAP, CARD_GAP);
        let avail = ui.available_width();
        let cols = ((avail + CARD_GAP) / (MIN_CARD_W + CARD_GAP)).floor().max(1.0) as usize;
        let card_w = ((avail - CARD_GAP * (cols as f32 - 1.0)) / cols as f32).floor();

        let mut clicked = None;
        for chunk in Menu::ALL.chunks(cols) {
            ui.horizontal(|ui| {
                for &menu in chunk {
                    if Self::menu_card(ui, menu, card_w) {
                        clicked = Some(menu);
                    }
                }
            });
        }
        if let Some(menu) = clicked {
            self.pane = Some(menu);
        }
    }

    /// The scrolling content area: the scrollbar is locked to the window's right
    /// edge (the scroll area is full-width) while the content is padded inside,
    /// so the bar never rides in from the margin. A `Frame` (not the panel) so
    /// the header/footer panels above and below stay flush.
    fn scroll_body(ui: &mut Ui, add: impl FnOnce(&mut Ui)) {
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            // Top margin matches the sides so the first card is evenly inset.
            egui::Frame::new()
                .inner_margin(egui::Margin { left: PAGE_PAD, right: PAGE_PAD, top: PAGE_PAD, bottom: PAGE_PAD })
                .show(ui, |ui| add(ui));
        });
    }

    /// The header shown on the homepage: the tool suite's name, version, repo
    /// link and the always-on-top toggle.
    fn draw_header(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.heading("brz tools");
            ui.label(format!("v{}", env!("CARGO_PKG_VERSION")));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if widgets::toggle(ui, &mut self.always_on_top, "Always on top").changed() {
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
        ui.hyperlink_to(
            format!("{}  brickadia-community/heightmap2brz", icons::UP_RIGHT_FROM_SQUARE),
            "https://github.com/brickadia-community/heightmap2brz",
        );
        ui.label("Convert images, heightmaps, video, audio, and MIDI into Brickadia save files.");
        egui::warn_if_debug_build(ui);
    }

    /// One clickable tool card of the given width: a big per-tool icon in a
    /// rounded accent-colored square on the left, with the title and wrapped
    /// description stacked to its right (like omegga's menu buttons). Returns
    /// whether it was clicked.
    ///
    /// Allocated at an EXACT `width` x [`CARD_H`] so every card is identical
    /// regardless of description length; the text is drawn in a clipped child so
    /// an over-long description is trimmed rather than growing the card (which
    /// would desync the grid rows and gaps).
    fn menu_card(ui: &mut Ui, menu: Menu, width: f32) -> bool {
        let (rect, response) = ui.allocate_exact_size(egui::vec2(width, CARD_H), egui::Sense::click());

        // Fill only, no stroke — cards have no outline. Painted from the card's
        // own interaction visuals so it highlights on hover.
        let fill = ui.style().interact(&response).bg_fill;
        ui.painter().rect_filled(rect, egui::CornerRadius::same(8), fill);

        let inner = rect.shrink(CARD_PAD);
        // Accent square with the big white tool icon, vertically centered.
        let sq = egui::Rect::from_min_size(
            egui::pos2(inner.left(), inner.center().y - CARD_ICON / 2.0),
            egui::vec2(CARD_ICON, CARD_ICON),
        );
        ui.painter().rect_filled(sq, egui::CornerRadius::same(12), menu.accent());
        ui.painter().text(
            sq.center().round(),
            Align2::CENTER_CENTER,
            menu.icon(),
            FontId::new(CARD_ICON * 0.55, theme::icon_family()),
            Color32::WHITE,
        );

        // Title (white) + wrapped description, painted DIRECTLY (galleys, not
        // widgets) and clipped to the remaining area. Painting rather than
        // adding widgets is essential: a child `Ui` would allocate space in the
        // parent row and shove the next card over.
        let text_rect =
            egui::Rect::from_min_max(egui::pos2(sq.right() + 8.0, inner.top()), inner.max);
        // Intersect with the current clip (the scroll viewport) — NOT replace it
        // — or a card scrolled off-screen would paint its text over the log below.
        let painter = ui.painter().with_clip_rect(text_rect.intersect(ui.clip_rect()));
        let title = painter.layout_no_wrap(
            menu.as_ref().to_owned(),
            egui::TextStyle::Heading.resolve(ui.style()),
            Color32::WHITE,
        );
        let title_h = title.size().y;
        painter.galley(text_rect.min, title, Color32::WHITE);
        let desc = painter.layout(
            menu.description().to_owned(),
            egui::TextStyle::Body.resolve(ui.style()),
            theme::TEXT,
            text_rect.width(),
        );
        painter.galley(
            egui::pos2(text_rect.left(), text_rect.top() + title_h + 2.0),
            desc,
            theme::TEXT,
        );

        response.on_hover_cursor(egui::CursorIcon::PointingHand).clicked()
    }

    /// An open tool's top bar: back button, then the tool's title and
    /// description. Lives in the dark header panel.
    fn draw_view_header(&mut self, ui: &mut Ui, menu: Menu) {
        ui.horizontal(|ui| {
            // Font Awesome's arrow, from the theme's bundled solid face — no
            // more relying on the emoji font's heavy arrow to dodge tofu.
            if widgets::neutral(ui, format!("{}  Back", icons::ARROW_LEFT))
                .on_hover_text("Return to the tool list")
                .clicked()
            {
                self.pane = None;
            }
            ui.add_space(4.0);
            ui.vertical(|ui| {
                ui.heading(format!("{}  {}", menu.icon(), menu.as_ref()));
                ui.label(menu.description());
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                ui.label(egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION"))).weak());
            });
        });
    }

    /// A tool's scrolling content (file selection + settings).
    fn draw_view_content(&mut self, ui: &mut Ui, menu: Menu) {
        match menu {
            Menu::Image => self.heightmap.draw(ui, &mut self.shared, true),
            Menu::Heightmap => self.heightmap.draw(ui, &mut self.shared, false),
            Menu::Text => self.text.draw(ui, &mut self.shared),
            Menu::Video => self.video.draw(ui, &mut self.shared),
            Menu::Audio => self.audio.draw(ui, &mut self.shared),
            Menu::Midi => self.midi.draw(ui, &mut self.shared),
        }
    }

    /// A tool's fixed footer (progress + submit buttons), in the dark footer
    /// panel between the scroll area and the log.
    fn draw_view_footer(&mut self, ui: &mut Ui, ctx: &Context, menu: Menu) {
        match menu {
            Menu::Image => self.heightmap.draw_footer(ui, ctx, &mut self.shared, true),
            Menu::Heightmap => self.heightmap.draw_footer(ui, ctx, &mut self.shared, false),
            Menu::Text => self.text.draw_footer(ui, &mut self.shared),
            Menu::Video => self.video.draw_footer(ui, &mut self.shared),
            Menu::Audio => self.audio.draw_footer(ui, &mut self.shared),
            Menu::Midi => self.midi.draw_footer(ui, &mut self.shared),
        }
    }
}

impl App for BrzApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        let pane = self.pane;

        // Dark header panel (flush with the scroll below — no gap).
        TopBottomPanel::top(Id::new("header"))
            .frame(header_frame(egui::Margin { left: PAGE_PAD, right: PAGE_PAD, top: 8, bottom: 8 }))
            .show(ctx, |ui| match pane {
                None => self.draw_header(ui),
                Some(menu) => self.draw_view_header(ui, menu),
            });

        // The log console at the very bottom (added first so it reserves the
        // lowest strip; the footer panel then stacks above it).
        TopBottomPanel::bottom(Id::new("logs"))
            .min_height(30.0)
            .resizable(true)
            .frame(egui::Frame {
                // Near-black, darker than the navy footer above it, so the log
                // reads as its own distinct strip.
                fill: Color32::from_rgb(0x02, 0x04, 0x09),
                inner_margin: 4.0.into(),
                outer_margin: 0.0.into(),
                ..Default::default()
            })
            .show(ctx, |ui| {
                logger::draw(ui);
            });

        // Fixed footer with the tool's progress + submit, above the log.
        if let Some(menu) = pane {
            TopBottomPanel::bottom(Id::new("footer"))
                .frame(header_frame(egui::Margin {
                    left: PAGE_PAD,
                    right: PAGE_PAD,
                    top: 8,
                    bottom: 8,
                }))
                .show(ctx, |ui| self.draw_view_footer(ui, ctx, menu));
        }

        // No panel margin: the scroll area's scrollbar reaches the window edge;
        // content padding is applied inside (`scroll_body`).
        CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::SURFACE_PAGE))
            .show(ctx, |ui| {
                Self::scroll_body(ui, |ui| match pane {
                    None => self.draw_home_grid(ui),
                    Some(menu) => self.draw_view_content(ui, menu),
                });
            });
    }
}

/// A dark header/footer panel frame (section-header color) with the given
/// content margin.
fn header_frame(inner: egui::Margin) -> egui::Frame {
    egui::Frame::new()
        .fill(theme::SURFACE_HEADER)
        .inner_margin(inner)
}
