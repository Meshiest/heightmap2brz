use crate::{
    gui::{
        SharedOptions,
        util::{PickedImage, deliver_save, pick_images, thumb},
    },
    text::{
        FontMetrics, FontPreset, PixelMode, TextOptions, add_text_tiles, build_measuring_world,
        encode_tiles, make_text_prefab,
    },
};
use brdb::World;
use egui::{Button, Color32, Ui};
use log::{error, info};
use poll_promise::Promise;
use std::collections::HashMap;

pub struct TextApp {
    image: Option<PickedImage>,
    pending_pick: Option<Promise<Vec<PickedImage>>>,
    preset: FontPreset,
    fill_char: String,
    empty_char: String,
    char_repeat: usize,
    alpha_threshold: u8,
    pixel_size: f32,
    // manual calibration; reseeded when the preset or pixel size changes
    line_height: f32,
    line_offset: f32,
    kerning: f32,
    offset_x: f32,
    offset_y: f32,
    offset_z: f32,
    pitch_scale: f32,
    mode: PixelMode,
    luma_threshold: u8,
    invert: bool,
    // measured spans of the measuring block (world units) and per-font
    // metrics solved from them
    measured_v: f32,
    measured_h: f32,
    metrics: HashMap<&'static str, FontMetrics>,
}

impl Default for TextApp {
    fn default() -> Self {
        let preset = FontPreset::MonaspaceArgon;
        let d = preset.options(1.0);
        Self {
            image: None,
            pending_pick: None,
            preset,
            fill_char: d.fill_char.to_string(),
            empty_char: d.empty_char.to_string(),
            char_repeat: d.char_repeat,
            alpha_threshold: d.alpha_threshold,
            pixel_size: d.line_world_height,
            line_height: d.line_height,
            line_offset: d.line_offset,
            kerning: d.kerning,
            offset_x: d.offset_x,
            offset_y: d.offset_y,
            offset_z: d.offset_z,
            pitch_scale: d.pitch_scale,
            mode: d.mode,
            luma_threshold: d.luma_threshold,
            invert: d.invert,
            measured_v: 0.0,
            measured_h: 0.0,
            metrics: HashMap::new(),
        }
    }
}

impl TextApp {
    fn options(&self) -> TextOptions {
        let d = self.preset.options(self.pixel_size);
        TextOptions {
            fill_char: self.fill_char.chars().next().unwrap_or(d.fill_char),
            empty_char: self.empty_char.chars().next().unwrap_or(d.empty_char),
            char_repeat: self.char_repeat,
            alpha_threshold: self.alpha_threshold,
            line_height: self.line_height,
            line_offset: self.line_offset,
            kerning: self.kerning,
            offset_x: self.offset_x,
            offset_y: self.offset_y,
            offset_z: self.offset_z,
            pitch_scale: self.pitch_scale,
            mode: self.mode,
            luma_threshold: self.luma_threshold,
            invert: self.invert,
            ..d
        }
    }

    /// Reseed glyphs + calibration fields from the current preset/pixel size.
    /// Measured metrics for the font, when present, override the preset's
    /// guesses with exact solved values.
    fn load_preset(&mut self) {
        let d = self.preset.options(self.pixel_size);
        self.fill_char = d.fill_char.to_string();
        self.empty_char = d.empty_char.to_string();
        self.char_repeat = d.char_repeat;
        self.line_height = d.line_height;
        self.line_offset = d.line_offset;
        self.kerning = d.kerning;
        self.offset_x = d.offset_x;
        self.offset_y = d.offset_y;
        self.offset_z = d.offset_z;
        self.pitch_scale = d.pitch_scale;
        if let Some(m) = self.metrics.get(self.preset.font_asset()) {
            let (lh, kerning) = m.solve(self.pixel_size, self.char_repeat);
            self.line_height = lh;
            self.kerning = kerning;
        }
    }

    /// Solve the calibration from the measured block spans and apply it.
    fn apply_measurements(&mut self) {
        if self.measured_v <= 0.0 || self.measured_h <= 0.0 {
            return error!("enter the measured V and H spans (world units) first");
        }
        let m = FontMetrics::from_measured(self.measured_v, self.measured_h);
        self.metrics.insert(self.preset.font_asset(), m);
        let (lh, kerning) = m.solve(self.pixel_size, self.char_repeat);
        self.line_height = lh;
        self.kerning = kerning;
        info!(
            "{}: line advance {:.4}/LH, char advance {:.4}/LH -> LineHeight {:.4}, Kerning {:.4}",
            self.preset.name(),
            m.line_advance,
            m.char_advance,
            lh,
            kerning
        );
    }

    /// Write the font-measuring save using the current calibration values.
    fn generate_measuring(&self, shared: &SharedOptions) {
        let world = build_measuring_world(&self.options());
        let out_file = "measure.brz";
        info!("Writing measuring save to {out_file}");
        let data = match world.to_brz_vec() {
            Ok(d) => d,
            Err(e) => return error!("failed to encode brz: {e}"),
        };
        if let Err(e) = deliver_save(data, out_file, shared.out_clipboard) {
            return error!("{e}");
        }
        info!("Paste it: read the 10x10 block's V/H spans off the rulers,");
        info!("enter them below and Apply; the sweeps verify the result.");
    }

    /// Encoding and writing are fast enough to run on the UI thread.
    fn generate(&self, shared: &SharedOptions) {
        let Some(picked) = &self.image else { return };
        let img = (*picked.image).clone();
        let opts = self.options();
        let tiles = match encode_tiles(&img, &opts) {
            Ok(t) => t,
            Err(e) => return error!("{e}"),
        };
        info!(
            "Encoded {} tile(s), {} text band(s)",
            tiles.len(),
            tiles.iter().map(|t| t.bands.len()).sum::<usize>()
        );

        let (img_w, img_h) = img.dimensions();
        let mut world = World::new();
        add_text_tiles(&mut world, tiles, &opts);
        world.meta.bundle.description = "Text image generated from image file".to_string();
        // prefab pivots/bounds cover the full rendered image so ground
        // placement holds the anchor cubes up in the air
        make_text_prefab(&mut world, img_w, img_h, &opts);

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

    fn draw_settings(&mut self, ui: &mut Ui, shared: &mut SharedOptions) {
        ui.heading("Settings");
        ui.label("Render an image as TextDisplay component bricks — one colored glyph per pixel.");

        egui::Grid::new("text_settings_grid")
            .striped(true)
            .spacing([40.0, 4.0])
            .show(ui, |ui| {
                ui.label("Save Destination")
                    .on_hover_text("The save will be created relative to the location of the exe.");
                ui.horizontal(|ui| {
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

                ui.label("Font")
                    .on_hover_text("Font preset; selecting one reseeds glyphs and calibration");
                ui.horizontal(|ui| {
                    let mut changed = false;
                    egui::ComboBox::from_id_salt("text_font_preset")
                        .selected_text(self.preset.name())
                        .show_ui(ui, |ui| {
                            for p in FontPreset::ALL {
                                changed |=
                                    ui.selectable_value(&mut self.preset, p, p.name()).changed();
                            }
                        });
                    if self.preset == FontPreset::Orbitron {
                        ui.colored_label(
                            Color32::from_rgb(255, 200, 100),
                            "1 glyph/px; transparency does not align (proportional font)",
                        );
                    }
                    if changed {
                        self.load_preset();
                    }
                });
                ui.end_row();

                ui.label("Pixel Glyphs").on_hover_text(
                    "Glyphs emitted per pixel: fill for opaque pixels, empty for transparent",
                );
                ui.horizontal(|ui| {
                    ui.label("Fill");
                    ui.add(egui::TextEdit::singleline(&mut self.fill_char).desired_width(24.0))
                        .on_hover_text("Glyph for opaque pixels (first character is used)");
                    ui.label("Empty");
                    ui.add(egui::TextEdit::singleline(&mut self.empty_char).desired_width(24.0))
                        .on_hover_text("Glyph for transparent pixels (first character is used)");
                    ui.label("Repeat");
                    ui.add(egui::Slider::new(&mut self.char_repeat, 1..=4))
                        .on_hover_text(
                            "Glyphs per pixel; 2 makes square pixels with the monospace font",
                        );
                });
                ui.end_row();

                ui.label("Mode").on_hover_text(
                    "Color: one colored glyph run per pixel. Braille: monochrome, 8 pixels per character. Blocks: monochrome quadrants, 4 per character",
                );
                ui.horizontal(|ui| {
                    for m in PixelMode::ALL {
                        ui.radio_value(&mut self.mode, m, m.name());
                    }
                    if self.mode != PixelMode::Color {
                        ui.label("Luma");
                        ui.add(egui::Slider::new(&mut self.luma_threshold, 0..=255))
                            .on_hover_text("Pixels at least this bright are drawn");
                        ui.checkbox(&mut self.invert, "Invert")
                            .on_hover_text("Draw dark pixels instead of bright ones");
                    }
                });
                ui.end_row();

                ui.label("Alpha Threshold")
                    .on_hover_text("Pixels with alpha below this are rendered as transparent");
                ui.add(egui::Slider::new(&mut self.alpha_threshold, 0..=255));
                ui.end_row();

                ui.label("Pixel Size")
                    .on_hover_text("World units per pixel row (1.0 = calibrated glyph fit)");
                if ui
                    .add(egui::Slider::new(&mut self.pixel_size, 0.01..=8.0).text("units"))
                    .changed()
                {
                    self.load_preset();
                }
                ui.end_row();

                ui.label("Calibration").on_hover_text(
                    "Manual glyph-fit tuning for the selected font; the preset seeds these \
                     and changing the preset or pixel size reseeds them",
                );
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("LineHeight");
                        ui.add(egui::DragValue::new(&mut self.line_height).speed(0.01))
                            .on_hover_text("Font size: scales the glyphs");
                        ui.label("LineOffset");
                        ui.add(egui::DragValue::new(&mut self.line_offset).speed(0.05));
                        ui.label("Kerning");
                        ui.add(egui::DragValue::new(&mut self.kerning).speed(0.01));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Offset X");
                        ui.add(egui::DragValue::new(&mut self.offset_x).speed(0.01));
                        ui.label("Offset Y");
                        ui.add(egui::DragValue::new(&mut self.offset_y).speed(0.01));
                        ui.label("Grid Scale");
                        ui.add(egui::DragValue::new(&mut self.pitch_scale).speed(0.002))
                            .on_hover_text(
                                "Brick-grid spacing as a fraction of the nominal pixel size, matching the font's actual rendered size. Tile gaps = lower it, overlaps = raise it",
                            );
                        ui.label("Offset Z");
                        ui.add(egui::DragValue::new(&mut self.offset_z).speed(0.1))
                            .on_hover_text(
                                "Out-of-plane: pushes the text off the anchor wall's face so the cubes hide behind the image",
                            );
                    });
                    ui.horizontal(|ui| {
                        if ui
                            .button("Generate measuring save")
                            .on_hover_text(
                                "Writes measure.brz: a 10x10 glyph block at LineHeight 1 \
                                 against brick rulers, plus LineHeight and Offset sweep \
                                 grids with corner markers. Read the block's V/H spans off \
                                 the rulers and enter them here.",
                            )
                            .clicked()
                        {
                            self.generate_measuring(shared);
                        }
                        ui.label("Measured V");
                        ui.add(egui::DragValue::new(&mut self.measured_v).speed(0.05))
                            .on_hover_text("Height of the 10-row block in world units");
                        ui.label("Measured H");
                        ui.add(egui::DragValue::new(&mut self.measured_h).speed(0.05))
                            .on_hover_text("Width of the 10-column block in world units");
                        if ui
                            .button("Apply")
                            .on_hover_text(
                                "Solve LineHeight, Kerning, and Row Advance exactly from \
                                 the measurements; remembered per font while the app runs",
                            )
                            .clicked()
                        {
                            self.apply_measurements();
                        }
                    });
                });
                ui.end_row();
            });

        ui.add_space(8.0);
        ui.separator();

        ui.heading("Image");
        ui.label("Select the image to render as text.");
        if ui
            .add(Button::new("Select image").fill(Color32::from_rgb(60, 60, 120)))
            .clicked()
            && self.pending_pick.is_none()
        {
            self.pending_pick = Some(pick_images(false));
        }

        if let Some(img) = &self.image {
            let mut clear = false;
            egui::Grid::new("text_image_grid")
                .striped(true)
                .spacing([8.0, 4.0])
                .min_col_width(4.0)
                .show(ui, |ui| {
                    if ui.button("✖").clicked() {
                        clear = true;
                    }
                    thumb(ui, img);
                    ui.label(&img.name);
                });
            if clear {
                self.image = None;
            }
        }
    }

    fn draw_submit(&mut self, ui: &mut Ui, shared: &mut SharedOptions) {
        if self.image.is_some() {
            if ui
                .add(Button::new("Generate image2text save").fill(Color32::from_rgb(50, 90, 50)))
                .clicked()
            {
                self.generate(shared);
            }
        } else {
            ui.label("Select an image file to continue...");
        }
    }

    pub fn draw(&mut self, ui: &mut Ui, shared: &mut SharedOptions) {
        if let Some(promise) = self.pending_pick.take() {
            match promise.try_take() {
                Ok(images) => {
                    if let Some(img) = images.into_iter().next() {
                        info!("Selected image: {}", img.name);
                        self.image = Some(img);
                    }
                }
                Err(promise) => self.pending_pick = Some(promise),
            }
        }
        self.draw_settings(ui, shared);
        ui.separator();
        self.draw_submit(ui, shared);
    }
}
