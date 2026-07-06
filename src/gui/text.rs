use crate::{
    gui::{
        SharedOptions,
        util::{PickedImage, deliver_save, pick_images, thumb},
    },
    text::{
        FontPreset, PixelMode, TILE_PX, TextOptions, add_text_tiles, build_calibration_world,
        encode_tiles, make_text_prefab, mono_geometry,
    },
};
use brdb::World;
use egui::{Button, Color32, Ui};
use log::{error, info};
use poll_promise::Promise;

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
    pitch_x: f32,
    pitch_y: f32,
    mode: PixelMode,
    luma_threshold: u8,
    invert: bool,
    /// world units between calibration tile anchors (tiny = tiny displays)
    cube_spacing: u32,
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
            pitch_x: d.pitch_x,
            pitch_y: d.pitch_y,
            mode: d.mode,
            luma_threshold: d.luma_threshold,
            invert: d.invert,
            cube_spacing: 30,
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
            pitch_x: self.pitch_x,
            pitch_y: self.pitch_y,
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
        self.pitch_x = d.pitch_x;
        self.pitch_y = d.pitch_y;
        self.seed_mode_geometry();
        self.cube_spacing = self.natural_spacing();
    }

    /// The mono modes use their own measured component geometry; reseed the
    /// calibration fields whenever the mode (or preset/pixel size) changes.
    fn seed_mode_geometry(&mut self) {
        if self.mode != PixelMode::Color {
            let (lh, kerning, line_offset, pitch_x, pitch_y) =
                mono_geometry(self.mode, self.pixel_size);
            self.line_height = lh;
            self.kerning = kerning;
            self.line_offset = line_offset;
            if let Some(pitch_x) = pitch_x {
                self.pitch_x = pitch_x;
            }
            self.pitch_y = pitch_y;
        }
    }

    /// The calibration grid spacing matching the current mode, pixel size,
    /// and grid pitch — one tile's rendered span.
    fn natural_spacing(&self) -> u32 {
        // the calibration grid always uses TILE_PX tiles (mono modes force
        // seams), so spacing is one 32px tile's span regardless of mode
        (TILE_PX as f32 * self.pixel_size * self.pitch_x)
            .round()
            .max(1.0) as u32
    }

    /// Write the live calibration save: a checkerboard grid with number
    /// Variable gates wired into every text component for in-game tuning.
    fn generate_calibration(&self, shared: &SharedOptions) {
        let world = build_calibration_world(&self.options(), self.cube_spacing as f32);
        let out_file = "calibrate.brz";
        info!("Writing calibration save to {out_file}");
        let data = match world.to_brz_vec() {
            Ok(d) => d,
            Err(e) => return error!("failed to encode brz: {e}"),
        };
        if let Err(e) = deliver_save(data, out_file, shared.out_clipboard) {
            return error!("{e}");
        }
        info!("Paste it, then edit the labeled Variable gates in-game:");
        info!("every tile updates live. Copy the dialed-in numbers back here.");
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
                    let mut mode_changed = false;
                    for m in PixelMode::ALL {
                        mode_changed |= ui.radio_value(&mut self.mode, m, m.name()).changed();
                    }
                    if mode_changed {
                        // re-pull the preset's color geometry, then apply the
                        // mono overrides when applicable
                        self.load_preset();
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
                    .add(
                        egui::Slider::new(&mut self.pixel_size, 0.01..=8.0)
                            .logarithmic(true)
                            .text("units"),
                    )
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
                        ui.label("Font Size");
                        ui.add(egui::DragValue::new(&mut self.line_height).speed(0.01))
                            .on_hover_text(
                                "The TextDisplay LineHeight field — the game's font size.                                  Scales the glyphs; presets derive it from Pixel Size",
                            );
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
                        ui.label("Offset Z");
                        ui.add(egui::DragValue::new(&mut self.offset_z).speed(0.1))
                            .on_hover_text(
                                "Out-of-plane: pushes the text off the anchor wall's face so the cubes hide behind the image",
                            );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Gap X");
                        ui.add(egui::DragValue::new(&mut self.pitch_x).speed(0.002))
                            .on_hover_text(
                                "Horizontal tile spacing as a fraction of the nominal pixel size. Gaps between tile columns = lower it, overlaps = raise it",
                            );
                        ui.label("Gap Y");
                        ui.add(egui::DragValue::new(&mut self.pitch_y).speed(0.002))
                            .on_hover_text(
                                "Vertical tile spacing as a fraction of the nominal pixel size. Gaps between tile rows = lower it, overlaps = raise it",
                            );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Cube Spacing");
                        ui.add(
                            egui::Slider::new(&mut self.cube_spacing, 1..=512)
                                .logarithmic(true)
                                .text("units"),
                        )
                        .on_hover_text(
                            "World distance between the calibration grid's anchor cubes: \
                             tiny spacing = tiny text displays, large = billboards",
                        );
                        if ui
                            .button("Generate calibration save")
                            .on_hover_text(
                                "Writes calibrate.brz: a 128px checkerboard rendered like a \
                                 normal export (current mode included), with labeled number \
                                 Variable gates wired into EVERY text component — edit a \
                                 variable in-game and the whole grid updates live.",
                            )
                            .clicked()
                        {
                            self.generate_calibration(shared);
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
