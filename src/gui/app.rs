#![allow(dead_code, unused_variables)]
use std::{
    borrow::Cow,
    collections::HashSet,
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender},
    thread::{self},
    time::Duration,
};

use super::logger;
use crate::{gui::util::maps_from_files, opt::*, util::bricks_to_save, util::*};
use brdb::assets::bricks::{
    PB_DEFAULT_BRICK, PB_DEFAULT_MICRO_BRICK, PB_DEFAULT_STUDDED, PB_DEFAULT_TILE,
};
use eframe::App;
use egui::{
    Button, CentralPanel, Color32, Context, Id, ImageSource, ProgressBar, ScrollArea,
    TopBottomPanel, Ui, vec2,
};
use log::{error, info};
use poll_promise::Promise;
use std::path::Path;

#[derive(PartialEq, Clone)]
enum BrickMode {
    Default,
    Tile,
    Stud,
    Micro,
}

#[derive(PartialEq, Clone)]
enum OptimizationMode {
    None,
    Quad,
    Greedy,
}

type Progress = (&'static str, f32);

pub struct HeightmapApp {
    // options for the generator
    heightmaps: Vec<PathBuf>,
    colormap: Option<PathBuf>,
    out_file: String,
    out_clipboard: bool,
    vertical_scale: u32,
    horizontal_size: u16,
    optimization: OptimizationMode,
    opt_cull: bool,
    opt_nocollide: bool,
    opt_lrgb: bool,
    opt_hdmap: bool,
    opt_snap: bool,
    opt_glow: bool,
    mode: BrickMode,
    progress: Progress,
    progress_channel: (Sender<Progress>, Receiver<Progress>),
    promise: Option<Promise<Result<(), String>>>,
    gen_interrupt: Option<Sender<()>>,
}

impl Default for HeightmapApp {
    fn default() -> Self {
        Self {
            // default generator options
            heightmaps: vec![],
            colormap: None,
            out_file: "out.brz".to_string(),
            out_clipboard: true,
            vertical_scale: 1,
            horizontal_size: 1,
            optimization: OptimizationMode::Quad,
            opt_cull: false,
            opt_nocollide: false,
            opt_lrgb: false,
            opt_snap: false,
            opt_glow: false,
            opt_hdmap: false,
            mode: BrickMode::Micro,
            promise: None,
            progress: ("Pending", 0.),
            progress_channel: mpsc::channel(),
            gen_interrupt: None,
        }
    }
}

impl HeightmapApp {
    fn has_large_image(&self) -> bool {
        // Check if any heightmap or colormap is larger than 1024px in either dimension
        let check_image = |path: &PathBuf| -> bool {
            if let Ok(img) = image::open(path) {
                img.width() > 1024 || img.height() > 1024
            } else {
                false
            }
        };

        self.heightmaps.iter().any(check_image) || self.colormap.as_ref().map_or(false, check_image)
    }

    fn options(&self) -> GenOptions {
        // output options
        let mut options = GenOptions {
            size: self.horizontal_size * 5,
            scale: self.vertical_scale,
            cull: self.opt_cull,
            asset: PB_DEFAULT_BRICK,
            tile: self.mode == BrickMode::Tile,
            micro: self.mode == BrickMode::Micro,
            stud: self.mode == BrickMode::Stud,
            snap: self.opt_snap,
            img: self.heightmaps.is_empty() && self.colormap.is_some(),
            glow: self.opt_glow,
            hdmap: self.opt_hdmap,
            lrgb: self.opt_lrgb,
            nocollide: self.opt_nocollide,
            quadtree: self.optimization == OptimizationMode::Quad,
            greedy: self.optimization == OptimizationMode::Greedy,
        };

        if options.tile {
            options.asset = PB_DEFAULT_TILE;
        } else if options.micro {
            options.size /= 5;
            options.asset = PB_DEFAULT_MICRO_BRICK;
        }
        if options.stud {
            options.asset = PB_DEFAULT_STUDDED;
        }

        options
    }

    fn run_converter(&mut self) {
        let out_file = self.out_file.clone();
        let is_clipboard = self.out_clipboard;
        let options = self.options();
        let heightmap_files = self.heightmaps.clone();
        let colormap_file = self.colormap.clone();

        let progress_tx = self.progress_channel.0.clone();
        let progress = move |status, p| progress_tx.send((status, p)).unwrap();

        // handle interrupts
        let (tx, rx) = mpsc::channel::<()>();
        self.gen_interrupt = Some(tx);
        let is_stopped = move || rx.try_recv().is_ok();

        self.promise.get_or_insert_with(|| {
            info!("Preparing converter...");
            let (sender, promise) = Promise::new();

            progress("Reading", 0.);

            thread::spawn(move || {
                macro_rules! stop_if_stopped {
                    () => {
                        if is_stopped() {
                            sender.send(Err("Stopped by user".to_string()));
                            return;
                        }
                    };
                }

                info!("Reading image files...");
                let (heightmap, colormap) =
                    match maps_from_files(&options, heightmap_files, colormap_file) {
                        Ok(hc) => hc,
                        Err(err) => {
                            error!("{err}");
                            return sender.send(Err(err));
                        }
                    };

                stop_if_stopped!();
                progress("Generating", 0.10);

                let bricks = match gen_opt_heightmap(&*heightmap, &*colormap, options, |p| {
                    progress("Generating", 0.1 + 0.85 * p);
                    !is_stopped()
                }) {
                    Ok(b) => b,
                    Err(err) => {
                        error!("{err}");
                        return sender.send(Err(err));
                    }
                };
                stop_if_stopped!();

                info!("Writing Save to {}", out_file);
                progress("Writing", 0.95);
                let data = bricks_to_save(bricks);

                if out_file.to_lowercase().ends_with(".brz") {
                    let brz = match data.to_brz_vec() {
                        Ok(b) => b,
                        Err(e) => {
                            let err = format!("failed to encode brz: {e}");
                            error!("{err}");
                            return sender.send(Err(err));
                        }
                    };
                    if let Err(e) = std::fs::write(&out_file, brz) {
                        let err = format!("failed to write file: {e}");
                        error!("{err}");
                        return sender.send(Err(err));
                    }
                } else if out_file.to_lowercase().ends_with(".brdb") {
                    if let Err(e) = data.write_brdb(&out_file) {
                        let err = format!("failed to write file: {e}");
                        error!("{err}");
                        return sender.send(Err(err));
                    };
                } else {
                    let err = "output file must end with .brz or .brdb".to_string();
                    error!("{err}");
                    return sender.send(Err(err));
                }

                if is_clipboard {
                    // If the path is not absolute, make it absolute relative to the current exe location
                    let mut full_path = Path::new(&out_file)
                        .canonicalize()
                        .unwrap_or_else(|_| PathBuf::from(&out_file))
                        .to_string_lossy()
                        .to_string();

                    // lowercase the first letter
                    full_path.get_mut(0..1).map(|s| s.make_ascii_lowercase());

                    #[cfg(target_os = "windows")]
                    {
                        if let Err(e) = clipboard_win::raw::open() {
                            error!("failed to open clipboard: {e}");
                            return sender.send(Err(format!("failed to open clipboard: {e}")));
                        }

                        if let Err(e) = clipboard_win::raw::set_file_list(&[full_path.clone()]) {
                            error!("failed to open clipboard: {e}");
                            return sender.send(Err(format!("failed to open clipboard: {e}")));
                        } else {
                            info!("Wrote path {full_path} to clipboard");
                        }

                        if let Err(e) = clipboard_win::raw::close() {
                            error!("failed to close clipboard: {e}");
                            return sender.send(Err(format!("failed to close clipboard: {e}")));
                        }
                    }

                    #[cfg(not(target_os = "windows"))]
                    {
                        info!("Clipboard file path support is only available on Windows");
                        info!("File saved to: {}", full_path);
                    }
                }

                stop_if_stopped!();
                progress("Finished", 1.0);

                info!("Done!");
                sender.send(Ok(()));
                thread::sleep(Duration::from_millis(500));
                progress("", 2.0);
            });
            // thread::self.gen_thread.unwrap().thread().
            promise
        });
    }

    fn draw_header(&self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.heading("heightmap2brz");
            ui.label(format!("v{}", env!("CARGO_PKG_VERSION")));
        });
        ui.hyperlink("https://github.com/brickadia-community/heightmap2brz");
        ui.label(
            "Converts heightmap images (PNG/JPG) to Brickadia save files, also works as img2brick",
        );
        egui::warn_if_debug_build(ui);
    }

    fn draw_settings(&mut self, ui: &mut Ui) {
        ui.heading("Settings");
        ui.label("Configure how the generator outputs the saves as bricks");

        // list of settings
        egui::Grid::new("settings_grid")
            .striped(true)
            .spacing([40.0, 4.0])
            .show(ui, |ui| {
                ui.label("Save Destination")
                    .on_hover_text("The save will be created relative to the location of the exe.");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.out_clipboard, "Copy to clipboard")
                        .on_hover_text("Copy the save file path to clipboard after generation");

                    ui.add(egui::TextEdit::singleline(&mut self.out_file).hint_text("File Name"));
                });
                ui.end_row();
                let out_file_lowercase = self.out_file.to_lowercase();
                let is_brz = out_file_lowercase.ends_with(".brz");
                if !is_brz && !out_file_lowercase.ends_with(".brdb") {
                    ui.label("Warning:");
                    ui.colored_label(Color32::RED, "Output file must end with .brz or .brdb");
                    ui.end_row();
                }

                ui.label("Horizontal Scale")
                    .on_hover_text("The size of each pixel in studs (or microbricks)");
                ui.add(egui::Slider::new(&mut self.horizontal_size, 1..=100).text("studs"));
                ui.end_row();
                ui.label("Vertical Size")
                    .on_hover_text("The height of each shade of grey from the heightmap");
                ui.add(egui::Slider::new(&mut self.vertical_scale, 1..=100).text("units"));
                ui.end_row();

                ui.label("Optimization")
                    .on_hover_text("Algorithm used to reduce brick count");
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.optimization, OptimizationMode::None, "None")
                            .on_hover_text("No optimization (~one brick per pixel)");
                        ui.radio_value(&mut self.optimization, OptimizationMode::Quad, "Quadtree")
                        .on_hover_text("Use quadtree based optimization. Looks prettier. May use more bricks. Uses a lot of memory for larger maps");
                        ui.radio_value(&mut self.optimization, OptimizationMode::Greedy, "Greedy")
                            .on_hover_text("Use greedy mesh for each height level. Uses fewer bricks but slower for images with many colors/heights");
                    });
                    if self.optimization == OptimizationMode::Greedy && !self.heightmaps.is_empty() {
                        ui.colored_label(
                            Color32::from_rgb(255, 200, 100),
                            "Note: Greedy meshing does not properly calculate brick heights based on neighbor heights"
                        );
                    }
                    if self.optimization == OptimizationMode::Greedy && self.has_large_image() {
                        ui.colored_label(
                            Color32::from_rgb(255, 100, 100),
                            "Warning: Large images (>1024px) may use excessive memory with greedy optimization"
                        );
                    }});
                ui.end_row();

                ui.label("Options")
                    .on_hover_text("A list of options for modifying how the generator works");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.opt_snap, "Snap")
                        .on_hover_text("Snap bricks to the brick grid");
                    ui.checkbox(&mut self.opt_cull, "Cull").on_hover_text(
                        "Automatically remove bottom level bricks and fully transparent bricks\n\
                            In image mode, only transparent bricks are removed",
                    );
                    ui.checkbox(&mut self.opt_nocollide, "No Collide")
                        .on_hover_text("Disable brick collision");
                    ui.checkbox(&mut self.opt_lrgb, "LRGB")
                        .on_hover_text("Use linear rgb input color instead of sRGB");
                    ui.checkbox(&mut self.opt_glow, "Glow")
                        .on_hover_text("Glow bricks at lowest intensity");
                    ui.checkbox(&mut self.opt_hdmap, "HD Map")
                        .on_hover_text("Using a high detail rgb color encoded heightmap");
                });
                ui.end_row();

                ui.label("Brick Type")
                    .on_hover_text("Change which brick type is used for the save file");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.mode, BrickMode::Default, "Default")
                        .on_hover_text("Use default bricks");
                    ui.radio_value(&mut self.mode, BrickMode::Tile, "Tile")
                        .on_hover_text("Use tile bricks");
                    ui.radio_value(&mut self.mode, BrickMode::Stud, "Stud")
                        .on_hover_text("Use studded bricks");
                    ui.radio_value(&mut self.mode, BrickMode::Micro, "Micro")
                        .on_hover_text("Use micro bricks");
                });
                ui.end_row();
            });

        ui.add_space(8.0);
        ui.separator();

        ui.heading("Heightmap Images");
        ui.label("Select image files to use for save generation.");

        // handle heightmap multiple file selection
        if ui.button("Select heightmaps").clicked() {
            let result = native_dialog::DialogBuilder::file()
                .add_filter("Image Files", ["png", "jpg", "jpeg"])
                .open_multiple_file()
                .show();

            match result {
                Ok(files) => {
                    self.heightmaps = files;
                    info!("Selected heightmap files: {:?}", &self.heightmaps);
                }
                Err(e) => {
                    error!("Error selecting heightmap files: {e}");
                }
            }
        }

        egui::Grid::new("heightmap_grid")
            .striped(true)
            .spacing([8.0, 4.0])
            .min_col_width(4.0)
            .show(ui, |ui| {
                let mut to_remove = HashSet::new();
                for img in self.heightmaps.clone() {
                    if ui.add(Button::new("✖")).clicked() {
                        to_remove.insert(img.clone());
                    }
                    self.thumb(ui, &img);
                    ui.label(Path::new(&img).file_name().unwrap().to_str().unwrap());
                    ui.end_row();
                }
                self.heightmaps.retain(|i| !to_remove.contains(i));
            });

        ui.separator();

        ui.heading("Colormap Image");
        ui.label("Select image file to use for heightmap coloring. Select only a colormap for img2brick mode.");

        // handle colormap single file selection
        if ui
            .add(Button::new("Select colormap").fill(Color32::from_rgb(60, 60, 120)))
            .clicked()
        {
            let result = native_dialog::DialogBuilder::file()
                .add_filter("Image Files", ["png", "jpg", "jpeg"])
                .open_single_file()
                .show();

            match result {
                Ok(file_path) => {
                    info!("Selected colormap file: {:?}", file_path);
                    self.colormap = file_path;
                }
                Err(e) => {
                    error!("Error selecting colormap file: {e}");
                }
            }
        }

        if let Some(path) = self.colormap.clone() {
            egui::Grid::new("colormap_grid")
                .striped(true)
                .spacing([8.0, 4.0])
                .min_col_width(4.0)
                .show(ui, |ui| {
                    if ui.button("✖").clicked() {
                        self.colormap = None;
                    }
                    self.thumb(ui, &path);
                    ui.label(Path::new(&path).file_name().unwrap().to_str().unwrap());
                });
        }
    }

    fn draw_progress(&mut self, ctx: &Context, ui: &mut Ui) -> bool {
        while let Ok(p) = self.progress_channel.1.try_recv() {
            self.progress = p;
        }
        let (progress_text, progress) = self.progress;

        let mut clear_promise = progress > 1.0;
        let mut rendered = false;

        if let Some(p) = &self.promise {
            match p.ready() {
                Some(Ok(())) => {
                    ui.add(
                        ProgressBar::new(ctx.animate_value_with_time(
                            Id::new("progress"),
                            1.0,
                            0.1,
                        ))
                        .text("Finished"),
                    );
                }
                Some(Err(e)) => {
                    ui.horizontal(|ui| {
                        if ui.button("ok").clicked() {
                            clear_promise = true;
                        }
                        ui.colored_label(Color32::RED, format!("Error: {e}"));
                    });
                }
                None => {
                    ui.horizontal(|ui| {
                        let stop_btn = ui.button("Stop");
                        ui.add(
                            ProgressBar::new(ctx.animate_value_with_time(
                                Id::new("progress"),
                                progress,
                                0.1,
                            ))
                            .text(progress_text)
                            .animate(true),
                        );
                        if let (true, Some(tx)) = (stop_btn.clicked(), &self.gen_interrupt) {
                            info!("Sending interrupt...");
                            if let Err(e) = tx.send(()) {
                                error!("error sending interrupt {e}");
                            }
                        }
                    });
                }
            }
            rendered = true;
        }

        if clear_promise {
            self.promise = None
        }

        rendered
    }

    fn draw_submit(&mut self, ui: &mut Ui) {
        // display different text based on the selected image files
        let heightmap_ok = !self.heightmaps.is_empty();
        let colormap_ok = self.colormap.is_some();

        if self.promise.is_some() {
            return;
        }

        if heightmap_ok || colormap_ok {
            if ui
                .add(
                    Button::new(match (heightmap_ok, colormap_ok) {
                        (true, true) => "Generate save",
                        (true, false) => "Generate colorless save",
                        (false, true) => "Generate image2brick save",
                        (false, false) => unreachable!(),
                    })
                    .fill(Color32::from_rgb(50, 90, 50)),
                )
                .clicked()
            {
                self.run_converter();
            }
        } else {
            ui.label("Select some image files to continue...");
        }
    }

    fn thumb(&mut self, ui: &mut Ui, image: &PathBuf) {
        ui.add(
            egui::Image::new(ImageSource::Uri(Cow::from(format!(
                "file://{}",
                image.display().to_string().replace("\\", "/")
            ))))
            .fit_to_exact_size(vec2(32.0, 32.0))
            .maintain_aspect_ratio(false),
        );
    }
}

impl App for HeightmapApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        CentralPanel::default().show(ctx, |ui| {
            self.draw_header(ui);
            ScrollArea::vertical().show(ui, |ui| {
                ui.separator();
                self.draw_settings(ui);
                ui.separator();
                if !self.draw_progress(ctx, ui) {
                    self.draw_submit(ui);
                }
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
