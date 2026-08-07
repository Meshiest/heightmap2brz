use std::sync::mpsc::{self, Receiver, Sender};

use crate::{
    gui::{
        SharedOptions,
        util::{
            PickedImage, bound_pane_width, deliver_world, maps_from_images, out_file_warning_row,
            pick_images, refuse_bad_out_file, save_destination_row, thumb,
        },
    },
    opt::*,
    util::{bricks_to_save, *},
};
use brdb::assets::bricks::{
    PB_DEFAULT_BRICK, PB_DEFAULT_MICRO_BRICK, PB_DEFAULT_SMOOTH_TILE, PB_DEFAULT_STUDDED,
};
use crate::gui::theme::{icons, widgets};
use egui::{Color32, Context, Id, ProgressBar, Ui};
use log::{error, info};
use poll_promise::Promise;

type Progress = (&'static str, f32);

/// Why a render stopped early.
///
/// The two are NOT the same outcome and must not share one `Err(String)`: a
/// failure is the user's to see and fix, while a cancellation is the user's own
/// doing and `progress.rs` requires that it "must never surface as an error
/// dialog or a crash-looking exit". `From<String>` is what lets every fallible
/// step in the worker keep its plain `?`.
enum Halt {
    Cancelled,
    Failed(String),
}

impl From<String> for Halt {
    fn from(e: String) -> Self {
        Halt::Failed(e)
    }
}

/// What the worker's promise carries, from how the render ended.
///
/// **A cancelled render reports SUCCESS.** `progress.rs` states the policy --
/// a cancel "must never surface as an error dialog or a crash-looking exit" --
/// and the Video and Audio panes follow it
/// (`gui::util::deliver_world_unless_cancelled` logs INFO and returns `Ok`).
/// This pane was the odd one out: Stop produced `Err("Stopped by user")`, which
/// `draw_progress` painted as a red "Error: Stopped by user" with an ok button,
/// so pressing the button the UI offered looked like a crash.
///
/// A free function rather than a `match` inside the closure so the policy can
/// be asserted without spawning a worker thread and racing its cancel flag.
fn finish(result: Result<(), Halt>) -> Result<(), String> {
    match result {
        Ok(()) => Ok(()),
        Err(Halt::Cancelled) => {
            info!("Render cancelled -- no save written");
            Ok(())
        }
        Err(Halt::Failed(e)) => Err(e),
    }
}

/// Which selection an in-flight file pick fills.
#[derive(Clone, Copy, PartialEq)]
enum PickTarget {
    Heightmaps,
    Colormap,
}

/// What the Brick Type row selects. The first five values select the asset
/// that the usual renderer puts above itself. The last two values replace the
/// renderer, because one brick used again and again cannot make a sloped
/// surface.
#[derive(PartialEq, Clone)]
enum BrickMode {
    Default,
    Tile,
    SmoothTile,
    Stud,
    Micro,
    /// Smooth micro wedge terrain (`opt::terrain`).
    Terrain,
    /// The Wrapperup rampifier over the height columns (`opt::rampify`).
    Rampify,
    /// Terraced wedge terrain with 45-degree chamfered outlines
    /// (`opt::wedge`).
    Wedge,
}

impl BrickMode {
    fn surface(&self) -> SurfaceMode {
        match self {
            BrickMode::Terrain => SurfaceMode::Terrain,
            BrickMode::Rampify => SurfaceMode::Rampify,
            BrickMode::Wedge => SurfaceMode::Wedge,
            _ => SurfaceMode::Blocks,
        }
    }
}

#[derive(PartialEq, Clone)]
enum OptimizationMode {
    None,
    Quad,
    Greedy,
}

pub struct HeightmapApp {
    // options for the generator
    heightmaps: Vec<PickedImage>,
    colormap: Option<PickedImage>,
    pending_pick: Option<(PickTarget, Promise<Vec<PickedImage>>)>,
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
            pending_pick: None,
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
        let check_image =
            |img: &PickedImage| -> bool { img.image.width() > 1024 || img.image.height() > 1024 };

        self.heightmaps.iter().any(check_image) || self.colormap.as_ref().map_or(false, check_image)
    }

    /// Poll an in-flight file pick and apply the result.
    fn poll_pick(&mut self) {
        if let Some((target, promise)) = self.pending_pick.take() {
            match promise.try_take() {
                Ok(images) => match target {
                    PickTarget::Heightmaps => {
                        if !images.is_empty() {
                            info!(
                                "Selected heightmaps: {:?}",
                                images.iter().map(|i| &i.name).collect::<Vec<_>>()
                            );
                            self.heightmaps = images;
                        }
                    }
                    PickTarget::Colormap => {
                        if let Some(img) = images.into_iter().next() {
                            info!("Selected image: {}", img.name);
                            self.colormap = Some(img);
                        }
                    }
                },
                Err(promise) => self.pending_pick = Some((target, promise)),
            }
        }
    }

    fn options(&self, img_only: bool) -> GenOptions {
        let img = img_only || (self.heightmaps.is_empty() && self.colormap.is_some());
        // The sloped renderers are for a heightmap only. A flat image has no
        // ground to slope, so the Image2Brick page uses blocks. If it did not,
        // it would fill a level plane with wedges.
        let surface = if img {
            SurfaceMode::Blocks
        } else {
            self.mode.surface()
        };
        GenOptions {
            // `--size` counts STUDS in each mode, but the micro mode counts
            // micro units.
            size: if self.mode == BrickMode::Micro {
                self.horizontal_size
            } else {
                self.horizontal_size * 5
            },
            scale: self.vertical_scale,
            cull: self.opt_cull,
            asset: match self.mode {
                BrickMode::Default => PB_DEFAULT_BRICK,
                BrickMode::Tile => PB_DEFAULT_BRICK,
                BrickMode::SmoothTile => PB_DEFAULT_SMOOTH_TILE,
                BrickMode::Stud => PB_DEFAULT_STUDDED,
                BrickMode::Micro => PB_DEFAULT_MICRO_BRICK,
                // The sloped and terraced renderers select their own asset
                // for each cell. They never read this value.
                BrickMode::Terrain | BrickMode::Rampify | BrickMode::Wedge => {
                    PB_DEFAULT_MICRO_BRICK
                }
            },
            micro: self.mode == BrickMode::Micro,
            stud: self.mode == BrickMode::Stud,
            snap: self.opt_snap,
            img,
            glow: self.opt_glow,
            hdmap: self.opt_hdmap,
            lrgb: self.opt_lrgb,
            nocollide: self.opt_nocollide,
            quadtree: self.optimization == OptimizationMode::Quad,
            greedy: self.optimization == OptimizationMode::Greedy,
            surface,
        }
    }

    fn run_converter(&mut self, shared: SharedOptions, img_only: bool) {
        let out_file = shared.out_file.clone();
        let is_clipboard = shared.out_clipboard;
        let options = self.options(img_only);
        // the Image2Brick pane renders the image flat, ignoring any
        // heightmaps picked while on the Heightmap pane
        let heightmaps = if img_only {
            vec![]
        } else {
            self.heightmaps.clone()
        };
        let colormap = self.colormap.clone();

        let progress_tx = self.progress_channel.0.clone();
        // Send failures are IGNORED, as in the Video and Audio panes: a closed
        // channel means the UI went away, which must never take a render that
        // is otherwise fine down with it. This used to `.unwrap()`, i.e. panic
        // the worker thread on a dropped receiver.
        let progress = move |status, p| {
            let _ = progress_tx.send((status, p));
        };

        // handle interrupts
        let (tx, rx) = mpsc::channel::<()>();
        self.gen_interrupt = Some(tx);
        let is_stopped = move || rx.try_recv().is_ok();

        self.promise.get_or_insert_with(|| {
            info!("Preparing converter...");
            let (sender, promise) = Promise::new();

            progress("Reading", 0.);
            let end_progress = progress.clone();

            let render = move || -> Result<(), Halt> {
                let stopped = || -> Result<(), Halt> {
                    if is_stopped() { Err(Halt::Cancelled) } else { Ok(()) }
                };

                info!("Reading image files...");
                let (heightmap, colormap) =
                    maps_from_images(&options, &heightmaps, colormap.as_ref())?;

                stopped()?;
                progress("Generating", 0.10);

                let bricks = gen_opt_heightmap(&*heightmap, &*colormap, options, |p| {
                    progress("Generating", 0.1 + 0.85 * p);
                    !is_stopped()
                })?;
                stopped()?;

                info!("Writing Save to {}", out_file);
                progress("Writing", 0.95);
                let data = bricks_to_save(bricks);

                // This pane's own extension branch, now shared with the other
                // four -- see `gui::util::deliver_world`. It was the only one
                // that honoured the extension at all, so making it the shared
                // implementation is what stops the panes disagreeing again.
                deliver_world(&data, &out_file, is_clipboard)?;

                stopped()?;
                info!("Done!");
                Ok(())
            };

            // The cancel is turned back into a success HERE rather than inside
            // `render`, so every `?` above still short-circuits the work. See
            // [`finish`] for the policy and why this pane needed changing.
            let work = move || -> Result<(), String> { finish(render()) };

            #[cfg(not(target_arch = "wasm32"))]
            std::thread::spawn(move || {
                let result = work();
                if let Err(e) = &result {
                    error!("{e}");
                    sender.send(result);
                } else {
                    end_progress("Finished", 1.0);
                    sender.send(result);
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    end_progress("", 2.0);
                }
            });

            #[cfg(target_arch = "wasm32")]
            {
                // no threads on the web: run synchronously (the tab blocks
                // for the duration of the generation)
                let result = work();
                if let Err(e) = &result {
                    error!("{e}");
                } else {
                    end_progress("", 2.0);
                }
                sender.send(result);
            }

            promise
        });
    }

    fn draw_settings(&mut self, ui: &mut Ui, shared: &mut SharedOptions, img_only: bool) {
        bound_pane_width(ui);
        ui.label("Configure how the generator outputs the saves as bricks");

        // Full-width, square-striped settings table (shared widget).
        widgets::settings_table(ui, |ui, t| {
            save_destination_row(t, ui, shared);
            out_file_warning_row(t, ui, &shared.out_file);

            t.row_hover(ui, "Horizontal Scale", Some("The size of each pixel in studs (or microbricks)"), |ui| {
                widgets::slider(ui, egui::Slider::new(&mut self.horizontal_size, 1..=100).text("studs"));
            });
            if !img_only {
                t.row_hover(ui, "Vertical Size", Some("The height of each shade of grey from the heightmap"), |ui| {
                    widgets::slider(ui, egui::Slider::new(&mut self.vertical_scale, 1..=100).text("units"));
                });
            }

            t.row_hover(ui, "Optimization", Some("Algorithm used to reduce brick count"), |ui| {
                // Vertical for the same reason as the Brick Type row below: the
                // control column is horizontal, so these two notes went to the
                // RIGHT of the buttons and each wrapped into a narrow column.
                ui.vertical(|ui| {
                    ui.horizontal_wrapped(|ui| {
                        widgets::radio(ui, &mut self.optimization, OptimizationMode::None, "None")
                            .on_hover_text("No optimization (~one brick per pixel)");
                        widgets::radio(ui, &mut self.optimization, OptimizationMode::Quad, "Quadtree")
                            .on_hover_text("Use quadtree based optimization. Looks prettier. May use more bricks. Uses a lot of memory for larger maps");
                        widgets::radio(ui, &mut self.optimization, OptimizationMode::Greedy, "Greedy")
                            .on_hover_text("Use greedy mesh for each height level. Uses fewer bricks but slower for images with many colors/heights");
                    });
                    if self.optimization == OptimizationMode::Greedy && !self.heightmaps.is_empty() {
                        ui.colored_label(
                            Color32::from_rgb(255, 200, 100),
                            "Note: Greedy meshing does not properly calculate brick heights based on neighbor heights",
                        );
                    }
                    if self.optimization == OptimizationMode::Greedy && self.has_large_image() {
                        ui.colored_label(
                            Color32::from_rgb(255, 100, 100),
                            "Warning: Large images (>1024px) may use excessive memory with greedy optimization",
                        );
                    }
                });
            });

            t.row_hover(ui, "Options", Some("A list of options for modifying how the generator works"), |ui| {
                ui.horizontal_wrapped(|ui| {
                    widgets::toggle(ui, &mut self.opt_snap, "Snap")
                        .on_hover_text("Snap bricks to the brick grid");
                    widgets::toggle(ui, &mut self.opt_cull, "Cull").on_hover_text(
                        "Automatically remove bottom level bricks and fully transparent bricks\n\
                            In image mode, only transparent bricks are removed",
                    );
                    widgets::toggle(ui, &mut self.opt_nocollide, "No Collide")
                        .on_hover_text("Disable brick collision");
                    widgets::toggle(ui, &mut self.opt_lrgb, "LRGB")
                        .on_hover_text("Use linear rgb input color instead of sRGB");
                    widgets::toggle(ui, &mut self.opt_glow, "Glow")
                        .on_hover_text("Glow bricks at lowest intensity");
                    if !img_only {
                        widgets::toggle(ui, &mut self.opt_hdmap, "HD Map")
                            .on_hover_text("Using a high detail rgb color encoded heightmap");
                    }
                });
            });

            t.row_hover(ui, "Brick Type", Some("Change which brick type is used for the save file"), |ui| {
                // `row_hover` gives the control a HORIZONTAL layout, so a note
                // after the buttons becomes the next item in that flow and goes
                // to the RIGHT of them. The space that stays is a few
                // characters wide, and the note then wraps into a tall, narrow
                // column. This vertical layout puts the note below the buttons,
                // where it gets the full width of the control column.
                ui.vertical(|ui| {
                    ui.horizontal_wrapped(|ui| {
                        widgets::radio(ui, &mut self.mode, BrickMode::Default, "Default")
                            .on_hover_text("Use default bricks");
                        widgets::radio(ui, &mut self.mode, BrickMode::Tile, "Tile")
                            .on_hover_text("Use tile bricks");
                        widgets::radio(ui, &mut self.mode, BrickMode::SmoothTile, "Smooth")
                            .on_hover_text("Use smooth tile bricks");
                        widgets::radio(ui, &mut self.mode, BrickMode::Stud, "Stud")
                            .on_hover_text("Use studded bricks");
                        widgets::radio(ui, &mut self.mode, BrickMode::Micro, "Micro")
                            .on_hover_text("Use micro bricks");
                        widgets::radio(ui, &mut self.mode, BrickMode::Terrain, "Smooth Terrain")
                            .on_hover_text(
                                "Build a SMOOTH surface out of micro wedges instead of flat-topped tiles: \n\
                                 every pixel gets a sloped top fitted to the heights of the four shared \n\
                                 grid vertices around it, so neighbouring cells meet instead of stepping.\n\
                                 Uses roughly 1.5 to 2.5 bricks per pixel",
                            );
                        widgets::radio(ui, &mut self.mode, BrickMode::Rampify, "Rampify")
                            .on_hover_text(
                                "Smooth the surface with Wrapperup's rampifier: fit full-size ramps, \n\
                                 wedges and ramp corners onto the height columns and fill the rest with \n\
                                 plain bricks. Coarser than Smooth Terrain (one plate of vertical \n\
                                 resolution) but builds from ordinary bricks",
                            );
                        widgets::radio(ui, &mut self.mode, BrickMode::Wedge, "Wedge Terrain")
                            .on_hover_text(
                                "Build TERRACED terrain in the style of the BRWorldSculptor sculpting \n\
                                 tool: tops stay flat, and terrace outlines are cut at 45 degrees by \n\
                                 vertical side wedges -- corners chamfered, staircases merged into \n\
                                 large wedges, flat tops merged into boxes. Slopes are not \n\
                                 approximated; this is the look of hand-built brick terrain",
                            );
                    });
                    if self.mode.surface() != SurfaceMode::Blocks {
                        ui.colored_label(
                            Color32::from_rgb(255, 200, 100),
                            "Note: this mode picks its own bricks per cell, so the Optimization and \
                             Snap settings above do not apply",
                        );
                    }
                });
            });
        });
    }

    /// The heightmap multi-select card body (heightmap mode only).
    fn draw_heightmaps(&mut self, ui: &mut Ui) {
        bound_pane_width(ui);
        ui.label("Select image files to use for save generation.");
        if widgets::info(ui, format!("{}  Select heightmaps", icons::IMAGE)).clicked()
            && self.pending_pick.is_none()
        {
            self.pending_pick = Some((PickTarget::Heightmaps, pick_images(true)));
        }
        // Only draw the (striped) list grid when there are rows — an empty grid
        // still reserves height, which read as odd extra bottom padding.
        if !self.heightmaps.is_empty() {
            egui::Grid::new("heightmap_grid")
                .striped(true)
                .spacing([8.0, 4.0])
                .min_col_width(4.0)
                .show(ui, |ui| {
                    let mut to_remove = Vec::new();
                    for (i, img) in self.heightmaps.iter().enumerate() {
                        if widgets::danger_icon(ui, icons::XMARK).clicked() {
                            to_remove.push(i);
                        }
                        thumb(ui, img);
                        ui.add(egui::Label::new(&img.name).truncate());
                        ui.end_row();
                    }
                    for i in to_remove.into_iter().rev() {
                        self.heightmaps.remove(i);
                    }
                });
        }
    }

    /// The colormap / single-image select card body.
    fn draw_colormap(&mut self, ui: &mut Ui, img_only: bool) {
        bound_pane_width(ui);
        ui.label(if img_only {
            "Select the image to convert into bricks (one brick per pixel, optimized)."
        } else {
            "Select image file to use for heightmap coloring."
        });
        let pick_label = if img_only { "Select image" } else { "Select colormap" };
        if widgets::info(ui, format!("{}  {}", icons::IMAGE, pick_label)).clicked()
            && self.pending_pick.is_none()
        {
            self.pending_pick = Some((PickTarget::Colormap, pick_images(false)));
        }
        if let Some(img) = &self.colormap {
            let mut clear = false;
            egui::Grid::new("colormap_grid")
                .striped(true)
                .spacing([8.0, 4.0])
                .min_col_width(4.0)
                .show(ui, |ui| {
                    if widgets::danger_icon(ui, icons::XMARK).clicked() {
                        clear = true;
                    }
                    thumb(ui, img);
                    ui.add(egui::Label::new(&img.name).truncate());
                });
            if clear {
                self.colormap = None;
            }
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
                        let stop_btn = widgets::neutral(ui, format!("{}  Stop", icons::STOP));
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

    fn draw_submit(&mut self, ui: &mut Ui, shared: &mut SharedOptions, img_only: bool) {
        // display different text based on the selected image files
        let heightmap_ok = !self.heightmaps.is_empty();
        let colormap_ok = self.colormap.is_some();

        if self.promise.is_some() {
            return;
        }

        // Refused before the button is offered -- see `util::refuse_bad_out_file`.
        // This pane already honoured the extension when it WROTE the file, but
        // only after the whole render, and only as an error dialog afterwards.
        if refuse_bad_out_file(ui, &shared.out_file) {
            return;
        }

        if img_only {
            if colormap_ok {
                if widgets::primary(ui, format!("{}  Generate image2brick save", icons::DOWNLOAD))
                    .clicked()
                {
                    self.run_converter(shared.clone(), true);
                }
            } else {
                ui.label("Select an image file to continue...");
            }
            return;
        }

        if heightmap_ok || colormap_ok {
            let label = match (heightmap_ok, colormap_ok) {
                (true, true) => "Generate save",
                (true, false) => "Generate colorless save",
                (false, true) => "Generate image2brick save",
                (false, false) => unreachable!(),
            };
            if widgets::primary(ui, format!("{}  {}", icons::DOWNLOAD, label)).clicked() {
                self.run_converter(shared.clone(), false);
            }
        } else {
            ui.label("Select some image files to continue...");
        }
    }

    pub fn draw(&mut self, ui: &mut Ui, shared: &mut SharedOptions, img_only: bool) {
        self.poll_pick();
        // File selection cards above the settings card.
        if img_only {
            widgets::section(ui, "Image", |ui| self.draw_colormap(ui, true));
        } else {
            widgets::section(ui, "Heightmap Images", |ui| self.draw_heightmaps(ui));
            ui.add_space(10.0);
            widgets::section(ui, "Colormap Image", |ui| self.draw_colormap(ui, false));
        }
        ui.add_space(10.0);
        widgets::section(ui, "Settings", |ui| self.draw_settings(ui, shared, img_only));
    }

    /// The fixed footer: the render progress bar or the Generate button.
    pub fn draw_footer(
        &mut self,
        ui: &mut Ui,
        ctx: &Context,
        shared: &mut SharedOptions,
        img_only: bool,
    ) {
        if !self.draw_progress(ctx, ui) {
            self.draw_submit(ui, shared, img_only);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Stop is not an error, on this pane either.**
    ///
    /// `draw_progress` paints any `Err` the worker's promise carries as a red
    /// "Error: ..." with an ok button, and this pane used to hand it
    /// `Err("Stopped by user")` -- so using the Stop button the UI offered
    /// looked exactly like a crash. `progress.rs` states the opposite policy and
    /// the other three panes follow it.
    ///
    /// Asserted on [`finish`] rather than by spawning a real worker and racing
    /// its cancel flag: the outcome of that race is timing, and a test that
    /// passes when the render simply finished first would prove nothing.
    #[test]
    fn a_cancelled_render_reports_success_rather_than_an_error_dialog() {
        assert!(
            finish(Err(Halt::Cancelled)).is_ok(),
            "a cancel must never reach draw_progress as an Err -- that is the red \
             'Error: Stopped by user' dialog the policy forbids"
        );
    }

    /// The complementary case: a REAL failure must still be reported, or the
    /// fix above would have swallowed every error on this pane.
    #[test]
    fn a_real_failure_still_reaches_the_error_dialog() {
        assert_eq!(
            finish(Err(Halt::Failed("no images selected".to_string()))),
            Err("no images selected".to_string())
        );
        assert_eq!(finish(Ok(())), Ok(()));
    }

    /// Every fallible step in the worker keeps its plain `?`, which needs the
    /// `String` errors those steps return to become `Halt::Failed` and not
    /// `Halt::Cancelled`.
    #[test]
    fn a_string_error_converts_into_a_failure_not_a_cancellation() {
        let halt: Halt = "failed to write file".to_string().into();
        assert!(matches!(halt, Halt::Failed(ref e) if e == "failed to write file"));
        assert_eq!(
            finish(Err(halt)),
            Err("failed to write file".to_string()),
            "a converted error must still be reported to the user"
        );
    }

    /// The Brick Type and Optimization notes must sit BELOW their buttons, at
    /// the full width of the control column.
    ///
    /// `SettingsTable::row_hover` gives the control a HORIZONTAL layout. A note
    /// added after the buttons is thus the next item in that flow and goes to
    /// the RIGHT of them, in the few characters of space that stay. It then
    /// wraps into a tall, narrow column. The correction is one `ui.vertical`
    /// around each control body, which no type or compile check can hold, and
    /// which a person could remove during a tidy of the code.
    ///
    /// The test drives a real `egui::Context` and reads the text that the pane
    /// paints. egui needs no window for this.
    #[test]
    fn a_mode_note_is_painted_below_the_buttons_and_not_beside_them() {
        const NOTE: &str = "Note: this mode picks its own bricks per cell, so the Optimization \
                            and Snap settings above do not apply";

        let ctx = Context::default();
        crate::gui::theme::install(&ctx);
        let mut app = HeightmapApp::default();
        app.mode = BrickMode::Terrain;
        let mut shared = SharedOptions::default();

        let mut texts: Vec<(egui::Rect, String)> = Vec::new();
        // Four frames to settle: a table learns its column widths from the
        // frame before, so the first frame is not representative.
        for _ in 0..4 {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::pos2(0.0, 0.0),
                    egui::vec2(900.0, 2400.0),
                )),
                ..Default::default()
            };
            let out = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    app.draw_settings(ui, &mut shared, false);
                });
            });
            texts = out
                .shapes
                .iter()
                .filter_map(|c| match &c.shape {
                    egui::epaint::Shape::Text(t) => Some((
                        egui::Rect::from_min_size(t.pos, t.galley.rect.size()),
                        t.galley.text().to_string(),
                    )),
                    _ => None,
                })
                .collect();
        }

        let rect = |want: &str| {
            texts
                .iter()
                .find(|(_, s)| s == want)
                .map(|(r, _)| *r)
                .unwrap_or_else(|| panic!("the pane painted no text {want:?}"))
        };
        let note = rect(NOTE);
        // The LAST button on the row, so the note must paint below it.
        let rampify = rect("Wedge Terrain");

        assert!(
            note.top() >= rampify.bottom(),
            "the note is beside the buttons: the note is at y {}..{} and the last button is at \
             y {}..{}",
            note.top(),
            note.bottom(),
            rampify.top(),
            rampify.bottom(),
        );
        assert!(
            note.left() <= rampify.left() + 1.0,
            "the note starts at x {} but the buttons start at x {}, so it is in a column to \
             their right",
            note.left(),
            rampify.left(),
        );
        // The narrow column is what looks incorrect, so measure it: a note that
        // gets the width of the control column is wide and short.
        assert!(
            note.width() > 300.0,
            "the note is {} wide in a pane of 900, so it still wraps too much",
            note.width(),
        );
    }
}
