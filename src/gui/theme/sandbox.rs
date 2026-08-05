//! A self-contained gallery of the themed widgets and icons — the visual
//! acceptance check for [`super`]. Rendered by `examples/theme_sandbox.rs`, and
//! handy to drop into a debug panel while tweaking the palette.
//!
//! Holds its own demo state so a single `Sandbox::default()` plus a call to
//! [`Sandbox::ui`] per frame is all a host needs. Every `pub` helper in
//! [`super::widgets`] is exercised here at least once, always through the shared
//! helper rather than a hand-rolled copy — the gallery doubles as living
//! documentation of how a real pane is expected to call them.
//!
//! Kept, like the rest of `theme/`, to an `egui`-only dependency: it touches
//! [`super::widgets`], [`super::icons`] and the palette constants, and nothing
//! from the rest of the crate. That is why the accordion + value-chip helpers
//! (`crate::gui::util::section` / `chip`) are deliberately NOT demoed here —
//! reaching for them would drag a crate dependency into the otherwise-liftable
//! `theme/` module. They belong to a gallery that lives beside `util`.
use super::{icons, widgets};
use egui::{DragValue, Frame, Grid, RichText, Slider, Ui};

/// A tiny enum for the [`widgets::radio`] group and the [`widgets::combo`]
/// dropdown to select over. Both demo widgets edit the *same* [`Sandbox::wave`],
/// so picking a value in one is reflected in the other — the sort of shared
/// selection state a real pane threads through several controls.
#[derive(PartialEq, Eq, Clone, Copy)]
enum Wave {
    Sine,
    Square,
    Saw,
}

impl Wave {
    /// Every variant, for building the radio group and the dropdown items with
    /// one loop each.
    const ALL: [Wave; 3] = [Wave::Sine, Wave::Square, Wave::Saw];

    /// Display label — also the [`widgets::combo`] button's selected text.
    fn label(self) -> &'static str {
        match self {
            Wave::Sine => "Sine",
            Wave::Square => "Square",
            Wave::Saw => "Saw",
        }
    }
}

/// Live state for the interactive demo widgets.
pub struct Sandbox {
    /// Gain for the [`widgets::slider`]; also feeds the `ProgressBar`.
    slider: f32,
    /// A bounded integer behind the stock `DragValue` (numeric entry has no
    /// bespoke themed wrapper — it inherits the palette from [`super::install`]).
    drag: i32,
    /// The three on/off [`widgets::toggle`] switches.
    loop_on: bool,
    in_chip: bool,
    controls: bool,
    /// The three multi-select [`widgets::checkbox`] boxes (distinct from the
    /// on/off toggle switch).
    checks: [bool; 3],
    /// Selected waveform — shared by the radio group and the combo dropdown.
    wave: Wave,
    /// Buffer for the [`widgets::text_field`] demo.
    text: String,
    /// Stands in for a running background job so the toggle-gated
    /// [`widgets::loading`] indicator has something to show.
    busy: bool,
}

impl Default for Sandbox {
    fn default() -> Self {
        Self {
            slider: 0.4,
            drag: 3,
            loop_on: true,
            in_chip: false,
            controls: true,
            checks: [true, false, true],
            wave: Wave::Sine,
            text: String::from("out.brz"),
            busy: true,
        }
    }
}

impl Sandbox {
    /// Render the whole gallery into `ui`. Each group is a [`widgets::section`]
    /// card (the same header-band-over-body panel the real panes use) so the
    /// layout matches production and the section helper is itself on display.
    pub fn ui(&mut self, ui: &mut Ui) {
        ui.heading(format!("{}  Brickadia egui theme", icons::MUSIC));
        ui.label("A sandbox of the themed widgets, toggles and Font Awesome icons.");
        ui.add_space(8.0);

        widgets::section(ui, "Buttons", |ui| self.buttons(ui));
        ui.add_space(10.0);
        widgets::section(ui, "Toggles & checkboxes", |ui| self.toggles(ui));
        ui.add_space(10.0);
        widgets::section(ui, "Radios", |ui| self.radios(ui));
        ui.add_space(10.0);
        widgets::section(ui, "Combo / dropdown", |ui| self.dropdown(ui));
        ui.add_space(10.0);
        widgets::section(ui, "Inputs", |ui| self.inputs(ui));
        ui.add_space(10.0);
        widgets::section(ui, "Settings table", |ui| self.settings(ui));
        ui.add_space(10.0);
        widgets::section(ui, "Loading", |ui| self.loading_indicator(ui));
        ui.add_space(10.0);
        widgets::section(ui, "Outlined title", |ui| self.title_demo(ui));
        ui.add_space(10.0);
        widgets::section(ui, "Icons", |ui| self.icon_gallery(ui));
    }

    /// The five semantic button variants — each recolors but never grows on
    /// hover and sinks its label on press — followed by the red rounded-square
    /// [`widgets::danger_icon`] for an icon-only destructive action.
    fn buttons(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            widgets::primary(ui, format!("{}  Generate", icons::DOWNLOAD))
                .on_hover_text("primary — the green go button");
            widgets::info(ui, format!("{}  Pick file", icons::FOLDER_OPEN));
            widgets::neutral(ui, format!("{}  Back", icons::ARROW_LEFT));
            widgets::warn(ui, format!("{}  Careful", icons::CIRCLE_XMARK));
            widgets::danger(ui, format!("{}  Remove", icons::XMARK));
            widgets::danger_icon(ui, icons::XMARK)
                .on_hover_text("danger_icon — square icon-only remove");
        });
        ui.label(RichText::new("Hover doesn't grow them; click sinks the label.").small());
    }

    /// The sliding on/off [`widgets::toggle`] switch (knob slides red ✖ → green
    /// ✔) beside the square [`widgets::checkbox`] for multi-select lists. Both
    /// are drop-ins for `ui.checkbox` whose `changed()` fires on click.
    fn toggles(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new("toggle — on/off switch").small());
                widgets::toggle(ui, &mut self.loop_on, "Loop");
                widgets::toggle(ui, &mut self.in_chip, "In microchip");
                widgets::toggle(ui, &mut self.controls, "Control buttons");
            });
            ui.add_space(24.0);
            ui.vertical(|ui| {
                ui.label(RichText::new("checkbox — multi-select").small());
                widgets::checkbox(ui, &mut self.checks[0], "Red");
                widgets::checkbox(ui, &mut self.checks[1], "Green");
                widgets::checkbox(ui, &mut self.checks[2], "Blue");
            });
        });
    }

    /// The larger hand-painted [`widgets::radio`] (egui's own is tiny), sized to
    /// the control row. Edits the shared [`Wave`], so the dropdown below tracks
    /// whatever is picked here.
    fn radios(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            for w in Wave::ALL {
                widgets::radio(ui, &mut self.wave, w, w.label());
            }
        });
    }

    /// The themed [`widgets::combo`] dropdown (Font Awesome caret, pointer
    /// cursor) with its options added via [`widgets::combo_item`] so they get a
    /// pointer cursor too. Drives the same [`Wave`] as the radios above.
    fn dropdown(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Waveform");
            widgets::combo(ui, "sandbox_wave", self.wave.label(), 120.0, |ui| {
                for w in Wave::ALL {
                    widgets::combo_item(ui, &mut self.wave, w, w.label());
                }
            });
        });
    }

    /// Stock egui numeric inputs wearing the theme, plus the pointer-cursor
    /// [`widgets::slider`] and the full-height [`widgets::text_field`] (egui's
    /// bare `TextEdit` ignores the control height, so panes always use the
    /// helper). The `DragValue` and `ProgressBar` have no bespoke wrapper — they
    /// pick up the palette straight from [`super::install`].
    fn inputs(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Gain");
            widgets::slider(ui, Slider::new(&mut self.slider, 0.0..=1.0));
        });
        ui.horizontal(|ui| {
            ui.label("Count");
            ui.add(DragValue::new(&mut self.drag).range(0..=32));
        });
        ui.horizontal(|ui| {
            ui.label("Output");
            widgets::text_field(ui, &mut self.text, "file name");
        });
        ui.add(egui::ProgressBar::new(self.slider).show_percentage());
        ui.hyperlink_to(
            format!("{}  brickadia-community/heightmap2brz", icons::UP_RIGHT_FROM_SQUARE),
            "https://github.com/brickadia-community/heightmap2brz",
        );
    }

    /// The full-width, square-striped [`widgets::settings_table`] every
    /// generator's settings pane is built from: a padded label column and a
    /// control column, rows added via [`SettingsTable::row`](widgets::SettingsTable::row)
    /// (and [`row_hover`](widgets::SettingsTable::row_hover) for a label
    /// tooltip). Reuses the same demo state as the cards above so the controls
    /// are live.
    ///
    /// Below the table, the two lower-level primitives the table is assembled
    /// from — [`widgets::full_bleed`] (pull content out to the card edges) and
    /// [`widgets::cell_label`] (a label re-inset by `CELL_PAD`) — are shown
    /// directly, both to cover them in the gallery and to make the striped-row
    /// trick legible.
    fn settings(&mut self, ui: &mut Ui) {
        widgets::settings_table(ui, |ui, t| {
            t.row(ui, "Gain", |ui| {
                widgets::slider(ui, Slider::new(&mut self.slider, 0.0..=1.0));
            });
            t.row(ui, "Count", |ui| {
                ui.add(DragValue::new(&mut self.drag).range(0..=32));
            });
            t.row(ui, "Waveform", |ui| {
                widgets::combo(ui, "sandbox_table_wave", self.wave.label(), 120.0, |ui| {
                    for w in Wave::ALL {
                        widgets::combo_item(ui, &mut self.wave, w, w.label());
                    }
                });
            });
            t.row_hover(
                ui,
                "Output",
                Some("Where the save is written, relative to the exe."),
                |ui| {
                    widgets::text_field(ui, &mut self.text, "file name");
                },
            );
            t.row(ui, "Loop", |ui| {
                widgets::toggle(ui, &mut self.loop_on, "");
            });
        });

        ui.add_space(6.0);
        // `full_bleed` cancels the section body's `CELL_PAD`, so this fill
        // reaches the card edges exactly as a striped table row does;
        // `cell_label` pads the label text back in so it stays inset while the
        // stripe bleeds full-width.
        widgets::full_bleed(ui, |ui| {
            Frame::new().fill(super::SURFACE_STRIPE).show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                widgets::cell_label(ui, "full_bleed + cell_label: full-width stripe, inset label.");
            });
        });
    }

    /// The inline [`widgets::loading`] indicator (spinner + label) a pane shows
    /// while a background render runs, gated behind a [`widgets::toggle`]
    /// standing in for the job. `loading` requests a repaint every frame — the
    /// same thing that keeps a real pane polling its promise — so the spinner
    /// keeps turning while `busy`.
    fn loading_indicator(&mut self, ui: &mut Ui) {
        widgets::toggle(ui, &mut self.busy, "Simulate a running job");
        ui.add_space(4.0);
        if self.busy {
            widgets::loading(ui, "Generating...");
        } else {
            widgets::primary(ui, format!("{}  Generate", icons::DOWNLOAD));
        }
    }

    /// The outlined [`widgets::title`] — a heading painted over a thick, rounded
    /// dark-blue outline. Its doc reserves it for titles drawn over busy image
    /// backdrops (where the outline earns its keep); over the solid card here it
    /// reads heavier than a plain `Ui::heading`, and it is shown only so the
    /// gallery covers every widget.
    fn title_demo(&self, ui: &mut Ui) {
        widgets::title(ui, format!("{}  Heightmap", icons::MOUNTAIN));
        ui.label(
            RichText::new("Reserved for titles over image backdrops; use Ui::heading on solid panels.")
                .small(),
        );
    }

    /// Every declared icon at a readable size, with its Font Awesome name.
    fn icon_gallery(&self, ui: &mut Ui) {
        Grid::new("theme_sandbox_icons")
            .num_columns(4)
            .spacing([18.0, 6.0])
            .show(ui, |ui| {
                for (i, (name, glyph)) in icons::ALL.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(icons::icon(glyph).size(20.0));
                        ui.label(RichText::new(*name).weak());
                    });
                    if i % 2 == 1 {
                        ui.end_row();
                    }
                }
            });
    }
}
