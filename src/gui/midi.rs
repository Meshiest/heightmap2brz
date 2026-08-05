//! The MIDI2Brick pane: pick a Standard MIDI File, assign a synth tone to each
//! discovered instrument, and generate a swarm of wired, pitched speaker bricks
//! that play the piece back -- with an audible offline preview.
//!
//! Mirrors [`crate::gui::audio::AudioApp`]'s machinery (a `poll_promise` picker,
//! a background render thread on native, [`deliver_world`]) but is much smaller:
//! a MIDI carries its own pitches, so there is no FFT, no band grid and no
//! envelope, only the shared spatialization/playback controls plus a per-note
//! gain, a polyphony cap and one tone per instrument. Parsing and building are
//! both fast, so this pane shows a plain spinner rather than the audio pane's
//! full progress bar and cancel machinery.
//!
//! Picking is native-only, exactly like the audio pane: the browser build draws
//! the settings and says plainly that loading a file needs the desktop. The
//! [`crate::gui::midi_preview::Preview`] backend, by contrast, is real on both
//! targets (rodio on desktop, Web Audio in the browser).
use crate::{
    audio::{speakers::build_midi_event_world, track::SynthWave},
    gui::{
        SharedOptions,
        midi_preview::{self, Preview},
        util::{bound_pane_width, deliver_world, out_file_warning_row, refuse_bad_out_file, save_destination_row},
    },
    gui::util::pick_midi_bytes,
    gui::theme::{icons, widgets},
    midi::{Instrument, MidiOptions, MidiSummary, ToneAssignment, analyze_midi, discover, preview::synthesize},
};
use egui::Ui;
use log::{error, info};
use poll_promise::Promise;

/// A picked and parsed MIDI file: its display name, the raw bytes (kept for the
/// build and the preview, both of which re-parse from them), and what `discover`
/// found. The browser reads an uploaded file into these same bytes, so this is
/// live on both targets.
struct PickedMidi {
    name: String,
    bytes: Vec<u8>,
    instruments: Vec<Instrument>,
    summary: MidiSummary,
}

pub struct MidiApp {
    /// The picked file, or `None`. Never `Some` on wasm (no picker), which is
    /// also why `draw_submit` never offers Generate or Preview there.
    input: Option<PickedMidi>,
    /// The in-flight file picker (native dialog or browser upload), if any.
    pending_pick: Option<Promise<Option<(String, Vec<u8>)>>>,
    /// One tone per discovered instrument, resized (to all-Sine) each time a new
    /// file is parsed. Kept beside `opts` rather than inside its
    /// [`ToneAssignment`] so the per-row combos can bind a plain slice; the
    /// assignment is assembled from it in [`Self::midi_opts`].
    tones: Vec<SynthWave>,
    /// One volume multiplier per discovered instrument (0..=1), parallel to
    /// `tones`, so one part (say the bass) can be turned down. Resized to all
    /// 1.0 when a file is parsed; fed to [`MidiOptions::instrument_volumes`].
    volumes: Vec<f32>,
    /// Which instrument rows are checked, parallel to `tones`. Used by the bulk
    /// "apply to selected" control.
    selected: Vec<bool>,
    /// The tone the bulk control assigns to all / selected rows.
    bulk_tone: SynthWave,
    /// Every numeric/toggle setting, in the pipeline's own struct. Its own
    /// `tones` field is unused -- [`Self::midi_opts`] overrides it from `tones`.
    opts: MidiOptions,
    /// The in-flight build, if any. `Some` from the moment `generate` spawns the
    /// worker until `poll_generate` observes it is ready; while `Some`,
    /// `draw_submit` shows a spinner instead of the button.
    pending_generate: Option<Promise<Result<(), String>>>,
    /// The in-flight preview synthesis, if any. Synthesized off the UI thread so
    /// a long preview never freezes the app; `poll_preview` plays it when ready
    /// and `draw_submit` shows a spinner while it is `Some`.
    pending_preview: Option<Promise<Result<Vec<f32>, String>>>,
    /// The audible preview device (rodio on desktop, Web Audio in the browser).
    preview: Preview,
    /// Preview playback volume, 0..=1.
    preview_volume: f32,
    /// The egui-clock time the current preview started, and its length in
    /// seconds -- together they drive the progress bar. `None` when idle.
    preview_started: Option<f64>,
    preview_len: f32,
    /// The synthesized PCM, retained so scrubbing can re-play from any sample
    /// (rodio/Web Audio have no seek). `None` until a preview is ready; while it
    /// is `Some` the scrubber stays on screen even after playback ends, so the
    /// preview can be re-scrubbed without re-synthesizing.
    preview_pcm: Option<Vec<f32>>,
    /// The playhead's resting fraction (0..=1) to draw when nothing is actively
    /// playing -- 1.0 after a preview runs out, or wherever Stop / a scrub left
    /// it. While `preview_started` is `Some` it tracks the live playhead.
    preview_pos: f32,
    /// While the user is dragging the progress bar, the scrubbed fraction
    /// (0..=1) to show; the actual re-play happens on release.
    preview_scrub: Option<f32>,
}

impl Default for MidiApp {
    fn default() -> Self {
        Self {
            input: None,
            pending_pick: None,
            tones: Vec::new(),
            volumes: Vec::new(),
            selected: Vec::new(),
            bulk_tone: SynthWave::Sine,
            opts: MidiOptions::default(),
            pending_generate: None,
            pending_preview: None,
            preview: Preview::default(),
            preview_volume: 1.0,
            preview_started: None,
            preview_len: 0.0,
            preview_pcm: None,
            preview_pos: 0.0,
            preview_scrub: None,
        }
    }
}

impl MidiApp {
    /// The build options for the current UI state -- the single source for both
    /// `generate` and `preview_play`, so a preview can never describe a
    /// different piece from the one Generate builds. Everything but the tone map
    /// comes straight from `opts`; the tones are assembled from the per-row
    /// slice as a [`ToneAssignment::PerInstrument`].
    fn midi_opts(&self) -> MidiOptions {
        MidiOptions {
            tones: ToneAssignment::PerInstrument(self.tones.clone()),
            instrument_volumes: self.volumes.clone(),
            ..self.opts.clone()
        }
    }

    /// Poll the in-flight picker: on a resolved upload, parse the (small) bytes
    /// right here -- a `.mid` is kilobytes and `discover` is cheap, so there is
    /// nothing to gain from a second worker thread. A parse failure is logged and
    /// clears the input rather than surfacing half a file. Works on both targets:
    /// native and browser both hand back `(name, bytes)`.
    fn poll_picks(&mut self) {
        let Some(promise) = self.pending_pick.take() else {
            return;
        };
        match promise.try_take() {
            Ok(result) => {
                if let Some((name, bytes)) = result {
                    self.load_midi(name, bytes);
                }
            }
            Err(promise) => self.pending_pick = Some(promise),
        }
    }

    /// Parse the picked file's bytes, storing the result (or logging and clearing
    /// on failure). Split out of `poll_picks` so the parse path is one place.
    fn load_midi(&mut self, name: String, bytes: Vec<u8>) {
        match discover(&bytes) {
            Ok((instruments, summary)) => {
                info!(
                    "Loaded MIDI {name}: format {}, {} instrument(s), {} note(s), {:.1}s",
                    summary.format,
                    instruments.len(),
                    summary.total_notes,
                    summary.duration_s,
                );
                // One tone slot per instrument, all Sine (the default); full
                // volume; none selected.
                self.tones = vec![SynthWave::Sine; instruments.len()];
                self.volumes = vec![1.0; instruments.len()];
                self.selected = vec![false; instruments.len()];
                self.reset_preview();
                self.input = Some(PickedMidi { name, bytes, instruments, summary });
            }
            Err(e) => {
                error!("could not parse {name}: {e}");
                self.input = None;
            }
        }
    }

    /// Poll the in-flight build. The worker already logged any failure through
    /// `log::error!`, so once the promise resolves there is nothing left but to
    /// drop it, which frees `draw_submit` to show the button again.
    fn poll_generate(&mut self) {
        if let Some(promise) = self.pending_generate.take() {
            match promise.try_take() {
                Ok(_) => {}
                Err(promise) => self.pending_generate = Some(promise),
            }
        }
    }

    /// The always-visible settings: destination plus the shared
    /// spatialization/playback controls.
    fn draw_settings(&mut self, ui: &mut Ui, shared: &mut SharedOptions) {
        ui.label(
            "Turn a MIDI file into a cluster of wired, pitched speaker bricks: each instrument \
             becomes a small bank of speakers that plays its notes back through a chosen synth \
             tone, driven by an in-chip clock.",
        );

        widgets::settings_table(ui, |ui, t| {
            save_destination_row(t, ui, shared);
            out_file_warning_row(t, ui, &shared.out_file);
            self.draw_control_rows(t, ui);
        });
    }

    fn draw_control_rows(&mut self, t: &mut widgets::SettingsTable, ui: &mut Ui) {
        t.row_hover(
            ui,
            "Gain",
            Some("Multiplier on each note's velocity-derived volume, applied and then CLAMPED at 1.0 -- a way to make a render quieter, never louder."),
            |ui| {
                widgets::slider(ui, egui::Slider::new(&mut self.opts.gain, 0.0..=1.0));
            },
        );
        t.row_hover(
            ui,
            "Playback Rate",
            Some("Speed multiplier baked into the clock: 1.0 = the file's own tempo, 2.0 = double speed, 0.5 = half. The generated Rate pin still overrides it at runtime."),
            |ui| {
                widgets::slider(ui, egui::Slider::new(&mut self.opts.playback_rate, 0.25..=4.0).logarithmic(true));
            },
        );
        t.row_hover(
            ui,
            "Polyphony Cap",
            Some("The most speakers any one instrument gets, however many notes it plays at once. Overflow steals the oldest sounding note. Each speaker is a brick and a stream, so this bounds the build size of a dense instrument."),
            |ui| {
                widgets::slider(ui, egui::Slider::new(&mut self.opts.polyphony_cap, 1..=32));
            },
        );
        t.row_hover(
            ui,
            "Playback",
            Some("Loop: repeat the piece forever (the default). Off: play through once and stop on the last note. Costs nothing either way."),
            |ui| {
                widgets::toggle(ui, &mut self.opts.loop_playback, "Loop");
            },
        );
        t.row_hover(
            ui,
            "Controls",
            Some("Pre-generate physical Pause/Restart/Resume buttons on the main grid, wired into the clock so the render is controllable out of the box. Off means you wire the clock's control pins yourself."),
            |ui| {
                widgets::toggle(ui, &mut self.opts.control_buttons, "Control buttons");
            },
        );
        t.row_hover(
            ui,
            "Placement",
            Some("Where the speaker cluster goes. Beside the chip on the main grid (the default), or IN the microchip's own inner grid, which makes the whole audio device one portable microchip. Moves the bricks only; the sound is unchanged."),
            |ui| {
                widgets::toggle(ui, &mut self.opts.speakers_in_chip, "In microchip");
            },
        );
        t.row_hover(
            ui,
            "Inner Radius",
            Some("The radius inside which there is NO distance attenuation, in units (10 units = 1 brick). Baked onto every speaker."),
            |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.opts.inner_radius)
                        .speed(10.0)
                        .range(1.0..=100_000.0),
                );
            },
        );
        t.row_hover(
            ui,
            "Max Distance",
            Some("Where the sound stops, in units (10 units = 1 brick). Must be larger than Inner Radius; the builder refuses an inverted pair rather than building a silent save."),
            |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.opts.max_distance)
                        .speed(10.0)
                        .range(1.0..=1_000_000.0),
                );
            },
        );
    }

    fn draw_input(&mut self, ui: &mut Ui) {
        ui.label("Pick a Standard MIDI File (.mid / .midi).");

        // The picker reads bytes on both targets: a native file dialog, or a
        // browser file-upload blob read into memory.
        let picking = self.pending_pick.is_some();
        ui.horizontal_wrapped(|ui| {
            if widgets::info(ui, format!("{}  Pick MIDI file", icons::FOLDER_OPEN)).clicked()
                && !picking
            {
                self.pending_pick = Some(pick_midi_bytes());
            }
            if picking {
                ui.spinner();
                ui.label("reading...");
            }
        });

        let mut clear = false;
        match &self.input {
            None => {
                ui.label("No MIDI file selected.");
            }
            Some(input) => {
                ui.horizontal_wrapped(|ui| {
                    if widgets::danger_icon(ui, icons::XMARK).clicked() {
                        clear = true;
                    }
                    ui.label(&input.name);
                });
            }
        }
        if clear {
            self.input = None;
            self.tones.clear();
            self.volumes.clear();
            self.selected.clear();
            self.reset_preview();
        }

        self.draw_summary(ui);
        self.draw_instrument_table(ui);
    }

    /// The whole-file facts, once a file is parsed.
    fn draw_summary(&self, ui: &mut Ui) {
        let Some(input) = &self.input else {
            return;
        };
        let s = &input.summary;
        ui.add_space(4.0);
        ui.label(format!(
            "Format {} -- {} track(s), {:.1}s, {:.0} bpm, {} note(s){}",
            s.format,
            s.track_count,
            s.duration_s,
            s.initial_bpm,
            s.total_notes,
            if s.has_percussion {
                " -- has percussion (the percussion channel is not built)"
            } else {
                ""
            },
        ));
    }

    /// A proper table of discovered instruments -- a checkbox, the name, note
    /// and polyphony counts, and a per-row tone dropdown -- with a bulk control
    /// above it to set every row (or just the checked ones) to one tone.
    ///
    /// The per-row facts are collected into owned values first so the immutable
    /// borrow of `input` is released before the table body mutably touches
    /// `tones`/`selected`. Columns have fixed widths and the name column CLIPS
    /// rather than wraps: the pane forces `Wrap` by default, which is what
    /// squished the old single-column layout on a narrow window.
    fn draw_instrument_table(&mut self, ui: &mut Ui) {
        use egui_extras::{Column, TableBuilder};

        struct Row {
            label: String,
            notes: usize,
            polyphony: usize,
            dropped: usize,
        }
        let rows: Vec<Row> = match &self.input {
            Some(input) => input
                .instruments
                .iter()
                .map(|inst| Row {
                    label: inst.label.clone(),
                    notes: inst.note_count,
                    polyphony: inst.max_polyphony,
                    dropped: inst.dropped_notes,
                })
                .collect(),
            None => return,
        };
        if rows.is_empty() {
            return;
        }

        ui.add_space(6.0);
        ui.label("Instruments");

        // Bulk control: check rows, pick a tone, apply it to the checked ones or
        // to every instrument at once.
        ui.horizontal_wrapped(|ui| {
            if widgets::neutral(ui, "Select all").clicked() {
                self.selected.iter_mut().for_each(|s| *s = true);
            }
            if widgets::neutral(ui, "None").clicked() {
                self.selected.iter_mut().for_each(|s| *s = false);
            }
            ui.label("Set");
            widgets::combo(ui, "midi_bulk_tone", self.bulk_tone.name(), 100.0, |ui| {
                for w in SynthWave::ALL {
                    widgets::combo_item(ui, &mut self.bulk_tone, w, w.name());
                }
            });
            if widgets::neutral(ui, "For selected").clicked() {
                for (i, tone) in self.tones.iter_mut().enumerate() {
                    if self.selected.get(i).copied().unwrap_or(false) {
                        *tone = self.bulk_tone;
                    }
                }
            }
            if widgets::neutral(ui, "For all").clicked() {
                self.tones.iter_mut().for_each(|t| *t = self.bulk_tone);
            }
        });
        ui.add_space(2.0);

        let row_h = 40.0;
        // The stripes bleed to the card edges, but the outer columns' CONTENT
        // gets an inset so the checkbox and the volume slider aren't jammed
        // against the card's left/right edge.
        let pad = widgets::CELL_PAD;
        // Full-bleed to the card edges (like the settings tables); the name
        // column takes the slack so rows fill the width.
        widgets::full_bleed(ui, |ui| {
        TableBuilder::new(ui)
            .striped(true)
            // Fill the card width: don't shrink horizontally, and let the name
            // column take up the slack.
            .auto_shrink([false, true])
            // No nested scrollbar: grow to fit all rows and let the page scroll.
            .vscroll(false)
            .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
            .column(Column::auto()) // checkbox
            .column(Column::remainder().at_least(60.0).clip(true)) // name
            .column(Column::auto().at_least(44.0)) // notes
            .column(Column::auto().at_least(40.0)) // poly
            .column(Column::auto().at_least(100.0)) // tone
            .column(Column::auto().at_least(110.0)) // volume
            .header(20.0, |mut header| {
                header.col(|ui| {
                    ui.add_space(pad);
                });
                header.col(|ui| {
                    ui.strong("Instrument");
                });
                header.col(|ui| {
                    ui.strong("Notes");
                });
                header.col(|ui| {
                    ui.strong("Poly");
                });
                header.col(|ui| {
                    ui.strong("Tone");
                });
                header.col(|ui| {
                    ui.strong("Vol");
                });
            })
            .body(|mut body| {
                for (i, row) in rows.iter().enumerate() {
                    body.row(row_h, |mut tr| {
                        tr.col(|ui| {
                            ui.add_space(pad);
                            if let Some(sel) = self.selected.get_mut(i) {
                                widgets::checkbox(ui, sel, "");
                            }
                        });
                        tr.col(|ui| {
                            ui.add(egui::Label::new(row.label.as_str()).truncate())
                                .on_hover_text(row.label.as_str());
                        });
                        tr.col(|ui| {
                            let notes = ui.label(row.notes.to_string());
                            if row.dropped > 0 {
                                notes.on_hover_text(format!(
                                    "{} note(s) outside the playable pitch range, not built",
                                    row.dropped
                                ));
                            }
                        });
                        tr.col(|ui| {
                            ui.label(row.polyphony.to_string());
                        });
                        tr.col(|ui| {
                            if let Some(tone) = self.tones.get_mut(i) {
                                widgets::combo(ui, format!("midi_tone_{i}"), tone.name(), 90.0, |ui| {
                                    for w in SynthWave::ALL {
                                        widgets::combo_item(ui, tone, w, w.name());
                                    }
                                });
                            }
                        });
                        tr.col(|ui| {
                            if let Some(vol) = self.volumes.get_mut(i) {
                                ui.spacing_mut().slider_width = 60.0;
                                widgets::slider(ui, egui::Slider::new(vol, 0.0..=1.0));
                            }
                            ui.add_space(pad);
                        });
                    });
                }
            });
        });
    }

    fn draw_submit(&mut self, ui: &mut Ui, shared: &mut SharedOptions) {
        let has_input = self.input.is_some();

        // Preview needs a parsed file but not a valid destination -- it never
        // writes one -- so it is offered independently of the Generate gate.
        if has_input {
            let now = ui.input(|i| i.time);
            self.poll_preview(now);
            ui.horizontal_wrapped(|ui| {
                // A real vertical gap between wrapped lines (and horizontal gap
                // between items). Wrapped lines already center their items.
                ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
                if self.pending_preview.is_some() {
                    widgets::loading(ui, "Synthesizing preview...");
                } else if widgets::info(ui, format!("{}  Preview", icons::PLAY))
                    .on_hover_text(
                        "Synthesize the first Preview Length seconds and play them here -- a \
                         rough approximation of the notes, timing and tones, not the game's \
                         exact synth or spatialization.",
                    )
                    .clicked()
                {
                    self.preview_play();
                }
                if widgets::neutral(ui, format!("{}  Stop", icons::STOP)).clicked() {
                    // Freeze the playhead where it is (the scrubber stays up,
                    // resting at `preview_pos`); it does not clear the preview.
                    self.preview.stop();
                    self.preview_started = None;
                }
                ui.separator();
                ui.label("Volume");
                if ui
                    .add(egui::Slider::new(&mut self.preview_volume, 0.0..=1.0))
                    .changed()
                {
                    self.preview.set_volume(self.preview_volume);
                }
                ui.label("Length").on_hover_text(
                    "Seconds of the piece the Preview synthesizes and plays. 0 = the whole file. \
                     Does NOT affect the generated build.",
                );
                ui.add(
                    egui::DragValue::new(&mut self.opts.preview_seconds)
                        .speed(1.0)
                        .suffix(" s")
                        .range(0.0..=600.0),
                );
            });

            // Seekable progress bar. Shown as long as a preview EXISTS
            // (`preview_pcm`), not just while it plays, so playback ending
            // leaves the bar on screen to keep scrubbing. Click or drag to
            // scrub (the actual re-play happens on release).
            if self.preview_pcm.is_some() {
                // Advance the playhead while audio is actually playing; when it
                // runs out, rest at the end and stop repainting -- but keep the
                // bar so it can still be scrubbed.
                let mut playing = false;
                if let Some(start) = self.preview_started {
                    let elapsed = ((now - start) as f32).max(0.0);
                    if self.preview_len > 0.0 && elapsed >= self.preview_len {
                        self.preview_started = None;
                        self.preview_pos = 1.0;
                    } else {
                        self.preview_pos = if self.preview_len > 0.0 {
                            (elapsed / self.preview_len).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        playing = true;
                    }
                }

                // While dragging, show the drag position instead of the playhead.
                let frac = self.preview_scrub.unwrap_or(self.preview_pos);
                let resp = ui
                    .add(
                        egui::ProgressBar::new(frac)
                            .desired_width(ui.available_width().min(320.0))
                            .text(format!("{:.0} / {:.0} s", frac * self.preview_len, self.preview_len)),
                    )
                    .interact(egui::Sense::click_and_drag())
                    .on_hover_cursor(egui::CursorIcon::ResizeHorizontal);

                if let Some(pos) = resp.interact_pointer_pos() {
                    let f = ((pos.x - resp.rect.left()) / resp.rect.width().max(1.0)).clamp(0.0, 1.0);
                    if resp.dragged() {
                        self.preview_scrub = Some(f);
                    }
                    if resp.drag_stopped() || resp.clicked() {
                        self.preview_seek(f, now);
                        self.preview_scrub = None;
                    }
                }

                if playing {
                    ui.ctx().request_repaint();
                }
            }
            ui.add_space(4.0);
        }

        // A build already in flight: a spinner, no button, so a second click
        // cannot start a second one.
        if self.pending_generate.is_some() {
            widgets::loading(ui, "Generating...");
            return;
        }

        // A destination that cannot be written is refused before the button is
        // offered -- the same red label the settings grid shows above.
        if refuse_bad_out_file(ui, &shared.out_file) {
            return;
        }

        if has_input {
            if widgets::primary(ui, format!("{}  Generate midi2brick save", icons::DOWNLOAD)).clicked() {
                self.generate(shared);
            }
        } else {
            ui.label("Pick a MIDI file to continue...");
        }
    }

    /// Stop any playback and forget the current preview entirely. The scrubber
    /// is drawn only while `preview_pcm` is `Some`, so this also hides it --
    /// used when the file changes (a stale scrubber for the old file would keep
    /// re-playing the old audio).
    fn reset_preview(&mut self) {
        self.preview.stop();
        self.preview_started = None;
        self.preview_pcm = None;
        self.preview_scrub = None;
        self.preview_pos = 0.0;
    }

    /// On click: schedule + synthesize the first Preview Length seconds on a
    /// background thread (native) or inline (wasm), so a long preview never
    /// freezes the UI. `poll_preview` plays the PCM once it is ready.
    fn preview_play(&mut self) {
        if self.pending_preview.is_some() {
            return;
        }
        let Some(input) = &self.input else {
            return;
        };
        let bytes = input.bytes.clone();
        let opts = self.midi_opts();
        let work = move || -> Result<Vec<f32>, String> {
            let score = analyze_midi(&bytes, &opts)?;
            Ok(synthesize(
                &score,
                midi_preview::SAMPLE_RATE,
                opts.preview_seconds,
                opts.playback_rate,
            ))
        };

        let (sender, promise) = Promise::new();
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || sender.send(work()));
        #[cfg(target_arch = "wasm32")]
        sender.send(work());
        self.pending_preview = Some(promise);
    }

    /// Poll the in-flight preview synthesis; when it resolves, play it (a
    /// failure or a silent piece is logged, not surfaced). `now` seeds the
    /// playback progress bar.
    fn poll_preview(&mut self, now: f64) {
        let Some(promise) = self.pending_preview.take() else {
            return;
        };
        match promise.try_take() {
            Ok(Ok(pcm)) if !pcm.is_empty() => {
                self.preview_len = pcm.len() as f32 / midi_preview::SAMPLE_RATE as f32;
                self.preview_started = Some(now);
                self.preview_pos = 0.0;
                self.preview.set_volume(self.preview_volume);
                self.preview.play(&pcm, midi_preview::SAMPLE_RATE);
                self.preview_pcm = Some(pcm); // retained for scrubbing
            }
            Ok(Ok(_)) => info!("Nothing to preview -- the file scheduled no playable notes"),
            Ok(Err(e)) => error!("{e}"),
            Err(promise) => self.pending_preview = Some(promise),
        }
    }

    /// Scrub: re-play the retained PCM from fraction `f` (0..=1) and re-anchor
    /// the progress clock so the bar tracks the new position (rodio/Web Audio
    /// have no seek, so a scrub is a fresh play of the tail).
    fn preview_seek(&mut self, f: f32, now: f64) {
        let tail = {
            let Some(pcm) = &self.preview_pcm else {
                return;
            };
            if pcm.is_empty() {
                return;
            }
            let offset = ((f * pcm.len() as f32) as usize).min(pcm.len() - 1);
            pcm[offset..].to_vec()
        };
        self.preview.set_volume(self.preview_volume);
        self.preview.play(&tail, midi_preview::SAMPLE_RATE);
        self.preview_pos = f;
        self.preview_started = Some(now - (f * self.preview_len) as f64);
    }

    /// On click: schedule -> build the speaker world -> deliver. The same path
    /// as `main.rs`'s midi branch, dispatched on the same [`MidiOptions`].
    /// Native runs it on a background thread; the browser (no threads) runs it
    /// inline, then `deliver_world` triggers a download of the `.brz`.
    fn generate(&mut self, shared: &SharedOptions) {
        if self.pending_generate.is_some() {
            return error!("a build is already in progress");
        }
        let Some(input) = &self.input else {
            return error!("pick a MIDI file first");
        };

        // Owned clones so nothing borrows `self` or `shared` inside the worker.
        let bytes = input.bytes.clone();
        let opts = self.midi_opts();
        let out_file = shared.out_file.clone();
        let out_clipboard = shared.out_clipboard;

        let work = move || -> Result<(), String> {
            let score = analyze_midi(&bytes, &opts)?;
            info!(
                "Scheduled {} speaker(s) over {:.1}s",
                score.voices.len(),
                score.duration_s
            );
            let world = build_midi_event_world(&score, &opts)?;
            info!("Writing Save to {out_file}");
            deliver_world(&world, &out_file, out_clipboard)?;
            info!("Done!");
            Ok(())
        };

        let (sender, promise) = Promise::new();

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            let result = work();
            if let Err(e) = &result {
                error!("{e}");
            }
            sender.send(result);
        });

        #[cfg(target_arch = "wasm32")]
        {
            let result = work();
            if let Err(e) = &result {
                error!("{e}");
            }
            sender.send(result);
        }

        self.pending_generate = Some(promise);
    }

    pub fn draw(&mut self, ui: &mut Ui, shared: &mut SharedOptions) {
        bound_pane_width(ui);
        self.poll_picks();
        self.poll_generate();

        // File selection above the settings.
        widgets::section(ui, "Source", |ui| self.draw_input(ui));
        ui.add_space(10.0);
        widgets::section(ui, "Settings", |ui| self.draw_settings(ui, shared));

        // A picker resolving on its own thread (or an async browser upload) has
        // nothing else to wake the event loop, so without this the picked file
        // would only appear on the user's next mouse move.
        if self.pending_pick.is_some() {
            ui.ctx().request_repaint();
        }
    }

    /// The fixed footer: preview + Generate (progress lives here too).
    pub fn draw_footer(&mut self, ui: &mut Ui, shared: &mut SharedOptions) {
        bound_pane_width(ui);
        self.draw_submit(ui, shared);
    }
}
