//! The Audio2Brick pane: pick a song (or a video to pull a track out of),
//! tune the analysis, and generate a swarm of wired, pitched speaker bricks.
//! Follows [`crate::gui::video::VideoApp`]'s shape (option state, `poll_promise`
//! picking, background render thread over an `mpsc` channel, cancel flag,
//! [`deliver_world_unless_cancelled`]), but stores an [`AudioOptions`] directly
//! and binds widgets to its own fields, rather than assembling one from loose
//! fields like the video pane does. `--max-voices` means two different things
//! depending on [`AudioApp::mode`]; see [`AudioApp::max_voices_label`].
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::{
    audio::{
        AudioMode,
        backend::{AudioBackend, DownloadConsent},
        cost::{self, AudioCost},
        presets::AudioPreset,
        source::AudioInfo,
        speakers::{build_speaker_world, build_voice_world},
        track::{AudioOptions, SynthWave, analyze},
        voices::analyze_voices,
    },
    gui::{
        SharedOptions,
        util::{
            ChannelProgress as UtilChannelProgress, RenderMsg, bound_pane_width,
            deliver_world_unless_cancelled, draw_out_file_warnings, draw_progress_bar, note,
            refuse_bad_out_file, section, settings_grid,
        },
    },
    progress::Progress,
};
use egui::{Button, Color32, Ui};
use log::{error, info};
use poll_promise::Promise;

// The picker returns a PATH, and every audio decoder in this crate opens one
// -- so, exactly like the video pane's video source, the whole picking half of
// this pane is native-only. A browser file handle has no filesystem path, and
// a song is decoded TWICE (once to find the normalisation peak, once to emit),
// which is what the re-openable `AudioSource` handle exists for. On wasm the
// pane still draws its settings and its options-derived cost model -- those
// are real and useful -- and says plainly that rendering needs the desktop
// build, rather than offering a button that cannot work.
#[cfg(not(target_arch = "wasm32"))]
use crate::{
    audio::backend::open_audio_ensuring,
    gui::util::{FfmpegModal, draw_cancel_button, pick_audio_path},
    video::ffmpeg::{ensure_ffmpeg, ffmpeg_available},
};

/// STFT window sizes offered by the Window dropdown.
///
/// Powers of two only: the transform is a radix FFT, and every window the
/// listening sessions settled on ([`AudioPreset`]) is one of these. A
/// free-text box would let a value through that runs an order of magnitude
/// slower for no musical gain.
///
/// The range is deliberately wide at both ends because the right answer is
/// material-dependent and spans the whole list -- 2048 for speech (formants
/// move, and a long window smears them together) against 16384 for a music
/// box (a sustained partial deserves the pitch resolution).
const WINDOW_SIZES: [usize; 7] = [512, 1024, 2048, 4096, 8192, 16384, 32768];

/// Tonal subdivisions offered by the Subdiv dropdown.
///
/// **MULTIPLES OF 12 ONLY, and this is why the control is a dropdown rather
/// than a number box.** Only a multiple of 12 puts every band on a real
/// semitone; anything else is heard in game as out of tune (14 sharp, 18
/// flat), and the failure is musical rather than obvious. The renderer
/// rejects a bad value ([`crate::audio::track::check_subdiv`]) and so does
/// this pane's cost readout, but the widget itself is what makes a bad value
/// unreachable in the first place.
const SUBDIVS: [u32; 4] = [12, 24, 36, 48];

/// Range of the Bands slider.
///
/// **Static, and deliberately WIDER than any one subdivision allows.** The
/// pitch range holds 79 tonal bands at Subdiv 12, 159 at 24 and 319 at 48, so
/// a range that tracked the current subdivision would have to move the user's
/// band count when they changed subdivision -- and `AudioOptions::bands` says
/// in as many words that asking for more bands than the range holds is "an
/// error naming the maximum, never a silent clamp", because a clamp hands
/// back a different number of speakers than was asked for. So the slider
/// covers every value that is legal at SOME subdivision, and an impossible
/// combination surfaces as the renderer's own error in the cost readout,
/// which names the maximum and how to reach it.
///
/// Logarithmic, because the interesting part is the low end: the presets sit
/// at 60-72 and the full span at the default subdivision is 79.
const BANDS_RANGE: std::ops::RangeInclusive<usize> = 2..=320;

/// Largest speaker count the Max Voices / Speakers control offers.
///
/// 32 simultaneous emitters are VERIFIED to sound in game; above that is
/// untested, and if 48 comes back thinner than 32 that is engine voice
/// stealing rather than the analysis. The slider goes past 32 because bank
/// mode's value is only an upper bound (the peak gate usually binds first,
/// leaving 12-24 candidates), so a large number there costs nothing.
const MAX_SPEAKERS: usize = 128;

/// What the render thread tells the UI about its progress.
///
/// `Extra` is `Infallible`, unlike the video pane's (which carries a
/// `Preview` frame): `Progress::frame` is for showing the picture being
/// built, and an audio render has no picture. The analysis is the only phase
/// that reports at all, which is also the only phase that can be cancelled.
/// See [`crate::gui::util::RenderMsg`] for the shared Begin/Tick/Finish core.
type ProgressMsg = RenderMsg<std::convert::Infallible>;

/// Reports progress from the render thread to the UI over a channel. This
/// pane needs nothing beyond the shared core (see [`ProgressMsg`]'s doc), so
/// it uses [`crate::gui::util::ChannelProgress`] directly rather than
/// wrapping it -- contrast the video pane's `ChannelProgress`, which adds
/// throttled frame-preview reporting on top.
///
/// **A cancelled audio analysis returns a SHORT TRACK, not an error** --
/// `analyze` breaks out of its frame loop and normalises what it has, the
/// same way `build_brick_world` returns a partial world. That is why
/// [`deliver_world_unless_cancelled`] has to ask again afterwards: the build
/// succeeds, and writing its save would hand the user a silently truncated
/// song.
type ChannelProgress = UtilChannelProgress<std::convert::Infallible>;

/// The picked audio source: a path, its display name, and whatever the
/// decoder could tell us about it without decoding.
///
/// `info` is `None` until the background probe resolves, and stays `None` if
/// the probe failed -- a file the selected backend cannot open, or one whose
/// container carries no duration. That is a real answer and not an error: the
/// cost readout simply reports the part of the build that does not depend on
/// length (see [`AudioApp::draw_cost`]), and Generate still works, because
/// the render measures the real frame count as it analyses.
struct PickedAudio {
    path: std::path::PathBuf,
    /// Shown by `draw_input`, which is native-only -- there is no picker on
    /// wasm, so nothing there ever constructs one of these to display.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    name: String,
    info: Option<AudioInfo>,
}

/// The result of [`AudioApp::check_ffmpeg_consent`]: whether opening the
/// picked file needs a download the user has not consented to yet.
#[cfg(not(target_arch = "wasm32"))]
enum FfmpegCheck {
    /// ffmpeg either isn't needed for this file/backend or is already
    /// installed -- `generate` proceeds straight to spawning the worker.
    Ready,
    /// The selected backend needs ffmpeg and it is not installed --
    /// `generate` shows the consent modal instead of starting a worker.
    NeedsConsent,
    /// A real failure unrelated to ffmpeg (an unreadable file, a codec the
    /// builtin decoder does not carry under an explicit `Builtin`). Already
    /// logged via `log::error!`.
    Failed,
}

pub struct AudioApp {
    /// The picked file, or `None`. Never `Some` on wasm -- there is no picker
    /// there -- which is also why `draw_submit` can never offer Generate on
    /// that target.
    input: Option<PickedAudio>,
    #[cfg(not(target_arch = "wasm32"))]
    pending_pick: Option<Promise<Option<std::path::PathBuf>>>,
    /// The in-flight duration probe. Opens the picked file with the SELECTED
    /// backend and `DownloadConsent::Never` purely to read
    /// `AudioSource::info`, then drops it: the whole point is to give the
    /// cost readout a real frame count before a render is committed to. It
    /// runs on a background thread because even the cheap path (a container
    /// parse, or an `ffprobe` spawn) is not something to do on the UI thread.
    #[cfg(not(target_arch = "wasm32"))]
    pending_probe: Option<Promise<Result<AudioInfo, String>>>,
    /// Which decode backend opens the file. Not `#[cfg]`'d, unlike the video
    /// pane's: `AudioBackend` exists on every target (wasm's `Ffmpeg` refuses
    /// cleanly rather than failing to compile).
    backend: AudioBackend,
    /// Which audio stream to decode, 0 = first. Dual-audio releases carry the
    /// original language first and the dub second, so which one is "first" is
    /// a container-ordering accident. Honoured by the ffmpeg backend only.
    audio_track: usize,
    /// The shared ffmpeg download-consent + download modal. Bundles what used
    /// to be a pending-consent path and an in-flight download `Promise`; see
    /// [`FfmpegModal`].
    #[cfg(not(target_arch = "wasm32"))]
    modal: FfmpegModal,

    /// The in-flight render, if any. `Some` from the moment `generate` spawns
    /// the worker until `poll_generate` observes it is ready -- while `Some`,
    /// `draw_submit` hides the generate button so a second render cannot
    /// start.
    pending_generate: Option<Promise<Result<(), String>>>,
    /// The receiving half of the worker's progress channel. `Some` for
    /// exactly as long as `pending_generate` is.
    progress_rx: Option<std::sync::mpsc::Receiver<ProgressMsg>>,
    progress_label: String,
    progress_pos: u64,
    progress_total: Option<u64>,
    /// The in-flight render's cancel flag, set by the Cancel button and read
    /// by the worker's `ChannelProgress::is_cancelled`. A FRESH flag backs
    /// every render, so returning to idle can never leave the next one
    /// pre-cancelled.
    #[cfg(not(target_arch = "wasm32"))]
    cancel_flag: Option<Arc<AtomicBool>>,

    /// Which renderer runs. Decides what several controls MEAN, not just
    /// whether they are shown -- see the module doc.
    mode: AudioMode,
    /// The dropdown's current selection. Purely a record of what was last
    /// applied: selecting a preset writes into `opts` and then every one of
    /// those controls stays editable, so this does not track whether `opts`
    /// still matches it.
    preset: AudioPreset,
    /// **Every analysis setting, in the pipeline's own struct.** See the
    /// module doc for why this is not a pile of loose fields.
    opts: AudioOptions,
    /// The band count to restore when "Limit" is re-ticked, so unticking and
    /// re-ticking does not silently reset a tuned span. `opts.bands` is
    /// authoritative; this is only the shadow.
    bands_value: usize,
}

impl Default for AudioApp {
    fn default() -> Self {
        let opts = AudioOptions::default();
        Self {
            input: None,
            #[cfg(not(target_arch = "wasm32"))]
            pending_pick: None,
            #[cfg(not(target_arch = "wasm32"))]
            pending_probe: None,
            backend: AudioBackend::Auto,
            audio_track: 0,
            #[cfg(not(target_arch = "wasm32"))]
            modal: FfmpegModal::default(),
            pending_generate: None,
            progress_rx: None,
            progress_label: String::new(),
            progress_pos: 0,
            progress_total: None,
            #[cfg(not(target_arch = "wasm32"))]
            cancel_flag: None,
            mode: AudioMode::Bank,
            preset: AudioPreset::Default,
            // Straight from the module default, which already carries the two
            // settings that came out right on EVERY source listened to:
            // `noise_bands: 0` and `leveling: 1.0`. They are the pane's
            // defaults because they are the pipeline's, not because a preset
            // sets them -- a fresh session is already on them with no preset
            // selected at all.
            bands_value: opts.bands.unwrap_or(60),
            opts,
        }
    }
}

/// Native only: the backend selector lives in `draw_input`, which only offers
/// a picker on a target that has real file paths.
#[cfg(not(target_arch = "wasm32"))]
fn backend_name(b: AudioBackend) -> &'static str {
    match b {
        AudioBackend::Auto => "Auto",
        AudioBackend::Builtin => "Builtin",
        AudioBackend::Ffmpeg => "ffmpeg",
    }
}

impl AudioApp {
    /// The analysis options for the current UI state.
    ///
    /// **The single source of these for both `generate` and `live_cost`**, so
    /// the readout can never describe a different build from the one that
    /// gets made. Adding a control reaches the readout for free, because
    /// there is nothing to remember to add it to.
    ///
    /// Almost a copy of `self.opts`, by design (see the module doc). The one
    /// thing it does is clamp `max_frames`, defensively and exactly as the
    /// CLI's `audio_options` does: the sentinel that would re-enable an
    /// unbounded analysis loop must not be reachable from a fat-fingered
    /// control here either.
    pub fn audio_opts(&self) -> AudioOptions {
        AudioOptions {
            max_frames: self.opts.max_frames.clamp(1, crate::anim::pack::MAX_FRAMES),
            ..self.opts
        }
    }

    /// How many analysis frames the picked source would produce at the
    /// current settings, or `None` when nothing is picked or the probe could
    /// not read a duration.
    ///
    /// Routed through [`cost::frames_for_duration`], which is the analyser's
    /// own arithmetic -- an STFT emits nothing until it holds a full window,
    /// so a duration times fps overshoots by most of a window at every
    /// setting and by a third of a second at `--window 16384`.
    fn estimated_frames(&self) -> Option<usize> {
        let info = self.input.as_ref()?.info?;
        let duration = info.duration_hint?;
        cost::frames_for_duration(duration, info.sample_rate, &self.audio_opts())
    }

    /// The build cost of the current settings.
    ///
    /// `Err` is the renderer's own refusal for options that cannot be
    /// rendered at all (a `--subdiv` off the semitone grid, a voice-mode
    /// speaker count of 0, an impossible band span). Everything in the `Ok`
    /// case that does not depend on the track's LENGTH is exact; see
    /// `draw_cost` for how the length-dependent half is labelled when the
    /// frame count is not known yet.
    fn live_cost(&self) -> Result<AudioCost, String> {
        cost::estimate(
            self.mode,
            self.estimated_frames().unwrap_or(0),
            &self.audio_opts(),
        )
    }

    /// Whether the current settings can be rendered at all.
    ///
    /// [`cost::check`] rather than a second set of rules, so the Generate
    /// button and the cost readout can never disagree about whether a render
    /// is possible.
    pub fn validate(&self) -> Result<(), String> {
        cost::check(self.mode, &self.audio_opts())
    }

    /// The Max Voices control's label for the current mode.
    ///
    /// **`--max-voices` means two different things and the label has to say
    /// which.** In bank mode it is an upper bound on how many of a fixed bank
    /// may sound at once, and 0 means "every band"; in voice mode it is the
    /// number of speakers the build contains, and 0 is a save with no
    /// speakers in it. A single fixed label would mislead in one mode or the
    /// other, whichever wording it chose.
    pub fn max_voices_label(&self) -> &'static str {
        match self.mode {
            AudioMode::Bank => "Max Voices",
            AudioMode::Voice => "Speakers",
        }
    }

    fn max_voices_hint(&self) -> &'static str {
        match self.mode {
            AudioMode::Bank => {
                "An UPPER BOUND on how many bands may sound at once; every other band in \
                 the frame is set to exactly zero. 0 = every band, which is the setting \
                 that made the render sound like noise. This is usually NOT what limits a \
                 dense frame -- Peak Gate is, and it leaves only 12-24 candidates, so \
                 raising this past that changes nothing at all."
            }
            AudioMode::Voice => {
                "The NUMBER OF SPEAKERS BUILT, each one sounding most frames. 32 \
                 simultaneous emitters are verified to sound in game; above that is \
                 untested, and if a higher count comes back thinner that is engine voice \
                 stealing rather than the analysis. 0 is refused: it would be a save with \
                 no speakers in it."
            }
        }
    }

    /// The smallest legal value of the Max Voices control in the current
    /// mode. The widget's own half of the "0 means two different things"
    /// rule; `validate` is the other half, and catches a 0 carried over from
    /// bank mode by a later switch to voice.
    fn min_voices(&self) -> usize {
        match self.mode {
            AudioMode::Bank => 0,
            AudioMode::Voice => 1,
        }
    }

    /// Whether the band-grid controls (Bands, Subdiv, Noise Bands, Peak Gate)
    /// do anything in the current mode.
    ///
    /// Voice mode has no band grid at all -- that is the whole reason it
    /// exists -- so every one of them is inert there, and the CLI says so flag
    /// by flag rather than dropping them silently. A control that provably
    /// does nothing is worse than no control, so these are hidden rather than
    /// merely greyed.
    pub fn shows_band_grid_controls(&self) -> bool {
        self.mode.uses_band_grid()
    }

    /// Apply a preset to `opts`.
    ///
    /// Everything stays editable afterwards -- [`AudioPreset::apply`] writes
    /// exactly the five settings the listening sessions varied plus the two
    /// that came out the same on every source, and touches nothing else.
    fn load_preset(&mut self, preset: AudioPreset) {
        self.preset = preset;
        preset.apply(&mut self.opts);
        if let Some(b) = self.opts.bands {
            self.bands_value = b;
        }
        info!("Applied audio preset: {}", preset.name());
    }

    /// Start (or restart) the background duration probe for the picked file.
    ///
    /// Uses the SELECTED backend, so the duration the readout works from is
    /// the one the render's own decoder will report, and
    /// `DownloadConsent::Never`, so a probe can never trigger a download
    /// behind the user's back. A failure is not an error the user must act
    /// on -- Generate still works, and measures the real frame count as it
    /// analyses -- so it is logged at info level and leaves `info` as `None`.
    #[cfg(not(target_arch = "wasm32"))]
    fn start_probe(&mut self) {
        let Some(input) = &self.input else { return };
        let path = input.path.clone();
        let backend = self.backend;
        let track = self.audio_track;
        self.pending_probe = Some(Promise::spawn_thread("audio_probe", move || {
            crate::audio::backend::open_audio_track(
                &path,
                backend,
                DownloadConsent::Never,
                track,
            )
            .map(|s| s.info())
        }));
    }

    /// Poll the in-flight picker and probe, and apply whichever resolves.
    #[cfg(not(target_arch = "wasm32"))]
    fn poll_picks(&mut self) {
        if let Some(promise) = self.pending_pick.take() {
            match promise.try_take() {
                Ok(result) => {
                    if let Some(path) = result {
                        let name = path
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| path.display().to_string());
                        info!("Selected audio: {}", path.display());
                        self.input = Some(PickedAudio { path, name, info: None });
                        self.start_probe();
                    }
                }
                Err(promise) => self.pending_pick = Some(promise),
            }
        }

        if let Some(promise) = self.pending_probe.take() {
            match promise.try_take() {
                Ok(result) => match result {
                    Ok(info) => {
                        if let Some(input) = &mut self.input {
                            input.info = Some(info);
                        }
                        match info.duration_hint {
                            Some(d) => info!(
                                "Audio is {:.1}s at {} Hz",
                                d, info.sample_rate
                            ),
                            None => info!(
                                "Audio opened at {} Hz; the container reports no duration, so \
                                 the frame count is measured during the render",
                                info.sample_rate
                            ),
                        }
                    }
                    // Deliberately not `error!`: the render can still run and
                    // will measure the real length itself. Only the estimate
                    // is lost.
                    Err(e) => info!(
                        "Could not read the audio's length up front ({e}); the cost estimate \
                         will show only the part that does not depend on length"
                    ),
                },
                Err(promise) => self.pending_probe = Some(promise),
            }
        }
    }

    /// Poll the in-flight render, draining every `ProgressMsg` the worker has
    /// sent so far. The worker already logged any failure through
    /// `log::error!` before sending it, so there is nothing left to do once
    /// the promise resolves but drop it -- which frees `draw_submit` to show
    /// the generate button again.
    fn poll_generate(&mut self) {
        if let Some(rx) = &self.progress_rx {
            while let Ok(msg) = rx.try_recv() {
                // `apply_core` is the shared Begin/Tick/Finish bookkeeping
                // (see `crate::gui::util::RenderMsg`); the `Extra` it could
                // hand back is `Infallible`, so there is never a payload
                // left to handle here.
                let _: Option<std::convert::Infallible> = msg.apply_core(
                    &mut self.progress_label,
                    &mut self.progress_pos,
                    &mut self.progress_total,
                );
            }
        }

        if let Some(promise) = self.pending_generate.take() {
            match promise.try_take() {
                Ok(_) => {
                    self.progress_rx = None;
                    // Returning to idle clears the flag -- `generate` hands
                    // the next render a brand new one regardless, but
                    // clearing this means the Cancel button cannot linger
                    // visible into the idle view between renders.
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        self.cancel_flag = None;
                    }
                }
                Err(promise) => self.pending_generate = Some(promise),
            }
        }
    }

    /// Whether opening `path` with the selected backend needs a download the
    /// user has not consented to yet.
    ///
    /// Performs the SAME open `generate`'s worker will make, with
    /// [`ensure_ffmpeg`]'s role played by a closure that records the request
    /// and refuses. Running it here, synchronously on the UI thread, is what
    /// lets the modal ask the question before any background work starts -- a
    /// modal needs the UI thread to answer it, and a worker thread has none.
    /// The source this opens is dropped immediately: it exists only to answer
    /// the question, and the worker opens its own via the identical call. This
    /// does briefly block the UI thread on a real probe, the same tradeoff the
    /// video pane documents.
    #[cfg(not(target_arch = "wasm32"))]
    fn check_ffmpeg_consent(&self, path: &std::path::Path) -> FfmpegCheck {
        let mut needs_consent = false;
        let mut ensure = |_consent| {
            if ffmpeg_available() {
                Ok(())
            } else {
                needs_consent = true;
                Err("ffmpeg download consent required".to_string())
            }
        };
        match open_audio_ensuring(
            path,
            self.backend,
            DownloadConsent::Never,
            self.audio_track,
            &mut ensure,
        ) {
            Ok(_source) => FfmpegCheck::Ready,
            Err(_) if needs_consent => FfmpegCheck::NeedsConsent,
            Err(e) => {
                error!("{e}");
                FfmpegCheck::Failed
            }
        }
    }

    /// The always-visible half of the pane: destination, preset, mode, and the
    /// speaker count.
    ///
    /// **Nothing on the path from "I picked a song" to "I have a save" is
    /// allowed behind a collapsed section.** Everything that only TUNES a
    /// render lives in [`Self::draw_advanced_sections`] instead, so a
    /// first-time user meets four rows rather than twenty.
    ///
    /// The speaker count is on this side of the line and the rest of the band
    /// grid is not, because it is the one grid setting that is not a tuning
    /// knob: in Pitch Switching it IS the number of speakers the save
    /// contains, so hiding it would hide the size of the build.
    fn draw_settings(&mut self, ui: &mut Ui, shared: &mut SharedOptions) {
        ui.heading("Settings");
        ui.label(
            "Turn a song into a cluster of wired, pitched speaker bricks: the spectrum is \
             analysed once and written into wire arrays, and a chip plays it back.",
        );

        self.draw_preset_block(ui);

        settings_grid(ui, "audio_settings_grid").show(ui, |ui| {
            ui.label("Save Destination")
                .on_hover_text("The save will be created relative to the location of the exe.");
            // A plain `horizontal`, NOT `horizontal_wrapped`: a wrapping
            // horizontal layout inside a grid cell reports a height that is
            // short of what it drew, and the next row lands on top of it (this
            // row and the preset row were where that showed). Nothing in here
            // is text that needs to wrap -- the text field sizes itself to the
            // cell, which `settings_grid`'s `max_col_width` already bounds.
            ui.horizontal(|ui| {
                #[cfg(not(target_arch = "wasm32"))]
                ui.checkbox(&mut shared.out_clipboard, "Copy to clipboard")
                    .on_hover_text("Copy the save file path to clipboard after generation");
                ui.add(egui::TextEdit::singleline(&mut shared.out_file).hint_text("File Name"));
            });
            ui.end_row();
            draw_out_file_warnings(ui, &shared.out_file);

            self.draw_mode_row(ui);
            self.draw_voices_row(ui);

            // Playback is on the always-visible critical path, beside the mode
            // it plays back through, rather than buried in the Analysis
            // section: whether a track loops is a decision every render makes,
            // not an advanced analysis knob.
            ui.label("Playback").on_hover_text(
                "Loop: repeat the track forever (the default). Off: play through once and \
                 stop on the last analysis frame -- the timer is given a limit of \
                 (frames - 0.5) / fps, which expires halfway through the final frame. \
                 Costs nothing either way: same gates, same wires, same speakers.",
            );
            ui.checkbox(&mut self.opts.loop_playback, "Loop");
            ui.end_row();
        });

        self.draw_advanced_sections(ui);
    }

    /// Everything that tunes a render, grouped behind collapsing headers.
    ///
    /// Each header carries the VALUES inside it as chips, not a bare noun, so a
    /// collapsed section still says why this render differs from the last one.
    /// The `*_is_tuned` predicates then open a section on its first draw when it
    /// holds something other than the module default -- see
    /// [`crate::gui::util::section`] for why that only applies to the first
    /// frame, and why the chips are the load-bearing half of the pair.
    ///
    /// The row bodies are the same functions the flat grid called, and the
    /// per-mode conditions inside them are untouched: a section is a container,
    /// not a second place for a rule about when a control applies.
    fn draw_advanced_sections(&mut self, ui: &mut Ui) {
        let (chips, open) = (self.analysis_chips(), self.analysis_is_tuned());
        section(ui, "audio_analysis_section", "Analysis", &chips, open, |ui| {
            settings_grid(ui, "audio_analysis_grid").show(ui, |ui| self.draw_analysis_rows(ui));
        });

        let (chips, open) = (self.band_chips(), self.band_grid_is_tuned());
        section(ui, "audio_band_section", "Band grid", &chips, open, |ui| {
            settings_grid(ui, "audio_band_grid").show(ui, |ui| self.draw_band_rows(ui));
        });

        let (chips, open) = (self.envelope_chips(), self.envelope_is_tuned());
        section(ui, "audio_envelope_section", "Envelope", &chips, open, |ui| {
            settings_grid(ui, "audio_envelope_grid").show(ui, |ui| self.draw_envelope_rows(ui));
        });

        let (chips, open) = (self.level_chips(), self.levels_are_tuned());
        section(ui, "audio_level_section", "Levels", &chips, open, |ui| {
            settings_grid(ui, "audio_level_grid").show(ui, |ui| self.draw_level_rows(ui));
        });

        let (chips, open) = (self.speaker_chips(), self.speakers_are_tuned());
        section(
            ui,
            "audio_speaker_section",
            "Speaker placement",
            &chips,
            open,
            |ui| {
                settings_grid(ui, "audio_speaker_grid").show(ui, |ui| self.draw_speaker_rows(ui));
            },
        );
    }

    fn analysis_chips(&self) -> Vec<String> {
        // No "no loop" chip: the Loop checkbox is on the always-visible grid
        // now, so its state is already in view without a header summary.
        vec![
            format!("{:.0} fps", self.opts.fps),
            format!("window {}", self.opts.window),
            format!("max {} frames", self.opts.max_frames),
        ]
    }

    fn analysis_is_tuned(&self) -> bool {
        let d = AudioOptions::default();
        self.opts.fps != d.fps
            || self.opts.window != d.window
            || self.opts.max_frames != d.max_frames
    }

    fn band_chips(&self) -> Vec<String> {
        if !self.shows_band_grid_controls() {
            // Named rather than summarised: in Pitch Switching every control
            // in here is inert, and a row of numbers would imply otherwise.
            return vec![format!("{} only", AudioMode::Bank.name())];
        }
        vec![
            match self.opts.bands {
                Some(b) => format!("{b} bands"),
                None => "every band".to_string(),
            },
            format!("subdiv {}", self.opts.subdiv),
            format!("{} noise", self.opts.noise_bands),
            format!("gate {:.2}", self.opts.peak_gate),
        ]
    }

    fn band_grid_is_tuned(&self) -> bool {
        let d = AudioOptions::default();
        // Gated on the mode for the same reason the rows inside are: opening a
        // section whose every control is inert would be pointing at something
        // that cannot be affecting this render.
        self.shows_band_grid_controls()
            && (self.opts.bands != d.bands
                || self.opts.subdiv != d.subdiv
                || self.opts.noise_bands != d.noise_bands
                || self.opts.peak_gate != d.peak_gate)
    }

    fn envelope_chips(&self) -> Vec<String> {
        let mut chips = vec![
            format!("attack {:.0} ms", self.opts.attack_ms),
            format!("release {:.0} ms", self.opts.release_ms),
        ];
        // Only the mode that reads them shows them, so a chip never claims a
        // setting is shaping a render that ignores it.
        if self.mode == AudioMode::Voice {
            chips.push(format!("voice release {:.0} ms", self.opts.voice_release_ms));
            chips.push(format!("snap {:.0} cents", self.opts.pitch_snap_cents));
        }
        chips
    }

    fn envelope_is_tuned(&self) -> bool {
        let d = AudioOptions::default();
        self.opts.attack_ms != d.attack_ms
            || self.opts.release_ms != d.release_ms
            // Only counted in the mode that reads them, so a Pitch-Per-Speaker
            // render is not made to open a section over a voice setting it
            // ignores.
            || (self.mode == AudioMode::Voice
                && (self.opts.voice_release_ms != d.voice_release_ms
                    || self.opts.pitch_snap_cents != d.pitch_snap_cents))
    }

    fn level_chips(&self) -> Vec<String> {
        vec![
            format!("gain {:.2}", self.opts.gain),
            format!("leveling {:.2}", self.opts.leveling),
            format!("floor {:.0} dB", self.opts.floor_db),
        ]
    }

    fn levels_are_tuned(&self) -> bool {
        let d = AudioOptions::default();
        self.opts.gain != d.gain
            || self.opts.leveling != d.leveling
            || self.opts.floor_db != d.floor_db
    }

    fn speaker_chips(&self) -> Vec<String> {
        let mut chips = vec![
            format!("inner {:.0}", self.opts.inner_radius),
            format!("max {:.0} units", self.opts.max_distance),
        ];
        // Sine is the default, so only a NON-sine waveform earns a chip -- the
        // same "only the difference is worth a word" idiom as the two below.
        if self.opts.tonal_synth != SynthWave::default() {
            chips.push(format!("{} wave", self.opts.tonal_synth.flag()));
        }
        if self.opts.speakers_in_chip {
            chips.push("in chip".to_string());
        }
        // Buttons are ON by default, so only their ABSENCE earns a chip.
        if !self.opts.control_buttons {
            chips.push("no buttons".to_string());
        }
        chips
    }

    fn speakers_are_tuned(&self) -> bool {
        let d = AudioOptions::default();
        self.opts.inner_radius != d.inner_radius
            || self.opts.max_distance != d.max_distance
            || self.opts.tonal_synth != d.tonal_synth
            || self.opts.speakers_in_chip != d.speakers_in_chip
            || self.opts.control_buttons != d.control_buttons
    }

    /// Every number behind it came from a listening session, not analysis,
    /// and several reverse what the measurements said -- see [`AudioPreset`].
    /// Selecting one seeds five settings and leaves them all editable.
    ///
    /// Drawn outside the settings grid: a `Grid` cell doesn't report the
    /// height of wrapped text, so the next row draws on top of it (the
    /// "Preset overlaps Pitch-Per-Speaker" bug). The pane's own vertical
    /// layout doesn't have that problem.
    fn draw_preset_block(&mut self, ui: &mut Ui) {
        let mut chosen = None;
        ui.horizontal(|ui| {
            ui.label("Preset").on_hover_text(
                "Settings found BY EAR, per content type, across dozens of renders -- window, \
                 band count, voices, peak gate and release move together, because that is how \
                 they were listened to. Selecting one seeds those five and leaves every control \
                 below editable. It does NOT touch the analysis rate, gain, subdivision, attack, \
                 the speaker radii or the frame cap: those were not what the sessions were \
                 listening for.",
            );
            egui::ComboBox::from_id_salt("audio_preset")
                .selected_text(self.preset.name())
                .show_ui(ui, |ui| {
                    for p in AudioPreset::ALL {
                        if ui
                            .selectable_label(self.preset == p, p.name())
                            .on_hover_text(p.hint())
                            .clicked()
                        {
                            chosen = Some(p);
                        }
                    }
                });
        });
        // The paragraph, full width, in the vertical layout -- see the doc.
        ui.label(self.preset.hint());
        if let Some(p) = chosen {
            self.load_preset(p);
        }
        ui.add_space(4.0);
    }

    fn draw_mode_row(&mut self, ui: &mut Ui) {
        ui.label("Mode").on_hover_text(
            "Pitch-Per-Speaker: a fixed bank of speakers, each one owning a single pitch, \
             with only their volumes written -- best for speech and broadband material, and \
             it never writes a pitch, so it depends on no unverified in-game behaviour. \
             Pitch Switching: Speakers speakers that TRACK spectral peaks, re-pitching and \
             changing volume every frame -- no band grid, so no tuning error; best for tonal \
             material such as piano. THE 'Max Voices' CONTROL MEANS SOMETHING DIFFERENT IN \
             EACH.",
        );
        ui.horizontal(|ui| {
            for m in AudioMode::ALL {
                ui.radio_value(&mut self.mode, m, m.name());
            }
        });
        ui.end_row();
    }

    fn draw_analysis_rows(&mut self, ui: &mut Ui) {
        ui.label("Analysis FPS").on_hover_text(
            "Analysis frames per second -- how often every speaker's volume is rewritten. \
             The wire graph ticks at 60 Hz on a dedicated server and faster locally, so 30 \
             is comfortable; raising it multiplies the array data without adding detail the \
             window can resolve.",
        );
        ui.add(
            egui::DragValue::new(&mut self.opts.fps)
                .speed(0.5)
                .range(1.0..=120.0),
        );
        ui.end_row();

        ui.label("Window").on_hover_text(
            "STFT (short-time Fourier transform) window size, in samples -- the single \
             biggest lever, and the one the \
             presets differ on most. Long windows resolve pitch (a sustained partial gets a \
             clean note) and smear transients; short ones do the reverse, which is why \
             speech wants 2048 and a music box wants 16384. Powers of two only: the \
             transform is a radix FFT.",
        );
        egui::ComboBox::from_id_salt("audio_window")
            .selected_text(self.opts.window.to_string())
            .show_ui(ui, |ui| {
                for w in WINDOW_SIZES {
                    ui.selectable_value(&mut self.opts.window, w, w.to_string());
                }
            });
        ui.end_row();

        ui.label("Max Frames").on_hover_text(
            "Hard cap on analysed frames. Frames past 65535 spill into extra wire arrays, \
             which costs two gates per stream per extra array and nothing else -- audio \
             writes numbers straight into arrays with no packing, so a long render is cheap \
             here in a way a long video render is not.",
        );
        ui.add(
            egui::Slider::new(&mut self.opts.max_frames, 1..=crate::anim::pack::MAX_FRAMES)
                .logarithmic(true),
        );
        ui.end_row();
    }

    fn draw_band_rows(&mut self, ui: &mut Ui) {
        ui.label("Bands").on_hover_text(
            "Total speakers, noise bands included. Bands sit on EXACT equal-tempered steps, \
             so this selects the SPAN (the steps nearest A440), never the spacing -- and it \
             is symmetric around A440, so narrowing trims the bass and the treble together. \
             Narrowing helps a music box or solo piano and CRUSHES chiptune bass. Unlimited \
             = every step the emitter's pitch range holds: 79 at Subdiv 12, 159 at 24.",
        );
        let mut unlimited = false;
        if self.shows_band_grid_controls() {
            ui.horizontal(|ui| {
                let mut limited = self.opts.bands.is_some();
                if ui.checkbox(&mut limited, "Limit").changed() {
                    self.opts.bands = limited.then_some(self.bands_value);
                }
                ui.add_enabled_ui(limited, |ui| {
                    if let Some(bands) = &mut self.opts.bands {
                        ui.add(egui::Slider::new(bands, BANDS_RANGE).logarithmic(true));
                    } else {
                        // A disabled stand-in so the row does not change
                        // width when the checkbox is cleared.
                        ui.add_enabled(
                            false,
                            egui::Slider::new(&mut self.bands_value, BANDS_RANGE)
                                .logarithmic(true),
                        );
                    }
                });
                unlimited = !limited;
            });
            if let Some(b) = self.opts.bands {
                self.bands_value = b;
            }
        } else {
            note(ui, "(Pitch-Per-Speaker only -- Pitch Switching has no band grid)");
        }
        ui.end_row();
        // Its OWN row, for the same reason the video pane's "(using source
        // size)" gets one: beside a checkbox and a slider there is less width
        // left than the sentence's longest word, and egui then breaks it
        // mid-word instead of at a space.
        if unlimited {
            ui.label("");
            note(ui, "(every band the pitch range holds)");
            ui.end_row();
        }

        ui.label("Subdiv").on_hover_text(
            "Tonal bands per octave. MULTIPLES OF 12 ONLY, which is why this is a dropdown: \
             only then does every band land on a real semitone, and anything else was heard \
             in game as out of tune -- 14 sharp, 18 flat. 12 (one per semitone) is \
             preferred; 24 halves the error on a source not tuned to A440 but splits each \
             note across two speakers.",
        );
        if self.shows_band_grid_controls() {
            egui::ComboBox::from_id_salt("audio_subdiv")
                .selected_text(self.opts.subdiv.to_string())
                .show_ui(ui, |ui| {
                    for s in SUBDIVS {
                        ui.selectable_value(&mut self.opts.subdiv, s, s.to_string());
                    }
                });
        } else {
            note(ui, "(Pitch-Per-Speaker only)");
        }
        ui.end_row();

        ui.label("Noise Bands").on_hover_text(
            "White/pink noise speakers carrying the energy off either end of the tonal \
             range. 0 is the default because every source tried in game -- speech, solo \
             piano and a full pop mix -- was reported WORSE with them than without. 2 is \
             worth trying on percussion, which nothing else in the bank can render at all.",
        );
        if self.shows_band_grid_controls() {
            ui.add(egui::Slider::new(&mut self.opts.noise_bands, 0..=2));
        } else {
            note(ui, "(Pitch-Per-Speaker only)");
        }
        ui.end_row();

        ui.label("Peak Gate").on_hover_text(
            "How far above the mean of its neighbourhood a band must stand to count as a \
             note at all, as an amplitude ratio (1.5 = 3.5 dB). THIS, not Max Voices, is \
             what limits a dense frame -- lower it to sound more bands at once. 1.0 \
             disables it, which is the wall-of-sound behaviour that made an early render \
             sound like noise: 94 of 96 bands sounding at once is noise by construction.",
        );
        if self.shows_band_grid_controls() {
            ui.add(egui::Slider::new(&mut self.opts.peak_gate, 1.0..=6.0));
        } else {
            note(ui, "(Pitch-Per-Speaker only -- Pitch Switching gates over FFT bins)");
        }
        ui.end_row();
    }

    /// The speaker-count row, kept OUT of the band-grid section on purpose.
    ///
    /// It sits with the band-grid controls in the CLI's flag list, but it is
    /// not one of them: in Pitch Switching it is the NUMBER OF SPEAKERS THE
    /// SAVE CONTAINS (see [`Self::max_voices_hint`]), which is the size of the
    /// build, not a tuning knob -- so it stays visible next to the mode that
    /// decides what it means. Label, hint and range are unchanged; only the
    /// row's position moved.
    fn draw_voices_row(&mut self, ui: &mut Ui) {
        // THE label that has to change with the mode -- see
        // `max_voices_label`'s doc.
        ui.label(self.max_voices_label())
            .on_hover_text(self.max_voices_hint());
        // Read before the mutable borrow of `opts.max_voices` below.
        let range = self.min_voices()..=MAX_SPEAKERS;
        ui.add(egui::Slider::new(&mut self.opts.max_voices, range));
        ui.end_row();
    }

    fn draw_envelope_rows(&mut self, ui: &mut Ui) {
        ui.label("Attack").on_hover_text(
            "How fast a speaker's level RISES toward what the analysis measured, in \
             milliseconds. Short by design -- an attack is what makes a render sound like an \
             instrument being struck rather than a pad swelling.",
        );
        ui.add(
            egui::DragValue::new(&mut self.opts.attack_ms)
                .speed(0.5)
                .suffix(" ms")
                .range(0.0..=1000.0),
        );
        ui.end_row();

        ui.label("Release").on_hover_text(
            "How fast a speaker's level FALLS, including after its band stops being \
             selected, in milliseconds. THIS is what stops one-frame selections sounding \
             like beeps. 150 suits sustained music; SPEECH WANTS 30-60, because a phoneme \
             is 50-100 ms long and a 150 ms release smears clean across it. Do not tune this \
             by maximising mean run length -- that was tried and made speech much worse.",
        );
        ui.add(
            egui::DragValue::new(&mut self.opts.release_ms)
                .speed(1.0)
                .suffix(" ms")
                .range(0.0..=2000.0),
        );
        ui.end_row();

        ui.label("Voice Release").on_hover_text(
            "Pitch Switching only: how long a voice takes to fade to EXACTLY zero once its \
             partial has gone. Distinct from Release, which is a time constant on a voice \
             that is still sounding and never reaches zero. The material decides it -- a \
             sustained note can afford 150, a spoken phoneme cannot.",
        );
        if self.mode == AudioMode::Voice {
            ui.add(
                egui::DragValue::new(&mut self.opts.voice_release_ms)
                    .speed(1.0)
                    .suffix(" ms")
                    .range(0.0..=2000.0),
            );
        } else {
            note(ui, "(Pitch Switching only -- a fixed-pitch band has no note-off to time)");
        }
        ui.end_row();

        ui.label("Pitch Snap").on_hover_text(
            "Pitch Switching only: pull a continuing voice onto the nearest equal-tempered \
             semitone when it is within this many cents. 0 = off, which is the default \
             because Pitch Switching's whole claim is that it needs no grid -- snapping quantises \
             real vibrato and glissando away along with the jitter.",
        );
        if self.mode == AudioMode::Voice {
            // The analysis owns the bound (`voices::MAX_PITCH_SNAP_CENTS` --
            // half a semitone, past which "snap if within" cannot mean
            // anything a listener hears differently), and the widget follows
            // it rather than carrying its own number: a range of 0..=100 here
            // let the GUI offer values the CLI refuses by name.
            ui.add(
                egui::DragValue::new(&mut self.opts.pitch_snap_cents)
                    .speed(1.0)
                    .suffix(" cents")
                    .range(0.0..=crate::audio::voices::MAX_PITCH_SNAP_CENTS),
            );
        } else {
            note(ui, "(Pitch Switching only)");
        }
        ui.end_row();
    }

    fn draw_level_rows(&mut self, ui: &mut Ui) {
        ui.label("Gain").on_hover_text(
            "Post-normalisation multiplier, applied and then CLAMPED at 1.0 -- so this is a \
             way to make a render quieter, never louder. Above 1.0 it only clips (the file \
             even gets smaller as it rises), which is why the slider stops there.",
        );
        ui.add(egui::Slider::new(&mut self.opts.gain, 0.0..=1.0));
        ui.end_row();

        ui.label("Leveling").on_hover_text(
            "Per-frame automatic gain control: 0 keeps the track's own dynamics, 1 drags \
             every frame toward full scale. 1.0 was best on EVERY source tried in game -- \
             piano, speech and a music box alike -- which reversed the argument that it \
             would flatten music. A bank of sine emitters has ~30 dB of usable range where a \
             master has 60, so a track's dynamics do not survive the trip regardless.",
        );
        ui.add(egui::Slider::new(&mut self.opts.leveling, 0.0..=1.0));
        ui.end_row();

        ui.label("Floor").on_hover_text(
            "Bands this many dB below THE LOUDEST BAND IN THE SAME FRAME become exactly \
             zero. Frame-relative, not track-relative: an absolute floor is measured against \
             a scale one loud transient sets for the whole track, and in the limit silences \
             a quiet passage outright.",
        );
        ui.add(egui::Slider::new(&mut self.opts.floor_db, -120.0..=0.0).suffix(" dB"));
        ui.end_row();
    }

    fn draw_speaker_rows(&mut self, ui: &mut Ui) {
        // The waveform is a baked emitter property, exactly like the two radii
        // below (it is written next to them in `add_emitter`), so it lives in
        // this section rather than under Analysis -- and it applies in BOTH
        // modes, so it is not gated on the band grid. The four basic waves are
        // the whole list, Sine first (the default the selector opens on).
        ui.label("Waveform").on_hover_text(
            "The synth the TONAL bands play through -- the four basic waves. Sine (the \
             default) is the classic bank tone; square, triangle and sawtooth are brighter \
             and buzzier. Applies to tonal bands in both modes; white/pink noise bands keep \
             their own noise assets and are unaffected.",
        );
        egui::ComboBox::from_id_salt("audio_synth")
            .selected_text(self.opts.tonal_synth.name())
            .show_ui(ui, |ui| {
                for w in SynthWave::ALL {
                    ui.selectable_value(&mut self.opts.tonal_synth, w, w.name());
                }
            });
        ui.end_row();

        ui.label("Placement").on_hover_text(
            "Where the speaker cluster goes. Beside the chip on the main grid (the \
             default), or IN the microchip's own inner grid, which makes the whole audio \
             device one portable microchip. The speakers play from the chip's ORIGIN \
             either way -- an AudioEmitter on a microchip inner grid emits from the chip's \
             world position -- so this only moves the bricks, it does not change the \
             sound, and costs nothing (same speakers, gates and wires).",
        );
        ui.checkbox(&mut self.opts.speakers_in_chip, "In microchip");
        ui.end_row();

        ui.label("Controls").on_hover_text(
            "Pre-generate three physical Pause/Restart/Resume buttons on the main grid, \
             wired into the clock so the render is controllable out of the box. Off means \
             you wire the clock's control pins yourself. Adds 9 bricks and 6 wires; no \
             extra gate.",
        );
        ui.checkbox(&mut self.opts.control_buttons, "Control buttons");
        ui.end_row();

        ui.label("Inner Radius").on_hover_text(
            "The radius inside which there is NO distance attenuation, in units (10 units = \
             1 brick). NOT COSMETIC: turning spatialization off kills PANNING, not distance \
             attenuation, so a listener outside this radius hears a distance-filtered slice \
             of the spectrum that changes as they walk. That was the root cause of 'it \
             doesn't sound like the song'.",
        );
        ui.add(
            egui::DragValue::new(&mut self.opts.inner_radius)
                .speed(10.0)
                .range(1.0..=100_000.0),
        );
        ui.end_row();

        ui.label("Max Distance").on_hover_text(
            "Where the sound stops, in units (10 units = 1 brick). Must be larger than Inner \
             Radius; the renderer refuses an inverted pair rather than building a silent save.",
        );
        ui.add(
            egui::DragValue::new(&mut self.opts.max_distance)
                .speed(10.0)
                .range(1.0..=1_000_000.0),
        );
        ui.end_row();
    }

    fn draw_input(&mut self, ui: &mut Ui) {
        ui.add_space(8.0);
        ui.separator();
        ui.heading("Source");
        ui.label(
            "Pick an audio file (mp3/wav/flac/ogg/m4a), or a video container to pull an \
             audio track out of (mp4/mov/mkv/webm/avi/m4v).",
        );

        #[cfg(target_arch = "wasm32")]
        {
            ui.colored_label(
                Color32::from_rgb(255, 140, 60),
                "Audio rendering needs the desktop build: every decoder here opens a file \
                 PATH and streams it twice, which a browser file handle cannot provide. The \
                 settings and cost model above are live and correct.",
            );
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let picking = self.pending_pick.is_some();
            ui.horizontal_wrapped(|ui| {
                if ui
                    .add(Button::new("Pick audio file").fill(Color32::from_rgb(60, 60, 120)))
                    .clicked()
                    && !picking
                {
                    self.pending_pick = Some(pick_audio_path());
                }
                if self.pending_probe.is_some() {
                    ui.spinner();
                    ui.label("reading length...");
                }
            });

            let mut reprobe = false;
            ui.horizontal_wrapped(|ui| {
                ui.label("Decode Backend").on_hover_text(
                    "Auto: ffmpeg when it is already installed (it covers codecs the builtin \
                     decoder is not built for), otherwise the pure-Rust builtin -- and it \
                     never downloads. Builtin: pure Rust only. ffmpeg: always, downloading it \
                     first if it is missing and consented to.",
                );
                for b in [AudioBackend::Auto, AudioBackend::Builtin, AudioBackend::Ffmpeg] {
                    reprobe |= ui
                        .radio_value(&mut self.backend, b, backend_name(b))
                        .changed();
                }
            });

            ui.horizontal_wrapped(|ui| {
                ui.label("Audio Track").on_hover_text(
                    "Which audio stream to decode, 0 = first. Dual-audio releases commonly \
                     carry the original language first and the dub second, so which one is \
                     'first' is a container-ordering accident. Needs the ffmpeg backend -- \
                     the builtin decoder always uses the container's default track.",
                );
                reprobe |= ui
                    .add(egui::DragValue::new(&mut self.audio_track).range(0..=32))
                    .changed();
            });
            // The probe reports the SELECTED backend's answer, so a change of
            // backend or track invalidates it -- a stale duration from
            // another track is worse than none.
            if reprobe && self.input.is_some() {
                if let Some(input) = &mut self.input {
                    input.info = None;
                }
                self.start_probe();
            }

            let mut clear_input = false;
            match &self.input {
                None => {
                    ui.label("No source selected.");
                }
                Some(input) => {
                    // Wrapped: a long file name would otherwise run the row
                    // off the edge of the pane.
                    ui.horizontal_wrapped(|ui| {
                        if ui.button("✖").clicked() {
                            clear_input = true;
                        }
                        let detail = match input.info {
                            Some(AudioInfo { sample_rate, duration_hint: Some(d) }) => {
                                format!("{:.1}s at {sample_rate} Hz", d)
                            }
                            Some(AudioInfo { sample_rate, duration_hint: None }) => {
                                format!("{sample_rate} Hz, length unknown until render")
                            }
                            None => "length not read yet".to_string(),
                        };
                        ui.label(format!("{} -- {detail}", input.name));
                    });
                }
            }
            if clear_input {
                self.input = None;
                self.pending_probe = None;
            }
        }
    }

    /// The live cost readout.
    ///
    /// Split in two on purpose, because the two halves have different
    /// standing. Everything derived from the options -- speakers, streams,
    /// gates, wires, bricks -- is EXACT and pinned against real builds by
    /// `audio::cost`'s own tests. The frame count is an estimate from the
    /// source's duration hint, and when there is no hint the length-dependent
    /// half is labelled as the single-bank floor it is rather than dressed up
    /// as a measurement.
    fn draw_cost(&self, ui: &mut Ui) {
        let cost = match self.live_cost() {
            // The renderer's own refusal, for options that cannot be rendered
            // at all. Surfacing it here rather than a plausible-looking
            // number is what stops the readout describing a build that will
            // never exist -- Generate refuses on the identical check.
            Err(msg) => {
                ui.colored_label(Color32::RED, msg);
                return;
            }
            Ok(c) => c,
        };

        let unit = match self.mode {
            AudioMode::Bank => "band",
            AudioMode::Voice => "voice",
        };
        match self.estimated_frames() {
            Some(frames) => {
                let seconds = frames as f32 / self.opts.fps.max(0.001);
                ui.label(format!(
                    "Estimated: {} speaker(s) ({} {unit}(s), {} stream(s)), {} frame(s) \
                     ~= {:.0}s -> {} gate(s), {} wire(s), {} brick(s), {} bank(s), {} array \
                     element(s)",
                    cost.speakers,
                    cost.speakers,
                    cost.streams,
                    cost.frames,
                    seconds,
                    cost.gates,
                    cost.wires,
                    cost.bricks,
                    cost.banks,
                    cost.elements,
                ));
            }
            None => {
                ui.label(format!(
                    "{} speaker(s) ({} stream(s)) -> {} gate(s), {} wire(s), {} brick(s) for \
                     a single bank. The frame count is measured when Generate opens the \
                     file; each extra 65535 frames adds 2 gates per stream.",
                    cost.speakers, cost.streams, cost.gates, cost.wires, cost.bricks,
                ));
            }
        }

        // Frame drops start around 20 000 gates. An audio build is nowhere
        // near it -- a full 79-band bank is under 300 -- and saying so is
        // worth a line, because the number that limits a video render does
        // not limit this one.
        if cost.gates > 6000 {
            ui.colored_label(
                Color32::from_rgb(255, 140, 60),
                format!(
                    "{} gates is a large build -- frame drops start around 20000 and it may \
                     be slow to paste in-game",
                    cost.gates
                ),
            );
        }
    }

    fn draw_submit(&mut self, ui: &mut Ui, shared: &mut SharedOptions) {
        // A render already in flight: no button at all, so a second click
        // cannot start a second one.
        if self.pending_generate.is_some() {
            draw_progress_bar(ui, &self.progress_label, self.progress_pos, self.progress_total);
            // Only the ANALYSIS polls the cancel flag; the world build and
            // the brz encode that follow it are not interruptible, so a
            // cancel late in a long render still has to finish the build
            // before it can decline to write it.
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(flag) = &self.cancel_flag {
                draw_cancel_button(ui, flag);
            }
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        if self.modal.is_open() {
            ui.label("Waiting on the ffmpeg download prompt above...");
            return;
        }

        // A destination that cannot be written is refused before the button is
        // offered -- the red label in the settings grid above is the same
        // condition, and it used to be advisory. See
        // `util::refuse_bad_out_file`.
        if refuse_bad_out_file(ui, &shared.out_file) {
            return;
        }

        // The SAME check the cost readout refuses on, so the button and the
        // readout can never disagree about whether a render is possible.
        if let Err(msg) = self.validate() {
            ui.colored_label(Color32::RED, format!("Cannot render: {msg}"));
            return;
        }

        if self.input.is_some() {
            if ui
                .add(Button::new("Generate audio2brick save").fill(Color32::from_rgb(50, 90, 50)))
                .clicked()
            {
                self.generate(shared);
            }
        } else {
            #[cfg(not(target_arch = "wasm32"))]
            ui.label("Pick an audio or video file to continue...");
            #[cfg(target_arch = "wasm32")]
            ui.label("Audio rendering is not available in the browser build.");
        }
    }

    /// On click: open the source -> analyse -> build the speaker world ->
    /// encode -> deliver. Same path as `main.rs`'s `--audio-mode` branch,
    /// dispatched on the same [`AudioMode`]/[`AudioOptions`] -- nothing here
    /// reimplements a step of it. Runs on a background thread on native (the
    /// captured state is all `Copy` or one `PathBuf`); `wasm32` has no
    /// `std::thread::spawn` and no picker, so this path is unreachable there.
    /// Every failure is logged via `log::error!`, never panicked.
    fn generate(&mut self, shared: &SharedOptions) {
        if self.pending_generate.is_some() {
            return error!("a render is already in progress");
        }
        // Belt and suspenders: `draw_submit` already refuses on this, but a
        // render that the readout has called impossible must not be reachable
        // from anywhere.
        if let Err(e) = self.validate() {
            return error!("{e}");
        }

        #[cfg(not(target_arch = "wasm32"))]
        if self.modal.is_open() {
            return error!("waiting on the ffmpeg download prompt");
        }

        let Some(input) = &self.input else {
            return error!("pick an audio or video file first");
        };
        let path = input.path.clone();

        // A source may need ffmpeg, and whether the user consents to
        // downloading it is a question only the UI thread can ask. The same
        // open the worker will make, with the consent decision recorded
        // rather than answered -- see `check_ffmpeg_consent`.
        #[cfg(not(target_arch = "wasm32"))]
        match self.check_ffmpeg_consent(&path) {
            FfmpegCheck::Ready => {}
            FfmpegCheck::NeedsConsent => {
                self.modal.request(path);
                return;
            }
            // Already logged by `check_ffmpeg_consent`.
            FfmpegCheck::Failed => return,
        }

        // Everything below is plain data or an owned clone -- nothing borrows
        // `self` or `shared`, so it is all free to move into `work`.
        let mode = self.mode;
        // THE SAME value `live_cost` just costed. `AudioOptions` is `Copy`,
        // so this is a plain value capture and there is no second assembly
        // step where the two could diverge.
        let opts = self.audio_opts();
        let backend = self.backend;
        let track = self.audio_track;
        let out_file = shared.out_file.clone();
        let out_clipboard = shared.out_clipboard;

        let (progress_tx, progress_rx) = std::sync::mpsc::channel::<ProgressMsg>();
        self.progress_rx = Some(progress_rx);
        self.progress_label.clear();
        self.progress_pos = 0;
        self.progress_total = None;

        // A FRESH flag every render -- never a reused one a previous render
        // might have left `true`.
        let cancel_flag = Arc::new(AtomicBool::new(false));
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.cancel_flag = Some(Arc::clone(&cancel_flag));
        }

        let work = move || -> Result<(), String> {
            info!("Opening audio {}", path.display());
            // `DownloadConsent::Never` is safe here, not silent:
            // `check_ffmpeg_consent` already ran this exact call on the UI
            // thread, so either ffmpeg was not needed or it is already
            // installed (the modal's Download button ran
            // `ensure_ffmpeg(Always)` and this worker only starts once that
            // succeeded).
            #[cfg(not(target_arch = "wasm32"))]
            let source = open_audio_ensuring(
                &path,
                backend,
                DownloadConsent::Never,
                track,
                &mut |_| ensure_ffmpeg(DownloadConsent::Never),
            )?;
            #[cfg(target_arch = "wasm32")]
            let source = crate::audio::backend::open_audio_track(
                &path,
                backend,
                DownloadConsent::Never,
                track,
            )?;

            let mut progress = ChannelProgress::new(progress_tx, cancel_flag);
            // The CLI's two-branch dispatch, unchanged: one analysis front
            // end, two renderers.
            //
            // CANCELLATION lands BETWEEN the analysis and the renderer, which
            // is why each arm re-checks before it builds.
            // `analyze`/`analyze_voices` stop pulling frames as soon as the
            // flag goes up, but the renderer behind them would otherwise go on
            // to build the whole speaker cluster out of the truncated track,
            // only for `deliver_world_unless_cancelled` to throw it away. An
            // empty `World` skips that while leaving the one cancellation
            // policy -- write nothing, log INFO not an error -- in the single
            // place that owns it. This is the caller-side half of the same
            // early return the three animation renderers make internally
            // (see `anim::bricks::build_brick_world`'s doc); the audio
            // builders take no `Progress` of their own, so this seam is the
            // only place their work can be skipped.
            let world = match mode {
                AudioMode::Voice => {
                    let streams = analyze_voices(source.as_ref(), &opts, &mut progress)?;
                    info!(
                        "Analyzed {} frame(s) at {} fps across {} voice(s); {:.2} sounding per \
                         frame, mean lifetime {:.1} frames",
                        streams.frame_count,
                        streams.fps,
                        streams.voice_count(),
                        streams.stats.mean_voices_per_frame(streams.frame_count),
                        streams.stats.mean_lifetime(),
                    );
                    if progress.is_cancelled() {
                        // Fully qualified rather than imported: this file
                        // names no other `brdb` type, so a one-off keeps the
                        // import block untouched.
                        brdb::World::new()
                    } else {
                        build_voice_world(&streams, &opts)?
                    }
                }
                AudioMode::Bank => {
                    let track = analyze(source.as_ref(), &opts, &mut progress)?;
                    let sounding: usize = track
                        .volumes
                        .iter()
                        .flat_map(|b| b.iter())
                        .filter(|v| **v > 0.0)
                        .count();
                    info!(
                        "Analyzed {} frame(s) at {} fps across {} band(s); {:.2} sounding per \
                         frame (Max Voices {}, Peak Gate {})",
                        track.frame_count,
                        track.fps,
                        track.plan.len(),
                        sounding as f64 / track.frame_count.max(1) as f64,
                        opts.max_voices,
                        opts.peak_gate,
                    );
                    if progress.is_cancelled() {
                        brdb::World::new()
                    } else {
                        build_speaker_world(&track, &opts)?
                    }
                }
            };
            deliver_world_unless_cancelled(world, &progress, &out_file, out_clipboard)
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
            // No threads on the web: run synchronously. Unreachable in
            // practice -- there is no picker there, so `self.input` is always
            // `None` -- but kept so the two targets have one `generate`.
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
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_picks();
        self.poll_generate();
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.modal.poll(ui.ctx());
            // Reworded slightly from the video pane's prompt: this pane also
            // accepts a plain audio file, so "This file" rather than "This
            // video". The URL is the mirror ffmpeg-sidecar fetches from.
            let prompt = format!(
                "This file needs the ffmpeg decode backend, and no ffmpeg install was found \
                 on this machine. Download it now from {}?",
                ffmpeg_sidecar::download::ffmpeg_download_url()
                    .unwrap_or("the official ffmpeg build server")
            );
            self.modal
                .draw(ui.ctx(), "audio", &prompt, "the audio was not converted");
        }
        self.draw_settings(ui, shared);
        self.draw_input(ui);
        ui.add_space(8.0);
        ui.separator();
        self.draw_cost(ui);
        ui.separator();
        self.draw_submit(ui, shared);
        // A probe resolving on its own thread has nothing else to wake the
        // event loop, so the readout would otherwise only pick up the
        // duration on the user's next mouse move.
        #[cfg(not(target_arch = "wasm32"))]
        if self.pending_probe.is_some() || self.pending_pick.is_some() {
            ui.ctx().request_repaint();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::source::{AudioSource, SampleClip};
    use crate::audio::track::DEFAULT_RELEASE_MS;
    use crate::progress::NoProgress;

    /// A few seconds of a three-partial tone, in memory. Real enough for the
    /// analyser to produce real frames, with an exact duration so the
    /// readout's frame count and the render's can be compared for EQUALITY
    /// rather than for being roughly similar.
    fn tone(seconds: f64) -> SampleClip {
        let sr = 48_000u32;
        let n = (sr as f64 * seconds) as usize;
        let samples = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let tau = std::f32::consts::TAU;
                0.4 * (tau * 440.0 * t).sin()
                    + 0.25 * (tau * 660.0 * t).sin()
                    + 0.15 * (tau * 880.0 * t).sin()
            })
            .collect();
        SampleClip::new(sr, samples)
    }

    /// Gates are every inner-grid brick that is not one of the chip's I/O
    /// pins, counted the way this crate's other build tests count them.
    fn built_gates(w: &brdb::World) -> usize {
        w.grids[0].1.len() - cost::CHIP_PINS
    }

    /// A pane with a source whose length is already known, so the
    /// length-dependent half of the readout has real numbers without a real
    /// file or a real decode.
    fn with_source(duration_s: f64) -> AudioApp {
        let mut app = AudioApp::default();
        app.input = Some(PickedAudio {
            path: std::path::PathBuf::from("song.mp3"),
            name: "song.mp3".to_string(),
            info: Some(AudioInfo { sample_rate: 48_000, duration_hint: Some(duration_s) }),
        });
        app
    }

    #[test]
    fn ticking_past_an_estimated_total_grows_it_rather_than_overflowing_the_bar() {
        let mut app = AudioApp::default();
        let (tx, rx) = std::sync::mpsc::channel();
        app.progress_rx = Some(rx);

        tx.send(ProgressMsg::Begin {
            label: "analyzing audio (estimated total)".into(),
            total: Some(100),
        })
        .expect("send");
        tx.send(ProgressMsg::Tick(50)).expect("send");
        tx.send(ProgressMsg::Tick(137)).expect("send");
        app.poll_generate();

        assert_eq!(app.progress_pos, 137, "the count must not freeze at the estimate");
        assert_eq!(app.progress_total, Some(137), "the total must grow to meet it");
        assert!(
            app.progress_pos <= app.progress_total.unwrap(),
            "a bar must never be drawn past its own end"
        );
    }

    #[test]
    fn ticking_a_totalless_phase_never_invents_a_denominator() {
        let mut app = AudioApp::default();
        let (tx, rx) = std::sync::mpsc::channel();
        app.progress_rx = Some(rx);
        tx.send(ProgressMsg::Begin { label: "analyzing audio".into(), total: None }).expect("send");
        tx.send(ProgressMsg::Tick(9_999)).expect("send");
        app.poll_generate();
        assert_eq!(app.progress_total, None);
        assert_eq!(app.progress_pos, 9_999);
    }

    #[test]
    fn a_fresh_pane_starts_with_no_noise_bands_and_full_leveling() {
        let app = AudioApp::default();
        assert_eq!(app.preset, AudioPreset::Default);
        assert_eq!(app.audio_opts().noise_bands, 0);
        assert_eq!(app.audio_opts().leveling, 1.0);
    }

    #[test]
    fn selecting_a_preset_reaches_the_options_the_render_uses() {
        let mut app = AudioApp::default();
        app.load_preset(AudioPreset::Speech);
        let o = app.audio_opts();
        let v = AudioPreset::Speech.values();
        assert_eq!(o.window, v.window);
        assert_eq!(o.bands, v.bands);
        assert_eq!(o.max_voices, v.max_voices);
        assert_eq!(o.peak_gate, v.peak_gate);
        assert_eq!(o.release_ms, v.release_ms);
        assert!(
            o.release_ms < DEFAULT_RELEASE_MS,
            "the speech preset's whole point is a shorter release than the default"
        );
    }

    #[test]
    fn every_control_is_still_editable_after_a_preset_is_applied() {
        let mut app = AudioApp::default();
        app.load_preset(AudioPreset::Tonal);
        assert_eq!(app.audio_opts().window, 16384);

        // The user then edits, exactly as the widgets do.
        app.opts.window = 2048;
        app.opts.bands = Some(30);
        app.opts.max_voices = 7;
        app.opts.peak_gate = 3.0;
        app.opts.release_ms = 12.0;

        let o = app.audio_opts();
        assert_eq!(o.window, 2048);
        assert_eq!(o.bands, Some(30));
        assert_eq!(o.max_voices, 7);
        assert_eq!(o.peak_gate, 3.0);
        assert_eq!(o.release_ms, 12.0);
        // The dropdown still SHOWS what was last applied; it does not silently
        // snap the edits back.
        assert_eq!(app.preset, AudioPreset::Tonal);
    }

    #[test]
    fn switching_presets_replaces_every_value_rather_than_merging_them() {
        let mut app = AudioApp::default();
        app.load_preset(AudioPreset::Tonal);
        app.load_preset(AudioPreset::Speech);
        let o = app.audio_opts();
        assert_eq!(o.window, AudioPreset::Speech.values().window);
        assert_eq!(o.bands, AudioPreset::Speech.values().bands);
        assert_eq!(o.release_ms, AudioPreset::Speech.values().release_ms);

        // Including the one preset whose band count is "every band": going
        // from a limited span back to the full one must actually clear it.
        app.load_preset(AudioPreset::Chiptune);
        assert_eq!(app.audio_opts().bands, None, "chiptune wants the full span");
    }

    // The Bands checkbox's shadow value must survive a round trip through
    // "every band" without resetting a tuned span.
    #[test]
    fn the_band_limit_remembers_its_value_across_a_round_trip() {
        let mut app = AudioApp::default();
        app.opts.bands = Some(42);
        app.bands_value = 42;
        // Untick.
        app.opts.bands = None;
        assert_eq!(app.audio_opts().bands, None);
        // Re-tick, the way the checkbox handler does.
        app.opts.bands = Some(app.bands_value);
        assert_eq!(app.audio_opts().bands, Some(42));
    }

    // Only multiples of 12 land on real semitones.
    #[test]
    fn every_subdiv_the_widget_offers_is_accepted_by_the_renderer() {
        for s in SUBDIVS {
            assert_eq!(s % 12, 0, "the dropdown must offer multiples of 12 only");
            let mut app = AudioApp::default();
            app.opts.subdiv = s;
            assert!(app.validate().is_ok(), "--subdiv {s} must be renderable");
            assert!(app.live_cost().is_ok(), "--subdiv {s} must be costable");
        }
    }

    #[test]
    fn a_subdiv_off_the_semitone_grid_is_refused_by_the_readout_and_the_button() {
        let mut app = AudioApp::default();
        app.opts.subdiv = 14;
        let err = app.validate().expect_err("14 is not a multiple of 12");
        assert!(err.contains("--subdiv must be a multiple of 12"), "{err}");
        assert_eq!(
            app.live_cost().expect_err("the readout must refuse it too"),
            err,
            "the readout and the button must refuse with the same words"
        );
    }

    // 0 is the documented "every band" escape hatch in bank mode and a save
    // with no speakers in voice mode.
    #[test]
    fn a_zero_speaker_count_is_legal_in_bank_mode_and_refused_after_switching_to_voice() {
        let mut app = AudioApp::default();
        app.opts.max_voices = 0;
        app.mode = AudioMode::Bank;
        assert!(app.validate().is_ok(), "0 = every band is a legal bank render");
        assert_eq!(app.min_voices(), 0, "bank mode's slider must allow 0");

        app.mode = AudioMode::Voice;
        let err = app.validate().expect_err("0 speakers is not a build");
        assert!(err.contains("--max-voices must be at least 1"), "{err}");
        assert_eq!(app.min_voices(), 1, "voice mode's slider must not offer 0");
    }

    #[test]
    fn the_speaker_control_is_labelled_for_the_mode_it_is_in() {
        let mut app = AudioApp::default();
        app.mode = AudioMode::Bank;
        let bank_label = app.max_voices_label();
        let bank_hint = app.max_voices_hint();
        app.mode = AudioMode::Voice;
        assert_ne!(bank_label, app.max_voices_label());
        assert_ne!(bank_hint, app.max_voices_hint());
        // And each says which meaning it is.
        assert!(bank_hint.contains("UPPER BOUND"));
        assert!(app.max_voices_hint().contains("NUMBER OF SPEAKERS BUILT"));
    }

    #[test]
    fn the_band_grid_controls_are_shown_only_where_they_do_something() {
        let mut app = AudioApp::default();
        app.mode = AudioMode::Bank;
        assert!(app.shows_band_grid_controls());
        app.mode = AudioMode::Voice;
        assert!(!app.shows_band_grid_controls());
    }

    // Both the readout and `generate` read the one `audio_opts()`.
    #[test]
    fn the_live_estimate_follows_every_setting_that_changes_the_build() {
        let mut app = with_source(60.0);
        let base = app.live_cost().expect("default options are renderable");
        assert_eq!(base.speakers, 79, "the full span at --subdiv 12");
        assert!(base.frames > 0, "a 60s source must produce frames");

        // Fewer bands -> fewer speakers, fewer streams, fewer gates.
        app.opts.bands = Some(24);
        let narrow = app.live_cost().unwrap();
        assert_eq!(narrow.speakers, 24);
        assert!(narrow.gates < base.gates);

        // A longer window -> fewer analysis frames for the same audio.
        app.opts.bands = None;
        let short_window = app.live_cost().unwrap().frames;
        app.opts.window = 32768;
        assert!(
            app.live_cost().unwrap().frames < short_window,
            "a longer window must yield fewer frames over the same audio"
        );

        // A lower frame cap binds.
        app.opts.window = 4096;
        app.opts.max_frames = 10;
        assert_eq!(app.live_cost().unwrap().frames, 10);
    }

    #[test]
    fn the_readout_counts_two_streams_per_speaker_in_voice_mode() {
        let mut app = with_source(30.0);
        app.mode = AudioMode::Voice;
        app.opts.max_voices = 16;
        let c = app.live_cost().unwrap();
        assert_eq!(c.speakers, 16);
        assert_eq!(c.streams, 32, "one pitch stream and one volume stream per voice");
        assert_eq!(c.elements, 32 * c.frames);
    }

    // Reports the single-bank floor rather than inventing a frame count.
    #[test]
    fn an_unprobed_source_still_costs_everything_that_does_not_depend_on_length() {
        let app = AudioApp::default();
        assert_eq!(app.estimated_frames(), None, "nothing picked, nothing probed");
        let c = app.live_cost().expect("the options alone are renderable");
        assert_eq!(c.speakers, 79);
        assert_eq!(c.banks, 1);
        assert_eq!(c.frames, 0);
    }

    #[test]
    fn a_source_with_no_duration_hint_has_no_frame_estimate() {
        let mut app = AudioApp::default();
        app.input = Some(PickedAudio {
            path: std::path::PathBuf::from("stream.mkv"),
            name: "stream.mkv".to_string(),
            info: Some(AudioInfo { sample_rate: 48_000, duration_hint: None }),
        });
        assert_eq!(app.estimated_frames(), None);
        assert!(app.live_cost().is_ok());
    }

    #[test]
    fn the_frame_cap_is_clamped_the_way_the_cli_clamps_it() {
        let mut app = AudioApp::default();
        app.opts.max_frames = usize::MAX;
        assert_eq!(app.audio_opts().max_frames, crate::anim::pack::MAX_FRAMES);
        app.opts.max_frames = 0;
        assert_eq!(app.audio_opts().max_frames, 1, "0 frames is not a render");
    }

    #[test]
    fn every_window_the_widget_offers_is_costable() {
        for w in WINDOW_SIZES {
            let mut app = with_source(120.0);
            app.opts.window = w;
            let c = app.live_cost().unwrap_or_else(|e| panic!("window {w}: {e}"));
            assert!(c.frames > 0, "window {w} must still yield frames over 120s");
        }
    }

    #[test]
    fn the_bands_slider_covers_every_preset_and_the_widest_legal_span() {
        for p in AudioPreset::ALL {
            if let Some(b) = p.values().bands {
                assert!(
                    BANDS_RANGE.contains(&b),
                    "{}: {b} bands is off the slider",
                    p.name()
                );
            }
        }
        // The widest subdivision the Subdiv dropdown offers, which is what
        // sets the top of the range.
        let widest = SUBDIVS.iter().copied().max().expect("non-empty");
        let max_tonal = crate::audio::bands::max_tonal_bands(widest).expect("valid subdiv");
        assert!(
            *BANDS_RANGE.end() <= max_tonal + 2,
            "the slider must not offer more bands than any subdivision could hold"
        );
    }

    // Gain above 1.0 only clamps; pinning the constant here since no test
    // can click the widget itself.
    #[test]
    fn the_gain_default_sits_at_the_top_of_the_offered_range() {
        assert_eq!(
            AudioApp::default().audio_opts().gain,
            1.0,
            "1.0 is both the default and the maximum the slider offers -- above it the \
             pipeline only clamps"
        );
    }

    // Drives the real pipeline (`analyze` -> `build_speaker_world`, the same
    // calls `generate` and `main.rs` make) rather than checking the estimator
    // against its own formula -- a readout that disagreed with its render has
    // shipped before, and only building the thing would have caught it.
    #[test]
    fn the_readout_describes_the_world_the_bank_pipeline_actually_builds() {
        let clip = tone(4.0);
        let mut app = AudioApp::default();
        // A narrow span and a preset's window, so this exercises settings a
        // user would really be on rather than only the defaults.
        app.opts.bands = Some(12);
        app.opts.window = 2048;
        app.opts.max_voices = 6;
        app.input = Some(PickedAudio {
            path: std::path::PathBuf::from("tone.wav"),
            name: "tone.wav".to_string(),
            info: Some(clip.info()),
        });

        let predicted = app.live_cost().expect("renderable");
        let opts = app.audio_opts();
        let track = analyze(&clip, &opts, &mut NoProgress).expect("analyze");
        let world = build_speaker_world(&track, &opts).expect("build");

        assert_eq!(
            predicted.frames, track.frame_count,
            "the readout's frame count must be the analyser's own, not a duration-times-fps \
             guess"
        );
        assert_eq!(predicted.speakers, track.plan.len(), "speakers");
        assert_eq!(predicted.gates, built_gates(&world), "gates");
        assert_eq!(predicted.wires, world.wires.len(), "wires");
        assert_eq!(predicted.bricks, world.bricks.len(), "bricks");
    }

    // Voice mode writes two streams per speaker, so a readout right about
    // bank mode could still be exactly half right here.
    #[test]
    fn the_readout_describes_the_world_the_voice_pipeline_actually_builds() {
        let clip = tone(4.0);
        let mut app = AudioApp::default();
        app.mode = AudioMode::Voice;
        app.opts.max_voices = 6;
        app.opts.window = 2048;
        app.input = Some(PickedAudio {
            path: std::path::PathBuf::from("tone.wav"),
            name: "tone.wav".to_string(),
            info: Some(clip.info()),
        });

        let predicted = app.live_cost().expect("renderable");
        let opts = app.audio_opts();
        let streams = analyze_voices(&clip, &opts, &mut NoProgress).expect("analyze");
        let world = build_voice_world(&streams, &opts).expect("build");

        assert_eq!(predicted.frames, streams.frame_count, "frames");
        assert_eq!(predicted.speakers, streams.voice_count(), "speakers");
        assert_eq!(predicted.gates, built_gates(&world), "gates");
        assert_eq!(predicted.wires, world.wires.len(), "wires");
        assert_eq!(predicted.bricks, world.bricks.len(), "bricks");
    }

    #[test]
    fn every_preset_renders_a_real_world_at_the_settings_it_carries() {
        let clip = tone(3.0);
        for p in AudioPreset::ALL {
            let mut app = AudioApp::default();
            app.load_preset(p);
            app.input = Some(PickedAudio {
                path: std::path::PathBuf::from("tone.wav"),
                name: "tone.wav".to_string(),
                info: Some(clip.info()),
            });
            let predicted = app
                .live_cost()
                .unwrap_or_else(|e| panic!("{}: readout refused it: {e}", p.name()));
            let opts = app.audio_opts();
            let track = analyze(&clip, &opts, &mut NoProgress)
                .unwrap_or_else(|e| panic!("{}: analyze refused it: {e}", p.name()));
            let world = build_speaker_world(&track, &opts)
                .unwrap_or_else(|e| panic!("{}: build refused it: {e}", p.name()));
            assert_eq!(predicted.gates, built_gates(&world), "{}: gates", p.name());
            assert_eq!(predicted.frames, track.frame_count, "{}: frames", p.name());
        }
    }

    // Inner Radius and Max Distance are plain DragValues with nothing coupling
    // them, so both `validate` (draw_submit's gate) and `live_cost`
    // (draw_cost's) must refuse an inverted pair, or the readout would show a
    // cost beside a "Cannot render".
    #[test]
    fn an_inverted_attenuation_pair_is_refused_before_the_render() {
        for mode in [AudioMode::Bank, AudioMode::Voice] {
            let mut app = with_source(240.0);
            app.mode = mode;
            assert!(
                app.validate().is_ok(),
                "{mode:?}: the defaults must be renderable, or this test proves nothing"
            );

            app.opts.inner_radius = 5000.0;
            app.opts.max_distance = 4000.0;

            let err = app.validate().expect_err(&format!(
                "{mode:?}: an inner radius outside the audible range must be refused"
            ));
            assert!(
                err.contains("--inner-radius") && err.contains("--max-distance"),
                "{mode:?}: the refusal must name both controls: {err}"
            );
            assert!(
                app.live_cost().is_err(),
                "{mode:?}: the cost readout must not describe a build that cannot happen"
            );
        }
    }

    // A non-positive or non-finite radius encodes fine and is audible nowhere.
    #[test]
    fn a_zero_or_non_finite_radius_is_refused_before_the_render() {
        for (inner, max) in [(0.0, 4000.0), (400.0, 0.0), (f32::NAN, 4000.0)] {
            let mut app = with_source(240.0);
            app.opts.inner_radius = inner;
            app.opts.max_distance = max;
            assert!(
                app.validate().is_err(),
                "inner {inner} / max {max} must be refused before a render starts"
            );
        }
    }

    #[test]
    fn the_two_modes_build_different_graphs_from_the_same_source() {
        let mut app = with_source(45.0);
        app.opts.max_voices = 20;
        let bank = app.live_cost().unwrap();
        app.mode = AudioMode::Voice;
        let voice = app.live_cost().unwrap();
        assert_ne!(bank.speakers, voice.speakers);
        assert_ne!(bank.gates, voice.gates);
        assert_eq!(voice.speakers, 20, "voice mode builds Max Voices speakers");
    }
}
