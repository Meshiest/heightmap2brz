pub mod map;
pub mod opt;
pub mod util;
// NOT `pub mod text;`. `src/text.rs` is a LIBRARY module (`heightmap::text`),
// and declaring it here as well would compile a second, distinct copy of every
// type in it into this binary -- at which point the `TextOptions` the `--text`
// path builds is a different type from the `heightmap::text::TextOptions` that
// `AnimOptions::text` holds, and the one cannot be assigned to the other. The
// two text paths must share one options builder (see `text_options`), so they
// must share one type.

#[cfg(not(target_arch = "wasm32"))]
mod progress_cli;

#[cfg(not(target_arch = "wasm32"))]
use crate::{map::*, opt::*, util::*};
#[cfg(not(target_arch = "wasm32"))]
use brdb::World;
#[cfg(not(target_arch = "wasm32"))]
use brdb::assets::bricks::{
    PB_DEFAULT_BRICK, PB_DEFAULT_MICRO_BRICK, PB_DEFAULT_SMOOTH_TILE, PB_DEFAULT_STUDDED,
    PB_DEFAULT_TILE,
};
#[cfg(not(target_arch = "wasm32"))]
use clap::clap_app;
#[cfg(not(target_arch = "wasm32"))]
use env_logger::Builder;
#[cfg(not(target_arch = "wasm32"))]
use heightmap::{
    anim::{
        AnimEncoding, AnimMode,
        bricks::{AnimOptions, DisplayBrickStyle},
        color_bricks, cost,
        pack::MAX_FRAMES,
        subtitle_display::{DEFAULT_SUBTITLE_LIFT, DEFAULT_SUBTITLE_SCALE},
        text_layout,
    },
    audio::{
        AudioMode,
        backend::{AudioBackend, open_audio_track},
        cost as audio_cost,
        speakers::{build_midi_event_world, build_speaker_world, build_voice_world},
        track::{AudioOptions, SynthWave, analyze},
        voices::{MAX_PITCH_SNAP_CENTS, analyze_voices},
    },
    midi::{MidiOptions, ToneAssignment, analyze_midi, discover},
    subs::{self, Subtitles},
    text::*,
    video::{
        backend::{self, Backend},
        ffmpeg::{DownloadConsent, ensure_ffmpeg},
        scale::{Filter, FitMode, estimated_frame_count, max_frames_error},
        source::{Source, decode, is_animated, is_video_path},
        stream::{AdaptedSource, FrameSource},
    },
};
#[cfg(not(target_arch = "wasm32"))]
use log::{LevelFilter, error, info, warn};
#[cfg(not(target_arch = "wasm32"))]
use std::{boxed::Box, io::Write, path::PathBuf};

/// The CLI is native-only; the web build ships only the GUI bin.
#[cfg(target_arch = "wasm32")]
fn main() {}

/// Print via `error!` and exit non-zero. `error!` alone leaves `main`
/// returning `()` -> exit 0, which a caller scripting `&& upload out.brz`
/// would not catch.
#[cfg(not(target_arch = "wasm32"))]
macro_rules! fail {
    ($($arg:tt)+) => {
        fail(format!($($arg)+))
    };
}

/// The whole CLI surface, as its own function so the tests at the bottom of
/// this file can parse argument lists against the REAL flag set rather than a
/// hand-copied stand-in that could drift away from it.
#[cfg(not(target_arch = "wasm32"))]
fn cli() -> clap::App<'static, 'static> {
    clap_app!(heightmap =>
        (version: env!("CARGO_PKG_VERSION"))
        (author: "github.com/Meshiest")
        (about: "Converts heightmap images (PNG/JPG) to Brickadia save files")
        (@arg INPUT: +required +multiple "Input heightmap image files (PNG/JPG)")
        (@arg output: -o --output +takes_value "Output file (BRDB, BRZ)")
        (@arg colormap: -c --colormap +takes_value "Input colormap image (PNG/JPG)")
        (@arg vertical: -v --vertical +takes_value "Vertical scale multiplier (default 1)")
        (@arg size: -s --size +takes_value "Brick stud size (default 1)")
        (@arg cull: --cull "Automatically remove bottom level bricks and fully transparent bricks")
        (@arg tile: --tile "Render bricks as tiles")
        (@arg smooth: --smooth "Render bricks as smooth tiles")
        (@arg micro: --micro "Render bricks as micro bricks")
        (@arg stud: --stud "Render bricks as stud cubes")
        (@arg terrain: --terrain "Render the terrain as SMOOTH micro bricks instead of flat-topped tiles: every pixel gets a sloped top chosen from Brickadia's micro wedge family (ramp, wedge corner, inner corner, and the stacked diagonal corner+triangle), fitted to the four shared vertex heights around it. Heights are sampled on a shared (w+1)x(h+1) vertex grid so neighbouring cells MEET rather than step. Replaces --tile/--smooth/--micro/--stud and the optimizers, which have no meaning once the top face is not flat")
        (@arg rampify: --rampify "Rampify the terrain with Wrapperup's rampifier: fit full-size ramps, wedges and ramp corners onto the height column surface and fill the rest with plain bricks. Coarser than --terrain (one plate of vertical resolution, runs of at most 4 studs) but uses ordinary bricks rather than micro pieces. Replaces --tile/--smooth/--micro/--stud and the optimizers")
        (@arg wedge: --wedge "Build TERRACED wedge terrain: tops stay flat, every height is a whole terrace step, and the outlines of the terraces are cut at 45 degrees by vertical side wedges (PB_DefaultSideWedge) -- convex corners chamfered, concave corners filled, collinear staircases merged into single large wedges, flat tops greedy-merged into boxes. Unbuildable configurations (diagonal crossings, spikes) are eroded first. Unlike --terrain and --rampify, slopes are not approximated: this is the terraced 'brick terrain' look of hand-built Brickadia maps. Replaces --tile/--smooth/--micro/--stud and the optimizers")
        (@arg prefab: --prefab "Heightmap/image renders: write a PREFAB bundle instead of a world, so the save can be dropped in Brickadia's Prefabs folder and spawned from the prefab browser rather than loaded as a level")
        (@arg snap: --snap "Snap bricks to the brick grid")
        (@arg lrgb: --lrgb "DEPRECATED and ignored. Colormap pixels always go into the bricks without conversion, because a save file stores brick colours in the same encoding as an image")
        (@arg img: -i --img "Make the heightmap flat and render an image")
        (@arg glow: --glow "Make the heightmap (or animation display) glow at 0 intensity")
        (@arg srgb2lin: --("srgb-to-linear") "Animation: convert sRGB frame colors to linear before encoding (use if the render looks too bright)")
        (@arg hdmap: --hdmap "Using a high detail rgb color encoded heightmap")
        (@arg nocollide: --nocollide "Disable brick collision")
        (@arg greedy: --greedy "Use greedy optimization")
        (@arg text: --text "Render the input image as TextDisplay component bricks")
        (@arg fillchar: --("fill-char") +takes_value "Text mode: glyph for opaque pixels (default █)")
        (@arg emptychar: --("empty-char") +takes_value "Text mode: glyph for transparent pixels (default space)")
        (@arg charrepeat: --("char-repeat") +takes_value "Text mode: glyphs emitted per pixel (default 2)")
        (@arg alphathreshold: --("alpha-threshold") +takes_value "Text mode: alpha below this is transparent (default 128)")
        (@arg lineheight: --("line-height-world") +takes_value "Text mode: world units per pixel row / pixel size (default 1)")
        (@arg font: --font +takes_value "Text mode: font preset (monaspace, iosevka, orbitron; default monaspace)")
        (@arg braille: --braille "Text mode: monochrome braille glyphs (8 pixels per character)")
        (@arg blocks: --blocks "Text mode: monochrome quadrant-block glyphs (4 pixels per character)")
        (@arg lumathreshold: --("luma-threshold") +takes_value "Braille/blocks: pixels at least this bright are drawn (default 128)")
        (@arg invert: --invert "Braille/blocks: draw dark pixels instead of bright ones")
        (@arg material: --material +takes_value "Text mode: material (unlit, graffiti, plastic, metallic, glow, translucent, glass; default unlit)")
        (@arg audiomode: --("audio-mode") +takes_value "Audio output mode: 'bank' = Pitch-Per-Speaker (a fixed bank of ~79 speakers, each owning one pitch, only their volumes written -- best for speech and broadband material) or 'voice' = Pitch Switching (--max-voices speakers that TRACK spectral peaks, re-pitching every frame -- no band grid, so no tuning error; best for tonal material such as piano)")
        (@arg audiotrack: --("audio-track") +takes_value "Audio: which audio stream to decode, 0 = first (default 0). Dual-audio releases commonly carry the original language first and the dub second, so which is 'first' is a container-ordering accident. Needs the ffmpeg backend")
        (@arg audiobands: --bands +takes_value "Audio: total speakers including noise bands. Tonal bands sit on exact equal-tempered steps, so this SELECTS THE SPAN (the steps nearest A440), not the spacing. Default: every step the emitter's pitch range holds -- 79 tonal at --subdiv 12 (F#1..C8), 159 at 24")
        (@arg audiosubdiv: --subdiv +takes_value "Audio: tonal bands per octave. MUST BE A MULTIPLE OF 12 -- only then does every band land on a real semitone, and anything else was heard in game as out of tune (14 sharp, 18 flat). 12 (default, one per semitone) is preferred; 24 (quarter-tones) halves the error on a source not tuned to A440 but splits each note across two speakers, which was heard as 'a little weird'")
        (@arg audionoise: --("noise-bands") +takes_value "Audio: white/pink noise speakers carrying the energy off either end of the tonal range, 0-2 (default 0 -- reported worse in game on speech, piano and a pop mix alike; 2 is worth trying on percussion, which nothing else in the bank can render)")
        (@arg audiofps: --("audio-fps") +takes_value "Audio: analysis frames per second (default 30)")
        (@arg audiowindow: --window +takes_value "Audio: STFT (short-time Fourier transform) window size (default 4096)")
        (@arg audiogain: --gain +takes_value "Audio: post-normalization gain (default 1.0)")
        (@arg audioleveling: --leveling +takes_value "Audio: per-frame leveling, 0 = keep the track's dynamics (default), 1 = full AGC")
        (@arg audiomaxvoices: --("max-voices") +takes_value "Audio: in --audio-mode bank, an upper bound on bands sounding at once (default 12; 0 = every band); in --audio-mode voice, the number of speakers built, each sounding most frames (32 simultaneous emitters are verified to work). In bank mode, --peak-gate is usually the real limit")
        (@arg audiopeakgate: --("peak-gate") +takes_value "Audio (Pitch-Per-Speaker, --audio-mode bank): how far above the mean of its neighbourhood a band must stand to count as a note, as an amplitude ratio (default 1.5 = 3.5 dB). THIS, not --max-voices, is what limits a dense frame -- lower it to sound more bands at once. 1.0 disables it (every local maximum), which is the pre-gate wall-of-sound behaviour")
        (@arg audioattack: --attack +takes_value "Audio: envelope attack in milliseconds (default 10). How fast a speaker's level RISES toward what the analysis measured. Short by design, so struck notes stay punchy")
        (@arg audiorelease: --release +takes_value "Audio: envelope release in milliseconds (default 150). How fast a speaker's level FALLS, including after its band stops being selected. THIS is what stops one-frame selections sounding like beeps. 0 for both = no smoothing, i.e. the pre-envelope renderer")
        (@arg audiovoicerelease: --("voice-release") +takes_value "Audio (Pitch Switching, --audio-mode voice): how long a voice takes to fade to EXACTLY zero once its partial has gone, in milliseconds (default 50). Distinct from --release, which is a time constant on a voice that is still sounding and never reaches zero. THE MATERIAL DECIDES IT: a spoken phoneme is 50-100 ms long and anything near 150 smears across it (speech came back 'disembodied'), while a sustained note can afford more -- 150 restores the old behaviour. 0 clamps to one analysis frame, the shortest fade the format can express")
        (@arg audiopitchsnap: --("pitch-snap") +takes_value "Audio (Pitch Switching, --audio-mode voice): pull a continuing voice onto the nearest equal-tempered semitone when it is within this many cents (default 0 = off, max 50 = half a semitone, i.e. always). Trades vibrato and glissando for a rock-steady pitch; the frame-to-frame wobble is already smoothed without it")
        (@arg audioinner: --("inner-radius") +takes_value "Audio: speaker no-attenuation radius in units (default 15; 10 units = 1 brick). Raise it (e.g. 400) for one flat equal-level field across a large build")
        (@arg audiomaxdist: --("max-distance") +takes_value "Audio: speaker audible range in units (default 400; 10 units = 1 brick). Raise it (e.g. 4000) to be heard across a big build")
        (@arg audiospeakersinchip: --("speakers-in-chip") "Audio: place the speaker cluster INSIDE the microchip's own inner grid instead of beside it on the main grid, so the whole audio device is one portable microchip. The speakers play from the chip's ORIGIN regardless of their inner-grid layout (an AudioEmitter on a microchip inner grid emits from the chip's world position), so the layout is physical placement only, not spatial audio. Default off (speakers beside the chip)")
        (@arg audiosynth: --synth +takes_value "Audio: the synth waveform every TONAL band plays through -- sine (default), square, triangle or sawtooth. Applies to tonal bands in BOTH modes; white/pink --noise-bands keep their own noise assets and are unaffected. Default sine renders exactly as before this flag existed")
        (@arg midi: --midi "MIDI (midi2brick): read the input as a Standard MIDI File and build an EVENT-BASED speaker world -- each track's notes are stored as spans and played by a runtime playhead. The input is the .mid; -o is the .brz. Tones: a waveform is picked per instrument from its General MIDI program (bass/electric guitar sawtooth, lead square, piano/vocals sine, ...); --synth forces one wave for the whole file. Per-track manual tones are a GUI feature. Reuses --inner-radius, --max-distance, --gain, --no-loop, --no-control-buttons, --speakers-in-chip and --polyphony-cap")
        (@arg midilist: --("midi-list") "MIDI: with --midi, print the discovered instruments (name, channel, note count, max polyphony) and the file's format/duration/tempo, then exit without building")
        (@arg midipolyphony: --("polyphony-cap") +takes_value "MIDI: maximum speakers per instrument, however many notes it plays at once (default 8). A busier instrument steals its oldest sounding note")
        (@arg midirate: --("playback-rate") +takes_value "MIDI: playback speed multiplier baked into the clock (default 1.0; 2.0 = double speed, 0.5 = half). The generated Rate pin still overrides it at runtime")
        (@arg nopercussion: --("no-percussion") "MIDI: skip the percussion channel (10). By default each drum note plays a oneshot sample, mapped from its General MIDI drum note through a fold table; this builds only the pitched instruments")
        (@arg animmode: --("anim-mode") +takes_value "Animation output mode (brick, text). 'brick' builds one display brick per pixel, driven by the encoding --anim-encoding selects. 'text' builds one animated Component_TextDisplay per BAND of image rows instead -- roughly two orders of magnitude fewer gates (a 192x108 clip is 113 gates against 4613), at the cost of glyph-grid rendering rather than real bricks. Text mode reuses --font, --char-repeat, --fill-char, --empty-char, --alpha-threshold and --line-height-world, and adds --colors")
        (@arg animcolors: --("colors") +takes_value "Text mode: quantize to at most N colours with a median-cut palette (default 0 = full 24-bit colour). Fewer colours means longer same-colour runs and a smaller save; useful values are 16 to 64")
        (@arg animencoding: --("anim-encoding") +takes_value "Animation pixel encoding (hex, color-array; default hex). 'hex' packs each frame into a shared RRGGBB string per chunk; 'color-array' gives each pixel its own colour array -- fewer gate evaluations and no string work, at the cost of more host RAM to build")
        (@arg animfps: --fps +takes_value "Animation output frame rate (default 10)")
        (@arg animstart: --start +takes_value "Start offset into the source, seconds")
        (@arg animduration: --duration +takes_value "Duration taken from the source, seconds")
        (@arg animmaxframes: --("max-frames") +takes_value "Cap on emitted frames (default 1048560; frames past 65535 spill into extra arrays)")
        (@arg animwidth: --width +takes_value "Target width in pixels")
        (@arg animheight: --height +takes_value "Target height in pixels")
        (@arg animfit: --fit +takes_value "Fit mode (exact, contain, cover; default contain)")
        (@arg animfilter: --filter +takes_value "Resample filter (lanczos, nearest; default lanczos)")
        (@arg subtitles: --subtitles +takes_value "Animation: render this subtitle file (.srt, .ass/.ssa) as a single wired TextDisplay overlaying the bottom of the screen -- 2 gates for the whole track, centred and outlined, at vector-glyph size rather than the screen's pixel grid. The file is read in SOURCE time, so --start is honoured. Mutually exclusive with --subtitle-track")
        (@arg subtitletrack: --("subtitle-track") +takes_value "Animation: extract subtitles from the input container instead, 0 = the first SUBTITLE stream (not the first stream). Text tracks only -- a PGS/DVD/DVB track is an image sequence with no text in it, and is refused by name rather than rendered as an empty track. Needs ffmpeg. Mutually exclusive with --subtitles")
        (@arg subtitlescale: --("subtitle-scale") +takes_value "Animation: how much bigger a subtitle line is than one row of the screen (default 6). At 192 px wide the screen is hundreds of glyph cells across while a subtitle line is 40-60 characters, so at equal size the text would occupy a seventh of the width; 6 covers about half. Ignored without --subtitles/--subtitle-track")
        (@arg subtitlelift: --("subtitle-lift") +takes_value "Animation: world units to lift the subtitle anchor toward the top of the picture (default 8, measured by eye against --anim-mode text at 192x108 with --subtitle-scale 6). --anim-mode brick/color-array lay their screen flat, so the lift there moves the OPPOSITE horizontal axis and is unverified by eye. Refused, not clamped, if it would push the anchor to a negative coordinate on a picture shorter than the lift. Ignored without --subtitles/--subtitle-track")
        (@arg externalclock: --("external-clock") "Expose Frame as a chip input instead of running a timer")
        (@arg noloop: --("no-loop") "Play through once and stop on the last frame, instead of looping forever (the default). Applies to video and audio alike. Inert with --external-clock, which builds no timer at all")
        (@arg nocontrolbuttons: --("no-control-buttons") "Do NOT pre-generate the three physical Pause/Restart/Resume BUTTON bricks on the main grid, wired into the clock's control pins (default: buttons ON, so a fresh render is pausable/restartable/resumable with no manual wiring). Video and audio alike. Adds 9 main-grid bricks and 6 wires per render; no extra microchip gate. Inert with --external-clock, which builds no timer and so exposes no control pins")
        (@arg animbrickstyle: --("brick-style") +takes_value "Animation display-brick style (micro, tile; default micro)")
        (@arg animpixelextent: --("pixel-extent") +takes_value "Animation display-brick half-extent in units (default 1; 1 = smallest, 2 units wide; tile style is always 4 units tall)")
        (@arg yesdownload: --yes "Consent to downloading ffmpeg if it is missing and a video backend needs it")
        (@arg nodownload: --("no-download") "Never download ffmpeg; error instead if it is missing")
        (@arg backend: --backend +takes_value "Video decode backend (auto, builtin, ffmpeg; default auto)")
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    Builder::new()
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .filter(None, LevelFilter::Info)
        .init();

    let matches = cli().get_matches();

    // get files from matches
    let heightmap_files = matches
        .values_of("INPUT")
        .unwrap()
        .map(|s| PathBuf::from(s))
        .collect::<Vec<_>>();
    let colormap_file = matches
        .value_of("colormap")
        .map(PathBuf::from)
        .unwrap_or(heightmap_files[0].clone());
    let out_file = matches
        .value_of("output")
        .unwrap_or("./out.brz")
        .to_string();
    // **Checked HERE, before anything is decoded.** `write_world` owns the rule
    // and is the last call in every branch, so `-o out` used to decode the whole
    // clip, print a cost line, build the world and only then refuse it -- a
    // 40-minute video or a full-song audio render thrown away over a name that
    // was knowably wrong before a byte was read. The message is `write_world`'s
    // own, so the two cannot drift.
    {
        let lower = out_file.to_lowercase();
        if !lower.ends_with(".brz") && !lower.ends_with(".brdb") {
            fail!("output file must end with .brz or .brdb (got '{out_file}')");
        }
    }

    // Task 3 added these; this is where they finally do something.
    // `Ask` is the default because it downgrades to `Never` on a
    // non-terminal stdin, so a headless run errors rather than hanging.
    // Validated here regardless of render mode (cheap, and catches a
    // contradictory pair immediately); only a video render ever actually
    // consults `consent`, via `ensure_ffmpeg`.
    let consent = match (matches.is_present("yesdownload"), matches.is_present("nodownload")) {
        (true, true) => fail!("--yes and --no-download contradict each other"),
        (true, false) => DownloadConsent::Always,
        (false, true) => DownloadConsent::Never,
        (false, false) => DownloadConsent::Ask,
    };

    let backend_choice = match matches
        .value_of("backend")
        .map(|s| s.to_lowercase())
        .as_deref()
    {
        None | Some("auto") => Backend::Auto,
        Some("builtin") => Backend::Builtin,
        Some("ffmpeg") => Backend::Ffmpeg,
        Some(other) => {
            fail!("unknown --backend '{other}' (auto, builtin, ffmpeg)");
        }
    };

    // Two ways to name ONE subtitle track. Refused here rather than
    // silently preferring one, and refused before anything is read: a user
    // who passed both has two different tracks in mind and no way to know
    // which one a render used. Checked regardless of render mode, exactly as
    // `--yes`/`--no-download` above is -- a contradictory command line is
    // wrong whatever it was pointed at.
    if matches.is_present("subtitles") && matches.is_present("subtitletrack") {
        fail(
            "--subtitles and --subtitle-track name two different subtitle sources and \
             cannot be combined: --subtitles reads a file you supply, --subtitle-track \
             extracts one out of the input container. Pass exactly one",
        );
    }
    // Warned about and ignored rather than refused, the same way `--colormap`
    // is under `--anim-mode`: there is no animated screen for a subtitle to
    // sit below in the heightmap, `--text` or `--audio-mode` paths, but a
    // user who passed the flag and saw no subtitles would reasonably conclude
    // the feature was broken.
    if !matches.is_present("animmode") {
        for (flag, name) in [
            ("--subtitles", "subtitles"),
            ("--subtitle-track", "subtitletrack"),
            ("--subtitle-scale", "subtitlescale"),
            ("--subtitle-lift", "subtitlelift"),
        ] {
            if matches.is_present(name) {
                warn!(
                    "{flag} applies to --anim-mode renders only (a subtitle is wired to an \
                     animated screen's frame index, and the other paths have no frames)"
                );
            }
        }
    } else {
        // Likewise named out loud, one flag at a time: with no track to
        // size or lift, either number does nothing at all. Same
        // warn-and-ignore shape as `--colors` under brick mode.
        if matches.is_present("subtitlescale")
            && !matches.is_present("subtitles")
            && !matches.is_present("subtitletrack")
        {
            warn!(
                "--subtitle-scale is ignored without --subtitles or --subtitle-track: it sizes \
                 a subtitle line against one screen row, and there is no subtitle to size"
            );
        }
        if matches.is_present("subtitlelift")
            && !matches.is_present("subtitles")
            && !matches.is_present("subtitletrack")
        {
            warn!(
                "--subtitle-lift is ignored without --subtitles or --subtitle-track: it lifts \
                 the subtitle anchor toward the top of the picture, and there is no subtitle \
                 to lift"
            );
        }
    }

    // `--srgb-to-linear` shapes the animated hex encoder's colours
    // (`pack::Packer::linearize`) and nothing else; named out loud rather
    // than silently ignored on any other path.
    if matches.is_present("srgb2lin") && !matches.is_present("animmode") {
        warn!(
            "--srgb-to-linear applies to --anim-mode renders only (it converts a FRAME's \
             colours before they are encoded). The heightmap and --img paths convert nothing \
             at all; --text has no colour encoding to convert"
        );
    }

    // `--lrgb` disabled a default sRGB to linear conversion of the colormap.
    // That conversion is removed. A save file stores brick colours in the same
    // encoding as an image, thus the conversion only made each render darker
    // than its colormap. The flag stays as a no-op for existing scripts.
    if matches.is_present("lrgb") {
        warn!(
            "--lrgb is deprecated and has no effect. Colormap pixels always go into the \
             bricks without conversion, which is what --lrgb selected"
        );
    }

    // **Which branch wins is decided by the order these are checked in, so say
    // so.** The audio branch already names a dozen lesser flags it cannot
    // honour one at a time; the MODE flags -- the ones that decide which
    // renderer runs at all -- were the omission. `--text --anim-mode text`
    // renders the animated wired build and never mentions that `--text` was
    // dropped, which looks like `--text` being broken rather than overridden.
    if matches.is_present("audiomode") {
        for (flag, name) in [("--anim-mode", "animmode"), ("--text", "text")] {
            if matches.is_present(name) {
                warn!(
                    "--audio-mode takes precedence over {flag}; this render is audio only. \
                     Pass one output mode"
                );
            }
        }
    } else if matches.is_present("animmode") && matches.is_present("text") {
        warn!(
            "--anim-mode takes precedence over --text; this render is the ANIMATED wired \
             build, not the static text export. Pass one output mode"
        );
    }

    // These options apply to the heightmap only. If another renderer wins the
    // selection below, the code tells the user. This is the rule that
    // `--subtitles` above uses: a user who gives `--terrain --text` and gets a
    // text export would think that the terrain renderer is defective.
    if ["midi", "audiomode", "animmode", "text"]
        .iter()
        .any(|m| matches.is_present(m))
    {
        for flag in ["terrain", "rampify", "wedge", "prefab"] {
            if matches.is_present(flag) {
                warn!(
                    "--{flag} applies to heightmap and --img renders only; this render is \
                     built by another mode and ignores it"
                );
            }
        }
    }

    if matches.is_present("midi") {
        run_midi(&matches, &heightmap_files, &out_file);
        return;
    }

    if matches.is_present("audiomode") {
        run_audio(&matches, &heightmap_files, consent, &out_file);
        return;
    }

    if matches.is_present("animmode") {
        run_anim(&matches, heightmap_files, consent, backend_choice, &out_file);
        return;
    }

    if matches.is_present("text") {
        run_text(&matches, &heightmap_files, &out_file);
        return;
    }

    run_heightmap(&matches, &heightmap_files, &colormap_file, &out_file);
}

/// The `--audio-mode` render branch: builds `AudioOptions`, opens the first
/// input file on the audio backend, runs either the equal-tempered speaker
/// bank analysis or the spectral-peak-tracking voice analysis, and writes the
/// result. Split out of `main`'s dispatch verbatim -- every `fail!`/`warn!`
/// here still exits or logs under exactly the same condition it did inline.
#[cfg(not(target_arch = "wasm32"))]
fn run_audio(
    matches: &clap::ArgMatches,
    heightmap_files: &[PathBuf],
    consent: DownloadConsent,
    out_file: &str,
) {
    let mode = matches.value_of("audiomode").unwrap_or("bank");
    // Two renderers, one analysis front end. `bank` puts every speaker on
    // a fixed equal-tempered pitch and writes only volumes; `voice` builds
    // `--max-voices` speakers that follow spectral peaks and writes both
    // pitch and volume. See `audio::voices` for why the second exists.
    // `AudioMode::parse`, not a local match: the GUI offers the same two
    // choices and branches on the same distinctions (most sharply on what
    // `--max-voices` means), so both front ends read one type. The error
    // text is unchanged.
    let voice_mode = match AudioMode::parse(mode) {
        Ok(m) => m == AudioMode::Voice,
        Err(e) => fail!("{e}"),
    };

    // Everything below is a flag the audio path cannot honour. Each is
    // named out loud rather than quietly dropped: a render that silently
    // ignored `--width` would look like it worked and sound identical to
    // one that never got the flag at all.
    if heightmap_files.len() > 1 {
        warn!(
            "--audio-mode uses only the first input file; ignoring {} extra input(s)",
            heightmap_files.len() - 1
        );
    }
    if matches.is_present("colormap") {
        warn!("--audio-mode ignores --colormap");
    }
    for (flag, name) in [
        ("--width", "animwidth"),
        ("--height", "animheight"),
        ("--fit", "animfit"),
        ("--filter", "animfilter"),
    ] {
        if matches.is_present(name) {
            warn!("--audio-mode ignores {flag} (there is no image to scale)");
        }
    }
    // `--fps` and `--audio-fps` are separate flags on purpose (a video
    // render and an audio analysis are different rates), and reaching for
    // the wrong one is the easy mistake -- so say so rather than analysing
    // at the default 30 while the user believes they set it.
    if matches.is_present("animfps") {
        warn!("--audio-mode ignores --fps; use --audio-fps for the analysis rate");
    }
    // Voice mode has no band grid at all -- that is the whole reason it
    // exists -- so every flag that shapes the grid is meaningless to it.
    // Named out loud rather than dropped: a listener who passed `--subdiv
    // 24` and heard no change would reasonably conclude the render was
    // broken.
    if voice_mode {
        for (flag, name) in [
            ("--bands", "audiobands"),
            ("--subdiv", "audiosubdiv"),
            ("--noise-bands", "audionoise"),
        ] {
            if matches.is_present(name) {
                warn!(
                    "--audio-mode voice (Pitch Switching) ignores {flag}: there is no \
                     band grid, and --max-voices is the speaker count"
                );
            }
        }
        // The two modes both gate on prominence, but over different
        // neighbourhoods (a band's neighbours vs a fixed span of FFT bins),
        // so one number could not mean the same thing in both. Voice mode's
        // gate is a module constant.
        if matches.is_present("audiopeakgate") {
            warn!(
                "--peak-gate applies to --audio-mode bank (Pitch-Per-Speaker) only; \
                 Pitch Switching's peak gate is measured over FFT bins and is not \
                 exposed"
            );
        }
    } else if matches.is_present("audiovoicerelease") {
        // The other direction, and worth saying: a bank band never STOPS
        // EXISTING, so there is no note-off for this flag to time. Its
        // levels are shaped by `--release` alone.
        warn!(
            "--voice-release applies to --audio-mode voice (Pitch Switching) only; a \
             Pitch-Per-Speaker band has no note-off to time, and its level is shaped \
             by --release"
        );
    }

    // REJECTED, not ignored. `AudioOptions::external_clock` exists and
    // `build_speaker_world` reads none of it -- it always bakes its own
    // clock -- so accepting this flag would hand back a timer-driven save
    // while the user believes they got a chip input to drive.
    if matches.is_present("externalclock") {
        fail!(
            "--external-clock is not supported in audio mode yet: the speaker chip always \
             builds its own clock, so the flag would have no effect on the save"
        );
    }
    // Likewise rejected: nothing on the audio decode path seeks or trims
    // (`open_audio` takes a path and nothing else), so honouring either
    // would mean silently rendering the whole file from its start.
    for (flag, name) in [("--start", "animstart"), ("--duration", "animduration")] {
        if matches.is_present(name) {
            fail!(
                "{flag} is not supported in audio mode yet: the audio decode path cannot \
                 seek or trim. Use --max-frames to bound the render instead \
                 (frames = seconds x --audio-fps)"
            );
        }
    }

    // Parsed as an `AudioBackend`, not reused from `backend_choice`
    // above: the two enums take the same three spellings but are separate
    // types, and `open_audio` owns the audio-side fallback policy.
    let audio_backend = match matches
        .value_of("backend")
        .unwrap_or("auto")
        .parse::<AudioBackend>()
    {
        Ok(b) => b,
        Err(e) => fail!("{e}"),
    };

    // Parsed through `parse_arg`, never `.expect(...)`: a typo'd
    // `--bands abc` must be a CLI error, not a Rust panic trace. See the
    // same reasoning on `--width` in the video branch below.
    let audio_opts = match audio_options(&matches) {
        Ok(o) => o,
        Err(e) => fail!("{e}"),
    };
    // **Before the file is opened, let alone analysed.** The GUI's Generate
    // button refuses on exactly this call (`AudioApp::validate`), and every
    // rule in it is knowable from the flags alone -- an inverted
    // `--inner-radius`/`--max-distance` pair used to survive the entire
    // analysis and then kill `build_speaker_world`, printing the analysis
    // summary first and the refusal after it.
    if let Err(e) = audio_cost::check(
        if voice_mode {
            AudioMode::Voice
        } else {
            AudioMode::Bank
        },
        &audio_opts,
    ) {
        fail!("{e}");
    }

    let input = &heightmap_files[0];
    info!("Opening audio {}", input.display());
    let audio_track = match matches.value_of("audiotrack") {
        Some(s) => match s.parse::<usize>() {
            Ok(v) => v,
            Err(e) => fail!("--audio-track must be a non-negative integer: {e}"),
        },
        None => 0,
    };
    let source = match open_audio_track(input, audio_backend, consent, audio_track) {
        Ok(s) => s,
        Err(e) => fail!("{e}"),
    };

    let world = if voice_mode {
        let streams = match analyze_voices(
            source.as_ref(),
            &audio_opts,
            &mut progress_cli::CliProgress::new(),
        ) {
            Ok(s) => s,
            Err(e) => fail!("{e}"),
        };
        let st = &streams.stats;
        info!(
            "Analyzed {} frame(s) at {} fps across {} voice(s), {} bank(s)",
            streams.frame_count,
            streams.fps,
            streams.voice_count(),
            streams
                .frame_count
                .div_ceil(audio_opts.bank_size.max(1))
                .max(1),
        );
        // Reported every render, not hidden behind a probe. Each of these
        // answers a question that decides whether the mode worked, and
        // none is visible in the save: a mean lifetime near 1 frame is
        // chimes however right the pitches are, and the REAL voices per
        // frame is the number to compare between renders -- the flag value
        // is not, and a run where a gate rather than the flag was setting
        // the count has shipped before.
        info!(
            "  {:.2} voices sounding per frame (--max-voices {}), mean voice lifetime \
             {:.1} frames",
            st.mean_voices_per_frame(streams.frame_count),
            streams.voice_count(),
            st.mean_lifetime(),
        );
        info!(
            "  {} voice-frames tracked, mean {:.1} cents from equal temperament; {:.2}% \
             of sounding frame pairs jump more than a semitone",
            st.peak_count,
            st.mean_abs_cents,
            st.pitch_jump_fraction() * 100.0,
        );
        // The wobble metric, measured before and after smoothing over the
        // SAME frames -- two runs with different settings would not
        // continue the same voices, so their jitter would not be
        // comparable.
        info!(
            "  pitch jitter {:.1} cents rms as tracked -> {:.1} written \
             (--pitch-snap {})",
            st.raw_jitter_rms_cents(),
            st.jitter_rms_cents(),
            audio_opts.pitch_snap_cents,
        );
        // **The bleed.** A voice that is sounding while nothing in the
        // spectrum matches it is playing a note the source has stopped
        // playing, and the tail is how long it takes to reach exactly
        // zero. Reported every render because none of the metrics above
        // can see it -- a voice droning through a whole phrase scores
        // BETTER on lifetime, on jitter and on jumps alike, which is how
        // "it bleeds together" and "disembodied" got shipped.
        let (tail_mean, tail_p95) = st.tail_ms(streams.fps);
        info!(
            "  {:.1}% of sounding voice-frames are unmatched (a note that has ended); \
             time from a partial's end to zero: mean {:.0} ms, p95 {:.0} ms \
             (--voice-release {})",
            st.unmatched_fraction() * 100.0,
            tail_mean,
            tail_p95,
            audio_opts.voice_release_ms,
        );
        // The voice BUDGET: how much of the build goes on one note's
        // overtones rather than on separate notes.
        info!(
            "  {:.1}% of sounding voices sit on a harmonic of another; {:.1} distinct \
             fundamentals per frame",
            st.harmonic_fraction() * 100.0,
            st.mean_fundamentals(),
        );
        match build_voice_world(&streams, &audio_opts) {
            Ok(w) => w,
            Err(e) => fail!("{e}"),
        }
    } else {
        let track = match analyze(
            source.as_ref(),
            &audio_opts,
            &mut progress_cli::CliProgress::new(),
        ) {
            Ok(t) => t,
            Err(e) => fail!("{e}"),
        };
        let sounding: usize = track
            .volumes
            .iter()
            .flat_map(|b| b.iter())
            .filter(|v| **v > 0.0)
            .count();
        info!(
            "Analyzed {} frame(s) at {} fps across {} band(s), {} bank(s)",
            track.frame_count,
            track.fps,
            track.plan.len(),
            track.frame_count.div_ceil(audio_opts.bank_size.max(1)).max(1),
        );
        // The same "what did the flag actually do" report as voice mode's.
        // `--max-voices` is an upper bound that `--peak-gate` usually
        // binds first, so the flag value says nothing about the render.
        // Mean length of a run of one band being continuously non-zero:
        // THE BEEPING METRIC. A run of one or two frames is a 33-66 ms
        // blip, heard as a beep rather than as a note, and it is what
        // `--attack`/`--release` exist to lengthen.
        let mut runs = 0usize;
        for band in &track.volumes {
            let mut on = false;
            for &v in band {
                if v > 0.0 && !on {
                    runs += 1;
                }
                on = v > 0.0;
            }
        }
        info!(
            "  {:.2} bands sounding per frame (--max-voices {}, --peak-gate {}); mean run \
             {:.1} frames (--attack {} --release {})",
            sounding as f64 / track.frame_count as f64,
            audio_opts.max_voices,
            audio_opts.peak_gate,
            if runs > 0 {
                sounding as f64 / runs as f64
            } else {
                0.0
            },
            audio_opts.attack_ms,
            audio_opts.release_ms,
        );
        match build_speaker_world(&track, &audio_opts) {
            Ok(w) => w,
            Err(e) => fail!("{e}"),
        }
    };

    info!("Writing Save to {}", out_file);
    if let Err(e) = write_world(&world, &out_file) {
        fail!("{e}");
    }
    return info!("Done!");
}

/// The `--midi` (midi2brick) branch: read a Standard MIDI File and either list
/// its instruments (`--midi-list`) or build an event-based speaker world from
/// it. Lite CLI: one `--synth` tone for the whole file; per-track tones are a
/// GUI feature.
#[cfg(not(target_arch = "wasm32"))]
fn run_midi(matches: &clap::ArgMatches, heightmap_files: &[PathBuf], out_file: &str) {
    let input = &heightmap_files[0];
    if heightmap_files.len() > 1 {
        warn!("--midi reads a single MIDI file; ignoring the extra inputs");
    }
    let bytes = match std::fs::read(input) {
        Ok(b) => b,
        Err(e) => fail!("could not read {}: {e}", input.display()),
    };

    if matches.is_present("midilist") {
        match discover(&bytes) {
            Ok((instruments, summary)) => print_midi_info(&instruments, &summary),
            Err(e) => fail!("{e}"),
        }
        return;
    }

    let opts = match midi_options(matches) {
        Ok(o) => o,
        Err(e) => fail!("{e}"),
    };
    let score = match analyze_midi(&bytes, &opts) {
        Ok(s) => s,
        Err(e) => fail!("{e}"),
    };
    info!(
        "midi2brick: {} speaker(s), {:.1}s{}",
        score.voices.len(),
        score.duration_s,
        if opts.loop_playback { ", looping" } else { "" }
    );
    let world = match build_midi_event_world(&score, &opts) {
        Ok(w) => w,
        Err(e) => fail!("{e}"),
    };
    if let Err(e) = write_world(&world, out_file) {
        fail!("{e}");
    }
    info!("Done!");
}

/// Build [`MidiOptions`] from the CLI flags. The tone defaults to an automatic
/// per-instrument pick from each track's GM program ([`ToneAssignment::Auto`]);
/// `--synth` forces one wave for the whole file. Per-track manual tones are a
/// GUI feature. Spatialization/playback flags are shared with the audio path by
/// name.
#[cfg(not(target_arch = "wasm32"))]
fn midi_options(matches: &clap::ArgMatches) -> Result<MidiOptions, String> {
    let d = MidiOptions::default();
    // Default: pick a waveform per instrument from its GM program. `--synth`
    // forces one wave for the whole file, as before.
    let tones = match matches.value_of("audiosynth") {
        Some(s) => ToneAssignment::Uniform(SynthWave::parse(s)?),
        None => ToneAssignment::Auto,
    };
    Ok(MidiOptions {
        inner_radius: parse_arg(matches, "audioinner", "--inner-radius", "a number", d.inner_radius)?,
        max_distance: parse_arg(matches, "audiomaxdist", "--max-distance", "a number", d.max_distance)?,
        gain: parse_arg(matches, "audiogain", "--gain", "a number", d.gain)?,
        polyphony_cap: parse_arg(
            matches,
            "midipolyphony",
            "--polyphony-cap",
            "an integer",
            d.polyphony_cap,
        )?,
        loop_playback: !matches.is_present("noloop"),
        control_buttons: !matches.is_present("nocontrolbuttons"),
        speakers_in_chip: matches.is_present("audiospeakersinchip"),
        // Lite CLI path: uniform tone, and no per-instrument volume (that is a
        // GUI feature) -- an empty list plays every instrument at 1.0.
        instrument_volumes: Vec::new(),
        playback_rate: parse_arg(matches, "midirate", "--playback-rate", "a number", d.playback_rate)?,
        preview_seconds: d.preview_seconds,
        tones,
        build_percussion: !matches.is_present("nopercussion"),
        drum_kit: Vec::new(),
    })
}

/// Print the `--midi-list` report: the file summary and one line per discovered
/// instrument.
#[cfg(not(target_arch = "wasm32"))]
fn print_midi_info(instruments: &[heightmap::midi::Instrument], summary: &heightmap::midi::MidiSummary) {
    info!(
        "MIDI format {}, {} track(s), {:.1}s, {:.0} BPM, {} note(s){}",
        summary.format,
        summary.track_count,
        summary.duration_s,
        summary.initial_bpm,
        summary.total_notes,
        if summary.has_percussion { " (has percussion)" } else { "" }
    );
    info!("{} instrument(s):", instruments.len());
    for (i, inst) in instruments.iter().enumerate() {
        let dropped = if inst.dropped_notes > 0 {
            format!(", {} out of range", inst.dropped_notes)
        } else {
            String::new()
        };
        info!(
            "  [{i}] {} (ch {}): {} note(s), max polyphony {}{}",
            inst.label,
            inst.channel + 1,
            inst.note_count,
            inst.max_polyphony,
            dropped
        );
    }
}

/// The `--anim-mode` render branch: resolves the mode/encoding pair, the
/// subtitle track, and every animation flag; then either decodes a video
/// source through a decode backend or reads image file(s)/a directory/an
/// animated image into a `Clip`, adapts it (resize + resample), estimates and
/// logs the cost, builds the wired world, and writes it. Split out of
/// `main`'s dispatch verbatim -- every `fail!`/`warn!` here still exits or
/// logs under exactly the same condition it did inline. Takes `heightmap_files`
/// by value because this branch (uniquely) may replace it wholesale when the
/// input names a directory of frames.
#[cfg(not(target_arch = "wasm32"))]
fn run_anim(
    matches: &clap::ArgMatches,
    heightmap_files: Vec<PathBuf>,
    consent: DownloadConsent,
    backend_choice: Backend,
    out_file: &str,
) {
    // The BRZ/BRDB string-array encoding this renderer builds on tops
    // out at `MAX_FRAMES` frames; passing an unbounded sentinel here
    // would re-enable an unbounded resampling loop (a fat-fingered --fps
    // could OOM).
    // `--anim-mode` and `--anim-encoding` are parsed TOGETHER, by
    // `AnimMode::parse`, because they are not independent: text mode has
    // exactly one pixel encoding, so an explicit `--anim-encoding` under it
    // is a hard error rather than a silently-ignored flag. Its messages are
    // propagated verbatim so the CLI and the GUI can never disagree about
    // what a mode/encoding pair means. Defaults to `Brick(Hex)`, the
    // combination every render before these flags existed produced, so an
    // unchanged command line still produces an unchanged save.
    let mode = match AnimMode::parse(
        matches.value_of("animmode").unwrap_or("brick"),
        matches.value_of("animencoding"),
    ) {
        Ok(m) => m,
        Err(e) => fail(e),
    };

    if matches.is_present("colormap") {
        warn!("--anim-mode ignores --colormap");
    }

    // Median-cut palette size, text mode only. Through `parse_arg` rather
    // than `.expect(...)`: a mistyped `--colors many` must be a CLI error
    // naming the flag, not a Rust panic trace (see `parse_arg`'s doc).
    let colors = match parse_arg(&matches, "animcolors", "--colors", "an integer", 0usize) {
        Ok(v) => v,
        Err(e) => fail(e),
    };
    // Warned about and ignored rather than refused, exactly as `--colormap`
    // is above. Brick mode spends a fixed cost per pixel that no palette
    // can change, so the flag is meaningless there -- but it is named out
    // loud, because a user who passed it and got a byte-for-byte
    // equivalent save would reasonably conclude quantization was broken.
    if colors > 0 && mode != AnimMode::Text {
        warn!(
            "--colors applies to --anim-mode text only (it lengthens colour RUNS in the \
             encoded strings); brick mode's cost is fixed per pixel, so the flag is ignored \
             here"
        );
    }

    // Resolved HERE, before a single frame is decoded, so a typo'd
    // subtitle path or a bitmap track fails in a second rather than after
    // a long render. `heightmap_files[0]` is the pre-expansion input --
    // `--subtitle-track` extracts from a CONTAINER, which is the video
    // path itself, never one image out of a sequence.
    let subtitles = match load_subtitles(&matches, &heightmap_files[0], consent) {
        Ok(s) => s,
        Err(e) => fail(e),
    };
    let subtitle_scale = match parse_arg(
        &matches,
        "subtitlescale",
        "--subtitle-scale",
        "a number",
        DEFAULT_SUBTITLE_SCALE,
    ) {
        Ok(v) => v,
        Err(e) => fail(e),
    };
    // Modelled on `--subtitle-scale` exactly: same parser, same default
    // source, same place. Legality (whether this lift fits the picture
    // that's actually rendered) is a renderer concern -- `subtitle_extent`
    // and `text_bricks::build_text_world` reject it there, where the
    // geometry is known, rather than here.
    let subtitle_lift = match parse_arg(
        &matches,
        "subtitlelift",
        "--subtitle-lift",
        "a number",
        DEFAULT_SUBTITLE_LIFT,
    ) {
        Ok(v) => v,
        Err(e) => fail(e),
    };
    // **Said out loud rather than left to be discovered in game.** A
    // `TextDisplay` draws in the plane of the anchor face it is given, and
    // a brick-mode screen lies FLAT -- so the subtitle is placed on the
    // screen's upward face (`bricks::subtitle_extent`) rather than the
    // upright one text mode uses. That much is read off the save schema's
    // own `EBrickDirection`; which way the line RUNS within that plane is
    // a property of the component nothing here can know, and has not been
    // checked by eye.
    if subtitles.is_some() && mode != AnimMode::Text {
        warn!(
            "subtitles over --anim-mode brick lie FLAT in the screen's own plane (the \
             screen is on the ground, so they are read from above, not from the front), \
             just above its surface and centred on the image's last row. Which \
             horizontal axis the line runs along -- and so whether that row reads as \
             the BOTTOM of the picture -- is unverified in game; --anim-mode text, \
             whose screen is an upright wall, is the mode this was designed against"
        );
    }

    // The SAME `TextOptions` the static `--text` path builds, from the same
    // flags, through the same function -- two parsers over one set of flags
    // drift apart silently, because each looks right on its own. Brick mode
    // ignores the whole struct; text mode reads `char_repeat` (which drives
    // the band layout), the font geometry, and the glyph settings.
    let text_opts = match text_options(&matches) {
        Ok(o) => o,
        Err(e) => fail(e),
    };
    // Named out loud for the same reason as `--colors` above: the band
    // layout is a closed-form bound on the COLOUR encoder's row width (16
    // characters of colour tag plus `char_repeat` glyph characters per
    // pixel), and a monochrome glyph mode packs several pixels into one
    // character, so it does not obey that bound or that row geometry.
    // REFUSED, not warned about. This used to warn and then build the save
    // anyway, which shipped a render that is WRONG rather than merely
    // unsupported: `text_options` above has already replaced `line_height`,
    // `kerning`, `line_offset`, `pitch_x` and `pitch_y` with
    // `mono_geometry(Braille, ..)`, so the world is laid out on braille
    // geometry -- while `text::encode_bands`/`encode_row`, the anim text
    // encoder, never consults `opts.mode` at all and emits colour tags plus
    // `char_repeat` glyphs regardless. The cost readout describes the colour
    // render too. A save the tool itself knows to be wrong must not be
    // written.
    if mode == AnimMode::Text && (matches.is_present("braille") || matches.is_present("blocks"))
    {
        fail!(
            "--braille/--blocks are not supported under --anim-mode text: the band layout \
             assumes the colour encoder's row geometry and the animated text encoder emits \
             colour runs whatever the mode says, so the save would be laid out for one \
             encoder and filled by the other. Use --text for a still monochrome export"
        );
    }
    // Colour-array mode always converts sRGB -> linear itself, unlike hex
    // mode where `MakeColorHex` makes `--srgb-to-linear` a real question.
    if mode == AnimMode::Brick(AnimEncoding::ColorArray) && matches.is_present("srgb2lin") {
        warn!(
            "--srgb-to-linear is implied by --anim-encoding color-array (that encoding \
             stores linear colors, so it always converts) -- the flag changes nothing here"
        );
    }
    // Third and last inert case: text mode writes colour tags straight
    // from the frame and never consults it.
    if mode == AnimMode::Text && matches.is_present("srgb2lin") {
        warn!(
            "--srgb-to-linear applies to --anim-mode brick with the hex encoding; the text \
             encoder writes colour tags from the frame's own sRGB values and ignores it"
        );
    }

    // All four through `parse_arg`, exactly as the audio branch's flags and
    // the video branch's `--width` are: a mistyped `--fps abc` is a CLI
    // error naming the flag, never the `.expect("fps must be a number")`
    // panic trace (exit 101, "note: run with `RUST_BACKTRACE=1`") these were.
    // `--max-frames` in particular reached a clean error under
    // `--audio-mode` and a panic under `--anim-mode`, from the same typo.
    let fps = match parse_arg(&matches, "animfps", "--fps", "a number", 10.0f32) {
        Ok(v) => v,
        Err(e) => fail(e),
    };
    // Both clamped at 0 here rather than left raw: `FpsStream` clamps
    // them internally the same way, so this changes no render, but it
    // lets the pre-flight frame-count check below mirror the stream's
    // arithmetic exactly instead of approximating it. The GUI clamps
    // these at the same point for the same reason.
    let start = match parse_arg(&matches, "animstart", "--start", "a number", 0.0f32) {
        Ok(v) => v.max(0.0),
        Err(e) => fail(e),
    };
    let duration =
        match parse_opt_arg::<f32>(&matches, "animduration", "--duration", "a number") {
            Ok(v) => v.map(|d| d.max(0.0)),
            Err(e) => fail(e),
        };
    let max_frames = match parse_arg(
        &matches,
        "animmaxframes",
        "--max-frames",
        "an integer",
        MAX_FRAMES,
    ) {
        Ok(v) => v.min(MAX_FRAMES),
        Err(e) => fail(e),
    };

    let fit = match matches
        .value_of("animfit")
        .map(|s| s.to_lowercase())
        .as_deref()
    {
        None | Some("contain") => FitMode::Contain,
        Some("exact") => FitMode::Exact,
        Some("cover") => FitMode::Cover,
        Some(other) => {
            fail!("unknown fit mode '{other}' (exact, contain, cover)");
        }
    };
    let filter = match matches
        .value_of("animfilter")
        .map(|s| s.to_lowercase())
        .as_deref()
    {
        None | Some("lanczos") => Filter::Lanczos,
        Some("nearest") => Filter::Nearest,
        Some(other) => {
            fail!("unknown filter '{other}' (lanczos, nearest)");
        }
    };

    // A frame sequence outruns the OS argument limit long before it
    // outruns anything else this renderer cares about. At ~11 characters
    // per relative filename, Windows' 32767-character command line runs
    // dry around 3000 frames -- but a 90-minute clip at 12fps is 65391 of
    // them: inside one bank's 65535 entries, far inside the 1048560-frame
    // overall cap, and ~20x past argv. Naming the directory sidesteps it
    // entirely, so the real limit stays the one that means something.
    //
    // Scoped to anim mode on purpose: a directory of images is a frame
    // sequence, which has no meaning for the heightmap or text paths.
    let heightmap_files = if heightmap_files.len() == 1 && heightmap_files[0].is_dir() {
        let dir = &heightmap_files[0];
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => fail!("Error reading directory {}: {e}", dir.display()),
        };
        let mut found = Vec::new();
        for entry in entries {
            let path = match entry {
                Ok(e) => e.path(),
                Err(e) => fail!("Error reading directory {}: {e}", dir.display()),
            };
            // Filter on extension rather than by attempting a decode: a
            // stray .txt or a nested directory should be skipped in
            // silence, while a genuinely corrupt .png must still be the
            // hard error it becomes below.
            let is_image = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_ascii_lowercase())
                .is_some_and(|e| {
                    matches!(e.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "tga" | "webp")
                });
            if is_image && path.is_file() {
                found.push(path);
            }
        }
        if found.is_empty() {
            fail!("No image files found in directory {}", dir.display());
        }
        // `decode_sequence` re-sorts by natural key (so f_2 precedes
        // f_10); this sort only settles the order for the single-image
        // case below, which never reaches it.
        found.sort();
        info!("Found {} image(s) in {}", found.len(), dir.display());
        found
    } else {
        heightmap_files
    };

    // A video file goes to a decode backend, which pushes fps and
    // scaling into its own filters rather than materialising a Clip --
    // the whole point of this path. The directory and animated-image
    // branches below are unchanged.
    if heightmap_files.len() == 1 && is_video_path(&heightmap_files[0]) {
        // ffmpeg's availability is consulted only where it is actually
        // needed, and `open_video_ensuring` owns that decision so it can
        // be unit tested (see its doc, and this task's report):
        //
        // - a GIF/PNG/frame-sequence render never reaches this branch at
        //   all, so it is never prompted about ffmpeg;
        // - `--backend builtin` never consults it -- that backend either
        //   decodes in builtin or refuses by name;
        // - `--backend ffmpeg` consults it eagerly, because the user
        //   named the backend and a missing binary should fail fast;
        // - `--backend auto` (the DEFAULT) TRIES the builtin backend
        //   first and only consults ffmpeg once that has failed with
        //   something ffmpeg could actually help with. An earlier version
        //   checked eagerly here, which made a machine without ffmpeg
        //   refuse every video by default -- including CABAC H.264 files
        //   the builtin backend handles perfectly well on its own.
        info!("Opening video {}", heightmap_files[0].display());
        // Hand the decode backend the target size when BOTH axes were
        // given explicitly, so `FfmpegSource` scales inside its own
        // filtergraph instead of piping native-resolution frames for
        // `ResizeStream` to shrink. On a 1080p source that is ~8.3 MB per
        // frame over the pipe versus ~9 KB at 64x36 -- measured, a
        // 24-minute episode fell from 5m21s to seconds.
        //
        // Only when both are present: with one axis missing the other is
        // derived from `native` below, which is not known until the source
        // is already open. That case keeps the old behaviour.
        //
        // `AdaptedSource` still receives the same `size` either way and
        // remains the single place that defines the output dimensions --
        // `ResizeStream` simply passes a frame through untouched when it
        // already matches (see `ResizeStream::fit_frame`). So the
        // guarantee the old `None` protected still holds: `--width` and
        // `--height` cannot work on one backend and silently do nothing
        // on the other.
        //
        // fps stays `None` deliberately. Pushing THAT into ffmpeg changes
        // which frames are selected, not merely how they are scaled, and
        // needs a cross-backend selection test first.
        let preset_size = match (matches.value_of("animwidth"), matches.value_of("animheight")) {
            (Some(w), Some(h)) => match (w.parse::<u32>(), h.parse::<u32>()) {
                (Ok(w), Ok(h)) => Some((w, h)),
                // Leave the error itself to the parse below, which already
                // reports which flag was bad.
                _ => None,
            },
            _ => None,
        };
        let raw = match backend::open_video_ensuring(
            &heightmap_files[0],
            backend_choice,
            preset_size,
            fit,
            filter,
            None,
            &mut || ensure_ffmpeg(consent),
        ) {
            Ok(s) => s,
            Err(e) => fail!("{e}"),
        };

        // `open_video` does not apply target size/rate to the builtin
        // backend -- `BuiltinVideoSource::open_path` takes no such
        // parameters at all -- so passing `--width`/`--height`/`--fps`
        // straight through to it here would make those flags work on the
        // ffmpeg backend and silently do nothing on the builtin one.
        // `AdaptedSource` is layered over whatever `open_video` returned
        // instead, exactly the way the image-sequence path below layers
        // it over a `Clip`, so every scaling/rate/window flag behaves
        // identically no matter which backend produced the raw frames.
        // `open_video` above is deliberately called with `None`/`None`
        // for size/fps so the raw source stays native and untouched on
        // EITHER backend -- there is exactly one place, this
        // `AdaptedSource`, that ever resizes or resamples.
        let native = raw.info();
        let size = match target_size(&matches, native.width, native.height) {
            Ok(s) => s,
            Err(e) => fail(e),
        };

        let adapted = AdaptedSource {
            inner: raw.as_ref(),
            size,
            fit,
            filter,
            target_fps: fps,
            start_s: start,
            duration_s: duration,
            max_frames,
        };

        // Same builder the image/sequence path below uses; see `anim_options`.
        let anim_opts = match anim_options(
            &matches,
            colors,
            &text_opts,
            subtitles.clone(),
            subtitle_scale,
            subtitle_lift,
            start,
        ) {
            Ok(o) => o,
            Err(e) => fail(e),
        };

        // `AdaptedSource::info` folds the resample/window math into the
        // hint whenever the raw source can say its own frame count up
        // front (see its doc) -- a video probed from a real container
        // almost always can. When it genuinely can't, the hint is `None`:
        // there is no pre-flight number to refuse on, but `FpsStream`
        // itself still enforces `max_frames` mid-render (an `Err`, not a
        // silent truncation), so nothing is left unguarded -- only the
        // refusal-before-printing-a-cost-line optimization is unavailable.
        let info = adapted.info();
        // The approximate count (`duration * fps`, folded through the same
        // resample/window math) for when the exact one is unavailable --
        // which is every Matroska file, since the container stores no
        // frame count. Used for the cost line and the progress bar, and
        // deliberately NOT for the refusal below: an estimate that came
        // out high would refuse a render the stream would have completed.
        let est = adapted.frame_count_estimate();
        if let Some(n) = info.frame_count_hint {
            if n > max_frames {
                fail!("{}", max_frames_error(max_frames));
            }
        } else if let Some(n) = est {
            warn!(
                "source did not report an exact frame count ahead of decode; the cost \
                 below and the progress bar are sized from an estimate of ~{n} frames \
                 (duration x fps) and may be off by a frame or two. max_frames is not \
                 pre-checked against an estimate, but is still enforced during the render"
            );
        } else {
            warn!(
                "source reported neither a frame count nor a duration ahead of decode; the \
                 cost estimate below counts 0 frames rather than guess and the progress bar \
                 has no total, but max_frames is still enforced during the render"
            );
        }

        // `estimate` errors only for a text geometry `plan_bands` cannot
        // lay out -- the same error `mode.build` below would fail with, so
        // reporting it here just fails a few milliseconds earlier instead
        // of printing a plausible cost for a render that cannot run.
        let cost = match mode.estimate(
            info.width,
            info.height,
            info.frame_count_hint.or(est).unwrap_or(0),
            &anim_opts,
        ) {
            Ok(c) => c,
            Err(e) => fail!("{e}"),
        };
        log_cost(mode, &cost, info.width, info.height, text_opts.char_repeat);

        let world = match mode.build(
            &adapted,
            &anim_opts,
            &mut progress_cli::CliProgress::new(),
        ) {
            Ok(w) => w,
            Err(e) => fail!("{e}"),
        };

        info!("Writing Save to {}", out_file);
        if let Err(e) = write_world(&world, &out_file) {
            fail!("{e}");
        }
        return info!("Done!");
    }

    info!("Reading image file(s)");
    let source = if heightmap_files.len() == 1 {
        let input = &heightmap_files[0];
        let bytes = match std::fs::read(input) {
            Ok(b) => b,
            Err(e) => fail!("Error reading file {}: {e}", input.display()),
        };
        if is_animated(&bytes) {
            Source::Animated(bytes)
        } else {
            match image::load_from_memory(&bytes) {
                Ok(i) => Source::Still(i.to_rgba8()),
                Err(e) => fail!("Error reading image: {e:?}"),
            }
        }
    } else {
        let mut named = Vec::with_capacity(heightmap_files.len());
        for input in &heightmap_files {
            let img = match image::open(input) {
                Ok(i) => i.to_rgba8(),
                Err(e) => fail!("Error reading image {}: {e:?}", input.display()),
            };
            named.push((input.display().to_string(), img));
        }
        Source::Sequence(named)
    };

    let clip = match decode(source, fps) {
        Ok(c) => c,
        Err(e) => fail!("{e}"),
    };

    // The SAME parser the video branch above uses -- see `target_size`.
    let size = match target_size(&matches, clip.width, clip.height) {
        Ok(s) => s,
        Err(e) => fail(e),
    };

    // Resize (if requested) then resample, streamed rather than
    // materialized twice: `AdaptedSource` layers `ResizeStream` under
    // `FpsStream` so frames are scaled before selection, never the other
    // way around.
    let adapted = AdaptedSource {
        inner: &clip,
        size,
        fit,
        filter,
        target_fps: fps,
        start_s: start,
        duration_s: duration,
        max_frames,
    };

    // Built before the estimate so the readout costs the options the
    // render will actually consume -- `bank_size` in particular decides
    // how many arrays the frames spill across. Same builder the video
    // path above uses.
    let anim_opts = match anim_options(
        &matches,
        colors,
        &text_opts,
        subtitles.clone(),
        subtitle_scale,
        subtitle_lift,
        start,
    ) {
        Ok(o) => o,
        Err(e) => fail(e),
    };

    // The resampled frame count isn't knowable from the *stream* (see
    // `AdaptedSource::info`, whose hint is `None` once the rate changes),
    // but it is computable from the source's length plus the scalars --
    // counts and timings only, no frame data. `estimated_frame_count` is
    // pinned to `FpsStream`'s real output by a test sweep, so this is the
    // number the render will actually produce, not an approximation.
    let est_frames =
        estimated_frame_count(clip.frames.len(), clip.fps, fps, start, duration, max_frames);

    // Deliberately checked BEFORE the cost line, and against the
    // unclamped count. Clamping to `max_frames` and printing anyway
    // would report the cap as though it were the real answer and then
    // have the render immediately refuse the same request -- a cost
    // line the CLI itself does not believe. `resample_fps` used to
    // error before this block was ever reached; this restores that
    // ordering now that the refusal happens mid-stream instead. The
    // message comes from `scale::max_frames_error`, the same one
    // `FpsStream` would emit, so the two can never drift.
    if est_frames > max_frames {
        fail!("{}", max_frames_error(max_frames));
    }

    let info = adapted.info();

    // See the sibling call above: an `Err` here is a text geometry
    // `plan_bands` cannot lay out, i.e. exactly what `mode.build` would
    // refuse a moment later.
    let cost = match mode.estimate(info.width, info.height, est_frames, &anim_opts) {
        Ok(c) => c,
        Err(e) => fail!("{e}"),
    };
    log_cost(mode, &cost, info.width, info.height, text_opts.char_repeat);

    let world =
        match mode.build(&adapted, &anim_opts, &mut progress_cli::CliProgress::new()) {
            Ok(w) => w,
            Err(e) => fail!("{e}"),
        };

    info!("Writing Save to {}", out_file);
    if let Err(e) = write_world(&world, &out_file) {
        fail!("{e}");
    }
    return info!("Done!");
}

/// The `--text` render branch: builds `TextOptions`, reads the first input
/// image, encodes it into `TextDisplay` tiles, and writes the result. Split
/// out of `main`'s dispatch verbatim -- every `fail!`/`warn!` here still
/// exits or logs under exactly the same condition it did inline.
#[cfg(not(target_arch = "wasm32"))]
fn run_text(matches: &clap::ArgMatches, heightmap_files: &[PathBuf], out_file: &str) {
    if heightmap_files.len() > 1 {
        warn!(
            "--text uses only the first input image; ignoring {} extra input(s)",
            heightmap_files.len() - 1
        );
    }
    if matches.is_present("colormap") {
        warn!("--text ignores --colormap");
    }

    // The same builder `--anim-mode text` uses -- see `text_options`.
    let text_opts = match text_options(&matches) {
        Ok(o) => o,
        Err(e) => fail!("{e}"),
    };

    let input = &heightmap_files[0];
    info!("Reading image file {}", input.display());
    let img = match image::open(input) {
        Ok(i) => i.to_rgba8(),
        Err(e) => fail!("Error reading image: {e:?}"),
    };
    let tiles = match encode_tiles(&img, &text_opts) {
        Ok(t) => t,
        Err(e) => fail!("{e}"),
    };
    info!(
        "Encoded {} tile(s), {} text band(s)",
        tiles.len(),
        tiles.iter().map(|t| t.bands.len()).sum::<usize>()
    );

    let (img_w, img_h) = img.dimensions();
    let mut world = World::new();
    add_text_tiles(&mut world, tiles, &text_opts);
    world.meta.bundle.description = "Text image generated from image file".to_string();
    // prefab pivots/bounds cover the full rendered image so ground
    // placement holds the anchor cubes up in the air
    make_text_prefab(&mut world, img_w, img_h, &text_opts);

    info!("Writing Save to {}", out_file);
    if let Err(e) = write_world(&world, &out_file) {
        fail!("{e}");
    }
    return info!("Done!");
}

/// The default (no `--audio-mode`/`--anim-mode`/`--text`) render branch:
/// parses `--size`/`--vertical` into `GenOptions`, reads the colormap and
/// heightmap image(s), generates bricks, and writes the result. Split out of
/// `main`'s fallthrough tail verbatim -- every `fail!` here still exits under
/// exactly the same condition it did inline.
#[cfg(not(target_arch = "wasm32"))]
fn run_heightmap(
    matches: &clap::ArgMatches,
    heightmap_files: &[PathBuf],
    colormap_file: &PathBuf,
    out_file: &str,
) {
    // Both through `parse_arg` rather than `.expect(...)`, like every other
    // numeric flag in this file, and both bounded at 1: a brick 0 studs wide or
    // 0 units tall is not a smaller brick, and `--size 0` used to report
    // "Reduced 512 to 496" and write a save. The GUI's sliders start at 1, so
    // only the CLI could ever express it.
    let size = match parse_arg(&matches, "size", "--size", "an integer", 1u16) {
        Ok(0) => fail!("--size must be at least 1 (it is the width of one pixel in studs)"),
        Ok(v) => v,
        Err(e) => fail(e),
    };
    let scale = match parse_arg(&matches, "vertical", "--vertical", "an integer", 1u32) {
        Ok(0) => fail!(
            "--vertical must be at least 1 (it is the height of one shade of grey in units)"
        ),
        Ok(v) => v,
        Err(e) => fail(e),
    };

    // The three surface renderers. The code refuses a combination and does
    // not select one of them. They are different methods with different
    // bricks. A user who gives several must select one of them.
    let surface = match (
        matches.is_present("terrain"),
        matches.is_present("rampify"),
        matches.is_present("wedge"),
    ) {
        (true, false, false) => SurfaceMode::Terrain,
        (false, true, false) => SurfaceMode::Rampify,
        (false, false, true) => SurfaceMode::Wedge,
        (false, false, false) => SurfaceMode::Blocks,
        _ => fail!(
            "--terrain, --rampify and --wedge are three different surface techniques and \
             cannot be combined: --terrain fits micro wedges to a shared vertex grid, \
             --rampify fits full-size ramps to the height columns, --wedge builds terraces \
             with 45-degree chamfered outlines. Pass exactly one"
        ),
    };
    // Each option below controls a box with a FLAT TOP, which is the one
    // thing that a sloped renderer does not make. The code names each option,
    // in the same way as the audio part names the options that it cannot use.
    // A render that ignored `--greedy` without a message would appear
    // correct.
    if surface != SurfaceMode::Blocks {
        let mode = match surface {
            SurfaceMode::Terrain => "--terrain",
            SurfaceMode::Rampify => "--rampify",
            _ => "--wedge",
        };
        for (flag, name) in [
            ("--tile", "tile"),
            ("--smooth", "smooth"),
            ("--stud", "stud"),
            ("--greedy", "greedy"),
            ("--snap", "snap"),
        ] {
            if matches.is_present(name) {
                warn!(
                    "{mode} ignores {flag}: it picks its own bricks from a shape grammar rather \
                     than stacking one asset, so there is no brick style or optimizer to choose"
                );
            }
        }
        // `--micro` also changes what `--size` COUNTS, so it gets its own
        // message. To obey it would make the map 5 times smaller.
        if matches.is_present("micro") {
            warn!(
                "{mode} ignores --micro; --size stays a count of STUDS per pixel. --terrain \
                 already builds from micro bricks, and --rampify and --wedge need full-size \
                 assets"
            );
        }
        if matches.is_present("img") {
            warn!("{mode} ignores --img: a flat image has no terrain to slope");
        }
    }
    let blocks = surface == SurfaceMode::Blocks;

    // `--size` counts STUDS, which are 5 units of half extent each. With
    // `--micro` it counts micro units. Use `checked_mul`: the half extent is a
    // u16, thus `--size 20000` overflowed and rendered a map at an incorrect
    // size.
    let micro_size = matches.is_present("micro") && blocks;
    let units_per_stud = if micro_size { 1 } else { 5 };
    let Some(half_extent) = size.checked_mul(units_per_stud) else {
        fail!(
            "--size {size} is too large. One pixel would be more than {} units across, which \
             is more than one brick can be. The maximum brick size is {} units",
            u16::MAX,
            MAX_BRICK_HALF_EXTENT * 2
        );
    };

    // output options
    let options = GenOptions {
        size: half_extent,
        scale,
        cull: matches.is_present("cull"),
        asset: if matches.is_present("micro") {
            PB_DEFAULT_MICRO_BRICK
        } else if matches.is_present("tile") {
            PB_DEFAULT_TILE
        } else if matches.is_present("smooth") {
            PB_DEFAULT_SMOOTH_TILE
        } else if matches.is_present("stud") {
            PB_DEFAULT_STUDDED
        } else {
            PB_DEFAULT_BRICK
        },
        micro: micro_size,
        stud: matches.is_present("stud") && blocks,
        snap: matches.is_present("snap"),
        img: matches.is_present("img") && blocks,
        glow: matches.is_present("glow"),
        hdmap: matches.is_present("hdmap"),
        nocollide: matches.is_present("nocollide"),
        quadtree: true,
        greedy: matches.is_present("greedy"),
        surface,
    };

    info!("Reading image files");

    // colormap file parsing
    let colormap = match file_ext(&colormap_file)
        .map(|s| s.to_lowercase())
        .as_deref()
    {
        Some("png") | Some("jpg") | Some("jpeg") => match ColormapPNG::new(&colormap_file) {
            Ok(map) => map,
            Err(err) => {
                fail!("Error reading colormap: {:?}", err);
            }
        },
        Some(ext) => {
            fail!("Unsupported colormap format '{}'", ext);
        }
        None => {
            fail!("Missing colormap format for '{}'", colormap_file.display());
        }
    };

    // heightmap file parsing
    let heightmap: Box<dyn Heightmap> = if heightmap_files.iter().all(|f| {
        matches!(
            file_ext(f).map(|s| s.to_lowercase()).as_deref(),
            Some("png") | Some("jpg") | Some("jpeg")
        )
    }) {
        if options.img {
            Box::new(HeightmapFlat::new(colormap.size()).unwrap())
        } else {
            match HeightmapPNG::new(heightmap_files.iter().collect(), options.hdmap) {
                Ok(map) => Box::new(map),
                Err(error) => {
                    fail!("Error reading heightmap: {:?}", error);
                }
            }
        }
    } else {
        fail!("Unsupported heightmap format");
    };

    // The size of the render, before it runs. The GUI shows the same values
    // below its scale sliders.
    let plan = footprint(
        heightmap.size(),
        options.size,
        if options.img { 0 } else { options.scale },
        255,
    );
    info!("Build size: {}", plan.size_text());
    info!(
        "  {} x {} units, {} at 1 unit = 1 inch",
        commas(plan.units.0),
        commas(plan.units.1),
        plan.real_text()
    );
    if !options.img {
        info!("  {}", plan.height_text());
    }
    if plan.over_brick_limit() {
        warn!("--size {size}: {}", plan.brick_limit_text());
    }

    // Not `.expect(...)`: `gen_opt_heightmap` returns a `String` describing a
    // real user-facing condition (an image it cannot use), and a panic trace
    // reads as a crash rather than as the refusal it is.
    let bricks = match gen_opt_heightmap(&*heightmap, &colormap, options, |_| true) {
        Ok(b) => b,
        Err(e) => fail!("{e}"),
    };

    // Do the check BEFORE the write. The game cannot load a save above the
    // chunk limit, and a failure in the game wastes the full render.
    match check_chunk_limit(&bricks) {
        Ok(chunks) => info!(
            "Save spans {} of the {} chunks the game will load",
            commas(chunks as u64),
            commas(MAX_SAVE_CHUNKS as u64)
        ),
        Err(e) => fail!("{e}"),
    }

    info!("Writing Save to {}", out_file);
    let mut data = bricks_to_save(bricks);

    // Each save gets a preview, world or prefab. The game shows a grid of
    // pictures, and a generated save has no screenshot of itself. The source
    // map gives a view from above. Use the COLORMAP, which shows the colours
    // of the build. Without `-c`, `colormap_file` is already the heightmap.
    //
    // An encode failure only writes a warning, because it must not stop the
    // render.
    match save_screenshot(colormap.image()) {
        Ok(screenshot) => {
            data.meta.screenshot = Some(screenshot);
            info!("Save preview: {}", colormap_file.display());
        }
        Err(e) => warn!("no save preview embedded: {e}"),
    }
    // The game names its own bundles, and the browser tile shows this name.
    // Use the output file name: `-o "Big Island.brz"` gives "Big Island".
    if data.meta.bundle.name.is_empty() {
        if let Some(stem) = std::path::Path::new(&out_file).file_stem() {
            data.meta.bundle.name = stem.to_string_lossy().to_string();
        }
    }

    // A prefab bundle and a world bundle differ in their metadata only. The
    // difference is `level_type` and the block of pivots and bounds that
    // `make_prefab` calculates from the brick positions. The code thus changes
    // the completed world and does not send this option to each generator.
    if matches.is_present("prefab") {
        data.make_prefab();
    }
    if let Err(e) = write_world(&data, &out_file) {
        fail!("{e}");
    }

    info!("Done!");
}

/// Print the pre-render cost readout for whichever mode was chosen. The
/// three modes report different fields -- hex mode's chunk/character counts
/// don't apply to colour-array, which tiles nothing and writes no strings.
/// `width`/`height`/`char_repeat` are only used in text mode, for the
/// per-band character bound computed in the `AnimMode::Text` arm.
#[cfg(not(target_arch = "wasm32"))]
fn log_cost(mode: AnimMode, cost: &cost::Cost, width: u32, height: u32, char_repeat: usize) {
    info!(
        "Estimated cost ({}): {} pixel(s), {} gate(s), {} wire(s), {} brick(s), {} bank(s), {} frame(s)",
        match mode {
            AnimMode::Brick(AnimEncoding::Hex) => "hex",
            AnimMode::Brick(AnimEncoding::ColorArray) => "color-array",
            AnimMode::Text => "text",
        },
        cost.pixels,
        cost.gates,
        cost.wires,
        cost.bricks,
        cost.banks,
        cost.frames
    );
    match mode {
        AnimMode::Brick(AnimEncoding::Hex) => {
            info!("  {} chunk(s), {} character(s)", cost.chunks, cost.chars)
        }
        // `Cost::chars` is 0 in text mode by design (see `cost::estimate_text`):
        // real length is content-dependent, so this prints the closed-form
        // worst-case bound the band layout was derived from instead.
        AnimMode::Text => {
            let repeat = char_repeat.max(1);
            match text_layout::plan_bands(width as usize, height as usize, repeat) {
                Ok(plan) if !plan.is_empty() => {
                    let row = text_layout::worst_case_row_chars(width as usize, repeat);
                    let rows = plan[0].rows;
                    let per_band = rows * row + rows.saturating_sub(1);
                    info!(
                        "  {} band(s) of {rows} row(s); UPPER BOUND {per_band} character(s) per \
                         band per frame ({row} per {width}-pixel row: 16 for a colour tag + \
                         {repeat} glyph char(s) per pixel, worst case every pixel starting its \
                         own run). NOT an estimate -- real length is content-dependent, which \
                         is why no character total is reported above; --colors is what shortens \
                         it",
                        plan.len(),
                    );
                }
                // An unsupportable layout or a zero-height clip: not reported
                // here, the render fails below with `plan_bands`' own message.
                _ => {}
            }
        }
        AnimMode::Brick(AnimEncoding::ColorArray) => {
            // 0 here means the source couldn't report its length ahead of
            // decode -- honest rather than guessed.
            let elements = color_bricks::array_elements(cost.pixels, cost.frames);
            let bytes = color_bricks::accumulator_bytes(cost.pixels, cost.frames);
            info!(
                "  {elements} color element(s), ~{:.1} MiB held while packing (16 bytes per \
                 pixel per frame, vs 6 in hex mode)",
                bytes as f64 / (1024.0 * 1024.0)
            );
        }
    }
}

/// Report a fatal CLI error and exit non-zero. The only way this file reports
/// a failure; [`fail!`] is this with a `format!` in front of it.
#[cfg(not(target_arch = "wasm32"))]
fn fail(msg: impl std::fmt::Display) -> ! {
    error!("{msg}");
    std::process::exit(1);
}

/// Resolve `--subtitles` / `--subtitle-track` into a parsed track, or `None`
/// when neither was passed.
///
/// The two flags are already known to be mutually exclusive by the time this
/// runs (`main` refuses the pair before anything is read), so this only has to
/// pick whichever is present.
///
/// `input` is the container `--subtitle-track` extracts from, and is ignored
/// entirely by `--subtitles`.
///
/// **An empty track is warned about, never swallowed.** A subtitle track with
/// no cues renders as a perfectly good video with no dialogue -- which is
/// indistinguishable from a correct render of a scene where nobody speaks, so
/// the only place the mistake can surface is here.
#[cfg(not(target_arch = "wasm32"))]
fn load_subtitles(
    matches: &clap::ArgMatches,
    input: &std::path::Path,
    consent: DownloadConsent,
) -> Result<Option<std::sync::Arc<Subtitles>>, String> {
    let track = if let Some(path) = matches.value_of("subtitles") {
        let path = PathBuf::from(path);
        // `read` + `from_utf8_lossy`, not `read_to_string`: subtitle files in
        // the wild are frequently Latin-1 or CP1251, and refusing the whole
        // render over one mangled accent would be a worse trade than
        // rendering that accent as a replacement character. The timings --
        // the part that has to be right -- are ASCII in both formats.
        let bytes = std::fs::read(&path)
            .map_err(|e| format!("could not read the subtitle file {}: {e}", path.display()))?;
        let text = String::from_utf8_lossy(&bytes);
        // Parser chosen by extension, falling back to sniffing the content --
        // see `subs::parse_auto`, which the GUI's picker shares so the two can
        // never read one file as two different formats.
        let track = subs::parse_auto(&text, path.extension().and_then(|e| e.to_str()))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        info!("Read {} subtitle cue(s) from {}", track.len(), path.display());
        track
    } else if let Some(s) = matches.value_of("subtitletrack") {
        let n = s
            .parse::<usize>()
            .map_err(|e| format!("--subtitle-track must be a non-negative integer: {e} (got '{s}')"))?;
        // Extraction spawns `ffprobe` and `ffmpeg`, so the same consent gate
        // every other ffmpeg path in this file goes through applies -- and
        // eagerly, because the user named the flag that needs it.
        ensure_ffmpeg(consent)?;
        // `n` indexes SUBTITLE streams (ffmpeg's `0:s:<n>`), not the
        // container's absolute stream list -- same convention as
        // `--audio-track`. A bitmap (PGS/DVD/DVB) track is refused here by
        // name, never extracted into an empty track.
        let track = subs::extract::extract(input, n)?;
        info!(
            "Extracted {} subtitle cue(s) from track {n} of {}",
            track.len(),
            input.display()
        );
        track
    } else {
        return Ok(None);
    };

    if track.is_empty() {
        warn!(
            "the subtitle track has no cues at all -- the render will show no subtitles, \
             which looks exactly like a correct render of a scene with no dialogue"
        );
    }
    Ok(Some(std::sync::Arc::new(track)))
}

// The next few option builders (`target_size`, `anim_options`,
// `text_options`, `audio_options`) each cover flags that more than one
// `main` branch needs, in one function instead of one copy per branch: two
// parsers over the same flags drift apart silently, and the only symptom is
// that identical input builds two different results.

/// The target size `--width`/`--height` name, or `None` when neither was
/// passed and the source's own dimensions stand. `0` is refused: a
/// zero-pixel screen is not a smaller render, it renders nothing.
#[cfg(not(target_arch = "wasm32"))]
fn target_size(
    matches: &clap::ArgMatches,
    native_w: u32,
    native_h: u32,
) -> Result<Option<(u32, u32)>, String> {
    if !matches.is_present("animwidth") && !matches.is_present("animheight") {
        return Ok(None);
    }
    let target_w = parse_arg(matches, "animwidth", "--width", "an integer", native_w)?;
    let target_h = parse_arg(matches, "animheight", "--height", "an integer", native_h)?;
    for (flag, value) in [("--width", target_w), ("--height", target_h)] {
        if value == 0 {
            return Err(format!(
                "{flag} must be at least 1: a {target_w}x{target_h} screen has no pixels to \
                 drive, so the save would encode perfectly and contain nothing"
            ));
        }
    }
    Ok(Some((target_w, target_h)))
}

/// The `AnimOptions` both `--anim-mode` branches share: display flags
/// (`--brick-style`, `--pixel-extent`, `--glow`, `--srgb-to-linear`), clock
/// flags (`--external-clock`, `--no-loop`), and what the caller already
/// parsed for its own use.
#[cfg(not(target_arch = "wasm32"))]
fn anim_options(
    matches: &clap::ArgMatches,
    colors: usize,
    text_opts: &TextOptions,
    subtitles: Option<std::sync::Arc<Subtitles>>,
    subtitle_scale: f32,
    subtitle_lift: f32,
    start: f32,
) -> Result<AnimOptions, String> {
    let brick_style = match matches
        .value_of("animbrickstyle")
        .map(|s| s.to_lowercase())
        .as_deref()
    {
        None | Some("micro") => DisplayBrickStyle::Micro,
        Some("tile") => DisplayBrickStyle::SmoothTile,
        Some(other) => return Err(format!("unknown brick style '{other}' (micro, tile)")),
    };
    // No upper bound beyond `u16`'s own range: any half-extent tiles flush at
    // `2 * pixel_extent` (see `AnimOptions::pixel_extent`), so there is no
    // value that can trip the overlap check.
    let pixel_extent = parse_arg(
        matches,
        "animpixelextent",
        "--pixel-extent",
        "an integer",
        1u16,
    )?
    .max(1);
    Ok(AnimOptions {
        external_clock: matches.is_present("externalclock"),
        // NEGATED: the flag is `--no-loop`, the field is "do loop". Looping is
        // the default so that every command line that predates the flag renders
        // exactly what it did before.
        loop_playback: !matches.is_present("noloop"),
        // NEGATED, and default-on: `--no-control-buttons` is an off-switch, so
        // an unchanged command line ships the buttons.
        control_buttons: !matches.is_present("nocontrolbuttons"),
        brick_style,
        pixel_extent,
        glow: matches.is_present("glow"),
        srgb_to_linear: matches.is_present("srgb2lin"),
        colors,
        // Both fields, one flag: the renderer culls against `alpha_threshold`
        // while the palette builds against `text.alpha_threshold`, the
        // encoder's own visibility rule. Setting only one would desync which
        // pixels each treats as visible.
        alpha_threshold: text_opts.alpha_threshold,
        text: text_opts.clone(),
        subtitles,
        subtitle_scale,
        subtitle_lift,
        // THE SAME `start` the caller's `AdaptedSource` was given. A subtitle
        // file is in SOURCE time, so this is what stops `--start 120` putting
        // the whole track two minutes early -- see
        // `AnimOptions::source_start_s`.
        source_start_s: start as f64,
        ..AnimOptions::default()
    })
}

/// The `TextOptions` every text-rendering path shares: `--font`,
/// `--fill-char`, `--empty-char`, `--char-repeat`, `--alpha-threshold`,
/// `--line-height-world`, `--braille`/`--blocks`, `--luma-threshold`,
/// `--invert`, `--material`. Numeric values go through [`parse_arg`], so a
/// mistyped `--char-repeat two` is a CLI error naming the flag.
#[cfg(not(target_arch = "wasm32"))]
fn text_options(matches: &clap::ArgMatches) -> Result<TextOptions, String> {
    let preset = match matches.value_of("font").map(|s| s.to_lowercase()).as_deref() {
        None | Some("monaspace") | Some("argon") => FontPreset::MonaspaceArgon,
        Some("iosevka") => FontPreset::IosevkaTerm,
        Some("orbitron") => FontPreset::Orbitron,
        Some(other) => {
            return Err(format!(
                "unknown font preset '{other}' (monaspace, iosevka, orbitron)"
            ));
        }
    };
    let material = match matches.value_of("material").map(|s| s.to_lowercase()).as_deref() {
        None | Some("unlit") => TextMaterial::Unlit,
        Some("graffiti") => TextMaterial::Graffiti,
        Some("plastic") => TextMaterial::Plastic,
        Some("metallic") | Some("metal") => TextMaterial::Metallic,
        Some("glow") => TextMaterial::Glow,
        Some("translucent") => TextMaterial::TranslucentPlastic,
        Some("glass") => TextMaterial::Glass,
        Some(other) => {
            return Err(format!(
                "unknown material '{other}' (unlit, graffiti, plastic, metallic, glow, \
                 translucent, glass)"
            ));
        }
    };
    let pixel_size = parse_arg(
        matches,
        "lineheight",
        "--line-height-world",
        "a number",
        1.0f32,
    )?;
    let d = preset.options(pixel_size);
    let text_opts = TextOptions {
        fill_char: char_arg(matches, "fillchar", "--fill-char", d.fill_char)?,
        empty_char: char_arg(matches, "emptychar", "--empty-char", d.empty_char)?,
        char_repeat: parse_arg(
            matches,
            "charrepeat",
            "--char-repeat",
            "an integer",
            d.char_repeat,
        )?,
        alpha_threshold: parse_arg(
            matches,
            "alphathreshold",
            "--alpha-threshold",
            "0-255",
            d.alpha_threshold,
        )?,
        mode: if matches.is_present("braille") {
            PixelMode::Braille
        } else if matches.is_present("blocks") {
            PixelMode::Blocks
        } else {
            PixelMode::Color
        },
        luma_threshold: parse_arg(
            matches,
            "lumathreshold",
            "--luma-threshold",
            "0-255",
            d.luma_threshold,
        )?,
        invert: matches.is_present("invert"),
        material,
        ..d
    };
    let text_opts = if text_opts.mode == PixelMode::Color {
        text_opts
    } else {
        // mono modes use their own measured component geometry
        let (line_height, kerning, line_offset, pitch_x, pitch_y) =
            mono_geometry(text_opts.mode, pixel_size);
        TextOptions {
            line_height,
            kerning,
            line_offset,
            pitch_x: pitch_x.unwrap_or(text_opts.pitch_x),
            pitch_y,
            ..text_opts
        }
    };
    if text_opts.char_repeat == 0 {
        return Err("--char-repeat must be at least 1".to_string());
    }
    Ok(text_opts)
}

/// A numeric flag's value, or `default` when it was not passed.
///
/// Deliberately NOT the `.expect("... must be a number")` the older branches
/// in this file use: that turns a mistyped value into a Rust panic trace with
/// a backtrace hint, which reads as a crash rather than as the user error it
/// is. `flag` and `what` are spelled out by the caller so the message names
/// the flag the user actually typed, not the internal arg id.
#[cfg(not(target_arch = "wasm32"))]
fn parse_arg<T>(
    matches: &clap::ArgMatches,
    name: &str,
    flag: &str,
    what: &str,
    default: T,
) -> Result<T, String>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match matches.value_of(name) {
        Some(s) => s
            .parse::<T>()
            .map_err(|e| format!("{flag} must be {what}: {e} (got '{s}')")),
        None => Ok(default),
    }
}

/// Like [`parse_arg`], but for a flag whose ABSENCE is itself meaningful and
/// so has no substitute default.
///
/// `--bands` is the one: with the band spacing musically fixed, "as many bands
/// as the pitch range holds" is not a number this file can pick -- it depends on
/// `--subdiv`, which `BandPlan` owns. Handing `audio_options` a concrete
/// default here would silently override the derived span whenever `--subdiv`
/// moved it.
#[cfg(not(target_arch = "wasm32"))]
fn parse_opt_arg<T>(
    matches: &clap::ArgMatches,
    name: &str,
    flag: &str,
    what: &str,
) -> Result<Option<T>, String>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match matches.value_of(name) {
        Some(s) => s
            .parse::<T>()
            .map(Some)
            .map_err(|e| format!("{flag} must be {what}: {e} (got '{s}')")),
        None => Ok(None),
    }
}

/// Every `--audio-mode` numeric flag, or its default. Returns `Result` so one
/// `?` per flag reports the first bad value. Range/consistency checks live in
/// the modules that own them (`BandPlan`, `analyze`) rather than duplicated
/// here.
#[cfg(not(target_arch = "wasm32"))]
fn audio_options(matches: &clap::ArgMatches) -> Result<AudioOptions, String> {
    let d = AudioOptions::default();
    // `SynthWave::parse`, not `parse_arg`: the value is an enum spelling, and
    // its own error already names the flag and every valid word -- the same
    // division of labour as `--audio-mode`, which surfaces `AudioMode::parse`'s
    // message verbatim. Absent means the default (sine).
    let tonal_synth = match matches.value_of("audiosynth") {
        Some(s) => SynthWave::parse(s)?,
        None => d.tonal_synth,
    };
    let opts = AudioOptions {
        fps: parse_arg(matches, "audiofps", "--audio-fps", "a number", d.fps)?,
        // `Option`, not a number with a default: absent means "every step the
        // pitch range holds at --subdiv", which is a different count for a
        // different subdivision. See `AudioOptions::bands`.
        bands: parse_opt_arg(matches, "audiobands", "--bands", "an integer")?,
        subdiv: parse_arg(matches, "audiosubdiv", "--subdiv", "an integer", d.subdiv)?,
        noise_bands: parse_arg(
            matches,
            "audionoise",
            "--noise-bands",
            "an integer",
            d.noise_bands,
        )?,
        window: parse_arg(matches, "audiowindow", "--window", "an integer", d.window)?,
        gain: parse_arg(matches, "audiogain", "--gain", "a number", d.gain)?,
        // Range left to `analyze`, which owns the 0..=1 meaning and rejects
        // anything outside it by name -- same division of labour as `--bands`
        // and `--audio-fps` below.
        leveling: parse_arg(matches, "audioleveling", "--leveling", "a number", d.leveling)?,
        // No range check here on purpose: every `usize` is meaningful. 0 is
        // the documented escape hatch (no selection, every band sounds), and
        // anything at or above the band count simply keeps every spectral
        // peak. A value that will not parse still errors, via `parse_arg`.
        max_voices: parse_arg(
            matches,
            "audiomaxvoices",
            "--max-voices",
            "an integer",
            d.max_voices,
        )?,
        // Range (>= 1.0) left to `analyze`, which owns the meaning. The flag
        // exists at all because the gate -- not `--max-voices` -- is what limits
        // a dense frame in bank mode, so without it every voice count above the
        // candidate count renders identically. See `track::DEFAULT_PEAK_GATE`.
        peak_gate: parse_arg(matches, "audiopeakgate", "--peak-gate", "a number", d.peak_gate)?,
        // Ranges left to `analyze` / `analyze_voices`, which own the meaning of
        // 0 (no smoothing) and reject a negative or non-finite time by name.
        attack_ms: parse_arg(matches, "audioattack", "--attack", "a number", d.attack_ms)?,
        release_ms: parse_arg(matches, "audiorelease", "--release", "a number", d.release_ms)?,
        voice_release_ms: parse_arg(
            matches,
            "audiovoicerelease",
            "--voice-release",
            "a number",
            d.voice_release_ms,
        )?,
        pitch_snap_cents: parse_arg(
            matches,
            "audiopitchsnap",
            "--pitch-snap",
            "a number",
            d.pitch_snap_cents,
        )?,
        // Range, not balance: these two decide whether the whole bank is
        // audible from where the listener stands. `build_speaker_world`
        // rejects a non-positive, non-finite or inverted pair, so the range
        // check is not duplicated here -- same division of labour as
        // `--bands` (checked by `BandPlan::new`) and `--audio-fps`
        // (checked by `analyze`).
        inner_radius: parse_arg(
            matches,
            "audioinner",
            "--inner-radius",
            "a number",
            d.inner_radius,
        )?,
        max_distance: parse_arg(
            matches,
            "audiomaxdist",
            "--max-distance",
            "a number",
            d.max_distance,
        )?,
        // Shared with the video path, and clamped the same way: the sentinel
        // that re-enables an unbounded loop must not be reachable from a
        // fat-fingered flag here either.
        max_frames: parse_arg(
            matches,
            "animmaxframes",
            "--max-frames",
            "an integer",
            d.max_frames,
        )?
        .min(MAX_FRAMES),
        // READ HERE, unlike `external_clock` just below: `--no-loop` is one
        // setting on one shared clock, and `build_speaker_world` builds that
        // clock exactly as the video path does. Negated and defaulting to
        // looping, same as the video path's `AnimOptions`.
        loop_playback: !matches.is_present("noloop"),
        // A plain boolean flag, off by default: the beside-the-chip placement
        // is unchanged for every existing render. `build_speaker_world` /
        // `build_voice_world` read this to put the speakers on the chip's inner
        // grid instead.
        speakers_in_chip: matches.is_present("audiospeakersinchip"),
        // NEGATED, default-on: the same `--no-control-buttons` off-switch the
        // video path reads, shared so one flag governs both pipelines.
        control_buttons: !matches.is_present("nocontrolbuttons"),
        // Parsed above, before the struct literal, so the enum error can `?`
        // out. Sine by default; reaches tonal bands only.
        tonal_synth,
        // Not `matches.is_present("externalclock")`: the audio branch has
        // already refused that flag outright, because nothing in
        // `build_speaker_world` reads this field.
        ..d
    };
    // THE one range check not left to the module that owns it, and the
    // exception earns itself. `analyze_voices` does check `--pitch-snap` -- but
    // only once it has been handed a source, and before it did, an out-of-band
    // value survived the ENTIRE analysis and then killed `build_voice_world`
    // with an error naming the data rather than the flag. A value that is
    // knowable wrong before a byte is decoded must not cost a full-length
    // render first. Checked against the module's own constant, so the front end
    // and the analysis cannot drift apart about what is legal.
    if !opts.pitch_snap_cents.is_finite()
        || !(0.0..=MAX_PITCH_SNAP_CENTS).contains(&opts.pitch_snap_cents)
    {
        return Err(format!(
            "--pitch-snap must be between 0 (off) and {MAX_PITCH_SNAP_CENTS} cents -- half a \
             semitone, and no pitch is ever further than that from the nearest one, so a \
             larger value cannot mean anything a listener would hear differently from \
             {MAX_PITCH_SNAP_CENTS}. Got {}",
            opts.pitch_snap_cents
        ));
    }
    Ok(opts)
}

/// The single character an `--fill-char`/`--empty-char` value names, or
/// `default` when the flag was absent.
///
/// **Absent and empty are not the same thing, and neither is "more than one".**
/// This used to be `.chars().next().unwrap_or(default)`, so `--fill-char ""`
/// rendered the preset glyph -- indistinguishable from never passing the flag --
/// and `--fill-char AB` rendered `A` with nothing said about the `B`. A glyph is
/// the most visible thing in a text render, so a value that cannot be honoured
/// is refused by name rather than quietly reinterpreted.
#[cfg(not(target_arch = "wasm32"))]
fn char_arg(
    matches: &clap::ArgMatches,
    name: &str,
    flag: &str,
    default: char,
) -> Result<char, String> {
    let Some(s) = matches.value_of(name) else {
        return Ok(default);
    };
    let mut chars = s.chars();
    let Some(c) = chars.next() else {
        return Err(format!(
            "{flag} needs a character: an empty value is not the same as leaving the flag off"
        ));
    };
    if chars.next().is_some() {
        return Err(format!(
            "{flag} takes exactly one character, got '{s}' ({} of them)",
            s.chars().count()
        ));
    }
    Ok(c)
}

/// Tests for the audio flag parsing only.
///
/// The rest of this file is unavoidably untested here: every render branch
/// ends in `fail!(...)` (which exits the process) or a file write, so
/// exercising one means running the binary -- `tests/cli_exit_codes.rs` and
/// `tests/cli_text_mode.rs` do exactly that. These cover the one failure mode
/// that is SILENT when
/// it regresses -- a bad numeric value falling back to its default and
/// rendering something the user never asked for. Everything else the audio
/// branch does (rejecting an unknown `--audio-mode`, refusing
/// `--external-clock`, warning about the image flags) announces itself on
/// stderr the moment it is wrong.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    /// Parsed against the REAL `cli()`, so a flag renamed or dropped up there
    /// fails these tests rather than silently testing a stale copy.
    fn args(extra: &[&str]) -> clap::ArgMatches<'static> {
        let mut argv = vec!["heightmap", "input.wav"];
        argv.extend_from_slice(extra);
        cli()
            .get_matches_from_safe(argv)
            .unwrap_or_else(|e| panic!("the real CLI must accept {extra:?}: {e}"))
    }

    #[test]
    fn absent_audio_flags_take_the_module_defaults() {
        let d = AudioOptions::default();
        let o = audio_options(&args(&[])).expect("no flags is valid");
        assert_eq!(o.fps, d.fps);
        // `None`, not a number: an absent `--bands` means "every step the
        // emitter's pitch range holds at --subdiv", which `BandPlan` alone
        // can work out. A concrete default here would override --subdiv.
        assert_eq!(o.bands, None);
        assert_eq!(o.bands, d.bands);
        assert_eq!(o.subdiv, d.subdiv);
        assert_eq!(o.noise_bands, d.noise_bands);
        assert_eq!(o.window, d.window);
        assert_eq!(o.gain, d.gain);
        assert_eq!(o.leveling, d.leveling);
        assert_eq!(o.max_voices, d.max_voices);
        assert_eq!(o.peak_gate, d.peak_gate);
        assert_eq!(o.attack_ms, d.attack_ms);
        assert_eq!(o.release_ms, d.release_ms);
        assert_eq!(o.voice_release_ms, d.voice_release_ms);
        assert_eq!(o.pitch_snap_cents, d.pitch_snap_cents);
        assert_eq!(o.max_frames, d.max_frames);
        assert_eq!(o.inner_radius, d.inner_radius);
        assert_eq!(o.max_distance, d.max_distance);
    }

    /// The two range flags default to the game's own single-prop values,
    /// 15 / 400, and the CLI parses them straight through.
    ///
    /// This is a deliberate near-field default: 15 is smaller than the cluster's
    /// own ~44-unit diagonal, so the bank attenuates within itself and reads as
    /// local rather than as one flat equal-level field. The larger 400 / 4000
    /// field is available through the flags; see `speakers::DEFAULT_INNER_RADIUS`
    /// for the full history of why this moved back and forth. The one invariant
    /// that holds either way: the no-attenuation radius sits inside the audible
    /// range, or the "flat" zone would extend past where sound is heard at all.
    #[test]
    fn the_range_defaults_are_the_games_near_field_prop_values() {
        let d = AudioOptions::default();
        assert_eq!(d.inner_radius, 15.0, "the game's single-prop InnerRadius");
        assert_eq!(d.max_distance, 400.0, "the game's single-prop MaxDistance");
        assert!(
            d.inner_radius < d.max_distance,
            "the no-attenuation radius ({}) must sit inside the audible range ({})",
            d.inner_radius,
            d.max_distance
        );
    }

    /// Each flag must reach its OWN field. Five same-typed values read out of
    /// one `ArgMatches` is exactly the shape a copy-paste swaps, and a swap
    /// renders a perfectly valid save of the wrong thing.
    #[test]
    fn each_audio_flag_reaches_its_own_field() {
        let o = audio_options(&args(&[
            "--audio-fps",
            "12",
            "--bands",
            "24",
            "--subdiv",
            "24",
            "--noise-bands",
            "1",
            "--window",
            "2048",
            "--gain",
            "2.5",
            "--leveling",
            "0.25",
            // Deliberately not 12 (the default) and not equal to `--bands`:
            // this is the flag that decides how many speakers sound at once,
            // and a default-valued assertion would pass with it unwired.
            "--max-voices",
            "7",
            // The gate that actually limits a dense frame. Not the default,
            // because a default-valued assertion would pass with it unwired --
            // which is precisely how four renders 40 voices apart came back
            // twelve bytes apart.
            "--peak-gate",
            "1.2",
            // Three more same-typed values in a row, none of them a default:
            // exactly the shape a copy-paste swaps.
            "--attack",
            "3.5",
            "--release",
            "222.0",
            // Deliberately unequal to `--release`: the two were ONE flag, and
            // an assertion that let them share a value would pass with the
            // split reverted.
            "--voice-release",
            "80.0",
            "--pitch-snap",
            "17.0",
            "--max-frames",
            "1000",
            // Deliberately unequal, and neither one a default: the two are
            // the same type, sit next to each other, and swapping them is
            // invisible to every other check in this file.
            "--inner-radius",
            "123.0",
            "--max-distance",
            "4567.0",
            // Not sine (the default): a default-valued assertion would pass
            // with the flag unwired.
            "--synth",
            "square",
        ]))
        .expect("valid values");
        assert_eq!(o.fps, 12.0);
        assert_eq!(o.bands, Some(24));
        assert_eq!(o.subdiv, 24);
        assert_eq!(o.noise_bands, 1);
        assert_eq!(o.window, 2048);
        assert_eq!(o.gain, 2.5);
        assert_eq!(o.leveling, 0.25);
        assert_eq!(o.max_voices, 7);
        assert_eq!(o.peak_gate, 1.2);
        assert_eq!(o.attack_ms, 3.5);
        assert_eq!(o.release_ms, 222.0);
        assert_eq!(o.voice_release_ms, 80.0);
        assert_eq!(o.pitch_snap_cents, 17.0);
        assert_eq!(o.max_frames, 1000);
        assert_eq!(o.inner_radius, 123.0);
        assert_eq!(o.max_distance, 4567.0);
        assert_eq!(o.tonal_synth, SynthWave::Square);
    }

    /// `--synth` parses each of the four waves, defaults to sine when absent,
    /// and refuses an unknown value with the flag and every valid word named --
    /// not a silent fall back to sine, which would render the wrong timbre and
    /// say nothing.
    #[test]
    fn synth_flag_parses_every_wave_and_refuses_the_rest() {
        assert_eq!(
            audio_options(&args(&[])).expect("no flag is valid").tonal_synth,
            SynthWave::Sine,
            "absent --synth is the default"
        );
        for w in SynthWave::ALL {
            let o = audio_options(&args(&["--synth", w.flag()])).expect("a valid wave");
            assert_eq!(o.tonal_synth, w, "--synth {} must reach the field", w.flag());
        }
        let err = audio_options(&args(&["--synth", "sawblade"]))
            .expect_err("an unknown wave must be rejected, not defaulted to sine");
        assert!(err.contains("--synth"), "the error must name the flag: {err}");
        for word in ["sine", "square", "triangle", "sawtooth"] {
            assert!(err.contains(word), "the error must list {word}: {err}");
        }
    }

    /// **The one that matters.** A value that will not parse must be an
    /// `Err` naming the flag -- never a panic (which reads as a crash) and
    /// never a silent fall back to the default (which renders the wrong
    /// track and says nothing at all).
    #[test]
    fn a_bad_numeric_flag_errors_instead_of_defaulting() {
        for (flag, bad) in [
            ("--audio-fps", "abc"),
            ("--bands", "abc"),
            ("--subdiv", "half"),
            // Not "-1": clap 2 reads a leading `-` as another flag and
            // refuses the whole command line first, which is a clean error of
            // its own but never reaches `audio_options`.
            ("--noise-bands", "two"),
            ("--window", "4096.5"),
            ("--gain", "loud"),
            ("--leveling", "flat"),
            // Not a panic: `--max-voices` is a `usize`, and "8.5" is exactly
            // the fat-finger that an `.expect(...)` would turn into a
            // backtrace instead of a CLI error.
            ("--max-voices", "8.5"),
            ("--peak-gate", "loose"),
            ("--attack", "fast"),
            ("--release", "slow"),
            ("--voice-release", "quick"),
            ("--pitch-snap", "close"),
            ("--max-frames", "1e6"),
            ("--inner-radius", "near"),
            ("--max-distance", "far"),
        ] {
            let err = audio_options(&args(&[flag, bad]))
                .expect_err(&format!("{flag} {bad} must be rejected, not defaulted"));
            assert!(
                err.contains(flag),
                "the error for {flag} must name the flag the user typed: {err}"
            );
            assert!(err.contains(bad), "the error must quote the bad value: {err}");
        }
    }

    /// **`--pitch-snap` is bounded HERE, before anything is decoded.**
    ///
    /// It used to be bounded nowhere. A value at or above 13.69 cents snapped a
    /// pitch at the edge of the playable band onto a semitone outside it, and
    /// the render died in `build_voice_world` -- after the entire analysis had
    /// run -- with an error naming the DATA rather than the flag. The snap
    /// itself no longer does that (`voices::VoiceShaping::snap`), but a value
    /// past half a semitone still cannot mean what the user thinks it means,
    /// and finding that out at the end of a full-length render is the worst
    /// place to find it out.
    #[test]
    fn an_out_of_range_pitch_snap_is_rejected_at_parse_time() {
        // No "-1": clap 2 reads a leading `-` as another flag and refuses the
        // command line before `audio_options` ever sees it -- see
        // `a_bad_numeric_flag_errors_instead_of_defaulting`.
        for bad in ["51", "60", "100", "1200", "nan", "inf"] {
            let err = audio_options(&args(&["--pitch-snap", bad]))
                .expect_err(&format!("--pitch-snap {bad} must be rejected at parse time"));
            assert!(
                err.contains("--pitch-snap"),
                "the error must name the flag the user typed: {err}"
            );
        }
        // ...and every legal value still parses, including the bound itself.
        for ok in ["0", "3", "13.69", "20", "50"] {
            audio_options(&args(&["--pitch-snap", ok]))
                .unwrap_or_else(|e| panic!("--pitch-snap {ok} is legal: {e}"));
        }
    }

    /// `--max-frames` is shared with the video path and clamped the same way
    /// there: the cap is what keeps a fat-fingered value from re-enabling an
    /// unbounded render.
    #[test]
    fn max_frames_is_clamped_to_the_overall_cap() {
        let o = audio_options(&args(&["--max-frames", "99999999"])).expect("valid");
        assert_eq!(o.max_frames, MAX_FRAMES);
    }

    /// `build_speaker_world` reads no `external_clock`, so the audio branch
    /// refuses the flag outright; these options must never carry it set,
    /// whatever was passed.
    #[test]
    fn audio_options_never_carry_external_clock() {
        assert!(!audio_options(&args(&["--external-clock"])).expect("valid").external_clock);
    }

    // Parsed through the real `cli()` (see `args`), so this also proves the
    // flag is registered and spelled `--no-loop`.
    #[test]
    fn looping_is_the_default_and_no_loop_is_what_turns_it_off() {
        assert!(
            audio_options(&args(&[])).expect("valid").loop_playback,
            "no flag must mean looping -- the pre-flag behaviour"
        );
        assert!(
            !audio_options(&args(&["--no-loop"])).expect("valid").loop_playback,
            "--no-loop must turn looping off"
        );
        // The field's own default has to agree, since `audio_options` fills
        // the rest of the struct from it via `..d`.
        assert!(AudioOptions::default().loop_playback);
        // And the video path's, which is the same setting on the same shared
        // clock -- the two must not be able to drift apart.
        assert!(AnimOptions::default().loop_playback);
    }

    #[test]
    fn parse_arg_returns_the_default_only_when_the_flag_is_absent() {
        let m = args(&[]);
        assert_eq!(parse_arg(&m, "audiobands", "--bands", "an integer", 7usize), Ok(7));
        let m = args(&["--bands", "9"]);
        assert_eq!(parse_arg(&m, "audiobands", "--bands", "an integer", 7usize), Ok(9));
    }

    /// `anim_options` with nothing but the flags -- the other arguments are
    /// whatever the caller already parsed, and none of these tests is about
    /// them.
    fn anim_opts(extra: &[&str]) -> Result<AnimOptions, String> {
        let matches = args(extra);
        let text_opts = text_options(&matches)?;
        anim_options(&matches, 0, &text_opts, None, 6.0, 8.0, 0.0)
    }

    #[test]
    fn srgb_to_linear_reaches_the_options_it_is_read_from() {
        assert!(
            !anim_opts(&[]).expect("valid").srgb_to_linear,
            "absent must mean off -- the pre-flag behaviour of every render"
        );
        assert!(
            anim_opts(&["--srgb-to-linear"]).expect("valid").srgb_to_linear,
            "--srgb-to-linear must reach the field the hex packer reads"
        );
    }

    /// The display flags this builder owns must each reach their own field --
    /// and `--pixel-extent` must be a CLI error rather than the panic the
    /// image path's copy produced.
    #[test]
    fn each_display_flag_reaches_its_own_field() {
        let o = anim_opts(&["--brick-style", "tile", "--pixel-extent", "3", "--glow"])
            .expect("valid");
        assert_eq!(o.brick_style, DisplayBrickStyle::SmoothTile);
        assert_eq!(o.pixel_extent, 3);
        assert!(o.glow);
        assert!(o.loop_playback, "looping is the default");
        assert!(!o.external_clock);

        let o = anim_opts(&["--no-loop", "--external-clock"]).expect("valid");
        assert!(!o.loop_playback);
        assert!(o.external_clock);
        assert_eq!(o.brick_style, DisplayBrickStyle::Micro, "the default style");
        assert_eq!(o.pixel_extent, 1, "the default extent");

        // `.err()` rather than `expect_err`: `AnimOptions` is not `Debug`.
        let err = anim_opts(&["--pixel-extent", "abc"])
            .err()
            .expect("a typo must be a CLI error, not a panic");
        assert!(err.contains("--pixel-extent"), "the error must name the flag: {err}");
        let err = anim_opts(&["--brick-style", "hologram"])
            .err()
            .expect("an unknown style must be refused");
        assert!(err.contains("brick style"), "{err}");
    }

    /// `--width 0` is refused rather than rendered as a screen with no pixels
    /// on it, and the source's own size stands when neither flag is passed.
    #[test]
    fn a_zero_target_size_is_refused_and_an_absent_one_keeps_the_source_size() {
        assert_eq!(target_size(&args(&[]), 320, 240), Ok(None));
        assert_eq!(
            target_size(&args(&["--width", "64"]), 320, 240),
            Ok(Some((64, 240))),
            "a missing axis takes the source's own"
        );
        for bad in [&["--width", "0"][..], &["--height", "0"][..]] {
            let err = target_size(&args(bad), 320, 240)
                .expect_err(&format!("{bad:?} must be refused"));
            assert!(err.contains(bad[0]), "the error must name the flag: {err}");
        }
        let err = target_size(&args(&["--width", "abc"]), 320, 240).expect_err("a typo");
        assert!(err.contains("--width"), "{err}");
    }

    /// **A glyph flag that cannot be honoured is refused, not reinterpreted.**
    ///
    /// `--fill-char ""` used to render the preset glyph -- indistinguishable
    /// from not passing the flag at all -- and `--fill-char AB` rendered `A`
    /// with nothing said about the `B`.
    #[test]
    fn a_multi_character_or_empty_glyph_is_refused() {
        for (flag, value) in [
            ("--fill-char", "AB"),
            ("--empty-char", "xyz"),
            ("--fill-char", ""),
            ("--empty-char", ""),
        ] {
            let err = text_options(&args(&[flag, value]))
                .err()
                .unwrap_or_else(|| panic!("{flag} {value:?} must be refused"));
            assert!(err.contains(flag), "the error must name the flag: {err}");
        }
        // ...and exactly one character still works, including a multi-byte one.
        let o = text_options(&args(&["--fill-char", "#", "--empty-char", "▒"])).expect("valid");
        assert_eq!(o.fill_char, '#');
        assert_eq!(o.empty_char, '▒');
    }
}
