//! **Every failure path of the CLI must exit NON-ZERO.**
//!
//! Measured before this suite existed: only 7 of roughly 57 failure sites did.
//! `return error!(...)` prints and returns `()` from `main`, so the process
//! still exited 0 -- and
//! `heightmap song.mp3 --audio-mode voice --max-voices 0 -o out.brz && upload out.brz`
//! refused the render, wrote nothing, and ran the upload anyway. The same error
//! could even exit differently depending on which branch raised it (an unknown
//! `--font` exited 1 under `--anim-mode text` and 0 under `--text`), so a script
//! could not learn the rule by trying it.
//!
//! It is exactly the class of defect that regresses silently: nothing about a
//! wrong exit code is visible in the log, which still carries the right message.
//! So this drives the REAL binary and asserts the status, at least one case out
//! of every branch of `main`:
//!
//! - the pre-branch flag validation (`--backend`, download consent, subtitles,
//!   the output name),
//! - the `--audio-mode` branch,
//! - the `--anim-mode` branch, on a video-less input (the image/sequence path),
//! - the `--text` branch,
//! - the heightmap/`--img` branch.
//!
//! Three things are asserted every time, because each hides a different
//! regression: a non-zero status (the point), a message NAMING the flag or file
//! at fault (an exit code with an unattributable message is barely better), and
//! the absence of "panicked" (an exit 101 backtrace is technically non-zero and
//! is not what any of these should look like).
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

fn heightmap() -> Command {
    Command::new(env!("CARGO_BIN_EXE_heightmap"))
}

/// A tiny real PNG, written once per test process.
///
/// Several of these flags are only parsed AFTER the input has been decoded
/// (`--width` on the image path, for one), so a nonexistent input would fail
/// earlier for the wrong reason and the test would pass without proving
/// anything.
fn tiny_png() -> &'static PathBuf {
    static PNG: OnceLock<PathBuf> = OnceLock::new();
    PNG.get_or_init(|| {
        let path =
            std::env::temp_dir().join(format!("h2b_exit_codes_{}.png", std::process::id()));
        let mut img = image::RgbaImage::new(8, 8);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgba([(x * 32) as u8, (y * 32) as u8, 0x40, 0xFF]);
        }
        img.save(&path).expect("write the test png");
        path
    })
}

/// An output path this run can refuse to write, unique per case.
fn out_path(case: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "h2b_exit_codes_{}_{case}.brz",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    path
}

/// Run the CLI with `args` and assert it FAILED, naming `must_name`.
///
/// `case` names the temp output file, which is also asserted not to exist: a
/// refusal that still wrote a save would be a worse bug than the exit code.
fn assert_fails(case: &str, must_name: &str, args: &[&str]) {
    let out = out_path(case);
    let mut argv: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    argv.push("-o".to_string());
    argv.push(out.to_string_lossy().to_string());

    let result = heightmap().args(&argv).output().expect("spawn the CLI");
    let log = String::from_utf8_lossy(&result.stdout).into_owned()
        + &String::from_utf8_lossy(&result.stderr);

    assert!(
        !result.status.success(),
        "{case}: `{}` must exit non-zero, got {:?}\n{log}",
        argv.join(" "),
        result.status.code(),
    );
    assert!(
        log.contains(must_name),
        "{case}: the error must name {must_name:?} so a script's user can act on it:\n{log}"
    );
    assert!(
        !log.contains("panicked"),
        "{case}: must be a clean refusal, not a panic trace:\n{log}"
    );
    assert!(
        !out.exists(),
        "{case}: a refused render must write no save file"
    );
    let _ = std::fs::remove_file(&out);
}

/// The flag validation that runs before any branch is chosen.
#[test]
fn the_pre_branch_flag_validation_exits_nonzero() {
    let png = tiny_png().to_string_lossy().to_string();
    let png = png.as_str();

    assert_fails("backend", "--backend", &[png, "--backend", "nope"]);
    assert_fails("consent", "--no-download", &[png, "--yes", "--no-download"]);
    assert_fails(
        "two_subtitle_sources",
        "--subtitle-track",
        &[
            png,
            "--anim-mode",
            "brick",
            "--subtitles",
            "a.srt",
            "--subtitle-track",
            "0",
        ],
    );
}

/// **The output name is refused BEFORE the render, not after it.**
///
/// `write_world` is the last call in every branch, so `-o out` used to decode
/// the whole clip, print a cost line, build the world and only then refuse the
/// name -- discarding a 40-minute render over something knowable from argv
/// alone. This asserts both halves: non-zero, and no cost line printed.
#[test]
fn a_bad_output_extension_is_refused_before_anything_is_rendered() {
    let png = tiny_png().to_string_lossy().to_string();
    let result = heightmap()
        .args([png.as_str(), "--anim-mode", "brick", "-o", "out"])
        .output()
        .expect("spawn the CLI");
    let log = String::from_utf8_lossy(&result.stdout).into_owned()
        + &String::from_utf8_lossy(&result.stderr);

    assert!(!result.status.success(), "a bad output name must exit non-zero:\n{log}");
    assert!(log.contains("must end with .brz or .brdb"), "{log}");
    assert!(
        !log.contains("Estimated cost"),
        "the name must be refused before the render is costed, let alone run:\n{log}"
    );
    assert!(!PathBuf::from("out").exists(), "nothing may be written");
}

/// The `--audio-mode` branch. Every one of these is knowable from the flags
/// alone, and every one used to exit 0 -- the inverted radius pair after
/// running the whole analysis first.
#[test]
fn the_audio_branch_exits_nonzero() {
    let wav = "nonexistent_song.wav";

    assert_fails("audio_mode", "--audio-mode", &[wav, "--audio-mode", "nope"]);
    assert_fails(
        "audio_external_clock",
        "--external-clock",
        &[wav, "--audio-mode", "bank", "--external-clock"],
    );
    assert_fails(
        "audio_start",
        "--start",
        &[wav, "--audio-mode", "bank", "--start", "10"],
    );
    assert_fails(
        "audio_fps_value",
        "--audio-fps",
        &[wav, "--audio-mode", "bank", "--audio-fps", "abc"],
    );
    assert_fails(
        "audio_pitch_snap",
        "--pitch-snap",
        &[wav, "--audio-mode", "voice", "--pitch-snap", "100"],
    );
    assert_fails(
        "audio_subdiv",
        "--subdiv",
        &[wav, "--audio-mode", "bank", "--subdiv", "14"],
    );
    assert_fails(
        "audio_max_voices",
        "--max-voices",
        &[wav, "--audio-mode", "voice", "--max-voices", "0"],
    );
    // The M2 case: an Inner Radius above Max Distance. Refused from the flags,
    // before the file is even opened -- it used to print the analysis summary
    // first and the refusal after it.
    assert_fails(
        "audio_attenuation",
        "--inner-radius",
        &[
            wav,
            "--audio-mode",
            "bank",
            "--inner-radius",
            "400",
            "--max-distance",
            "10",
        ],
    );
    // ...and a source that cannot be opened at all.
    assert_fails("audio_missing", "nonexistent_song.wav", &[wav, "--audio-mode", "bank"]);
}

/// The `--anim-mode` branch, on the image/sequence path.
///
/// The numeric flags here are the H2 half: `--fps`, `--start`, `--duration`,
/// `--max-frames`, `--width`, `--height` and `--pixel-extent` all used to
/// `.expect(...)` and exit 101 with a backtrace hint, while the identical flags
/// in the audio and video branches errored cleanly.
#[test]
fn the_anim_branch_exits_nonzero() {
    let png = tiny_png().to_string_lossy().to_string();
    let png = png.as_str();
    let anim = [png, "--anim-mode", "brick"];
    let with = |extra: &[&'static str]| -> Vec<&str> {
        let mut v = anim.to_vec();
        v.extend_from_slice(extra);
        v
    };

    assert_fails("anim_mode", "text", &[png, "--anim-mode", "hologram"]);
    assert_fails("anim_fps", "--fps", &with(&["--fps", "abc"]));
    assert_fails("anim_start", "--start", &with(&["--start", "abc"]));
    assert_fails("anim_duration", "--duration", &with(&["--duration", "abc"]));
    assert_fails("anim_max_frames", "--max-frames", &with(&["--max-frames", "abc"]));
    assert_fails("anim_width", "--width", &with(&["--width", "abc"]));
    assert_fails("anim_height", "--height", &with(&["--height", "abc"]));
    assert_fails(
        "anim_pixel_extent",
        "--pixel-extent",
        &with(&["--pixel-extent", "abc"]),
    );
    assert_fails("anim_fit", "fit mode", &with(&["--fit", "nope"]));
    assert_fails("anim_filter", "filter", &with(&["--filter", "nope"]));
    assert_fails("anim_style", "brick style", &with(&["--brick-style", "nope"]));
    assert_fails("anim_colors", "--colors", &with(&["--colors", "many"]));
    assert_fails(
        "anim_subtitles",
        "subtitle",
        &with(&["--subtitles", "nonexistent.srt"]),
    );
    assert_fails(
        "anim_missing",
        "nonexistent_clip.png",
        &["nonexistent_clip.png", "--anim-mode", "brick"],
    );
}

/// The `--text` branch. `--font bogus` exited 1 under `--anim-mode text` and 0
/// here, from the identical `text_options` error -- the sharpest instance of
/// the whole finding.
#[test]
fn the_text_branch_exits_nonzero() {
    let png = tiny_png().to_string_lossy().to_string();
    let png = png.as_str();

    assert_fails("text_font", "font", &[png, "--text", "--font", "bogus"]);
    assert_fails("text_material", "material", &[png, "--text", "--material", "nope"]);
    assert_fails(
        "text_char_repeat",
        "--char-repeat",
        &[png, "--text", "--char-repeat", "0"],
    );
    assert_fails(
        "text_missing",
        "nonexistent_text.png",
        &["nonexistent_text.png", "--text"],
    );
}

/// The heightmap / `--img` branch, the oldest one and the last to keep its
/// `.expect(...)`s.
#[test]
fn the_heightmap_branch_exits_nonzero() {
    let png = tiny_png().to_string_lossy().to_string();
    let png = png.as_str();

    assert_fails("hm_size_value", "--size", &[png, "--img", "--size", "abc"]);
    assert_fails(
        "hm_vertical_value",
        "--vertical",
        &[png, "--vertical", "abc"],
    );
    assert_fails("hm_missing", "nonexistent_map.png", &["nonexistent_map.png", "--img"]);
    // An input the heightmap path has no decoder for.
    let txt = std::env::temp_dir().join(format!("h2b_exit_codes_{}.txt", std::process::id()));
    std::fs::write(&txt, b"not an image").expect("write the stub");
    let txt_s = txt.to_string_lossy().to_string();
    assert_fails("hm_format", "olormap format", &[txt_s.as_str()]);
    let _ = std::fs::remove_file(&txt);
}

/// **A zero-pixel render is refused, not "Done!".**
///
/// `--width 0 --height 0` in brick mode used to print a cost line reading "0
/// pixel(s)", write a 5678-byte save and exit 0, while `--anim-mode text`
/// refused the identical input and the GUI's sliders could not express it at
/// all. `--size 0` / `--vertical 0` are the same shape on the heightmap path,
/// where the GUI's sliders likewise start at 1.
#[test]
fn a_zero_sized_render_is_refused_everywhere() {
    let png = tiny_png().to_string_lossy().to_string();
    let png = png.as_str();

    assert_fails(
        "zero_size",
        "--width",
        &[png, "--anim-mode", "brick", "--width", "0", "--height", "0"],
    );
    assert_fails(
        "zero_width_only",
        "--width",
        &[png, "--anim-mode", "brick", "--width", "0"],
    );
    assert_fails(
        "zero_height_only",
        "--height",
        &[png, "--anim-mode", "brick", "--height", "0"],
    );
    assert_fails("zero_studs", "--size", &[png, "--img", "--size", "0"]);
    assert_fails("zero_vertical", "--vertical", &[png, "--vertical", "0"]);
}

/// **`--braille`/`--blocks` under `--anim-mode text` is refused, not warned
/// about and then shipped.**
///
/// The two disagree by construction: `text_options` replaces the geometry with
/// `mono_geometry(Braille, ..)`, so the world is laid out for braille, while the
/// animated text encoder (`text::encode_bands`/`encode_row`) never consults
/// `opts.mode` and emits colour runs regardless. The old behaviour printed the
/// warning, printed a cost readout describing the COLOUR render, wrote a
/// 7721-byte save and exited 0.
#[test]
fn braille_under_anim_text_is_refused_rather_than_rendered_wrong() {
    let png = tiny_png().to_string_lossy().to_string();
    let png = png.as_str();

    assert_fails(
        "anim_text_braille",
        "--braille",
        &[png, "--anim-mode", "text", "--braille"],
    );
    assert_fails(
        "anim_text_blocks",
        "--blocks",
        &[png, "--anim-mode", "text", "--blocks"],
    );
}

/// The complementary case for every refusal above: a legal command line still
/// renders and still exits 0.
///
/// Without this, "exit non-zero" is satisfiable by exiting non-zero always.
#[test]
fn a_good_command_line_still_exits_zero_and_writes_its_save() {
    let png = tiny_png().to_string_lossy().to_string();
    let out = out_path("success");

    let result = heightmap()
        .args([
            png.as_str(),
            "--anim-mode",
            "brick",
            "--width",
            "4",
            "--height",
            "4",
            "-o",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("spawn the CLI");
    let log = String::from_utf8_lossy(&result.stdout).into_owned()
        + &String::from_utf8_lossy(&result.stderr);

    assert!(result.status.success(), "a valid render must exit 0:\n{log}");
    assert!(out.exists(), "a valid render must write its save:\n{log}");
    let _ = std::fs::remove_file(&out);
}
