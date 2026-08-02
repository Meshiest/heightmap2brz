//! CLI-level coverage for `--anim-mode text` and `--colors`.
//!
//! These drive the real binary rather than calling into the library, because
//! the thing under test is the argument wiring itself -- the code path that
//! turns typed flags into an `AnimMode` and an `AnimOptions`. A library-level
//! test cannot see a flag that was parsed into the wrong field, or one that
//! was never read at all.
use std::process::Command;

fn heightmap() -> Command {
    Command::new(env!("CARGO_BIN_EXE_heightmap"))
}

#[test]
fn text_mode_is_no_longer_rejected() {
    let out = heightmap().args(["--anim-mode", "text", "--help"]).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    assert!(!text.contains("later phase"), "the placeholder rejection must be gone");
}

#[test]
fn an_unknown_mode_exits_nonzero_and_names_the_valid_modes() {
    let out = heightmap()
        .args(["nonexistent.png", "--anim-mode", "hologram"])
        .output()
        .unwrap();
    assert!(!out.status.success(), "an unknown mode must not exit 0");
    let text = String::from_utf8_lossy(&out.stderr);
    assert!(text.contains("brick") && text.contains("text"), "{text}");
}

#[test]
fn colors_rejects_a_non_numeric_value_without_panicking() {
    let out = heightmap()
        .args(["nonexistent.png", "--anim-mode", "text", "--colors", "many"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let text = String::from_utf8_lossy(&out.stderr);
    assert!(text.contains("--colors"), "names the flag: {text}");
    assert!(!text.contains("panicked"), "must be a clean error, not a panic: {text}");
}

/// **The one that proves text mode is reachable at all.** The three tests
/// above all describe error paths, and every one of them was already
/// satisfied by the placeholder rejection this task removes (clap's own
/// "unexpected argument" message even names `--colors`). This one renders a
/// real save through the binary, which the placeholder could never do.
#[test]
fn text_mode_renders_a_save_through_the_cli() {
    let dir = std::env::temp_dir();
    let png = dir.join(format!("h2b_cli_text_{}.png", std::process::id()));
    let brz = dir.join(format!("h2b_cli_text_{}.brz", std::process::id()));
    // A small gradient: enough distinct colours that `--colors` has something
    // to quantize, small enough that the render is instant.
    let mut img = image::RgbaImage::new(32, 16);
    for (x, y, p) in img.enumerate_pixels_mut() {
        *p = image::Rgba([(x * 8) as u8, (y * 16) as u8, 0x40, 0xFF]);
    }
    img.save(&png).unwrap();

    let out = heightmap()
        .args([
            png.to_str().unwrap(),
            "--anim-mode",
            "text",
            "--colors",
            "8",
            "-o",
            brz.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let log = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "text render must succeed: {log}");
    assert!(brz.exists(), "no save written: {log}");
    // The cost readout must name the mode and report the character BOUND,
    // never a bogus "0 character(s)" estimate (`Cost::chars` is 0 in text
    // mode by design).
    assert!(log.contains("text"), "the cost readout must name the mode: {log}");
    assert!(
        log.to_lowercase().contains("bound"),
        "the character figure must be labelled a bound, not an estimate: {log}"
    );

    let _ = std::fs::remove_file(&png);
    let _ = std::fs::remove_file(&brz);
}
