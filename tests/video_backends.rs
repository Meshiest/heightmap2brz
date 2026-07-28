//! Task 9: cross-backend verification and `frame_count_hint` honesty for the
//! video decode backends -- see
//! `.superpowers/sdd/2026-07-27-video-decode-backends/task-9-brief.md` and
//! `task-9-report.md` for the full writeup, including the memory
//! measurement (deliberately NOT a test here -- see that report for why: a
//! 65 392-frame clip is far too slow for a committed suite, so it is run
//! once, by hand, and reported).
//!
//! **Which case is in force, and why:** the brief's Step 1 asks for a real
//! differential test "if both backends exist", or a stability-only fallback
//! "if only ffmpeg exists". `rust_h264` is an unconditional native
//! dependency (`Cargo.toml` -- not behind any feature flag), and Task 1's
//! evaluation (`task-1-evaluation.md`) found it decodes CABAC H.264 --
//! exactly what every clip generated below uses (`-coder 1`) -- correctly.
//! So on any machine that can build this crate natively, the pure-Rust
//! backend genuinely exists and works; the only backend whose *presence* is
//! ever actually in question at runtime is ffmpeg, an external binary that
//! may or may not be on `PATH`. Every test below is therefore the real
//! differential comparison (case A). Each one SKIPS -- prints a notice,
//! does not fail -- when ffmpeg itself is unavailable, matching the
//! convention every other clip-generating test in this crate already
//! follows (see `video::ffmpeg::tests::sample_clip`): without ffmpeg there
//! is no way to even manufacture a test clip here, so case B's "only ffmpeg
//! exists, pin its output for stability" fallback is not reachable from
//! this file either -- it would need a *pre-baked* fixture clip checked
//! into the repo, which this task's brief did not ask for and this file
//! does not add.
//!
//! **Why this duplicates a clip generator instead of importing
//! `video::ffmpeg::tests::sample_clip_args`:** that helper is `pub(crate)`,
//! gated by `#[cfg(test)]`. Verified directly against this file's
//! predecessor: `use heightmap::video::ffmpeg::tests::sample_clip;` fails to
//! compile here with "could not find `tests` in `ffmpeg` ... found an item
//! that was configured out" -- `#[cfg(test)]` is not set when the library is
//! compiled as a dependency of an integration-test binary (only for the
//! crate's own `--lib` test harness), and `pub(crate)` would hide it from an
//! external test crate even if it were. So this file re-implements the same
//! small technique (`ffmpeg -f lavfi -i testsrc2=... -coder 1`) rather than
//! reaching for an interface that does not actually exist in this build
//! configuration, despite looking identical to the in-crate one.
//!
//! **Why this file exists at all, given `rustvideo.rs` already has its own
//! `decoded_frames_match_the_ffmpeg_backend` oracle test:** that test
//! constructs `RustVideoSource`/`FfmpegSource` directly. This one goes
//! through the public `video::backend::open_video` *selection* API instead,
//! at both `Backend::Rust` and `Backend::Ffmpeg` explicitly -- the surface
//! an actual caller (the CLI, the GUI) uses -- so a bug introduced in
//! selection itself, rather than in either decoder, would also be caught
//! here.

use heightmap::video::backend::{Backend, open_video};
use heightmap::video::scale::{FitMode, Filter};
use heightmap::video::stream::FrameSource;
use image::RgbaImage;

/// Generates a tiny CABAC H.264 clip with a spawned `ffmpeg`. `-coder 1`
/// forces CABAC, the only entropy coding the pure-Rust backend accepts
/// (`video::rustvideo::RustVideoSource::open_path` refuses everything else).
/// The fixed-rate `testsrc2=...:rate={fps}` lavfi source with no timestamp
/// manipulation is also deliberately constant frame rate -- see
/// `both_backends_agree_within_the_established_tolerance`'s doc for why that
/// is load-bearing, not incidental.
/// Skips (returns `None`), never fails, when ffmpeg is not on `PATH` -- see
/// this file's module doc for why that is the right behavior here.
fn sample_cabac_clip(name: &str, secs: u32, w: u32, h: u32, fps: u32) -> Option<std::path::PathBuf> {
    if !heightmap::video::ffmpeg::ffmpeg_available() {
        eprintln!("SKIPPING {name}: ffmpeg not on PATH");
        return None;
    }
    let path = std::env::temp_dir().join(format!("h2b_xbk_{name}_{}.mp4", std::process::id()));
    let ok = std::process::Command::new("ffmpeg")
        .args([
            "-v", "error", "-y", "-f", "lavfi", "-i",
            &format!("testsrc2=size={w}x{h}:rate={fps}"),
            "-t", &secs.to_string(), "-pix_fmt", "yuv420p", "-coder", "1",
        ])
        .arg(&path)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    ok.then_some(path)
}

/// Drains every frame `source` produces via a fresh `.open()`.
fn drain(source: &dyn FrameSource) -> Vec<RgbaImage> {
    let mut stream = source.open().expect("open");
    let mut out = Vec::new();
    while let Some(f) = stream.next().expect("next") {
        out.push(f);
    }
    out
}

/// Mean absolute per-channel difference between two equally-sized RGBA
/// frames -- the same measurement Task 5's oracle test uses.
fn mean_abs_diff(a: &RgbaImage, b: &RgbaImage) -> f64 {
    let total: u64 = a
        .as_raw()
        .iter()
        .zip(b.as_raw())
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
        .sum();
    total as f64 / a.as_raw().len() as f64
}

/// **Step 1: the differential test.** Decodes the same CABAC clip through
/// `open_video(..., Backend::Rust, ...)` and `open_video(..., Backend::Ffmpeg,
/// ...)` and compares every frame at the tolerance Task 5 established: a
/// mean absolute per-channel difference under 3.0 ("structurally wrong"
/// above that), with CABAC content typically landing at 0.44-0.60. C
/// (ffmpeg) is B (the pure-Rust backend)'s oracle; this exercises that
/// relationship through the same `open_video` selection surface the CLI and
/// GUI actually call, not the concrete source types directly.
///
/// **Deliberately CFR-only, and this is load-bearing, not incidental.**
/// `sample_cabac_clip` generates its clip from a fixed-rate `testsrc2=...
/// :rate={fps}` lavfi source with no timestamp manipulation, so every frame
/// lands on an exact, evenly-spaced grid -- genuinely constant frame rate.
/// That matters because the ffmpeg backend's `frame_count_hint` fix (see
/// `video::ffmpeg::FfmpegSource::probe`'s doc) intentionally keeps ffmpeg's
/// own gap-filling behaviour on a real variable-frame-rate source: ffmpeg
/// DUPLICATES frames to conform variable input timing to the output's fixed
/// rate, so the frame COUNT the two backends produce can legitimately
/// diverge on VFR content -- the pure-Rust backend has no equivalent
/// conformance step at all, since `Demuxer`/`rust_h264` decode exactly the
/// samples the container declares, one per packet. A cross-backend test
/// over VFR input would therefore fail `assert_eq!(got.len(), want.len())`
/// below for a reason that has nothing to do with decode correctness, and
/// weakening that assertion (or the 3.0 tolerance) to tolerate it would blur
/// the exact signal this test exists to catch: a structurally wrong
/// pure-Rust decode on content BOTH backends agree how many frames there
/// are. So this test stays restricted to CFR sources on purpose, and any
/// future case added here must stay CFR too, or drop the frame-count
/// equality assertion deliberately rather than by accident.
#[test]
fn both_backends_agree_within_the_established_tolerance() {
    let Some(path) = sample_cabac_clip("xbk_cross", 1, 64, 48, 10) else {
        return;
    };

    let rust = open_video(&path, Backend::Rust, None, FitMode::Contain, Filter::Lanczos, None)
        .expect("Backend::Rust must open a CABAC clip");
    let ffmpeg = open_video(&path, Backend::Ffmpeg, None, FitMode::Contain, Filter::Lanczos, None)
        .expect("Backend::Ffmpeg must open the same clip");

    let got = drain(rust.as_ref());
    let want = drain(ffmpeg.as_ref());

    assert_eq!(got.len(), want.len(), "frame counts must agree between backends");
    assert!(!got.is_empty(), "the clip must actually contain frames");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.dimensions(), w.dimensions(), "frame {i} dimensions must agree");
        let mean = mean_abs_diff(g, w);
        assert!(
            mean < 3.0,
            "frame {i} mean abs channel diff {mean:.2} through the public backend::open_video \
             selection API -- the pure-Rust decoder is wrong, not merely different from ffmpeg"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// **Step 2: `frame_count_hint` honesty.** Spec A required
/// `SourceInfo::frame_count_hint` to be either exactly the number of frames
/// actually emitted or `None` -- never an approximation a caller could be
/// misled by -- and a later review found this violated once already
/// (`AdaptedSource::info`'s duration-window bug; see `src/video/stream.rs`'s
/// module doc and its regression test). This checks the same property per
/// backend, on a clip both can decode, going through `open_video` rather
/// than either concrete source type directly.
#[test]
fn frame_count_hint_is_honest_for_each_backend() {
    let Some(path) = sample_cabac_clip("xbk_hint", 1, 48, 32, 8) else {
        return;
    };

    for backend in [Backend::Rust, Backend::Ffmpeg] {
        let source = open_video(&path, backend, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("both backends must open a CABAC clip");
        let hint = source.info().frame_count_hint;
        let actual = drain(source.as_ref()).len();
        assert!(actual > 0, "{backend:?}: the clip must actually contain frames");
        if let Some(n) = hint {
            assert_eq!(
                n, actual,
                "{backend:?}: frame_count_hint {n} must equal the {actual} frames actually \
                 emitted, not merely approximate it"
            );
        }
        // `hint == None` is itself an honest answer (see `SourceInfo`'s own
        // doc) and needs no further assertion here.
    }
    let _ = std::fs::remove_file(&path);
}
