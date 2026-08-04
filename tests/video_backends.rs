//! Cross-backend verification and `frame_count_hint` honesty for the video
//! decode backends, exercised through the public
//! `video::backend::open_video` selection API rather than the concrete
//! source types directly, so a bug in selection itself (not just in either
//! decoder) would be caught here too.
//!
//! Every test skips (prints a notice, does not fail) when ffmpeg is not on
//! `PATH`, since there is no way to manufacture a test clip here without it.
//! The clip generator below duplicates
//! `video::ffmpeg::tests::sample_clip_args` rather than importing it: that
//! helper is `pub(crate)` behind `#[cfg(test)]`, neither of which an
//! external integration-test binary can reach.

use heightmap::video::backend::{Backend, open_video};
use heightmap::video::scale::{FitMode, Filter};
use heightmap::video::stream::FrameSource;
use image::RgbaImage;

/// Generates a tiny CABAC H.264 clip with a spawned `ffmpeg`. `-coder 1`
/// forces CABAC, the only entropy coding the builtin backend accepts. The
/// fixed-rate `testsrc2=...:rate={fps}` lavfi source is deliberately
/// constant frame rate -- see
/// `both_backends_agree_within_the_established_tolerance`'s doc for why.
/// Skips (returns `None`), never fails, when ffmpeg is not on `PATH`.
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
/// frames.
fn mean_abs_diff(a: &RgbaImage, b: &RgbaImage) -> f64 {
    let total: u64 = a
        .as_raw()
        .iter()
        .zip(b.as_raw())
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
        .sum();
    total as f64 / a.as_raw().len() as f64
}

/// Decodes the same CABAC clip through both backends via `open_video` and
/// compares every frame at the established tolerance: a mean absolute
/// per-channel difference under 3.0 ("structurally wrong" above that).
///
/// Deliberately CFR-only, and that is load-bearing: on a genuinely
/// variable-frame-rate source, ffmpeg's decode-time conformance duplicates
/// frames to fill timing gaps while the builtin backend (which decodes
/// exactly the samples the container declares) does not, so the two
/// backends' frame counts can legitimately diverge for reasons that have
/// nothing to do with decode correctness. Any future case added here must
/// stay CFR too, or drop the frame-count equality assertion deliberately.
#[test]
fn both_backends_agree_within_the_established_tolerance() {
    let Some(path) = sample_cabac_clip("xbk_cross", 1, 64, 48, 10) else {
        return;
    };

    let rust = open_video(&path, Backend::Builtin, None, FitMode::Contain, Filter::Lanczos, None)
        .expect("Backend::Builtin must open a CABAC clip");
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
             selection API -- the builtin decoder is wrong, not merely different from ffmpeg"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// `SourceInfo::frame_count_hint` must be either exactly the number of
/// frames actually emitted or `None` -- never an approximation a caller
/// could be misled by (see `AdaptedSource::info`'s duration-window
/// regression test in `src/video/stream.rs` for why this was violated once
/// already). Checked per backend through `open_video`.
#[test]
fn frame_count_hint_is_honest_for_each_backend() {
    let Some(path) = sample_cabac_clip("xbk_hint", 1, 48, 32, 8) else {
        return;
    };

    for backend in [Backend::Builtin, Backend::Ffmpeg] {
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
