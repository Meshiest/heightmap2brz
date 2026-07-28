//! Backend selection: picks which [`FrameSource`] implementation opens a
//! video file, and enforces the two safety guards that keep the pure-Rust
//! decoder away from streams it decodes incorrectly.
//!
//! Two guards, same shape:
//!
//! - **CAVLC entropy** ([`crate::video::demux::EntropyCoding::Cavlc`], and
//!   [`crate::video::demux::EntropyCoding::Unknown`] treated the same way,
//!   unsafe on purpose). `rust_h264` decodes these to visibly wrong pixels
//!   with no error and no panic (measured: mean abs channel diff up to ~27
//!   luma, ~49 chroma -- see `.superpowers/sdd/2026-07-27-video-decode-
//!   backends/task-1-evaluation.md`). [`RustVideoSource::open_path`] already
//!   refuses this before this module exists at all; this module's job is to
//!   make [`Backend::Auto`] route AROUND it rather than surface the refusal
//!   as a hard error.
//! - **BT.2020 colour matrix** (`matrix_coefficients` 9 or 10). Not in the
//!   original task brief -- added because Task 5's fix-round-1 report
//!   (`progress.md`) flagged it as a known, MEASURED gap: falling back to
//!   BT.601 scores 2.83-2.91 mean abs per-channel diff against ffmpeg on
//!   synthetic content, close enough to this project's 3.0 "structurally
//!   wrong" threshold that real saturated footage could plausibly exceed it
//!   silently. `colour_info_from_sps_rbsp` in `rustvideo.rs` folds 9/10 into
//!   BT.601 without erroring -- safe only because THIS module is the one
//!   meant to intercept it first, exactly the same division of labour as the
//!   CAVLC guard (rustvideo.rs's own checks are belt, this module is
//!   suspenders -- see `RustVideoSource::open_path`'s doc for that framing).
//!
//! **Selection happens exactly once**, inside [`open_video`]. The returned
//! `Box<dyn FrameSource>` is a concrete, already-chosen backend; nothing
//! stored in it re-probes on a later [`FrameSource::open`] call. This matters
//! beyond tidiness: a later feature's two-pass scan calls `open()` twice on
//! the same source, and a source that could land on a different backend
//! between those two calls would silently compare two different decoders'
//! output against each other.
//!
//! **On wasm there is no ffmpeg backend at all** (`video::ffmpeg` is
//! `#[cfg(not(target_arch = "wasm32"))]`). A CAVLC or BT.2020 stream there has
//! no correct decode path and [`open_video`] fails with a clear message --
//! never a best-effort decode through the guard.

use crate::video::demux::Demuxer;
use crate::video::rustvideo::{self, RustVideoSource};
use crate::video::scale::{FitMode, Filter};
use crate::video::stream::FrameSource;
use std::path::Path;

/// Which decode backend handles a video file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Pick whichever backend can decode this file correctly: the pure-Rust
    /// path when it is safe (H.264, CABAC entropy, not BT.2020), ffmpeg
    /// otherwise. Never CAVLC/`Unknown`/BT.2020 into the pure-Rust path.
    Auto,
    /// The pure-Rust path, or an error naming the reason when it is unsafe
    /// for this specific file -- never a silent wrong decode.
    Rust,
    /// The known-good ffmpeg subprocess path, unconditionally.
    Ffmpeg,
}

/// Opens `path` with `backend`, resolving [`Backend::Auto`] to a concrete
/// backend exactly once. `target`/`fit`/`filter`/`fps` are the ffmpeg
/// backend's own decode-time resize/resample settings (see
/// [`crate::video::ffmpeg::FfmpegSource::probe`]) -- the pure-Rust backend has
/// no equivalent of its own (`RustVideoSource::open_path` takes no such
/// parameters at all), so a caller that needs a specific size or rate out of
/// that backend layers `video::stream::AdaptedSource` over the result itself,
/// the same way the pre-existing image-sequence path already does over a
/// `Clip`.
pub fn open_video(
    path: &Path,
    backend: Backend,
    target: Option<(u32, u32)>,
    fit: FitMode,
    filter: Filter,
    fps: Option<f32>,
) -> Result<Box<dyn FrameSource>, String> {
    match backend {
        Backend::Ffmpeg => open_ffmpeg(path, target, fit, filter, fps),
        Backend::Rust => {
            open_rust_checked(path).map(|s| Box::new(s) as Box<dyn FrameSource>)
        }
        Backend::Auto => match open_rust_checked(path) {
            Ok(source) => Ok(Box::new(source)),
            // Whatever blocked the pure-Rust path (wrong codec, CAVLC,
            // Unknown entropy, BT.2020, a demux failure) -- fall back to
            // ffmpeg silently, per this module's doc. The reason itself is
            // discarded here on purpose: it is only user-actionable when the
            // user explicitly asked for `Backend::Rust`, which is the branch
            // above, not this one.
            Err(_reason) => open_ffmpeg(path, target, fit, filter, fps),
        },
    }
}

/// Builds a [`RustVideoSource`] for `path`, refusing (with a message naming
/// the reason) anything this task's two guards say is unsafe.
///
/// Layers the BT.2020 guard on top of [`RustVideoSource::open_path`]'s own
/// codec/CAVLC/Unknown refusals (Task 5) rather than re-implementing them, so
/// there is exactly one wording for each of those, and this only adds the one
/// check `open_path` does not make itself.
fn open_rust_checked(path: &Path) -> Result<RustVideoSource, String> {
    let source = RustVideoSource::open_path(path)?;

    // `open_path` above already guarantees codec == h264, entropy == Cabac,
    // and an out-of-band avcC/CodecPrivate record exists -- everything
    // `sps_declares_bt2020` needs is present, so this is checking a
    // genuinely new condition, not re-deriving one `open_path` already ruled
    // out. A second `Demuxer::open` (a second read of the file from disk)
    // rather than threading one through: `RustVideoSource` does not expose
    // its internal demuxer, and `RustFrameStream::new` already re-opens one
    // fresh per `open()` for the same reason -- this module treats each
    // check as re-derived from the file, not trusted from an earlier step.
    let demuxer = Demuxer::open(path)?;
    if rustvideo::sps_declares_bt2020(&demuxer) {
        return Err(bt2020_refusal(path));
    }

    Ok(source)
}

/// Why the pure-Rust backend could not open a file -- specifically, whether
/// ffmpeg is a plausible next step or whether nothing would help.
///
/// Derived STRUCTURALLY, by re-running the two steps that come *before* the
/// guards (can the file be opened for reading at all, and does its container
/// parse), rather than by matching on the text of [`open_video`]'s error.
/// Every failure in this module and in `RustVideoSource::open_path` is a plain
/// `String`, so a guard's refusal is not otherwise distinguishable from a
/// corrupt file's parse error; matching on "use --backend ffmpeg" would work
/// today and break silently the first time one of those messages is reworded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustFailure {
    /// The container parsed fine, but a guard refused this specific stream:
    /// CAVLC or unknown entropy coding, a BT.2020 colour matrix, a non-H.264
    /// codec, or no out-of-band parameter sets. ffmpeg decodes all of these
    /// correctly, so it is exactly the right next step.
    Guard,
    /// The file is readable, but [`Demuxer::open`] could not parse it: an AVI
    /// (this demuxer handles only MP4/MOV and MKV/WebM, yet `avi` is one of
    /// `video::source::VIDEO_EXTENSIONS`), or a corrupt or otherwise
    /// unsupported file. ffmpeg may still manage it -- it demuxes far more
    /// containers, and tolerates damage this one refuses -- so this warrants a
    /// fallback too; lumping it in with [`RustFailure::Unreadable`] would make
    /// `--backend auto` refuse every AVI outright, which it handles today.
    Container,
    /// The file could not be opened for reading at all: missing, a directory,
    /// or permission denied. No decode backend can fix that, so this is the
    /// one class that must NOT be retried under ffmpeg -- doing so would
    /// report a missing-ffmpeg problem for a file that was simply not there.
    Unreadable,
}

/// Classifies why the pure-Rust backend failed on `path`; see [`RustFailure`].
///
/// Deliberately re-derived from the file rather than threaded out of the
/// failing call: this module already treats every check as re-read from disk
/// (see [`open_rust_checked`], which opens a second [`Demuxer`] for exactly
/// the same reason), and the alternative would mean changing
/// `RustVideoSource::open_path`'s established `Result<_, String>` signature.
/// Only ever called on a path that has ALREADY failed, so the extra read costs
/// nothing on the success path.
pub fn classify_rust_failure(path: &Path) -> RustFailure {
    if std::fs::File::open(path).is_err() {
        return RustFailure::Unreadable;
    }
    if Demuxer::open(path).is_err() {
        return RustFailure::Container;
    }
    // Readable, and the container parsed -- so whatever `open_rust_checked`
    // refused, it refused *after* those two steps, which leaves exactly the
    // guards.
    RustFailure::Guard
}

/// [`open_video`], but resolving [`Backend::Auto`] by *trying* the pure-Rust
/// backend before ever consulting `ensure_ffmpeg`.
///
/// This exists because `Auto` is the CLI default, and eagerly requiring ffmpeg
/// for it made a machine without ffmpeg refuse every video by default --
/// including the CABAC H.264 files the pure-Rust backend decodes perfectly
/// well on its own. Trying first costs nothing (the attempt *is* the probe)
/// and needs no duplicate of this module's private CAVLC/BT.2020 logic.
///
/// `ensure_ffmpeg` is injected rather than called directly so this stays free
/// of `video::ffmpeg` (which does not exist on wasm at all) and so a test can
/// simulate "ffmpeg is unavailable and consent was denied" without mutating
/// the process-wide `PATH` a parallel test suite shares. `main.rs` passes
/// `|| ffmpeg::ensure_ffmpeg(consent)`.
///
/// Per backend:
/// - [`Backend::Rust`]: never calls `ensure_ffmpeg`. The guards still refuse
///   an unsafe stream by name -- an error, never a silently wrong decode.
/// - [`Backend::Ffmpeg`]: calls `ensure_ffmpeg` up front, as before. The user
///   named the backend, so failing fast on a missing binary is right there.
/// - [`Backend::Auto`]: pure-Rust first; `ensure_ffmpeg` only once that has
///   failed with a [`RustFailure`] ffmpeg could actually help with.
pub fn open_video_ensuring(
    path: &Path,
    backend: Backend,
    target: Option<(u32, u32)>,
    fit: FitMode,
    filter: Filter,
    fps: Option<f32>,
    ensure_ffmpeg: &mut dyn FnMut() -> Result<(), String>,
) -> Result<Box<dyn FrameSource>, String> {
    match backend {
        Backend::Rust => open_video(path, Backend::Rust, target, fit, filter, fps),
        Backend::Ffmpeg => {
            ensure_ffmpeg()?;
            open_video(path, Backend::Ffmpeg, target, fit, filter, fps)
        }
        Backend::Auto => {
            let rust_err = match open_video(path, Backend::Rust, target, fit, filter, fps) {
                // The pure-Rust backend handled it, so ffmpeg was never
                // needed and was never asked about. That is the whole point.
                Ok(source) => return Ok(source),
                Err(e) => e,
            };

            // A file that cannot be read is not an ffmpeg problem and must
            // not be reported as one.
            if classify_rust_failure(path) == RustFailure::Unreadable {
                return Err(rust_err);
            }

            if let Err(ffmpeg_err) = ensure_ffmpeg() {
                // BOTH reasons, never just the second: the pure-Rust refusal
                // is the one that says what is actually wrong with the file
                // (e.g. "this stream uses CAVLC entropy coding"), and
                // swallowing it in favour of "ffmpeg was not found" would
                // leave the user holding the less actionable half.
                //
                // Each half's own text carries a "next step" suggestion
                // written for when IT is the only backend that failed --
                // `rust_err` says "use --backend ffmpeg instead", and
                // `ffmpeg_err` (from `ffmpeg::refusal`) says "run with
                // --backend rust". Concatenated as-is those two sentences
                // point in opposite directions and neither is actually
                // actionable here, since BOTH backends have already refused
                // this exact file. The closing paragraph below is what
                // resolves that: it says so explicitly, and names what
                // genuinely would help, so a reader doesn't have to notice
                // the contradiction and guess which suggestion (if either)
                // still applies.
                return Err(format!(
                    "{}: neither decode backend can handle this file.\n  \
                     pure-Rust backend: {rust_err}\n  ffmpeg backend: {ffmpeg_err}\n\n\
                     Switching backends will not help -- both have already refused this exact \
                     file. Install ffmpeg (and make sure it is on PATH) if it is genuinely \
                     unavailable, or try a different source file.",
                    path.display()
                ));
            }
            open_video(path, Backend::Ffmpeg, target, fit, filter, fps)
        }
    }
}

/// The refusal [`open_rust_checked`] returns for a BT.2020-tagged stream.
/// Names the reason (so `err.contains("BT.2020")` is a stable, testable
/// contract) and suggests the same next step every other refusal in this
/// backend does.
fn bt2020_refusal(path: &Path) -> String {
    format!(
        "{}: this H.264 stream's SPS declares a BT.2020 colour matrix (matrix_coefficients 9 \
         or 10), which the pure-Rust decoder falls back to a BT.601 conversion for rather than \
         decoding correctly -- measured 2.83-2.91 mean abs per-channel difference against \
         ffmpeg on synthetic content, close enough to this project's 3.0 \"wrong\" threshold \
         that real saturated footage could silently exceed it; use --backend ffmpeg instead",
        path.display()
    )
}

/// The ffmpeg backend, present only on targets that can spawn a subprocess.
#[cfg(not(target_arch = "wasm32"))]
fn open_ffmpeg(
    path: &Path,
    target: Option<(u32, u32)>,
    fit: FitMode,
    filter: Filter,
    fps: Option<f32>,
) -> Result<Box<dyn FrameSource>, String> {
    let source = crate::video::ffmpeg::FfmpegSource::probe(path, target, fit, filter, fps)?;
    Ok(Box::new(source))
}

/// On wasm there is no `video::ffmpeg` module at all (it is `#[cfg(not(
/// target_arch = "wasm32"))]`, since it spawns a subprocess) -- so a file that
/// the pure-Rust guards refuse has no correct decode path here. This errors
/// with a clear message rather than falling back to a "best effort" decode,
/// per this module's own doc and the task brief's explicit constraint.
#[cfg(target_arch = "wasm32")]
fn open_ffmpeg(
    path: &Path,
    _target: Option<(u32, u32)>,
    _fit: FitMode,
    _filter: Filter,
    _fps: Option<f32>,
) -> Result<Box<dyn FrameSource>, String> {
    Err(format!(
        "{}: no ffmpeg decode backend is available in this build (the browser build cannot \
         spawn a subprocess), and this file is not safe for the pure-Rust decoder either; \
         there is no correct way to decode it here",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_explicit_backend_is_honoured() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip("sel", 1, 32, 32, 5) else { return };
        let s = open_video(&path, Backend::Ffmpeg, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("explicit ffmpeg");
        assert_eq!(s.info().width, 32);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn auto_selects_a_backend_that_can_actually_open_the_file() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip("auto", 1, 32, 32, 5) else { return };
        let s = open_video(&path, Backend::Auto, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("auto must find a working backend");
        let mut st = s.open().expect("open");
        assert!(st.next().expect("next").is_some(), "the selected backend must actually decode");
        let _ = std::fs::remove_file(&path);
    }

    /// Selection is made ONCE. Two opens of the same source must not be able
    /// to land on different backends -- text mode's two-pass scan would then
    /// compare two different decoders' output.
    #[test]
    fn selection_is_stable_across_opens() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip("stable", 1, 24, 24, 5) else { return };
        let s = open_video(&path, Backend::Auto, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("auto");
        let drain = || {
            let mut st = s.open().expect("open");
            let mut v = Vec::new();
            while let Some(f) = st.next().expect("next") { v.push(f); }
            v
        };
        assert_eq!(drain(), drain());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_unopenable_file_errors_from_every_backend() {
        let path = std::env::temp_dir().join(format!("h2b_bad_{}.mp4", std::process::id()));
        std::fs::write(&path, b"not a container").expect("write");
        assert!(open_video(&path, Backend::Auto, None, FitMode::Contain, Filter::Lanczos, None).is_err());
        let _ = std::fs::remove_file(&path);
    }

    /// `rust_h264` decodes CAVLC streams to visibly wrong pixels while
    /// returning no error (measured: mean abs channel diff up to ~27 luma,
    /// ~49 chroma). So CAVLC must never reach it, even when the user asks
    /// for the pure-Rust backend by name.
    #[test]
    fn cavlc_never_reaches_the_pure_rust_decoder() {
        // -coder 0 forces CAVLC; testsrc2 gives a deterministic stream.
        let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(
            "cavlc", &["-coder", "0", "-bf", "0"], 1, 32, 32, 5,
        ) else { return };

        // Auto must silently route it to ffmpeg and still decode correctly.
        let s = open_video(&path, Backend::Auto, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("auto must find a correct backend for CAVLC");
        let mut st = s.open().expect("open");
        assert!(st.next().expect("next").is_some(), "the CAVLC fallback must actually decode");

        // Asking for pure Rust by name must ERROR, not decode wrongly.
        //
        // Not `.expect_err(...)`: that requires the `Ok` side (`Box<dyn
        // FrameSource>`) to be `Debug`, which it is not (the trait has no
        // such bound, so a trait object can't blanket-impl it) -- the same
        // reason `scale.rs`'s `fps_stream_non_positive_target_fps_is_an_error`
        // matches on `FpsStream::new`'s `Result` instead of unwrapping it.
        let err = match open_video(&path, Backend::Rust, None, FitMode::Contain, Filter::Lanczos, None) {
            Err(e) => e,
            Ok(_) => panic!("explicit --backend rust must refuse a CAVLC stream"),
        };
        assert!(
            err.to_uppercase().contains("CAVLC"),
            "the refusal must name the reason so the user can act on it: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// A CABAC stream is what the pure-Rust decoder is correct on, so it must
    /// NOT be diverted -- otherwise the guard has quietly disabled backend B.
    #[test]
    fn cabac_still_reaches_the_pure_rust_decoder() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(
            "cabac", &["-coder", "1"], 1, 32, 32, 5,
        ) else { return };
        let s = open_video(&path, Backend::Rust, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("explicit --backend rust must accept a CABAC stream");
        let mut st = s.open().expect("open");
        assert!(st.next().expect("next").is_some());
        let _ = std::fs::remove_file(&path);
    }

    /// GUARD 2, not in the original brief: added because Task 5's fix-round-1
    /// report flagged BT.2020 (`matrix_coefficients` 9/10 falling back to
    /// BT.601) as a known, MEASURED gap -- 2.83-2.91 mean abs per-channel
    /// diff against ffmpeg on synthetic content, close enough to the 3.0
    /// "wrong" threshold that real saturated footage could silently exceed
    /// it. Treated exactly like CAVLC: `Auto` routes around it silently,
    /// `Backend::Rust` refuses it by name.
    #[test]
    fn bt2020_never_reaches_the_pure_rust_decoder() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(
            "bt2020",
            &["-coder", "1", "-colorspace", "bt2020nc", "-color_primaries", "bt2020",
              "-color_trc", "bt2020-10"],
            1, 32, 32, 5,
        ) else { return };

        let s = open_video(&path, Backend::Auto, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("auto must find a correct backend for a BT.2020-tagged stream");
        let mut st = s.open().expect("open");
        assert!(st.next().expect("next").is_some(), "the BT.2020 fallback must actually decode");

        // See the CAVLC test above for why this is a match, not `.expect_err`.
        let err = match open_video(&path, Backend::Rust, None, FitMode::Contain, Filter::Lanczos, None) {
            Err(e) => e,
            Ok(_) => panic!("explicit --backend rust must refuse a BT.2020-tagged stream"),
        };
        assert!(
            err.contains("BT.2020"),
            "the refusal must name the reason so the user can act on it: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    // --- `open_video_ensuring`: when ffmpeg's availability is consulted ---
    //
    // The bug these pin: `Backend::Auto` is the CLI DEFAULT, and an earlier
    // version confirmed ffmpeg was installed BEFORE calling `open_video` for
    // it. On a machine without ffmpeg, run headlessly (so consent downgrades
    // to `Never`), that refused every video by default -- including CABAC
    // H.264 files the pure-Rust backend decodes perfectly well by itself.
    //
    // Availability is simulated through the injected `ensure_ffmpeg` closure
    // rather than by scrubbing `PATH`, deliberately: `PATH` is process-wide
    // and `cargo test` runs these in parallel with tests that need a real
    // ffmpeg to generate their fixtures, so mutating it would make unrelated
    // tests fail nondeterministically. (Task 3 hit exactly this and deleted
    // its own `PATH`-scrubbing test for the same reason.) The real
    // `PATH`-stripped case is covered end-to-end by hand at the CLI level --
    // see this task's report.

    /// A closure standing in for an ffmpeg that is absent with consent
    /// denied, recording whether it was ever consulted.
    fn unavailable_ffmpeg(calls: &mut usize) -> impl FnMut() -> Result<(), String> + '_ {
        move || {
            *calls += 1;
            Err("ffmpeg was not found and downloading it was declined.".to_string())
        }
    }

    /// THE regression test for this fix: a CABAC file, ffmpeg unavailable,
    /// consent denied -- `Auto` must still render it, and must not so much as
    /// ask about ffmpeg.
    #[test]
    fn auto_decodes_a_cabac_file_without_ever_consulting_ffmpeg() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(
            "auto_no_ffmpeg", &["-coder", "1"], 1, 32, 32, 5,
        ) else { return };

        let mut calls = 0usize;
        let source = match open_video_ensuring(
            &path, Backend::Auto, None, FitMode::Contain, Filter::Lanczos, None,
            &mut unavailable_ffmpeg(&mut calls),
        ) {
            Ok(s) => s,
            Err(e) => panic!("auto must decode a CABAC file with no ffmpeg available: {e}"),
        };
        let mut stream = source.open().expect("open");
        assert!(stream.next().expect("next").is_some(), "it must actually decode");
        drop(stream);
        drop(source);
        assert_eq!(calls, 0, "ffmpeg's availability must never even be consulted here");
        let _ = std::fs::remove_file(&path);
    }

    /// The complementary case: a stream the guards refuse DOES need ffmpeg,
    /// so `Auto` must consult it -- exactly once, and only after the
    /// pure-Rust attempt has failed.
    #[test]
    fn auto_consults_ffmpeg_only_after_the_pure_rust_path_refuses() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(
            "auto_needs_ffmpeg", &["-coder", "0", "-bf", "0"], 1, 32, 32, 5,
        ) else { return };

        let mut calls = 0usize;
        let mut ensure = || {
            calls += 1;
            Ok(())
        };
        let source = match open_video_ensuring(
            &path, Backend::Auto, None, FitMode::Contain, Filter::Lanczos, None, &mut ensure,
        ) {
            Ok(s) => s,
            Err(e) => panic!("auto must fall back to ffmpeg for a CAVLC file: {e}"),
        };
        let mut stream = source.open().expect("open");
        assert!(stream.next().expect("next").is_some(), "the fallback must actually decode");
        drop(stream);
        drop(source);
        drop(ensure);
        assert_eq!(calls, 1, "ffmpeg must be consulted exactly once, on the fallback");
        let _ = std::fs::remove_file(&path);
    }

    /// Guards must not weaken when ffmpeg is missing: a CAVLC file with no
    /// ffmpeg available is an ERROR, never a silently wrong pure-Rust decode.
    /// The message must carry BOTH halves -- what is wrong with the file, and
    /// why the fallback could not run.
    #[test]
    fn a_guarded_stream_with_no_ffmpeg_errors_naming_both_reasons() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(
            "cavlc_no_ffmpeg", &["-coder", "0", "-bf", "0"], 1, 32, 32, 5,
        ) else { return };

        let mut calls = 0usize;
        let err = match open_video_ensuring(
            &path, Backend::Auto, None, FitMode::Contain, Filter::Lanczos, None,
            &mut unavailable_ffmpeg(&mut calls),
        ) {
            Err(e) => e,
            Ok(_) => panic!("a CAVLC stream with no ffmpeg must refuse, never decode wrongly"),
        };
        assert!(
            err.to_uppercase().contains("CAVLC"),
            "the pure-Rust refusal must survive into the message: {err}"
        );
        assert!(
            err.to_lowercase().contains("ffmpeg was not found"),
            "the fallback's own reason must be there too: {err}"
        );
        assert_eq!(calls, 1);
        let _ = std::fs::remove_file(&path);
    }

    /// **MINOR 4's regression test.** Each half's own refusal names the
    /// OTHER backend as its next step -- the pure-Rust refusal says "use
    /// --backend ffmpeg instead", and a real `video::ffmpeg::refusal()` (its
    /// actual wording is reproduced here rather than called directly: it is
    /// private to that module, and PATH cannot be scrubbed in this suite to
    /// provoke the real thing -- see the block comment above) says "run
    /// with --backend rust". Concatenated naively, those two sentences give
    /// directly OPPOSITE advice once both backends have already refused
    /// this exact file, so neither is actually actionable. The combined
    /// message must resolve that rather than just print it.
    #[test]
    fn combined_refusal_gives_coherent_advice_not_contradictory_suggestions() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(
            "cavlc_coherent", &["-coder", "0", "-bf", "0"], 1, 32, 32, 5,
        ) else { return };

        let mut simulated_ffmpeg_refusal = || {
            Err("ffmpeg was not found and downloading it was declined. Install ffmpeg \
                 yourself and make sure it is on PATH, or run with --backend rust to use the \
                 pure-Rust decode path instead."
                .to_string())
        };
        let err = match open_video_ensuring(
            &path, Backend::Auto, None, FitMode::Contain, Filter::Lanczos, None,
            &mut simulated_ffmpeg_refusal,
        ) {
            Err(e) => e,
            Ok(_) => panic!("a CAVLC stream with no ffmpeg must refuse"),
        };

        // Both raw suggestions are still present in the diagnostic detail --
        // this pins that the contradiction really is there in the raw text,
        // so the coherence check below isn't vacuously true.
        assert!(
            err.contains("use --backend ffmpeg instead"),
            "the pure-Rust half's own suggestion must still be visible: {err}"
        );
        assert!(
            err.contains("run with --backend rust"),
            "the ffmpeg half's own suggestion must still be visible: {err}"
        );

        // But the message must ALSO say, explicitly, that neither backend
        // suggestion above applies here, and name what actually helps.
        assert!(
            err.to_lowercase().contains("will not help"),
            "the combined message must say switching backends won't help: {err}"
        );
        assert!(
            err.to_lowercase().contains("install ffmpeg"),
            "the combined message must say what actually helps: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// An unreadable file is not an ffmpeg problem and must not be reported
    /// as one -- the real error surfaces, and ffmpeg is never consulted.
    #[test]
    fn an_unreadable_file_is_not_retried_under_ffmpeg() {
        let path = std::env::temp_dir()
            .join(format!("h2b_missing_{}_{}.mp4", std::process::id(), line!()));
        let _ = std::fs::remove_file(&path);

        let mut calls = 0usize;
        let err = match open_video_ensuring(
            &path, Backend::Auto, None, FitMode::Contain, Filter::Lanczos, None,
            &mut unavailable_ffmpeg(&mut calls),
        ) {
            Err(e) => e,
            Ok(_) => panic!("a missing file must not open"),
        };
        assert_eq!(calls, 0, "a missing file must not be blamed on ffmpeg");
        assert!(
            !err.to_lowercase().contains("was not found and downloading"),
            "the ffmpeg refusal must not be what the user sees here: {err}"
        );
    }

    /// An explicitly named backend keeps its eager behaviour: the user asked
    /// for ffmpeg, so a missing binary should fail fast rather than after a
    /// wasted decode attempt.
    #[test]
    fn explicit_ffmpeg_checks_availability_up_front() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(
            "explicit_ffmpeg", &["-coder", "1"], 1, 24, 24, 5,
        ) else { return };

        let mut calls = 0usize;
        let err = match open_video_ensuring(
            &path, Backend::Ffmpeg, None, FitMode::Contain, Filter::Lanczos, None,
            &mut unavailable_ffmpeg(&mut calls),
        ) {
            Err(e) => e,
            Ok(_) => panic!("--backend ffmpeg must fail when ffmpeg is unavailable"),
        };
        assert!(err.to_lowercase().contains("ffmpeg"), "got: {err}");
        assert_eq!(calls, 1, "explicit ffmpeg must check availability itself");
        let _ = std::fs::remove_file(&path);
    }

    /// `--backend rust` must never consult ffmpeg, on any file. The closure
    /// panics if called, so a regression here is unmissable.
    #[test]
    fn explicit_rust_never_consults_ffmpeg() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(
            "explicit_rust", &["-coder", "1"], 1, 24, 24, 5,
        ) else { return };
        let source = match open_video_ensuring(
            &path, Backend::Rust, None, FitMode::Contain, Filter::Lanczos, None,
            &mut || panic!("--backend rust must never consult ffmpeg"),
        ) {
            Ok(s) => s,
            Err(e) => panic!("explicit rust must accept a CABAC stream: {e}"),
        };
        assert!(source.open().expect("open").next().expect("next").is_some());
        let _ = std::fs::remove_file(&path);
    }

    /// ...including on a file it refuses: the refusal is the answer, and it
    /// must not turn into an ffmpeg question.
    #[test]
    fn explicit_rust_refuses_a_guarded_stream_without_consulting_ffmpeg() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(
            "explicit_rust_cavlc", &["-coder", "0", "-bf", "0"], 1, 24, 24, 5,
        ) else { return };
        let err = match open_video_ensuring(
            &path, Backend::Rust, None, FitMode::Contain, Filter::Lanczos, None,
            &mut || panic!("--backend rust must never consult ffmpeg"),
        ) {
            Err(e) => e,
            Ok(_) => panic!("--backend rust must refuse a CAVLC stream"),
        };
        assert!(err.to_uppercase().contains("CAVLC"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    /// The three failure classes, pinned directly rather than only through
    /// `open_video_ensuring`'s branching. `Container` matters most: an AVI is
    /// a declared input extension that this demuxer cannot parse but ffmpeg
    /// can, so classifying it as `Unreadable` would make `--backend auto`
    /// refuse every AVI outright.
    #[test]
    fn a_pure_rust_failure_is_classified_by_what_actually_went_wrong() {
        let missing = std::env::temp_dir()
            .join(format!("h2b_absent_{}_{}.mp4", std::process::id(), line!()));
        let _ = std::fs::remove_file(&missing);
        assert_eq!(classify_rust_failure(&missing), RustFailure::Unreadable);

        let garbage = std::env::temp_dir()
            .join(format!("h2b_garbage_{}_{}.mp4", std::process::id(), line!()));
        std::fs::write(&garbage, b"not a container at all").expect("write");
        assert_eq!(
            classify_rust_failure(&garbage),
            RustFailure::Container,
            "a readable file whose container will not parse is not 'unreadable'"
        );
        let _ = std::fs::remove_file(&garbage);

        // A real, readable, parseable container that a GUARD refuses.
        let Some(cavlc) = crate::video::ffmpeg::tests::sample_clip_args(
            "classify_cavlc", &["-coder", "0", "-bf", "0"], 1, 16, 16, 5,
        ) else { return };
        assert_eq!(
            classify_rust_failure(&cavlc),
            RustFailure::Guard,
            "the container parses fine here -- only a guard refused it"
        );
        let _ = std::fs::remove_file(&cavlc);
    }

    /// The complementary case: a CABAC stream carrying an ORDINARY (BT.709)
    /// matrix tag must not be caught by the BT.2020 guard -- otherwise the
    /// guard has quietly disabled the pure-Rust backend for every tagged
    /// clip, not just BT.2020 ones.
    #[test]
    fn bt709_is_not_mistaken_for_bt2020() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(
            "notbt2020",
            &["-coder", "1", "-colorspace", "bt709", "-color_primaries", "bt709",
              "-color_trc", "bt709"],
            1, 32, 32, 5,
        ) else { return };
        let s = open_video(&path, Backend::Rust, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("explicit --backend rust must accept a BT.709-tagged CABAC stream");
        let mut st = s.open().expect("open");
        assert!(st.next().expect("next").is_some());
        let _ = std::fs::remove_file(&path);
    }
}
