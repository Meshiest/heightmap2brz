//! Backend selection: picks which [`AudioSource`] implementation opens an
//! audio file, and enforces the availability policy that keeps [`Auto`]
//! from surprising a user who never asked for ffmpeg.
//!
//! [`Auto`]: AudioBackend::Auto
//!
//! Deliberately smaller than [`crate::video::backend`]'s counterpart: there
//! is no guard here comparable to the video side's CAVLC/BT.2020 checks --
//! `symphonia` either decodes a file correctly or errors, it does not
//! silently produce wrong SAMPLES the way `rust_h264` silently produces
//! wrong PIXELS. So `Auto` here is a plain try-then-fallback, not a
//! route-around-a-known-bad-path.
//!
//! **On wasm there is no ffmpeg backend at all** (`audio::ffmpeg_src` is
//! `#[cfg(not(target_arch = "wasm32"))]`, mirroring `video::ffmpeg`'s own
//! gate). [`AudioBackend::Ffmpeg`] there fails with a clear message rather
//! than failing to compile.
//!
//! **`DownloadConsent` cannot simply be imported from
//! [`crate::video::ffmpeg`] here**, even though that is conceptually the
//! right type to reuse: that whole module is `#[cfg(not(target_arch =
//! "wasm32"))]` (it spawns a subprocess), so it does not exist to import
//! from when this crate is built for wasm -- and [`open_audio`] must
//! compile there regardless. On native, [`DownloadConsent`] here IS
//! literally `video::ffmpeg`'s type (a `pub use`), so a caller wiring one
//! CLI flag to both `video::open_video_ensuring` and this module's
//! [`open_audio`] passes the same value to both with no conversion. On
//! wasm it is a local mirror with identical variants, existing only so this
//! module's signature does not change per target -- nothing on wasm ever
//! has a real ffmpeg decision to make with it, since there is no ffmpeg
//! backend there to ask.
//!
//! [`ensure_ffmpeg`] is injected into the private [`open_audio_with`]
//! rather than called directly, for the same reason
//! `video::backend::open_video_ensuring` injects it: so a test can simulate
//! "ffmpeg is unavailable and consent was denied" deterministically,
//! without mutating the process-wide `PATH` a parallel test suite shares.

use crate::audio::source::AudioSource;
use crate::audio::symphonia_src::SymphoniaSource;
use std::path::Path;

/// Reused as-is from [`crate::video::ffmpeg`] on native, so one CLI flag
/// drives both the video and audio backends with no conversion between two
/// separate types. See this module's doc for why wasm cannot do the same.
#[cfg(not(target_arch = "wasm32"))]
pub use crate::video::ffmpeg::DownloadConsent;

/// wasm's local mirror of [`crate::video::ffmpeg::DownloadConsent`] -- see
/// this module's doc for why it has to be a separate type here rather than
/// an import.
#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownloadConsent {
    Ask,
    Always,
    Never,
}

/// Which decode backend handles an audio file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AudioBackend {
    /// Try the builtin (`symphonia`) path first; fall back to ffmpeg only if
    /// it is already available or consent permits fetching it. If ffmpeg is
    /// unavailable, the ORIGINAL symphonia error is what the user sees --
    /// their file is the actual problem, not ffmpeg.
    #[default]
    Auto,
    /// The pure-Rust `symphonia` path only. Its error propagates unchanged;
    /// never consults ffmpeg.
    Builtin,
    /// The ffmpeg subprocess path, unconditionally. Refuses cleanly on
    /// wasm, where there is no such backend at all.
    Ffmpeg,
}

impl std::str::FromStr for AudioBackend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "builtin" => Ok(Self::Builtin),
            "ffmpeg" => Ok(Self::Ffmpeg),
            other => Err(format!("unknown audio backend '{other}' (auto, builtin, ffmpeg)")),
        }
    }
}

/// Opens `path` with `backend`, resolving [`AudioBackend::Auto`] to a
/// concrete backend exactly once.
///
/// - [`AudioBackend::Builtin`]: [`SymphoniaSource`] only; its error
///   propagates unchanged, and ffmpeg is never consulted.
/// - [`AudioBackend::Ffmpeg`]: `ensure_ffmpeg(consent)` first, then the
///   ffmpeg source. Native only; refuses on wasm.
/// - [`AudioBackend::Auto`]: try `SymphoniaSource`; on failure, fall back to
///   ffmpeg only if it is already available or `consent` permits fetching
///   it. When neither holds, the symphonia error is returned exactly as it
///   was, never replaced by an ffmpeg complaint -- the user's actual
///   problem is the file.
pub fn open_audio(
    path: &Path,
    backend: AudioBackend,
    consent: DownloadConsent,
) -> Result<Box<dyn AudioSource>, String> {
    open_audio_track(path, backend, consent, 0)
}

/// [`open_audio`], selecting which audio stream to decode.
///
/// `track` is an index among the container's AUDIO streams, so 0 is the first
/// audio track regardless of how many video or subtitle streams precede it.
/// Dual-audio releases commonly carry the original language first and the dub
/// second, so which one is "first" is a container-ordering accident.
///
/// Honoured by the ffmpeg backend (`-map 0:a:<track>`). The builtin
/// (symphonia) backend currently always decodes the container's default
/// track, so a non-zero `track` is reported rather than silently ignored --
/// under EVERY backend, including [`AudioBackend::Auto`], where it takes
/// symphonia out of the running entirely rather than letting the fallback
/// quietly hand back the wrong stream.
pub fn open_audio_track(
    path: &Path,
    backend: AudioBackend,
    consent: DownloadConsent,
    track: usize,
) -> Result<Box<dyn AudioSource>, String> {
    let mut ensure = default_ensure_ffmpeg;
    open_audio_with_track(path, backend, consent, track, &mut ensure)
}

/// [`open_audio_track`], with [`ensure_ffmpeg`]'s role played by a caller's
/// closure.
///
/// The public form of the injection this module already uses internally, and
/// the audio counterpart of [`crate::video::backend::open_video_ensuring`].
/// It exists for one caller shape: a GUI, which cannot answer
/// [`DownloadConsent::Ask`] because that prompt reads stdin and a window has
/// no terminal. Passing a closure that RECORDS the request and refuses lets
/// the UI thread discover that a download is needed, ask in a modal, and try
/// again -- without this, a GUI's only options are to download silently or to
/// never download at all.
///
/// The resolution order, the fallbacks and the error precedence are
/// [`open_audio_track`]'s exactly; only who answers the ffmpeg question
/// changes.
///
/// [`ensure_ffmpeg`]: crate::video::ffmpeg::ensure_ffmpeg
pub fn open_audio_ensuring(
    path: &Path,
    backend: AudioBackend,
    consent: DownloadConsent,
    track: usize,
    ensure_ffmpeg: &mut dyn FnMut(DownloadConsent) -> Result<(), String>,
) -> Result<Box<dyn AudioSource>, String> {
    open_audio_with_track(path, backend, consent, track, ensure_ffmpeg)
}

/// [`open_audio`], but with [`ensure_ffmpeg`]'s real implementation injected
/// as a closure rather than called directly -- see this module's doc for
/// why. `FnMut` rather than `Fn` so a test can count how many times it was
/// called (see e.g. `auto_consults_ffmpeg_only_once_after_symphonia_fails`).
///
/// [`ensure_ffmpeg`]: crate::video::ffmpeg::ensure_ffmpeg
fn open_audio_with(
    path: &Path,
    backend: AudioBackend,
    consent: DownloadConsent,
    ensure_ffmpeg: &mut dyn FnMut(DownloadConsent) -> Result<(), String>,
) -> Result<Box<dyn AudioSource>, String> {
    open_audio_with_track(path, backend, consent, 0, ensure_ffmpeg)
}

fn open_audio_with_track(
    path: &Path,
    backend: AudioBackend,
    consent: DownloadConsent,
    track: usize,
    ensure_ffmpeg: &mut dyn FnMut(DownloadConsent) -> Result<(), String>,
) -> Result<Box<dyn AudioSource>, String> {
    // symphonia has no track-selection plumbing here yet, so refuse rather
    // than silently decode the wrong language -- the exact failure mode
    // `--audio-track` exists to fix.
    if track != 0 && matches!(backend, AudioBackend::Builtin) {
        return Err(format!(
            "--audio-track {track} needs the ffmpeg backend; the builtin decoder always uses \
             the container's default audio track"
        ));
    }
    match backend {
        AudioBackend::Builtin => {
            SymphoniaSource::open_path(path).map(|s| Box::new(s) as Box<dyn AudioSource>)
        }
        AudioBackend::Ffmpeg => {
            ensure_ffmpeg(consent)?;
            open_ffmpeg_source_track(path, track)
        }
        // A non-zero `--audio-track` names something ONLY ffmpeg can do, so
        // symphonia is not a candidate on this path at all: not as the first
        // try, and not as the fallback either. Both of those used to reach
        // `SymphoniaSource::open_path`, which takes no track parameter and
        // decodes the container's default stream -- so `--audio-track 1` on a
        // dual-audio release rendered the FIRST language, successfully, with
        // no warning anywhere. The user got the wrong dub and a build that
        // looked perfect.
        //
        // Honoured where it can be, refused by name where it cannot. Refusing
        // is the same policy `Builtin` above already has, and it is what
        // `open_audio_track`'s doc has always claimed for every backend.
        AudioBackend::Auto if track != 0 => {
            ensure_ffmpeg(consent).map_err(|e| {
                format!(
                    "--audio-track {track} needs the ffmpeg backend (the builtin decoder \
                     always uses the container's default audio track), and ffmpeg is not \
                     available: {e}"
                )
            })?;
            open_ffmpeg_source_track(path, track)
        }
        // BUILTIN FIRST, and ffmpeg only when symphonia has actually failed.
        //
        // Deliberately NOT `video::backend`'s "prefer ffmpeg when present"
        // policy, and this module's own doc says why: that one routes around a
        // decoder known to produce silently WRONG pixels, where symphonia
        // either decodes correctly or errors. There is no wrong-output class to
        // route around here, so preferring ffmpeg would buy nothing and cost
        // three things -- a subprocess per open on the common path, the
        // documented error precedence below (it inverts: the user's undecodable
        // file starts reporting "ffprobe failed" instead of what symphonia
        // said), and a suite whose result depends on whether the developer has
        // ffmpeg on PATH. It was tried; all three happened.
        //
        // The codec coverage that motivated it is not lost: an Ogg/Opus file
        // still opens under `Auto`, via the fallback immediately below -- see
        // `auto_really_falls_back_to_ffmpeg_when_symphonia_cannot_decode_the_format`.
        AudioBackend::Auto => match SymphoniaSource::open_path(path) {
            Ok(source) => Ok(Box::new(source)),
            Err(symphonia_err) => {
                // `ensure_ffmpeg`'s real implementation already encodes
                // "already available OR consent permits fetching it" --
                // see `video::ffmpeg::ensure_ffmpeg`'s own doc. Its `Err`
                // here means neither held, so ffmpeg was never a genuine
                // option and the user's real problem is the one symphonia
                // already named.
                if ensure_ffmpeg(consent).is_err() {
                    return Err(symphonia_err);
                }
                open_ffmpeg_source_track(path, track).map_err(|_| symphonia_err)
            }
        },
    }
}

/// The real [`ensure_ffmpeg`](crate::video::ffmpeg::ensure_ffmpeg), or its
/// wasm stand-in -- the default `ensure_ffmpeg` argument [`open_audio`]
/// passes to [`open_audio_with`].
#[cfg(not(target_arch = "wasm32"))]
fn default_ensure_ffmpeg(consent: DownloadConsent) -> Result<(), String> {
    crate::video::ffmpeg::ensure_ffmpeg(consent)
}

/// On wasm there is nothing to ensure: no ffmpeg backend exists to fetch or
/// find, so this always refuses.
#[cfg(target_arch = "wasm32")]
fn default_ensure_ffmpeg(_consent: DownloadConsent) -> Result<(), String> {
    Err("no ffmpeg decode backend is available in this build (the browser build cannot spawn \
         a subprocess)"
        .to_string())
}

/// The ffmpeg backend, present only on targets that can spawn a subprocess.
#[cfg(not(target_arch = "wasm32"))]
fn open_ffmpeg_source_track(path: &Path, track: usize) -> Result<Box<dyn AudioSource>, String> {
    crate::audio::ffmpeg_src::FfmpegAudioSource::open_path(path)
        .map(|s| Box::new(s.track(track)) as Box<dyn AudioSource>)
}

#[cfg(target_arch = "wasm32")]
fn open_ffmpeg_source_track(path: &Path, _track: usize) -> Result<Box<dyn AudioSource>, String> {
    open_ffmpeg_source(path)
}

#[cfg(not(target_arch = "wasm32"))]
fn open_ffmpeg_source(path: &Path) -> Result<Box<dyn AudioSource>, String> {
    let source = crate::audio::ffmpeg_src::FfmpegAudioSource::open_path(path)?;
    Ok(Box::new(source))
}

/// On wasm there is no `audio::ffmpeg_src` module at all (it is
/// `#[cfg(not(target_arch = "wasm32"))]`, since it spawns a subprocess) --
/// so this errors with a clear message rather than failing to compile.
#[cfg(target_arch = "wasm32")]
fn open_ffmpeg_source(path: &Path) -> Result<Box<dyn AudioSource>, String> {
    Err(format!(
        "{}: no ffmpeg decode backend is available in this build (the browser build cannot \
         spawn a subprocess); use the builtin backend instead",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **This test used to assert nothing.** Its whole body was
    /// `assert_eq!(AudioBackend::default(), AudioBackend::Auto)` -- true, and
    /// silent about the preference its name claims. Meanwhile the code had
    /// grown an "ffmpeg first when it is on PATH" arm, so on a developer
    /// machine with ffmpeg installed the thing this is named for was not
    /// happening at all and nothing said so.
    ///
    /// The real assertion is that a decodable file opens under `Auto` with an
    /// `ensure_ffmpeg` that PANICS if it is called: `ensure_ffmpeg` is the only
    /// route from `Auto` to ffmpeg, so a shut door there is the preference.
    /// [`auto_reports_symphonias_error_when_ffmpeg_is_unavailable_and_consent_refuses`]
    /// is the other half of the pin -- it fails if an ffmpeg-first arm comes
    /// back, because the error precedence inverts with it.
    #[test]
    fn auto_prefers_the_builtin_backend_for_a_decodable_file() {
        assert_eq!(AudioBackend::default(), AudioBackend::Auto, "Auto is what a user gets");
        let path = decodable_wav("prefers_builtin");
        let source = match open_audio_with(&path, AudioBackend::Auto, DownloadConsent::Never, &mut |_| {
            panic!("Auto must reach for the builtin decoder first, not for ffmpeg")
        }) {
            Ok(s) => s,
            Err(e) => panic!("a real WAV must decode under Auto via symphonia alone: {e}"),
        };
        assert!(source.open().is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_missing_file_errors_under_every_backend() {
        let missing = std::path::Path::new("definitely_not_a_real_file.wav");
        for b in [AudioBackend::Auto, AudioBackend::Builtin] {
            assert!(open_audio(missing, b, DownloadConsent::Never).is_err());
        }
    }

    #[test]
    fn backend_parses_from_its_flag_spelling() {
        assert_eq!("auto".parse::<AudioBackend>().unwrap(), AudioBackend::Auto);
        assert_eq!("builtin".parse::<AudioBackend>().unwrap(), AudioBackend::Builtin);
        assert_eq!("ffmpeg".parse::<AudioBackend>().unwrap(), AudioBackend::Ffmpeg);
        assert!("rust".parse::<AudioBackend>().is_err(), "the old spelling is gone");
        assert!("nonsense".parse::<AudioBackend>().is_err());
    }

    // --- Tests added beyond the brief's three. A mutation campaign (see
    // --- this task's report) proved several real properties of `Auto`'s
    // --- fallback policy were completely unprotected by the brief's tests,
    // --- which never exercise the fallback branch at all (their one
    // --- decodable file never fails symphonia, and their one failing file
    // --- is a plain missing path, which errors identically under every
    // --- backend before the fallback logic is ever reached).

    fn garbage_file(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("h2b_audio_backend_{name}_{}.wav", std::process::id()));
        std::fs::write(&path, b"this is not audio at all").expect("write");
        path
    }

    /// A real, tiny, 16-bit PCM WAV -- symphonia handles this on its own, with
    /// no ffmpeg anywhere.
    fn decodable_wav(name: &str) -> std::path::PathBuf {
        let data: Vec<u8> = vec![0i16; 100].iter().flat_map(|s| s.to_le_bytes()).collect();
        let path =
            std::env::temp_dir().join(format!("h2b_audio_backend_{name}_{}.wav", std::process::id()));
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&48_000u32.to_le_bytes());
        out.extend_from_slice(&(48_000u32 * 2).to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&data);
        std::fs::write(&path, out).expect("write");
        path
    }

    /// **THE regression test for this fallback policy.** With ffmpeg
    /// simulated unavailable and consent refused, `Auto` on a file
    /// symphonia cannot decode must surface symphonia's OWN error, never an
    /// ffmpeg-shaped complaint -- the user's real problem is the file.
    #[test]
    fn auto_reports_symphonias_error_when_ffmpeg_is_unavailable_and_consent_refuses() {
        // Deliberately NOT named e.g. "no_ffmpeg" -- symphonia's own error
        // message embeds the file's path, and a fixture name containing the
        // substring "ffmpeg" would make the assertion below pass for the
        // wrong reason (a path collision, not an actually ffmpeg-free
        // error). Caught by running this test: it failed against exactly
        // that fixture name before this fix.
        let path = garbage_file("unavailable");
        // Not `.expect_err(...)`: that requires the `Ok` side (`Box<dyn
        // AudioSource>`) to be `Debug`, which it is not (the trait has no
        // such bound) -- the same reason `video::backend`'s tests match on
        // the `Result` instead of unwrapping it.
        let err = match open_audio_with(&path, AudioBackend::Auto, DownloadConsent::Never, &mut |_| {
            Err("ffmpeg was not found and downloading it was declined.".to_string())
        }) {
            Err(e) => e,
            Ok(_) => panic!("a garbage file must still error"),
        };
        assert!(
            !err.to_lowercase().contains("ffmpeg"),
            "the fallback's ffmpeg-shaped error must not replace symphonia's own: {err}"
        );
        // ...and it is not merely ffmpeg-free, it is symphonia's own message,
        // verbatim. The substring check above is not enough on its own: while
        // `Auto` had an ffmpeg-FIRST arm this test passed on a machine with
        // ffmpeg installed for entirely the wrong reason, returning
        // `probe_audio`'s "ffprobe failed for {path}: ..." -- which is an
        // ffmpeg complaint that happens not to contain the letters "ffmpeg".
        let symphonias_own = match SymphoniaSource::open_path(&path) {
            Err(e) => e,
            Ok(_) => panic!("the fixture must be undecodable for this test to mean anything"),
        };
        assert_eq!(
            err, symphonias_own,
            "Auto must hand back symphonia's error unchanged when ffmpeg was never a genuine \
             option -- the user's real problem is the file"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// `Auto` must consult `ensure_ffmpeg` exactly once, and only after
    /// symphonia has already failed -- never speculatively, and never twice.
    ///
    /// **This is the test that was red on every machine with ffmpeg on PATH**,
    /// and it was not flaky: `Auto` had an arm that took ffmpeg FIRST whenever
    /// `ffmpeg_present()`, which reaches ffmpeg without going through
    /// `ensure_ffmpeg` at all, so the count came back 0. A suite whose result
    /// depends on the developer's PATH is worth as much as no suite, and the
    /// arm is gone -- see `open_audio_with_track`.
    #[test]
    fn auto_consults_ffmpeg_only_once_after_symphonia_fails() {
        let path = garbage_file("consult_once");
        let mut calls = 0usize;
        let mut ensure = |_: DownloadConsent| {
            calls += 1;
            Err("simulated: ffmpeg unavailable".to_string())
        };
        let _ = open_audio_with(&path, AudioBackend::Auto, DownloadConsent::Never, &mut ensure);
        assert_eq!(calls, 1, "ensure_ffmpeg must be consulted exactly once on the fallback path");
        let _ = std::fs::remove_file(&path);
    }

    /// The complementary case: when symphonia already succeeds, `Auto` must
    /// never so much as ask about ffmpeg. The closure panics if called, so
    /// a regression here is unmissable -- mirrors
    /// `video::backend::tests::auto_decodes_a_cabac_file_without_ever_consulting_ffmpeg`.
    #[test]
    fn auto_never_consults_ffmpeg_when_symphonia_already_succeeds() {
        let path = decodable_wav("ok");
        let source = match open_audio_with(&path, AudioBackend::Auto, DownloadConsent::Never, &mut |_| {
            panic!("Auto must never consult ffmpeg when symphonia already succeeded")
        }) {
            Ok(s) => s,
            Err(e) => panic!("a real WAV must decode via symphonia alone: {e}"),
        };
        assert!(source.open().is_ok());
        let _ = std::fs::remove_file(&path);
    }

    // -- `--audio-track` under every backend --------------------------------

    /// **M2.** `--audio-track N` must never be silently dropped.
    ///
    /// The refusal used to name `Builtin` alone, and both of `Auto`'s paths
    /// reached `SymphoniaSource::open_path`, which takes no track parameter
    /// and decodes the container's default stream. So on the default backend
    /// with no ffmpeg available, `--audio-track 1` on a dual-audio release --
    /// the exact case the flag was added for -- rendered the FIRST language,
    /// returned `Ok`, and said nothing. The user got the wrong dub and a build
    /// that looked perfect.
    ///
    /// A DECODABLE fixture on purpose: with a garbage one every backend errors
    /// for its own reasons and the silent-success path is never taken.
    #[test]
    fn a_non_zero_track_is_never_silently_dropped_under_auto() {
        let path = decodable_wav("auto_track");
        let err = match open_audio_with_track(
            &path,
            AudioBackend::Auto,
            DownloadConsent::Never,
            1,
            &mut |_| Err("simulated: ffmpeg unavailable and declined".to_string()),
        ) {
            Err(e) => e,
            Ok(_) => panic!(
                "Auto with no ffmpeg available must refuse --audio-track 1, not decode the \
                 container's default track and call it success"
            ),
        };
        assert!(
            err.contains("--audio-track"),
            "the refusal must name the flag that could not be honoured: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The same flag under `Auto` when ffmpeg IS available: honoured, not
    /// refused. Refusing whenever the builtin decoder cannot do the job would
    /// be no better than ignoring it -- the point is that the flag works
    /// wherever it can and says so where it cannot.
    #[test]
    fn a_non_zero_track_is_honoured_under_auto_when_ffmpeg_is_available() {
        let Some(path) = sample_opus("auto_track_ffmpeg") else { return };
        // Track 0 of a single-track file, addressed explicitly: the point is
        // that `Auto` routed to ffmpeg rather than refusing, and a one-track
        // fixture is all this crate can make without a second encoder.
        let mut consulted = 0usize;
        let source = open_audio_with_track(
            &path,
            AudioBackend::Auto,
            DownloadConsent::Never,
            0,
            &mut |_| {
                consulted += 1;
                Ok(())
            },
        );
        assert!(source.is_ok(), "Auto must open an Opus file through its ffmpeg fallback");
        assert_eq!(consulted, 1, "and it must have gone through ensure_ffmpeg to get there");
        let _ = std::fs::remove_file(&path);
    }

    /// `Builtin` still refuses by name, unchanged -- the policy `Auto` above
    /// was brought into line with.
    #[test]
    fn a_non_zero_track_is_refused_by_name_under_builtin() {
        let path = decodable_wav("builtin_track");
        let err = match open_audio_with_track(
            &path,
            AudioBackend::Builtin,
            DownloadConsent::Never,
            2,
            &mut |_| panic!("Builtin must never consult ffmpeg"),
        ) {
            Err(e) => e,
            Ok(_) => panic!("Builtin cannot select a track and must say so"),
        };
        assert!(err.contains("--audio-track"), "the refusal must name the flag: {err}");
        let _ = std::fs::remove_file(&path);
    }

    /// `Builtin` must NEVER consult ffmpeg, on any file -- including one it
    /// refuses. The closure panics if called, so a silent fallback is
    /// unmissable.
    #[test]
    fn builtin_never_consults_ffmpeg_even_on_a_file_it_refuses() {
        let path = garbage_file("builtin_never");
        let err = match open_audio_with(&path, AudioBackend::Builtin, DownloadConsent::Never, &mut |_| {
            panic!("Builtin must never consult ffmpeg")
        }) {
            Err(e) => e,
            Ok(_) => panic!("a garbage file must error under Builtin"),
        };
        assert!(!err.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    /// An explicitly requested `Ffmpeg` backend must propagate
    /// `ensure_ffmpeg`'s own refusal verbatim rather than silently trying
    /// symphonia instead -- the user named the backend, so a missing binary
    /// is the whole answer.
    #[test]
    fn explicit_ffmpeg_propagates_ensures_refusal_without_trying_symphonia() {
        let path = garbage_file("explicit_ffmpeg_refuses");
        let err = match open_audio_with(&path, AudioBackend::Ffmpeg, DownloadConsent::Never, &mut |_| {
            Err("simulated: ffmpeg not found".to_string())
        }) {
            Err(e) => e,
            Ok(_) => panic!("must refuse"),
        };
        assert_eq!(err, "simulated: ffmpeg not found", "ensure_ffmpeg's own error must pass through unchanged");
        let _ = std::fs::remove_file(&path);
    }

    // --- End-to-end tests exercising the REAL ffmpeg fallback, not a
    // --- simulated one. Ogg/Opus is chosen deliberately: this crate's
    // --- `symphonia` feature list (mp3, aac, isomp4, mkv, flac, vorbis,
    // --- wav, pcm -- see Cargo.toml) has neither the `ogg` container nor
    // --- an `opus` decoder, so `SymphoniaSource::open_path` genuinely
    // --- cannot open one -- while a real installed ffmpeg (built with
    // --- `--enable-libopus`) decodes it without trouble. This is the one
    // --- real file format gap between the two backends available on this
    // --- project's dependency list, which is exactly what `Auto`'s
    // --- fallback and `Builtin`'s refusal need a genuine (not simulated)
    // --- case to prove against.

    fn ffmpeg_available() -> bool {
        crate::video::ffmpeg::ffmpeg_available()
    }

    fn sample_opus(name: &str) -> Option<std::path::PathBuf> {
        if !ffmpeg_available() {
            eprintln!("SKIPPING {name}: ffmpeg not on PATH");
            return None;
        }
        let path = std::env::temp_dir().join(format!("h2b_audio_backend_{name}_{}.ogg", std::process::id()));
        let ok = std::process::Command::new("ffmpeg")
            .args([
                "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3",
                "-c:a", "libopus",
            ])
            .arg(&path)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        ok.then_some(path)
    }

    #[test]
    fn symphonia_genuinely_cannot_open_the_opus_fixture() {
        // Non-vacuousness check for the two tests below: if this ever stops
        // being true (e.g. the `ogg`/`opus` symphonia features get added to
        // Cargo.toml), those tests would start passing for the wrong reason.
        let Some(path) = sample_opus("vacuous_check") else { return };
        assert!(
            SymphoniaSource::open_path(&path).is_err(),
            "this test suite assumes symphonia cannot open Ogg/Opus with this crate's feature list"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn auto_really_falls_back_to_ffmpeg_when_symphonia_cannot_decode_the_format() {
        let Some(path) = sample_opus("real_fallback") else { return };
        let source = open_audio(&path, AudioBackend::Auto, DownloadConsent::Never)
            .expect("Auto must fall back to a real ffmpeg for a format symphonia cannot open");
        let mut stream = source.open().expect("open stream");
        let mut n = 0usize;
        while let Some(b) = stream.next().expect("next") {
            n += b.len();
        }
        assert!(n > 0, "the fallback must actually decode real samples");
        let _ = std::fs::remove_file(&path);
    }

    /// **The mutation this pins directly:** if `Builtin` were changed to
    /// silently fall back to ffmpeg (one of this task's required
    /// mutations), THIS test would start passing when it must fail --
    /// asking for `Builtin` by name on a format only ffmpeg can decode must
    /// error, never quietly succeed via the other backend.
    #[test]
    fn builtin_refuses_the_opus_fixture_rather_than_silently_using_ffmpeg() {
        let Some(path) = sample_opus("builtin_refuses") else { return };
        let err = match open_audio(&path, AudioBackend::Builtin, DownloadConsent::Never) {
            Err(e) => e,
            Ok(_) => panic!("Builtin must refuse a format only ffmpeg can decode, not use it anyway"),
        };
        assert!(!err.is_empty());
        let _ = std::fs::remove_file(&path);
    }
}
