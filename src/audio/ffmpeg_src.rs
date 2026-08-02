//! The ffmpeg-backed [`AudioSource`]: the audio sibling of
//! [`crate::video::ffmpeg`], and the same known-good decode path for
//! containers/codecs `symphonia` was not built with support for.
//!
//! Spawns the `ffmpeg` binary per [`FfmpegAudioSource::open`] and reads raw
//! `f32le` mono samples off its stdout pipe, with ffmpeg itself doing the
//! resample to [`TARGET_RATE`] via `-ar` -- the same division of labour as
//! the video path pushing scale/fps into its own `-vf` filtergraph rather
//! than doing it in-process.
//!
//! Native only: a browser cannot spawn a process. Gated at the `mod
//! ffmpeg_src;` declaration in `mod.rs`, mirroring `video::ffmpeg`'s own
//! gate.

use crate::audio::source::{AudioInfo, AudioSource, AudioStream};
use crate::audio::symphonia_src::TARGET_RATE;
use ffmpeg_sidecar::command::FfmpegCommand;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ChildStdout;
use std::thread::JoinHandle;

/// One block's worth of `f32` samples read per [`AudioStream::next`] call.
/// An implementation detail -- [`AudioStream`]'s contract only guarantees
/// the concatenation, never where the boundaries fall.
const BLOCK_SAMPLES: usize = 4096;
const BLOCK_BYTES: usize = BLOCK_SAMPLES * 4;

/// A re-openable [`AudioSource`] backed by a spawned `ffmpeg` process.
///
/// Holds only the path and the probed duration hint -- no process handle or
/// decoded state -- so `open` always spawns a brand new `ffmpeg`, exactly
/// what makes two opens of the same source agree sample for sample (the
/// same re-openability contract [`crate::video::ffmpeg::FfmpegSource`]
/// documents).
#[derive(Debug)]
pub struct FfmpegAudioSource {
    path: PathBuf,
    duration_hint: Option<f64>,
    /// Which audio stream to pull, as ffmpeg's `-map 0:a:<track>` index.
    ///
    /// 0 (the first audio track) unless `--audio-track` says otherwise.
    /// Dual-audio releases routinely carry the original language first and
    /// the dub second, so "the first track" is a container-ordering
    /// accident, not a choice.
    track: usize,
}

impl FfmpegAudioSource {
    /// Probes `path` with `ffprobe` and builds a re-openable source over it.
    ///
    /// Confirms an audio track actually exists (mirroring
    /// [`crate::audio::symphonia_src::SymphoniaSource::open_path`]'s own
    /// upfront check) rather than discovering that only after spawning
    /// `ffmpeg` for real -- a missing file, a non-media file, or a
    /// video-only container are all errors here, at open time.
    pub fn open_path(path: &Path) -> Result<Self, String> {
        let probed = probe_audio(path)?;
        Ok(Self { path: path.to_path_buf(), duration_hint: probed.duration, track: 0 })
    }

    /// Select which audio stream to decode (ffmpeg's `-map 0:a:<track>`).
    ///
    /// A builder rather than an `open_path` parameter so existing call sites
    /// and tests keep compiling unchanged.
    pub fn track(mut self, track: usize) -> Self {
        self.track = track;
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl AudioSource for FfmpegAudioSource {
    fn info(&self) -> AudioInfo {
        AudioInfo {
            // Always the ADAPTED rate, never whatever the source natively
            // was -- ffmpeg is asked for exactly this via `-ar` below, the
            // same contract `SymphoniaSource` upholds via its own resampler.
            sample_rate: TARGET_RATE,
            duration_hint: self.duration_hint,
        }
    }

    fn open(&self) -> Result<Box<dyn AudioStream + '_>, String> {
        let mut cmd = FfmpegCommand::new();
        // Mirrors `video::ffmpeg::FfmpegSource::open`'s `-hide_banner` +
        // `-v error`: stderr must carry only genuine errors, since `finish`
        // below treats any leftover stderr text as a decode error.
        cmd.hide_banner();
        cmd.args(["-v", "error"]);
        cmd.arg("-i").arg(&self.path);
        // Exactly the audio track, explicitly -- belt (`-map`) and
        // suspenders (`-vn`) against a container that also carries video,
        // the mirror of the video path's `-map 0:v:0` + `-an`.
        cmd.args(["-map", &format!("0:a:{}", self.track)]).arg("-vn");
        // Raw `f32le` mono at `TARGET_RATE`: ffmpeg does the resample and
        // the downmix itself, so nothing decoded off this pipe ever needs a
        // second pass to reach the shape everything downstream assumes.
        cmd.args(["-f", "f32le", "-ac", "1", "-ar", &TARGET_RATE.to_string(), "-"]);

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("failed to spawn ffmpeg for {}: {e}", self.path.display()))?;
        let stdout = child
            .take_stdout()
            .ok_or_else(|| "ffmpeg stdout was not piped".to_string())?;
        let stderr = child
            .take_stderr()
            .ok_or_else(|| "ffmpeg stderr was not piped".to_string())?;

        // Drained on a background thread so a chatty stderr can never fill
        // its OS pipe buffer and deadlock the process against our stdout
        // reads -- identical reasoning to `video::ffmpeg::FfmpegSource::open`.
        let stderr_thread = std::thread::spawn(move || {
            let mut stderr = stderr;
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text);
            text
        });

        Ok(Box::new(FfmpegAudioStream {
            child,
            reader: SampleReader::new(stdout),
            stderr_thread: Some(stderr_thread),
            finished: false,
        }))
    }
}

/// A one-shot cursor over one spawned `ffmpeg` process's raw `f32le` output.
struct FfmpegAudioStream {
    child: ffmpeg_sidecar::child::FfmpegChild,
    reader: SampleReader<ChildStdout>,
    stderr_thread: Option<JoinHandle<String>>,
    /// Whether the process has already been waited on (via [`Self::finish`]
    /// or [`Drop`]) -- distinct from [`SampleReader::done`], which only
    /// tracks whether reading has stopped.
    finished: bool,
}

impl FfmpegAudioStream {
    /// Called once the pipe reaches a clean end of stream. Waits for ffmpeg
    /// to exit and surfaces a non-zero status or any leftover stderr as
    /// `Err` rather than a quiet `Ok(None)` -- a crash or a decode error
    /// must not be read as "the track simply ended". Idempotent, so both
    /// `next` and [`Drop`] can call it safely.
    fn finish(&mut self) -> Result<Option<Vec<f32>>, String> {
        if self.finished {
            return Ok(None);
        }
        self.finished = true;
        let status = self
            .child
            .wait()
            .map_err(|e| format!("waiting for ffmpeg to exit: {e}"))?;
        let stderr = self
            .stderr_thread
            .take()
            .and_then(|h| h.join().ok())
            .unwrap_or_default();
        let stderr = stderr.trim();

        if !status.success() {
            return Err(if stderr.is_empty() {
                format!("ffmpeg exited with {status}")
            } else {
                format!("ffmpeg exited with {status}: {stderr}")
            });
        }
        if !stderr.is_empty() {
            return Err(format!("ffmpeg reported an error: {stderr}"));
        }
        Ok(None)
    }
}

impl AudioStream for FfmpegAudioStream {
    fn next(&mut self) -> Result<Option<Vec<f32>>, String> {
        match self.reader.next_block() {
            Ok(Some(block)) => Ok(Some(block)),
            Ok(None) => self.finish(),
            Err(e) => {
                self.finished = true;
                Err(e)
            }
        }
    }
}

impl Drop for FfmpegAudioStream {
    /// Best-effort cleanup for a stream dropped before it drained -- e.g. a
    /// caller that errors out elsewhere mid-render.
    ///
    /// `kill()` alone only SIGNALS the child -- `ffmpeg-sidecar`'s `kill`
    /// forwards straight to `std::process::Child::kill`, which does not
    /// reap. Without the `wait()` below the process is never collected: a
    /// zombie on Unix, and on Windows a race against the input file's handle
    /// being released, since nothing confirms the process actually exited.
    /// `wait()` cannot deadlock here: stderr is drained by its own thread
    /// from `open`, so there is no full pipe for the child to block on. The
    /// result is ignored because the process is being torn down and a
    /// failure to reap is not actionable. Identical reasoning to
    /// `video::ffmpeg::FfmpegFrameStream`'s `Drop`.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The byte-level reading and sample-validation logic, generic over the
/// underlying reader so it can be unit tested against an in-memory buffer
/// without ever spawning `ffmpeg` -- the same reasoning
/// `video::ffmpeg::resolve_consent` gives for being split out of
/// `ensure_ffmpeg` as its own pure-ish function.
struct SampleReader<R> {
    reader: R,
    /// Samples handed out so far, so a rejected sample can be named by its
    /// global index rather than an offset into whichever block contained it.
    frames_seen: u64,
    done: bool,
}

impl<R: Read> SampleReader<R> {
    fn new(reader: R) -> Self {
        Self { reader, frames_seen: 0, done: false }
    }

    /// Reads up to [`BLOCK_SAMPLES`] `f32` units from the underlying reader.
    ///
    /// Distinguishes a clean end of stream (nothing read at all) from a
    /// truncated final sample (some whole samples read, then a 1-3 byte
    /// remainder before the pipe closed) -- `read_fully` alone cannot make
    /// this distinction, and collapsing them would silently drop a partial
    /// trailing `f32` instead of erroring on it. Also rejects any non-finite
    /// sample, naming its global index -- the same rule
    /// `symphonia_src::SymphoniaStream` enforces, and for the same reason:
    /// `f32::NAN.min(1.0)` is `1.0`, so a single NaN launders into a whole
    /// bank of speakers at maximum volume with nothing having errored.
    fn next_block(&mut self) -> Result<Option<Vec<f32>>, String> {
        if self.done {
            return Ok(None);
        }
        let mut buf = vec![0u8; BLOCK_BYTES];
        let filled = read_fully(&mut self.reader, &mut buf)?;

        if filled == 0 {
            self.done = true;
            return Ok(None);
        }
        if filled % 4 != 0 {
            self.done = true;
            let whole = filled / 4;
            return Err(format!(
                "ffmpeg's output ended with {} leftover byte(s) after sample {} -- a partial \
                 trailing f32, not a clean end of stream",
                filled % 4,
                self.frames_seen + whole as u64
            ));
        }

        let samples = bytes_to_samples(&buf[..filled]);
        if let Some((at, value)) = first_non_finite(&samples, self.frames_seen) {
            self.done = true;
            return Err(format!(
                "ffmpeg decoded a non-finite sample ({value}) at sample {at} -- the file is \
                 corrupt or truncated"
            ));
        }
        self.frames_seen += samples.len() as u64;
        Ok(Some(samples))
    }
}

/// Reads from `reader` until `buf` is completely filled or the reader hits a
/// clean end of stream (`Ok(0)`), returning how much was actually filled.
/// `read` alone can return short of `buf.len()` even mid-stream (a pipe
/// filling slower than the request), so this loops rather than trusting a
/// single call -- identical shape to
/// `video::ffmpeg::FfmpegFrameStream::read_frame_bytes`.
fn read_fully<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<usize, String> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(format!("reading ffmpeg output: {e}")),
        }
    }
    Ok(filled)
}

/// Converts a byte slice -- already validated as a whole multiple of 4
/// bytes long -- into little-endian `f32` samples.
fn bytes_to_samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The global index and value of the first non-finite sample in `block`,
/// where `base` is `block[0]`'s index in the whole decoded stream. Split out
/// as a free function, same as `symphonia_src::first_non_finite`, so the
/// rejection rule can be tested directly against a NaN ffmpeg would only
/// ever produce on a corrupt file.
fn first_non_finite(block: &[f32], base: u64) -> Option<(u64, f32)> {
    block
        .iter()
        .position(|s| !s.is_finite())
        .map(|i| (base + i as u64, block[i]))
}

/// Metadata read from `ffprobe`, before this source is ever opened.
struct ProbedAudio {
    duration: Option<f64>,
}

/// Reads whether `path` has an audio track and, if the container can say so
/// without decoding, its duration.
///
/// A missing file or a file `ffprobe` cannot parse at all surfaces as the
/// command's own non-zero exit here -- the same way
/// `video::ffmpeg::probe_metadata` reports it -- so `FfmpegAudioSource::
/// open_path` never has to spawn a real decode just to discover the file
/// does not exist.
fn probe_audio(path: &Path) -> Result<ProbedAudio, String> {
    let ffprobe_bin = ffmpeg_sidecar::ffprobe::ffprobe_path();
    let mut cmd = std::process::Command::new(&ffprobe_bin);
    // No console window: see `crate::video::ffmpeg::hide_console`. This probe
    // is the one the GUI runs the moment a file is picked, so it is the spawn
    // a user is most likely to SEE flash.
    crate::video::ffmpeg::hide_console(&mut cmd);
    let output = cmd
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type:format=duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .map_err(|e| format!("failed to run ffprobe on {}: {e}", path.display()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(if stderr.trim().is_empty() {
            format!("ffprobe exited with {} for {}", output.status, path.display())
        } else {
            format!("ffprobe failed for {}: {}", path.display(), stderr.trim())
        });
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut has_audio = false;
    let mut duration = None;
    for line in stdout.lines() {
        let Some((key, value)) = line.split_once('=') else { continue };
        let value = value.trim();
        match key {
            // `-select_streams a:0` means this only ever appears for an
            // actual audio stream at index 0 -- its mere presence IS the
            // check, not just its value.
            "codec_type" => has_audio = true,
            "duration" => duration = value.parse::<f64>().ok(),
            _ => {}
        }
    }

    if !has_audio {
        return Err(format!("{} contains no audio track", path.display()));
    }

    Ok(ProbedAudio { duration })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::ffmpeg::ffmpeg_available;
    use std::io::Cursor;

    // --- Unit tests over `SampleReader`/`read_fully`/`first_non_finite`:
    // --- no `ffmpeg` process involved at all, so these run on every
    // --- machine regardless of whether `ffmpeg` is installed.

    #[test]
    fn read_fully_drains_an_in_memory_reader_completely() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut cursor = Cursor::new(data.clone());
        let mut buf = vec![0u8; 8];
        let n = read_fully(&mut cursor, &mut buf).expect("read");
        assert_eq!(n, 8);
        assert_eq!(buf, data);
    }

    #[test]
    fn read_fully_reports_a_short_read_at_true_eof() {
        let mut cursor = Cursor::new(vec![9u8, 9, 9]);
        let mut buf = vec![0u8; 8];
        let n = read_fully(&mut cursor, &mut buf).expect("read");
        assert_eq!(n, 3, "only 3 bytes ever existed");
    }

    #[test]
    fn bytes_to_samples_round_trips_known_values() {
        let values = [0.0f32, 1.0, -1.0, 0.5, std::f32::consts::PI];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(bytes_to_samples(&bytes), values);
    }

    #[test]
    fn first_non_finite_indexes_globally_not_within_the_block() {
        assert_eq!(first_non_finite(&[0.0, 1.0, -1.0], 0), None);
        let (at, value) = first_non_finite(&[0.0, f32::NAN], 5_000).expect("a NaN is non-finite");
        assert_eq!(at, 5_001, "the index must be global, not the offset within the block");
        assert!(value.is_nan());
        assert_eq!(
            first_non_finite(&[0.0, 0.0, f32::NEG_INFINITY], 10),
            Some((12, f32::NEG_INFINITY))
        );
    }

    /// **The partial-trailing-f32 regression test.** A reader that ends with
    /// 1-3 leftover bytes after some whole samples must be a hard error
    /// naming the sample index it was truncated at -- not a silently
    /// shortened block.
    #[test]
    fn a_partial_trailing_sample_is_rejected_naming_its_index() {
        let samples = vec![0.25f32; 10];
        let mut bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        bytes.truncate(bytes.len() - 3); // 10 whole samples minus 3 bytes -> 9 whole + 1 leftover byte

        // The whole 37-byte input is smaller than one `BLOCK_BYTES` read, so
        // `read_fully` drains it entirely (hitting true EOF) in this single
        // call -- the truncation is detected and reported right here, not on
        // a later call.
        let mut reader = SampleReader::new(Cursor::new(bytes));
        let err = reader.next_block().expect_err("a partial trailing f32 must be an error");
        assert!(err.contains('9'), "the error must name the sample index it was truncated at: {err}");

        // The reader must not pretend to keep going after a fatal error.
        assert!(reader.next_block().expect("stays done").is_none());
    }

    /// The complementary case: a byte count that is an exact multiple of 4,
    /// but shorter than a full block (a legitimate final partial block),
    /// must NOT be treated as truncation.
    #[test]
    fn a_short_but_whole_final_block_is_not_an_error() {
        let samples = vec![0.1f32, 0.2, 0.3];
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut reader = SampleReader::new(Cursor::new(bytes));
        let block = reader.next_block().expect("must not error").expect("some data");
        assert_eq!(block, samples);
        assert!(reader.next_block().expect("clean end").is_none());
    }

    /// A NaN (or infinity) must die at the reader, naming the sample index,
    /// the same contract `symphonia_src`'s decoder enforces -- see that
    /// module's doc for why: `f32::NAN.min(1.0)` is `1.0`, so an unrejected
    /// NaN launders into a whole speaker bank at maximum volume.
    #[test]
    fn a_non_finite_sample_is_rejected_naming_its_index() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut samples = vec![0.1f32; 20];
            samples[7] = bad;
            let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
            let mut reader = SampleReader::new(Cursor::new(bytes));
            let err = reader.next_block().expect_err("a non-finite sample must be an error");
            assert!(err.contains('7'), "the error must name the offending sample's index: {err}");
        }
    }

    #[test]
    fn a_missing_file_errors_rather_than_panicking() {
        if !ffmpeg_available() {
            return;
        }
        let err = FfmpegAudioSource::open_path(std::path::Path::new(
            "definitely_not_a_real_audio_file_12345.wav",
        ))
        .expect_err("must error");
        assert!(!err.is_empty());
    }

    // --- End-to-end tests: these spawn real `ffmpeg`/`ffprobe` processes,
    // --- and skip (never fail) when ffmpeg is not on PATH, the same
    // --- convention `video::ffmpeg::tests` uses.

    /// Generates a mono sine-tone WAV with `ffmpeg` itself, at an arbitrary
    /// sample rate, so this module's own resample-to-48k behaviour can be
    /// exercised against a source that is NOT already 48 kHz.
    fn sample_tone(name: &str, secs: f32, freq: u32, sr: u32) -> Option<PathBuf> {
        if !ffmpeg_available() {
            eprintln!("SKIPPING {name}: ffmpeg not on PATH");
            return None;
        }
        let path = std::env::temp_dir().join(format!("h2b_audio_ff_{name}_{}.wav", std::process::id()));
        let ok = std::process::Command::new("ffmpeg")
            .args([
                "-v", "error", "-y", "-f", "lavfi", "-i",
                &format!("sine=frequency={freq}:duration={secs}:sample_rate={sr}"),
            ])
            .arg(&path)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        ok.then_some(path)
    }

    /// A video-only clip (no audio stream at all) -- `probe_audio` must
    /// refuse this, naming the reason, rather than opening a stream that
    /// silently reads zero samples forever.
    fn sample_video_only(name: &str) -> Option<PathBuf> {
        if !ffmpeg_available() {
            eprintln!("SKIPPING {name}: ffmpeg not on PATH");
            return None;
        }
        let path = std::env::temp_dir().join(format!("h2b_audio_ff_{name}_{}.mp4", std::process::id()));
        let ok = std::process::Command::new("ffmpeg")
            .args([
                "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=32x32:rate=5:duration=1",
                "-an", "-pix_fmt", "yuv420p",
            ])
            .arg(&path)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        ok.then_some(path)
    }

    /// A clip with BOTH a video and an audio track -- exercises `-map
    /// 0:a:0` + `-vn` actually pulling just the audio out of a multiplexed
    /// container, the mirror of `video::ffmpeg`'s own `-map 0:v:0` + `-an`.
    fn sample_av(name: &str, secs: f32, freq: u32, sr: u32) -> Option<PathBuf> {
        if !ffmpeg_available() {
            eprintln!("SKIPPING {name}: ffmpeg not on PATH");
            return None;
        }
        let path = std::env::temp_dir().join(format!("h2b_audio_ff_{name}_{}.mp4", std::process::id()));
        let ok = std::process::Command::new("ffmpeg")
            .args([
                "-v", "error", "-y",
                "-f", "lavfi", "-i", &format!("testsrc2=size=32x32:rate=5:duration={secs}"),
                "-f", "lavfi", "-i", &format!("sine=frequency={freq}:duration={secs}:sample_rate={sr}"),
                "-shortest", "-pix_fmt", "yuv420p",
            ])
            .arg(&path)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        ok.then_some(path)
    }

    fn drain(src: &FfmpegAudioSource) -> Vec<f32> {
        let mut st = src.open().expect("open");
        let mut out = Vec::new();
        while let Some(b) = st.next().expect("next") {
            out.extend_from_slice(&b);
        }
        out
    }

    #[test]
    fn a_wav_decodes_to_mono_48k() {
        let Some(path) = sample_tone("mono48", 0.5, 1000, 48_000) else { return };
        let src = FfmpegAudioSource::open_path(&path).expect("open");
        assert_eq!(src.info().sample_rate, 48_000);
        let n = drain(&src).len();
        assert!((23_000..=25_000).contains(&n), "0.5s at 48kHz should be ~24000 samples, got {n}");
        let _ = std::fs::remove_file(&path);
    }

    /// A source at a different rate must be resampled to [`TARGET_RATE`],
    /// not passed through at its native rate. If `-ar 48000` were dropped
    /// from the ffmpeg invocation, a 44.1 kHz source would emit ~22050
    /// samples for 0.5s -- well outside this range -- so this test would
    /// catch that mutation directly, without needing to inspect the
    /// spawned command's arguments at all.
    #[test]
    fn a_44100_source_is_resampled_to_48k() {
        let Some(path) = sample_tone("mono44", 0.5, 1000, 44_100) else { return };
        let src = FfmpegAudioSource::open_path(&path).expect("open");
        assert_eq!(src.info().sample_rate, 48_000, "everything adapts to 48 kHz");
        let n = drain(&src).len();
        assert!(
            (23_000..=25_000).contains(&n),
            "0.5s resampled to 48kHz should be ~24000 samples, got {n}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn two_streams_over_one_source_agree() {
        let Some(path) = sample_tone("twice", 0.2, 440, 48_000) else { return };
        let src = FfmpegAudioSource::open_path(&path).expect("open");
        assert_eq!(drain(&src), drain(&src));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_video_only_file_errors_naming_the_missing_audio_track() {
        let Some(path) = sample_video_only("noaudio") else { return };
        let err = FfmpegAudioSource::open_path(&path).expect_err("must error: no audio track");
        assert!(
            err.to_lowercase().contains("audio"),
            "the error must say there is no audio track: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// `-map 0:a:0 -vn` must pull only the audio out of a multiplexed
    /// container -- dropping either flag risks video bytes reaching a pipe
    /// this module parses strictly as `f32le`, which would show up here as
    /// either a spawn/parse failure or a wildly wrong sample count.
    #[test]
    fn an_audio_video_file_decodes_just_the_audio_track() {
        let Some(path) = sample_av("avmix", 0.5, 1000, 48_000) else { return };
        let src = FfmpegAudioSource::open_path(&path).expect("open");
        let n = drain(&src).len();
        assert!(
            (23_000..=25_000).contains(&n),
            "0.5s of audio out of a muxed a/v file should be ~24000 samples, got {n}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_duration_hint_is_in_seconds() {
        let Some(path) = sample_tone("dur", 0.5, 1000, 48_000) else { return };
        let hint = FfmpegAudioSource::open_path(&path).expect("open").info().duration_hint;
        let hint = hint.unwrap_or_else(|| panic!("this container states a duration"));
        assert!((hint - 0.5).abs() < 0.05, "0.5s of audio should hint 0.5, got {hint}");
        let _ = std::fs::remove_file(&path);
    }

    /// **The `Drop` regression test.** Dropping a stream before it has
    /// drained must both `kill` AND reap (`wait`) the child. `kill()` alone
    /// only signals; on Windows the input file's handle is not guaranteed
    /// released until the process is actually reaped, so a `kill`-only
    /// `Drop` can race an immediate delete of the source file. A 3s clip
    /// (long enough that the process is still very much alive when dropped)
    /// makes that race exercised rather than won by a process that happened
    /// to exit before the check ran anyway.
    #[test]
    fn dropping_before_drain_releases_the_input_file() {
        let Some(path) = sample_tone("droprelease", 3.0, 220, 48_000) else { return };
        {
            let src = FfmpegAudioSource::open_path(&path).expect("open");
            let mut stream = src.open().expect("open stream");
            let _ = stream.next().expect("first block");
            // `stream` (and its child process) is dropped here, mid-decode.
        }
        std::fs::remove_file(&path)
            .expect("the input file must be releasable immediately after the stream is dropped");
    }

    /// The same NaN-rejection contract as `symphonia_src`'s decoder, proven
    /// through a REAL ffmpeg process this time (the unit tests above prove
    /// the logic; this proves the wiring). ffmpeg's own `-af volume` filter
    /// can be pushed to produce a `-inf` sample is unreliable across
    /// builds, so this instead confirms the wiring the cheap way: a real
    /// decode of ordinary audio produces no non-finite rejection at all,
    /// i.e. the check does not false-positive on real ffmpeg output.
    #[test]
    fn real_ffmpeg_output_never_false_positives_the_non_finite_check() {
        let Some(path) = sample_tone("nofalsepositive", 0.3, 300, 48_000) else { return };
        let src = FfmpegAudioSource::open_path(&path).expect("open");
        let mut st = src.open().expect("open stream");
        loop {
            match st.next() {
                Ok(Some(_)) => continue,
                Ok(None) => break,
                Err(e) => panic!("real ffmpeg output must never trip the non-finite check: {e}"),
            }
        }
        let _ = std::fs::remove_file(&path);
    }
}
