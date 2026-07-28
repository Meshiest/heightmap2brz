//! The ffmpeg-backed [`FrameSource`]: the known-good decode path.
//!
//! Spawns the `ffmpeg` binary per [`FfmpegSource::open`] and reads raw video
//! frames off its stdout pipe. Scaling and frame-rate resampling are pushed
//! into ffmpeg's own `-vf` filtergraph rather than done in-process, so the
//! bytes read off the pipe are already the target size and count -- see
//! `FfmpegSource::filtergraph`.
//!
//! Frames are piped as `rgba`, not the `rgb24` that `ffmpeg-sidecar`'s
//! `.rawvideo()` helper hard-codes, so that `FitMode::Contain`'s letterbox
//! padding can be genuinely transparent and get culled downstream exactly as
//! the in-process [`crate::video::scale::ResizeStream`] path's is. Keeping the
//! two backends pixel-comparable is what lets Task 5 use this one as an
//! oracle.

use crate::video::scale::{FitMode, Filter};
use crate::video::stream::{FrameSource, FrameStream, SourceInfo};
use ffmpeg_sidecar::command::FfmpegCommand;
use image::RgbaImage;
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ChildStdout;
use std::thread::JoinHandle;

/// Whether an `ffmpeg` binary can be found and executed right now.
///
/// Delegates to `ffmpeg-sidecar`'s own check, which looks in `PATH` and next
/// to the running executable -- the same place a download from a later task
/// would have placed a fetched copy.
pub fn ffmpeg_available() -> bool {
    ffmpeg_sidecar::command::ffmpeg_is_installed()
}

/// How much permission the caller has given [`ensure_ffmpeg`] to fetch and
/// run a downloaded `ffmpeg` binary.
///
/// Downloading and executing a binary pulled from the internet is not
/// something to do quietly. Every variant here resolves to either a real
/// "yes" from a person or a hard refusal -- never a silent fetch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownloadConsent {
    /// Prompt on stdin -- but only when stdin is actually a terminal. In a
    /// non-interactive context (a pipe, CI, or the GUI, which has no console
    /// to prompt on at all) this behaves exactly like [`DownloadConsent::Never`]
    /// instead of blocking forever on a read that will never return. Wired
    /// from the CLI's absence of both `--yes` and `--no-download`.
    Ask,
    /// Download without asking. Wired from an explicit `--yes` flag -- the
    /// caller already said yes, so there is nothing left to confirm.
    Always,
    /// Refuse outright, no prompt, no download. Wired from an explicit
    /// `--no-download` flag, and also what [`DownloadConsent::Ask`] becomes
    /// when stdin is not a terminal.
    Never,
}

/// Downgrades [`DownloadConsent::Ask`] to [`DownloadConsent::Never`] when
/// `stdin_is_terminal` is false, and passes every other variant through
/// unchanged.
///
/// Split out from [`ensure_ffmpeg`] as its own pure function -- taking the
/// terminal state as a plain `bool` rather than calling `std::io::stdin()`
/// itself -- so this decision can be unit tested directly instead of only
/// indirectly through whatever stdin happens to be wired to in a given test
/// run.
fn resolve_consent(consent: DownloadConsent, stdin_is_terminal: bool) -> DownloadConsent {
    match consent {
        DownloadConsent::Ask if !stdin_is_terminal => DownloadConsent::Never,
        other => other,
    }
}

/// The refusal every "no download happened" path returns. Names ffmpeg
/// explicitly and gives two concrete next steps, because "ffmpeg not found"
/// alone tells the user nothing they can act on.
fn refusal() -> String {
    "ffmpeg was not found and downloading it was declined. Install ffmpeg yourself and \
     make sure it is on PATH, or run with --backend rust to use the pure-Rust decode \
     path instead."
        .to_string()
}

/// Makes sure an `ffmpeg` binary is available, per `consent`.
///
/// Already-installed ffmpeg (found the same way [`ffmpeg_available`] finds
/// it -- `PATH` or next to the running executable) short-circuits every
/// variant of `consent` to `Ok`, including [`DownloadConsent::Never`]: an
/// existing install is not something to refuse.
///
/// Otherwise: [`DownloadConsent::Always`] downloads immediately via
/// `ffmpeg_sidecar::download::auto_download`. [`DownloadConsent::Ask`]
/// prompts on stdin, unless stdin is not a terminal, in which case it is
/// downgraded to [`DownloadConsent::Never`] by [`resolve_consent`] before it
/// ever reaches a read. [`DownloadConsent::Never`] always refuses with
/// [`refusal`]'s message.
pub fn ensure_ffmpeg(consent: DownloadConsent) -> Result<(), String> {
    if ffmpeg_available() {
        return Ok(());
    }

    match resolve_consent(consent, std::io::stdin().is_terminal()) {
        DownloadConsent::Always => download_ffmpeg(),
        DownloadConsent::Never => Err(refusal()),
        DownloadConsent::Ask => {
            // `resolve_consent` above has already ruled out a non-terminal
            // stdin, so this read cannot block forever on an input that will
            // never arrive.
            print!(
                "ffmpeg was not found. Download it now from the official ffmpeg build \
                 ffmpeg-sidecar fetches from ({})? [y/N] ",
                ffmpeg_sidecar::download::ffmpeg_download_url().unwrap_or("official mirror")
            );
            // A prompt with no flush is a prompt nobody sees until the
            // process exits -- stdout is line-buffered here only when it
            // happens to be a terminal, and this print has no trailing
            // newline to trigger that anyway.
            let _ = std::io::stdout().flush();

            let mut answer = String::new();
            match std::io::stdin().read_line(&mut answer) {
                Ok(_) if answer.trim().eq_ignore_ascii_case("y") => download_ffmpeg(),
                _ => Err(refusal()),
            }
        }
    }
}

/// The one call in this file that actually reaches the network. Kept as its
/// own function so [`ensure_ffmpeg`]'s two consenting branches
/// (`Always`, and `Ask` answered "y") share one code path and one error
/// message rather than two copies that could drift.
fn download_ffmpeg() -> Result<(), String> {
    ffmpeg_sidecar::download::auto_download().map_err(|e| format!("failed to download ffmpeg: {e}"))
}

/// A re-openable [`FrameSource`] backed by a spawned `ffmpeg` process.
///
/// Deliberately holds only the inputs to [`FfmpegSource::probe`] -- the path
/// and the scaling/fps settings -- and no process handle or decoded state.
/// `open` always spawns a brand new `ffmpeg`, which is what makes two opens of
/// the same source agree frame for frame (the re-openability contract
/// documented on [`FrameSource`] itself).
#[derive(Debug)]
pub struct FfmpegSource {
    path: PathBuf,
    target: Option<(u32, u32)>,
    fit: FitMode,
    filter: Filter,
    fps: Option<f32>,
    width: u32,
    height: u32,
    out_fps: f32,
    frame_count_hint: Option<usize>,
}

/// Metadata read from `ffprobe`, before folding in any target size/fps.
struct ProbedInfo {
    width: u32,
    height: u32,
    fps: f32,
    nb_frames: Option<usize>,
    duration: Option<f64>,
    /// Whether `avg_frame_rate` and `r_frame_rate` disagree -- see
    /// [`frame_rates_disagree`] for the detection method and why it was
    /// chosen. Computed once here, in `probe_metadata`, rather than kept as
    /// the two raw rational strings on this struct, because nothing past
    /// this point needs anything about them other than this one bit.
    is_vfr: bool,
}

impl FfmpegSource {
    /// Probes `path` with `ffprobe` and builds a re-openable source over it.
    ///
    /// `target`/`fit`/`filter` become an ffmpeg `scale`/`pad`/`crop` filter,
    /// and `fps` becomes an ffmpeg `fps` filter -- both applied by ffmpeg
    /// itself during decode (see [`FfmpegSource::open`]), not by a second
    /// in-process resize pass.
    pub fn probe(
        path: &Path,
        target: Option<(u32, u32)>,
        fit: FitMode,
        filter: Filter,
        fps: Option<f32>,
    ) -> Result<Self, String> {
        let probed = probe_metadata(path)?;

        let (width, height) = target.unwrap_or((probed.width, probed.height));
        if width == 0 || height == 0 {
            return Err(format!(
                "invalid target size {width}x{height} for {}",
                path.display()
            ));
        }

        // A target fps re-samples the stream, which changes the frame count,
        // so ffprobe's `nb_frames` (which describes the UNTOUCHED stream) can
        // only be trusted when no fps filter is applied. When one is, the
        // honest answer is duration * target fps -- and if duration itself
        // isn't known, `None`, never a guess. This is exact regardless of
        // whether the SOURCE is itself variable frame rate: an `fps` filter
        // forces ffmpeg to emit precisely `duration * target_fps` frames.
        //
        // When no `fps` filter is applied, `open` below still asks for no
        // `-fps_mode`, so ffmpeg's default output timing CONFORMS whatever
        // frame timing the container declares to a fixed rate -- and for a
        // genuinely variable-frame-rate source, that conformance step
        // DUPLICATES frames to fill each timestamp gap (see this module's
        // doc and `open`'s own comment on `-fps_mode` for why that duplicate
        // is deliberately kept, not passed through). The duplicate count is
        // not knowable without actually decoding, so `nb_frames` -- which
        // describes the untouched, pre-conformance stream -- can only be
        // trusted when the source is genuinely constant frame rate.
        // `probed.is_vfr` is exactly that check; see `frame_rates_disagree`
        // for the detection method.
        let frame_count_hint = match fps {
            Some(target_fps) if target_fps.is_finite() && target_fps > 0.0 => probed
                .duration
                .map(|d| (d * target_fps as f64).round().max(0.0) as usize),
            _ if probed.is_vfr => None,
            _ => probed.nb_frames,
        };

        Ok(Self {
            path: path.to_path_buf(),
            target,
            fit,
            filter,
            fps,
            width,
            height,
            out_fps: fps.unwrap_or(probed.fps),
            frame_count_hint,
        })
    }

    /// The `-vf` filtergraph for this source's scaling/fps settings, or
    /// `None` when neither is requested (a plain decode at native size/rate).
    fn filtergraph(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(fps) = self.fps {
            parts.push(format!("fps={fps}"));
        }
        if let Some((w, h)) = self.target {
            let flag = match self.filter {
                Filter::Lanczos => "lanczos",
                // ffmpeg's `scale` filter spells nearest-neighbor
                // "neighbor", not "nearest".
                Filter::Nearest => "neighbor",
            };
            match self.fit {
                FitMode::Exact => parts.push(format!("scale={w}:{h}:flags={flag}")),
                FitMode::Contain => {
                    // `black@0` -- fully TRANSPARENT, not opaque black --
                    // matching `scale::resize_frame`'s `Rgba([0, 0, 0, 0])`
                    // letterbox canvas exactly. Downstream, transparent
                    // pixels are culled and emit no brick, so letterboxing is
                    // free; opaque black padding would emit a wall of black
                    // display bricks and gates that the in-process
                    // `ResizeStream` emits none for. Measured on a 16:9
                    // source contained into 48x36, that is 432 of 1728 pixels
                    // -- 25% of the canvas -- and it would also swamp Task
                    // 5's cross-backend frame comparison with divergence that
                    // has nothing to do with decode correctness.
                    //
                    // This is precisely why `open` asks for `-pix_fmt rgba`
                    // rather than using the `.rawvideo()` helper's rgb24:
                    // rgb24 has no alpha channel to be transparent with.
                    parts.push(format!(
                        "scale={w}:{h}:force_original_aspect_ratio=decrease:flags={flag}"
                    ));
                    parts.push(format!("pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black@0"));
                }
                FitMode::Cover => {
                    parts.push(format!(
                        "scale={w}:{h}:force_original_aspect_ratio=increase:flags={flag}"
                    ));
                    parts.push(format!("crop={w}:{h}"));
                }
            }
        }
        (!parts.is_empty()).then(|| parts.join(","))
    }
}

impl FrameSource for FfmpegSource {
    fn info(&self) -> SourceInfo {
        SourceInfo {
            width: self.width,
            height: self.height,
            fps: self.out_fps,
            frame_count_hint: self.frame_count_hint,
        }
    }

    fn open(&self) -> Result<Box<dyn FrameStream + '_>, String> {
        let mut cmd = FfmpegCommand::new();
        // Overrides the constructor's default `-loglevel level+info`: stderr
        // must carry only genuine errors, since `FfmpegFrameStream::finish`
        // below treats any leftover stderr text as a decode error.
        //
        // `hide_banner` is not just cosmetic here: measured directly against
        // this ffmpeg build, the startup version/build banner prints to
        // stderr at "info" severity regardless of a later `-v error`
        // overriding the *effective* level for the rest of the run -- only
        // `-hide_banner` suppresses it. Without this, every successful decode
        // would surface that banner text as a false "ffmpeg reported an
        // error".
        cmd.hide_banner();
        cmd.args(["-v", "error"]);
        cmd.arg("-i").arg(&self.path);
        // Exactly the video track, explicitly -- belt (`-map`) and suspenders
        // (`-an`) against a container that also carries audio, which `-f
        // rawvideo` cannot mux alongside video on a single stdout pipe.
        cmd.args(["-map", "0:v:0"]).arg("-an");
        if let Some(vf) = self.filtergraph() {
            cmd.args(["-vf", &vf]);
        }
        // Deliberately NOT `ffmpeg-sidecar`'s `.rawvideo()` helper, which is
        // hard-coded to `-f rawvideo -pix_fmt rgb24 -`. Appending a
        // `-pix_fmt rgba` before it does not win: ffmpeg takes the LAST
        // occurrence of an output option, so the helper's rgb24 silently
        // overrides the override. Measured directly -- piping a 64x36 clip
        // through `-pix_fmt rgba -f rawvideo -pix_fmt rgb24` produced 34560
        // bytes for 5 frames (64*36*3*5, i.e. rgb24), not the 46080 rgba
        // would give. So the args are built explicitly here instead.
        //
        // rgba, not rgb24, because `FitMode::Contain`'s letterbox padding
        // must be genuinely transparent so it is culled downstream (see
        // `filtergraph`). ffmpeg converts yuv420p content to rgba with alpha
        // 255 for real pixels, so no per-pixel widening is needed on our side
        // at all.
        cmd.args(["-f", "rawvideo", "-pix_fmt", "rgba", "-"]);

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
        // reads. Joined in `finish` once stdout hits a frame boundary with
        // nothing left to read, so the full text is available before
        // deciding whether this was a clean end of stream or an ffmpeg error.
        let stderr_thread = std::thread::spawn(move || {
            let mut stderr = stderr;
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text);
            text
        });

        Ok(Box::new(FfmpegFrameStream {
            child,
            stdout,
            stderr_thread: Some(stderr_thread),
            frame_bytes: self.width as usize * self.height as usize * 4,
            width: self.width,
            height: self.height,
            done: false,
        }))
    }
}

/// A one-shot cursor over one spawned `ffmpeg` process's raw `rgba` output.
struct FfmpegFrameStream {
    child: ffmpeg_sidecar::child::FfmpegChild,
    stdout: ChildStdout,
    stderr_thread: Option<JoinHandle<String>>,
    frame_bytes: usize,
    width: u32,
    height: u32,
    done: bool,
}

impl FfmpegFrameStream {
    /// Reads exactly one frame's worth of bytes from the pipe, distinguishing
    /// a clean end of stream (nothing read at all) from a truncated final
    /// frame (some bytes read, then the pipe closed). `read_exact` alone
    /// cannot make this distinction -- both report the same `UnexpectedEof`
    /// -- and collapsing them would let a short read at the end of a pipe
    /// masquerade as an on-time end of stream instead of the fatal error it
    /// actually is.
    fn read_frame_bytes(&mut self, buf: &mut [u8]) -> Result<usize, String> {
        let mut filled = 0;
        while filled < buf.len() {
            match self.stdout.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(format!("reading ffmpeg output: {e}")),
            }
        }
        Ok(filled)
    }

    /// Called once the pipe reaches a frame boundary with nothing left to
    /// read. Waits for ffmpeg to exit and surfaces a non-zero status or any
    /// leftover stderr as `Err` rather than a quiet `Ok(None)` -- a crash or a
    /// decode error must not be read as "the clip simply ended".
    fn finish(&mut self) -> Result<Option<RgbaImage>, String> {
        self.done = true;
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

impl FrameStream for FfmpegFrameStream {
    fn next(&mut self) -> Result<Option<RgbaImage>, String> {
        if self.done {
            return Ok(None);
        }
        let mut buf = vec![0u8; self.frame_bytes];
        let filled = self.read_frame_bytes(&mut buf)?;

        if filled == 0 {
            return self.finish();
        }
        if filled != buf.len() {
            self.done = true;
            return Err(format!(
                "ffmpeg's output ended mid-frame: got {filled} of {} bytes for a {}x{} rgba frame",
                buf.len(),
                self.width,
                self.height
            ));
        }

        // `open` asks ffmpeg for `-pix_fmt rgba` directly, so the pipe is
        // already in `RgbaImage`'s exact layout -- no per-pixel widening, and
        // no chance of the shifted-colour-channel bug that assuming the wrong
        // format would cause silently. ffmpeg gives real content alpha 255
        // and `FitMode::Contain`'s pad filter gives letterboxing alpha 0,
        // which is the whole reason for using rgba over rgb24.
        let frame = RgbaImage::from_raw(self.width, self.height, buf)
            .ok_or_else(|| "decoded frame buffer had the wrong size".to_string())?;
        Ok(Some(frame))
    }
}

impl Drop for FfmpegFrameStream {
    /// Best-effort cleanup for a stream dropped before it drained -- e.g. a
    /// caller that errors out elsewhere mid-render. A no-op error (ignored)
    /// if the process already exited via `finish`.
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

/// Reads width, height, frame rate, frame count and duration with `ffprobe`.
///
/// The reported `fps` is derived from `avg_frame_rate` alone, never
/// `r_frame_rate` -- the latter is a guessed lowest-common-denominator base
/// that lies on variable-frame-rate sources. `r_frame_rate` IS read here too,
/// but only to detect that lie: compared against `avg_frame_rate` by
/// [`frame_rates_disagree`] to decide [`ProbedInfo::is_vfr`], never folded
/// into the `fps` value itself.
fn probe_metadata(path: &Path) -> Result<ProbedInfo, String> {
    let ffprobe_bin = ffmpeg_sidecar::ffprobe::ffprobe_path();
    let output = std::process::Command::new(&ffprobe_bin)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames:format=duration",
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
    let mut width = None;
    let mut height = None;
    let mut avg_frame_rate = None;
    let mut r_frame_rate = None;
    let mut nb_frames = None;
    let mut duration = None;
    for line in stdout.lines() {
        let Some((key, value)) = line.split_once('=') else { continue };
        let value = value.trim();
        match key {
            "width" => width = value.parse::<u32>().ok(),
            "height" => height = value.parse::<u32>().ok(),
            "avg_frame_rate" => avg_frame_rate = Some(value.to_string()),
            "r_frame_rate" => r_frame_rate = Some(value.to_string()),
            "nb_frames" => nb_frames = value.parse::<usize>().ok(),
            "duration" => duration = value.parse::<f64>().ok(),
            _ => {}
        }
    }

    let width =
        width.ok_or_else(|| format!("ffprobe found no video width for {}", path.display()))?;
    let height =
        height.ok_or_else(|| format!("ffprobe found no video height for {}", path.display()))?;
    let fps = avg_frame_rate
        .as_deref()
        .and_then(parse_frame_rate)
        .ok_or_else(|| format!("ffprobe found no usable frame rate for {}", path.display()))?;
    // `avg_frame_rate` parsed above (it must, or the `ok_or_else` already
    // returned) so re-borrow the raw string rather than re-probing.
    let is_vfr = match (avg_frame_rate.as_deref(), r_frame_rate.as_deref()) {
        (Some(avg), Some(r)) => frame_rates_disagree(avg, r),
        // No `r_frame_rate` at all to compare against -- cannot confirm this
        // is genuinely constant, so `frame_count_hint` must not trust
        // `nb_frames` here either.
        _ => true,
    };

    Ok(ProbedInfo { width, height, fps, nb_frames, duration, is_vfr })
}

/// Whether ffprobe's `avg_frame_rate` and `r_frame_rate` disagree -- the
/// standard signal that a stream is genuinely variable frame rate rather
/// than constant, chosen over the alternative (comparing `nb_frames` against
/// `duration * avg_frame_rate`) because `avg_frame_rate` is itself commonly
/// COMPUTED from `nb_frames`/`duration` by ffprobe, which would make that
/// comparison close to a tautology.
///
/// For genuinely constant frame rate content the two rationals coincide
/// EXACTLY -- confirmed here against real `ffprobe` output on a plain
/// integer rate (10/1 both) and a fractional NTSC-style rate (30000/1001
/// both). For variable frame rate content `r_frame_rate` reports a finer
/// time base than the true average, since it has to be able to represent
/// every distinct inter-frame gap the stream actually contains: measured
/// here on a clip built from a 1s/6fps segment concatenated with a
/// 1s/24fps one (`ffmpeg -f lavfi -i testsrc2=rate=6:duration=1 -f lavfi -i
/// testsrc2=rate=24:duration=1 -filter_complex concat=n=2:v=1:a=0 -fps_mode
/// vfr ...`), `ffprobe` reports `avg_frame_rate=40/3` against
/// `r_frame_rate=24/1` -- clearly disjoint, not rounding noise.
///
/// Compared as exact rationals (cross-multiplied in [`parse_rational`]), not
/// as floats, so no tolerance needs tuning and no floating-point rounding
/// can paper over a genuine mismatch or manufacture a fake one.
fn frame_rates_disagree(avg: &str, r: &str) -> bool {
    match (parse_rational(avg), parse_rational(r)) {
        (Some((an, ad)), Some((rn, rd))) => an * rd != rn * ad,
        // Either side didn't parse as `num/den` at all -- cannot confirm
        // this is genuinely CFR, so treat it the same as a disagreement:
        // `frame_count_hint`'s contract cares more about never being wrong
        // than about always having an answer (see `FfmpegSource::probe`).
        _ => true,
    }
}

/// Parses ffprobe's `"num/den"` rational into `(num, den)` as `i64`s,
/// without collapsing to a lossy `f32` first -- used only by
/// [`frame_rates_disagree`]'s exact cross-multiplication, which needs the
/// unrounded integers, not [`parse_frame_rate`]'s `f32` (that one is for the
/// reported `fps` value itself, a different use with different precision
/// needs). Rejects a zero denominator, same as `parse_frame_rate`.
fn parse_rational(s: &str) -> Option<(i64, i64)> {
    let (num, den) = s.split_once('/')?;
    let num: i64 = num.trim().parse().ok()?;
    let den: i64 = den.trim().parse().ok()?;
    (den != 0).then_some((num, den))
}

/// Parses ffprobe's `"num/den"` rational frame rate into an `f32`, rejecting
/// a zero denominator (ffprobe's own spelling of "unknown", e.g. `"0/0"`).
fn parse_frame_rate(s: &str) -> Option<f32> {
    let (num, den) = s.split_once('/')?;
    let num: f64 = num.trim().parse().ok()?;
    let den: f64 = den.trim().parse().ok()?;
    if den == 0.0 {
        return None;
    }
    let fps = (num / den) as f32;
    (fps.is_finite() && fps > 0.0).then_some(fps)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Generates a tiny clip with ffmpeg itself, then decodes it back. Skips
    /// (rather than fails) when ffmpeg is absent, so the suite still runs on a
    /// machine without it — but prints a clear notice, because a silently
    /// skipped test is a test that never runs.
    ///
    /// `pub(crate)` so the demux, rustvideo and backend tests can reuse it.
    pub(crate) fn sample_clip(name: &str, secs: u32, w: u32, h: u32, fps: u32) -> Option<std::path::PathBuf> {
        sample_clip_args(name, &[], secs, w, h, fps)
    }

    /// Like [`sample_clip`], but with `extra` encoder arguments spliced in
    /// just before the output path (e.g. `&["-coder", "0"]` to force CAVLC).
    /// `sample_clip` is implemented in terms of this so there is one
    /// generator, not two -- Task 6's CAVLC-guard tests need the extra-args
    /// form, and duplicating the generator would let the two drift apart.
    pub(crate) fn sample_clip_args(
        name: &str,
        extra: &[&str],
        secs: u32,
        w: u32,
        h: u32,
        fps: u32,
    ) -> Option<std::path::PathBuf> {
        if !ffmpeg_available() {
            eprintln!("SKIPPING {name}: ffmpeg not on PATH");
            return None;
        }
        let path = std::env::temp_dir().join(format!("h2b_{name}_{}.mp4", std::process::id()));
        let ok = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i",
                   &format!("testsrc2=size={w}x{h}:rate={fps}"),
                   "-t", &secs.to_string(), "-pix_fmt", "yuv420p"])
            .args(extra)
            .arg(&path)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        ok.then_some(path)
    }

    #[test]
    fn probing_reports_the_clips_real_shape() {
        let Some(path) = sample_clip("probe", 2, 64, 48, 10) else { return };
        let src = FfmpegSource::probe(&path, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("probe");
        let info = src.info();
        assert_eq!((info.width, info.height), (64, 48));
        assert!((info.fps - 10.0).abs() < 0.01, "fps was {}", info.fps);
        assert_eq!(info.frame_count_hint, Some(20), "2s at 10fps");
        let _ = std::fs::remove_file(&path);
    }

    /// Builds a genuinely variable-frame-rate clip: a 1s@6fps segment
    /// concatenated with a 1s@24fps one via `-filter_complex concat`,
    /// encoded with `-fps_mode vfr` so the CONTAINER keeps the real, unequal
    /// inter-frame gaps rather than ffmpeg conforming them away at ENCODE
    /// time -- this file's DECODE-time conformance (see `open`'s comment on
    /// `-fps_mode`, and `frame_count_hint`'s doc in `probe`) is what needs a
    /// source that genuinely varies to exercise. `-coder 1` matches every
    /// other generator in this file (CABAC), though entropy coding itself is
    /// irrelevant here.
    ///
    /// Measured against this exact clip: `ffprobe` reports `nb_frames=30`
    /// (6 + 24, correct for the untouched container), but decoding it the
    /// same way `FfmpegSource::open` does (no `-fps_mode`, so ffmpeg's
    /// default conforms to a fixed output rate) emits 50 frames -- the
    /// duplicate-frame gap-fill this task's fix has to account for.
    ///
    /// Skips (returns `None`), never fails, when ffmpeg is not on `PATH`,
    /// the same convention as `sample_clip`/`sample_clip_args`.
    fn sample_vfr_clip(name: &str) -> Option<std::path::PathBuf> {
        if !ffmpeg_available() {
            eprintln!("SKIPPING {name}: ffmpeg not on PATH");
            return None;
        }
        let path = std::env::temp_dir().join(format!("h2b_{name}_{}.mp4", std::process::id()));
        let ok = std::process::Command::new("ffmpeg")
            .args([
                "-v", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=6:duration=1",
                "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=24:duration=1",
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[outv]",
                "-map", "[outv]", "-fps_mode", "vfr", "-pix_fmt", "yuv420p", "-coder", "1",
            ])
            .arg(&path)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        ok.then_some(path)
    }

    /// **The regression test for Important 1/2.** Before this fix,
    /// `frame_count_hint` blindly trusted ffprobe's `nb_frames` whenever no
    /// `fps` filter was requested -- for a genuinely variable-frame-rate
    /// source, that is a lie: `open`'s default output timing duplicates
    /// frames to conform the variable input to a fixed rate. Measured on
    /// `sample_vfr_clip`'s exact clip: `nb_frames` says 30, the pipe emits
    /// 50. Confirmed this test FAILS against the pre-fix code (which always
    /// returned `probed.nb_frames` here) -- it asserted `Some(30)` while the
    /// stream actually produced 50 frames, i.e. exactly the "hint doesn't
    /// match what was emitted" violation this task exists to close (checked
    /// by hand against the prior `frame_count_hint` implementation before
    /// this fix landed; see this task's report).
    #[test]
    fn a_variable_frame_rate_clip_gets_an_honest_hint() {
        let Some(path) = sample_vfr_clip("vfr_hint") else { return };
        let src = FfmpegSource::probe(&path, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("probe");
        let hint = src.info().frame_count_hint;

        let mut stream = src.open().expect("open");
        let mut actual = 0usize;
        while stream.next().expect("next").is_some() {
            actual += 1;
        }
        assert!(actual > 0, "the clip must actually contain frames");

        // This exact clip is detected as VFR (avg_frame_rate 40/3 vs
        // r_frame_rate 24/1, measured via ffprobe -- see `sample_vfr_clip`'s
        // doc), so the fix must report `None` here, not a guessed `Some`.
        assert_eq!(
            hint, None,
            "a genuinely variable-frame-rate source cannot honestly report a Some(_) hint -- \
             the post-conformance frame count isn't knowable without decoding"
        );

        // Belt and suspenders: the general contract holds regardless of
        // which branch produced the hint -- a `Some(_)` must equal what was
        // actually emitted, never merely approximate it.
        if let Some(n) = hint {
            assert_eq!(n, actual, "a Some(_) hint must equal the frames actually emitted");
        }
        let _ = std::fs::remove_file(&path);
    }

    /// The complementary case: constant frame rate content (every OTHER clip
    /// generator in this file) must keep getting a real `Some(_)` hint, not
    /// regress to `None` just because VFR detection now exists.
    /// `probing_reports_the_clips_real_shape` above already pins this via
    /// its `Some(20)` assertion; this test names the property directly so a
    /// future change to the VFR heuristic that starts false-flagging CFR
    /// content fails loudly here, not just as a side effect of that older
    /// test.
    #[test]
    fn a_constant_frame_rate_clip_keeps_a_real_hint() {
        let Some(path) = sample_clip("cfr_hint", 1, 32, 24, 8) else { return };
        let src = FfmpegSource::probe(&path, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("probe");
        assert_eq!(
            src.info().frame_count_hint,
            Some(8),
            "1s at 8fps, genuinely constant -- VFR detection must not false-flag this"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The contract every consumer relies on: emitted frames match info().
    #[test]
    fn every_frame_matches_the_reported_dimensions() {
        let Some(path) = sample_clip("dims", 1, 32, 24, 10) else { return };
        let src = FfmpegSource::probe(&path, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("probe");
        let info = src.info();
        let mut stream = src.open().expect("open");
        let mut n = 0;
        while let Some(f) = stream.next().expect("next") {
            assert_eq!(f.dimensions(), (info.width, info.height), "frame {n}");
            n += 1;
        }
        assert_eq!(n, 10, "1s at 10fps");
        let _ = std::fs::remove_file(&path);
    }

    /// Re-openability, which text mode's two-pass banding scan depends on.
    #[test]
    fn two_opens_yield_identical_frames() {
        let Some(path) = sample_clip("reopen", 1, 16, 16, 5) else { return };
        let src = FfmpegSource::probe(&path, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("probe");
        let drain = || {
            let mut s = src.open().expect("open");
            let mut out = Vec::new();
            while let Some(f) = s.next().expect("next") { out.push(f); }
            out
        };
        let a = drain();
        let b = drain();
        assert_eq!(a.len(), 5);
        assert_eq!(a, b, "two opens must agree frame for frame");
        let _ = std::fs::remove_file(&path);
    }

    /// Scaling and fps go into ffmpeg's own filters, so the emitted frames
    /// must already be the target size and count.
    #[test]
    fn filters_resize_and_resample_during_decode() {
        let Some(path) = sample_clip("filters", 2, 64, 48, 20) else { return };
        let src = FfmpegSource::probe(&path, Some((32, 24)), FitMode::Exact, Filter::Nearest, Some(5.0))
            .expect("probe");
        let info = src.info();
        assert_eq!((info.width, info.height), (32, 24));
        let mut s = src.open().expect("open");
        let mut n = 0;
        while let Some(f) = s.next().expect("next") {
            assert_eq!(f.dimensions(), (32, 24));
            n += 1;
        }
        assert_eq!(n, 10, "2s at 5fps");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_missing_file_errors_rather_than_panicking() {
        if !ffmpeg_available() { return }
        let err = FfmpegSource::probe(
            std::path::Path::new("does_not_exist_12345.mp4"),
            None, FitMode::Contain, Filter::Lanczos, None,
        ).expect_err("must error");
        assert!(!err.is_empty());
    }

    /// With consent denied, a missing ffmpeg must produce an actionable error
    /// rather than hanging on a prompt or downloading anyway.
    #[test]
    fn refusing_the_download_errors_with_instructions() {
        if ffmpeg_available() {
            // Already installed: ensure() must succeed without prompting.
            ensure_ffmpeg(DownloadConsent::Never).expect("installed ffmpeg needs no download");
            return;
        }
        let err = ensure_ffmpeg(DownloadConsent::Never).expect_err("must refuse");
        assert!(err.to_lowercase().contains("ffmpeg"), "error must name ffmpeg: {err}");
        assert!(
            err.contains("--backend") || err.to_lowercase().contains("install"),
            "error must tell the user what to do: {err}"
        );
    }

    /// A headless run must never block on stdin.
    #[test]
    fn never_consent_does_not_prompt() {
        let start = std::time::Instant::now();
        let _ = ensure_ffmpeg(DownloadConsent::Never);
        assert!(start.elapsed().as_secs() < 5, "Never must not wait for input");
    }

    /// The terminal-detection decision, pinned directly rather than only
    /// indirectly through whatever `cargo test`'s real stdin happens to be.
    /// This is the rule the GUI and CI both rely on: `Ask` must never reach
    /// a blocking read when there is no person on the other end of stdin.
    #[test]
    fn ask_downgrades_to_never_without_a_terminal() {
        assert_eq!(
            resolve_consent(DownloadConsent::Ask, false),
            DownloadConsent::Never,
            "a non-terminal stdin must never be prompted"
        );
    }

    /// The complementary case: with a real terminal on stdin, `Ask` is left
    /// alone to actually prompt. Without this, `resolve_consent` could be
    /// (wrongly) simplified to always return `Never`.
    #[test]
    fn ask_stays_ask_with_a_terminal() {
        assert_eq!(resolve_consent(DownloadConsent::Ask, true), DownloadConsent::Ask);
    }

    /// `Always` and `Never` are explicit consent already; terminal state
    /// must not change either one.
    #[test]
    fn always_and_never_are_unaffected_by_terminal_state() {
        assert_eq!(resolve_consent(DownloadConsent::Always, false), DownloadConsent::Always);
        assert_eq!(resolve_consent(DownloadConsent::Always, true), DownloadConsent::Always);
        assert_eq!(resolve_consent(DownloadConsent::Never, false), DownloadConsent::Never);
        assert_eq!(resolve_consent(DownloadConsent::Never, true), DownloadConsent::Never);
    }

    /// Exercises the refusal message directly, rather than only through
    /// `ensure_ffmpeg`'s "ffmpeg absent" branch -- on a machine where ffmpeg
    /// IS installed (this one, per the task constraints), that branch never
    /// runs, so this is the only way the message's actual wording gets
    /// checked here. Same two assertions `refusing_the_download_errors_with_
    /// instructions` makes on whatever `ensure_ffmpeg` happens to return.
    #[test]
    fn refusal_message_names_ffmpeg_and_a_next_step() {
        let err = refusal();
        assert!(err.to_lowercase().contains("ffmpeg"), "error must name ffmpeg: {err}");
        assert!(
            err.contains("--backend") || err.to_lowercase().contains("install"),
            "error must tell the user what to do: {err}"
        );
    }

    /// `FitMode::Contain`'s letterbox padding must be fully TRANSPARENT, the
    /// same as `scale::resize_frame`'s `Rgba([0, 0, 0, 0])` canvas -- not
    /// opaque black.
    ///
    /// This is a save-content assertion, not a cosmetic one. Transparent
    /// pixels are culled downstream and emit no brick, so letterboxing is
    /// free; opaque black padding emits a wall of real display bricks and
    /// gates that the in-process `ResizeStream` emits none for. It also has to
    /// hold for Task 5 to use this backend as a decode oracle at all: a
    /// Contain clip whose padding disagreed between backends would swamp a
    /// frame-for-frame comparison with divergence unrelated to decode
    /// correctness.
    ///
    /// Deliberately asserts on the ALPHA CHANNEL rather than just the frame
    /// size -- the pre-existing `FitMode::Contain` test passes `target: None`,
    /// so no pad filter is ever built and this path went entirely unexercised.
    /// A 16:9 source into a 4:3 box is chosen so padding is guaranteed.
    #[test]
    fn contain_letterboxing_is_transparent_not_opaque_black() {
        // 64x36 (16:9) contained into 48x36 (4:3): scaled by
        // min(48/64, 36/36) = 0.75 to 48x27, leaving 9 rows (4 top, 5 bottom)
        // of padding -- 432 of 1728 pixels, 25% of the canvas.
        let Some(path) = sample_clip("contain_alpha", 1, 64, 36, 5) else { return };
        let src = FfmpegSource::probe(&path, Some((48, 36)), FitMode::Contain, Filter::Lanczos, None)
            .expect("probe");
        let mut s = src.open().expect("open");
        let frame = s.next().expect("next").expect("at least one frame");
        assert_eq!(frame.dimensions(), (48, 36));

        // The top and bottom rows are padding for this geometry.
        for x in 0..48 {
            assert_eq!(
                frame.get_pixel(x, 0).0[3], 0,
                "top letterbox row must be transparent at x={x}, not opaque black"
            );
            assert_eq!(
                frame.get_pixel(x, 35).0[3], 0,
                "bottom letterbox row must be transparent at x={x}, not opaque black"
            );
        }

        // The middle row is real content, which must stay fully opaque --
        // otherwise "transparent padding" could be satisfied by a frame that
        // is uniformly transparent, culling the whole render.
        for x in 0..48 {
            assert_eq!(
                frame.get_pixel(x, 18).0[3], 255,
                "content must stay opaque at x={x}"
            );
        }

        // Pin the exact split, so a future filtergraph change that pads with
        // the right alpha but the wrong geometry still fails here.
        let transparent = frame.pixels().filter(|p| p.0[3] == 0).count();
        assert_eq!(
            transparent, 432,
            "9 padding rows of 48px = 432 of 1728 pixels must be culled"
        );
        let _ = std::fs::remove_file(&path);
    }
}
