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

/// Keep a spawned helper process from flashing a console window on Windows.
///
/// Every raw `ffprobe` spawn in this crate goes through here. Without
/// `CREATE_NO_WINDOW`, a console-subsystem child launched from the GUI
/// (built `windows_subsystem = "windows"`, so it has no console of its own)
/// gets one allocated for it, flashing a black box open and shut.
/// `FfmpegCommand` needs no such call -- `ffmpeg-sidecar` sets the same flag
/// itself -- so only the raw `std::process::Command` spawns need this.
///
/// A no-op off Windows, where the flag does not exist.
pub fn hide_console(cmd: &mut std::process::Command) -> &mut std::process::Command {
    #[cfg(windows)]
    {
        // winbase.h's CREATE_NO_WINDOW, spelled out rather than pulling in a
        // winapi crate for one constant.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        std::os::windows::process::CommandExt::creation_flags(cmd, CREATE_NO_WINDOW);
    }
    cmd
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
     make sure it is on PATH, or run with --backend builtin to use the builtin decode \
     path instead."
        .to_string()
}

/// A crate-owned mirror of `ffmpeg_sidecar`'s download progress events.
///
/// Deliberately NOT the sidecar's own `FfmpegDownloadProgressEvent`: keeping a
/// type of our own here means the sidecar enum never leaks into this crate's
/// signatures (the GUI's shared progress cell and the CLI's stderr printer both
/// speak this), so a sidecar bump that reshapes its event type is a one-line
/// change in [`translate_progress`] rather than a ripple through the panes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfmpegDownloadProgress {
    /// The fetch has begun but no bytes have arrived yet (connecting).
    Starting,
    /// Bytes are arriving. `total` is 0 when the server sent no
    /// `Content-Length`, in which case a percentage is a fabrication -- see
    /// [`download_fraction`].
    Downloading { done: u64, total: u64 },
    /// The archive is downloaded and is being unpacked to its final location.
    Unpacking,
    /// ffmpeg is installed and ready.
    Done,
}

/// Translate one `ffmpeg_sidecar` download event into this crate's own
/// [`FfmpegDownloadProgress`]. The one place the sidecar's enum is named, so a
/// version bump that reshapes it touches only here.
fn translate_progress(
    event: ffmpeg_sidecar::download::FfmpegDownloadProgressEvent,
) -> FfmpegDownloadProgress {
    use ffmpeg_sidecar::download::FfmpegDownloadProgressEvent as E;
    match event {
        E::Starting => FfmpegDownloadProgress::Starting,
        E::Downloading { total_bytes, downloaded_bytes } => FfmpegDownloadProgress::Downloading {
            done: downloaded_bytes,
            total: total_bytes,
        },
        E::UnpackingArchive => FfmpegDownloadProgress::Unpacking,
        E::Done => FfmpegDownloadProgress::Done,
    }
}

/// The fraction downloaded so far, in `[0.0, 1.0]`, or `None` when `total` is 0
/// -- which is how the sidecar reports a server that sent no `Content-Length`.
/// A percentage of an unknown total would be a fabrication, so callers show an
/// indeterminate bar in that case rather than a made-up number.
pub fn download_fraction(done: u64, total: u64) -> Option<f32> {
    (total > 0).then(|| (done as f32 / total as f32).clamp(0.0, 1.0))
}

/// How long a download may go without a single new byte before it is treated
/// as stalled. The gyan.dev Windows mirror is notoriously slow and
/// rate-limited under load, and a dead connection there is indistinguishable
/// from a crawling one except by this silence -- so 30s of no bytes is
/// reported as a stall rather than as normal slowness.
pub const STALL_AFTER: std::time::Duration = std::time::Duration::from_secs(30);

/// Whether a download whose most recent byte arrived `since_last_byte` ago
/// should be reported as stalled. A pure function of the elapsed time so the
/// threshold decision is unit-testable without a real socket or a 30s sleep --
/// nothing here aborts anything, this only decides whether to show the "looks
/// stalled" message driven off the timestamp the callback records.
pub fn download_is_stalled(since_last_byte: std::time::Duration) -> bool {
    since_last_byte >= STALL_AFTER
}

/// Prints ffmpeg-download progress to stderr as a single carriage-return line.
///
/// This is what keeps `--yes` (and an answered `Ask`) from being a silent
/// multi-minute hang: the gyan.dev Windows build is slow, and without a
/// visible percentage a working-but-crawling download looks identical to a
/// dead one. `\r` keeps it to one rewritten line; the trailing spaces clear a
/// longer previous line, and `Done` ends it with a newline. A no-op-on-error
/// sink -- a failed write to stderr must never fail the install.
fn cli_stderr_progress(progress: FfmpegDownloadProgress) {
    let mut err = std::io::stderr();
    match progress {
        FfmpegDownloadProgress::Starting => {
            let _ = write!(err, "\rDownloading ffmpeg: connecting...          ");
        }
        FfmpegDownloadProgress::Downloading { done, total } => {
            let done_mb = done as f64 / 1_000_000.0;
            match download_fraction(done, total) {
                Some(frac) => {
                    let _ = write!(
                        err,
                        "\rDownloading ffmpeg: {:>3.0}% ({:.1}/{:.1} MB)   ",
                        frac * 100.0,
                        done_mb,
                        total as f64 / 1_000_000.0,
                    );
                }
                None => {
                    let _ = write!(err, "\rDownloading ffmpeg: {done_mb:.1} MB   ");
                }
            }
        }
        FfmpegDownloadProgress::Unpacking => {
            let _ = write!(err, "\rDownloading ffmpeg: unpacking...            ");
        }
        FfmpegDownloadProgress::Done => {
            let _ = writeln!(err, "\rDownloading ffmpeg: done.                   ");
        }
    }
    let _ = err.flush();
}

/// Makes sure an `ffmpeg` binary is available, per `consent`.
///
/// The CLI entry point: identical to [`ensure_ffmpeg_with_progress`] except
/// that progress goes to stderr as a carriage-return percentage line (see
/// [`cli_stderr_progress`]). Every existing caller -- the CLI's
/// `--yes`/`--no-download`/prompt path, and the GUI workers that call this with
/// [`DownloadConsent::Never`] purely to reuse one open path -- keeps working
/// unchanged; only the consenting branches actually reach the sink, so a
/// `Never` call prints nothing. The GUI's download modal calls
/// [`ensure_ffmpeg_with_progress`] directly instead, with a sink that drives a
/// real progress bar.
pub fn ensure_ffmpeg(consent: DownloadConsent) -> Result<(), String> {
    ensure_ffmpeg_with_progress(consent, cli_stderr_progress)
}

/// Makes sure an `ffmpeg` binary is available, per `consent`, reporting
/// download progress through `progress`.
///
/// Already-installed ffmpeg (found the same way [`ffmpeg_available`] finds
/// it -- `PATH` or next to the running executable) short-circuits every
/// variant of `consent` to `Ok`, including [`DownloadConsent::Never`]: an
/// existing install is not something to refuse, and `progress` is never called.
///
/// Otherwise: [`DownloadConsent::Always`] downloads immediately.
/// [`DownloadConsent::Ask`] prompts on stdin, unless stdin is not a terminal,
/// in which case it is downgraded to [`DownloadConsent::Never`] by
/// [`resolve_consent`] before it ever reaches a read. [`DownloadConsent::Never`]
/// always refuses with [`refusal`]'s message. Both consenting branches funnel
/// through [`download_ffmpeg`], so both report progress through the same sink.
pub fn ensure_ffmpeg_with_progress(
    consent: DownloadConsent,
    progress: impl Fn(FfmpegDownloadProgress),
) -> Result<(), String> {
    if ffmpeg_available() {
        return Ok(());
    }

    match resolve_consent(consent, std::io::stdin().is_terminal()) {
        DownloadConsent::Always => download_ffmpeg(progress),
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
                Ok(_) if answer.trim().eq_ignore_ascii_case("y") => download_ffmpeg(progress),
                _ => Err(refusal()),
            }
        }
    }
}

/// The one call in this file that actually reaches the network. Kept as its
/// own function so [`ensure_ffmpeg_with_progress`]'s two consenting branches
/// (`Always`, and `Ask` answered "y") share one code path and one error
/// message rather than two copies that could drift.
///
/// Uses `auto_download_with_progress` rather than the blocking, silent
/// `auto_download`: the download is a real 100 MB fetch from a slow mirror, and
/// a caller with no progress feed cannot tell a crawling download from a dead
/// one. Each `read` off the socket fires the sidecar's callback, which is
/// translated into a crate-owned [`FfmpegDownloadProgress`] before reaching
/// `progress` -- the sidecar's own event type never escapes this function.
fn download_ffmpeg(progress: impl Fn(FfmpegDownloadProgress)) -> Result<(), String> {
    ffmpeg_sidecar::download::auto_download_with_progress(move |event| {
        progress(translate_progress(event));
    })
    .map_err(|e| format!("failed to download ffmpeg: {e}"))
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
    /// The approximate frame count for a progress denominator, when the exact
    /// one above is unavailable. See [`FrameSource::frame_count_estimate`].
    frame_count_estimate: Option<usize>,
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

        // A denominator for the progress bar when the exact count above came
        // out `None` -- which is the COMMON case for Matroska, whose container
        // stores no frame count at all. `duration * out_fps` is then what the
        // render really produces (measured: within a frame on a 6s/30fps mkv
        // resampled to 10fps), and a bar a frame or two out is enormously
        // better than a spinner that looks frozen for minutes. Nothing
        // refuses or allocates off this -- see
        // `FrameSource::frame_count_estimate`'s doc for that line.
        //
        // **Not for a VFR source**, though, even though its duration is known
        // too. `out_fps` there is the AVERAGE rate, while `open`'s default
        // output timing conforms the stream to a fixed rate by DUPLICATING
        // frames -- so `duration * average` is systematically low, not
        // approximately right. Measured on the 6fps+24fps concat clip
        // `tests::sample_vfr_clip` builds: it estimates 27 against 50 frames
        // actually emitted, 46% short. A bar that fills at 54% of the way
        // through and then keeps growing is a worse lie than an honest
        // spinner that says it has no total, so this leaves VFR to the
        // spinner.
        let out_fps = fps.unwrap_or(probed.fps);
        let frame_count_estimate = frame_count_hint.or_else(|| {
            probed
                .duration
                .filter(|_| !probed.is_vfr && out_fps.is_finite() && out_fps > 0.0)
                .map(|d| (d * out_fps as f64).round().max(0.0) as usize)
        });

        Ok(Self {
            path: path.to_path_buf(),
            target,
            fit,
            filter,
            fps,
            width,
            height,
            out_fps,
            frame_count_hint,
            frame_count_estimate,
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

    fn frame_count_estimate(&self) -> Option<usize> {
        self.frame_count_estimate
    }

    fn open(&self) -> Result<Box<dyn FrameStream + '_>, String> {
        let mut cmd = FfmpegCommand::new();
        // Overrides the constructor's default `-loglevel level+info`: stderr
        // must carry only genuine errors, since `finish` below treats any
        // leftover stderr as a decode error. `hide_banner` matters too --
        // ffmpeg's startup banner prints at "info" severity regardless of
        // `-v error`, and only `-hide_banner` suppresses it.
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
        // Not `ffmpeg-sidecar`'s `.rawvideo()` helper: it hard-codes
        // `-pix_fmt rgb24`, and ffmpeg takes the last occurrence of an
        // output option, so appending `-pix_fmt rgba` first would not win.
        // rgba (not rgb24) is what lets `FitMode::Contain`'s letterbox
        // padding be genuinely transparent so it is culled downstream (see
        // `filtergraph`).
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
            scratch: Vec::new(),
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
    /// One frame's worth of bytes, reused by [`FrameStream::advance`] for the
    /// frames it only passes over. Allocated on first use so a stream that is
    /// never `advance`d past more than one frame at a time never pays for it.
    ///
    /// The bytes still have to come off the pipe -- ffmpeg has already
    /// produced them and there is no way to un-ask -- but a passed-over frame
    /// need not get its own fresh multi-megabyte allocation and `RgbaImage`
    /// on top. At 1080p that is 8.3 MB mapped, faulted in and dropped again
    /// per frame, two of every three on a 30fps→10fps render.
    scratch: Vec<u8>,
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

        // `open` asks ffmpeg for `-pix_fmt rgba`, so the pipe is already in
        // `RgbaImage`'s exact layout -- no per-pixel widening.
        let frame = RgbaImage::from_raw(self.width, self.height, buf)
            .ok_or_else(|| "decoded frame buffer had the wrong size".to_string())?;
        Ok(Some(frame))
    }

    /// Read `n` frames off the pipe, but allocate an image for only one.
    ///
    /// The bytes themselves cannot be skipped: ffmpeg has already decoded and
    /// written them, and the pipe has to be drained in order or the next read
    /// lands mid-frame. What can be skipped is everything `next` does around
    /// the read -- a fresh `frame_bytes`-sized allocation and an `RgbaImage`
    /// built over it -- for the `n - 1` frames being passed over.
    ///
    /// `n <= 1` delegates straight to `next`, because there is nothing to
    /// pass over and routing it through the scratch buffer would add a copy.
    /// The drain paths hand back whatever is in the scratch buffer, which is
    /// the last frame actually read -- exactly what the default body would
    /// have returned, and what `FpsStream` repeats when a source ends between
    /// two output times.
    fn advance(&mut self, n: usize) -> Result<(usize, Option<RgbaImage>), String> {
        if n <= 1 {
            let f = self.next()?;
            return Ok((usize::from(f.is_some()), f));
        }
        if self.scratch.len() != self.frame_bytes {
            self.scratch = vec![0u8; self.frame_bytes];
        }

        // The last frame read into `scratch` during THIS call, rebuilt as an
        // image only if the stream drains before the final `next` below.
        let held = |s: &Self| {
            RgbaImage::from_raw(s.width, s.height, s.scratch.clone())
                .ok_or_else(|| "decoded frame buffer had the wrong size".to_string())
        };

        let mut got = 0;
        for _ in 0..n - 1 {
            if self.done {
                return Ok((got, if got > 0 { Some(held(self)?) } else { None }));
            }
            // `read_frame_bytes` needs `&mut self`, so the buffer is moved out
            // and put straight back -- including on the error path, so a
            // subsequent call still finds a correctly sized scratch.
            let mut buf = std::mem::take(&mut self.scratch);
            let read = self.read_frame_bytes(&mut buf);
            self.scratch = buf;
            let filled = read?;

            if filled == 0 {
                // Same as `next`: this is the clean-end-of-stream check, and
                // `finish` still turns a non-zero exit or any stderr text
                // into a fatal error rather than a quiet end.
                self.finish()?;
                return Ok((got, if got > 0 { Some(held(self)?) } else { None }));
            }
            if filled != self.frame_bytes {
                self.done = true;
                return Err(format!(
                    "ffmpeg's output ended mid-frame: got {filled} of {} bytes for a {}x{} \
                     rgba frame",
                    self.frame_bytes, self.width, self.height
                ));
            }
            got += 1;
        }

        match self.next()? {
            Some(f) => Ok((got + 1, Some(f))),
            None => Ok((got, if got > 0 { Some(held(self)?) } else { None })),
        }
    }
}

impl Drop for FfmpegFrameStream {
    /// Best-effort cleanup for a stream dropped before it drained (e.g. a
    /// mid-render cancel). `kill()` alone only signals the child; `wait()`
    /// is what actually reaps it (a zombie on Unix otherwise), and cannot
    /// deadlock here since stderr is drained on its own thread from `open`.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
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
    let mut cmd = std::process::Command::new(&ffprobe_bin);
    // No console window: see `hide_console`.
    hide_console(&mut cmd);
    let output = cmd
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
/// standard signal that a stream is genuinely variable frame rate, chosen
/// over comparing `nb_frames` against `duration * avg_frame_rate` because
/// `avg_frame_rate` is itself commonly computed from those, making that
/// comparison close to a tautology. Compared as exact rationals
/// (cross-multiplied in [`parse_rational`]), not as floats, so there is no
/// tolerance to tune.
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
    /// machine without it -- but prints a clear notice, because a silently
    /// skipped test is a test that never runs.
    ///
    /// `pub(crate)` so the demux, builtin and backend tests can reuse it.
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
    /// concatenated with a 1s@24fps one, encoded with `-fps_mode vfr` so the
    /// container keeps the real, unequal inter-frame gaps instead of ffmpeg
    /// conforming them away at encode time -- `open`'s own decode-time
    /// conformance needs a source that genuinely varies to exercise. ffprobe
    /// reports `nb_frames=30` for this clip (6 + 24), but decoding it the way
    /// `FfmpegSource::open` does emits 50.
    ///
    /// Skips (returns `None`), never fails, when ffmpeg is not on `PATH`.
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

    /// Matroska stores no frame count, so `frame_count_hint` is honestly
    /// `None`. The duration is known, and for constant-rate content
    /// `duration * fps` is what the stream really emits, so the estimate has
    /// to be within a frame or two of that or it is not worth showing.
    #[test]
    fn an_mkv_has_no_hint_but_estimates_a_total_that_matches_what_it_emits() {
        let Some(path) = sample_mkv("ffmpeg_mkv_estimate", 2, 64, 48, 10) else { return };
        let src = FfmpegSource::probe(&path, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("probe");
        assert_eq!(
            src.info().frame_count_hint,
            None,
            "matroska states no frame count; if that changes this test is moot"
        );

        let mut stream = src.open().expect("open");
        let mut actual = 0usize;
        while stream.next().expect("next").is_some() {
            actual += 1;
        }

        let est = src.frame_count_estimate().expect("a known duration must yield an estimate");
        assert!(
            est.abs_diff(actual) <= 2,
            "the estimate ({est}) must be within a frame or two of the {actual} frames \
             actually emitted, or it is not a useful denominator"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// A VFR source gets no estimate, deliberately: its duration is known,
    /// but `out_fps` is the AVERAGE rate while `open`'s conformance
    /// duplicates frames up to a fixed one, so `duration * average` is
    /// systematically short of what actually gets emitted. A bar that fills
    /// partway and then keeps growing is a worse lie than an honest spinner.
    #[test]
    fn a_variable_frame_rate_clip_gets_no_estimate_rather_than_a_systematically_short_one() {
        let Some(path) = sample_vfr_clip("vfr_no_estimate") else { return };
        let src = FfmpegSource::probe(&path, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("probe");
        assert_eq!(src.info().frame_count_hint, None, "the VFR hint is None by design");
        assert_eq!(
            src.frame_count_estimate(),
            None,
            "an average-rate estimate is systematically short of what conformance emits"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// An `.mkv` of `secs` seconds at `fps`, or `None` when ffmpeg is missing.
    /// Matroska specifically: an `.mp4`'s sample table gives an exact frame
    /// count, so it cannot exercise the no-`nb_frames` case above.
    fn sample_mkv(name: &str, secs: u32, w: u32, h: u32, fps: u32) -> Option<std::path::PathBuf> {
        if !ffmpeg_available() {
            eprintln!("SKIPPING {name}: ffmpeg not on PATH");
            return None;
        }
        let path = std::env::temp_dir().join(format!("h2b_{name}_{}.mkv", std::process::id()));
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

    /// `frame_count_hint` must not blindly trust ffprobe's `nb_frames` when
    /// no `fps` filter is requested: for a genuinely variable-frame-rate
    /// source, `open`'s default output timing duplicates frames to conform
    /// the input to a fixed rate, so `nb_frames` (30 on this clip) understates
    /// what the pipe actually emits (50).
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

    /// [`FrameStream::advance`]'s override on this stream reads passed-over
    /// frames into a reused scratch buffer instead of giving each one its own
    /// allocation and `RgbaImage`. That must be invisible: same frames, same
    /// counts, same drain point as the default body's `n` calls to `next`.
    ///
    /// Every step size from 1 up is swept because the override has three
    /// distinct paths -- `n <= 1` delegates straight to `next`, a full run
    /// ends on `next`, and a run that hits the end of the pipe part-way has
    /// to hand back the last frame sitting in the scratch buffer.
    #[test]
    fn advance_matches_repeated_next() {
        let Some(path) = sample_clip("fadv", 1, 32, 24, 8) else { return };
        let src = FfmpegSource::probe(&path, None, FitMode::Contain, Filter::Lanczos, None)
            .expect("probe");
        for step in 1..=5 {
            let mut by_next = src.open().expect("open");
            let mut by_advance = src.open().expect("open");
            loop {
                let mut want_n = 0;
                let mut want_last = None;
                for _ in 0..step {
                    match by_next.next().expect("next") {
                        Some(f) => { want_last = Some(f); want_n += 1; }
                        None => break,
                    }
                }
                let (got_n, got_last) = by_advance.advance(step).expect("advance");
                assert_eq!(got_n, want_n, "step {step}: pulled count");
                assert_eq!(got_last, want_last, "step {step}: kept frame");
                if want_n == 0 { break }
            }
        }
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

    /// Every sidecar download event must map onto the crate-owned type, byte
    /// counts preserved -- this translation is the only place the sidecar's
    /// enum is named, so if it drifts nothing else in the crate would notice.
    #[test]
    fn every_sidecar_event_translates_to_the_crate_type() {
        use ffmpeg_sidecar::download::FfmpegDownloadProgressEvent as E;
        assert_eq!(translate_progress(E::Starting), FfmpegDownloadProgress::Starting);
        assert_eq!(
            translate_progress(E::Downloading { total_bytes: 101, downloaded_bytes: 42 }),
            FfmpegDownloadProgress::Downloading { done: 42, total: 101 },
            "downloaded_bytes -> done and total_bytes -> total, not swapped"
        );
        assert_eq!(translate_progress(E::UnpackingArchive), FfmpegDownloadProgress::Unpacking);
        assert_eq!(translate_progress(E::Done), FfmpegDownloadProgress::Done);
    }

    /// The percentage is honest: a real fraction while the total is known, and
    /// `None` -- not a fabricated 0% or a divide-by-zero -- when the server
    /// sent no `Content-Length` (the sidecar reports `total = 0` then).
    #[test]
    fn download_fraction_is_none_without_a_known_total() {
        assert_eq!(download_fraction(50, 0), None, "no Content-Length means no percentage");
        assert_eq!(download_fraction(0, 100), Some(0.0));
        assert_eq!(download_fraction(25, 100), Some(0.25));
        assert_eq!(download_fraction(100, 100), Some(1.0));
        // A server that over-reports (done past total) must clamp, never draw
        // a bar past its own end.
        assert_eq!(download_fraction(150, 100), Some(1.0), "clamped, never over 1.0");
    }

    /// The stall decision is purely the elapsed time crossing [`STALL_AFTER`]:
    /// a fresh byte is not stalled, and a long silence is -- no socket, no
    /// sleep, just the timestamp arithmetic the modal drives off.
    #[test]
    fn a_download_is_stalled_only_after_the_threshold_of_silence() {
        use std::time::Duration;
        assert!(!download_is_stalled(Duration::from_secs(0)), "a fresh byte is not a stall");
        assert!(
            !download_is_stalled(STALL_AFTER - Duration::from_millis(1)),
            "just under the threshold is not yet a stall"
        );
        assert!(download_is_stalled(STALL_AFTER), "exactly the threshold counts");
        assert!(
            download_is_stalled(STALL_AFTER + Duration::from_secs(60)),
            "a long silence is certainly a stall"
        );
    }

    /// The plumbing: `download_ffmpeg` must forward every event to its sink,
    /// translated. When ffmpeg is already installed the sidecar returns before
    /// emitting anything, so this only asserts what it CAN without a network --
    /// that a no-op sink type-checks through the whole chain, and that on an
    /// already-installed machine the sink is simply never called (no spurious
    /// events). The real event forwarding is exercised end to end by the
    /// GUI-side plumbing test and, if run, a live download.
    #[test]
    fn download_forwards_translated_events_to_its_sink() {
        use std::sync::Mutex;
        let seen: Mutex<Vec<FfmpegDownloadProgress>> = Mutex::new(Vec::new());
        // Only exercise the sink wiring when ffmpeg is already present, so this
        // never actually reaches out to the network in CI.
        if ffmpeg_available() {
            let _ = download_ffmpeg(|p| seen.lock().unwrap().push(p));
            assert!(
                seen.lock().unwrap().is_empty(),
                "an already-installed ffmpeg short-circuits before any event is emitted"
            );
        }
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
