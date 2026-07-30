//! Pulling a subtitle track out of a video container and handing it to the
//! [`super::srt`] or [`super::ass`] parser.
//!
//! Native only: spawns `ffprobe` and `ffmpeg` subprocesses, so this module is
//! gated `#[cfg(not(target_arch = "wasm32"))]` at its `mod extract;`
//! declaration in `mod.rs`, the same way `video::ffmpeg` and
//! `audio::ffmpeg_src` are gated -- `ffmpeg-sidecar` cannot build for wasm at
//! all.
//!
//! **The PGS case is the point of this module.** `hdmv_pgs_subtitle`,
//! `dvd_subtitle` and `dvb_subtitle` are bitmap (image sequence) subtitle
//! formats -- there is no text in them to extract without OCR, which is out
//! of scope. [`extract`] must never return an empty [`Subtitles`] for one of
//! these: a silently empty track renders as a perfectly good video with no
//! dialogue, indistinguishable from a correct render of a scene where nobody
//! speaks. Instead [`extract`] fails loudly with [`bitmap_error`], which
//! names the codec, says why (it's an image format), and points at the
//! `--subtitles <file>` escape hatch (supplying an external text subtitle
//! file bypasses extraction entirely).

use super::{ass, srt, Subtitles};
use ffmpeg_sidecar::command::FfmpegCommand;
use std::io::Read;
use std::path::Path;

/// One subtitle stream found in a container by [`list_streams`].
///
/// `index` is the stream's position among the container's subtitle streams
/// only (0-based) -- i.e. exactly the `<n>` in ffmpeg's `-map 0:s:<n>`, and
/// exactly the `track` argument [`extract`] expects. Mirrors
/// `audio::backend::open_audio_track`'s doc: this is an index among streams
/// of one type, not the container's absolute stream index, so it does not
/// shift depending on how many video or audio streams happen to precede it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubtitleStream {
    pub index: usize,
    pub codec: String,
    pub language: Option<String>,
    pub title: Option<String>,
}

/// Lists every subtitle stream in `path`, via `ffprobe -select_streams s`.
///
/// A container with no subtitle streams at all is not an error -- it
/// succeeds with an empty `Vec`, mirroring `srt::parse`'s and `ass::parse`'s
/// own "no cues is valid" stance. A missing file or one `ffprobe` cannot open
/// at all surfaces as `ffprobe`'s own non-zero exit, the same way
/// `video::ffmpeg::probe_metadata` and `audio::ffmpeg_src::probe_audio`
/// report it.
pub fn list_streams(path: &Path) -> Result<Vec<SubtitleStream>, String> {
    let ffprobe_bin = ffmpeg_sidecar::ffprobe::ffprobe_path();
    let mut cmd = std::process::Command::new(&ffprobe_bin);
    // No console window: see `crate::video::ffmpeg::hide_console`.
    crate::video::ffmpeg::hide_console(&mut cmd);
    let output = cmd
        .args([
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=codec_name:stream_tags=language,title",
            // Unlike the single-stream probes elsewhere in this crate (which
            // pass `noprint_wrappers=1` because there is only ever one
            // relevant stream to read), this can match several subtitle
            // streams, so the `[STREAM]`/`[/STREAM]` wrappers are kept ON --
            // they are the only thing that marks where one stream's fields
            // end and the next one's begin.
            "-of",
            "default=noprint_wrappers=0",
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
    let mut streams = Vec::new();
    let mut codec: Option<String> = None;
    let mut language: Option<String> = None;
    let mut title: Option<String> = None;
    for line in stdout.lines() {
        let line = line.trim();
        match line {
            "[STREAM]" => {
                codec = None;
                language = None;
                title = None;
                continue;
            }
            "[/STREAM]" => {
                streams.push(SubtitleStream {
                    index: streams.len(),
                    codec: codec.take().unwrap_or_default(),
                    language: language.take(),
                    title: title.take(),
                });
                continue;
            }
            _ => {}
        }
        let Some((key, value)) = line.split_once('=') else { continue };
        let value = value.trim();
        match key {
            "codec_name" => codec = Some(value.to_string()),
            "TAG:language" => language = Some(value.to_string()),
            "TAG:title" => title = Some(value.to_string()),
            _ => {}
        }
    }

    Ok(streams)
}

/// Whether `codec` (an ffprobe `codec_name`) is a BITMAP (image sequence)
/// subtitle format, which has no text in it to extract without OCR.
///
/// A DENY-list, not an allow-list, and deliberately so. The two ways to be
/// wrong here are not symmetric:
///
/// - wrongly calling a bitmap codec "text" costs one failed ffmpeg run and a
///   clear error (ffmpeg refuses `-c:s srt` on a bitmap stream and exits
///   non-zero, which `extract` reports verbatim);
/// - wrongly calling a text codec "bitmap" tells the user, in
///   [`bitmap_error`]'s own words, that their track needs OCR and sends them
///   away from a track that would have extracted cleanly.
///
/// The previous allow-list (`ass|ssa|subrip|srt|webvtt|mov_text|text`) made
/// the second, worse mistake for every other text subtitle codec ffmpeg can
/// transcode with `-c:s srt`: `eia_608`, `eia_708`, `arib_caption`, `ttml`,
/// `stl`, `microdvd`, `jacosub`, `sami`, `realtext`, `subviewer`,
/// `subviewer1`, `vplayer`, `pjs`, `mpl2`, `hdmv_text_subtitle`. A broadcast
/// MPEG-TS with EIA-608 closed captions was told it was an image format.
///
/// Matched case-insensitively, since ffprobe's own `codec_name` values are
/// always lowercase in practice but nothing here should depend on that
/// holding forever.
pub fn is_bitmap_codec(codec: &str) -> bool {
    matches!(
        codec.trim().to_lowercase().as_str(),
        // ffprobe decoder names, plus the encoder-side aliases of each, since
        // a name is cheap to add and a wrong "needs OCR" is not.
        "hdmv_pgs_subtitle"
            | "pgssub"
            | "dvd_subtitle"
            | "dvdsub"
            | "dvb_subtitle"
            | "dvbsub"
            | "xsub"
    )
}

/// Whether `codec` is a text-based subtitle format this crate can extract.
///
/// Everything that is not a known bitmap codec is treated as text -- see
/// [`is_bitmap_codec`] for why the deny-list is the safe direction. An EMPTY
/// codec name is neither: ffprobe reporting no codec at all is a "we do not
/// know what this is" answer, and [`extract`] reports it as exactly that
/// rather than as a bitmap (the old allow-list said `""` was a bitmap, and
/// produced "subtitle track 0 is , an image-based (bitmap) subtitle format").
pub fn is_text_codec(codec: &str) -> bool {
    !codec.trim().is_empty() && !is_bitmap_codec(codec)
}

/// The error `extract` returns for a bitmap subtitle codec
/// (`hdmv_pgs_subtitle`, `dvd_subtitle`, `dvb_subtitle`, `xsub`). Names the
/// codec, says why (an image format, not text), and names the way out
/// (`--subtitles <file>`) -- see the module doc for why this must never be
/// silently swallowed into an empty track instead.
fn bitmap_error(codec: &str, track: usize) -> String {
    format!(
        "subtitle track {track} is {codec}, an image-based (bitmap) subtitle format -- there is \
         no text in it to extract without OCR, which is out of scope. Supply a text subtitle \
         file directly with --subtitles <file> (.srt or .ass) instead."
    )
}

/// The error `extract` returns when ffprobe named no codec for the track at
/// all. Distinct from [`bitmap_error`] because "we could not identify this"
/// and "this is an image format" call for different next steps, and claiming
/// the latter for the former is a lie the user cannot act on.
fn unknown_codec_error(track: usize, path: &Path) -> String {
    format!(
        "ffprobe reported no codec for subtitle track {track} of {}, so there is no way to \
         tell whether it holds text or images. Supply a text subtitle file directly with \
         --subtitles <file> (.srt or .ass) instead.",
        path.display()
    )
}

/// Extracts subtitle stream `track` (an index among the container's
/// subtitle streams -- see [`SubtitleStream::index`]) from `path` and parses
/// it into a [`Subtitles`] track.
///
/// Dispatches ffmpeg's output format and the parser by codec: `ass`/`ssa`
/// goes through [`ass::parse`] via ffmpeg's `-f ass` muxer, everything else
/// text-based goes through [`srt::parse`] via ffmpeg's `-f srt` muxer (ffmpeg
/// transcodes `mov_text`/`webvtt`/etc. into SubRip text on the way out, the
/// same way `-c:s ass` transcodes non-ASS text formats into ASS). A bitmap
/// codec is refused up front with [`bitmap_error`] and never reaches ffmpeg
/// at all.
pub fn extract(path: &Path, track: usize) -> Result<Subtitles, String> {
    let streams = list_streams(path)?;
    let stream = streams.get(track).ok_or_else(|| {
        format!(
            "{} has no subtitle track {track} ({} subtitle track(s) found)",
            path.display(),
            streams.len()
        )
    })?;

    if stream.codec.trim().is_empty() {
        return Err(unknown_codec_error(track, path));
    }
    if is_bitmap_codec(&stream.codec) {
        return Err(bitmap_error(&stream.codec, track));
    }

    let fmt = if stream.codec.eq_ignore_ascii_case("ass") || stream.codec.eq_ignore_ascii_case("ssa")
    {
        "ass"
    } else {
        "srt"
    };

    let mut cmd = FfmpegCommand::new();
    // Mirrors `video::ffmpeg::FfmpegSource::open`'s `-hide_banner` + `-v
    // error`: stderr must carry only genuine errors, since the status/stderr
    // check below treats any leftover stderr text as a decode error.
    cmd.hide_banner();
    cmd.args(["-v", "error"]);
    cmd.arg("-i").arg(path);
    // Exactly this subtitle stream, transcoded to a text format ffmpeg can
    // mux to a plain pipe.
    cmd.args(["-map", &format!("0:s:{track}")]);
    cmd.args(["-c:s", fmt, "-f", fmt, "-"]);

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to spawn ffmpeg for {}: {e}", path.display()))?;
    let mut stdout = child
        .take_stdout()
        .ok_or_else(|| "ffmpeg stdout was not piped".to_string())?;
    let stderr = child
        .take_stderr()
        .ok_or_else(|| "ffmpeg stderr was not piped".to_string())?;

    // Drained on a background thread so a chatty stderr can never fill its OS
    // pipe buffer and deadlock the process against the stdout read below --
    // the same reasoning `video::ffmpeg::FfmpegSource::open` and
    // `audio::ffmpeg_src::FfmpegAudioSource::open` give for their own
    // stderr threads. Subtitle text is small (kilobytes, not the megabytes a
    // video frame is), so reading it into one `String` rather than streaming
    // it is fine -- there is no per-frame pipeline here to feed incrementally.
    let stderr_thread = std::thread::spawn(move || {
        let mut stderr = stderr;
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text);
        text
    });

    let mut text = String::new();
    let read_err = stdout.read_to_string(&mut text).err();

    let status = child.wait().map_err(|e| format!("waiting for ffmpeg to exit: {e}"))?;
    let stderr_text = stderr_thread.join().unwrap_or_default();
    let stderr_text = stderr_text.trim();

    if !status.success() {
        return Err(if stderr_text.is_empty() {
            format!(
                "ffmpeg exited with {status} extracting subtitle track {track} from {}",
                path.display()
            )
        } else {
            format!(
                "ffmpeg exited with {status} extracting subtitle track {track} from {}: {stderr_text}",
                path.display()
            )
        });
    }
    if !stderr_text.is_empty() {
        return Err(format!(
            "ffmpeg reported an error extracting subtitle track {track} from {}: {stderr_text}",
            path.display()
        ));
    }
    if let Some(e) = read_err {
        return Err(format!(
            "reading ffmpeg's subtitle output for {}: {e}",
            path.display()
        ));
    }

    match fmt {
        "ass" => ass::parse(&text),
        _ => srt::parse(&text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_codecs_are_recognised() {
        for c in ["ass", "ssa", "subrip", "srt", "webvtt", "mov_text"] {
            assert!(is_text_codec(c), "{c} is a text codec");
        }
    }

    #[test]
    fn the_less_common_text_codecs_are_text_too_not_bitmaps() {
        // Every one of these is a TEXT subtitle codec ffmpeg transcodes with
        // `-c:s srt`. The old allow-list rejected all of them and then told
        // the user, in `bitmap_error`'s words, that their track was an image
        // format needing OCR -- factually false, and it sends the user away
        // from a track that would have extracted cleanly.
        for c in [
            "eia_608", "eia_708", "arib_caption", "ttml", "stl", "microdvd", "jacosub",
            "sami", "realtext", "subviewer", "subviewer1", "vplayer", "pjs", "mpl2",
            "hdmv_text_subtitle",
        ] {
            assert!(is_text_codec(c), "{c} is a text codec, not a bitmap");
            assert!(!is_bitmap_codec(c), "{c} must not be called an image format");
        }
    }

    #[test]
    fn bitmap_codecs_are_not_text() {
        for c in ["hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub", "pgssub",
                  "dvdsub", "dvbsub"] {
            assert!(!is_text_codec(c), "{c} is a bitmap codec");
            assert!(is_bitmap_codec(c), "{c} is a bitmap codec");
        }
    }

    #[test]
    fn an_unnamed_codec_is_neither_text_nor_a_bitmap() {
        // ffprobe naming no codec is a "do not know" answer. Calling it a
        // bitmap produced "subtitle track 0 is , an image-based (bitmap)
        // subtitle format", which is both wrong and unactionable.
        for c in ["", "   "] {
            assert!(!is_text_codec(c), "an unnamed codec is not known-good text");
            assert!(!is_bitmap_codec(c), "an unnamed codec is not an image format either");
        }
        let err = unknown_codec_error(0, Path::new("clip.mkv"));
        assert!(!err.to_lowercase().contains("bitmap"), "must not claim bitmap: {err}");
        assert!(err.contains("no codec"), "says what actually happened: {err}");
    }

    #[test]
    fn codec_matching_is_case_insensitive() {
        assert!(is_text_codec("ASS"));
        assert!(!is_text_codec("HDMV_PGS_SUBTITLE"));
        assert!(is_bitmap_codec("HDMV_PGS_SUBTITLE"));
    }

    #[test]
    fn the_bitmap_error_names_the_codec_and_the_escape_hatch() {
        let err = bitmap_error("hdmv_pgs_subtitle", 0);
        assert!(err.contains("hdmv_pgs_subtitle"), "names the codec: {err}");
        assert!(err.contains("--subtitles"), "points at the way out: {err}");
        assert!(
            err.to_lowercase().contains("image") || err.to_lowercase().contains("bitmap"),
            "says WHY: {err}"
        );
    }
}
