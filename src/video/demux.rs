//! Pure-Rust container demux: yields CODED (not decoded) video packets plus
//! track metadata, off of `re_mp4` (MP4/MOV) and `matroska-demuxer`
//! (MKV/WebM). Both are builtin and must build on wasm, which is why this
//! module carries no decoder dependency at all -- decoding whatever
//! [`Packet`] bytes come out of here is Task 5's job (`video::builtin`),
//! wired to whichever crate that task's own evaluation picked.
//!
//! The one piece of this file that is more than plumbing is
//! [`EntropyCoding`]: see its own doc for why getting it right (and returning
//! [`EntropyCoding::Unknown`] rather than guessing when it can't be
//! determined) is load-bearing for a later task, not merely descriptive.

use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A `Read + Seek + Send` reader, boxed so the demuxer works over either a
/// streamed file (native) or an in-memory blob (the web build's uploaded
/// bytes). `Send` so a native render thread may own the demuxer.
trait ReadSeek: Read + Seek + Send {}
impl<T: Read + Seek + Send> ReadSeek for T {}

/// Where the demuxer's readers come from: a path re-opened per handle (native,
/// streamed from disk) or a blob re-cursored per handle (web, where an uploaded
/// file is already fully in memory and has no path). Both `re_mp4` and
/// `matroska_demuxer` take any `Read + Seek`, and the MP4 path needs two
/// independent handles (one consumed parsing the `moov`, one retained to read
/// sample bytes on demand), so this hands out FRESH readers rather than sharing
/// one -- which for a `File` preserves the streamed, never-whole-file memory
/// characteristic documented on [`Demuxer::open`].
enum Src {
    Path(PathBuf),
    Bytes { bytes: Arc<[u8]>, label: String },
}

impl Src {
    fn reader(&self) -> Result<Box<dyn ReadSeek>, String> {
        match self {
            Src::Path(p) => Ok(Box::new(
                File::open(p).map_err(|e| format!("failed to open {}: {e}", p.display()))?,
            )),
            Src::Bytes { bytes, .. } => Ok(Box::new(Cursor::new(Arc::clone(bytes)))),
        }
    }

    fn len(&self) -> Result<u64, String> {
        match self {
            Src::Path(p) => Ok(File::open(p)
                .and_then(|f| f.metadata())
                .map_err(|e| format!("failed to stat {}: {e}", p.display()))?
                .len()),
            Src::Bytes { bytes, .. } => Ok(bytes.len() as u64),
        }
    }

    fn label(&self) -> String {
        match self {
            Src::Path(p) => p.display().to_string(),
            Src::Bytes { label, .. } => label.clone(),
        }
    }
}

/// H.264 entropy coding mode, from the PPS `entropy_coding_mode_flag`.
///
/// Not informational: `rust_h264` decodes CAVLC streams to visibly wrong
/// pixels while returning no error, so this decides which backend may handle
/// the file at all. `Unknown` must be treated as unsafe for the builtin
/// path -- a guess that lands on CABAC when the stream is CAVLC produces a
/// wrong render with no warning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntropyCoding {
    Cavlc,
    Cabac,
    Unknown,
}

/// What the container's metadata says about its one video track.
///
/// `frame_count` mirrors [`crate::video::stream::SourceInfo::frame_count_hint`]'s
/// honesty rule: `None` is a real answer for a container that cannot say its
/// length up front (Matroska/WebM has no equivalent of MP4's sample table),
/// never a guess.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoTrack {
    /// A short codec name ("h264", "h265", "vp8", "vp9", "av1", or the
    /// container's own codec id lowercased as a fallback) so Task 6's probe
    /// can refuse a codec no decoder here supports.
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub frame_count: Option<usize>,
    /// Track length in seconds, when the container states one.
    ///
    /// Distinct from `frame_count` and held to a LOOSER standard on purpose:
    /// this is only ever used to size a progress bar
    /// ([`crate::video::stream::FrameSource::frame_count_estimate`]), never to
    /// refuse or truncate a render. Matroska is precisely the case that needs
    /// it -- the container has no frame count at all, so `frame_count` is
    /// honestly `None`, but it does declare a Duration, and `duration * fps`
    /// is a perfectly good denominator for a bar that would otherwise be an
    /// indeterminate spinner for the whole render.
    pub duration_s: Option<f64>,
    pub entropy: EntropyCoding,
}

/// One coded (still-compressed) frame's bytes, exactly as the container
/// stored them. For MP4/MOV this is length-prefixed NAL units (AVCC form,
/// per the track's `avcC` box); for Matroska/WebM it is one Block's payload.
/// Decoding -- including any reshaping a decoder needs -- is Task 5's job,
/// not this file's.
#[derive(Debug)]
pub struct Packet {
    pub data: Vec<u8>,
}

/// A re-openable-per-file (not per-stream -- see [`Demuxer::open`]) cursor
/// over one video file's container, dispatched to whichever of the two pure
/// the builtin backend can parse it.
pub struct Demuxer {
    track: VideoTrack,
    backend: Backend,
    /// Raw AVCDecoderConfigurationRecord bytes (MP4's `avcC`, or MKV's
    /// `CodecPrivate` for a `V_MPEG4/ISO/AVC` track) for an H.264 track with
    /// out-of-band parameter sets. `None` for any other codec, or an H.264
    /// track that has no out-of-band config to read (see
    /// `entropy_coding_from_avcc`'s doc for the same gap). This is the exact
    /// same byte layout `entropy_coding_from_avcc` below already parses one
    /// bit out of -- Task 5's `BuiltinVideoSource` needs the rest of it (the
    /// SPS/PPS NAL units themselves) to actually decode, not just to read the
    /// entropy coding mode.
    avc_config: Option<Vec<u8>>,
}

enum Backend {
    Mp4(Mp4Demux),
    Mkv(MkvDemux),
}

impl Demuxer {
    /// Opens `path` and sniffs its container from a magic number at the very
    /// front of the file, not the file extension -- the same philosophy
    /// `video::source::is_video_path`'s doc describes for the higher-level
    /// routing decision: the backend that actually parses the container is
    /// the one source of truth for whether it's valid, so sniffing here must
    /// not duplicate a naming convention that could disagree with it.
    ///
    /// Streams from a live file handle rather than reading `path` into
    /// memory up front. Both `re_mp4::Mp4::read` and
    /// `matroska_demuxer::MatroskaFile::open` accept ANY `Read + Seek`, not
    /// only an in-memory buffer -- `re_mp4::Mp4::read_bytes` and
    /// `Cursor::new(bytes)` (the previous shape of this function) were a
    /// convenience wrapper and a workaround respectively, not a requirement
    /// of either crate. A `std::fs::File` works directly for both.
    ///
    /// **Real memory characteristic**, since this replaced a whole-file
    /// `std::fs::read` (measured at the time: 218,095,616 bytes peak on a
    /// 61,669,350-byte input, a 3.5x blow-up from parser overhead on top of
    /// the file itself -- see this task's report): peak memory for the life
    /// of the returned `Demuxer` is the parsed sample/frame index
    /// (proportional to frame COUNT, not file size -- each `re_mp4::Sample`
    /// is a small `Copy` struct) plus at most one packet's coded bytes at a
    /// time, decoded lazily by [`Demuxer::next_packet`]. It is never the
    /// whole file, regardless of the file's size -- for MP4, `Mp4Demux`
    /// seeks to each sample's own byte range and reads exactly that sample's
    /// bytes on demand; for MKV, `matroska_demuxer::MatroskaFile` already
    /// pulls one block at a time off whatever `Read + Seek` it was given, so
    /// handing it a `File` instead of an in-memory `Cursor` was the entire
    /// fix for that half.
    pub fn open(path: &Path) -> Result<Self, String> {
        Self::open_src(Src::Path(path.to_path_buf()))
    }

    /// Like [`open`](Self::open), but over an in-memory blob (the web build,
    /// whose uploaded files have no path). `name` supplies the error label; the
    /// bytes are re-cursored per reader, exactly as a path is re-opened.
    pub fn open_bytes(name: &str, bytes: Arc<[u8]>) -> Result<Self, String> {
        Self::open_src(Src::Bytes { bytes, label: name.to_string() })
    }

    fn open_src(src: Src) -> Result<Self, String> {
        let label = src.label();

        // Sniff from a small magic-number read off a throwaway reader; the
        // dispatched `open_*` each take their own fresh reader(s) from `src`, so
        // there is nothing to rewind.
        let mut reader = src.reader()?;
        let mut magic = [0u8; 8];
        let n = reader
            .read(&mut magic)
            .map_err(|e| format!("failed to read {label}: {e}"))?;
        drop(reader);

        match sniff(&magic[..n]) {
            Sniff::Mp4 => open_mp4(&src),
            Sniff::Matroska => open_mkv(&src),
            Sniff::Unknown => Err(format!(
                "{label}: not a recognised video container (expected an MP4/MOV \
                 `ftyp` box or an MKV/WebM EBML header)"
            )),
        }
    }

    /// The track's metadata. Cheap to call repeatedly -- it is computed once
    /// in [`Demuxer::open`] and cloned out, not re-derived.
    pub fn track(&self) -> VideoTrack {
        self.track.clone()
    }

    /// The raw AVCDecoderConfigurationRecord bytes for an H.264 track with
    /// out-of-band parameter sets -- `None` for every other case (see the
    /// field's own doc). [`Packet`]s carry only sample data, never the SPS/PPS
    /// that live out-of-band in this record, so a decoder needs both.
    pub fn avc_decoder_config(&self) -> Option<&[u8]> {
        self.avc_config.as_deref()
    }

    /// The next coded packet, or `None` once every sample/frame belonging to
    /// the video track has been yielded. A drained demuxer keeps returning
    /// `None`, matching [`crate::video::stream::FrameStream::next`]'s
    /// contract even though this type does not implement that trait itself
    /// (it yields coded packets, not decoded [`image::RgbaImage`] frames --
    /// Task 5's `BuiltinVideoSource` is the actual `FrameStream`, built on top
    /// of this).
    pub fn next_packet(&mut self) -> Result<Option<Packet>, String> {
        match &mut self.backend {
            Backend::Mp4(m) => m.next_packet(),
            Backend::Mkv(m) => m.next_packet(),
        }
    }
}

enum Sniff {
    Mp4,
    Matroska,
    Unknown,
}

/// Sniffs a container from its own bytes: the EBML magic number
/// (`1A 45 DF A3`) for Matroska/WebM, or an ISO base media `ftyp` box (its
/// type field at byte offset 4) for MP4/MOV. Deliberately not extension
/// based -- a wrong extension must not make an otherwise-valid file
/// unreadable, and a garbage file with a `.mp4` name must not be handed to
/// `re_mp4` just because of its name.
fn sniff(bytes: &[u8]) -> Sniff {
    const EBML_MAGIC: [u8; 4] = [0x1A, 0x45, 0xDF, 0xA3];
    if bytes.len() >= 4 && bytes[0..4] == EBML_MAGIC {
        Sniff::Matroska
    } else if bytes.len() >= 8 && &bytes[4..8] == b"ftyp" {
        Sniff::Mp4
    } else {
        Sniff::Unknown
    }
}

// ---------------------------------------------------------------------------
// MP4 / MOV, via `re_mp4`.
// ---------------------------------------------------------------------------

/// A packet's coded bytes live only on the file, at the byte range
/// [`re_mp4::Sample::byte_range`] names -- `file` is seeked there and read
/// exactly that many bytes on demand by [`Mp4Demux::next_packet`], never
/// materialized for the whole clip up front. See [`Demuxer::open`]'s doc for
/// the memory characteristic this gives the whole `Demuxer`.
struct Mp4Demux {
    reader: Box<dyn ReadSeek>,
    samples: Vec<re_mp4::Sample>,
    next_index: usize,
}

impl Mp4Demux {
    fn next_packet(&mut self) -> Result<Option<Packet>, String> {
        let Some(sample) = self.samples.get(self.next_index).copied() else {
            return Ok(None);
        };
        self.next_index += 1;
        let range = sample.byte_range();
        let len = range.end.saturating_sub(range.start);

        self.reader.seek(SeekFrom::Start(range.start as u64)).map_err(|e| {
            format!("seeking to mp4 sample {} at byte {}: {e}", self.next_index - 1, range.start)
        })?;
        let mut data = vec![0u8; len];
        self.reader.read_exact(&mut data).map_err(|e| {
            format!(
                "mp4 sample {} claims byte range {:?}, past the end of the file: {e}",
                self.next_index - 1,
                range
            )
        })?;
        Ok(Some(Packet { data }))
    }
}

fn open_mp4(src: &Src) -> Result<Demuxer, String> {
    let size = src.len()?;
    // `re_mp4::Mp4::read` consumes its reader parsing metadata (`moov` etc.)
    // and does not hand it back, but `Mp4Demux` below needs a live handle
    // afterwards to read sample bytes on demand -- so each gets its own fresh
    // reader from `src` (a second `File::open` on native, a second `Cursor`
    // over the shared `Arc` on the web). Every `next_packet` seeks explicitly
    // before reading, so the retained reader's starting position is irrelevant.
    let read_handle = src.reader()?;

    let mp4 = re_mp4::Mp4::read(read_handle, size)
        .map_err(|e| format!("invalid mp4/mov container: {e}"))?;

    let track = mp4
        .tracks()
        .values()
        .find(|t| t.kind == Some(re_mp4::TrackKind::Video))
        .ok_or_else(|| "mp4/mov container has no video track".to_string())?;

    let frame_count = Some(track.samples.len());

    // `fps` comes from the track's own total duration (mdhd, authoritative)
    // and sample count, mirroring how `video::ffmpeg::probe_metadata` treats
    // an undiscoverable rate as an error rather than a guessed default (see
    // that file's `parse_frame_rate`) -- never fabricate a frame rate.
    if track.timescale == 0 || track.duration == 0 || track.samples.is_empty() {
        return Err(
            "mp4/mov video track has no usable duration/sample data to derive a frame rate from"
                .to_string(),
        );
    }
    let fps = (track.samples.len() as f64 * track.timescale as f64 / track.duration as f64) as f32;
    if !fps.is_finite() || fps <= 0.0 {
        return Err(format!(
            "mp4/mov video track reported a non-finite or non-positive frame rate ({fps})"
        ));
    }

    // Dimensions come from the SAMPLE DESCRIPTION (`stsd`), not the track
    // header (`tkhd`).
    //
    // `re_mp4::Track::width`/`height` are `tkhd.width`/`height`, which is the
    // PRESENTATION size: coded size times the pixel aspect ratio. For
    // square-pixel content the two agree, which is why this went unnoticed;
    // for anamorphic content (AVCHD/HDV-style 1440x1080 stored for 1920x1080
    // display) they differ, and it is the CODED size a decoder emits.
    // Reporting the display size made `BuiltinVideoSource::open_path` accept
    // such a file -- it never decodes a frame -- and then made
    // `BuiltinFrameStream::accept` fail on frame 0 with "decoded frame
    // 1440x1080 does not match the container's reported 1920x1080". Because
    // `Backend::Auto` returns as soon as the builtin backend OPENS, that
    // failure landed mid-render instead of triggering the ffmpeg fallback.
    //
    // A `0` in the sample entry means the box did not carry a size at all
    // (legal, if rare), and `tkhd` is then the only thing left to go on.
    let (codec, entropy, avc_config, coded) = match &track.trak(&mp4).mdia.minf.stbl.stsd.contents {
        re_mp4::StsdBoxContent::Avc1(c) => (
            "h264".to_string(),
            entropy_coding_from_avcc(&c.avcc.raw),
            Some(c.avcc.raw.clone()),
            (c.width, c.height),
        ),
        re_mp4::StsdBoxContent::Hvc1(c) | re_mp4::StsdBoxContent::Hev1(c) => {
            ("h265".to_string(), EntropyCoding::Unknown, None, (c.width, c.height))
        }
        re_mp4::StsdBoxContent::Vp08(c) => {
            ("vp8".to_string(), EntropyCoding::Unknown, None, (c.width, c.height))
        }
        re_mp4::StsdBoxContent::Vp09(c) => {
            ("vp9".to_string(), EntropyCoding::Unknown, None, (c.width, c.height))
        }
        re_mp4::StsdBoxContent::Av01(c) => {
            ("av1".to_string(), EntropyCoding::Unknown, None, (c.width, c.height))
        }
        // Only video-kind `StsdBoxContent` variants can reach here (the
        // `find` above filtered on `TrackKind::Video`); this arm exists so
        // the match stays exhaustive against future/audio variants without
        // panicking, not because it is expected to run.
        _ => ("unknown".to_string(), EntropyCoding::Unknown, None, (0, 0)),
    };
    let width = u32::from(if coded.0 != 0 { coded.0 } else { track.width });
    let height = u32::from(if coded.1 != 0 { coded.1 } else { track.height });

    let samples = track.samples.clone();

    // Both non-zero: checked above, where the same two fields are what the
    // frame rate is derived from.
    let duration_s = Some(track.duration as f64 / track.timescale as f64);

    Ok(Demuxer {
        track: VideoTrack { codec, width, height, fps, frame_count, duration_s, entropy },
        backend: Backend::Mp4(Mp4Demux { reader: src.reader()?, samples, next_index: 0 }),
        avc_config,
    })
}

// ---------------------------------------------------------------------------
// MKV / WebM, via `matroska-demuxer`.
// ---------------------------------------------------------------------------

/// `matroska_demuxer::MatroskaFile` itself pulls one block at a time off
/// whatever `Read + Seek` it holds -- it is already the lazy shape
/// `Mp4Demux` above had to be given explicitly, so wrapping a `File` here
/// instead of `Cursor<Vec<u8>>` was the entire memory fix for this half of
/// the demuxer. See [`Demuxer::open`]'s doc for the resulting characteristic.
struct MkvDemux {
    file: matroska_demuxer::MatroskaFile<Box<dyn ReadSeek>>,
    video_track: u64,
}

impl MkvDemux {
    fn next_packet(&mut self) -> Result<Option<Packet>, String> {
        let mut frame = matroska_demuxer::Frame::default();
        loop {
            let got = self
                .file
                .next_frame(&mut frame)
                .map_err(|e| format!("mkv/webm demux error: {e}"))?;
            if !got {
                return Ok(None);
            }
            if frame.track == self.video_track {
                return Ok(Some(Packet { data: std::mem::take(&mut frame.data) }));
            }
            // A frame from some other track (audio, subtitles) -- keep
            // reading until the next video frame or true end of file.
        }
    }
}

fn open_mkv(src: &Src) -> Result<Demuxer, String> {
    let file = matroska_demuxer::MatroskaFile::open(src.reader()?)
        .map_err(|e| format!("invalid mkv/webm container: {e}"))?;

    let track = file
        .tracks()
        .iter()
        .find(|t| t.track_type() == matroska_demuxer::TrackType::Video)
        .ok_or_else(|| "mkv/webm container has no video track".to_string())?;

    let video = track.video().ok_or_else(|| {
        "mkv/webm video track is missing its Video settings element".to_string()
    })?;
    let width = video.pixel_width().get() as u32;
    let height = video.pixel_height().get() as u32;

    // Unlike MP4's mdhd/stts (an explicit duration and per-sample deltas),
    // Matroska's only per-track rate signal is DefaultDuration -- ns per
    // frame. Absent that, there is no honest fps to report (no equivalent of
    // MP4's sample table to derive one from without a full scan, which would
    // defeat this file's whole point), so this errors rather than guessing,
    // matching `video::ffmpeg::probe_metadata`'s treatment of an
    // undiscoverable rate.
    let default_duration_ns = track.default_duration().ok_or_else(|| {
        "mkv/webm video track has no default frame duration to derive a frame rate from"
            .to_string()
    })?;
    let fps = (1_000_000_000.0 / default_duration_ns.get() as f64) as f32;
    if !fps.is_finite() || fps <= 0.0 {
        return Err(format!(
            "mkv/webm video track reported a non-finite or non-positive frame rate ({fps})"
        ));
    }

    let (codec, entropy, avc_config) = match track.codec_id() {
        "V_MPEG4/ISO/AVC" => {
            let private = track.codec_private();
            (
                "h264".to_string(),
                private.map(entropy_coding_from_avcc).unwrap_or(EntropyCoding::Unknown),
                private.map(<[u8]>::to_vec),
            )
        }
        "V_MPEGH/ISO/HEVC" => ("h265".to_string(), EntropyCoding::Unknown, None),
        "V_VP8" => ("vp8".to_string(), EntropyCoding::Unknown, None),
        "V_VP9" => ("vp9".to_string(), EntropyCoding::Unknown, None),
        "V_AV1" => ("av1".to_string(), EntropyCoding::Unknown, None),
        other => (other.to_ascii_lowercase(), EntropyCoding::Unknown, None),
    };

    let video_track = track.track_number().get();

    // Matroska's Segment `Duration` is a float in TimestampScale units, and
    // TimestampScale is nanoseconds -- so this is the one length signal the
    // container gives, and the only reason `frame_count: None` above does not
    // have to mean a totalless progress bar. Absent or non-finite stays
    // `None`; nothing here guesses.
    let duration_s = file.info().duration().filter(|d| d.is_finite() && *d > 0.0).map(|d| {
        d * file.info().timestamp_scale().get() as f64 / 1_000_000_000.0
    });

    Ok(Demuxer {
        track: VideoTrack {
            codec,
            width,
            height,
            fps,
            frame_count: None,
            duration_s,
            entropy,
        },
        backend: Backend::Mkv(MkvDemux { file, video_track }),
        avc_config,
    })
}

// ---------------------------------------------------------------------------
// Shared H.264 PPS parsing, used by both containers above.
//
// `re_mp4`'s `RawBox<AvcCBox>::raw` (an MP4 `avc1` sample entry's `avcC`
// child) and `matroska_demuxer::TrackEntry::codec_private()` (an
// `V_MPEG4/ISO/AVC` track's CodecPrivate) hand back the exact same binary
// layout: an ISO/IEC 14496-15 AVCDecoderConfigurationRecord, sans any box
// header. One parser below serves both.
// ---------------------------------------------------------------------------

/// Extracts the H.264 entropy coding mode from a raw
/// AVCDecoderConfigurationRecord. Returns [`EntropyCoding::Unknown`] on
/// anything that does not parse cleanly -- truncated data, an empty PPS list,
/// whatever -- never a guess. See [`EntropyCoding`]'s own doc for why a wrong
/// guess here is worse than admitting ignorance.
fn entropy_coding_from_avcc(avcc: &[u8]) -> EntropyCoding {
    first_pps(avcc)
        .and_then(|pps| entropy_coding_from_pps_nal(&pps))
        .unwrap_or(EntropyCoding::Unknown)
}

/// Pulls the first PPS NAL unit's bytes (NAL header included) out of an
/// AVCDecoderConfigurationRecord. Layout (ISO/IEC 14496-15 §5.3.3.1.2):
/// 1 byte configurationVersion, 1 byte AVCProfileIndication, 1 byte
/// profile_compatibility, 1 byte AVCLevelIndication, 1 byte
/// (reserved | lengthSizeMinusOne), 1 byte (reserved | numOfSequenceParameterSets),
/// then that many `[u16 length][length bytes]` SPS entries, then 1 byte
/// numOfPictureParameterSets, then that many `[u16 length][length bytes]`
/// PPS entries.
fn first_pps(avcc: &[u8]) -> Option<Vec<u8>> {
    let num_sps = usize::from(*avcc.get(5)? & 0x1F);
    let mut pos = 6usize;
    for _ in 0..num_sps {
        let len = usize::from(u16::from_be_bytes(avcc.get(pos..pos + 2)?.try_into().ok()?));
        pos += 2 + len;
    }
    let num_pps = usize::from(*avcc.get(pos)?);
    pos += 1;
    if num_pps == 0 {
        return None;
    }
    let len = usize::from(u16::from_be_bytes(avcc.get(pos..pos + 2)?.try_into().ok()?));
    pos += 2;
    avcc.get(pos..pos + len).map(<[u8]>::to_vec)
}

/// Reads `entropy_coding_mode_flag` out of one PPS NAL unit's bytes.
///
/// The PPS RBSP (ITU-T H.264 §7.3.2.2) opens with two Exp-Golomb fields --
/// `pic_parameter_set_id` and `seq_parameter_set_id` -- before the single bit
/// this needs. Emulation prevention bytes (the `00 00 03` → `00 00` escaping
/// every NAL unit's RBSP uses to keep a real start code from appearing in the
/// payload) are stripped first; skipping that step could desync the bit
/// position the moment a real `0x03` shows up where the escape check would
/// have consumed it.
fn entropy_coding_from_pps_nal(pps: &[u8]) -> Option<EntropyCoding> {
    let rbsp = remove_emulation_prevention(pps.get(1..)?); // skip the 1-byte NAL header
    let mut bits = BitReader::new(&rbsp);
    let _pic_parameter_set_id = bits.read_ue()?;
    let _seq_parameter_set_id = bits.read_ue()?;
    Some(if bits.read_bit()? == 1 { EntropyCoding::Cabac } else { EntropyCoding::Cavlc })
}

/// Undoes the `00 00 03` -> `00 00` escaping every NAL unit's RBSP uses.
///
/// `zero_run` SATURATES rather than wrapping. It was a `u8` that only ever
/// reset on a `0x03` after two zeros, so 256 consecutive zero bytes overflowed
/// it -- `attempt to add with overflow`, a panic in any build with debug
/// assertions on, which is every `cargo test` run and every `cargo build`
/// without `--release`. A valid H.264 stream cannot contain three consecutive
/// zero bytes, but this input is not validated: `first_pps` hands over
/// whatever byte range the avcC record's PPS-length field points at, and
/// `Demuxer::open` runs on the DEFAULT `--backend auto` path, so a corrupt or
/// crafted `.mp4` reaches it. Only the "have we seen two zeros" question is
/// ever asked of this counter, so saturating loses nothing.
fn remove_emulation_prevention(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut zero_run = 0u8;
    for &b in data {
        if zero_run >= 2 && b == 0x03 {
            zero_run = 0;
            continue;
        }
        zero_run = if b == 0 { zero_run.saturating_add(1) } else { 0 };
        out.push(b);
    }
    out
}

/// A minimal big-endian, MSB-first bitstream reader for H.264's Exp-Golomb
/// and fixed-width fields.
///
/// `pub(crate)`: [`crate::video::builtin`]'s SPS walk uses
/// [`BitReader::read_bits`] and [`BitReader::read_se`] too, so nothing in
/// this file calling them is not dead code.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(crate) fn read_bit(&mut self) -> Option<u8> {
        let byte = self.pos / 8;
        let bit = self.pos % 8;
        let b = *self.data.get(byte)?;
        self.pos += 1;
        Some((b >> (7 - bit)) & 1)
    }

    /// An unsigned fixed-width field (`u(n)`), MSB first. `n` must be at most
    /// 32; the widest this crate actually reads is `u(16)` (VUI's
    /// `sar_width`/`sar_height`).
    pub(crate) fn read_bits(&mut self, n: u32) -> Option<u32> {
        if n > 32 {
            return None;
        }
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.read_bit()?);
        }
        Some(value)
    }

    /// Exp-Golomb signed (`se(v)`), ITU-T H.264 §9.1.1: the `ue(v)` codeNum
    /// `k` maps to `(-1)^(k+1) * ceil(k / 2)`.
    ///
    /// Computed through `i64` so a malformed stream's huge `k` cannot
    /// overflow the `k + 1` before it is range-checked back down to `i32`.
    pub(crate) fn read_se(&mut self) -> Option<i32> {
        let k = i64::from(self.read_ue()?);
        let value = if k % 2 == 1 { (k + 1) / 2 } else { -(k / 2) };
        i32::try_from(value).ok()
    }

    /// Exp-Golomb unsigned (`ue(v)`), ITU-T H.264 §9.1.
    pub(crate) fn read_ue(&mut self) -> Option<u32> {
        let mut leading_zeros = 0u32;
        while self.read_bit()? == 0 {
            leading_zeros += 1;
            // A real PPS's first two fields are tiny (almost always 0),
            // i.e. essentially no leading zeros. This guard exists only so
            // that malformed/garbage input can't walk this up to 32 and
            // panic the `1u32 << leading_zeros` below on overflow.
            if leading_zeros >= 32 {
                return None;
            }
        }
        let mut value = 0u32;
        for _ in 0..leading_zeros {
            value = (value << 1) | u32::from(self.read_bit()?);
        }
        Some((1u32 << leading_zeros) - 1 + value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reuses the ffmpeg-generated sample helper; see src/video/ffmpeg.rs.
    #[test]
    fn an_mp4s_track_metadata_matches_what_made_it() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip("demux", 2, 64, 48, 10) else { return };
        let d = Demuxer::open(&path).expect("open");
        let t = d.track();
        assert_eq!((t.width, t.height), (64, 48));
        assert!((t.fps - 10.0).abs() < 0.01, "fps was {}", t.fps);
        assert_eq!(t.frame_count, Some(20), "2s at 10fps");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn every_coded_packet_is_yielded_once_in_order() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip("demux2", 1, 32, 32, 10) else { return };
        let mut d = Demuxer::open(&path).expect("open");
        let mut n = 0;
        while d.next_packet().expect("packet").is_some() { n += 1; }
        assert_eq!(n, 10, "one packet per frame for this encode");
        assert!(d.next_packet().expect("drained").is_none(), "a drained demuxer stays drained");
        let _ = std::fs::remove_file(&path);
    }

    /// Anamorphic content is the case where `tkhd` (display size) and `stsd`
    /// (coded size) part company. `-aspect 8:3` on a 64x48 encode gives a
    /// pixel aspect ratio of 2:1, so `tkhd` says 128x48 while the decoder
    /// emits 64x48. Reporting `tkhd` made `BuiltinVideoSource::open_path`
    /// accept the file and `BuiltinFrameStream::accept` then kill the render
    /// on frame 0 -- past the point where `Backend::Auto` could still fall
    /// back to ffmpeg.
    #[test]
    fn anamorphic_mp4_reports_its_coded_size_not_its_display_size() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(
            "demux_anamorphic", &["-aspect", "8:3"], 1, 64, 48, 10,
        ) else { return };
        let track = Demuxer::open(&path).expect("open").track();
        assert_eq!(
            (track.width, track.height),
            (64, 48),
            "the coded size is what a decoder emits; tkhd would say 128x48"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_non_video_file_errors_rather_than_panicking() {
        let path = std::env::temp_dir().join(format!("h2b_notvideo_{}.mp4", std::process::id()));
        std::fs::write(&path, b"this is not a container").expect("write");
        assert!(Demuxer::open(&path).is_err(), "garbage must error");
        let _ = std::fs::remove_file(&path);
    }

    /// A long zero run used to overflow the `u8` zero counter and panic under
    /// debug assertions -- i.e. in every `cargo test` and every non-release
    /// build. Reachable from a corrupt `.mp4` on the default `--backend auto`
    /// path, since `first_pps` reads an unvalidated length out of the avcC
    /// record and hands the bytes straight here.
    #[test]
    fn a_long_zero_run_does_not_overflow_the_escape_counter() {
        let out = remove_emulation_prevention(&vec![0u8; 300]);
        assert_eq!(out.len(), 300, "no 0x03 to remove, so nothing is removed");
        assert!(out.iter().all(|&b| b == 0));
    }

    #[test]
    fn emulation_prevention_bytes_are_still_removed_after_a_long_zero_run() {
        // 300 zeros, then `03`, then a payload byte: the `03` is an escape
        // (it follows at least two zeros) and must go; the rest must stay.
        let mut input = vec![0u8; 300];
        input.push(0x03);
        input.push(0xAB);
        let out = remove_emulation_prevention(&input);
        assert_eq!(out.len(), 301);
        assert_eq!(out[300], 0xAB, "the escape is dropped, the payload kept");
    }

    /// The load-bearing assertion this whole task exists for: Task 6's guard
    /// routes purely on `track().entropy`, so if this test is wrong the guard
    /// silently protects nothing. `-coder 0`/`-coder 1` force CAVLC/CABAC
    /// respectively (see `video::ffmpeg::tests::sample_clip_args`).
    #[test]
    fn entropy_coding_mode_matches_the_encoders_coder_choice() {
        let Some(cavlc_path) = crate::video::ffmpeg::tests::sample_clip_args(
            "demux_cavlc", &["-coder", "0"], 1, 32, 32, 5,
        ) else { return };
        let cavlc_entropy = Demuxer::open(&cavlc_path).expect("open cavlc clip").track().entropy;
        assert_eq!(
            cavlc_entropy,
            EntropyCoding::Cavlc,
            "-coder 0 must be read back from the PPS as CAVLC"
        );
        let _ = std::fs::remove_file(&cavlc_path);

        let Some(cabac_path) = crate::video::ffmpeg::tests::sample_clip_args(
            "demux_cabac", &["-coder", "1"], 1, 32, 32, 5,
        ) else { return };
        let cabac_entropy = Demuxer::open(&cabac_path).expect("open cabac clip").track().entropy;
        assert_eq!(
            cabac_entropy,
            EntropyCoding::Cabac,
            "-coder 1 must be read back from the PPS as CABAC"
        );
        let _ = std::fs::remove_file(&cabac_path);
    }
}
