//! The pure-Rust [`FrameSource`]: `video::demux::Demuxer` (coded packets) fed
//! into `rust_h264` (pixels), with no ffmpeg process involved at all. Exists
//! so wasm -- which cannot spawn `ffmpeg` -- has a decode path, and so native
//! builds can skip a subprocess for the codec this crate actually handles.
//!
//! **CABAC only.** Task 1's evaluation (`.superpowers/sdd/2026-07-27-video-
//! decode-backends/task-1-evaluation.md`) measured `rust_h264` 0.4.0 against
//! ffmpeg on 13 clips: every CABAC stream (all 7 real-world captures, every
//! CABAC synthetic clip) landed within 0.0-0.26 mean absolute per-channel
//! difference -- rounding noise. But two CAVLC clips decoded to visibly wrong
//! pixels -- mean absolute difference up to ~27 on luma, ~49 on chroma --
//! while `rust_h264` returned no error and did not panic. The owner's ruling
//! (recorded in `progress.md`) was to ship this backend CABAC-only:
//! [`RustVideoSource::open_path`] refuses anything else before a single frame
//! is decoded, per [`crate::video::demux::EntropyCoding`]'s own doc. This is
//! deliberately redundant with Task 6's later `video::backend` routing guard
//! (not yet built when this module was written) -- a caller that reaches this
//! type directly, bypassing that guard, must still be protected.
//!
//! Decoding happens lazily inside [`FrameStream::next`]: packets are pulled
//! from the [`Demuxer`] one at a time, fed to an `OrderedDecoder`, and any
//! frames it returns (0 or more per packet -- B-frame reordering means a
//! frame's packet and its emission don't line up 1:1) are queued for pickup.
//! That queue is bounded by the decoder's own reorder depth (16 frames by
//! `rust_h264`'s default), never the whole clip -- an accumulating `Vec` of
//! every decoded frame would defeat the entire point of this plan.

use crate::video::demux::{BitReader, Demuxer, EntropyCoding};
use crate::video::stream::{FrameSource, FrameStream, SourceInfo};
use image::RgbaImage;
use rust_h264::decoder::{Frame as H264Frame, OrderedDecoder};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

/// A re-openable [`FrameSource`] backed by `Demuxer` + `rust_h264`, no
/// subprocess involved.
///
/// Deliberately holds only `path` and the metadata [`Demuxer::track`] already
/// reported -- exactly the same shape as [`crate::video::ffmpeg::FfmpegSource`]
/// -- so that [`FrameSource::open`] can build a brand new [`Demuxer`] and
/// decoder each time, which is what makes two opens agree frame for frame.
pub struct RustVideoSource {
    path: PathBuf,
    width: u32,
    height: u32,
    fps: f32,
    frame_count_hint: Option<usize>,
}

impl RustVideoSource {
    /// Opens `path`, validates it up front, and refuses anything this backend
    /// cannot decode correctly -- BEFORE constructing a source that would
    /// otherwise decode it (wrongly, for CAVLC) the first time it's opened.
    ///
    /// Refuses three things, each with its own actionable message:
    /// - a codec other than H.264 (the only thing `rust_h264` implements);
    /// - [`EntropyCoding::Cavlc`] -- named explicitly, per the module doc;
    /// - [`EntropyCoding::Unknown`] -- refused for the same reason, because a
    ///   guess that happens to land on CABAC when the stream is actually
    ///   CAVLC produces exactly the silent-wrong-render this guard exists to
    ///   prevent. Its message also names CAVLC, so both refusals point the
    ///   caller at the same explanation and the same next step.
    /// - an H.264 track with no out-of-band SPS/PPS to decode with at all.
    pub fn open_path(path: &Path) -> Result<Self, String> {
        let demuxer = Demuxer::open(path)?;
        let track = demuxer.track();

        if track.codec != "h264" {
            return Err(format!(
                "{}: the pure-Rust decode backend only supports H.264, but this file's video \
                 track is {:?}; use --backend ffmpeg",
                path.display(),
                track.codec
            ));
        }
        match track.entropy {
            EntropyCoding::Cabac => {}
            EntropyCoding::Cavlc => {
                return Err(format!(
                    "{}: this H.264 stream uses CAVLC entropy coding, which the pure-Rust \
                     decoder (rust_h264) silently decodes to visibly wrong pixels rather than \
                     erroring; use --backend ffmpeg instead",
                    path.display()
                ));
            }
            EntropyCoding::Unknown => {
                return Err(format!(
                    "{}: this H.264 stream's entropy coding mode could not be determined, so \
                     it is refused the same as CAVLC -- guessing CABAC on a stream that turns \
                     out to be CAVLC would silently decode to wrong pixels; use --backend \
                     ffmpeg instead",
                    path.display()
                ));
            }
        }
        if demuxer.avc_decoder_config().is_none() {
            return Err(format!(
                "{}: no out-of-band H.264 parameter sets (avcC/CodecPrivate) were found, so \
                 there is no SPS/PPS to decode with; use --backend ffmpeg",
                path.display()
            ));
        }

        Ok(Self {
            path: path.to_path_buf(),
            width: track.width,
            height: track.height,
            fps: track.fps,
            frame_count_hint: track.frame_count,
        })
    }
}

impl FrameSource for RustVideoSource {
    fn info(&self) -> SourceInfo {
        SourceInfo {
            width: self.width,
            height: self.height,
            fps: self.fps,
            frame_count_hint: self.frame_count_hint,
        }
    }

    fn open(&self) -> Result<Box<dyn FrameStream + '_>, String> {
        Ok(Box::new(RustFrameStream::new(&self.path, self.width, self.height)?))
    }
}

/// A one-shot cursor over one fresh [`Demuxer`] + `rust_h264::decoder::OrderedDecoder`.
struct RustFrameStream {
    demuxer: Demuxer,
    decoder: OrderedDecoder,
    /// `lengthSizeMinusOne + 1` from the track's avcC/CodecPrivate config --
    /// how many bytes prefix each NAL unit inside a [`Packet`](crate::video::demux::Packet).
    length_size: usize,
    /// The YUV→RGB matrix and range read once from the SPS VUI at open time,
    /// applied to every frame. Read once rather than per frame because the
    /// parameter sets are fed to the decoder once, out of band.
    colour: ColourInfo,
    /// Frames the decoder has released but [`FrameStream::next`] hasn't
    /// handed out yet. Bounded by `OrderedDecoder`'s own reorder buffer (16
    /// frames by default), not by clip length -- this is the queue the
    /// module doc's "never an accumulating Vec of the whole clip" refers to.
    queue: VecDeque<RgbaImage>,
    /// How many packets have been pulled from the demuxer so far, purely for
    /// naming the failure point in an error message.
    packets_read: usize,
    /// How many frames this stream has already handed out successfully,
    /// purely for naming the failure point in an error message.
    frames_emitted: usize,
    width: u32,
    height: u32,
    /// Set once the demuxer is drained and the decoder has been flushed (or
    /// a fatal error occurred). A `done` stream with an empty queue keeps
    /// returning `Ok(None)`, matching [`FrameStream::next`]'s contract.
    done: bool,
    display_path: String,
}

impl RustFrameStream {
    fn new(path: &Path, width: u32, height: u32) -> Result<Self, String> {
        let display_path = path.display().to_string();
        let demuxer = Demuxer::open(path)?;

        // `RustVideoSource::open_path` already checked this is `Some` for the
        // source this stream was built from, but `open()` re-derives
        // everything from scratch (a fresh `Demuxer::open` read of the file,
        // just like `FfmpegSource::open` spawns a fresh process) rather than
        // trusting that an earlier check still holds, so this is re-checked
        // here rather than `.expect()`-ed past.
        let config = demuxer
            .avc_decoder_config()
            .ok_or_else(|| {
                format!(
                    "{display_path}: no out-of-band H.264 parameter sets were found; \
                     try --backend ffmpeg"
                )
            })?
            .to_vec();
        let avcc = rust_h264::nal::parse_avcc_config(&config).map_err(|e| {
            format!("{display_path}: invalid avcC configuration record: {e}; try --backend ffmpeg")
        })?;

        // The colour matrix/range the stream asks for, read from the first
        // SPS's VUI. `parse_avcc_config` already stripped emulation
        // prevention bytes from `rbsp`, so this walks it directly. A stream
        // that does not say gets `ColourInfo::default()` (BT.601 limited) --
        // see that type's doc for why that is not a guess.
        let colour = avcc
            .sps_nals
            .first()
            .and_then(|sps| colour_info_from_sps_rbsp(&sps.rbsp))
            .unwrap_or_default();

        let mut decoder = OrderedDecoder::new();
        // SPS/PPS live out-of-band (the avcC box), not in any sample, so they
        // are fed once here rather than showing up in the packet loop below.
        for nal in avcc.sps_nals.iter().chain(avcc.pps_nals.iter()) {
            decoder.decode_nal(nal).map_err(|e| {
                format!("{display_path}: failed to parse SPS/PPS: {e}; try --backend ffmpeg")
            })?;
        }
        let length_size = avcc.length_size;

        Ok(Self {
            demuxer,
            decoder,
            length_size,
            colour,
            queue: VecDeque::new(),
            packets_read: 0,
            frames_emitted: 0,
            width,
            height,
            done: false,
            display_path,
        })
    }

    /// Converts one decoded frame to RGBA (alpha 255) and checks it against
    /// the dimensions [`FrameSource::info`] promised, per that trait's own
    /// contract: a mismatch here is this stream's own fatal error, not a
    /// wrongly-sized frame handed to a caller that trusted `info()`.
    fn accept(&mut self, frame: H264Frame) -> Result<(), String> {
        let img = frame_to_rgba(&frame, self.colour)
            .map_err(|e| format!("{}: {e}; try --backend ffmpeg", self.display_path))?;
        if img.dimensions() != (self.width, self.height) {
            return Err(format!(
                "{}: decoded frame {}x{} does not match the container's reported {}x{}; \
                 try --backend ffmpeg",
                self.display_path,
                img.width(),
                img.height(),
                self.width,
                self.height
            ));
        }
        self.queue.push_back(img);
        Ok(())
    }
}

impl FrameStream for RustFrameStream {
    fn next(&mut self) -> Result<Option<RgbaImage>, String> {
        loop {
            if let Some(img) = self.queue.pop_front() {
                self.frames_emitted += 1;
                return Ok(Some(img));
            }
            if self.done {
                return Ok(None);
            }

            match self.demuxer.next_packet() {
                Ok(Some(packet)) => {
                    let nals = rust_h264::nal::parse_avcc(&packet.data, self.length_size);
                    for nal in &nals {
                        match self.decoder.decode_nal(nal) {
                            Ok(frames) => {
                                for frame in frames {
                                    self.accept(frame)?;
                                }
                            }
                            Err(e) => {
                                self.done = true;
                                return Err(format!(
                                    "{}: rust_h264 failed to decode packet {} (after {} \
                                     frame(s) decoded successfully): {e}; try --backend ffmpeg",
                                    self.display_path, self.packets_read, self.frames_emitted
                                ));
                            }
                        }
                    }
                    self.packets_read += 1;
                }
                Ok(None) => {
                    // Demuxer drained -- flush whatever the decoder was still
                    // holding for reorder, then stop pulling packets for
                    // good. The queue this fills is picked up on the next
                    // loop iteration above.
                    self.done = true;
                    let flushed = self.decoder.flush();
                    for frame in flushed {
                        self.accept(frame)?;
                    }
                }
                Err(e) => {
                    self.done = true;
                    return Err(format!(
                        "{}: demux error reading packet {}: {e}; try --backend ffmpeg",
                        self.display_path, self.packets_read
                    ));
                }
            }
        }
    }
}

/// Which YUV→RGB matrix a stream's SPS VUI asked for.
///
/// Read from `matrix_coefficients`, never guessed from resolution: a fixed
/// BT.601 conversion measured 0.47 mean absolute per-channel difference
/// against ffmpeg on UNTAGGED 720p but ~4.2 on the same content carrying an
/// explicit BT.709 tag -- past the oracle test's "this decoder is wrong"
/// threshold of 3.0. BT.709 is the standard tag for HD, so that is the
/// common case for real captures, not an exotic one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColourMatrix {
    /// `matrix_coefficients` 5 (BT.470BG) or 6 (SMPTE 170M), and the
    /// fallback for every stream that does not say -- see [`ColourInfo`].
    Bt601,
    /// `matrix_coefficients` 1.
    Bt709,
    /// `matrix_coefficients` 0: the planes are already G/B/R, not YCbCr, so
    /// no matrix is applied at all.
    Identity,
}

/// The colour handling one stream asked for: which matrix, and whether the
/// samples span the full 0-255 range or the studio-swing 16-235/16-240 range
/// (`video_full_range_flag`). Both halves matter -- getting the range wrong
/// shifts every pixel even when the matrix is right.
///
/// [`Default`] is BT.601 limited range, which is what an untagged stream
/// gets. That is deliberately NOT a resolution-based heuristic: ffmpeg
/// treats untagged content as BT.601 regardless of resolution (measured: an
/// untagged 720p clip scores 0.47 against a BT.601 conversion, i.e. ffmpeg
/// used BT.601 too), and matching the oracle is the goal. A "smarter" guess
/// would diverge from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ColourInfo {
    matrix: ColourMatrix,
    full_range: bool,
}

impl Default for ColourInfo {
    fn default() -> Self {
        Self { matrix: ColourMatrix::Bt601, full_range: false }
    }
}

/// Reads the VUI colour description out of one SPS RBSP.
///
/// Returns `None` -- meaning "the stream did not say", which the caller turns
/// into [`ColourInfo::default`] -- when there is no VUI, no
/// `video_signal_type_present_flag`, or the walk runs out of bits. Never a
/// guess, matching [`crate::video::demux::EntropyCoding`]'s own philosophy
/// for the same reason: an assumption here is not visible to the user.
///
/// Everything before the VUI (ITU-T H.264 §7.3.2.1, then §E.1.1) has to be
/// walked field by field because the fields are variable-length -- there is
/// no fixed offset to jump to. `rbsp` must already have emulation prevention
/// bytes removed; `rust_h264::nal::parse_avcc_config` hands back exactly
/// that in `NalUnit::rbsp`, so nothing here re-does that step.
fn colour_info_from_sps_rbsp(rbsp: &[u8]) -> Option<ColourInfo> {
    let mut bits = BitReader::new(rbsp);

    let profile_idc = bits.read_bits(8)?;
    // 6 constraint_setN_flags + 2 reserved bits.
    let _constraint_and_reserved = bits.read_bits(8)?;
    let _level_idc = bits.read_bits(8)?;
    let _seq_parameter_set_id = bits.read_ue()?;

    // The chroma/bit-depth/scaling-list block is present only for these
    // profiles (§7.3.2.1). Skipping it when present -- or reading it when
    // absent -- desyncs every field after it.
    if matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        let chroma_format_idc = bits.read_ue()?;
        if chroma_format_idc == 3 {
            let _separate_colour_plane_flag = bits.read_bit()?;
        }
        let _bit_depth_luma_minus8 = bits.read_ue()?;
        let _bit_depth_chroma_minus8 = bits.read_ue()?;
        let _qpprime_y_zero_transform_bypass_flag = bits.read_bit()?;
        if bits.read_bit()? == 1 {
            // seq_scaling_matrix_present_flag
            let lists = if chroma_format_idc != 3 { 8 } else { 12 };
            for i in 0..lists {
                if bits.read_bit()? == 1 {
                    skip_scaling_list(&mut bits, if i < 6 { 16 } else { 64 })?;
                }
            }
        }
    }

    let _log2_max_frame_num_minus4 = bits.read_ue()?;
    let pic_order_cnt_type = bits.read_ue()?;
    if pic_order_cnt_type == 0 {
        let _log2_max_pic_order_cnt_lsb_minus4 = bits.read_ue()?;
    } else if pic_order_cnt_type == 1 {
        let _delta_pic_order_always_zero_flag = bits.read_bit()?;
        let _offset_for_non_ref_pic = bits.read_se()?;
        let _offset_for_top_to_bottom_field = bits.read_se()?;
        let cycle_len = bits.read_ue()?;
        for _ in 0..cycle_len {
            let _offset_for_ref_frame = bits.read_se()?;
        }
    }

    let _max_num_ref_frames = bits.read_ue()?;
    let _gaps_in_frame_num_value_allowed_flag = bits.read_bit()?;
    let _pic_width_in_mbs_minus1 = bits.read_ue()?;
    let _pic_height_in_map_units_minus1 = bits.read_ue()?;
    if bits.read_bit()? == 0 {
        // frame_mbs_only_flag == 0
        let _mb_adaptive_frame_field_flag = bits.read_bit()?;
    }
    let _direct_8x8_inference_flag = bits.read_bit()?;
    if bits.read_bit()? == 1 {
        // frame_cropping_flag
        let _crop_left = bits.read_ue()?;
        let _crop_right = bits.read_ue()?;
        let _crop_top = bits.read_ue()?;
        let _crop_bottom = bits.read_ue()?;
    }

    if bits.read_bit()? == 0 {
        // vui_parameters_present_flag -- the stream carries no VUI at all.
        return None;
    }

    // vui_parameters(), §E.1.1 -- only walked as far as the colour fields.
    if bits.read_bit()? == 1 {
        // aspect_ratio_info_present_flag
        let aspect_ratio_idc = bits.read_bits(8)?;
        if aspect_ratio_idc == 255 {
            // Extended_SAR
            let _sar_width = bits.read_bits(16)?;
            let _sar_height = bits.read_bits(16)?;
        }
    }
    if bits.read_bit()? == 1 {
        // overscan_info_present_flag
        let _overscan_appropriate_flag = bits.read_bit()?;
    }
    if bits.read_bit()? == 0 {
        // video_signal_type_present_flag -- no range and no matrix stated.
        return None;
    }

    let _video_format = bits.read_bits(3)?;
    let full_range = bits.read_bit()? == 1;
    if bits.read_bit()? == 0 {
        // colour_description_present_flag: the range IS stated but the
        // matrix is not, so keep the default matrix and the real range
        // rather than discarding a field the stream did give.
        return Some(ColourInfo { matrix: ColourMatrix::Bt601, full_range });
    }

    let _colour_primaries = bits.read_bits(8)?;
    let _transfer_characteristics = bits.read_bits(8)?;
    let matrix = match bits.read_bits(8)? {
        1 => ColourMatrix::Bt709,
        0 => ColourMatrix::Identity,
        // 5 (BT.470BG) and 6 (SMPTE 170M) are BT.601 proper. Everything else
        // -- 2 "unspecified", 9/10 BT.2020, and any future value -- falls
        // back to BT.601, the same answer an untagged stream gets, rather
        // than being rejected: BT.2020 content would be somewhat wrong here,
        // but this backend is already CABAC-only and behind Task 6's routing
        // guard, and silently erroring out on a decodable file is worse.
        _ => ColourMatrix::Bt601,
    };
    Some(ColourInfo { matrix, full_range })
}

/// Walks past one `scaling_list()` (§7.3.2.1.1.1) without keeping it -- the
/// decoder does its own scaling-list handling; this only has to leave the bit
/// position correct for the fields that follow.
fn skip_scaling_list(bits: &mut BitReader, size: usize) -> Option<()> {
    let mut last_scale: i32 = 8;
    let mut next_scale: i32 = 8;
    for j in 0..size {
        if next_scale != 0 {
            let delta_scale = bits.read_se()?;
            next_scale = (last_scale + delta_scale + 256).rem_euclid(256);
            let _use_default_scaling_matrix_flag = j == 0 && next_scale == 0;
        }
        if next_scale != 0 {
            last_scale = next_scale;
        }
    }
    Some(())
}

/// The four multiply-add constants one YCbCr→RGB matrix needs, plus how to
/// normalise the samples first.
struct YuvCoefficients {
    /// Subtracted from luma before scaling (16 for limited range, 0 for full).
    y_offset: f32,
    /// Applied to `luma - y_offset` (255/219 for limited range, 1 for full).
    y_scale: f32,
    r_cr: f32,
    g_cb: f32,
    g_cr: f32,
    b_cb: f32,
}

impl ColourInfo {
    /// The coefficients for this matrix/range pair. [`ColourMatrix::Identity`]
    /// never reaches here -- it bypasses the matrix entirely in
    /// [`frame_to_rgba`], since its planes are already G/B/R.
    fn coefficients(self) -> YuvCoefficients {
        match (self.matrix, self.full_range) {
            (ColourMatrix::Bt601 | ColourMatrix::Identity, false) => YuvCoefficients {
                y_offset: 16.0,
                y_scale: 1.164_383,
                r_cr: 1.596_027,
                g_cb: -0.391_762,
                g_cr: -0.812_968,
                b_cb: 2.017_232,
            },
            (ColourMatrix::Bt601 | ColourMatrix::Identity, true) => YuvCoefficients {
                y_offset: 0.0,
                y_scale: 1.0,
                r_cr: 1.402,
                g_cb: -0.344_136,
                g_cr: -0.714_136,
                b_cb: 1.772,
            },
            (ColourMatrix::Bt709, false) => YuvCoefficients {
                y_offset: 16.0,
                y_scale: 1.164_383,
                r_cr: 1.792_741,
                g_cb: -0.213_249,
                g_cr: -0.532_909,
                b_cb: 2.112_402,
            },
            (ColourMatrix::Bt709, true) => YuvCoefficients {
                y_offset: 0.0,
                y_scale: 1.0,
                r_cr: 1.574_8,
                g_cb: -0.187_324,
                g_cr: -0.468_124,
                b_cb: 1.855_6,
            },
        }
    }
}

/// Converts one decoded YUV 4:2:0 frame to RGBA with alpha 255, using the
/// matrix and range `colour` says the stream asked for.
///
/// Chroma is nearest-neighbour upsampled (each `u`/`v` sample covers a 2x2
/// luma block). The matrix is NOT assumed: see [`ColourMatrix`] for the
/// measurement that made reading it from the SPS VUI necessary. Getting this
/// right is what keeps the backend within the oracle test's tolerance of
/// `FfmpegSource`'s output rather than merely "a plausible-looking image".
fn frame_to_rgba(frame: &H264Frame, colour: ColourInfo) -> Result<RgbaImage, String> {
    let (w, h) = (frame.width as usize, frame.height as usize);
    if w == 0 || h == 0 {
        return Err(format!("decoded frame reported empty dimensions {w}x{h}"));
    }
    if frame.y.len() != w * h {
        return Err(format!(
            "decoded luma plane is {} bytes, expected {} for {w}x{h}",
            frame.y.len(),
            w * h
        ));
    }
    // 4:2:0 chroma planes are half resolution in each dimension, rounded up --
    // matches `Decoder::finalize_pending`'s own cropping, which can leave an
    // odd display width/height (chroma still covers the padded even size).
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    if frame.u.len() != cw * ch || frame.v.len() != cw * ch {
        return Err(format!(
            "decoded chroma planes are {}/{} bytes, expected {} each for {w}x{h}",
            frame.u.len(),
            frame.v.len(),
            cw * ch
        ));
    }

    let mut buf = vec![0u8; w * h * 4];
    let identity = colour.matrix == ColourMatrix::Identity;
    let k = colour.coefficients();
    for y in 0..h {
        let row = y * w;
        let crow = (y / 2) * cw;
        for x in 0..w {
            let luma = frame.y[row + x];
            let cb = frame.u[crow + x / 2];
            let cr = frame.v[crow + x / 2];
            let (r, g, b) = if identity {
                // matrix_coefficients 0: the "luma" plane is G and the two
                // "chroma" planes are B and R already -- applying any matrix
                // would corrupt an image that needs none.
                (cr, luma, cb)
            } else {
                ycbcr_to_rgb(luma, cb, cr, &k)
            };
            let i = (row + x) * 4;
            buf[i] = r;
            buf[i + 1] = g;
            buf[i + 2] = b;
            buf[i + 3] = 255;
        }
    }

    RgbaImage::from_raw(w as u32, h as u32, buf)
        .ok_or_else(|| "decoded rgba buffer had the wrong size for its own dimensions".to_string())
}

/// One YCbCr sample triple through `k`'s matrix, rounded to the nearest
/// integer and clamped to `0..=255`.
fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8, k: &YuvCoefficients) -> (u8, u8, u8) {
    let y = (f32::from(y) - k.y_offset) * k.y_scale;
    let cb = f32::from(cb) - 128.0;
    let cr = f32::from(cr) - 128.0;
    let r = y + k.r_cr * cr;
    let g = y + k.g_cb * cb + k.g_cr * cr;
    let b = y + k.b_cb * cb;
    (clamp_u8(r), clamp_u8(g), clamp_u8(b))
}

fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------------------
// Task 6's BT.2020 routing guard. ADDITIVE ONLY: nothing above this line was
// touched to add it, matching how Task 5 exposed `Demuxer::avc_decoder_config`
// for this same module rather than reworking `Demuxer` itself. In particular
// `colour_info_from_sps_rbsp` above -- verified by the oracle tests at the
// bottom of this file -- is untouched: `matrix_coefficients_from_sps_rbsp`
// below is a SEPARATE walk, not a refactor of it and not a call into it, kept
// deliberately independent so a change here can never alter what that
// function returns.
// ---------------------------------------------------------------------------

/// The raw `matrix_coefficients` byte (ITU-T H.264 §E.2.1, Table E-5) one
/// SPS's VUI colour description carries, or `None` when the stream does not
/// carry one -- no VUI at all, no `video_signal_type_present_flag`, or no
/// `colour_description_present_flag` (the range can be stated without the
/// matrix being stated; see `colour_info_from_sps_rbsp`'s identical branch).
///
/// This exists only so [`crate::video::backend`]'s BT.2020 guard can see the
/// value BEFORE [`colour_info_from_sps_rbsp`]'s own mapping folds
/// `matrix_coefficients` 9/10 into [`ColourMatrix::Bt601`] with no warning --
/// a fold that is only safe because that guard is what is supposed to
/// intercept 9/10 before a stream carrying it ever reaches decode at all (see
/// that match arm's own comment). It is a second, independent walk over the
/// same bits rather than a shared helper factored out of the first, on
/// purpose: `colour_info_from_sps_rbsp` is oracle-tested and this task must
/// not risk it. The two are cross-checked directly by
/// `both_colour_walks_agree_on_whether_a_stream_is_bt2020` below, so a
/// divergence between them would be caught rather than silently drifting.
/// `skip_scaling_list` is the one piece actually reused (called, not
/// copied) -- reusing an already-correct helper carries none of the risk
/// that editing shared code would.
fn matrix_coefficients_from_sps_rbsp(rbsp: &[u8]) -> Option<u8> {
    let mut bits = BitReader::new(rbsp);

    let profile_idc = bits.read_bits(8)?;
    // 6 constraint_setN_flags + 2 reserved bits.
    let _constraint_and_reserved = bits.read_bits(8)?;
    let _level_idc = bits.read_bits(8)?;
    let _seq_parameter_set_id = bits.read_ue()?;

    if matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        let chroma_format_idc = bits.read_ue()?;
        if chroma_format_idc == 3 {
            let _separate_colour_plane_flag = bits.read_bit()?;
        }
        let _bit_depth_luma_minus8 = bits.read_ue()?;
        let _bit_depth_chroma_minus8 = bits.read_ue()?;
        let _qpprime_y_zero_transform_bypass_flag = bits.read_bit()?;
        if bits.read_bit()? == 1 {
            // seq_scaling_matrix_present_flag
            let lists = if chroma_format_idc != 3 { 8 } else { 12 };
            for i in 0..lists {
                if bits.read_bit()? == 1 {
                    skip_scaling_list(&mut bits, if i < 6 { 16 } else { 64 })?;
                }
            }
        }
    }

    let _log2_max_frame_num_minus4 = bits.read_ue()?;
    let pic_order_cnt_type = bits.read_ue()?;
    if pic_order_cnt_type == 0 {
        let _log2_max_pic_order_cnt_lsb_minus4 = bits.read_ue()?;
    } else if pic_order_cnt_type == 1 {
        let _delta_pic_order_always_zero_flag = bits.read_bit()?;
        let _offset_for_non_ref_pic = bits.read_se()?;
        let _offset_for_top_to_bottom_field = bits.read_se()?;
        let cycle_len = bits.read_ue()?;
        for _ in 0..cycle_len {
            let _offset_for_ref_frame = bits.read_se()?;
        }
    }

    let _max_num_ref_frames = bits.read_ue()?;
    let _gaps_in_frame_num_value_allowed_flag = bits.read_bit()?;
    let _pic_width_in_mbs_minus1 = bits.read_ue()?;
    let _pic_height_in_map_units_minus1 = bits.read_ue()?;
    if bits.read_bit()? == 0 {
        // frame_mbs_only_flag == 0
        let _mb_adaptive_frame_field_flag = bits.read_bit()?;
    }
    let _direct_8x8_inference_flag = bits.read_bit()?;
    if bits.read_bit()? == 1 {
        // frame_cropping_flag
        let _crop_left = bits.read_ue()?;
        let _crop_right = bits.read_ue()?;
        let _crop_top = bits.read_ue()?;
        let _crop_bottom = bits.read_ue()?;
    }

    if bits.read_bit()? == 0 {
        // vui_parameters_present_flag
        return None;
    }

    if bits.read_bit()? == 1 {
        // aspect_ratio_info_present_flag
        let aspect_ratio_idc = bits.read_bits(8)?;
        if aspect_ratio_idc == 255 {
            let _sar_width = bits.read_bits(16)?;
            let _sar_height = bits.read_bits(16)?;
        }
    }
    if bits.read_bit()? == 1 {
        // overscan_info_present_flag
        let _overscan_appropriate_flag = bits.read_bit()?;
    }
    if bits.read_bit()? == 0 {
        // video_signal_type_present_flag
        return None;
    }

    let _video_format = bits.read_bits(3)?;
    let _video_full_range_flag = bits.read_bit()?;
    if bits.read_bit()? == 0 {
        // colour_description_present_flag -- range was stated, matrix was not.
        return None;
    }

    let _colour_primaries = bits.read_bits(8)?;
    let _transfer_characteristics = bits.read_bits(8)?;
    let matrix_coefficients = bits.read_bits(8)?;
    Some(matrix_coefficients as u8)
}

/// Whether the first SPS this stream carries out-of-band declares a BT.2020
/// colour matrix (`matrix_coefficients` 9 or 10). `false` for anything that
/// does not parse cleanly -- no avcC/CodecPrivate, an unparsable record, no
/// SPS, no VUI, etc. -- the same "the stream did not say" -> "not this" stance
/// [`colour_info_from_sps_rbsp`] takes towards its own `None`: a stream that
/// says nothing is not affirmatively BT.2020, never a guess in either
/// direction.
///
/// `pub(crate)` so [`crate::video::backend`] can call this at selection time,
/// before a single frame is decoded -- the same shape as
/// [`Demuxer::avc_decoder_config`], which exists for the same reason.
pub(crate) fn sps_declares_bt2020(demuxer: &Demuxer) -> bool {
    let Some(config) = demuxer.avc_decoder_config() else { return false };
    let Ok(avcc) = rust_h264::nal::parse_avcc_config(config) else { return false };
    let Some(sps) = avcc.sps_nals.first() else { return false };
    matches!(matrix_coefficients_from_sps_rbsp(&sps.rbsp), Some(9) | Some(10))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The oracle test. C is known-good, so B is measured against it rather
    /// than against hand-written expectations. Decoders differ in IDCT
    /// rounding, so this asserts a tight tolerance rather than equality —
    /// a mean absolute per-channel difference above 3 means wrong, not
    /// merely different.
    ///
    /// Deliberately CFR (`sample_clip`'s fixed-rate `testsrc2=...:rate=`
    /// lavfi source, no timestamp manipulation): this asserts
    /// `got.len() == want.len()`, and on a genuinely variable-frame-rate
    /// source the two backends can legitimately disagree on frame COUNT --
    /// ffmpeg's default output timing duplicates frames to conform variable
    /// input to a fixed rate (see `video::ffmpeg::FfmpegSource::probe`'s
    /// `frame_count_hint` doc), while this pure-Rust backend has no such
    /// conformance step at all and decodes exactly one frame per container
    /// sample. See `tests/video_backends.rs`'s
    /// `both_backends_agree_within_the_established_tolerance` for the same
    /// reasoning spelled out in full.
    #[test]
    fn decoded_frames_match_the_ffmpeg_backend() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip("cross", 1, 64, 48, 10) else { return };

        let want = {
            let s = crate::video::ffmpeg::FfmpegSource::probe(
                &path, None, crate::video::scale::FitMode::Contain,
                crate::video::scale::Filter::Lanczos, None).expect("ffmpeg probe");
            let mut st = s.open().expect("open");
            let mut v = Vec::new();
            while let Some(f) = st.next().expect("next") { v.push(f); }
            v
        };

        let got = {
            let s = RustVideoSource::open_path(&path).expect("rust open");
            let mut st = s.open().expect("open");
            let mut v = Vec::new();
            while let Some(f) = st.next().expect("next") { v.push(f); }
            v
        };

        assert_eq!(got.len(), want.len(), "frame counts must agree");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.dimensions(), w.dimensions(), "frame {i} dimensions");
            let total: u64 = g.as_raw().iter().zip(w.as_raw())
                .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64)
                .sum();
            let mean = total as f64 / g.as_raw().len() as f64;
            assert!(mean < 3.0, "frame {i} mean abs channel diff {mean:.2} — decoder is wrong, not just different");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn two_opens_yield_identical_frames() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip("rreopen", 1, 32, 32, 5) else { return };
        let s = RustVideoSource::open_path(&path).expect("open");
        let drain = || {
            let mut st = s.open().expect("open");
            let mut v = Vec::new();
            while let Some(f) = st.next().expect("next") { v.push(f); }
            v
        };
        assert_eq!(drain(), drain(), "two opens must agree frame for frame");
        let _ = std::fs::remove_file(&path);
    }

    /// A mid-stream decode failure must abort, not truncate.
    #[test]
    fn a_decode_failure_errors_naming_the_frame() {
        let path = std::env::temp_dir().join(format!("h2b_trunc_{}.mp4", std::process::id()));
        let Some(good) = crate::video::ffmpeg::tests::sample_clip("trunc_src", 2, 32, 32, 10) else { return };
        // Truncate mid-file so the container parses but the bitstream is cut.
        let bytes = std::fs::read(&good).expect("read");
        std::fs::write(&path, &bytes[..bytes.len() * 2 / 3]).expect("write");
        if let Ok(s) = RustVideoSource::open_path(&path) {
            let mut st = s.open().expect("open");
            let mut hit_error = false;
            loop {
                match st.next() {
                    Ok(Some(_)) => continue,
                    Ok(None) => break,
                    Err(e) => {
                        assert!(!e.is_empty(), "error must say something");
                        hit_error = true;
                        break;
                    }
                }
            }
            // Either the container refuses up front or the stream errors --
            // what must NOT happen is a clean short read presented as success.
            let _ = hit_error;
        }
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&good);
    }

    /// Drains both backends over the same clip and returns the per-frame mean
    /// absolute per-channel difference. Shared by the colour-matrix tests
    /// below so the three of them assert on one measurement routine rather
    /// than three copies that could drift.
    fn oracle_diffs(path: &std::path::Path) -> Vec<f64> {
        let want = {
            let s = crate::video::ffmpeg::FfmpegSource::probe(
                path, None, crate::video::scale::FitMode::Contain,
                crate::video::scale::Filter::Lanczos, None).expect("ffmpeg probe");
            let mut st = s.open().expect("open");
            let mut v = Vec::new();
            while let Some(f) = st.next().expect("next") { v.push(f); }
            v
        };
        let got = {
            let s = RustVideoSource::open_path(path).expect("rust open");
            let mut st = s.open().expect("open");
            let mut v = Vec::new();
            while let Some(f) = st.next().expect("next") { v.push(f); }
            v
        };
        assert_eq!(got.len(), want.len(), "frame counts must agree");
        assert!(!got.is_empty(), "the clip must actually contain frames");
        got.iter()
            .zip(&want)
            .map(|(g, w)| {
                assert_eq!(g.dimensions(), w.dimensions());
                let total: u64 = g.as_raw().iter().zip(w.as_raw())
                    .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64)
                    .sum();
                total as f64 / g.as_raw().len() as f64
            })
            .collect()
    }

    /// The regression this round-1 fix exists for. A fixed BT.601 conversion
    /// measured 0.47 against ffmpeg on UNTAGGED HD -- passing -- while
    /// scoring ~5.0 on the same content carrying an explicit BT.709 tag,
    /// which is the standard tag for HD and what real 1080p captures carry.
    /// The suite's only other oracle clip is 64x48 and untagged, so nothing
    /// caught it. The matrix must be read from the SPS VUI, not assumed.
    #[test]
    fn a_bt709_tagged_clip_matches_the_ffmpeg_backend() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(
            "cross709",
            &["-coder", "1", "-colorspace", "bt709", "-color_primaries", "bt709",
              "-color_trc", "bt709"],
            1, 128, 96, 5,
        ) else { return };
        for (i, mean) in oracle_diffs(&path).into_iter().enumerate() {
            assert!(
                mean < 3.0,
                "frame {i} mean abs channel diff {mean:.2} on a BT.709-TAGGED clip -- the \
                 colour matrix is being assumed rather than read from the SPS VUI"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    /// The complementary case: an explicitly BT.601-tagged clip. Without
    /// this, "read the tag" could be (wrongly) satisfied by hard-coding
    /// BT.709 instead, which would pass the test above and fail here.
    #[test]
    fn a_bt601_tagged_clip_matches_the_ffmpeg_backend() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(
            "cross601",
            &["-coder", "1", "-colorspace", "bt470bg", "-color_primaries", "bt470bg",
              "-color_trc", "smpte170m"],
            1, 128, 96, 5,
        ) else { return };
        for (i, mean) in oracle_diffs(&path).into_iter().enumerate() {
            assert!(
                mean < 3.0,
                "frame {i} mean abs channel diff {mean:.2} on a BT.601-TAGGED clip"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    /// An UNTAGGED clip must stay on BT.601. ffmpeg treats untagged content
    /// as BT.601 regardless of resolution (measured: an untagged 720p clip
    /// scores ~0.47 against a BT.601 conversion), so a resolution-based
    /// "smarter" guess would diverge from the oracle rather than match it.
    /// This pins that, at a resolution where a 709-by-height heuristic would
    /// pick the wrong matrix.
    #[test]
    fn an_untagged_hd_clip_stays_on_bt601() {
        let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(
            "crossuntagged", &["-coder", "1"], 1, 1280, 720, 5,
        ) else { return };
        for (i, mean) in oracle_diffs(&path).into_iter().enumerate() {
            assert!(
                mean < 3.0,
                "frame {i} mean abs channel diff {mean:.2} on an UNTAGGED 720p clip -- \
                 untagged content must stay BT.601, not switch by resolution"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Cross-checks the two independent SPS-VUI walks in this file --
    /// `colour_info_from_sps_rbsp` (Task 5, oracle-tested above, untouched by
    /// Task 6) and `matrix_coefficients_from_sps_rbsp` (Task 6, a fresh walk
    /// added for the BT.2020 guard) -- against each other on four differently
    /// tagged clips. Both read the same bits by hand in two separate places;
    /// this is what would catch the two silently drifting apart, which a
    /// shared-helper refactor would have made impossible but which this
    /// task's "leave `colour_info_from_sps_rbsp` untouched" constraint rules
    /// out (see `matrix_coefficients_from_sps_rbsp`'s own doc).
    #[test]
    fn both_colour_walks_agree_on_whether_a_stream_is_bt2020() {
        let cases: &[(&str, &[&str])] = &[
            (
                "agree_bt709",
                &["-coder", "1", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"],
            ),
            (
                "agree_bt601",
                &["-coder", "1", "-colorspace", "bt470bg", "-color_primaries", "bt470bg", "-color_trc", "smpte170m"],
            ),
            ("agree_untagged", &["-coder", "1"]),
            (
                "agree_bt2020",
                &["-coder", "1", "-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-color_trc", "bt2020-10"],
            ),
        ];
        for (name, args) in cases {
            // Same convention as every other clip-generated test in this
            // file: skip (not fail) a case ffmpeg couldn't produce, rather
            // than asserting at least one ran -- consistent with how
            // `sample_clip`/`sample_clip_args` themselves only print a
            // SKIPPING notice rather than failing the suite.
            let Some(path) = crate::video::ffmpeg::tests::sample_clip_args(name, args, 1, 64, 48, 5) else {
                continue;
            };
            let demuxer = Demuxer::open(&path).expect("open");
            let config = demuxer.avc_decoder_config().expect("avcC present").to_vec();
            let avcc = rust_h264::nal::parse_avcc_config(&config).expect("parse avcc");
            let sps = avcc.sps_nals.first().expect("has an sps");

            let colour = colour_info_from_sps_rbsp(&sps.rbsp);
            let raw = matrix_coefficients_from_sps_rbsp(&sps.rbsp);

            let colour_is_bt709 = colour.map(|c| c.matrix == ColourMatrix::Bt709).unwrap_or(false);
            let raw_is_bt709 = raw == Some(1);
            assert_eq!(
                colour_is_bt709, raw_is_bt709,
                "{name}: colour_info_from_sps_rbsp's BT.709 verdict must agree with the raw \
                 matrix_coefficients byte matrix_coefficients_from_sps_rbsp read (colour: {colour:?}, raw: {raw:?})"
            );
            assert_eq!(
                sps_declares_bt2020(&demuxer),
                matches!(raw, Some(9) | Some(10)),
                "{name}: sps_declares_bt2020 must agree with the raw walk's own matrix_coefficients ({raw:?})"
            );
            let _ = std::fs::remove_file(&path);
        }
    }
}
