//! The pure-Rust [`AudioSource`]: `symphonia` container demux + codec decode,
//! adapted to the one shape everything downstream assumes -- mono `f32` at
//! [`TARGET_RATE`].
//!
//! [`SymphoniaSource`] holds nothing but the path, and every
//! [`AudioSource::open`] re-probes from scratch. That is what makes two opens
//! agree sample for sample, which the two-pass band analysis in
//! [`crate::audio::track`] depends on.
//!
//! This is the only decode path the web build has: there is no ffmpeg on
//! wasm, which is why `symphonia` sits in the unconditional `[dependencies]`
//! block next to `rustfft` rather than the native-only one.

use crate::audio::source::{AudioInfo, AudioSource, AudioStream, first_non_finite};
use std::path::{Path, PathBuf};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, Decoder, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// The one sample rate anything past this module ever sees.
///
/// `StftStream`'s hop arithmetic and `BandPlan`'s bin mapping are both
/// computed from `AudioInfo::sample_rate`, so a source that reported its own
/// rate instead of this one would produce a spectrum mapped to the wrong
/// frequencies -- quietly, with every band still finite and in range.
pub const TARGET_RATE: u32 = 48_000;

/// A re-openable handle to an audio file on disk.
///
/// Holds the path and the probed metadata, and deliberately no reader,
/// decoder or decoded state: [`AudioSource::open`] re-probes every time.
#[derive(Debug, Clone)]
pub struct SymphoniaSource {
    path: PathBuf,
    info: AudioInfo,
}

impl SymphoniaSource {
    /// Probes `path` once to confirm it holds a decodable audio track and to
    /// read its duration, and builds a re-openable source over it.
    ///
    /// Everything that can be wrong with the file -- missing, not audio, no
    /// audio track, a codec this build was not compiled with -- is an error
    /// here, at open time, rather than a surprise part-way through a render.
    pub fn open_path(path: &Path) -> Result<Self, String> {
        let probed = probe_file(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            info: AudioInfo {
                // Always the adapted rate, never `probed.rate`. Every stream
                // this source hands out resamples to it.
                sample_rate: TARGET_RATE,
                duration_hint: probed.duration,
            },
        })
    }
}

impl AudioSource for SymphoniaSource {
    fn info(&self) -> AudioInfo {
        self.info
    }

    fn open(&self) -> Result<Box<dyn AudioStream + '_>, String> {
        let probed = probe_file(&self.path)?;
        Ok(Box::new(SymphoniaStream {
            label: self.path.display().to_string(),
            format: probed.format,
            decoder: probed.decoder,
            track_id: probed.track_id,
            resampler: LinearResampler::default(),
            sample_buf: None,
            buf_shape: None,
            mono: Vec::new(),
            frames_seen: 0,
            done: false,
        }))
    }
}

/// One probe's worth of state: a demuxer positioned at the start of the file,
/// a decoder for its audio track, and what little metadata the container can
/// state without decoding.
struct Probed {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    track_id: u32,
    duration: Option<f64>,
}

/// Opens and probes `path`, building a demuxer and a decoder for its first
/// real audio track.
///
/// Called by both [`SymphoniaSource::open_path`] and [`AudioSource::open`],
/// from the same path with no retained state in between, which is exactly
/// what makes two opens agree sample for sample.
fn probe_file(path: &Path) -> Result<Probed, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());

    // The extension is a hint, not a decision: symphonia still sniffs the
    // actual bytes, so a `.wav` full of MP3 (or of junk, as the tests check)
    // is handled on its contents.
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    // `enable_gapless` applies the encoder delay/padding trim that MP3's
    // LAME/Xing header (and Ogg/FLAC's equivalents) describe, so the decoded
    // sample count matches the track's real duration instead of running a few
    // dozen milliseconds long with silence on both ends.
    let format_opts = FormatOptions { enable_gapless: true, ..Default::default() };
    let probe = symphonia::default::get_probe()
        .format(&hint, stream, &format_opts, &MetadataOptions::default())
        .map_err(|e| format!("{} is not a supported audio file: {e}", path.display()))?;
    let format = probe.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| format!("{} contains no audio track", path.display()))?;
    let track_id = track.id;
    let params = track.codec_params.clone();

    let decoder = symphonia::default::get_codecs()
        .make(&params, &DecoderOptions::default())
        .map_err(|e| format!("no decoder for the audio in {}: {e}", path.display()))?;

    // A hint, so `None` is a real answer rather than a failure -- it drives
    // spinner-vs-bar in progress, and `track::analyze` treats it as optional.
    // Never guessed from file size: a wrong duration silently mis-sizes the
    // progress bar and, worse, the frame-count estimate built on it.
    let duration = match (params.n_frames, params.sample_rate) {
        (Some(n), Some(rate)) if rate > 0 => Some(n as f64 / rate as f64),
        (Some(n), _) => params.time_base.map(|tb| {
            let t = tb.calc_time(n);
            t.seconds as f64 + t.frac
        }),
        _ => None,
    };

    Ok(Probed { format, decoder, track_id, duration })
}

/// A one-shot cursor over one probe's worth of demuxer + decoder.
struct SymphoniaStream {
    /// The path, pre-rendered for error messages -- the stream outlives no
    /// borrow of the source, and an error naming no file is useless.
    label: String,
    format: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    track_id: u32,
    /// Carried across packets on purpose. See [`LinearResampler`].
    resampler: LinearResampler,
    /// Reused between packets: interleaved `f32` in the decoder's own layout.
    sample_buf: Option<SampleBuffer<f32>>,
    /// `(rate, channels, capacity_frames)` the current `sample_buf` was built
    /// for. Tracked here rather than inferred from the buffer so a mid-stream
    /// format change reallocates instead of writing past the end.
    buf_shape: Option<(u32, usize, u64)>,
    /// Reused between packets: the downmixed mono block.
    mono: Vec<f32>,
    /// Source-rate frames decoded so far, so a rejected sample can be named
    /// by its index in the source rather than by its offset in some packet
    /// the caller never sees.
    frames_seen: u64,
    done: bool,
}

impl AudioStream for SymphoniaStream {
    fn next(&mut self) -> Result<Option<Vec<f32>>, String> {
        if self.done {
            return Ok(None);
        }
        // Loops rather than returning: a packet can legitimately yield no
        // output samples (another track's packet, a packet gapless-trimmed to
        // nothing, or -- when downsampling -- a packet shorter than one
        // output step). `AudioStream`'s contract allows any non-zero block
        // length, so an empty `Vec` would be a lie about the stream ending.
        loop {
            let packet = match self.format.next_packet() {
                Ok(p) => p,
                // The end of the stream. Symphonia spells it as an EOF read
                // error rather than a distinct variant.
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    self.done = true;
                    return Ok(None);
                }
                Err(e) => {
                    self.done = true;
                    return Err(format!("reading audio from {}: {e}", self.label));
                }
            };
            if packet.track_id() != self.track_id {
                continue;
            }

            let decoded = match self.decoder.decode(&packet) {
                Ok(d) => d,
                // Symphonia documents `DecodeError` as recoverable: the
                // packet is malformed, the next one may be fine. Real MP3s in
                // the wild routinely carry a junk frame or two at a splice
                // point, and failing the whole render over one is worse than
                // dropping ~26 ms. It is logged, never silent.
                Err(SymphoniaError::DecodeError(e)) => {
                    log::warn!("skipping a malformed audio packet in {}: {e}", self.label);
                    continue;
                }
                Err(e) => {
                    self.done = true;
                    return Err(format!("decoding audio from {}: {e}", self.label));
                }
            };

            let spec = *decoded.spec();
            let channels = spec.channels.count();
            if channels == 0 || spec.rate == 0 {
                self.done = true;
                return Err(format!(
                    "{} decoded a block with {channels} channels at {} Hz",
                    self.label, spec.rate
                ));
            }
            let shape = (spec.rate, channels, decoded.capacity() as u64);
            if self.buf_shape != Some(shape) {
                self.sample_buf = Some(SampleBuffer::new(shape.2, spec));
                self.buf_shape = Some(shape);
            }
            let sample_buf = self.sample_buf.as_mut().expect("just built above");
            sample_buf.copy_interleaved_ref(decoded);

            // Mono is the average of the channels, never their sum. Summing a
            // correlated stereo pair -- which nearly every mixed track is,
            // especially in the bass -- doubles it straight into clipping,
            // and the clipping happens here, before any of the normalisation
            // downstream could scale it back.
            let inv = 1.0 / channels as f32;
            self.mono.clear();
            self.mono.extend(
                sample_buf
                    .samples()
                    .chunks_exact(channels)
                    .map(|frame| frame.iter().sum::<f32>() * inv),
            );

            // Non-finite samples die here, at the source: `f32::NAN.min(1.0)`
            // is `1.0`, so `track::analyze`'s clamp would otherwise launder a
            // single NaN into a whole bank of speakers at maximum volume.
            if let Some((at, value)) = first_non_finite(&self.mono, self.frames_seen) {
                self.done = true;
                return Err(format!(
                    "{} decoded a non-finite sample ({value}) at source sample {at} \
                     ({:.3}s in) -- the file is corrupt or truncated",
                    self.label,
                    at as f64 / spec.rate as f64,
                ));
            }
            self.frames_seen += self.mono.len() as u64;

            let mut out = Vec::new();
            self.resampler.push(&self.mono, spec.rate, &mut out);
            if !out.is_empty() {
                return Ok(Some(out));
            }
        }
    }
}

/// Linear interpolation from the source rate to [`TARGET_RATE`], with the
/// read position carried across packet boundaries.
///
/// Linear is adequate here because the save gets band energies, not a raw
/// waveform, so interpolation error does not matter the way it would for
/// audio samples.
///
/// State (`pos_*`, `consumed`, `prev`) must persist across packets, or
/// resetting the read position at every boundary buzzes at the packet rate.
///
/// The position is an exact rational (`pos_int + pos_frac / TARGET_RATE`)
/// rather than an accumulated `f64`, specifically so it cannot drift over the
/// length of a long file.
#[derive(Debug, Default)]
struct LinearResampler {
    /// Global index of the input sample at or before the next output.
    pos_int: u64,
    /// Fractional part of the read position, over [`TARGET_RATE`].
    pos_frac: u32,
    /// Global index one past the last input sample consumed.
    consumed: u64,
    /// The last input sample of the previous block -- the left-hand end of
    /// the interpolation for an output that straddles the boundary.
    prev: f32,
}

impl LinearResampler {
    /// Appends the outputs that `block` (at `rate` Hz) makes available.
    ///
    /// Emits only outputs whose interpolation ends are both already known,
    /// so the last sample of every block is held back in `prev` and used by
    /// the next call. Nothing is dropped and nothing is re-read.
    fn push(&mut self, block: &[f32], rate: u32, out: &mut Vec<f32>) {
        if block.is_empty() {
            return;
        }
        let base = self.consumed;
        // The global index of the last sample this block makes readable.
        let last = base + block.len() as u64 - 1;

        // Invariant: `pos_int >= base - 1` on entry (the loop below only
        // stops once the next output needs a sample this block does not
        // have), so the left end is either in `block` or is exactly `prev`.
        debug_assert!(base == 0 || self.pos_int + 1 >= base);
        while self.pos_int + 1 <= last {
            let i = self.pos_int;
            let left = if i >= base { block[(i - base) as usize] } else { self.prev };
            let right = block[(i + 1 - base) as usize];
            let t = self.pos_frac as f32 / TARGET_RATE as f32;
            out.push(left + (right - left) * t);

            // pos += rate / TARGET_RATE, in exact integer arithmetic.
            self.pos_frac += rate;
            self.pos_int += (self.pos_frac / TARGET_RATE) as u64;
            self.pos_frac %= TARGET_RATE;
        }

        self.consumed = base + block.len() as u64;
        self.prev = block[block.len() - 1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::source::AudioSource;
    use std::f32::consts::TAU;
    use std::io::Write;

    /// A 16-bit mono WAV of a 1 kHz tone, written to the temp dir. WAV is
    /// used because it can be synthesised here byte-for-byte with no encoder
    /// dependency, which keeps this test hermetic.
    fn wav_tone(name: &str, sr: u32, secs: f32) -> std::path::PathBuf {
        let n = (sr as f32 * secs) as usize;
        let samples: Vec<i16> = (0..n)
            .map(|i| ((TAU * 1000.0 * i as f32 / sr as f32).sin() * 16000.0) as i16)
            .collect();
        let data_len = (samples.len() * 2) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM
        out.extend_from_slice(&1u16.to_le_bytes()); // mono
        out.extend_from_slice(&sr.to_le_bytes());
        out.extend_from_slice(&(sr * 2).to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for s in samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        let path = std::env::temp_dir().join(format!("h2b_audio_{name}_{}.wav", std::process::id()));
        std::fs::File::create(&path).expect("create").write_all(&out).expect("write");
        path
    }

    #[test]
    fn a_wav_decodes_to_mono_48k() {
        let path = wav_tone("mono48", 48_000, 0.5);
        let src = SymphoniaSource::open_path(&path).expect("open");
        assert_eq!(src.info().sample_rate, 48_000);
        let mut st = src.open().expect("stream");
        let mut n = 0;
        while let Some(b) = st.next().expect("next") {
            n += b.len();
        }
        assert!(
            (23_000..=25_000).contains(&n),
            "0.5s at 48kHz should be ~24000 samples, got {n}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// A source at a different rate must be resampled, not passed through --
    /// the STFT's hop arithmetic assumes 48 kHz.
    #[test]
    fn a_44100_source_is_resampled_to_48k() {
        let path = wav_tone("mono44", 44_100, 0.5);
        let src = SymphoniaSource::open_path(&path).expect("open");
        assert_eq!(src.info().sample_rate, 48_000, "everything adapts to 48 kHz");
        let mut st = src.open().expect("stream");
        let mut n = 0;
        while let Some(b) = st.next().expect("next") {
            n += b.len();
        }
        assert!(
            (23_000..=25_000).contains(&n),
            "0.5s resampled to 48kHz should be ~24000 samples, got {n}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn two_streams_over_one_source_agree() {
        let path = wav_tone("twice", 48_000, 0.2);
        let src = SymphoniaSource::open_path(&path).expect("open");
        let drain = || {
            let mut st = src.open().expect("stream");
            let mut out = Vec::new();
            while let Some(b) = st.next().expect("next") {
                out.extend_from_slice(&b);
            }
            out
        };
        assert_eq!(drain(), drain());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_missing_file_is_an_error_not_a_panic() {
        let missing = std::env::temp_dir().join("h2b_audio_definitely_not_here.wav");
        assert!(SymphoniaSource::open_path(&missing).is_err());
    }

    #[test]
    fn a_non_audio_file_is_an_error() {
        let path = std::env::temp_dir().join(format!("h2b_audio_junk_{}.wav", std::process::id()));
        std::fs::write(&path, b"this is not audio at all").expect("write");
        assert!(SymphoniaSource::open_path(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    // --- The tests above are all mono, all length-range assertions, and all
    // --- on files with no non-finite sample in them -- a shape that cannot
    // --- see a bad downmix, a per-packet resampler reset, or a NaN. The
    // --- tests below target those specifically.

    /// A WAV of arbitrary format tag / channel count / bit depth. Same
    /// hermetic, no-encoder-dependency approach as `wav_tone`, generalised so
    /// the stereo and IEEE-float cases below need no new file format.
    fn write_wav(
        name: &str,
        fmt_tag: u16,
        sr: u32,
        channels: u16,
        bits: u16,
        data: &[u8],
    ) -> std::path::PathBuf {
        // Non-PCM tags need the `cbSize` field, so the chunk is 18 bytes for
        // IEEE float and the classic 16 for integer PCM.
        let fmt_len: u32 = if fmt_tag == 1 { 16 } else { 18 };
        let block_align = channels * (bits / 8);
        let data_len = data.len() as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(4 + (8 + fmt_len) + (8 + data_len)).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&fmt_len.to_le_bytes());
        out.extend_from_slice(&fmt_tag.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sr.to_le_bytes());
        out.extend_from_slice(&(sr * block_align as u32).to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        if fmt_len == 18 {
            out.extend_from_slice(&0u16.to_le_bytes()); // cbSize
        }
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        out.extend_from_slice(data);
        let path = std::env::temp_dir().join(format!("h2b_audio_{name}_{}.wav", std::process::id()));
        std::fs::File::create(&path).expect("create").write_all(&out).expect("write");
        path
    }

    fn drain(src: &SymphoniaSource) -> Vec<f32> {
        let mut st = src.open().expect("stream");
        let mut out = Vec::new();
        while let Some(b) = st.next().expect("next") {
            out.extend_from_slice(&b);
        }
        out
    }

    /// The samples `wav_tone` writes, as the decoder should hand them back.
    fn tone_samples(sr: u32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| ((TAU * 1000.0 * i as f32 / sr as f32).sin() * 16000.0) as i16 as f32 / 32768.0)
            .collect()
    }

    /// Linear resampling to 48 kHz computed offline, over the whole signal at
    /// once, with the read position recomputed from scratch for every output
    /// rather than carried in a state machine. The oracle for the streaming
    /// implementation: any packet-boundary bug shows up as divergence here
    /// even though the two agree on length to within a sample or two.
    fn reference_resample(x: &[f32], rate: u32) -> Vec<f32> {
        let mut out = Vec::new();
        let mut k: u64 = 0;
        loop {
            let num = k * rate as u64;
            let i = (num / TARGET_RATE as u64) as usize;
            let frac = (num % TARGET_RATE as u64) as f32 / TARGET_RATE as f32;
            if i + 1 >= x.len() {
                return out;
            }
            out.push(x[i] + (x[i + 1] - x[i]) * frac);
            k += 1;
        }
    }

    /// The resampler's fractional read position and the last sample of the
    /// previous packet must survive across packets, or every packet restarts
    /// from its own first sample -- a discontinuity that a plain length
    /// assertion cannot see.
    ///
    /// Compared against an offline reference rather than a length. A 1 kHz
    /// tone is used because its sample-to-sample delta is bounded, so a phase
    /// slip of even a fraction of one input sample is a visible outlier.
    #[test]
    fn resampling_is_continuous_across_packet_boundaries() {
        let path = wav_tone("cont44", 44_100, 0.5);
        let src = SymphoniaSource::open_path(&path).expect("open");
        let got = drain(&src);
        let want = reference_resample(&tone_samples(44_100, 0.5), 44_100);

        assert_eq!(got.len(), want.len(), "the streamed and offline resamplers must agree on length");
        // Non-vacuous: the tone really is at full level, so a phase slip has
        // something to be visible against.
        let peak = want.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!((peak - 16_000.0 / 32_768.0).abs() < 0.01, "the tone's peak should be ~0.488, got {peak}");

        let (at, err) = got
            .iter()
            .zip(&want)
            .enumerate()
            .map(|(i, (a, b))| (i, (a - b).abs()))
            .fold((0, 0.0f32), |acc, x| if x.1 > acc.1 { x } else { acc });
        assert!(
            err < 1e-3,
            "sample {at} is {err} away from the offline resample -- the streaming resampler's \
             position or previous-sample state is not surviving the packet boundary"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The same property at the unit level, and the one that pins it hardest:
    /// how the input is chunked must not change the output at all. Any
    /// per-packet reset makes irregular chunking diverge from one big push.
    ///
    /// Swept across every rate this project is likely to meet, in both
    /// directions. Downsampling is the case the end-to-end tests cannot
    /// reach and where the state machine is least obvious: a packet shorter
    /// than one output step produces no output at all, and the read position
    /// then sits past the end of that packet when the next one arrives.
    #[test]
    fn chunking_does_not_change_the_resampled_output() {
        for rate in [44_100u32, 48_000, 96_000, 192_000, 32_000, 22_050, 8_000] {
            let x: Vec<f32> =
                (0..5_000).map(|i| (TAU * 997.0 * i as f32 / rate as f32).sin()).collect();

            let mut whole = Vec::new();
            LinearResampler::default().push(&x, rate, &mut whole);

            // Irregular on purpose: a fixed chunk size that happens to divide
            // evenly into the resampling ratio could hide a reset. The
            // single-sample pattern is the extreme case -- at 192 kHz most of
            // those pushes emit nothing at all.
            for sizes in [&[1usize, 2, 3, 5, 7, 11, 13][..], &[1152][..], &[4096, 1][..], &[1][..]] {
                let mut chunked = Vec::new();
                let mut r = LinearResampler::default();
                let mut at = 0;
                let mut i = 0;
                while at < x.len() {
                    let end = (at + sizes[i % sizes.len()]).min(x.len());
                    r.push(&x[at..end], rate, &mut chunked);
                    at = end;
                    i += 1;
                }
                assert_eq!(chunked, whole, "{rate} Hz: chunk pattern {sizes:?} changed the output");
            }

            // The ratio really is applied, and in the right direction. Exact,
            // not approximate: outputs are the `k` with `k*rate/48000` strictly
            // below the last input index 4999, since the final input sample
            // has no right-hand end to interpolate towards and is held back.
            let want = (4_999u64 * TARGET_RATE as u64).div_ceil(rate as u64) as usize;
            assert_eq!(
                whole.len(),
                want,
                "{rate} Hz: 5000 input samples must resample to exactly {want} at 48 kHz"
            );
        }
    }

    /// Mono is the average of the channels, not their sum and not the left
    /// channel. A constant-DC stereo pair with unequal channels separates
    /// all three by construction (avg 0.3, sum 0.6, left 0.5).
    #[test]
    fn stereo_is_downmixed_by_averaging_not_summing() {
        let (l, r) = (16_384i16, 3_277i16); // 0.5 and ~0.1
        let mut data = Vec::new();
        for _ in 0..24_000 {
            data.extend_from_slice(&l.to_le_bytes());
            data.extend_from_slice(&r.to_le_bytes());
        }
        let path = write_wav("stereo", 1, 48_000, 2, 16, &data);
        let src = SymphoniaSource::open_path(&path).expect("open");
        let got = drain(&src);
        assert!(!got.is_empty(), "the file must decode to something");
        let want = (l as f32 + r as f32) / 2.0 / 32_768.0;
        for (i, &s) in got.iter().enumerate() {
            assert!(
                (s - want).abs() < 0.01,
                "sample {i} is {s}, expected the channel average {want} \
                 (a sum would be {}, the left channel alone {})",
                want * 2.0,
                l as f32 / 32_768.0
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    /// A NaN reaching the analyser is laundered into full scale by
    /// `track::analyze`'s `min(1.0)` clamp -- `f32::NAN.min(1.0)` is `1.0` --
    /// so one corrupt sample silently slams the whole speaker bank to
    /// maximum volume. It has to die at the decoder, naming the sample.
    ///
    /// A 32-bit IEEE float WAV is the hermetic way to get a real decoder to
    /// emit one: the bit pattern is carried through the PCM path verbatim,
    /// exactly as a corrupt or truncated compressed file would produce it.
    #[test]
    fn a_non_finite_sample_is_rejected_naming_its_index() {
        for (label, bad) in [("nan", f32::NAN), ("inf", f32::INFINITY)] {
            let mut samples = vec![0.25f32; 4_000];
            samples[1_234] = bad;
            let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
            let path = write_wav(&format!("{label}_sample"), 3, 48_000, 1, 32, &data);

            // The file itself is decodable -- this is not a probe failure.
            let src = SymphoniaSource::open_path(&path).expect("open");
            let mut st = src.open().expect("stream");
            let mut err = None;
            loop {
                match st.next() {
                    Ok(Some(_)) => continue,
                    Ok(None) => break,
                    Err(e) => {
                        err = Some(e);
                        break;
                    }
                }
            }
            let err = err.unwrap_or_else(|| {
                panic!("a {label} sample must be a hard error, not passed downstream")
            });
            assert!(
                err.contains("1234"),
                "the error must name the offending sample's index, got: {err}"
            );
            let _ = std::fs::remove_file(&path);
        }
    }

    /// The index in that error is a global sample index, not an offset into
    /// whichever packet happened to contain it -- a per-packet offset points
    /// at nothing the caller can find.
    #[test]
    fn a_rejected_sample_is_indexed_globally() {
        assert_eq!(first_non_finite(&[0.0, 1.0, -1.0], 0), None);
        // NaN is not equal to itself, so the value is checked by predicate.
        let (at, value) = first_non_finite(&[0.0, f32::NAN], 5_000).expect("a NaN is non-finite");
        assert_eq!(at, 5_001, "the index must be global, not the offset within the block");
        assert!(value.is_nan());
        assert_eq!(
            first_non_finite(&[0.0, 0.0, f32::NEG_INFINITY], 10),
            Some((12, f32::NEG_INFINITY))
        );
    }

    /// Nothing is dropped at the end of the stream. A loose length range is
    /// wide enough for a whole final packet (1152 frames for MP3, 4096 for
    /// this WAV reader) to go missing unnoticed, which is precisely the tail
    /// a "write a save missing its end" bug looks like.
    #[test]
    fn the_final_partial_packet_is_not_dropped() {
        // 0.5s at each rate; only the last input sample is legitimately held
        // back, since it has no right-hand end to interpolate towards.
        // 48 kHz: 24000 in, 24000 out minus the unpaired last. 44.1 kHz:
        // 22050 in, and the last output is the largest k with
        // floor(k*147/160) <= 22048, i.e. k = 23998 -- 23999 samples.
        for (rate, want) in [(48_000u32, 23_999usize), (44_100, 23_999)] {
            let path = wav_tone(&format!("tail{rate}"), rate, 0.5);
            let n = drain(&SymphoniaSource::open_path(&path).expect("open")).len();
            assert_eq!(
                n, want,
                "{rate} Hz: expected every sample but the unpaired last one; a shortfall of a \
                 whole packet means the tail was truncated"
            );
            let _ = std::fs::remove_file(&path);
        }
    }

    /// `duration_hint` is the source's real duration in seconds, not its
    /// sample count and not a value scaled by the resampling ratio.
    #[test]
    fn the_duration_hint_is_in_seconds_at_the_source_rate() {
        for rate in [48_000u32, 44_100] {
            let path = wav_tone(&format!("dur{rate}"), rate, 0.5);
            let hint = SymphoniaSource::open_path(&path).expect("open").info().duration_hint;
            let hint = hint.unwrap_or_else(|| panic!("{rate} Hz: a WAV can state its duration"));
            assert!(
                (hint - 0.5).abs() < 0.01,
                "{rate} Hz: 0.5s of audio should hint 0.5, got {hint}"
            );
            let _ = std::fs::remove_file(&path);
        }
    }
}
