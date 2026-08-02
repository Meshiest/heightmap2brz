//! Short-time Fourier transform: mono samples in, magnitude spectra out.
//!
//! One spectrum per output frame. The window is Hann, and consecutive
//! windows overlap by `window - hop` samples -- at the defaults (4096 window,
//! 1600 hop for 30 fps at 48 kHz) that is 61% overlap and an 85.3 ms window,
//! giving 11.72 Hz bins. The hop is honoured exactly even when it exceeds the
//! window (low `--audio-fps`), in which case the samples between windows are
//! skipped rather than folded into the next one.
//!
//! A partial trailing window is DROPPED rather than zero-padded. Padding
//! would emit a final frame whose energy is artificially low, which reads as
//! a fade-out that is not in the source.
//!
//! Magnitudes are the UNNORMALISED `|X[k]|` that `rustfft` returns: no `1/N`
//! division and no window-gain compensation. Nothing downstream depends on
//! the absolute scale -- band levels are normalised against the track's own
//! peak, so only a spectrum's relative shape matters.
use super::source::AudioStream;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

/// Samples advanced per output frame.
///
/// Returns `Err` rather than a nonsense hop for a rate that is not a positive
/// finite number. `0.0` would otherwise produce `usize::MAX` and `NaN` would
/// produce `1` -- and NaN slips past a plain `fps <= 0.0` guard, because every
/// comparison against NaN is false.
pub fn hop_for(sample_rate: u32, fps: f32) -> Result<usize, String> {
    if !fps.is_finite() || fps <= 0.0 {
        return Err(format!("frame rate must be a positive finite number, got {fps}"));
    }
    Ok((sample_rate as f32 / fps).round().max(1.0) as usize)
}

/// How many frames a stream of `duration_s` actually emits.
///
/// This MUST match `StftStream` exactly, because it is the progress bar's
/// denominator: the stream emits one spectrum once it holds a full window,
/// then advances by `hop`. So `n` samples yield `1 + (n - window) / hop`
/// frames, and none at all below one full window.
pub fn frame_count_for(duration_s: f64, sample_rate: u32, window: usize, hop: usize) -> usize {
    if duration_s <= 0.0 || hop == 0 {
        return 0;
    }
    let n = (duration_s * sample_rate as f64).floor() as usize;
    if n < window {
        return 0;
    }
    1 + (n - window) / hop
}

pub struct StftStream<'a> {
    inner: Box<dyn AudioStream + 'a>,
    /// Precomputed Hann coefficients, `window` long.
    window_fn: Vec<f32>,
    window: usize,
    hop: usize,
    /// Samples not yet consumed by a window.
    buf: Vec<f32>,
    /// Samples still owed to the CURRENT advance, when `hop > window` left
    /// the buffer short. Consumed off the front of incoming blocks before
    /// they are appended.
    ///
    /// This field is the whole reason the hop is honest. Draining
    /// `hop.min(buf.len())` instead -- which an earlier version did, to stop
    /// `drain` panicking -- silently under-advances whenever `hop > window`,
    /// by an amount set by the DECODER'S BLOCK SIZE.
    pending_skip: usize,
    fft: Arc<dyn Fft<f32>>,
    drained: bool,
}

impl<'a> StftStream<'a> {
    pub fn new(inner: Box<dyn AudioStream + 'a>, window: usize, hop: usize) -> Result<Self, String> {
        if window < 2 {
            return Err(format!("STFT window must be at least 2, got {window}"));
        }
        if window % 2 != 0 {
            return Err(format!("STFT window must be even, got {window}"));
        }
        if hop == 0 {
            return Err("STFT hop must be non-zero (check --audio-fps)".to_string());
        }
        // Periodic Hann (divisor `window`, not `window - 1`): the symmetric
        // form would make overlapping windows sum unevenly.
        let window_fn = (0..window)
            .map(|i| {
                0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / window as f32).cos()
            })
            .collect();
        let fft = FftPlanner::<f32>::new().plan_fft_forward(window);
        Ok(Self {
            inner,
            window_fn,
            window,
            hop,
            buf: Vec::with_capacity(window * 2),
            pending_skip: 0,
            fft,
            drained: false,
        })
    }

    /// The next magnitude spectrum (`window / 2 + 1` bins), or `None` once
    /// fewer than a full window of samples remains.
    pub fn next_spectrum(&mut self) -> Result<Option<Vec<f32>>, String> {
        while self.buf.len() < self.window {
            if self.drained {
                return Ok(None);
            }
            match self.inner.next()? {
                Some(block) => {
                    if block.is_empty() {
                        // The contract says blocks are non-zero length. A
                        // decoder that breaks it would spin this loop
                        // forever, so refuse loudly instead of hanging.
                        return Err(
                            "audio stream produced an empty block; blocks must be non-empty"
                                .to_string(),
                        );
                    }
                    // Pay down any skip owed from a hop larger than the
                    // window BEFORE buffering, so skipped samples never
                    // enter a window.
                    let skipped = self.pending_skip.min(block.len());
                    self.pending_skip -= skipped;
                    if skipped < block.len() {
                        self.buf.extend_from_slice(&block[skipped..]);
                    }
                }
                None => {
                    self.drained = true;
                    if self.buf.len() < self.window {
                        return Ok(None);
                    }
                }
            }
        }

        let mut scratch: Vec<Complex<f32>> = (0..self.window)
            .map(|i| Complex { re: self.buf[i] * self.window_fn[i], im: 0.0 })
            .collect();
        self.fft.process(&mut scratch);

        let bins = self.window / 2 + 1;
        let out: Vec<f32> = scratch[..bins].iter().map(|c| c.norm()).collect();

        // Advance by exactly `hop`. When the hop overruns what is buffered
        // (only possible for `hop > window`), clear it and carry the
        // remainder as a skip against future blocks rather than clamping.
        if self.hop <= self.buf.len() {
            self.buf.drain(..self.hop);
        } else {
            self.pending_skip = self.hop - self.buf.len();
            self.buf.clear();
        }
        Ok(Some(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::source::{AudioSource, AudioStream, SampleClip};
    use std::f32::consts::TAU;

    const SR: u32 = 48_000;

    fn sine(freq: f32, secs: f32) -> SampleClip {
        let n = (SR as f32 * secs) as usize;
        SampleClip::new(
            SR,
            (0..n).map(|i| (TAU * freq * i as f32 / SR as f32).sin()).collect(),
        )
    }

    fn spectra(clip: &SampleClip, window: usize, hop: usize) -> Vec<Vec<f32>> {
        let mut s = StftStream::new(clip.open().expect("open"), window, hop).expect("stft");
        let mut out = Vec::new();
        while let Some(sp) = s.next_spectrum().expect("spectrum") {
            out.push(sp);
        }
        out
    }

    /// The bin a pure tone lands in is the single most load-bearing fact in
    /// the whole pipeline: every band mapping downstream is indexed off it.
    ///
    /// Asserted as EXACT equality, at four window sizes. A `<= 1` tolerance
    /// is precisely the size of the error it is meant to catch: a systematic
    /// one-bin shift of every bin would slide straight through it, and
    /// checking one window size alone lets a shift hide as a rounding
    /// artefact of that particular size.
    #[test]
    fn a_pure_tone_peaks_in_the_exact_expected_bin() {
        let freq = 1000.0f32;
        let clip = sine(freq, 0.5);
        for (window, expected) in [(1024usize, 21usize), (2048, 43), (4096, 85), (8192, 171)] {
            assert_eq!(
                expected,
                (freq as f64 * window as f64 / SR as f64).round() as usize,
                "the hard-coded expectation must be the analytic bin"
            );
            let sp = spectra(&clip, window, 1600);
            assert!(!sp.is_empty(), "window {window}: a 0.5s clip must produce spectra");
            let mid = &sp[sp.len() / 2];
            let peak = mid
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .expect("a non-empty spectrum");
            assert_eq!(
                peak, expected,
                "1000 Hz at window {window} must peak exactly at bin {expected}, got {peak}"
            );
        }
    }

    /// `norm()` gives a phase-independent magnitude; `re.abs()` does not.
    /// Dropping the imaginary part would make a steady tone's level flicker
    /// frame to frame -- audible as tremolo on every sustained note, with
    /// nothing erroring anywhere. No other test distinguishes the two, and a
    /// magnitude BOUND cannot: the scale is deliberately unnormalised, so
    /// `norm_sqr()` and a `1/window` factor are legitimate and must keep
    /// passing.
    #[test]
    fn a_steady_tones_magnitude_is_stable_across_frames() {
        let window = 4096;
        let sp = spectra(&sine(1000.0, 2.0), window, 1600);
        assert!(sp.len() >= 10, "need enough frames to see flicker, got {}", sp.len());
        let bin = (1000.0f64 * window as f64 / SR as f64).round() as usize;
        let peaks: Vec<f32> = sp.iter().map(|s| s[bin]).collect();
        let lo = peaks.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = peaks.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            hi / lo < 1.05,
            "a steady tone's peak magnitude must not vary across frames \
             (min {lo}, max {hi}, ratio {}) -- a phase-dependent magnitude such as \
             |re| instead of |X[k]| makes a sustained note flicker",
            hi / lo
        );
    }

    #[test]
    fn a_spectrum_has_half_the_window_plus_one_bins() {
        let sp = spectra(&sine(440.0, 0.5), 2048, 800);
        assert!(!sp.is_empty());
        assert_eq!(sp[0].len(), 2048 / 2 + 1);
    }

    /// Silence must be silent all the way through. A floor of numerical
    /// noise here becomes 32 speakers of audible hiss downstream.
    #[test]
    fn silence_produces_no_energy() {
        let clip = SampleClip::new(SR, vec![0.0; SR as usize]);
        for sp in spectra(&clip, 2048, 800) {
            for m in sp {
                assert!(m < 1e-6, "silence must not produce energy, got {m}");
            }
        }
    }

    /// Two tones must both appear. A single-peak-only implementation would
    /// pass the first test and fail here.
    #[test]
    fn two_tones_both_appear() {
        let window = 4096;
        let n = SR as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                0.5 * (TAU * 500.0 * t).sin() + 0.5 * (TAU * 2000.0 * t).sin()
            })
            .collect();
        let sp = spectra(&SampleClip::new(SR, samples), window, 1600);
        let mid = &sp[sp.len() / 2];
        let bin = |f: f32| (f as f64 * window as f64 / SR as f64).round() as usize;
        let quiet: f32 = mid[bin(1200.0)];
        assert!(mid[bin(500.0)] > quiet * 10.0, "500 Hz must be present");
        assert!(mid[bin(2000.0)] > quiet * 10.0, "2000 Hz must be present");
    }

    /// A source shorter than one window yields no spectra rather than a
    /// partial one built over uninitialised or zero-padded memory.
    #[test]
    fn a_source_shorter_than_one_window_yields_nothing() {
        let clip = SampleClip::new(SR, vec![0.5; 100]);
        assert!(spectra(&clip, 4096, 1600).is_empty());
    }

    /// The hop must be exactly the hop, whatever the decoder's block size is.
    ///
    /// Blocks are explicitly not contractual, so an advance that clamps to
    /// "however much happened to be buffered" makes the frame spacing -- and
    /// therefore the playback speed -- a function of the decoder. It only
    /// misbehaves when `hop > window`, so both regimes are covered here.
    #[test]
    fn consecutive_spectra_are_exactly_hop_apart_regardless_of_block_size() {
        let window = 4096usize;
        // `n - window` is an exact multiple of the hop, so an off-by-one hop
        // changes the frame count. A rounder length would absorb it.
        for hop in [1600usize, 4800] {
            let n = window + hop * 20;
            let expected = 1 + (n - window) / hop;
            assert_eq!(expected, 21, "the arithmetic this test rests on");
            let samples: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let mut reference: Option<Vec<Vec<f32>>> = None;
            for block in [1usize, 1024, 1152, 8192, 200_000] {
                let got = spectra(
                    &SampleClip { sample_rate: SR, samples: samples.clone(), block },
                    window,
                    hop,
                );
                assert_eq!(
                    got.len(),
                    expected,
                    "hop {hop} block {block}: frame count must not depend on the block size"
                );
                match &reference {
                    None => reference = Some(got),
                    Some(r) => assert!(
                        *r == got,
                        "hop {hop} block {block}: spectra must be identical across block sizes"
                    ),
                }
            }
        }
    }

    /// `hop > window` (low `--audio-fps`) must skip the samples between
    /// windows outright, not fold them into the next one.
    #[test]
    fn a_hop_larger_than_the_window_skips_samples_between_frames() {
        let window = 1024usize;
        let hop = 3000usize;
        let frames = 6usize;
        let n = window + hop * (frames - 1);
        // Silence exactly where each window is expected to land, a loud
        // constant everywhere in between. Any drift in the hop -- or an
        // under-advance that clamps to what happens to be buffered -- pulls
        // the loud region into a window and shows up as energy.
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let inside = (0..frames).any(|k| i >= k * hop && i < k * hop + window);
                if inside { 0.0 } else { 100.0 }
            })
            .collect();
        let sp = spectra(&SampleClip::new(SR, samples), window, hop);
        assert_eq!(sp.len(), frames, "a hop past the window must still emit every frame");
        for (k, s) in sp.iter().enumerate() {
            for m in s {
                assert!(*m < 1e-3, "frame {k} must land on its silent region, got {m}");
            }
        }
        // One sample short of the last full window is one frame fewer, not a
        // padded tail frame.
        let short = SampleClip::new(SR, vec![0.0; n - 1]);
        assert_eq!(spectra(&short, window, hop).len(), frames - 1);
    }

    /// The window shape is invisible to every signal test here, so assert the
    /// coefficients directly. Periodic Hann has `w[0] == 0`, `w[N/2] == 1`,
    /// and `w[i] == w[N-i]`; the symmetric form (divisor `N-1`) fails the
    /// last two, and rectangular/Hamming fail the first.
    #[test]
    fn the_window_function_is_periodic_hann() {
        for n in [64usize, 1024] {
            let clip = SampleClip::new(SR, Vec::new());
            let s = StftStream::new(clip.open().expect("open"), n, 16).expect("stft");
            let w = &s.window_fn;
            assert_eq!(w.len(), n);
            assert_eq!(w[0], 0.0, "periodic Hann starts at exactly 0 (window {n})");
            assert_eq!(w[n / 2], 1.0, "periodic Hann peaks at exactly 1 at N/2 (window {n})");
            for i in 1..n {
                let mirrored = n - i;
                assert!(
                    (w[i] - w[mirrored]).abs() < 1e-6,
                    "window {n}: w[{i}] and w[{mirrored}] must match, got {} and {}",
                    w[i],
                    w[mirrored]
                );
            }
        }
    }

    /// The progress bar's denominator has to be the number the stream will
    /// actually produce, not an estimate that drifts by a window's worth.
    #[test]
    fn frame_count_for_agrees_with_what_the_stream_emits() {
        // Durations are exact binary fractions of a second, so
        // `duration * rate` is exact in f64 and the helper and the stream
        // are looking at the same sample count.
        for (secs, window, fps) in [
            (0.0f64, 4096usize, 30.0f32),
            (0.0625, 4096, 30.0), // shorter than one window
            (0.125, 4096, 30.0),
            (0.5, 4096, 30.0),
            (1.0, 4096, 30.0),
            (1.0, 4096, 60.0),
            (1.0, 8192, 10.0),  // a long window, hop 4800 still inside it
            (0.25, 1024, 10.0), // hop 4800 LARGER than the 1024 window
        ] {
            let hop = hop_for(SR, fps).expect("a positive finite rate");
            let n = (secs * SR as f64) as usize;
            let clip = SampleClip::new(SR, vec![0.0; n]);
            let actual = spectra(&clip, window, hop).len();
            assert_eq!(
                frame_count_for(secs, SR, window, hop),
                actual,
                "{secs}s window {window} hop {hop}: the denominator must match the stream"
            );
        }
    }

    #[test]
    fn hop_is_derived_from_the_frame_rate() {
        assert_eq!(hop_for(48_000, 30.0).expect("positive"), 1600);
        assert_eq!(hop_for(48_000, 60.0).expect("positive"), 800);
        assert_eq!(hop_for(44_100, 30.0).expect("positive"), 1470);
    }

    /// `0.0` would yield `usize::MAX` and `NaN` would yield `1`, and NaN
    /// slips past a plain `<= 0.0` guard because every NaN comparison is
    /// false. All four must be refused.
    #[test]
    fn hop_for_rejects_a_rate_that_is_not_positive_and_finite() {
        assert!(hop_for(48_000, 0.0).is_err(), "zero fps");
        assert!(hop_for(48_000, -30.0).is_err(), "negative fps");
        assert!(hop_for(48_000, f32::NAN).is_err(), "NaN fps");
        assert!(hop_for(48_000, f32::INFINITY).is_err(), "infinite fps");
    }

    /// A zero or negative rate must be rejected, not turned into a zero hop
    /// that would spin forever emitting the same window.
    #[test]
    fn a_nonpositive_frame_rate_is_rejected() {
        let clip = SampleClip::new(SR, vec![0.0; 10_000]);
        assert!(StftStream::new(clip.open().expect("open"), 1024, 0).is_err());
    }

    /// Blocks are contractually non-empty. A decoder that breaks that would
    /// spin the fill loop forever, so it must be an error, not a hang.
    #[test]
    fn an_empty_block_is_an_error_not_a_hang() {
        struct EmptyBlocks;
        impl AudioStream for EmptyBlocks {
            fn next(&mut self) -> Result<Option<Vec<f32>>, String> {
                Ok(Some(Vec::new()))
            }
        }
        let mut s = StftStream::new(Box::new(EmptyBlocks), 1024, 400).expect("stft");
        assert!(s.next_spectrum().is_err(), "an empty block must be refused, not looped on");
    }
}
