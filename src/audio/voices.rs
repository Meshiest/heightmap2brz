//! Peak-tracking resynthesis: a handful of speakers that MOVE.
//!
//! This is the alternative to [`super::track`]'s fixed band bank, and it exists
//! because the bank has a structural tuning problem that no amount of tuning
//! the bank can remove.
//!
//! # What is wrong with a grid, however fine
//!
//! A bank puts each speaker on a fixed pitch and modulates only its volume, so
//! every note the source plays has to be rounded onto the nearest grid step.
//! At the equal-tempered semitone grid the error is zero **for a source in
//! tune with A440 and playing equal temperament**, and up to 50 cents for
//! anything else -- a recording pitched a little sharp, a bend, a vibrato, a
//! string section, or any of the inharmonic upper partials that real
//! instruments are largely made of. Halving it costs twice the speakers
//! (`--subdiv 24`) and still leaves 25 cents.
//!
//! Worse, a bank is *mostly idle*: at 79 tonal bands and `--max-voices 12`,
//! 85% of every array is zero, and all 79 arrays are still written in full.
//!
//! # What this mode does instead
//!
//! Both problems have the same answer. A speaker's `PitchMultiplier` is a
//! continuous float, so a speaker parked on a **detected spectral peak** can
//! sit at *any* frequency. There is then no grid at all and therefore no
//! quantisation error -- the tuning question dissolves rather than being
//! mitigated. And because a voice only exists while it has something to play,
//! `--max-voices` speakers is the whole build, not a bank of 79 with 12 awake.
//!
//! Per frame:
//!
//! 1. find local maxima in the **raw FFT magnitude spectrum**, at full bin
//!    resolution -- not on any band grid ([`find_peaks`]);
//! 2. refine each one with **parabolic interpolation** over the three bins
//!    around it, which is what buys sub-bin -- and therefore sub-cent --
//!    frequency accuracy ([`interpolate_peak`]);
//! 3. gate on **prominence**, so a lump of broadband noise does not become a
//!    note ([`MIN_PROMINENCE`]);
//! 4. keep the strongest `max_voices`;
//! 5. **match them against the previous frame's voices by frequency
//!    proximity**, so a voice glides along a partial instead of hopping
//!    between unrelated ones ([`assign`]).
//!
//! Step 5 is the one that decides whether this sounds like music. Without it,
//! voice *k* plays "whatever the k-th loudest peak happened to be this frame",
//! which changes identity several times a second and is heard as random chimes.
//!
//! # The pitch-write assumption
//!
//! Every frame writes `PitchMultiplier` as well as `VolumeMultiplier`. That a
//! volume write RAMPS a running voice rather than retriggering it is confirmed
//! in game (`examples/audio_diagnostics.rs`, diagnostic 3). That a **pitch**
//! write behaves the same way is the subject of **diagnostic 5**
//! (`diag_5_pitch_ramp.brz`) and must be listened to before this mode's output
//! is trusted: if a pitch write retriggers, a voice re-pitched at 30 fps is a
//! 30 Hz click train.
//!
//! Should that turn out to be so, the design still stands -- a pitch change at a
//! note boundary is an attack, which is musically correct -- but
//! [`MATCH_TOLERANCE_SEMITONES`] would have to become 0 for continuing voices
//! (freeze the pitch for the life of a note), giving up vibrato and glissando.
use super::bands::{BASE_HZ, PITCH_MAX, PITCH_MIN};
use super::source::AudioSource;
use super::stft::{StftStream, frame_count_for, hop_for};
use super::track::{AudioOptions, Envelope};
use crate::progress::{FrameTotal, Progress};

/// The lowest and highest frequency a speaker can actually play: `BASE_HZ`
/// times the emitter's `PitchMultiplier` limits, i.e. 44 Hz to 4400 Hz.
///
/// Peaks outside this are not merely uninteresting, they are **unplayable** --
/// the game clamps `PitchMultiplier` silently, so a 6 kHz peak assigned to a
/// voice would come out at 4400 Hz as a wrong note rather than as nothing.
/// Filtering here is what keeps that from happening.
///
/// The bank folds this same off-the-ends energy onto white and pink noise
/// bands. This mode has no noise bands: a moving sine voice cannot carry
/// broadband content, and spending one of a handful of voices on a noise bed
/// was measured on the bank to be a poor trade even when noise bands were free
/// (see `track::select_peaks`). Cymbals and sub-bass are therefore dropped
/// rather than mis-rendered.
pub fn min_hz() -> f64 {
    BASE_HZ as f64 * PITCH_MIN as f64
}
pub fn max_hz() -> f64 {
    BASE_HZ as f64 * PITCH_MAX as f64
}

/// Half-width, in bins, of a periodic Hann window's main lobe.
///
/// A windowed sinusoid does not occupy one bin; it occupies its window's main
/// lobe, which for a periodic Hann is 4 bins wide -- the apex plus two either
/// side. Those bins are the peak ITSELF, so they are excluded from the
/// neighbourhood its [`prominence`] is measured against. Including them makes
/// every peak its own reference and drives every prominence toward 1, which
/// then either passes everything (gate useless) or nothing (silence).
///
/// This is a property of the window function, not of the window SIZE: a longer
/// window makes each bin narrower in Hz but the lobe is still 4 bins.
const LOBE_HALF_BINS: usize = 2;

/// Half-width, in bins, of the neighbourhood a peak's [`prominence`] is
/// measured against. The bins within [`LOBE_HALF_BINS`] of the apex are cut
/// out of it, so the mean is taken over `2 * (this - LOBE_HALF_BINS)` bins.
///
/// Fixed in BINS rather than as a frequency ratio, and that is the deliberate
/// choice. The two lengths it has to sit between are both roughly constant in
/// bins:
///
/// * **The peak's own footprint** is exactly [`LOBE_HALF_BINS`], everywhere in
///   the spectrum -- a window property, not a frequency-dependent one.
/// * **The spacing between the partials of one note** is the fundamental, in
///   Hz, and therefore a CONSTANT number of bins across that note's whole
///   harmonic series. A constant-Q neighbourhood (say ±3 semitones) would be
///   right at the fundamental and then swallow four neighbouring harmonics by
///   the tenth partial, where a minor third spans far more than one harmonic
///   gap.
///
/// 8 is four lobe half-widths: wide enough that the mean is taken over 12 bins
/// well clear of the peak's own skirt, and narrow enough to stay inside a 24 Hz
/// window at `--window 16384` -- under the ~44 Hz spacing of the lowest
/// fundamental this mode can play at all.
const PROMINENCE_HALF_BINS: usize = 8;

/// The [`prominence`] a local maximum needs before it counts as a note.
///
/// Same role, and the same measured value, as `track::MIN_PROMINENCE`: a peak
/// standing 3.5 dB above the mean of its neighbourhood. The band bank's
/// measurements are the justification (the constant's doc there carries the
/// table); the quantity is the same ratio measured over a different
/// neighbourhood, so it is restated here rather than imported -- the two are
/// free to diverge as each is tuned against its own axis.
const MIN_PROMINENCE: f32 = 1.5;

/// Exponent on a peak's own magnitude in the rank key
/// `prominence * magnitude^this`.
///
/// Mirrors `track::RANK_MAGNITUDE_EXP` and for the same measured reason: a
/// pure prominence rank has no level term, so nothing stops it promoting a
/// quiet-but-pointy ripple over a loud note.
const RANK_MAGNITUDE_EXP: f32 = 0.5;

/// Ceiling on a reported [`prominence`], so a peak standing in perfect silence
/// stays a finite, ORDERABLE number rather than tying with every other such
/// peak at infinity.
const MAX_PROMINENCE: f32 = 1e6;

/// How far, in semitones, a peak may sit from a voice's current pitch and
/// still be taken as **the same partial, continuing**.
///
/// **This constant is the whole design.** Above it a peak is a different note
/// and gets a different voice; below it the voice glides. Too wide and a voice
/// follows whatever is nearest, which across a chord change is an unrelated
/// partial -- the "random chime" failure. Too narrow and nothing ever matches,
/// every voice dies after one frame, and the result is the same chimes by the
/// opposite route.
///
/// 0.5 semitones is half the smallest interval Western music uses, so two
/// simultaneous notes can never both fall inside one voice's window, while a
/// single note is free to bend, drift or vibrate by up to a quarter-tone
/// between frames -- 15 semitones per second at 30 fps, far faster than any
/// real portamento.
const MATCH_TOLERANCE_SEMITONES: f64 = 0.5;

/// How many candidate peaks are found per voice, before tracking.
///
/// **Not a tuning knob -- the tracker is wrong without it.** Peaks arrive ranked
/// by `prominence * magnitude^k`, which has no memory: a partial a voice is
/// already following can be pushed out of the top-`max_voices` by a slightly
/// louder newcomer, killing that voice, and be back in the top V the next
/// frame. The voice then dies and is reborn every few frames on content that
/// is, to a listener, one held note.
///
/// Truncating to `max_voices * this` instead and letting the MATCHING decide
/// what survives is the McAulay-Quatieri rule: a running voice continues as
/// long as its partial is a detectable peak AT ALL, and only the leftover
/// slots go to newcomers by rank. Measured on speech at 8 voices, truncating
/// to V first left 20.7% of sounding frame pairs jumping more than a semitone;
/// the fix is what that number is for.
///
/// 4 is generous enough that a tracked partial has to fall out of the top ~32
/// peaks of the frame before it is given up, and cheap: the work is a sort of a
/// list that is already short.
const PEAK_OVERSAMPLE: usize = 4;

/// The largest deviation from an exact integer frequency ratio, in cents, that
/// still counts as "this voice is a harmonic of that one" for
/// [`VoiceStats::harmonic_voices`].
///
/// A measurement threshold only -- nothing in the renderer branches on it. 35
/// cents is chosen to admit real piano inharmonicity, which stretches the upper
/// partials measurably sharp (tens of cents by the tenth), while staying inside
/// the ~100-cent spacing at which an unrelated note could be mistaken for one.
const HARMONIC_TOLERANCE_CENTS: f64 = 35.0;

/// Ceiling on the frames a released voice keeps its speaker.
///
/// The release ramp's length comes from `--release`, but a voice's release is
/// not only an envelope -- it is also a LOCK on one of very few speakers, and no
/// peak can be born while every speaker is held. A `--release 2000` would
/// otherwise silence the build for two seconds after the first chord.
const MAX_VOICE_RELEASE_FRAMES: usize = 12;

/// Time constant, in milliseconds, for a CONTINUING voice's pitch.
///
/// **The fix for "sounds weird because the pitch wobbles a bit".** A sustained
/// note's detected peak moves a few cents every frame -- the bin grid, the
/// interpolation's own noise, and partials genuinely beating against one
/// another -- and written straight to `PitchMultiplier` that is an audible
/// warble on a note that should be steady.
///
/// 120 ms at 30 fps is a coefficient of 0.24, so a continuing voice covers a
/// quarter of the distance to the newly-measured frequency each frame.
/// Frame-to-frame jitter alternates sign and is attenuated hard; a real bend or
/// portamento does not, and still arrives within a few frames.
///
/// It applies ONLY to a voice continuing the same partial. A voice being born
/// takes its pitch outright -- see [`assign`].
///
/// The cost is vibrato: at 30 fps a 5 Hz wobble is already near the sampling
/// limit, and this attenuates what is left. That is the deliberate trade -- the
/// wobble being removed is measurement noise, and at this frame rate nothing
/// can tell the two apart.
const PITCH_GLIDE_MS: f32 = 120.0;

/// [`PITCH_GLIDE_MS`] as a one-pole coefficient at `fps`. Same discretisation
/// as [`Envelope`], so the time constant means the same thing at any rate.
fn pitch_coefficient(fps: f32) -> f64 {
    if !fps.is_finite() || fps <= 0.0 || PITCH_GLIDE_MS <= 0.0 {
        return 1.0;
    }
    let dt_ms = 1000.0 / fps as f64;
    (1.0 - (-dt_ms / PITCH_GLIDE_MS as f64).exp()).clamp(0.0, 1.0)
}

/// The pitch a voice holds before it has ever sounded.
///
/// 1.0 is A440, the identity multiplier and the one value guaranteed legal.
/// It is inaudible (the voice's volume is 0 until it is born) but it is what
/// the speaker's baked `PitchMultiplier` will be, so it must be a real note
/// rather than 0.0 -- see [`VoiceStreams::pitches`].
const IDLE_PITCH: f64 = 1.0;

/// The largest per-frame boost `leveling` may apply. Mirrors
/// `track::MAX_LEVELING_BOOST`; see [`analyze_voices`].
const MAX_LEVELING_BOOST: f32 = 10.0;

/// The largest meaningful `--pitch-snap`, in cents.
///
/// Half a semitone: no pitch is ever further than 50 cents from the nearest
/// equal-tempered step, so at 50 every continuing voice is quantised and 100
/// means exactly what 50 does. Bounding it is not a safety clamp -- illegal
/// pitches are impossible either way, see [`VoiceShaping::snap`] -- it is so a
/// value that cannot mean what the user thinks it means is refused by name
/// instead of silently meaning something else. Both front ends check against
/// this same constant.
pub const MAX_PITCH_SNAP_CENTS: f32 = 50.0;

/// One refined spectral peak.
#[derive(Clone, Copy, Debug)]
pub struct Peak {
    /// Frequency in Hz, sub-bin accurate -- see [`interpolate_peak`].
    pub hz: f64,
    /// The interpolated apex magnitude, i.e. an estimate of the partial's own
    /// amplitude rather than of whichever bin happened to catch it.
    pub mag: f32,
    /// How far this peak stands above its neighbourhood. See [`prominence`].
    pub prominence: f32,
}

/// The output of a voice-mode analysis: `2V` dense per-frame streams.
///
/// Voice-major for the same reason the bank is band-major: each row is exactly
/// one `ArrayVar`, and a frame-major layout would force a transpose of millions
/// of elements at build time.
pub struct VoiceStreams {
    /// `pitches[voice][frame]` -- the `PitchMultiplier` to write, always inside
    /// the emitter's legal `PITCH_MIN..=PITCH_MAX`.
    ///
    /// **Never 0, not even while the voice is silent.** A silent voice HOLDS
    /// its last pitch. Writing 0.0 would be a 44 Hz lurch at the end of every
    /// note (0 is below `PITCH_MIN`, so the game would clamp it to 0.1 and
    /// play it) and, if a pitch write turns out to retrigger, an audible click
    /// on every note end as well. Holding costs nothing and is silent by
    /// construction, because the volume is 0 in exactly those frames.
    pub pitches: Vec<Vec<f64>>,
    /// `volumes[voice][frame]`, linear 0..=1.
    pub volumes: Vec<Vec<f64>>,
    pub fps: f32,
    pub frame_count: usize,
    pub stats: VoiceStats,
}

impl VoiceStreams {
    pub fn voice_count(&self) -> usize {
        self.pitches.len()
    }
}

/// What the analysis measured about itself.
///
/// Every field here answers a question that decides whether this mode works at
/// all, and none of them is visible in the output arrays without being
/// computed. They are carried out of the analysis rather than recomputed by a
/// caller so that the CLI, the tests and any report are all reading the same
/// numbers.
#[derive(Clone, Debug, Default)]
pub struct VoiceStats {
    /// Peaks assigned to a voice, over the whole track.
    pub peak_count: usize,
    /// Signed offset from the nearest equal-tempered semitone, in cents, one
    /// bucket per cent over `-50..=50`. **This is the direct evidence that the
    /// bank's tuning problem is gone**: a tonal source should pile up near 0.
    pub cents_hist: Vec<u32>,
    /// Mean |cents| offset over every assigned peak.
    pub mean_abs_cents: f64,
    /// Counts of assigned peaks per equal-tempered semitone step from A440,
    /// index 0 = step `NOTE_HIST_LOW`. The frequency distribution, on the axis
    /// that matters.
    pub note_hist: Vec<u32>,
    /// The length in frames of every completed run of a voice sounding
    /// continuously. **If these are 1-2 frames the tracking is not working**
    /// and the render is chimes whatever else is true.
    pub lifetimes: Vec<usize>,
    /// (voice, frame) pairs whose volume is above zero. Divided by
    /// `frame_count` this is the real voices-per-frame, which is NOT the
    /// `--max-voices` flag.
    pub sounding: usize,
    /// Consecutive frame pairs in which one voice sounded in both.
    pub continuations: usize,
    /// ...of which the pitch moved more than a semitone. Should be rare: with
    /// no voice stealing, the only way to move a sounding speaker that far is a
    /// tracking failure.
    pub pitch_jumps: usize,
    /// Voice-frames whose frequency is an integer multiple (within
    /// [`HARMONIC_TOLERANCE_CENTS`]) of another sounding voice's.
    ///
    /// A harmonic is a real, audible part of an instrument's timbre -- a piano
    /// rendered as bare fundamentals sounds like an organ -- so this is not a
    /// defect count. It is a **budget** measurement: with too few voices a
    /// single note's overtone series can consume the whole build, and a chord
    /// then comes out as one note's harmonics rather than as three notes.
    /// Divided by [`Self::sounding`] it is the fraction of the build spent on
    /// timbre rather than on notes.
    pub harmonic_voices: usize,
    /// How many frames had `k` distinct fundamentals sounding -- a voice being
    /// counted as a fundamental exactly when it is not a harmonic of a lower
    /// sounding voice. Index `k`, so `[0]` is frames with no voice at all.
    ///
    /// This is the other half of the budget question: whether the voices
    /// available are spread across the notes actually being played.
    pub fundamentals_hist: Vec<u32>,
    /// Voice-frames written with a NON-ZERO volume in which the voice was not
    /// matched to any peak of that frame.
    ///
    /// **This is the bleed, quantified.** A voice that is sounding while
    /// nothing in the spectrum is within [`MATCH_TOLERANCE_SEMITONES`] of it is
    ///, by construction, playing a note the source has stopped playing. Some of
    /// this is legitimate -- a release ramp is a few frames of exactly this -- so
    /// the number is read against [`Self::sounding`] and against
    /// [`Self::tail_frames`], which says how LONG each such stretch is.
    pub unmatched_sounding: usize,
    /// For every completed run of a voice sounding continuously: the frames
    /// from the last frame the run was MATCHED to a peak, to the first frame it
    /// reads exactly zero.
    ///
    /// **Time from a partial's true end to the voice going silent**, in frames.
    /// The defect stated directly: a voice whose partial has ended must reach
    /// zero promptly, and this is how long it actually takes.
    pub tail_frames: Vec<usize>,
    /// Sum of the WRITTEN volume over matched and over unmatched sounding
    /// frames respectively.
    ///
    /// The pair is what says whether a release ramp is heard as a fade at all.
    /// `--leveling` divides every frame by its own peak, so a frame whose
    /// loudest content IS a decaying tail is normalised straight back up to
    /// full scale: the ramp exists in `raw_mag` and is flattened out of the
    /// written values. If these two means are close, the "release" is a HOLD.
    matched_volume: f64,
    unmatched_volume: f64,
    /// Frame-to-frame pitch changes of a CONTINUING voice, as sums of squared
    /// cents -- the wobble metric, measured twice over the same frames.
    ///
    /// `raw` is what the peak tracker handed over, `out` is what was written
    /// after [`PITCH_GLIDE_MS`] smoothing and any snap. The pair is what says
    /// whether the smoothing did anything, and it must come from ONE run: two
    /// runs with different settings differ in which voices continued at all,
    /// so their jitter is not measured over the same frames.
    raw_jitter_sq: f64,
    out_jitter_sq: f64,
    jitter_n: usize,
}

/// Split a frame's sounding frequencies into (harmonics, distinct
/// fundamentals).
///
/// A frequency is a HARMONIC if some LOWER sounding frequency divides it to
/// within [`HARMONIC_TOLERANCE_CENTS`] of a whole number at least 2. Everything
/// else is a fundamental. Lower-first so a harmonic series collapses onto its
/// own root rather than each partial claiming the one above it.
fn harmonic_split(sounding: &[f64]) -> (usize, usize) {
    let mut sorted = sounding.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut harmonics = 0usize;
    for (i, &hi) in sorted.iter().enumerate() {
        let is_harmonic = sorted[..i].iter().any(|&lo| {
            if lo <= 0.0 {
                return false;
            }
            let ratio = hi / lo;
            let n = ratio.round();
            n >= 2.0 && (1200.0 * (ratio / n).log2()).abs() <= HARMONIC_TOLERANCE_CENTS
        });
        if is_harmonic {
            harmonics += 1;
        }
    }
    (harmonics, sorted.len() - harmonics)
}

/// Lowest semitone step from A440 that [`VoiceStats::note_hist`] indexes.
/// `12 * log2(0.1)` is -39.86, so -40 is the lowest step any playable peak can
/// round to, and +40 the highest.
pub const NOTE_HIST_LOW: i32 = -40;
pub const NOTE_HIST_LEN: usize = 81;

impl VoiceStats {
    fn new(n_voices: usize) -> Self {
        Self {
            cents_hist: vec![0; 101],
            note_hist: vec![0; NOTE_HIST_LEN],
            fundamentals_hist: vec![0; n_voices + 1],
            ..Default::default()
        }
    }

    /// The share of sounding voice-frames spent on a harmonic of another
    /// sounding voice rather than on an independent note. See
    /// [`Self::harmonic_voices`].
    pub fn harmonic_fraction(&self) -> f64 {
        if self.sounding == 0 {
            return 0.0;
        }
        self.harmonic_voices as f64 / self.sounding as f64
    }

    /// Record one continuing voice's frame-to-frame pitch move, before and
    /// after smoothing. Both in cents, both over the same frame.
    fn record_jitter(&mut self, raw_prev: f64, raw_now: f64, out_prev: f64, out_now: f64) {
        let cents = |a: f64, b: f64| {
            if a > 0.0 && b > 0.0 {
                1200.0 * (b / a).log2()
            } else {
                0.0
            }
        };
        let r = cents(raw_prev, raw_now);
        let o = cents(out_prev, out_now);
        if r.is_finite() && o.is_finite() {
            self.raw_jitter_sq += r * r;
            self.out_jitter_sq += o * o;
            self.jitter_n += 1;
        }
    }

    /// RMS frame-to-frame pitch movement of a continuing voice, in cents, as
    /// the TRACKER measured it -- the wobble before smoothing.
    pub fn raw_jitter_rms_cents(&self) -> f64 {
        if self.jitter_n == 0 {
            return 0.0;
        }
        (self.raw_jitter_sq / self.jitter_n as f64).sqrt()
    }

    /// The same, as WRITTEN -- the wobble a listener actually hears.
    pub fn jitter_rms_cents(&self) -> f64 {
        if self.jitter_n == 0 {
            return 0.0;
        }
        (self.out_jitter_sq / self.jitter_n as f64).sqrt()
    }

    /// Mean number of distinct fundamentals sounding, over frames that had any
    /// voice at all.
    pub fn mean_fundamentals(&self) -> f64 {
        let frames: u32 = self.fundamentals_hist.iter().skip(1).sum();
        if frames == 0 {
            return 0.0;
        }
        let total: u64 = self
            .fundamentals_hist
            .iter()
            .enumerate()
            .map(|(k, &c)| k as u64 * c as u64)
            .sum();
        total as f64 / frames as f64
    }

    pub fn mean_lifetime(&self) -> f64 {
        if self.lifetimes.is_empty() {
            return 0.0;
        }
        self.lifetimes.iter().sum::<usize>() as f64 / self.lifetimes.len() as f64
    }

    pub fn mean_voices_per_frame(&self, frame_count: usize) -> f64 {
        if frame_count == 0 {
            return 0.0;
        }
        self.sounding as f64 / frame_count as f64
    }

    pub fn pitch_jump_fraction(&self) -> f64 {
        if self.continuations == 0 {
            return 0.0;
        }
        self.pitch_jumps as f64 / self.continuations as f64
    }

    /// The share of sounding voice-frames in which the voice was matched to no
    /// peak at all. See [`Self::unmatched_sounding`].
    pub fn unmatched_fraction(&self) -> f64 {
        if self.sounding == 0 {
            return 0.0;
        }
        self.unmatched_sounding as f64 / self.sounding as f64
    }

    /// Mean written volume on frames where the voice WAS following a peak, and
    /// on frames where it was not. See [`Self::matched_volume`].
    pub fn mean_volumes(&self) -> (f64, f64) {
        let matched = self.sounding - self.unmatched_sounding;
        (
            if matched == 0 { 0.0 } else { self.matched_volume / matched as f64 },
            if self.unmatched_sounding == 0 {
                0.0
            } else {
                self.unmatched_volume / self.unmatched_sounding as f64
            },
        )
    }

    /// Mean and 95th-percentile time from a partial's end to the voice reading
    /// zero, in MILLISECONDS at `fps`. See [`Self::tail_frames`].
    pub fn tail_ms(&self, fps: f32) -> (f64, f64) {
        if self.tail_frames.is_empty() || !(fps > 0.0) {
            return (0.0, 0.0);
        }
        let ms = 1000.0 / fps as f64;
        let mut v = self.tail_frames.clone();
        v.sort_unstable();
        let mean = v.iter().sum::<usize>() as f64 / v.len() as f64 * ms;
        let p95 = v[((v.len() as f64 * 0.95) as usize).min(v.len() - 1)] as f64 * ms;
        (mean, p95)
    }
}

/// How far `hz` sits from the nearest equal-tempered semitone, in cents, and
/// which semitone step from A440 that is.
///
/// The measurement the whole mode is justified by. A band bank can only ever
/// report the offset of its own grid; a peak tracker reports the offset of the
/// SOURCE, and for a tonal recording that should cluster near zero.
pub fn cents_from_equal_temperament(hz: f64) -> (f64, i32) {
    let semitones = 12.0 * (hz / BASE_HZ as f64).log2();
    let nearest = semitones.round();
    ((semitones - nearest) * 100.0, nearest as i32)
}

/// How sharply bin `i` stands above its local neighbourhood: its magnitude
/// divided by the MEAN of the bins from [`LOBE_HALF_BINS`] to
/// [`PROMINENCE_HALF_BINS`] either side, the peak's own main lobe excluded.
///
/// This is what separates a note from a lump. A musical partial is a narrow,
/// high-contrast spike against its surroundings; cymbal wash, sibilance, room
/// tone and bass rumble are broad and flat, and carry local maxima that are
/// indistinguishable from partials by magnitude alone.
pub fn prominence(spectrum: &[f32], i: usize) -> f32 {
    let lo = i.saturating_sub(PROMINENCE_HALF_BINS);
    let hi = (i + PROMINENCE_HALF_BINS).min(spectrum.len().saturating_sub(1));
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for (k, &m) in spectrum.iter().enumerate().take(hi + 1).skip(lo) {
        if k.abs_diff(i) > LOBE_HALF_BINS {
            sum += m as f64;
            count += 1;
        }
    }
    // Nothing to stand above -- a spectrum too short to have a neighbourhood, or
    // a digitally silent one -- makes the peak as prominent as the
    // representation allows, rather than infinite or NaN.
    if count == 0 {
        return MAX_PROMINENCE;
    }
    let mean = sum / count as f64;
    if mean <= 0.0 {
        return MAX_PROMINENCE;
    }
    ((spectrum[i] as f64 / mean) as f32).min(MAX_PROMINENCE)
}

/// Sub-bin peak location and height, by fitting a parabola through the three
/// bins around `i` in the LOG magnitude domain.
///
/// Returns `(delta, apex)`: `delta` in `-0.5..=0.5` bins from `i`, and `apex`
/// the interpolated magnitude at that point.
///
/// # Why this is not optional
///
/// Without it a peak is quantised to its bin, and a bin is a fixed number of
/// HERTZ while pitch is logarithmic -- so the error in cents grows without
/// bound toward the bass. At `--window 16384` (2.93 Hz bins) half a bin is
/// 1.46 Hz: 2.5 cents at 1 kHz, but **25 cents at 100 Hz and 57 cents at
/// 44 Hz**, which is the band bank's own worst case back again. At the 4096
/// default it is four times worse. Interpolation is what makes "no grid" mean
/// something at the frequencies music actually lives at.
///
/// # Why the log domain
///
/// The main lobe of a Hann-windowed sinusoid is very close to a Gaussian in
/// shape, and a Gaussian is exactly a parabola once logged. Fitting the
/// parabola to the raw magnitudes instead is the same arithmetic against the
/// wrong curve and leaves several times the residual error.
///
/// A non-concave triple (equal or rising neighbours, which a strict local
/// maximum rules out, or floating-point noise that makes the denominator
/// vanish) falls back to `delta = 0` -- the bin centre, i.e. exactly what an
/// uninterpolated peak finder would have said.
pub fn interpolate_peak(spectrum: &[f32], i: usize) -> (f64, f32) {
    if i == 0 || i + 1 >= spectrum.len() {
        return (0.0, spectrum[i]);
    }
    // A floor rather than a zero guard: `ln(0)` is -inf, and one infinity in
    // the triple makes the denominator NaN, which then compares false against
    // every bound and would slip through as a `delta` of NaN.
    const TINY: f64 = 1e-30;
    let a = (spectrum[i - 1] as f64).max(TINY).ln();
    let b = (spectrum[i] as f64).max(TINY).ln();
    let c = (spectrum[i + 1] as f64).max(TINY).ln();
    let denom = a - 2.0 * b + c;
    if !denom.is_finite() || denom >= -f64::EPSILON {
        return (0.0, spectrum[i]);
    }
    let delta = (0.5 * (a - c) / denom).clamp(-0.5, 0.5);
    let apex = (b - 0.25 * (a - c) * delta).exp();
    if !apex.is_finite() {
        return (delta, spectrum[i]);
    }
    (delta, apex as f32)
}

/// Every prominent local maximum in a raw magnitude spectrum, refined to
/// sub-bin accuracy, ranked, and truncated to the strongest `max_peaks`.
///
/// Operates on the FFT's own bins. There is no band grid anywhere in this
/// function and that is the point of the mode.
///
/// # The level floor is not optional here, unlike in the bank
///
/// `floor_ratio` (`10^(floor_db/20)`, so 0.001 at the default -60 dB) discards
/// peaks that far below **the loudest peak in the same frame**. The band bank
/// applies the same ratio, but at the very END of its pipeline, where all it
/// does is silence a band that was going to be inaudible anyway.
///
/// Here it has to happen at SELECTION, and leaving it out was a real defect
/// caught by this module's own tests. The reason is that
/// [`prominence`] is a CONTRAST measure with no level term: a ripple in the
/// noise floor 100 dB down stands just as far above its neighbours as a real
/// partial does, so it passes the gate, takes a voice, and -- because floor
/// noise moves every frame -- that voice is re-pointed somewhere new every
/// frame. On a pure synthetic tone through this pipeline it produced eight
/// "notes" scattered across 2-4 kHz from nothing but FFT round-off, dragged the
/// measured tuning error from ~0 to 10.8 cents, and cut mean voice lifetime to
/// 1.9 frames. Those wandering voices are silenced by the floor at the far end,
/// so the save is not audibly wrong -- but the slots they consumed are gone, and
/// on real music they are slots a real note needed.
pub fn find_peaks(
    spectrum: &[f32],
    sample_rate: u32,
    window: usize,
    max_peaks: usize,
    floor_ratio: f32,
) -> Vec<Peak> {
    if spectrum.len() < 3 || window == 0 || max_peaks == 0 {
        return Vec::new();
    }
    let hz_per_bin = sample_rate as f64 / window as f64;
    if !(hz_per_bin > 0.0) {
        return Vec::new();
    }
    // Bin 0 and the last bin have no neighbour on one side, so they can be
    // neither a strict local maximum nor interpolated. Unlike the bank -- where
    // an axis endpoint is a real band and dropping it silences the ends of the
    // spectrum -- here the endpoints are DC and Nyquist, neither of which is
    // inside the playable range at any supported sample rate.
    let lo = 1usize;
    let hi = spectrum.len() - 2;

    let mut peaks: Vec<Peak> = Vec::new();
    for i in lo..=hi {
        let m = spectrum[i];
        // Silence is never a peak, however its neighbours compare.
        if m <= 0.0 {
            continue;
        }
        if !(m > spectrum[i - 1] && m > spectrum[i + 1]) {
            continue;
        }
        let p = prominence(spectrum, i);
        if p < MIN_PROMINENCE {
            continue;
        }
        let (delta, apex) = interpolate_peak(spectrum, i);
        let hz = (i as f64 + delta) * hz_per_bin;
        // Checked AFTER interpolation, against the refined frequency: a peak
        // whose bin centre is inside the range but whose true position is
        // outside it is unplayable, and would be clamped in game into a wrong
        // note rather than into nothing.
        if hz < min_hz() || hz > max_hz() || !apex.is_finite() || apex <= 0.0 {
            continue;
        }
        peaks.push(Peak { hz, mag: apex, prominence: p });
    }

    // The level floor, measured against the loudest peak IN THIS FRAME. See
    // the doc above for why this cannot wait until the end of the pipeline.
    // Frame-relative rather than track-relative for the same reason
    // `track::analyze`'s floor is: an absolute floor takes a fixed amount out
    // of every frame and therefore proportionally more of a quiet one, which
    // in the limit silences a quiet passage outright.
    let loudest = peaks.iter().map(|p| p.mag).fold(0.0f32, f32::max);
    if loudest > 0.0 && floor_ratio > 0.0 {
        let cutoff = loudest * floor_ratio;
        peaks.retain(|p| p.mag >= cutoff);
    }

    // Prominence decides WHICH peaks sound; magnitude tempers the order so a
    // quiet ripple cannot outrank a loud note. Neither ever changes a volume:
    // what is written is always the peak's own magnitude.
    let key = |p: &Peak| p.prominence * p.mag.powf(RANK_MAGNITUDE_EXP);
    peaks.sort_by(|a, b| key(b).total_cmp(&key(a)));
    peaks.truncate(max_peaks);
    peaks
}

/// One speaker's running state.
#[derive(Clone, Copy, Debug)]
struct VoiceState {
    /// The pitch multiplier WRITTEN, after smoothing and any snap. **Held
    /// across silence** -- see [`VoiceStreams::pitches`].
    pitch: f64,
    /// The pitch multiplier the tracker was handed this frame, before
    /// smoothing. Kept only so the wobble can be measured before and after in
    /// the same run -- see [`VoiceStats::raw_jitter_rms_cents`].
    raw_pitch: f64,
    /// The level follower's output: what this voice would be emitting if it
    /// were still following its partial.
    ///
    /// **Frozen, not ramped, while the voice releases** -- the fade lives in
    /// [`Self::gain`] instead. That split is the fix for "it holds level after
    /// the note is done": normalisation reads this field, and a per-frame AGC
    /// that reads a ramp divides the ramp straight back out. See
    /// [`analyze_voices`].
    level: f32,
    /// The release ramp, 1.0 while following a peak and stepping linearly to
    /// exactly 0.0 over `release_frames` once the partial has gone.
    ///
    /// Multiplied onto the volume AFTER normalisation, so nothing downstream
    /// can undo it and 0 means silence with no residual.
    gain: f32,
    /// True while this voice is following a peak.
    active: bool,
    /// Frames of release ramp left. 0 and `!active` means free.
    release_left: usize,
    /// Whether this voice was matched to a peak on the frame just assigned.
    /// Measurement only -- see [`VoiceStats::unmatched_sounding`].
    matched: bool,
}

impl VoiceState {
    fn idle() -> Self {
        Self {
            pitch: IDLE_PITCH,
            raw_pitch: IDLE_PITCH,
            level: 0.0,
            gain: 0.0,
            active: false,
            release_left: 0,
            matched: false,
        }
    }

    /// What this voice actually emits: the follower level through the release
    /// ramp. Zero exactly when the ramp has finished.
    fn emitted(&self) -> f32 {
        self.level * self.gain
    }

    fn is_free(&self) -> bool {
        !self.active && self.release_left == 0
    }

    /// Whether a peak at `hz` is close enough to be this same partial,
    /// continuing -- and how far, in semitones.
    ///
    /// Measured against the RAW tracked pitch, not the smoothed output: the
    /// smoothed one lags the partial by design, and matching against a lagging
    /// reference would shrink the effective tolerance on any note that is
    /// genuinely moving.
    fn distance(&self, hz: f64) -> f64 {
        (12.0 * (hz / (self.raw_pitch * BASE_HZ as f64)).log2()).abs()
    }
}

/// How a voice's level and pitch are allowed to move between frames.
///
/// Bundled rather than passed as four arguments because they are one decision:
/// all of it comes from `--attack`, `--release` and `--pitch-snap` against the
/// analysis frame rate.
#[derive(Clone, Copy, Debug)]
pub struct VoiceShaping {
    /// The level follower, shared with the band bank. Applied only while a
    /// voice is FOLLOWING a peak; the end of a note is the linear release
    /// below, which has to reach exactly zero.
    level: Envelope,
    /// Frames a released voice takes to fade to exactly zero, and for which
    /// its speaker stays unavailable.
    ///
    /// From `--voice-release`, NOT from `--release`. The two are different
    /// quantities: `--release` is a one-pole time constant on a voice that is
    /// still following something and never reaches zero, and taking a note's
    /// END from it gave every voice a 133 ms tail -- measured at 52.7% of all
    /// sounding voice-frames on speech. See
    /// [`crate::audio::track::DEFAULT_VOICE_RELEASE_MS`].
    release_frames: usize,
    /// One-pole coefficient for a CONTINUING voice's pitch, in log-frequency.
    /// 1.0 = no smoothing.
    pitch_coeff: f64,
    /// A continuing voice within this many cents of an equal-tempered semitone
    /// is pulled onto it. 0 disables snapping.
    snap_cents: f64,
}

impl VoiceShaping {
    /// Derive the shaping from the render options.
    pub fn new(opts: &AudioOptions) -> Self {
        let frames = (opts.voice_release_ms as f64 * opts.fps as f64 / 1000.0).round();
        let release_frames = if frames.is_finite() {
            (frames as usize).clamp(1, MAX_VOICE_RELEASE_FRAMES)
        } else {
            1
        };
        Self {
            level: Envelope::new(opts.attack_ms, opts.release_ms, opts.fps),
            release_frames,
            pitch_coeff: pitch_coefficient(opts.fps),
            snap_cents: opts.pitch_snap_cents.max(0.0) as f64,
        }
    }

    /// One smoothing step for a CONTINUING voice's pitch, in log-frequency,
    /// with the optional snap applied afterwards.
    fn glide(&self, from: f64, to: f64) -> f64 {
        let smoothed = if self.pitch_coeff >= 1.0 {
            to
        } else {
            // Interpolating in log-frequency, not in the multiplier itself: a
            // linear blend between two pitches is not a pitch halfway between
            // them, and the error grows with the interval.
            (from.ln() + (to.ln() - from.ln()) * self.pitch_coeff).exp()
        };
        self.snap(smoothed)
    }

    /// Pull a pitch onto the nearest equal-tempered semitone if it is within
    /// [`Self::snap_cents`] of one.
    ///
    /// OFF by default, and deliberately so. The whole argument for this mode is
    /// that a voice needs no grid, and snapping puts one back -- a shallower one
    /// (it only bites within a few cents) but a grid nonetheless, and it
    /// quantises real vibrato and real glissando away with the jitter. It is
    /// offered because a *stable* pitch that is 3 cents wrong may well sound
    /// better than a correct one that wobbles, and only a listener can settle
    /// that.
    /// A pitch at either end of the playable band whose nearest semitone lies
    /// OUTSIDE it is left where it is. `12*log2(10.0)` is 39.863, so every
    /// pitch from 4309 Hz to the top of the band rounds to step +40, which is
    /// `2^(40/12)` = 10.0794 -- past [`PITCH_MAX`]; symmetrically 44.00..45.3 Hz
    /// rounds to step -40 = 0.09921, under [`PITCH_MIN`]. Writing either put an
    /// illegal `PitchMultiplier` in the streams and killed `build_voice_world`
    /// AFTER the whole analysis had run, blaming the data for what was a flag
    /// value (any `--pitch-snap` at or above 13.69 cents reached it).
    ///
    /// Declining rather than snapping to the nearest LEGAL semitone: the legal
    /// one is a whole semitone further off (step 39 is 82 cents from 4390 Hz),
    /// so it would be a lurch, and the pitch that arrived here was already in
    /// range and already within half a semitone of the grid.
    fn snap(&self, pitch: f64) -> f64 {
        if self.snap_cents <= 0.0 {
            return pitch;
        }
        let (cents, step) = cents_from_equal_temperament(pitch * BASE_HZ as f64);
        if cents.abs() > self.snap_cents {
            return pitch;
        }
        let snapped = 2.0f64.powf(step as f64 / 12.0);
        if (PITCH_MIN as f64..=PITCH_MAX as f64).contains(&snapped) {
            snapped
        } else {
            pitch
        }
    }
}

/// Match this frame's peaks onto the running voices, then give the leftovers
/// somewhere to go.
///
/// **The single most important function in the mode.** Peaks arrive ranked by
/// strength and nothing else; without matching, voice *k* would play the k-th
/// loudest peak of each frame, an identity that changes several times a second.
/// What comes out is heard as random chimes -- which is precisely the complaint
/// this design answers.
///
/// The rule is McAulay-Quatieri partial tracking, in three steps:
///
/// 1. **Continue.** Every (voice, peak) pair within
///    [`MATCH_TOLERANCE_SEMITONES`] is a candidate; they are taken
///    closest-first, each voice and each peak used once. Greedy on distance
///    rather than optimal assignment because the two only disagree when two
///    voices are within half a semitone of each other AND of each other's
///    peaks, at which point the two orderings are inaudibly different.
/// 2. **Release.** A voice with no peak within tolerance does NOT jump to the
///    nearest one going spare; it ramps down over [`RELEASE_FRAMES`], holding
///    its pitch. A voice that is re-matched during its release resumes -- a
///    partial that dips below the prominence gate for one frame is a dropout,
///    not the end of a note.
/// 3. **Birth.** Unmatched peaks, strongest first, take a FREE voice. Nothing
///    else: a voice that is sounding -- whether it is following a peak or fading
///    out of one -- is never re-pointed at an unrelated frequency, and a peak
///    with nowhere to go is dropped. There are only so many speakers, and that
///    is the honest consequence.
///
/// Step 1 sees ALL the candidates, not just the `max_voices` strongest -- see
/// [`PEAK_OVERSAMPLE`], without which a held note dies whenever a louder
/// newcomer pushes its partial out of the top V for one frame.
///
/// Step 3's restriction is the whole difference between a note ending and a
/// chime. An earlier draft let a homeless peak steal the quietest RELEASING
/// voice, reasoning that its note was ending anyway. On speech at 8 voices that
/// fired constantly -- there is never a free slot -- and **20.7% of sounding
/// frame pairs jumped more than a semitone**: a speaker at half volume yanked
/// to an unrelated pitch, which is exactly the sound being designed out. The
/// cost of not stealing is up to [`RELEASE_FRAMES`] (67 ms) of latency before a
/// new note can start while every voice is busy, which is inaudible by
/// comparison.
///
/// Determinism matters as much as correctness here -- ties break on index
/// throughout, so the same audio always produces the same save.
fn assign(
    voices: &mut [VoiceState],
    peaks: &[Peak],
    shaping: &VoiceShaping,
    stats: &mut VoiceStats,
) {
    let n = voices.len();
    let mut peak_of_voice: Vec<Option<usize>> = vec![None; n];
    let mut voice_of_peak: Vec<Option<usize>> = vec![None; peaks.len()];

    // Step 1: continue. Candidate pairs, closest first; index order breaks
    // exact ties so the result never depends on sort stability.
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (v, voice) in voices.iter().enumerate() {
        if voice.is_free() {
            continue;
        }
        for (p, peak) in peaks.iter().enumerate() {
            let d = voice.distance(peak.hz);
            if d.is_finite() && d <= MATCH_TOLERANCE_SEMITONES {
                pairs.push((d, v, p));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    for (_, v, p) in pairs {
        if peak_of_voice[v].is_none() && voice_of_peak[p].is_none() {
            peak_of_voice[v] = Some(p);
            voice_of_peak[p] = Some(v);
        }
    }

    // Which voices were ALREADY free when the frame began. Births use this
    // rather than the post-release state, so a voice that finishes its release
    // on this frame writes its final zero and becomes available on the NEXT
    // one. Without the one-frame gap a voice is reborn on the same frame its
    // ramp reaches zero, so the frame before the new note still carries the old
    // note's last non-zero step -- a pitch jump on a sounding speaker, measured
    // at 3.74% of continuations on speech.
    let was_free: Vec<bool> = voices.iter().map(|v| v.is_free()).collect();

    // Step 2: release.
    //
    // A LINEAR ramp to exactly zero, not the level follower's exponential: a
    // one-pole never reaches zero, so a released voice would hold a shrinking
    // tail forever and its speaker could never be reused without a
    // discontinuity.
    //
    // The ramp is kept in `gain`, SEPARATE from the follower's `level`, and the
    // two are only multiplied at the far end of `analyze_voices` -- after
    // normalisation. Ramping `level` itself is what the previous version did
    // and it is the bug the owner heard as "it bleeds together": `--leveling`
    // divides every frame by its own loudest voice, so a frame whose loudest
    // content IS a fade gets that fade divided straight back out and the
    // release is written as a HOLD. Measured on speech at 32 voices, a released
    // voice was written at 0.050 mean volume against 0.071 for a voice actually
    // following a peak -- a "fade" to 70% of full level, sustained for four
    // frames, across 52.7% of all sounding voice-frames.
    for (v, voice) in voices.iter_mut().enumerate() {
        if peak_of_voice[v].is_some() {
            continue;
        }
        voice.matched = false;
        if voice.active {
            voice.active = false;
            voice.release_left = shaping.release_frames;
        }
        if voice.release_left > 0 {
            voice.release_left -= 1;
            voice.gain = voice.release_left as f32 / shaping.release_frames as f32;
        }
        // Exactly zero, and the LEVEL is cleared with it: a voice reborn on the
        // next frame must attack from silence, not from whatever the last note
        // was left frozen at.
        if voice.release_left == 0 {
            voice.gain = 0.0;
            voice.level = 0.0;
        }
        // `voice.pitch` is deliberately untouched: a silent voice holds its
        // last note.
    }

    // Step 3: birth. FREE slots only, in index order; peaks are already in rank
    // order, so the loudest homeless peak gets the lowest free slot.
    for p in 0..peaks.len() {
        if voice_of_peak[p].is_some() {
            continue;
        }
        let slot = (0..n)
            .filter(|&v| peak_of_voice[v].is_none() && was_free[v])
            .min();
        // No free voice: every speaker is already sounding something. The peak
        // is DROPPED rather than taking one over -- see the doc above.
        let Some(v) = slot else { break };
        peak_of_voice[v] = Some(p);
        voice_of_peak[p] = Some(v);
    }

    // Apply every match. Done last, in one pass, so the distances in step 1
    // were all measured against the PREVIOUS frame's pitches rather than
    // against pitches some earlier iteration had already moved.
    for (v, voice) in voices.iter_mut().enumerate() {
        let Some(p) = peak_of_voice[v] else { continue };
        let peak = peaks[p];
        let raw = (peak.hz / BASE_HZ as f64).clamp(PITCH_MIN as f64, PITCH_MAX as f64);
        // CONTINUING vs BORN, and the distinction is audible.
        //
        // A continuing voice glides: a sustained note's detected peak moves a
        // little every frame -- bin quantisation, interpolation noise, and
        // partials genuinely beating against each other -- and writing that
        // straight to `PitchMultiplier` is heard as a warble on what should be
        // a steady note ("sounds weird because the pitch wobbles a bit",
        // reported in game).
        //
        // A voice being BORN jumps. Gliding into a new note from wherever the
        // speaker happened to be left is a portamento swoop between unrelated
        // pitches, which is worse than the wobble it would be fixing.
        if voice.active {
            let glided = shaping.glide(voice.pitch, raw);
            stats.record_jitter(voice.raw_pitch, raw, voice.pitch, glided);
            voice.pitch = glided;
        } else {
            voice.pitch = shaping.snap(raw);
        }
        voice.raw_pitch = raw;
        // The level follower runs only while a voice is following a peak. Its
        // attack is what keeps a struck note punchy; its release is what stops
        // a level chattering between frames.
        //
        // It steps from what the voice was EMITTING, not from its frozen
        // `level`: a voice re-matched part-way down its release ramp -- a
        // partial that dipped under the prominence gate for a frame -- must
        // resume from where it audibly was, not jump back up to where the note
        // last peaked.
        voice.level = shaping.level.step(voice.emitted(), peak.mag);
        voice.gain = 1.0;
        voice.active = true;
        voice.matched = true;
        voice.release_left = 0;
    }
}

/// Reject a voice-mode speaker count of 0.
///
/// `--max-voices` is the voice COUNT in this mode, not a cap on how many of a
/// fixed bank may sound, so the flag's two meanings disagree about 0: a bank
/// with no selection is a legal render (every band sounds), a build with no
/// speakers is not.
///
/// Its own function, rather than four lines inside [`analyze_voices`], for the
/// same reason as [`crate::audio::track::check_subdiv`]: a front end that lets
/// the user switch a `0` from bank mode into voice mode has to refuse it with
/// the renderer's own words, and before it costs a build that cannot happen.
pub fn check_voice_count(max_voices: usize) -> Result<(), String> {
    if max_voices == 0 {
        return Err(
            "--max-voices must be at least 1 in --audio-mode voice: it is the number of \
             speakers built, not a cap on a fixed bank, so 0 would be a save with no \
             speakers in it"
                .to_string(),
        );
    }
    Ok(())
}

/// Decode, transform, find peaks, track them onto voices, and normalise in one
/// streaming pass.
///
/// # Normalisation is the bank's, verbatim
///
/// One global scale puts the largest per-frame `sqrt(sum of squares)` at
/// exactly `gain`, then `floor_db` zeroes anything that far below its own
/// frame's peak, then `leveling` optionally drags each frame toward full scale.
/// The reasoning -- why the incoherent sum and not the plain sum or the loudest
/// single voice, why the floor is frame-relative -- is on `track::analyze` and
/// applies here unchanged: a handful of uncorrelated sine voices is exactly the
/// same mixing problem as a bank of them, and the two modes must be comparable
/// at the same `--gain` or no A/B between them means anything.
///
/// `--max-voices` is the voice COUNT here, not a cap on how many of a fixed
/// bank may sound. 0 is rejected rather than meaning "no limit": a bank with no
/// selection is a legal render, a build with no speakers is not.
pub fn analyze_voices(
    source: &dyn AudioSource,
    opts: &AudioOptions,
    progress: &mut dyn Progress,
) -> Result<VoiceStreams, String> {
    // `!is_finite()` first throughout: NaN fails EVERY comparison, so a bare
    // `<= 0.0` lets `nan` straight through.
    if !opts.fps.is_finite() || opts.fps <= 0.0 {
        return Err(format!(
            "--audio-fps must be a positive finite number, got {}",
            opts.fps
        ));
    }
    if opts.max_frames == 0 {
        return Err("--max-frames must be at least 1".to_string());
    }
    if !opts.gain.is_finite() || opts.gain < 0.0 {
        return Err(format!(
            "--gain must be a non-negative finite number, got {}",
            opts.gain
        ));
    }
    if !opts.leveling.is_finite() || !(0.0..=1.0).contains(&opts.leveling) {
        return Err(format!(
            "--leveling must be between 0 (keep the track's dynamics) and 1 (flatten \
             every frame to full scale), got {}",
            opts.leveling
        ));
    }
    // Word for word `track::analyze`'s check, and it was missing here.
    // `Envelope::new` maps a negative or non-finite time constant to a
    // coefficient of exactly 1.0 -- the documented "no smoothing" setting -- so
    // `--audio-mode voice --attack nan` rendered a whole save with the level
    // follower silently disabled while the identical flags in bank mode stopped
    // with a message naming the flag. Same flag, same value, two behaviours.
    for (flag, value) in [("--attack", opts.attack_ms), ("--release", opts.release_ms)] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "{flag} must be a non-negative finite number of milliseconds \
                 (0 = no smoothing), got {value}"
            ));
        }
    }
    // Checked HERE, before a single frame is analysed, and not left to the
    // point of use: `VoiceShaping::new` reads this through `f32::max(0.0)`,
    // which IGNORES a NaN operand and would quietly turn `--pitch-snap nan`
    // into "off". See [`MAX_PITCH_SNAP_CENTS`] for the upper bound.
    if !opts.pitch_snap_cents.is_finite()
        || !(0.0..=MAX_PITCH_SNAP_CENTS).contains(&opts.pitch_snap_cents)
    {
        return Err(format!(
            "--pitch-snap must be between 0 (off) and {MAX_PITCH_SNAP_CENTS} cents -- half a \
             semitone, and no pitch is ever further than that from the nearest one, so a \
             larger value cannot mean anything a listener would hear differently from \
             {MAX_PITCH_SNAP_CENTS}. Got {}",
            opts.pitch_snap_cents
        ));
    }
    let n_voices = opts.max_voices;
    check_voice_count(n_voices)?;

    let info = source.info();
    let hop = hop_for(info.sample_rate, opts.fps)?;
    // See `track::build_audio_world`'s call for why this is `Exact` and what
    // the `None` arm now says instead of showing a bare, totalless spinner.
    let hint = info
        .duration_hint
        .map(|d| frame_count_for(d, info.sample_rate, opts.window, hop).min(opts.max_frames));
    FrameTotal::new(hint, None).begin(progress, "analyzing audio");

    // One ratio, used twice: to keep a peak out of the tracker (see
    // `find_peaks`) and to zero a written volume at the far end. They MUST be
    // the same number -- a peak that survives selection and is then floored
    // would leave a voice whose pitch moves while its volume reads zero.
    let floor_ratio = 10f32.powf(opts.floor_db / 20.0);
    let shaping = VoiceShaping::new(opts);
    let mut voices = vec![VoiceState::idle(); n_voices];
    let mut raw_pitch: Vec<Vec<f64>> = vec![Vec::new(); n_voices];
    // The follower level and the release ramp, kept APART all the way to the
    // end. Everything normalisation looks at is the level; the ramp is applied
    // once, last, so no per-frame scaling can flatten it. See the release step
    // in `assign`.
    let mut raw_level: Vec<Vec<f32>> = vec![Vec::new(); n_voices];
    let mut raw_gain: Vec<Vec<f32>> = vec![Vec::new(); n_voices];
    // Measurement only: whether each voice was following a peak on each frame.
    // Kept alongside the levels rather than counted on the fly because the
    // question it answers -- was this voice sounding while nothing in the
    // spectrum matched it -- can only be asked of the FINISHED volumes, which
    // do not exist until normalisation has run.
    let mut raw_matched: Vec<Vec<bool>> = vec![Vec::new(); n_voices];
    let mut stats = VoiceStats::new(n_voices);

    let collected: Result<usize, String> = (|| {
        let mut stft = StftStream::new(source.open()?, opts.window, hop)?;
        let mut n = 0usize;
        while n < opts.max_frames {
            let Some(spectrum) = stft.next_spectrum()? else {
                break;
            };
            let peaks = find_peaks(
                &spectrum,
                info.sample_rate,
                opts.window,
                n_voices.saturating_mul(PEAK_OVERSAMPLE),
                floor_ratio,
            );
            assign(&mut voices, &peaks, &shaping, &mut stats);
            for (v, voice) in voices.iter().enumerate() {
                raw_pitch[v].push(voice.pitch);
                raw_level[v].push(voice.level);
                raw_gain[v].push(voice.gain);
                raw_matched[v].push(voice.matched);
            }
            // Tuning and harmonic statistics are taken over the voices that are
            // FOLLOWING a peak this frame, not over the candidate peaks. Only
            // the former reach a speaker: a candidate that found no free voice
            // is never heard, and counting it would report a tuning the render
            // does not have.
            let sounding: Vec<f64> = voices
                .iter()
                .filter(|v| v.active)
                .map(|v| v.pitch * BASE_HZ as f64)
                .collect();
            for &hz in &sounding {
                let (cents, step) = cents_from_equal_temperament(hz);
                stats.peak_count += 1;
                stats.mean_abs_cents += cents.abs();
                let bucket = (cents.round() as i64 + 50).clamp(0, 100) as usize;
                stats.cents_hist[bucket] += 1;
                let idx = (step - NOTE_HIST_LOW).clamp(0, NOTE_HIST_LEN as i32 - 1) as usize;
                stats.note_hist[idx] += 1;
            }
            let (harmonics, roots) = harmonic_split(&sounding);
            stats.harmonic_voices += harmonics;
            if !sounding.is_empty() {
                let slot = roots.min(stats.fundamentals_hist.len() - 1);
                stats.fundamentals_hist[slot] += 1;
            }
            n += 1;
            progress.tick(n as u64);
            if progress.is_cancelled() {
                break;
            }
        }
        Ok(n)
    })();
    progress.finish();
    let frame_count = collected?;

    if frame_count == 0 {
        return Err(
            "audio produced 0 analysis frames -- the source is shorter than one STFT window \
             (try a smaller --window, or check --start/--duration)"
                .to_string(),
        );
    }
    if stats.peak_count > 0 {
        stats.mean_abs_cents /= stats.peak_count as f64;
    }

    // Four statistics in one pass over the voice-major store, exactly as
    // `track::analyze` takes its three over the band-major one.
    //
    // Two of them are MIXES -- `sqrt(sum of squares)`, the height N
    // uncorrelated sinusoids actually reach together -- and they are taken over
    // DIFFERENT quantities, which is the point:
    //
    // * `out_mix[f]` (and its maximum `mix_peak`) is the sound that is actually
    //   made, i.e. the level THROUGH the release ramp. It sets the one global
    //   scale, and it is what the output must be bounded against.
    // * `level_mix[f]` (and its maximum `level_peak`) is the same sum with the
    //   ramp left OUT. It is the per-frame AGC reference, and it must be
    //   ramp-free: a frame whose loudest content is a note fading out is the END
    //   of a loud passage, not a quiet one, and normalising a fade against
    //   itself is what deleted the fade.
    //
    // `frame_peak[f]`, the loudest single voice in the frame, is only the
    // floor's reference. It USED to be the AGC's as well, and that was H1: the
    // global scale targeted a mix while the boost divided by a peak, so a frame
    // with six voices sounding took the boost a one-voice frame had earned and
    // the output reached a measured 2.413x full scale at the shipped
    // `--leveling 1.0`. See `track::analyze`.
    let mut mix_peak = 0.0f64;
    let mut level_peak = 0.0f64;
    let mut frame_peak = vec![0.0f32; frame_count];
    let mut out_mix = vec![0.0f64; frame_count];
    let mut level_mix = vec![0.0f64; frame_count];
    for (f, fp) in frame_peak.iter_mut().enumerate() {
        let mut out_sq = 0.0f64;
        let mut level_sq = 0.0f64;
        for (v, voice) in raw_level.iter().enumerate() {
            let level = voice[f];
            if level > *fp {
                *fp = level;
            }
            level_sq += level as f64 * level as f64;
            let out = level * raw_gain[v][f];
            out_sq += out as f64 * out as f64;
        }
        out_mix[f] = out_sq.sqrt();
        level_mix[f] = level_sq.sqrt();
        if out_mix[f] > mix_peak {
            mix_peak = out_mix[f];
        }
        if level_mix[f] > level_peak {
            level_peak = level_mix[f];
        }
    }

    let base = if mix_peak > 0.0 {
        opts.gain / mix_peak as f32
    } else {
        0.0
    };
    let scale: Vec<f32> = (0..frame_count)
        .map(|f| {
            let levelled = if opts.leveling <= 0.0 || level_mix[f] <= 0.0 {
                base
            } else {
                base * ((level_peak / level_mix[f]) as f32)
                    .powf(opts.leveling)
                    .min(MAX_LEVELING_BOOST)
            };
            // The headroom this frame actually has, and the last word.
            //
            // In the bank the AGC reference and the normaliser are the same
            // quantity, so `mix * scale <= gain` is algebraic and no ceiling is
            // needed. Here they deliberately are not -- the AGC ignores the
            // release ramp so it cannot flatten a fade -- and the gap between
            // them is real: if the frame with the loudest ramp-FREE mix happens
            // to be one where every voice is mid-fade, `level_peak` exceeds
            // anything `mix_peak` saw and the boost derived from it can ask for
            // more than full scale. This is what stops it, and it is inert
            // wherever the two agree.
            if out_mix[f] > 0.0 {
                levelled.min(opts.gain / out_mix[f] as f32)
            } else {
                levelled
            }
        })
        .collect();

    // The release ramp is applied HERE, after the floor and after both scales,
    // so that nothing downstream can undo it and a finished ramp is exactly
    // 0.0 rather than a small residual.
    let volumes: Vec<Vec<f64>> = raw_level
        .into_iter()
        .zip(raw_gain.iter())
        .map(|(voice, gains)| {
            voice
                .into_iter()
                .enumerate()
                .map(|(f, v)| {
                    if v < frame_peak[f] * floor_ratio {
                        0.0
                    } else {
                        (v * scale[f]).min(1.0) as f64 * gains[f] as f64
                    }
                })
                .collect()
        })
        .collect();

    // Lifetimes, sounding counts and pitch jumps are measured on the FINISHED
    // arrays, not on the tracker's own bookkeeping. A tracker that believed it
    // was continuing a voice while writing an unrelated pitch would report
    // itself healthy; the arrays cannot.
    for (v, vol) in volumes.iter().enumerate() {
        let mut run = 0usize;
        // Frames since this run was last matched to a peak. When the run ends,
        // that count IS the time from the partial's end to the voice's zero.
        let mut since_matched = 0usize;
        for f in 0..frame_count {
            if vol[f] > 0.0 {
                stats.sounding += 1;
                run += 1;
                if raw_matched[v][f] {
                    since_matched = 0;
                    stats.matched_volume += vol[f];
                } else {
                    stats.unmatched_sounding += 1;
                    stats.unmatched_volume += vol[f];
                    since_matched += 1;
                }
                if f > 0 && vol[f - 1] > 0.0 {
                    stats.continuations += 1;
                    let ratio = raw_pitch[v][f] / raw_pitch[v][f - 1];
                    if (12.0 * ratio.log2()).abs() > 1.0 {
                        stats.pitch_jumps += 1;
                    }
                }
            } else if run > 0 {
                stats.lifetimes.push(run);
                stats.tail_frames.push(since_matched);
                run = 0;
                since_matched = 0;
            }
        }
        if run > 0 {
            stats.lifetimes.push(run);
            stats.tail_frames.push(since_matched);
        }
    }

    Ok(VoiceStreams {
        pitches: raw_pitch,
        volumes,
        fps: opts.fps,
        frame_count,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::source::SampleClip;
    use crate::audio::track::DEFAULT_VOICE_RELEASE_MS;
    use crate::progress::NoProgress;
    use std::f32::consts::TAU;

    const SR: u32 = 48_000;

    fn opts(max_voices: usize, window: usize) -> AudioOptions {
        AudioOptions {
            max_voices,
            window,
            ..Default::default()
        }
    }

    /// Sum of sinusoids, each `(hz, amplitude)`, held for `secs`.
    fn tones(parts: &[(f32, f32)], secs: f32) -> SampleClip {
        let n = (SR as f32 * secs) as usize;
        SampleClip::new(
            SR,
            (0..n)
                .map(|i| {
                    let t = i as f32 / SR as f32;
                    parts.iter().map(|(f, a)| a * (TAU * f * t).sin()).sum()
                })
                .collect(),
        )
    }

    /// Sustained tones with a real vibrato: `depth_cents` peak deviation at
    /// [`VIBRATO_HZ`], held for `secs`.
    ///
    /// The modulation goes in the PHASE, integrated, not on the frequency term
    /// of a `sin(TAU * f * v(t) * t)` expression. That shortcut looks like
    /// vibrato and is not: differentiating it leaves a `t * v'(t)` term, so the
    /// excursion grows without bound with time -- an early draft of this fixture
    /// swung ±45% by three seconds, which is a glissando across half the
    /// keyboard rather than a note wobbling.
    fn vibrato(freqs: &[f32], depth_cents: f32, secs: f32) -> SampleClip {
        const VIBRATO_HZ: f32 = 4.0;
        let n = (SR as f32 * secs) as usize;
        SampleClip::new(
            SR,
            (0..n)
                .map(|i| {
                    let t = i as f32 / SR as f32;
                    freqs
                        .iter()
                        .map(|f| {
                            // Peak deviation in Hz, then in radians of phase.
                            let dev = f * (2f32.powf(depth_cents / 1200.0) - 1.0);
                            let beta = dev / VIBRATO_HZ;
                            0.3 * (TAU * f * t + beta * (TAU * VIBRATO_HZ * t).sin()).sin()
                        })
                        .sum()
                })
                .collect(),
        )
    }

    fn analyze(clip: &SampleClip, o: &AudioOptions) -> VoiceStreams {
        analyze_voices(clip, o, &mut NoProgress).expect("analysis")
    }

    // -- --pitch-snap ------------------------------------------------------

    /// **The unit gate on H2.** The nearest equal-tempered semitone to a pitch
    /// at either end of the playable band lies OUTSIDE the band, and snapping
    /// onto it wrote a `PitchMultiplier` the emitter cannot play.
    ///
    /// `12*log2(10.0)` = 39.863, so every pitch from 4309 Hz up rounds to step
    /// +40 = 10.0794 (past `PITCH_MAX` = 10.0); symmetrically 44.00..45.3 Hz
    /// rounds to step -40 = 0.09921 (under `PITCH_MIN` = 0.1). Any
    /// `--pitch-snap` at or above 13.69 cents -- squarely inside the range the
    /// flag is documented for -- reached it.
    #[test]
    fn snapping_at_the_edge_of_the_band_never_leaves_the_playable_range() {
        let shaping = VoiceShaping::new(&AudioOptions {
            pitch_snap_cents: MAX_PITCH_SNAP_CENTS,
            ..Default::default()
        });
        // Every Hz the tracker can admit (`find_peaks` gates on
        // `min_hz()..=max_hz()`), at a resolution fine enough to land inside
        // both edge zones.
        let mut hz = min_hz();
        while hz <= max_hz() {
            let snapped = shaping.snap(hz / BASE_HZ as f64);
            assert!(
                (PITCH_MIN as f64..=PITCH_MAX as f64).contains(&snapped),
                "snapping {hz:.2} Hz gave {snapped}, outside the emitter's legal \
                 {PITCH_MIN}..{PITCH_MAX} -- `build_voice_world` refuses this, AFTER the \
                 whole analysis has run, with an error that blames the data"
            );
            hz *= 1.0005;
        }
    }

    /// The same property end to end, on the fixture H2 was measured with: a
    /// tone at each end of the band, `--pitch-snap` well past the 13.69 cent
    /// threshold. Nothing may reach the streams that
    /// `speakers::build_voice_world` would refuse.
    #[test]
    fn a_pitch_snap_past_the_threshold_still_writes_only_legal_pitches() {
        let n = (SR as f32 * 3.0) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                0.5 * (TAU * 4390.0 * t).sin() + 0.5 * (TAU * 44.05 * t).sin()
            })
            .collect();
        for snap in [0.0, 14.0, 20.0, MAX_PITCH_SNAP_CENTS] {
            let o = AudioOptions { pitch_snap_cents: snap, ..opts(4, 16384) };
            let t = analyze(&SampleClip::new(SR, samples.clone()), &o);
            for (v, row) in t.pitches.iter().enumerate() {
                for (f, &p) in row.iter().enumerate() {
                    assert!(
                        p.is_finite() && (PITCH_MIN as f64..=PITCH_MAX as f64).contains(&p),
                        "--pitch-snap {snap}: voice {v} frame {f} pitch {p} is outside \
                         {PITCH_MIN}..{PITCH_MAX}"
                    );
                }
            }
        }
    }

    /// Snapping still SNAPS. The legality guard above is a decline at the two
    /// edges of the band, not a way to turn the feature off everywhere.
    #[test]
    fn a_pitch_within_the_snap_window_is_still_pulled_onto_the_semitone() {
        let shaping =
            VoiceShaping::new(&AudioOptions { pitch_snap_cents: 10.0, ..Default::default() });
        // 442 Hz is 7.85 cents sharp of A440: inside the window, and nowhere
        // near either edge of the band.
        let snapped = shaping.snap(442.0 / BASE_HZ as f64);
        assert!(
            (snapped - 1.0).abs() < 1e-9,
            "442 Hz is 7.85 cents from A440 and must snap onto it, got {snapped}"
        );
        // 450 Hz is 38.9 cents sharp: outside it, and must be left alone.
        let untouched = shaping.snap(450.0 / BASE_HZ as f64);
        assert!(
            (untouched - 450.0 / BASE_HZ as f64).abs() < 1e-9,
            "450 Hz is 38.9 cents from A440, outside a 10 cent window, got {untouched}"
        );
    }

    /// An out-of-band `--pitch-snap` is refused BY NAME and BEFORE the
    /// analysis, not carried through it. NaN included: `VoiceShaping::new`
    /// reads this through `f32::max(0.0)`, which ignores a NaN operand, so a
    /// bare check written the other way round turns `nan` into "off".
    #[test]
    fn an_out_of_range_pitch_snap_is_an_error() {
        for bad in [-1.0f32, MAX_PITCH_SNAP_CENTS + 0.5, 1200.0, f32::NAN, f32::INFINITY] {
            let o = AudioOptions { pitch_snap_cents: bad, ..opts(4, 4096) };
            let Err(err) = analyze_voices(&tones(&[(440.0, 0.5)], 1.0), &o, &mut NoProgress)
            else {
                panic!("--pitch-snap {bad} must be rejected")
            };
            assert!(err.contains("--pitch-snap"), "the error must name the flag: {err}");
        }
    }

    /// **M3.** `--attack` / `--release` are validated on THIS path too, not
    /// only in bank mode. `Envelope::new` maps a negative or non-finite time
    /// constant to a coefficient of 1.0 -- "no smoothing" -- so an unchecked
    /// value here rendered a whole save with the level follower silently off,
    /// while the identical flags in bank mode stopped with a named error.
    #[test]
    fn a_non_finite_or_negative_attack_or_release_is_an_error() {
        for bad in [f32::NAN, f32::INFINITY, -5.0] {
            for (flag, o) in [
                ("--attack", AudioOptions { attack_ms: bad, ..opts(4, 4096) }),
                ("--release", AudioOptions { release_ms: bad, ..opts(4, 4096) }),
            ] {
                let Err(err) = analyze_voices(&tones(&[(440.0, 0.5)], 1.0), &o, &mut NoProgress)
                else {
                    panic!("{flag} {bad} must be rejected in voice mode too")
                };
                assert!(err.contains(flag), "the error must name the flag: {err}");
            }
        }
    }

    fn spectrum_of(clip: &SampleClip, window: usize) -> Vec<f32> {
        let mut s = StftStream::new(clip.open().expect("open"), window, 1600).expect("stft");
        let mut last = None;
        for _ in 0..3 {
            if let Some(sp) = s.next_spectrum().expect("spectrum") {
                last = Some(sp);
            }
        }
        last.expect("at least one spectrum")
    }

    fn semitones_between(a: f64, b: f64) -> f64 {
        (12.0 * (a / b).log2()).abs()
    }

    /// [`find_peaks`] at the renderer's own default level floor
    /// (`AudioOptions::floor_db`, -60 dB below the frame's loudest peak), so
    /// the peak tests and a real render agree about what is a note.
    fn peaks_of(spectrum: &[f32], sr: u32, window: usize, max_peaks: usize) -> Vec<Peak> {
        let floor = 10f32.powf(AudioOptions::default().floor_db / 20.0);
        find_peaks(spectrum, sr, window, max_peaks, floor)
    }

    // -- peak refinement ---------------------------------------------------

    /// **The mutation gate on parabolic interpolation.**
    ///
    /// The tone sits exactly half a bin above a bin centre, which is the
    /// worst case for a bin-quantised peak finder, and it sits LOW, where a
    /// bin is many cents wide. At `--window 4096` a bin is 11.72 Hz, so half a
    /// bin at ~220 Hz is 42 cents -- audibly out of tune, and no better than the
    /// band grid this mode exists to remove. Interpolation must bring that
    /// under 5 cents, the threshold where detuning stops being audible.
    ///
    /// Asserted against the ANALYTIC bin, so the test cannot drift with the
    /// implementation.
    #[test]
    fn parabolic_interpolation_puts_a_between_bins_tone_within_five_cents() {
        let window = 4096usize;
        let hz_per_bin = SR as f64 / window as f64;
        let bin = 19.0;
        let freq = (bin + 0.5) * hz_per_bin; // 228.8 Hz, dead between two bins
        let uninterpolated_error =
            (1200.0 * ((bin * hz_per_bin) / freq).log2()).abs();
        assert!(
            uninterpolated_error > 30.0,
            "the fixture must be a hard case for a bin-quantised finder: \
             rounding to bin {bin} is {uninterpolated_error:.1} cents off"
        );

        let sp = spectrum_of(&tones(&[(freq as f32, 1.0)], 0.5), window);
        let peaks = peaks_of(&sp, SR, window, 4);
        assert!(!peaks.is_empty(), "a pure tone must produce a peak");
        let err = (1200.0 * (peaks[0].hz / freq).log2()).abs();
        assert!(
            err < 5.0,
            "a tone half a bin off centre must be located within 5 cents, got \
             {err:.2} ({} Hz vs {freq} Hz) -- without parabolic interpolation this \
             is {uninterpolated_error:.1}",
            peaks[0].hz
        );
    }

    /// The same property across the range, and at the window a real render
    /// uses. One frequency could pass by luck.
    #[test]
    fn every_peak_is_located_within_a_few_cents_of_the_true_tone() {
        for window in [4096usize, 16384] {
            let hz_per_bin = SR as f64 / window as f64;
            for bin_offset in [0.0f64, 0.25, 0.5, -0.3] {
                for base_bin in [20.0f64, 90.0, 400.0] {
                    let freq = (base_bin + bin_offset) * hz_per_bin;
                    if freq < min_hz() || freq > max_hz() {
                        continue;
                    }
                    let sp = spectrum_of(&tones(&[(freq as f32, 1.0)], 0.6), window);
                    let peaks = peaks_of(&sp, SR, window, 4);
                    assert!(
                        !peaks.is_empty(),
                        "window {window}, {freq} Hz: a pure tone must produce a peak"
                    );
                    let err = (1200.0 * (peaks[0].hz / freq).log2()).abs();
                    assert!(
                        err < 8.0,
                        "window {window}, {freq} Hz: located at {} Hz, {err:.2} cents off",
                        peaks[0].hz
                    );
                }
            }
        }
    }

    /// A peak is the note, not the three bins around it. Without the
    /// local-maximum test one tone would burn three voice slots.
    #[test]
    fn one_tone_produces_exactly_one_peak() {
        let sp = spectrum_of(&tones(&[(880.0, 1.0)], 0.5), 4096);
        let peaks = peaks_of(&sp, SR, 4096, 8);
        assert_eq!(
            peaks.len(),
            1,
            "a single sine must light exactly one peak, got {:?}",
            peaks.iter().map(|p| p.hz).collect::<Vec<_>>()
        );
    }

    #[test]
    fn separate_tones_produce_separate_peaks() {
        let sp = spectrum_of(&tones(&[(220.0, 1.0), (660.0, 1.0), (1760.0, 1.0)], 0.5), 8192);
        let peaks = peaks_of(&sp, SR, 8192, 8);
        for want in [220.0f64, 660.0, 1760.0] {
            assert!(
                peaks.iter().any(|p| semitones_between(p.hz, want) < 0.1),
                "{want} Hz must be found; got {:?}",
                peaks.iter().map(|p| p.hz).collect::<Vec<_>>()
            );
        }
    }

    /// Unplayable content must be DROPPED, not clamped. A 6 kHz peak handed to
    /// a voice becomes 4400 Hz in game -- a wrong note rather than nothing.
    #[test]
    fn peaks_outside_the_emitters_pitch_range_are_dropped() {
        let sp = spectrum_of(&tones(&[(8000.0, 1.0), (20.0, 1.0)], 0.5), 8192);
        let peaks = peaks_of(&sp, SR, 8192, 16);
        for p in &peaks {
            assert!(
                p.hz >= min_hz() && p.hz <= max_hz(),
                "{} Hz is outside the playable {}..{} range",
                p.hz,
                min_hz(),
                max_hz()
            );
        }
    }

    #[test]
    fn silence_produces_no_peaks() {
        let sp = spectrum_of(&SampleClip::new(SR, vec![0.0; SR as usize]), 4096);
        assert!(peaks_of(&sp, SR, 4096, 8).is_empty());
    }

    /// The prominence gate is what stops broadband content becoming notes.
    /// White noise has local maxima everywhere; almost none of them stand
    /// clear of their neighbourhood.
    #[test]
    fn broadband_noise_yields_far_fewer_peaks_than_it_has_local_maxima() {
        // Deterministic pseudo-noise -- a fixed LCG, so the test cannot flake.
        let mut state = 0x2545F491_4F6CDD1Du64;
        let samples: Vec<f32> = (0..SR as usize)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect();
        let sp = spectrum_of(&SampleClip::new(SR, samples), 4096);
        let maxima = (1..sp.len() - 1)
            .filter(|&i| sp[i] > 0.0 && sp[i] > sp[i - 1] && sp[i] > sp[i + 1])
            .count();
        let gated = peaks_of(&sp, SR, 4096, usize::MAX).len();
        assert!(maxima > 100, "noise must have many local maxima, got {maxima}");
        assert!(
            (gated as f64) < 0.35 * maxima as f64,
            "the prominence gate must reject most of noise's local maxima: \
             {gated} of {maxima} survived"
        );
    }

    // -- tracking ----------------------------------------------------------

    /// **The property the whole design rests on, and the one nothing else
    /// asserts.**
    ///
    /// Three sustained tones with a slow ±10 cent vibrato -- real notes wander,
    /// and a tracker that only ever matches an EXACTLY repeated frequency is
    /// not tracking. Every voice that sounds must stay on its note for most of
    /// the clip, not fire and die.
    ///
    /// A match tolerance set far too narrow makes every voice die after one
    /// frame and fails here. So does removing the matching step in a way that
    /// leaves voices unable to continue.
    #[test]
    fn sustained_tones_give_voices_lifetimes_of_many_frames() {
        let t = analyze(&vibrato(&[220.0, 330.0, 550.0], 10.0, 3.0), &opts(6, 4096));
        assert!(t.frame_count > 60, "need a long clip, got {}", t.frame_count);
        let mean = t.stats.mean_lifetime();
        assert!(
            mean > 20.0,
            "sustained tones must hold their voices: mean lifetime {mean:.1} frames \
             over {} run(s) in {} frames. Anything near 1-2 means the tracker is \
             re-assigning voices every frame, which sounds like chimes however \
             correct the pitches are.",
            t.stats.lifetimes.len(),
            t.frame_count
        );
    }

    /// **The mutation gate on [`PEAK_OVERSAMPLE`].**
    ///
    /// Three sustained tones against two voices. The 220 Hz tone is steady at
    /// medium level while the other two swap loud and quiet either side of it,
    /// so 220 Hz alternates between the loudest peak in the frame and the
    /// THIRD loudest -- repeatedly, while never going anywhere.
    ///
    /// Truncating the candidate list to `max_voices` before matching therefore
    /// kills the voice on 220 Hz every time it slips to third and reborns it a
    /// second later. Measured on this fixture: mean voice lifetime 175 frames
    /// (the whole clip, 2 runs) with the oversampled list, **49.3 frames over 7
    /// runs** without it. On real speech the same defect put 20.7% of sounding
    /// frame pairs more than a semitone apart.
    ///
    /// Handing the matcher a longer list is the McAulay-Quatieri rule: a
    /// running voice keeps its partial as long as the partial is a detectable
    /// peak AT ALL, and only the leftover slots go to newcomers by rank.
    #[test]
    fn a_tracked_partial_survives_falling_out_of_the_top_ranks() {
        let secs = 6.0f32;
        let n = (SR as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                let swap = (TAU * 0.5 * t).sin();
                0.30 * (TAU * 220.0 * t).sin()
                    + (0.20 + 0.15 * swap) * (TAU * 330.0 * t).sin()
                    + (0.20 - 0.15 * swap) * (TAU * 550.0 * t).sin()
            })
            .collect();
        let t = analyze(&SampleClip::new(SR, samples), &opts(2, 8192));
        assert!(t.frame_count > 100, "need a long clip, got {}", t.frame_count);
        let mean = t.stats.mean_lifetime();
        assert!(
            mean > t.frame_count as f64 * 0.8,
            "no tone ever stops, so a voice that takes one must keep it for the whole clip:              mean lifetime {mean:.1} frames of {} over {} run(s). Truncating the candidate              list to --max-voices before matching drops a partial the moment two others              outrank it, and the voice dies and is reborn instead of holding.",
            t.frame_count,
            t.stats.lifetimes.len()
        );
    }

    /// **The mutation gate on tracking itself.**
    ///
    /// Two tones a fifth apart whose levels CROSS OVER: A starts loud and
    /// fades, B starts quiet and grows. Ranked by strength, the "first" peak is
    /// A at the start and B at the end, so a renderer that assigns peaks to
    /// voices by rank each frame swaps both voices at the crossover -- a
    /// 7-semitone jump on a sounding speaker, in both directions at once.
    /// Matching by frequency keeps each voice on its own tone throughout.
    ///
    /// There are more voices than tones, so no voice is ever stolen: every
    /// pitch jump this test can see is a tracking failure.
    #[test]
    fn a_level_crossover_does_not_swap_voices_between_notes() {
        let secs = 2.0f32;
        let n = (SR as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                let k = t / secs;
                0.4 * (1.0 - k) * (TAU * 440.0 * t).sin() + 0.4 * k * (TAU * 660.0 * t).sin()
            })
            .collect();
        let t = analyze(&SampleClip::new(SR, samples), &opts(6, 4096));

        assert!(
            t.stats.continuations > 50,
            "the fixture must produce sustained voices, got {} continuations",
            t.stats.continuations
        );
        assert_eq!(
            t.stats.pitch_jumps, 0,
            "no sounding voice may jump more than a semitone here: the two tones \
             never move, so every jump is a voice swapping notes. {} of {} \
             continuations jumped.",
            t.stats.pitch_jumps, t.stats.continuations
        );

        // ...and each voice that sounds must be ON one of the two tones,
        // not somewhere between them.
        for v in 0..t.voice_count() {
            for f in 0..t.frame_count {
                if t.volumes[v][f] <= 0.0 {
                    continue;
                }
                let hz = t.pitches[v][f] * BASE_HZ as f64;
                assert!(
                    semitones_between(hz, 440.0) < 0.6 || semitones_between(hz, 660.0) < 0.6,
                    "voice {v} frame {f} sounds at {hz:.1} Hz, which is neither tone"
                );
            }
        }
    }

    /// A tolerance set far too WIDE lets a voice follow whatever is nearest,
    /// including the other note. Two tones three semitones apart, alternating
    /// in level, must still never share a voice.
    #[test]
    fn voices_do_not_wander_between_notes_a_minor_third_apart() {
        let secs = 2.0f32;
        let n = (SR as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                let k = (TAU * 0.75 * t).sin() * 0.5 + 0.5;
                0.4 * k * (TAU * 440.0 * t).sin() + 0.4 * (1.0 - k) * (TAU * 523.25 * t).sin()
            })
            .collect();
        let t = analyze(&SampleClip::new(SR, samples), &opts(6, 4096));
        for v in 0..t.voice_count() {
            let mut seen: Option<f64> = None;
            for f in 0..t.frame_count {
                if t.volumes[v][f] <= 0.0 {
                    // A voice that went silent may legitimately be reborn on
                    // the other note -- that is a new note, not a wander.
                    seen = None;
                    continue;
                }
                let hz = t.pitches[v][f] * BASE_HZ as f64;
                if let Some(prev) = seen {
                    assert!(
                        semitones_between(hz, prev) < 1.0,
                        "voice {v} moved from {prev:.1} Hz to {hz:.1} Hz at frame {f} \
                         while still sounding -- a match tolerance wide enough to \
                         span a minor third makes voices wander between notes"
                    );
                }
                seen = Some(hz);
            }
        }
    }

    /// A voice whose partial ends must RAMP DOWN, not cut. The last frame
    /// before silence must be quieter than the note was.
    ///
    /// Measured with `--leveling 0`, and that is not the test dodging the
    /// default. Full leveling divides each frame by its own peak, so a frame
    /// whose loudest content IS the decaying tail is normalised straight back
    /// up (to the `MAX_LEVELING_BOOST` cap) and the fade is flattened out of
    /// the written values. That is what an automatic gain control does and it
    /// is what the owner asked for; the release still shapes the tail relative
    /// to everything else sounding, and still lengthens the run of non-zero
    /// frames, which is the beeping metric. This test is about the ramp
    /// existing at all, so it looks at it without the AGC on top.
    #[test]
    fn a_voice_whose_note_ends_fades_rather_than_cutting() {
        // One tone for the first half, silence after.
        let secs = 2.0f32;
        let n = (SR as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                if t < secs / 2.0 { 0.5 * (TAU * 440.0 * t).sin() } else { 0.0 }
            })
            .collect();
        let t = analyze(
            &SampleClip::new(SR, samples),
            &AudioOptions { leveling: 0.0, ..opts(4, 4096) },
        );
        let mut faded = false;
        for v in 0..t.voice_count() {
            for f in 1..t.frame_count {
                if t.volumes[v][f] > 0.0
                    && t.volumes[v][f] < t.volumes[v][f - 1]
                    && f + 1 < t.frame_count
                    && t.volumes[v][f + 1] == 0.0
                {
                    faded = true;
                }
            }
        }
        assert!(
            faded,
            "at least one voice must end on a frame quieter than the one before it -- \
             a note that cuts straight to zero is a click"
        );
    }

    // -- output invariants -------------------------------------------------

    /// **The mutation gate on "write 0.0 when silent".**
    ///
    /// 0.0 is below the emitter's `PITCH_MIN`, so the game clamps it to 0.1 and
    /// plays 44 Hz -- a lurch to the bottom of the range at the end of every
    /// note, on every voice, audible and inexplicable.
    #[test]
    fn every_pitch_is_always_a_legal_playable_multiplier() {
        let t = analyze(&tones(&[(440.0, 0.5), (554.0, 0.4)], 2.0), &opts(5, 4096));
        for (v, row) in t.pitches.iter().enumerate() {
            assert_eq!(row.len(), t.frame_count);
            for (f, &p) in row.iter().enumerate() {
                assert!(
                    p.is_finite() && p >= PITCH_MIN as f64 && p <= PITCH_MAX as f64,
                    "voice {v} frame {f} pitch {p} is outside the emitter's legal \
                     {PITCH_MIN}..{PITCH_MAX} range -- a silent voice must HOLD its \
                     last pitch, never be written as 0"
                );
            }
        }
    }

    /// A silent voice holds the pitch it last sounded at. Anything else is a
    /// jump on a speaker that may be about to be reused.
    #[test]
    fn a_silent_voice_holds_its_last_pitch() {
        let secs = 2.0f32;
        let n = (SR as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                if t < secs / 2.0 { 0.5 * (TAU * 440.0 * t).sin() } else { 0.0 }
            })
            .collect();
        let t = analyze(&SampleClip::new(SR, samples), &opts(4, 4096));
        for v in 0..t.voice_count() {
            for f in 1..t.frame_count {
                // Silent this frame AND silent last frame: nothing can have
                // moved the pitch, because nothing was assigned to it.
                if t.volumes[v][f] == 0.0 && t.volumes[v][f - 1] == 0.0 {
                    assert_eq!(
                        t.pitches[v][f], t.pitches[v][f - 1],
                        "voice {v} pitch changed at frame {f} while silent in both frames"
                    );
                }
            }
        }
    }

    #[test]
    fn the_streams_are_rectangular_and_two_per_voice() {
        let t = analyze(&tones(&[(440.0, 0.5)], 1.5), &opts(7, 4096));
        assert_eq!(t.voice_count(), 7);
        assert_eq!(t.volumes.len(), 7);
        assert!(t.frame_count > 0);
        for row in t.pitches.iter().chain(t.volumes.iter()) {
            assert_eq!(row.len(), t.frame_count);
        }
    }

    #[test]
    fn every_volume_is_finite_and_in_range() {
        let t = analyze(&tones(&[(440.0, 0.5), (880.0, 0.3)], 2.0), &opts(6, 4096));
        for row in &t.volumes {
            for &v in row {
                assert!((0.0..=1.0).contains(&v) && v.is_finite(), "volume {v}");
            }
        }
    }

    /// Normalisation must reach full scale, exactly as the bank's does -- the
    /// two modes are compared at the same `--gain` and a quieter one would
    /// read as "worse" for no reason but a missing scale.
    #[test]
    fn the_loudest_frames_incoherent_mix_reaches_full_scale() {
        let t = analyze(&tones(&[(440.0, 0.5), (660.0, 0.4)], 2.0), &opts(6, 4096));
        let mut best = 0.0f64;
        for f in 0..t.frame_count {
            let sq: f64 = t.volumes.iter().map(|v| v[f] * v[f]).sum();
            best = best.max(sq.sqrt());
        }
        assert!(
            (best - 1.0).abs() < 0.02,
            "the loudest incoherent mix must land on 1.0, got {best}"
        );
    }

    /// A loud passage carried by ONE partial, then a quieter, denser one --
    /// `track::tests::one_partial_then_six`'s fixture, in voice mode.
    fn one_partial_then_six(loud: f32, quiet: f32) -> SampleClip {
        let freqs = [440.0f32, 554.365, 659.255, 880.0, 1108.731, 1318.51];
        let n = (SR as f32 * 4.0) as usize;
        SampleClip::new(
            SR,
            (0..n)
                .map(|i| {
                    let t = i as f32 / SR as f32;
                    if t < 2.0 {
                        loud * (TAU * freqs[0] * t).sin()
                    } else {
                        freqs.iter().map(|f| quiet * (TAU * f * t).sin()).sum()
                    }
                })
                .collect(),
        )
    }

    /// **NO frame's mix may exceed full scale, at the shipped `--leveling` too.**
    ///
    /// [`the_loudest_frames_incoherent_mix_reaches_full_scale`] does run at the
    /// real default, but it feeds two steady tones -- every frame has the same
    /// spread, so leveling is a no-op on it -- and it asserts only that the
    /// MAXIMUM mix is 1.0, never that no frame passes it. On material where the
    /// spread changes, the shipped default measured **2.413x** full scale here:
    /// the leveling boost divided by the frame's loudest single VOICE while the
    /// global scale targeted the frame MIX, so a frame with six voices sounding
    /// took the boost a one-voice frame had earned. See
    /// `track::tests::no_frame_exceeds_full_scale_at_the_shipped_leveling_default`.
    #[test]
    fn no_frame_exceeds_full_scale_at_the_shipped_leveling_default() {
        let clip = one_partial_then_six(0.9, 0.09);
        let d = AudioOptions::default();
        assert_eq!(d.leveling, 1.0, "this test exists to cover the SHIPPED default");
        let measured: Vec<(f32, f64, usize)> = [0.0, 0.25, 0.5, 0.75, d.leveling]
            .into_iter()
            .map(|leveling| {
                let o = AudioOptions { leveling, ..opts(12, 4096) };
                let t = analyze_voices(&clip, &o, &mut NoProgress).expect("analyze");
                let (mix, frame) = (0..t.frame_count)
                    .map(|f| (t.volumes.iter().map(|v| v[f] * v[f]).sum::<f64>().sqrt(), f))
                    .fold((0.0f64, 0usize), |a, b| if b.0 > a.0 { b } else { a });
                println!("voice --leveling {leveling}: worst mix {mix:.3}x at frame {frame}");
                (leveling, mix, frame)
            })
            .collect();
        for (leveling, mix, frame) in measured {
            assert!(
                mix <= 1.0 + 1e-6,
                "--leveling {leveling} mixes frame {frame} to {mix}x full scale -- the \
                 emitters sum incoherently in the game's mixer and anything above 1.0 clips"
            );
        }
    }

    /// **`--gain` is the level this mode's normalisation lands on too**, and
    /// the suite could not tell either: every other test here runs at the
    /// default 1.0 or measures a ratio. The two modes must agree on what the
    /// flag means, or an A/B between them at the same `--gain` compares two
    /// different things.
    #[test]
    fn gain_is_the_level_the_loudest_mix_lands_on() {
        let clip = tones(&[(440.0, 0.5), (660.0, 0.4)], 2.0);
        for leveling in [0.0, AudioOptions::default().leveling] {
            for gain in [0.125, 0.25, 0.5, 1.0] {
                let o = AudioOptions { gain, leveling, ..opts(6, 4096) };
                let t = analyze(&clip, &o);
                let mix = (0..t.frame_count)
                    .map(|f| t.volumes.iter().map(|v| v[f] * v[f]).sum::<f64>().sqrt())
                    .fold(0.0f64, f64::max);
                assert!(
                    (mix - gain as f64).abs() < 0.02,
                    "--gain {gain} at --leveling {leveling} must put the loudest incoherent \
                     mix on exactly {gain}, got {mix}"
                );
            }
        }
        // ...and it scales the WHOLE render, not just its loudest instant. A
        // scale that pinned only the peak -- or a limiter that happened to
        // carry `gain` while the global scale had stopped doing so -- would
        // leave every quieter frame at the wrong level and still pass above.
        let full = analyze(&clip, &opts(6, 4096));
        let half = analyze(&clip, &AudioOptions { gain: 0.5, ..opts(6, 4096) });
        for (v, row) in full.volumes.iter().enumerate() {
            for (f, &loud) in row.iter().enumerate() {
                let quiet = half.volumes[v][f];
                assert!(
                    (quiet - loud * 0.5).abs() < 1e-9,
                    "voice {v} frame {f}: --gain 0.5 must be exactly half of --gain 1.0 \
                     ({quiet} vs {loud})"
                );
            }
        }
        let t = analyze(&clip, &AudioOptions { gain: 0.0, ..opts(6, 4096) });
        assert!(
            t.volumes.iter().flat_map(|v| v.iter()).all(|&v| v == 0.0),
            "--gain 0 must write silence"
        );
    }

    /// The other half of the pair: the bound must not have been bought by
    /// making full leveling quiet. Every frame with content in it must still
    /// arrive AT full scale at `--leveling 1`.
    #[test]
    fn full_leveling_still_takes_every_frame_all_the_way_to_full_scale() {
        let clip = one_partial_then_six(0.9, 0.09);
        let t = analyze_voices(&clip, &opts(12, 4096), &mut NoProgress).expect("analyze");
        let worst = (t.frame_count * 2 / 3..t.frame_count)
            .map(|f| t.volumes.iter().map(|v| v[f] * v[f]).sum::<f64>().sqrt())
            .fold(1.0f64, f64::min);
        assert!(
            worst > 0.85,
            "the quiet half must still be dragged to full scale by --leveling 1 (worst \
             frame mix {worst}) -- bounding the mix by attenuating is the loudness \
             regression the bound was supposed to avoid"
        );
    }

    #[test]
    fn silence_produces_exactly_zero_everywhere() {
        let clip = SampleClip::new(SR, vec![0.0; SR as usize * 2]);
        let t = analyze(&clip, &opts(4, 4096));
        for row in &t.volumes {
            for &v in row {
                assert_eq!(v, 0.0);
            }
        }
        for row in &t.pitches {
            for &p in row {
                assert_eq!(p, IDLE_PITCH, "an unused voice holds the idle pitch");
            }
        }
    }

    /// A tonal source's peaks must land ON the scale. This is the measurement
    /// the whole mode is justified by: the band bank puts a note up to 50 cents
    /// (geometric grid: 42) from the nearest speaker, and a peak tracker has no
    /// grid at all.
    #[test]
    fn peaks_of_an_in_tune_source_sit_within_a_few_cents_of_equal_temperament() {
        // A440, C#5, E5 -- an A major triad in exact equal temperament.
        let t = analyze(
            &tones(&[(440.0, 0.5), (554.365, 0.4), (659.255, 0.4)], 2.0),
            &opts(6, 8192),
        );
        assert!(t.stats.peak_count > 0);
        assert!(
            t.stats.mean_abs_cents < 5.0,
            "an exactly-tuned triad must be located within a few cents of the \
             scale, mean |offset| was {:.2} cents",
            t.stats.mean_abs_cents
        );
    }

    #[test]
    fn the_cents_histogram_counts_every_assigned_peak() {
        let t = analyze(&tones(&[(440.0, 0.5), (660.0, 0.4)], 1.5), &opts(4, 4096));
        assert_eq!(
            t.stats.cents_hist.iter().sum::<u32>() as usize,
            t.stats.peak_count
        );
        assert_eq!(
            t.stats.note_hist.iter().sum::<u32>() as usize,
            t.stats.peak_count
        );
    }

    /// The voice count is the flag, and the REAL voices per frame is not --
    /// that is the number the owner needs reported, and a build that quietly
    /// sounded the same number whatever the flag said has shipped before.
    #[test]
    fn more_voices_sound_more_of_a_dense_source() {
        let parts: Vec<(f32, f32)> = (1..=12).map(|k| (110.0 * k as f32, 0.3)).collect();
        let clip = tones(&parts, 2.0);
        let few = analyze(&clip, &opts(3, 8192));
        let many = analyze(&clip, &opts(10, 8192));
        let a = few.stats.mean_voices_per_frame(few.frame_count);
        let b = many.stats.mean_voices_per_frame(many.frame_count);
        assert!(a > 2.0, "3 voices on a 12-partial source must nearly fill, got {a:.2}");
        assert!(
            b > a + 2.0,
            "10 voices must sound audibly more than 3 on a 12-partial source: \
             {a:.2} vs {b:.2} voices/frame"
        );
    }

    // -- pitch smoothing ---------------------------------------------------

    /// **The fix for "sounds weird because the pitch wobbles a bit".**
    ///
    /// A sustained tone's detected peak moves a few cents every frame -- bin
    /// quantisation, interpolation noise, partials beating -- and written
    /// straight through, that is an audible warble on a note that should be
    /// steady. The written pitch must move materially less than the tracked
    /// one, measured over the SAME frames of the SAME run.
    #[test]
    fn a_continuing_voices_pitch_is_smoothed_against_frame_to_frame_wobble() {
        let t = analyze(&vibrato(&[220.0, 330.0, 550.0], 10.0, 3.0), &opts(6, 4096));
        let raw = t.stats.raw_jitter_rms_cents();
        let out = t.stats.jitter_rms_cents();
        assert!(
            raw > 1.0,
            "the fixture must actually wobble as tracked, got {raw:.2} cents rms -- \
             otherwise there is nothing for the smoothing to remove"
        );
        assert!(
            out < raw * 0.6,
            "the written pitch must wobble materially less than the tracked one: \
             {raw:.2} cents rms tracked, {out:.2} written"
        );
    }

    /// ...but a voice being BORN must jump straight to its note. Smoothing a
    /// rebirth is a portamento swoop between unrelated pitches, which is worse
    /// than the wobble it would be fixing.
    ///
    /// One tone, then silence long enough for every voice to release, then a
    /// tone a fifth away. Whichever voice takes the second note must be ON it
    /// from its first sounding frame, not gliding up from the first note.
    #[test]
    fn a_newly_born_voice_takes_its_pitch_outright_rather_than_gliding() {
        let secs = 3.0f32;
        let n = (SR as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                if t < 1.0 {
                    0.5 * (TAU * 440.0 * t).sin()
                } else if t > 2.0 {
                    0.5 * (TAU * 660.0 * t).sin()
                } else {
                    0.0
                }
            })
            .collect();
        let t = analyze(&SampleClip::new(SR, samples), &opts(4, 4096));
        // From the middle of the silent gap onward, so the frame a voice is
        // BORN on for the second tone is inside the scan.
        let from = (1.5 * t.fps as f64) as usize;
        let mut checked = 0usize;
        for v in 0..t.voice_count() {
            for f in from..t.frame_count {
                if t.volumes[v][f] <= 0.0 || t.volumes[v][f - 1] > 0.0 {
                    continue;
                }
                // f is this voice's FIRST sounding frame of the second note.
                let hz = t.pitches[v][f] * BASE_HZ as f64;
                assert!(
                    semitones_between(hz, 660.0) < 0.6,
                    "voice {v} starts its new note at {hz:.1} Hz instead of 660 -- a birth \
                     must jump, not glide up from wherever the speaker was left"
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "the fixture must actually start a voice in the second tone");
    }

    /// Snapping is OFF by default: the argument for this mode is that a
    /// tracked voice needs no grid, and a default that put one back would give
    /// that away silently.
    #[test]
    fn pitch_snapping_is_off_by_default_and_pulls_onto_the_scale_when_asked() {
        assert_eq!(AudioOptions::default().pitch_snap_cents, 0.0);
        // A tone 12 cents sharp of A440: inside a generous snap window, and
        // far outside a tight one.
        let sharp = 440.0 * 2f32.powf(12.0 / 1200.0);
        let plain = analyze(&tones(&[(sharp, 0.5)], 2.0), &opts(4, 8192));
        let snapped = analyze(
            &tones(&[(sharp, 0.5)], 2.0),
            &AudioOptions {
                pitch_snap_cents: 25.0,
                ..opts(4, 8192)
            },
        );
        let last = |t: &VoiceStreams| {
            (0..t.voice_count())
                .filter(|&v| t.volumes[v][t.frame_count - 1] > 0.0)
                .map(|v| t.pitches[v][t.frame_count - 1])
                .next()
                .expect("a sounding voice")
        };
        let plain_cents = cents_from_equal_temperament(last(&plain) * BASE_HZ as f64).0;
        let snapped_cents = cents_from_equal_temperament(last(&snapped) * BASE_HZ as f64).0;
        assert!(
            plain_cents.abs() > 5.0,
            "unsnapped, the render must keep the source's own 12-cent offset, got \
             {plain_cents:.1}"
        );
        assert!(
            snapped_cents.abs() < 0.001,
            "snapped, the voice must sit exactly on the semitone, got {snapped_cents:.3}"
        );
    }

    // -- the level envelope ------------------------------------------------

    /// A released voice must reach EXACTLY zero and hold its speaker until it
    /// does. A one-pole release never reaches zero, so a voice would carry a
    /// shrinking tail forever and could never be reused without a
    /// discontinuity.
    #[test]
    fn a_released_voice_reaches_exactly_zero() {
        let secs = 2.0f32;
        let n = (SR as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                if t < secs / 2.0 { 0.5 * (TAU * 440.0 * t).sin() } else { 0.0 }
            })
            .collect();
        let t = analyze(&SampleClip::new(SR, samples), &opts(4, 4096));
        for v in 0..t.voice_count() {
            assert_eq!(
                t.volumes[v][t.frame_count - 1],
                0.0,
                "voice {v} is still sounding a second after the source went silent"
            );
        }
    }

    /// **The one-frame gap, and why it is not cosmetic.**
    ///
    /// A voice must write its final zero BEFORE it can be reborn. Without the
    /// gap a voice is reborn on the very frame its release ramp reaches zero,
    /// so the frame before the new note still carries the old note's last
    /// non-zero step -- a pitch jump on a speaker that is sounding in both
    /// frames, measured at 3.74% of continuations on speech.
    ///
    /// The invariant is checkable straight off the output: no voice may change
    /// pitch by more than [`MATCH_TOLERANCE_SEMITONES`] while sounding in two
    /// consecutive frames, ever.
    #[test]
    fn a_voice_never_changes_note_without_passing_through_silence() {
        // Six tones coming and going on different schedules against four
        // voices: every voice is busy, so births have to wait for a release.
        let secs = 3.0f32;
        let n = (SR as f32 * secs) as usize;
        let freqs = [220.0f32, 277.18, 329.63, 440.0, 554.37, 659.26];
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                freqs
                    .iter()
                    .enumerate()
                    .map(|(k, f)| {
                        // Each tone on its own duty cycle, so the set of
                        // sounding notes turns over constantly.
                        let gate = ((t * (0.7 + k as f32 * 0.31)).fract() < 0.5) as i32 as f32;
                        0.25 * gate * (TAU * f * t).sin()
                    })
                    .sum()
            })
            .collect();
        let t = analyze(&SampleClip::new(SR, samples), &opts(4, 4096));
        assert!(
            t.stats.continuations > 100,
            "the fixture must produce plenty of sounding frame pairs, got {}",
            t.stats.continuations
        );
        assert_eq!(
            t.stats.pitch_jumps, 0,
            "{} of {} sounding frame pairs changed note by more than a semitone. A voice \
             must pass through zero before it takes a new note; anything else is a \
             speaker being yanked to an unrelated pitch mid-sound.",
            t.stats.pitch_jumps, t.stats.continuations
        );
    }

    /// A voice's peak level must not be pulled down by the attack: transients
    /// are what make a struck note sound struck.
    #[test]
    fn the_attack_is_fast_enough_to_keep_a_transient() {
        let shaping = VoiceShaping::new(&AudioOptions::default());
        let after_one = shaping.level.step(0.0, 1.0);
        assert!(
            after_one > 0.9,
            "one frame of attack must get most of the way to the target, got {after_one}"
        );
    }

    /// The release must lock a voice's speaker for several frames and no more
    /// -- it is a fade, but it is also an unavailable speaker.
    ///
    /// **And it comes from `--voice-release`, not from `--release`.** The two
    /// were the same flag and that was the bug: `--release` is a one-pole time
    /// constant on a voice that is still following a peak, 150 ms by default,
    /// and taking the END OF A NOTE from it gave every voice a 133 ms tail --
    /// measured at 52.7% of all sounding voice-frames on speech, where a
    /// phoneme is 50-100 ms long.
    #[test]
    fn the_release_length_follows_its_own_flag_and_is_capped() {
        let at = |ms: f32, fps: f32| {
            VoiceShaping::new(&AudioOptions {
                voice_release_ms: ms,
                fps,
                ..Default::default()
            })
            .release_frames
        };
        assert_eq!(at(0.0, 30.0), 1, "no release still has to reach zero");
        assert_eq!(at(150.0, 30.0), 5);
        assert_eq!(at(150.0, 60.0), 9, "the same time is more frames at a higher rate");
        assert_eq!(
            at(10_000.0, 30.0),
            MAX_VOICE_RELEASE_FRAMES,
            "a release is also a LOCK on one of very few speakers, so it is capped"
        );

        // `--release` must NOT reach it. Changing the level follower's time
        // constant is a change to how a SOUNDING voice tracks its partial, and
        // silently re-timing every note ending with it is what shipped.
        let by_release = VoiceShaping::new(&AudioOptions {
            release_ms: 2_000.0,
            fps: 30.0,
            ..Default::default()
        })
        .release_frames;
        assert_eq!(
            by_release,
            at(DEFAULT_VOICE_RELEASE_MS, 30.0),
            "--release must not change how long a note takes to reach zero"
        );
    }

    /// **The mutation gate on applying the release ramp AFTER normalisation,
    /// and the direct regression test for "it bleeds together".**
    ///
    /// **The mutation gate on applying the release ramp AFTER normalisation,
    /// and the direct regression test for "it bleeds together".**
    ///
    /// A LOUD tone that stops dead over a quiet one that does not, rendered at
    /// `--leveling 1` -- the default, and full automatic gain control.
    ///
    /// The arithmetic is what makes this decisive. Once the loud tone has gone,
    /// the only thing in the frame near its pitch is its own release ramp. If
    /// the per-frame divisor is the level THROUGH the ramp, then in tail frame
    /// `k` it is `L * g_k`, the scale is `peak / (L * g_k)`, and what gets
    /// written is `L * g_k * peak / (L * g_k)` = **exactly what the note was
    /// written at while it was still playing, on every frame of the ramp**.
    /// The fade cancels itself out algebraically and the note is held at full
    /// level for the whole release. That is the bug, and it is why a "release"
    /// that provably reached zero still sounded like a drone: measured on
    /// speech at 32 voices, a released voice was written at 0.050 mean volume
    /// against 0.071 for a voice actually following a peak.
    ///
    /// Dividing by the FROZEN level instead leaves `g_k` standing, and the
    /// written tail is the ramp.
    #[test]
    fn a_release_survives_full_leveling_and_still_reaches_zero() {
        let secs = 3.0f32;
        let n = (SR as f32 * secs) as usize;
        // FOUR equally loud tones that all stop at one second, against a very
        // quiet one that does not. Four, not one, deliberately: the written
        // volume is clamped at 1.0, and a lone tone is normalised so close to
        // full scale that the clamp alone would flatten the difference this
        // test is looking for. With four voices in the loudest frame the whole
        // ramp sits near half scale, well clear of the clamp.
        let notes = [233.08f32, 349.23, 523.25, 783.99];
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                let chord: f32 = if t < 1.0 {
                    notes.iter().map(|f| 0.4 * (TAU * f * t).sin()).sum()
                } else {
                    0.0
                };
                chord + 0.004 * (TAU * 1660.0 * t).sin()
            })
            .collect();
        let clip = SampleClip::new(SR, samples);
        let o = AudioOptions {
            leveling: 1.0,
            voice_release_ms: 150.0,
            ..opts(5, 4096)
        };
        let t = analyze(&clip, &o);
        let release_frames = VoiceShaping::new(&o).release_frames;
        assert!(release_frames >= 4, "the fixture needs a ramp with steps in it");

        // The voice that carried the lowest chord tone, and its last sounding
        // frame.
        let (v, end) = (0..t.voice_count())
            .filter_map(|v| {
                let last = (0..t.frame_count)
                    .filter(|&f| {
                        t.volumes[v][f] > 0.0
                            && semitones_between(t.pitches[v][f] * BASE_HZ as f64, notes[0] as f64)
                                < 0.6
                    })
                    .next_back()?;
                Some((v, last))
            })
            .next()
            .expect("some voice must carry the lowest chord tone");

        // The written ramp: the last `release_frames - 1` non-zero frames.
        let ramp: Vec<f64> = ((end + 2 - release_frames)..=end)
            .map(|f| t.volumes[v][f])
            .collect();
        assert_eq!(t.volumes[v][end + 1], 0.0, "the ramp must end at exactly zero");
        for w in ramp.windows(2) {
            assert!(
                w[1] < w[0],
                "the written release must FALL on every frame: {ramp:?}. Equal steps mean \
                 the per-frame AGC is dividing by a divisor that falls with the ramp, \
                 which cancels the ramp out exactly and holds the note at full level."
            );
        }
        // ...and it must fall by roughly the ramp's own shape, not merely
        // decrease: the last step is 1/release_frames of the first.
        let want = 1.0 / (release_frames - 1) as f64;
        let got = ramp[ramp.len() - 1] / ramp[0];
        assert!(
            (got - want).abs() < 0.15,
            "the last frame of the ramp must be about {want:.2} of the first, got \
             {got:.2} from {ramp:?}"
        );
    }

    /// **The measurement the owner asked for, as an assertion: time from a
    /// partial's true end to the voice reading zero.**
    ///
    /// Stated in the units of the complaint. A phoneme is 50-100 ms, so a
    /// release that takes longer than the note smears across the next one.
    #[test]
    fn a_finished_partial_reaches_zero_within_the_release_time() {
        let secs = 3.0f32;
        let n = (SR as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                if t < 1.0 { 0.5 * (TAU * 440.0 * t).sin() } else { 0.0 }
            })
            .collect();
        let t = analyze(&SampleClip::new(SR, samples), &opts(4, 4096));
        let (mean, p95) = t.stats.tail_ms(t.fps);
        let budget = DEFAULT_VOICE_RELEASE_MS as f64;
        assert!(
            p95 <= budget,
            "a voice whose partial has ended must reach zero within --voice-release \
             ({budget} ms); mean {mean:.0} ms, p95 {p95:.0} ms"
        );
        // And the default must be short enough for speech: a spoken phoneme is
        // 50-100 ms, so a note ending must not outlast one.
        assert!(
            DEFAULT_VOICE_RELEASE_MS <= 50.0,
            "the default release must fit inside a phoneme, got {DEFAULT_VOICE_RELEASE_MS} ms"
        );
    }

    /// The bleed statistics must count what they say they count. A silent
    /// stretch after a note is exactly the tail, and nothing sounds unmatched
    /// once the ramp is over.
    #[test]
    fn the_bleed_statistics_count_unmatched_sounding_frames_and_tails() {
        let secs = 3.0f32;
        let n = (SR as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                if t < 1.0 { 0.5 * (TAU * 440.0 * t).sin() } else { 0.0 }
            })
            .collect();
        let t = analyze(&SampleClip::new(SR, samples), &opts(4, 4096));
        let st = &t.stats;
        assert!(st.sounding > 0);
        assert!(
            st.unmatched_sounding <= st.sounding,
            "an unmatched frame is a sounding frame"
        );
        assert_eq!(
            st.tail_frames.len(),
            st.lifetimes.len(),
            "every completed run has exactly one tail"
        );
        // A clean tone into digital silence has no bleed to speak of: at the
        // default release the ramp reaches zero on the frame after the last
        // match, so almost nothing is written unmatched.
        assert!(
            st.unmatched_fraction() < 0.15,
            "a tone into silence must not leave voices sounding on nothing: \
             {:.1}% of sounding voice-frames were unmatched",
            st.unmatched_fraction() * 100.0
        );
        let (matched_v, unmatched_v) = st.mean_volumes();
        assert!(
            unmatched_v < matched_v,
            "a released voice must be written quieter than a sounding one: \
             {unmatched_v:.3} vs {matched_v:.3}"
        );
    }

    /// A longer `--voice-release` must actually hold the note longer, and a
    /// shorter one must let go sooner. The flag is the whole point.
    #[test]
    fn a_longer_voice_release_holds_a_finished_note_for_longer() {
        let secs = 3.0f32;
        let n = (SR as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                if t < 1.0 { 0.5 * (TAU * 440.0 * t).sin() } else { 0.0 }
            })
            .collect();
        let clip = SampleClip::new(SR, samples);
        let tail = |ms: f32| {
            let t = analyze(
                &clip,
                &AudioOptions { voice_release_ms: ms, ..opts(4, 4096) },
            );
            t.stats.tail_ms(t.fps).1
        };
        let short = tail(0.0);
        let long = tail(300.0);
        assert!(
            long > short + 100.0,
            "--voice-release 300 must hold a finished note materially longer than \
             --voice-release 0: {long:.0} ms vs {short:.0} ms"
        );
        assert_eq!(
            short, 0.0,
            "--voice-release 0 clamps to one frame, which writes zero on the frame after \
             the partial ends -- no tail at all"
        );
    }

    // -- the voice budget --------------------------------------------------

    /// `harmonic_split` must collapse a harmonic series onto its own root
    /// rather than counting each partial as a note.
    #[test]
    fn a_harmonic_series_counts_as_one_fundamental() {
        let series: Vec<f64> = (1..=6).map(|k| 110.0 * k as f64).collect();
        let (harmonics, roots) = harmonic_split(&series);
        assert_eq!(roots, 1, "six harmonics of 110 Hz are one note");
        assert_eq!(harmonics, 5);

        // Three unrelated notes are three notes. A major triad's ratios
        // (1 : 1.26 : 1.5) are nowhere near an integer.
        let (h, r) = harmonic_split(&[440.0, 554.365, 659.255]);
        assert_eq!((h, r), (0, 3));

        assert_eq!(harmonic_split(&[]), (0, 0));
    }

    // -- guards ------------------------------------------------------------

    #[test]
    fn zero_voices_is_an_error_rather_than_a_speakerless_build() {
        let clip = tones(&[(440.0, 0.5)], 1.0);
        assert!(analyze_voices(&clip, &opts(0, 4096), &mut NoProgress).is_err());
    }

    #[test]
    fn a_source_with_no_frames_is_an_error() {
        let clip = SampleClip::new(SR, vec![0.0; 100]);
        assert!(analyze_voices(&clip, &opts(4, 4096), &mut NoProgress).is_err());
    }

    #[test]
    fn max_frames_bounds_the_track() {
        let o = AudioOptions {
            max_frames: 7,
            ..opts(4, 4096)
        };
        let t = analyze(&tones(&[(440.0, 0.5)], 3.0), &o);
        assert_eq!(t.frame_count, 7);
    }

    /// NaN slips past a plain `<= 0.0` guard, because every comparison against
    /// NaN is false.
    #[test]
    fn non_finite_options_are_rejected() {
        let clip = tones(&[(440.0, 0.5)], 1.0);
        for bad in [
            AudioOptions { fps: f32::NAN, ..opts(4, 4096) },
            AudioOptions { fps: 0.0, ..opts(4, 4096) },
            AudioOptions { gain: f32::NAN, ..opts(4, 4096) },
            AudioOptions { gain: -1.0, ..opts(4, 4096) },
            AudioOptions { leveling: f32::NAN, ..opts(4, 4096) },
            AudioOptions { leveling: 2.0, ..opts(4, 4096) },
            AudioOptions { max_frames: 0, ..opts(4, 4096) },
        ] {
            assert!(analyze_voices(&clip, &bad, &mut NoProgress).is_err());
        }
    }

    #[test]
    fn cents_from_equal_temperament_is_exact_on_the_scale() {
        for (hz, step) in [(440.0, 0), (880.0, 12), (220.0, -12), (523.2511, 3)] {
            let (cents, s) = cents_from_equal_temperament(hz);
            assert_eq!(s, step, "{hz} Hz is step {step} from A440");
            assert!(cents.abs() < 0.5, "{hz} Hz is on the scale, got {cents} cents");
        }
        // Exactly a quarter-tone up is +50 cents and rounds to the step below.
        let (cents, _) = cents_from_equal_temperament(440.0 * 2f64.powf(0.5 / 12.0));
        assert!((cents.abs() - 50.0).abs() < 0.01, "got {cents}");
    }
}
