//! The fixed band bank: which speaker sits at which pitch, and how a
//! magnitude spectrum folds onto them.
//!
//! Pitches are set once at build time and never written again, so this
//! design does not depend on how the game handles a live pitch change.
//!
//! Every tonal band is an exact equal-tempered interval from A440:
//! `PitchMultiplier = 2^(step / subdiv)`, `step` an integer, so step 0 is
//! exactly 1.0 -- A440 itself, because [`BASE_HZ`] is what the synth assets
//! are authored at. At the default [`DEFAULT_SUBDIV`] of 12 that is one band
//! per semitone, and an in-tune equal-tempered recording has its notes land
//! on band centres with zero quantisation error.
//!
//! `2^(-40/12)` is 0.0992 and `2^(40/12)` is 10.079, both outside the
//! emitter's legal `PitchMultiplier`, so the usable span at 12 steps per
//! octave is exactly -39..=39 semitones -- 79 bands, 46.25 Hz (F#1) to
//! 4186.0 Hz (C8), the top 79 keys of an 88-key piano. See [`max_step`].

/// The synth assets' authored frequency. Every pitch is a multiplier on it,
/// and it is also the tuning anchor: a band at `PitchMultiplier` 1.0 plays
/// exactly this, so the grid is anchored on A440 by construction.
pub const BASE_HZ: f32 = 440.0;
/// `Component_AudioEmitter::PitchMultiplier` hardware limits.
pub const PITCH_MIN: f32 = 0.1;
pub const PITCH_MAX: f32 = 10.0;
/// At most white + pink.
pub const MAX_NOISE_BANDS: usize = 2;

/// Bands per octave in the tonal grid.
///
/// 12 -- one band per semitone -- is the default, matching the scale sources
/// are actually played in: an in-tune equal-tempered recording then lands
/// exactly on a band centre with zero tuning error.
///
/// 24 (quarter-tones, via `--subdiv`) halves the worst-case error for
/// mistuned sources (50 to 25 cents) at the cost of doubling the speaker
/// count; it changes nothing for in-tune material, since the semitone bands
/// are a subset of the quarter-tone ones.
pub const DEFAULT_SUBDIV: u32 = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandKind {
    Tonal,
    WhiteNoise,
    PinkNoise,
}

/// The `PitchMultiplier` of grid step `step` at `subdiv` steps per octave.
/// The only place a band pitch is computed: `2^(step/subdiv)` in f64, cast to
/// f32 once -- never an accumulated product, which would drift.
///
/// Step 0 returns exactly 1.0, so one speaker always sits precisely on A440.
pub fn step_pitch(step: i32, subdiv: u32) -> f32 {
    2.0f64.powf(step as f64 / subdiv.max(1) as f64) as f32
}

/// The largest `|step|` whose pitch is still a legal `PitchMultiplier`, at
/// `subdiv` steps per octave.
///
/// At `subdiv` 12 this is 39: `2^(39/12)` is 9.5137 and `2^(-39/12)` is
/// 0.10511, both legal, while 40 steps gives 10.079 and 0.09921, which the
/// game would clamp silently.
///
/// The floor of the logarithm is only the first guess. The invariant that
/// matters is the pitch, not the exponent, so the guess is walked back until
/// both ends really are inside the range -- that keeps this correct if the
/// hardware limits ever move, or become asymmetric, and costs one f64 compare.
pub fn max_step(subdiv: u32) -> Result<i32, String> {
    if subdiv == 0 {
        return Err("--subdiv must be at least 1 band per octave".to_string());
    }
    let up = subdiv as f64 * (PITCH_MAX as f64).log2();
    let down = subdiv as f64 * -(PITCH_MIN as f64).log2();
    let mut s = up.min(down).floor().max(0.0) as i32;
    while s > 0 && (step_pitch(s, subdiv) > PITCH_MAX || step_pitch(-s, subdiv) < PITCH_MIN) {
        s -= 1;
    }
    Ok(s)
}

/// How many tonal bands the emitter's legal pitch range holds at `subdiv`
/// steps per octave: `2 * max_step + 1`, the steps `-max_step ..= max_step`.
/// 79 at the default 12 (one per semitone), 159 at 24 (quarter-tones).
pub fn max_tonal_bands(subdiv: u32) -> Result<usize, String> {
    Ok(2 * max_step(subdiv)? as usize + 1)
}

#[derive(Debug)]
pub struct BandPlan {
    /// `PitchMultiplier` per speaker. Noise bands carry 1.0 -- a noise asset
    /// has no pitch to speak of, and 1.0 is the identity rather than a
    /// meaningful frequency.
    pub pitches: Vec<f32>,
    pub kinds: Vec<BandKind>,
    /// Steps per octave. See [`DEFAULT_SUBDIV`].
    subdiv: u32,
    /// Grid step of tonal band 0, in `1/subdiv` octaves relative to A440.
    lowest_step: i32,
}

impl BandPlan {
    /// `bands` total speakers, `noise_bands` of them noise, tonal bands on
    /// the equal-tempered semitone grid centred on A440.
    pub fn new(bands: usize, noise_bands: usize) -> Result<Self, String> {
        Self::with_subdiv(bands, noise_bands, DEFAULT_SUBDIV)
    }

    /// Every tonal band the legal pitch range holds at `subdiv`, plus the
    /// noise bands. What a render gets when `--bands` is not passed: the
    /// count follows from the hardware range and subdivision, since the
    /// spacing is musically fixed. 79 tonal at `subdiv` 12, 159 at 24.
    pub fn full(noise_bands: usize, subdiv: u32) -> Result<Self, String> {
        let tonal = max_tonal_bands(subdiv)?;
        Self::with_subdiv(tonal + noise_bands, noise_bands, subdiv)
    }

    /// The general constructor. `bands - noise_bands` tonal steps, centred on
    /// A440 so step 0 (`PitchMultiplier` exactly 1.0) is always one of them.
    ///
    /// The count selects the span; it cannot change the spacing, which is
    /// fixed by `subdiv`. Asking for more bands than the pitch range holds is
    /// an error naming the maximum, never a silent clamp: a clamp would hand
    /// back a different number of speakers than the flag asked for.
    pub fn with_subdiv(bands: usize, noise_bands: usize, subdiv: u32) -> Result<Self, String> {
        if noise_bands > MAX_NOISE_BANDS {
            return Err(format!(
                "at most {MAX_NOISE_BANDS} noise bands are available (white, pink), got {noise_bands}"
            ));
        }
        let tonal = bands
            .checked_sub(noise_bands)
            .ok_or_else(|| format!("{noise_bands} noise bands exceed {bands} total bands"))?;
        if tonal < 2 {
            return Err(format!(
                "need at least 2 tonal bands to span a range, got {tonal} \
                 ({bands} total minus {noise_bands} noise)"
            ));
        }
        let max_step = max_step(subdiv)?;
        let max_tonal = 2 * max_step as usize + 1;
        if tonal > max_tonal {
            return Err(format!(
                "{tonal} tonal bands do not fit the equal-tempered grid: at {subdiv} band(s) \
                 per octave the emitter's {PITCH_MIN}x..{PITCH_MAX}x pitch range holds at most \
                 {max_tonal} tonal bands ({} total with {noise_bands} noise). Lower --bands, \
                 or raise --subdiv (24 = quarter-tones, {} tonal bands)",
                max_tonal + noise_bands,
                max_tonal_bands(24)?,
            ));
        }

        // Centred on A440: an even count takes the extra band downward, where
        // the bass lives, and an odd count is symmetric. Either way step 0 is
        // in range, so exactly one speaker plays A440 itself.
        //
        // The two clamps only bind at the widest counts, where `tonal` fills
        // the legal window and there is nowhere left to centre.
        let lowest = (-((tonal as i32) / 2))
            .max(-max_step)
            .min(max_step - tonal as i32 + 1);

        let mut pitches: Vec<f32> =
            (0..tonal).map(|k| step_pitch(lowest + k as i32, subdiv)).collect();
        let mut kinds = vec![BandKind::Tonal; tonal];
        if noise_bands >= 1 {
            pitches.push(1.0);
            kinds.push(BandKind::WhiteNoise);
        }
        if noise_bands >= 2 {
            pitches.push(1.0);
            kinds.push(BandKind::PinkNoise);
        }
        Ok(Self { pitches, kinds, subdiv, lowest_step: lowest })
    }

    pub fn len(&self) -> usize {
        self.pitches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pitches.is_empty()
    }

    /// Steps per octave. See [`DEFAULT_SUBDIV`].
    pub fn subdiv(&self) -> u32 {
        self.subdiv
    }

    /// The grid step of tonal band `band`, in `1/subdiv` octaves from A440.
    /// `None` for a noise band, which has no place on the scale.
    pub fn step_of(&self, band: usize) -> Option<i32> {
        match self.kinds.get(band) {
            Some(BandKind::Tonal) => Some(self.lowest_step + band as i32),
            _ => None,
        }
    }

    pub fn tonal_count(&self) -> usize {
        self.kinds.iter().filter(|k| **k == BandKind::Tonal).count()
    }

    /// Half a grid step as a frequency ratio: `2^(1/(2*subdiv))`.
    ///
    /// At the default 12 steps per octave this is 50 cents, so a band's edges
    /// are ±50 cents around its centre and it captures exactly one semitone --
    /// the semitone it is tuned to and no part of its neighbours'.
    fn half_step(&self) -> f64 {
        2.0f64.powf(0.5 / self.subdiv as f64)
    }

    /// The edges of tonal band `k`, in Hz: half a grid step either side of
    /// its centre. Half-open -- `[lower, upper)` -- matching [`BandPlan::fold`].
    pub fn edges_hz(&self, k: usize) -> (f32, f32) {
        let half = self.half_step();
        let centre = self.pitches[k] as f64 * BASE_HZ as f64;
        ((centre / half) as f32, (centre * half) as f32)
    }

    /// Where `hz` falls on the tonal grid, as a *continuous* band index:
    /// `subdiv * log2(hz / 440) - lowest_step`. Band `k` owns `[k-0.5, k+0.5)`.
    ///
    /// `fold` uses this and nothing else, so the floor test, the ceiling test
    /// and the bin->band choice are all one comparison against one
    /// expression and cannot drift apart.
    ///
    /// `hz == 0.0` (the DC bin) gives `-inf`, which compares below band 0 and
    /// folds to pink like any other sub-bank content. It never produces a NaN.
    fn position(&self, hz: f64) -> f64 {
        self.subdiv as f64 * (hz / BASE_HZ as f64).log2() - self.lowest_step as f64
    }

    /// Fold a magnitude spectrum onto the bank: each band gets
    /// `sqrt(sum of |X[k]|^2)` over the bins it owns -- the band's energy
    /// (Parseval) expressed as the amplitude of one equivalent sine, not a
    /// width-biased sum of magnitudes.
    ///
    /// Energy off either end of the tonal span has to go somewhere rather
    /// than disappear silently: above the top tonal band's upper edge folds
    /// to the white-noise band (sines cannot render cymbals or sibilance;
    /// broadband hiss can), and below the bottom tonal band's lower edge
    /// folds to the pink-noise band (its 1/f spectrum is itself low-heavy,
    /// the right carrier for sub-bass). Either fold is skipped, and that
    /// energy discarded, when the corresponding noise band does not exist in
    /// this plan.
    pub fn fold(&self, spectrum: &[f32], sample_rate: u32, window: usize) -> Vec<f32> {
        if spectrum.is_empty() || window == 0 {
            return vec![0.0f32; self.len()];
        }
        // f64 accumulator: squaring puts ~1e6 into each of ~1600 bins for the
        // white band, and an f32 running total loses the small terms entirely
        // once it passes ~1.7e7.
        let mut acc = vec![0.0f64; self.len()];
        let tonal = self.tonal_count();
        let white = self.kinds.iter().position(|k| *k == BandKind::WhiteNoise);
        let pink = self.kinds.iter().position(|k| *k == BandKind::PinkNoise);
        let hz_per_bin = sample_rate as f64 / window as f64;

        for (bin, &mag) in spectrum.iter().enumerate() {
            if mag == 0.0 {
                continue;
            }
            let power = mag as f64 * mag as f64;
            let hz = bin as f64 * hz_per_bin;
            // `+ 0.5` then `floor` is round-half-up, keeping bands half-open
            // `[lower, upper)` on both sides of A440. `hz == 0` gives `-inf`
            // here, so DC folds to pink without ever producing a NaN.
            let k = (self.position(hz) + 0.5).floor();
            if k < 0.0 {
                if let Some(p) = pink {
                    acc[p] += power;
                }
            } else if k >= tonal as f64 {
                if let Some(w) = white {
                    acc[w] += power;
                }
            } else {
                acc[k as usize] += power;
            }
        }
        acc.into_iter().map(|p| p.sqrt() as f32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;
    const WIN: usize = 4096;

    fn plan() -> BandPlan {
        BandPlan::new(32, 2).expect("the default plan must be valid")
    }

    /// A spectrum that is all zeros except one bin at `freq`.
    fn impulse_at(freq: f32) -> Vec<f32> {
        let bins = WIN / 2 + 1;
        let mut sp = vec![0.0; bins];
        let b = (freq as f64 * WIN as f64 / SR as f64).round() as usize;
        sp[b.min(bins - 1)] = 1.0;
        sp
    }

    /// A spectrum with `1.0` in the bin nearest each of `freqs`.
    fn impulses_at(freqs: &[f32]) -> Vec<f32> {
        let bins = WIN / 2 + 1;
        let mut sp = vec![0.0; bins];
        for &f in freqs {
            let b = (f as f64 * WIN as f64 / SR as f64).round() as usize;
            sp[b.min(bins - 1)] = 1.0;
        }
        sp
    }

    /// Signed cents of `pitch` from the nearest equal-tempered semitone.
    fn cents_off_semitone(pitch: f32) -> f64 {
        let c = 1200.0 * (pitch as f64).log2();
        c - (c / 100.0).round() * 100.0
    }

    /// Every tonal pitch must be an exact power of `2^(1/subdiv)` relative to
    /// 1.0 -- an exact equal-tempered interval from A440, since the synth
    /// assets are authored at [`BASE_HZ`]. Tolerance is 0.02 cents, roughly
    /// 400x tighter than audibility.
    #[test]
    fn every_tonal_pitch_is_an_exact_equal_tempered_interval() {
        for subdiv in [1u32, 3, 12, 19, 24, 31] {
            let max = max_tonal_bands(subdiv).expect("a legal subdivision");
            for tonal in [2usize, 5, 30, 79, max] {
                if tonal > max {
                    continue;
                }
                let p = BandPlan::with_subdiv(tonal, 0, subdiv)
                    .unwrap_or_else(|e| panic!("{tonal} bands at subdiv {subdiv}: {e}"));
                for (k, &pitch) in p.pitches.iter().enumerate() {
                    // The exponent this pitch claims, in grid steps, must be
                    // an integer -- that is what "exact equal-tempered
                    // interval" means.
                    let steps = subdiv as f64 * (pitch as f64).log2();
                    let off_cents = (steps - steps.round()) * 1200.0 / subdiv as f64;
                    assert!(
                        off_cents.abs() < 0.02,
                        "subdiv {subdiv}, {tonal} bands: band {k} pitch {pitch} is \
                         {off_cents:.4} cents off grid step {} -- not an exact interval",
                        steps.round()
                    );
                    // ...and it must be the step the plan claims it is.
                    assert_eq!(
                        p.step_of(k),
                        Some(steps.round() as i32),
                        "subdiv {subdiv}: band {k} pitch {pitch} is not the step it reports"
                    );
                }
            }
        }
    }

    /// At the default 12 steps per octave, "exact interval" means exactly a
    /// semitone, so every band is a real note and an in-tune recording
    /// quantises to zero error.
    #[test]
    fn at_twelve_per_octave_every_band_is_a_real_note() {
        for p in [BandPlan::new(32, 2).unwrap(), BandPlan::full(0, 12).unwrap()] {
            for (k, &pitch) in p.pitches.iter().enumerate() {
                if p.kinds[k] != BandKind::Tonal {
                    continue;
                }
                assert!(
                    cents_off_semitone(pitch).abs() < 0.02,
                    "band {k} pitch {pitch} is {:.3} cents off the nearest semitone",
                    cents_off_semitone(pitch)
                );
            }
        }
    }

    /// The anchor. One speaker must play A440 itself, at `PitchMultiplier`
    /// exactly 1.0 -- not 0.99999. An anchor off by a rounding step puts the
    /// whole grid off by that step.
    #[test]
    fn one_band_sits_exactly_on_a440() {
        for p in [
            BandPlan::new(32, 2).unwrap(),
            BandPlan::new(9, 0).unwrap(),
            BandPlan::full(0, 12).unwrap(),
            BandPlan::full(2, 24).unwrap(),
        ] {
            let tonal = p.tonal_count();
            let at_a440: Vec<usize> =
                (0..tonal).filter(|&i| p.pitches[i] == 1.0).collect();
            assert_eq!(
                at_a440.len(),
                1,
                "exactly one tonal band must sit at PitchMultiplier 1.0 = A440, got {at_a440:?}"
            );
            assert_eq!(p.step_of(at_a440[0]), Some(0));
        }
        assert_eq!(step_pitch(0, 12), 1.0);
        assert_eq!(step_pitch(0, 24), 1.0);
    }

    /// One octave up is a factor of two, exactly. A grid that is only
    /// "nearly" equal-tempered fails this long before it fails anything
    /// audible.
    #[test]
    fn a_full_octave_of_steps_is_exactly_a_factor_of_two() {
        for subdiv in [12u32, 24] {
            let p = BandPlan::full(0, subdiv).unwrap();
            let anchor = p.pitches.iter().position(|&x| x == 1.0).unwrap();
            let octave = anchor + subdiv as usize;
            assert!(
                (p.pitches[octave] as f64 - 2.0).abs() < 1e-6,
                "subdiv {subdiv}: {subdiv} steps above A440 must be exactly 2.0, got {}",
                p.pitches[octave]
            );
        }
    }

    /// Band edges are half a grid step either side of the centre, so at the
    /// default subdivision each band captures exactly one semitone: ±50
    /// cents. Wider and it steals its neighbours' notes; narrower and the
    /// gaps between bands drop energy on the floor.
    #[test]
    fn band_edges_are_half_a_step_either_side() {
        for (subdiv, want_cents) in [(12u32, 50.0f64), (24, 25.0)] {
            let p = BandPlan::full(0, subdiv).unwrap();
            for k in 0..p.tonal_count() {
                let (lo, hi) = p.edges_hz(k);
                let centre = p.pitches[k] as f64 * BASE_HZ as f64;
                let lo_c = 1200.0 * (lo as f64 / centre).log2();
                let hi_c = 1200.0 * (hi as f64 / centre).log2();
                assert!(
                    (lo_c + want_cents).abs() < 0.02 && (hi_c - want_cents).abs() < 0.02,
                    "subdiv {subdiv} band {k}: edges are {lo_c:.3}..{hi_c:.3} cents, \
                     want -{want_cents}..+{want_cents}"
                );
            }
            // ...and consecutive bands must therefore meet, with no gap and
            // no overlap: band k-1's upper edge is band k's lower edge.
            for k in 1..p.tonal_count() {
                let (lo, _) = p.edges_hz(k);
                let (_, hi_prev) = p.edges_hz(k - 1);
                assert!(
                    ((lo as f64 - hi_prev as f64) / lo as f64).abs() < 1e-5,
                    "subdiv {subdiv}: band {} ends at {hi_prev} but band {k} starts at {lo}",
                    k - 1
                );
            }
        }
    }

    /// `edges_hz` and `fold` must classify identically: both derive from one
    /// expression ([`BandPlan::position`]), so this is the test that catches
    /// it if a change ever reintroduces a second copy that can disagree.
    #[test]
    fn fold_agrees_with_edges_hz_on_every_band() {
        // A window fine enough to place a bin inside any band of the full
        // plan: the narrowest band (46.2 Hz, one semitone) is 2.7 Hz wide,
        // and 65536 points at 48 kHz gives 0.73 Hz bins.
        const W: usize = 65_536;
        let p = BandPlan::full(0, 12).unwrap();
        let hz_per_bin = SR as f64 / W as f64;
        let bins = W / 2 + 1;
        // Every bin, not a probe near each edge: a rounded probe can land
        // outside the band it was aimed at, and then the test is measuring
        // its own arithmetic. The band each bin belongs to is read off
        // `edges_hz`, and `fold` must agree on all of them.
        let mut checked = 0usize;
        for bin in 1..bins {
            let hz = bin as f64 * hz_per_bin;
            let Some(want) = (0..p.tonal_count()).find(|&k| {
                let (lo, hi) = p.edges_hz(k);
                hz >= lo as f64 && hz < hi as f64
            }) else {
                continue; // off the ends of the bank; the noise-fold tests own those
            };
            let mut sp = vec![0.0f32; bins];
            sp[bin] = 1.0;
            let folded = p.fold(&sp, SR, W);
            assert_eq!(
                folded[want],
                1.0,
                "{hz:.4} Hz is inside band {want}'s edges {:?} but fold put it in {:?}",
                p.edges_hz(want),
                folded.iter().position(|&v| v > 0.0)
            );
            assert_eq!(folded.iter().sum::<f32>(), 1.0, "bin {bin}: energy was duplicated");
            checked += 1;
        }
        assert!(checked > 5_000, "only {checked} bins fell inside the bank; the test is vacuous");
    }

    /// Nothing may be dropped. With both noise bands present every bin has a
    /// home, so the folded power must equal the spectrum's power exactly
    /// (Parseval). This is the invariant that catches a bin rounding to a
    /// band index that does not exist -- silent destruction, which no
    /// single-impulse test can see.
    #[test]
    fn every_bin_reaches_exactly_one_band() {
        let p = plan();
        let bins = WIN / 2 + 1;
        // Deterministic and non-flat, so a fold that swallows a contiguous
        // run shows up as clearly as one that swallows a stray bin.
        let sp: Vec<f32> = (0..bins).map(|i| 1.0 + (i % 7) as f32 * 0.5).collect();
        let want: f64 = sp.iter().map(|&v| v as f64 * v as f64).sum();
        let got: f64 = p.fold(&sp, SR, WIN).iter().map(|&v| v as f64 * v as f64).sum();
        assert!(
            (got - want).abs() / want < 1e-6,
            "fold lost energy: {want} in, {got} out ({:.4}% missing)",
            100.0 * (want - got) / want
        );
    }

    // ------------------------------------------------------------------
    // Range, span, and the --bands / --subdiv contract.
    // ------------------------------------------------------------------

    #[test]
    fn the_default_plan_has_30_tonal_and_2_noise_bands() {
        let p = plan();
        assert_eq!(p.len(), 32);
        assert_eq!(p.kinds.iter().filter(|k| **k == BandKind::Tonal).count(), 30);
        assert_eq!(p.kinds[30], BandKind::WhiteNoise);
        assert_eq!(p.kinds[31], BandKind::PinkNoise);
    }

    /// A semitone grid anchored on A440 cannot end exactly on `PITCH_MIN` or
    /// `PITCH_MAX` -- 0.1 and 10.0 are not equal-tempered intervals from
    /// 440 Hz. A plan fills as much of the legal range as the scale allows
    /// and never leaves it.
    #[test]
    fn the_full_plan_is_the_widest_legal_equal_tempered_span() {
        let p = BandPlan::full(0, 12).unwrap();
        assert_eq!(p.len(), 79, "12 per octave inside 0.1x..10x is 79 semitones");
        assert_eq!(p.step_of(0), Some(-39));
        assert_eq!(p.step_of(78), Some(39));
        // Inside the hardware range...
        for (i, &pitch) in p.pitches.iter().enumerate() {
            assert!(
                (PITCH_MIN..=PITCH_MAX).contains(&pitch),
                "band {i} pitch {pitch} is outside the legal range"
            );
        }
        // ...and one more step at either end would leave it, which is what
        // makes this the widest span rather than merely a legal one.
        assert!(step_pitch(-40, 12) < PITCH_MIN, "step -40 must be below the floor");
        assert!(step_pitch(40, 12) > PITCH_MAX, "step 40 must be above the ceiling");
        // F#1 to C8: the top 79 keys of an 88-key piano.
        assert!((p.pitches[0] * BASE_HZ - 46.249).abs() < 0.01);
        assert!((p.pitches[78] * BASE_HZ - 4186.0).abs() < 0.1);
    }

    /// Quarter-tones: half the step, twice the bands, still exact intervals.
    #[test]
    fn a_finer_subdivision_halves_the_step_and_doubles_the_span_count() {
        assert_eq!(max_tonal_bands(12).unwrap(), 79);
        assert_eq!(max_tonal_bands(24).unwrap(), 159);
        let p = BandPlan::full(0, 24).unwrap();
        assert_eq!(p.len(), 159);
        assert_eq!(p.subdiv(), 24);
        // Adjacent bands are 50 cents apart, not 100.
        let c = 1200.0 * (p.pitches[1] as f64 / p.pitches[0] as f64).log2();
        assert!((c - 50.0).abs() < 0.02, "a quarter-tone step must be 50 cents, got {c:.3}");
        // Every semitone of the 12-grid is still present, exactly: the even
        // steps -78..=78 are the 79 semitones, and the 80 odd ones between
        // them are the new quarter-tones.
        let semis = p.pitches.iter().filter(|&&x| cents_off_semitone(x).abs() < 0.02).count();
        assert_eq!(semis, 79, "the quarter-tone grid must contain the whole semitones too");
        assert_eq!(p.len() - semis, 80, "the rest must be quarter-tones");
    }

    /// Asking for more bands than the pitch range holds is an error, not a
    /// silent clamp. A clamp would hand back a different speaker count than
    /// `--bands` asked for and the user would never know.
    #[test]
    fn too_many_bands_for_the_subdivision_is_rejected_not_clamped() {
        assert!(BandPlan::new(79, 0).is_ok(), "79 tonal is exactly the semitone maximum");
        let e = BandPlan::new(80, 0).expect_err("80 tonal must not fit at 12 per octave");
        assert!(e.contains("79"), "the error must name the maximum: {e}");
        assert!(e.contains("--subdiv"), "the error must point at the way out: {e}");
        assert!(BandPlan::new(96, 0).is_err());
        assert!(BandPlan::new(96, 2).is_err());
        // ...and --subdiv 24 really is the way out.
        assert!(BandPlan::with_subdiv(96, 0, 24).is_ok());
        assert!(BandPlan::with_subdiv(160, 0, 24).is_err());
    }

    #[test]
    fn a_zero_subdivision_is_rejected() {
        assert!(max_step(0).is_err());
        assert!(max_tonal_bands(0).is_err());
        assert!(BandPlan::with_subdiv(32, 2, 0).is_err());
    }

    #[test]
    fn tonal_pitches_are_strictly_increasing() {
        for p in [plan(), BandPlan::full(2, 12).unwrap(), BandPlan::full(0, 24).unwrap()] {
            for i in 1..p.tonal_count() {
                assert!(
                    p.pitches[i] > p.pitches[i - 1],
                    "band {i} ({}) must be above band {} ({})",
                    p.pitches[i],
                    i - 1,
                    p.pitches[i - 1]
                );
            }
        }
    }

    /// A tone at a band's own centre frequency must land in that band and
    /// nowhere else -- catches an off-by-one in the bin->band mapping that
    /// reading the code will not.
    #[test]
    fn a_tone_at_a_band_centre_lands_in_that_band() {
        let p = plan();
        let hz_per_bin = SR as f64 / WIN as f64;
        let half = p.half_step();
        let mut checked = 0;
        for band in 0..30 {
            let centre = p.pitches[band] as f64 * BASE_HZ as f64;
            // Skip bands narrower than one FFT bin at this window: placing an
            // impulse "at the centre" is meaningless there, because the
            // nearest bin can be outside the band entirely. At 4096/48 kHz
            // that is the bottom two bands of a 30-band plan.
            if centre * (1.0 - 1.0 / half) <= hz_per_bin / 2.0 {
                continue;
            }
            let folded = p.fold(&impulse_at(centre as f32), SR, WIN);
            let loudest = folded
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .expect("a non-empty fold");
            assert_eq!(
                loudest, band,
                "a tone at {centre} Hz (band {band}'s centre) landed in band {loudest}"
            );
            checked += 1;
        }
        assert!(checked >= 28, "the sub-bin guard skipped too much: only {checked} bands tested");
    }

    /// The same oracle over every band of the default full plan, at a window
    /// fine enough that no band is sub-bin.
    #[test]
    fn a_tone_at_a_band_centre_lands_in_that_band_across_the_full_plan() {
        const W: usize = 65_536;
        let p = BandPlan::full(2, 12).unwrap();
        let bins = W / 2 + 1;
        for band in 0..p.tonal_count() {
            let centre = p.pitches[band] as f64 * BASE_HZ as f64;
            let bin = (centre * W as f64 / SR as f64).round() as usize;
            let mut sp = vec![0.0f32; bins];
            sp[bin] = 1.0;
            let folded = p.fold(&sp, SR, W);
            assert_eq!(
                folded[band],
                1.0,
                "band {band} ({centre:.2} Hz) did not receive its own centre tone; it \
                 landed in {:?}",
                folded.iter().position(|&v| v > 0.0)
            );
        }
    }

    /// Energy above the top tonal band must be folded into white noise, not
    /// dropped. Dropping it makes cymbals vanish silently.
    #[test]
    fn energy_above_the_ceiling_folds_into_white_noise() {
        let p = plan();
        let folded = p.fold(&impulse_at(12_000.0), SR, WIN);
        assert!(folded[30] > 0.0, "12 kHz must reach the white-noise band");
        let tonal_total: f32 = folded[..30].iter().sum();
        assert!(
            tonal_total < folded[30] * 1e-3,
            "12 kHz must not leak into tonal bands (tonal {tonal_total}, white {})",
            folded[30]
        );
    }

    #[test]
    fn silence_folds_to_silence() {
        let p = plan();
        let folded = p.fold(&vec![0.0; WIN / 2 + 1], SR, WIN);
        assert_eq!(folded.len(), 32);
        assert!(folded.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn a_plan_with_no_room_for_tonal_bands_is_rejected() {
        assert!(BandPlan::new(2, 2).is_err(), "2 bands with 2 noise leaves no tonal bands");
        assert!(BandPlan::new(0, 0).is_err());
        assert!(BandPlan::new(32, 3).is_err(), "at most 2 noise bands exist");
    }

    /// A plan with no noise bands is legal and must still be on the scale.
    #[test]
    fn a_plan_with_no_noise_bands_is_all_tonal() {
        let p = BandPlan::new(16, 0).expect("16 tonal bands is valid");
        assert_eq!(p.len(), 16);
        assert!(p.kinds.iter().all(|k| *k == BandKind::Tonal));
        assert_eq!(p.step_of(0), Some(-8));
        assert_eq!(p.step_of(15), Some(7));
        for &pitch in &p.pitches {
            assert!((PITCH_MIN..=PITCH_MAX).contains(&pitch));
        }
    }

    /// The ceiling and the bin->band rounding must coincide exactly: a
    /// ceiling that is too low routes real top-octave tone into hiss, and one
    /// that is too high lets bins round to band `tonal`, which `fold` would
    /// then have to drop on the floor -- the silent disappearance the module
    /// doc explicitly forbids.
    #[test]
    fn the_white_fold_point_is_the_top_bands_upper_edge() {
        let p = plan();
        // The top tonal band (B5, 987.77 Hz) has an upper edge at 1016.71 Hz.
        // Bin 86 (1007.81 Hz) sits just inside it; bin 87 (1019.53 Hz) just
        // outside.
        let inside = p.fold(&impulse_at(1_007.0), SR, WIN);
        assert_eq!(inside[29], 1.0, "just below the edge belongs to the top tonal band");
        assert_eq!(inside[30], 0.0, "just below the edge must not reach white noise");
        assert_eq!(inside.iter().sum::<f32>(), 1.0, "energy below the edge was dropped");

        let outside = p.fold(&impulse_at(1_020.0), SR, WIN);
        assert_eq!(outside[30], 1.0, "just above the edge becomes white noise");
        assert_eq!(
            outside[..30].iter().sum::<f32>(),
            0.0,
            "just above the edge must leave the tonal bands"
        );
        assert_eq!(outside.iter().sum::<f32>(), 1.0, "energy above the edge was dropped");
    }

    /// No tonal speaker of this plan can reproduce sub-180 Hz content, so it
    /// folds onto PINK noise -- whose 1/f spectrum is itself low-heavy, making
    /// it the carrier that puts sub-bass energy back in roughly the right
    /// octaves. Letting it fall through to band 0 instead would pile every DC
    /// offset and rumble onto the lowest speaker at its exact pitch.
    #[test]
    fn energy_below_the_bottom_band_folds_into_pink_noise() {
        let p = plan();
        // Band 0 (F#3, 185.00 Hz) has a lower edge at 179.73 Hz; bin 3 is
        // 35.2 Hz.
        let folded = p.fold(&impulse_at(35.0), SR, WIN);
        assert_eq!(folded[31], 1.0, "35 Hz must reach the pink-noise band");
        assert_eq!(
            folded[..31].iter().sum::<f32>(),
            0.0,
            "sub-bank rumble must not pile into band 0 or white: {folded:?}"
        );
    }

    /// The two folds must not be crossed. White carries the top end and pink
    /// the bottom; swapping them puts sibilance in the sub and rumble in the
    /// hiss, which every single-ended test above still passes.
    #[test]
    fn the_two_noise_folds_go_to_opposite_ends() {
        let p = plan();
        let mut sp = impulse_at(35.0);
        let hi = (12_000.0f64 * WIN as f64 / SR as f64).round() as usize;
        sp[hi] = 2.0;
        let folded = p.fold(&sp, SR, WIN);
        assert_eq!(folded[30], 2.0, "white must carry the 12 kHz energy, not the 35 Hz");
        assert_eq!(folded[31], 1.0, "pink must carry the 35 Hz energy, not the 12 kHz");
        assert_eq!(folded[..30].iter().sum::<f32>(), 0.0, "neither belongs to a tonal band");
    }

    /// DC (bin 0) is below the bottom edge like any other sub-bass bin, and
    /// `log2(0)` is -inf there. It must fold to pink, not panic, NaN, or land
    /// in band 0 via a saturating cast on a garbage position.
    #[test]
    fn dc_folds_into_pink_noise_rather_than_a_tonal_band() {
        let p = plan();
        let folded = p.fold(&impulse_at(0.0), SR, WIN);
        assert!(folded.iter().all(|v| v.is_finite()), "DC produced a non-finite fold: {folded:?}");
        assert_eq!(folded[31], 1.0, "DC must reach the pink-noise band");
        assert_eq!(folded[..31].iter().sum::<f32>(), 0.0, "DC must not reach a tonal band");
    }

    /// Symmetric partner to `the_white_fold_point_is_the_top_bands_upper_edge`.
    /// A floor that is too high steals real bottom-octave tone into pink, and
    /// one that is too low lets sub-bass round up into band 0.
    #[test]
    fn the_pink_fold_point_is_the_bottom_bands_lower_edge() {
        let p = plan();
        // Band 0's lower edge is 179.73 Hz. Bin 16 (187.50 Hz) sits just
        // inside it; bin 15 (175.78 Hz) just outside.
        let inside = p.fold(&impulse_at(187.0), SR, WIN);
        assert_eq!(inside[0], 1.0, "just above the edge belongs to the bottom tonal band");
        assert_eq!(inside[31], 0.0, "just above the edge must not reach pink noise");
        assert_eq!(inside.iter().sum::<f32>(), 1.0, "energy above the edge was dropped");

        let outside = p.fold(&impulse_at(175.0), SR, WIN);
        assert_eq!(outside[31], 1.0, "just below the edge becomes pink noise");
        assert_eq!(
            outside[..31].iter().sum::<f32>(),
            0.0,
            "just below the edge must leave the tonal bands"
        );
        assert_eq!(outside.iter().sum::<f32>(), 1.0, "energy below the edge was dropped");
    }

    /// `--noise-bands 1` buys white only. Sub-bass has nowhere legal to go,
    /// so it is discarded -- not redirected into white (the wrong end of the
    /// spectrum) and not written to a band index that does not exist. All
    /// three plans below have 30 tonal bands, so they share the default
    /// plan's 179.73 Hz floor.
    #[test]
    fn with_only_a_white_band_sub_bass_is_still_discarded() {
        let p = BandPlan::new(31, 1).expect("30 tonal + white is valid");
        assert_eq!(p.len(), 31);
        assert_eq!(p.kinds[30], BandKind::WhiteNoise);
        assert!(!p.kinds.contains(&BandKind::PinkNoise), "this plan must have no pink band");

        let folded = p.fold(&impulse_at(35.0), SR, WIN);
        assert!(
            folded.iter().all(|&v| v == 0.0),
            "with no pink band, sub-bass must be dropped, not rerouted: {folded:?}"
        );
        // The white fold is unaffected by the absence of pink.
        assert_eq!(p.fold(&impulse_at(12_000.0), SR, WIN)[30], 1.0);
    }

    /// `--noise-bands 0` buys neither, so both ends are discarded.
    #[test]
    fn with_no_noise_bands_both_ends_are_discarded() {
        let p = BandPlan::new(30, 0).expect("30 tonal bands is valid");
        assert!(p.kinds.iter().all(|k| *k == BandKind::Tonal));

        for hz in [0.0, 35.0, 12_000.0] {
            let folded = p.fold(&impulse_at(hz), SR, WIN);
            assert!(
                folded.iter().all(|&v| v == 0.0),
                "with no noise bands, {hz} Hz must be dropped, not rerouted: {folded:?}"
            );
        }
    }

    // --- Aggregation. Every test above uses a single unit impulse, whose
    // --- fold is identical under a magnitude sum and a power sum, so none
    // --- of them can see the width bias. These are the ones that can.

    /// `n` equal bins inside one band must fold to `sqrt(n)` times a single
    /// bin, not `n` times: a band reports the energy it owns, as the
    /// amplitude of one equivalent sine. Magnitude-summing instead grows
    /// linearly with a band's bin count, and because the bank is constant-Q
    /// that count scales with centre frequency -- so the error is a systematic
    /// brightness tilt, not a wash.
    #[test]
    fn a_band_aggregates_power_not_magnitude() {
        let p = plan();
        // Band 29 (B5) spans 959.65..1016.71 Hz, so both of these land in it.
        let one = p.fold(&impulse_at(984.0), SR, WIN);
        assert_eq!(one[29], 1.0, "one unit bin is one unit of amplitude");

        let two = p.fold(&impulses_at(&[984.0, 996.0]), SR, WIN);
        assert!(
            (two[29] - 2.0f32.sqrt()).abs() < 1e-5,
            "two equal bins in band 29 must fold to sqrt(2) = 1.414, got {} \
             (2.0 means magnitude-summing)",
            two[29]
        );

        // Same law on the noise folds, which are the widest bands of all.
        let four_hi = p.fold(&impulses_at(&[9_000.0, 11_000.0, 13_000.0, 15_000.0]), SR, WIN);
        assert!(
            (four_hi[30] - 2.0).abs() < 1e-5,
            "four equal bins above the ceiling must fold to sqrt(4) = 2, got {} \
             (4.0 means magnitude-summing)",
            four_hi[30]
        );
        let four_lo = p.fold(&impulses_at(&[0.0, 11.7, 23.4, 35.2]), SR, WIN);
        assert!(
            (four_lo[31] - 2.0).abs() < 1e-5,
            "four equal bins below the floor must fold to sqrt(4) = 2, got {}",
            four_lo[31]
        );
    }

    /// The white band alone covers ceiling..Nyquist, hundreds of times more
    /// bins than any one semitone-wide tonal band. Energy aggregation makes
    /// the gap the square root of the bin ratio, the honest amount of energy
    /// a wide band holds.
    ///
    /// Expected values are computed from the band edges rather than
    /// hardcoded, so this keeps testing the aggregation law and not the
    /// geometry of one particular plan.
    #[test]
    fn a_flat_spectrum_does_not_let_the_wide_white_band_dominate() {
        let p = plan();
        let folded = p.fold(&vec![1.0f32; WIN / 2 + 1], SR, WIN);
        let hz_per_bin = SR as f64 / WIN as f64;
        let (top_lo, top_hi) = p.edges_hz(29);

        let white_bins = (0..=WIN / 2).filter(|&b| b as f64 * hz_per_bin >= top_hi as f64).count();
        let top_bins = (0..=WIN / 2)
            .filter(|&b| {
                let hz = b as f64 * hz_per_bin;
                hz >= top_lo as f64 && hz < top_hi as f64
            })
            .count();
        assert!(white_bins > 100 && top_bins > 1, "this test needs a wide white and a narrow top");

        assert!(
            (folded[30] - (white_bins as f32).sqrt()).abs() < 1e-2,
            "white must fold to sqrt({white_bins}) = {}, got {}",
            (white_bins as f32).sqrt(),
            folded[30]
        );
        assert!(
            (folded[29] - (top_bins as f32).sqrt()).abs() < 1e-4,
            "the top tonal band must fold to sqrt({top_bins}), got {}",
            folded[29]
        );

        let ratio = folded[30] / folded[29];
        let want = (white_bins as f32 / top_bins as f32).sqrt();
        assert!(
            (ratio - want).abs() < 0.05,
            "white must sit sqrt({white_bins}/{top_bins}) = {want:.2}x the top tonal band, \
             got {ratio:.2}x ({:.0}x means magnitude-summing is back)",
            white_bins as f32 / top_bins as f32
        );
    }

    /// PITCH_MIN/PITCH_MAX are the emitter's hardware limits, so they bind
    /// every speaker in the bank at every plan size -- noise bands included,
    /// which a tonal-only range check never looks at.
    #[test]
    fn every_speaker_pitch_is_inside_the_hardware_range() {
        for subdiv in [1u32, 12, 24, 31] {
            let max_tonal = max_tonal_bands(subdiv).unwrap();
            for noise in 0..=MAX_NOISE_BANDS {
                for tonal in 2..=max_tonal {
                    let bands = tonal + noise;
                    let p = BandPlan::with_subdiv(bands, noise, subdiv).unwrap_or_else(|e| {
                        panic!("{bands} bands with {noise} noise at subdiv {subdiv}: {e}")
                    });
                    assert_eq!(p.len(), bands);
                    for (i, &pitch) in p.pitches.iter().enumerate() {
                        assert!(
                            pitch.is_finite() && (PITCH_MIN..=PITCH_MAX).contains(&pitch),
                            "subdiv {subdiv}, {bands} bands / {noise} noise: band {i} pitch \
                             {pitch} is illegal"
                        );
                    }
                }
            }
        }
    }

    /// A lone tonal band cannot span a range: a bank of one speaker is not a
    /// filterbank, and accepting it would silently produce a save that plays
    /// exactly one note.
    #[test]
    fn a_single_tonal_band_is_rejected() {
        assert!(BandPlan::new(1, 0).is_err(), "1 band cannot span a range");
        assert!(BandPlan::new(3, 2).is_err(), "3 bands minus 2 noise leaves 1 tonal band");
    }
}
