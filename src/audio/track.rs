//! Spectrum -> per-band linear volumes, normalised across the whole track.
use super::bands::{BandKind, BandPlan};
use super::source::AudioSource;
use super::speakers::{DEFAULT_INNER_RADIUS, DEFAULT_MAX_DISTANCE};
use super::stft::{frame_count_for, hop_for, StftStream};
use crate::anim::pack::BANK_FRAMES;
use crate::progress::{FrameTotal, Progress};

#[derive(Clone, Copy, Debug)]
pub struct AudioOptions {
    pub fps: f32,
    /// Total speakers, noise bands included -- or `None` for "as many as the
    /// hardware pitch range holds at [`Self::subdiv`]".
    ///
    /// The tonal spacing is no longer free: bands sit on exact equal-tempered
    /// steps ([`BandPlan`]), so a count cannot change the interval, only how
    /// much of the scale is covered. `Some(n)` therefore **selects the span**
    /// -- the `n - noise_bands` steps closest to A440 -- and asking for more
    /// than the range holds is an error naming the maximum, never a clamp.
    ///
    /// `None` is the default because with the spacing fixed there is no other
    /// honest one: 79 tonal bands at 12 steps per octave (F#1..C8), 159 at 24.
    pub bands: Option<usize>,
    pub noise_bands: usize,
    /// Tonal bands per octave. 12 = one per semitone; see
    /// [`crate::audio::bands::DEFAULT_SUBDIV`].
    pub subdiv: u32,
    pub window: usize,
    /// Post-normalisation multiplier. Clamped at 1.0 after application.
    pub gain: f32,
    /// Band energies at or below this many dB below **the loudest band in
    /// the same frame** become exactly zero.
    ///
    /// Relative to the frame, not to the track: see [`analyze`] for the
    /// measurements that forced the change.
    pub floor_db: f32,
    /// Per-frame leveling, 0.0 (off) to 1.0 (full automatic gain control,
    /// **and the default** -- see the `Default` impl below for the listening
    /// that put it there). Every setting keeps the frame mix inside full
    /// scale; see [`analyze`].
    pub leveling: f32,
    /// How many bands may sound at once. Every other band in the frame is set
    /// to exactly zero. `0` disables selection entirely and lets every band
    /// through, which is the behaviour that made the render sound like noise.
    ///
    /// **This is the flag that turns a filterbank back into music.** See
    /// [`select_peaks`] for the measurements.
    pub max_voices: usize,
    /// How far a spectral peak must stand above its neighbourhood before it
    /// counts as a note at all, as an amplitude RATIO -- see [`prominence`].
    ///
    /// **This, not [`Self::max_voices`], is what limits a dense frame.** The
    /// gate leaves ~12-24 candidates per frame at the default 1.5, so
    /// `--max-voices` above that is inert and every render at 24, 32, 48 and 64
    /// voices is the same render. Lowering it is the only way to sound more
    /// bands at once; 1.0 disables it entirely and lets every local maximum
    /// through. See [`DEFAULT_PEAK_GATE`] for the measurements behind the
    /// default and for what going below it costs.
    ///
    /// Voice mode has its own gate of the same name and value
    /// (`voices::MIN_PROMINENCE`), measured over a different neighbourhood;
    /// this flag does not reach it.
    pub peak_gate: f32,
    /// Envelope attack time in MILLISECONDS: how quickly a band's output level
    /// rises toward the level the analysis just measured. See [`Envelope`].
    pub attack_ms: f32,
    /// Envelope release time in MILLISECONDS: how quickly it falls. Longer
    /// than the attack by design -- see [`Envelope`], and
    /// [`DEFAULT_RELEASE_MS`] for what it fixes.
    pub release_ms: f32,
    /// **Voice mode only.** How long a voice takes to fade to EXACTLY zero once
    /// its partial has gone, in milliseconds.
    ///
    /// Separate from [`Self::release_ms`], which is a one-pole time constant on
    /// a *sounding* voice's level and never reaches zero. This one is the end
    /// of a note, and the material decides it: a piano note has a natural decay
    /// and can afford a slow one, a spoken phoneme is 50-100 ms long and a
    /// release anywhere near [`DEFAULT_RELEASE_MS`] smears clean across it --
    /// which is what the owner heard as speech coming out "disembodied".
    ///
    /// See [`DEFAULT_VOICE_RELEASE_MS`].
    pub voice_release_ms: f32,
    /// **Voice mode only.** A continuing voice whose pitch is within this many
    /// cents of an equal-tempered semitone is pulled onto it. 0 (the default)
    /// disables snapping entirely; `voices::MAX_PITCH_SNAP_CENTS` (50, half a
    /// semitone) is the largest value that can mean anything, and larger ones
    /// are refused by name at parse time rather than at the end of the render.
    ///
    /// Offered, and off, deliberately. The argument for voice mode is that a
    /// tracked voice needs no grid, and this puts a shallow one back -- it
    /// quantises real vibrato and glissando away along with the jitter. It
    /// exists because a *stable* pitch a few cents wrong may still sound better
    /// than a correct one that wobbles, which only a listener can settle. The
    /// jitter itself is addressed first by `voices::PITCH_GLIDE_MS`, which
    /// costs nothing musically.
    pub pitch_snap_cents: f32,
    /// Baked `InnerRadius` on every speaker, in game units: the radius inside
    /// which there is NO distance attenuation, so every band reaches the
    /// listener at exactly the same level.
    ///
    /// Not cosmetic. `bSpatialization` is false, which turns panning off but
    /// leaves distance attenuation on, so this and [`Self::max_distance`] are
    /// what decide whether the bank sounds like one instrument or like a
    /// distance-filtered slice of the spectrum. See `speakers::DEFAULT_INNER_RADIUS`.
    pub inner_radius: f32,
    /// Baked `MaxDistance` on every speaker, in game units: where the sound
    /// stops. See [`Self::inner_radius`].
    pub max_distance: f32,
    pub bank_size: usize,
    pub max_frames: usize,
    pub external_clock: bool,
    /// Repeat the track forever (`true`, the default) or stop on its last
    /// analysis frame (`false`).
    ///
    /// The same field, the same meaning and the same single effect as
    /// [`crate::anim::bricks::AnimOptions::loop_playback`]: both render paths
    /// drive one shared clock (`clock::build_clock`), and this becomes the
    /// value inlined on its `Timer.Limit` -- see
    /// [`crate::anim::clock::stop_limit`] for the arithmetic and for what
    /// about it is still unverified in game. Costs nothing either way: same
    /// gates, same wires, same speakers.
    ///
    /// Unlike [`Self::external_clock`], which the audio path refuses outright
    /// because `speakers::build_speaker_world` reads none of it, this one IS
    /// read there -- the speaker chip builds the shared clock like every other
    /// render does.
    pub loop_playback: bool,
    /// Place the speaker cluster INSIDE the microchip's own inner grid rather
    /// than beside it on the world's main grid (`false`, the default and every
    /// render before this flag existed).
    ///
    /// On (`true`) the whole audio device is one portable microchip: the
    /// speakers sit in their own block of the inner grid clear of the gate
    /// lattice (see `speakers::speaker_inner_position`), and the per-band
    /// volume/pitch wires that were cross-grid remote wires become same-grid
    /// internal ones. Nothing else changes -- same speakers, same gates, same
    /// wires, same baked emitter data.
    ///
    /// **The inner-grid layout is physical placement only, NOT spatialisation.**
    /// An `AudioEmitter` on a microchip inner grid emits from the CHIP'S WORLD
    /// POSITION regardless of where its brick sits inside the grid, so the
    /// cluster's shape here buys none of the near-field equal-level geometry the
    /// beside-the-chip layout does (which is moot anyway with
    /// `bSpatialization = false`). Whether an in-chip speaker plays at all, and
    /// that it plays from the chip origin, are game-runtime facts the owner must
    /// confirm on a test render.
    pub speakers_in_chip: bool,
}

impl Default for AudioOptions {
    fn default() -> Self {
        Self {
            fps: 30.0,
            // Fill the scale. See the field doc.
            bands: None,
            // OFF. Every source tried in game -- speech, solo piano and a full
            // pop mix -- was reported worse with the noise bands than without:
            // "the noise sounds sound bad on all of those". They are kept
            // behind the flag rather than deleted because percussion is the one
            // case where a broadband bed plausibly earns its slot, and nothing
            // else in the bank can render a cymbal at all.
            noise_bands: 0,
            subdiv: crate::audio::bands::DEFAULT_SUBDIV,
            window: 4096,
            gain: 1.0,
            floor_db: -60.0,
            // FULL. Opt-in was the wrong default and was reverted on
            // listening evidence: 1.0 beat 0.5 and beat off, on every source
            // tried in game -- piano, speech and a music box alike.
            //
            // The reasoning that made it opt-in ("a track that arrives with
            // dynamics leaves with them") is true of a normal playback medium
            // and false of this one. A bank of sine emitters has perhaps 30 dB
            // of usable range before its quiet passages fall under the room,
            // where a master has 60 or more, so the source's own dynamics do
            // not survive the trip regardless -- and what a listener hears
            // without leveling is not dynamics but a render that vanishes for
            // half its length. `MAX_LEVELING_BOOST` is what keeps that from
            // becoming an amplified noise floor.
            leveling: 1.0,
            // The knee of the measured energy/voice-count curve. See
            // `select_peaks`.
            max_voices: 12,
            // The measured knee of the prominence/frame-power curve. See
            // `DEFAULT_PEAK_GATE`.
            peak_gate: DEFAULT_PEAK_GATE,
            attack_ms: DEFAULT_ATTACK_MS,
            release_ms: DEFAULT_RELEASE_MS,
            voice_release_ms: DEFAULT_VOICE_RELEASE_MS,
            // Off. Voice mode's whole claim is that it needs no grid.
            pitch_snap_cents: 0.0,
            // Sourced from `speakers`, not spelled out again: these are the
            // values baked into every emitter, and the reasoning for them
            // lives next to the field they are written into.
            inner_radius: DEFAULT_INNER_RADIUS,
            max_distance: DEFAULT_MAX_DISTANCE,
            bank_size: BANK_FRAMES,
            max_frames: BANK_FRAMES * 16,
            external_clock: false,
            // LOOPING, matching `AnimOptions` and matching what every audio
            // render did before the flag existed.
            loop_playback: true,
            // OFF: speakers beside the chip on the main grid, unchanged from
            // every render before the flag. See the field doc.
            speakers_in_chip: false,
        }
    }
}

pub struct VoiceTrack {
    pub plan: BandPlan,
    /// `volumes[band][frame]`, linear 0..=1. Band-major because that is
    /// exactly one `ArrayVar` per band -- frame-major would force a transpose
    /// of millions of elements at build time.
    pub volumes: Vec<Vec<f64>>,
    pub fps: f32,
    pub frame_count: usize,
}

/// Default envelope attack, in milliseconds.
///
/// One analysis frame at the default 30 fps is 33 ms, so 10 ms is a coefficient
/// of `1 - exp(-33/10)` = 0.96: a band that is asked to get louder is
/// essentially there within one frame. That is deliberate -- attacks are what
/// make a render sound like an instrument being struck rather than a pad
/// swelling, and the whole point of the asymmetry is that the fix for chatter
/// must not cost transient definition.
pub const DEFAULT_ATTACK_MS: f32 = 10.0;

/// Default envelope release, in milliseconds.
///
/// **This is the fix for "a lot of random beep pitch sounds".**
///
/// The renderer had no envelope at all: a band's output was whatever that
/// frame's FFT said, so a band selected for a single frame produced a 33 ms
/// blip -- a beep. Piano was reported "really good" with the same code, and that
/// contrast is the diagnosis rather than a coincidence: a piano partial is
/// sustained, so a band selected once stays selected for many frames, while
/// speech formants and percussion are transient and select for one or two.
/// Every real channel vocoder has an attack/release envelope for exactly this
/// reason and this one did not.
///
/// 150 ms is the middle of the standard vocoder range (50-200 ms). At 30 fps
/// the coefficient is `1 - exp(-33/150)` = 0.20, so a band that loses its
/// selection decays about 1.9 dB per frame instead of cutting to zero, and a
/// one-frame blip becomes a short note with a tail. It applies **after
/// deselection as well as during it**, which is most of the fix: the abrupt
/// part of a blip is the end, not the start.
///
/// This is complementary to [`VOICE_HYSTERESIS`] and does not replace it.
/// Hysteresis stabilises WHICH bands are selected; the envelope stabilises HOW
/// THEIR LEVEL MOVES. A band can still be deselected and, without an envelope,
/// still cuts.
pub const DEFAULT_RELEASE_MS: f32 = 150.0;

/// Default `--voice-release`, in milliseconds: how long a voice-mode speaker
/// takes to fade to exactly zero once its partial has gone.
///
/// **This is the fix for "you might need to zero the volume after a note is
/// done playing" and for speech coming out "disembodied".** Voice mode used to
/// take this time from `--release` (150 ms), which is a level-follower time
/// constant meant for a bank whose bands never stop existing. Fifty ms is one
/// and a half analysis frames at the default 30 fps and roughly a third of a
/// short phoneme, so a word's consonants no longer smear into its vowels.
///
/// # Why not shorter, and why shorter would not click
///
/// The floor is the format, not the ear. Volumes are written once per analysis
/// frame, so the shortest fade that can be EXPRESSED is one frame -- 33 ms at 30
/// fps -- and a linear fade only becomes audible as a click somewhere under
/// about 5-10 ms on a full-scale sine. A one-frame release is therefore already
/// 3-7x longer than the shortest inaudible fade, and there is no shorter
/// setting to be had at any `--audio-fps` a build actually uses. (`--voice-release 0`
/// is accepted and clamps to that one frame; it is the documented A/B.) The
/// separate worry -- that a volume write RETRIGGERS the emitter rather than
/// ramping it, which would click regardless of length -- was settled in game:
/// diagnostic 3 fades smoothly.
///
/// 50 ms is one frame more than that minimum at 30 fps, which buys the one
/// thing a hard cut gives up: a partial that dips under the prominence gate for
/// a single frame is a dropout, not a note ending, and a voice with a frame of
/// ramp in hand resumes it instead of dying and being reborn somewhere else.
///
/// Sustained material can afford far more -- `--voice-release 150` restores the
/// old behaviour -- and that is exactly why this is a flag.
pub const DEFAULT_VOICE_RELEASE_MS: f32 = 50.0;

/// An asymmetric one-pole level follower: fast up, slow down.
///
/// `level += (target - level) * coeff`, with `coeff` chosen by direction.
/// `coeff = 1 - exp(-dt / tau)` is the exact discretisation of a first-order
/// lag, so the time constants mean the same thing at any `--audio-fps` -- a
/// coefficient written directly would silently change meaning with the frame
/// rate.
///
/// A time of 0 gives a coefficient of exactly 1, i.e. no smoothing at all, and
/// `--attack 0 --release 0` is therefore the documented A/B against the
/// unsmoothed renderer.
#[derive(Clone, Copy, Debug)]
pub struct Envelope {
    pub attack: f32,
    pub release: f32,
}

impl Envelope {
    /// Build from times in milliseconds at `fps` frames per second.
    pub fn new(attack_ms: f32, release_ms: f32, fps: f32) -> Self {
        let dt_ms = 1000.0 / fps;
        let coeff = |tau: f32| {
            if tau <= 0.0 || !tau.is_finite() {
                1.0
            } else {
                (1.0 - (-dt_ms / tau).exp()).clamp(0.0, 1.0)
            }
        };
        Self {
            attack: coeff(attack_ms),
            release: coeff(release_ms),
        }
    }

    /// One step toward `target`.
    pub fn step(&self, level: f32, target: f32) -> f32 {
        let c = if target >= level { self.attack } else { self.release };
        level + (target - level) * c
    }

    /// True when neither coefficient smooths anything, i.e. the follower is the
    /// identity and the render is bit-identical to the pre-envelope one.
    pub fn is_off(&self) -> bool {
        self.attack >= 1.0 && self.release >= 1.0
    }

    /// Run the follower over one band's whole time series, in place.
    ///
    /// Starts from the first value rather than from zero: an envelope
    /// initialised at 0 fades the first note of the track in over its attack,
    /// which is a fade that is not in the source.
    pub fn apply(&self, series: &mut [f32]) {
        if self.is_off() || series.is_empty() {
            return;
        }
        let mut level = series[0];
        for v in series.iter_mut() {
            level = self.step(level, *v);
            *v = level;
        }
    }
}

/// The largest per-frame boost [`AudioOptions::leveling`] may apply, as a
/// linear amplitude factor (20 dB).
///
/// Without a cap, `leveling = 1.0` divides by the frame's own mix, so a
/// near-silent frame gets an unbounded boost and the decoder's noise floor
/// arrives at full scale between the notes. Measured on a pop track: the
/// quietest 1% of frames peak at 0.004 of the track peak, which uncapped is a
/// 250x boost.
///
/// It is NOT what bounds the output -- the cap only ever makes the scale
/// smaller than the leveling asked for, and the bound comes from the ratio
/// being taken between mixes. See [`analyze`].
const MAX_LEVELING_BOOST: f32 = 10.0;

/// How much louder a band that is ALREADY sounding is treated as being when
/// it competes for one of the [`AudioOptions::max_voices`] slots.
///
/// Pure rank bias: it never changes a volume, only the order candidates are
/// ranked in, so the selected set is still exactly `max_voices` wide. A
/// challenger has to beat an incumbent by this factor (3.5 dB) to take its
/// slot.
///
/// Measured on a pop master at 96 bands / `--max-voices 12`: the fraction of
/// "on" runs that last a single 33 ms frame -- a band that fires and dies
/// inside one frame, which is a chirp rather than a note -- falls from 16.8%
/// to 8.1%, and the per-frame set turnover from 4.97 bands to 3.93. The cost
/// is 0.06 dB of captured frame power (0.4986 -> 0.4922), because the band it
/// declines to swap in was, by construction, within 3.5 dB of the one it kept.
/// Higher values keep buying stability at an accelerating price: 2.0 gives
/// 7.2% / 3.48 for 0.13 dB, 3.0 gives 7.4% / 3.08 for 0.23 dB -- 3.0 is already
/// past the point where the chatter stops falling.
const VOICE_HYSTERESIS: f32 = 1.5;

/// Half-width, in bands ON THE FREQUENCY AXIS, of the neighbourhood a band's
/// [`prominence`] is measured against. The window is `2 * this + 1` bands wide
/// and excludes the band itself.
///
/// Measured, not guessed. It has to straddle two lengths:
///
/// * **Wider than one tone's own footprint**, or a note's leakage skirt becomes
///   its own neighbourhood and every prominence collapses toward 1. A pure sine
///   through the real pipeline (96 bands, window 16384) lights **1 band** above
///   10% of its apex at 208 Hz and above, and **3 bands** at the very bottom of
///   the bank, where a band is narrower than the FFT's own bin spacing.
/// * **Narrower than the gap between separate notes**, or one chord tone is
///   measured against another and both look flat. At 96 bands over the emitter's
///   0.1..10 pitch range a band is 0.845 semitones, so the closest interval that
///   is still a chord -- a minor third, 3 semitones -- is 3.6 bands away.
///
/// 3 is the tightest half-width that clears the 3-band worst-case footprint,
/// and it stays inside the 3.6-band note spacing. Measured mean prominence of
/// selected bands falls monotonically with width (1.84/1.61/1.50/1.47/1.41/1.39
/// at half-widths 1/2/3/4/6/8 in the low third), which is the skirt-dilution
/// effect above; the number itself carries no information about the right width,
/// only the two lengths do.
const PROMINENCE_WIDTH: usize = 3;

/// The [`prominence`] a local maximum needs before it counts as a note at all.
///
/// **This constant, not the ranking, is what sparsifies a real track.** The
/// local-maximum test leaves 29.02 candidates in an average frame of a pop
/// master at 96 bands, so at `--max-voices 32` or `48` the top-N truncation
/// never fires and the rank key -- whatever it is -- selects nothing. Every
/// scheme measured (magnitude, pure prominence, three prominence/magnitude
/// blends) produced byte-identical output at `--max-voices 32` for exactly that
/// reason. A render at 32 voices was therefore "every local maximum", i.e. 29
/// sine oscillators sounding at once, which is the wall this module exists to
/// prevent.
///
/// 1.5 is a band standing 3.5 dB above the mean of its neighbourhood. Measured
/// on the same track:
///
/// | gate | candidates/frame | frame power | mean prominence | HF voices/frame | silent frames |
/// |------|------------------|-------------|-----------------|-----------------|---------------|
/// | none | 29.02            | 54.1%       | 1.82            | 8.18            | 0             |
/// | 1.3  | 16.36            | 45.2%       | 2.12            | 5.22            | 0             |
/// | 1.5  | 12.20            | 38.6%       | 2.37            | 3.67            | 0             |
/// | 2.0  |  6.30            | 28.7%       | 2.98            | 1.66            | 0.607%        |
///
/// At 1.5 the pool drops 58% while frame power drops 29%, so what leaves is
/// worth less per voice than what stays (2.27% -> 3.16% of frame power each).
/// "HF voices" is the count above 979 Hz, which is the reported "random high
/// pitched sounds": broadband cymbal and sibilance content has local maxima
/// like anything else, and rendering 8 of them per frame as pure sines is what
/// they sound like. 1.5 more than halves that.
///
/// 2.0 is the next step and is rejected: it starts leaving frames that have
/// real energy with NO voice at all (0.607% of sounding frames), which is
/// audible as chopping.
///
/// # It is a flag, and it has to be
///
/// This constant is now only the DEFAULT of [`AudioOptions::peak_gate`]. It was
/// a fixed constant for one release and that was a mistake: **the gate, not
/// `--max-voices`, is what limits a dense frame**, so above the number of
/// candidates it leaves, raising `--max-voices` does nothing at all. Measured
/// on a solo piano master at `--window 16384`, renders at `--max-voices` 24,
/// 32, 48 and 64 came back 311925 / 311919 / 311917 / 311913 bytes -- twelve
/// bytes apart across a 40-voice range, i.e. the same render four times. A
/// listener asking for more voices got none, with nothing anywhere to say so.
///
/// Lowering the gate is what unlocks the range; `--peak-gate 1.0` disables it
/// entirely (every local maximum is a candidate) and restores the pre-gate
/// "wall of sound" behaviour on demand.
pub const DEFAULT_PEAK_GATE: f32 = 1.5;

/// The exponent on a candidate's OWN magnitude in the rank key
/// `prominence * magnitude^this`.
///
/// A pure prominence rank (exponent 0) has no level term at all, so nothing
/// stops it promoting a quiet-but-pointy ripple over a loud note. Measured at
/// `--max-voices 8`, where the truncation actually binds:
///
/// | rank key           | frame power | mean prominence | selected >40 dB down |
/// |--------------------|-------------|-----------------|----------------------|
/// | magnitude (before) | 40.0%       | 2.36            | 0.00%                |
/// | prominence         | 32.4%       | 2.63            | 0.26%                |
/// | prom * mag^0.25    | 35.3%       | 2.62            | 0.02%                |
/// | prom * mag^0.5     | 37.1%       | 2.60            | 0.00%                |
/// | prom * mag^1       | 39.0%       | 2.55            | 0.00%                |
///
/// 0.5 keeps 89% of the prominence gain a pure rank buys (2.36 -> 2.60 of
/// 2.36 -> 2.63) for 38% of its cost in frame power (7.6 points -> 2.9), and it
/// is the smallest exponent that puts the quiet-band promotion back to zero.
/// It also beats plain magnitude on run stability (82.4% vs 81.1% of selected
/// bands persisting to the next frame).
const RANK_MAGNITUDE_EXP: f32 = 0.5;

/// Ceiling on a reported [`prominence`], so that a peak standing in perfect
/// silence -- neighbourhood mean exactly 0, which is a real case on synthetic
/// signals and on digital fades -- stays a finite, ORDERABLE number. Two such
/// peaks must still be ranked against each other by their magnitudes rather
/// than tying at infinity.
const MAX_PROMINENCE: f32 = 1e6;

/// How sharply the band at frequency-axis position `i` stands above its local
/// neighbourhood: its magnitude divided by the MEAN of the
/// [`PROMINENCE_WIDTH`] bands either side of it, itself excluded.
///
/// This is the measure that separates a note from a lump. A real musical
/// partial is a narrow, high-contrast spike; bass rumble, reverb tails and the
/// broadband hiss of cymbals and sibilance are broad and flat, and they carry
/// local maxima that are indistinguishable from notes by magnitude alone.
///
/// Indexing is by AXIS POSITION, not by storage index: with noise bands present
/// the two differ, and a neighbourhood taken in storage order would compare a
/// 40 Hz band against a cymbal wash. See [`frequency_axis`].
fn prominence(frame: &[f32], axis: &[usize], i: usize) -> f32 {
    let lo = i.saturating_sub(PROMINENCE_WIDTH);
    let hi = (i + PROMINENCE_WIDTH).min(axis.len() - 1);
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for k in lo..=hi {
        if k != i {
            sum += frame[axis[k]] as f64;
            count += 1;
        }
    }
    // A bank with no neighbours at all, and a neighbourhood that is digitally
    // silent, are the same case: nothing to stand above, so the peak is as
    // prominent as the representation allows.
    if count == 0 {
        return MAX_PROMINENCE;
    }
    let mean = sum / count as f64;
    if mean <= 0.0 {
        return MAX_PROMINENCE;
    }
    ((frame[axis[i]] as f64 / mean) as f32).min(MAX_PROMINENCE)
}

/// Band indices in FREQUENCY order, which is not storage order.
///
/// `BandPlan` stores the tonal bands in pitch order and then appends white and
/// pink. But those two are not off to one side of the spectrum -- `fold` sends
/// everything ABOVE the top tonal band's edge to white and everything BELOW
/// the bottom one's to pink, so in frequency they are the two ends of the same
/// axis: `pink < tonal[0] .. tonal[n-1] < white`.
///
/// Ordering them that way is what lets [`select_peaks`] treat all bands
/// alike. It also all but retires the endpoint question: with the default two
/// noise bands the tonal ends have neighbours on both sides, and only pink and
/// white are true endpoints.
fn frequency_axis(plan: &BandPlan) -> Vec<usize> {
    let mut axis = Vec::with_capacity(plan.len());
    axis.extend(plan.kinds.iter().position(|k| *k == BandKind::PinkNoise));
    axis.extend(
        plan.kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == BandKind::Tonal)
            .map(|(i, _)| i),
    );
    axis.extend(plan.kinds.iter().position(|k| *k == BandKind::WhiteNoise));
    debug_assert_eq!(axis.len(), plan.len(), "every band must sit on the frequency axis");
    axis
}

/// Keep only the `max_voices` strongest spectral peaks in `frame`; zero every
/// other band. `sounding` carries the previous frame's selection in and the
/// new one out, and must be `frame.len()` long.
///
/// # Why a filterbank has to be sparsified at all
///
/// A band bank gives every band whatever energy falls in its slot, so noise
/// floor, reverb tails, cymbal wash and the skirts of every real note all get
/// a speaker. Measured on a pop master at 96 bands, **94.40 of 96 bands were
/// non-zero in an average frame**. Ninety-four sine oscillators 0.85 semitones
/// apart sounding at once is not a chord, it is broadband noise, and the
/// previous fix -- which corrected a real 24 dB loudness deficit -- only made
/// that noise louder. Music is sparse; the render has to be too.
///
/// # Step 1: local maxima, not just "the loud ones"
///
/// A band qualifies only if it is strictly greater than both its neighbours
/// **on the frequency axis** ([`frequency_axis`]). Without this, one strong
/// note lights its own band and the two either side of it and burns three
/// voice slots on one note. The test is cheap and does real work: it cuts the
/// average frame from 94.40 candidates to **29.28**.
///
/// It is not, however, enough on its own -- 29 simultaneous oscillators is
/// still a wall. Top-N below is what actually sparsifies; the local-maximum
/// test is what makes the N slots land on N DIFFERENT notes. The two together
/// capture less raw energy than top-N alone at the same N (49.9% vs 57.9% of
/// frame power at N=12) and that gap is precisely the skirt energy belonging
/// to notes already selected -- paying for it twice is what the test prevents.
///
/// ## Endpoints
///
/// A band at either end of the axis has one neighbour, and qualifies if it
/// beats that one. It is NOT silently dropped: an axis endpoint is the single
/// loudest peak in the frame 5.5% of the time (a bass drop, a cymbal wash),
/// and excluding endpoints costs 5.7% of the captured power at N=12 (0.4986 ->
/// 0.4699) while barely sparsifying anything (29.28 candidates -> 27.64). It
/// buys nothing and silences the ends of the spectrum.
///
/// # Step 2: the prominence gate -- which candidates are NOTES
///
/// A local maximum is not yet a note. Broadband content -- cymbal wash,
/// sibilance, room tone, bass rumble -- has local maxima everywhere, and by
/// magnitude alone they are indistinguishable from a real partial. What
/// separates them is CONTRAST: see [`prominence`], and [`MIN_PROMINENCE`] for
/// the measurements that set the threshold.
///
/// **This step, not the ranking, is what sparsifies a real track**, and it is
/// the step whose absence made every voice count from 24 to 48 sound alike. The
/// local-maximum test leaves 29.02 candidates in an average frame, so a top-N
/// truncation at N=32 or N=48 never fires at all.
///
/// # Step 3: the top N by prominence * magnitude^k
///
/// Ranking is by [`prominence`] scaled by the band's own magnitude to the
/// [`RANK_MAGNITUDE_EXP`] power -- the measurements that chose that blend over
/// both a pure magnitude rank and a pure prominence rank are on the constant.
///
/// **Prominence decides WHICH bands sound; it never decides how loud they
/// are.** The volume written is always the band's own magnitude, untouched, so
/// the track's dynamics survive selection intact.
///
/// Captured share of each frame's power, averaged over a 3m57s pop master at
/// 96 bands, under the ORIGINAL magnitude-ranked, ungated scheme:
///
/// | N  | share | vs. N=6 |
/// |----|-------|---------|
/// | 6  | 38.6% | --       |
/// | 8  | 43.3% | +0.50 dB|
/// | 12 | 49.9% | +1.11 dB|
/// | 16 | 54.0% | +1.45 dB|
/// | 24 | 58.4% | +1.79 dB|
/// | all local maxima | 59.6% | +1.88 dB |
///
/// The knee is at 12, which is the default. Note what the last row means now
/// that it has been measured properly: "all local maxima" is only 29 bands, so
/// **`--max-voices` above about 29 is inert** -- the gate, not N, is what limits
/// a dense frame there.
///
/// # The noise bands compete; they are not exempt
///
/// White and pink are ranked by exactly the same rule as every tonal band. The
/// alternative -- letting them always through -- was measured and rejected: they
/// are non-zero in 99.4% of frames, so exempting them adds **1.99 permanent
/// voices**, a noise bed under the entire render, which is half of the
/// complaint this function exists to answer. Under competition at N=12 white
/// sounds in 69.0% of frames and pink in 18.6%: still often, but earned, and
/// each time at the cost of a tonal slot rather than on top of one. Excluding
/// them instead would silence cymbals, sibilance and sub-bass outright, since
/// no sine band covers that content.
///
/// # Pitch is never touched
///
/// Selection only decides which bands have a non-zero volume. Band pitches are
/// baked at build time and never written, so a band fading in and out carries
/// no retrigger risk -- `VolumeMultiplier` was verified in game to ramp a
/// running voice rather than restart it.
fn select_peaks(
    frame: &mut [f32],
    axis: &[usize],
    max_voices: usize,
    peak_gate: f32,
    sounding: &mut [bool],
) {
    // 0 is the documented escape hatch: no selection at all, every band
    // sounds, exactly the pre-selection behaviour. Kept reachable so an A/B
    // against this change is one flag away.
    if max_voices == 0 {
        sounding.fill(true);
        return;
    }
    let n = axis.len();
    // (band, prominence) for every candidate that survives both the
    // local-maximum test and the prominence gate.
    let mut peaks: Vec<(usize, f32)> = Vec::with_capacity(n);
    for i in 0..n {
        let b = axis[i];
        let v = frame[b];
        // Silence is never a peak, however its neighbours compare. Without
        // this a digitally-silent frame would "select" bands at volume 0 and
        // the run-length statistics would be measuring nothing.
        if v <= 0.0 {
            continue;
        }
        // `i == 0` / `i + 1 == n`: an endpoint has no neighbour on that side,
        // and a missing neighbour cannot outrank it. See the doc above.
        let over_lower = i == 0 || v > frame[axis[i - 1]];
        let over_upper = i + 1 == n || v > frame[axis[i + 1]];
        if !(over_lower && over_upper) {
            continue;
        }
        // A local maximum on a broad, flat lump is not a note. See
        // `DEFAULT_PEAK_GATE`: this gate is what limits a dense frame once
        // `max_voices` is above the ~29 local maxima a real track offers, and
        // it is `--peak-gate` precisely because of that.
        let p = prominence(frame, axis, i);
        if p >= peak_gate {
            peaks.push((b, p));
        }
    }

    // Rank by prominence scaled by the band's own magnitude, then keep the top
    // `max_voices`.
    //
    // The incumbent bonus is applied INSIDE the magnitude factor, not to the
    // finished key. `VOICE_HYSTERESIS` is a measured amplitude ratio (1.5x,
    // 3.5 dB) and multiplying a key that already carries `magnitude^0.5` by it
    // would silently demand 1.5^(1/0.5) = 2.25x, i.e. 7 dB, of a challenger --
    // doubling the constant's documented meaning without changing its value.
    //
    // Either way it biases RANK ONLY: no volume is ever scaled by it, and the
    // value written for a band is always its own magnitude.
    let key = |b: usize, p: f32| {
        let mag = if sounding[b] { frame[b] * VOICE_HYSTERESIS } else { frame[b] };
        p * mag.powf(RANK_MAGNITUDE_EXP)
    };
    peaks.sort_by(|&(a, pa), &(b, pb)| key(b, pb).total_cmp(&key(a, pa)));
    peaks.truncate(max_voices);

    let mut keep = vec![false; frame.len()];
    for &(b, _) in &peaks {
        keep[b] = true;
    }
    for (b, v) in frame.iter_mut().enumerate() {
        if !keep[b] {
            *v = 0.0;
        }
    }
    sounding.copy_from_slice(&keep);
}

/// Reject a `--subdiv` that does not put every band on a real semitone.
///
/// ONLY multiples of 12 do, and the failure of anything else is musical
/// rather than obvious -- so it is an error, not a warning.
///
/// A subdivision of `s` spaces bands `12/s` semitones apart, so a band lands
/// on a semitone only when `12/s` divides 1, i.e. when `s` is a multiple of
/// 12. Listened to in game on solo piano, one binary, only this flag
/// varying: 12 "was good"; 14 (0.857 semitones/step, no step on any note,
/// worst error 43 cents) "sharp/out of tune"; 18 (0.667, only the even
/// semitones) "also out of tune, flat??"; 24 (0.5, every note hit by every
/// second band) "okay, sounded a little weird". The sign of the error even
/// follows from the step size, which is why the two wrong ones were heard as
/// wrong in opposite directions.
///
/// This is also, retroactively, the original "sounds off key": the geometric
/// grid this bank replaced spaced bands 0.848 semitones apart, i.e. an
/// effective subdivision of 14.15 -- the same defect as `--subdiv 14`.
///
/// Its own function, rather than four lines inside [`analyze`], because the
/// GUI has to refuse the same value in its own widget and its cost readout
/// has to refuse it before it can count anything -- and a second copy of a
/// rule this expensive to learn is how a front end ends up accepting what the
/// renderer rejects.
pub fn check_subdiv(subdiv: u32) -> Result<(), String> {
    if subdiv == 0 || subdiv % 12 != 0 {
        return Err(format!(
            "--subdiv must be a multiple of 12, got {}: only then does every band land on a              real semitone. {} bands per octave spaces them {:.3} semitones apart, so the              whole render is pulled off pitch (and audibly so -- 14 was heard as sharp, 18              as flat). Use 12 (one per semitone, preferred), 24 (quarter-tones, for a source              not tuned to A440) or 36",
            subdiv,
            subdiv,
            12.0 / subdiv.max(1) as f32,
        ));
    }
    Ok(())
}

/// The [`BandPlan`] a set of options asks for: the one [`analyze`] builds, and
/// therefore the one anything estimating a bank-mode render must count.
///
/// `--bands` selects the SPAN of the equal-tempered grid; absent, the span is
/// everything the emitter's pitch range holds at `--subdiv`. See
/// [`AudioOptions::bands`].
pub fn band_plan(opts: &AudioOptions) -> Result<BandPlan, String> {
    check_subdiv(opts.subdiv)?;
    match opts.bands {
        Some(bands) => BandPlan::with_subdiv(bands, opts.noise_bands, opts.subdiv),
        None => BandPlan::full(opts.noise_bands, opts.subdiv),
    }
}

/// Decode, transform, fold, **select the sounding bands**, and normalise in
/// one streaming pass.
///
/// # Selection comes first
///
/// [`select_peaks`] runs on each folded frame before anything measures it.
/// That ordering is deliberate: the normaliser's track peak, each frame's own
/// peak, and the `floor_db` comparison must all be taken over the bands that
/// will actually sound. A band that lost the top-N competition contributing to
/// the scale, or to the floor the winners are measured against, would let
/// discarded energy quietly change the level of what is kept.
///
/// # Normalisation: the loudest per-frame INCOHERENT SUM
///
/// One global scale puts the largest per-frame `sqrt(sum of squares)` across
/// the bank at exactly `gain` (then clamped to 1.0). With `--leveling 0` that
/// is the ONLY scale applied, so every level relationship in the source
/// survives verbatim into the volumes. The shipped default is full leveling,
/// which adds a per-frame scale on top -- referenced to the same incoherent
/// mix, so it moves frames toward that ceiling and never through it. See
/// "# Leveling" below.
///
/// `sqrt(sum of squares)` is the physically correct peak for a bank of
/// UNCORRELATED sinusoids, which is what the emitters are: N bands at
/// unrelated frequencies neither add coherently (the plain sum) nor stay at
/// the height of the tallest (the single-band max). It sits between the two
/// and it is the number the game's mixer actually sees.
///
/// Both alternatives were shipped and both were wrong, in opposite directions:
///
/// * **The per-frame SUM of amplitudes.** Chosen to guarantee the bank could
///   never add past full scale. It costs a factor of `sum-peak /
///   max-band-peak` -- measured at 8.9x (32 bands) and 16.2x (96 bands) -- which
///   put the loudest instant of a whole track at 0.112 (-19 dB) and 0.062
///   (-24 dB). Inaudible in game, and reported as beeping rather than music.
/// * **The loudest single band.** The correction for the above, and an
///   overshoot: with the loudest band pinned to 1.0 the incoherent mix reaches
///   **1.575x** full scale at the loudest frame (measured, 96 bands,
///   `--max-voices 32`, `--window 16384`). That is the distortion reported at
///   `--gain 1.0`, and it is why `--gain 0.5` was needed to clean it up -- which
///   then gave away 6 dB everywhere else, so the same render was both distorted
///   and too quiet.
///
/// Targeting the incoherent sum lands the peak mix at exactly 1.000 and makes
/// `--gain 1.0` both clean and as loud as the medium allows. An individual
/// band is then no longer pinned to full scale -- six equal tones sit at
/// `1/sqrt(6)` each -- which is correct: they are six voices sharing one output.
///
/// **And the leveling boost is referenced to the same quantity**, which is the
/// only reason the bound above survives the shipped default. It did not
/// originally: the boost divided by the frame's loudest single BAND while the
/// scale targeted the frame MIX, so a frame with six bands sounding was handed
/// the boost a one-band frame had earned and the mix came out
/// `sqrt(N_this_frame / N_reference_frame)` too high -- **measured 2.447x full
/// scale at `--leveling 1.0`, the default**, against the 1.575x above that a
/// listener had already reported as distortion. Pinned by
/// `tests::no_frame_exceeds_full_scale_at_the_shipped_leveling_default`.
///
/// The sum is taken over the SELECTED, PRE-FLOOR magnitudes. Pre-floor so that
/// `floor_db` cannot move the scale around under the render (the property
/// [`the_floor_zeroes_exactly_what_is_below_it_in_amplitude_db`] pins); the
/// bands it removes are 60 dB down and contribute under 0.001% of the power, so
/// nothing measurable is bought by the alternative.
///
/// [`the_floor_zeroes_exactly_what_is_below_it_in_amplitude_db`]: self
///
/// # The floor is relative to the frame
///
/// `floor_db` zeroes bands more than that many dB below the loudest band **in
/// the same frame**, not below the track peak. An absolute floor is measured
/// against a scale that one loud transient sets for the whole track, so it
/// eats a fixed absolute amount out of every frame and takes proportionally
/// more of a quiet one -- in the limit it silences a quiet passage outright.
/// A frame-relative floor removes only what is masked by something louder the
/// listener is hearing at the same moment, so a quiet passage keeps exactly as
/// much detail as a loud one. Measured on the same track, the switch drops the
/// fraction of exactly-zero values from 1.86% to 0.62% (32 bands) and 3.29% to
/// 1.67% (96 bands), all of it recovered from the quiet frames.
///
/// Digital silence still reads exactly zero: a frame whose peak is 0 has a
/// floor of 0, and every band in it is already 0.
///
/// # Leveling
///
/// `leveling` interpolates toward per-frame automatic gain control: the frame's
/// scale is multiplied by `(loudest frame mix / this frame's mix) ^ leveling`,
/// capped at [`MAX_LEVELING_BOOST`]. At 0.0 nothing happens and the track keeps
/// its dynamics; at 1.0 -- **the shipped default**, chosen by ear over 0.5 and
/// over off on every source tried -- every frame is dragged to full scale and
/// the music is flattened into a wall. It is a knob rather than a boolean so
/// the useful middle (0.3 or so lifts the quietest 10% of frames from 0.19 to
/// 0.32 while clamping nothing) is reachable.
///
/// The ratio is taken between MIXES, not between the frames' loudest bands, and
/// that is what keeps full leveling inside full scale: `(mix_peak / mix) ^ l <=
/// mix_peak / mix` for every `l` in 0..=1, so `mix * scale <= gain` falls out
/// algebraically at every setting of the knob. It costs nothing musically -- a
/// frame is still lifted all the way to the ceiling at 1.0, which is what full
/// AGC means -- and it is a correction to the reference, not an attenuation.
pub fn analyze(
    source: &dyn AudioSource,
    opts: &AudioOptions,
    progress: &mut dyn Progress,
) -> Result<VoiceTrack, String> {
    // `!is_finite()` first: NaN fails EVERY comparison, so a bare
    // `opts.fps <= 0.0` lets `--audio-fps nan` straight through.
    if !opts.fps.is_finite() || opts.fps <= 0.0 {
        return Err(format!("--audio-fps must be a positive finite number, got {}", opts.fps));
    }
    if opts.max_frames == 0 {
        return Err("--max-frames must be at least 1".to_string());
    }
    // Same NaN-first reasoning as `fps`, and it matters more here: `f32::min`
    // IGNORES a NaN operand, so a NaN scale would sail through the 1.0 clamp
    // as 1.0 and write EVERY speaker at full volume for the whole track.
    if !opts.gain.is_finite() || opts.gain < 0.0 {
        return Err(format!("--gain must be a non-negative finite number, got {}", opts.gain));
    }
    if !opts.leveling.is_finite() || !(0.0..=1.0).contains(&opts.leveling) {
        return Err(format!(
            "--leveling must be between 0 (keep the track's dynamics) and 1 (flatten \
             every frame to full scale), got {}",
            opts.leveling
        ));
    }
    // Negative is meaningless (a negative time constant makes `1 - exp(x)`
    // negative, i.e. a follower that runs away from its target) and NaN
    // poisons every level it touches. 0 is legal and documented: no smoothing.
    for (flag, value) in [
        ("--attack", opts.attack_ms),
        ("--release", opts.release_ms),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "{flag} must be a non-negative finite number of milliseconds \
                 (0 = no smoothing), got {value}"
            ));
        }
    }
    // A prominence is a magnitude divided by a mean magnitude, so it is never
    // below 0 and a gate below 1.0 admits every local maximum exactly as 1.0
    // does -- but silently, which invites "--peak-gate 0.5 must be half as
    // strict as 1.0" when it is the same thing. NaN first, as everywhere else
    // here: it fails every comparison.
    if !opts.peak_gate.is_finite() || opts.peak_gate < 1.0 {
        return Err(format!(
            "--peak-gate must be at least 1.0 (a peak's magnitude as a multiple of its \
             neighbourhood's mean; 1.0 admits every local maximum, {DEFAULT_PEAK_GATE} is \
             the default), got {}",
            opts.peak_gate
        ));
    }
    let plan = band_plan(opts)?;
    let info = source.info();
    let hop = hop_for(info.sample_rate, opts.fps)?;

    // `frame_count_for` takes window and hop, not fps, so the hint matches
    // what `StftStream` actually emits -- this is the progress denominator.
    //
    // `Exact` rather than `Estimated` because `frame_count_for` is the very
    // recurrence `StftStream` steps: given a correct duration it IS the
    // emitted count, so the only slack is whatever the container's own
    // duration carries. What this DOES change is the `None` arm -- a source
    // that cannot report a duration used to get a bare, totalless spinner that
    // was indistinguishable from a hung one, and now says so. (And a duration
    // that turns out short cannot overflow the bar either -- see
    // `FrameTotal::position`.)
    let hint = info
        .duration_hint
        .map(|d| frame_count_for(d, info.sample_rate, opts.window, hop).min(opts.max_frames));
    FrameTotal::new(hint, None).begin(progress, "analyzing audio");

    let mut raw: Vec<Vec<f32>> = vec![Vec::new(); plan.len()];
    // Selection is per frame and reads across bands, so it happens HERE, on
    // the frame-major `folded` vector, before it is scattered into the
    // band-major store. Doing it afterwards would mean gathering a column out
    // of `plan.len()` separate Vecs for every one of millions of frames.
    let axis = frequency_axis(&plan);
    let mut sounding = vec![false; plan.len()];
    let collected: Result<usize, String> = (|| {
        let mut stft = StftStream::new(source.open()?, opts.window, hop)?;
        let mut n = 0usize;
        while n < opts.max_frames {
            let Some(spectrum) = stft.next_spectrum()? else { break };
            let mut folded = plan.fold(&spectrum, info.sample_rate, opts.window);
            // Before the peaks are found, so the normaliser sees only what
            // will actually sound: an unselected band must not set the scale
            // or the floor for the bands that beat it.
            select_peaks(
                &mut folded,
                &axis,
                opts.max_voices,
                opts.peak_gate,
                &mut sounding,
            );
            for (b, v) in folded.into_iter().enumerate() {
                raw[b].push(v);
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

    // The envelope, per band, over the whole track -- BEFORE anything measures
    // the result. It changes every level that will be written, so the
    // normaliser's track peak, each frame's own peak and the floor comparison
    // must all see the smoothed values or the render is normalised against
    // levels nothing plays. Band-major storage makes this a walk down each row.
    let envelope = Envelope::new(opts.attack_ms, opts.release_ms, opts.fps);
    for band in raw.iter_mut() {
        envelope.apply(band);
    }

    // Three statistics in one pass over the band-major store, so no
    // frame-major copy is ever materialised:
    //
    // * `frame_peak[f]` -- the loudest band within each frame. What the floor
    //   is measured against.
    // * `frame_mix[f]` -- that frame's `sqrt(sum of squares)`, i.e. what the
    //   whole bank sums to at that instant. What leveling divides by.
    // * `mix_peak` -- the largest `frame_mix`. THE NORMALISER: see this
    //   function's doc for why it is the incoherent sum and not the loudest
    //   single band or the plain sum.
    let mut mix_peak = 0.0f64;
    let mut frame_peak = vec![0.0f32; frame_count];
    let mut frame_mix = vec![0.0f64; frame_count];
    for (f, fp) in frame_peak.iter_mut().enumerate() {
        // f64: 96 bands of squared magnitudes, and an f32 running total loses
        // the small terms once it passes ~1.7e7 -- the same reason
        // `BandPlan::fold` accumulates in f64.
        let mut sq = 0.0f64;
        for band in &raw {
            let v = band[f];
            if v > *fp {
                *fp = v;
            }
            sq += v as f64 * v as f64;
        }
        let mix = sq.sqrt();
        frame_mix[f] = mix;
        if mix > mix_peak {
            mix_peak = mix;
        }
    }

    let floor_ratio = 10f32.powf(opts.floor_db / 20.0);
    // The loudest incoherent mix in the track lands on exactly `gain`.
    let base = if mix_peak > 0.0 { opts.gain / mix_peak as f32 } else { 0.0 };
    let scale: Vec<f32> = frame_mix
        .iter()
        .map(|&mix| {
            if opts.leveling <= 0.0 || mix <= 0.0 {
                base
            } else {
                base * ((mix_peak / mix) as f32).powf(opts.leveling).min(MAX_LEVELING_BOOST)
            }
        })
        .collect();

    let volumes: Vec<Vec<f64>> = raw
        .into_iter()
        .map(|band| {
            band.into_iter()
                .enumerate()
                .map(|(f, v)| {
                    // Below the floor is exactly zero, not the floor itself:
                    // a floor across every band is audible hiss under
                    // everything, and silence must be silent. Compared before
                    // scaling, against this frame's own peak -- the ratio is
                    // what `floor_db` names, and it is scale-free, so
                    // leveling cannot move the floor around under it.
                    if v < frame_peak[f] * floor_ratio {
                        0.0
                    } else {
                        (v * scale[f]).min(1.0) as f64
                    }
                })
                .collect()
        })
        .collect();

    Ok(VoiceTrack { plan, volumes, fps: opts.fps, frame_count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::source::SampleClip;
    use crate::progress::{NoProgress, Progress};
    use std::f32::consts::TAU;

    const SR: u32 = 48_000;

    /// The baseline for every test below, and it deliberately differs from
    /// `AudioOptions::default()` in exactly two ways.
    ///
    /// * **Two noise bands.** The shipping default is 0 -- every source tried
    ///   in game sounded worse with them -- but these tests are about how the
    ///   bank folds, ranks and normalises, and half of them are written against
    ///   a frequency axis with both ends present. The DEFAULT has its own test;
    ///   these have the mechanism.
    /// * **No envelope.** `--attack`/`--release` shape how a level MOVES
    ///   between frames, which is a separate mechanism with its own tests
    ///   below. Leaving it on here would put a release tail into every
    ///   measurement of selection, of the floor and of the dynamics: a 20 dB
    ///   step down takes ~10 frames to clear at the default 150 ms, so a test
    ///   asserting a quiet passage stays quiet would be measuring the loud
    ///   passage's tail instead.
    ///
    /// **The `leveling: 0.0` here is a third difference and it is a trap.** It
    /// is right for the mechanism tests -- an AGC on top of them would be a
    /// second variable -- but the normalisation GUARDS used to inherit it too,
    /// which meant the suite tested every value of the knob except the 1.0 that
    /// ships, and a 2.447x overshoot lived under them. Anything asserting a
    /// bound on the output must set `leveling` explicitly: see
    /// [`the_incoherent_mix_never_exceeds_full_scale`] and
    /// [`no_frame_exceeds_full_scale_at_the_shipped_leveling_default`].
    fn opts() -> AudioOptions {
        AudioOptions {
            noise_bands: 2,
            attack_ms: 0.0,
            release_ms: 0.0,
            leveling: 0.0,
            ..AudioOptions::default()
        }
    }

    fn sine(freq: f32, secs: f32) -> SampleClip {
        let n = (SR as f32 * secs) as usize;
        SampleClip::new(SR, (0..n).map(|i| (TAU * freq * i as f32 / SR as f32).sin()).collect())
    }

    #[test]
    fn a_track_is_band_major_and_rectangular() {
        let t = analyze(&sine(1000.0, 2.0), &opts(), &mut NoProgress).expect("analyze");
        assert_eq!(t.volumes.len(), t.plan.len(), "one array per band");
        // The derived default: 79 equal-tempered semitones plus white and pink.
        assert_eq!(t.plan.len(), 81, "an absent --bands must fill the scale");
        for (b, v) in t.volumes.iter().enumerate() {
            assert_eq!(v.len(), t.frame_count, "band {b} must cover every frame");
        }
        assert!(t.frame_count > 0);
    }

    /// 2 seconds at 30 fps is about 60 frames. The trailing partial window is
    /// dropped, so a small shortfall is expected and a large one is a bug.
    #[test]
    fn frame_count_tracks_duration_and_rate() {
        let t = analyze(&sine(1000.0, 2.0), &opts(), &mut NoProgress).expect("analyze");
        assert!(
            (58..=60).contains(&t.frame_count),
            "2s at 30fps should be ~60 frames, got {}",
            t.frame_count
        );
    }

    /// Every value must be a legal `VolumeMultiplier`. A NaN or a negative
    /// would be written straight into the save.
    #[test]
    fn every_volume_is_finite_and_in_range() {
        let t = analyze(&sine(1000.0, 1.0), &opts(), &mut NoProgress).expect("analyze");
        for (b, band) in t.volumes.iter().enumerate() {
            for (f, &v) in band.iter().enumerate() {
                assert!(v.is_finite(), "band {b} frame {f} is not finite: {v}");
                assert!((0.0..=1.0).contains(&v), "band {b} frame {f} out of range: {v}");
            }
        }
    }

    /// Silence must produce exactly zero, not a -60 dB floor. A floor across
    /// 32 speakers is audible hiss under everything.
    #[test]
    fn silence_produces_exactly_zero_everywhere() {
        let clip = SampleClip::new(SR, vec![0.0; SR as usize]);
        let t = analyze(&clip, &opts(), &mut NoProgress).expect("analyze");
        for band in &t.volumes {
            for &v in band {
                assert_eq!(v, 0.0, "silence must be exactly zero, got {v}");
            }
        }
    }

    /// Normalisation must make the loudest moment reach full scale,
    /// otherwise every render comes out needlessly quiet.
    ///
    /// A single sine cannot check this: all the energy is in one band, so the
    /// frame sum and the frame max are the same number and every candidate
    /// normaliser agrees. The interesting case is energy SPREAD across bands,
    /// which is what music is, and it must be checked at more than one band
    /// count because the old sum normaliser's error grew with the count.
    /// See [`the_loudest_value_in_a_track_reaches_full_scale`], which is the
    /// version with teeth; this one stays as the single-tone smoke test.
    #[test]
    fn normalisation_reaches_full_scale() {
        let t = analyze(&sine(1000.0, 1.0), &opts(), &mut NoProgress).expect("analyze");
        let peak = t
            .volumes
            .iter()
            .flat_map(|b| b.iter())
            .fold(0.0f64, |a, &b| a.max(b));
        assert!(peak > 0.5, "the loudest band should approach full scale, got {peak}");
    }

    #[test]
    fn gain_scales_the_result_and_still_clamps() {
        let mut o = opts();
        o.gain = 100.0;
        let t = analyze(&sine(1000.0, 1.0), &o, &mut NoProgress).expect("analyze");
        let peak = t.volumes.iter().flat_map(|b| b.iter()).fold(0.0f64, |a, &b| a.max(b));
        assert!(peak <= 1.0, "gain must not push volumes past 1.0, got {peak}");
    }

    /// Cancellation must stop the analysis early rather than draining the
    /// whole source.
    #[test]
    fn cancellation_stops_analysis_early() {
        struct CancelAfter {
            n: u64,
            seen: std::cell::Cell<u64>,
        }
        impl Progress for CancelAfter {
            fn begin(&mut self, _l: &str, _t: Option<u64>) {}
            fn tick(&mut self, n: u64) {
                self.seen.set(n);
            }
            fn finish(&mut self) {}
            fn is_cancelled(&self) -> bool {
                self.seen.get() >= self.n
            }
        }
        let mut p = CancelAfter { n: 5, seen: std::cell::Cell::new(0) };
        let t = analyze(&sine(1000.0, 10.0), &opts(), &mut p).expect("analyze");
        assert!(
            t.frame_count < 30,
            "cancelling after 5 frames must not analyse a 10s clip ({} frames)",
            t.frame_count
        );
    }

    #[test]
    fn a_source_with_no_frames_is_an_error() {
        let clip = SampleClip::new(SR, vec![0.0; 10]);
        assert!(analyze(&clip, &opts(), &mut NoProgress).is_err());
    }

    /// `max_frames` bounds the track regardless of source length -- the same
    /// OOM guard the video path carries.
    #[test]
    fn max_frames_bounds_the_track() {
        let mut o = opts();
        o.max_frames = 10;
        let t = analyze(&sine(1000.0, 10.0), &o, &mut NoProgress).expect("analyze");
        assert_eq!(t.frame_count, 10);
    }

    // --- Tests added beyond the task brief. A mutation campaign proved each
    // --- of these properties was completely unprotected, even though the
    // --- module's own doc comments state every one of them. All the brief's
    // --- signals are single constant-amplitude sines, which is exactly the
    // --- one shape that cannot distinguish these mutations.

    /// A loud second followed by a second at `quiet` times the amplitude, on
    /// a band wide enough to hold one FFT bin comfortably. Window 4096 / hop
    /// 1600 puts frames 0..=27 wholly in the loud half and 30.. wholly in the
    /// quiet half.
    fn loud_then_quiet(quiet: f32) -> SampleClip {
        let f = band_centres(6)[4];
        let n = SR as usize * 2;
        SampleClip::new(
            SR,
            (0..n)
                .map(|i| {
                    let t = i as f32 / SR as f32;
                    let a = if i < SR as usize { 1.0 } else { quiet };
                    a * (TAU * f * t).sin()
                })
                .collect(),
        )
    }

    /// The loudest volume in `lo..=hi`.
    fn peak_in(t: &VoiceTrack, lo: usize, hi: usize) -> f64 {
        (lo..=hi).flat_map(|f| t.volumes.iter().map(move |b| b[f])).fold(0.0f64, f64::max)
    }

    /// Leveling is FULL by default, and the boost that makes that safe is
    /// capped.
    ///
    /// It shipped opt-in first, on the reasoning that a track arriving with
    /// dynamics should leave with them. Listening said otherwise on every
    /// source tried -- piano, speech and a music box -- and the reasoning is
    /// wrong for this medium: a bank of sine emitters has a fraction of a
    /// master's usable range, so what a listener hears without leveling is not
    /// the source's dynamics but a render that disappears for half its length.
    ///
    /// The cap is asserted alongside because it is what stops the default from
    /// being an amplified noise floor between the notes.
    #[test]
    fn leveling_is_full_by_default_and_the_boost_is_capped() {
        assert_eq!(
            AudioOptions::default().leveling,
            1.0,
            "per-frame leveling was measured better than 0.5 and better than off on              every source; the default follows the listening, not the theory"
        );
        assert!(
            MAX_LEVELING_BOOST > 1.0 && MAX_LEVELING_BOOST <= 20.0,
            "full leveling divides by the frame's own peak, so the cap is the only thing              between a near-silent frame and a full-scale noise floor: {MAX_LEVELING_BOOST}"
        );
    }

    /// Full leveling exists, is reachable, and does what it says: every frame
    /// dragged to full scale. This is the automatic gain control that
    /// [`a_quiet_passage_stays_quiet_relative_to_a_loud_one`] forbids by
    /// default -- the pair is what makes "opt-in" mean something, since a test
    /// that only forbids AGC is also passed by a build where the knob is
    /// wired to nothing.
    #[test]
    fn full_leveling_is_opt_in_and_flattens_the_dynamics() {
        let clip = loud_then_quiet(0.1);
        let mut o = opts();
        o.leveling = 1.0;
        let t = analyze(&clip, &o, &mut NoProgress).expect("analyze");
        let loud = peak_in(&t, 0, 27);
        let quiet = peak_in(&t, 30, t.frame_count - 1);
        assert!(
            quiet / loud > 0.9,
            "--leveling 1 must pull the quiet half up to the loud half (loud {loud}, \
             quiet {quiet})"
        );
    }

    /// The leveling boost is capped. Uncapped, `leveling = 1` divides by the
    /// frame's own peak, so a near-silent frame is multiplied by whatever
    /// number that takes and the decoder's noise floor arrives at full scale
    /// between the notes.
    #[test]
    fn the_leveling_boost_is_capped() {
        // 40 dB down: a 100x boost, which the cap must cut to 10x.
        let clip = loud_then_quiet(0.01);
        let mut o = opts();
        o.leveling = 1.0;
        let t = analyze(&clip, &o, &mut NoProgress).expect("analyze");
        let loud = peak_in(&t, 0, 27);
        let quiet = peak_in(&t, 30, t.frame_count - 1);
        let boost = (quiet / loud) / 0.01;
        assert!(
            boost <= MAX_LEVELING_BOOST as f64 + 0.5,
            "a 40 dB-down passage must not be boosted past {MAX_LEVELING_BOOST}x, got \
             {boost}x (loud {loud}, quiet {quiet})"
        );
        assert!(boost > 5.0, "the cap must not disable leveling altogether, got {boost}x");
    }

    /// A `--leveling` outside 0..=1 is a user error, not a silently clamped
    /// or wildly-scaled render. NaN included: it fails every comparison, so a
    /// bare range check written the other way round lets it through.
    #[test]
    fn an_out_of_range_leveling_is_an_error() {
        for bad in [-0.1f32, 1.5, f32::NAN, f32::INFINITY] {
            let mut o = opts();
            o.leveling = bad;
            let Err(err) = analyze(&sine(1000.0, 1.0), &o, &mut NoProgress) else {
                panic!("--leveling {bad} must be rejected")
            };
            assert!(err.contains("--leveling"), "the error must name the flag: {err}");
        }
    }

    /// **`--gain` is the level the normalisation lands ON, and until this test
    /// nothing in the suite could tell.** Pinning `opts.gain` to a constant
    /// anywhere in `analyze` broke no test: every other test either uses the
    /// default 1.0 or measures a RATIO, both of which survive the flag being
    /// wired to nothing.
    ///
    /// Asserted at both ends of `--leveling` because the two compose: the
    /// global scale is `gain / mix_peak` and the per-frame boost multiplies it,
    /// so a leveling that referenced the wrong quantity would show up here as a
    /// mix that misses `gain` at 1.0 while hitting it at 0.
    #[test]
    fn gain_is_the_level_the_loudest_mix_lands_on() {
        let clip = tones(&band_centres(6), &[1.0 / 6.0; 6], 1.0);
        for leveling in [0.0, AudioOptions::default().leveling] {
            for gain in [0.125, 0.25, 0.5, 1.0] {
                let o = AudioOptions { gain, leveling, ..opts() };
                let t = analyze(&clip, &o, &mut NoProgress).expect("analyze");
                let mix = (0..t.frame_count)
                    .map(|f| t.volumes.iter().map(|b| b[f] * b[f]).sum::<f64>().sqrt())
                    .fold(0.0f64, f64::max);
                assert!(
                    (mix - gain as f64).abs() < 1e-3,
                    "--gain {gain} at --leveling {leveling} must put the loudest incoherent \
                     mix on exactly {gain}, got {mix}"
                );
            }
        }
        // ...and it scales the WHOLE render, not just its loudest instant: a
        // scale that pinned only the peak would leave every quieter frame at
        // the wrong level and still pass above.
        let full = analyze(&clip, &opts(), &mut NoProgress).expect("analyze");
        let half = analyze(&clip, &AudioOptions { gain: 0.5, ..opts() }, &mut NoProgress)
            .expect("analyze");
        for (b, row) in full.volumes.iter().enumerate() {
            for (f, &loud) in row.iter().enumerate() {
                let quiet = half.volumes[b][f];
                assert!(
                    (quiet - loud * 0.5).abs() < 1e-9,
                    "band {b} frame {f}: --gain 0.5 must be exactly half of --gain 1.0 \
                     ({quiet} vs {loud})"
                );
            }
        }
        // ...and 0 is silence, not "no scaling applied".
        let o = AudioOptions { gain: 0.0, ..opts() };
        let t = analyze(&clip, &o, &mut NoProgress).expect("analyze");
        assert!(
            t.volumes.iter().flat_map(|b| b.iter()).all(|&v| v == 0.0),
            "--gain 0 must write silence"
        );
    }

    /// `--gain nan` must be an error rather than the LOUDEST possible render.
    /// `f32::min` IGNORES a NaN operand, so a NaN scale walks through the 1.0
    /// clamp as 1.0 and writes every speaker at full volume for the whole
    /// track -- finite, in range, and deafening.
    #[test]
    fn a_non_finite_or_negative_gain_is_an_error() {
        for bad in [f32::NAN, f32::INFINITY, -1.0] {
            let mut o = opts();
            o.gain = bad;
            let Err(err) = analyze(&sine(1000.0, 1.0), &o, &mut NoProgress) else {
                panic!("--gain {bad} must be rejected")
            };
            assert!(err.contains("--gain"), "the error must name the flag: {err}");
        }
    }

    /// The frequencies of `n` bands, spread across the bank.
    fn band_centres(n: usize) -> Vec<f32> {
        let p = BandPlan::new(32, 2).expect("the default plan");
        (0..n).map(|i| p.pitches[8 + i * 4] * crate::audio::bands::BASE_HZ).collect()
    }

    fn tones(freqs: &[f32], amps: &[f32], secs: f32) -> SampleClip {
        let n = (SR as f32 * secs) as usize;
        SampleClip::new(
            SR,
            (0..n)
                .map(|i| {
                    let t = i as f32 / SR as f32;
                    freqs
                        .iter()
                        .zip(amps)
                        .map(|(f, a)| a * (TAU * f * t).sin())
                        .sum()
                })
                .collect(),
        )
    }

    fn frame_sum(t: &VoiceTrack, f: usize) -> f64 {
        t.volumes.iter().map(|b| b[f]).sum()
    }

    /// The largest per-frame INCOHERENT SUM in a track must land on exactly
    /// full scale, whatever the band count -- no lower (the 19-24 dB deficit
    /// this test was written to catch) and no higher (the 1.575x overshoot that
    /// replaced it, heard as distortion at `--gain 1.0`).
    ///
    /// Six equal tones is the shape that separates all three normalisers. Each
    /// band lands at:
    ///
    /// | normaliser                | per band | peak mix |
    /// |---------------------------|----------|----------|
    /// | per-frame SUM (original)  | 0.167    | 0.408    |
    /// | loudest single band       | 1.0      | 2.449    |
    /// | incoherent sum (current)  | 0.408    | 1.0      |
    ///
    /// so a check on the mix peak alone would be passed by a build that scaled
    /// everything by a constant, and a check on the per-band value alone was
    /// what let the overshoot ship. Both are asserted.
    #[test]
    fn the_loudest_incoherent_mix_in_a_track_reaches_exactly_full_scale() {
        let freqs = band_centres(6);
        let clip = tones(&freqs, &[1.0 / 6.0; 6], 1.0);
        // Swept, because the original normaliser's error grew with the band
        // count: a check at one count could be passed by a scheme that is
        // still 1/N quiet at another.
        for bands in [8, 32, 81] {
            let o = AudioOptions { bands: Some(bands), ..opts() };
            let t = analyze(&clip, &o, &mut NoProgress).expect("analyze");
            let mix = (0..t.frame_count)
                .map(|f| t.volumes.iter().map(|b| b[f] * b[f]).sum::<f64>().sqrt())
                .fold(0.0f64, f64::max);
            assert!(
                (mix - 1.0).abs() < 1e-3,
                "at {bands} bands the loudest incoherent mix must be exactly full scale, got \
                 {mix} -- below 1 is the 19-24 dB deficit that normalising by the per-frame \
                 SUM of all bands costs, above 1 is the overshoot that normalising by the \
                 loudest single band causes, and both were shipped"
            );
            // ...and the individual bands really are SHARING that mix, rather
            // than one of them holding it alone. Only checked where the six
            // tones land in six distinct bands: at 8 bands the bank is 6 tonal
            // bands spanning 370..494 Hz, so most of the tones fold onto a
            // noise band and there is no "per tone" level to speak of.
            if bands >= 32 {
                let peak =
                    t.volumes.iter().flat_map(|b| b.iter()).fold(0.0f64, |a, &b| a.max(b));
                let expected = 1.0 / 6f64.sqrt();
                assert!(
                    (peak - expected).abs() < 0.05,
                    "six equal tones must each sit at 1/sqrt(6) = {expected:.3} of full scale \
                     at {bands} bands, got {peak} -- 1/6 = 0.167 is the per-frame-sum defect, \
                     1.0 is the single-band overshoot, and a lone band at 1.0 with the other \
                     five silent would pass the mix check above"
                );
            }
        }
    }

    /// **The property that makes `--gain 1.0` usable.** No frame's incoherent
    /// mix may exceed full scale, on a signal with energy spread across many
    /// bands at once.
    ///
    /// The previous normaliser pinned the loudest single BAND to 1.0, which let
    /// the mix of everything sounding beside it reach a measured 1.575x on a
    /// real track -- the reported distortion. Nothing asserted a bound on the
    /// mix, so it shipped.
    ///
    /// **Swept over `leveling`, and that sweep is not decoration.** This guard
    /// used to run through [`opts`] alone, which pins `leveling` to 0 -- so it
    /// tested every value of the knob except the one that ships, and a second
    /// overshoot (2.447x, [`no_frame_exceeds_full_scale_at_the_shipped_leveling_default`])
    /// shipped underneath it for exactly that reason. The default is included
    /// explicitly rather than spelled as a literal so the sweep follows the
    /// default if it ever moves again.
    #[test]
    fn the_incoherent_mix_never_exceeds_full_scale() {
        let freqs = band_centres(6);
        for bands in [8, 32, 81] {
            for leveling in [0.0, 0.5, AudioOptions::default().leveling] {
                let o = AudioOptions { bands: Some(bands), leveling, ..opts() };
                let t = analyze(&tones(&freqs, &[1.0 / 6.0; 6], 1.0), &o, &mut NoProgress)
                    .expect("analyze");
                for f in 0..t.frame_count {
                    let mix = t.volumes.iter().map(|b| b[f] * b[f]).sum::<f64>().sqrt();
                    assert!(
                        mix <= 1.0 + 1e-6,
                        "at {bands} bands and --leveling {leveling}, frame {f} mixes to {mix}x \
                         full scale -- N uncorrelated sinusoids peak at sqrt(sum of squares), \
                         and anything above 1.0 there is distortion the listener hears at \
                         --gain 1.0"
                    );
                }
            }
        }
    }

    /// A loud passage carried by ONE partial, then a quieter, denser one.
    ///
    /// The shape that separates a per-frame scale referenced to the frame's
    /// loudest BAND from one referenced to the frame's own MIX: the two frames
    /// have the same loudest band relative to the track (so the first scheme
    /// hands the second passage the full `peak/frame_peak` boost) but wildly
    /// different band counts (so that boost multiplies an incoherent sum of six
    /// where the reference frame summed one).
    fn one_partial_then_six(loud: f32, quiet: f32) -> SampleClip {
        // A4, C#5, E5, A5, C#6, E6 -- an A major triad over two octaves, so
        // every tone lands in a band of its own at the default subdivision.
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

    /// The loudest per-frame incoherent mix in a track, as a multiple of full
    /// scale, with the frame it happened on. The measurement the overshoot was
    /// reported with.
    fn worst_mix(t: &VoiceTrack) -> (f64, usize) {
        (0..t.frame_count)
            .map(|f| (t.volumes.iter().map(|b| b[f] * b[f]).sum::<f64>().sqrt(), f))
            .fold((0.0, 0), |a, b| if b.0 > a.0 { b } else { a })
    }

    /// **The regression test for the leveling overshoot, on the SHIPPED
    /// defaults and nothing else.**
    ///
    /// Everything above runs through [`opts`], which pins `leveling` to 0 for
    /// reasons that are good for the mechanism tests and fatal for this one:
    /// with the knob at 0 the per-frame scale is the global scale and the bound
    /// is trivial. At the shipped `leveling: 1.0` it was not: measured **2.447x**
    /// full scale on this fixture, against the 1.575x the incoherent-sum
    /// normaliser was written to remove and which a listener reported as
    /// distortion. The owner independently described a real render as "super
    /// crunchy"; this is that.
    ///
    /// The cause was that the two decisions referenced different quantities.
    /// The global scale targets the loudest per-frame incoherent MIX; the
    /// leveling boost divided by the frame's loudest single BAND. A frame with
    /// six bands sounding therefore got the boost a one-band frame earned, and
    /// the mix came out `sqrt(N_quiet / N_loud)` too high -- unbounded by
    /// anything but [`MAX_LEVELING_BOOST`]. Referencing both to the frame's own
    /// mix makes `mix[f] * scale[f] <= gain` algebraically true for every
    /// `leveling` in 0..=1, which is what this asserts.
    #[test]
    fn no_frame_exceeds_full_scale_at_the_shipped_leveling_default() {
        let clip = one_partial_then_six(0.9, 0.09);
        let d = AudioOptions::default();
        assert_eq!(d.leveling, 1.0, "this test exists to cover the SHIPPED default");
        // Measured for every value first and asserted afterwards, so a failure
        // prints the whole curve rather than stopping at the first overshoot.
        let measured: Vec<(f32, f64, usize)> = [0.0, 0.25, 0.5, 0.75, d.leveling]
            .into_iter()
            .map(|leveling| {
                let o = AudioOptions { leveling, ..AudioOptions::default() };
                let t = analyze(&clip, &o, &mut NoProgress).expect("analyze");
                let (mix, frame) = worst_mix(&t);
                println!("bank --leveling {leveling}: worst mix {mix:.3}x at frame {frame}");
                (leveling, mix, frame)
            })
            .collect();
        for (leveling, mix, frame) in measured {
            assert!(
                mix <= 1.0 + 1e-6,
                "--leveling {leveling} mixes frame {frame} to {mix}x full scale. The bank \
                 sums incoherently in the game's mixer, so anything above 1.0 clips -- and \
                 the fix is not to turn the leveling down, it is to reference its boost to \
                 the same quantity the global scale targets"
            );
        }
    }

    /// The other half of the pair: bounding the mix must not have been bought
    /// by making full leveling quiet. Every frame that has any content at all
    /// must still arrive AT full scale when the knob is at 1.0 -- that is what
    /// "full automatic gain control" means, and it is the loudness the default
    /// was chosen by ear for.
    #[test]
    fn full_leveling_still_takes_every_frame_all_the_way_to_full_scale() {
        let clip = one_partial_then_six(0.9, 0.09);
        let o = AudioOptions { leveling: 1.0, ..AudioOptions::default() };
        let t = analyze(&clip, &o, &mut NoProgress).expect("analyze");
        // Skip the frames straddling the 2 s boundary: they are half of each
        // passage and belong to neither.
        let mixes: Vec<f64> = (0..t.frame_count)
            .map(|f| t.volumes.iter().map(|b| b[f] * b[f]).sum::<f64>().sqrt())
            .collect();
        let quiet_half: Vec<f64> = mixes[t.frame_count * 2 / 3..].to_vec();
        let worst = quiet_half.iter().cloned().fold(1.0f64, f64::min);
        assert!(
            worst > 0.9,
            "the quiet half must still be dragged to full scale by --leveling 1 (worst \
             frame mix {worst}) -- a fix that bounds the mix by attenuating is the \
             loudness regression the bound was supposed to avoid"
        );
    }

    /// **Replaces `the_whole_bank_together_never_exceeds_full_scale`**, which
    /// asserted the OLD contract: that the sum across every band never passed
    /// 1.0. That guarantee is what made the render inaudible -- it costs a
    /// factor of `frame-sum-peak / loudest-band`, measured at 8.9x (32 bands)
    /// and 16.2x (96 bands) on a real track -- and it modelled the mix wrongly,
    /// since bands at unrelated frequencies do not add coherently. The old
    /// assertion is deliberately INVERTED here rather than deleted, so the
    /// reversal is visible in the suite instead of being a silent absence.
    #[test]
    fn many_bands_sounding_at_once_do_not_scale_the_bank_down() {
        let freqs = band_centres(6);
        let t = analyze(&tones(&freqs, &[1.0 / 6.0; 6], 1.0), &opts(), &mut NoProgress)
            .expect("analyze");
        let peak_sum = (0..t.frame_count).map(|f| frame_sum(&t, f)).fold(0.0f64, f64::max);
        assert!(
            peak_sum > 1.0,
            "six equal tones must put six speakers near full scale at once (bank sum \
             {peak_sum}) -- a bank sum pinned at 1.0 means each band was scaled down by \
             the number of bands sounding, which is the defect this replaced"
        );
        // The mix is still bounded: N incoherent bands peak near
        // `sqrt(sum of squares)`, not the plain sum, and every individual
        // speaker is still a legal 0..=1 volume.
        let peak = t.volumes.iter().flat_map(|b| b.iter()).fold(0.0f64, |a, &b| a.max(b));
        assert!(peak <= 1.0, "no single speaker may exceed full scale, got {peak}");
    }

    /// Normalisation is across the WHOLE TRACK, not per frame. A per-frame
    /// scale is an automatic gain control: it drags every quiet passage up to
    /// full scale and flattens the track's dynamics into a wall of noise,
    /// with every value still finite, in range, and summing to 1.0.
    #[test]
    fn a_quiet_passage_stays_quiet_relative_to_a_loud_one() {
        // `loud_then_quiet` puts the tone on a high band, whose width
        // comfortably exceeds one FFT bin. The bottom bands are narrower than
        // the 11.7 Hz bin spacing, so a low tone smears across several of them
        // and no single band approaches full scale -- nothing to do with
        // normalisation.
        let t = analyze(&loud_then_quiet(0.1), &opts(), &mut NoProgress).expect("analyze");
        // window 4096 / hop 1600: frames 0..=27 lie wholly in the loud half,
        // frames 30.. wholly in the quiet half.
        assert!(t.frame_count >= 58, "need both halves, got {}", t.frame_count);
        let loud = peak_in(&t, 0, 27);
        let quiet = peak_in(&t, 30, t.frame_count - 1);
        assert!(loud > 0.3, "the loud half must carry real level, got {loud}");
        // One scale for the whole track means the 10:1 amplitude step in the
        // source survives verbatim into the volumes. A per-frame scale
        // collapses it to 1:1.
        let ratio = quiet / loud;
        assert!(
            (0.05..0.2).contains(&ratio),
            "the quiet half is 1/10 the amplitude of the loud half and must stay that way \
             (loud {loud}, quiet {quiet}, ratio {ratio}) -- a per-frame scale is an automatic \
             gain control that normalises the dynamics away"
        );
    }

    /// The floor zeroes everything more than `floor_db` below THE LOUDEST
    /// BAND IN THE SAME FRAME, and `floor_db` is an AMPLITUDE ratio:
    /// `10^(dB/20)`, not `10^(dB/10)`.
    ///
    /// Digital silence cannot test this at all -- it normalises to zero before
    /// the floor is ever consulted -- so the silence test passes with the floor
    /// deleted outright, or with the exponent an octave wrong. Here a loud
    /// 220 Hz tone carries a companion 80 dB below it: correct code zeroes the
    /// companion, `10^(dB/10)` leaves it ringing.
    ///
    /// The reference changed from the TRACK peak to the FRAME peak when
    /// normalisation stopped being a per-frame-sum scale; this signal is one
    /// steady level throughout, so the two references coincide here and the
    /// exponent and the zeroing are what is under test. The frame-vs-track
    /// distinction is [`the_floor_is_measured_against_the_frame_not_the_track`].
    #[test]
    fn the_floor_zeroes_exactly_what_is_below_it_in_amplitude_db() {
        let n = SR as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                (TAU * 220.0 * t).sin() + 1e-4 * (TAU * 3000.0 * t).sin()
            })
            .collect();
        let clip = SampleClip::new(SR, samples);
        let floored = analyze(&clip, &opts(), &mut NoProgress).expect("analyze");
        // The floor is a comparison against the frame peak and does not touch
        // the scale, so an effectively-disabled floor gives the same values
        // unfloored -- the oracle for what the floor should have removed.
        let mut open = opts();
        open.floor_db = -400.0;
        let unfloored = analyze(&clip, &open, &mut NoProgress).expect("analyze");

        let ratio = 10f64.powf(-60.0 / 20.0);
        let mut below = 0usize;
        let mut above = 0usize;
        for f in 0..unfloored.frame_count {
            // Normalisation is one global scale, so the frame's peak VOLUME
            // is the frame's peak magnitude in the same units the values are
            // in -- the floor for this frame is that times the ratio.
            let peak = unfloored.volumes.iter().map(|b| b[f]).fold(0.0f64, f64::max);
            let floor = peak * ratio;
            for b in 0..unfloored.volumes.len() {
                let raw = unfloored.volumes[b][f];
                let got = floored.volumes[b][f];
                if raw < floor {
                    below += 1;
                    assert_eq!(got, 0.0, "band {b} frame {f}: {raw} is more than 60 dB below \
                        this frame's peak {peak} (floor {floor}) and must be exactly zero, \
                        got {got}");
                } else {
                    above += 1;
                    assert_eq!(got, raw, "band {b} frame {f}: {raw} is above the floor and \
                        must pass through untouched, got {got}");
                }
            }
        }
        // Non-vacuous in both directions.
        assert!(below > 0, "the signal must put something below the floor");
        assert!(above > 0, "the signal must put something above the floor");
    }

    /// The floor is measured against the frame the listener is hearing, not
    /// against the whole track's peak.
    ///
    /// One global scale is set by the loudest moment in the track, so an
    /// ABSOLUTE floor eats a fixed amount out of every frame and takes
    /// proportionally more of a quiet one -- quiet passages lose their detail
    /// first and, far enough down, go silent entirely. That is what made a
    /// render sound like beeps between the loud hits.
    ///
    /// Signal: a loud second, then a second 30 dB down, each carrying a
    /// companion tone 40 dB below its own half. The quiet half's companion is
    /// therefore 70 dB below the TRACK peak (an absolute -60 dB floor zeroes
    /// it) but only 40 dB below its OWN frame (a relative floor keeps it).
    #[test]
    fn the_floor_is_measured_against_the_frame_not_the_track() {
        let c = band_centres(6);
        let (loud_hz, quiet_hz) = (c[3], c[5]);
        let n = SR as usize * 2;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                let a = if i < SR as usize { 1.0 } else { 0.03 };
                a * ((TAU * loud_hz * t).sin() + 0.01 * (TAU * quiet_hz * t).sin())
            })
            .collect();
        let t = analyze(&SampleClip::new(SR, samples), &opts(), &mut NoProgress).expect("analyze");
        assert!(t.frame_count >= 58, "need both halves, got {}", t.frame_count);

        // Which band the companion landed in, found rather than assumed: it
        // is whichever band is loudest in the LOUD half after excluding the
        // main tone's own band and its neighbours.
        let peak_at = |f: usize| {
            (0..t.volumes.len()).max_by(|&a, &b| t.volumes[a][f].total_cmp(&t.volumes[b][f]))
        };
        let main = peak_at(10).expect("a loudest band");
        let companion = (0..t.volumes.len())
            .filter(|&b| b.abs_diff(main) > 1)
            .max_by(|&a, &b| t.volumes[a][10].total_cmp(&t.volumes[b][10]))
            .expect("a companion band");

        let absolute_floor = 10f64.powf(-60.0 / 20.0);
        // Frames 30.. lie wholly inside the quiet half (window 4096, hop 1600).
        for f in 30..t.frame_count {
            let v = t.volumes[companion][f];
            assert!(
                v > 0.0,
                "band {companion} frame {f} is 40 dB below its own frame's peak and must \
                 survive the floor, got exactly zero -- a floor measured against the TRACK \
                 peak silences a quiet passage's detail while keeping a loud one's"
            );
            assert!(
                v < absolute_floor,
                "the test is vacuous unless the companion really is below the old absolute \
                 -60 dB floor ({absolute_floor}); band {companion} frame {f} is {v}"
            );
        }
    }

    // ================= spectral peak selection =================
    //
    // The defect these cover: a filterbank gives EVERY band whatever energy
    // lands in its slot, so on a real pop master 94.40 of 96 bands were
    // non-zero in an average frame. Ninety-four sine oscillators 0.85
    // semitones apart, sounding at once, is broadband noise -- and the
    // preceding loudness fix only made that noise louder. Nothing in the
    // suite above asserts sparsity, so every one of these mutations shipped
    // green: selection removed, top-N without the local-maximum test, the
    // local-maximum test without top-N, N off by one, endpoints always or
    // never selected, and the noise bands exempted from the competition.

    /// `select_peaks` over a real plan's axis, on a hand-built frame.
    fn sel(plan: &BandPlan, frame: &mut [f32], n: usize, sounding: &mut [bool]) -> Vec<usize> {
        let axis = frequency_axis(plan);
        select_peaks(frame, &axis, n, DEFAULT_PEAK_GATE, sounding);
        frame
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 0.0)
            .map(|(b, _)| b)
            .collect()
    }

    /// The axis is FREQUENCY order, not storage order. `fold` sends everything
    /// above the bank to white and everything below it to pink, so the two
    /// noise bands are the ends of the same axis the tonal bands sit on -- and
    /// getting that wrong would make the local-maximum test compare a 40 Hz
    /// band against a cymbal wash.
    #[test]
    fn the_frequency_axis_runs_pink_then_tonal_then_white() {
        let p = BandPlan::new(32, 2).expect("plan");
        let axis = frequency_axis(&p);
        assert_eq!(axis.len(), p.len(), "every band must appear exactly once");
        assert_eq!(p.kinds[axis[0]], BandKind::PinkNoise, "pink is below the bank");
        assert_eq!(p.kinds[axis[31]], BandKind::WhiteNoise, "white is above it");
        assert_eq!(&axis[1..31], &(0..30).collect::<Vec<_>>()[..], "tonal, in pitch order");
        // With no noise bands the axis is just the tonal bands.
        let p0 = BandPlan::new(8, 0).expect("plan");
        assert_eq!(frequency_axis(&p0), (0..8).collect::<Vec<_>>());
    }

    /// One strong note must cost ONE voice, not three. Its skirt lights the
    /// bands either side of it, and a plain top-N spends slots on them.
    #[test]
    fn a_hump_selects_only_its_apex() {
        let p = BandPlan::new(32, 2).expect("plan");
        let mut f = vec![0.0f32; 32];
        // A hump over tonal bands 4..=8, apex at 6.
        for (i, v) in [(4, 1.0), (5, 3.0), (6, 9.0), (7, 3.0), (8, 1.0)] {
            f[i] = v;
        }
        let got = sel(&p, &mut f, 8, &mut vec![false; 32]);
        assert_eq!(
            got,
            vec![6],
            "only the apex is a spectral peak; the shoulders belong to the same note and \
             must not each consume a voice slot"
        );
    }

    /// Top-N is not optional. The local-maximum test alone leaves 29.28
    /// candidates in an average frame on a real track -- still a wall.
    #[test]
    fn only_the_n_strongest_peaks_survive() {
        let p = BandPlan::new(32, 2).expect("plan");
        // Ten isolated peaks of descending height on alternate tonal bands.
        let mut f = vec![0.0f32; 32];
        for k in 0..10 {
            f[2 * k + 1] = 10.0 - k as f32;
        }
        for n in [1usize, 3, 4, 7] {
            let mut g = f.clone();
            let got = sel(&p, &mut g, n, &mut vec![false; 32]);
            assert_eq!(got.len(), n, "--max-voices {n} must leave exactly {n} bands sounding");
            // and they must be THE strongest ones, not any n of them
            assert_eq!(
                got,
                (0..n).map(|k| 2 * k + 1).collect::<Vec<_>>(),
                "the {n} kept must be the {n} loudest peaks"
            );
        }
    }

    /// `--max-voices 0` is the documented escape hatch and the A/B control: no
    /// selection at all. It must really mean "every band", not "no bands" and
    /// not "one band".
    #[test]
    fn zero_max_voices_disables_selection_entirely() {
        let p = BandPlan::new(32, 2).expect("plan");
        let mut f: Vec<f32> = (0..32).map(|i| 1.0 + (i % 3) as f32).collect();
        let before = f.clone();
        let got = sel(&p, &mut f, 0, &mut vec![false; 32]);
        assert_eq!(f, before, "with selection off no band may be touched");
        assert_eq!(got.len(), 32);
    }

    /// An axis endpoint has one neighbour and qualifies by beating it. Not
    /// dropped: measured on a real track, an endpoint is the single loudest
    /// peak in the frame 5.5% of the time -- that is a bass drop or a cymbal
    /// wash, and it IS the music at that moment.
    #[test]
    fn an_axis_endpoint_qualifies_against_its_one_neighbour() {
        let p = BandPlan::new(32, 2).expect("plan");
        let (white, pink) = (30usize, 31usize);
        let mut f = vec![1.0f32; 32];
        f[pink] = 5.0;
        f[white] = 5.0;
        let got = sel(&p, &mut f, 8, &mut vec![false; 32]);
        assert!(got.contains(&pink), "pink beats its one neighbour and must sound: {got:?}");
        assert!(got.contains(&white), "white beats its one neighbour and must sound: {got:?}");
        // ...and the flat tonal middle, where nothing is strictly greater than
        // both neighbours, contributes nothing.
        assert_eq!(got.len(), 2, "a plateau has no peaks in it: {got:?}");
    }

    /// The other half of the endpoint rule: an endpoint is not selected just
    /// for BEING an endpoint. This is the mutation where the two noise bands
    /// pass the local-maximum test unconditionally, which is a permanent
    /// noise bed by another route.
    #[test]
    fn an_endpoint_that_loses_to_its_neighbour_is_not_a_peak() {
        let p = BandPlan::new(32, 2).expect("plan");
        let (white, pink) = (30usize, 31usize);
        let mut f = vec![1.0f32; 32];
        // Both noise bands are beaten by the tonal band next to them.
        f[0] = 4.0; // lowest tonal, pink's neighbour
        f[29] = 4.0; // highest tonal, white's neighbour
        f[15] = 9.0;
        let got = sel(&p, &mut f, 12, &mut vec![false; 32]);
        assert!(!got.contains(&pink), "pink is below its neighbour and is not a peak: {got:?}");
        assert!(!got.contains(&white), "white is below its neighbour and is not a peak: {got:?}");
        assert_eq!(got, vec![0, 15, 29], "only the three real peaks: {got:?}");
    }

    /// The noise bands compete on magnitude like everything else. Letting them
    /// always through was measured and rejected: they are non-zero in 99.4% of
    /// frames on a real track, so an exemption is 1.99 permanent voices of
    /// hiss and rumble under the whole render.
    #[test]
    fn the_noise_bands_compete_for_a_slot_rather_than_being_exempt() {
        let p = BandPlan::new(32, 2).expect("plan");
        let (white, pink) = (30usize, 31usize);
        // Twelve loud tonal peaks; both noise bands carry real but ordinary
        // energy and are genuine local maxima.
        let mut f = vec![0.1f32; 32];
        for k in 0..12 {
            f[2 * k + 1] = 10.0 + k as f32;
        }
        f[white] = 1.0;
        f[pink] = 1.0;
        assert!(f[white] > f[29] && f[pink] > f[0], "both noise bands must be local maxima");
        let got = sel(&p, &mut f, 6, &mut vec![false; 32]);
        assert_eq!(got.len(), 6);
        assert!(
            !got.contains(&white) && !got.contains(&pink),
            "both noise bands are far below the six strongest peaks and must be silent, \
             got {got:?}"
        );
        // Non-vacuous: when they ARE among the strongest, they sound.
        let mut f2 = vec![0.1f32; 32];
        f2[15] = 1.0;
        f2[white] = 9.0;
        f2[pink] = 9.0;
        let got2 = sel(&p, &mut f2, 6, &mut vec![false; 32]);
        assert!(
            got2.contains(&white) && got2.contains(&pink),
            "a cymbal wash and a sub-bass drop must be able to win a slot, got {got2:?}"
        );
    }

    /// A band already sounding must be beaten by [`VOICE_HYSTERESIS`], not
    /// merely equalled, before it loses its slot. At 30 fps a slot that
    /// changes hands every frame is a 33 ms chirp; measured on a real track,
    /// this bias cuts the share of one-frame "on" runs from 16.8% to 8.1%.
    ///
    /// It biases RANK only. The volume written is the band's own magnitude,
    /// never a boosted one.
    #[test]
    fn an_incumbent_band_keeps_its_slot_until_it_is_clearly_beaten() {
        let p = BandPlan::new(32, 2).expect("plan");
        let (a, b) = (5usize, 15usize);
        let mut sounding = vec![false; 32];

        let mut f1 = vec![0.0f32; 32];
        f1[a] = 10.0;
        f1[b] = 1.0;
        assert_eq!(sel(&p, &mut f1, 1, &mut sounding), vec![a], "the louder band wins first");

        // A challenger 1.2x the incumbent: a plain top-1 hands the slot over,
        // and the pair flips back and forth for as long as the two hover.
        let mut f2 = vec![0.0f32; 32];
        f2[a] = 10.0;
        f2[b] = 12.0;
        assert_eq!(
            sel(&p, &mut f2, 1, &mut sounding),
            vec![a],
            "a challenger within {VOICE_HYSTERESIS}x must not take the slot"
        );
        // **The bonus biases RANK ONLY.** Applying it to the value instead
        // would write every sustained band {VOICE_HYSTERESIS}x (3.5 dB) louder
        // than it is, which is inaudible as a bug and wrong in every frame --
        // a mutation campaign found this the one hole in the tests above.
        assert_eq!(
            f2[a], 10.0,
            "the incumbent's VOLUME must be its own magnitude, not the boosted rank key"
        );

        // Clearly beaten: the slot changes hands, so this is stickiness and
        // not a latch.
        let mut f3 = vec![0.0f32; 32];
        f3[a] = 10.0;
        f3[b] = 20.0;
        assert_eq!(sel(&p, &mut f3, 1, &mut sounding), vec![b], "a clear winner takes the slot");
        // The kept band's VOLUME is its own, never the boosted rank key.
        assert_eq!(f3[b], 20.0);
    }

    /// A held-over band must still be a spectral peak. Hysteresis makes an
    /// incumbent hard to displace, not immortal.
    #[test]
    fn hysteresis_cannot_keep_a_band_that_stopped_being_a_peak() {
        let p = BandPlan::new(32, 2).expect("plan");
        let mut sounding = vec![false; 32];
        let mut f1 = vec![0.0f32; 32];
        f1[5] = 10.0;
        assert_eq!(sel(&p, &mut f1, 4, &mut sounding), vec![5]);
        // Band 5 is now a shoulder of a peak at 6, and silent besides.
        let mut f2 = vec![0.0f32; 32];
        f2[5] = 3.0;
        f2[6] = 9.0;
        f2[7] = 3.0;
        assert_eq!(sel(&p, &mut f2, 4, &mut sounding), vec![6], "a shoulder is not held over");
    }

    /// `n` tones on evenly-spaced tonal bands of a `bands`-wide plan.
    fn spread_tones(bands: usize, n: usize, step: usize, secs: f32) -> SampleClip {
        let p = BandPlan::new(bands, 2).expect("plan");
        let freqs: Vec<f32> =
            (0..n).map(|i| p.pitches[6 + i * step] * crate::audio::bands::BASE_HZ).collect();
        tones(&freqs, &vec![1.0 / n as f32; n], secs)
    }

    /// **The headline property, and the entire point of this change.** The
    /// number of bands sounding at once must be about `--max-voices`, whatever
    /// the source throws at the bank.
    ///
    /// Nothing in this suite asserted it before. Measured on a real pop master
    /// at 96 bands, 94.40 of 96 bands were non-zero in an average frame, and
    /// every test above passed on that render -- they check levels, ranges and
    /// dynamics, none of which notice that the output is a wall of sound.
    ///
    /// The signal is 24 simultaneous tones: more than any N under test, so the
    /// limit is what decides the answer rather than the source running out of
    /// content.
    #[test]
    fn the_mean_number_of_sounding_bands_tracks_max_voices() {
        let clip = spread_tones(81, 24, 3, 1.0);
        for n in [4usize, 8, 12, 16] {
            let o = AudioOptions { bands: Some(81), max_voices: n, ..opts() };
            let t = analyze(&clip, &o, &mut NoProgress).expect("analyze");
            let nz: usize =
                t.volumes.iter().flat_map(|b| b.iter()).filter(|v| **v > 0.0).count();
            let mean = nz as f64 / t.frame_count as f64;
            assert!(
                (mean - n as f64).abs() < 0.5,
                "--max-voices {n} must leave about {n} bands sounding per frame, got {mean:.2} \
                 of 81 -- a bank with most of its speakers on at once is noise, not music"
            );
        }
    }

    /// The other half of the headline: with selection OFF the same signal
    /// lights most of the bank. Without this, a build where selection is
    /// deleted outright and `max_voices` ignored fails the test above only
    /// because it happens to be dense -- this pins the contrast to the flag.
    #[test]
    fn selection_is_what_makes_the_bank_sparse() {
        let clip = spread_tones(81, 24, 3, 1.0);
        let mean_for = |n: usize| {
            let o = AudioOptions { bands: Some(81), max_voices: n, ..opts() };
            let t = analyze(&clip, &o, &mut NoProgress).expect("analyze");
            t.volumes.iter().flat_map(|b| b.iter()).filter(|v| **v > 0.0).count() as f64
                / t.frame_count as f64
        };
        let off = mean_for(0);
        let on = mean_for(12);
        // 56.6 of 96 here; a real pop master, whose noise floor and reverb
        // tails leave nothing below the frame floor, measured 94.40 of 96.
        assert!(
            off > 40.0,
            "24 tones across a 96-band bank must light most of the bank with selection off \
             (got {off:.2}) -- otherwise the comparison below proves nothing"
        );
        assert!(
            on < off / 4.0,
            "selection must be what sparsifies the bank: off {off:.2}, on {on:.2}"
        );
    }

    /// Selection is ON by default. A default of 0 is a build that ships the
    /// defect with the fix sitting unreachable behind a flag.
    #[test]
    fn selection_is_on_by_default_at_the_measured_knee() {
        let d = AudioOptions::default();
        assert_ne!(d.max_voices, 0, "the default must not disable peak selection");
        assert!(
            (6..=16).contains(&d.max_voices),
            "the default must sit at the measured knee of the energy/voice curve -- 6 to 12 \
             buys 1.11 dB of frame power for six voices, 12 to 24 buys 0.68 dB for twelve -- \
             got {}",
            d.max_voices
        );
    }

    /// Deterministic broadband pseudo-noise: energy in every band, and shallow
    /// local maxima everywhere. This is what the prominence gate exists to
    /// reject, and what a real master is full of (reverb tails, cymbal wash,
    /// room tone) -- a bank of pure tones cannot exercise the gate at all,
    /// because every one of its maxima is genuinely prominent.
    ///
    /// A fixed LCG rather than `rand`: the test must not flake, and the crate
    /// has no RNG dependency.
    fn noise(secs: f32) -> SampleClip {
        let mut state = 0x2545F491_4F6CDD1Du64;
        let n = (48_000.0 * secs) as usize;
        SampleClip::new(
            48_000,
            (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
                })
                .collect(),
        )
    }

    /// **The regression this flag exists for.**
    ///
    /// `--max-voices` is an UPPER bound; the prominence gate is what decides
    /// how many candidates there are to bound. On dense material the gate binds
    /// first, so above it `--max-voices` does nothing whatsoever -- measured in
    /// the field as four renders of a piano master at 24, 32, 48 and 64 voices
    /// coming back 311925 / 311919 / 311917 / 311913 bytes. Twelve bytes apart
    /// across a 40-voice range: the same render, four times, with no error and
    /// nothing to see.
    ///
    /// Both halves are asserted, because either alone is satisfiable by an
    /// implementation that ignores one of the two flags:
    ///
    /// * at the default gate, raising `--max-voices` far past the candidate
    ///   count must change (almost) nothing -- the bug, pinned so it cannot be
    ///   "fixed" by making `--max-voices` bind where it should not;
    /// * lowering the gate at that same `--max-voices` must sound audibly more
    ///   bands -- the escape hatch actually working.
    #[test]
    fn the_peak_gate_and_not_max_voices_is_what_limits_a_dense_frame() {
        let clip = noise(1.0);
        let mean_for = |voices: usize, gate: f32| {
            let o = AudioOptions {
                bands: Some(81),
                max_voices: voices,
                peak_gate: gate,
                ..opts()
            };
            let t = analyze(&clip, &o, &mut NoProgress).expect("analyze");
            t.volumes.iter().flat_map(|b| b.iter()).filter(|v| **v > 0.0).count() as f64
                / t.frame_count as f64
        };

        let at_24 = mean_for(24, DEFAULT_PEAK_GATE);
        let at_64 = mean_for(64, DEFAULT_PEAK_GATE);
        assert!(
            (at_64 - at_24).abs() < 1.0,
            "at the default gate, 24 and 64 voices must land on the same handful of \
             candidates ({at_24:.2} vs {at_64:.2}) -- if this ever stops being true the \
             flag documentation is wrong, not the code"
        );

        let open = mean_for(64, 1.0);
        assert!(
            open > at_64 * 1.5,
            "lowering --peak-gate to 1.0 at --max-voices 64 must sound substantially more \
             bands: {at_64:.2} at the {DEFAULT_PEAK_GATE} default, {open:.2} with the gate \
             off. If these are equal the flag does not reach the gate."
        );
        assert!(
            open > at_24,
            "the gate, not the voice cap, is the binding constraint here"
        );
    }

    /// A prominence is a ratio of magnitudes, so anything below 1.0 admits
    /// every local maximum exactly as 1.0 does. Accepting it silently would
    /// invite "0.5 is half as strict as 1.0", which it is not.
    #[test]
    fn an_out_of_range_peak_gate_is_an_error() {
        let clip = tones(&[440.0], &[1.0], 0.5);
        for bad in [0.0f32, 0.5, -1.0, f32::NAN] {
            let o = AudioOptions { peak_gate: bad, ..opts() };
            assert!(
                analyze(&clip, &o, &mut NoProgress).is_err(),
                "--peak-gate {bad} must be rejected"
            );
        }
        let o = AudioOptions { peak_gate: 1.0, ..opts() };
        assert!(
            analyze(&clip, &o, &mut NoProgress).is_ok(),
            "1.0 is the documented 'no gate' setting and must be accepted"
        );
    }

    #[test]
    fn the_peak_gate_defaults_to_the_measured_knee() {
        assert_eq!(AudioOptions::default().peak_gate, DEFAULT_PEAK_GATE);
        assert!(
            DEFAULT_PEAK_GATE > 1.0,
            "the default must actually gate; 1.0 is the off switch"
        );
    }

    // -- the envelope --------------------------------------------------------

    /// Mean length, in frames, of a run of one band being continuously
    /// non-zero. **The beeping metric**: a band that switches on for one or two
    /// frames and off again is a 33-66 ms blip, which is heard as a beep rather
    /// than as a note.
    fn mean_run(t: &VoiceTrack) -> f64 {
        let mut runs = Vec::new();
        for band in &t.volumes {
            let mut run = 0usize;
            for &v in band {
                if v > 0.0 {
                    run += 1;
                } else if run > 0 {
                    runs.push(run);
                    run = 0;
                }
            }
            if run > 0 {
                runs.push(run);
            }
        }
        if runs.is_empty() {
            return 0.0;
        }
        runs.iter().sum::<usize>() as f64 / runs.len() as f64
    }

    /// **The fix for "a lot of random beep pitch sounds".**
    ///
    /// Broadband noise makes the band selection churn: a band wins a slot for a
    /// frame or two and loses it, which without an envelope is an abrupt blip.
    /// The release must turn those into short notes with tails, and the metric
    /// that says so is the mean run of consecutive non-zero frames.
    ///
    /// Asserted as a RATIO between two runs of the same analysis, so it cannot
    /// be satisfied by a source that happens to be sustained anyway.
    #[test]
    fn the_release_envelope_lengthens_the_runs_that_sound_like_beeps() {
        let clip = noise(2.0);
        let run_for = |attack: f32, release: f32| {
            let o = AudioOptions { attack_ms: attack, release_ms: release, ..opts() };
            mean_run(&analyze(&clip, &o, &mut NoProgress).expect("analyze"))
        };
        let bare = run_for(0.0, 0.0);
        let smoothed = run_for(DEFAULT_ATTACK_MS, DEFAULT_RELEASE_MS);
        assert!(
            bare < 5.0,
            "the fixture must actually chatter without an envelope, got {bare:.2} frames per \
             run -- otherwise there is nothing here for the envelope to fix"
        );
        assert!(
            smoothed > bare * 2.0,
            "the release envelope must materially lengthen a band's runs: {bare:.2} frames \
             bare, {smoothed:.2} smoothed. A one- or two-frame run is a 33-66 ms blip, which \
             is the beeping."
        );
    }

    /// The envelope must be ON by default. A default of 0/0 ships the defect
    /// with the fix sitting unreachable behind a flag.
    #[test]
    fn the_envelope_is_on_by_default_and_asymmetric() {
        let d = AudioOptions::default();
        assert_eq!(d.attack_ms, DEFAULT_ATTACK_MS);
        assert_eq!(d.release_ms, DEFAULT_RELEASE_MS);
        assert!(
            d.release_ms > d.attack_ms * 4.0,
            "attack and release are asymmetric BY DESIGN -- fast up keeps transients punchy, \
             slow down stops the chatter -- and collapsing them into one constant loses one \
             or the other. Got attack {} release {}",
            d.attack_ms,
            d.release_ms
        );
        let e = Envelope::new(d.attack_ms, d.release_ms, 30.0);
        assert!(!e.is_off());
        assert!(
            e.attack > e.release,
            "a shorter attack TIME must give the LARGER coefficient: {} vs {}",
            e.attack,
            e.release
        );
    }

    /// A note ENDING and a level FALLING are two different times, and voice
    /// mode used to take both from `--release`.
    ///
    /// 150 ms is a reasonable one-pole time constant for a band that never
    /// stops existing, and a disastrous note-off for a spoken phoneme 50-100 ms
    /// long -- measured at 52.7% of all sounding voice-frames on speech being a
    /// voice playing a note the source had already stopped playing. The default
    /// note-off must fit inside a phoneme, and it must be a SEPARATE number.
    #[test]
    fn the_voice_release_is_its_own_default_and_fits_inside_a_phoneme() {
        let d = AudioOptions::default();
        assert_eq!(d.voice_release_ms, DEFAULT_VOICE_RELEASE_MS);
        assert!(
            d.voice_release_ms <= 50.0,
            "a spoken phoneme is 50-100 ms; a note-off longer than one smears across the \
             next. Got {} ms",
            d.voice_release_ms
        );
        assert!(
            d.voice_release_ms < d.release_ms,
            "a note must stop sooner than a level settles: --voice-release {} vs \
             --release {}",
            d.voice_release_ms,
            d.release_ms
        );
        // ...and it must still be at least one analysis frame, the shortest
        // fade the format can express (33 ms at the default 30 fps), so that a
        // fade exists at all.
        assert!(
            d.voice_release_ms > 0.0,
            "a note-off of exactly 0 would be a cut with no ramp to express"
        );
    }

    /// The time constants must mean the same thing at any `--audio-fps`. The
    /// coefficient is `1 - exp(-dt/tau)`, so the same elapsed time must give
    /// the same decay however it is divided into frames; a coefficient written
    /// directly would silently change meaning with the rate.
    #[test]
    fn the_envelope_coefficients_follow_the_frame_rate() {
        let slow = Envelope::new(10.0, 150.0, 15.0);
        let fast = Envelope::new(10.0, 150.0, 60.0);
        assert!(fast.release < slow.release);
        let one = Envelope::new(10.0, 150.0, 30.0).step(1.0, 0.0);
        let e60 = Envelope::new(10.0, 150.0, 60.0);
        let two = e60.step(e60.step(1.0, 0.0), 0.0);
        assert!(
            (one - two).abs() < 1e-5,
            "one frame at 30 fps and two at 60 must decay the same: {one} vs {two}"
        );
    }

    /// `0` is the documented off switch and must be exactly the identity --
    /// that is what makes an A/B against the unsmoothed renderer meaningful.
    #[test]
    fn a_zero_time_is_no_smoothing_at_all() {
        let e = Envelope::new(0.0, 0.0, 30.0);
        assert!(e.is_off());
        let mut series = vec![0.0f32, 1.0, 0.0, 0.5];
        let before = series.clone();
        e.apply(&mut series);
        assert_eq!(series, before);
    }

    /// The follower starts from the first value, not from zero: initialised at
    /// 0 it fades the first note of the track in over its attack, which is a
    /// fade that is not in the source.
    #[test]
    fn the_envelope_does_not_fade_the_first_frame_in() {
        let e = Envelope::new(200.0, 200.0, 30.0);
        let mut series = vec![1.0f32; 8];
        e.apply(&mut series);
        assert!(
            (series[0] - 1.0).abs() < 1e-6,
            "a track that starts loud must start loud, got {}",
            series[0]
        );
    }

    #[test]
    fn a_negative_or_non_finite_envelope_time_is_an_error() {
        let clip = tones(&[440.0], &[1.0], 0.5);
        for (a, r) in [(-1.0f32, 150.0f32), (10.0, -1.0), (f32::NAN, 150.0), (10.0, f32::NAN)] {
            let o = AudioOptions { attack_ms: a, release_ms: r, ..opts() };
            assert!(
                analyze(&clip, &o, &mut NoProgress).is_err(),
                "attack {a} release {r} must be rejected"
            );
        }
    }

    // -- the grid has to be a musical one ------------------------------------

    /// **Only multiples of 12 put bands on real notes**, and every other value
    /// was heard in game as out of tune -- 14 as sharp, 18 as flat, from a
    /// controlled sweep on solo piano with one binary and only this flag
    /// varying. There is no source for which 14 is what someone wants, and the
    /// failure is musical rather than obvious, so it is an error and not a
    /// warning.
    #[test]
    fn a_subdivision_that_is_not_a_multiple_of_twelve_is_an_error() {
        let clip = tones(&[440.0], &[1.0], 0.5);
        for bad in [0u32, 1, 5, 7, 13, 14, 18, 20, 25] {
            let o = AudioOptions { subdiv: bad, ..opts() };
            let Err(err) = analyze(&clip, &o, &mut NoProgress) else {
                panic!("--subdiv {bad} must be rejected");
            };
            assert!(
                err.contains("12"),
                "the error must name the value the user should have used: {err}"
            );
        }
        for good in [12u32, 24, 36] {
            let o = AudioOptions { subdiv: good, bands: Some(24), ..opts() };
            assert!(
                analyze(&clip, &o, &mut NoProgress).is_ok(),
                "--subdiv {good} is a multiple of 12 and must be accepted"
            );
        }
    }

    /// Noise bands are OFF by default: reported worse on every source tried in
    /// game -- speech, solo piano and a full pop mix alike.
    #[test]
    fn noise_bands_are_off_by_default() {
        assert_eq!(AudioOptions::default().noise_bands, 0);
        let t = analyze(&tones(&[440.0], &[1.0], 0.5), &AudioOptions::default(), &mut NoProgress)
            .expect("analyze");
        assert!(
            t.plan.kinds.iter().all(|k| *k == BandKind::Tonal),
            "a default render must contain no noise speakers at all"
        );
    }

    /// Hysteresis measured the way it is actually heard: how often the set of
    /// sounding bands changes, frame to frame, when two bands hover around
    /// each other. Two tones whose levels cross repeatedly, one voice slot.
    ///
    /// The oracle is the same track analysed with selection off: the number of
    /// times the two bands genuinely swap rank is what a memoryless top-N
    /// would toggle on, and the selected set must toggle far less than that.
    #[test]
    fn a_hovering_pair_does_not_swap_the_voice_slot_every_frame() {
        // `--bands 32` explicitly: the two band indices below are read out
        // of this plan and fed to `analyze`, so the two must be the SAME
        // plan. Leaving `analyze` on the derived default would put the tones
        // in bands nobody is watching and the oracle would count 0 crossings.
        let base = AudioOptions { bands: Some(32), ..opts() };
        let p = BandPlan::new(32, 2).expect("plan");
        let (ba, bb) = (10usize, 20usize);
        let (fa, fb) =
            (p.pitches[ba] * crate::audio::bands::BASE_HZ, p.pitches[bb] * crate::audio::bands::BASE_HZ);
        // B wobbles +-20% around A at 2 Hz: it crosses A repeatedly and never
        // gets anywhere near VOICE_HYSTERESIS times louder.
        let n = SR as usize * 3;
        let clip = SampleClip::new(
            SR,
            (0..n)
                .map(|i| {
                    let t = i as f32 / SR as f32;
                    let wob = 1.0 + 0.2 * (TAU * 2.0 * t).sin();
                    0.5 * (TAU * fa * t).sin() + 0.5 * wob * (TAU * fb * t).sin()
                })
                .collect(),
        );

        // Oracle: how many times do the two bands swap rank?
        let loose = AudioOptions { max_voices: 0, ..base };
        let raw = analyze(&clip, &loose, &mut NoProgress).expect("analyze");
        let crossings = (1..raw.frame_count)
            .filter(|&f| {
                let now = raw.volumes[ba][f] > raw.volumes[bb][f];
                let was = raw.volumes[ba][f - 1] > raw.volumes[bb][f - 1];
                now != was
            })
            .count();
        assert!(crossings >= 6, "the pair must really cross repeatedly, got {crossings}");

        let one = AudioOptions { max_voices: 1, ..base };
        let t = analyze(&clip, &one, &mut NoProgress).expect("analyze");
        let handovers = (1..t.frame_count)
            .filter(|&f| {
                let sel = |f: usize| (0..t.volumes.len()).find(|&b| t.volumes[b][f] > 0.0);
                sel(f) != sel(f - 1)
            })
            .count();
        assert!(
            handovers * 3 < crossings,
            "the voice slot must not change hands on every rank swap ({handovers} handovers \
             vs {crossings} crossings) -- at 30 fps that is a 33 ms chirp per swap"
        );
    }

    /// [`prominence`] is the ratio to the MEAN of a neighbourhood of exactly
    /// [`PROMINENCE_WIDTH`] bands either side, and the width is load-bearing:
    /// too narrow and a note's own leakage skirt becomes its neighbourhood,
    /// too wide and the next note in the chord does.
    ///
    /// The frame is built so the answer names the width. Distances 1 and 2
    /// carry 2.0, distance 3 carries 0.5 and distance 4 carries 8.0, so the
    /// mean -- and therefore the prominence -- is different for every width in
    /// the plausible range:
    ///
    /// | half-width | neighbourhood mean | prominence of a 3.0 peak |
    /// |------------|--------------------|--------------------------|
    /// | 2          | 8/4   = 2.0        | 1.50                     |
    /// | **3**      | 9/6   = 1.5        | **2.00**                 |
    /// | 4          | 25/8  = 3.125      | 0.96                     |
    #[test]
    fn prominence_is_the_ratio_to_the_mean_of_a_fixed_width_neighbourhood() {
        let p = BandPlan::new(32, 0).expect("plan");
        let axis = frequency_axis(&p);
        let mut f = vec![0.0f32; 32];
        f[10] = 3.0;
        for d in [1usize, 2] {
            f[10 - d] = 2.0;
            f[10 + d] = 2.0;
        }
        f[7] = 0.5;
        f[13] = 0.5;
        f[6] = 8.0;
        f[14] = 8.0;
        let got = prominence(&f, &axis, 10);
        assert!(
            (got - 2.0).abs() < 1e-5,
            "prominence must average the {PROMINENCE_WIDTH} bands either side and no others: \
             expected 3.0/1.5 = 2.0, got {got} -- 1.5 means the neighbourhood is one band too \
             narrow, 0.96 means it is one band too wide"
        );
    }

    /// A peak standing in perfect silence has nothing to be measured against.
    /// It must come back finite and ORDERABLE rather than infinite or NaN, so
    /// that two such peaks are still ranked against each other by magnitude --
    /// which is the whole of a synthetic test signal, and every digital fade.
    #[test]
    fn a_peak_in_silence_has_a_finite_maximal_prominence() {
        let p = BandPlan::new(32, 0).expect("plan");
        let axis = frequency_axis(&p);
        let mut f = vec![0.0f32; 32];
        f[10] = 0.001;
        let got = prominence(&f, &axis, 10);
        assert!(got.is_finite(), "an isolated peak's prominence must be finite, got {got}");
        assert_eq!(got, MAX_PROMINENCE);
    }

    /// **The headline behavioural change.** A quiet, NARROW spike must outrank
    /// a louder, BROADER one.
    ///
    /// Band 8 is a 10.0 sitting on a plateau of 6.0 -- the loudest thing in the
    /// frame, and under the previous magnitude rank it took the slot every
    /// time. Band 20 is a 6.0 standing over a floor of 1.0: quieter, but six
    /// times its surroundings instead of 1.67 times. That is the difference
    /// between a note and a ripple on a lump of bass rumble or cymbal wash, and
    /// it is invisible to a magnitude rank.
    #[test]
    fn a_narrow_spike_outranks_a_louder_broad_lump() {
        let p = BandPlan::new(32, 0).expect("plan");
        let mut f = vec![0.0f32; 32];
        // A broad lump, apex 10.0 over a plateau of 6.0: prominence 1.67.
        for b in 4..=12 {
            f[b] = 6.0;
        }
        f[8] = 10.0;
        // A narrow spike, 6.0 over a floor of 1.0: prominence 6.0.
        for b in 16..=24 {
            f[b] = 1.0;
        }
        f[20] = 6.0;

        let got = sel(&p, &mut f, 1, &mut vec![false; 32]);
        assert_eq!(
            got,
            vec![20],
            "the one voice must go to the spike that stands 6x over its neighbourhood, not to \
             the louder apex that stands 1.67x over its own -- ranking by raw magnitude picks \
             band 8 here, which is how a bank spends its slots on rumble and hiss"
        );
        // **Prominence decides WHICH band sounds, never HOW LOUD it is.** The
        // volume written is the band's own magnitude; scaling it by the rank
        // key would make every selected band 6x louder than it is and destroy
        // the dynamics the rest of this module works to preserve.
        assert_eq!(
            f[20], 6.0,
            "the selected band's volume must be its own magnitude, not its prominence or its \
             rank key"
        );
    }

    /// The prominence GATE: a local maximum that is not prominent enough is not
    /// a note, and must stay silent even when voice slots are going spare.
    ///
    /// This is the step that actually sparsifies a real track. Measured on a
    /// pop master at 96 bands, the local-maximum test alone leaves 29.02
    /// candidates per frame, so at `--max-voices 32` or `48` the top-N
    /// truncation never fires and EVERY local maximum sounds -- 29 sine
    /// oscillators at once. Every ranking scheme produced identical output
    /// there; only the gate changes anything.
    #[test]
    fn a_local_maximum_that_is_not_prominent_enough_is_not_a_note() {
        let p = BandPlan::new(32, 0).expect("plan");
        // A gentle bump on a broad plateau: 5.0 over 4.5 is prominence 1.11.
        let mut f = vec![0.0f32; 32];
        for b in 4..=16 {
            f[b] = 4.5;
        }
        f[10] = 5.0;
        // Twelve slots for one candidate, so nothing but the gate can silence it.
        let got = sel(&p, &mut f, 12, &mut vec![false; 32]);
        assert!(
            got.is_empty(),
            "a bump only {:.2}x its neighbourhood is a ripple on a lump, not a note, and must \
             not take a voice however many are free: {got:?}",
            5.0 / 4.5
        );

        // Non-vacuous: the SAME band, made genuinely prominent, does sound.
        let mut f2 = vec![0.0f32; 32];
        for b in 4..=16 {
            f2[b] = 4.5;
        }
        f2[10] = 9.0; // prominence 2.0
        assert_eq!(
            sel(&p, &mut f2, 12, &mut vec![false; 32]),
            vec![10],
            "the gate must pass a peak that really does stand above its neighbourhood"
        );
    }

    /// The gate is what makes `--max-voices` mean anything at a high setting.
    ///
    /// A frame of 20 low-contrast bumps has 20 local maxima, so a build without
    /// the gate hands out 20 voices at `--max-voices 32` -- which is what a
    /// render at 32 or 48 voices actually was. With the gate only the genuine
    /// spikes sound, and the count is set by the music rather than by N.
    #[test]
    fn a_high_max_voices_does_not_resurrect_the_wall_of_sound() {
        let p = BandPlan::new(32, 0).expect("plan");
        let mut f = vec![4.0f32; 32];
        // 15 low-contrast bumps: local maxima, but only 1.12x their neighbours.
        for k in 0..15 {
            f[2 * k + 1] = 4.5;
        }
        // ...and two real spikes.
        f[5] = 20.0;
        f[25] = 16.0;
        let got = sel(&p, &mut f, 32, &mut vec![false; 32]);
        assert_eq!(
            got,
            vec![5, 25],
            "with more slots than candidates the GATE, not the top-N truncation, has to be \
             what keeps the bank sparse -- got {got:?}"
        );
    }

    /// **Equal-amplitude tones spread across the bank must win voices across
    /// the bank.** The property the whole selection scheme exists to hold, and
    /// nothing asserted it before.
    ///
    /// Music's energy is concentrated at the low end, so a selector that ranks
    /// by anything correlated with raw level can spend every slot down there
    /// and leave the melody and the vocal without a voice -- "muffled". The
    /// signal here removes that excuse entirely: every tone carries the same
    /// amplitude, so a fair selector must spread its slots over the whole
    /// range, and any concentration is the selector's own bias.
    ///
    /// Tones are placed from band 12 up. Below that a band is narrower than the
    /// FFT's own bin spacing (at 96 bands over 44..4400 Hz, band 8 is 3.3 Hz
    /// wide against a 2.9 Hz bin and an 11.7 Hz Hann main lobe), so a low tone
    /// smears across several bands whatever the selector does -- a measurement
    /// artefact of the bank's geometry, not a selection bias, and it would make
    /// this test lie about which one it caught.
    #[test]
    fn equal_tones_across_the_range_are_not_selected_only_from_the_low_bands() {
        const BANDS: usize = 79;
        const LO: usize = 12;
        let p = BandPlan::new(BANDS, 0).expect("plan");
        // Eight equal tones, evenly spaced over bands 12..=75.
        let idx: Vec<usize> = (0..8).map(|i| LO + i * 9).collect();
        let freqs: Vec<f32> =
            idx.iter().map(|&b| p.pitches[b] * crate::audio::bands::BASE_HZ).collect();
        let clip = tones(&freqs, &vec![1.0 / 8.0; 8], 2.0);

        let o = AudioOptions {
            bands: Some(BANDS),
            noise_bands: 0,
            window: 16384,
            max_voices: 8,
            ..opts()
        };
        let t = analyze(&clip, &o, &mut NoProgress).expect("analyze");

        // Thirds of the range the tones actually occupy.
        let span = BANDS - LO;
        let mut third = [0usize; 3];
        for b in 0..t.volumes.len() {
            let r = ((b.saturating_sub(LO)) * 3 / span).min(2);
            third[r] += t.volumes[b].iter().filter(|v| **v > 0.0).count();
        }
        let total: usize = third.iter().sum();
        assert!(total > 0, "nothing was selected at all");
        let share = |r: usize| 100.0 * third[r] as f64 / total as f64;

        for (r, name) in ["low", "mid", "high"].iter().enumerate() {
            assert!(
                share(r) > 15.0,
                "the {name} third won only {:.1}% of the voice slots (low {:.1}%, mid {:.1}%, \
                 high {:.1}%) -- with every tone at the SAME amplitude a fair selector spreads \
                 its slots over the whole range, and starving a third of the spectrum is \
                 exactly what makes a render sound muffled",
                share(r),
                share(0),
                share(1),
                share(2)
            );
            assert!(
                share(r) < 55.0,
                "the {name} third took {:.1}% of the voice slots (low {:.1}%, mid {:.1}%, \
                 high {:.1}%) -- equal-amplitude tones must not let one region hog the bank",
                share(r),
                share(0),
                share(1),
                share(2)
            );
        }
    }

    /// Selection must not resurrect silence. A frame with no energy has no
    /// peaks, so it stays exactly zero -- the property
    /// [`silence_produces_exactly_zero_everywhere`] pins for a whole clip,
    /// checked here for the selector alone so a "always fill N slots"
    /// implementation cannot pass by writing zeros into N bands.
    #[test]
    fn a_silent_frame_selects_nothing() {
        let p = BandPlan::new(32, 2).expect("plan");
        let mut f = vec![0.0f32; 32];
        let mut sounding = vec![true; 32];
        assert!(sel(&p, &mut f, 12, &mut sounding).is_empty());
        assert!(sounding.iter().all(|s| !s), "nothing may be left marked as sounding");
    }
}

