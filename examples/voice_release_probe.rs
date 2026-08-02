//! Measures how long a voice keeps sounding after its partial has ended.
//!
//! The complaint this answers is "it bleeds together" / "disembodied": voices
//! that hold level after the note is over, so successive notes overlap into a
//! drone and speech never articulates into words. The metrics the mode already
//! carried (lifetime, jitter, cents) cannot see it -- a voice droning through a
//! whole phrase scores *better* on every one of them.
//!
//! Two measurements, because neither alone is enough:
//!
//! * **Ground truth** (`--synthetic`): a tone that stops dead at a known frame.
//!   The time from that frame to the voice reading exactly zero is the defect
//!   in milliseconds, with nothing inferred.
//! * **Real material**: `VoiceStats::tail_ms` and `unmatched_fraction`, which
//!   ask the same question of a track whose note ends are not known in advance
//!   -- the tail is measured from the last frame the voice was MATCHED to a
//!   peak, which is the tracker's own opinion of when the partial ended.
//!
//! ```text
//! cargo run --release --example voice_release_probe -- <file.mp3> [V] [window]
//! cargo run --release --example voice_release_probe -- --synthetic
//! ```
use heightmap::audio::backend::{AudioBackend, DownloadConsent, open_audio};
use heightmap::audio::source::{AudioSource, SampleClip};
use heightmap::audio::track::AudioOptions;
use heightmap::audio::stft::{StftStream, hop_for};
use heightmap::audio::voices::{analyze_voices, find_peaks, max_hz, min_hz};
use heightmap::progress::NoProgress;
use std::f32::consts::TAU;
use std::path::PathBuf;

const SR: u32 = 48_000;

/// A tone that plays for `on_secs` and then stops dead, followed by digital
/// silence. Nothing can be sounding after the cut, so any voice that still is
/// is measuring exactly the defect.
fn gated_tone(hz: f32, on_secs: f32, total_secs: f32) -> SampleClip {
    let n = (SR as f32 * total_secs) as usize;
    SampleClip::new(
        SR,
        (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                if t < on_secs { 0.5 * (TAU * hz * t).sin() } else { 0.0 }
            })
            .collect(),
    )
}

/// The same, but with a real noise bed under it -- the case the synthetic tone
/// cannot show. A tracker that continues a voice onto whatever peak is nearest
/// keeps that voice alive on the noise after the tone has gone.
fn gated_tone_in_noise(hz: f32, on_secs: f32, total_secs: f32, noise: f32) -> SampleClip {
    let n = (SR as f32 * total_secs) as usize;
    let mut state = 0x2545F491_4F6CDD1Du64;
    SampleClip::new(
        SR,
        (0..n)
            .map(|i| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let r = ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
                let t = i as f32 / SR as f32;
                let tone = if t < on_secs { 0.5 * (TAU * hz * t).sin() } else { 0.0 };
                tone + noise * r
            })
            .collect(),
    )
}

fn report(label: &str, source: &dyn AudioSource, opts: &AudioOptions) {
    let t = match analyze_voices(source, opts, &mut NoProgress) {
        Ok(t) => t,
        Err(e) => {
            println!("{label}: ERROR {e}");
            return;
        }
    };
    let st = &t.stats;
    let (mean_tail, p95_tail) = st.tail_ms(t.fps);
    let (vm, vu) = st.mean_volumes();
    println!(
        "{label}\n  frames {}  voices/frame {:.2}/{}  lifetime {:.1}f\n  \
         TAIL  mean {:.0} ms  p95 {:.0} ms  (n={})\n  \
         UNMATCHED-BUT-SOUNDING  {:.1}% of sounding voice-frames\n  \
         mean written volume: matched {:.3} / unmatched {:.3}\n  \
         cents {:.1}  jitter {:.2}  jumps {}",
        t.frame_count,
        st.mean_voices_per_frame(t.frame_count),
        t.voice_count(),
        st.mean_lifetime(),
        mean_tail,
        p95_tail,
        st.tail_frames.len(),
        st.unmatched_fraction() * 100.0,
        vm,
        vu,
        st.mean_abs_cents,
        st.jitter_rms_cents(),
        st.pitch_jumps,
    );
    tuning(st);
    for gate in [10.0, 20.0, 35.0] {
        et_rejection("  ", &t, gate);
    }
}

/// Is a large mean cents offset a DETUNED SOURCE or genuine inharmonic scatter?
///
/// The two sound completely different and the mean |offset| cannot tell them
/// apart. A master cut 20 cents flat of A440 puts every peak 20 cents off and
/// is not out of tune with itself at all -- it is a transposed key, and it
/// sounds fine. Peaks scattered at random across the semitone are notes landing
/// between the keys, which is what "off key" means.
///
/// The cents axis wraps at +-50, so the statistic is a circular one. Map each
/// offset onto the unit circle at `2*pi*c/100` and average:
///
/// * `|mean|` is the CONCENTRATION, 1.0 for peaks all at the same offset and
///   0.0 for peaks spread uniformly over the semitone -- i.e. for content with
///   no pitch grid in it at all.
/// * `arg(mean)` is that common offset: the source's own detuning from A440.
/// * the residual after removing it is the error that is actually audible as
///   wrong notes.
fn tuning(st: &heightmap::audio::voices::VoiceStats) {
    let n: u32 = st.cents_hist.iter().sum();
    if n == 0 {
        return;
    }
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (i, &c) in st.cents_hist.iter().enumerate() {
        let cents = i as f64 - 50.0;
        let th = std::f64::consts::TAU * cents / 100.0;
        re += th.cos() * c as f64;
        im += th.sin() * c as f64;
    }
    re /= n as f64;
    im /= n as f64;
    let concentration = (re * re + im * im).sqrt();
    let offset = im.atan2(re) / std::f64::consts::TAU * 100.0;
    // Mean |offset| once the source's own detuning is taken out, on the
    // wrapped axis.
    let mut resid = 0.0f64;
    for (i, &c) in st.cents_hist.iter().enumerate() {
        let mut d = (i as f64 - 50.0) - offset;
        while d > 50.0 {
            d -= 100.0;
        }
        while d < -50.0 {
            d += 100.0;
        }
        resid += d.abs() * c as f64;
    }
    println!(
        "  TUNING  concentration {concentration:.3} (1 = one grid, 0 = uniform over the \
         semitone)\n          source detuning {offset:+.1} cents; residual after removing \
         it {:.1} cents\n          (uniform noise scores concentration 0.00, residual 25.0)",
        resid / n as f64
    );
}

/// Ground truth: milliseconds from the frame the source went silent to the
/// frame every voice reads exactly zero.
fn synthetic(label: &str, clip: &SampleClip, on_secs: f32, opts: &AudioOptions) {
    let t = analyze_voices(clip, opts, &mut NoProgress).expect("analysis");
    // The analysis frame whose window first contains no tone at all. The STFT
    // window straddles the cut, so the earliest frame that CAN be silent is
    // `on_secs` plus one window.
    let cut = ((on_secs as f64 + opts.window as f64 / SR as f64) * t.fps as f64).ceil() as usize;
    let mut last_sounding = 0usize;
    for v in 0..t.voice_count() {
        for f in 0..t.frame_count {
            if t.volumes[v][f] > 0.0 {
                last_sounding = last_sounding.max(f);
            }
        }
    }
    let over = last_sounding.saturating_sub(cut);
    println!(
        "{label}\n  source silent from frame {cut}, last non-zero volume at frame \
         {last_sounding} => {over} frames = {:.0} ms after the note ended",
        over as f64 * 1000.0 / t.fps as f64
    );
    // ...and how much level is still being written during that overhang.
    let mut worst = 0.0f64;
    for v in 0..t.voice_count() {
        for f in cut..t.frame_count {
            worst = worst.max(t.volumes[v][f]);
        }
    }
    println!("  loudest volume written after the note ended: {worst:.3}");
}

/// What the peak finder actually hands the tracker: how many peaks a frame
/// yields at `V * PEAK_OVERSAMPLE`, and how far down the weakest of them are.
///
/// The question this answers is why voices never release. If a frame offers 100
/// peaks scattered across the spectrum, then almost any voice, at almost any
/// pitch, has SOMETHING within half a semitone to continue onto -- so a note
/// that has stopped keeps being "matched" and never enters its release at all.
fn peak_census(label: &str, source: &dyn AudioSource, opts: &AudioOptions, oversample: usize) {
    let info = source.info();
    let hop = hop_for(info.sample_rate, opts.fps).expect("hop");
    let mut stft = StftStream::new(source.open().expect("open"), opts.window, hop).expect("stft");
    let floor = 10f32.powf(opts.floor_db / 20.0);
    let mut frames = 0usize;
    let mut total = 0usize;
    // Fraction of the playable log-frequency range covered by a +-0.5 semitone
    // window around each peak -- i.e. the chance a voice at a random pitch
    // finds something to continue onto.
    let mut covered = 0.0f64;
    let span = 12.0 * (max_hz() / min_hz()).log2();
    let mut weakest_db = 0.0f64;
    while frames < opts.max_frames {
        let Some(sp) = stft.next_spectrum().expect("spectrum") else {
            break;
        };
        let peaks = find_peaks(
            &sp,
            info.sample_rate,
            opts.window,
            opts.max_voices * oversample,
            floor,
        );
        if peaks.is_empty() {
            frames += 1;
            continue;
        }
        total += peaks.len();
        let loudest = peaks.iter().map(|p| p.mag).fold(0.0f32, f32::max);
        let quietest = peaks.iter().map(|p| p.mag).fold(f32::MAX, f32::min);
        weakest_db += 20.0 * (quietest.max(1e-30) / loudest.max(1e-30)).log10() as f64;
        // Union of the +-0.5 semitone windows, on the semitone axis.
        let mut iv: Vec<(f64, f64)> = peaks
            .iter()
            .map(|p| {
                let s = 12.0 * (p.hz / min_hz()).log2();
                (s - 0.5, s + 0.5)
            })
            .collect();
        iv.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut lo, mut hi) = iv[0];
        let mut union = 0.0;
        for &(a, b) in &iv[1..] {
            if a > hi {
                union += hi - lo;
                lo = a;
                hi = b;
            } else {
                hi = hi.max(b);
            }
        }
        union += hi - lo;
        covered += union / span;
        frames += 1;
    }
    if frames == 0 {
        return;
    }
    println!(
        "{label}\n  {:.1} peaks/frame (cap {}), weakest {:.0} dB below the loudest\n  \
         a voice at a random pitch finds a match {:.1}% of the time",
        total as f64 / frames as f64,
        opts.max_voices * oversample,
        weakest_db / frames as f64,
        covered / frames as f64 * 100.0,
    );
}

/// Would rejecting peaks that are not near an equal-tempered pitch fix "off
/// key" -- or delete the timbre?
///
/// The proposal only makes sense if off-grid voices are junk. They are not:
/// **a harmonic series is not equal-tempered.** Against the fundamental, the
/// 3rd harmonic is 2 cents from the ET fifth, the 5th is **14 cents** from the
/// ET major third, the 7th is **31 cents** from ET, the 11th is 49. Every real
/// instrument's own overtones therefore fail an ET gate tighter than about 35
/// cents, and a render with them removed is the bare fundamentals that were
/// reported as sounding like an organ.
///
/// So the measurement is: of the voice-frames an ET gate would silence, how
/// many are harmonics of another sounding voice -- i.e. timbre -- and how much of
/// the written level goes with them.
fn et_rejection(label: &str, t: &heightmap::audio::voices::VoiceStreams, gate_cents: f64) {
    use heightmap::audio::bands::BASE_HZ;
    use heightmap::audio::voices::cents_from_equal_temperament;
    let (mut kept, mut cut) = (0usize, 0usize);
    let (mut kept_vol, mut cut_vol) = (0.0f64, 0.0f64);
    let (mut cut_harmonic, mut kept_harmonic) = (0usize, 0usize);
    for f in 0..t.frame_count {
        let mut sounding: Vec<(f64, f64)> = (0..t.voice_count())
            .filter(|&v| t.volumes[v][f] > 0.0)
            .map(|v| (t.pitches[v][f] * BASE_HZ as f64, t.volumes[v][f]))
            .collect();
        sounding.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (i, &(hz, vol)) in sounding.iter().enumerate() {
            // Is it an overtone of a lower sounding voice? Same rule as
            // `VoiceStats::harmonic_voices`.
            let harmonic = sounding[..i].iter().any(|&(lo, _)| {
                let r = hz / lo;
                let n = r.round();
                n >= 2.0 && (1200.0 * (r / n).log2()).abs() <= 35.0
            });
            if cents_from_equal_temperament(hz).0.abs() <= gate_cents {
                kept += 1;
                kept_vol += vol;
                kept_harmonic += harmonic as usize;
            } else {
                cut += 1;
                cut_vol += vol;
                cut_harmonic += harmonic as usize;
            }
        }
    }
    let tot = (kept + cut).max(1);
    println!(
        "{label} ET gate +-{gate_cents:.0} cents\n  would silence {:.1}% of sounding \
         voice-frames and {:.1}% of the written level\n  of the silenced, {:.1}% are \
         harmonics of another sounding voice (kept: {:.1}%)",
        cut as f64 / tot as f64 * 100.0,
        cut_vol / (kept_vol + cut_vol).max(1e-12) * 100.0,
        cut_harmonic as f64 / cut.max(1) as f64 * 100.0,
        kept_harmonic as f64 / kept.max(1) as f64 * 100.0,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let base = AudioOptions::default();

    if args.first().map(String::as_str) == Some("--synthetic") || args.is_empty() {
        let o = AudioOptions { max_voices: 8, window: 4096, ..base };
        synthetic(
            "-- clean tone, 1.0 s on, 3.0 s total --",
            &gated_tone(440.0, 1.0, 3.0),
            1.0,
            &o,
        );
        synthetic(
            "-- tone in noise (-34 dB bed), 1.0 s on, 3.0 s total --",
            &gated_tone_in_noise(440.0, 1.0, 3.0, 0.01),
            1.0,
            &o,
        );
        report(
            "-- tone in noise, full stats --",
            &gated_tone_in_noise(440.0, 1.0, 3.0, 0.01),
            &o,
        );
        return;
    }

    let path = PathBuf::from(&args[0]);
    let voices: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(32);
    let window: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let source = open_audio(&path, AudioBackend::Auto, DownloadConsent::Never).expect("open");
    // A sweep, not a single point: the release is the thing under test, and one
    // number for one setting cannot say whether it is the right setting.
    let sweep: Vec<f32> = match args.get(3) {
        Some(s) => s.split(',').filter_map(|x| x.parse().ok()).collect(),
        None => vec![0.0, 50.0, 100.0, 150.0],
    };
    for ms in sweep {
        let o = AudioOptions {
            max_voices: voices,
            window,
            voice_release_ms: ms,
            ..base
        };
        report(
            &format!("{} V={voices} window={window} --voice-release {ms}", path.display()),
            source.as_ref(),
            &o,
        );
    }
    peak_census(
        "  peak census",
        source.as_ref(),
        &AudioOptions { max_voices: voices, window, ..base },
        4,
    );
}
