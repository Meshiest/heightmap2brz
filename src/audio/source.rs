//! The audio source contract, split the same way `video::stream` splits
//! frames: a source is a cheap re-openable handle, a stream is a one-shot
//! cursor. The split is not decoration -- band analysis needs one pass to
//! find the normalisation peak and another to emit, and a `rewind()` every
//! backend must implement correctly is exactly the burden the video side
//! already declined to take on.

/// File extensions the audio path accepts as a bare audio file.
///
/// Exactly the five the builtin (`symphonia`) decoder is built for -- mp3,
/// wav/pcm, flac, vorbis-in-ogg and aac-in-mp4 -- so a file offered by a
/// picker filtered on this list is one every build can open, with or without
/// ffmpeg. A picker that accepts a file the tool then refuses is worse than
/// one that never offered it.
///
/// A video container is equally valid input (the pipeline pulls its audio
/// track out, and `--audio-track` picks which), but those extensions live in
/// [`crate::video::source::VIDEO_EXTENSIONS`] and are not duplicated here.
pub const AUDIO_EXTENSIONS: [&str; 5] = ["mp3", "wav", "flac", "ogg", "m4a"];

/// What a source knows before decoding anything.
///
/// There is deliberately no frame count here. A "frame" is an STFT concept
/// that depends on `--audio-fps` and `--window`; a decoder knows neither, so
/// asking it for one would invite a guess. `StftStream` derives the count
/// from `duration_hint`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioInfo {
    /// Always 48 000 once adapted. Decoders resample to this.
    pub sample_rate: u32,
    /// Seconds, when the source can say without decoding. `None` is a real
    /// answer, not a failure -- it drives spinner-vs-bar in progress.
    pub duration_hint: Option<f64>,
}

/// A re-openable handle to a stream of mono samples.
pub trait AudioSource {
    fn info(&self) -> AudioInfo;
    fn open(&self) -> Result<Box<dyn AudioStream + '_>, String>;
}

/// A one-shot cursor. Blocks may be any non-zero length; only the
/// concatenation is contractual.
pub trait AudioStream {
    /// The next block of mono f32 samples, or `None` once drained.
    ///
    /// An error is fatal to the render. A stream that fails halfway must not
    /// be treated as a short track -- that would silently write a save
    /// missing its tail.
    fn next(&mut self) -> Result<Option<Vec<f32>>, String>;
}

/// The global index and value of the first non-finite sample in `block`,
/// where `base` is `block[0]`'s index in the whole decoded stream.
///
/// Shared by every decode backend ([`crate::audio::symphonia_src`] and
/// [`crate::audio::ffmpeg_src`]): both decoders must reject a NaN or
/// infinity at the source rather than let it reach `track::analyze`'s
/// `min(1.0)` clamp, where `f32::NAN.min(1.0)` is `1.0` and a single corrupt
/// sample would launder into a whole bank of speakers at maximum volume.
/// Split out as a free function so the rejection rule can be tested directly
/// against a NaN a decoder would only ever produce on a corrupt file.
pub(crate) fn first_non_finite(block: &[f32], base: u64) -> Option<(u64, f32)> {
    block
        .iter()
        .position(|s| !s.is_finite())
        .map(|i| (base + i as u64, block[i]))
}

/// In-memory source. Used by every test, and by the GUI for short previews.
pub struct SampleClip {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
    /// Samples handed out per `next()` call.
    pub block: usize,
}

impl SampleClip {
    pub fn new(sample_rate: u32, samples: Vec<f32>) -> Self {
        Self { sample_rate, samples, block: 4096 }
    }
}

pub struct SampleClipStream<'a> {
    samples: &'a [f32],
    at: usize,
    block: usize,
}

impl AudioStream for SampleClipStream<'_> {
    fn next(&mut self) -> Result<Option<Vec<f32>>, String> {
        if self.at >= self.samples.len() {
            return Ok(None);
        }
        let end = (self.at + self.block).min(self.samples.len());
        let out = self.samples[self.at..end].to_vec();
        self.at = end;
        Ok(Some(out))
    }
}

impl AudioSource for SampleClip {
    fn info(&self) -> AudioInfo {
        AudioInfo {
            sample_rate: self.sample_rate,
            duration_hint: Some(self.samples.len() as f64 / self.sample_rate as f64),
        }
    }

    fn open(&self) -> Result<Box<dyn AudioStream + '_>, String> {
        Ok(Box::new(SampleClipStream {
            samples: &self.samples,
            at: 0,
            block: self.block.max(1),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(n: usize) -> SampleClip {
        SampleClip::new(48_000, (0..n).map(|i| i as f32 / n as f32).collect())
    }

    /// The property the two-pass consumers depend on: opening the same
    /// source twice must yield identical samples.
    #[test]
    fn two_streams_over_one_source_agree() {
        let c = clip(1000);
        let drain = |s: &dyn AudioSource| -> Vec<f32> {
            let mut st = s.open().expect("open");
            let mut out = Vec::new();
            while let Some(b) = st.next().expect("next") {
                out.extend_from_slice(&b);
            }
            out
        };
        assert_eq!(drain(&c), drain(&c));
        assert_eq!(drain(&c).len(), 1000);
    }

    #[test]
    fn a_drained_stream_stays_drained() {
        let c = clip(4);
        let mut s = c.open().expect("open");
        while s.next().expect("next").is_some() {}
        assert!(s.next().expect("still done").is_none());
        assert!(s.next().expect("still done").is_none());
    }

    #[test]
    fn an_empty_source_yields_nothing() {
        let c = SampleClip::new(48_000, Vec::new());
        assert!(c.open().expect("open").next().expect("next").is_none());
        assert_eq!(c.info().duration_hint, Some(0.0));
    }

    #[test]
    fn duration_is_derived_from_the_sample_count() {
        let c = SampleClip::new(48_000, vec![0.0; 24_000]);
        assert_eq!(c.info().duration_hint, Some(0.5));
    }

    /// Blocks are an implementation detail, never a contract: a consumer
    /// must be able to rely only on the concatenation, not on where the
    /// boundaries fall.
    #[test]
    fn block_size_does_not_change_the_concatenation() {
        let samples: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let a = SampleClip { sample_rate: 48_000, samples: samples.clone(), block: 7 };
        let b = SampleClip { sample_rate: 48_000, samples: samples.clone(), block: 999 };
        let drain = |s: &dyn AudioSource| -> Vec<f32> {
            let mut st = s.open().expect("open");
            let mut out = Vec::new();
            while let Some(x) = st.next().expect("next") {
                out.extend_from_slice(&x);
            }
            out
        };
        assert_eq!(drain(&a), drain(&b));
    }
}
