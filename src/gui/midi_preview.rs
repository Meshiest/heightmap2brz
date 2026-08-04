//! Audible preview playback for the MIDI pane, split by target: rodio on the
//! desktop, Web Audio in the browser. The pane synthesizes the score to mono
//! PCM off in [`crate::midi::preview::synthesize`] (pure, wasm-safe) and hands
//! the buffer here to be played -- so this module only owns the platform audio
//! device, never any synthesis.
//!
//! Both backends expose the same tiny surface -- `Preview::default()`,
//! `play(&mut self, pcm, sample_rate)` and `stop(&mut self)` -- so the pane
//! calls them without any `#[cfg]` of its own. A failed device init (no output
//! device, or the browser blocking autoplay before a user gesture) is logged at
//! warn level and swallowed: a preview that cannot play must never block
//! Generate, which does not touch this at all.

/// Sample rate the pane synthesizes and plays previews at. One place so the
/// buffer's rate and the device's rate can never disagree.
pub const SAMPLE_RATE: u32 = 44_100;

// --------------------------------------------------------------------------
// Desktop: rodio. The OutputStream MUST stay alive for the whole time sound is
// playing -- dropping it silences the sink -- so it lives in the struct beside
// the sink rather than in a local. The stream is opened lazily on the first
// play so a machine with no audio device only ever logs on a real preview
// attempt, not at startup.
// --------------------------------------------------------------------------
#[cfg(not(target_arch = "wasm32"))]
pub struct Preview {
    /// The output stream and its handle, opened on first use. Kept together so
    /// the stream (whose `Drop` stops playback) outlives every sink made from
    /// the handle.
    stream: Option<(rodio::OutputStream, rodio::OutputStreamHandle)>,
    /// The sink the current preview plays through, if any.
    sink: Option<rodio::Sink>,
    /// Playback volume, 0..=1. Applied to a new sink on `play` and live to the
    /// current sink via `set_volume`.
    volume: f32,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for Preview {
    fn default() -> Self {
        Self { stream: None, sink: None, volume: 1.0 }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Preview {
    /// Play `pcm` (mono, `sample_rate` Hz), replacing whatever was playing.
    /// A device that cannot be opened is logged and left alone -- the button
    /// simply does nothing rather than surfacing an error.
    pub fn play(&mut self, pcm: &[f32], sample_rate: u32) {
        if pcm.is_empty() {
            return;
        }
        if self.stream.is_none() {
            match rodio::OutputStream::try_default() {
                Ok(pair) => self.stream = Some(pair),
                Err(e) => {
                    log::warn!("MIDI preview: no audio output device ({e})");
                    return;
                }
            }
        }
        // `stream` was just ensured `Some`.
        let handle = &self.stream.as_ref().unwrap().1;
        let sink = match rodio::Sink::try_new(handle) {
            Ok(sink) => sink,
            Err(e) => {
                log::warn!("MIDI preview: could not open a playback sink ({e})");
                return;
            }
        };
        sink.set_volume(self.volume);
        sink.append(rodio::buffer::SamplesBuffer::new(1, sample_rate, pcm.to_vec()));
        sink.play();
        // Replaces (and so drops/stops) any previous sink.
        self.sink = Some(sink);
    }

    /// Set the playback volume (0..=1), live on the current preview if one is
    /// playing and stored for the next.
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume;
        if let Some(sink) = &self.sink {
            sink.set_volume(volume);
        }
    }

    /// Stop the current preview, if any.
    pub fn stop(&mut self) {
        if let Some(sink) = &self.sink {
            sink.stop();
        }
        self.sink = None;
    }
}

// --------------------------------------------------------------------------
// Browser: Web Audio. The AudioContext and the source node must both outlive
// the sound, so both are kept in the struct. Every call is fallible (autoplay
// is blocked until a user gesture, and a JsValue error is the browser's way of
// saying so); we log and no-op, exactly as the desktop path does for a missing
// device.
// --------------------------------------------------------------------------
#[cfg(target_arch = "wasm32")]
pub struct Preview {
    ctx: Option<web_sys::AudioContext>,
    src: Option<web_sys::AudioBufferSourceNode>,
    /// A gain node between the source and the destination, so volume is live.
    gain: Option<web_sys::GainNode>,
    /// Volume, 0..=1. Applied to a new gain node on `play` and live via
    /// `set_volume`.
    volume: f32,
}

#[cfg(target_arch = "wasm32")]
impl Default for Preview {
    fn default() -> Self {
        Self { ctx: None, src: None, gain: None, volume: 1.0 }
    }
}

#[cfg(target_arch = "wasm32")]
impl Preview {
    /// Play `pcm` (mono, `sample_rate` Hz), replacing whatever was playing.
    pub fn play(&mut self, pcm: &[f32], sample_rate: u32) {
        if pcm.is_empty() {
            return;
        }
        // Tear down any prior playback first, so a second click does not leave
        // an orphaned context running.
        self.stop();
        if let Err(e) = self.try_play(pcm, sample_rate) {
            log::warn!("MIDI preview (web audio) could not start: {e:?}");
            // Leave nothing half-built behind.
            self.stop();
        }
    }

    fn try_play(&mut self, pcm: &[f32], sample_rate: u32) -> Result<(), wasm_bindgen::JsValue> {
        let ctx = web_sys::AudioContext::new()?;
        let buf = ctx.create_buffer(1, pcm.len() as u32, sample_rate as f32)?;
        // copy_to_channel takes an immutable slice; the PCM the pane synthesized
        // is already exactly the mono channel we want.
        buf.copy_to_channel(pcm, 0)?;
        let src = ctx.create_buffer_source()?;
        src.set_buffer(Some(&buf));
        // source -> gain -> destination, so volume is a live control.
        let gain = ctx.create_gain()?;
        gain.gain().set_value(self.volume);
        src.connect_with_audio_node(&gain)?;
        gain.connect_with_audio_node(&ctx.destination())?;
        src.start()?;
        self.ctx = Some(ctx);
        self.src = Some(src);
        self.gain = Some(gain);
        Ok(())
    }

    /// Set the playback volume (0..=1), live on the current preview if one is
    /// playing and stored for the next.
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume;
        if let Some(gain) = &self.gain {
            gain.gain().set_value(volume);
        }
    }

    /// Stop the current preview, if any, and release its context.
    pub fn stop(&mut self) {
        if let Some(src) = &self.src {
            // Both `stop()` and `stop_with_when()` are marked deprecated in this
            // web-sys, with no non-deprecated replacement -- the no-arg form is
            // exactly what "stop now" wants.
            #[allow(deprecated)]
            let _ = src.stop();
        }
        if let Some(ctx) = &self.ctx {
            // Returns a Promise we do not need to await -- closing frees the
            // audio hardware; a failure here is nothing the user can act on.
            let _ = ctx.close();
        }
        self.src = None;
        self.gain = None;
        self.ctx = None;
    }
}
