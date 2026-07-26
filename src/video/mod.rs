use image::RgbaImage;

pub mod animated;
pub mod scale;
pub mod sequence;
pub mod source;

/// A decoded sequence of frames at a fixed size and frame rate.
///
/// This is the single contract between the decode layer (whatever reads a
/// gif/webp/apng/image sequence off disk or the wasm file picker) and the
/// rest of the pipeline (resizing, fps resampling, and eventually the
/// renderer). Every input source funnels into a `Clip`; every downstream
/// stage consumes and produces one.
#[derive(Clone, Debug)]
pub struct Clip {
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub frames: Vec<RgbaImage>,
}
