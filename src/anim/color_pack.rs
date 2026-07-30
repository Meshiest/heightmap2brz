//! Pixel-major linear-colour packing: turns a stream of frames into the
//! per-pixel colour arrays a Brickadia microchip reads in colour-array mode.
//!
//! This is the sibling of [`super::pack`], and the two are laid out along
//! OPPOSITE axes. Hex mode is **frame-major**: one array entry per frame,
//! each holding every pixel's colour as text, tiled into
//! [`super::pack::PIXELS_PER_CHUNK`]-pixel chunks so an entry fits the
//! component character limit. Colour-array mode is **pixel-major**: one array
//! per *pixel*, holding that pixel's own colour for every frame. There is no
//! chunking at all -- the character limit that forced it does not apply to a
//! `WireGraphLinearColorArray`, whose length is bounded only by
//! [`super::pack::BANK_FRAMES`] like every other wire array.
//!
//! ## Colour space
//!
//! `WireArrayVariant::LinearColorArray` is documented in `brdb` as `(R, G, B,
//! A)` f32 elements, **linear 0-1**. Video frames are sRGB. Unlike the hex
//! path -- where the in-game `MakeColorHex` gate does its own conversion from
//! whatever bytes it is handed -- nothing downstream of this module transforms
//! colour, so the sRGB -> linear transfer MUST happen here or the whole render
//! comes out a gamma step too bright. That is [`crate::util::to_linear_rgb_f32`],
//! the float version: `to_linear_rgb` quantizes back to `u8` and would
//! collapse every linear value below `1/255` to zero, crushing shadows.
//!
//! Because the conversion is mandatory rather than optional here, this path
//! ignores `AnimOptions::srgb_to_linear` (which exists for the hex path, where
//! it genuinely is a question about what the game's gate expects).
//!
//! ## Memory
//!
//! **This mode's accumulator retains 16 bytes per pixel per frame** -- four
//! `f32` -- against hex mode's 6 (one `RRGGBB` per pixel per frame). Both are
//! streaming, in the sense that no *image* is ever retained, but the encoded
//! form is what dominates a long render and colour-array mode's is ~2.7x
//! larger. A 128x72 clip of 65 535 frames is ~9.7 GB here versus ~3.6 GB in
//! hex mode. That raises the memory floor on long renders and is the main
//! cost of the encoding; the gate-count and per-frame-work savings are what
//! it buys.
//!
//! That 16-bytes-per-pixel-per-frame figure is only true because
//! [`ColorPacker::new`] **reserves each pixel's `Vec` exactly**. It did not
//! always: with a plain `Vec::new()` grown one `push` per frame, Rust's
//! amortized doubling leaves every inner `Vec` at a capacity of
//! `next_power_of_two(frames)` (minimum 4), so the process really held
//! `pixels * next_power_of_two(frames) * 16`. At 128x72 over 40,000 frames
//! that is 9.66 GB actually retained against the 5.90 GB this figure -- and
//! the CLI readout built on it -- reported, and the worst case, one frame past
//! a power of two, is just over 2x. An under-reported memory figure is how a
//! user starts an hour-long render that then gets OOM-killed, so the
//! reservation is load-bearing rather than a micro-optimisation (it also
//! removes the reallocation churn: a 65 535-frame render otherwise re-copies
//! every pixel's array 16 times).
//!
//! [`super::color_bricks::unreserved_accumulator_bytes`] is what the figure
//! would be without the reservation, and
//! `the_reservation_is_what_makes_the_memory_figure_true` pins the two apart.
//!
//! The headline example above happens to be accurate under EITHER scheme:
//! 65 535 is one *under* a power of two, so its doubling slack was 0.0015%.
//! Every other frame count was worse, which is exactly why the example never
//! showed the problem.
use image::RgbaImage;

/// One array element: linear `(R, G, B, A)`, matching
/// `WireArrayVariant::LinearColorArray`'s element type exactly so a finished
/// pixel's `Vec` can be handed to it with no conversion.
pub type LinearColor = (f32, f32, f32, f32);

/// What a pixel below the alpha threshold contributes to a frame.
///
/// Fully transparent black -- the colour-array analogue of the `"000000"` the
/// hex path writes into a culled pixel's slot. Its colour carries no meaning
/// in that frame, since a pixel culled in EVERY frame gets no display brick at
/// all and one culled in only *some* frames is being told not to show
/// anything. Zeroed alpha rather than hex's implicit opaque black, because
/// unlike a 6-character hex string this encoding actually has an alpha
/// channel to say it with.
pub const CULLED: LinearColor = (0.0, 0.0, 0.0, 0.0);

/// Builds the per-pixel colour arrays and the per-pixel visibility bitmap in
/// ONE traversal of the frames, retaining no frame -- the same contract
/// [`super::pack::Packer`] holds for hex mode, so both renderers can share
/// `build_*_world`'s single streaming pull loop.
///
/// Pixels are indexed row-major (`index = row * width + col`), the same order
/// [`super::pack`] uses and the same order the display-brick loop walks.
pub struct ColorPacker {
    width: usize,
    height: usize,
    alpha_threshold: u8,
    /// `pixels[pixel_index][frame]`. Pixel-major: the inner `Vec` IS what one
    /// `ArrayVar` will hold, so no transposition happens later.
    pixels: Vec<Vec<LinearColor>>,
    visible: Vec<bool>,
    frames_pushed: usize,
}

impl ColorPacker {
    /// `frame_count_hint` is the source's own `SourceInfo::frame_count_hint`,
    /// and it is what keeps this mode's memory figure honest -- see the
    /// module's Memory section. Each pixel's array is reserved EXACTLY that
    /// long, so the accumulator holds `pixels * frames * 16` bytes and not the
    /// `pixels * next_power_of_two(frames) * 16` that one `push` per frame
    /// into an unreserved `Vec` would leave it at.
    ///
    /// `None` (a source that cannot report its length ahead of decode) falls
    /// back to that doubling, which is the only thing available without a
    /// length -- [`super::color_bricks::unreserved_accumulator_bytes`] is the
    /// figure for that case.
    ///
    /// A hint that turns out to be WRONG is safe either way: too small and the
    /// tail doubles from there, too large and the excess is capacity that was
    /// never touched. It is a reservation, not a limit -- `push_frame` still
    /// accepts as many frames as the source produces, up to
    /// [`super::pack::MAX_FRAMES`].
    pub fn new(
        width: u32,
        height: u32,
        alpha_threshold: u8,
        frame_count_hint: Option<usize>,
    ) -> Self {
        let total_pixels = width as usize * height as usize;
        let capacity = frame_count_hint.unwrap_or(0);
        Self {
            width: width as usize,
            height: height as usize,
            alpha_threshold,
            // Built one at a time, NOT `vec![Vec::with_capacity(n); total]`:
            // that clones the prototype, and `Vec::clone` allocates for its
            // LENGTH, not its capacity -- so the macro form would hand back
            // `total` empty, capacity-0 vectors and silently undo the whole
            // reservation. `with_capacity(0)` does not allocate, so the
            // no-hint case costs nothing.
            pixels: (0..total_pixels).map(|_| Vec::with_capacity(capacity)).collect(),
            visible: vec![false; total_pixels],
            frames_pushed: 0,
        }
    }

    /// Push one frame's contribution to every pixel's colour array.
    ///
    /// `frame` MUST be exactly `width x height` -- the dimensions `new` was
    /// built with, which are the ones the caller's `SourceInfo` reported (see
    /// [`crate::video::stream::FrameStream::next`]'s contract). Checked here,
    /// before any pixel is read, for the same reason
    /// [`super::pack::Packer::push_frame`] checks: an undersized frame would
    /// otherwise panic deep in `get_pixel` and an oversized one would decode
    /// silently with its excess ignored.
    pub fn push_frame(&mut self, frame: &RgbaImage) -> Result<(), String> {
        if self.frames_pushed >= super::pack::MAX_FRAMES {
            return Err(format!(
                "clip exceeds the {}-frame limit ({} banks of {})",
                super::pack::MAX_FRAMES,
                super::pack::MAX_BANKS,
                super::pack::BANK_FRAMES
            ));
        }
        let (frame_w, frame_h) = (frame.width() as usize, frame.height() as usize);
        if frame_w != self.width || frame_h != self.height {
            return Err(format!(
                "frame {} is {frame_w}x{frame_h}, but the source's SourceInfo reported \
                 {}x{} -- every frame a FrameStream emits must match info()'s dimensions",
                self.frames_pushed, self.width, self.height
            ));
        }
        // Pixel `idx` is `raw[idx * 4..][..4]`: `RgbaImage`'s buffer is
        // row-major RGBA with no padding, which is the same order `pixels` is
        // in. Walking the two together replaces the `%`, the `/` and the
        // bounds-checked `get_pixel` this used to do per pixel per frame with
        // a plain zip. The dimension check above is what makes the two
        // lengths agree.
        let raw = frame.as_raw();
        debug_assert_eq!(raw.len(), self.pixels.len() * 4);
        {
            let Self { pixels, visible, alpha_threshold, .. } = self;
            let alpha_threshold = *alpha_threshold;
            let lut = crate::util::srgb_to_linear_f32_table();
            push_pixels(pixels, visible, raw, alpha_threshold, lut);
        }
        self.frames_pushed += 1;
        Ok(())
    }

    /// The per-pixel colour arrays (row-major, one `Vec` per pixel, each
    /// exactly as long as the number of frames pushed), plus row-major
    /// per-pixel visibility (`true` where the pixel was opaque enough in at
    /// least one frame).
    pub fn finish(self) -> (Vec<Vec<LinearColor>>, Vec<bool>) {
        (self.pixels, self.visible)
    }
}

/// Push one frame's colour into every pixel's array.
///
/// `raw` is the frame's row-major RGBA buffer, so pixel `i` is
/// `raw[i * 4..][..4]` and the three slices simply zip -- no `%`, no `/`, no
/// bounds-checked `get_pixel`. `lut` is
/// [`crate::util::srgb_to_linear_f32_table`], whose entries are bit-identical
/// to [`to_linear_rgb_f32`]'s own `powf` (pinned by
/// `the_f32_linear_table_is_exactly_the_function`), so this is a speed change
/// and never a colour change.
///
/// **Deliberately serial, unlike [`super::pack::Packer`]'s per-chunk fan-out.**
/// It was written parallel first and measured: a rayon walk over these same
/// three zipped slices was **11% SLOWER** at 192x108 (0.107s against 0.096s
/// for 400 frames) and only won at sizes this mode cannot reach anyway -- 2.2x
/// at 640x360, which at 16 bytes per pixel per frame is 3.7 MB of accumulator
/// *per frame* and blows the memory ceiling in the module doc long before the
/// speed matters. The reason the hex packer's fan-out pays and this one does
/// not is the access pattern: a hex chunk is ~1 666 pixels of contiguous
/// string building, while this scatters one 16-byte push into each of tens of
/// thousands of separate heap allocations, where the limit is memory latency
/// rather than CPU and extra threads buy nothing. So it is not here.
fn push_pixels(
    pixels: &mut [Vec<LinearColor>],
    visible: &mut [bool],
    raw: &[u8],
    alpha_threshold: u8,
    lut: &[f32; 256],
) {
    for ((colors, vis), p) in pixels.iter_mut().zip(visible.iter_mut()).zip(raw.chunks_exact(4))
    {
        if p[3] < alpha_threshold {
            colors.push(CULLED);
        } else {
            *vis = true;
            // Alpha is a plain 0..=1 rescale, never transferred -- see
            // `to_linear_rgb_f32`, which this must stay identical to.
            colors.push((
                lut[p[0] as usize],
                lut[p[1] as usize],
                lut[p[2] as usize],
                p[3] as f32 / 255.0,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::srgb_to_linear_f32;
    use image::Rgba;

    fn frame(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> RgbaImage {
        RgbaImage::from_fn(w, h, |x, y| Rgba(f(x, y)))
    }

    /// The core shape guarantee: one array per pixel, each exactly as long as
    /// the clip. A renderer wiring an `ArrayVar_Get` at frame index N into an
    /// array shorter than N reads out of bounds in-game.
    #[test]
    fn every_pixels_array_is_exactly_the_frame_count_long() {
        let mut p = ColorPacker::new(3, 2, 128, None);
        for f in 0..7u32 {
            p.push_frame(&frame(3, 2, |_, _| [f as u8, 0, 0, 255])).unwrap();
        }
        let (pixels, _) = p.finish();
        assert_eq!(pixels.len(), 6, "one array per pixel of a 3x2 screen");
        for (i, colors) in pixels.iter().enumerate() {
            assert_eq!(colors.len(), 7, "pixel {i}'s array must hold every frame");
        }
    }

    /// A zero-frame clip still reserves every pixel's array -- empty, not
    /// missing. The renderer rejects a zero-frame source before it gets this
    /// far, but a `Vec` short by one pixel would be an indexing panic there
    /// rather than the clear error it emits.
    #[test]
    fn a_zero_frame_clip_still_reserves_every_pixels_array() {
        let (pixels, visible) = ColorPacker::new(4, 3, 128, None).finish();
        assert_eq!(pixels.len(), 12);
        assert!(pixels.iter().all(|c| c.is_empty()));
        assert_eq!(visible, vec![false; 12]);
    }

    /// **The value test.** Pixel p at frame f must hold exactly that pixel's
    /// own source colour, converted -- not a neighbour's, not a different
    /// frame's, and not the raw sRGB bytes. Every pixel and every frame gets a
    /// distinct colour so a transposition, an off-by-one or a missing
    /// conversion all show up as a mismatch.
    #[test]
    fn pixel_p_at_frame_f_holds_that_pixels_own_converted_colour() {
        let (w, h, n) = (5u32, 3u32, 4u32);
        let source = |x: u32, y: u32, f: u32| -> [u8; 4] {
            [(x * 17 + f) as u8, (y * 53 + f * 7) as u8, (x * 31 + y * 11 + f * 3) as u8, 255]
        };
        let mut p = ColorPacker::new(w, h, 128, None);
        for f in 0..n {
            p.push_frame(&frame(w, h, |x, y| source(x, y, f))).unwrap();
        }
        let (pixels, _) = p.finish();
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                for f in 0..n {
                    let want = source(x, y, f);
                    let got = pixels[idx][f as usize];
                    assert_eq!(
                        got,
                        (
                            srgb_to_linear_f32(want[0]),
                            srgb_to_linear_f32(want[1]),
                            srgb_to_linear_f32(want[2]),
                            1.0
                        ),
                        "pixel (col {x}, row {y}) frame {f}"
                    );
                }
            }
        }
    }

    /// The conversion is not optional here and must actually have happened.
    /// Mid-grey is the clearest witness: sRGB 128 is linear ~0.216, and a path
    /// that forgot to convert would hand the array ~0.502 instead -- the
    /// gamma-step-too-bright render this mode exists to avoid.
    #[test]
    fn colours_are_linear_not_raw_srgb() {
        let mut p = ColorPacker::new(1, 1, 128, None);
        p.push_frame(&frame(1, 1, |_, _| [128, 128, 128, 255])).unwrap();
        let (pixels, _) = p.finish();
        let (r, _, _, _) = pixels[0][0];
        assert!(
            (r - 0.2158).abs() < 1e-3,
            "sRGB 128 must arrive as linear ~0.216, got {r} (0.502 means no conversion ran)"
        );
    }

    /// Row-major indexing, checked on a NON-square screen so the two
    /// candidate formulas actually differ: `row * width + col` puts (col 2,
    /// row 1) of a 5-wide screen at index 7, while the column-major
    /// transposition would put it at 5.
    #[test]
    fn pixels_are_indexed_row_major() {
        let mut p = ColorPacker::new(5, 2, 128, None);
        p.push_frame(&frame(5, 2, |x, y| {
            if (x, y) == (2, 1) { [255, 0, 0, 255] } else { [0, 0, 0, 0] }
        }))
        .unwrap();
        let (pixels, visible) = p.finish();
        assert_eq!(pixels[7][0], (1.0, 0.0, 0.0, 1.0), "(col 2, row 1) is index 7");
        assert!(visible[7]);
        assert_eq!(visible.iter().filter(|v| **v).count(), 1);
    }

    /// A pixel below the threshold contributes a transparent-black slot for
    /// that frame and does not become visible -- but a pixel opaque in even one
    /// frame stays visible for the whole render, because its display brick has
    /// to exist to show that one frame.
    #[test]
    fn a_pixel_opaque_in_any_single_frame_is_visible_and_culled_slots_are_transparent() {
        let mut p = ColorPacker::new(2, 1, 128, None);
        // Pixel 0 opaque only in frame 1; pixel 1 never opaque.
        p.push_frame(&frame(2, 1, |_, _| [9, 9, 9, 0])).unwrap();
        p.push_frame(&frame(2, 1, |x, _| {
            if x == 0 { [255, 255, 255, 255] } else { [9, 9, 9, 127] }
        }))
        .unwrap();
        let (pixels, visible) = p.finish();
        assert_eq!(visible, vec![true, false]);
        assert_eq!(pixels[0][0], CULLED, "a culled frame is transparent black");
        assert_eq!(pixels[0][1], (1.0, 1.0, 1.0, 1.0));
        assert_eq!(pixels[1], vec![CULLED, CULLED]);
    }

    /// The threshold is `>=`, matching `pack::Packer` exactly. A `>` here
    /// would cull the boundary alpha in one mode and keep it in the other, so
    /// the two encodings would disagree about which pixels exist.
    #[test]
    fn the_alpha_threshold_boundary_matches_the_hex_packer() {
        let mut p = ColorPacker::new(3, 1, 128, None);
        p.push_frame(&frame(3, 1, |x, _| [200, 100, 50, [127u8, 128, 129][x as usize]]))
            .unwrap();
        let (_, visible) = p.finish();
        assert_eq!(
            visible,
            vec![false, true, true],
            "alpha == threshold must count as visible, the same way pack::Packer treats it"
        );

        // Cross-checked against the real hex packer rather than asserted from
        // memory, so the two can never drift apart.
        let mut hex = super::super::pack::Packer::new(3, 1, 128, super::super::pack::HEX_STRIDE);
        hex.push_frame(&frame(3, 1, |x, _| {
            [200, 100, 50, [127u8, 128, 129][x as usize]]
        }))
        .unwrap();
        let (_, hex_visible) = hex.finish();
        assert_eq!(hex_visible, visible, "both packers must cull exactly the same pixels");
    }

    /// The dimension contract, checked before any pixel is read. Without it an
    /// undersized frame panics inside `get_pixel` and an oversized one is
    /// silently cropped.
    #[test]
    fn a_frame_of_the_wrong_size_is_a_descriptive_error() {
        let mut p = ColorPacker::new(4, 3, 128, None);
        let err = p
            .push_frame(&frame(4, 2, |_, _| [0, 0, 0, 255]))
            .expect_err("a 4x2 frame must not be accepted by a 4x3 packer");
        assert!(err.contains("4x2"), "the error must name the actual size: {err}");
        assert!(err.contains("4x3"), "the error must name the expected size: {err}");
    }

    /// Banking is `pack::bank_frames`, generic over the element type, so the
    /// seam behaviour is literally the same code the hex path uses: the last
    /// bank is short rather than padded (padding would play phantom frames),
    /// and the concatenation is the original list.
    #[test]
    fn banking_splits_a_pixels_colours_at_the_seam_without_padding() {
        let mut p = ColorPacker::new(1, 1, 128, None);
        for f in 0..7u32 {
            p.push_frame(&frame(1, 1, |_, _| [f as u8 * 10, 0, 0, 255])).unwrap();
        }
        let (pixels, _) = p.finish();
        let banks = super::super::pack::bank_frames(&pixels[0], 3);
        assert_eq!(
            banks.iter().map(|b| b.len()).collect::<Vec<_>>(),
            vec![3, 3, 1],
            "7 frames at bank size 3 -> 3 + 3 + 1, last one short, never padded"
        );
        let flat: Vec<LinearColor> = banks.iter().flat_map(|b| b.iter().copied()).collect();
        assert_eq!(flat, pixels[0], "the banks must reassemble into the original list");
        // The seam itself: bank 1 must START at frame 3, not repeat frame 0.
        assert_eq!(banks[1][0], pixels[0][3], "bank 1 element 0 is global frame 3");
        assert_eq!(banks[2][0], pixels[0][6], "bank 2 element 0 is global frame 6");
    }
}
