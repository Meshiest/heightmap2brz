//! Build-cost estimation. Gate count is what actually limits a build, so the
//! UI shows this before the user commits to a render.
use super::pack::{HEX_STRIDE, PIXELS_PER_CHUNK};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    pub pixels: usize,
    pub gates: usize,
    pub wires: usize,
    pub bricks: usize,
    pub chunks: usize,
    pub banks: usize,
    pub frames: usize,
    pub chars: usize,
}

/// Estimate the build cost of a `width * height` screen over `frames` frames,
/// spilling across arrays of at most `bank_size` frames each.
///
/// This is an **upper bound**, and the bound is not uniform across fields.
/// `gates`, `wires` and `bricks` assume every pixel survives; a pixel that is
/// fully transparent across the whole clip is culled and emits no display
/// brick and no gates, so a real render of a mostly-transparent clip can come
/// in well under these numbers. That divergence is by design — this feeds a UI
/// readout shown before a render is committed to, and it deliberately never
/// under-promises.
///
/// `chars` is **exact**, not a bound. A culled pixel still reserves its
/// [`HEX_STRIDE`] characters in every frame string, so that each surviving
/// pixel's offset stays a plain `pixel_in_chunk * HEX_STRIDE` with no remap
/// table — see [`super::pack`].
pub fn estimate(width: u32, height: u32, frames: usize, bank_size: usize) -> Cost {
    let pixels = width as usize * height as usize;
    let chunks = pixels.div_ceil(PIXELS_PER_CHUNK).max(1);
    let banks = frames.div_ceil(bank_size.max(1)).max(1);
    let boundaries = banks - 1;
    Cost {
        pixels,
        // 2 per pixel + 2 per chunk per bank (ArrayVar, Get) + 4 clock
        // + 1 detector, plus per boundary: comparator, branch, index subtract,
        // and one value select per chunk
        gates: 2 * pixels + 2 * chunks * banks + 5 + boundaries * 3 + boundaries * chunks,
        // 3 per pixel + 2 per chunk per bank + exec chain + detector feed
        // + 3 clock chain + 3 control pins, plus per boundary: comparator
        // InputA (1) + subtract InputA (1) + branch bCond/Exec (2) + one
        // select's bSelectB/InputA/InputB (3) per chunk
        wires: 3 * pixels
            + 2 * chunks * banks
            + (chunks * banks - 1)
            + 1
            + 3
            + 3
            + boundaries * (3 * chunks + 4),
        // one display brick per pixel + the microchip shell
        bricks: pixels + 1,
        chunks,
        banks,
        frames,
        chars: pixels * frames * HEX_STRIDE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::pack::BANK_FRAMES;

    #[test]
    fn matches_the_spec_formula() {
        // 64x36 = 2304 px; ceil(2304/1666) = 2 chunks
        // gates = 2*2304 + 2*2 + 5 = 4617
        let c = estimate(64, 36, 300, BANK_FRAMES);
        assert_eq!(c.pixels, 2304);
        assert_eq!(c.chunks, 2);
        assert_eq!(c.gates, 4617);
        // wires = 3/pixel + 2/chunk (ArrayVarRef, Index) + (chunks-1) exec chain
        //         + 1 detector feed + 3 clock chain + 3 control pins
        //       = 6912 + 4 + 1 + 1 + 3 + 3 = 6924
        assert_eq!(c.wires, 6924);
    }

    #[test]
    fn a_single_chunk_screen_needs_one_array() {
        assert_eq!(estimate(10, 10, 5, BANK_FRAMES).chunks, 1);
    }

    #[test]
    fn char_total_is_pixels_times_frames_times_stride() {
        assert_eq!(estimate(10, 10, 5, BANK_FRAMES).chars, 100 * 5 * 6);
    }

    #[test]
    fn the_chunk_boundary_is_exact() {
        assert_eq!(
            estimate(1666, 1, 2, BANK_FRAMES).chunks,
            1,
            "exactly one chunk's worth"
        );
        assert_eq!(
            estimate(1667, 1, 2, BANK_FRAMES).chunks,
            2,
            "one pixel over spills"
        );
    }

    #[test]
    fn a_sub_limit_clip_is_one_bank_and_costs_what_it_used_to() {
        let c = estimate(64, 36, 300, BANK_FRAMES);
        assert_eq!(c.banks, 1);
        assert_eq!(c.gates, 4617, "unchanged from the pre-spillover formula");
    }

    #[test]
    fn spilling_adds_three_gates_per_boundary_plus_one_select_per_chunk() {
        // 2304 px -> 2 chunks; 5 frames at bank size 2 -> 3 banks, 2 boundaries
        let one = estimate(64, 36, 5, 65_535);
        let three = estimate(64, 36, 5, 2);
        assert_eq!(three.banks, 3);
        // per boundary: comparator + branch + subtract, plus a select per chunk
        // plus the extra array/Get pair per chunk per extra bank
        let boundaries = 2;
        let chunks = 2;
        let extra = boundaries * 3 + boundaries * chunks + 2 * chunks * boundaries;
        assert_eq!(three.gates - one.gates, extra);
    }

    #[test]
    fn spilling_adds_the_boundary_wiring_to_the_wires_estimate() {
        // Same setup as the gates counterpart above: 2304 px -> 2 chunks;
        // 5 frames at bank size 2 -> 3 banks, 2 boundaries.
        let one = estimate(64, 36, 5, 65_535);
        let three = estimate(64, 36, 5, 2);
        assert_eq!(three.banks, 3);
        let boundaries = 2;
        let chunks = 2;
        // the extra array/Get pair per chunk per extra bank contributes 2
        // wires each (ArrayVarRef, Index), plus one more link in the exec
        // chain per extra bank per chunk; each boundary itself adds a
        // comparator InputA + subtract InputA + branch bCond/Exec (2) + one
        // select's bSelectB/InputA/InputB (3) per chunk
        let extra_bank_wires = 2 * chunks * boundaries + chunks * boundaries;
        let extra_boundary_wires = boundaries * (3 * chunks + 4);
        assert_eq!(three.wires - one.wires, extra_bank_wires + extra_boundary_wires);
        assert_eq!(three.wires - one.wires, 32, "12 bank wires + 20 boundary wires");
    }

    #[test]
    fn banking_moves_characters_between_arrays_without_changing_the_total() {
        assert_eq!(estimate(64, 36, 100, 65_535).chars, estimate(64, 36, 100, 7).chars);
    }
}
