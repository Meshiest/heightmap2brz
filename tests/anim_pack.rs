use heightmap::anim::pack::{Chunk, HEX_STRIDE, PIXELS_PER_CHUNK, Packer, pack, slice_of};
use heightmap::video::Clip;
use image::{Rgba, RgbaImage};

fn clip_from(frames: Vec<Vec<[u8; 4]>>, w: u32, h: u32) -> Clip {
    let frames = frames
        .into_iter()
        .map(|px| {
            let mut img = RgbaImage::new(w, h);
            for (i, c) in px.into_iter().enumerate() {
                img.put_pixel(i as u32 % w, i as u32 / w, Rgba(c));
            }
            img
        })
        .collect();
    Clip { width: w, height: h, fps: 10.0, frames }
}

#[test]
fn packs_one_frame_row_major_as_uppercase_hex_without_hash() {
    let clip = clip_from(
        vec![vec![[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 255, 255]]],
        2,
        2,
    );
    let chunks = pack(&clip, 128).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].frames[0], "FF000000FF000000FFFFFFFF");
    assert!(!chunks[0].frames[0].contains('#'), "MakeColorHex takes bare RRGGBB");
}

#[test]
fn every_entry_is_exactly_stride_times_pixel_count() {
    let clip = clip_from(vec![vec![[1, 2, 3, 255]; 6], vec![[4, 5, 6, 255]; 6]], 3, 2);
    let chunks = pack(&clip, 128).unwrap();
    for c in &chunks {
        for f in &c.frames {
            assert_eq!(f.chars().count(), c.pixel_count * HEX_STRIDE);
        }
    }
}

#[test]
fn one_array_entry_per_frame() {
    let clip = clip_from(vec![vec![[0, 0, 0, 255]; 4]; 7], 2, 2);
    let chunks = pack(&clip, 128).unwrap();
    assert_eq!(chunks[0].frames.len(), 7);
}

/// The invariant everything else rests on.
#[test]
fn slicing_recovers_every_pixel_of_every_frame() {
    let w = 5u32;
    let h = 3u32;
    let frames: Vec<Vec<[u8; 4]>> = (0..4)
        .map(|f| (0..(w * h)).map(|i| [(i * 7) as u8, f as u8 * 40, (i + f) as u8, 255]).collect())
        .collect();
    let clip = clip_from(frames.clone(), w, h);
    let chunks = pack(&clip, 128).unwrap();

    for (fi, frame) in frames.iter().enumerate() {
        for (pi, px) in frame.iter().enumerate() {
            let chunk = chunks
                .iter()
                .find(|c: &&Chunk| pi >= c.first_pixel && pi < c.first_pixel + c.pixel_count)
                .expect("every pixel belongs to a chunk");
            let got = slice_of(chunk, fi, pi - chunk.first_pixel);
            let want = format!("{:02X}{:02X}{:02X}", px[0], px[1], px[2]);
            assert_eq!(got, want, "frame {fi} pixel {pi}");
        }
    }
}

#[test]
fn chunks_split_at_the_ten_thousand_char_limit_and_tile_the_screen() {
    let n = PIXELS_PER_CHUNK * 2 + 5;
    let clip = clip_from(vec![vec![[9, 9, 9, 255]; n]], n as u32, 1);
    let chunks = pack(&clip, 128).unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].pixel_count, PIXELS_PER_CHUNK);
    assert_eq!(chunks[2].pixel_count, 5);
    // contiguous, no gaps, no overlap
    let mut next = 0;
    for c in &chunks {
        assert_eq!(c.first_pixel, next);
        next += c.pixel_count;
        assert!(c.frames[0].chars().count() <= 10_000);
    }
    assert_eq!(next, n);
}

#[test]
fn culled_pixels_still_reserve_their_slots() {
    let clip = clip_from(
        vec![vec![[255, 0, 0, 255], [0, 0, 0, 0], [0, 0, 255, 255], [1, 2, 3, 255]]],
        2,
        2,
    );
    let chunks = pack(&clip, 128).unwrap();
    assert_eq!(chunks[0].pixel_count, 4, "transparent pixels keep their slot");
    assert_eq!(slice_of(&chunks[0], 0, 2), "0000FF", "offsets stay index*6");
}

#[test]
fn exceeding_the_array_limit_is_an_error() {
    let clip = Clip {
        width: 1,
        height: 1,
        fps: 10.0,
        frames: vec![RgbaImage::new(1, 1); MAX_FRAMES + 1],
    };
    let err = pack(&clip, 128).unwrap_err();
    assert!(err.contains(&MAX_FRAMES.to_string()), "got: {err}");
}

// --- Beyond the brief: edge cases it deliberately leaves unpinned ---------

/// The brief's own `culled_pixels_still_reserve_their_slots` picks a culled
/// pixel whose real color already happens to be `000000`, so it cannot tell
/// whether a culled slot is blacked out or left at its source color. This
/// pins the choice made here: a pixel below threshold in a given frame
/// encodes as `000000` in *that* frame's string (its real color carries no
/// meaning there -- nothing displays it), independent of what it encodes to
/// in frames where it clears the threshold.
#[test]
fn a_culled_pixel_encodes_as_black_not_its_source_color() {
    let clip = clip_from(vec![vec![[200, 100, 50, 0]]], 1, 1);
    let chunks = pack(&clip, 128).unwrap();
    assert_eq!(
        slice_of(&chunks[0], 0, 0),
        "000000",
        "a culled pixel's real (non-black) color must not leak into its slot"
    );
}

/// Pins the boundary direction: alpha exactly equal to the threshold counts
/// as opaque enough (`>=`), matching the codebase's existing convention in
/// `text.rs` (`p[3] < opts.alpha_threshold` is the *culled* branch, so `>=`
/// survives) -- not the stricter `>` a naive reading of "below the threshold"
/// might suggest for the excluded side.
#[test]
fn alpha_exactly_at_threshold_is_kept_one_below_is_culled() {
    let clip = clip_from(vec![vec![[10, 20, 30, 127], [10, 20, 30, 128]]], 2, 1);
    let chunks = pack(&clip, 128).unwrap();
    assert_eq!(
        slice_of(&chunks[0], 0, 0),
        "000000",
        "alpha 127 is strictly below threshold 128: culled"
    );
    assert_eq!(
        slice_of(&chunks[0], 0, 1),
        "0A141E",
        "alpha == threshold is kept: boundary is >=, not >"
    );
}

/// `alpha_threshold: 0` is a legal (if degenerate) input: every pixel,
/// including fully transparent ones, clears `alpha >= 0` and keeps its real
/// color.
#[test]
fn alpha_threshold_zero_keeps_even_fully_transparent_pixels() {
    let clip = clip_from(vec![vec![[1, 2, 3, 0]]], 1, 1);
    let chunks = pack(&clip, 0).unwrap();
    assert_eq!(slice_of(&chunks[0], 0, 0), "010203");
}

/// A clip with frames but zero pixels (a zero-width or zero-height image)
/// has nothing to tile: `pack` must not divide by the zero dimension or
/// otherwise misbehave, it should simply report no chunks.
#[test]
fn zero_dimension_clip_packs_into_no_chunks() {
    let clip = Clip {
        width: 0,
        height: 5,
        fps: 10.0,
        frames: vec![RgbaImage::new(0, 5)],
    };
    let chunks = pack(&clip, 128).unwrap();
    assert!(chunks.is_empty(), "a screen with zero pixels needs zero chunks");

    let clip = Clip {
        width: 5,
        height: 0,
        fps: 10.0,
        frames: vec![RgbaImage::new(5, 0)],
    };
    let chunks = pack(&clip, 128).unwrap();
    assert!(chunks.is_empty(), "a screen with zero pixels needs zero chunks");
}

/// A clip with zero frames still has a screen to tile: chunking is driven by
/// pixel count, not frame count, so the pixel slots exist with simply no
/// frame strings filling them (nothing to encode from).
#[test]
fn zero_frame_clip_reserves_pixel_slots_with_no_frame_strings() {
    let clip = clip_from(vec![], 2, 2);
    let chunks = pack(&clip, 128).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].pixel_count, 4);
    assert!(chunks[0].frames.is_empty(), "no frames means no frame strings");
}

/// 65535 frames was the boundary of the legal range before spillover widened
/// `MAX_FRAMES` to `BANK_FRAMES * MAX_BANKS`; it is now comfortably inside
/// the (much larger) legal range rather than the boundary itself, but still
/// worth pinning as a non-error.
#[test]
fn exactly_the_frame_limit_is_not_an_error() {
    let clip = Clip {
        width: 0,
        height: 0,
        fps: 10.0,
        frames: vec![RgbaImage::new(0, 0); 65_535],
    };
    assert!(pack(&clip, 128).is_ok(), "65535 frames is within the limit");
}

use heightmap::anim::pack::{BANK_FRAMES, MAX_BANKS, MAX_FRAMES, bank_frames};

fn frames(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("f{i}")).collect()
}

/// The seams are where off-by-ones live. Tested at a bank size of 3 rather
/// than 65535 because building a real 65536-entry clip costs minutes of CPU
/// and gigabytes of RAM -- which is exactly why `bank_frames` takes its size
/// as a parameter.
#[test]
fn banks_split_exactly_at_the_boundary() {
    let cases = [
        (2usize, 3usize, vec![2usize]),
        (3, 3, vec![3]),
        (4, 3, vec![3, 1]),
        (6, 3, vec![3, 3]),
        (7, 3, vec![3, 3, 1]),
    ];
    for (n, size, want) in cases {
        let f = frames(n);
        let got: Vec<usize> = bank_frames(&f, size).iter().map(|b| b.len()).collect();
        assert_eq!(got, want, "{n} frames at bank size {size}");
    }
}

/// A short final bank must stay short. Padding it would play phantom frames
/// at the end of the clip.
#[test]
fn the_last_bank_is_short_not_padded() {
    let f = frames(7);
    let banks = bank_frames(&f, 3);
    assert_eq!(banks.last().unwrap().len(), 1, "7 = 3 + 3 + 1");
}

/// Concatenating the banks must reproduce the original list -- no frame
/// dropped or duplicated at a seam.
#[test]
fn every_frame_survives_banking_exactly_once() {
    let f = frames(1000);
    let flat: Vec<&String> = bank_frames(&f, 7).into_iter().flatten().collect();
    assert_eq!(flat.len(), f.len());
    for (i, s) in flat.iter().enumerate() {
        assert_eq!(**s, f[i], "frame {i} moved during banking");
    }
}

/// The wiring always needs at least one bank to hang an array off, even for
/// a degenerate frame list.
#[test]
fn an_empty_frame_list_still_yields_one_bank() {
    let f: Vec<String> = Vec::new();
    let banks = bank_frames(&f, 3);
    assert_eq!(banks.len(), 1);
    assert!(banks[0].is_empty());
}

#[test]
fn the_overall_cap_is_banks_times_bank_size() {
    assert_eq!(BANK_FRAMES, 65_535);
    assert_eq!(MAX_BANKS, 16);
    assert_eq!(MAX_FRAMES, 65_535 * 16);
}

/// Over the absolute ceiling `pack` must refuse, naming both numbers, rather
/// than silently truncating the clip.
///
/// Uses a 1x1 clip so a million frames costs a million 6-char strings rather
/// than a million images.
#[test]
fn over_the_absolute_cap_errors_naming_both_limits() {
    let img = image::RgbaImage::from_pixel(1, 1, image::Rgba([1, 2, 3, 255]));
    let clip = heightmap::video::Clip {
        width: 1,
        height: 1,
        fps: 10.0,
        frames: vec![img; MAX_FRAMES + 1],
    };
    let err = heightmap::anim::pack::pack(&clip, 128).expect_err("must refuse");
    assert!(err.contains(&MAX_FRAMES.to_string()), "error must name the cap: {err}");
    assert!(err.contains(&BANK_FRAMES.to_string()), "error must name the bank size: {err}");
}

// --- Task 3: the fused Packer ---------------------------------------------

/// The regression gate for the whole change: the fused packer must agree with
/// today's two-pass `pack` + `visible` byte for byte. Uses a clip with a
/// pixel that is transparent in some frames and opaque in others, plus one
/// transparent throughout, so visibility is not trivially all-true.
#[test]
fn the_fused_packer_matches_the_two_pass_result_exactly() {
    let (w, h, n) = (5u32, 4u32, 6usize);
    let mut frames = Vec::new();
    for f in 0..n {
        let mut img = image::RgbaImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                // pixel (0,0) is never opaque; (1,0) only on odd frames
                let a = match (x, y) {
                    (0, 0) => 0u8,
                    (1, 0) => if f % 2 == 1 { 255 } else { 0 },
                    _ => 255,
                };
                img.put_pixel(x, y, image::Rgba([(x * 7) as u8, (y * 11) as u8, f as u8, a]));
            }
        }
        frames.push(img);
    }
    let clip = heightmap::video::Clip { width: w, height: h, fps: 10.0, frames };

    let want_chunks = heightmap::anim::pack::pack(&clip, 128).expect("pack");
    let want_visible: Vec<bool> = (0..(w * h) as usize)
        .map(|i| {
            let (col, row) = ((i as u32) % w, (i as u32) / w);
            clip.frames.iter().any(|f| f.get_pixel(col, row).0[3] >= 128)
        })
        .collect();

    let mut p = Packer::new(w, h, 128, HEX_STRIDE);
    for f in &clip.frames {
        p.push_frame(f).expect("push");
    }
    let (got_chunks, got_visible) = p.finish();

    assert_eq!(got_chunks.len(), want_chunks.len(), "chunk count");
    for (g, wc) in got_chunks.iter().zip(&want_chunks) {
        assert_eq!(g.first_pixel, wc.first_pixel);
        assert_eq!(g.pixel_count, wc.pixel_count);
        assert_eq!(g.frames, wc.frames, "frame strings must be byte-identical");
    }
    assert_eq!(got_visible, want_visible, "visibility must match the old scan");
    assert!(!got_visible[0], "a never-opaque pixel must be invisible");
    assert!(got_visible[1], "a sometimes-opaque pixel must be visible");
}

/// A culled pixel still reserves its stride in every frame string, so every
/// surviving pixel's offset stays a plain `pixel_in_chunk * stride`.
#[test]
fn a_culled_pixel_still_reserves_its_slot() {
    let img = image::RgbaImage::from_pixel(3, 1, image::Rgba([0, 0, 0, 0]));
    let mut p = Packer::new(3, 1, 128, HEX_STRIDE);
    p.push_frame(&img).expect("push");
    let (chunks, visible) = p.finish();
    assert_eq!(chunks[0].frames[0].len(), 3 * HEX_STRIDE);
    assert_eq!(visible, vec![false; 3]);
}

#[test]
fn pushing_past_the_frame_cap_errors() {
    let img = image::RgbaImage::from_pixel(1, 1, image::Rgba([1, 2, 3, 255]));
    let mut p = Packer::new(1, 1, 128, HEX_STRIDE);
    for _ in 0..heightmap::anim::pack::MAX_FRAMES {
        p.push_frame(&img).expect("within the cap");
    }
    assert!(p.push_frame(&img).is_err(), "one past the cap must error");
}

#[test]
fn a_zero_frame_packer_still_produces_its_chunks() {
    let (chunks, visible) = Packer::new(4, 2, 128, HEX_STRIDE).finish();
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].frames.is_empty(), "no frames pushed, no frame strings");
    assert_eq!(visible, vec![false; 8]);
}

// --- Important 4: `push_frame` must reject a frame that disagrees with the
// dimensions `Packer::new` was built with, rather than panicking (undersized)
// or silently cropping (oversized). Nothing in the `FrameStream` trait
// enforced this before; `build_brick_world` sizes the `Packer` from
// `source.info()` once and then indexes every subsequent frame with
// `get_pixel(col, row)`, so a disagreeing frame reached that call unchecked.

/// A 4x4 `Packer` fed a 2x2 frame used to panic inside `get_pixel` the moment
/// it tried to read a column/row past the smaller frame's edge -- aborting
/// the CLI process and killing the GUI's render thread. It must now return a
/// descriptive `Err` instead, before any pixel is read.
#[test]
fn push_frame_rejects_an_undersized_frame_instead_of_panicking() {
    let mut p = Packer::new(4, 4, 128, HEX_STRIDE);
    let small = RgbaImage::from_pixel(2, 2, Rgba([1, 2, 3, 255]));
    let err = p.push_frame(&small).expect_err("an undersized frame must error, not panic");
    assert!(err.contains("4x4"), "error must name the expected size: {err}");
    assert!(err.contains("2x2"), "error must name the actual size: {err}");
}

/// A 2x2 `Packer` fed a 4x4 frame used to silently succeed: `get_pixel`
/// stayed in-bounds for the smaller footprint, so the extra rows/columns
/// were read and then simply never referenced again -- no error, no
/// indication anything was wrong, just a quietly cropped result.
#[test]
fn push_frame_rejects_an_oversized_frame_instead_of_silently_cropping() {
    let mut p = Packer::new(2, 2, 128, HEX_STRIDE);
    let big = RgbaImage::from_pixel(4, 4, Rgba([1, 2, 3, 255]));
    let err = p.push_frame(&big).expect_err("an oversized frame must error, not be cropped");
    assert!(err.contains("2x2"), "error must name the expected size: {err}");
    assert!(err.contains("4x4"), "error must name the actual size: {err}");
}

/// The dimension check must fire before the frame-count cap is otherwise
/// exhausted -- i.e. it applies per push, not just once -- and must not
/// corrupt a `Packer` that already has good frames pushed: a later mismatch
/// errors without touching the chunks already built from valid frames.
#[test]
fn a_dimension_mismatch_after_valid_frames_still_errors_without_corrupting_prior_chunks() {
    let mut p = Packer::new(2, 1, 128, HEX_STRIDE);
    let good = RgbaImage::from_pixel(2, 1, Rgba([9, 9, 9, 255]));
    p.push_frame(&good).expect("first frame matches info()");
    let bad = RgbaImage::from_pixel(3, 1, Rgba([9, 9, 9, 255]));
    assert!(p.push_frame(&bad).is_err(), "a later mismatched frame must still error");
    let (chunks, _) = p.finish();
    assert_eq!(chunks[0].frames.len(), 1, "the one valid push must be preserved, not rolled back");
    assert_eq!(chunks[0].frames[0], "090909090909");
}

// --- Differential sweep: Packer vs. the two-pass pack()+visible() ---------
//
// The single regression test above pins one hand-picked clip. This sweep
// goes wider: many widths/heights/frame-counts/alpha patterns, including
// ones that straddle a chunk boundary, are fully transparent, are fully
// opaque, or are a bare 1x1 clip -- run through both the old two-pass path
// (`pack` + a re-implementation of `bricks::visible`'s rule) and the new
// fused `Packer`, then diffed byte for byte.

fn alpha_opaque(_x: u32, _y: u32, _f: usize) -> u8 {
    255
}
fn alpha_transparent(_x: u32, _y: u32, _f: usize) -> u8 {
    0
}
fn alpha_checkerboard(x: u32, y: u32, f: usize) -> u8 {
    if (x + y + f as u32) % 2 == 0 { 255 } else { 0 }
}
fn alpha_threshold_boundary(x: u32, _y: u32, _f: usize) -> u8 {
    if x % 2 == 0 { 127 } else { 128 }
}
fn alpha_odd_frames_only(_x: u32, _y: u32, f: usize) -> u8 {
    if f % 2 == 1 { 255 } else { 0 }
}
/// A cheap deterministic bit-mixer, not a real PRNG -- just enough spread
/// that (x, y, f) triples don't fall into an accidental pattern the other
/// alpha functions might share.
fn alpha_pseudo_random(x: u32, y: u32, f: usize) -> u8 {
    let v = x
        .wrapping_mul(2_654_435_761)
        .wrapping_add(y.wrapping_mul(40_503))
        .wrapping_add(f as u32)
        .wrapping_mul(2_246_822_519);
    ((v >> 24) & 0xFF) as u8
}

struct SweepCase {
    label: &'static str,
    w: u32,
    h: u32,
    n: usize,
    threshold: u8,
    alpha: fn(u32, u32, usize) -> u8,
}

/// Builds a clip from `case`, runs it through the old two-pass path (`pack`
/// plus a fresh re-implementation of `bricks::visible`'s rule -- `bricks.rs`
/// is off limits for this task, so it is restated here rather than imported)
/// and through `Packer`, and appends a description to `mismatches` for every
/// place they disagree. Returns without mismatching only if every chunk's
/// `first_pixel`/`pixel_count`/`frames` and the full visibility vector are
/// identical.
///
/// Every case is run with `linearize` BOTH off and on. `pack` has no
/// `linearize` of its own, so the oracle for the `on` half is `pack` over a
/// clip whose pixels have already been through `crate::util::to_linear_rgb`.
/// That is a genuine oracle rather than a restatement of the optimized code:
/// `to_linear_rgb` passes alpha through untouched (see its doc), so culling
/// -- which reads alpha only, before any colour conversion -- is bit-for-bit
/// the same decision on both clips, and the surviving pixels' encoded colour
/// is exactly `to_linear_rgb(p)` either way.
fn run_sweep_case(case: &SweepCase, mismatches: &mut Vec<String>) {
    let SweepCase { label, w, h, n, threshold, alpha } = *case;
    let mut frames = Vec::with_capacity(n);
    for f in 0..n {
        let mut img = image::RgbaImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let a = alpha(x, y, f);
                // `x * 7` / `y * 11` wrap deliberately on the wide cases
                // below, which is fine: the point is a spread of byte values,
                // not a unique one per pixel.
                img.put_pixel(
                    x,
                    y,
                    image::Rgba([
                        (x.wrapping_mul(7)) as u8,
                        (y.wrapping_mul(11)) as u8,
                        (f as u32).wrapping_mul(29) as u8,
                        a,
                    ]),
                );
            }
        }
        frames.push(img);
    }
    let clip = heightmap::video::Clip { width: w, height: h, fps: 10.0, frames };

    let want_visible: Vec<bool> = (0..(w as usize * h as usize))
        .map(|i| {
            let (col, row) = ((i as u32) % w.max(1), (i as u32) / w.max(1));
            clip.frames.iter().any(|fr| fr.get_pixel(col, row).0[3] >= threshold)
        })
        .collect();

    for linearize in [false, true] {
        let label = format!("{label} [linearize={linearize}]");

        // The oracle clip: identical pixels for `linearize=false`, and the
        // sRGB->linear transfer already applied for `linearize=true`.
        let oracle = if linearize {
            let frames = clip
                .frames
                .iter()
                .map(|fr| {
                    let mut out = fr.clone();
                    for px in out.pixels_mut() {
                        px.0 = heightmap::util::to_linear_rgb(px.0);
                    }
                    out
                })
                .collect();
            heightmap::video::Clip { width: w, height: h, fps: 10.0, frames }
        } else {
            clip.clone()
        };

        let want_chunks = match heightmap::anim::pack::pack(&oracle, threshold) {
            Ok(c) => c,
            Err(e) => {
                mismatches.push(format!("{label}: pack() itself errored: {e}"));
                return;
            }
        };

        let mut p = Packer::new(w, h, threshold, HEX_STRIDE).linearize(linearize);
        for f in &clip.frames {
            if let Err(e) = p.push_frame(f) {
                mismatches.push(format!("{label}: Packer::push_frame errored unexpectedly: {e}"));
                return;
            }
        }
        let (got_chunks, got_visible) = p.finish();

        if got_chunks.len() != want_chunks.len() {
            mismatches.push(format!(
                "{label}: chunk count {} != {}",
                got_chunks.len(),
                want_chunks.len()
            ));
            return;
        }
        for (i, (g, wc)) in got_chunks.iter().zip(&want_chunks).enumerate() {
            if g.first_pixel != wc.first_pixel {
                mismatches.push(format!("{label}: chunk {i} first_pixel differs"));
            }
            if g.pixel_count != wc.pixel_count {
                mismatches.push(format!("{label}: chunk {i} pixel_count differs"));
            }
            if g.frames != wc.frames {
                mismatches.push(format!("{label}: chunk {i} frame strings differ"));
            }
        }
        if got_visible != want_visible {
            mismatches.push(format!("{label}: visibility vector differs"));
        }
    }
}

#[test]
fn a_differential_sweep_of_configurations_matches_the_two_pass_result() {
    let cases = vec![
        SweepCase { label: "1x1 opaque single frame", w: 1, h: 1, n: 1, threshold: 128, alpha: alpha_opaque },
        SweepCase { label: "1x1 transparent single frame", w: 1, h: 1, n: 1, threshold: 128, alpha: alpha_transparent },
        SweepCase { label: "1x1 zero frames", w: 1, h: 1, n: 0, threshold: 128, alpha: alpha_opaque },
        SweepCase { label: "1x1 alternating across many frames", w: 1, h: 1, n: 10, threshold: 128, alpha: alpha_odd_frames_only },
        SweepCase { label: "fully opaque clip", w: 6, h: 4, n: 5, threshold: 128, alpha: alpha_opaque },
        SweepCase { label: "fully transparent clip", w: 6, h: 4, n: 5, threshold: 128, alpha: alpha_transparent },
        SweepCase { label: "checkerboard alpha", w: 9, h: 7, n: 6, threshold: 128, alpha: alpha_checkerboard },
        SweepCase { label: "alpha exactly at / one below threshold", w: 8, h: 1, n: 1, threshold: 128, alpha: alpha_threshold_boundary },
        SweepCase { label: "pseudo-random alpha, mid threshold", w: 13, h: 11, n: 9, threshold: 128, alpha: alpha_pseudo_random },
        SweepCase { label: "pseudo-random alpha, low threshold", w: 13, h: 11, n: 9, threshold: 1, alpha: alpha_pseudo_random },
        SweepCase { label: "pseudo-random alpha, high threshold", w: 13, h: 11, n: 9, threshold: 255, alpha: alpha_pseudo_random },
        SweepCase { label: "threshold zero keeps everything", w: 5, h: 5, n: 3, threshold: 0, alpha: alpha_transparent },
        SweepCase { label: "chunk boundary exact width", w: PIXELS_PER_CHUNK as u32, h: 1, n: 2, threshold: 128, alpha: alpha_checkerboard },
        SweepCase { label: "chunk boundary minus one", w: (PIXELS_PER_CHUNK - 1) as u32, h: 1, n: 2, threshold: 128, alpha: alpha_checkerboard },
        SweepCase { label: "chunk boundary plus one", w: (PIXELS_PER_CHUNK + 1) as u32, h: 1, n: 2, threshold: 128, alpha: alpha_checkerboard },
        SweepCase { label: "multi-row straddling several chunks", w: (PIXELS_PER_CHUNK * 2 + 5) as u32, h: 1, n: 2, threshold: 128, alpha: alpha_checkerboard },
        // Every chunk-boundary case above is a single row, so a chunk always
        // began at column 0 -- the one arrangement in which "a chunk is a
        // contiguous run of the frame buffer" is trivially true. These are
        // taller than one row AND not a whole number of chunks per row, so
        // chunk seams land in the MIDDLE of a row and a chunk spans a row
        // wrap. That is exactly the case a flat-slice walk of `as_raw()` (or
        // any surviving `col`/`row` arithmetic) can get wrong.
        SweepCase { label: "chunk seams mid-row, odd width", w: 97, h: 53, n: 3, threshold: 128, alpha: alpha_pseudo_random },
        SweepCase { label: "chunk seam exactly one row in", w: PIXELS_PER_CHUNK as u32, h: 4, n: 2, threshold: 128, alpha: alpha_checkerboard },
        SweepCase { label: "chunk seam one pixel past a row", w: (PIXELS_PER_CHUNK - 1) as u32, h: 4, n: 2, threshold: 128, alpha: alpha_pseudo_random },
        SweepCase { label: "many rows, several chunks, all opaque", w: 64, h: 64, n: 2, threshold: 128, alpha: alpha_opaque },
        SweepCase { label: "many rows, several chunks, all culled", w: 64, h: 64, n: 2, threshold: 128, alpha: alpha_transparent },
        SweepCase { label: "zero-frame multi-pixel clip", w: 5, h: 3, n: 0, threshold: 128, alpha: alpha_opaque },
        SweepCase { label: "zero-width clip", w: 0, h: 5, n: 3, threshold: 128, alpha: alpha_opaque },
        SweepCase { label: "zero-height clip", w: 5, h: 0, n: 3, threshold: 128, alpha: alpha_opaque },
        SweepCase { label: "zero-width and zero-height clip", w: 0, h: 0, n: 3, threshold: 128, alpha: alpha_opaque },
    ];

    let mut mismatches = Vec::new();
    for case in &cases {
        run_sweep_case(case, &mut mismatches);
    }
    println!(
        "differential sweep: {} configurations compared, {} diverged",
        cases.len(),
        mismatches.len()
    );
    assert!(
        mismatches.is_empty(),
        "{} of {} configurations diverged:\n{}",
        mismatches.len(),
        cases.len(),
        mismatches.join("\n")
    );
}

// --- --srgb-to-linear --------------------------------------------------------

/// **`--srgb-to-linear` must actually convert.**
///
/// `Packer::linearize` had no behavioural test at all: emptying its body left
/// the whole suite green, so the flag could have been inert and nothing would
/// have said so. Video frames are sRGB and whether the in-game `MakeColorHex`
/// gate wants sRGB or linear is a property of the GAME -- nothing between this
/// packer and that gate touches colour -- so this flag is the only thing that
/// can compensate, and it has to be the exact transfer `crate::util` defines
/// rather than approximately darker.
#[test]
fn srgb_to_linear_rewrites_every_visible_pixel_with_the_linear_transfer() {
    // Mid-tones, where the sRGB transfer bites hardest. 0 and 255 are fixed
    // points of it, so a test built only on black and white would pass against
    // a no-op.
    let source = vec![[0x80, 0x40, 0xC0, 255], [0x10, 0xF0, 0x7F, 255]];
    let build = |on: bool| -> String {
        let mut p = Packer::new(2, 1, 128, HEX_STRIDE).linearize(on);
        let mut img = RgbaImage::new(2, 1);
        for (i, c) in source.iter().enumerate() {
            img.put_pixel(i as u32, 0, Rgba(*c));
        }
        p.push_frame(&img).expect("push");
        let (chunks, _) = p.finish();
        chunks[0].frames[0].clone()
    };

    let plain = build(false);
    let linear = build(true);
    assert_ne!(plain, linear, "the flag must change the bytes the gate receives");

    // Exactly the transfer, pixel by pixel -- not merely "different".
    let expected: String = source
        .iter()
        .map(|c| {
            let l = heightmap::util::to_linear_rgb(*c);
            format!("{:02X}{:02X}{:02X}", l[0], l[1], l[2])
        })
        .collect();
    assert_eq!(linear, expected, "every channel must be the sRGB->linear transfer");
    assert_eq!(plain, "8040C010F07F", "off must still be the raw sRGB bytes");

    // And it really is a DARKENING transfer, so a `linearize` that silently
    // ran the inverse would not pass either. Black is a fixed point, which is
    // why this test's source is all mid-tones -- a black-and-white clip could
    // not tell the transfer from a no-op.
    assert_eq!(heightmap::util::to_linear_rgb([0, 0, 0, 255])[0], 0);
    assert!(
        heightmap::util::to_linear_rgb([0x80, 0x80, 0x80, 255])[0] < 0x80,
        "mid grey must come out darker"
    );
}

/// A culled pixel writes its `'0'` padding either way: the flag governs
/// COLOUR, not the reserved-slot invariant every `Substring` offset depends on.
#[test]
fn srgb_to_linear_does_not_disturb_a_culled_pixels_reserved_slot() {
    for on in [false, true] {
        let mut p = Packer::new(2, 1, 128, HEX_STRIDE).linearize(on);
        let mut img = RgbaImage::new(2, 1);
        img.put_pixel(0, 0, Rgba([0, 0, 0, 0])); // below the threshold
        img.put_pixel(1, 0, Rgba([0x80, 0x80, 0x80, 255]));
        p.push_frame(&img).expect("push");
        let (chunks, _) = p.finish();
        let frame = &chunks[0].frames[0];
        assert_eq!(frame.len(), 2 * HEX_STRIDE, "linearize {on}: stride must be untouched");
        assert_eq!(&frame[..HEX_STRIDE], "000000", "linearize {on}: culled slot stays zeroed");
    }
}
