use heightmap::anim::pack::{Chunk, HEX_STRIDE, PIXELS_PER_CHUNK, pack, slice_of};
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
/// meaning there — nothing displays it), independent of what it encodes to
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
/// survives) — not the stricter `>` a naive reading of "below the threshold"
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
