//! Treat a set of numbered image files as an animation frame sequence.

use image::RgbaImage;
use std::path::Path;

use super::Clip;

/// Extract the natural sort key from a filename.
///
/// Returns a tuple of (prefix, trailing_number) where:
/// - `prefix` is the filename stem with the trailing digit run stripped
/// - `trailing_number` is the parsed u64 value of that run (0 if absent)
///
/// This makes "frame_2" sort before "frame_10" despite lexicographic ordering.
/// Zero-padding is irrelevant: "f_0002" and "f_2" sort identically.
///
/// A name with NO trailing digits gets `0`, so `frame.png` sorts before
/// `frame1.png` -- the order every file manager and `ls -v` gives, and the
/// order this doc has always claimed. The previous code ran the empty digit
/// run through the same `parse().unwrap_or(u64::MAX)` the overflow case uses,
/// so an unnumbered first frame sorted LAST among its siblings and a rendered
/// sequence opened on what should have been its final frame. The existing
/// tests missed it: `names_without_numbers_fall_back_to_lexicographic` uses
/// two unnumbered names (both got `u64::MAX`, so the prefix decided it), and
/// `digit_run_overflow_caps_to_u64_max` pins the fallback that the empty case
/// was wrongly sharing.
pub fn natural_key(name: &str) -> (String, u64) {
    let stem = Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    // Find the trailing digit run
    let mut end = stem.len();
    while end > 0 && stem.as_bytes()[end - 1].is_ascii_digit() {
        end -= 1;
    }

    let prefix = stem[..end].to_string();
    let trailing_digits = &stem[end..];

    let number = if trailing_digits.is_empty() {
        // No number at all is not the same thing as a number too large to
        // hold -- see this function's doc.
        0
    } else {
        // A digit run wider than u64 caps rather than wrapping; such a name
        // sorts last among its siblings, which is the least surprising place
        // for something unrepresentable to land.
        trailing_digits.parse::<u64>().unwrap_or(u64::MAX)
    };

    (prefix, number)
}

/// Decode a sequence of named images into a single `Clip`.
///
/// Sorts the images by their natural key (numeric suffix awareness), rejects empty
/// sequences, and ensures all frames have matching dimensions.
///
/// # Errors
///
/// Returns an error if:
/// - The sequence is empty
/// - Frames have mismatched dimensions (error message includes both sizes)
pub fn decode_sequence(
    named: Vec<(String, RgbaImage)>,
    fps: f32,
) -> Result<Clip, String> {
    if named.is_empty() {
        return Err("Frame sequence cannot be empty".to_string());
    }

    // Sort by natural key, preserving the filename association
    let mut sorted: Vec<(String, RgbaImage)> = named;
    sorted.sort_by_key(|(name, _)| natural_key(name));

    // Extract frames and verify all have the same dimensions
    let (width, height) = sorted[0].1.dimensions();

    for (name, img) in &sorted {
        let (w, h) = img.dimensions();
        if w != width || h != height {
            return Err(format!(
                "Frame dimension mismatch: '{}' is {}x{}, expected {}x{}",
                name, w, h, width, height
            ));
        }
    }

    let frames: Vec<RgbaImage> = sorted.into_iter().map(|(_, img)| img).collect();

    Ok(Clip {
        width,
        height,
        fps,
        frames,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_numerically_not_lexicographically() {
        let mut names = vec!["frame_10.png", "frame_2.png", "frame_1.png"];
        names.sort_by_key(|n| natural_key(n));
        assert_eq!(names, vec!["frame_1.png", "frame_2.png", "frame_10.png"]);
    }

    #[test]
    fn zero_padding_does_not_change_order() {
        let mut names = vec!["f_0010.png", "f_0002.png"];
        names.sort_by_key(|n| natural_key(n));
        assert_eq!(names, vec!["f_0002.png", "f_0010.png"]);
    }

    #[test]
    fn names_without_numbers_fall_back_to_lexicographic() {
        let mut names = vec!["beta.png", "alpha.png"];
        names.sort_by_key(|n| natural_key(n));
        assert_eq!(names, vec!["alpha.png", "beta.png"]);
    }

    #[test]
    fn an_unnumbered_name_sorts_before_its_numbered_siblings() {
        // The unnumbered name used to share the overflow fallback and get
        // u64::MAX, so it sorted LAST -- a sequence whose first frame is
        // `frame.png` played it at the end.
        let mut names = vec!["frame2.png", "frame.png", "frame10.png", "frame1.png"];
        names.sort_by_key(|n| natural_key(n));
        assert_eq!(names, vec!["frame.png", "frame1.png", "frame2.png", "frame10.png"]);
    }

    #[test]
    fn an_absent_number_is_zero_not_the_overflow_sentinel() {
        assert_eq!(natural_key("intro.png"), ("intro".to_string(), 0));
        // ...and the overflow sentinel is still reserved for real overflow.
        assert_eq!(natural_key("intro99999999999999999999.png").1, u64::MAX);
    }

    #[test]
    fn decode_sequence_puts_an_unnumbered_frame_first() {
        // The ordering bug's actual consequence: the frames come out in the
        // wrong order, and nothing anywhere reports it.
        let mut first = RgbaImage::new(1, 1);
        first.put_pixel(0, 0, image::Rgba([1, 0, 0, 255]));
        let mut second = RgbaImage::new(1, 1);
        second.put_pixel(0, 0, image::Rgba([2, 0, 0, 255]));

        let named = vec![("f1.png".to_string(), second), ("f.png".to_string(), first)];
        let clip = decode_sequence(named, 10.0).unwrap();
        assert_eq!(clip.frames[0].get_pixel(0, 0)[0], 1, "f.png is frame 0");
        assert_eq!(clip.frames[1].get_pixel(0, 0)[0], 2);
    }

    #[test]
    fn mismatched_dimensions_are_rejected() {
        let named = vec![
            ("a_1.png".to_string(), RgbaImage::new(4, 4)),
            ("a_2.png".to_string(), RgbaImage::new(8, 4)),
        ];
        let err = decode_sequence(named, 10.0).unwrap_err();
        assert!(err.to_lowercase().contains("size") || err.to_lowercase().contains("dimension"));
    }

    #[test]
    fn an_empty_sequence_is_rejected() {
        assert!(decode_sequence(vec![], 10.0).is_err());
    }

    // Edge case tests
    #[test]
    fn digit_run_overflow_caps_to_u64_max() {
        // A digit string longer than u64::MAX should cap to u64::MAX
        let input = "frame_99999999999999999999.png";
        let (prefix, number) = natural_key(input);
        assert_eq!(prefix, "frame_");
        assert_eq!(number, u64::MAX);
    }

    #[test]
    fn names_differing_only_by_extension_have_identical_keys() {
        let key1 = natural_key("frame_5.png");
        let key2 = natural_key("frame_5.jpg");
        assert_eq!(key1, key2);
    }

    #[test]
    fn sort_is_stable_for_identical_keys() {
        let mut names = vec!["frame_1.png", "frame_1.jpg"];
        names.sort_by_key(|n| natural_key(n));
        // Both should have identical keys; since Rust's sort is stable,
        // the relative order should be preserved.
        assert_eq!(names[0], "frame_1.png");
        assert_eq!(names[1], "frame_1.jpg");
    }

    #[test]
    fn trailing_digit_run_extracted_correctly() {
        let (prefix, num) = natural_key("shot2_015.png");
        assert_eq!(prefix, "shot2_");
        assert_eq!(num, 15);
    }

    #[test]
    fn decode_sequence_returns_clip_with_correct_fps() {
        let named = vec![
            ("f_1.png".to_string(), RgbaImage::new(2, 2)),
            ("f_2.png".to_string(), RgbaImage::new(2, 2)),
        ];
        let clip = decode_sequence(named, 24.0).unwrap();
        assert_eq!(clip.fps, 24.0);
        assert_eq!(clip.frames.len(), 2);
        assert_eq!((clip.width, clip.height), (2, 2));
    }

    #[test]
    fn decode_sequence_preserves_frame_content_order() {
        // Create frames with different pixel values so we can verify order
        let mut frame1 = RgbaImage::new(1, 1);
        frame1.put_pixel(0, 0, image::Rgba([1, 0, 0, 255]));

        let mut frame2 = RgbaImage::new(1, 1);
        frame2.put_pixel(0, 0, image::Rgba([2, 0, 0, 255]));

        let named = vec![
            ("f_2.png".to_string(), frame2),
            ("f_1.png".to_string(), frame1),
        ];
        let clip = decode_sequence(named, 10.0).unwrap();

        // After natural sorting, frame_1 should come first
        assert_eq!(clip.frames[0].get_pixel(0, 0)[0], 1);
        assert_eq!(clip.frames[1].get_pixel(0, 0)[0], 2);
    }
}
