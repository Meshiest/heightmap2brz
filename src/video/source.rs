//! Turn any supported input into a `Clip`.
use super::{Clip, animated::decode_animated, sequence::decode_sequence};
use image::RgbaImage;

pub enum Source {
    Animated(Vec<u8>),
    Sequence(Vec<(String, RgbaImage)>),
    Still(RgbaImage),
}

/// Extensions routed to a video decode backend rather than the image decoders.
///
/// Extension-based rather than content-sniffed on purpose: the backends probe
/// the real container themselves and error clearly on a mismatch, so sniffing
/// here would duplicate that and disagree with it eventually.
pub const VIDEO_EXTENSIONS: [&str; 6] = ["mp4", "mov", "mkv", "webm", "avi", "m4v"];

pub fn is_video_path(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .is_some_and(|e| VIDEO_EXTENSIONS.contains(&e.as_str()))
}

/// True when the bytes are a container that can hold multiple frames. A
/// single-frame GIF still counts -- `decode_animated` handles it and yields a
/// one-frame clip.
pub fn is_animated(bytes: &[u8]) -> bool {
    matches!(
        image::guess_format(bytes),
        Ok(image::ImageFormat::Gif) | Ok(image::ImageFormat::WebP)
    ) || is_apng(bytes)
}

fn is_apng(bytes: &[u8]) -> bool {
    if !matches!(image::guess_format(bytes), Ok(image::ImageFormat::Png)) {
        return false;
    }
    image::codecs::png::PngDecoder::new(std::io::Cursor::new(bytes))
        .and_then(|d| d.is_apng())
        .unwrap_or(false)
}

pub fn decode(source: Source, fallback_fps: f32) -> Result<Clip, String> {
    match source {
        Source::Animated(bytes) => decode_animated(&bytes),
        Source::Sequence(named) => decode_sequence(named, fallback_fps),
        Source::Still(img) => {
            let (width, height) = img.dimensions();
            Ok(Clip { width, height, fps: fallback_fps, frames: vec![img] })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbaImage;

    #[test]
    fn a_still_image_becomes_a_one_frame_clip() {
        let clip = decode(Source::Still(RgbaImage::new(3, 2)), 10.0).unwrap();
        assert_eq!(clip.frames.len(), 1);
        assert_eq!((clip.width, clip.height), (3, 2));
    }

    #[test]
    fn a_sequence_uses_the_fallback_fps() {
        let named = vec![
            ("f_1.png".to_string(), RgbaImage::new(2, 2)),
            ("f_2.png".to_string(), RgbaImage::new(2, 2)),
        ];
        let clip = decode(Source::Sequence(named), 24.0).unwrap();
        assert_eq!(clip.frames.len(), 2);
        assert_eq!(clip.fps, 24.0);
    }

    #[test]
    fn png_bytes_are_not_treated_as_an_animation() {
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(RgbaImage::new(1, 1))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        assert!(!is_animated(&png));
    }

    #[test]
    fn garbage_bytes_are_rejected_not_panicked_on() {
        assert!(decode(Source::Animated(b"nonsense".to_vec()), 10.0).is_err());
    }

    #[test]
    fn video_extensions_are_recognised_case_insensitively() {
        for ext in ["mp4", "MP4", "mov", "mkv", "webm", "avi", "m4v", "M4v"] {
            let path = std::path::PathBuf::from(format!("clip.{ext}"));
            assert!(is_video_path(&path), "{ext} should be treated as a video");
        }
    }

    #[test]
    fn non_video_extensions_are_not_routed_to_a_video_backend() {
        for ext in ["png", "jpg", "jpeg", "gif", "webp", "bmp", "tga"] {
            let path = std::path::PathBuf::from(format!("image.{ext}"));
            assert!(!is_video_path(&path), "{ext} should not be treated as a video");
        }
    }

    #[test]
    fn an_extensionless_path_is_not_a_video() {
        assert!(!is_video_path(&std::path::PathBuf::from("no_extension")));
    }
}
