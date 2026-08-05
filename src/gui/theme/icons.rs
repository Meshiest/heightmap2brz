//! Font Awesome Free (Solid) icons for the Brickadia egui theme.
//!
//! The face is registered by [`super::install`] both as a fallback on the
//! proportional family and as the standalone [`super::icon_family`], so an icon
//! reaches the screen two ways:
//!
//! * inline in any string — `format!("{}  Play", icons::PLAY)` — because FA is a
//!   proportional fallback and the codepoints live in the Private Use Area where
//!   they never collide with real text;
//! * explicitly sized — [`icon`]`(PLAY).size(20.0)` — pinned to the FA family.
//!
//! Only the glyphs the app actually uses are declared. To add one, look up its
//! Unicode value in the Font Awesome cheatsheet (Solid, Free) and add a const.
//! Every value here is verified present in the bundled `fa-solid-900.ttf`.
use egui::RichText;

/// Font Awesome Free v6.7.2 — Solid (`fa-solid-900.ttf`). See
/// `assets/fonts/LICENSES.md` for attribution. Loaded by [`super::install`].
pub const FA_SOLID: &[u8] = include_bytes!("../../../assets/fonts/fa-solid-900.ttf");

pub const XMARK: &str = "\u{f00d}"; // fa-xmark — close / remove / toggle-off
pub const CHECK: &str = "\u{f00c}"; // fa-check — toggle-on
pub const CIRCLE_XMARK: &str = "\u{f057}"; // fa-circle-xmark
pub const ARROW_LEFT: &str = "\u{f060}"; // fa-arrow-left — back
pub const IMAGE: &str = "\u{f03e}"; // fa-image
pub const MOUNTAIN: &str = "\u{f6fc}"; // fa-mountain — heightmap
pub const FONT: &str = "\u{f031}"; // fa-font — text
pub const FILM: &str = "\u{f008}"; // fa-film — video
pub const WAVE_SQUARE: &str = "\u{f83e}"; // fa-wave-square — audio
pub const MUSIC: &str = "\u{f001}"; // fa-music — midi
pub const PLAY: &str = "\u{f04b}"; // fa-play — preview
pub const STOP: &str = "\u{f04d}"; // fa-stop
pub const FOLDER_OPEN: &str = "\u{f07c}"; // fa-folder-open — pick file
pub const DOWNLOAD: &str = "\u{f019}"; // fa-download — generate
pub const UP_RIGHT_FROM_SQUARE: &str = "\u{f35d}"; // fa-up-right-from-square — external link
pub const CHEVRON_DOWN: &str = "\u{f078}"; // fa-chevron-down — accordion open
pub const CHEVRON_RIGHT: &str = "\u{f054}"; // fa-chevron-right — accordion closed

/// A [`RichText`] for `glyph` pinned to the Font Awesome solid family, so it
/// renders at whatever `.size(..)` you give it regardless of the surrounding
/// text style. For icon-plus-label buttons, prefer inline
/// `format!("{glyph}  label")` instead — no wrapper needed.
pub fn icon(glyph: &str) -> RichText {
    RichText::new(glyph).family(super::icon_family())
}

/// Every declared icon as `(name, glyph)`, for galleries/demos like
/// [`super::sandbox`]. Not used at runtime by the app.
pub const ALL: &[(&str, &str)] = &[
    ("xmark", XMARK),
    ("check", CHECK),
    ("circle-xmark", CIRCLE_XMARK),
    ("arrow-left", ARROW_LEFT),
    ("image", IMAGE),
    ("mountain", MOUNTAIN),
    ("font", FONT),
    ("film", FILM),
    ("wave-square", WAVE_SQUARE),
    ("music", MUSIC),
    ("play", PLAY),
    ("stop", STOP),
    ("folder-open", FOLDER_OPEN),
    ("download", DOWNLOAD),
    ("up-right-from-square", UP_RIGHT_FROM_SQUARE),
    ("chevron-down", CHEVRON_DOWN),
    ("chevron-right", CHEVRON_RIGHT),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Each icon const must be exactly one Private-Use-Area char — a typo'd
    /// escape or an accidental ASCII glyph would render as tofu at runtime.
    #[test]
    fn icon_consts_are_single_pua_glyphs() {
        for (name, glyph) in ALL {
            let mut chars = glyph.chars();
            let c = chars.next().unwrap_or_else(|| panic!("{name} is empty"));
            assert!(chars.next().is_none(), "{name} is more than one char");
            assert!(
                ('\u{e000}'..='\u{f8ff}').contains(&c),
                "{name} (U+{:04X}) is not in the Private Use Area",
                c as u32
            );
        }
    }
}
