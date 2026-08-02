pub mod bricks;
pub mod chip;
pub mod clock;
pub mod color_bricks;
pub mod color_pack;
pub mod cost;
pub mod layout;
pub mod pack;
pub mod palette;
pub mod subtitle_display;
pub mod text_bricks;
pub mod text_layout;
pub mod text_pack;

use crate::progress::Progress;
use crate::video::stream::FrameSource;
use brdb::World;

/// How a render encodes a pixel's colour for the wire graph.
///
/// The two encodings produce the same *picture* from the same source; they
/// differ in what the microchip does per frame to produce it, and the whole
/// point of having both is to measure that difference in game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnimEncoding {
    /// Frame-major hex strings (see [`bricks`] and [`pack`]). One `ArrayVar`
    /// per 1666-pixel chunk holding one ~10 KB `RRGGBB...` string per frame;
    /// per pixel a `Substring` slices its own 6 characters out and a
    /// `MakeColorHex` parses them.
    ///
    /// The default, and unchanged: every existing render takes this path.
    #[default]
    Hex,
    /// Pixel-major linear colour arrays (see [`color_bricks`] and
    /// [`color_pack`]). One `ArrayVar` per PIXEL holding that pixel's colour
    /// for every frame, read by that pixel's own `ArrayVar_Get` straight into
    /// its display brick's `Color`.
    ///
    /// Same component count as [`Hex`](Self::Hex) -- two per pixel -- but half
    /// the per-frame gate executions (no `Substring`, no `MakeColorHex`) and
    /// no string allocation at all. Costs more host memory to build (see
    /// [`color_pack`]) and far more gates at a bank boundary (see
    /// [`color_bricks`]).
    ColorArray,
}

impl AnimEncoding {
    /// Parse a CLI/GUI spelling. `None` for anything unrecognised -- the
    /// caller owns the error message, since it knows which flag was typed.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "hex" => Some(Self::Hex),
            "color-array" | "colour-array" | "color" | "colour" => Some(Self::ColorArray),
            _ => None,
        }
    }

    /// Every spelling a user may type, for an error message.
    pub const NAMES: &'static str = "hex, color-array";

    /// Render `source` with this encoding. The two renderers have identical
    /// signatures on purpose, so every caller (CLI, GUI, tests) can pick an
    /// encoding without branching over anything else.
    pub fn build(
        self,
        source: &dyn FrameSource,
        opts: &bricks::AnimOptions,
        progress: &mut dyn Progress,
    ) -> Result<World, String> {
        match self {
            Self::Hex => bricks::build_brick_world(source, opts, progress),
            Self::ColorArray => color_bricks::build_color_array_world(source, opts, progress),
        }
    }

    /// The build-cost estimate for this encoding. The two formulas are
    /// genuinely different -- colour-array mode has no chunks, no characters,
    /// and a per-PIXEL select at every bank boundary -- so this dispatches
    /// rather than sharing one approximate number between them.
    ///
    /// Takes the whole [`bricks::AnimOptions`] for the same reason
    /// [`AnimMode::estimate`] does: every option that changes the graph --
    /// `bank_size`, and now a subtitle track's two gates -- has to reach the
    /// readout through the identical value the render is built from, or the
    /// two describe different graphs. See [`cost`]'s module doc.
    pub fn estimate(
        self,
        width: u32,
        height: u32,
        frames: usize,
        opts: &bricks::AnimOptions,
    ) -> cost::Cost {
        match self {
            Self::Hex => cost::estimate(width, height, frames, opts),
            Self::ColorArray => cost::estimate_color_array(width, height, frames, opts),
        }
    }
}

/// Which MEDIUM an animated render is built from.
///
/// Text mode is not a third [`AnimEncoding`]: it uses different bricks
/// (`Component_TextDisplay` instead of a display brick per pixel), different
/// components, and a completely different cost shape (per-band rather than
/// per-pixel or per-chunk) -- see [`text_bricks`] and [`cost::estimate_text`].
/// `AnimEncoding` only has a meaningful choice between [`Hex`](AnimEncoding::Hex)
/// and [`ColorArray`](AnimEncoding::ColorArray) *within* [`Brick`](Self::Brick);
/// `--anim-encoding` is therefore ignored -- and rejected, if the caller passed
/// one explicitly -- under `--anim-mode text`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimMode {
    /// One display brick per pixel, in either [`AnimEncoding`].
    Brick(AnimEncoding),
    /// A stack of animated `Component_TextDisplay` bricks, one per band of
    /// image rows (see [`text_layout::plan_bands`] and [`text_bricks`]).
    Text,
}

impl AnimMode {
    /// Every spelling a user may type for `--anim-mode`, for an error message.
    pub const NAMES: &'static str = "brick, text";

    /// Parse `--anim-mode` plus (optionally) `--anim-encoding`.
    ///
    /// `encoding` is `None` when the flag was not passed at all, and
    /// `Some(s)` with the raw string when it was -- that distinction matters
    /// because `--anim-mode text --anim-encoding hex` must be a hard error
    /// naming both the flag and the mode, not a silently-ignored encoding: a
    /// user who typed it would otherwise have no way to know it did nothing.
    pub fn parse(mode: &str, encoding: Option<&str>) -> Result<Self, String> {
        match mode.to_lowercase().as_str() {
            "brick" => {
                let enc = match encoding {
                    None => AnimEncoding::default(),
                    Some(s) => AnimEncoding::parse(s).ok_or_else(|| {
                        format!("unknown --anim-encoding '{s}' ({})", AnimEncoding::NAMES)
                    })?,
                };
                Ok(Self::Brick(enc))
            }
            "text" => match encoding {
                None => Ok(Self::Text),
                Some(s) => Err(format!(
                    "--anim-encoding '{s}' has no effect under --anim-mode text -- only brick \
                     mode has more than one pixel encoding; drop --anim-encoding to render text \
                     mode"
                )),
            },
            other => Err(format!("unknown --anim-mode '{other}' ({})", Self::NAMES)),
        }
    }

    /// Render `source` in this mode. Identical signature to
    /// [`AnimEncoding::build`] and each of the three renderers it and
    /// [`text_bricks::build_text_world`] wrap, so dispatch never branches on
    /// anything but the mode itself.
    pub fn build(
        &self,
        source: &dyn FrameSource,
        opts: &bricks::AnimOptions,
        progress: &mut dyn Progress,
    ) -> Result<World, String> {
        match self {
            Self::Brick(enc) => enc.build(source, opts, progress),
            Self::Text => text_bricks::build_text_world(source, opts, progress),
        }
    }

    /// The build-cost estimate for this mode. Dispatches to
    /// [`AnimEncoding::estimate`] or [`cost::estimate_text`], which is exactly
    /// why text mode is not folded into [`AnimEncoding`]: the two shapes
    /// aren't reconcilable into one formula.
    ///
    /// Takes the whole [`bricks::AnimOptions`] rather than loose numbers so the
    /// estimate can never describe a different graph from the one
    /// [`Self::build`] would produce from the same options. Text mode's band
    /// count depends on `opts.text.char_repeat`, and passing that separately
    /// invited exactly the mismatch it caused once already: a `--font` with a
    /// single-glyph cell bands 192x108 36 ways while the readout still claimed
    /// 54. `opts.subtitles` is the second instance of the same class -- two
    /// gates built and not counted -- and the reason the estimators BELOW now
    /// take the options too rather than a `bank_size` plucked out of them.
    ///
    /// `Err` only from text mode, and only for a geometry
    /// [`text_layout::plan_bands`] cannot lay out at all (too wide for the
    /// component character limit). Both brick encodings are total. The error is
    /// the one [`Self::build`] would fail with, so a caller that shows this in
    /// place of a cost tells the user exactly what pressing Generate would
    /// tell them -- rather than the plausible, unusually cheap "5 gate(s), 1
    /// brick(s)" a swallowed layout error used to read as. See
    /// [`cost::estimate_text`].
    pub fn estimate(
        &self,
        width: u32,
        height: u32,
        frames: usize,
        opts: &bricks::AnimOptions,
    ) -> Result<cost::Cost, String> {
        match self {
            Self::Brick(enc) => Ok(enc.estimate(width, height, frames, opts)),
            Self::Text => cost::estimate_text(width, height, frames, opts),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rejects_an_encoding_under_text_mode() {
        let err = AnimMode::parse("text", Some("hex")).expect_err("must reject");
        assert!(err.contains("--anim-encoding"), "names the offending flag: {err}");
        assert!(err.contains("text"), "names the mode: {err}");
    }

    #[test]
    fn brick_mode_defaults_to_hex() {
        assert_eq!(AnimMode::parse("brick", None).unwrap(), AnimMode::Brick(AnimEncoding::Hex));
    }

    #[test]
    fn text_mode_parses() {
        assert_eq!(AnimMode::parse("text", None).unwrap(), AnimMode::Text);
    }

    #[test]
    fn an_unknown_mode_names_the_valid_ones() {
        let err = AnimMode::parse("hologram", None).expect_err("must reject");
        assert!(err.contains("brick") && err.contains("text"), "{err}");
    }
}
