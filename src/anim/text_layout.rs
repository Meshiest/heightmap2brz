//! The closed-form band layout for text mode.
//!
//! Text components are physical bricks at fixed positions with fixed wiring,
//! so the image rows a given component draws MUST NOT move between frames.
//! The layout is therefore decided ONCE, from a worst-case bound on row width,
//! before a single frame is looked at: a row of `width` pixels costs at most
//! `width * (TAG_CHARS + char_repeat)` characters, because the worst case is
//! every pixel changing colour and so emitting its own `<color="RRGGBB">` tag.
//! The bound depends only on the width, so the layout never needs a scan pass
//! over the clip -- which is what lets the render stream: frames can be
//! encoded and discarded one at a time, because the band boundaries are
//! already fixed.
//!
//! Because every row costs the same bound, packing rows into bands degenerates
//! to a constant: `rows_per_band = (MAX_COMPONENT_CHARS + 1) / (bound + 1)`,
//! where the `+ 1`s account for the newline that joins rows within a band (a
//! band of `rows` rows costs `rows * bound + (rows - 1)` characters at worst,
//! and `rows * (bound + 1) <= MAX + 1` is the same inequality rearranged so
//! integer division rounds correctly). That constant is 2 rows at width 192,
//! 5 at width 96, and 8 at width 64 -- the values this module's tests pin.
//!
//! This module is pure arithmetic over `width` / `height` / `char_repeat`; it
//! knows nothing about images, bricks, or `World`.
use crate::text::MAX_COMPONENT_CHARS;

/// Characters one `<color="RRGGBB">` tag costs: `<color="` (8) + 6 hex + `">`
/// (2). The worst-case row bound assumes EVERY pixel emits one, which is what
/// `heightmap::text::encode_row` does when no two neighbours share a colour.
const TAG_CHARS: usize = 16;

/// One `TextDisplay`'s worth of image rows, fixed for the whole clip.
///
/// Each band is meant to get its OWN anchor cube at its own image row (see
/// `heightmap::text::add_text_tiles`), rather than one shared anchor with
/// leading newlines pushing each band down: that alternative stacks bands in
/// depth and needs a component `Offset.Z` the game does not honour past a
/// handful of bands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BandPlan {
    /// First image row this band draws.
    pub start_row: usize,
    pub rows: usize,
}

/// Upper bound on the characters one image row can encode to.
pub fn worst_case_row_chars(width: usize, char_repeat: usize) -> usize {
    width * (TAG_CHARS + char_repeat)
}

/// Decide the band layout for a `width` x `height` clip, once, from the
/// worst-case row bound alone -- no frame is consulted, so every frame is
/// guaranteed to produce exactly this layout, and no scan pass is needed.
///
/// Every row costs the same bound, so the packing is uniform: `rows` rows cost
/// `rows * bound + (rows - 1)` (the separating newlines), and the largest
/// `rows` satisfying that is the same for every band.
pub fn plan_bands(width: usize, height: usize, char_repeat: usize) -> Result<Vec<BandPlan>, String> {
    let row = worst_case_row_chars(width, char_repeat);
    if row > MAX_COMPONENT_CHARS {
        return Err(format!(
            "row 0: a {width}-pixel row encodes to at most {row} chars ({TAG_CHARS} for a \
             colour tag + {char_repeat} glyph chars per pixel), over the \
             {MAX_COMPONENT_CHARS}-char TextDisplay limit -- no band layout can fit it, and \
             every other row of this image costs the same; render narrower or with a smaller \
             char_repeat"
        ));
    }
    // rows * row + (rows - 1) <= MAX  <=>  rows * (row + 1) <= MAX + 1
    let rows_per_band = ((MAX_COMPONENT_CHARS + 1) / (row + 1)).max(1);
    Ok((0..height)
        .step_by(rows_per_band)
        .map(|start_row| BandPlan {
            start_row,
            rows: rows_per_band.min(height - start_row),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::MAX_COMPONENT_CHARS;

    #[test]
    fn the_bound_is_sixteen_plus_repeat_per_pixel() {
        assert_eq!(worst_case_row_chars(10, 2), 10 * 18);
        assert_eq!(worst_case_row_chars(10, 1), 10 * 17);
        assert_eq!(worst_case_row_chars(0, 2), 0);
    }

    #[test]
    fn bands_tile_every_row_exactly_once_in_order() {
        let plan = plan_bands(192, 108, 2).expect("192 wide fits");
        let mut next = 0;
        for b in &plan {
            assert_eq!(b.start_row, next, "bands must be contiguous and ordered");
            assert!(b.rows > 0, "no empty bands");
            next += b.rows;
        }
        assert_eq!(next, 108, "every row covered exactly once");
    }

    #[test]
    fn no_band_can_exceed_the_component_limit_at_its_worst_case() {
        for width in [1usize, 63, 64, 96, 192, 400] {
            for repeat in [1usize, 2] {
                let plan = plan_bands(width, 108, repeat).expect("fits");
                let row = worst_case_row_chars(width, repeat);
                for b in &plan {
                    let worst = b.rows * row + b.rows.saturating_sub(1);
                    assert!(
                        worst <= MAX_COMPONENT_CHARS,
                        "width {width} repeat {repeat}: band of {} rows worst-cases to {worst}",
                        b.rows
                    );
                }
            }
        }
    }

    #[test]
    fn rows_per_band_matches_the_documented_constants() {
        assert_eq!(plan_bands(192, 108, 2).unwrap()[0].rows, 2);
        assert_eq!(plan_bands(96, 108, 2).unwrap()[0].rows, 5);
        assert_eq!(plan_bands(64, 108, 2).unwrap()[0].rows, 8);
    }

    #[test]
    fn the_last_band_is_short_rather_than_padded() {
        // 108 rows at 5 per band -> 21 full bands + 3 rows.
        let plan = plan_bands(96, 108, 2).unwrap();
        assert_eq!(plan.last().unwrap().rows, 3);
    }

    #[test]
    fn a_row_too_wide_for_one_component_is_an_error_naming_the_width() {
        // 10000 / 18 = 555 px is the ceiling at char_repeat 2.
        assert!(plan_bands(555, 10, 2).is_ok(), "555 fits");
        let err = plan_bands(556, 10, 2).expect_err("556 must not fit");
        assert!(err.contains("556"), "error must name the width: {err}");
        assert!(
            err.contains(&MAX_COMPONENT_CHARS.to_string()),
            "error must name the limit: {err}"
        );
    }

    #[test]
    fn a_zero_height_screen_plans_no_bands() {
        assert!(plan_bands(64, 0, 2).unwrap().is_empty());
    }
}
