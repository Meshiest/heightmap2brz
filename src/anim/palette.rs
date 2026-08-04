//! Median-cut colour quantization.
//!
//! Text mode writes an explicit `<color="RRGGBB">` tag at the start of every
//! colour run, so its size is governed by how long runs are, not how wide a
//! tag is. Collapsing the palette lengthens runs: measured on a real animated
//! episode at 192x108, going from 24-bit to a 32-entry median-cut palette cuts
//! characters per cell from 14.65 to 4.74 -- from 2.4x brick mode's cost to
//! 0.79x -- because the tag count falls by 7.4x.
//!
//! Median cut rather than uniform bit truncation: at equal palette size uniform
//! gives longer runs (it merges broad swathes of colour space), but that is the
//! wrong comparison. On the cost/quality frontier median cut dominates --
//! median-cut 32 is both cheaper and ~1.8 dB better than uniform 256. There is
//! no in-game palette table; the tags carry RGB, so any palette is free to use.
//!
//! No dithering, deliberately: it would improve appearance while destroying the
//! run lengths this exists to create.
use image::RgbaImage;
use std::collections::HashMap;

/// A fixed set of colours, plus a memoized nearest-entry lookup.
#[derive(Debug, Clone, Default)]
pub struct Palette {
    entries: Vec<[u8; 3]>,
    cache: HashMap<[u8; 3], [u8; 3]>,
}

/// One box in the median-cut subdivision: the colours it holds, with counts.
struct Box3 {
    colors: Vec<([u8; 3], u32)>,
}

impl Box3 {
    /// Extent along each axis. The longest is the one worth splitting.
    fn extents(&self) -> [u8; 3] {
        let mut lo = [u8::MAX; 3];
        let mut hi = [0u8; 3];
        for (c, _) in &self.colors {
            for ch in 0..3 {
                lo[ch] = lo[ch].min(c[ch]);
                hi[ch] = hi[ch].max(c[ch]);
            }
        }
        [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]
    }

    fn longest_axis(&self) -> usize {
        let e = self.extents();
        let mut best = 0;
        for ch in 1..3 {
            if e[ch] > e[best] {
                best = ch;
            }
        }
        best
    }

    /// Total pixel count, which is what "largest box" means for splitting --
    /// splitting by population beats splitting by extent, because it puts
    /// entries where the image actually spends its pixels.
    fn population(&self) -> u64 {
        self.colors.iter().map(|(_, n)| *n as u64).sum()
    }

    /// The population-weighted mean, rounded. This is the box's palette entry.
    fn average(&self) -> [u8; 3] {
        let total = self.population().max(1);
        let mut acc = [0u64; 3];
        for (c, n) in &self.colors {
            for ch in 0..3 {
                acc[ch] += c[ch] as u64 * *n as u64;
            }
        }
        [
            ((acc[0] + total / 2) / total) as u8,
            ((acc[1] + total / 2) / total) as u8,
            ((acc[2] + total / 2) / total) as u8,
        ]
    }

    /// Split at the population-weighted median along `axis`. Returns `None`
    /// when the box cannot be split into two non-empty halves (one distinct
    /// colour, or all colours identical on this axis).
    fn split(mut self, axis: usize) -> Option<(Box3, Box3)> {
        if self.colors.len() < 2 {
            return None;
        }
        // Sort by the axis, tie-broken by the full colour so the order -- and
        // therefore the resulting palette -- is deterministic.
        self.colors.sort_by_key(|(c, _)| (c[axis], *c));
        let half = self.population() / 2;
        let mut acc = 0u64;
        let mut cut = 0usize;
        for (i, (_, n)) in self.colors.iter().enumerate() {
            acc += *n as u64;
            if acc > half {
                cut = i;
                break;
            }
        }
        // Keep both sides non-empty: a first entry already over half would
        // otherwise cut at 0 and produce an empty left box.
        let cut = cut.clamp(1, self.colors.len() - 1);
        let right = self.colors.split_off(cut);
        Some((Box3 { colors: self.colors }, Box3 { colors: right }))
    }
}

impl Palette {
    /// Build a palette of at most `size` entries from `samples`.
    ///
    /// Pixels with alpha strictly below `alpha_threshold` are excluded: they
    /// are never drawn, so letting them vote would spend palette entries on
    /// colours nothing displays.
    ///
    /// An empty sample set, or `size == 0`, yields an empty palette, which
    /// [`Palette::map`] passes through unchanged. That is the "no quantization"
    /// path and it must stay free of special cases at the call site.
    pub fn build(samples: &[RgbaImage], size: usize, alpha_threshold: u8) -> Self {
        if size == 0 {
            return Self::default();
        }
        let mut counts: HashMap<[u8; 3], u32> = HashMap::new();
        for img in samples {
            for p in img.pixels() {
                if p.0[3] < alpha_threshold {
                    continue;
                }
                *counts.entry([p.0[0], p.0[1], p.0[2]]).or_insert(0) += 1;
            }
        }
        if counts.is_empty() {
            return Self::default();
        }
        // Sorted, not HashMap order: the split order below must not depend on
        // hash iteration, or two runs would produce different palettes.
        let mut colors: Vec<([u8; 3], u32)> = counts.into_iter().collect();
        colors.sort_unstable();

        let mut boxes = vec![Box3 { colors }];
        while boxes.len() < size {
            // Split the most populous splittable box.
            let Some(idx) = boxes
                .iter()
                .enumerate()
                .filter(|(_, b)| b.colors.len() > 1)
                .max_by_key(|(i, b)| (b.population(), std::cmp::Reverse(*i)))
                .map(|(i, _)| i)
            else {
                break; // every box is a single colour
            };
            let b = boxes.swap_remove(idx);
            let axis = b.longest_axis();
            match b.split(axis) {
                Some((l, r)) => {
                    boxes.push(l);
                    boxes.push(r);
                }
                None => break,
            }
        }

        let mut entries: Vec<[u8; 3]> = boxes.iter().map(|b| b.average()).collect();
        entries.sort_unstable();
        entries.dedup();
        Self { entries, cache: HashMap::new() }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[[u8; 3]] {
        &self.entries
    }

    /// The nearest palette entry to `rgb` by squared Euclidean RGB distance,
    /// or `rgb` itself for an empty palette.
    pub fn map(&mut self, rgb: [u8; 3]) -> [u8; 3] {
        if self.entries.is_empty() {
            return rgb;
        }
        if let Some(hit) = self.cache.get(&rgb) {
            return *hit;
        }
        let mut best = self.entries[0];
        let mut best_d = u32::MAX;
        for e in &self.entries {
            let dr = e[0] as i32 - rgb[0] as i32;
            let dg = e[1] as i32 - rgb[1] as i32;
            let db = e[2] as i32 - rgb[2] as i32;
            let d = (dr * dr + dg * dg + db * db) as u32;
            if d < best_d {
                best_d = d;
                best = *e;
            }
        }
        self.cache.insert(rgb, best);
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    fn img(pixels: &[[u8; 4]]) -> RgbaImage {
        let mut i = RgbaImage::new(pixels.len() as u32, 1);
        for (x, p) in pixels.iter().enumerate() {
            i.put_pixel(x as u32, 0, Rgba(*p));
        }
        i
    }

    const RED: [u8; 4] = [0xFF, 0, 0, 0xFF];
    const GREEN: [u8; 4] = [0, 0xFF, 0, 0xFF];
    const BLUE: [u8; 4] = [0, 0, 0xFF, 0xFF];

    #[test]
    fn exact_when_palette_is_at_least_the_distinct_colour_count() {
        let f = img(&[RED, GREEN, BLUE]);
        let mut p = Palette::build(&[f], 4, 1);
        assert_eq!(p.map([0xFF, 0, 0]), [0xFF, 0, 0]);
        assert_eq!(p.map([0, 0xFF, 0]), [0, 0xFF, 0]);
        assert_eq!(p.map([0, 0, 0xFF]), [0, 0, 0xFF]);
    }

    #[test]
    fn never_exceeds_the_requested_size() {
        let mut pixels = Vec::new();
        for r in 0..16u8 {
            for g in 0..16u8 {
                pixels.push([r * 16, g * 16, 0, 0xFF]);
            }
        }
        let p = Palette::build(&[img(&pixels)], 8, 1);
        assert!(p.len() <= 8, "got {} entries", p.len());
    }

    #[test]
    fn fewer_distinct_colours_than_requested_yields_no_duplicates() {
        let f = img(&[RED, RED, GREEN]);
        let p = Palette::build(&[f], 16, 1);
        assert_eq!(p.len(), 2, "two distinct colours, not sixteen");
        let mut sorted = p.entries().to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), p.len(), "entries must be distinct");
    }

    #[test]
    fn a_single_entry_palette_maps_everything_to_one_colour() {
        let f = img(&[RED, GREEN, BLUE]);
        let mut p = Palette::build(&[f], 1, 1);
        assert_eq!(p.len(), 1);
        let a = p.map([0xFF, 0, 0]);
        let b = p.map([0, 0, 0xFF]);
        assert_eq!(a, b);
    }

    #[test]
    fn maps_to_the_nearest_entry() {
        let f = img(&[[0, 0, 0, 0xFF], [0xFF, 0xFF, 0xFF, 0xFF]]);
        let mut p = Palette::build(&[f], 2, 1);
        assert_eq!(p.map([0x10, 0x10, 0x10]), [0, 0, 0], "near black");
        assert_eq!(p.map([0xF0, 0xF0, 0xF0]), [0xFF, 0xFF, 0xFF], "near white");
    }

    #[test]
    fn transparent_pixels_do_not_contribute_to_the_palette() {
        // A hugely-outnumbered opaque red, and many transparent greens. If
        // alpha were ignored, green would dominate a 1-entry palette.
        let mut pixels = vec![RED];
        pixels.extend(std::iter::repeat([0, 0xFF, 0, 0]).take(50));
        let p = Palette::build(&[img(&pixels)], 1, 1);
        assert_eq!(p.entries(), &[[0xFF, 0, 0]]);
    }

    #[test]
    fn building_is_deterministic() {
        let f = img(&[RED, GREEN, BLUE, [0x80, 0x80, 0x80, 0xFF]]);
        let a = Palette::build(&[f.clone()], 3, 1);
        let b = Palette::build(&[f], 3, 1);
        assert_eq!(a.entries(), b.entries(), "same input must give same palette");
    }

    #[test]
    fn an_empty_sample_set_yields_an_empty_palette_that_maps_identically() {
        let mut p = Palette::build(&[], 8, 1);
        assert_eq!(p.len(), 0);
        assert_eq!(p.map([0x12, 0x34, 0x56]), [0x12, 0x34, 0x56], "pass through");
    }

    /// `--colors N` must actually produce N entries.
    ///
    /// Every test above constrains the palette from one side only:
    /// `never_exceeds_the_requested_size` asserts `<= 8`,
    /// `fewer_distinct_colours_than_requested_yields_no_duplicates` uses a
    /// 2-colour source, and the rest use 1-3 entries. A `build` that silently
    /// capped every palette at 3 entries would pass all of them -- so
    /// `--colors 32` could have been rendering 3 colours with nothing in the
    /// suite to say otherwise. This pins the count exactly, at sizes well past
    /// that cap, on a source with plenty of distinct colours to go round.
    #[test]
    fn a_palette_really_has_as_many_entries_as_were_asked_for() {
        // 256 distinct colours spread along all three axes, so median cut has
        // something genuine to subdivide at every size below.
        let pixels: Vec<[u8; 4]> = (0..256u32)
            .map(|i| {
                [
                    (i & 0xFF) as u8,
                    ((i * 7) & 0xFF) as u8,
                    ((i * 13) & 0xFF) as u8,
                    0xFF,
                ]
            })
            .collect();
        let source = img(&pixels);
        for size in [2usize, 4, 8, 16, 32, 64] {
            let p = Palette::build(&[source.clone()], size, 1);
            assert_eq!(
                p.len(),
                size,
                "--colors {size} must give {size} entries, got {}",
                p.len()
            );
            let mut sorted = p.entries().to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), size, "--colors {size}: entries must all be distinct");
        }
    }

    /// The other half: the entry count has to reach the mapping, not merely
    /// sit in `entries()`. A bigger palette must resolve a gradient to more
    /// distinct outputs, and every output must be one of the entries.
    #[test]
    fn a_bigger_palette_maps_a_gradient_to_more_distinct_colours() {
        let pixels: Vec<[u8; 4]> = (0..256u32)
            .map(|i| [(i & 0xFF) as u8, (255 - i) as u8, ((i * 3) & 0xFF) as u8, 0xFF])
            .collect();
        let source = img(&pixels);

        let distinct_outputs = |size: usize| -> usize {
            let mut p = Palette::build(&[source.clone()], size, 1);
            let entries: std::collections::HashSet<[u8; 3]> =
                p.entries().iter().copied().collect();
            let mut seen = std::collections::HashSet::new();
            for px in &pixels {
                let mapped = p.map([px[0], px[1], px[2]]);
                assert!(
                    entries.contains(&mapped),
                    "size {size}: map returned {mapped:?}, which is not a palette entry"
                );
                seen.insert(mapped);
            }
            seen.len()
        };

        let (three, thirty_two) = (distinct_outputs(3), distinct_outputs(32));
        assert_eq!(three, 3, "a 3-entry palette must resolve exactly 3 colours");
        // Not exactly 32: a median-cut box average can land where no source
        // colour is nearest to it, so a handful of entries go unused on any
        // given clip. The point is the order of magnitude -- a cap at 3 would
        // read as 3 here, not as the high twenties.
        assert!(
            thirty_two >= 24,
            "a 32-entry palette must resolve most of its entries, got {thirty_two}"
        );
        assert!(
            thirty_two > 4 * three,
            "asking for 32 colours instead of 3 must render many more of them: \
             {thirty_two} vs {three}"
        );
    }
}
