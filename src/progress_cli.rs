//! Terminal progress for the CLI. Native only: the wasm build has no
//! terminal and the GUI reports through egui instead.
use indicatif::{ProgressBar, ProgressStyle};

use heightmap::progress::{Progress, reconcile_total};

pub struct CliProgress {
    bar: Option<ProgressBar>,
}

impl CliProgress {
    pub fn new() -> Self {
        Self { bar: None }
    }
}

impl Default for CliProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl Progress for CliProgress {
    fn begin(&mut self, label: &str, total: Option<u64>) {
        // An unknown total gets a spinner, never a bar: a bar needs a
        // denominator, and inventing one produces a gauge that reaches 100%
        // and keeps going.
        let bar = match total {
            Some(n) => {
                let b = ProgressBar::new(n);
                b.set_style(
                    ProgressStyle::with_template("{msg} [{bar:40}] {pos}/{len} ({eta})")
                        .expect("static template")
                        .progress_chars("=> "),
                );
                b
            }
            None => {
                let b = ProgressBar::new_spinner();
                b.set_style(
                    ProgressStyle::with_template("{msg} {spinner} {pos}").expect("static template"),
                );
                b
            }
        };
        bar.set_message(label.to_string());
        self.bar = Some(bar);
    }

    /// A total that came from an ESTIMATE (see `progress::FrameTotal`) can be
    /// low, and a render that runs past it must not draw a bar over 100% or a
    /// `pos/len` that reads `137/100`. The length grows to meet the position
    /// rather than the position being clamped to the length: clamping would
    /// freeze the count at the estimate while the render kept going, which is
    /// the exact "looks stalled" impression the estimate was added to remove.
    ///
    /// `progress::reconcile_total` is the shared rule, unit-tested in the
    /// library and used by both GUI panes too, so no reporter can drift onto
    /// a different one.
    fn tick(&mut self, n: u64) {
        let Some(b) = &self.bar else { return };
        let was = b.length();
        let (pos, len) = reconcile_total(was, n);
        // Only when it actually moves: `set_length` forces a redraw, and this
        // runs once per frame. A spinner (`None`) has no length to reconcile.
        if len != was {
            if let Some(len) = len {
                b.set_length(len);
            }
        }
        b.set_position(pos);
    }

    fn finish(&mut self) {
        if let Some(b) = self.bar.take() {
            b.finish_and_clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The estimate must degrade gracefully.** A total derived from
    /// `duration * fps` can be low, and the render then runs past it. The bar
    /// must keep counting, must never end up with a position past its length
    /// (which is what a fraction above 100% would be drawn from), and must not
    /// panic.
    #[test]
    fn ticking_past_an_estimated_total_grows_the_bar_instead_of_overflowing_it() {
        let mut p = CliProgress::new();
        p.begin("packing frames (estimated total)", Some(100));
        for n in 1..=100 {
            p.tick(n);
        }
        {
            let b = p.bar.as_ref().expect("begin installed a bar");
            assert_eq!((b.position(), b.length()), (100, Some(100)));
        }
        // Past the estimate.
        for n in 101..=137 {
            p.tick(n);
        }
        let b = p.bar.as_ref().expect("the bar must survive over-ticking");
        assert_eq!(b.position(), 137, "the count must stay honest, not freeze at the estimate");
        assert_eq!(b.length(), Some(137), "the length must grow to meet it");
        assert!(b.position() <= b.length().unwrap(), "a bar must never be past its own end");
        p.finish();
    }

    /// An estimate that OVERSHOOTS needs no special handling -- the bar simply
    /// never fills before `finish` clears it. Pinned so a future "clamp the
    /// position" change cannot quietly make a short render report 100%.
    #[test]
    fn an_overshooting_estimate_leaves_the_bar_short_rather_than_faking_completion() {
        let mut p = CliProgress::new();
        p.begin("packing frames (estimated total)", Some(1000));
        p.tick(12);
        let b = p.bar.as_ref().expect("bar");
        assert_eq!((b.position(), b.length()), (12, Some(1000)));
        p.finish();
    }

    /// A phase with no total at all gets a spinner, and ticking it must be
    /// harmless -- there is no length to reconcile against.
    #[test]
    fn a_totalless_phase_gets_a_spinner_that_still_counts() {
        let mut p = CliProgress::new();
        p.begin("packing frames (total unknown -- source did not report one)", None);
        p.tick(1);
        p.tick(9_999);
        let b = p.bar.as_ref().expect("bar");
        assert_eq!(b.length(), None, "a spinner must not invent a denominator");
        assert_eq!(b.position(), 9_999);
        p.finish();
    }

    /// `finish` before any `begin`, and `tick` before any `begin`, must both
    /// be no-ops rather than panics -- an error can end a render at any point.
    #[test]
    fn ticking_or_finishing_without_a_phase_is_harmless() {
        let mut p = CliProgress::new();
        p.tick(5);
        p.finish();
        p.finish();
    }
}
