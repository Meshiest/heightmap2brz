//! Progress reporting, kept out of the core so the library and its tests
//! carry no UI dependency.
//!
//! Phases are named rather than anonymous because a later stage runs two of
//! them back to back ("scanning" then "packing"), and a bar that silently
//! restarts at zero reads as a bug.

pub trait Progress {
    /// Start a phase. `total` is `None` when the work length is unknown, in
    /// which case a reporter should show an indeterminate spinner rather than
    /// inventing a denominator.
    fn begin(&mut self, label: &str, total: Option<u64>);
    /// Report cumulative progress within the current phase.
    fn tick(&mut self, n: u64);
    fn finish(&mut self);

    /// The frame just processed, for implementations with a live preview.
    /// Called every frame, so an implementation that cares must throttle --
    /// copying/uploading tens of thousands of images costs more than the
    /// render itself. Default is a no-op.
    fn frame(&mut self, _width: u32, _height: u32, _rgba: &[u8]) {}

    /// Polled once per frame. `true` aborts the render cleanly: no output is
    /// written, and this is NOT an error. Takes `&self` so a caller can re-ask
    /// after a callee it lent `&mut dyn Progress` to returns.
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// How well a phase knows its length. Exact/Estimated both draw a bar (an
/// estimate may be low; see `reconcile_total`); Unknown draws a labelled
/// spinner so it does not read as frozen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameTotal {
    Exact(u64),
    Estimated(u64),
    Unknown,
}

impl FrameTotal {
    /// Best of what a source can say: exact count, else estimate, else
    /// nothing. A zero estimate counts as no estimate -- it would render as
    /// an already-finished bar in front of a render that hasn't started.
    pub fn new(exact: Option<usize>, estimate: Option<usize>) -> Self {
        match (exact, estimate) {
            (Some(n), _) => FrameTotal::Exact(n as u64),
            (None, Some(n)) if n > 0 => FrameTotal::Estimated(n as u64),
            _ => FrameTotal::Unknown,
        }
    }

    /// The denominator to hand [`Progress::begin`], or `None` for a spinner.
    pub fn total(&self) -> Option<u64> {
        match self {
            FrameTotal::Exact(n) | FrameTotal::Estimated(n) => Some(*n),
            FrameTotal::Unknown => None,
        }
    }

    /// `phase` with a suffix naming which case this is -- the only place the
    /// user learns a source reported no length at all.
    pub fn label(&self, phase: &str) -> String {
        match self {
            FrameTotal::Exact(_) => phase.to_string(),
            FrameTotal::Estimated(_) => format!("{phase} (estimated total)"),
            FrameTotal::Unknown => format!("{phase} (total unknown -- source did not report one)"),
        }
    }

    /// `begin` this phase on `progress` with the right label and denominator.
    pub fn begin(&self, progress: &mut dyn Progress, phase: &str) {
        progress.begin(&self.label(phase), self.total());
    }

    /// Position and length a bar should display at cumulative tick `ticks`,
    /// or `None` for [`FrameTotal::Unknown`]. See [`reconcile_total`].
    pub fn position(&self, ticks: u64) -> Option<(u64, u64)> {
        let (pos, len) = reconcile_total(self.total(), ticks);
        len.map(|len| (pos, len))
    }
}

/// Grows the length to meet an over-run tick rather than clamping, so an
/// under-estimate neither freezes the readout nor underflows a remainder.
pub fn reconcile_total(total: Option<u64>, ticks: u64) -> (u64, Option<u64>) {
    (ticks, total.map(|len| len.max(ticks)))
}

/// The default reporter: renders nothing.
pub struct NoProgress;

impl Progress for NoProgress {
    fn begin(&mut self, _label: &str, _total: Option<u64>) {}
    fn tick(&mut self, _n: u64) {}
    fn finish(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Recorder {
        events: Vec<String>,
    }
    impl Progress for Recorder {
        fn begin(&mut self, label: &str, total: Option<u64>) {
            self.events.push(format!("begin:{label}:{total:?}"));
        }
        fn tick(&mut self, n: u64) {
            self.events.push(format!("tick:{n}"));
        }
        fn finish(&mut self) {
            self.events.push("finish".into());
        }
    }

    #[test]
    fn a_custom_reporter_sees_every_event() {
        let mut r = Recorder::default();
        r.begin("packing", Some(3));
        r.tick(1);
        r.tick(2);
        r.finish();
        assert_eq!(r.events, ["begin:packing:Some(3)", "tick:1", "tick:2", "finish"]);
    }

    #[test]
    fn no_progress_swallows_everything_without_panicking() {
        let mut p = NoProgress;
        p.begin("scanning", None);
        p.tick(1);
        p.frame(1, 1, &[0, 0, 0, 0]);
        p.finish();
    }

    #[test]
    fn an_unknown_total_is_accepted() {
        let mut r = Recorder::default();
        r.begin("decoding", None);
        assert_eq!(r.events[0], "begin:decoding:None");
    }

    #[test]
    fn an_estimate_is_used_only_when_there_is_no_exact_count() {
        assert_eq!(FrameTotal::new(Some(10), Some(99)), FrameTotal::Exact(10));
        assert_eq!(FrameTotal::new(None, Some(99)), FrameTotal::Estimated(99));
        assert_eq!(FrameTotal::new(None, None), FrameTotal::Unknown);
        assert_eq!(FrameTotal::new(Some(0), None), FrameTotal::Exact(0));
    }

    #[test]
    fn a_zero_estimate_is_treated_as_no_estimate() {
        assert_eq!(FrameTotal::new(None, Some(0)), FrameTotal::Unknown);
    }

    #[test]
    fn each_case_labels_itself_and_only_the_known_ones_carry_a_total() {
        assert_eq!(FrameTotal::Exact(5).label("packing frames"), "packing frames");
        assert_eq!(FrameTotal::Exact(5).total(), Some(5));

        let est = FrameTotal::Estimated(5).label("packing frames");
        assert!(est.contains("packing frames") && est.contains("estimated"), "{est}");
        assert_eq!(FrameTotal::Estimated(5).total(), Some(5));

        let unknown = FrameTotal::Unknown.label("packing frames");
        assert!(
            unknown.contains("packing frames") && unknown.contains("unknown"),
            "a spinner must say why it has no total, got {unknown:?}"
        );
        assert_eq!(FrameTotal::Unknown.total(), None);
    }

    #[test]
    fn ticking_past_an_estimate_grows_the_length_instead_of_overflowing_it() {
        let est = FrameTotal::Estimated(100);
        assert_eq!(est.position(0), Some((0, 100)));
        assert_eq!(est.position(50), Some((50, 100)));
        assert_eq!(est.position(100), Some((100, 100)));
        // Past the estimate: full bar, still-advancing count, never pos > len.
        assert_eq!(est.position(137), Some((137, 137)));
        assert_eq!(est.position(u64::MAX), Some((u64::MAX, u64::MAX)));
        for ticks in [0u64, 1, 99, 100, 101, 10_000] {
            let (pos, len) = est.position(ticks).expect("an estimate has a length");
            assert!(pos <= len, "{pos} > {len} at tick {ticks}");
        }
    }

    #[test]
    fn an_overshooting_estimate_just_never_fills() {
        assert_eq!(FrameTotal::Estimated(1000).position(12), Some((12, 1000)));
    }

    #[test]
    fn an_unknown_total_has_no_position_to_report() {
        assert_eq!(FrameTotal::Unknown.position(0), None);
        assert_eq!(FrameTotal::Unknown.position(1_000_000), None);
    }

    #[test]
    fn begin_passes_the_label_and_the_total_together() {
        let mut r = Recorder::default();
        FrameTotal::Exact(7).begin(&mut r, "packing frames");
        FrameTotal::Estimated(7).begin(&mut r, "packing frames");
        FrameTotal::Unknown.begin(&mut r, "packing frames");
        assert_eq!(r.events[0], "begin:packing frames:Some(7)");
        assert!(r.events[1].starts_with("begin:packing frames (estimated"), "{}", r.events[1]);
        assert!(r.events[2].ends_with(":None"), "{}", r.events[2]);
    }

    #[test]
    fn no_progress_never_cancels() {
        let p = NoProgress;
        assert!(!p.is_cancelled());
    }

    #[test]
    fn a_reporter_that_never_overrides_is_cancelled_defaults_to_false() {
        let r = Recorder::default();
        assert!(!r.is_cancelled());
    }
}
