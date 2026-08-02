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

    /// The frame just processed, for implementations that show a live
    /// preview. Called for every frame, so an implementation that cares
    /// MUST throttle: copying and uploading tens of thousands of images
    /// would cost far more than the render itself.
    ///
    /// Default is a no-op, so terminal and headless reporters pay nothing
    /// beyond an ignored call.
    fn frame(&mut self, _width: u32, _height: u32, _rgba: &[u8]) {}

    /// Polled once per frame (not per pixel). Returning `true` aborts the
    /// render cleanly: the frame loop stops consuming its source and no
    /// output file is written, but this is NOT treated as an error -- a
    /// cancelled render must never surface as an error dialog or a
    /// crash-looking exit.
    ///
    /// Takes `&self`, not `&mut self`, on purpose: a caller that holds its
    /// own `Progress` value (rather than only a borrowed `&mut dyn Progress`
    /// handed to a callee) can ask again, after that callee returns, whether
    /// the render it just ran was cancelled -- without needing a second
    /// mutable borrow. That is exactly how the GUI decides whether to write
    /// the save a completed `build_brick_world` call just produced.
    ///
    /// Default is `false`: nothing except an explicit UI cancel button ever
    /// sets this, so the CLI, the library's tests, and the wasm build (none
    /// of which offer a way to cancel) can never trip it by accident.
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// How well a phase knows how much work it has, and what to tell the user.
///
/// A phase that streams frames has three genuinely different situations and
/// they must not be reported as though they were two. Before this existed, the
/// renderers passed `frame_count_hint` straight to [`Progress::begin`], which
/// collapsed the last two together: an `.mkv` (whose container stores no frame
/// count, so the hint is honestly `None`) got an indeterminate spinner that
/// sat there for minutes with no total and no explanation, which reads as a
/// hang rather than as "this source could not say".
///
/// - [`FrameTotal::Exact`] -- the source knows its own length. A plain bar.
/// - [`FrameTotal::Estimated`] -- only a computed estimate is available
///   (`FrameSource::frame_count_estimate`, e.g. `duration * fps`). Still a
///   bar, because a bar that is a frame or two out is far more useful than no
///   bar at all -- but LABELLED as an estimate, and the reporter must tolerate
///   the real count overshooting it (see [`FrameTotal::position`]).
/// - [`FrameTotal::Unknown`] -- nothing to go on. A spinner, but one that says
///   so, so it does not read as frozen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameTotal {
    Exact(u64),
    Estimated(u64),
    Unknown,
}

impl FrameTotal {
    /// Pick the best of what a source could say: its exact count if it has
    /// one, otherwise its estimate, otherwise nothing.
    ///
    /// An estimate of 0 is treated as no estimate: a zero-length bar has no
    /// denominator to show and would render as an instantly-complete gauge in
    /// front of a render that is about to do real work.
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

    /// `phase`, with a suffix naming which of the three cases this is.
    ///
    /// The suffix is the whole point of the `Unknown` arm: an unlabelled
    /// spinner is indistinguishable from a stalled one, and this is the only
    /// place the user is told the source never reported a length.
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

    /// The position and length a bar should DISPLAY at cumulative tick
    /// `ticks`, or `None` for [`FrameTotal::Unknown`] -- a spinner has no
    /// length to reconcile. See [`reconcile_total`] for the rule.
    pub fn position(&self, ticks: u64) -> Option<(u64, u64)> {
        let (pos, len) = reconcile_total(self.total(), ticks);
        len.map(|len| (pos, len))
    }
}

/// Reconcile a cumulative tick against a total that may have come from an
/// ESTIMATE, giving the position and length a reporter should display.
///
/// An estimate can be low, and a render that runs past it must not produce a
/// fraction above 1 -- or, in a reporter that computes a remainder by
/// subtraction, an underflow. **The length grows to meet the position rather
/// than the position being clamped to the length**: clamping would freeze the
/// `pos/len` readout at the estimate while the render visibly kept going,
/// which is the exact "looks stalled" impression the estimate was introduced
/// to remove. Growing keeps the count honest and pins the bar at full, which
/// is precisely what "we are at or past the estimate" means.
///
/// Every reporter routes through this one function -- the CLI bar, both GUI
/// panes, and [`FrameTotal::position`] -- so none of them can quietly adopt a
/// different rule.
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

    /// The library default must be inert -- tests and the wasm build carry no
    /// terminal or UI dependency.
    #[test]
    fn no_progress_swallows_everything_without_panicking() {
        let mut p = NoProgress;
        p.begin("scanning", None);
        p.tick(1);
        p.frame(1, 1, &[0, 0, 0, 0]);
        p.finish();
    }

    /// `frame`'s default body must not break object safety: the trait is
    /// used everywhere as `&mut dyn Progress`, so a `&[u8]` parameter and an
    /// unimplemented default both have to survive dynamic dispatch.
    #[test]
    fn frame_is_callable_through_a_trait_object_and_defaults_to_a_no_op() {
        let mut p = NoProgress;
        let dyn_p: &mut dyn Progress = &mut p;
        dyn_p.frame(2, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

        let mut r = Recorder::default();
        let dyn_r: &mut dyn Progress = &mut r;
        dyn_r.frame(1, 1, &[0, 0, 0, 0]);
        assert!(
            r.events.is_empty(),
            "Recorder never overrides `frame`, so the default no-op must leave it untouched"
        );
    }

    /// An unknown total is a real answer, not an error: a source that cannot
    /// say its length yields a spinner rather than a lying bar.
    #[test]
    fn an_unknown_total_is_accepted() {
        let mut r = Recorder::default();
        r.begin("decoding", None);
        assert_eq!(r.events[0], "begin:decoding:None");
    }

    /// An exact count wins over an estimate, and an estimate is used when
    /// there is no exact count. Before this, an absent exact count went
    /// straight to `begin(.., None)` and the estimate -- which the CLI already
    /// computes for its cost line -- was simply thrown away, so an `.mkv` got
    /// a totalless spinner for the whole render.
    #[test]
    fn an_estimate_is_used_only_when_there_is_no_exact_count() {
        assert_eq!(FrameTotal::new(Some(10), Some(99)), FrameTotal::Exact(10));
        assert_eq!(FrameTotal::new(None, Some(99)), FrameTotal::Estimated(99));
        assert_eq!(FrameTotal::new(None, None), FrameTotal::Unknown);
        assert_eq!(FrameTotal::new(Some(0), None), FrameTotal::Exact(0));
    }

    /// A zero estimate is no estimate. A `Some(0)` denominator would render as
    /// an already-finished bar in front of a render that has not started.
    #[test]
    fn a_zero_estimate_is_treated_as_no_estimate() {
        assert_eq!(FrameTotal::new(None, Some(0)), FrameTotal::Unknown);
    }

    /// The three cases must be distinguishable in the UI. In particular the
    /// spinner has to SAY it has no total -- an unlabelled spinner is
    /// indistinguishable from a hung one, which is the complaint that started
    /// this.
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

    /// **The graceful-degradation guarantee.** An estimate can be low, and a
    /// render that runs past it must not produce a fraction above 1 or panic.
    /// The length grows to meet the position rather than the position being
    /// clamped, so the count stays honest instead of freezing at the estimate.
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

    /// An estimate that overshoots (the render finishes early) is the other
    /// half of the same problem, and needs no special handling: the bar simply
    /// never reaches its end before `finish` clears it.
    #[test]
    fn an_overshooting_estimate_just_never_fills() {
        assert_eq!(FrameTotal::Estimated(1000).position(12), Some((12, 1000)));
    }

    /// A spinner has no length to reconcile, at any tick.
    #[test]
    fn an_unknown_total_has_no_position_to_report() {
        assert_eq!(FrameTotal::Unknown.position(0), None);
        assert_eq!(FrameTotal::Unknown.position(1_000_000), None);
    }

    /// `begin` must pass the labelled phase and the matching denominator
    /// through in one call, so no call site can pair one case's label with
    /// another's total.
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

    /// The default reporter never cancels -- the CLI, this library's own
    /// tests, and the wasm build have no cancel button, and must never trip
    /// this by accident.
    #[test]
    fn no_progress_never_cancels() {
        let p = NoProgress;
        assert!(!p.is_cancelled());
    }

    /// `Recorder` never overrides `is_cancelled` either, so it must fall
    /// back to the same default -- pinning that the default applies to any
    /// implementer, not just `NoProgress`.
    #[test]
    fn a_reporter_that_never_overrides_is_cancelled_defaults_to_false() {
        let r = Recorder::default();
        assert!(!r.is_cancelled());
    }

    /// `is_cancelled` must survive dynamic dispatch (the trait is used
    /// everywhere as `&mut dyn Progress`) AND remain callable on a plain
    /// `&dyn Progress` taken *after* a mutable borrow ends -- the exact
    /// shape the GUI relies on to ask "was this cancelled?" once the
    /// `&mut dyn Progress` handed to `build_brick_world` is no longer
    /// borrowed.
    #[test]
    fn is_cancelled_is_callable_through_a_trait_object_and_after_a_mutable_borrow_ends() {
        struct CancelOnce {
            cancelled: bool,
        }
        impl Progress for CancelOnce {
            fn begin(&mut self, _label: &str, _total: Option<u64>) {}
            fn tick(&mut self, _n: u64) {}
            fn finish(&mut self) {}
            fn is_cancelled(&self) -> bool {
                self.cancelled
            }
        }

        let mut c = CancelOnce { cancelled: false };
        {
            let dyn_c: &mut dyn Progress = &mut c;
            dyn_c.tick(1);
            assert!(!dyn_c.is_cancelled(), "not cancelled yet");
        }
        c.cancelled = true;
        // No more mutable borrow in scope here -- `&c` (immutable) is enough,
        // exactly what a caller re-checking after a callee returns needs.
        assert!(c.is_cancelled());
    }
}
