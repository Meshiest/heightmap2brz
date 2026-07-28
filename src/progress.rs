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
}
