//! Terminal progress for the CLI. Native only: the wasm build has no
//! terminal and the GUI reports through egui instead.
use indicatif::{ProgressBar, ProgressStyle};

use heightmap::progress::Progress;

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

    fn tick(&mut self, n: u64) {
        if let Some(b) = &self.bar {
            b.set_position(n);
        }
    }

    fn finish(&mut self) {
        if let Some(b) = self.bar.take() {
            b.finish_and_clear();
        }
    }
}
