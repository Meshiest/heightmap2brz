//! Interaction and layout guards for the two accordion panes.
//!
//! egui is deterministic and needs no window to run, so both of these are real
//! end-to-end checks rather than approximations: `Context::run` is driven with
//! synthetic pointer events and the resulting paint list is inspected.
//!
//! # Why a test for "the header toggles"
//!
//! It regressed once, silently, and in a way no type or compile check could
//! catch: `Interaction::selectable_labels` defaults to ON, so the plain
//! `ui.label` that drew a section title was a CLICK-SENSING widget (egui lets
//! you select label text) and won the hit-test over the clickable header row
//! underneath it. The section then opened only from the little triangle. The
//! fix -- drawing the header's title and chips with `.selectable(false)` so
//! clicks fall through -- is invisible in the source unless you know that
//! story, which is exactly the kind of thing that gets "tidied" away.
#![cfg(feature = "gui")]

use heightmap::gui::{SharedOptions, audio::AudioApp, video::VideoApp};

/// Drives a pane the way `gui::app` composes it, and remembers what was
/// painted.
struct Pane<'a> {
    ctx: egui::Context,
    draw: &'a mut dyn FnMut(&mut egui::Ui),
    width: f32,
    texts: Vec<(egui::Rect, String)>,
}

impl<'a> Pane<'a> {
    fn new(width: f32, draw: &'a mut dyn FnMut(&mut egui::Ui)) -> Self {
        let mut p = Pane { ctx: egui::Context::default(), draw, width, texts: Vec::new() };
        // A few frames to settle: an `egui::Grid` learns its column widths
        // from the previous frame, so the first one is not representative.
        for _ in 0..4 {
            p.frame(Vec::new());
        }
        p
    }

    fn frame(&mut self, events: Vec<egui::Event>) {
        let screen =
            egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(self.width, 2400.0));
        let input = egui::RawInput { screen_rect: Some(screen), events, ..Default::default() };
        let draw = &mut *self.draw;
        let out = self.ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .max_height((ui.available_height() - 50.0).max(50.0))
                    .show(ui, |ui| draw(ui));
            });
        });
        self.texts = out
            .shapes
            .iter()
            .filter_map(|c| match &c.shape {
                egui::epaint::Shape::Text(t) => Some((
                    egui::Rect::from_min_size(t.pos, t.galley.rect.size()),
                    t.galley.text().to_string(),
                )),
                _ => None,
            })
            .collect();
    }

    fn click(&mut self, pos: egui::Pos2) {
        let modifiers = egui::Modifiers::default();
        let button = egui::PointerButton::Primary;
        // Hover on its own frame first: egui resolves interaction against the
        // PREVIOUS frame's widget rects, so a press arriving in the same frame
        // as the pointer's first appearance has nothing to land on yet.
        self.frame(vec![egui::Event::PointerMoved(pos)]);
        self.frame(vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton { pos, button, pressed: true, modifiers },
        ]);
        self.frame(vec![egui::Event::PointerButton {
            pos,
            button,
            pressed: false,
            modifiers,
        }]);
        // Let the open/close animation finish, so `shows` reads a settled pane.
        for _ in 0..40 {
            self.frame(Vec::new());
        }
    }

    fn rect_of(&self, text: &str) -> Option<egui::Rect> {
        self.texts.iter().find(|(_, s)| s == text).map(|(r, _)| *r)
    }

    fn shows(&self, text: &str) -> bool {
        self.rect_of(text).is_some()
    }

    /// The first chip drawn after `title` on the same header row.
    fn chip_after(&self, title: &str) -> Option<egui::Rect> {
        let t = self.rect_of(title)?;
        self.texts
            .iter()
            .filter(|(r, s)| {
                s != title && (r.center().y - t.center().y).abs() < 6.0 && r.left() > t.right()
            })
            .min_by(|a, b| a.0.left().total_cmp(&b.0.left()))
            .map(|(r, _)| *r)
    }
}

/// Where in a section header to click.
#[derive(Clone, Copy, Debug)]
enum Target {
    Title,
    Chip,
    /// The empty space to the right of the last chip -- a header row is only
    /// as wide as its content unless it is deliberately stretched, so this is
    /// its own case and not a duplicate of `Title`.
    EmptySpace,
    /// The collapsing triangle, which worked even while the rest did not.
    Arrow,
}

fn point(pane: &Pane, title: &str, target: Target) -> Option<egui::Pos2> {
    let t = pane.rect_of(title)?;
    Some(match target {
        Target::Title => t.center(),
        Target::Chip => pane.chip_after(title)?.center(),
        Target::EmptySpace => egui::pos2(pane.width - 40.0, t.center().y),
        Target::Arrow => egui::pos2(t.left() - 9.0, t.center().y),
    })
}

/// Section header title -> a label that only its BODY draws, so "is this
/// section open" can be read off the paint list.
const AUDIO_SECTIONS: [(&str, &str); 5] = [
    ("Analysis", "Analysis FPS"),
    ("Band grid", "Subdiv"),
    ("Envelope", "Attack"),
    ("Levels", "Gain"),
    ("Speaker placement", "Inner Radius"),
];

const VIDEO_SECTIONS: [(&str, &str); 4] = [
    ("Text options", "Colours"),
    ("Subtitles", "Subtitle Scale"),
    ("Scaling & timing", "Fit Mode"),
    ("Picture", "Alpha Threshold"),
];

const TARGETS: [Target; 4] =
    [Target::Title, Target::Chip, Target::EmptySpace, Target::Arrow];

fn assert_toggles(
    pane_name: &str,
    title: &str,
    body_marker: &str,
    target: Target,
    pane: &mut Pane,
) {
    assert!(
        !pane.shows(body_marker),
        "{pane_name}/{title}: expected to start collapsed, but {body_marker:?} was already drawn"
    );

    let pos = point(pane, title, target)
        .unwrap_or_else(|| panic!("{pane_name}/{title}: no {target:?} to click"));
    pane.click(pos);
    assert!(
        pane.shows(body_marker),
        "{pane_name}/{title}: clicking the {target:?} did not OPEN the section"
    );

    // Re-locate: opening the section may have moved the header of the ones
    // below it, and clicking a stale position would prove nothing.
    let pos = point(pane, title, target)
        .unwrap_or_else(|| panic!("{pane_name}/{title}: header vanished once open"));
    pane.click(pos);
    assert!(
        !pane.shows(body_marker),
        "{pane_name}/{title}: clicking the {target:?} did not CLOSE the section again \
         (this is the exact regression the module doc describes)"
    );
}

#[test]
fn every_audio_section_opens_and_closes_from_anywhere_in_its_header() {
    for target in TARGETS {
        for (title, body_marker) in AUDIO_SECTIONS {
            let mut shared = SharedOptions::default();
            let mut app = AudioApp::default();
            let mut draw = |ui: &mut egui::Ui| app.draw(ui, &mut shared);
            let mut pane = Pane::new(900.0, &mut draw);
            assert_toggles("audio", title, body_marker, target, &mut pane);
        }
    }
}

#[test]
fn every_video_section_opens_and_closes_from_anywhere_in_its_header() {
    for target in TARGETS {
        for (title, body_marker) in VIDEO_SECTIONS {
            let mut shared = SharedOptions::default();
            let mut app = VideoApp::default();
            let mut draw = |ui: &mut egui::Ui| app.draw(ui, &mut shared);
            let mut pane = Pane::new(900.0, &mut draw);
            assert_toggles("video", title, body_marker, target, &mut pane);
        }
    }
}

/// Nothing may run off the right edge, and no two pieces of text may overlap.
///
/// Both failures were real: a wrapping horizontal layout inside an
/// `egui::Grid` cell reports a height short of what it drew, so the next row
/// was placed on top of it, and rows too wide for the pane widened the grid
/// until a horizontal scrollbar appeared. Neither is visible to a type check.
fn assert_lays_out_cleanly(name: &str, width: f32, draw: &mut dyn FnMut(&mut egui::Ui)) {
    let pane = Pane::new(width, draw);
    assert_pane_is_clean(&pane, name, width);
}

/// The body of [`assert_lays_out_cleanly`], against a pane the caller already
/// holds -- so a test that had to CLICK a section open first can reuse the
/// same rules rather than restating them.
fn assert_pane_is_clean(pane: &Pane, name: &str, width: f32) {
    for (rect, text) in &pane.texts {
        assert!(
            rect.right() <= width - 2.0,
            "{name} @ {width}: {text:?} runs to {} past the right edge",
            rect.right()
        );
    }
    for (i, (a, sa)) in pane.texts.iter().enumerate() {
        for (b, sb) in pane.texts.iter().skip(i + 1) {
            let overlap_y = a.bottom().min(b.bottom()) - a.top().max(b.top());
            let overlap_x = a.right().min(b.right()) - a.left().max(b.left());
            assert!(
                overlap_y <= 1.5 || overlap_x <= 1.5,
                "{name} @ {width}: {sa:?} and {sb:?} are drawn on top of each other"
            );
        }
    }
}

/// Open every section of a pane, so the rows inside can be measured.
fn open_every_section(pane: &mut Pane, name: &str, sections: &[(&str, &str)]) {
    for (title, body_marker) in sections {
        if pane.shows(body_marker) {
            continue;
        }
        // Re-located every time: opening one section moves the headers below it.
        let pos = point(pane, title, Target::Title)
            .unwrap_or_else(|| panic!("{name}: no {title:?} header to open"));
        pane.click(pos);
        assert!(
            pane.shows(body_marker),
            "{name}: {title:?} did not open, so its rows cannot be measured"
        );
    }
}

/// Every row in every section is label-in-column-1, control-in-column-2.
///
/// The failure this catches: wrapping a label AND its control in one
/// `add_enabled_ui` (or any single child `Ui`) makes the grid treat the pair as
/// ONE cell, so the control sits just right of its own label instead of at the
/// column boundary. `Fit Mode` and `Filter` did that, and the section read as
/// two unrelated tables stacked on top of each other.
///
/// Stated as a geometric invariant rather than a list of expected labels, so it
/// keeps holding as rows are added: **nothing may be drawn in the middle of the
/// label column.** A label starts at the column's left edge; a control starts at
/// the next column. Anything landing in between is a miscounted cell.
///
/// The second assertion guards `LABEL_COLUMN_WIDTH` itself. That constant is
/// what makes separate sections agree on where column 2 begins (each section is
/// its own `Grid` and would otherwise size column 1 to its own widest label), so
/// a label growing past it would silently pull one section out of line.
fn assert_columns_line_up(pane: &Pane, name: &str, section_titles: &[&str]) {
    let column_width = heightmap::gui::util::LABEL_COLUMN_WIDTH;

    // Header rows carry the title and its chips, which are not grid cells and
    // do not obey the column geometry.
    let header_bands: Vec<egui::Rect> = section_titles
        .iter()
        .filter_map(|t| pane.rect_of(t))
        .collect();
    let first_header = header_bands
        .iter()
        .map(|r| r.top())
        .fold(f32::INFINITY, f32::min);
    // Everything below the sections is the Source / cost / submit block, which
    // is not a grid either.
    let end = pane
        .rect_of("Source")
        .map_or(f32::INFINITY, |r| r.top());

    let rows: Vec<&(egui::Rect, String)> = pane
        .texts
        .iter()
        .filter(|(r, _)| r.top() > first_header && r.bottom() < end)
        .filter(|(r, _)| {
            !header_bands
                .iter()
                .any(|h| r.center().y > h.top() - 2.0 && r.center().y < h.bottom() + 2.0)
        })
        .collect();
    assert!(
        rows.len() > 10,
        "{name}: only {} row texts found, the section scan is wrong",
        rows.len()
    );

    let c1 = rows.iter().map(|(r, _)| r.left()).fold(f32::INFINITY, f32::min);

    for (rect, text) in &rows {
        assert!(
            rect.left() <= c1 + 1.0 || rect.left() >= c1 + column_width - 1.0,
            "{name}: {text:?} starts at {:.1}, inside the label column (which runs \
             {c1:.1}..{:.1}) -- its row is putting a label and a control in one grid cell",
            rect.left(),
            c1 + column_width
        );
        if rect.left() <= c1 + 1.0 {
            assert!(
                rect.right() <= c1 + column_width - 2.0,
                "{name}: the label {text:?} is wider than LABEL_COLUMN_WIDTH, so its \
                 section's column 2 no longer lines up with the others -- raise the constant",
            );
        }
    }
}

/// Collapsed section headers sit directly under one another.
///
/// The failure this catches: widening the header's click target by ALLOCATING
/// a zero-height, full-width rect inside it. That looks free and is not -- the
/// header is a `main_wrap` layout, an exact-width item is allowed to spill onto
/// a second line, and a wrapped zero-height item still costs a full
/// `item_spacing.y`. Every collapsed header grew a blank line under it, and the
/// sections drifted twice as far apart as they should be (measured: 42 point
/// pitch against the correct 21).
///
/// The bound is expressed in terms of the header's own text height rather than
/// as a magic number, so it survives a font or zoom change: the gap between one
/// header and the next may not be as tall as a line of text.
fn assert_headers_are_not_spread_apart(pane: &Pane, name: &str, section_titles: &[&str]) {
    let mut rects: Vec<egui::Rect> = section_titles
        .iter()
        .filter_map(|t| pane.rect_of(t))
        .collect();
    assert_eq!(
        rects.len(),
        section_titles.len(),
        "{name}: not every section header was drawn"
    );
    rects.sort_by(|a, b| a.top().total_cmp(&b.top()));

    for pair in rects.windows(2) {
        let (upper, lower) = (pair[0], pair[1]);
        let gap = lower.top() - upper.bottom();
        assert!(
            gap <= upper.height(),
            "{name}: {gap:.1} points between two collapsed headers, more than the \
             {:.1} a line of text occupies -- something in the header row is \
             allocating layout space it should not",
            upper.height()
        );
    }
}

#[test]
fn collapsed_sections_stack_without_a_blank_line_between_them() {
    for (name, titles) in [
        ("audio", AUDIO_SECTIONS.iter().map(|(t, _)| *t).collect::<Vec<_>>()),
        ("video", VIDEO_SECTIONS.iter().map(|(t, _)| *t).collect::<Vec<_>>()),
    ] {
        let mut shared = SharedOptions::default();
        let mut audio = AudioApp::default();
        let mut video = VideoApp::default();
        let mut draw = |ui: &mut egui::Ui| {
            if name == "audio" {
                audio.draw(ui, &mut shared);
            } else {
                video.draw(ui, &mut shared);
            }
        };
        let pane = Pane::new(900.0, &mut draw);
        assert_headers_are_not_spread_apart(&pane, name, &titles);
    }
}

#[test]
fn every_section_uses_the_same_two_column_geometry() {
    let audio_titles: Vec<&str> = AUDIO_SECTIONS.iter().map(|(t, _)| *t).collect();
    let video_titles: Vec<&str> = VIDEO_SECTIONS.iter().map(|(t, _)| *t).collect();

    let mut shared = SharedOptions::default();
    let mut app = AudioApp::default();
    let mut draw = |ui: &mut egui::Ui| app.draw(ui, &mut shared);
    let mut pane = Pane::new(900.0, &mut draw);
    open_every_section(&mut pane, "audio", &AUDIO_SECTIONS);
    assert_columns_line_up(&pane, "audio", &audio_titles);

    let mut shared = SharedOptions::default();
    let mut app = VideoApp::default();
    let mut draw = |ui: &mut egui::Ui| app.draw(ui, &mut shared);
    let mut pane = Pane::new(900.0, &mut draw);
    open_every_section(&mut pane, "video", &VIDEO_SECTIONS);
    assert_columns_line_up(&pane, "video", &video_titles);
}

#[test]
fn both_panes_lay_out_without_overlap_or_overflow() {
    // The narrow end is below the 600px default window, the wide end well past
    // it: the grid's column sizing behaves differently at each.
    for width in [520.0, 600.0, 900.0] {
        let mut shared = SharedOptions::default();
        let mut app = AudioApp::default();
        assert_lays_out_cleanly("audio", width, &mut |ui| app.draw(ui, &mut shared));

        let mut shared = SharedOptions::default();
        let mut app = VideoApp::default();
        assert_lays_out_cleanly("video", width, &mut |ui| app.draw(ui, &mut shared));
    }
}

/// The Loop toggle is on the ALWAYS-VISIBLE critical path in both panes -- not
/// behind a collapsing section -- and lays out cleanly.
///
/// Whether a clip loops is a decision every render makes, so it sits on the flat
/// grid beside the mode and the frame rate rather than in an advanced section a
/// user has to open. This asserts it is drawn with NO section clicked, which is
/// the whole point of the move: opening a section first would let the test pass
/// even if the toggle had been left buried, the exact failure this shape guards
/// against.
///
/// The label AND the checkbox text are both asserted: a row whose label rendered
/// but whose widget did not would otherwise pass.
#[test]
fn the_loop_toggle_is_always_visible_in_both_panes() {
    for width in [520.0, 600.0, 900.0] {
        for name in ["video", "audio"] {
            let mut shared = SharedOptions::default();
            let mut video = VideoApp::default();
            let mut audio = AudioApp::default();
            let mut draw = |ui: &mut egui::Ui| {
                if name == "video" {
                    video.draw(ui, &mut shared);
                } else {
                    audio.draw(ui, &mut shared);
                }
            };
            // No click: every collapsing section is shut on a default first
            // draw, so anything visible here is on the always-visible grid.
            let pane = Pane::new(width, &mut draw);

            assert!(
                pane.shows("Playback"),
                "{name} @ {width}: the Playback row's label must be drawn with no section opened"
            );
            assert!(
                pane.shows("Loop"),
                "{name} @ {width}: the Loop checkbox must be drawn with no section opened"
            );
            assert_pane_is_clean(&pane, &format!("{name}/loop"), width);
        }
    }
}

/// **A Save Destination the tool cannot write must BLOCK Generate, not just
/// colour a label red.**
///
/// Every pane drew the "must end with .brz or .brdb" warning and then offered
/// the button anyway, so the surface that showed the warning was the one that
/// wrote to the bad name -- while the CLI, which showed nothing up front,
/// refused it. Driven through a real frame here rather than by calling a
/// predicate, because the thing that regressed is what the pane DRAWS: the
/// refusal has to be painted and the Generate button has to be gone.
#[test]
fn a_bad_save_destination_blocks_generate_in_both_panes() {
    for (name, button) in [
        ("video", "Generate video2brick save"),
        ("audio", "Generate audio2brick save"),
    ] {
        // A legal name first, so the test can tell "the button is missing
        // because of the destination" from "this pane never draws one".
        for (out_file, refused) in [("out.brz", false), ("out", true), ("", true)] {
            let mut shared = SharedOptions { out_file: out_file.to_string(), out_clipboard: false };
            let mut video = VideoApp::default();
            let mut audio = AudioApp::default();
            let mut draw = |ui: &mut egui::Ui| {
                if name == "video" {
                    video.draw(ui, &mut shared);
                } else {
                    audio.draw(ui, &mut shared);
                }
            };
            let pane = Pane::new(900.0, &mut draw);

            let shows_refusal = pane
                .texts
                .iter()
                .any(|(_, s)| s.starts_with("Cannot render:"));
            assert_eq!(
                shows_refusal, refused,
                "{name} with out_file {out_file:?}: expected refused={refused}, drew: {:?}",
                pane.texts.iter().map(|(_, s)| s.as_str()).collect::<Vec<_>>()
            );
            assert!(
                !pane.shows(button),
                "{name} with out_file {out_file:?}: no input is picked, so Generate must not \
                 be offered at all"
            );
        }
    }
}

/// The warning ROW is drawn for the same names, and only for them: the label
/// and the gate above must agree, or the pane is refusing something it never
/// explained.
#[test]
fn the_save_destination_warning_row_matches_the_gate() {
    for (out_file, warned) in [("out.brz", false), ("out.brdb", false), ("out", true)] {
        let mut shared = SharedOptions { out_file: out_file.to_string(), out_clipboard: false };
        let mut app = VideoApp::default();
        let mut draw = |ui: &mut egui::Ui| app.draw(ui, &mut shared);
        let pane = Pane::new(900.0, &mut draw);

        assert_eq!(
            pane.shows("Output file must end with .brz or .brdb"),
            warned,
            "out_file {out_file:?}: the warning row must follow the same rule as the gate"
        );
    }
}
