//! Brickadia-styled widgets: semantic buttons, a sliding on/off toggle,
//! outlined title text and a header/body section panel. All are hand-painted
//! (rather than restyled egui widgets) so they can match the game exactly —
//! buttons that recolor but never grow on hover and sink their text on press, a
//! toggle whose square knob slides red→green, and titles with a chunky dark
//! outline.
use super::*;
use egui::{
    Align2, Color32, CornerRadius, FontId, Frame, Margin, Rect, Response, Sense, TextStyle, Ui,
    Vec2, pos2, vec2,
};

/// Pixels the text/symbol drops while a button/knob is held — the game's
/// `translateY` on `:active`.
const SINK: f32 = 2.0;
/// Radius of the thick, rounded title outline.
const OUTLINE_W: f32 = 3.0;
/// Constant downward nudge for the toggle knob symbol so the (up-tweaked) icon
/// font sits centered in the knob.
const SYMBOL_NUDGE: f32 = 2.0;
/// Horizontal padding inside a [`section`] body — and the amount [`full_bleed`]
/// cancels so a table's rows reach the panel edges.
pub const CELL_PAD: f32 = 10.0;
/// Button horizontal padding — smaller on the left so a leading icon reads
/// visually centered inside the pill rather than pushed right.
const PAD_L: f32 = 12.0;
const PAD_R: f32 = 18.0;

// ---------------------------------------------------------------------------
// Semantic buttons
// ---------------------------------------------------------------------------

/// Green primary action (Generate, Build, …). `$br-main-*`.
pub fn primary(ui: &mut Ui, label: impl Into<String>) -> Response {
    button(ui, label, MAIN, MAIN_HOVER, MAIN_PRESSED, BUTTON_FG)
}

/// Blue informational action (Pick file, Preview, …). `$br-info-*`.
pub fn info(ui: &mut Ui, label: impl Into<String>) -> Response {
    button(ui, label, INFO, INFO_HOVER, INFO_PRESSED, BUTTON_FG)
}

/// Red destructive action (Remove, Clear, …). `$br-error-*`.
pub fn danger(ui: &mut Ui, label: impl Into<String>) -> Response {
    button(ui, label, ERROR, ERROR_HOVER, ERROR_PRESSED, BUTTON_FG)
}

/// Orange cautionary action. `$br-warn-*`.
pub fn warn(ui: &mut Ui, label: impl Into<String>) -> Response {
    button(ui, label, WARN, WARN_HOVER, WARN_PRESSED, BUTTON_FG)
}

/// Neutral blue-gray button (Back, Stop, secondary actions). `$br-element-*`.
pub fn neutral(ui: &mut Ui, label: impl Into<String>) -> Response {
    button(ui, label, ELEMENT, ELEMENT_HOVER, ELEMENT_PRESSED, BORING_FG)
}

/// A small inline loading indicator (spinner + label) shown while a background
/// job runs — the same treatment for Generate and Preview. Requests a repaint
/// so the owning pane keeps polling its promise.
pub fn loading(ui: &mut Ui, label: &str) {
    ui.horizontal(|ui| {
        ui.spinner();
        ui.label(label);
    });
    ui.ctx().request_repaint();
}

/// A slider with a pointing-hand cursor on hover (egui gives sliders none).
/// Drop-in: `widgets::slider(ui, egui::Slider::new(&mut x, a..=b).text("u"))`.
pub fn slider(ui: &mut Ui, slider: egui::Slider) -> Response {
    ui.add(slider).on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// A red rounded-square icon-only button (Remove/Clear ✖) — square, with the
/// theme's corner radius, like the toggle knob.
pub fn danger_icon(ui: &mut Ui, glyph: &str) -> Response {
    icon_button(ui, glyph, ERROR, ERROR_HOVER, ERROR_PRESSED, BUTTON_FG)
}

/// One rounded-square icon-only button: a square with the theme corner radius
/// (matching the toggle knob), the glyph centered (and sunk on press).
fn icon_button(
    ui: &mut Ui,
    glyph: &str,
    normal: Color32,
    hover: Color32,
    pressed: Color32,
    fg: Color32,
) -> Response {
    let h = control_height(ui);
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(h), Sense::click());
    let down = resp.is_pointer_button_down_on();
    let bg = if down {
        pressed
    } else if resp.hovered() {
        hover
    } else {
        normal
    };
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(RADIUS), bg);
    let mut c = rect.center();
    c.y += SYMBOL_NUDGE; // compensate the icon font's upward tweak so it centers
    if down {
        c.y += SINK;
    }
    painter.text(
        c.round(),
        Align2::CENTER_CENTER,
        glyph,
        FontId::new(h * 0.5, icon_family()),
        fg,
    );
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// A larger, hand-painted radio (egui's is tiny) sized to the control row, with
/// a blue-gray ring, an info-blue dot when selected and a pointer cursor.
/// Drop-in: `widgets::radio(ui, &mut current, value, "label")`.
pub fn radio<V: PartialEq>(
    ui: &mut Ui,
    current: &mut V,
    value: V,
    label: impl Into<String>,
) -> Response {
    let selected = *current == value;
    let d = 20.0;
    let gap = 6.0;
    let galley = ui
        .painter()
        .layout_no_wrap(label.into(), TextStyle::Body.resolve(ui.style()), TEXT);
    let h = control_height(ui);
    let (rect, mut resp) =
        ui.allocate_exact_size(vec2(d + gap + galley.size().x, h), Sense::click());
    if resp.clicked() {
        *current = value;
        resp.mark_changed();
    }
    let hovered = resp.hovered();
    {
        let painter = ui.painter();
        let center = pos2(rect.left() + d / 2.0, rect.center().y);
        painter.circle_filled(center, d / 2.0, if hovered { ELEMENT_HOVER } else { ELEMENT });
        if selected {
            painter.circle_filled(center, d * 0.30, BUTTON_FG);
        }
        let lpos = pos2(rect.left() + d + gap, rect.center().y - galley.size().y / 2.0);
        painter.galley(lpos.round(), galley, TEXT);
    }
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Shared height for buttons and toggles so they line up — the game's 32px
/// element height, never shorter than a button's own label needs.
fn control_height(ui: &Ui) -> f32 {
    let text_h = ui.text_style_height(&TextStyle::Button);
    (text_h + 2.0 * ui.spacing().button_padding.y).max(ui.spacing().interact_size.y)
}

/// One uppercase, bold, pill-shaped button. Fully hand-painted: it recolors
/// across normal/hover/pressed but keeps a fixed rect (no hover growth), and its
/// label sinks by [`SINK`] while held.
fn button(
    ui: &mut Ui,
    label: impl Into<String>,
    normal: Color32,
    hover: Color32,
    pressed: Color32,
    fg: Color32,
) -> Response {
    let font = TextStyle::Button.resolve(ui.style());
    let galley = ui
        .painter()
        .layout_no_wrap(label.into().to_uppercase(), font, fg);

    let h = control_height(ui);
    let desired = vec2(galley.size().x + PAD_L + PAD_R, h);
    let (rect, resp) = ui.allocate_exact_size(desired, Sense::click());

    let down = resp.is_pointer_button_down_on();
    let bg = if down {
        pressed
    } else if resp.hovered() {
        hover
    } else {
        normal
    };

    let painter = ui.painter();
    // Pill: radius = half the height, like the game's 16px-on-32px buttons.
    painter.rect_filled(rect, CornerRadius::same((h * 0.5) as u8), bg);
    // Left-align with a smaller left pad (a pill's leading icon otherwise reads
    // pushed right); vertical-center the label; sink on press.
    let mut text_pos = pos2(rect.left() + PAD_L, rect.center().y - galley.size().y / 2.0);
    if down {
        text_pos.y += SINK;
    }
    // Snap to whole pixels so the text stays crisp.
    painter.galley(text_pos.round(), galley, fg);

    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

// ---------------------------------------------------------------------------
// Toggle — a sliding on/off switch (omegga's `.toggle`)
// ---------------------------------------------------------------------------

/// An on/off toggle: a blue-gray track with a square knob that slides from the
/// left (off, red, ✖) to the right (on, green, ✔), the symbol sinking while
/// pressed. Drop-in for `ui.checkbox`: returns a [`Response`] whose `changed()`
/// fires on toggle, so `.on_hover_text(..)` and `.changed()` both work.
pub fn toggle(ui: &mut Ui, on: &mut bool, label: impl Into<String>) -> Response {
    let label = label.into();
    let h = control_height(ui);
    let knob = h;
    let slide = 8.0; // omegga: track width = element height + 8
    let track_w = knob + slide;
    let gap = 8.0;

    // Measure the label so the WHOLE control (track + label) is one clickable
    // rect — clicking the label toggles the switch too.
    let label_galley = (!label.is_empty()).then(|| {
        let font = TextStyle::Body.resolve(ui.style());
        ui.painter().layout_no_wrap(label, font, TEXT)
    });
    let total_w = track_w + label_galley.as_ref().map_or(0.0, |g| gap + g.size().x);

    let (rect, mut resp) = ui.allocate_exact_size(vec2(total_w, h), Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }

    let t = ui.ctx().animate_bool(resp.id, *on);
    let down = resp.is_pointer_button_down_on();
    let hovered = resp.hovered();
    {
        let painter = ui.painter();
        let round = CornerRadius::same(RADIUS);
        let track = Rect::from_min_size(rect.min, vec2(track_w, h));
        painter.rect_filled(track, round, ELEMENT);

        // Knob: slides left→right, color lerps red→green (darker pressed,
        // lighter hovered), matching the button states.
        let off = pick(ERROR, ERROR_HOVER, ERROR_PRESSED, hovered, down);
        let onc = pick(MAIN, MAIN_HOVER, MAIN_PRESSED, hovered, down);
        let knob_rect = Rect::from_min_size(pos2(track.left() + t * slide, track.top()), vec2(knob, h));
        painter.rect_filled(knob_rect, round, lerp_color(off, onc, t));

        // Cross-fade ✖→✔, nudged down to sit centered, sinking while held.
        let mut c = knob_rect.center();
        c.y += SYMBOL_NUDGE;
        if down {
            c.y += SINK;
        }
        let c = c.round();
        let font = FontId::new(h * 0.5, icon_family());
        painter.text(c, Align2::CENTER_CENTER, icons::XMARK, font.clone(), Color32::WHITE.gamma_multiply(1.0 - t));
        painter.text(c, Align2::CENTER_CENTER, icons::CHECK, font, Color32::WHITE.gamma_multiply(t));

        if let Some(g) = label_galley {
            let pos = pos2(track.right() + gap, rect.center().y - g.size().y / 2.0);
            painter.galley(pos.round(), g, TEXT);
        }
    }
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// A real checkbox — a square box that shows a check when set — for
/// multi-select lists (distinct from the on/off [`toggle`] switch). Compact so
/// it fits table rows.
pub fn checkbox(ui: &mut Ui, checked: &mut bool, label: impl Into<String>) -> Response {
    let label = label.into();
    let box_sz = 22.0;
    let gap = 6.0;
    let galley = (!label.is_empty()).then(|| {
        ui.painter()
            .layout_no_wrap(label, TextStyle::Body.resolve(ui.style()), TEXT)
    });
    let h = control_height(ui);
    let w = box_sz + galley.as_ref().map_or(0.0, |g| gap + g.size().x);
    let (rect, mut resp) = ui.allocate_exact_size(vec2(w, h), Sense::click());
    if resp.clicked() {
        *checked = !*checked;
        resp.mark_changed();
    }
    let hovered = resp.hovered();
    {
        let painter = ui.painter();
        let bx = Rect::from_min_size(
            pos2(rect.left(), rect.center().y - box_sz / 2.0),
            Vec2::splat(box_sz),
        );
        let bg = if *checked {
            MAIN
        } else if hovered {
            ELEMENT_HOVER
        } else {
            ELEMENT
        };
        painter.rect_filled(bx, CornerRadius::same(5), bg);
        if *checked {
            painter.text(
                bx.center().round(),
                Align2::CENTER_CENTER,
                icons::CHECK,
                FontId::new(box_sz * 0.6, icon_family()),
                BUTTON_FG,
            );
        }
        if let Some(g) = galley {
            let pos = pos2(bx.right() + gap, rect.center().y - g.size().y / 2.0);
            painter.galley(pos.round(), g, TEXT);
        }
    }
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Pick the normal/hover/pressed colour for a state.
fn pick(normal: Color32, hover: Color32, pressed: Color32, hovered: bool, down: bool) -> Color32 {
    if down {
        pressed
    } else if hovered {
        hover
    } else {
        normal
    }
}

/// Linear-space colour lerp (`a` at t=0, `b` at t=1).
fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    egui::lerp(egui::Rgba::from(a)..=egui::Rgba::from(b), t.clamp(0.0, 1.0)).into()
}

// ---------------------------------------------------------------------------
// Outlined title text
// ---------------------------------------------------------------------------

/// A heading with a thick, rounded dark-blue outline (the game's chunky title
/// look). Any Font Awesome glyphs embedded in `text` are outlined too, since
/// they lay out in the same run.
///
/// Reserved for titles drawn over busy/blurred image backdrops (gameplay-style
/// menus), where the outline earns its keep. Over solid panel/card colors — the
/// sandbox and this tool's GUI — use a plain [`Ui::heading`] instead; the
/// outline only muddies text there. Kept here for that future use.
pub fn title(ui: &mut Ui, text: impl Into<String>) -> Response {
    outlined(ui, text.into(), TextStyle::Heading.resolve(ui.style()))
}

/// Paint `text` in [`TITLE_FG`] over two rings of [`OUTLINE`] copies — a thick,
/// rounded dark-blue outline. Positions snap to whole pixels for crispness.
fn outlined(ui: &mut Ui, text: String, font: FontId) -> Response {
    let measure = ui.painter().layout_no_wrap(text.clone(), font.clone(), TITLE_FG);
    let size = measure.size() + Vec2::splat(2.0 * OUTLINE_W);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::hover());
    let base = (rect.min + Vec2::splat(OUTLINE_W)).round();

    let painter = ui.painter();
    const N: usize = 12;
    for &r in &[OUTLINE_W, OUTLINE_W * 0.5] {
        for i in 0..N {
            let a = i as f32 / N as f32 * std::f32::consts::TAU;
            let off = vec2(a.cos(), a.sin()) * r;
            painter.text((base + off).round(), Align2::LEFT_TOP, &text, font.clone(), OUTLINE);
        }
    }
    painter.text(base, Align2::LEFT_TOP, &text, font, TITLE_FG);
    resp
}

// ---------------------------------------------------------------------------
// Section — a titled panel that replaces separator lines with a header band
// over a lighter content body.
// ---------------------------------------------------------------------------

/// A titled section: a dark [`SURFACE_HEADER`] band with a plain bold title
/// (no outline — that's reserved for top-level page titles) over a lighter
/// [`SURFACE_PANEL`] body holding `add`'s content. Grouping the UI into these —
/// instead of `ui.separator()` lines — is what gives the strong header↔content
/// contrast. Returns whatever `add` returns.
pub fn section<R>(ui: &mut Ui, title_text: &str, add: impl FnOnce(&mut Ui) -> R) -> R {
    let width = ui.available_width();
    let pad = CELL_PAD as i8;
    let top = CornerRadius { nw: RADIUS, ne: RADIUS, sw: 0, se: 0 };

    Frame::new()
        .fill(SURFACE_PANEL)
        .corner_radius(CornerRadius::same(RADIUS))
        .show(ui, |ui| {
            ui.set_min_width(width);
            // Header and body frames flush (no inter-frame gap) so the body's
            // top padding equals its side padding.
            ui.spacing_mut().item_spacing.y = 0.0;

            Frame::new()
                .fill(SURFACE_HEADER)
                .corner_radius(top)
                .inner_margin(Margin { left: pad, right: pad, top: 5, bottom: 5 })
                .show(ui, |ui| {
                    ui.set_min_width(width - 2.0 * CELL_PAD);
                    ui.add(
                        egui::Label::new(egui::RichText::new(title_text).heading())
                            .selectable(false),
                    );
                });

            Frame::new()
                .inner_margin(Margin::same(pad))
                .show(ui, |ui| {
                    ui.set_min_width(width - 2.0 * CELL_PAD);
                    // Restore a normal vertical gap inside the body (the outer
                    // frame zeroed it to keep the header flush).
                    ui.spacing_mut().item_spacing.y = 8.0;
                    add(ui)
                })
                .inner
        })
        .inner
}

/// Pull `add`'s content out to the section panel edges, canceling the body's
/// [`CELL_PAD`] horizontal padding — for full-width striped tables. Pad the row
/// content back with [`cell_label`] (and the table's natural right-column space).
pub fn full_bleed<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    let pad = CELL_PAD as i8;
    Frame::new()
        .outer_margin(Margin { left: -pad, right: -pad, top: 0, bottom: 0 })
        .show(ui, |ui| add(ui))
        .inner
}

/// A themed dropdown: egui's `ComboBox` with a Font Awesome caret and a pointer
/// cursor on the button. Add the options inside `items` with [`combo_item`] so
/// they get a pointer cursor too. Reused by every pane instead of each rolling
/// its own `ComboBox`.
pub fn combo<R>(
    ui: &mut Ui,
    id: impl std::hash::Hash,
    selected: impl Into<egui::WidgetText>,
    width: f32,
    items: impl FnOnce(&mut Ui) -> R,
) -> Response {
    egui::ComboBox::from_id_salt(id)
        .selected_text(selected)
        .width(width)
        .truncate() // ellipsize a long selected value instead of wrapping it
        .icon(|ui, rect, visuals, _open| {
            ui.painter().text(
                rect.center(),
                Align2::CENTER_CENTER,
                icons::CHEVRON_DOWN,
                FontId::new(rect.height() * 0.55, icon_family()),
                visuals.fg_stroke.color,
            );
        })
        .show_ui(ui, items)
        .response
        .on_hover_cursor(egui::CursorIcon::PointingHand)
}

// ---------------------------------------------------------------------------
// Settings table — full-width, SQUARE striped rows (egui `Grid` can do neither:
// it hardcodes a 2px stripe radius at the content width). A top-aligned padded
// label column + a control column. Reused by every generator's settings.
// ---------------------------------------------------------------------------

/// Per-table state passed to the body closure; add rows with [`SettingsTable::row`].
pub struct SettingsTable {
    row: usize,
    label_w: f32,
    width: f32,
}

impl SettingsTable {
    /// One row: a top-aligned, left-padded `label` and a `control`.
    pub fn row(&mut self, ui: &mut Ui, label: &str, control: impl FnOnce(&mut Ui)) {
        self.row_hover(ui, label, None, control);
    }

    /// [`row`](Self::row) with a hover tooltip on the label.
    pub fn row_hover(
        &mut self,
        ui: &mut Ui,
        label: &str,
        hover: Option<&str>,
        control: impl FnOnce(&mut Ui),
    ) {
        let left = ui.max_rect().left();
        let full = self.width;
        let striped = self.row % 2 == 1;
        let min_h = control_height(ui);
        // Reserve the stripe behind the row; fill it once the row height is known.
        let bg = ui.painter().add(egui::Shape::Noop);

        let inner = ui.horizontal(|ui| {
            ui.set_min_height(min_h);
            ui.add_space(CELL_PAD);
            // Fixed-width label column: paint the label vertically centered in a
            // min-height box. The center-aligned row keeps the label and the
            // control (widget OR single-line text like the Note row) on the same
            // vertical center.
            let (lrect, lresp) =
                ui.allocate_exact_size(egui::vec2(self.label_w, min_h), egui::Sense::hover());
            let galley = ui.painter().layout_no_wrap(
                label.to_owned(),
                egui::TextStyle::Body.resolve(ui.style()),
                TEXT,
            );
            ui.painter().galley(
                egui::pos2(lrect.left(), lrect.center().y - galley.size().y / 2.0).round(),
                galley,
                TEXT,
            );
            if let Some(h) = hover {
                lresp.on_hover_text(h);
            }
            control(ui);
        });

        if striped {
            let rect = inner.response.rect;
            let stripe = egui::Rect::from_min_max(
                egui::pos2(left, rect.top() - 2.0),
                egui::pos2(left + full, rect.bottom() + 2.0),
            );
            ui.painter().set(
                bg,
                egui::epaint::RectShape::filled(stripe, CornerRadius::same(0), SURFACE_STRIPE),
            );
        }
        self.row += 1;
    }
}

/// Run a full-width, square-striped settings table. `add` adds rows via
/// `t.row(ui, "Label", |ui| { <control> })`. Bleeds to the card edges; the
/// labels stay padded.
pub fn settings_table(ui: &mut Ui, add: impl FnOnce(&mut Ui, &mut SettingsTable)) {
    full_bleed(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 4.0;
        let width = ui.available_width();
        let mut t = SettingsTable { row: 0, label_w: 130.0, width };
        add(ui, &mut t);
    });
}

/// A dropdown option with a pointer cursor (egui's `selectable_value` gives
/// none). Use inside [`combo`]'s `items` closure.
pub fn combo_item<V: PartialEq>(
    ui: &mut Ui,
    current: &mut V,
    value: V,
    label: impl Into<egui::WidgetText>,
) -> Response {
    ui.selectable_value(current, value, label)
        .on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// A grid/table label cell with [`CELL_PAD`] of left padding, so a [`full_bleed`]
/// row's stripe reaches the panel edge while its text stays inset. Returns the
/// label's [`Response`] so `.on_hover_text(..)` still chains.
pub fn cell_label(ui: &mut Ui, text: impl Into<String>) -> Response {
    // Top-aligned (Align::Min) so in a tall row (wrapped controls) the label
    // sits at the top rather than vertically centered.
    ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
        ui.add_space(CELL_PAD);
        ui.label(text.into())
    })
    .inner
}

/// A single-line text field sized to the same [`control_height`] as the numeric
/// `DragValue`/`Slider` and dropdown controls (egui's `TextEdit` otherwise
/// ignores `interact_size`). The vertical margin both sets the height and keeps
/// the text centered.
pub fn text_field(ui: &mut Ui, text: &mut String, hint: &str) -> Response {
    let row = ui.text_style_height(&TextStyle::Body);
    let vpad = (((control_height(ui) - row) * 0.5).max(0.0)) as i8;
    ui.add(
        egui::TextEdit::singleline(text)
            .hint_text(hint)
            .margin(Margin::symmetric(8, vpad)),
    )
}
