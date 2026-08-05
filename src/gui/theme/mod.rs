//! Brickadia-themed egui styling: the game's navy/green palette, its
//! typeface (Glacial Indifference), Font Awesome icon support and a set of
//! semantic buttons that mirror the in-game / web-UI look.
//!
//! This module is deliberately self-contained and depends only on `egui` (the
//! [`sandbox`] demo aside, which is pure `egui` too). It borrows no types from
//! the rest of this crate, so the whole `theme/` directory can later be lifted
//! into a standalone `brickadia-egui` crate with no untangling: move the folder,
//! move `assets/fonts/`, fix the `include_bytes!` paths, done.
//!
//! The palette is transcribed from Brickadia's web UI theme
//! (`omegga/frontend/src/css/theme.scss`); each constant keeps its source name.
//!
//! # Usage
//! (Illustrative, using the paths as seen from inside the host crate; marked
//! `ignore` so the crate-relative names don't have to resolve as a doctest.)
//! ```ignore
//! theme::install(&ctx); // once, at startup (after image loaders)
//! ```
//! then in UI code:
//! ```ignore
//! use crate::gui::theme::{widgets, icons};
//! if widgets::primary(ui, format!("{}  Generate", icons::DOWNLOAD)).clicked() {
//!     // ...
//! }
//! ```
use egui::{
    Color32, Context, FontData, FontDefinitions, FontFamily, FontId, Stroke, Style, TextStyle,
    Visuals,
};
use std::sync::Arc;

pub mod icons;
pub mod sandbox;
pub mod widgets;

pub use sandbox::Sandbox;

// ---------------------------------------------------------------------------
// Palette — from omegga/frontend/src/css/theme.scss. Names match the SCSS vars
// (`$br-…`) so the two can be diffed by eye when the game theme changes.
// ---------------------------------------------------------------------------

/// Backgrounds & panels
pub const BG_PRIMARY: Color32 = Color32::from_rgb(0x1b, 0x2f, 0x4a); // $br-bg-primary
pub const BG_PRIMARY_ALT: Color32 = Color32::from_rgb(0x18, 0x2b, 0x44); // $br-bg-primary-alt
pub const BG_SECONDARY: Color32 = Color32::from_rgb(0x13, 0x1d, 0x30); // $br-bg-secondary
pub const BG_SECONDARY_ALT: Color32 = Color32::from_rgb(0x11, 0x1a, 0x2b); // $br-bg-secondary-alt
pub const BG_FOOTER: Color32 = Color32::from_rgb(0x09, 0x10, 0x21); // $br-bg-footer / header

/// Text & icon foregrounds
pub const TEXT: Color32 = Color32::from_rgb(0xcd, 0xd8, 0xe6); // body text on navy
pub const BORING_FG: Color32 = Color32::from_rgb(0xa7, 0xbb, 0xce); // $br-boring-button-fg
pub const BUTTON_FG: Color32 = Color32::from_rgb(0xff, 0xff, 0xff); // $br-button-fg

/// Main / primary action (green)
pub const MAIN: Color32 = Color32::from_rgb(0x5d, 0xa9, 0x3d); // $br-main-normal
pub const MAIN_HOVER: Color32 = Color32::from_rgb(0x8d, 0xc3, 0x77); // $br-main-hover
pub const MAIN_PRESSED: Color32 = Color32::from_rgb(0x41, 0x77, 0x2b); // $br-main-pressed

/// Info (blue)
pub const INFO: Color32 = Color32::from_rgb(0x00, 0x9b, 0xee); // $br-info-normal
pub const INFO_HOVER: Color32 = Color32::from_rgb(0x4c, 0xb9, 0xf3); // $br-info-hover
pub const INFO_PRESSED: Color32 = Color32::from_rgb(0x00, 0x7d, 0xa7); // $br-info-pressed

/// Error (red)
pub const ERROR: Color32 = Color32::from_rgb(0xe0, 0x2d, 0x2d); // $br-error-normal
pub const ERROR_HOVER: Color32 = Color32::from_rgb(0xe9, 0x6c, 0x6c); // $br-error-hover
pub const ERROR_PRESSED: Color32 = Color32::from_rgb(0x9d, 0x20, 0x20); // $br-error-pressed

/// Warning (orange)
pub const WARN: Color32 = Color32::from_rgb(0xff, 0xa1, 0x0b); // $br-warn-normal
pub const WARN_HOVER: Color32 = Color32::from_rgb(0xff, 0xbd, 0x54); // $br-warn-hover
pub const WARN_PRESSED: Color32 = Color32::from_rgb(0xb3, 0x71, 0x08); // $br-warn-pressed

/// Neutral button / element (blue-gray)
pub const ELEMENT: Color32 = Color32::from_rgb(0x45, 0x5a, 0x7e); // $br-element-normal
pub const ELEMENT_HOVER: Color32 = Color32::from_rgb(0x5e, 0x71, 0x90); // $br-element-fg/hover
pub const ELEMENT_PRESSED: Color32 = Color32::from_rgb(0x26, 0x3f, 0x61); // $br-element-pressed
pub const ELEMENT_HEADER_BG: Color32 = Color32::from_rgb(0x16, 0x28, 0x41); // $br-element-header-bg

/// Corner radius used across the theme (`$br-radius-sm`, 8px).
pub const RADIUS: u8 = 8;

// --- Surfaces: a deliberate light→dark ladder so headers, panels, the page and
// table stripes each sit on a visibly different shade (no separator lines). ---

/// The page behind everything (central panel). Mid-dark.
pub const SURFACE_PAGE: Color32 = BG_SECONDARY;
/// A content panel / section body — lighter than the page so it stands out.
pub const SURFACE_PANEL: Color32 = BG_PRIMARY;
/// A section's header band — the darkest surface, for strong header↔content
/// contrast.
pub const SURFACE_HEADER: Color32 = BG_FOOTER;
/// Alternating ("faint") table/grid row fill — darker than a panel so striped
/// rows read clearly against [`SURFACE_PANEL`].
pub const SURFACE_STRIPE: Color32 = BG_SECONDARY_ALT;

/// Dark-blue (not black) outline painted around top-level title text & icons
/// for the chunky game-style look. Reserved for the topmost page titles.
pub const OUTLINE: Color32 = Color32::from_rgb(0x06, 0x14, 0x2b);
/// Title text fill (sits inside [`OUTLINE`]).
pub const TITLE_FG: Color32 = Color32::from_rgb(0xf2, 0xf6, 0xfc);

// ---------------------------------------------------------------------------
// Font families
// ---------------------------------------------------------------------------

/// `font_data` keys for the vendored faces.
const FD_REGULAR: &str = "brk-regular";
const FD_BOLD: &str = "brk-bold";
const FD_ICON: &str = "fa-solid-900";

/// Named [`FontFamily`] for bold text (headings & buttons): Glacial Bold, with
/// Font Awesome and the default proportional chain as fallbacks.
pub const FAMILY_BOLD: &str = "brk-bold";
/// Named [`FontFamily`] holding only the Font Awesome solid face — for glyphs
/// that want an explicit size independent of the surrounding text style.
pub const FAMILY_ICON: &str = "fa";

/// The bold family, for `RichText::family(..)`.
pub fn bold_family() -> FontFamily {
    FontFamily::Name(FAMILY_BOLD.into())
}
/// The Font Awesome icon family, for `RichText::family(..)`.
pub fn icon_family() -> FontFamily {
    FontFamily::Name(FAMILY_ICON.into())
}

const GLACIAL_REGULAR: &[u8] = include_bytes!("../../../assets/fonts/GlacialIndifferenceRegular.ttf");
const GLACIAL_BOLD: &[u8] = include_bytes!("../../../assets/fonts/GlacialIndifferenceBold.ttf");

// ---------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------

/// Apply the Brickadia theme to `ctx`: fonts, visuals and spacing. Call once at
/// startup. Idempotent, so re-calling (e.g. on a settings reset) is harmless.
pub fn install(ctx: &Context) {
    ctx.set_fonts(fonts());

    let mut style: Style = (*ctx.style()).clone();
    style.visuals = visuals();
    apply_text_styles(&mut style);
    apply_spacing(&mut style);
    ctx.set_style(style);
}

/// Build the font set: Glacial Indifference as the proportional face, Font
/// Awesome appended as a fallback (so bare FA codepoints render inline in any
/// label), plus the two named families used by [`bold_family`]/[`icon_family`].
/// egui's built-in fonts are kept as trailing fallbacks for emoji & monospace.
fn fonts() -> FontDefinitions {
    let mut f = FontDefinitions::default();

    f.font_data
        .insert(FD_REGULAR.into(), Arc::new(FontData::from_static(GLACIAL_REGULAR)));
    f.font_data
        .insert(FD_BOLD.into(), Arc::new(FontData::from_static(GLACIAL_BOLD)));
    // Font Awesome glyphs read small next to the bold text, so scale them up a
    // little (and nudge them up to stay vertically centered). Applies to every
    // family the face is a member of — inline icons, toggle symbols and all.
    f.font_data.insert(
        FD_ICON.into(),
        Arc::new(FontData::from_static(icons::FA_SOLID).tweak(egui::FontTweak {
            scale: 1.2,
            y_offset_factor: -0.08,
            ..Default::default()
        })),
    );

    // Proportional: Glacial first, then FA (fallback for inline icon glyphs),
    // then whatever egui already had (emoji etc.).
    let prop = f.families.entry(FontFamily::Proportional).or_default();
    prop.insert(0, FD_REGULAR.into());
    prop.insert(1, FD_ICON.into());

    // Monospace keeps its default face but also gains the FA fallback.
    f.families
        .entry(FontFamily::Monospace)
        .or_default()
        .push(FD_ICON.into());

    // Bold family = Glacial Bold + FA + the (already-augmented) proportional
    // chain as fallback, so headings/buttons get true bold and still render
    // icons and emoji.
    let mut bold_chain = vec![FD_BOLD.to_owned(), FD_ICON.to_owned()];
    bold_chain.extend(f.families.get(&FontFamily::Proportional).cloned().unwrap_or_default());
    f.families.insert(FontFamily::Name(FAMILY_BOLD.into()), bold_chain);

    // Icon-only family, for explicitly-sized glyphs.
    f.families
        .insert(FontFamily::Name(FAMILY_ICON.into()), vec![FD_ICON.to_owned()]);

    f
}

/// Almost everything is bold (the game's chunky look): body, buttons and
/// headings all use the bold family. Only fine print (`Small`) and code
/// (`Monospace`) stay in the regular/mono faces.
fn apply_text_styles(style: &mut Style) {
    style.text_styles = [
        (TextStyle::Small, FontId::new(11.0, FontFamily::Proportional)),
        (TextStyle::Body, FontId::new(15.0, bold_family())),
        (TextStyle::Monospace, FontId::new(13.0, FontFamily::Monospace)),
        (TextStyle::Button, FontId::new(15.0, bold_family())),
        (TextStyle::Heading, FontId::new(22.0, bold_family())),
    ]
    .into();
}

/// Match the game's element metrics: 32px-tall controls, 16px horizontal button
/// padding, 8px gaps (omegga `$br-element-height` / button `padding: 0 16px`).
fn apply_spacing(style: &mut Style) {
    style.spacing.button_padding = egui::vec2(16.0, 6.0);
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.interact_size.y = 32.0;
}

/// The dark navy visuals: the surface ladder, neutral blue-gray widgets that
/// recolor (but never grow) on hover, info-blue selection & links, rounded
/// corners and clearly-contrasting table stripes.
fn visuals() -> Visuals {
    let mut v = Visuals::dark();
    let radius = egui::CornerRadius::same(RADIUS);

    v.panel_fill = SURFACE_PAGE;
    v.window_fill = BG_SECONDARY_ALT; // combo popups / menus
    v.window_stroke = Stroke::new(1.0, ELEMENT_HEADER_BG);
    // Text-edit fill = the neutral widget fill, so a `TextEdit` matches the
    // blue-gray of the numeric `DragValue`/`Slider` inputs beside it.
    v.extreme_bg_color = ELEMENT;
    v.faint_bg_color = SURFACE_STRIPE; // alternating table/grid rows
    v.hyperlink_color = INFO;
    v.selection.bg_fill = Color32::from_rgba_unmultiplied(0x00, 0x9b, 0xee, 90);
    // `selection.stroke.color` is also what a `ProgressBar` paints its overlay
    // text in — keep it near-white for contrast on the blue fill.
    v.selection.stroke = Stroke::new(1.0, TITLE_FG);

    // Non-interactive: panels, plain labels.
    let w = &mut v.widgets;
    w.noninteractive.bg_fill = SURFACE_PANEL;
    w.noninteractive.weak_bg_fill = SURFACE_PANEL;
    w.noninteractive.bg_stroke = Stroke::new(1.0, ELEMENT_HEADER_BG);
    w.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
    w.noninteractive.corner_radius = radius;

    // Inactive: default (unhovered) interactive widgets — neutral blue-gray.
    // fg_stroke is the text color (glyphs are filled, so its *width* is
    // irrelevant to text) AND the slider handle's outline stroke — keep the
    // colors, zero the widths so the handle has no white ring.
    w.inactive.bg_fill = ELEMENT;
    w.inactive.weak_bg_fill = ELEMENT;
    w.inactive.bg_stroke = Stroke::NONE;
    w.inactive.fg_stroke = Stroke::new(0.0, BUTTON_FG);
    w.inactive.corner_radius = radius;

    w.hovered.bg_fill = ELEMENT_HOVER;
    w.hovered.weak_bg_fill = ELEMENT_HOVER;
    w.hovered.bg_stroke = Stroke::new(1.0, ELEMENT_HOVER);
    w.hovered.fg_stroke = Stroke::new(0.0, BUTTON_FG);
    w.hovered.corner_radius = radius;

    w.active.bg_fill = ELEMENT_PRESSED;
    w.active.weak_bg_fill = ELEMENT_PRESSED;
    w.active.bg_stroke = Stroke::new(1.0, INFO);
    w.active.fg_stroke = Stroke::new(0.0, BUTTON_FG);
    w.active.corner_radius = radius;

    w.open.bg_fill = ELEMENT_PRESSED;
    w.open.weak_bg_fill = ELEMENT_PRESSED;
    w.open.bg_stroke = Stroke::new(1.0, ELEMENT_HEADER_BG);
    w.open.fg_stroke = Stroke::new(1.0, TEXT);
    w.open.corner_radius = radius;

    // Widgets recolor on hover but never grow — kill egui's default expansion
    // on every state so buttons stay put under the cursor.
    for s in [
        &mut w.noninteractive,
        &mut w.inactive,
        &mut w.hovered,
        &mut w.active,
        &mut w.open,
    ] {
        s.expansion = 0.0;
    }

    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Install the theme and run one full headless frame that lays out a
    /// heading, body text, every icon and both a bold and an icon-bearing
    /// button. Layout + tessellation forces the bundled `.ttf`s through
    /// ab_glyph, so a corrupt or wrong-format font panics here rather than at
    /// the user's first launch. (No window/GL is needed — rasterization is CPU.)
    #[test]
    fn install_and_render_one_frame() {
        let ctx = Context::default();
        install(&ctx);
        let out = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading(format!("{}  Heading", icons::MUSIC));
                ui.label("body text renders in Glacial Indifference");
                for (_, glyph) in icons::ALL {
                    ui.label(icons::icon(glyph).size(18.0));
                }
                widgets::primary(ui, format!("{}  Generate", icons::DOWNLOAD));
                widgets::neutral(ui, "back");
            });
        });
        // A frame that laid out text produces at least one font-atlas texture.
        assert!(
            !out.textures_delta.set.is_empty(),
            "expected a font texture to be rasterized"
        );
    }
}
