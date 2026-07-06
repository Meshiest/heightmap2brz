use brdb::{
    AsBrdbValue, BrdbSchemaError, Brick, BrickSize, BrickType, Collision, Color, Direction,
    Position, PrefabJson, SavedBrickColor, Vector3f, WirePort, World,
    assets::{LiteralComponent, bricks::PB_DEFAULT_MICRO_BRICK},
    schema::{BrdbInterned, BrdbSchema, BrdbValue, WireVariant},
};
use image::RgbaImage;

/// Maximum characters (Unicode chars, not bytes) a TextDisplay component accepts.
pub const MAX_COMPONENT_CHARS: usize = 10_000;

/// Calibrated font presets for the text renderer. Each preset carries the
/// glyph scheme and the component geometry (LineHeight/LineOffset/Kerning/
/// Offset) tuned in-game for that font; values scale with pixel size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FontPreset {
    /// `██` / two spaces per pixel is square; monospace, so space-based
    /// transparency lines up.
    MonaspaceArgon,
    /// `██` / two spaces per pixel is square; monospace. Calibrated from the
    /// user's reference clipboards (2026-07-04).
    IosevkaTerm,
    /// Single `█` per pixel is square, halving the char budget — but the font
    /// is proportional, so space-based transparency does NOT line up.
    Orbitron,
}

impl FontPreset {
    pub const ALL: [FontPreset; 3] = [
        FontPreset::MonaspaceArgon,
        FontPreset::IosevkaTerm,
        FontPreset::Orbitron,
    ];

    /// Display name for UIs.
    pub fn name(&self) -> &'static str {
        match self {
            FontPreset::MonaspaceArgon => "Monaspace Argon",
            FontPreset::IosevkaTerm => "Iosevka Term",
            FontPreset::Orbitron => "Orbitron",
        }
    }

    /// BrickFontDescriptor asset name.
    pub fn font_asset(&self) -> &'static str {
        match self {
            FontPreset::MonaspaceArgon => "MonaspaceArgon",
            FontPreset::IosevkaTerm => "IosevkaTerm",
            FontPreset::Orbitron => "Orbitron",
        }
    }

    /// Calibrated options for this font at the given pixel size (world units
    /// per pixel row). Geometry values may need manual re-calibration; UIs
    /// expose them for tweaking after the preset seeds them.
    pub fn options(&self, pixel_size: f32) -> TextOptions {
        let base = TextOptions {
            font: self.font_asset(),
            fill_char: '█',
            empty_char: ' ',
            char_repeat: 2,
            alpha_threshold: 128,
            line_world_height: pixel_size,
            line_height: 0.61 * pixel_size,
            line_offset: 0.0,
            kerning: 0.0,
            offset_x: 0.0,
            offset_y: -0.2 * pixel_size,
            // the calibrated reference clipboards all carry Offset Z = 0:
            // front cubes get no forward offset at all
            offset_z: 0.0,
            pitch_x: 1.0,
            pitch_y: 1.0,
            mode: PixelMode::Color,
            luma_threshold: 128,
            invert: false,
            tile_override: None,
            // material defaults from the calibrated reference clipboards
            // (they also match the game's own field defaults)
            material: TextMaterial::Unlit,
            material_intensity: 10,
            scuff: 0.0,
            graffiti_depth_limit: 5.0,
            graffiti_angle_limit: 45.0,
            graffiti_priority: 0,
            shading: TextShading::None,
            shading_width: 2.0,
            invert_shading: false,
        };
        match self {
            // at LineHeight 0.61 Monaspace renders 30/32 of the nominal
            // pixel size (measured in-game via uniform one-cube tile gaps):
            // the brick grid spacing shrinks to match the rendering
            FontPreset::MonaspaceArgon => TextOptions {
                pitch_x: 30.0 / 32.0,
                pitch_y: 30.0 / 32.0,
                ..base
            },
            FontPreset::IosevkaTerm => base,
            // from the user's "Text Pixel" prefab (2 units/px: LineHeight 1.6,
            // LineOffset -8.5) and F-shape clipboard (0.5 units/px: LineHeight
            // 0.4, LineOffset -8, Kerning -0.1, Offset.Y -0.05)
            FontPreset::Orbitron => TextOptions {
                char_repeat: 1,
                line_height: 0.8 * pixel_size,
                line_offset: -8.0,
                kerning: -0.2 * pixel_size,
                offset_y: -0.1 * pixel_size,
                ..base
            },
        }
    }
}

/// TextDisplay material (`EBRTextMaterial` — declaration order matches the
/// game's enum values). Unlit and Graffiti have no shading settings; only
/// Graffiti projects onto nearby bricks (depth/angle limit, priority).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextMaterial {
    Unlit,
    Graffiti,
    Plastic,
    Metallic,
    Glow,
    TranslucentPlastic,
    Glass,
}

impl TextMaterial {
    pub const ALL: [TextMaterial; 7] = [
        TextMaterial::Unlit,
        TextMaterial::Graffiti,
        TextMaterial::Plastic,
        TextMaterial::Metallic,
        TextMaterial::Glow,
        TextMaterial::TranslucentPlastic,
        TextMaterial::Glass,
    ];

    /// Display name for UIs (matches the in-game dropdown).
    pub fn name(&self) -> &'static str {
        match self {
            TextMaterial::Unlit => "Unlit",
            TextMaterial::Graffiti => "Graffiti",
            TextMaterial::Plastic => "Plastic",
            TextMaterial::Metallic => "Metallic",
            TextMaterial::Glow => "Glow",
            TextMaterial::TranslucentPlastic => "Translucent Plastic",
            TextMaterial::Glass => "Glass",
        }
    }

    /// `EBRTextMaterial` enum value.
    pub fn byte(&self) -> u8 {
        *self as u8
    }

    /// The game enables the Material Intensity slider for every material
    /// except Unlit and Plastic.
    pub fn has_intensity(&self) -> bool {
        !matches!(self, TextMaterial::Unlit | TextMaterial::Plastic)
    }

    /// Only Graffiti projects: depth limit, angle limit, and priority.
    pub fn is_graffiti(&self) -> bool {
        matches!(self, TextMaterial::Graffiti)
    }

    /// Shading settings are not available for Unlit or Graffiti.
    pub fn has_shading(&self) -> bool {
        !matches!(self, TextMaterial::Unlit | TextMaterial::Graffiti)
    }
}

/// TextDisplay edge shading style (`EBRTextShading` — declaration order
/// matches the game's enum values). Unavailable for Unlit/Graffiti.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextShading {
    None,
    Chamfer,
    Fillet,
    DoubleChamfer,
    DoubleFillet,
    Rounded,
}

impl TextShading {
    pub const ALL: [TextShading; 6] = [
        TextShading::None,
        TextShading::Chamfer,
        TextShading::Fillet,
        TextShading::DoubleChamfer,
        TextShading::DoubleFillet,
        TextShading::Rounded,
    ];

    /// Display name for UIs (matches the in-game dropdown).
    pub fn name(&self) -> &'static str {
        match self {
            TextShading::None => "None",
            TextShading::Chamfer => "Chamfer",
            TextShading::Fillet => "Fillet",
            TextShading::DoubleChamfer => "Double Chamfer",
            TextShading::DoubleFillet => "Double Fillet",
            TextShading::Rounded => "Rounded",
        }
    }

    /// `EBRTextShading` enum value.
    pub fn byte(&self) -> u8 {
        *self as u8
    }
}

#[derive(Clone, Debug)]
pub struct TextOptions {
    /// BrickFontDescriptor asset name.
    pub font: &'static str,
    pub fill_char: char,
    pub empty_char: char,
    pub char_repeat: usize,
    pub alpha_threshold: u8,
    /// World units per pixel row (pixel size).
    pub line_world_height: f32,
    /// Component LineHeight (font size).
    pub line_height: f32,
    /// Component LineOffset.
    pub line_offset: f32,
    /// Component Kerning.
    pub kerning: f32,
    /// Component Offset.X — glyph-fit nudge.
    pub offset_x: f32,
    /// Component Offset.Y — glyph-fit nudge.
    pub offset_y: f32,
    /// Component Offset.Z — out-of-plane: pushes the text off the anchor
    /// wall's face so the cubes hide behind the image.
    pub offset_z: f32,
    /// Horizontal rendered size / nominal pixel size: tile spacing along the
    /// image's X is `tile_px × line_world_height × pitch_x` (gaps between
    /// tile columns ⇒ lower, overlaps ⇒ raise).
    pub pitch_x: f32,
    /// Vertical counterpart (gap Y): tile row spacing scale. Mono modes'
    /// glyph cells are not square, so the two differ there.
    pub pitch_y: f32,
    /// How pixels become glyphs (full color, or monochrome braille/blocks).
    pub mode: PixelMode,
    /// Monochrome modes: pixels at least this bright are drawn.
    pub luma_threshold: u8,
    /// Monochrome modes: draw dark pixels instead of bright ones.
    pub invert: bool,
    /// Override the mode's tile edge in pixels (None = mode default). The
    /// calibration grid uses this to force seams in mono modes.
    pub tile_override: Option<u32>,
    /// Component material.
    pub material: TextMaterial,
    /// Material Intensity — glow brightness / metal, glass, translucency
    /// strength. Inert for Unlit and Plastic.
    pub material_intensity: i32,
    /// Scuff width — worn-edge wear on the glyphs.
    pub scuff: f32,
    /// Graffiti only: projection depth limit in cm.
    pub graffiti_depth_limit: f32,
    /// Graffiti only: projection angle limit in degrees.
    pub graffiti_angle_limit: f32,
    /// Graffiti only: projection priority (the component's GraffitiLayer).
    pub graffiti_priority: i32,
    /// Edge shading style. Unavailable for Unlit/Graffiti.
    pub shading: TextShading,
    /// Shading width.
    pub shading_width: f32,
    /// Invert Shading (the component's bFlipShading).
    pub invert_shading: bool,
}

impl TextOptions {
    /// Tile edge in pixels for this configuration.
    pub fn tile_px(&self) -> u32 {
        self.tile_override.unwrap_or(self.mode.tile_px())
    }
}

/// Component geometry for the monochrome modes, measured in-game
/// (2026-07-04), returned as (font size, kerning, line offset, pitch_x,
/// pitch_y): braille wants font size 2.7 with kerning -4 and a constant
/// LineOffset of -8 (gap X from the font preset); blocks wants font size
/// 1.08, gap X 0.41, and matches the normal mode's zeroes. Both use gap Y
/// 0.8125.
pub fn mono_geometry(mode: PixelMode, pixel_size: f32) -> (f32, f32, f32, Option<f32>, f32) {
    match mode {
        PixelMode::Braille => (2.7 * pixel_size, -4.0 * pixel_size, -8.0, None, 0.8125),
        _ => (1.08 * pixel_size, 0.0, 0.0, Some(0.41), 0.8125),
    }
}

impl Default for TextOptions {
    fn default() -> Self {
        FontPreset::MonaspaceArgon.options(1.0)
    }
}

/// One TextDisplay component's worth of image rows. Bands after the first
/// begin with `start_row` newlines so the game's own line advance places
/// their rows at the exact image depth (no measured-advance error).
#[derive(Clone, Debug)]
pub struct TextBand {
    /// First image row rendered by this band (also its padding-line count).
    pub start_row: usize,
    /// Number of image rows in this band (excluding padding lines).
    pub rows: usize,
    pub text: String,
    /// Unicode char count of `text`, including padding newlines.
    pub chars: usize,
}

pub fn encode_bands(img: &RgbaImage, opts: &TextOptions) -> Result<Vec<TextBand>, String> {
    let (_, h) = img.dimensions();
    let mut bands: Vec<TextBand> = Vec::new();
    let mut cur = TextBand {
        start_row: 0,
        rows: 0,
        text: String::new(),
        chars: 0,
    };
    // Last emitted <color> tag; persists across pixels, gaps, and rows within a band.
    let mut last_color: Option<[u8; 3]> = None;

    for y in 0..h {
        let mut row_state = last_color;
        let (row_text, row_chars) = encode_row(img, y, opts, &mut row_state);
        let sep = if cur.rows == 0 { 0 } else { 1 };

        if cur.rows > 0 && cur.chars + sep + row_chars > MAX_COMPONENT_CHARS {
            // Close the band; the next one starts with y padding newlines
            // (the game's own line advance shifts its rows into place) and
            // fresh color state since a new component starts colorless.
            bands.push(std::mem::replace(
                &mut cur,
                TextBand {
                    start_row: y as usize,
                    rows: 0,
                    text: "\n".repeat(y as usize),
                    chars: y as usize,
                },
            ));
            let mut fresh = None;
            let (row_text, row_chars) = encode_row(img, y, opts, &mut fresh);
            if row_chars > MAX_COMPONENT_CHARS {
                return Err(row_too_wide(y, row_chars));
            }
            if cur.chars + row_chars > MAX_COMPONENT_CHARS {
                return Err(format!(
                    "row {y}: {} padding lines plus a {row_chars}-char row exceed the \
                     {MAX_COMPONENT_CHARS}-char TextDisplay limit; the image is too tall \
                     to band",
                    cur.chars
                ));
            }
            cur.text.push_str(&row_text);
            cur.chars += row_chars;
            cur.rows = 1;
            last_color = fresh;
            continue;
        }
        if row_chars > MAX_COMPONENT_CHARS {
            return Err(row_too_wide(y, row_chars));
        }
        if sep == 1 {
            cur.text.push('\n');
            cur.chars += 1;
        }
        cur.text.push_str(&row_text);
        cur.chars += row_chars;
        cur.rows += 1;
        last_color = row_state;
    }
    if cur.rows > 0 || bands.is_empty() {
        bands.push(cur);
    }
    Ok(bands)
}

/// Square pixel patch tiled across the image in both axes. Small enough
/// that even a worst-case patch (a color tag on every pixel) fits one or
/// two components, and each patch anchors to its own nearby cube so
/// component offsets stay tiny.
pub const TILE_PX: u32 = 32;

/// Braille sprite sheet: index bit `y*2 + x` set ⇒ dot at (x, y) in the
/// char's 2-wide × 4-tall pixel cell.
const BRAILLE_SPRITES: &str = "\
⠀⠁⠈⠉⠂⠃⠊⠋⠐⠑⠘⠙⠒⠓⠚⠛⠄⠅⠌⠍⠆⠇⠎⠏⠔⠕⠜⠝⠖⠗⠞⠟\
⠠⠡⠨⠩⠢⠣⠪⠫⠰⠱⠸⠹⠲⠳⠺⠻⠤⠥⠬⠭⠦⠧⠮⠯⠴⠵⠼⠽⠶⠷⠾⠿\
⡀⡁⡈⡉⡂⡃⡊⡋⡐⡑⡘⡙⡒⡓⡚⡛⡄⡅⡌⡍⡆⡇⡎⡏⡔⡕⡜⡝⡖⡗⡞⡟\
⡠⡡⡨⡩⡢⡣⡪⡫⡰⡱⡸⡹⡲⡳⡺⡻⡤⡥⡬⡭⡦⡧⡮⡯⡴⡵⡼⡽⡶⡷⡾⡿\
⢀⢁⢈⢉⢂⢃⢊⢋⢐⢑⢘⢙⢒⢓⢚⢛⢄⢅⢌⢍⢆⢇⢎⢏⢔⢕⢜⢝⢖⢗⢞⢟\
⢠⢡⢨⢩⢢⢣⢪⢫⢰⢱⢸⢹⢲⢳⢺⢻⢤⢥⢬⢭⢦⢧⢮⢯⢴⢵⢼⢽⢶⢷⢾⢿\
⣀⣁⣈⣉⣂⣃⣊⣋⣐⣑⣘⣙⣒⣓⣚⣛⣄⣅⣌⣍⣆⣇⣎⣏⣔⣕⣜⣝⣖⣗⣞⣟\
⣠⣡⣨⣩⣢⣣⣪⣫⣰⣱⣸⣹⣲⣳⣺⣻⣤⣥⣬⣭⣦⣧⣮⣯⣴⣵⣼⣽⣶⣷⣾⣿";

/// Quadrant-block sprite sheet: index bit `y*2 + x` ⇒ quadrant at (x, y)
/// in the char's 2×2 pixel cell.
const BLOCK_SPRITES: &str = " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█";

/// How pixels become glyphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelMode {
    /// One colored glyph run per pixel (full color).
    Color,
    /// Monochrome braille: each char is a 2-wide × 4-tall pixel cell —
    /// eight pixels per character, no color tags.
    Braille,
    /// Monochrome quadrant blocks: each char is a 2×2 pixel cell.
    Blocks,
}

impl PixelMode {
    pub const ALL: [PixelMode; 3] = [PixelMode::Color, PixelMode::Braille, PixelMode::Blocks];

    pub fn name(&self) -> &'static str {
        match self {
            PixelMode::Color => "Color",
            PixelMode::Braille => "Braille",
            PixelMode::Blocks => "Blocks",
        }
    }

    /// Pixel cell covered by one character: (width, height).
    pub fn cell(&self) -> (u32, u32) {
        match self {
            PixelMode::Color => (1, 1),
            PixelMode::Braille => (2, 4),
            PixelMode::Blocks => (2, 2),
        }
    }

    /// Tile edge in pixels. Mono modes pack 4–8 pixels per char, so their
    /// tiles are larger for the same char budget (fewer anchor bricks).
    pub fn tile_px(&self) -> u32 {
        match self {
            PixelMode::Color => TILE_PX,
            PixelMode::Braille | PixelMode::Blocks => 128,
        }
    }

    fn sprites(&self) -> &'static str {
        match self {
            PixelMode::Color => "",
            PixelMode::Braille => BRAILLE_SPRITES,
            PixelMode::Blocks => BLOCK_SPRITES,
        }
    }
}

/// Whether a pixel's dot is drawn in a monochrome mode.
fn mono_on(p: [u8; 4], opts: &TextOptions) -> bool {
    if p[3] < opts.alpha_threshold {
        return false;
    }
    // Rec.601 luma
    let luma = (p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114) / 1000;
    (luma as u8 >= opts.luma_threshold) != opts.invert
}

/// Encode one tile as monochrome sprite characters (braille or quadrant
/// blocks). Returns the text and its char count; edge cells pad with OFF
/// pixels like the reference implementation.
fn encode_mono_tile(img: &RgbaImage, opts: &TextOptions) -> (String, usize, bool) {
    let (cw, ch) = opts.mode.cell();
    let table: Vec<char> = opts.mode.sprites().chars().collect();
    let (w, h) = img.dimensions();
    let (cols, lines) = (w.div_ceil(cw), h.div_ceil(ch));
    let mut out = String::new();
    let mut chars = 0usize;
    let mut any_on = false;
    for cy in 0..lines {
        if cy > 0 {
            out.push('\n');
            chars += 1;
        }
        for cx in 0..cols {
            let mut idx = 0usize;
            for dy in 0..ch {
                for dx in 0..cw {
                    let (x, y) = (cx * cw + dx, cy * ch + dy);
                    if x < w && y < h && mono_on(img.get_pixel(x, y).0, opts) {
                        idx |= 1 << (dy * cw + dx);
                    }
                }
            }
            any_on |= idx != 0;
            out.push(table[idx]);
            chars += 1;
        }
    }
    (out, chars, any_on)
}

/// One tile of the image: a `TILE_PX`-square pixel patch (smaller at the
/// right/bottom edges) with its own bands.
#[derive(Clone, Debug)]
pub struct TextTile {
    /// First pixel column of this tile.
    pub start_col: usize,
    /// First pixel row of this tile.
    pub start_row: usize,
    pub bands: Vec<TextBand>,
}

/// Tile the image into square patches across BOTH axes (patch size per
/// [`PixelMode::tile_px`]), banding each patch at the char budget
/// (worst-case color patches split into a couple of bands; mono patches are
/// always one). Patches with nothing visible are skipped entirely — no
/// brick, no component.
pub fn encode_tiles(img: &RgbaImage, opts: &TextOptions) -> Result<Vec<TextTile>, String> {
    let (w, h) = img.dimensions();
    let tile_px = opts.tile_px();
    let mut tiles = Vec::new();
    let mut ty = 0u32;
    while ty < h {
        let th = tile_px.min(h - ty);
        let mut tx = 0u32;
        while tx < w {
            let tw = tile_px.min(w - tx);
            let sub = image::imageops::crop_imm(img, tx, ty, tw, th).to_image();
            let (bands, visible) = match opts.mode {
                PixelMode::Color => {
                    let bands = encode_bands(&sub, opts)?;
                    let visible = bands.iter().any(|b| b.text.chars().any(|c| c != '\n'));
                    (bands, visible)
                }
                PixelMode::Braille | PixelMode::Blocks => {
                    let (text, chars, any_on) = encode_mono_tile(&sub, opts);
                    let band = TextBand {
                        start_row: 0,
                        rows: th as usize,
                        text,
                        chars,
                    };
                    (vec![band], any_on)
                }
            };
            if visible {
                tiles.push(TextTile {
                    start_col: tx as usize,
                    start_row: ty as usize,
                    bands,
                });
            }
            tx += tw;
        }
        ty += th;
    }
    Ok(tiles)
}

fn row_too_wide(y: u32, chars: usize) -> String {
    format!(
        "row {y} encodes to {chars} chars, over the {MAX_COMPONENT_CHARS}-char \
         TextDisplay limit; use a narrower image or a smaller --char-repeat"
    )
}

/// Encode one image row. Updates `last_color` with the final emitted tag so
/// color runs can continue into following rows.
fn encode_row(
    img: &RgbaImage,
    y: u32,
    opts: &TextOptions,
    last_color: &mut Option<[u8; 3]>,
) -> (String, usize) {
    let (w, _) = img.dimensions();
    let mut out = String::new();
    let mut chars = 0usize;
    // Transparent pixels are buffered so a trailing run can be trimmed when the
    // empty glyph is a space (invisible at line end); any other glyph is kept.
    let mut pending_empty = 0usize;
    for x in 0..w {
        let p = img.get_pixel(x, y).0;
        if p[3] < opts.alpha_threshold {
            pending_empty += 1;
            continue;
        }
        for _ in 0..pending_empty * opts.char_repeat {
            out.push(opts.empty_char);
        }
        chars += pending_empty * opts.char_repeat;
        pending_empty = 0;

        let rgb = [p[0], p[1], p[2]];
        if *last_color != Some(rgb) {
            let tag = format!("<color=\"{:02X}{:02X}{:02X}\">", rgb[0], rgb[1], rgb[2]);
            chars += tag.chars().count();
            out.push_str(&tag);
            *last_color = Some(rgb);
        }
        for _ in 0..opts.char_repeat {
            out.push(opts.fill_char);
        }
        chars += opts.char_repeat;
    }
    if opts.empty_char != ' ' {
        for _ in 0..pending_empty * opts.char_repeat {
            out.push(opts.empty_char);
        }
        chars += pending_empty * opts.char_repeat;
    }
    (out, chars)
}

/// brdb ships `Vector3f` but no 2-component wrapper; the TextDisplay `Anchor`
/// field is a `Vector2f` struct in the component schema.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector2f {
    pub x: f32,
    pub y: f32,
}

impl AsBrdbValue for Vector2f {
    fn as_brdb_struct_prop_value(
        &self,
        schema: &BrdbSchema,
        _struct_name: BrdbInterned,
        prop_name: BrdbInterned,
    ) -> Result<&dyn AsBrdbValue, BrdbSchemaError> {
        match prop_name.get(schema).unwrap() {
            "X" => Ok(&self.x),
            "Y" => Ok(&self.y),
            n => unimplemented!("unimplemented Vector2f field {n}"),
        }
    }
}

/// The invisible, collision-less anchor cube all text rides on.
fn anchor_cube(position: Position, visible: bool) -> Brick {
    Brick {
        asset: BrickType::Procedural {
            asset: PB_DEFAULT_MICRO_BRICK,
            size: BrickSize::new(1, 1, 1),
        },
        position,
        collision: Collision {
            player: false,
            weapon: false,
            interact: false,
            tool: false,
            physics: false,
            ..Default::default()
        },
        visible,
        ..Default::default()
    }
}

/// Add a TextDisplay block with explicit geometry (LineHeight/Kerning/Offset)
/// on an anchor cube. `visible_anchor` shows the cube itself, useful when the
/// block is meant to be compared against the cube's edges.
pub fn add_text_block(
    world: &mut World,
    text: String,
    position: Position,
    line_height: f32,
    kerning: f32,
    offset: Vector3f,
    visible_anchor: bool,
    opts: &TextOptions,
) {
    let (font_idx, _) = world
        .global_data
        .external_asset_references
        .insert_full(("BrickFontDescriptor".to_string(), opts.font.to_string()));
    let block_opts = TextOptions {
        line_height,
        kerning,
        ..opts.clone()
    };
    world.add_brick(
        anchor_cube(position, visible_anchor).with_component(text_display_component(
            text,
            font_idx,
            offset,
            &block_opts,
            0,
            SavedBrickColor {
                r: 255,
                g: 255,
                b: 255,
                a: 255,
            },
            Vector2f { x: 0.0, y: 0.0 },
            0,
            0,
        )),
    );
}

/// Add a readable TextDisplay label brick (for annotating generated saves) —
/// plain text at the given LineHeight, not pixel art.
pub fn add_annotation(
    world: &mut World,
    text: String,
    position: Position,
    line_height: f32,
    opts: &TextOptions,
) {
    add_text_block(
        world,
        text,
        position,
        line_height,
        0.0,
        Vector3f {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        false,
        opts,
    );
}

/// Brickadia's number Variable gate brick.
const B_GATE_VARIABLE: BrickType = BrickType::str("B_1x1_Gate_Variable");

/// The in-game tunable geometry fields: (label, TextDisplay wire port,
/// current component value).
fn calibration_variables(opts: &TextOptions) -> [(&'static str, &'static str, f32); 6] {
    [
        (
            "Font
Size",
            "LineHeight",
            opts.line_height,
        ),
        ("Kerning", "Kerning", opts.kerning),
        (
            "Line
Offset",
            "LineOffset",
            opts.line_offset,
        ),
        (
            "Offset
X",
            "Offset.X",
            opts.offset_x,
        ),
        (
            "Offset
Y",
            "Offset.Y",
            opts.offset_y,
        ),
        (
            "Offset
Z",
            "Offset.Z",
            opts.offset_z,
        ),
    ]
}

/// Build the live calibration save: a 128×128 checkerboard rendered exactly
/// like a normal export (color, braille, and blocks modes alike), plus one
/// number Variable gate per tunable geometry field, wired into EVERY text
/// component — edit a variable in-game and the whole grid updates at once.
/// Each variable carries an upward-facing (+Z) label so it's clear which is
/// which. `cube_spacing` sets the world distance between tile anchors
/// (tiny spacing = tiny text displays, large = billboards).
pub fn build_calibration_world(opts: &TextOptions, cube_spacing: f32) -> World {
    const CHECKER_PX: u32 = 128;
    const CELL_PX: u32 = 8;

    // force small tiles in every mode (mono modes normally use 128px tiles)
    // so the checkerboard splits into a seamed grid — tile overlap/gaps are
    // exactly what calibration needs to expose
    let tile_px = TILE_PX.min(CHECKER_PX) as f32;
    // scale the export geometry so a tile's rendering matches the requested
    // anchor spacing (the variables re-tune everything live afterwards)
    let s = cube_spacing / (tile_px * opts.line_world_height * opts.pitch_x);
    let cal = TextOptions {
        line_world_height: opts.line_world_height * s,
        line_height: opts.line_height * s,
        kerning: opts.kerning * s,
        offset_x: opts.offset_x * s,
        offset_y: opts.offset_y * s,
        tile_override: Some(TILE_PX),
        ..opts.clone()
    };

    // dark/light nested checkerboard: alternating 32px regions use single-
    // and double-sized squares so scale errors read at two frequencies
    let mut img = RgbaImage::new(CHECKER_PX, CHECKER_PX);
    for y in 0..CHECKER_PX {
        for x in 0..CHECKER_PX {
            let cell = if ((x / 32) + (y / 32)) % 2 == 0 {
                CELL_PX
            } else {
                CELL_PX * 2
            };
            let on = ((x / cell) + (y / cell)) % 2 == 0;
            let c = if on {
                [240, 240, 240, 255]
            } else {
                [40, 40, 40, 255]
            };
            img.put_pixel(x, y, image::Rgba(c));
        }
    }

    let mut world = World::new();
    let tiles = encode_tiles(&img, &cal).expect("checkerboard must encode");
    let text_ids = add_text_tiles(&mut world, tiles, &cal);

    // sufficiently small displays depth-stagger their cubes, whose per-cube
    // Offset.Z compensation a shared wire would clobber — drop that
    // variable entirely there (the collidable graffiti canvas also sits at
    // x > 0 without being a stagger, so only collisionless cubes count)
    let staggered = world
        .bricks
        .iter()
        .any(|b| b.position.x > 0 && !b.collision.player);
    let variables: Vec<_> = calibration_variables(&cal)
        .into_iter()
        .filter(|(_, port, _)| !(staggered && *port == "Offset.Z"))
        .collect();

    // one visible Variable gate per tunable, in a labeled row above the grid
    let (font_idx, _) = world
        .global_data
        .external_asset_references
        .insert_full(("BrickFontDescriptor".to_string(), cal.font.to_string()));
    let grid_top = world.bricks.iter().map(|b| b.position.z).max().unwrap_or(1) + 10;
    let label_opts = TextOptions {
        line_height: 2.5,
        kerning: 0.0,
        // labels are plain text: never apply the mono modes' cell scaling
        mode: PixelMode::Color,
        // nor the image's material — a Graffiti/Glow label is unreadable
        material: TextMaterial::Unlit,
        ..cal.clone()
    };
    let mut var_ids = Vec::new();
    for (i, (label, _port, value)) in variables.iter().copied().enumerate() {
        let (brick, id) = Brick {
            asset: B_GATE_VARIABLE,
            position: Position::new(0, i as i32 * 20, grid_top),
            // rotate the gate so its display (and label) faces the viewer
            direction: Direction::XPositive,
            // dark brick so the white label reads
            color: Color {
                r: 25,
                g: 25,
                b: 30,
            },
            ..Default::default()
        }
        .with_component(
            LiteralComponent::new("BrickComponentType_WireGraphPseudo_Var").with_data([(
                "Value",
                Box::new(WireVariant::Number(value as f64)) as Box<dyn AsBrdbValue>,
            )]),
        )
        .with_component(text_display_component(
            label.to_string(),
            font_idx,
            Vector3f {
                x: 0.0,
                y: 0.0,
                // lift the label off the gate face (Offset acts in line
                // units; the label's line is ~4 world units)
                z: 0.5,
            },
            &label_opts,
            // the gate is rotated: its local top (+Z) faces the viewer
            4,
            // white with a black outline, on a dark brick
            SavedBrickColor {
                r: 255,
                g: 255,
                b: 255,
                a: 255,
            },
            // centered on the gate face
            Vector2f { x: 0.5, y: 0.5 },
            // bold
            1,
            // outlined
            2,
        ))
        .with_id_split();
        world.add_brick(brick);
        var_ids.push(id);
    }

    // wire every variable into every tile's text component
    for (var_id, (_, port, _)) in var_ids.iter().zip(variables) {
        for &text_id in &text_ids {
            world.add_wire_connection(
                WirePort::new(*var_id, "BrickComponentType_WireGraphPseudo_Var", "Value"),
                WirePort::new(text_id, "Component_TextDisplay", port),
            );
        }
    }

    world.meta.bundle.description = "Text render live calibration".to_string();
    make_text_prefab(&mut world, CHECKER_PX, CHECKER_PX, &cal);
    // re-register: the wires (and their port names) were added after
    // add_text_tiles registered the component types
    world.register_used_components();
    world
}

/// Mark the world as a prefab whose pivots/bounds cover the full rendered
/// image. The anchor-cube grid already spans most of the image; each
/// outermost tile's content extends up to one tile beyond its cube toward
/// the text's right (world -Y) and downward (-Z), so the bounds pad those
/// two edges — a ground placement then rests the image (not the cubes) on
/// the ground.
pub fn make_text_prefab(world: &mut World, _img_w: u32, _img_h: u32, opts: &TextOptions) {
    let span_x = (TILE_PX as f32 * opts.line_world_height * opts.pitch_x).ceil() as i32;
    let span_z = (TILE_PX as f32 * opts.line_world_height * opts.pitch_y).ceil() as i32;
    let (bmin, bmax) = world
        .brick_bounds()
        .unwrap_or((Position::ZERO, Position::ZERO));
    let min = Position::new(bmin.x, bmin.y - span_x, bmin.z - span_z);
    world.meta.bundle.level_type = "Prefab".to_string();
    world.meta.prefab = Some(PrefabJson::from_bounds(min, bmax));
}

/// A 1×1×1 micro brick is 2 world units tall; the anchor cubes pack into a
/// flush vertical column instead of spreading out one band-height apart.
const BRICK_STACK_STEP: i32 = 2;

/// Add one TextDisplay micro brick per band to the world, as a single tile
/// at the origin. See [`add_text_tiles`] for tiled images.
pub fn add_text_bricks(world: &mut World, bands: Vec<TextBand>, opts: &TextOptions) {
    add_text_tiles(
        world,
        vec![TextTile {
            start_col: 0,
            start_row: 0,
            bands,
        }],
        opts,
    );
}

/// Add all tiles' bricks. Each tile's anchor cube sits AT its patch's image
/// position (brick coordinates carry ALL the placement — the game does not
/// honor large component Offset values), forming a sparse cube grid with
/// the same footprint as the image, always covered by the rendered text.
/// A multi-band tile stacks its extra cubes BACKWARD (world +X, behind the
/// wall) so the grid's vertical/horizontal footprint stays one cube per
/// tile; their text returns to the common plane via local Z. Component
/// offsets carry only the glyph nudges and these tiny compensations.
pub fn add_text_tiles(world: &mut World, tiles: Vec<TextTile>, opts: &TextOptions) -> Vec<usize> {
    // Index into global_data.external_asset_references, written as the Font
    // object reference (same pattern bearilog uses for item assets).
    let (font_idx, _) = world
        .global_data
        .external_asset_references
        .insert_full(("BrickFontDescriptor".to_string(), opts.font.to_string()));

    // grid spacing follows the font's ACTUAL rendered size, not the nominal
    // pixel size — tiles anchor where their neighbors' glyphs end. The CUBE
    // POSITION carries each tile's in-plane placement exactly (rounded to
    // the brick grid); component offsets stay pure glyph nudges. Any cube
    // that would intersect an already-placed one — crowded tiles at tiny
    // pixel sizes, or a tile's extra bands — steps BACKWARD (world +X) one
    // cube at a time, its text returning via local Z.
    let tile_px = opts.tile_px() as usize;
    let step_true_x = tile_px as f32 * opts.line_world_height * opts.pitch_x;
    let step_true_z = tile_px as f32 * opts.line_world_height * opts.pitch_y;

    // brdb's chunk encoding mishandles negative coordinates (bricks land in
    // wrong chunks in-game), so the grid is translated to keep EVERY brick
    // coordinate non-negative: the rightmost tile sits at y=0 and the image
    // grows toward +Y (the text's right is world -Y, so the order mirrors).
    let max_kx = tiles
        .iter()
        .map(|t| t.start_col / tile_px)
        .max()
        .unwrap_or(0) as i32;
    let max_ky = tiles
        .iter()
        .map(|t| t.start_row / tile_px)
        .max()
        .unwrap_or(0) as i32;
    // world +X points TOWARD the viewer (verified in-game), and bricks
    // cannot use negative coordinates — so depth slots are assigned first,
    // then the whole grid is arranged with the front plane at the deepest
    // slot used and crowded cubes stepping down toward x=0 (away from the
    // viewer), their text coming forward by their distance behind the
    // front plane.
    let mut pending = Vec::new();
    let mut placed: Vec<Position> = Vec::new();
    for tile in tiles {
        let kx = max_kx - (tile.start_col / tile_px) as i32;
        let ky = max_ky - (tile.start_row / tile_px) as i32;
        let y_exact = kx as f32 * step_true_x;
        let z_exact = ky as f32 * step_true_z;
        let y = y_exact.round() as i32;
        // bottom tile row anchors at z=1; rows above it proportionally higher
        let z_tile = 1 + z_exact.round() as i32;
        // brick positions are integers; at tiny tile spans the rounding
        // residue (≤0.5 units) is a visible fraction of a tile, so the text
        // compensates it (offsets are world units on all axes)
        let res_y = y as f32 - y_exact;
        let res_z = (z_tile - 1) as f32 - z_exact;
        for band in tile.bands {
            // take the shallowest free depth slot at this in-plane spot
            let mut depth = 0;
            while placed
                .iter()
                .any(|p| (p.x - depth).abs() < 2 && (p.y - y).abs() < 2 && (p.z - z_tile).abs() < 2)
            {
                depth += BRICK_STACK_STEP;
            }
            placed.push(Position::new(depth, y, z_tile));
            pending.push((band, y, z_tile, depth, res_y, res_z));
        }
    }
    let max_depth = pending
        .iter()
        .map(|(_, _, _, d, _, _)| *d)
        .max()
        .unwrap_or(0);
    // graffiti renders only where it projects onto a collision surface, so
    // those exports get an invisible collidable wall flush in front of the
    // text plane. The wall must cover the content hanging one tile span
    // below/right of the outermost anchors — and negative brick coordinates
    // are forbidden, so the whole grid first shifts up/left by that overhang.
    let span_y = step_true_x.ceil() as i32;
    let span_z = step_true_z.ceil() as i32;
    let (shift_y, shift_z) = if opts.material.is_graffiti() {
        (span_y + 1, span_z + 1)
    } else {
        (0, 0)
    };
    let mut bricks = Vec::new();
    let mut ids = Vec::new();
    for (band, y, z_tile, depth, res_y, res_z) in pending {
        let offset = Vector3f {
            // cube rounded toward +Y (image-left) pulls its text back right
            // (local +X = world -Y)
            x: opts.offset_x + res_y,
            // cube rounded upward pulls its text back down
            y: opts.offset_y - res_z,
            // text comes forward by the cube's distance behind the front
            // plane (all offset axes share the same world units)
            z: opts.offset_z + depth as f32,
        };
        let (brick, id) = anchor_cube(
            Position::new(max_depth - depth, y + shift_y, z_tile + shift_z),
            false,
        )
        .with_component(text_display_component(
            band.text,
            font_idx,
            offset,
            opts,
            0,
            SavedBrickColor {
                r: 255,
                g: 255,
                b: 255,
                a: 255,
            },
            Vector2f { x: 0.0, y: 0.0 },
            0,
            0,
        ))
        .with_id_split();
        ids.push(id);
        bricks.push(brick);
    }
    if opts.material.is_graffiti() && !placed.is_empty() {
        bricks.extend(graffiti_canvas(
            &placed,
            (max_depth, shift_y, shift_z),
            (span_y, span_z),
        ));
    }
    world.add_bricks(bricks);
    // brdb 0.6 requires used component types registered in global data before
    // writing; rebuilds from the max schema, so calling once at the end is fine.
    world.register_used_components();
    ids
}

/// Graffiti projects onto collision surfaces — build one: invisible,
/// collidable micro-brick slabs covering the image rectangle (the anchor
/// grid plus the one-tile content overhang toward -Y/-Z), their back faces
/// flush against the text plane (the front cubes' viewer-side face).
fn graffiti_canvas(
    placed: &[Position],
    (max_depth, shift_y, shift_z): (i32, i32, i32),
    (span_y, span_z): (i32, i32),
) -> Vec<Brick> {
    // half-extent cap per slab, matching the heightmap generator's 250
    // ceiling for procedural brick sizes
    const MAX_HALF: i32 = 250;
    let y_lo = placed.iter().map(|p| p.y).min().unwrap() + shift_y - 1 - span_y;
    let y_hi = placed.iter().map(|p| p.y).max().unwrap() + shift_y + 1;
    let z_lo = placed.iter().map(|p| p.z).min().unwrap() + shift_z - 1 - span_z;
    let z_hi = placed.iter().map(|p| p.z).max().unwrap() + shift_z + 1;
    // one micro unit thick, just past the front cubes' faces (x=max_depth+1)
    let x = max_depth + 2;
    let mut slabs = Vec::new();
    let mut z = z_lo;
    while z < z_hi {
        let hz = ((z_hi - z).min(2 * MAX_HALF) + 1) / 2;
        let mut y = y_lo;
        while y < y_hi {
            let hy = ((y_hi - y).min(2 * MAX_HALF) + 1) / 2;
            slabs.push(Brick {
                asset: BrickType::Procedural {
                    asset: PB_DEFAULT_MICRO_BRICK,
                    size: BrickSize::new(1, hy as u16, hz as u16),
                },
                position: Position::new(x, y + hy, z + hz),
                visible: false,
                // default collision stays on: the wall is the canvas
                ..Default::default()
            });
            y += 2 * hy;
        }
        z += 2 * hz;
    }
    slabs
}

/// TextDisplay component data mirroring the user's calibrated reference
/// clipboards: Anchor top-left, glyph-fit Offset, LineHeight sized so pixel
/// rows land on world units — all seeded by the selected [`FontPreset`].
fn text_display_component(
    text: String,
    font_idx: usize,
    offset: Vector3f,
    opts: &TextOptions,
    face: u8,
    color: SavedBrickColor,
    anchor: Vector2f,
    typeface: u8,
    outline: u8,
) -> LiteralComponent {
    LiteralComponent::new("Component_TextDisplay").with_data([
        ("Text", Box::new(text) as Box<dyn AsBrdbValue>),
        ("Font", Box::new(BrdbValue::Asset(Some(font_idx)))),
        ("Anchor", Box::new(anchor)),
        ("Offset", Box::new(offset)),
        ("Rotation", Box::new(0.0f32)),
        ("Skew", Box::new(0.0f32)),
        ("Kerning", Box::new(opts.kerning)),
        ("LineHeight", Box::new(opts.line_height)),
        ("LineOffset", Box::new(opts.line_offset)),
        ("Color", Box::new(color)),
        ("MaterialSlider", Box::new(opts.material_intensity)),
        ("ShadingWidth", Box::new(opts.shading_width)),
        (
            "OutlineColor",
            Box::new(SavedBrickColor {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }),
        ),
        ("OutlineWidth", Box::new(2.0f32)),
        ("ScuffWidth", Box::new(opts.scuff)),
        ("GraffitiDepthLimit", Box::new(opts.graffiti_depth_limit)),
        ("GraffitiAngleLimit", Box::new(opts.graffiti_angle_limit)),
        ("GraffitiLayer", Box::new(opts.graffiti_priority)),
        ("bEnabled", Box::new(true)),
        ("Face", Box::new(face)),
        ("bAlignToWedge", Box::new(false)),
        ("bOverrideColor", Box::new(true)),
        ("Typeface", Box::new(typeface)),
        ("Billboard", Box::new(0u8)),
        ("Material", Box::new(opts.material.byte())),
        ("Shading", Box::new(opts.shading.byte())),
        ("bFlipShading", Box::new(opts.invert_shading)),
        ("Outline", Box::new(outline)),
        ("bSharpCorners", Box::new(true)),
        ("bSharpOutlines", Box::new(true)),
        ("bOverrideOutlineColor", Box::new(true)),
        ("bForeground", Box::new(false)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    const RED: Rgba<u8> = Rgba([255, 0, 0, 255]);
    const GREEN: Rgba<u8> = Rgba([0, 255, 0, 255]);
    const CLEAR: Rgba<u8> = Rgba([0, 0, 0, 0]);

    fn img(pixels: &[&[Rgba<u8>]]) -> RgbaImage {
        let h = pixels.len() as u32;
        let w = pixels[0].len() as u32;
        let mut img = RgbaImage::new(w, h);
        for (y, row) in pixels.iter().enumerate() {
            for (x, p) in row.iter().enumerate() {
                img.put_pixel(x as u32, y as u32, *p);
            }
        }
        img
    }

    /// Encode with defaults, asserting a single band results.
    fn text(i: &RgbaImage) -> String {
        let bands = encode_bands(i, &TextOptions::default()).unwrap();
        assert_eq!(bands.len(), 1);
        bands.into_iter().next().unwrap().text
    }

    #[test]
    fn single_opaque_pixel() {
        assert_eq!(text(&img(&[&[RED]])), "<color=\"FF0000\">██");
    }

    #[test]
    fn trailing_transparent_trimmed() {
        assert_eq!(text(&img(&[&[RED, CLEAR]])), "<color=\"FF0000\">██");
    }

    #[test]
    fn custom_empty_char_kept() {
        let opts = TextOptions {
            empty_char: '.',
            ..Default::default()
        };
        let bands = encode_bands(&img(&[&[RED, CLEAR]]), &opts).unwrap();
        assert_eq!(bands[0].text, "<color=\"FF0000\">██..");
    }

    #[test]
    fn color_run_spans_transparent_gap() {
        assert_eq!(
            text(&img(&[&[RED, CLEAR, RED]])),
            "<color=\"FF0000\">██  ██"
        );
    }

    #[test]
    fn color_run_spans_rows() {
        assert_eq!(text(&img(&[&[RED], &[RED]])), "<color=\"FF0000\">██\n██");
    }

    #[test]
    fn color_change_emits_new_tag() {
        assert_eq!(
            text(&img(&[&[RED, GREEN]])),
            "<color=\"FF0000\">██<color=\"00FF00\">██"
        );
    }

    #[test]
    fn alpha_threshold_boundary() {
        let below = Rgba([255, 0, 0, 127]);
        let at = Rgba([255, 0, 0, 128]);
        assert_eq!(text(&img(&[&[below, at]])), "  <color=\"FF0000\">██");
    }

    #[test]
    fn char_repeat_three() {
        let opts = TextOptions {
            char_repeat: 3,
            ..Default::default()
        };
        let bands = encode_bands(&img(&[&[RED]]), &opts).unwrap();
        assert_eq!(bands[0].text, "<color=\"FF0000\">███");
    }

    #[test]
    fn fully_transparent_rows_keep_line_breaks() {
        assert_eq!(text(&img(&[&[CLEAR], &[RED]])), "\n<color=\"FF0000\">██");
    }

    /// 20-wide row of per-pixel alternating colors = 20 × (16 + 2) = 360 chars.
    /// Band capacity: 360 + 361·(n−1) ≤ 10 000 ⇒ 27 rows. 60 rows ⇒ 27+27+6.
    #[test]
    fn bands_split_at_char_limit() {
        let mut i = RgbaImage::new(20, 60);
        for y in 0..60 {
            for x in 0..20 {
                i.put_pixel(x, y, if x % 2 == 0 { RED } else { GREEN });
            }
        }
        let bands = encode_bands(&i, &TextOptions::default()).unwrap();
        assert_eq!(
            bands.iter().map(|b| b.start_row).collect::<Vec<_>>(),
            vec![0, 27, 54]
        );
        assert_eq!(bands.iter().map(|b| b.rows).sum::<usize>(), 60);
        for b in &bands {
            assert!(b.chars <= MAX_COMPONENT_CHARS);
            assert_eq!(b.chars, b.text.chars().count());
            // start_row padding newlines first, then content with fresh color
            assert!(
                b.text[..b.start_row].chars().all(|c| c == '\n'),
                "band must start with start_row padding newlines"
            );
            assert!(
                b.text[b.start_row..].starts_with("<color=\""),
                "band must reset color state"
            );
        }
    }

    #[test]
    fn images_tile_in_both_axes() {
        // 72×72 alternating-color pixels: 3×3 tile grid (32+32+8 each axis)
        let mut i = RgbaImage::new(72, 72);
        for y in 0..72 {
            for x in 0..72 {
                i.put_pixel(x, y, if x % 2 == 0 { RED } else { GREEN });
            }
        }
        let tiles = encode_tiles(&i, &TextOptions::default()).unwrap();
        assert_eq!(tiles.len(), 9, "3x3 grid of {TILE_PX}px tiles");
        let expected_origins: Vec<(usize, usize)> = [0, 32, 64]
            .iter()
            .flat_map(|&r| [0usize, 32, 64].iter().map(move |&c| (c, r)))
            .collect();
        assert_eq!(
            tiles
                .iter()
                .map(|t| (t.start_col, t.start_row))
                .collect::<Vec<_>>(),
            expected_origins
        );
        for t in &tiles {
            assert!(t.bands.iter().all(|b| b.chars <= MAX_COMPONENT_CHARS));
            let rows_expected = if t.start_row == 64 { 8 } else { 32 };
            assert_eq!(t.bands.iter().map(|b| b.rows).sum::<usize>(), rows_expected);
        }

        // each tile's anchor cube sits AT its patch position scaled by the
        // rendered pitch, translated so all coordinates stay non-negative
        // (text right = world -Y ⇒ rightmost tile at y=0, image grows +Y;
        // brdb's negative-chunk encoding is unreliable in-game); z descends
        // with start_row from the top tile row (bottom row at z=1); extra
        // band cubes go BACKWARD (world +X) at constant y/z footprint
        let d = TextOptions::default();
        let pitch = d.line_world_height * d.pitch_x;
        let step = (32.0 * pitch).ceil().max(2.0) as i32;
        let per_tile: Vec<(i32, i32, i32)> = tiles
            .iter()
            .map(|t| {
                (
                    (2 - (t.start_col / 32) as i32) * step,
                    1 + (2 - (t.start_row / 32) as i32) * step,
                    t.bands.len() as i32,
                )
            })
            .collect();
        let max_depth = 2 * (per_tile.iter().map(|(_, _, n)| *n).max().unwrap() - 1);
        let mut world = brdb::World::new();
        add_text_tiles(&mut world, tiles, &TextOptions::default());
        for (y, z_tile, n) in per_tile {
            let mut xs: Vec<i32> = world
                .bricks
                .iter()
                .filter(|b| b.position.y == y && b.position.z == z_tile)
                .map(|b| b.position.x)
                .collect();
            xs.sort_unstable();
            // band 0 on the front plane (x = max_depth), extras stepping back
            let mut expected: Vec<i32> = (0..n).map(|i| max_depth - 2 * i).collect();
            expected.sort_unstable();
            assert_eq!(xs, expected, "tile at y={y} z={z_tile}: bands step back");
        }
        assert!(world.bricks.iter().all(|b| b.position.z >= 1));
        // every coordinate must be non-negative (brdb negative-chunk bug)
        assert!(world.bricks.iter().all(|b| b.position.x >= 0));
        assert!(world.bricks.iter().all(|b| b.position.y >= 0));
    }

    #[test]
    fn braille_cells_encode_dot_patterns() {
        let opts = TextOptions {
            mode: PixelMode::Braille,
            ..Default::default()
        };
        // white pixel at (0,0) of a 2×4 cell ⇒ dot 1 (⠁); all on ⇒ ⣿
        let mut i = RgbaImage::new(2, 4);
        i.put_pixel(0, 0, Rgba([255, 255, 255, 255]));
        let tiles = encode_tiles(&i, &opts).unwrap();
        assert_eq!(tiles[0].bands[0].text, "⠁");

        let full = RgbaImage::from_pixel(2, 4, Rgba([255, 255, 255, 255]));
        let tiles = encode_tiles(&full, &opts).unwrap();
        assert_eq!(tiles[0].bands[0].text, "⣿");

        // dark pixels are off by default, on when inverted
        let dark = RgbaImage::from_pixel(2, 4, Rgba([0, 0, 0, 255]));
        assert!(encode_tiles(&dark, &opts).unwrap().is_empty());
        let inv = TextOptions {
            invert: true,
            ..opts.clone()
        };
        assert_eq!(encode_tiles(&dark, &inv).unwrap()[0].bands[0].text, "⣿");
    }

    #[test]
    fn block_cells_encode_quadrants() {
        let opts = TextOptions {
            mode: PixelMode::Blocks,
            ..Default::default()
        };
        // 4×2: left cell top-left quadrant only; right cell all four
        let mut i = RgbaImage::new(4, 2);
        i.put_pixel(0, 0, Rgba([255, 255, 255, 255]));
        for (x, y) in [(2, 0), (3, 0), (2, 1), (3, 1)] {
            i.put_pixel(x, y, Rgba([255, 255, 255, 255]));
        }
        let tiles = encode_tiles(&i, &opts).unwrap();
        assert_eq!(tiles[0].bands[0].text, "▘█");

        // multi-line: 2×4 all-on in blocks = two stacked █ lines
        let full = RgbaImage::from_pixel(2, 4, Rgba([255, 255, 255, 255]));
        let tiles = encode_tiles(&full, &opts).unwrap();
        assert_eq!(tiles[0].bands[0].text, "█\n█");
        assert_eq!(tiles[0].bands[0].chars, 3);
    }

    #[test]
    fn tiny_pixel_sizes_never_overlap_cubes() {
        // 0.02 units/px: a 32px tile renders ~0.6 units wide, far below the
        // 2-unit cube size — crowded cubes must step BACKWARD (world +X)
        // instead of spreading in-plane, keeping their true y/z placement
        let opts = TextOptions {
            line_world_height: 0.02,
            ..Default::default()
        };
        let mut i = RgbaImage::new(96, 96);
        for y in 0..96 {
            for x in 0..96 {
                i.put_pixel(x, y, if (x / 8 + y / 8) % 2 == 0 { RED } else { GREEN });
            }
        }
        let tiles = encode_tiles(&i, &opts).unwrap();
        let mut world = brdb::World::new();
        add_text_tiles(&mut world, tiles, &opts);
        let positions: Vec<(i32, i32, i32)> = world
            .bricks
            .iter()
            .map(|b| (b.position.x, b.position.y, b.position.z))
            .collect();
        for (i, a) in positions.iter().enumerate() {
            for b in &positions[i + 1..] {
                assert!(
                    (a.0 - b.0).abs() >= 2 || (a.1 - b.1).abs() >= 2 || (a.2 - b.2).abs() >= 2,
                    "cubes {a:?} and {b:?} intersect"
                );
            }
        }
        // in-plane placement stays true (rounded), so crowded cubes go deep
        assert!(
            positions.iter().any(|p| p.0 > 0),
            "depth staggering engaged"
        );
        assert!(positions.iter().all(|p| p.1 <= 2 && p.2 <= 3));
        assert!(positions.iter().all(|p| p.0 >= 0 && p.1 >= 0 && p.2 >= 0));
    }

    /// Graffiti projects onto collision surfaces, so those exports carry an
    /// invisible collidable wall flush in front of the text plane, covering
    /// the image rectangle — with the grid shifted up/left so the wall's
    /// bricks stay in non-negative coordinates.
    #[test]
    fn graffiti_gets_invisible_collision_canvas() {
        let opts = TextOptions {
            material: TextMaterial::Graffiti,
            ..Default::default()
        };
        // 33×33 → a 2×2 tile grid
        let i = RgbaImage::from_pixel(33, 33, RED);
        let tiles = encode_tiles(&i, &opts).unwrap();
        let mut world = brdb::World::new();
        add_text_tiles(&mut world, tiles, &opts);

        let (slabs, cubes): (Vec<_>, Vec<_>) =
            world.bricks.iter().partition(|b| b.collision.player);
        assert!(!slabs.is_empty(), "graffiti must add a collision canvas");
        assert!(slabs.iter().all(|b| !b.visible));
        // the wall sits flush in front of the front plane's cube faces
        let front = cubes.iter().map(|b| b.position.x).max().unwrap();
        assert!(slabs.iter().all(|b| b.position.x == front + 2));
        assert!(
            world
                .bricks
                .iter()
                .all(|b| b.position.x >= 0 && b.position.y >= 0 && b.position.z >= 0)
        );
        // the wall covers the anchor grid plus the one-tile content overhang
        // beyond the outermost anchors toward -Y/-Z
        let span = (32.0 * (30.0 / 32.0_f32)).ceil() as i32; // default preset
        let (mut wy, mut wz) = ((i32::MAX, i32::MIN), (i32::MAX, i32::MIN));
        for b in &slabs {
            let BrickType::Procedural { size, .. } = &b.asset else {
                panic!("canvas must be procedural");
            };
            wy = (
                wy.0.min(b.position.y - size.y as i32),
                wy.1.max(b.position.y + size.y as i32),
            );
            wz = (
                wz.0.min(b.position.z - size.z as i32),
                wz.1.max(b.position.z + size.z as i32),
            );
        }
        let cy = cubes.iter().map(|b| b.position.y).collect::<Vec<_>>();
        let cz = cubes.iter().map(|b| b.position.z).collect::<Vec<_>>();
        let (cy_lo, cy_hi) = (*cy.iter().min().unwrap(), *cy.iter().max().unwrap());
        let (cz_lo, cz_hi) = (*cz.iter().min().unwrap(), *cz.iter().max().unwrap());
        assert!(
            wy.0 <= cy_lo - 1 - span && wy.1 >= cy_hi + 1,
            "wall covers Y"
        );
        assert!(
            wz.0 <= cz_lo - 1 - span && wz.1 >= cz_hi + 1,
            "wall covers Z"
        );

        // non-graffiti materials add no canvas
        let plain = TextOptions::default();
        let mut world = brdb::World::new();
        add_text_tiles(&mut world, encode_tiles(&i, &plain).unwrap(), &plain);
        assert!(world.bricks.iter().all(|b| !b.collision.player));
    }

    #[test]
    fn fully_transparent_tiles_are_skipped() {
        // 64×32: left half opaque, right half fully transparent
        let mut i = RgbaImage::new(64, 32);
        for y in 0..32 {
            for x in 0..32 {
                i.put_pixel(x, y, RED);
            }
        }
        let tiles = encode_tiles(&i, &TextOptions::default()).unwrap();
        assert_eq!(tiles.len(), 1, "transparent tile must be skipped");
        assert_eq!((tiles[0].start_col, tiles[0].start_row), (0, 0));
    }

    #[test]
    fn row_too_wide_is_error() {
        let mut i = RgbaImage::new(600, 1);
        for x in 0..600 {
            i.put_pixel(x, 0, if x % 2 == 0 { RED } else { GREEN });
        }
        let err = encode_bands(&i, &TextOptions::default()).unwrap_err();
        assert!(err.contains("row 0"), "unexpected error: {err}");
    }

    #[test]
    fn calibration_world_wires_variables_to_all_tiles() {
        // color mode: 128px checker / 32px tiles = 16 text cubes
        let world = build_calibration_world(&TextOptions::default(), 30.0);
        let text_cubes = world
            .bricks
            .iter()
            .filter(|b| matches!(b.asset, BrickType::Procedural { .. }))
            .count();
        let vars = world.bricks.len() - text_cubes;
        assert_eq!(text_cubes, 16, "4x4 tile grid");
        assert_eq!(vars, 6, "one variable gate per tunable");
        assert_eq!(world.wires.len(), 6 * 16, "every var wired to every tile");
        assert!(world.bricks.iter().all(|b| b.position.z >= 0));
        world
            .to_brz_vec()
            .expect("calibration world must encode to brz");

        // braille mode: tiles are forced down to 32px so seams exist to
        // check — same 4x4 grid, fully wired
        let opts = TextOptions {
            mode: PixelMode::Braille,
            ..Default::default()
        };
        let world = build_calibration_world(&opts, 120.0);
        assert_eq!(world.wires.len(), 6 * 16);
        world
            .to_brz_vec()
            .expect("braille calibration world must encode to brz");

        // a tiny display staggers cubes in depth: the Offset Z variable is
        // dropped so the shared wire can't clobber per-cube compensation
        let world = build_calibration_world(&TextOptions::default(), 1.0);
        let staggered = world.bricks.iter().any(|b| b.position.x > 0);
        assert!(staggered, "tiny spacing must depth-stagger");
        assert_eq!(world.wires.len() % 16, 0);
        assert_eq!(world.wires.len() / 16, 5, "Offset Z variable dropped");
    }

    #[test]
    fn bricks_stack_one_per_band() {
        use brdb::{Position, World};
        let mut i = RgbaImage::new(20, 60);
        for y in 0..60 {
            for x in 0..20 {
                i.put_pixel(x, y, if x % 2 == 0 { RED } else { GREEN });
            }
        }
        let opts = TextOptions::default();
        let bands = encode_bands(&i, &opts).unwrap();
        let mut world = World::new();
        add_text_bricks(&mut world, bands, &opts);

        assert_eq!(world.bricks.len(), 3);
        // a single tile anchors at z=1; band 0's cube sits on the front
        // plane (world +X faces the viewer) and extra bands step backward
        // toward x=0, their text returning via local Z
        assert_eq!(world.bricks[0].position, Position::new(4, 0, 1));
        assert_eq!(world.bricks[1].position, Position::new(2, 0, 1));
        assert_eq!(world.bricks[2].position, Position::new(0, 0, 1));
        assert_eq!(world.bricks[0].components.len(), 1);
        for b in &world.bricks {
            assert!(!b.visible, "anchor bricks must be invisible");
            assert!(
                !b.collision.player
                    && !b.collision.weapon
                    && !b.collision.interact
                    && !b.collision.tool
                    && !b.collision.physics,
                "anchor bricks must have no collision"
            );
        }
        assert!(world.global_data.external_asset_references.contains(&(
            "BrickFontDescriptor".to_string(),
            "MonaspaceArgon".to_string()
        )));
    }
}
