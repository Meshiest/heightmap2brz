use brdb::{AsBrdbValue, Brz, IntoReader, World, schema::BrdbValue};
use heightmap::text::{TextMaterial, TextOptions, TextShading, add_text_bricks, encode_bands};
use image::{Rgba, RgbaImage};

/// Raw value of an enum-typed component property.
fn enum_raw(value: &BrdbValue) -> u64 {
    match value {
        BrdbValue::Enum(e) => e.get_value_raw(),
        other => panic!("expected an enum value, got {other:?}"),
    }
}

/// Writes a tiny image's world to a temp .brz, reads it back with brdb, and
/// asserts the TextDisplay component round-trips with the expected encoding.
#[test]
fn text_component_roundtrips_through_brz() {
    let mut img = RgbaImage::new(3, 2);
    // row 0: red, red, transparent (trailing run trimmed)
    img.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
    img.put_pixel(1, 0, Rgba([255, 0, 0, 255]));
    img.put_pixel(2, 0, Rgba([0, 0, 0, 0]));
    // row 1: green, transparent, red (gap preserved, red re-tagged after green)
    img.put_pixel(0, 1, Rgba([0, 255, 0, 255]));
    img.put_pixel(1, 1, Rgba([0, 0, 0, 0]));
    img.put_pixel(2, 1, Rgba([255, 0, 0, 255]));

    let opts = TextOptions::default();
    let bands = encode_bands(&img, &opts).unwrap();
    assert_eq!(bands.len(), 1);

    let mut world = World::new();
    add_text_bricks(&mut world, bands, &opts);
    world.meta.bundle.description = "roundtrip test".to_string();

    let data = world.to_brz_vec().unwrap();
    let path = std::env::temp_dir().join(format!("h2b_text_roundtrip_{}.brz", std::process::id()));
    std::fs::write(&path, data).unwrap();

    let db = Brz::open(&path).unwrap().into_reader();
    let mut found = 0;
    for chunk in db.brick_chunk_index(1).unwrap() {
        // 1 = main brick grid
        let (_soa, comps) = db.component_chunk_soa(1, chunk.index).unwrap();
        for c in comps {
            let text = c.prop("Text").unwrap().as_brdb_str().unwrap();
            assert_eq!(
                text,
                "<color=\"FF0000\">████\n<color=\"00FF00\">██  <color=\"FF0000\">██"
            );
            let anchor = c.prop("Anchor").unwrap();
            assert_eq!(anchor.prop("X").unwrap().as_brdb_f32().unwrap(), 0.0);
            assert_eq!(anchor.prop("Y").unwrap().as_brdb_f32().unwrap(), 0.0);
            // user-calibrated glyph fit: LineHeight 0.61 + (0, -0.2, 0) offset
            let line_height = c.prop("LineHeight").unwrap().as_brdb_f32().unwrap();
            assert_eq!(line_height, 0.61);
            let offset = c.prop("Offset").unwrap();
            assert_eq!(offset.prop("X").unwrap().as_brdb_f32().unwrap(), 0.0);
            assert_eq!(offset.prop("Y").unwrap().as_brdb_f32().unwrap(), -0.2);
            // out-of-plane push so the anchor cube hides behind the image
            assert_eq!(offset.prop("Z").unwrap().as_brdb_f32().unwrap(), 0.0);
            found += 1;
        }
    }
    assert_eq!(found, 1, "expected exactly one TextDisplay component");
    std::fs::remove_file(&path).ok();
}

/// Non-default material settings survive the trip into the component data.
#[test]
fn material_settings_roundtrip_through_brz() {
    let mut img = RgbaImage::new(1, 1);
    img.put_pixel(0, 0, Rgba([255, 0, 0, 255]));

    let opts = TextOptions {
        material: TextMaterial::Glow,
        material_intensity: 7,
        scuff: 0.25,
        graffiti_depth_limit: 12.0,
        graffiti_angle_limit: 60.0,
        graffiti_priority: 3,
        shading: TextShading::Chamfer,
        shading_width: 1.5,
        invert_shading: true,
        ..Default::default()
    };
    let bands = encode_bands(&img, &opts).unwrap();

    let mut world = World::new();
    add_text_bricks(&mut world, bands, &opts);

    let data = world.to_brz_vec().unwrap();
    let path = std::env::temp_dir().join(format!("h2b_text_material_{}.brz", std::process::id()));
    std::fs::write(&path, data).unwrap();

    let db = Brz::open(&path).unwrap().into_reader();
    let mut found = 0;
    for chunk in db.brick_chunk_index(1).unwrap() {
        let (_soa, comps) = db.component_chunk_soa(1, chunk.index).unwrap();
        for c in comps {
            assert_eq!(
                enum_raw(c.prop("Material").unwrap()),
                TextMaterial::Glow.byte() as u64
            );
            assert_eq!(c.prop("MaterialSlider").unwrap().as_brdb_i32().unwrap(), 7);
            assert_eq!(c.prop("ScuffWidth").unwrap().as_brdb_f32().unwrap(), 0.25);
            assert_eq!(
                c.prop("GraffitiDepthLimit").unwrap().as_brdb_f32().unwrap(),
                12.0
            );
            assert_eq!(
                c.prop("GraffitiAngleLimit").unwrap().as_brdb_f32().unwrap(),
                60.0
            );
            assert_eq!(c.prop("GraffitiLayer").unwrap().as_brdb_i32().unwrap(), 3);
            assert_eq!(
                enum_raw(c.prop("Shading").unwrap()),
                TextShading::Chamfer.byte() as u64
            );
            assert_eq!(c.prop("ShadingWidth").unwrap().as_brdb_f32().unwrap(), 1.5);
            assert!(c.prop("bFlipShading").unwrap().as_brdb_bool().unwrap());
            found += 1;
        }
    }
    assert_eq!(found, 1, "expected exactly one TextDisplay component");
    std::fs::remove_file(&path).ok();
}
