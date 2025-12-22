use std::path::PathBuf;

use crate::{
    map::{Colormap, ColormapPNG, Heightmap, HeightmapFlat, HeightmapPNG},
    util::{GenOptions, file_ext},
};

type MapPair = (Box<dyn Heightmap>, Box<dyn Colormap>);

pub fn maps_from_files(
    options: &GenOptions,
    heightmap_files: Vec<PathBuf>,
    colormap_file: Option<PathBuf>,
) -> Result<MapPair, String> {
    let heightmap_files: Vec<PathBuf> = heightmap_files.into_iter().collect();
    let first_heightmap = heightmap_files
        .first()
        .map(|s| s.to_owned())
        .unwrap_or_else(|| "".into());
    let colormap_file = colormap_file.unwrap_or(first_heightmap);

    // colormap file parsing
    let colormap = match file_ext(&colormap_file)
        .map(|s| s.to_lowercase())
        .as_deref()
    {
        Some("png") | Some("jpg") | Some("jpeg") => ColormapPNG::new(&colormap_file, options.lrgb)
            .map_err(|e| format!("Error reading colormap: {:?}", e))?,
        Some(ext) => {
            return Err(format!("Unsupported colormap format '{}'", ext));
        }
        None => {
            return Err(format!(
                "Missing colormap format for '{}'",
                colormap_file.display()
            ));
        }
    };

    // heightmap file parsing
    let heightmap: Box<dyn Heightmap> = if heightmap_files.iter().all(|f| {
        matches!(
            file_ext(f).map(|s| s.to_lowercase()).as_deref(),
            Some("png") | Some("jpg") | Some("jpeg")
        )
    }) {
        if options.img {
            Box::new(HeightmapFlat::new(colormap.size()).unwrap())
        } else {
            match HeightmapPNG::new(heightmap_files.iter().collect(), options.hdmap) {
                Ok(map) => Box::new(map),
                Err(error) => {
                    return Err(format!("Error reading heightmap: {:?}", error));
                }
            }
        }
    } else {
        return Err("Unsupported heightmap format".to_string());
    };

    Ok((heightmap, Box::new(colormap)))
}
