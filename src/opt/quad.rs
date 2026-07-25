use crate::map::*;
use crate::util::*;
use brdb::{
    Brick, BrickSize, BrickType, Collision, Color, Position,
    assets::materials::{GLOW, PLASTIC},
};
use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
};

#[derive(Debug, Default)]
struct Tile {
    index: usize,
    center: (u32, u32),
    size: (u32, u32),
    color: [u8; 4],
    height: u32,
    neighbors: HashSet<u32>,
    parent: Option<usize>,
}

pub struct QuadTree {
    tiles: Box<[Tile]>,
    /// Additional tile layers, one per height above `gen_full_layers_above_height`.
    /// Each is a full grid of solid-fill tiles for that height band.
    height_layers: Vec<Box<[Tile]>>,
    /// Sorted list of the filtered heights used to build the layers.
    sorted_heights: Vec<u32>,
    /// Colors that appeared at height 0 (treated as water/lake).
    height_0_colors: HashSet<[u8; 4]>,
    /// Mapping from a kept height to the color used for its layer.
    filtered_heights: HashMap<u32, [u8; 4]>,
    width: u32,
    height: u32,
}

impl Tile {
    // determine if another tile is similar in all properties
    fn similar_quad(&self, other: &Self) -> bool {
        self.size == other.size
            && self.color == other.color
            && self.height == other.height
            && self.parent.is_none()
            && other.parent.is_none()
    }

    // determine if another tile is similar in all properties except potentially width or height as long as they are in a line
    fn similar_line(&self, other: &Self) -> bool {
        let is_vertical = self.center.0 == other.center.0;
        let is_horizontal = self.center.1 == other.center.1;

        (is_vertical && self.size.0 == other.size.0 || is_horizontal && self.size.1 == other.size.1)
            && self.color == other.color
            && self.height == other.height
            && self.parent.is_none()
            && other.parent.is_none()
    }

    // merge a few tiles with this one
    fn merge_quad(
        &mut self,
        top_right: &mut Self,
        bottom_left: &mut Self,
        bottom_right: &mut Self,
    ) {
        // update size
        self.size = (self.size.0 * 2, self.size.1 * 2);

        self.neighbors.extend(&top_right.neighbors);
        self.neighbors.extend(&bottom_left.neighbors);
        self.neighbors.extend(&bottom_right.neighbors);

        // update parents of merged nodes
        top_right.parent = Some(self.index);
        bottom_left.parent = Some(self.index);
        bottom_right.parent = Some(self.index);
    }
}

impl QuadTree {
    // create a heightmap grid from two images
    pub fn new(
        heightmap: &dyn Heightmap,
        colormap: &dyn Colormap,
        gen_full_layers_above_height: u32,
    ) -> Result<Self, String> {
        let (width, height) = heightmap.size();

        if colormap.size() != heightmap.size() {
            return Err("Heightmap and colormap must have same dimensions".to_string());
        }

        // First pass: collect all possible heights and their colors in the heightmap
        let mut all_heights = HashMap::new();
        let mut height_0_colors = HashSet::new();
        for x in 0..width {
            for y in 0..height {
                let h = heightmap.at(x, y);
                let color = colormap.at(x, y);
                if h == 0 {
                    height_0_colors.insert(color);
                }
                all_heights.insert(h, color);
            }
        }

        // Filter heights: keep all heights above gen_full_layers_above_height,
        // and only the highest height at or below gen_full_layers_above_height
        let filtered_heights: HashMap<u32, [u8; 4]> = if gen_full_layers_above_height > 0 {
            let mut heights_at_or_below: Vec<u32> = all_heights
                .keys()
                .cloned()
                .filter(|&h| h <= gen_full_layers_above_height)
                .collect();
            heights_at_or_below.sort();

            let mut result = HashMap::new();

            // Add all heights above the threshold
            for (&h, &color) in &all_heights {
                if h > gen_full_layers_above_height {
                    result.insert(h, color);
                }
            }

            // Add only the highest height at or below the threshold
            if let Some(&highest_at_or_below) = heights_at_or_below.last() {
                if let Some(&color) = all_heights.get(&highest_at_or_below) {
                    result.insert(highest_at_or_below, color);
                }
            }

            result
        } else {
            // If gen_full_layers_above_height is 0, keep all heights
            all_heights
        };

        if gen_full_layers_above_height > 0 && !filtered_heights.is_empty() {
            // Get minimum height from filtered_heights for capping
            let min_filtered_height = *filtered_heights.keys().min().unwrap();

            // Create a sorted vector of filtered heights for consistent ordering
            let mut sorted_heights: Vec<u32> = filtered_heights.keys().cloned().collect();
            sorted_heights.sort();

            // Create tiles vector for the first layer (capped heights)
            let mut first_layer_tiles = Vec::with_capacity((width * height) as usize);

            for x in 0..width as i32 {
                for y in 0..height as i32 {
                    let original_height = heightmap.at(x as u32, y as u32);
                    // For first layer: keep original height if it's <= min_filtered_height,
                    // otherwise cap it to min_filtered_height
                    let capped_height = if original_height > min_filtered_height {
                        min_filtered_height
                    } else {
                        original_height
                    };

                    first_layer_tiles.push(Tile {
                        index: (x + y * height as i32) as usize,
                        center: (x as u32, y as u32),
                        neighbors: vec![(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                            .into_iter()
                            .filter(|(x, y)| {
                                *x >= 0 && *x < width as i32 && *y >= 0 && *y < height as i32
                            })
                            .map(|(x, y)| heightmap.at(x as u32, y as u32))
                            .fold(HashSet::new(), |mut set, h| {
                                set.insert(h);
                                set
                            }),
                        size: (1, 1),
                        color: if capped_height == min_filtered_height {
                            filtered_heights[&min_filtered_height]
                        } else {
                            colormap.at(x as u32, y as u32)
                        },
                        height: capped_height,
                        parent: None,
                    })
                }
            }

            // Create additional layers for remaining heights
            let mut height_layers = Vec::new();
            for &layer_height in &sorted_heights[1..] {
                // Skip first height as it's already in main tiles
                let mut layer_tiles = Vec::with_capacity((width * height) as usize);
                let layer_color = filtered_heights[&layer_height];
                let is_lake_layer = height_0_colors.contains(&layer_color);

                for x in 0..width as i32 {
                    for y in 0..height as i32 {
                        let original_height = heightmap.at(x as u32, y as u32);
                        let pixel_color = colormap.at(x as u32, y as u32);

                        // Lake layers only fill where color and height both match the layer;
                        // land layers fill everywhere at or above the layer height.
                        let tile_height = if is_lake_layer {
                            if pixel_color == layer_color && original_height == layer_height {
                                layer_height
                            } else {
                                0
                            }
                        } else if original_height >= layer_height {
                            layer_height
                        } else {
                            0
                        };

                        layer_tiles.push(Tile {
                            index: (x + y * height as i32) as usize,
                            center: (x as u32, y as u32),
                            neighbors: vec![(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                                .into_iter()
                                .filter(|(x, y)| {
                                    *x >= 0 && *x < width as i32 && *y >= 0 && *y < height as i32
                                })
                                .map(|(x, y)| heightmap.at(x as u32, y as u32))
                                .fold(HashSet::new(), |mut set, h| {
                                    set.insert(h);
                                    set
                                }),
                            size: (1, 1),
                            color: layer_color,
                            height: tile_height,
                            parent: None,
                        })
                    }
                }
                height_layers.push(layer_tiles.into_boxed_slice());
            }

            Ok(QuadTree {
                tiles: first_layer_tiles.into_boxed_slice(),
                height_layers,
                sorted_heights,
                height_0_colors,
                filtered_heights,
                width,
                height,
            })
        } else {
            // Original behavior when gen_full_layers_above_height is 0
            let mut tiles = Vec::with_capacity((width * height) as usize);

            // add all the tiles to the heightmap
            for x in 0..width as i32 {
                for y in 0..height as i32 {
                    tiles.push(Tile {
                        index: (x + y * height as i32) as usize,
                        center: (x as u32, y as u32),
                        // store a set of the neighbor's heights with each tile
                        // they will be joined when the tiles merge
                        neighbors: vec![(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                            .into_iter()
                            .filter(|(x, y)| {
                                *x >= 0 && *x < width as i32 && *y >= 0 && *y < height as i32
                            })
                            .map(|(x, y)| heightmap.at(x as u32, y as u32))
                            .fold(HashSet::new(), |mut set, h| {
                                set.insert(h);
                                set
                            }),
                        size: (1, 1),
                        color: colormap.at(x as u32, y as u32),
                        height: heightmap.at(x as u32, y as u32),
                        parent: None,
                    })
                }
            }

            Ok(QuadTree {
                tiles: tiles.into_boxed_slice(),
                height_layers: Vec::new(),
                sorted_heights: Vec::new(),
                height_0_colors: HashSet::new(),
                filtered_heights: HashMap::new(),
                width,
                height,
            })
        }
    }

    // optimize bricks with size (level+1)
    pub fn quad_optimize_level(&mut self, level: u32) -> usize {
        let space = 2_u32.pow(level);
        let step_amt = space as usize * 2;

        let mut count =
            Self::quad_optimize_tiles(&mut self.tiles, self.width, self.height, space, step_amt);
        for layer in &mut self.height_layers {
            count += Self::quad_optimize_tiles(layer, self.width, self.height, space, step_amt);
        }
        count
    }

    fn quad_optimize_tiles(
        tiles: &mut [Tile],
        width: u32,
        height: u32,
        space: u32,
        step_amt: usize,
    ) -> usize {
        let mut count = 0;

        for x in (0..width - space).step_by(step_amt) {
            for y in (0..height - space).step_by(step_amt) {
                // split vertically (left/right columns)
                let (left, right) = tiles.split_at_mut(((x + space) * height) as usize);

                // split the columns horizontally
                let (top_left, bottom_left) =
                    left.split_at_mut((y + space + x * height) as usize);
                let (top_right, bottom_right) = right.split_at_mut((y + space) as usize);

                // first of each slice is the target cell
                let top_left = &mut top_left[(y + x * height) as usize];
                let bottom_left = &mut bottom_left[0];
                let top_right = &mut top_right[y as usize];
                let bottom_right = &mut bottom_right[0];

                // if these are not similar tiles, skip them
                if top_left.size.0 != space
                    || !top_left.similar_quad(top_right)
                    || !top_left.similar_quad(bottom_left)
                    || !top_left.similar_quad(bottom_right)
                {
                    continue;
                }

                count += 3;

                // merge the tiles into the first one
                top_left.merge_quad(top_right, bottom_left, bottom_right);
            }
        }

        count
    }

    // merge tiles that are arranged in a line
    fn merge_line(tiles: &mut [Tile], start_i: usize, children: Vec<usize>) {
        // there is nothing to merge, return
        if children.is_empty() {
            return;
        }

        let mut new_neighbors = vec![];

        // determine direction of this merge
        let is_vertical = tiles[children[0]].center.0 == tiles[start_i].center.0;

        // determine the new size of the parent tile, make children point at the parent
        let new_size = children.iter().fold(0, |sum, &i| {
            let t = &mut tiles[i];
            // assign parent, extend parent's neighbors
            t.parent = Some(start_i);
            new_neighbors.push(t.neighbors.clone());

            // sum size depending on merge direction
            sum + if is_vertical { t.size.1 } else { t.size.0 }
        });

        let start = &mut tiles[start_i];

        for n in new_neighbors {
            start.neighbors.extend(&n);
        }

        // add the size to its respective dimension
        if is_vertical {
            start.size.1 += new_size
        } else {
            start.size.0 += new_size
        }
    }

    // optimize by nearby bricks in line
    pub fn line_optimize(&mut self, tile_scale: u32) -> usize {
        let mut count =
            Self::line_optimize_tiles(&mut self.tiles, self.width, self.height, tile_scale);
        for layer in &mut self.height_layers {
            count += Self::line_optimize_tiles(layer, self.width, self.height, tile_scale);
        }
        count
    }

    fn line_optimize_tiles(
        tiles: &mut [Tile],
        width: u32,
        height: u32,
        tile_scale: u32,
    ) -> usize {
        let mut count = 0;
        for x in 0..width {
            for y in 0..height {
                let start_i = (y + x * height) as usize;
                let start = &tiles[start_i];
                if start.parent.is_some() {
                    continue;
                }

                let shift = start.size;
                let mut sx = shift.0;
                let mut horiz_tiles = vec![];
                let mut sy = shift.1;
                let mut vert_tiles = vec![];

                // determine longest horizontal merge
                while x + sx < width {
                    let i = (y + (x + sx) * height) as usize;
                    let t = &tiles[i];
                    if (sx + t.size.0) * tile_scale > 500 || !start.similar_line(t) {
                        break;
                    }
                    horiz_tiles.push(i);
                    sx += t.size.0;
                }

                // determine longest vertical merge
                while y + sy < height {
                    let i = (y + sy + x * height) as usize;
                    let t = &tiles[i];
                    if (sy + t.size.1) * tile_scale > 500 || !start.similar_line(t) {
                        break;
                    }
                    vert_tiles.push(i);
                    sy += t.size.1;
                }

                count += max(horiz_tiles.len(), vert_tiles.len());

                // merge whichever is largest
                Self::merge_line(
                    tiles,
                    start_i,
                    if horiz_tiles.len() > vert_tiles.len() {
                        horiz_tiles
                    } else {
                        vert_tiles
                    },
                );
            }
        }

        count
    }

    // convert quadtree state into bricks
    pub fn into_bricks(&self, options: GenOptions, width: u32, height: u32) -> Vec<Brick> {
        let mut all_bricks = Self::tiles_to_bricks(&self.tiles, &options, 0, width, height);

        // Each height layer stacks a solid fill for its band. The height
        // adjustment subtracts the height of the layer below so the fill only
        // spans the gap; lake layers reference the layer below the water.
        for (i, layer) in self.height_layers.iter().enumerate() {
            let height_adjustment = if self.sorted_heights.is_empty() {
                0
            } else if i < self.sorted_heights.len() {
                let current_height = self.sorted_heights[i];
                match self.filtered_heights.get(&current_height) {
                    Some(&color) if self.height_0_colors.contains(&color) => {
                        // lake/water layer: fill down to the layer below it
                        if i > 0 {
                            self.sorted_heights[i - 1]
                        } else {
                            0
                        }
                    }
                    _ => self.sorted_heights[i],
                }
            } else {
                self.sorted_heights[i]
            };
            all_bricks.extend(Self::tiles_to_bricks(
                layer,
                &options,
                height_adjustment,
                width,
                height,
            ));
        }

        all_bricks
    }

    fn tiles_to_bricks(
        tiles: &[Tile],
        options: &GenOptions,
        height_adjustment: u32,
        width: u32,
        height: u32,
    ) -> Vec<Brick> {
        // Calculate offsets to center the bricks
        let offset_x = -(width as i32 * options.size as i32);
        let offset_y = -(height as i32 * options.size as i32);

        // Fill layers (adjustment > 0) are lifted slightly so stacked layers
        // sit flush instead of z-fighting / floating.
        let pos_adjust: i32 = if height_adjustment == 0 { 0 } else { 4 };

        // Layered generation needs solid columns so the fill layers stack into
        // continuous terrain. Without it, keep the cheaper surface-shell model
        // that only spans the drop to the lowest neighbor.
        let layered = options.gen_full_layers_above_height > 0;

        tiles
            .iter()
            .flat_map(|t| {
                // skip merged tiles and (when culling) fully transparent tiles.
                // Layered mode keeps ground-level tiles, since a layer's fill
                // is what gives the terrain below it its body.
                if t.parent.is_some()
                    || options.cull && (t.color[3] == 0 || (!layered && t.height == 0))
                {
                    return vec![];
                }

                let mut z = (options.scale * t.height) as i32;

                let raw_height = if layered {
                    // solid column from the layer below up to this tile's height
                    max(t.height as i32 - height_adjustment as i32 + 1, 2)
                } else {
                    // difference of self and smallest neighbor
                    max(
                        t.height as i32
                            - t.neighbors.iter().cloned().min().unwrap_or(0) as i32
                            + 1,
                        2,
                    )
                };
                let mut desired_height = max(raw_height * options.scale as i32 / 2, 2);

                // snap bricks to grid
                if options.snap {
                    z += (4 - z % 4) % 4;
                    desired_height += (4 - desired_height % 4) % 4;
                }

                let mut bricks = vec![];
                // until we've made enough bricks to fill the height
                // add a brick with a max height of 250
                while desired_height > 0 {
                    // pick height for this brick
                    let height_u =
                        min(max(desired_height, if options.stud { 5 } else { 2 }), 250) as u16;
                    let height_u = height_u + height_u % (if options.stud { 5 } else { 2 });

                    bricks.push(Brick {
                        asset: BrickType::Procedural {
                            asset: options.asset.clone(),
                            size: BrickSize::new(
                                t.size.0 as u16 * options.size,
                                t.size.1 as u16 * options.size,
                                // if it's a microbrick image, just use the block size so it's cubes
                                if options.img && options.micro {
                                    options.size
                                } else {
                                    height_u.saturating_sub(pos_adjust as u16)
                                },
                            ),
                        },
                        position: Position::new(
                            (t.center.0 as i32 * 2 + t.size.0 as i32) * options.size as i32
                                + offset_x,
                            (t.center.1 as i32 * 2 + t.size.1 as i32) * options.size as i32
                                + offset_y,
                            options.base_height() - 5
                                + if options.img {
                                    0
                                } else {
                                    z - height_u as i32 + pos_adjust
                                },
                        ),
                        collision: Collision {
                            player: !options.nocollide,
                            weapon: !options.nocollide,
                            interact: !options.nocollide,
                            tool: !options.nocollide,
                            ..Default::default()
                        },
                        color: Color {
                            r: t.color[0],
                            g: t.color[1],
                            b: t.color[2],
                        },
                        owner_index: None,
                        material_intensity: 0,
                        material: if options.glow { GLOW } else { PLASTIC },
                        ..Default::default()
                    });

                    // update Z and remaining height
                    desired_height -= height_u as i32;
                    z -= height_u as i32 * 2;
                }
                bricks
            })
            .collect()
    }
}
