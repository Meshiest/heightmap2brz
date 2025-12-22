use crate::map::*;
use crate::util::*;
use brdb::{
    Brick, BrickSize, BrickType, Collision, Color, Position,
    assets::materials::{GLOW, PLASTIC},
};
use std::{
    cmp::{max, min},
    collections::HashSet,
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
    pub fn new(heightmap: &dyn Heightmap, colormap: &dyn Colormap) -> Result<Self, String> {
        let (width, height) = heightmap.size();

        if colormap.size() != heightmap.size() {
            return Err("Heightmap and colormap must have same dimensions".to_string());
        }

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
                        .fold(HashSet::new(), |mut set, height| {
                            set.insert(height);
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
            width,
            height,
        })
    }

    fn index(&self, x: u32, y: u32) -> usize {
        (y + x * self.height) as usize
    }

    // optimize bricks with size (level+1)
    pub fn quad_optimize_level(&mut self, level: u32) -> usize {
        let mut count = 0;

        // step amounts
        let space = 2_u32.pow(level);
        let step_amt = space as usize * 2;

        for x in (0..self.width - space).step_by(step_amt) {
            for y in (0..self.height - space).step_by(step_amt) {
                // split vertically (left/right columns)
                let (left, right) = self
                    .tiles
                    .split_at_mut(((x + space) * self.height) as usize);

                // split the columns horizontally
                let (top_left, bottom_left) =
                    left.split_at_mut((y + space + x * self.height) as usize);
                let (top_right, bottom_right) = right.split_at_mut((y + space) as usize);

                // first of each slice is the target cell
                let top_left = &mut top_left[(y + x * self.height) as usize];
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
    fn merge_line(&mut self, start_i: usize, children: Vec<usize>) {
        // there is nothing to merge, return
        if children.is_empty() {
            return;
        }

        let mut new_neighbors = vec![];

        // determine direction of this merge
        let is_vertical = self.tiles[children[0]].center.0 == self.tiles[start_i].center.0;

        // determine the new size of the parent tile, make children point at the parent
        let new_size = children.iter().fold(0, |sum, &i| {
            let t = &mut self.tiles[i];
            // assign parent, extend parent's neighbors
            t.parent = Some(start_i);
            new_neighbors.push(t.neighbors.clone());

            // sum size depending on merge direction
            sum + if is_vertical { t.size.1 } else { t.size.0 }
        });

        let start = &mut self.tiles[start_i];

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
        let mut count = 0;
        for x in 0..self.width {
            for y in 0..self.height {
                let start_i = self.index(x, y);
                let start = &self.tiles[start_i];
                if start.parent.is_some() {
                    continue;
                }

                let shift = start.size;
                let mut sx = shift.0;
                let mut horiz_tiles = vec![];
                let mut sy = shift.1;
                let mut vert_tiles = vec![];

                // determine longest horizontal merge
                while x + sx < self.width {
                    let i = self.index(x + sx, y);
                    let t = &self.tiles[i];
                    if (sx + t.size.0) * tile_scale > 500 || !start.similar_line(t) {
                        break;
                    }
                    horiz_tiles.push(i);
                    sx += t.size.0;
                }

                // determine longest vertical merge
                while y + sy < self.height {
                    let i = self.index(x, y + sy);
                    let t = &self.tiles[i];
                    if (sy + t.size.1) * tile_scale > 500 || !start.similar_line(t) {
                        break;
                    }
                    vert_tiles.push(i);
                    sy += t.size.1;
                }

                count += max(horiz_tiles.len(), vert_tiles.len());

                // merge whichever is largest
                self.merge_line(
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
        // Calculate offsets to center the bricks
        let offset_x = -(width as i32 * options.size as i32);
        let offset_y = -(height as i32 * options.size as i32);

        self.tiles
            .iter()
            .flat_map(|t| {
                if t.parent.is_some() || options.cull && (t.height == 0 || t.color[3] == 0) {
                    return vec![];
                }

                let mut z = (options.scale * t.height) as i32;

                // determine the height of this brick (difference of self and smallest neighbor)
                let raw_height = max(
                    t.height as i32 - t.neighbors.iter().cloned().min().unwrap_or(0) as i32 + 1,
                    2,
                );
                let mut desired_height = max(raw_height * options.scale as i32 / 2, 2);

                // snap bricks to grid
                if options.snap {
                    z += 4 - z % 4;
                    desired_height += 4 - desired_height % 4;
                }

                let mut bricks = vec![];
                // until we've made enough bricks to fill the height
                // add a brick with a max height of 250
                while desired_height > 0 {
                    // pick height for this brick

                    let height =
                        min(max(desired_height, if options.stud { 5 } else { 2 }), 250) as u16;
                    let height = height + height % (if options.stud { 5 } else { 2 });

                    bricks.push(Brick {
                        asset: BrickType::Procedural {
                            asset: options.asset.clone(),
                            size: BrickSize::new(
                                t.size.0 as u16 * options.size,
                                t.size.1 as u16 * options.size, // if it's a microbrick image, just use the block size so it's cubes
                                if options.img && options.micro {
                                    options.size
                                } else {
                                    height
                                },
                            ),
                        },
                        position: Position::new(
                            (t.center.0 as i32 * 2 + t.size.0 as i32) * options.size as i32
                                + offset_x,
                            (t.center.1 as i32 * 2 + t.size.1 as i32) * options.size as i32
                                + offset_y,
                            options.base_height() - 5
                                + if options.img { 0 } else { z - height as i32 },
                        ),
                        collision: Collision {
                            player: !options.nocollide,
                            weapon: !options.nocollide,
                            interact: !options.nocollide,
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
                    desired_height -= height as i32;
                    z -= height as i32 * 2;
                }
                bricks
            })
            .collect()
    }
}
