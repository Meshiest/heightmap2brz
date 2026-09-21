//! The quadtree optimizer: merge equal pixels into square tiles, then
//! extend the tiles into lines.
//!
//! The grid stores 4 bytes per pixel: the tile's width and height, with
//! width `0` marking a cell that was merged into another tile. Everything
//! else a merge decision needs is read where it already is:
//!
//!   - a tile's colour and height are the maps' pixel at its origin, and
//!     both maps stay in memory for the whole run
//!   - the brick height of a tile is its height less the smallest height
//!     around it, and that ring is walked once at emission
//!
//! A 4096x4096 map thus costs the grid 64 MB.

use crate::map::*;
use crate::util::*;
use brdb::{
    Brick, BrickSize, BrickType, Collision, Color, Position,
    assets::materials::{GLOW, PLASTIC},
};
use std::cmp::{max, min};

/// One cell of the grid. `w == 0` means the cell belongs to another tile.
///
/// `u16` holds every reachable size: the driver stops the quad passes once
/// `2^(level+1) * size` reaches 500 and the line pass stops a merge that
/// would pass `500 / size` pixels, and `size` is at least 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cell {
    w: u16,
    h: u16,
}

impl Cell {
    const MERGED: Cell = Cell { w: 0, h: 0 };

    #[inline]
    fn is_merged(self) -> bool {
        self.w == 0
    }
}

pub struct TileGrid<'a> {
    cells: Box<[Cell]>,
    width: u32,
    height: u32,
    heightmap: &'a dyn Heightmap,
    colormap: &'a dyn Colormap,
}

impl<'a> TileGrid<'a> {
    /// Build the grid over two maps of the same size. Every cell starts as
    /// its own 1x1 tile.
    pub fn new(heightmap: &'a dyn Heightmap, colormap: &'a dyn Colormap) -> Result<Self, String> {
        let (width, height) = heightmap.size();

        if colormap.size() != heightmap.size() {
            return Err("Heightmap and colormap must have same dimensions".to_string());
        }

        Ok(TileGrid {
            cells: vec![Cell { w: 1, h: 1 }; (width * height) as usize].into_boxed_slice(),
            width,
            height,
            heightmap,
            colormap,
        })
    }

    /// Cells are stored column-major, the order `into_bricks` walks, so the
    /// bricks of a save come out column by column.
    #[inline]
    fn index(&self, x: u32, y: u32) -> usize {
        (y + x * self.height) as usize
    }

    #[inline]
    fn cell(&self, x: u32, y: u32) -> Cell {
        self.cells[self.index(x, y)]
    }

    /// The pixel a tile at `(x, y)` carries: the maps' value at its origin.
    #[inline]
    fn pixel(&self, x: u32, y: u32) -> (u32, [u8; 4]) {
        (self.heightmap.at(x, y), self.colormap.at(x, y))
    }

    /// Can `b` join a quad with `a`: same size, same pixel, neither merged.
    #[inline]
    fn similar_quad(&self, a: Cell, pa: (u32, [u8; 4]), bx: u32, by: u32) -> bool {
        let b = self.cell(bx, by);
        !a.is_merged() && !b.is_merged() && a == b && pa == self.pixel(bx, by)
    }

    /// Merge aligned 2x2 blocks of `2^level`-sized square tiles into one.
    /// Returns the number of tiles removed, three per merge.
    pub fn quad_optimize_level(&mut self, level: u32) -> usize {
        let mut count = 0;

        let space = 2_u32.pow(level);
        let step_amt = space as usize * 2;

        // The driver never reaches a level whose tiles are wider than the
        // map: a merge at `level` needs `2^(level+1)` pixels of room.
        // `saturating_sub` makes a direct call a no-op instead of a panic.
        for x in (0..self.width.saturating_sub(space)).step_by(step_amt) {
            for y in (0..self.height.saturating_sub(space)).step_by(step_amt) {
                let top_left = self.cell(x, y);
                if top_left.w as u32 != space {
                    continue;
                }
                let p = self.pixel(x, y);
                if !self.similar_quad(top_left, p, x + space, y)
                    || !self.similar_quad(top_left, p, x, y + space)
                    || !self.similar_quad(top_left, p, x + space, y + space)
                {
                    continue;
                }

                count += 3;

                let i = self.index(x, y);
                self.cells[i] = Cell {
                    w: top_left.w * 2,
                    h: top_left.h * 2,
                };
                let i = self.index(x + space, y);
                self.cells[i] = Cell::MERGED;
                let i = self.index(x, y + space);
                self.cells[i] = Cell::MERGED;
                let i = self.index(x + space, y + space);
                self.cells[i] = Cell::MERGED;
            }
        }

        count
    }

    /// Extend each tile along whichever axis has the longer run of tiles
    /// that share its pixel and its cross-axis size. Returns the number of
    /// tiles removed.
    pub fn line_optimize(&mut self, tile_scale: u32) -> usize {
        let mut count = 0;
        for x in 0..self.width {
            for y in 0..self.height {
                let start = self.cell(x, y);
                if start.is_merged() {
                    continue;
                }
                let p = self.pixel(x, y);

                // longest horizontal run: tiles to the right with the same
                // height (size.1) and pixel
                let mut sx = start.w as u32;
                let mut horiz = 0usize;
                while x + sx < self.width {
                    let t = self.cell(x + sx, y);
                    if (sx + t.w as u32) * tile_scale > 500
                        || t.is_merged()
                        || t.h != start.h
                        || p != self.pixel(x + sx, y)
                    {
                        break;
                    }
                    horiz += 1;
                    sx += t.w as u32;
                }

                // longest vertical run: tiles below with the same width
                // (size.0) and pixel
                let mut sy = start.h as u32;
                let mut vert = 0usize;
                while y + sy < self.height {
                    let t = self.cell(x, y + sy);
                    if (sy + t.h as u32) * tile_scale > 500
                        || t.is_merged()
                        || t.w != start.w
                        || p != self.pixel(x, y + sy)
                    {
                        break;
                    }
                    vert += 1;
                    sy += t.h as u32;
                }

                count += max(horiz, vert);

                // merge whichever is longest; a tie goes vertical
                if horiz > vert {
                    if horiz > 0 {
                        // Every cell of row `y` between the old and new right
                        // edge is a child's origin or already inside a child.
                        for cx in x + start.w as u32..x + sx {
                            let i = self.index(cx, y);
                            self.cells[i] = Cell::MERGED;
                        }
                        let i = self.index(x, y);
                        self.cells[i].w = sx as u16;
                    }
                } else if vert > 0 {
                    for cy in y + start.h as u32..y + sy {
                        let i = self.index(x, cy);
                        self.cells[i] = Cell::MERGED;
                    }
                    let i = self.index(x, y);
                    self.cells[i].h = sy as u16;
                }
            }
        }

        count
    }

    /// The smallest height in the 4-neighborhood of the tile at `(x, y)`,
    /// or `None` for a 1x1 tile with no in-bounds neighbor.
    ///
    /// The brick height comes from the lowest neighbor of any pixel of the
    /// tile. For a rectangle, that set is its outer ring plus, once the tile
    /// has two or more pixels, its own height (each pixel then borders
    /// another pixel of the tile). The last part only matters for a tile
    /// that fills the whole map, where the ring is empty.
    fn min_neighbor(&self, x: u32, y: u32, c: Cell, own: u32) -> Option<u32> {
        let (w, h) = (c.w as u32, c.h as u32);
        let mut lowest: Option<u32> = if w * h > 1 { Some(own) } else { None };
        let mut see = |v: u32| lowest = Some(lowest.map_or(v, |m| min(m, v)));

        if y > 0 {
            for cx in x..x + w {
                see(self.heightmap.at(cx, y - 1));
            }
        }
        if y + h < self.height {
            for cx in x..x + w {
                see(self.heightmap.at(cx, y + h));
            }
        }
        if x > 0 {
            for cy in y..y + h {
                see(self.heightmap.at(x - 1, cy));
            }
        }
        if x + w < self.width {
            for cy in y..y + h {
                see(self.heightmap.at(x + w, cy));
            }
        }

        lowest
    }

    /// Convert the grid into bricks. One tile becomes one brick, or a stack
    /// when the column is taller than one brick can be.
    pub fn into_bricks(&self, options: GenOptions, width: u32, height: u32) -> Vec<Brick> {
        // Calculate offsets to center the bricks
        let offset_x = -(width as i32 * options.size as i32);
        let offset_y = -(height as i32 * options.size as i32);

        let mut bricks = vec![];

        for x in 0..self.width {
            for y in 0..self.height {
                let c = self.cell(x, y);
                if c.is_merged() {
                    continue;
                }
                let (t_height, t_color) = self.pixel(x, y);
                if options.cull.is_on() && (t_height == 0 || t_color[3] == 0) {
                    continue;
                }

                let mut z = (options.scale * t_height) as i32;

                // determine the height of this brick (difference of self and smallest neighbor)
                let raw_height = max(
                    t_height as i32 - self.min_neighbor(x, y, c, t_height).unwrap_or(0) as i32 + 1,
                    2,
                );
                let mut desired_height = max(raw_height * options.scale as i32 / 2, 2);

                // snap bricks to grid
                if options.snap {
                    z += 4 - z % 4;
                    desired_height += 4 - desired_height % 4;
                }

                // until we've made enough bricks to fill the height
                // add a brick with a max height of 250
                while desired_height > 0 {
                    // pick height for this brick
                    let brick_height =
                        min(max(desired_height, if options.stud { 5 } else { 2 }), 250) as u16;
                    let brick_height =
                        brick_height + brick_height % (if options.stud { 5 } else { 2 });

                    bricks.push(Brick {
                        asset: BrickType::Procedural {
                            asset: options.asset.clone(),
                            size: BrickSize::new(
                                c.w * options.size,
                                c.h * options.size, // if it's a microbrick image, just use the block size so it's cubes
                                if options.img && options.micro {
                                    options.size
                                } else {
                                    brick_height
                                },
                            ),
                        },
                        position: Position::new(
                            (x as i32 * 2 + c.w as i32) * options.size as i32 + offset_x,
                            (y as i32 * 2 + c.h as i32) * options.size as i32 + offset_y,
                            options.base_height() - 5
                                + if options.img {
                                    0
                                } else {
                                    z - brick_height as i32
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
                            r: t_color[0],
                            g: t_color[1],
                            b: t_color[2],
                        },
                        owner_index: None,
                        material_intensity: 0,
                        material: if options.glow { GLOW } else { PLASTIC },
                        ..Default::default()
                    });

                    // update Z and remaining height
                    desired_height -= brick_height as i32;
                    z -= brick_height as i32 * 2;
                }
            }
        }

        bricks
    }
}
