use super::{BitMask, QuadTree, greedy_mesh_binary_plane};
use crate::map::*;
use crate::util::*;
use brdb::{
    Brick, BrickSize, BrickType, Collision, Color, Position,
    assets::materials::{GLOW, PLASTIC},
};
use log::info;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

/// Generate a heightmap with brick conservation optimizations
pub fn gen_opt_heightmap<F: Fn(f32) -> bool>(
    heightmap: &dyn Heightmap,
    colormap: &dyn Colormap,
    options: GenOptions,
    progress_f: F,
) -> Result<Vec<Brick>, String> {
    // Use greedy mesh if requested
    if options.greedy {
        return gen_greedy_heightmap(heightmap, colormap, options, progress_f);
    }

    // Use quad tree optimization
    gen_quad_heightmap(heightmap, colormap, options, progress_f)
}

/// Generate a heightmap using quadtree optimization
pub fn gen_quad_heightmap<F: Fn(f32) -> bool>(
    heightmap: &dyn Heightmap,
    colormap: &dyn Colormap,
    options: GenOptions,
    progress_f: F,
) -> Result<Vec<Brick>, String> {
    macro_rules! progress {
        ($e:expr) => {
            if !progress_f($e) {
                return Err("Stopped by user".to_string());
            }
        };
    }
    progress!(0.0);

    info!("Building initial quadtree");
    let quadtree_build_start = Instant::now();
    let (width, height) = heightmap.size();
    let area = width * height;
    let mut quad = QuadTree::new(heightmap, colormap)?;
    let quadtree_build_duration = quadtree_build_start.elapsed();
    info!(
        "Built quadtree in {:.2}s",
        quadtree_build_duration.as_secs_f64()
    );
    progress!(0.2);

    let (prog_offset, prog_scale, quad_opt_duration) = if options.quadtree {
        info!("Optimizing quadtree");
        let quad_opt_start = Instant::now();
        let mut scale = 0;

        // loop until the bricks would be too wide or we stop optimizing bricks
        while 2_i32.pow(scale + 1) * (options.size as i32) < 500 {
            progress!(0.2 + 0.5 * (scale as f32 / (500.0 / (options.size as f32)).log2()));
            let count = quad.quad_optimize_level(scale);
            if count == 0 {
                break;
            } else {
                info!("  Removed {:?} {}x bricks", count, 2_i32.pow(scale));
            }
            scale += 1;
        }
        let quad_opt_duration = quad_opt_start.elapsed();
        info!(
            "Quadtree optimization in {:.2}s",
            quad_opt_duration.as_secs_f64()
        );
        progress!(0.7);

        (0.7, 0.25, quad_opt_duration)
    } else {
        (0.2, 0.75, std::time::Duration::ZERO)
    };

    info!("Optimizing linear");
    let linear_opt_start = Instant::now();
    let mut i = 0;
    loop {
        i += 1;

        let count = quad.line_optimize(options.size as u32);
        progress!(prog_offset + prog_scale * (i as f32 / 5.0).min(1.0));

        if count == 0 {
            break;
        }
        info!("  Removed {} bricks", count);
    }
    let linear_opt_duration = linear_opt_start.elapsed();
    info!(
        "Linear optimization in {:.2}s",
        linear_opt_duration.as_secs_f64()
    );

    progress!(0.95);

    let brick_convert_start = Instant::now();
    let bricks = quad.into_bricks(options, width, height);
    let brick_convert_duration = brick_convert_start.elapsed();
    let brick_count = bricks.len();
    info!(
        "Reduced {} to {} ({}%; -{} bricks)",
        area,
        brick_count,
        (100. - brick_count as f64 / area as f64 * 100.).floor(),
        area as i32 - brick_count as i32,
    );
    info!(
        "Converted to bricks in {:.2}s",
        brick_convert_duration.as_secs_f64()
    );

    let total_duration =
        quadtree_build_duration + quad_opt_duration + linear_opt_duration + brick_convert_duration;
    info!(
        "Total quadtree time: {:.2}s (build: {:.2}s, quad-opt: {:.2}s, linear-opt: {:.2}s, bricks: {:.2}s)",
        total_duration.as_secs_f64(),
        quadtree_build_duration.as_secs_f64(),
        quad_opt_duration.as_secs_f64(),
        linear_opt_duration.as_secs_f64(),
        brick_convert_duration.as_secs_f64()
    );

    progress!(1.0);
    Ok(bricks)
}

/// Generate a heightmap using greedy mesh optimization for each height level
pub fn gen_greedy_heightmap<F: Fn(f32) -> bool>(
    heightmap: &dyn Heightmap,
    colormap: &dyn Colormap,
    options: GenOptions,
    progress_f: F,
) -> Result<Vec<Brick>, String> {
    macro_rules! progress {
        ($e:expr) => {
            if !progress_f($e) {
                return Err("Stopped by user".to_string());
            }
        };
    }
    progress!(0.0);

    info!("Building greedy mesh planes");
    let (width, height) = heightmap.size();

    if colormap.size() != heightmap.size() {
        return Err("Heightmap and colormap must have same dimensions".to_string());
    }

    // Find all unique (height, color) combinations
    let mut height_color_pairs = std::collections::BTreeSet::new();
    for x in 0..width {
        for y in 0..height {
            let h = heightmap.at(x, y);
            let c = colormap.at(x, y);
            // Only add non-transparent pixels (or all if not culling)
            if !options.cull || (h > 0 && c[3] > 0) {
                height_color_pairs.insert((h, c));
            }
        }
    }

    let total_pairs = height_color_pairs.len();
    info!("Found {} unique (height, color) combinations", total_pairs);

    // Convert to Vec for processing
    let pairs_vec: Vec<_> = height_color_pairs.into_iter().collect();

    // Build all planes in a single pass over the image
    let plane_build_start = Instant::now();

    // Create a map from (h, color) to plane index
    let mut plane_map: HashMap<(u32, [u8; 4]), usize> = HashMap::with_capacity(pairs_vec.len());
    let mut all_planes: Vec<Vec<BitMask>> = Vec::with_capacity(pairs_vec.len());

    for (idx, &pair) in pairs_vec.iter().enumerate() {
        plane_map.insert(pair, idx);
        // Allocate BitMasks more efficiently - start with minimal capacity
        all_planes.push(Vec::with_capacity(width as usize));
        for _ in 0..width {
            all_planes[idx].push(BitMask::new());
        }
    }

    // Single pass over the image to populate all planes at once
    for x in 0..width {
        for y in 0..height {
            let h = heightmap.at(x, y);
            let c = colormap.at(x, y);

            if !options.cull || (h > 0 && c[3] > 0) {
                if let Some(&plane_idx) = plane_map.get(&(h, c)) {
                    all_planes[plane_idx][x as usize].set_bit(y);
                }
            }
        }
    }

    // Build planes_with_metadata from all_planes (consume all_planes to avoid clones)
    let planes_with_metadata: Vec<_> = all_planes
        .into_iter()
        .zip(pairs_vec.into_iter())
        .map(|(plane, (h, color))| (plane, h, color))
        .collect();

    let plane_build_duration = plane_build_start.elapsed();
    info!(
        "Built {} planes in {:.2}s",
        planes_with_metadata.len(),
        plane_build_duration.as_secs_f64()
    );

    progress!(0.4);

    // Run greedy mesh in parallel

    let brick_scale = if options.micro { 2 } else { 10 };
    let max_quad_size = 1000 / brick_scale;
    let greedy_mesh_start = Instant::now();
    let all_quads: Vec<_> = planes_with_metadata
        .into_par_iter()
        .flat_map(|(plane, h, color)| {
            let quads = greedy_mesh_binary_plane(plane, width, height, max_quad_size);
            quads
                .into_iter()
                .map(move |quad| (quad, h, color))
                .collect::<Vec<_>>()
        })
        .collect();
    let greedy_mesh_duration = greedy_mesh_start.elapsed();
    info!(
        "Greedy meshed {} quads in {:.2}s",
        all_quads.len(),
        greedy_mesh_duration.as_secs_f64()
    );

    progress!(0.7);

    // Convert quads to bricks sequentially
    let brick_build_start = Instant::now();
    let mut all_bricks = Vec::new();
    let total_quads = all_quads.len();

    // Calculate offsets to center the bricks
    let offset_x = -(width as i32 * options.size as i32);
    let offset_y = -(height as i32 * options.size as i32);

    for (idx, (quad, h, color)) in all_quads.into_iter().enumerate() {
        if idx % 1000 == 0 {
            progress!(0.7 + 0.25 * (idx as f32 / total_quads as f32));
        }
        let x = quad.x;
        let y = quad.y;
        let w = quad.w;
        let h_brick = quad.h;

        let mut z = (options.scale * h) as i32;
        let mut desired_height = (options.scale * 2) as i32;

        // Create vertical bricks if needed
        while desired_height > 0 {
            let brick_height = std::cmp::min(
                std::cmp::max(desired_height, if options.stud { 5 } else { 2 }),
                250,
            ) as u16;
            let brick_height = brick_height + brick_height % (if options.stud { 5 } else { 2 });

            all_bricks.push(Brick {
                asset: BrickType::Procedural {
                    asset: options.asset.clone(),
                    size: BrickSize::new(
                        w as u16 * options.size,
                        h_brick as u16 * options.size,
                        if options.img && options.micro {
                            options.size
                        } else {
                            brick_height
                        },
                    ),
                },
                position: Position::new(
                    x as i32 * options.size as i32 * 2 + w as i32 * options.size as i32 + offset_x,
                    y as i32 * options.size as i32 * 2
                        + h_brick as i32 * options.size as i32
                        + offset_y,
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
                    ..Default::default()
                },
                color: Color {
                    r: color[0],
                    g: color[1],
                    b: color[2],
                },
                owner_index: None,
                material_intensity: if options.glow { 0 } else { 5 },
                material: if options.glow { GLOW } else { PLASTIC },
                ..Default::default()
            });

            desired_height -= brick_height as i32;
            z -= brick_height as i32 * 2;
        }
    }
    let brick_build_duration = brick_build_start.elapsed();
    info!(
        "Converted to {} bricks in {:.2}s",
        all_bricks.len(),
        brick_build_duration.as_secs_f64()
    );

    let total_duration = plane_build_duration + greedy_mesh_duration + brick_build_duration;
    info!(
        "Total greedy mesh time: {:.2}s (planes: {:.2}s, mesh: {:.2}s, bricks: {:.2}s)",
        total_duration.as_secs_f64(),
        plane_build_duration.as_secs_f64(),
        greedy_mesh_duration.as_secs_f64(),
        brick_build_duration.as_secs_f64()
    );

    progress!(1.0);
    Ok(all_bricks)
}
