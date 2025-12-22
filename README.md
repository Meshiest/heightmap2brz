# Heightmap2BRZ

[Download here](https://github.com/Meshiest/heightmap2brz/releases)

![Example output](https://i.imgur.com/QdPLN09.png)
![GTAV Map](https://i.imgur.com/J9XpmT3.png)
![Gui](https://i.imgur.com/8v9MXnl.png)

### Compiling

You need [rust](https://www.rust-lang.org/).

Run `cargo build` for the CLI, `cargo build --bin heightmap_gui --features gui` for the gui.

### Usage

Compile or download from releases.

`heightmap.exe --help` for usage instructions:

    USAGE:
        heightmap.exe [FLAGS] [OPTIONS] <INPUT>...

    FLAGS:
            --cull         Automatically remove bottom level bricks and fully transparent bricks
            --glow         Make the heightmap glow at 0 intensity
            --greedy       Use greedy optimization
        -h, --help         Prints help information
            --hdmap        Using a high detail rgb color encoded heightmap
        -i, --img          Make the heightmap flat and render an image
            --lrgb         Use linear rgb input color instead of sRGB
            --micro        Render bricks as micro bricks
            --nocollide    Disable brick collision
            --snap         Snap bricks to the brick grid
            --stud         Render bricks as stud cubes
            --tile         Render bricks as tiles
        -V, --version      Prints version information

    OPTIONS:
        -c, --colormap <colormap>    Input colormap image (PNG/JPG)
        -o, --output <output>        Output file (BRDB, BRZ)
        -s, --size <size>            Brick stud size (default 1)
        -v, --vertical <vertical>    Vertical scale multiplier (default 1)

    ARGS:
        <INPUT>...    Input heightmap image files (PNG/JPG)

###  Examples

An example command for generating the GTA V map would be:

`heightmap example_maps/gta5_fixed2_height.png -c example_maps/gta5_fixed2_color.png -s 4 -v 20 --tile -o gta5.brz`

To use stacked heightmap for increased resolution, simply provide more input files. See the `stacked_N.png` files in the `example_maps` directory for example stacked heightmaps.

`heightmap ./example_maps/stacked_1.png ./example_maps/stacked_2.png ./example_maps/stacked_3.png ./example_maps/stacked_4.png --tile`

To generate HD heightmaps for the `--hdmap` flag, check out [Kmschr's GeoTIFF2Heightmap tool](https://github.com/Kmschr/GeoTIFF2Heightmap).
