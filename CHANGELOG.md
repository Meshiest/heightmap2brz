# Changelog

Release notes live here, newest first. `tools/release-notes.sh` reads this file
and `.github/workflows/release.yml` publishes the matching section as the GitHub
release body, so **the heading is the release title**: write it as
`## <version> - <theme>` before tagging, and `just tag` will refuse to tag a
version this file does not mention.

Entries up to 0.16.2 were backfilled from the GitHub releases they shipped as.

## 0.18.0 - Shorter Color Tags

- `--short-hex` forces every text-mode color tag into the three-digit
  `<color="FFF">` form by quantizing each channel to the nearest of 16 levels.
  Tags drop from 16 characters to 13, and neighbouring pixels that quantize to
  the same color merge into one run, so it saves space twice over, at the
  cost of some color fidelity. Available for both image and video text modes,
  on the CLI and in both GUI panels.
- The Monaspace Argon preset now defaults to `--width-scale 2.05` rather than
  2, closing the seam between glyph cells.

## 0.16.2 - Write Performance

Expect slight improvements to brz write performance

## 0.16.0 - QoL and MIDI Percussion

Added experimental support for midi percussion and automatic wave-instrument selection.

Also from https://github.com/Meshiest/heightmap2brz/pull/17 (thanks @Kmschr):

- Get rid of SRGB option, no longer need to do color conversion and makes no sense to keep it
- Show size of what you are converting
- Expand limits for height
- Show brickcount in save description
- Attach the heightmap as screenshot, or colormap if used
- Error when the save has too many chunks for Brickadia to be able to load

## 0.15.0 - Wedge Terrain

@F1shar contributed a wedge terrain feature: https://github.com/Meshiest/heightmap2brz/pull/16

## 0.14.0 - Ramp and Smooth Terrain

Thanks @Kmschr for [PR#15](https://github.com/Meshiest/heightmap2brz/pull/15)

Implements wrap's rampifier and @Kmschr's smooth terrain features.

## 0.13.0 - Brickadia Themed UI

Also adds a few more things:

- Cleaner midi circuit output
- Fixes midi speaker placement for in-chip
- Allows seeking in midi preview

## 0.12.0 - Midi

- Add support for midi2brick
- Move navigation away from tabs to a main menu
- Fix browser support for audio2brick and video2brick

## 0.11.1

- fixes pausing audio not actually pausing the components
- adds support for putting the speakers in the microchip

## 0.11.0 - Video and Audio

Adds video and audio support. Having ffmpeg installed locally makes it much faster

## 0.10.2 - More Tiny Text Fixes

More tiny text fixes.... Also on https://heightmap.brickadia.dev/

## 0.10.1 - Tiny text fixes

Also updated on https://heightmap.brickadia.dev/

Fixes tiny text rendering

## 0.10.0 - More text modes, Web Build

- You can now use this tool entirely online at: https://heightmap.brickadia.dev/
- The text mode now has support for a braille and black/white block text for higher resolution per-character images
- The image scaling now properly handles very tiny sizes
- This update also upgrades the calibrator to use variables in-game to configure a checkerboard pattern interactively!

## 0.9.0 - Img2Text

This update allows automatically generating brz files with images using the Text Renderer component!

## 0.8.1 - Smooth tiles

Added support for smooth tiles and disabling the always-on-top option!

## 0.8.0 - JPG and Greedy Optimization

Heightmap2brz now supports a greedy optimization and support for loading JPG images. A linux binary is also now available without clipboard support.

## 0.7.0 - BRZ and Clipboard features!

BRS support is removed. Can now generate brz/brdb files and save them to your clipboard!

## 0.6.1 - Clear Images and Interrupt Generation

You can now stop the generation partway through as well as remove images instead of clicking "select image"

Also the colormap button is blue!

## 0.6.0 - GUI QoL

Added some nice new quality of life features to the GUI:

- Image previews
- Progress bar
- Logs inside the GUI
- Run the generation in its own thread

## 0.5.0 - Glow flag, new save writer

Now using @voximity's [brickadia-rs](https://github.com/brickadia-community/brickadia-rs) instead of brickadia's brs for writing saves.

## 0.4.2 - Gui Slider Fixes

Fixes swapped slider values in the heightmap gui (thanks @mraware)

## 0.4.1 - Gui Fixes

Minor fixes with the gui for img2brick mode

## 0.4.0 - Gui

The long awaited heightmap2brs/img2brick gui application

## 0.3.8 - Linear RGB

Adds a flag `--lrgb` for those who want to use linear rgb colors

## 0.3.7 - Microbricks and More!

Added microbrick support via `--micro`
Added studded bricks support via `--stud`
Added fake owner insertion and transparency support
`--cull` will remove perfectly transparent bricks

## 0.3.6 - HD Heightmaps

Adds support for HD heightmaps exceeding the previous PNG8/Stacked heightmap limitation. Special thanks to @Kmschr for the feature and the [GeoTIFF2Heightmap](https://github.com/Kmschr/GeoTIFF2Heightmap) tool!

## 0.3.5 - img2brick ability

adds a new flag (`-i` or `--img`) to use the program as an img2brick. simply run `heightmap image.png -i` and render images as flat heightmaps!

## 0.3.4 - Color Update

Colors will no longer be washed out. heightmap2brs now converts sRGB to linear RGB on the fly!

## 0.3.3 - Resolution Update

By providing multiple heightmap input files, the generator will now add the colors together. See `example_maps/stacked_1.png` for an example stacked heightmap image

## 0.3.2 - Linear Optimization Improvements

Added even better linear optimization for bricks that are the same width/height and next to each other.

Also runs linear-optimization until no more bricks can be removed

## 0.3.1 - Hole Fixes

Fixed an issue that was creating holes in the terrain. No bricks overlap now.

## 0.3.0 - Optimization Update

Added snap and tile feature, fixed cull feature, added new optimized generator

Changed some args around so they make more sense

## 0.2.0 - Output fixes

Added output argument

## 0.0.1 - First release

`heightmap` is for unix systems

`heightmap.exe` is for windows
