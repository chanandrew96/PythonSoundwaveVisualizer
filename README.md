# Soundwave Visualizer

FFT-based polar soundwave visualizer: preview in a window and export to video (e.g. MP4).

## Install

```bash
pip install soundwave-visualizer
```

Requires **ffmpeg** on your system for video export.

## Usage

### Command line

```bash
soundwave-visualizer audio.mp3 -o output.mp4
soundwave-visualizer --no-prompt   # export right after preview
soundwave-visualizer --text "My Title" --background-image bg.jpg
```

### Python API

```python
from soundwave_visualizer import SoundwaveVisualizer

v = SoundwaveVisualizer(
    "audio.mp3",
    soundwave_text="Song Title",
    video_width=800,
    video_height=800,
)
v.preview()                    # show window
v.export("sound_wave.mp4")     # save video
# or
v.run_preview_then_export("sound_wave.mp4", prompt=True)
```

### Options (constructor / CLI)

| Option | Description |
|--------|-------------|
| `audio` | Audio file path or (samples, sr) |
| `video_width`, `video_height` | Canvas size (px) |
| `soundwave_text` | Center label (string, list of lines, or image path) |
| `soundwave_color`, `soundwave_opacity` | Bar/style |
| `background_image` | Background image path |
| `background_image_blur` | Blur radius for back layer |
| `thumbnail_image_scale`, `thumbnail_image_blur` | Front image size and blur |
| `background_color` | Fallback if no image |
| `fps` | Frames per second |

Video length follows the input audio length.
