"""
Local entry point: run with your config below, or use the library/CLI after pip install.
"""

from soundwave_visualizer import SoundwaveVisualizer

# --- Your config (edit for local runs) ---
AUDIO_PATH = "audio.mp3"
OUTPUT_PATH = "sound_wave.mp4"  # full path, or use OUTPUT_DIR + OUTPUT_FILENAME
OUTPUT_DIR = ""                 # e.g. "./exports"
OUTPUT_FILENAME = "sound_wave.mp4"
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 800
SOUNDWAVE_RADIUS = 30
SOUNDWAVE_TEXT = "Soundwave測試"
SOUNDWAVE_COLOR = "white"
SOUNDWAVE_OPACITY = 0.5
BACKGROUND_IMAGE = "74767a9a-8e05-452e-bcf1-4854764c9e26.jpg"
THUMBNAIL_IMAGE_BLUR = 0
THUMBNAIL_IMAGE_SCALE = 0.5
BACKGROUND_IMAGE_BLUR = 50
BACKGROUND_COLOR = "black"
FPS = 30


def main() -> None:
    v = SoundwaveVisualizer(
        AUDIO_PATH,
        video_width=VIDEO_WIDTH,
        video_height=VIDEO_HEIGHT,
        soundwave_radius=SOUNDWAVE_RADIUS,
        soundwave_text=SOUNDWAVE_TEXT,
        soundwave_color=SOUNDWAVE_COLOR,
        soundwave_opacity=SOUNDWAVE_OPACITY,
        background_image=BACKGROUND_IMAGE or None,
        background_image_blur=BACKGROUND_IMAGE_BLUR,
        background_color=BACKGROUND_COLOR,
        thumbnail_image_blur=THUMBNAIL_IMAGE_BLUR,
        thumbnail_image_scale=THUMBNAIL_IMAGE_SCALE,
        fps=FPS,
        output_dir=OUTPUT_DIR or None,
        output_filename=OUTPUT_FILENAME or None,
    )
    v.run_preview_then_export(
        output_path=OUTPUT_PATH or None,
        output_dir=OUTPUT_DIR or None,
        output_filename=OUTPUT_FILENAME or None,
        prompt=True,
    )


if __name__ == "__main__":
    main()
