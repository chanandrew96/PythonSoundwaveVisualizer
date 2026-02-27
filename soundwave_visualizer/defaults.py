"""Default configuration for SoundwaveVisualizer."""

# Video / canvas
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 800
OUTER_RADIUS = 100
DPI = 100

# Soundwave
SOUNDWAVE_RADIUS = 30
SOUNDWAVE_TEXT = "Soundwave"
SOUNDWAVE_COLOR = "white"
SOUNDWAVE_OPACITY = 0.5
NUM_BARS = 120
CHUNK_SIZE = 512
MAX_BAR_LENGTH = None  # None => OUTER_RADIUS - SOUNDWAVE_RADIUS
RISE_SPEED = 0.85
FALL_SPEED = 0.55

# Background
BACKGROUND_IMAGE = ""
BACKGROUND_IMAGE_BLUR = 50
BACKGROUND_COLOR = "black"
THUMBNAIL_IMAGE_BLUR = 0
THUMBNAIL_IMAGE_SCALE = 0.5
THUMBNAIL_SHADOW_MARGIN = 3   # pixels for progressive shadow (more room = visible gradient)
THUMBNAIL_SHADOW_BLUR = 3      # light blur so gradient stays visible (漸進式)
THUMBNAIL_SHADOW_ALPHA = 0.1   # max opacity at inner edge so dark part is clear (0–1)

# Export
FPS = 30
OUTPUT_FILE = "sound_wave.mp4"

# UTF-8 font names (first available is used)
UTF8_FONT_NAMES = [
    "Microsoft JhengHei",
    "SimHei",
    "SimSun",
    "Meiryo",
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "WenQuanYi Micro Hei",
    "DejaVu Sans",
    "sans-serif",
]
