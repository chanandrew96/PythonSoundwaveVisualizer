"""
Soundwave Visualizer: FFT-based polar audio visualization with preview and video export.
"""

from .visualizer import SoundwaveVisualizer, run_preview_and_export, check_ffmpeg

__all__ = ["SoundwaveVisualizer", "run_preview_and_export", "check_ffmpeg"]
__version__ = "0.1.0"
