"""
CLI entry point for soundwave-visualizer (pip install then run: soundwave-visualizer).
"""

import argparse
from . import SoundwaveVisualizer
from . import defaults as D


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Soundwave visualizer: preview and export FFT-based polar animation from audio."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default="audio.mp3",
        help="Path to audio file (default: audio.mp3)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output video path (full path; overrides --output-dir and --output-filename)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (used with --output-filename when -o is not set)",
    )
    parser.add_argument(
        "--output-filename",
        default=None,
        help=f"Output filename (e.g. my_video.mp4; default: {D.OUTPUT_FILE})",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Export immediately after preview without asking",
    )
    parser.add_argument("--video-width", type=int, default=D.VIDEO_WIDTH, help="Canvas width (px)")
    parser.add_argument("--video-height", type=int, default=D.VIDEO_HEIGHT, help="Canvas height (px)")
    parser.add_argument("--text", default=D.SOUNDWAVE_TEXT, help="Center text or path to image")
    parser.add_argument("--background-image", default=D.BACKGROUND_IMAGE, help="Background image path")
    parser.add_argument("--background-color", default=D.BACKGROUND_COLOR, help="Background color if no image")
    parser.add_argument("--fps", type=int, default=D.FPS, help="Frames per second")
    args = parser.parse_args()

    v = SoundwaveVisualizer(
        args.audio,
        video_width=args.video_width,
        video_height=args.video_height,
        soundwave_text=args.text,
        background_image=args.background_image or None,
        background_color=args.background_color or None,
        fps=args.fps,
        output_dir=(args.output_dir or "").strip() or None,
        output_filename=(args.output_filename or "").strip() or None,
    )
    v.run_preview_then_export(
        output_path=args.output,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        prompt=not args.no_prompt,
    )


if __name__ == "__main__":
    main()
