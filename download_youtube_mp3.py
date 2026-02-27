"""
Download audio from a YouTube video and save as MP3.
Requires: yt-dlp, ffmpeg (on PATH).
"""

import argparse
import os
import sys

try:
    import yt_dlp
except ImportError:
    print("Missing dependency. Install with: pip install yt-dlp")
    sys.exit(1)


def download_youtube_mp3(
    url: str,
    output_dir: str = ".",
    output_template: str = "%(title)s.%(ext)s",
) -> str | None:
    """
    Download audio from a YouTube URL and save as MP3.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the MP3 file.
        output_template: yt-dlp template for filename (default: title.ext).

    Returns:
        Path to the saved MP3 file, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_template)

    opts = {
        "format": "bestaudio/best",
        "outtmpl": out_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": False,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                return None
            # Actual file written by postprocessor has .mp3 extension
            base = os.path.splitext(ydl.prepare_filename(info))[0]
            mp3_path = base + ".mp3"
            return mp3_path if os.path.isfile(mp3_path) else None
    except Exception as e:
        print(f"Download error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube video audio as MP3."
    )
    parser.add_argument(
        "url",
        help="YouTube video URL (e.g. https://www.youtube.com/watch?v=...)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Output directory for the MP3 file (default: current directory)",
    )
    parser.add_argument(
        "-f",
        "--filename",
        default=None,
        help="Output filename (without .mp3). Default: video title",
    )
    args = parser.parse_args()

    template = f"{args.filename}.%(ext)s" if args.filename else "%(title)s.%(ext)s"
    result = download_youtube_mp3(
        url=args.url,
        output_dir=args.output_dir,
        output_template=template,
    )
    if result:
        print(f"Saved: {result}")
    else:
        print("Download failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
