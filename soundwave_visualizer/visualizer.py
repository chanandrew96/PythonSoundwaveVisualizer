"""
Core soundwave visualizer: FFT-based polar animation with configurable
background, center text/image, and export to video.
"""

import os
import subprocess
import tempfile
from typing import List, Optional, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties, fontManager
from scipy.fft import fft
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter as gaussian_filter_nd
import librosa

from . import defaults as D

try:
    from PIL import Image, ImageFilter
except ImportError:
    Image = None
    ImageFilter = None


def check_ffmpeg(ffmpeg_path: str = "ffmpeg") -> bool:
    """Return True if ffmpeg is available at the given path (or in PATH if path is 'ffmpeg')."""
    path = (ffmpeg_path or "ffmpeg").strip() or "ffmpeg"
    try:
        r = subprocess.run(
            [path, "-version"],
            capture_output=True,
            timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _resolve_font() -> FontProperties:
    """Return a font that supports UTF-8 / CJK."""
    for name in D.UTF8_FONT_NAMES:
        if any(name in f.name for f in fontManager.ttflist):
            return FontProperties(family=name)
    return FontProperties(family="sans-serif")


def _norm_img_to_float(img: np.ndarray) -> np.ndarray:
    """Normalize image to float [0, 1] for display."""
    if img.dtype == np.uint8 or (img.size > 0 and np.nanmax(img) > 1.0):
        img = img.astype(np.float32) / 255.0
    return np.clip(img.astype(np.float32), 0, 1)


class SoundwaveVisualizer:
    """
    Animated FFT-based polar soundwave visualizer with optional
    background image, center text/image, and video export.
    """

    def __init__(
        self,
        audio: Union[str, np.ndarray],
        sr: Optional[int] = None,
        *,
        video_width: int = D.VIDEO_WIDTH,
        video_height: int = D.VIDEO_HEIGHT,
        outer_radius: float = D.OUTER_RADIUS,
        dpi: int = D.DPI,
        soundwave_radius: float = D.SOUNDWAVE_RADIUS,
        soundwave_text: Union[str, List[str], None] = D.SOUNDWAVE_TEXT,
        soundwave_color: str = D.SOUNDWAVE_COLOR,
        soundwave_opacity: float = D.SOUNDWAVE_OPACITY,
        num_bars: int = D.NUM_BARS,
        chunk_size: int = D.CHUNK_SIZE,
        max_bar_length: Optional[float] = D.MAX_BAR_LENGTH,
        rise_speed: float = D.RISE_SPEED,
        fall_speed: float = D.FALL_SPEED,
        background_image: Union[str, None] = D.BACKGROUND_IMAGE,
        background_image_blur: float = D.BACKGROUND_IMAGE_BLUR,
        background_color: Union[str, None] = D.BACKGROUND_COLOR,
        thumbnail_image_blur: float = D.THUMBNAIL_IMAGE_BLUR,
        thumbnail_image_scale: float = D.THUMBNAIL_IMAGE_SCALE,
        thumbnail_shadow_margin: int = D.THUMBNAIL_SHADOW_MARGIN,
        thumbnail_shadow_blur: float = D.THUMBNAIL_SHADOW_BLUR,
        thumbnail_shadow_alpha: float = D.THUMBNAIL_SHADOW_ALPHA,
        fps: int = D.FPS,
        output_dir: str = "",
        output_filename: Optional[str] = None,
        ffmpeg_path: str = "ffmpeg",
    ):
        """
        Args:
            audio: Path to audio file (e.g. .mp3) or (samples, sample_rate) array.
            sr: Sample rate; required only when audio is an array.
            video_width, video_height: Canvas size in pixels.
            outer_radius: Radial limit for bars (data coordinates).
            soundwave_radius: Inner empty circle radius.
            soundwave_text: Center label (string, list of lines, or path to image).
            soundwave_color: Color of bars and text.
            soundwave_opacity: Opacity of soundwave bars (0–1).
            num_bars: Number of bars around the circle.
            chunk_size: Samples per frame for FFT.
            max_bar_length: Cap bar length (default: outer_radius - soundwave_radius).
            rise_speed, fall_speed: Smoothing (0–1) for bar animation.
            background_image: Path to background image or ""/None.
            background_image_blur: Blur radius for back layer.
            background_color: Fallback background color if no image.
            thumbnail_image_blur: Blur for front/thumbnail layer.
            thumbnail_image_scale: Front image scale 0–1.
            thumbnail_shadow_margin: Pixels around front image for soft shadow (0 = no shadow).
            thumbnail_shadow_blur: Blur radius for the shadow.
            thumbnail_shadow_alpha: Max opacity of shadow 0–1 (transparent so background shows).
            fps: Frames per second for preview and export.
            output_dir: Default directory for exported video (used when output_path not given).
            output_filename: Default output filename (e.g. "sound_wave.mp4"; used when output_path not given).
            ffmpeg_path: Path to ffmpeg executable, or "ffmpeg" to use system PATH.
        """
        if isinstance(audio, str):
            self._audio, self._sr = librosa.load(audio, sr=None)
            self._audio_source_path: Optional[str] = audio
        else:
            self._audio = np.asarray(audio)
            self._sr = int(sr) if sr is not None else 22050
            self._audio_source_path = None

        self._video_width = video_width
        self._video_height = video_height
        self._outer_radius = outer_radius
        self._dpi = dpi
        self._soundwave_radius = soundwave_radius
        self._soundwave_text = soundwave_text
        self._soundwave_color = soundwave_color
        self._soundwave_opacity = soundwave_opacity
        self._num_bars = num_bars
        self._chunk_size = chunk_size
        self._max_bar_length = max_bar_length if max_bar_length is not None else (outer_radius - soundwave_radius)
        self._rise_speed = rise_speed
        self._fall_speed = fall_speed
        self._background_image = (background_image or "").strip() or None
        self._background_image_blur = background_image_blur
        self._background_color = (background_color or "").strip() or None
        self._thumbnail_image_blur = thumbnail_image_blur
        self._thumbnail_image_scale = thumbnail_image_scale
        self._thumbnail_shadow_margin = max(0, int(thumbnail_shadow_margin))
        self._thumbnail_shadow_blur = max(0.0, float(thumbnail_shadow_blur))
        self._thumbnail_shadow_alpha = max(0.0, min(1.0, float(thumbnail_shadow_alpha)))
        self._thumbnail_extent_scale = 1.0  # set in _prepare_background when shadow is applied
        self._fps = fps
        self._output_dir = (output_dir or "").strip()
        self._output_filename = (output_filename or D.OUTPUT_FILE).strip() or D.OUTPUT_FILE
        self._ffmpeg_path = (ffmpeg_path or "ffmpeg").strip() or "ffmpeg"

        self._duration_sec = len(self._audio) / self._sr
        self._num_frames = max(1, int(round(self._duration_sec * fps)))

        self._has_bg_image = bool(self._background_image and os.path.isfile(self._background_image))
        self._has_bg_color = bool(self._background_color)
        self._use_transparent_bg = not self._has_bg_image and not self._has_bg_color

        self._background_image_back: Optional[np.ndarray] = None
        self._background_image_front: Optional[np.ndarray] = None
        self._center_image: Optional[np.ndarray] = None
        self._utf8_font = _resolve_font()

        self._prepare_background()
        self._prepare_center_content()
        matplotlib.rcParams["axes.unicode_minus"] = False

    def _prepare_background(self) -> None:
        if not self._has_bg_image:
            return
        path = self._background_image
        pw, ph = self._video_width, self._video_height

        if Image is not None:
            pil_img = Image.open(path).convert("RGB")
            w, h = pil_img.size
            scale = max(pw / w, ph / h)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            back_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            left, top = (new_w - pw) // 2, (new_h - ph) // 2
            back_img = back_img.crop((left, top, left + pw, top + ph))
            if self._background_image_blur > 0:
                back_img = back_img.filter(ImageFilter.GaussianBlur(radius=self._background_image_blur))
            self._background_image_back = np.array(back_img) / 255.0

            tw = max(1, int(round(pw * self._thumbnail_image_scale)))
            th = max(1, int(round(ph * self._thumbnail_image_scale)))
            crop_w, crop_h = min(w, pw), min(h, ph)
            if w > pw or h > ph:
                left = (w - crop_w) // 2
                top = (h - crop_h) // 2
                front_img = pil_img.crop((left, top, left + crop_w, top + crop_h))
                front_img = front_img.resize((tw, th), Image.Resampling.LANCZOS)
            else:
                front_img = pil_img.resize((tw, th), Image.Resampling.LANCZOS)
            if self._thumbnail_image_blur > 0:
                front_img = front_img.filter(ImageFilter.GaussianBlur(radius=self._thumbnail_image_blur))
            # Optional soft shadow: darker near the front image, fading and more blurred farther out
            if self._thumbnail_shadow_margin > 0 and self._thumbnail_shadow_blur > 0:
                m = self._thumbnail_shadow_margin
                blur = self._thumbnail_shadow_blur
                sw, sh = front_img.size
                H, W = sh + 2 * m, sw + 2 * m
                # Build gradient: distance from each pixel to the inner rect [m:m+sh, m:m+sw]
                y = np.arange(H, dtype=np.float32)[:, None]
                x = np.arange(W, dtype=np.float32)[None, :]
                # Distance from (x,y) to inner rectangle (0 inside, positive outside)
                dist_left = np.maximum(0, m - x)
                dist_right = np.maximum(0, (x - (m + sw - 1)))
                dist_top = np.maximum(0, m - y)
                dist_bottom = np.maximum(0, (y - (m + sh - 1)))
                # For outside pixels: min distance to an edge gives distance to boundary
                outside = (x < m) | (x >= m + sw) | (y < m) | (y >= m + sh)
                dist_to_edge = np.full((H, W), np.inf, dtype=np.float32)
                np.minimum(dist_to_edge, dist_left, out=dist_to_edge)
                np.minimum(dist_to_edge, dist_right, out=dist_to_edge)
                np.minimum(dist_to_edge, dist_top, out=dist_to_edge)
                np.minimum(dist_to_edge, dist_bottom, out=dist_to_edge)
                dist_to_edge[~outside] = 0
                dist_to_edge = np.minimum(dist_to_edge, m)
                # 漸進式陰影: inner edge dark (alpha max), smooth gradual fade to transparent at outer edge
                # Use power < 1 so more of the margin is visibly mid-tone (easier to see the gradient)
                t = np.clip(dist_to_edge / (m + 1e-6), 0, 1)
                alpha = (1.0 - t ** 0.7) * self._thumbnail_shadow_alpha
                alpha = np.clip(alpha, 0, 1)
                rgba = np.zeros((H, W, 4), dtype=np.float32)
                rgba[..., 3] = alpha
                # Light blur so the progressive gradient stays visible (漸進式)
                rgba[..., 3] = gaussian_filter_nd(rgba[..., 3], sigma=blur, mode="constant", cval=0)
                rgba = np.clip(rgba, 0, 1)
                canvas = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")
                # Paste front image in center (opaque)
                front_rgba = front_img.convert("RGBA")
                canvas.paste(front_rgba, (m, m), front_rgba)
                self._background_image_front = np.array(canvas) / 255.0
                self._thumbnail_extent_scale = 1.0 + (2 * m) / min(sw, sh)
            else:
                self._background_image_front = np.array(front_img) / 255.0
        else:
            img = plt.imread(path)
            img = _norm_img_to_float(img)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]
            self._background_image_back = img
            self._background_image_front = None

    def _prepare_center_content(self) -> None:
        t = self._soundwave_text
        if t is None:
            return
        if isinstance(t, str) and os.path.isfile(t):
            self._center_image = plt.imread(t)

    def _center_text_string(self) -> str:
        t = self._soundwave_text
        if t is None:
            return ""
        if isinstance(t, (list, tuple)):
            return "\n".join(str(line) for line in t)
        return str(t)

    def _draw_center_content(self, ax: plt.Axes) -> None:
        if self._center_image is not None:
            extent = 0.9 * self._soundwave_radius
            ax.imshow(
                self._center_image,
                extent=[-extent, extent, -extent, extent],
                zorder=10,
                aspect="equal",
            )
        else:
            text = self._center_text_string()
            if text:
                ax.text(
                    0, 0, text,
                    ha="center", va="center",
                    color=self._soundwave_color,
                    fontsize=12,
                    zorder=10,
                    fontproperties=self._utf8_font,
                )

    def _update(
        self,
        frame: int,
        ax: plt.Axes,
        transparent: bool,
        state: List[np.ndarray],
    ) -> None:
        ax.clear()
        R = self._outer_radius

        if self._background_image_back is not None:
            ax.imshow(
                self._background_image_back,
                extent=[-R, R, -R, R],
                zorder=0,
                aspect="equal",
                origin="upper",
            )
        if self._background_image_front is not None:
            r = R * self._thumbnail_image_scale * self._thumbnail_extent_scale
            ax.imshow(
                self._background_image_front,
                extent=[-r, r, -r, r],
                zorder=1,
                aspect="equal",
                origin="upper",
            )

        max_start = max(0, len(self._audio) - self._chunk_size)
        start = (frame * max_start) // max(1, self._num_frames - 1) if self._num_frames > 1 else 0
        chunk = self._audio[start : start + self._chunk_size]
        if len(chunk) < self._chunk_size:
            chunk = np.pad(chunk, (0, self._chunk_size - len(chunk)))

        fft_size = max(2 * self._num_bars, self._chunk_size)
        freq = np.abs(fft(chunk, n=fft_size))[: self._num_bars]
        theta = np.linspace(0, 2 * np.pi, self._num_bars, endpoint=False)
        raw_length = (R - self._soundwave_radius) * (freq / (np.max(freq) + 1e-8))
        target_length = np.minimum(raw_length, self._max_bar_length)

        if state:
            if frame == 0:
                state[0] = target_length.copy()
            smoothed = state[0]
            rise = target_length >= smoothed
            smoothed = np.where(
                rise,
                smoothed + self._rise_speed * (target_length - smoothed),
                smoothed + self._fall_speed * (target_length - smoothed),
            )
            smoothed = np.clip(smoothed, 0, self._max_bar_length)
            state[0] = smoothed
            bar_length = smoothed
        else:
            bar_length = target_length

        for i in range(self._num_bars):
            inner_x = self._soundwave_radius * np.cos(theta[i])
            inner_y = self._soundwave_radius * np.sin(theta[i])
            outer_x = (self._soundwave_radius + bar_length[i]) * np.cos(theta[i])
            outer_y = (self._soundwave_radius + bar_length[i]) * np.sin(theta[i])
            ax.plot(
                [inner_x, outer_x],
                [inner_y, outer_y],
                color=self._soundwave_color,
                alpha=self._soundwave_opacity,
            )

        self._draw_center_content(ax)
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_aspect("equal")
        if transparent or self._use_transparent_bg:
            ax.set_facecolor("none")
        elif self._has_bg_image:
            ax.set_facecolor("none")
        else:
            ax.set_facecolor(self._background_color or "none")
        ax.axis("off")

    def _resolve_output_path(
        self,
        output_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        output_filename: Optional[str] = None,
    ) -> str:
        """Return the path to use for export. output_path overrides dir + filename."""
        if output_path and str(output_path).strip():
            return str(output_path).strip()
        dir_ = (output_dir if output_dir is not None else self._output_dir) or ""
        name = (output_filename if output_filename is not None else self._output_filename) or D.OUTPUT_FILE
        return os.path.join(dir_, name) if dir_ else name

    def _mux_audio_into_video(self, video_path: str) -> None:
        """Mux original audio into the video file using ffmpeg. Overwrites video_path."""
        audio_path: Optional[str] = self._audio_source_path
        if audio_path is None:
            # Audio was from array: write a temporary WAV
            try:
                fd, audio_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                # librosa loads as float [-1, 1]; wavfile expects int16
                wav = (np.clip(self._audio, -1, 1) * 32767).astype(np.int16)
                wavfile.write(audio_path, self._sr, wav)
            except Exception:
                return
            try:
                self._mux_audio_into_video_impl(video_path, audio_path)
            finally:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            return
        self._mux_audio_into_video_impl(video_path, audio_path)

    def _mux_audio_into_video_impl(self, video_path: str, audio_path: str) -> None:
        """Run ffmpeg to mux audio into video. Overwrites video_path with result."""
        if not os.path.isfile(audio_path) or not os.path.isfile(video_path):
            return
        try:
            run = subprocess.run(
                [
                    self._ffmpeg_path,
                    "-y",
                    "-i", video_path,
                    "-i", audio_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    "-loglevel", "error",
                    video_path + ".tmp.mux.mp4",
                ],
                capture_output=True,
                text=True,
                timeout=3600,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return
        if run.returncode != 0:
            return
        try:
            os.replace(video_path + ".tmp.mux.mp4", video_path)
        except OSError:
            try:
                os.remove(video_path)
                os.rename(video_path + ".tmp.mux.mp4", video_path)
            except OSError:
                pass

    def create_animation(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        transparent: bool = False,
    ) -> FuncAnimation:
        """Build the animation (caller owns fig/ax)."""
        state = [np.zeros(self._num_bars)]
        return FuncAnimation(
            fig,
            self._update,
            fargs=(ax, transparent, state),
            frames=self._num_frames,
            interval=1000 / self._fps,
            repeat=True,
        )

    def preview(self) -> None:
        """Show the animation in a window."""
        figsize = (self._video_width / self._dpi, self._video_height / self._dpi)
        fig, ax = plt.subplots(figsize=figsize, dpi=self._dpi)
        fig.patch.set_facecolor("none" if self._use_transparent_bg else (self._background_color or "white"))
        ax.set_facecolor("none" if (self._has_bg_image or self._use_transparent_bg) else (self._background_color or "white"))
        anim = self.create_animation(fig, ax)
        plt.title("Close window to continue", color=self._soundwave_color)
        plt.tight_layout()
        plt.show()

    def export(
        self,
        output_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        output_filename: Optional[str] = None,
        transparent: Optional[bool] = None,
    ) -> str:
        """
        Render and save the animation to a video file (e.g. .mp4). Requires ffmpeg.
        Returns the path where the file was saved.
        """
        path = self._resolve_output_path(output_path, output_dir, output_filename)
        if transparent is None:
            transparent = self._use_transparent_bg
        figsize = (self._video_width / self._dpi, self._video_height / self._dpi)
        fig, ax = plt.subplots(figsize=figsize, dpi=self._dpi)
        if transparent:
            fig.patch.set_facecolor("none")
            fig.patch.set_alpha(0)
            ax.patch.set_facecolor("none")
            ax.patch.set_alpha(0)
        else:
            fig.patch.set_facecolor(self._background_color or "white")
            ax.set_facecolor("none" if self._has_bg_image else (self._background_color or "white"))
        ani = self.create_animation(fig, ax, transparent=transparent)
        # If custom ffmpeg path, prepend its directory to PATH so matplotlib's writer finds it
        old_path = os.environ.get("PATH", "")
        if os.sep in self._ffmpeg_path:
            ffmpeg_dir = os.path.dirname(os.path.abspath(self._ffmpeg_path))
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + old_path
        try:
            ani.save(path, writer="ffmpeg", fps=self._fps, savefig_kwargs={"transparent": transparent})
        finally:
            os.environ["PATH"] = old_path
        plt.close(fig)
        self._mux_audio_into_video(path)
        return path

    def run_preview_then_export(
        self,
        output_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        output_filename: Optional[str] = None,
        prompt: bool = True,
    ) -> None:
        """Show preview window, then optionally export to video (prompt in terminal)."""
        path = self._resolve_output_path(output_path, output_dir, output_filename)
        self.preview()
        if not prompt:
            self.export(output_path=path)
            print(f"Saved: {path}")
            return
        try:
            response = input("Export to video file? (y/n): ").strip().lower()
        except EOFError:
            response = "n"
        if response == "y":
            print(f"Exporting to {path} ...")
            self.export(output_path=path)
            print(f"Saved: {path}")
        else:
            print("Export skipped.")


def run_preview_and_export(
    audio_path: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    prompt: bool = True,
    **kwargs,
) -> None:
    """
    Convenience: load audio from path, preview, then optionally export.
    Use output_path for a full path, or output_dir + output_filename.
    Extra kwargs are passed to SoundwaveVisualizer.
    """
    v = SoundwaveVisualizer(audio_path, **kwargs)
    v.run_preview_then_export(
        output_path=output_path,
        output_dir=output_dir,
        output_filename=output_filename,
        prompt=prompt,
    )
