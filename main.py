"""
Soundwave Visualizer GUI: configure MP3, background image, soundwave text;
add tasks and export multiple videos in bulk.
"""

import os
import shutil
import subprocess
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Any, Dict, List, Optional

from soundwave_visualizer import SoundwaveVisualizer, check_ffmpeg
from soundwave_visualizer import defaults as D

# Default ffmpeg: use "ffmpeg" so it is looked up in system PATH
DEFAULT_FFMPEG_PATH = "ffmpeg"


def _str_or_none(s: str) -> Optional[str]:
    t = (s or "").strip()
    return t if t else None


def _float_or_default(s: str, default: float) -> float:
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return default


def _int_or_default(s: str, default: int) -> int:
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return default


def _auto_ffmpeg_path() -> Optional[str]:
    """Try to locate ffmpeg using Python and OS tools.

    Priority:
    1) shutil.which("ffmpeg")
    2) On Windows, `where ffmpeg`
    """
    # 1) Standard lookup via PATH
    p = shutil.which("ffmpeg")
    if p:
        return p
    # 2) Windows: use `where` as a fallback
    if os.name == "nt":
        try:
            run = subprocess.run(
                ["where", "ffmpeg"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if run.returncode == 0 and run.stdout:
                first = run.stdout.splitlines()[0].strip()
                if first:
                    return first
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
    return None


class SoundwaveGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Soundwave Visualizer")
        self.root.minsize(700, 580)
        self.root.geometry("640x620")

        self.tasks: List[Dict[str, Any]] = []
        self._advanced_visible = False
        self._tree_tooltip = None
        self._tree_tooltip_text = ""
        self._tree_font = None
        self._build_ui()
        self.root.after(100, self._check_ffmpeg_on_start)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Main inputs ---
        main_f = ttk.LabelFrame(main, text="Source & layout", padding=8)
        main_f.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(main_f, text="MP3 path:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.var_audio = tk.StringVar(value="audio.mp3")
        e_audio = ttk.Entry(main_f, textvariable=self.var_audio, width=50)
        e_audio.grid(row=0, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Button(main_f, text="Browse…", command=self._browse_audio).grid(row=0, column=2, pady=2)
        main_f.columnconfigure(1, weight=1)

        ttk.Label(main_f, text="Background image:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.var_bg_image = tk.StringVar(value="")
        e_bg = ttk.Entry(main_f, textvariable=self.var_bg_image, width=50)
        e_bg.grid(row=1, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Button(main_f, text="Browse…", command=self._browse_bg_image).grid(row=1, column=2, pady=2)

        ttk.Label(main_f, text="Soundwave text:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.var_text = tk.StringVar(value=D.SOUNDWAVE_TEXT)
        e_text = ttk.Entry(main_f, textvariable=self.var_text, width=50)
        e_text.grid(row=2, column=1, sticky=tk.EW, padx=4, pady=2)

        ttk.Label(main_f, text="FFmpeg path:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.var_ffmpeg = tk.StringVar(value=DEFAULT_FFMPEG_PATH)
        e_ffmpeg = ttk.Entry(main_f, textvariable=self.var_ffmpeg, width=50)
        e_ffmpeg.grid(row=3, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Button(main_f, text="Browse…", command=self._browse_ffmpeg).grid(row=3, column=2, pady=2)

        ttk.Label(main_f, text="Video width (px):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.var_video_width = tk.StringVar(value=str(D.VIDEO_WIDTH))
        ttk.Entry(main_f, textvariable=self.var_video_width, width=12).grid(
            row=4, column=1, sticky=tk.W, padx=4, pady=2
        )
        ttk.Label(main_f, text="Video height (px):").grid(row=4, column=2, sticky=tk.W, padx=(12, 0), pady=2)
        self.var_video_height = tk.StringVar(value=str(D.VIDEO_HEIGHT))
        ttk.Entry(main_f, textvariable=self.var_video_height, width=12).grid(
            row=4, column=3, sticky=tk.W, padx=4, pady=2
        )

        ttk.Label(main_f, text="Output path:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.var_output_dir = tk.StringVar(value="")
        e_out_dir = ttk.Entry(main_f, textvariable=self.var_output_dir, width=50)
        e_out_dir.grid(row=5, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Button(main_f, text="Browse…", command=self._browse_output_dir).grid(row=5, column=2, pady=2)

        ttk.Label(main_f, text="Output filename:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.var_output_filename = tk.StringVar(value=D.OUTPUT_FILE)
        ttk.Entry(main_f, textvariable=self.var_output_filename, width=50).grid(
            row=6, column=1, sticky=tk.EW, padx=4, pady=2
        )

        # Try to auto-detect ffmpeg on this system and pre-fill the field if found
        detected_ffmpeg = _auto_ffmpeg_path()
        if detected_ffmpeg:
            self.var_ffmpeg.set(detected_ffmpeg)

        # --- Expandable: All parameters ---
        self.advanced_frame = ttk.LabelFrame(main, text="All parameters", padding=6)
        self.advanced_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 6))
        self._advanced_inner: Optional[ttk.Frame] = None
        self._advanced_vars: Dict[str, tk.Variable] = {}

        def toggle_advanced() -> None:
            self._advanced_visible = not self._advanced_visible
            if self._advanced_inner is not None:
                if self._advanced_visible:
                    self._advanced_inner.pack(fill=tk.BOTH, expand=True)
                else:
                    self._advanced_inner.pack_forget()

        ttk.Button(
            self.advanced_frame,
            text="Show / hide all parameters",
            command=toggle_advanced,
        ).pack(anchor=tk.W, pady=(0, 4))

        inner = ttk.Frame(self.advanced_frame)
        self._advanced_inner = inner
        # Build parameter rows (label + entry) in two columns
        params = [
            ("soundwave_radius", "Soundwave radius", str(D.SOUNDWAVE_RADIUS)),
            ("soundwave_color", "Soundwave color", D.SOUNDWAVE_COLOR),
            ("soundwave_opacity", "Soundwave opacity (0–1)", str(D.SOUNDWAVE_OPACITY)),
            ("background_image_blur", "Background image blur", str(D.BACKGROUND_IMAGE_BLUR)),
            ("background_color", "Background color", D.BACKGROUND_COLOR),
            ("thumbnail_image_scale", "Thumbnail scale (0–1)", str(D.THUMBNAIL_IMAGE_SCALE)),
            ("thumbnail_image_blur", "Thumbnail blur", str(D.THUMBNAIL_IMAGE_BLUR)),
            ("thumbnail_shadow_margin", "Shadow margin", str(D.THUMBNAIL_SHADOW_MARGIN)),
            ("thumbnail_shadow_blur", "Shadow blur", str(D.THUMBNAIL_SHADOW_BLUR)),
            ("thumbnail_shadow_alpha", "Shadow alpha (0–1)", str(D.THUMBNAIL_SHADOW_ALPHA)),
            ("fps", "FPS", str(D.FPS)),
        ]
        for i, (key, label, default) in enumerate(params):
            r = i // 2
            c = (i % 2) * 2
            ttk.Label(inner, text=label + ":").grid(row=r, column=c, sticky=tk.W, padx=(0, 4), pady=2)
            v = tk.StringVar(value=default)
            self._advanced_vars[key] = v
            ttk.Entry(inner, textvariable=v, width=18).grid(row=r, column=c + 1, sticky=tk.W, pady=2)
        inner.columnconfigure(1, weight=1)
        inner.columnconfigure(3, weight=1)

        # --- Task list ---
        task_f = ttk.LabelFrame(main, text="Task list (bulk export)", padding=6)
        task_f.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        # Buttons above the list
        btn_top = ttk.Frame(task_f)
        btn_top.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(btn_top, text="Add current to task list", command=self._add_task).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_top, text="Remove selected", command=self._remove_selected).pack(side=tk.LEFT, padx=(0, 6))

        # The task list table (Status: pending / exporting / exported / error)
        cols = ("Audio", "Background", "Text", "Status", "Output path")
        self.tree = ttk.Treeview(task_f, columns=cols, show="headings", height=5)
        for c in cols:
            self.tree.heading(c, text=c)
            w = 200 if c == "Output path" else (90 if c == "Status" else 120)
            self.tree.column(c, width=w)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(task_f, orient=tk.VERTICAL, command=self.tree.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=sb.set)

        # Tooltip for overflowing output path
        self.tree.bind("<Motion>", self._on_tree_motion)
        self.tree.bind("<Leave>", self._hide_tree_tooltip)

        # --- Preview selected & Export ---
        preview_export_f = ttk.LabelFrame(main, text="Preview selected & Export", padding=6)
        preview_export_f.pack(fill=tk.X, expand=True, pady=(0, 6))
        
        ttk.Button(preview_export_f, text="Preview selected", command=self._preview_selected).pack(fill=tk.X, anchor=tk.W, side=tk.LEFT, expand=True)
        ttk.Button(preview_export_f, text="Export", command=self._export_all).pack(fill=tk.X, anchor=tk.W, side=tk.RIGHT, expand=True)

    def _browse_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio", "*.mp3 *.wav *.m4a"), ("All", "*.*")],
        )
        if path:
            self.var_audio.set(path)

    def _browse_bg_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select background image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")],
        )
        if path:
            self.var_bg_image.set(path)

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.var_output_dir.set(path)

    def _browse_ffmpeg(self) -> None:
        path = filedialog.askopenfilename(
            title="Select ffmpeg executable",
            filetypes=[("Executable (ffmpeg.exe)", "ffmpeg.exe"), ("All files", "*.*")],
        )
        if path:
            self.var_ffmpeg.set(path)

    def _check_ffmpeg_on_start(self) -> None:
        path = (self.var_ffmpeg.get() or "").strip() or DEFAULT_FFMPEG_PATH
        if not check_ffmpeg(path):
            messagebox.showwarning(
                "FFmpeg not found",
                "FFmpeg was not found at the given path (or in system PATH).\n\n"
                "Video export requires FFmpeg. Please install it and set the path above, "
                "or download from https://ffmpeg.org/download.html",
            )

    def _get_ffmpeg_path(self) -> str:
        return (self.var_ffmpeg.get() or "").strip() or DEFAULT_FFMPEG_PATH

    def _get_current_config(self) -> Dict[str, Any]:
        c: Dict[str, Any] = {
            "audio": self.var_audio.get().strip() or "audio.mp3",
            "background_image": _str_or_none(self.var_bg_image.get()),
            "soundwave_text": self.var_text.get().strip() or D.SOUNDWAVE_TEXT,
            "ffmpeg_path": self._get_ffmpeg_path(),
        }
        v = self._advanced_vars
        c["video_width"] = _int_or_default(self.var_video_width.get(), D.VIDEO_WIDTH)
        c["video_height"] = _int_or_default(self.var_video_height.get(), D.VIDEO_HEIGHT)
        c["output_dir"] = _str_or_none(self.var_output_dir.get())
        c["output_filename"] = self.var_output_filename.get().strip() or D.OUTPUT_FILE
        c["soundwave_radius"] = _float_or_default(v["soundwave_radius"].get(), D.SOUNDWAVE_RADIUS)
        c["soundwave_color"] = v["soundwave_color"].get().strip() or D.SOUNDWAVE_COLOR
        c["soundwave_opacity"] = _float_or_default(v["soundwave_opacity"].get(), D.SOUNDWAVE_OPACITY)
        c["background_image_blur"] = _float_or_default(v["background_image_blur"].get(), D.BACKGROUND_IMAGE_BLUR)
        c["background_color"] = v["background_color"].get().strip() or D.BACKGROUND_COLOR
        c["thumbnail_image_scale"] = _float_or_default(v["thumbnail_image_scale"].get(), D.THUMBNAIL_IMAGE_SCALE)
        c["thumbnail_image_blur"] = _float_or_default(v["thumbnail_image_blur"].get(), D.THUMBNAIL_IMAGE_BLUR)
        c["thumbnail_shadow_margin"] = _int_or_default(v["thumbnail_shadow_margin"].get(), D.THUMBNAIL_SHADOW_MARGIN)
        c["thumbnail_shadow_blur"] = _float_or_default(v["thumbnail_shadow_blur"].get(), D.THUMBNAIL_SHADOW_BLUR)
        c["thumbnail_shadow_alpha"] = _float_or_default(v["thumbnail_shadow_alpha"].get(), D.THUMBNAIL_SHADOW_ALPHA)
        c["fps"] = _int_or_default(v["fps"].get(), D.FPS)
        return c

    def _format_task_output_path(self, output_dir: Optional[str], output_filename: str) -> str:
        if output_dir and output_dir.strip():
            return os.path.normpath(os.path.join(output_dir.strip(), output_filename or D.OUTPUT_FILE))
        return f"(cwd){os.sep}{output_filename or D.OUTPUT_FILE}"

    def _show_tree_tooltip(self, text: str, x: int, y: int) -> None:
        if self._tree_tooltip_text == text and self._tree_tooltip is not None:
            return
        self._hide_tree_tooltip()
        self._tree_tooltip = tw = tk.Toplevel(self.root)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(tw, text=text, background="#ffffe0", relief="solid", borderwidth=1)
        label.pack(ipadx=4, ipady=2)
        self._tree_tooltip_text = text

    def _hide_tree_tooltip(self, event: Optional[tk.Event] = None) -> None:
        if self._tree_tooltip is not None:
            self._tree_tooltip.destroy()
            self._tree_tooltip = None
        self._tree_tooltip_text = ""

    def _on_tree_motion(self, event: tk.Event) -> None:
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            self._hide_tree_tooltip()
            return

        col_id = self.tree.identify_column(event.x)
        row_id = self.tree.identify_row(event.y)
        if not col_id or not row_id:
            self._hide_tree_tooltip()
            return

        try:
            col_index = int(col_id[1:]) - 1
        except (ValueError, TypeError):
            self._hide_tree_tooltip()
            return

        columns = self.tree["columns"]
        if col_index < 0 or col_index >= len(columns):
            self._hide_tree_tooltip()
            return

        col_name = columns[col_index]
        if col_name != "Output path":
            self._hide_tree_tooltip()
            return

        text = self.tree.set(row_id, col_name)
        if not text:
            self._hide_tree_tooltip()
            return

        col_width = self.tree.column(col_name, "width")
        if self._tree_font is None:
            self._tree_font = tkfont.nametofont("TkDefaultFont")
        text_width = self._tree_font.measure(text)

        if text_width <= max(col_width - 10, 0):
            self._hide_tree_tooltip()
            return

        x_root = self.tree.winfo_rootx() + event.x + 16
        y_root = self.tree.winfo_rooty() + event.y + 16
        self._show_tree_tooltip(text, x_root, y_root)

    def _add_task(self) -> None:
        cfg = self._get_current_config()
        if not cfg["audio"] or not os.path.isfile(cfg["audio"]):
            messagebox.showwarning("Invalid task", "Please set a valid MP3 (or audio) file path.")
            return
        self.tasks.append(cfg)
        out_path = self._format_task_output_path(cfg["output_dir"], cfg["output_filename"])
        self.tree.insert("", tk.END, values=(
            os.path.basename(cfg["audio"]),
            os.path.basename(cfg["background_image"] or "") or "(none)",
            (cfg["soundwave_text"] or "")[:20] + ("…" if len((cfg["soundwave_text"] or "")) > 20 else ""),
            "pending",
            out_path,
        ))
        messagebox.showinfo("Task added", "Current configuration added to the task list.")

    def _remove_selected(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a task in the list to remove.")
            return
        indices = sorted([self.tree.index(item) for item in sel], reverse=True)
        for idx in indices:
            if 0 <= idx < len(self.tasks):
                self.tasks.pop(idx)
        for item in sel:
            self.tree.delete(item)

    def _preview_selected(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a task in the list to preview.")
            return
        idx = self.tree.index(sel[0])
        if idx < 0 or idx >= len(self.tasks):
            messagebox.showwarning("Invalid selection", "Selected task index is out of range.")
            return
        cfg = self.tasks[idx]
        if not cfg.get("audio") or not os.path.isfile(cfg["audio"]):
            messagebox.showwarning("Invalid task", "Selected task has no valid audio file.")
            return
        try:
            v = SoundwaveVisualizer(
                cfg["audio"],
                video_width=cfg["video_width"],
                video_height=cfg["video_height"],
                soundwave_radius=cfg["soundwave_radius"],
                soundwave_text=cfg["soundwave_text"],
                soundwave_color=cfg["soundwave_color"],
                soundwave_opacity=cfg["soundwave_opacity"],
                background_image=cfg["background_image"],
                background_image_blur=cfg["background_image_blur"],
                background_color=cfg["background_color"],
                thumbnail_image_blur=cfg["thumbnail_image_blur"],
                thumbnail_image_scale=cfg["thumbnail_image_scale"],
                thumbnail_shadow_margin=cfg["thumbnail_shadow_margin"],
                thumbnail_shadow_blur=cfg["thumbnail_shadow_blur"],
                thumbnail_shadow_alpha=cfg["thumbnail_shadow_alpha"],
                fps=cfg["fps"],
                output_dir=cfg.get("output_dir") or "",
                output_filename=cfg.get("output_filename"),
                ffmpeg_path=cfg.get("ffmpeg_path", DEFAULT_FFMPEG_PATH),
            )
            v.preview()
        except Exception as e:
            messagebox.showerror("Preview error", str(e))

    def _set_task_status(self, index: int, status: str) -> None:
        """Update the Status column for the task at index (must be called from main thread)."""
        children = self.tree.get_children()
        if 0 <= index < len(children):
            self.tree.set(children[index], "Status", status)

    def _export_all(self) -> None:
        if not self.tasks:
            messagebox.showwarning("No tasks", "Add at least one task (Add current to task list) before exporting.")
            return
        ffmpeg_path = self._get_ffmpeg_path()
        if not check_ffmpeg(ffmpeg_path):
            messagebox.showerror(
                "FFmpeg required",
                "FFmpeg was not found. Please install FFmpeg and set the path in the FFmpeg path field.\n\n"
                "Download: https://ffmpeg.org/download.html",
            )
            return
        log_win = tk.Toplevel(self.root)
        log_win.title("Export log")
        log_win.geometry("500x300")
        log = scrolledtext.ScrolledText(log_win, wrap=tk.WORD)
        log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        def append_log(line: str) -> None:
            log.insert(tk.END, line)
            log.see(tk.END)
            log.update_idletasks()

        def run_export_thread() -> None:
            for i, cfg in enumerate(self.tasks):
                self.root.after(0, lambda idx=i: self._set_task_status(idx, "exporting"))
                self.root.after(0, lambda idx=i, p=cfg["audio"]: append_log(f"[{idx+1}/{len(self.tasks)}] {p} → exporting…\n"))
                try:
                    v = SoundwaveVisualizer(
                        cfg["audio"],
                        video_width=cfg["video_width"],
                        video_height=cfg["video_height"],
                        soundwave_radius=cfg["soundwave_radius"],
                        soundwave_text=cfg["soundwave_text"],
                        soundwave_color=cfg["soundwave_color"],
                        soundwave_opacity=cfg["soundwave_opacity"],
                        background_image=cfg["background_image"],
                        background_image_blur=cfg["background_image_blur"],
                        background_color=cfg["background_color"],
                        thumbnail_image_blur=cfg["thumbnail_image_blur"],
                        thumbnail_image_scale=cfg["thumbnail_image_scale"],
                        thumbnail_shadow_margin=cfg["thumbnail_shadow_margin"],
                        thumbnail_shadow_blur=cfg["thumbnail_shadow_blur"],
                        thumbnail_shadow_alpha=cfg["thumbnail_shadow_alpha"],
                        fps=cfg["fps"],
                        output_dir=cfg["output_dir"] or "",
                        output_filename=cfg["output_filename"],
                        ffmpeg_path=cfg.get("ffmpeg_path", DEFAULT_FFMPEG_PATH),
                    )
                    out_path = v._resolve_output_path(None, cfg["output_dir"], cfg["output_filename"])
                    v.export(output_path=out_path)
                    self.root.after(0, lambda idx=i: self._set_task_status(idx, "exported"))
                    self.root.after(0, lambda p=out_path: append_log(f"  → Saved: {p}\n"))
                except Exception as e:
                    err_msg = str(e)
                    self.root.after(0, lambda idx=i: self._set_task_status(idx, "error"))
                    self.root.after(0, lambda msg=err_msg: append_log(f"  → Error: {msg}\n"))
                self.root.after(0, lambda: log.see(tk.END))
            self.root.after(0, lambda: append_log("Done.\n"))
            self.root.after(0, lambda: messagebox.showinfo("Export complete", "Bulk export finished. Check the log window."))

        thread = threading.Thread(target=run_export_thread, daemon=True)
        thread.start()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = SoundwaveGUI()
    app.run()


if __name__ == "__main__":
    main()
