from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional, Tuple

import cv2
from PIL import Image, ImageTk

from .model import SessionStore
from .video import VideoStream, CameraConfig, probe_cameras
from .vision import draw_boxes, map_boxes_frame_to_canvas, Box
from .config import Config
from .tba_downloader import TBADownloader
from .cache import CacheManager
from .robot_detector import RobotDetector
from .tracker import RobotTracker, RobotDetection
from .video_processor import VideoProcessor


_CONF = 0.5


class ScoutingApp:
    """
    High-level app wrapper.
    Keeps UI code in one place, but delegates:
      - capture to VideoStream
      - detection to vision.py
      - data to model.py
    """
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Watchdog AI - Scouting Base UI")
        self.root.geometry("1200x760")

        # State
        self.store = SessionStore()
        self.vs = VideoStream()

        import os
        trained_model = "models/yolov8m_robots.pt"
        actual_model = trained_model if os.path.exists(trained_model) else "yolov8m.pt"
        self.model_loaded = actual_model
        self.robot_detector = RobotDetector(model_path=actual_model)
        # UI var to show which model is loaded
        self.model_var = tk.StringVar(value=f"Model: {os.path.basename(actual_model)}")
        self.robot_tracker = RobotTracker()
        self.enable_robot_tracking = tk.BooleanVar(value=True)
        
        # Video processor for processing stored videos
        self.video_processor = VideoProcessor(model_path=trained_model)
        self.processing_video = False

        self.available_cams: List[int] = []
        self.selected_cam_index: Optional[int] = None
        
        # TBA Integration
        self.tba_downloader: Optional[TBADownloader] = None
        self.cache_manager = CacheManager()
        self.event_data: Optional[dict] = None
        self.match_list: List[str] = []  # List of match labels for dropdown
        try:
            self.tba_downloader = TBADownloader(Config.get_tba_key())
        except Exception as e:
            print(f"Warning: TBA downloader not initialized: {e}")

        # Video display config
        self.video_w = 860
        self.video_h = 480

        # Canvas image handle + click-hit boxes
        self._canvas_img_id: Optional[int] = None
        self._last_imgtk: Optional[ImageTk.PhotoImage] = None
        self._hit_boxes: List[Box] = []

        # Build UI
        self._build_ui()

        # Events / timers
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.video_canvas.bind("<Button-1>", self._on_canvas_click)

        self._refresh_camera_list()

        # periodic updates
        self.root.after(33, self._tick_video)      # ~30fps UI render
        self.root.after(500, self._tick_table)     # stats refresh

    def run(self) -> None:
        self.root.mainloop()

    # ---------------- UI layout ----------------

    def _build_ui(self) -> None:
        root = ttk.Frame(self.root)
        root.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(root, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # Video area (fixed-size container)
        video_frame = ttk.Frame(left, width=self.video_w, height=self.video_h)
        video_frame.pack(fill=tk.BOTH, expand=True)
        video_frame.pack_propagate(False)

        self.video_canvas = tk.Canvas(video_frame, width=self.video_w, height=self.video_h, highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.status_var, anchor="w").pack(fill=tk.X, pady=(8, 0))

        # Right side controls
        self._build_camera_panel(right)
        self._build_tba_panel(right)
        self._build_session_panel(right)
        self._build_video_processor_panel(right)
        self._build_stats_panel(right)
        self._build_export_panel(right)

    def _build_camera_panel(self, parent: ttk.Frame) -> None:
        box = ttk.Labelframe(parent, text="Camera")
        box.pack(fill=tk.X, pady=(0, 10))

        self.cam_combo = ttk.Combobox(box, state="readonly", values=[])
        self.cam_combo.pack(fill=tk.X, padx=8, pady=(8, 6))
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_camera_selected)

        ttk.Checkbutton(box, text="Robot tracking overlay", variable=self.enable_robot_tracking)\
            .pack(anchor="w", padx=8, pady=(0, 8))

        # Show which model is currently loaded
        ttk.Label(box, textvariable=self.model_var).pack(anchor="w", padx=8, pady=(0, 4))

        row = ttk.Frame(box)
        row.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(row, text="Refresh", command=self._refresh_camera_list).pack(side=tk.LEFT)
        ttk.Button(row, text="Start", command=self._start_camera).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(row, text="Stop", command=self._stop_camera).pack(side=tk.LEFT, padx=(8, 0))

    def _build_tba_panel(self, parent: ttk.Frame) -> None:
        box = ttk.Labelframe(parent, text="TBA Event Data")
        box.pack(fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(box)
        row1.pack(fill=tk.X, padx=8, pady=(8, 6))
        ttk.Label(row1, text="Event key:").pack(side=tk.LEFT)
        self.event_entry = ttk.Entry(row1)
        self.event_entry.insert(0, Config.get_event_key())
        self.event_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        row2 = ttk.Frame(box)
        row2.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(row2, text="Download Data", command=self._download_tba_data).pack(side=tk.LEFT)
        ttk.Button(row2, text="View Data", command=self._view_tba_data).pack(side=tk.LEFT, padx=(8, 0))

    def _build_session_panel(self, parent: ttk.Frame) -> None:
        box = ttk.Labelframe(parent, text="Session")
        box.pack(fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(box)
        row1.pack(fill=tk.X, padx=8, pady=(8, 6))
        ttk.Label(row1, text="Match:").pack(side=tk.LEFT)
        self.match_combo = ttk.Combobox(row1, state="readonly", values=self.match_list)
        if self.match_list:
            self.match_combo.current(0)
        self.match_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        row2 = ttk.Frame(box)
        row2.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(row2, text="Start session", command=self._start_session).pack(side=tk.LEFT)
        ttk.Button(row2, text="New session", command=self._new_session).pack(side=tk.LEFT, padx=(8, 0))

    def _build_video_processor_panel(self, parent: ttk.Frame) -> None:
        box = ttk.Labelframe(parent, text="Video Processor")
        box.pack(fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(box)
        row1.pack(fill=tk.X, padx=8, pady=(8, 6))
        ttk.Button(row1, text="Select & Process Video", command=self._process_video_file).pack(side=tk.LEFT)

        self.video_status_var = tk.StringVar(value="Ready")
        ttk.Label(box, textvariable=self.video_status_var, wraplength=300).pack(fill=tk.X, padx=8, pady=(0, 8))

    def _build_stats_panel(self, parent: ttk.Frame) -> None:
        box = ttk.Labelframe(parent, text="Team Stats (placeholder)")
        box.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        cols = ("team", "made", "miss", "low", "mid", "high")
        self.tree = ttk.Treeview(box, columns=cols, show="headings", height=10)
        for c, title, w in [
            ("team", "Team", 60),
            ("made", "Made", 55),
            ("miss", "Miss", 55),
            ("low", "L", 40),
            ("mid", "M", 40),
            ("high", "H", 40),
        ]:
            self.tree.heading(c, text=title)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _build_export_panel(self, parent: ttk.Frame) -> None:
        box = ttk.Labelframe(parent, text="Export")
        box.pack(fill=tk.X)

        row = ttk.Frame(box)
        row.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(row, text="Export JSON", command=self._export_json).pack(side=tk.LEFT)
        ttk.Button(row, text="Export CSV", command=self._export_csv).pack(side=tk.LEFT, padx=(8, 0))

    # ---------------- Camera controls ----------------

    def _refresh_camera_list(self) -> None:
        self.status_var.set("Probing cameras...")
        self.root.update_idletasks()

        # Keep probe range small to avoid log spam
        self.available_cams = probe_cameras(max_index=3)

        if not self.available_cams:
            self.cam_combo["values"] = ["(no cameras found)"]
            self.cam_combo.current(0)
            self.selected_cam_index = None
            self.status_var.set("No cameras found. Check macOS Camera permissions.")
            return

        items = [f"Camera {i}" for i in self.available_cams]
        self.cam_combo["values"] = items
        self.cam_combo.current(0)
        self.selected_cam_index = self.available_cams[0]
        self.status_var.set(f"Found cameras: {self.available_cams}")

    def _on_camera_selected(self, _evt=None) -> None:
        idx = self.cam_combo.current()
        if idx < 0 or idx >= len(self.available_cams):
            return
        self.selected_cam_index = self.available_cams[idx]
        self.status_var.set(f"Selected camera {self.selected_cam_index}")

    def _start_camera(self) -> None:
        if self.selected_cam_index is None:
            messagebox.showerror("Camera", "No camera selected.")
            return

        ok = self.vs.open(CameraConfig(index=self.selected_cam_index, width=1280, height=720, fps=60))
        if not ok:
            messagebox.showerror("Camera", f"Failed to open camera {self.selected_cam_index}.")
            return

        self.vs.start()
        self.status_var.set(f"Camera {self.selected_cam_index} started.")

    def _stop_camera(self) -> None:
        self.vs.close()
        self.status_var.set("Camera stopped.")

    # ---------------- TBA controls ----------------

    def _download_tba_data(self) -> None:
        """Download event data from The Blue Alliance (or load from cache)."""
        if not self.tba_downloader:
            messagebox.showerror("TBA Error", "TBA downloader not initialized. Check your API key in .env file.")
            return

        event_key = self.event_entry.get().strip()
        if not event_key:
            messagebox.showerror("Event Key", "Please enter an event key (e.g., 2024week0)")
            return

        try:
            # Try to load from cache first
            cached_data = self.cache_manager.load_event_data(event_key)
            
            if cached_data:
                self.event_data = cached_data
                cached_at = cached_data.get('cached_at', 'Unknown')
                self.status_var.set(f"Loaded {event_key} from cache (cached at {cached_at})")
                
                num_teams = self.event_data['num_teams']
                num_matches = self.event_data['num_qual_matches']
                
                # Populate match dropdown
                self.match_list = [f"Qual {match.match_number}" for match in self.event_data['matches']]
                self.match_combo['values'] = self.match_list
                if self.match_list:
                    self.match_combo.current(0)
                
                msg = (f"Loaded cached data for {event_key}:\n\n"
                       f"Teams: {num_teams}\n"
                       f"Qualification Matches: {num_matches}\n\n"
                       f"Cached at: {cached_at}")
                
                messagebox.showinfo("Data Loaded (from cache)", msg)
            else:
                # Download from API if not in cache
                self.status_var.set(f"Downloading data for {event_key}...")
                self.root.update_idletasks()
                
                self.event_data = self.tba_downloader.get_event_data(event_key)
                
                # Save to cache
                self.cache_manager.save_event_data(event_key, self.event_data)
                
                num_teams = self.event_data['num_teams']
                num_matches = self.event_data['num_qual_matches']
                
                # Populate match dropdown
                self.match_list = [f"Qual {match.match_number}" for match in self.event_data['matches']]
                self.match_combo['values'] = self.match_list
                if self.match_list:
                    self.match_combo.current(0)
                
                msg = (f"Downloaded data for {event_key}:\n\n"
                       f"Teams: {num_teams}\n"
                       f"Qualification Matches: {num_matches}\n\n"
                       f"Data has been cached for future use.")
                
                self.status_var.set(f"Downloaded {num_teams} teams, {num_matches} qual matches")
                messagebox.showinfo("TBA Download Complete", msg)
            
        except Exception as e:
            self.status_var.set("TBA download failed")
            messagebox.showerror("TBA Error", f"Failed to download event data:\n{e}")

    def _view_tba_data(self) -> None:
        """Display downloaded TBA data in a new window."""
        if not self.event_data:
            messagebox.showinfo("No Data", "No event data downloaded yet. Click 'Download Data' first.")
            return

        # Create a new window to display the data
        view_window = tk.Toplevel(self.root)
        view_window.title(f"TBA Event Data: {self.event_data['event_key']}")
        view_window.geometry("700x600")

        # Create text widget with scrollbar
        frame = ttk.Frame(view_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text.yview)

        # Format and display the data
        content = f"Event: {self.event_data['event_key']}\n"
        content += f"{'=' * 60}\n\n"
        
        content += f"Teams ({self.event_data['num_teams']}):\n"
        content += f"{', '.join(map(str, self.event_data['teams']))}\n\n"
        
        content += f"{'=' * 60}\n"
        content += f"Qualification Matches ({self.event_data['num_qual_matches']}):\n"
        content += f"{'=' * 60}\n\n"
        
        for match in self.event_data['matches']:
            content += f"Match {match.match_number}:\n"
            content += f"  Red:  {', '.join(map(str, match.red_alliance.teams))}\n"
            content += f"  Blue: {', '.join(map(str, match.blue_alliance.teams))}\n\n"

        text.insert('1.0', content)
        text.config(state=tk.DISABLED)  # Make read-only

    # ---------------- Session controls ----------------

    def _start_session(self) -> None:
        match_label = self.match_combo.get()
        if not match_label:
            messagebox.showwarning("Match", "Please select a match first.")
            return
        self.store.set_match_label(match_label)
        self.store.start()
        self.status_var.set(f"Session started: {self.store.session_id} ({self.store.match_label})")

    def _new_session(self) -> None:
        if messagebox.askyesno("New session", "Start a new session? This clears current stats."):
            self.store.reset()
            if self.match_list:
                self.match_combo.current(0)
            self.status_var.set(f"New session created: {self.store.session_id}")



    # ---------------- Export ----------------

    def _export_json(self) -> None:
        default_name = f"{self.store.match_label}_{self.store.session_id}.json".replace(" ", "_")
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            self.store.export_json(path)
            self.status_var.set(f"Exported JSON: {path}")
        except Exception as e:
            messagebox.showerror("Export", f"Failed to export JSON:\n{e}")

    def _export_csv(self) -> None:
        default_name = f"{self.store.match_label}_{self.store.session_id}.csv".replace(" ", "_")
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        try:
            self.store.export_csv_summary(path)
            self.status_var.set(f"Exported CSV: {path}")
        except Exception as e:
            messagebox.showerror("Export", f"Failed to export CSV:\n{e}")

    # ---------------- Video Processing ----------------

    def _process_video_file(self) -> None:
        """Open a video file and process it with robot detection and tracking."""
        file_path = filedialog.askopenfilename(
            title="Select a video file to process",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        
        if not file_path:
            return
        
        if self.processing_video:
            messagebox.showwarning("Processing", "A video is already being processed. Please wait.")
            return
        
        self.processing_video = True
        self.video_status_var.set("Processing video... this may take a while")
        self.root.update_idletasks()
        
        try:
            # Reset tracker for new video
            self.video_processor.reset_tracker()
            
            # Define progress callback
            def progress_callback(current: int, total: int):
                percent = int(100 * current / total) if total > 0 else 0
                self.video_status_var.set(f"Processing: {percent}% ({current}/{total} frames)")
                self.root.update_idletasks()
            
            # Process the video
            output_path = self.video_processor.process_video(
                input_path=file_path,
                output_path=None,
                confidence_threshold=0.5,
                progress_callback=progress_callback
            )
            
            self.video_status_var.set(f"✓ Complete: {output_path}")
            self.status_var.set(f"Video processed: {output_path}")
            messagebox.showinfo("Success", f"Video processing complete!\n\nOutput saved to:\n{output_path}")
        
        except Exception as e:
            self.video_status_var.set(f"✗ Error: {str(e)}")
            self.status_var.set("Video processing failed")
            messagebox.showerror("Error", f"Failed to process video:\n{e}")
        
        finally:
            self.processing_video = False

    # ---------------- Video + Vision loop ----------------

    def _tick_video(self) -> None:
        frame = self.vs.get_frame()
        if frame is not None:
            # 1) Run robot detection + tracking + draw overlay in frame-space
            boxes_xywh = []
            if self.enable_robot_tracking.get():
                detections = self.robot_detector.detect(frame, _CONF)
                self.robot_tracker.update(detections)
                
                # Draw boxes for all active tracks
                for track_id, track in self.robot_tracker.tracks.items():
                    if track.positions:
                        cx, cy = track.positions[-1]
                        # Estimate box from track history (use average dimensions)
                        if len(track.positions) > 1:
                            # Simple box approximation: 40x40 centered at position
                            w, h = 40, 40
                            x = int(cx - w / 2)
                            y = int(cy - h / 2)
                            boxes_xywh.append((x, y, w, h))
                            # Draw with track ID
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID:{track_id}", (x, y - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 2) Convert to PIL for display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # 3) Resize to fit canvas while preserving aspect ratio (letterbox)
            draw_w, draw_h, x0, y0 = self._fit_to_canvas(pil_img.width, pil_img.height)

            pil_img = pil_img.resize((draw_w, draw_h), Image.Resampling.BILINEAR)
            imgtk = ImageTk.PhotoImage(image=pil_img)
            self._last_imgtk = imgtk

            # 4) Draw/update image
            if self._canvas_img_id is None:
                self._canvas_img_id = self.video_canvas.create_image(x0, y0, anchor="nw", image=imgtk)
            else:
                self.video_canvas.coords(self._canvas_img_id, x0, y0)
                self.video_canvas.itemconfig(self._canvas_img_id, image=imgtk)

            # 5) Build click-hit boxes in canvas coords
            fh, fw = frame.shape[:2]
            self._hit_boxes = map_boxes_frame_to_canvas(
                boxes_xywh=boxes_xywh,
                frame_size=(fw, fh),
                canvas_img_offset=(x0, y0),
                canvas_img_size=(draw_w, draw_h),
                label="robot",
            )

        self.root.after(33, self._tick_video)

    def _fit_to_canvas(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """
        Returns: (draw_w, draw_h, x0, y0) where the image will be drawn in the canvas.
        """
        canvas_w = self.video_w
        canvas_h = self.video_h

        img_ratio = img_w / img_h
        target_ratio = canvas_w / canvas_h

        if img_ratio > target_ratio:
            draw_w = canvas_w
            draw_h = int(canvas_w / img_ratio)
        else:
            draw_h = canvas_h
            draw_w = int(canvas_h * img_ratio)

        x0 = (canvas_w - draw_w) // 2
        y0 = (canvas_h - draw_h) // 2
        return draw_w, draw_h, x0, y0

    def _tick_table(self) -> None:
        # Refresh the table (simple approach)
        for item in self.tree.get_children():
            self.tree.delete(item)

        for team in sorted(self.store.teams.keys()):
            s = self.store.teams[team]
            self.tree.insert("", tk.END, values=(s.team, s.fuel_made, s.fuel_missed, s.climb_low, s.climb_mid, s.climb_high))

        self.root.after(500, self._tick_table)

    # ---------------- Click handling ----------------

    def _on_canvas_click(self, event) -> None:
        x, y = float(event.x), float(event.y)

        for box in reversed(self._hit_boxes):
            if box.contains(x, y):
                msg = f"Clicked: {box.label}"
                self.status_var.set(msg)
                messagebox.showinfo("Selection", msg)
                return

        self.status_var.set(f"Clicked at ({int(x)}, {int(y)}) - no robot detected")

    # ---------------- Shutdown ----------------

    def _on_close(self) -> None:
        try:
            self.vs.close()
        except Exception:
            pass
        self.root.destroy()

