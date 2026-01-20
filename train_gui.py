"""Simple Tkinter GUI to configure and run Ultralytics YOLO training.

Why: makes selecting data/model easier, shows live logs, and provides a simple
ETA estimate based on observed epoch durations.

Run:
  python3 train_gui.py

Or via script:
  ./scripts/yolo_train_gui.fish
"""

from __future__ import annotations

import queue
import re
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def _workspace_root() -> Path:
    return Path(__file__).resolve().parent


def _candidate_dataset_yamls(root: Path) -> list[Path]:
    candidates: list[Path] = []

    preferred = root / "dataset" / "combined_data.yaml"
    if preferred.exists():
        candidates.append(preferred)

    dataset_dir = root / "dataset"
    if dataset_dir.exists():
        for p in sorted(dataset_dir.glob("**/*.yaml")):
            if p.is_file() and p not in candidates:
                candidates.append(p)

    archive = root / "archive" / "datasets"
    if archive.exists():
        for p in sorted(archive.glob("**/data.yaml")):
            if p.is_file() and p not in candidates:
                candidates.append(p)

    # De-dup while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def _candidate_models(root: Path) -> list[Path]:
    candidates: list[Path] = []

    for p in sorted(root.glob("yolov8*.pt")):
        if p.is_file():
            candidates.append(p)

    models_dir = root / "models"
    if models_dir.exists():
        for p in sorted(models_dir.glob("*.pt")):
            if p.is_file() and p not in candidates:
                candidates.append(p)

    # De-dup while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def _format_hms(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:d}m {s:02d}s"


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value.strip())
    except Exception:
        return default


def _find_resumable_runs(root: Path, project: str = "runs_yolo") -> list[Path]:
    """Find runs with weights/last.pt in project directory."""
    project_dir = root / project
    if not project_dir.exists():
        return []
    
    runs: list[Path] = []
    for run_dir in sorted(project_dir.iterdir()):
        if run_dir.is_dir():
            last_pt = run_dir / "weights" / "last.pt"
            if last_pt.exists():
                runs.append(last_pt)
    return runs


@dataclass
class TrainProgress:
    total_epochs: int | None = None
    current_epoch: int | None = None  # 1-based
    save_dir: str | None = None

    start_time_s: float = field(default_factory=time.monotonic)
    completed_epoch_durations_s: list[float] = field(default_factory=list)
    _last_epoch_seen_ts_s: float | None = None

    def elapsed_s(self) -> float:
        return time.monotonic() - self.start_time_s

    def avg_epoch_s(self) -> float | None:
        if not self.completed_epoch_durations_s:
            return None
        return sum(self.completed_epoch_durations_s) / len(self.completed_epoch_durations_s)

    def eta_s(self) -> float | None:
        if self.total_epochs is None or self.current_epoch is None:
            return None
        avg = self.avg_epoch_s()
        if avg is None:
            return None
        remaining = max(self.total_epochs - self.current_epoch, 0)
        return remaining * avg

    def note_epoch_seen(self, epoch: int) -> None:
        now = time.monotonic()
        if self._last_epoch_seen_ts_s is not None:
            dt = now - self._last_epoch_seen_ts_s
            if 0.25 <= dt <= 24 * 3600:
                self.completed_epoch_durations_s.append(dt)
        self._last_epoch_seen_ts_s = now


class YoloTrainGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("seer — YOLO Train")
        self.geometry("1000x700")

        self._workspace = _workspace_root()
        self._proc: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._queue: queue.Queue[str] = queue.Queue()
        self._progress = TrainProgress()
        self._running = False
        self._resume_mode = False

        self._build_ui()
        self._populate_defaults()
        self._refresh_cmd_preview()

        self.after(100, self._poll_queue)
        self.after(250, self._refresh_stats)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # --- UI ---

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=10)
        top.grid(row=0, column=0, sticky="nsew")
        top.columnconfigure(0, weight=2)
        top.columnconfigure(1, weight=1)

        left = ttk.Labelframe(top, text="Training config", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(1, weight=1)

        right = ttk.Labelframe(top, text="Status", padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        # Dataset
        ttk.Label(left, text="Dataset YAML").grid(row=0, column=0, sticky="w")
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(left, textvariable=self.dataset_var, state="readonly")
        self.dataset_combo.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(left, text="Browse…", command=self._browse_dataset).grid(row=0, column=2, sticky="e")

        # Model
        ttk.Label(left, text="Model (.pt)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(left, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
        ttk.Button(left, text="Browse…", command=self._browse_model).grid(row=1, column=2, sticky="e", pady=(8, 0))

        # Resume from existing run
        self.resume_var = tk.BooleanVar(value=False)
        resume_check = ttk.Checkbutton(left, text="Resume from existing run", variable=self.resume_var, command=self._on_resume_toggle)
        resume_check.grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 4))

        self.resume_run_var = tk.StringVar()
        self.resume_combo = ttk.Combobox(left, textvariable=self.resume_run_var, state="readonly")
        self.resume_combo.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.resume_combo.grid_remove()  # Hidden by default

        # Basic params (2 columns)
        params = ttk.Frame(left)
        params.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        for c in range(4):
            params.columnconfigure(c, weight=1)

        ttk.Label(params, text="epochs").grid(row=0, column=0, sticky="w")
        self.epochs_var = tk.StringVar(value="50")
        ttk.Entry(params, textvariable=self.epochs_var, width=8).grid(row=0, column=1, sticky="ew", padx=(6, 12))

        ttk.Label(params, text="imgsz").grid(row=0, column=2, sticky="w")
        self.imgsz_var = tk.StringVar(value="640")
        ttk.Entry(params, textvariable=self.imgsz_var, width=8).grid(row=0, column=3, sticky="ew", padx=(6, 0))

        ttk.Label(params, text="batch").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.batch_var = tk.StringVar(value="16")
        ttk.Entry(params, textvariable=self.batch_var, width=8).grid(row=1, column=1, sticky="ew", padx=(6, 12), pady=(8, 0))

        ttk.Label(params, text="device (mps/cpu/0)").grid(row=1, column=2, sticky="w", pady=(8, 0))
        self.device_var = tk.StringVar(value="mps")
        ttk.Entry(params, textvariable=self.device_var, width=10).grid(row=1, column=3, sticky="ew", padx=(6, 0), pady=(8, 0))

        ttk.Label(params, text="project").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.project_var = tk.StringVar(value="runs_yolo")
        ttk.Entry(params, textvariable=self.project_var).grid(row=2, column=1, sticky="ew", padx=(6, 12), pady=(8, 0))

        ttk.Label(params, text="name").grid(row=2, column=2, sticky="w", pady=(8, 0))
        self.name_var = tk.StringVar(value="robots")
        ttk.Entry(params, textvariable=self.name_var).grid(row=2, column=3, sticky="ew", padx=(6, 0), pady=(8, 0))

        # Buttons
        btns = ttk.Frame(left)
        btns.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)

        self.start_btn = ttk.Button(btns, text="Start training", command=self._start_training)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.stop_btn = ttk.Button(btns, text="Stop", command=self._stop_training)
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.stop_btn.state(["disabled"])

        self.cmd_preview = ttk.Label(left, text="Command: —", wraplength=650, justify="left")
        self.cmd_preview.grid(row=6, column=0, columnspan=3, sticky="w", pady=(10, 0))

        # Status panel
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(right, textvariable=self.status_var, wraplength=260, justify="left").grid(row=0, column=0, sticky="w")

        self.epoch_var = tk.StringVar(value="Epoch: —")
        self.elapsed_var = tk.StringVar(value="Elapsed: —")
        self.avg_epoch_var = tk.StringVar(value="Avg epoch: —")
        self.eta_var = tk.StringVar(value="ETA: —")
        self.save_dir_var = tk.StringVar(value="Save dir: —")

        for i, var in enumerate([self.epoch_var, self.elapsed_var, self.avg_epoch_var, self.eta_var, self.save_dir_var], start=1):
            ttk.Label(right, textvariable=var, wraplength=260, justify="left").grid(row=i, column=0, sticky="w", pady=(6, 0))

        ttk.Label(
            right,
            text="Tip: set SEER_YOLO_DEVICE for a default device.",
            wraplength=260,
            justify="left",
            foreground="#666",
        ).grid(row=10, column=0, sticky="w", pady=(10, 0))

        # Logs
        logs = ttk.Labelframe(self, text="Logs", padding=10)
        logs.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        logs.rowconfigure(0, weight=1)
        logs.columnconfigure(0, weight=1)

        self.log_text = tk.Text(logs, wrap="word", height=20)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(logs, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

        # Hook updates
        self.dataset_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_cmd_preview())
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_cmd_preview())
        self.resume_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_cmd_preview())
        for var in [
            self.epochs_var,
            self.imgsz_var,
            self.batch_var,
            self.device_var,
            self.project_var,
            self.name_var,
        ]:
            var.trace_add("write", lambda *_: self._refresh_cmd_preview())
        
        # Update resume runs list when project changes
        self.project_var.trace_add("write", lambda *_: self._refresh_resume_runs())

    def _populate_defaults(self) -> None:
        datasets = _candidate_dataset_yamls(self._workspace)
        models = _candidate_models(self._workspace)

        self.dataset_combo["values"] = [str(p.relative_to(self._workspace)) for p in datasets]
        self.model_combo["values"] = [str(p.relative_to(self._workspace)) for p in models]

        if datasets:
            self.dataset_var.set(str(datasets[0].relative_to(self._workspace)))
        if models:
            preferred = next((p for p in models if p.name == "yolov8n.pt"), None)
            self.model_var.set(str((preferred or models[0]).relative_to(self._workspace)))
        
        self._refresh_resume_runs()

    def _on_resume_toggle(self) -> None:
        """Handle resume checkbox toggle."""
        self._resume_mode = self.resume_var.get()
        if self._resume_mode:
            self.resume_combo.grid()
            # Disable normal config when resuming
            self.dataset_combo.state(["disabled"])
            self.model_combo.state(["disabled"])
        else:
            self.resume_combo.grid_remove()
            self.dataset_combo.state(["!disabled"])
            self.model_combo.state(["!disabled"])
        self._refresh_cmd_preview()
    
    def _refresh_resume_runs(self) -> None:
        """Update the list of resumable runs."""
        project = (self.project_var.get() or "").strip() or "runs_yolo"
        runs = _find_resumable_runs(self._workspace, project)
        
        if runs:
            self.resume_combo["values"] = [str(p.relative_to(self._workspace)) for p in runs]
            if not self.resume_run_var.get() and runs:
                # Auto-select most recent (last in sorted list)
                self.resume_run_var.set(str(runs[-1].relative_to(self._workspace)))
        else:
            self.resume_combo["values"] = ["(no resumable runs found)"]
            self.resume_run_var.set("")

    def _browse_dataset(self) -> None:
        p = filedialog.askopenfilename(
            title="Select dataset YAML",
            initialdir=str(self._workspace / "dataset"),
            filetypes=[("YAML", "*.yaml"), ("All", "*")],
        )
        if p:
            try:
                rel = str(Path(p).resolve().relative_to(self._workspace))
            except Exception:
                rel = p
            self.dataset_var.set(rel)
            self._refresh_cmd_preview()

    def _browse_model(self) -> None:
        p = filedialog.askopenfilename(
            title="Select model .pt",
            initialdir=str(self._workspace),
            filetypes=[("PyTorch", "*.pt"), ("All", "*")],
        )
        if p:
            try:
                rel = str(Path(p).resolve().relative_to(self._workspace))
            except Exception:
                rel = p
            self.model_var.set(rel)
            self._refresh_cmd_preview()

    # --- Training lifecycle ---

    def _build_cmd(self) -> list[str] | None:
        if self._resume_mode:
            # Resume mode: use weights/last.pt from selected run
            run_rel = (self.resume_run_var.get() or "").strip()
            if not run_rel or run_rel == "(no resumable runs found)":
                return None
            
            last_pt = str((self._workspace / run_rel).resolve()) if not Path(run_rel).is_absolute() else run_rel
            return [
                "yolo",
                "train",
                "resume",
                f"model={last_pt}",
            ]
        else:
            # Normal training mode
            data_rel = (self.dataset_var.get() or "").strip()
            model_rel = (self.model_var.get() or "").strip()

            if not data_rel or not model_rel:
                return None

            data = str((self._workspace / data_rel).resolve()) if not Path(data_rel).is_absolute() else data_rel
            model = str((self._workspace / model_rel).resolve()) if not Path(model_rel).is_absolute() else model_rel

            epochs = _safe_int(self.epochs_var.get(), 50)
            imgsz = _safe_int(self.imgsz_var.get(), 640)
            batch = _safe_int(self.batch_var.get(), 16)
            device = (self.device_var.get() or "").strip() or "cpu"
            project = (self.project_var.get() or "").strip() or "runs_yolo"
            name = (self.name_var.get() or "").strip() or "robots"

            return [
                "yolo",
                "detect",
                "train",
                f"data={data}",
                f"model={model}",
                f"epochs={epochs}",
                f"imgsz={imgsz}",
                f"batch={batch}",
                f"device={device}",
                f"project={project}",
                f"name={name}",
            ]

    def _refresh_cmd_preview(self) -> None:
        cmd = self._build_cmd()
        self.cmd_preview.configure(text=("Command: —" if cmd is None else f"Command: {shlex.join(cmd)}"))

    def _start_training(self) -> None:
        if self._running:
            return

        if shutil.which("yolo") is None:
            messagebox.showerror("Missing yolo", "Could not find the 'yolo' CLI in PATH.\n\nTry: pip install ultralytics")
            return

        cmd = self._build_cmd()
        if cmd is None:
            messagebox.showwarning("Missing config", "Please select a dataset YAML and model.")
            return

        self.log_text.delete("1.0", "end")
        self._progress = TrainProgress()
        self._running = True

        self.start_btn.state(["disabled"])
        self.stop_btn.state(["!disabled"])
        self.status_var.set("Starting…")

        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(self._workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            self._running = False
            self.start_btn.state(["!disabled"])
            self.stop_btn.state(["disabled"])
            messagebox.showerror("Failed to start", str(e))
            return

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _stop_training(self) -> None:
        if not self._running or self._proc is None:
            return

        self.status_var.set("Stopping…")
        try:
            self._proc.terminate()
        except Exception:
            pass

        def _kill_later() -> None:
            if self._proc is None:
                return
            if self._proc.poll() is None:
                try:
                    self._proc.kill()
                except Exception:
                    pass

        self.after(2000, _kill_later)

    def _reader_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None

        for line in self._proc.stdout:
            self._queue.put(line)

        rc = self._proc.wait()
        self._queue.put(f"\n[process exited with code {rc}]\n")
        self._queue.put("__DONE__")

    # --- Log parsing + stats ---

    _epoch_re = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\b")
    _save_dir_re = re.compile(r"\bsave_dir=([^,\s]+)")

    def _handle_line(self, line: str) -> None:
        m = self._save_dir_re.search(line)
        if m:
            self._progress.save_dir = m.group(1)

        m2 = self._epoch_re.match(line)
        if m2:
            cur = int(m2.group(1))
            total = int(m2.group(2))
            if self._progress.current_epoch is None or cur != self._progress.current_epoch:
                self._progress.current_epoch = cur
                self._progress.total_epochs = total
                self._progress.note_epoch_seen(cur)

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self._queue.get_nowait()
                if item == "__DONE__":
                    self._on_process_done()
                    break

                self._handle_line(item)
                self.log_text.insert("end", item)
                self.log_text.see("end")
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    def _refresh_stats(self) -> None:
        if self._running:
            self.status_var.set("Running")
        else:
            if self.status_var.get() == "Running":
                self.status_var.set("Idle")

        ep = self._progress.current_epoch
        tot = self._progress.total_epochs
        self.epoch_var.set(f"Epoch: {ep}/{tot}" if ep and tot else "Epoch: —")

        self.elapsed_var.set(f"Elapsed: {_format_hms(self._progress.elapsed_s())}")

        avg = self._progress.avg_epoch_s()
        self.avg_epoch_var.set(f"Avg epoch: {_format_hms(avg)}" if avg is not None else "Avg epoch: —")

        eta = self._progress.eta_s()
        self.eta_var.set(f"ETA: {_format_hms(eta)}" if eta is not None else "ETA: —")

        self.save_dir_var.set(f"Save dir: {self._progress.save_dir}" if self._progress.save_dir else "Save dir: —")

        self.after(250, self._refresh_stats)

    def _on_process_done(self) -> None:
        self._running = False
        self.start_btn.state(["!disabled"])
        self.stop_btn.state(["disabled"])
        if self._proc is not None and self._proc.returncode == 0:
            self.status_var.set("Done")
        else:
            self.status_var.set("Exited (see logs)")

    def _on_close(self) -> None:
        if self._running:
            if not messagebox.askyesno("Quit", "Training is still running. Stop and quit?"):
                return
            self._stop_training()
        self.destroy()


def main() -> None:
    app = YoloTrainGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
