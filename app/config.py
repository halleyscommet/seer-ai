from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)


def _getenv_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _getenv_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _getenv_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _getenv_str(name: str, default: Optional[str]) -> Optional[str]:
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip()
    return val if val else default


@dataclass(frozen=True)
class AppConfig:
    # TBA
    tba_api_key: Optional[str]
    tba_event_key: str

    # Model
    model_path_preferred: str
    model_path_fallback: str

    # Detection
    detection_conf_live: float
    detection_conf_video: float

    # Tracking (custom tracker)
    tracking_enabled_default: bool
    tracker_max_robots: int
    tracker_max_misses: int
    tracker_max_lost_age: int

    # Tracking (Ultralytics)
    ultralytics_tracker_yaml: str

    # Runtime device selection (Ultralytics)
    # Examples: "mps" (Apple Metal), "cpu", "0" (CUDA device 0)
    ultralytics_device: Optional[str]


class Config:
    """Central app configuration.

    Reads from `.env` (and inherited environment variables), but avoids raising at
    import time so the UI can still launch without optional integrations.
    """

    _cfg: Optional[AppConfig] = None

    @classmethod
    def get(cls) -> AppConfig:
        if cls._cfg is None:
            _load_env()
            cls._cfg = AppConfig(
                # TBA
                tba_api_key=os.getenv("TBA_API_KEY"),
                tba_event_key=os.getenv("TBA_EVENT_KEY", "2026mimil"),

                # Model
                model_path_preferred=os.getenv("MODEL_PATH", "models/yolov8m_robots.pt"),
                model_path_fallback=os.getenv("MODEL_FALLBACK_PATH", "yolov8m.pt"),

                # Detection
                detection_conf_live=_getenv_float("DETECTION_CONF_LIVE", 0.5),
                detection_conf_video=_getenv_float("DETECTION_CONF_VIDEO", 0.5),

                # Tracking
                tracking_enabled_default=_getenv_bool("TRACKING_ENABLED", True),
                tracker_max_robots=_getenv_int("TRACKER_MAX_ROBOTS", 6),
                tracker_max_misses=_getenv_int("TRACKER_MAX_MISSES", 60),
                tracker_max_lost_age=_getenv_int("TRACKER_MAX_LOST_AGE", 150),

                # Ultralytics tracker
                ultralytics_tracker_yaml=os.getenv("ULTRALYTICS_TRACKER_YAML", "botsort.yaml"),

                # Device (Ultralytics)
                ultralytics_device=_getenv_str("ULTRALYTICS_DEVICE", None),
            )
        return cls._cfg

    @classmethod
    def get_tba_key(cls) -> str:
        cfg = cls.get()
        if not cfg.tba_api_key or cfg.tba_api_key == "your_tba_api_key_here":
            raise ValueError("TBA_API_KEY not configured. Please update your .env file.")
        return cfg.tba_api_key

    @classmethod
    def get_event_key(cls) -> str:
        return cls.get().tba_event_key

