#!/usr/bin/env python3
"""Deprecated video processing CLI.

This repo now standardizes CLI usage around the Ultralytics `yolo` command.

Use instead:
  - Predict (detect):  ./scripts/yolo_predict_video.(fish|sh) --source <video>
  - Track (BoT-SORT):  ./scripts/yolo_track_video.(fish|sh) --source <video>

If you want the UI's annotated/tracked export, use the UI's
"Select & Process Video" button (it uses the custom tracker).
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    print(
        "process_video.py is deprecated.\n\n"
        "Use:\n"
        "  ./scripts/yolo_predict_video.(fish|sh) --source videos/raw/your_video.mp4 --model models/yolov8m_robots.pt\n"
        "  ./scripts/yolo_track_video.(fish|sh)   --source videos/raw/your_video.mp4 --model models/yolov8m_robots.pt\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
