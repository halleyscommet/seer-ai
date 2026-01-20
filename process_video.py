#!/usr/bin/env python3
"""
CLI to process a video using the trained robot detector and tracker.
Usage:
  python process_video.py --input videos/raw/your_video.mp4 [--output out.mp4] [--conf 0.5]
"""

import argparse
import os
from app.video_processor import VideoProcessor


def main():
    parser = argparse.ArgumentParser(description="Process video with robot detector + tracker")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default=None, help="Output video path (optional)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    # Prefer trained model
    model_path = "models/yolov8m_robots.pt"
    vp = VideoProcessor(model_path=model_path)

    print(f"Processing video: {args.input}")
    out = vp.process_video(
        input_path=args.input,
        output_path=args.output,
        confidence_threshold=args.conf,
        progress_callback=lambda cur, tot: print(f"\r{int(100*cur/tot) if tot>0 else 0}% ({cur}/{tot})", end="")
    )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
