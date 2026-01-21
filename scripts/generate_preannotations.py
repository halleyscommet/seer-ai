#!/usr/bin/env python3
"""
Generate pre-annotations from a video using the trained YOLO model.
Outputs frames + YOLO format label files that can be refined by a human.

Usage:
    python scripts/generate_preannotations.py --video videos/raw/match.mp4 \
        --output dataset/preannotations --conf 0.3 --sample-rate 5
"""

import argparse
import cv2
from pathlib import Path
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.robot_detector import RobotDetector
from app.config import Config


def extract_frame_annotations(
    video_path: str,
    output_dir: str,
    model_path: str,
    confidence_threshold: float = 0.3,
    sample_rate: int = 1,
    device: Optional[str] = None,
) -> None:
    """
    Extract frames and generate YOLO format annotations.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save images/ and labels/
        model_path: Path to YOLO model
        confidence_threshold: Min confidence (lower = more false positives for human to filter)
        sample_rate: Save every Nth frame (1 = all frames, 5 = every 5th frame)
        device: Device to run inference on
    """
    # Setup output directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Also save classes.txt
    classes_file = output_path / "classes.txt"
    classes_file.write_text("robot\n")
    
    # Load detector
    print(f"Loading model from {model_path}...")
    detector = RobotDetector(model_path=model_path, device=device)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = Path(video_path).stem
    
    frame_idx = 0
    saved_count = 0
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, Sample rate: {sample_rate}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every Nth frame
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue
            
            # Run detection
            detections = detector.detect(frame, confidence_threshold=confidence_threshold)
            
            # Generate filename
            frame_name = f"{video_name}_frame_{frame_idx:04d}"
            image_path = images_dir / f"{frame_name}.jpg"
            label_path = labels_dir / f"{frame_name}.txt"
            
            # Save image
            cv2.imwrite(str(image_path), frame)
            
            # Save YOLO format labels
            # Format: class_id x_center y_center width height (all normalized 0-1)
            h, w = frame.shape[:2]
            
            with open(label_path, 'w') as f:
                for det in detections:
                    # Convert xyxy to xywh (normalized)
                    x_center = ((det.x1 + det.x2) / 2) / w
                    y_center = ((det.y1 + det.y2) / 2) / h
                    width = (det.x2 - det.x1) / w
                    height = (det.y2 - det.y1) / h
                    
                    # Class 0 = robot
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"Processed: {saved_count} frames ({frame_idx}/{total_frames})")
            
            frame_idx += 1
    
    finally:
        cap.release()
    
    print(f"\nDone! Saved {saved_count} annotated frames to {output_dir}")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review and correct labels using a tool like CVAT, Label Studio, or labelImg")
    print(f"  2. Remove false positives and add missed detections")
    print(f"  3. Merge corrected data into your training dataset")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-annotations from video using trained YOLO model",
        epilog="Example: ./scripts/generate_preannotations.py --video videos/raw/match.mp4 "
               "--output dataset/preannotations --sample-rate 4 --conf 0.3"
    )
    parser.add_argument("--video", required=True, 
                       help="Path to input video file")
    parser.add_argument("--output", required=True, 
                       help="Output directory for annotated frames (creates images/ and labels/ subdirs)")
    parser.add_argument("--model", default="models/yolov8m_robots.pt", 
                       help="Path to YOLO model weights (default: models/yolov8m_robots.pt)")
    parser.add_argument("--conf", type=float, default=0.3, 
                       help="Confidence threshold 0.0-1.0 (lower = more suggestions, default: 0.3)")
    parser.add_argument("--sample-rate", type=int, default=4,
                       help="Process every Nth frame to reduce workload. "
                            "For 30fps 3min video: rate=4 gives ~1350 frames, rate=10 gives ~540 frames (default: 4)")
    parser.add_argument("--device", default=None, 
                       help="Device to run inference on: cpu/mps/0/1/etc (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Get default device if not specified
    device = args.device
    if device is None:
        device = Config.get().ultralytics_device
    
    extract_frame_annotations(
        video_path=args.video,
        output_dir=args.output,
        model_path=args.model,
        confidence_threshold=args.conf,
        sample_rate=args.sample_rate,
        device=device,
    )


if __name__ == "__main__":
    main()
