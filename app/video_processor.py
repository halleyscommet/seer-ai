"""
Video processor for analyzing recorded videos with robot detection and tracking.
Processes videos frame-by-frame and generates annotated output.
"""

from __future__ import annotations

import cv2
from pathlib import Path
from typing import Optional, Callable
import os

from .robot_detector import RobotDetector
from .ultralytics_tracker import BoTSORTTracker


class VideoProcessor:
    """
    Process a video file with robot detection and tracking.
    Generates annotated output video with bounding boxes and track IDs.
    """
    
    def __init__(self, model_path: str = "models/yolov8m_robots.pt"):
        """
        Initialize video processor.
        
        Args:
            model_path: Path to YOLO model weights
        """
        # Prefer trained model if available; fall back to base weights
        chosen_model = model_path
        if not os.path.exists(chosen_model):
            fallback = "yolov8m.pt"
            print(f"Info: model '{chosen_model}' not found. Falling back to '{fallback}'.")
            chosen_model = fallback
        self.detector = RobotDetector(model_path=chosen_model)
        self.tracker = BoTSORTTracker(yolo_model=self.detector.model)
    
    def process_video(self, 
                     input_path: str, 
                     output_path: Optional[str] = None,
                     confidence_threshold: float = 0.5,
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> str:
        """
        Process a video file with detection and tracking.
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file (if None, saves to same dir with _tracked suffix)
            confidence_threshold: Minimum detection confidence
            progress_callback: Optional callback for progress updates (current_frame, total_frames)
        
        Returns:
            Path to the output video file
        """
        # Validate input file
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video file not found: {input_path}")
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Default output path - save to videos/tracked/ directory
        if output_path is None:
            input_stem = Path(input_path).stem
            # Use videos/tracked/ as output directory
            output_dir = Path("videos/tracked")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{input_stem}_tracked.mp4")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to create output video: {output_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                tracks = self.tracker.update(frame, confidence_threshold=confidence_threshold)

                # Draw tracked boxes on frame
                self._draw_tracks(frame, tracks)
                
                # Draw info overlay
                self._draw_overlay(frame, frame_count, total_frames, len(tracks))
                
                # Write frame
                out.write(frame)
                
                frame_count += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(frame_count, total_frames)
        
        finally:
            cap.release()
            out.release()
        
        return output_path
    
    def _draw_tracks(self, frame, tracks) -> None:
        """
        Draw bounding boxes and track IDs on frame.
        """
        # Track colors by ID for consistency
        track_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 165, 255),  # Orange
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 255, 0),  # Light green
            (255, 128, 0),  # Light blue
        ]
        
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.bbox_xyxy)

            # Assign color based on track ID for consistency
            color = track_colors[t.track_id % len(track_colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and confidence
            label = f"T{t.track_id} ({t.confidence:.2f})"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Note: Ultralytics tracker does not expose per-track history here.
    
    def _draw_overlay(self, frame, frame_num: int, total_frames: int, num_tracks: int) -> None:
        """Draw info overlay on frame."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background for info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (200, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Track count
        cv2.putText(frame, f"Tracking: {num_tracks} robots", (15, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def reset_tracker(self) -> None:
        """Reset the tracker for a new video."""
        self.tracker.reset()
