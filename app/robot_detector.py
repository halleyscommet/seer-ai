"""
Robot detection integration using YOLO or other detector.
Converts detections to tracking-compatible format.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from .tracker import RobotDetection, BumperColorDetector, RobotNumberOCR

from ultralytics import YOLO

class RobotDetector:
    def __init__(self, model_path: str = "models/yolov8m_robots.pt"):
        self.model = YOLO(model_path)
    
    def _load_model(self, model_path: str) -> None:
        """
        Load YOLO model.
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError:
            print("Warning: ultralytics not installed. Install with: pip install ultralytics")
        except Exception as e:
            print(f"Warning: Failed to load YOLO model from {model_path}: {e}")
    
    def detect(self, frame, confidence_threshold: float = 0.5):
        results = self.model(frame, conf=confidence_threshold)
        detections = []
        
        for result in results:
            for box in result.boxes:
                if box.conf[0] >= confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0]
                    detections.append(RobotDetection(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        confidence=float(box.conf[0])
                    ))
        
        return detections

    @staticmethod
    def post_process_detections(detections: List[RobotDetection], 
                               frame: np.ndarray,
                               extract_ocr: bool = True,
                               detect_bumper_color: bool = True) -> List[RobotDetection]:
        """
        Post-process detections with OCR and color info.
        """
        for det in detections:
            # Try to extract robot number via OCR
            if extract_ocr:
                number = RobotNumberOCR.extract_number(frame, det)
                # Could store in det if we extend RobotDetection
            
            # Try to detect bumper color
            if detect_bumper_color:
                color = BumperColorDetector.detect_color(frame, det)
                # Could store in det if we extend RobotDetection
        
        return detections
