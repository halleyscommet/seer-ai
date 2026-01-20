"""
Multi-object tracker for FRC robots with ID persistence.
Maintains 6 temp IDs and uses spatial proximity, OCR, and color matching to prevent ID swaps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import uuid

import cv2
import numpy as np


@dataclass
class RobotDetection:
    """A single robot detection from the model."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    label: str = "robot"

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def iou(self, other: RobotDetection) -> float:
        """Compute intersection-over-union with another detection."""
        x1_inter = max(self.x1, other.x1)
        y1_inter = max(self.y1, other.y1)
        x2_inter = min(self.x2, other.x2)
        y2_inter = min(self.y2, other.y2)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        self_area = self.width * self.height
        other_area = other.width * other.height
        union_area = self_area + other_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class RobotTrack:
    """Persistent track for a single robot."""
    track_id: int
    temp_team_id: Optional[int] = None  # 0-5 for 6 robots
    detected_number: Optional[int] = None  # From OCR on bumper
    bumper_color: Optional[str] = None  # "red", "blue", or None
    alliance_color: Optional[str] = None  # "red" or "blue" from field analysis
    
    # Spatial history
    positions: List[Tuple[float, float]] = field(default_factory=list)
    last_detection_time: float = field(default_factory=time.time)
    confidence_history: List[float] = field(default_factory=list)
    last_bbox: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2)
    
    # Tracking metadata
    age: int = 0  # frames since creation
    consecutive_misses: int = 0
    creation_time: float = field(default_factory=time.time)

    def update(self, detection: RobotDetection) -> None:
        """Update track with a new detection."""
        self.positions.append(detection.center)
        self.confidence_history.append(detection.confidence)
        self.last_detection_time = time.time()
        self.last_bbox = (detection.x1, detection.y1, detection.x2, detection.y2)
        self.consecutive_misses = 0
        self.age += 1
        
        # Keep only last 60 positions (2 sec at 30fps)
        if len(self.positions) > 60:
            self.positions.pop(0)
        if len(self.confidence_history) > 60:
            self.confidence_history.pop(0)

    def predict_position(self) -> Optional[Tuple[float, float]]:
        """Predict next position using velocity if we have history."""
        if len(self.positions) < 2:
            return self.positions[-1] if self.positions else None
        
        # Use last 5 positions to estimate velocity
        recent = self.positions[-5:]
        vx = sum(recent[i + 1][0] - recent[i][0] for i in range(len(recent) - 1)) / (len(recent) - 1)
        vy = sum(recent[i + 1][1] - recent[i][1] for i in range(len(recent) - 1)) / (len(recent) - 1)
        
        last_x, last_y = self.positions[-1]
        return (last_x + vx, last_y + vy)

    def get_avg_confidence(self) -> float:
        """Get average confidence of recent detections."""
        return sum(self.confidence_history) / len(self.confidence_history) if self.confidence_history else 0.0


class RobotTracker:
    """
    Main tracker: maintains up to 6 robot tracks with ID persistence.
    Uses Hungarian algorithm for optimal assignment and supports soft constraints from OCR/color.
    Handles occlusions by keeping "lost" tracks and re-associating them.
    """
    
    def __init__(self, max_robots: int = 6, max_age: int = 30, max_misses: int = 60):
        self.max_robots = max_robots
        self.max_age = max_age
        self.max_misses = max_misses  # ~2 seconds at 30fps before truly losing track
        
        self.tracks: Dict[int, RobotTrack] = {}
        self.lost_tracks: Dict[int, RobotTrack] = {}  # Tracks that disappeared but might come back
        self.next_track_id = 1
        self.frame_count = 0
        self.max_lost_age = 150  # Keep lost tracks for ~5 seconds at 30fps

    def update(self, detections: List[RobotDetection]) -> Dict[int, RobotTrack]:
        """
        Update tracker with new detections.
        Returns: dict of {track_id -> RobotTrack} for visible robots.
        """
        self.frame_count += 1
        
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)
        
        # Update matched tracks
        for track_id, det_idx in matched:
            self.tracks[track_id].update(detections[det_idx])
        
        # Try to re-associate unmatched detections with lost tracks first
        still_unmatched_dets = []
        for det_idx in unmatched_dets:
            reassociated = self._try_reassociate_lost(detections[det_idx])
            if not reassociated:
                still_unmatched_dets.append(det_idx)
        
        # Only create new tracks if we haven't hit max_robots and couldn't reuse an ID
        for det_idx in still_unmatched_dets:
            # Count total active tracks (including those briefly missing)
            active_count = len(self.tracks)
            if active_count < self.max_robots:
                # Reuse lowest available ID if possible
                new_id = self._get_next_available_id()
                new_track = RobotTrack(track_id=new_id)
                new_track.update(detections[det_idx])
                self.tracks[new_id] = new_track
        
        # Mark unmatched tracks as missed
        for track_id in unmatched_tracks:
            self.tracks[track_id].consecutive_misses += 1
            self.tracks[track_id].age += 1
        
        # Move tracks with too many misses to lost_tracks (don't delete immediately)
        newly_lost = [tid for tid, track in self.tracks.items() 
                      if track.consecutive_misses > self.max_misses]
        for tid in newly_lost:
            self.lost_tracks[tid] = self.tracks.pop(tid)
        
        # Age and clean up lost tracks
        dead_lost = []
        for tid, track in self.lost_tracks.items():
            track.consecutive_misses += 1
            if track.consecutive_misses > self.max_lost_age:
                dead_lost.append(tid)
        for tid in dead_lost:
            del self.lost_tracks[tid]
        
        return self.tracks
    
    def _try_reassociate_lost(self, detection: RobotDetection) -> bool:
        """
        Try to match a detection with a lost track.
        Returns True if successfully reassociated.
        """
        if not self.lost_tracks:
            return False
        
        det_x, det_y = detection.center
        best_tid = None
        best_dist = float('inf')
        
        # Find closest lost track (with larger search radius since they've been gone)
        for tid, track in self.lost_tracks.items():
            if not track.positions:
                continue
            
            # Use predicted position if possible
            pred_pos = track.predict_position()
            if pred_pos:
                pred_x, pred_y = pred_pos
            else:
                pred_x, pred_y = track.positions[-1]
            
            dist = np.sqrt((det_x - pred_x) ** 2 + (det_y - pred_y) ** 2)
            
            # Larger search radius for lost tracks (they could have moved during occlusion)
            max_dist = 150 + track.consecutive_misses * 2  # Grows over time
            
            if dist < max_dist and dist < best_dist:
                best_dist = dist
                best_tid = tid
        
        if best_tid is not None:
            # Reactivate the lost track
            track = self.lost_tracks.pop(best_tid)
            track.update(detection)
            self.tracks[best_tid] = track
            return True
        
        return False
    
    def _get_next_available_id(self) -> int:
        """
        Get the next available track ID, preferring to reuse IDs 1-6.
        """
        # IDs currently in use
        used_ids = set(self.tracks.keys()) | set(self.lost_tracks.keys())
        
        # Try to find an unused ID in the 1-6 range first
        for i in range(1, self.max_robots + 1):
            if i not in used_ids:
                return i
        
        # Otherwise use next sequential ID
        self.next_track_id = max(self.next_track_id, max(used_ids, default=0) + 1)
        return self.next_track_id

    def _match_detections(self, detections: List[RobotDetection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks using a greedy approach with spatial proximity.
        Returns: (matched_pairs, unmatched_detection_indices, unmatched_track_ids)
        """
        if not detections or not self.tracks:
            return [], list(range(len(detections))), list(self.tracks.keys())
        
        matched = []
        unmatched_dets = set(range(len(detections)))
        unmatched_track_ids = set(self.tracks.keys())
        
        # Build cost matrix: distance between predicted track position and detection
        cost_matrix = np.full((len(self.tracks), len(detections)), float('inf'))
        track_ids = list(self.tracks.keys())
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            pred_pos = track.predict_position()
            
            if pred_pos is None:
                continue
            
            for j, det in enumerate(detections):
                det_x, det_y = det.center
                pred_x, pred_y = pred_pos
                
                # Euclidean distance
                dist = np.sqrt((det_x - pred_x) ** 2 + (det_y - pred_y) ** 2)
                
                # Penalize distance more for older tracks (more confident)
                # Newer tracks get bigger search radius
                max_dist = 50 + 20 * (1 - track.get_avg_confidence())
                
                if dist < max_dist:
                    cost_matrix[i, j] = dist
        
        # Greedy matching: pair closest matches first
        while cost_matrix.min() < float('inf'):
            i, j = np.unravel_index(cost_matrix.argmin(), cost_matrix.shape)
            cost = cost_matrix[i, j]
            
            track_id = track_ids[i]
            matched.append((track_id, j))
            
            unmatched_dets.discard(j)
            unmatched_track_ids.discard(track_id)
            
            # Mark this row/col as used
            cost_matrix[i, :] = float('inf')
            cost_matrix[:, j] = float('inf')
        
        return matched, list(unmatched_dets), list(unmatched_track_ids)

    def set_robot_attributes(self, track_id: int, 
                            temp_team_id: Optional[int] = None,
                            detected_number: Optional[int] = None,
                            bumper_color: Optional[str] = None,
                            alliance_color: Optional[str] = None) -> None:
        """
        Set attributes from OCR/color detection to aid in ID matching.
        """
        if track_id in self.tracks:
            track = self.tracks[track_id]
            if temp_team_id is not None:
                track.temp_team_id = temp_team_id
            if detected_number is not None:
                track.detected_number = detected_number
            if bumper_color is not None:
                track.bumper_color = bumper_color
            if alliance_color is not None:
                track.alliance_color = alliance_color

    def get_track_positions(self) -> Dict[int, Tuple[float, float]]:
        """Get current positions of all active tracks."""
        return {tid: track.positions[-1] for tid, track in self.tracks.items() 
                if track.positions}

    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """Get full info about a track."""
        if track_id not in self.tracks:
            return None
        
        track = self.tracks[track_id]
        return {
            'track_id': track_id,
            'temp_team_id': track.temp_team_id,
            'detected_number': track.detected_number,
            'bumper_color': track.bumper_color,
            'alliance_color': track.alliance_color,
            'position': track.positions[-1] if track.positions else None,
            'age': track.age,
            'confidence': track.get_avg_confidence(),
            'time_since_detection': time.time() - track.last_detection_time,
        }

    def reset(self) -> None:
        """Clear all tracks."""
        self.tracks.clear()
        self.lost_tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0


class BumperColorDetector:
    """
    Detect bumper color (red/blue) in a robot detection region.
    FRC bumpers are brightly colored.
    """
    
    @staticmethod
    def detect_color(frame: np.ndarray, detection: RobotDetection) -> Optional[str]:
        """
        Detect bumper color from a detection region.
        Returns: "red", "blue", or None if unclear.
        """
        # Extract ROI with some padding
        x1, y1, x2, y2 = int(detection.x1), int(detection.y1), int(detection.x2), int(detection.y2)
        h, w = frame.shape[:2]
        
        x1 = max(0, x1 - 5)
        y1 = max(0, y1 - 5)
        x2 = min(w, x2 + 5)
        y2 = min(h, y2 + 5)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Red hue range (wraps around 0-180 in OpenCV HSV)
        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Blue hue range
        blue_mask = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
        
        red_pixels = cv2.countNonZero(red_mask)
        blue_pixels = cv2.countNonZero(blue_mask)
        
        # Return dominant color if significant
        if red_pixels > blue_pixels and red_pixels > 100:
            return "red"
        elif blue_pixels > red_pixels and blue_pixels > 100:
            return "blue"
        
        return None


class RobotNumberOCR:
    """
    Placeholder for OCR on robot bumper numbers.
    In practice, use pytesseract or a fine-tuned detector.
    """
    
    @staticmethod
    def extract_number(frame: np.ndarray, detection: RobotDetection) -> Optional[int]:
        """
        Extract robot number from bumper via OCR.
        Placeholder: returns None.
        TODO: Implement with pytesseract or CRAFT + CRNN
        """
        # This is where you'd run pytesseract on the bumper region
        # For now, return None
        return None
