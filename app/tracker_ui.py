"""
UI integration utilities for robot tracking.
Provides drawing functions and layout helpers.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Tuple, Optional

from .tracker import RobotTrack


def draw_robot_tracks(frame: np.ndarray, 
                      tracks: Dict[int, RobotTrack],
                      show_trails: bool = True,
                      show_ids: bool = True,
                      show_info: bool = True) -> None:
    """
    Draw tracked robots on frame with trails and metadata.
    
    Args:
        frame: BGR image to draw on (modified in place)
        tracks: dict of {track_id -> RobotTrack}
        show_trails: Draw position history
        show_ids: Draw track IDs
        show_info: Draw team/color info
    """
    colors = {
        0: (255, 0, 0),    # Blue (BGR)
        1: (0, 0, 255),    # Red
        2: (0, 165, 255),  # Orange
        3: (255, 255, 0),  # Cyan
        4: (255, 0, 255),  # Magenta
        5: (0, 255, 255),  # Yellow
    }
    
    for track_id, track in tracks.items():
        if not track.positions:
            continue
        
        color = colors.get(track.temp_team_id if track.temp_team_id is not None else track_id % 6, 
                          (200, 200, 200))
        
        # Draw trail
        if show_trails and len(track.positions) > 1:
            points = np.array(track.positions, dtype=np.int32)
            # Draw fading line
            for i in range(1, len(points)):
                alpha = i / len(points)
                pt_color = tuple(int(c * alpha + (1 - alpha) * 200) for c in color)
                cv2.line(frame, tuple(points[i-1]), tuple(points[i]), pt_color, 1)
        
        # Draw current position
        x, y = track.positions[-1]
        x, y = int(x), int(y)
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
        
        # Draw ID
        if show_ids:
            label = f"#{track.temp_team_id}" if track.temp_team_id is not None else f"T{track_id}"
            cv2.putText(frame, label, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
        
        # Draw extra info
        if show_info:
            info_lines = []
            if track.detected_number:
                info_lines.append(f"#{track.detected_number}")
            if track.bumper_color:
                info_lines.append(track.bumper_color)
            
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (x - 20, y + 15 + i * 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def draw_detection_stats(frame: np.ndarray, num_tracks: int, frame_count: int) -> None:
    """
    Draw FPS and tracking stats in corner of frame.
    """
    h, w = frame.shape[:2]
    text = f"Tracked: {num_tracks}/6 | Frame: {frame_count}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def get_robot_id_color(team_id: Optional[int]) -> str:
    """
    Get alliance color string for a robot temp ID.
    IDs 0-2: Red alliance, IDs 3-5: Blue alliance
    """
    if team_id is None:
        return "unknown"
    return "red" if team_id < 3 else "blue"
