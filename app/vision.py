from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2


@dataclass(frozen=True)
class Box:
    """
    Rectangle in pixel coordinates.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "object"

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def _haar_path() -> str:
    """
    Avoid cv2.data.* (often untyped). Compute path relative to cv2 package.
    """
    return os.path.join(
        os.path.dirname(cv2.__file__),
        "data",
        "haarcascade_frontalface_default.xml",
    )


class FaceDetector:
    """
    Simple OpenCV Haar cascade face detector.
    Good for testing integration; not intended as final “vision.”
    """
    def __init__(self) -> None:
        path = _haar_path()
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade at: {path}")

    def detect(self, bgr_frame) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        return list(faces)


def draw_boxes(bgr_frame, boxes_xywh: List[Tuple[int, int, int, int]]) -> None:
    for (x, y, w, h) in boxes_xywh:
        cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def map_boxes_frame_to_canvas(
    boxes_xywh: List[Tuple[int, int, int, int]],
    frame_size: Tuple[int, int],
    canvas_img_offset: Tuple[int, int],
    canvas_img_size: Tuple[int, int],
    label: str = "face",
) -> List[Box]:
    """
    Convert frame-space boxes to canvas-space boxes considering scale + letterboxing.

    frame_size: (frame_w, frame_h)
    canvas_img_offset: (x0, y0) where the image is drawn inside the canvas
    canvas_img_size: (draw_w, draw_h) of the image drawn inside the canvas
    """
    frame_w, frame_h = frame_size
    x0, y0 = canvas_img_offset
    draw_w, draw_h = canvas_img_size

    sx = draw_w / frame_w
    sy = draw_h / frame_h

    out: List[Box] = []
    for (fx, fy, fw, fh) in boxes_xywh:
        cx1 = x0 + fx * sx
        cy1 = y0 + fy * sy
        cx2 = x0 + (fx + fw) * sx
        cy2 = y0 + (fy + fh) * sy
        out.append(Box(cx1, cy1, cx2, cy2, label=label))
    return out

