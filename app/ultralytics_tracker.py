from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float


class BoTSORTTracker:
    """BoT-SORT tracking via Ultralytics.

    Uses `YOLO.track(..., persist=True, tracker=...)` so IDs persist across frames.
    """

    def __init__(self, yolo_model, tracker_yaml: str = "botsort.yaml") -> None:
        self._model = yolo_model
        self._tracker_yaml = tracker_yaml

    @property
    def tracker_yaml(self) -> str:
        return self._tracker_yaml

    def reset(self) -> None:
        """Best-effort reset of tracker state.

        Ultralytics stores tracker state in the model predictor; the most reliable
        reset is to create a new YOLO model instance. As a lightweight option, we
        clear the predictor if present.
        """

        predictor = getattr(self._model, "predictor", None)
        if predictor is not None:
            try:
                # Different ultralytics versions store trackers differently; try common fields.
                if hasattr(predictor, "trackers"):
                    predictor.trackers = None
                if hasattr(predictor, "tracker"):
                    predictor.tracker = None
            except Exception:
                pass

    def update(self, frame, confidence_threshold: float = 0.5, imgsz: int = 640, device: Optional[str] = None) -> List[TrackedObject]:
        """Run tracking on a single frame and return tracked detections."""

        kwargs = {
            "conf": confidence_threshold,
            "persist": True,
            "tracker": self._tracker_yaml,
            "imgsz": imgsz,
            "verbose": False,
        }
        if device is not None:
            kwargs["device"] = device

        results = self._model.track(frame, **kwargs)
        if not results:
            return []

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy
        conf = boxes.conf
        ids = getattr(boxes, "id", None)

        # Convert tensors/arrays to python lists without taking a hard dependency on torch.
        try:
            xyxy_list = xyxy.cpu().numpy().tolist()
        except Exception:
            xyxy_list = xyxy.tolist()

        try:
            conf_list = conf.cpu().numpy().tolist()
        except Exception:
            conf_list = conf.tolist()

        if ids is None:
            id_list: List[Optional[int]] = [None] * len(xyxy_list)
        else:
            try:
                id_list = ids.cpu().numpy().astype(int).tolist()
            except Exception:
                # Some versions return floats; cast best-effort.
                id_list = [int(v) if v is not None else None for v in ids.tolist()]

        out: List[TrackedObject] = []
        for bbox, c, tid in zip(xyxy_list, conf_list, id_list):
            if tid is None:
                continue
            x1, y1, x2, y2 = bbox
            out.append(
                TrackedObject(
                    track_id=int(tid),
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(c),
                )
            )

        return out
