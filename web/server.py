from __future__ import annotations

import asyncio
import base64
import os
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import Config
from app.ultralytics_tracker import BoTSORTTracker
from ultralytics import YOLO


def _b64_to_frame(jpg_b64: str):
    data = base64.b64decode(jpg_b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _frame_to_b64_jpg(frame, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _draw_tracks(frame, tracks) -> None:
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID:{t.track_id} {t.confidence:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


app = FastAPI(title="Seer Remote Tracker")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index() -> HTMLResponse:
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


def _require_token(websocket: WebSocket) -> bool:
    expected = os.getenv("STREAM_TOKEN")
    if not expected:
        return True
    token = websocket.query_params.get("token")
    return token == expected


class ViewerHub:
    def __init__(self) -> None:
        self._viewers: set[WebSocket] = set()

    async def add(self, ws: WebSocket) -> None:
        self._viewers.add(ws)

    def remove(self, ws: WebSocket) -> None:
        self._viewers.discard(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._viewers):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.remove(ws)


hub = ViewerHub()


@app.websocket("/ws/view")
async def ws_view(websocket: WebSocket):
    """Browser viewers connect here to receive the latest annotated frames."""

    if not _require_token(websocket):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    await hub.add(websocket)
    try:
        while True:
            # Keep the socket alive; viewers don't need to send anything.
            await websocket.receive_text()
    except WebSocketDisconnect:
        hub.remove(websocket)


@app.websocket("/ws/ingest")
async def ws_ingest(websocket: WebSocket):
    """Capture client connects here and pushes frames.

    Client sends JSON: {type:'frame', jpg_b64:'...', conf:0.5}
    Server broadcasts to all viewers: {type:'result', jpg_b64:'...', tracks:[...]}
    """

    if not _require_token(websocket):
        await websocket.close(code=1008)
        return

    await websocket.accept()

    cfg = Config.get()
    model_path = cfg.model_path_preferred
    if not os.path.exists(model_path):
        model_path = cfg.model_path_fallback

    model = YOLO(model_path)
    tracker = BoTSORTTracker(model, tracker_yaml=cfg.ultralytics_tracker_yaml)
    device = cfg.ultralytics_device

    try:
        while True:
            msg = await websocket.receive_json()
            if not isinstance(msg, dict):
                continue

            msg_type = msg.get("type")
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type != "frame":
                await websocket.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})
                continue

            jpg_b64 = msg.get("jpg_b64")
            if not isinstance(jpg_b64, str) or not jpg_b64:
                await websocket.send_json({"type": "error", "message": "Missing jpg_b64"})
                continue

            conf = msg.get("conf")
            if not isinstance(conf, (int, float)):
                conf = float(cfg.detection_conf_live)

            frame = _b64_to_frame(jpg_b64)
            if frame is None:
                await websocket.send_json({"type": "error", "message": "Failed to decode frame"})
                continue

            tracks = tracker.update(frame, confidence_threshold=float(conf), device=device)
            _draw_tracks(frame, tracks)

            tracks_json = [
                {
                    "id": int(t.track_id),
                    "x1": float(t.bbox_xyxy[0]),
                    "y1": float(t.bbox_xyxy[1]),
                    "x2": float(t.bbox_xyxy[2]),
                    "y2": float(t.bbox_xyxy[3]),
                    "conf": float(t.confidence),
                }
                for t in tracks
            ]

            out_jpg_b64 = _frame_to_b64_jpg(frame)
            payload = {"type": "result", "jpg_b64": out_jpg_b64, "tracks": tracks_json}
            await hub.broadcast(payload)

            # Small ack so client can measure RTT / drop rate.
            await websocket.send_json({"type": "ack"})

            await asyncio.sleep(0)
    except WebSocketDisconnect:
        return
