from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
from typing import Optional

import cv2
import websockets


def frame_to_b64_jpg(frame, quality: int = 75) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_url(server: str, token: Optional[str]) -> str:
    server = server.rstrip("/")
    url = f"{server}/ws/ingest"
    if token:
        url += f"?token={token}"
    return url


async def run(
    server: str,
    camera: int,
    width: int,
    jpeg_quality: int,
    conf: float,
    fps_limit: float,
    token: Optional[str],
) -> None:
    url = build_url(server, token)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera}")

    try:
        async with websockets.connect(url, max_size=32 * 1024 * 1024) as ws:
            print(f"Connected ingest -> {url}")
            last_send = 0.0
            sent = 0
            t0 = time.time()

            while True:
                ok, frame = cap.read()
                if not ok:
                    await asyncio.sleep(0.01)
                    continue

                if width > 0:
                    h, w = frame.shape[:2]
                    if w != width:
                        new_h = int(h * (width / w))
                        frame = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)

                now = time.time()
                if fps_limit > 0:
                    min_dt = 1.0 / fps_limit
                    if now - last_send < min_dt:
                        await asyncio.sleep(max(0.0, min_dt - (now - last_send)))
                        continue

                jpg_b64 = frame_to_b64_jpg(frame, quality=jpeg_quality)
                await ws.send(
                    json.dumps(
                        {
                            "type": "frame",
                            "jpg_b64": jpg_b64,
                            "conf": conf,
                        }
                    )
                )

                # Expect small ack; if it doesn't arrive, we still keep going.
                try:
                    _ = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except Exception:
                    pass

                last_send = time.time()
                sent += 1
                if sent % 30 == 0:
                    dt = time.time() - t0
                    rate = sent / dt if dt > 0 else 0
                    print(f"sent={sent} avg_fps={rate:.1f}")

    finally:
        cap.release()


def main() -> int:
    p = argparse.ArgumentParser(description="Send camera frames to Seer remote tracker server")
    p.add_argument("--server", default="ws://127.0.0.1:8000", help="Server base URL, e.g. ws://1.2.3.4:8000")
    p.add_argument("--token", default=os.getenv("STREAM_TOKEN"), help="Optional STREAM_TOKEN")
    p.add_argument("--camera", type=int, default=0, help="Camera index")
    p.add_argument("--width", type=int, default=640, help="Resize frames to this width (0 disables)")
    p.add_argument("--jpeg-quality", type=int, default=75, help="JPEG quality 1-100")
    p.add_argument("--conf", type=float, default=0.5, help="Detection confidence")
    p.add_argument("--fps", type=float, default=20.0, help="Max FPS to send (0 disables)")
    args = p.parse_args()

    asyncio.run(
        run(
            server=args.server,
            camera=args.camera,
            width=args.width,
            jpeg_quality=args.jpeg_quality,
            conf=args.conf,
            fps_limit=args.fps,
            token=args.token,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
