#!/usr/bin/env sh
# Competition laptop helper:
#  - starts the capture client (camera -> server)
#  - opens the viewer page in your browser
#
# Usage:
#   ./scripts/run_comp_laptop.sh --server 192.168.1.50 --camera 0
#   ./scripts/run_comp_laptop.sh --server example.com --port 8000 --fps 15 --width 640
#
# Notes:
# - This script runs the capture client until you Ctrl+C.
# - If STREAM_TOKEN is set on the server, pass --token or export STREAM_TOKEN locally.

set -eu

SERVER=""
PORT="8000"
CAMERA="0"
WIDTH="640"
FPS="20"
JPEG_QUALITY="75"
CONF="0.5"
TOKEN="${STREAM_TOKEN-}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --server) SERVER="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --camera) CAMERA="$2"; shift 2;;
    --width) WIDTH="$2"; shift 2;;
    --fps) FPS="$2"; shift 2;;
    --jpeg-quality) JPEG_QUALITY="$2"; shift 2;;
    --conf) CONF="$2"; shift 2;;
    --token) TOKEN="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 --server HOST [--port 8000] [--camera 0] [--width 640] [--fps 20] [--jpeg-quality 75] [--conf 0.5] [--token TOKEN]";
      exit 0;;
    *)
      echo "Unknown arg: $1" 1>&2
      exit 2;;
  esac
done

if [ -z "$SERVER" ]; then
  echo "Missing --server (IP or hostname of the GPU box)." 1>&2
  exit 2
fi

HTTP_URL="http://${SERVER}:${PORT}/"
WS_URL="ws://${SERVER}:${PORT}"

# Open browser best-effort.
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$HTTP_URL" >/dev/null 2>&1 &
elif command -v open >/dev/null 2>&1; then
  open "$HTTP_URL" >/dev/null 2>&1 &
elif command -v start >/dev/null 2>&1; then
  start "$HTTP_URL" >/dev/null 2>&1 &
else
  echo "Open this URL in a browser: $HTTP_URL"
fi

PY="python3"
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
fi

cleanup() {
  if [ -n "${CLIENT_PID-}" ]; then
    kill "$CLIENT_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup INT TERM EXIT

# Start capture client in background so the browser can open immediately.
if [ -n "$TOKEN" ]; then
  "$PY" web/capture_client.py --server "$WS_URL" --token "$TOKEN" --camera "$CAMERA" --width "$WIDTH" --fps "$FPS" --jpeg-quality "$JPEG_QUALITY" --conf "$CONF" &
else
  "$PY" web/capture_client.py --server "$WS_URL" --camera "$CAMERA" --width "$WIDTH" --fps "$FPS" --jpeg-quality "$JPEG_QUALITY" --conf "$CONF" &
fi
CLIENT_PID=$!

echo "Viewer:  $HTTP_URL"
echo "Capture: camera=$CAMERA width=$WIDTH fps=$FPS jpeg=$JPEG_QUALITY conf=$CONF"
wait "$CLIENT_PID"
