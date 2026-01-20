#!/usr/bin/env sh
# Track objects in a video using Ultralytics YOLO + BoT-SORT (or another tracker).
# Example:
#   ./scripts/yolo_track_video.sh --source videos/raw/foo.mp4 --model models/yolov8m_robots.pt
#   ./scripts/yolo_track_video.sh --source videos/raw/foo.mp4 --tracker botsort.yaml

set -eu

SOURCE=""
MODEL="models/yolov8m_robots.pt"
CONF="0.5"
IMGSZ="640"
DEVICE=""
DEVICE_SET="0"
TRACKER="botsort.yaml"
NAME="track"
PROJECT="runs_yolo"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --conf) CONF="$2"; shift 2;;
    --imgsz) IMGSZ="$2"; shift 2;;
    --device) DEVICE="$2"; DEVICE_SET="1"; shift 2;;
    --tracker) TRACKER="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --project) PROJECT="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 --source PATH [--model PATH] [--tracker botsort.yaml] [--conf N] [--imgsz N] [--device ID] [--name NAME] [--project DIR]";
      exit 0;;
    *)
      echo "Unknown arg: $1" 1>&2
      exit 2;;
  esac
done

if [ -n "${SEER_YOLO_DEVICE-}" ] && [ "$DEVICE_SET" = "0" ]; then
  DEVICE="$SEER_YOLO_DEVICE"
fi

if [ -z "$DEVICE" ] && [ "$DEVICE_SET" = "0" ]; then
  OS_NAME="$(uname)"
  if [ "$OS_NAME" = "Darwin" ]; then
    ARCH="$(uname -m)"
    if [ "$ARCH" = "arm64" ]; then
      DEVICE="mps"
    else
      DEVICE="cpu"
    fi
  else
    DEVICE="0"
  fi
fi

if [ -z "$SOURCE" ]; then
  echo "Missing --source (path to video)." 1>&2
  exit 2
fi

if ! command -v yolo >/dev/null 2>&1; then
  echo "Missing 'yolo' CLI. Install with: pip install ultralytics" 1>&2
  exit 127
fi

yolo track \
  model="$MODEL" \
  source="$SOURCE" \
  conf="$CONF" \
  imgsz="$IMGSZ" \
  device="$DEVICE" \
  tracker="$TRACKER" \
  project="$PROJECT" \
  name="$NAME" \
  save=true
