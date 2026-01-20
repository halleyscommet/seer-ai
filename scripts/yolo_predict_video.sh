#!/usr/bin/env sh
# Run YOLO detection on a video via the Ultralytics CLI.
# Example:
#   ./scripts/yolo_predict_video.sh --source videos/raw/foo.mp4 --model models/yolov8m_robots.pt

set -eu

SOURCE=""
MODEL="models/yolov8m_robots.pt"
CONF="0.5"
IMGSZ="640"
DEVICE="0"
NAME="predict"
PROJECT="runs_yolo"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --conf) CONF="$2"; shift 2;;
    --imgsz) IMGSZ="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --project) PROJECT="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 --source PATH [--model PATH] [--conf N] [--imgsz N] [--device ID] [--name NAME] [--project DIR]";
      exit 0;;
    *)
      echo "Unknown arg: $1" 1>&2
      exit 2;;
  esac
done

if [ -z "$SOURCE" ]; then
  echo "Missing --source (path to video)." 1>&2
  exit 2
fi

if ! command -v yolo >/dev/null 2>&1; then
  echo "Missing 'yolo' CLI. Install with: pip install ultralytics" 1>&2
  exit 127
fi

yolo detect predict \
  model="$MODEL" \
  source="$SOURCE" \
  conf="$CONF" \
  imgsz="$IMGSZ" \
  device="$DEVICE" \
  project="$PROJECT" \
  name="$NAME" \
  save=true
