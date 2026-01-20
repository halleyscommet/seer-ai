#!/usr/bin/env sh
# Validate a trained YOLO model via the Ultralytics CLI.
# Example:
#   ./scripts/yolo_val.sh --data dataset/combined_data.yaml --model runs_yolo/robots/weights/best.pt

set -eu

DATA="dataset/combined_data.yaml"
MODEL="models/yolov8m_robots.pt"
IMGSZ="640"
DEVICE="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --data) DATA="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --imgsz) IMGSZ="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [--data PATH] [--model PATH] [--imgsz N] [--device ID]";
      exit 0;;
    *)
      echo "Unknown arg: $1" 1>&2
      exit 2;;
  esac
done

if ! command -v yolo >/dev/null 2>&1; then
  echo "Missing 'yolo' CLI. Install with: pip install ultralytics" 1>&2
  exit 127
fi

yolo detect val \
  data="$DATA" \
  model="$MODEL" \
  imgsz="$IMGSZ" \
  device="$DEVICE"
