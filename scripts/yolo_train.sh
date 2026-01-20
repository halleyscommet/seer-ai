#!/usr/bin/env sh
# Train a YOLOv8 detector via the Ultralytics CLI.
# Example:
#   ./scripts/yolo_train.sh --data dataset/combined_data.yaml --model yolov8m.pt --epochs 50 --name robots

set -eu

DATA="dataset/combined_data.yaml"
MODEL="yolov8m.pt"
EPOCHS="50"
IMGSZ="640"
BATCH="16"
DEVICE="0"
NAME="robots"
PROJECT="runs_yolo"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --data) DATA="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --imgsz) IMGSZ="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --project) PROJECT="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [--data PATH] [--model PATH] [--epochs N] [--imgsz N] [--batch N] [--device ID] [--name NAME] [--project DIR]";
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

yolo detect train \
  data="$DATA" \
  model="$MODEL" \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  batch="$BATCH" \
  device="$DEVICE" \
  project="$PROJECT" \
  name="$NAME"
