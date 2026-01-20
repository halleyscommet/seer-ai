#!/usr/bin/env fish
# Validate a trained YOLO model via the Ultralytics CLI.
# Example:
#   ./scripts/yolo_val.fish --data dataset/combined_data.yaml --model runs_yolo/robots/weights/best.pt

set -e

argparse \
  'data=' \
  'model=' \
  'imgsz=' \
  'device=' \
  -- $argv
or begin
  echo "Failed to parse arguments." 1>&2
  exit 2
end

if not set -q _flag_data
  set _flag_data dataset/combined_data.yaml
end
if not set -q _flag_model
  set _flag_model models/yolov8m_robots.pt
end
if not set -q _flag_imgsz
  set _flag_imgsz 640
end
if not set -q _flag_device
  set _flag_device 0
end

command -q yolo
or begin
  echo "Missing 'yolo' CLI. Install with: pip install ultralytics" 1>&2
  exit 127
end

yolo detect val \
  data=$_flag_data \
  model=$_flag_model \
  imgsz=$_flag_imgsz \
  device=$_flag_device
