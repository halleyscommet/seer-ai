#!/usr/bin/env fish
# Train a YOLOv8 detector via the Ultralytics CLI.
# Example:
#   ./scripts/yolo_train.fish --data dataset/combined_data.yaml --model yolov8m.pt --epochs 50 --name robots

set -e

argparse \
  'data=' \
  'model=' \
  'epochs=' \
  'imgsz=' \
  'batch=' \
  'device=' \
  'name=' \
  'project=' \
  -- $argv
or begin
  echo "Failed to parse arguments." 1>&2
  exit 2
end

if not set -q _flag_data
  set _flag_data dataset/combined_data.yaml
end
if not set -q _flag_model
  set _flag_model yolov8m.pt
end
if not set -q _flag_epochs
  set _flag_epochs 50
end
if not set -q _flag_imgsz
  set _flag_imgsz 640
end
if not set -q _flag_batch
  set _flag_batch 16
end
if not set -q _flag_device
  set _flag_device 0
end
if not set -q _flag_name
  set _flag_name robots
end
if not set -q _flag_project
  set _flag_project runs_yolo
end

command -q yolo
or begin
  echo "Missing 'yolo' CLI. Install with: pip install ultralytics" 1>&2
  exit 127
end

echo "Training: data=$_flag_data model=$_flag_model epochs=$_flag_epochs imgsz=$_flag_imgsz batch=$_flag_batch device=$_flag_device"

yolo detect train \
  data=$_flag_data \
  model=$_flag_model \
  epochs=$_flag_epochs \
  imgsz=$_flag_imgsz \
  batch=$_flag_batch \
  device=$_flag_device \
  project=$_flag_project \
  name=$_flag_name
