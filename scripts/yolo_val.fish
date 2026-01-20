#!/usr/bin/env fish
# Validate a trained YOLO model via the Ultralytics CLI.
# Example:
#   ./scripts/yolo_val.fish --data dataset/combined_data.yaml --model runs_yolo/robots/weights/best.pt

set -l _usage "Usage: ./scripts/yolo_val.fish [--data PATH] [--model PATH] [--imgsz N] [--device cpu|mps|0]"

function _seer_default_device
  if set -q SEER_YOLO_DEVICE
    echo $SEER_YOLO_DEVICE
    return
  end

  set os (uname)
  if test "$os" = "Darwin"
    set arch (uname -m)
    if test "$arch" = "arm64"
      echo mps
    else
      echo cpu
    end
    return
  end

  echo 0
end

argparse \
  'h/help' \
  'data=' \
  'model=' \
  'imgsz=' \
  'device=' \
  -- $argv
or begin
  echo "Failed to parse arguments." 1>&2
  exit 2
end

if set -q _flag_help
  echo $_usage
  exit 0
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
  set _flag_device (_seer_default_device)
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
