#!/usr/bin/env fish
# Run YOLO detection on a video via the Ultralytics CLI.
# Example:
#   ./scripts/yolo_predict_video.fish --source videos/raw/foo.mp4 --model models/yolov8m_robots.pt

set -e

argparse \
  'source=' \
  'model=' \
  'conf=' \
  'imgsz=' \
  'device=' \
  'name=' \
  'project=' \
  -- $argv
or begin
  echo "Failed to parse arguments." 1>&2
  exit 2
end

if not set -q _flag_source
  echo "Missing --source (path to video)." 1>&2
  exit 2
end
if not set -q _flag_model
  set _flag_model models/yolov8m_robots.pt
end
if not set -q _flag_conf
  set _flag_conf 0.5
end
if not set -q _flag_imgsz
  set _flag_imgsz 640
end
if not set -q _flag_device
  set _flag_device 0
end
if not set -q _flag_name
  set _flag_name predict
end
if not set -q _flag_project
  set _flag_project runs_yolo
end

command -q yolo
or begin
  echo "Missing 'yolo' CLI. Install with: pip install ultralytics" 1>&2
  exit 127
end

yolo detect predict \
  model=$_flag_model \
  source=$_flag_source \
  conf=$_flag_conf \
  imgsz=$_flag_imgsz \
  device=$_flag_device \
  project=$_flag_project \
  name=$_flag_name \
  save=true
