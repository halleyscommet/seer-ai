#!/usr/bin/env fish
# Copy the newest 'best.pt' from YOLO training runs into ./models/ for the UI.
# Example:
#   ./scripts/yolo_copy_best_to_models.fish
#   ./scripts/yolo_copy_best_to_models.fish --dest models/yolov8m_robots.pt

set -e

argparse 'runs=' 'dest=' -- $argv
or begin
  echo "Failed to parse arguments." 1>&2
  exit 2
end

if not set -q _flag_runs
  set _flag_runs runs_yolo
end
if not set -q _flag_dest
  set _flag_dest models/yolov8m_robots.pt
end

set -l best (find "$_flag_runs" -type f -path '*/weights/best.pt' -exec stat -f '%m %N' {} \; 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)

if test -z "$best"
  echo "No best.pt found under '$_flag_runs'." 1>&2
  echo "Expected something like: $_flag_runs/.../weights/best.pt" 1>&2
  exit 1
end

mkdir -p (dirname "$_flag_dest")
cp -f "$best" "$_flag_dest"

echo "Copied: $best"
echo "    -> $_flag_dest"
