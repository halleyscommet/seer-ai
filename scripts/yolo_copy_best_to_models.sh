#!/usr/bin/env sh
# Copy the newest 'best.pt' from YOLO training runs into ./models/ for the UI.
# Example:
#   ./scripts/yolo_copy_best_to_models.sh
#   ./scripts/yolo_copy_best_to_models.sh --runs runs_yolo --dest models/yolov8m_robots.pt

set -eu

RUNS="runs_yolo"
DEST="models/yolov8m_robots.pt"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [--runs DIR] [--dest PATH]";
      exit 0;;
    *)
      echo "Unknown arg: $1" 1>&2
      exit 2;;
  esac
done

BEST=$(find "$RUNS" -type f -path '*/weights/best.pt' -exec stat -f '%m %N' {} \; 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)

if [ -z "$BEST" ]; then
  echo "No best.pt found under '$RUNS'." 1>&2
  echo "Expected something like: $RUNS/.../weights/best.pt" 1>&2
  exit 1
fi

mkdir -p "$(dirname "$DEST")"
cp -f "$BEST" "$DEST"

echo "Copied: $BEST"
echo "    -> $DEST"
