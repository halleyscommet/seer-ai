# YOLO CLI scripts

These wrappers standardize how this repo trains/validates/predicts using the Ultralytics `yolo` CLI.

## Train

Fish:

- `./scripts/yolo_train.fish --data dataset/combined_data.yaml --model yolov8m.pt --epochs 50 --name robots`

Sh:

- `./scripts/yolo_train.sh --data dataset/combined_data.yaml --model yolov8m.pt --epochs 50 --name robots`

Outputs go under `runs_yolo/` by default.

## Validate

- `./scripts/yolo_val.(fish|sh) --data dataset/combined_data.yaml --model runs_yolo/robots/weights/best.pt`

## Predict on video

- `./scripts/yolo_predict_video.(fish|sh) --source videos/raw/your_video.mp4 --model models/yolov8m_robots.pt`

## Track on video (BoT-SORT)

- `./scripts/yolo_track_video.(fish|sh) --source videos/raw/your_video.mp4 --model models/yolov8m_robots.pt`
- Optional: `--tracker botsort.yaml` (default) or another Ultralytics tracker yaml

## Copy newest trained weights into UI path

- `./scripts/yolo_copy_best_to_models.(fish|sh)`

This copies the newest `best.pt` found under `runs_yolo/` to `models/yolov8m_robots.pt` so the UI picks it up.
