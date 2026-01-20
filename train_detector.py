"""
TRAINING WORKFLOW: From FRC Video to Trained Robot Detector
============================================================

Complete guide to train your own robot detector on actual FRC footage.
"""

# STEP 1: COLLECT VIDEO FOOTAGE
# ================================

# When Week 0 arrives:
# 1. Record 5-10 matches (3-4 min each)
# 2. Use your camera/phone at good angles (side view best for seeing bumpers)
# 3. Vary: lighting, teams, field positions, distance
# 4. Save as .mp4 or .mov files in: videos/raw/

# Example structure:
# videos/
# ├── raw/
# │   ├── week0_qual1.mp4
# │   ├── week0_qual2.mp4
# │   └── ...
# └── frames/
#     └── (will create during extraction)


# STEP 2: EXTRACT FRAMES FROM VIDEO
# ==================================

import cv2
import os
from pathlib import Path

def extract_frames_from_video(video_path: str, output_dir: str, frame_skip: int = 5) -> int:
    """
    Extract frames from video at intervals.
    
    Args:
        video_path: Path to video file
        output_dir: Where to save frames
        frame_skip: Extract every Nth frame (5 = ~6fps from 30fps video)
    
    Returns:
        Number of frames extracted
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")
    
    frame_count = 0
    saved_count = 0
    video_name = Path(video_path).stem
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            filename = f"{output_dir}/{video_name}_frame_{saved_count:04d}.jpg"
            cv2.imwrite(filename, frame)
            saved_count += 1
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames, saved {saved_count}...", end='\r')
    
    cap.release()
    print(f"  Processed {frame_count} frames, saved {saved_count} frames ✓")
    return saved_count


# SCRIPT: Extract all videos
# ==========================

def extract_all_videos():
    """Extract frames from all videos in videos/raw/"""
    video_dir = "videos/raw"
    output_dir = "videos/frames"
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_extracted = 0
    for video_file in sorted(Path(video_dir).glob("*.mp4")):
        extracted = extract_frames_from_video(str(video_file), output_dir, frame_skip=5)
        total_extracted += extracted
    
    print(f"\nTotal frames extracted: {total_extracted}")
    print(f"Location: {output_dir}/")
    return output_dir


# Run extraction:
# ===============
# if __name__ == "__main__":
#     extract_all_videos()


# STEP 3: ANNOTATE FRAMES WITH BOUNDING BOXES
# =============================================

# Tools available (pick one):

# Option A: LabelImg (Free, GUI, simple)
# Install: pip install labelImg
# Run: labelimg videos/frames/ --default-save-dir dataset/annotations/
# Tutorial: https://github.com/heartexlabs/labelImg

# Option B: Roboflow (Web-based, free tier)
# 1. Create account at https://roboflow.com/
# 2. Create new project (Object Detection)
# 3. Upload frames: videos/frames/
# 4. Label all robots with class "robot"
# 5. Export as YOLOv8 format
# 6. Download: dataset.yaml + images/ + labels/

# Option C: CVAT (Complex, but powerful for team use)
# https://www.cvat.ai/

# RECOMMENDATION: Use Roboflow for first training
# - Easy web UI
# - Automatic data augmentation
# - Direct YOLOv8 export


# STEP 4: ORGANIZE DATASET IN YOLOV8 FORMAT
# ===========================================

# After annotation, your structure should be:

# dataset/
# ├── images/
# │   ├── train/         (70% of images)
# │   ├── val/           (15% of images)
# │   └── test/          (15% of images)
# ├── labels/
# │   ├── train/         (corresponding .txt files)
# │   ├── val/
# │   └── test/
# └── data.yaml

# Example data.yaml:
"""
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 1
names: ['robot']
"""

def verify_dataset_structure(dataset_dir: str) -> bool:
    """Verify YOLOv8 dataset is properly formatted using data.yaml.

    Supports both layouts:
      - images/train + labels/train
      - train/images + train/labels
    """
    import yaml

    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    if not os.path.exists(yaml_path):
        print("❌ Missing: data.yaml")
        return False

    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    ok = True
    for split_key in ('train', 'val', 'test'):
        split_images = cfg.get(split_key)
        if not split_images:
            print(f"❌ data.yaml missing key: {split_key}")
            ok = False
            continue

        # Resolve images path relative to dataset_dir if not absolute
        images_path = split_images
        if not os.path.isabs(images_path):
            images_path = os.path.normpath(os.path.join(dataset_dir, images_path))

        # Derive labels path by replacing 'images' with 'labels'
        if 'images' in split_images:
            labels_rel = split_images.replace('images', 'labels')
        else:
            # If path is like 'train/images', ensure 'train/labels'
            labels_rel = split_images.replace('/images', '/labels')
        labels_path = labels_rel
        if not os.path.isabs(labels_path):
            labels_path = os.path.normpath(os.path.join(dataset_dir, labels_path))

        if not os.path.isdir(images_path):
            print(f"❌ Missing images dir: {os.path.relpath(images_path, dataset_dir)}")
            ok = False
        else:
            print(f"✓ Found images: {os.path.relpath(images_path, dataset_dir)}")

        if not os.path.isdir(labels_path):
            print(f"❌ Missing labels dir: {os.path.relpath(labels_path, dataset_dir)}")
            ok = False
        else:
            print(f"✓ Found labels: {os.path.relpath(labels_path, dataset_dir)}")

        # Quick content check: at least one image and one label
        try:
            has_img = any(fn.lower().endswith(('.jpg', '.jpeg', '.png')) for fn in os.listdir(images_path))
            has_lbl = any(fn.lower().endswith('.txt') for fn in os.listdir(labels_path))
            if not has_img:
                print(f"⚠ No images found in: {os.path.relpath(images_path, dataset_dir)}")
            if not has_lbl:
                print(f"⚠ No labels found in: {os.path.relpath(labels_path, dataset_dir)}")
        except Exception:
            pass

    if ok:
        print("\n✅ Dataset structure is valid!")
    return ok


# STEP 5: TRAIN YOLO MODEL
# ========================

from ultralytics import YOLO

def train_robot_detector(dataset_yaml: str, output_dir: str = "runs/detect", base_model: str = 'yolov8m.pt', epochs: int = 100, exp_name: str = 'robot_detector') -> str:
    """
    Train YOLOv8 robot detector on your dataset.
    
    Args:
        dataset_yaml: Path to data.yaml
        output_dir: Where to save results
    
    Returns:
        Path to best trained model weights
    """
    
    # Load pretrained YOLOv8 model
    # Available sizes: nano (n), small (s), medium (m), large (l), xlarge (x)
    # For training: use 's' or 'm' (good balance)
    
    print("=" * 60)
    print("TRAINING YOLOV8 ROBOT DETECTOR")
    print("=" * 60)
    
    # Check if GPU is available
    import torch
    
    # Detect device
    if torch.cuda.is_available():
        device = 'cuda:0'
        batch_size = 16
        device_name = "NVIDIA GPU (CUDA)"
    elif torch.backends.mps.is_available():
        device = 'mps'  # Apple Metal Performance Shaders
        batch_size = 8
        device_name = "Apple Metal (M5 MacBook Pro) 🚀"
    else:
        device = 'cpu'
        batch_size = 4
        device_name = "CPU (slower)"
    
    print(f"Using device: {device_name}")
    
    # Load chosen YOLOv8 model
    model = YOLO(base_model)
    
    # Train
    results = model.train(
        data=dataset_yaml,
        
        # Training parameters
        epochs=epochs,           # Number of training epochs
        imgsz=640,               # Image size (640x640)
        batch=batch_size,        # Batch size (auto-adjusted for CPU)
        device=device,           # GPU device or CPU
        
        # Optimization
        optimizer='SGD',         # SGD or Adam
        lr0=0.01,                # Initial learning rate
        momentum=0.937,          # SGD momentum
        weight_decay=0.0005,     # L2 regularization
        
        # Early stopping
        patience=20,             # Stop if no improvement for 20 epochs
        
        # Augmentation
        augment=True,            # Data augmentation
        hsv_h=0.015,             # HSV-Hue augmentation
        hsv_s=0.7,               # HSV-Saturation augmentation
        hsv_v=0.4,               # HSV-Value augmentation
        degrees=10,              # Rotation augmentation
        flipud=0.5,              # Flip upside-down probability
        fliplr=0.5,              # Flip left-right probability
        mosaic=1.0,              # Mosaic augmentation
        
        # Validation & saving
        save=True,               # Save checkpoints
        save_period=10,          # Save every 10 epochs
        val=True,                # Validate each epoch
        
        # Logging
        project=output_dir,
        name=exp_name,           # Experiment name
        exist_ok=False,          # Create new folder each run
    )
    
    # Best model path
    best_model = os.path.join(output_dir, exp_name, 'weights', 'best.pt')
    
    print("\n" + "=" * 60)
    print(f"✅ Training complete!")
    print(f"Best model: {best_model}")
    print("=" * 60)
    
    return best_model


# STEP 6: EVALUATE & VALIDATE
# =============================

def evaluate_model(model_path: str, dataset_yaml: str) -> None:
    """
    Evaluate trained model on test set.
    """
    model = YOLO(model_path)
    
    # Validate on test set
    results = model.val(data=dataset_yaml)
    
    print("\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.3f}")
    print(f"  mAP50-95: {results.box.map:.3f}")
    print(f"  Precision: {results.box.p:.3f}")
    print(f"  Recall: {results.box.r:.3f}")


def test_on_video(model_path: str, video_path: str, output_path: str = "output.mp4") -> None:
    """
    Run inference on a video and save annotated output.
    """
    model = YOLO(model_path)
    
    # Inference
    results = model.predict(
        source=video_path,
        conf=0.5,              # Confidence threshold
        save=True,
        save_txt=True,
        line_thickness=2,
    )
    
    print(f"✓ Output saved to: {output_path}")


# STEP 7: INTEGRATE INTO YOUR APP
# ================================

# Once trained, update your robot_detector.py:

"""
from ultralytics import YOLO

class RobotDetector:
    def __init__(self, model_path: str = "models/yolov8m_robots.pt"):
        self.model = YOLO(model_path)
    
    def detect(self, frame, confidence_threshold: float = 0.5):
        results = self.model(frame, conf=confidence_threshold)
        detections = []
        
        for result in results:
            for box in result.boxes:
                if box.conf[0] >= confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0]
                    detections.append(RobotDetection(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        confidence=float(box.conf[0])
                    ))
        
        return detections
"""


# COMPLETE TRAINING PIPELINE
# ===========================

if __name__ == "__main__":
    import sys
    
    print("""
    FRC ROBOT DETECTOR TRAINING PIPELINE
    ====================================
    
    This script will:
    1. Extract frames from your videos
    2. Help you annotate them
    3. Train a YOLOv8 detector
    4. Evaluate the model
    
    Prerequisites:
      - Video files in videos/raw/
      - Annotated dataset in YOLO format
    
    Usage:
      python train_detector.py extract    # Extract frames from videos
    python train_detector.py train [--model yolov8s.pt] [--epochs 50] [--name exp]
                            # Train model on dataset
      python train_detector.py evaluate   # Evaluate trained model
      python train_detector.py test VIDEO # Test on video
    """)
    
    if len(sys.argv) < 2:
        print("Usage: python train_detector.py [extract|train|evaluate|test]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "extract":
        extract_all_videos()
    
    elif command == "train":
        # Simple arg parsing for overrides
        base_model = 'yolov8m.pt'
        epochs = 100
        exp_name = 'robot_detector'
        for arg in sys.argv[2:]:
            if arg.startswith('--model='):
                base_model = arg.split('=', 1)[1]
            elif arg.startswith('--epochs='):
                try:
                    epochs = int(arg.split('=', 1)[1])
                except ValueError:
                    pass
            elif arg.startswith('--name='):
                exp_name = arg.split('=', 1)[1]

        best_model = train_robot_detector(
            dataset_yaml="dataset/data.yaml",
            output_dir="runs/detect",
            base_model=base_model,
            epochs=epochs,
            exp_name=exp_name,
        )
        # Save to models/
        import shutil
        os.makedirs("models", exist_ok=True)
        shutil.copy(best_model, "models/yolov8m_robots.pt")
        print(f"Saved to: models/yolov8m_robots.pt")
    
    elif command == "evaluate":
        evaluate_model(
            model_path="models/yolov8m_robots.pt",
            dataset_yaml="dataset/data.yaml"
        )
    
    elif command == "test":
        if len(sys.argv) < 3:
            print("Usage: python train_detector.py test VIDEO_PATH")
            sys.exit(1)
        test_on_video(
            model_path="models/yolov8m_robots.pt",
            video_path=sys.argv[2]
        )
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
