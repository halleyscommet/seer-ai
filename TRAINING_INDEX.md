📚 TRAINING RESOURCES - COMPLETE INDEX
═════════════════════════════════════════════════════════════════════════════════

All files and guides for training your robot detector on FRC footage.


TRAINING FILES
═════════════════════════════════════════════════════════════════════════════════

EXECUTABLE SCRIPT:
──────────────────
✅ train_detector.py (200+ lines)
   Main script for the entire training pipeline
   
   Commands:
   • python train_detector.py extract    - Extract frames from videos
   • python train_detector.py train      - Train model on dataset
   • python train_detector.py evaluate   - Evaluate trained model
   • python train_detector.py test VIDEO - Test on video file
   
   Features:
   • Extracts every Nth frame (configurable)
   • Handles multiple video files
   • Trains YOLOv8 with optimized parameters
   • Saves best model to models/yolov8_robots.pt
   • Includes evaluation and testing

DOCUMENTATION GUIDES:
─────────────────────
✅ TRAINING_COMPLETE_GUIDE.txt (400+ lines)
   Full end-to-end guide from video to trained model
   
   Covers:
   • What you'll be doing (overview)
   • Step-by-step walkthrough (each phase)
   • Week 0 event recording tips
   • Frame extraction workflow
   • Roboflow annotation tutorial
   • Model training process
   • Integration into app
   • Tips for success
   • Updating model throughout season
   
   READ THIS: Complete start-to-finish reference

✅ WEEK0_TRAINING_GUIDE.txt (250+ lines)
   Week 0 event to deployment timeline
   
   Covers:
   • Timeline: Week 0 → Week 1
   • Recording tips and positioning
   • Frame extraction step-by-step
   • Roboflow tutorial (detailed)
   • Dataset organization
   • Training process
   • Evaluation and testing
   • Integration checklist
   • Troubleshooting
   • Command reference
   
   READ THIS: Quick timeline reference

✅ WEEK0_CHECKLIST.txt (200+ lines)
   Actionable checklist for each phase
   
   Covers:
   • Week 0 event checklist
   • Monday evening tasks
   • Tuesday annotation tasks
   • Training checklist
   • Integration checklist
   • Quick command reference
   • Troubleshooting quick fixes
   • Timeline summary
   
   READ THIS: Use during execution to track progress


HOW TO READ THESE GUIDES
═════════════════════════════════════════════════════════════════════════════════

For different needs:

Planning phase (before Week 0):
  → TRAINING_COMPLETE_GUIDE.txt (section: "What you'll be doing")
  → Get familiar with the overall process

During Week 0 event:
  → WEEK0_TRAINING_GUIDE.txt (section: "Recording tips")
  → Record good quality footage

After Week 0, Monday evening:
  → WEEK0_TRAINING_GUIDE.txt (section: "Frame extraction")
  → Extract frames quickly

During annotation (Monday-Tuesday):
  → TRAINING_COMPLETE_GUIDE.txt (section: "Annotation")
  → Detailed Roboflow tutorial
  → WEEK0_CHECKLIST.txt (section: "Annotation checklist")
  → Track your progress

During training (Tuesday evening):
  → WEEK0_TRAINING_GUIDE.txt (section: "Training")
  → Understand what's happening
  → WEEK0_CHECKLIST.txt (section: "Training checklist")
  → Verify each step

Integration (Wednesday):
  → TRAINING_COMPLETE_GUIDE.txt (section: "Integration")
  → Update app code
  → WEEK0_CHECKLIST.txt (section: "Integration checklist")
  → Verify everything works

Troubleshooting (anytime):
  → WEEK0_TRAINING_GUIDE.txt (section: "Troubleshooting")
  → Quick fixes for common issues
  → WEEK0_CHECKLIST.txt (section: "If something goes wrong")
  → Diagnosis and solutions


QUICK REFERENCE: THE FLOW
═════════════════════════════════════════════════════════════════════════════════

Videos (Week 0)
   ↓ python train_detector.py extract
   ↓
Frames in videos/frames/
   ↓ [Annotate on Roboflow]
   ↓
Annotated dataset in dataset/
   ↓ python train_detector.py train
   ↓
Trained model: models/yolov8_robots.pt
   ↓ [Update robot_detector.py]
   ↓
Production-ready app! 🚀


KEY COMMANDS (Always use these)
═════════════════════════════════════════════════════════════════════════════════

Activate virtual environment (REQUIRED FIRST):
  set -x VIRTUAL_ENV (pwd)/.venv && set -x PATH $VIRTUAL_ENV/bin $PATH

Extract frames from videos (Monday, 5 min):
  python train_detector.py extract

Train on annotated dataset (Tuesday, 30 min - 2 hours):
  python train_detector.py train

Evaluate trained model (Tuesday, 5 min):
  python train_detector.py evaluate

Test on video file (Tuesday, 10 min):
  python train_detector.py test videos/raw/week0_match1.mp4

Run app with trained model (Wednesday onwards):
  python main.py


TYPICAL TIMELINE
═════════════════════════════════════════════════════════════════════════════════

Friday/Saturday (Week 0 event):
  ~45 min: Record 5-10 matches

Monday evening:
  ~5 min: Extract frames
  ~1-2 hours: Annotate images on Roboflow
  ~5 min: Download and extract dataset

Tuesday evening:
  ~30 min - 2 hours: Train model
  ~5 min: Evaluate
  ~10 min: Test on video

Wednesday:
  ~10 min: Integrate into app
  ~5 min: Test on camera
  ✅ READY!

Total: 2-4 hours of active work
Result: Custom robot detector trained on YOUR footage!


EXPECTED RESULTS
═════════════════════════════════════════════════════════════════════════════════

After training you'll have:

models/yolov8_robots.pt (100-200 MB)
  • Your trained robot detector
  • Ready for inference
  • Trained on 300-500 annotated images from YOUR videos
  • Accuracy: 0.85+ mAP (with good training data)

runs/detect/robot_detector/
  • Training logs and metrics
  • Training curves (visualize improvement)
  • Checkpoint files
  • Best and last weights

app/robot_detector.py (updated)
  • Loads your trained model
  • Runs inference on frames
  • Returns detections for tracker

Integrated app
  • Detects robots in real-time
  • Tracks 6 robots with consistent IDs
  • Records scouting data
  • Ready for events!


HELPFUL TOOLS & SERVICES
═════════════════════════════════════════════════════════════════════════════════

Roboflow (Recommended for annotation):
  • Free account available
  • Web-based annotation tool
  • Automatic data augmentation
  • Exports directly in YOLOv8 format
  • Website: https://roboflow.com/

LabelImg (Alternative for annotation):
  • Free, open-source
  • Desktop GUI tool
  • pip install labelimg
  • More manual, but powerful

YOLOv8 Documentation:
  • Official docs: https://docs.ultralytics.com/
  • Training guide: https://docs.ultralytics.com/modes/train/
  • API reference: https://docs.ultralytics.com/usage/python/

OpenCV (Image processing):
  • For frame extraction
  • Already in your venv
  • Docs: https://opencv.org/

FRC Vision Resources:
  • GitHub: github.com/topics/frc-vision
  • Team forums: chiefdelphi.com
  • Vision libraries: wpilib.org


TROUBLESHOOTING INDEX
═════════════════════════════════════════════════════════════════════════════════

Problem: Can't extract frames
  Solution: See WEEK0_TRAINING_GUIDE.txt
  • Check videos/raw/ exists
  • Check video files are readable
  • Check free disk space

Problem: Training is very slow
  Solution: See TRAINING_COMPLETE_GUIDE.txt
  • Check if using GPU
  • Reduce batch size if CUDA out of memory
  • Reduce epochs to 50

Problem: Model accuracy is low (mAP < 0.70)
  Solution: See WEEK0_TRAINING_GUIDE.txt
  • Get more annotated data
  • Improve annotation quality
  • Vary recording angles/lighting
  • Train for longer epochs

Problem: Can't find trained model
  Solution: Check WEEK0_CHECKLIST.txt
  • Verify models/yolov8_robots.pt exists
  • Training may still be running
  • Check training completed successfully

Problem: App crashes with detector
  Solution: See TRAINING_COMPLETE_GUIDE.txt
  • Verify model path in code
  • Check CUDA compatibility
  • Try CPU mode


NEXT STEPS
═════════════════════════════════════════════════════════════════════════════════

Before Week 0:
  1. Read TRAINING_COMPLETE_GUIDE.txt (overview)
  2. Review WEEK0_TRAINING_GUIDE.txt (timeline)
  3. Get familiar with Roboflow

During Week 0:
  1. Follow WEEK0_TRAINING_GUIDE.txt (recording section)
  2. Use WEEK0_CHECKLIST.txt to track progress

After Week 0:
  1. Extract frames: python train_detector.py extract
  2. Annotate on Roboflow (1-2 hours)
  3. Download dataset
  4. Train: python train_detector.py train
  5. Integrate into app
  6. Ready to use!


SUPPORT
═════════════════════════════════════════════════════════════════════════════════

If stuck, check:
  1. WEEK0_CHECKLIST.txt (Troubleshooting section)
  2. WEEK0_TRAINING_GUIDE.txt (Troubleshooting section)
  3. TRAINING_COMPLETE_GUIDE.txt (Tips for success)

All commands are in the guides.
All steps are documented.
All common issues have solutions.

Good luck with your training! 🚀
