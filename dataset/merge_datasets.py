"""
merge_datasets.py  —  Seer dataset merger (portable, zip-ready)

Output structure (everything under dataset/merged/):

  merged/
    combined/
      images/train/
      images/val/
      labels/train/
      labels/val/
    robots/
      images/train/
      images/val/
      labels/train/  (only class 0 annotations)
      labels/val/
    balls/
      images/train/
      images/val/
      labels/train/  (class 1 → remapped to 0)
      labels/val/

  combined_data.yaml   — absolute paths to merged/
  robots_data.yaml     — absolute paths to merged/robots/
  balls_data.yaml      — absolute paths to merged/balls/

Zip workflow (paths stay consistent on server if unzipped to same relative location):
  zip -r dataset_robots.zip  dataset/merged/robots  dataset/robots_data.yaml
  zip -r dataset_balls.zip   dataset/merged/balls   dataset/balls_data.yaml

Usage:
    python dataset/merge_datasets.py           # uses hardlinks (fast, saves disk)
    python dataset/merge_datasets.py --copy    # full copies (needed across filesystems)
"""

import argparse
import os
import random
import shutil
from pathlib import Path
import yaml

# ── Config ────────────────────────────────────────────────────────────────────
SEED      = 42
VAL_RATIO = 0.20
BASE_DIR  = Path(__file__).parent          # dataset/
OUT_DIR   = BASE_DIR / "merged"

FOLDERS = [
    "yolo_dataset",
    "yolo_dataset (1)",
    "yolo_dataset (2)",
    "yolo_dataset (3)",
    "yolo_dataset (4)",
    # "yolo_dataset (5)",
]
# ──────────────────────────────────────────────────────────────────────────────

random.seed(SEED)


def collect_samples(folder_path: Path):
    images_dir = folder_path / "images"
    labels_dir = folder_path / "labels"
    if not images_dir.exists():
        print(f"  WARNING: no images/ in {folder_path.name}, skipping.")
        return []
    samples = []
    for img in sorted(images_dir.rglob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = labels_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        samples.append((img, lbl))
    return samples


def split_samples(samples, val_ratio):
    shuffled = samples[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def place_file(src: Path, dst: Path, use_copy: bool):
    """Hardlink (fast, no extra disk) or full copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def write_filtered_label(src_lbl: Path, dst_lbl: Path, keep_class: int, remap_to: int = 0):
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    lines_out = []
    with open(src_lbl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if int(parts[0]) == keep_class:
                parts[0] = str(remap_to)
                lines_out.append(" ".join(parts))
    with open(dst_lbl, "w") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))
    return len(lines_out)


def process_split(samples, split_name, dataset_name, keep_class, use_copy):
    """
    Copy images + write (filtered) labels for one train/val split.
    keep_class=None means keep all classes as-is (combined dataset).
    Returns annotation count.
    """
    images_out = OUT_DIR / dataset_name / "images" / split_name
    labels_out = OUT_DIR / dataset_name / "labels" / split_name
    total = 0
    for img, lbl in samples:
        # Deduplicate filenames across source folders by prefixing with parent folder
        safe_name = f"{img.parent.parent.name}_{img.name}".replace(" ", "_").replace("(", "").replace(")", "")
        place_file(img, images_out / safe_name, use_copy)
        dst_lbl = labels_out / (Path(safe_name).stem + ".txt")
        if keep_class is None:
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            if not dst_lbl.exists():
                shutil.copy2(lbl, dst_lbl)
            with open(lbl) as f:
                total += sum(1 for l in f if l.strip())
        else:
            total += write_filtered_label(lbl, dst_lbl, keep_class, remap_to=0)
    return total


def write_yaml(yaml_path: Path, train_images: Path, val_images: Path, names: dict):
    data = {
        "train": str(train_images.resolve()),
        "val":   str(val_images.resolve()),
        "names": names,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"  Wrote {yaml_path.relative_to(BASE_DIR.parent)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy", action="store_true",
                        help="Use file copies instead of hardlinks (needed across filesystems)")
    args = parser.parse_args()

    print("=" * 60)
    print("Scanning dataset folders …")
    print("=" * 60)

    all_train, all_val = [], []

    for folder_name in FOLDERS:
        folder_path = BASE_DIR / folder_name
        if not folder_path.exists():
            print(f"  SKIP (not found): {folder_name}")
            continue
        samples = collect_samples(folder_path)
        if not samples:
            print(f"  SKIP (empty): {folder_name}")
            continue

        train_s, val_s = split_samples(samples, VAL_RATIO)
        all_train.extend(train_s)
        all_val.extend(val_s)

        n_robots = n_balls = 0
        for _, lbl in samples:
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        if int(parts[0]) == 0: n_robots += 1
                        elif int(parts[0]) == 1: n_balls += 1

        print(f"  {folder_name}: {len(samples)} images "
              f"(train={len(train_s)}, val={len(val_s)}) "
              f"robots={n_robots} balls={n_balls}")

    print()
    print(f"TOTAL  train={len(all_train)}  val={len(all_val)}  "
          f"images={len(all_train)+len(all_val)}")
    print()
    print(f"Building merged/ structure in {OUT_DIR} …")
    print(f"  ({'copies' if args.copy else 'hardlinks — add --copy if you see errors'})")
    print()

    # Wipe and rebuild to avoid stale files from previous runs
    if OUT_DIR.exists():
        print(f"  Removing old {OUT_DIR} …")
        shutil.rmtree(OUT_DIR)

    c_tr = process_split(all_train, "train", "combined", None,  args.copy)
    c_va = process_split(all_val,   "val",   "combined", None,  args.copy)
    print(f"  combined  — train={c_tr} annotations  val={c_va}")

    r_tr = process_split(all_train, "train", "robots", 0, args.copy)
    r_va = process_split(all_val,   "val",   "robots", 0, args.copy)
    print(f"  robots    — train={r_tr} annotations  val={r_va}")

    b_tr = process_split(all_train, "train", "balls",  1, args.copy)
    b_va = process_split(all_val,   "val",   "balls",  1, args.copy)
    print(f"  balls     — train={b_tr} annotations  val={b_va}")

    # ── Yaml files ────────────────────────────────────────────────────────────
    print()
    print("Writing yaml files …")
    write_yaml(
        BASE_DIR / "combined_data.yaml",
        OUT_DIR / "combined" / "images" / "train",
        OUT_DIR / "combined" / "images" / "val",
        names={0: "robot", 1: "ball"},
    )
    write_yaml(
        BASE_DIR / "robots_data.yaml",
        OUT_DIR / "robots" / "images" / "train",
        OUT_DIR / "robots" / "images" / "val",
        names={0: "robot"},
    )
    write_yaml(
        BASE_DIR / "balls_data.yaml",
        OUT_DIR / "balls" / "images" / "train",
        OUT_DIR / "balls" / "images" / "val",
        names={0: "ball"},
    )

    # ── Instructions ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Done! Zip and upload to Google Drive:")
    print()
    print("  cd ~/Desktop/Code/seer")
    print()
    print("  zip -r dataset_robots.zip \\")
    print("      dataset/merged/robots dataset/robots_data.yaml")
    print()
    print("  zip -r dataset_balls.zip \\")
    print("      dataset/merged/balls dataset/balls_data.yaml")
    print()
    print("On the server (unzip into your seer repo root):")
    print()
    print("  unzip dataset_robots.zip")
    print("  unzip dataset_balls.zip")
    print()
    print("Then train:")
    print()
    print("  ./scripts/yolo_train.sh \\")
    print("    --model yolo26m.pt --data dataset/robots_data.yaml \\")
    print("    --epochs 100 --imgsz 1024 --batch 6 --name robots_v1")
    print()
    print("  ./scripts/yolo_train.sh \\")
    print("    --model yolo26s.pt --data dataset/balls_data.yaml \\")
    print("    --epochs 100 --imgsz 1024 --batch 6 --name balls_v1")
    print("=" * 60)


if __name__ == "__main__":
    main()
