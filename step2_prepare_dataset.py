"""
Step 2 — Labeling Guide & Dataset Preparation

PART A: HOW TO LABEL WITH LabelImg
─────────────────────────────────────────────────────────────────────────────
Install LabelImg:
    pip install labelImg

Launch it:
    labelImg

Inside LabelImg:
    1. Click "Open Dir"  → select  dataset/images/raw/<class_folder>
    2. Click "Change Save Dir" → select  dataset/labels/raw/<class_folder>
       (create this folder first if it doesn't exist)
    3. Press W  → draws a bounding box
    4. Draw a tight box around the fastener
    5. Type the class name exactly as in your CLASSES list (e.g. "bolt")
    6. Press Ctrl+S to save
    7. Press D  → next image, repeat

    IMPORTANT YOLO FORMAT SETTING:
    In the bottom-left dropdown, make sure it says "YOLO" not "Pascal VOC".
    Each saved label file will be a .txt with one line per object:
        <class_id> <x_center> <y_center> <width> <height>
    All values are normalized 0–1 relative to image size.

TIPS FOR GOOD LABELS:
    - Draw the box as tight as possible around the fastener body
    - Include the full head and threads but exclude shadow/reflection
    - Label every fastener visible in the image, not just the main one
    - Vary: angles (0°, 45°, 90°), distances (close, mid, far), lighting

─────────────────────────────────────────────────────────────────────────────
PART B: This script splits labeled images into train/val sets
         and generates the dataset.yaml YOLO needs for training.

Usage (run AFTER labeling):
    python step2_prepare_dataset.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import shutil
import random
import yaml

# ── Must match exactly what you typed as labels in LabelImg ───────────────────
CLASSES = ["bolt", "nut", "screw", "rivet", "washer"]

RAW_IMAGES = "dataset/images/raw"
RAW_LABELS = "dataset/labels/raw"

TRAIN_SPLIT = 0.85   # 85% train, 15% validation


def prepare():
    # Output dirs
    for split in ["train", "val"]:
        os.makedirs(f"dataset/images/{split}", exist_ok=True)
        os.makedirs(f"dataset/labels/{split}", exist_ok=True)

    all_pairs = []   # (image_path, label_path)

    for cls in CLASSES:
        img_dir = os.path.join(RAW_IMAGES, cls)
        lbl_dir = os.path.join(RAW_LABELS, cls)

        if not os.path.isdir(img_dir):
            print(f"[SKIP] No image folder for class '{cls}'")
            continue

        images = [f for f in os.listdir(img_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for img_file in images:
            label_file = os.path.splitext(img_file)[0] + ".txt"
            img_path   = os.path.join(img_dir, img_file)
            lbl_path   = os.path.join(lbl_dir, label_file)

            if not os.path.exists(lbl_path):
                print(f"[WARN] No label for {img_file} — skipping")
                continue

            all_pairs.append((img_path, lbl_path))

    if not all_pairs:
        print("[ERROR] No labeled image pairs found.")
        print("Make sure labels are saved to dataset/labels/raw/<class>/")
        return

    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * TRAIN_SPLIT)
    train_pairs = all_pairs[:split_idx]
    val_pairs   = all_pairs[split_idx:]

    def copy_pairs(pairs, split):
        for img_src, lbl_src in pairs:
            shutil.copy(img_src, f"dataset/images/{split}/{os.path.basename(img_src)}")
            shutil.copy(lbl_src, f"dataset/labels/{split}/{os.path.basename(lbl_src)}")

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs,   "val")

    # Write dataset.yaml
    yaml_content = {
        "path"  : os.path.abspath("dataset"),
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : len(CLASSES),
        "names" : CLASSES,
    }
    with open("dataset.yaml", "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print("\n=== DATASET PREPARED ===")
    print(f"  Total labeled images : {len(all_pairs)}")
    print(f"  Train                : {len(train_pairs)}")
    print(f"  Val                  : {len(val_pairs)}")
    print(f"  Classes ({len(CLASSES)})          : {CLASSES}")
    print(f"\n  dataset.yaml written.")
    print("  Next step: run step3_train.py")


if __name__ == "__main__":
    prepare()
