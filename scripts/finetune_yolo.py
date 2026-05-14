# ============================================================
# Fine-tune YOLOv8 on DeepFashion In-Shop Bounding Boxes
# ============================================================
# Kaggle-ready: hardcoded config, dual T4 GPU support.
# Run each "# %%" section as a separate Kaggle cell.
#
# Classes (from clothes_type in bbox annotations):
#   0 — upper_body  (shirts, sweaters, blouses, tees, jackets)
#   1 — lower_body  (pants, shorts, skirts, denim)
#   2 — full_body   (dresses, rompers, jumpsuits)
#
# Output:
#   /kaggle/working/yolo_ft/best.pt  — fine-tuned model
# ============================================================

# %% Install dependencies
# !pip install -q ultralytics tqdm Pillow pyyaml

# %% Imports and configuration
import os, shutil, gc
from pathlib import Path

import yaml
from PIL import Image
from tqdm.auto import tqdm

# ===== PATHS =====
DATA_ROOT      = "/kaggle/input/deepfashion-inshop"
PARTITION_FILE = f"{DATA_ROOT}/Eval/list_eval_partition.txt"
BBOX_FILE      = f"{DATA_ROOT}/Anno/list_bbox_inshop.txt"
DATASET_DIR    = Path("/kaggle/working/yolo_dataset")
OUTPUT_DIR     = Path("/kaggle/working/yolo_ft")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== TRAINING CONFIG =====
YOLO_BASE      = "yolov8m.pt"     # base model to fine-tune
EPOCHS         = 50
BATCH_SIZE     = 32               # per-GPU; total = 32×2 = 64 with dual T4
IMG_SIZE       = 640
WORKERS        = 4
SINGLE_CLASS   = False            # True → 1 "clothing" class; False → 3 sub-types
DEVICE         = [0, 1]           # Dual T4 GPUs on Kaggle

# Class mapping
CLOTHES_TYPE_TO_CLASS = {1: 0, 2: 1, 3: 2}
CLASS_NAMES = ["upper_body", "lower_body", "full_body"]

print(f"Dataset dir : {DATASET_DIR}")
print(f"Output dir  : {OUTPUT_DIR}")
print(f"GPUs        : {DEVICE}")
print(f"Batch/GPU   : {BATCH_SIZE}  (effective: {BATCH_SIZE * len(DEVICE)})")

# %% Parse bounding-box annotations
def load_bbox_annotations(bbox_file):
    """
    Parse Anno/list_bbox_inshop.txt.
    Returns: { rel_img_path: (clothes_type, x1, y1, x2, y2) }
    """
    bbox_map = {}
    with open(bbox_file) as f:
        lines = f.read().splitlines()
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        img_name = parts[0]
        clothes_type = int(parts[1])
        x1, y1, x2, y2 = int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
        bbox_map[img_name] = (clothes_type, x1, y1, x2, y2)
    print(f"Loaded {len(bbox_map)} bbox annotations")
    return bbox_map


def load_partition(partition_file):
    """Parse list_eval_partition.txt → dict of splits."""
    splits = {"train": [], "query": [], "gallery": []}
    with open(partition_file) as f:
        lines = f.read().splitlines()
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        if parts[2] in splits:
            splits[parts[2]].append(parts[0])
    for k, v in splits.items():
        print(f"  {k}: {len(v)} images")
    return splits


print("Loading annotations...")
bbox_map = load_bbox_annotations(BBOX_FILE)
partition = load_partition(PARTITION_FILE)

# %% Create YOLO dataset structure
def create_yolo_dataset(partition, bbox_map, img_root, dataset_dir, single_class):
    """
    Create YOLO-format dataset:
        dataset_dir/
        ├── images/train/  (symlinks to source images)
        ├── images/val/
        ├── labels/train/  (YOLO format .txt labels)
        └── labels/val/

    train partition → YOLO train, gallery partition → YOLO val.
    """
    split_map = {
        "train": partition["train"],
        "val": partition["gallery"],
    }

    stats = {}

    for split_name, img_list in split_map.items():
        img_dir = dataset_dir / "images" / split_name
        lbl_dir = dataset_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        ok, skip = 0, 0
        class_counts = {}

        for rel_path in tqdm(img_list, desc=f"Creating {split_name}"):
            if rel_path not in bbox_map:
                skip += 1; continue

            src_img = Path(img_root) / rel_path
            if not src_img.exists():
                skip += 1; continue

            # Get image dimensions for YOLO normalization
            try:
                img = Image.open(src_img)
                W, H = img.size
                img.close()
            except Exception:
                skip += 1; continue

            clothes_type, x1, y1, x2, y2 = bbox_map[rel_path]
            cls_id = 0 if single_class else CLOTHES_TYPE_TO_CLASS.get(clothes_type, 0)
            cls_name = "clothing" if single_class else CLASS_NAMES[cls_id]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            # YOLO format: class center_x center_y width height (all normalized)
            cx = max(0.0, min(1.0, ((x1 + x2) / 2.0) / W))
            cy = max(0.0, min(1.0, ((y1 + y2) / 2.0) / H))
            bw = max(0.001, min(1.0, (x2 - x1) / W))
            bh = max(0.001, min(1.0, (y2 - y1) / H))

            # Flatten filename: img/WOMEN/Dresses/id_xxx/file.jpg → WOMEN_Dresses_id_xxx_file
            flat_name = rel_path.replace("img/", "").replace("/", "_")
            stem = Path(flat_name).stem
            ext = Path(flat_name).suffix

            # Symlink the image (works on Linux/Kaggle without admin)
            dst_img = img_dir / f"{stem}{ext}"
            if not dst_img.exists():
                try:
                    os.symlink(src_img.resolve(), dst_img)
                except (OSError, NotImplementedError):
                    shutil.copy2(str(src_img), str(dst_img))

            # Write label
            lbl_file = lbl_dir / f"{stem}.txt"
            with open(lbl_file, "w") as f:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            ok += 1

        stats[split_name] = {"ok": ok, "skip": skip, "classes": class_counts}
        print(f"  {split_name}: {ok} images, {skip} skipped")
        for cn, cnt in sorted(class_counts.items()):
            print(f"    {cn}: {cnt}")

    return stats


# Check if dataset already exists
if (DATASET_DIR / "images" / "train").exists() and \
   len(list((DATASET_DIR / "images" / "train").iterdir())) > 100:
    print(f"\n[Cache] YOLO dataset already exists at {DATASET_DIR}, skipping creation.")
else:
    print(f"\nCreating YOLO dataset at {DATASET_DIR}...")
    create_yolo_dataset(partition, bbox_map, DATA_ROOT, DATASET_DIR, SINGLE_CLASS)

# %% Write data.yaml
if SINGLE_CLASS:
    names = {0: "clothing"}
    nc = 1
else:
    names = {i: n for i, n in enumerate(CLASS_NAMES)}
    nc = len(CLASS_NAMES)

data_cfg = {
    "path": str(DATASET_DIR.resolve()),
    "train": "images/train",
    "val": "images/val",
    "nc": nc,
    "names": names,
}

data_yaml = DATASET_DIR / "data.yaml"
with open(data_yaml, "w") as f:
    yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)

print(f"data.yaml → {data_yaml}")
print(f"  nc={nc}, names={names}")

# %% Fine-tune YOLO on dual T4 GPUs
from ultralytics import YOLO

print("=" * 60)
print(f"  YOLO Fine-tuning: {YOLO_BASE}")
print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}×{len(DEVICE)} GPUs")
print(f"  Image size: {IMG_SIZE}")
print("=" * 60)

model = YOLO(YOLO_BASE)

results = model.train(
    data=str(data_yaml),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    workers=WORKERS,
    device=DEVICE,              # [0, 1] → dual GPU via DDP
    project=str(OUTPUT_DIR),
    name="deepfashion",
    exist_ok=True,
    # Optimizer
    optimizer="AdamW",
    lr0=1e-3,
    lrf=0.01,
    warmup_epochs=3,
    weight_decay=0.0005,
    # Augmentations (fashion-appropriate)
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.3,
    degrees=10.0,               # slight rotation
    translate=0.1,
    scale=0.3,
    fliplr=0.5,
    flipud=0.0,                 # clothes don't appear upside down
    mosaic=0.5,
    mixup=0.1,
    # Saving
    save=True,
    save_period=10,
    patience=15,                # early stopping
    verbose=True,
)

# %% Copy best model to convenient location
best_pt = OUTPUT_DIR / "deepfashion" / "weights" / "best.pt"
final_pt = OUTPUT_DIR / "best.pt"

if best_pt.exists():
    shutil.copy2(best_pt, final_pt)
    print(f"\n✅ Best model → {final_pt}")
    print(f"   To use: YOLO('{final_pt}')")
else:
    # Check alternative paths
    for candidate in [
        OUTPUT_DIR / "deepfashion" / "weights" / "last.pt",
        OUTPUT_DIR / "deepfashion2" / "weights" / "best.pt",
    ]:
        if candidate.exists():
            shutil.copy2(candidate, final_pt)
            print(f"\n✅ Model → {final_pt} (from {candidate})")
            break
    else:
        print(f"\n⚠️ No model found. Check {OUTPUT_DIR / 'deepfashion' / 'weights'}/")

# %% Quick validation — run inference on a few images
print("\n--- Quick inference test ---")
test_model = YOLO(str(final_pt)) if final_pt.exists() else YOLO(YOLO_BASE)

# Pick 5 random gallery images
import random
random.seed(42)
test_imgs = random.sample(partition["gallery"], min(5, len(partition["gallery"])))

for rel_path in test_imgs:
    full_path = os.path.join(DATA_ROOT, rel_path)
    results = test_model.predict(source=full_path, conf=0.25, verbose=False)
    n_det = sum(len(r.boxes) for r in results)
    print(f"  {rel_path}  →  {n_det} detections")

print("\nDone! Download best.pt from /kaggle/working/yolo_ft/best.pt")
