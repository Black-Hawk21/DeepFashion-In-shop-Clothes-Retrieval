"""
scripts/finetune_yolo.py
--------------------------
Fine-tune YOLOv8 on DeepFashion In-Shop bounding-box annotations so the
detector learns to crop clothing items (upper-body, lower-body, full-body)
instead of relying on COCO person detections.

Steps:
  1. Parse Anno/list_bbox_inshop.txt for ground-truth bounding boxes.
  2. Convert to YOLO label format (normalized center-x/y, width, height).
  3. Split by partition (train → YOLO train, gallery → YOLO val).
  4. Generate a data.yaml config.
  5. Fine-tune YOLOv8.
  6. Export best model to checkpoints/.

Classes (from clothes_type in bbox annotations):
    0 — upper_body  (shirts, sweaters, blouses, tees, jackets, etc.)
    1 — lower_body  (pants, shorts, skirts, denim, leggings)
    2 — full_body   (dresses, rompers, jumpsuits)

Usage:
    # Default: fine-tune yolov8m on DeepFashion for 50 epochs
    python scripts/finetune_yolo.py --config configs/config.yaml

    # Quick test run
    python scripts/finetune_yolo.py --config configs/config.yaml --epochs 5

    # Use fine-tuned model in other scripts:
    # Update config.yaml →  yolo.model_name: "checkpoints/yolo/best.pt"
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from PIL import Image
from tqdm import tqdm

from src.utils.helpers import get_logger, load_config

# Mapping from bbox file clothes_type to YOLO class id
CLOTHES_TYPE_TO_CLASS = {1: 0, 2: 1, 3: 2}
CLASS_NAMES = ["upper_body", "lower_body", "full_body"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on DeepFashion clothing bounding boxes"
    )
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to project config YAML")
    parser.add_argument("--yolo_base", default="yolov8m.pt",
                        help="Base YOLO model to fine-tune (default: yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--img_size", type=int, default=640,
                        help="Training image size")
    parser.add_argument("--single_class", action="store_true",
                        help="Use a single 'clothing' class instead of 3 sub-types")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save the fine-tuned model "
                             "(default: checkpoints/yolo)")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Where to create the YOLO dataset "
                             "(default: data/deepfashion/yolo_dataset)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Step 1: Parse bounding-box annotations                             #
# ------------------------------------------------------------------ #

def load_bbox_annotations(bbox_file: str, logger=None):
    """
    Parse Anno/list_bbox_inshop.txt.

    Returns:
        dict: { rel_img_path: (clothes_type, x1, y1, x2, y2) }

    File format (first two lines are header):
        52712
        image_name  clothes_type  pose_type  x_1  y_1  x_2  y_2
        img/WOMEN/Dresses/id_00000002/02_1_front.jpg  3  1  065  045  233  252
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

    if logger:
        logger.info(f"Loaded {len(bbox_map)} bbox annotations from {bbox_file}")
    return bbox_map


def load_partition(partition_file: str):
    """Parse list_eval_partition.txt → dict of splits."""
    splits = {"train": [], "query": [], "gallery": []}
    with open(partition_file) as f:
        lines = f.read().splitlines()
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        img_name, item_id, status = parts[0], parts[1], parts[2]
        if status in splits:
            splits[status].append(img_name)
    return splits


# ------------------------------------------------------------------ #
#  Step 2 & 3: Create YOLO dataset directory with labels               #
# ------------------------------------------------------------------ #

def create_yolo_dataset(
    partition: dict,
    bbox_map: dict,
    img_root: Path,
    dataset_dir: Path,
    single_class: bool,
    logger,
):
    """
    Create the YOLO dataset directory structure:
        dataset_dir/
        ├── images/train/   (symlinks to source images)
        ├── images/val/
        ├── labels/train/   (YOLO format .txt labels)
        └── labels/val/

    Uses partition 'train' → YOLO train, 'gallery' → YOLO val.
    """
    split_map = {
        "train": partition["train"],
        "val": partition["gallery"],   # use gallery split for validation
    }

    stats = {"train": {"total": 0, "ok": 0, "skip": 0, "classes": {}},
             "val":   {"total": 0, "ok": 0, "skip": 0, "classes": {}}}

    for split_name, img_list in split_map.items():
        img_dir = dataset_dir / "images" / split_name
        lbl_dir = dataset_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for rel_path in tqdm(img_list, desc=f"Creating {split_name} labels"):
            stats[split_name]["total"] += 1

            # Check bbox annotation exists
            if rel_path not in bbox_map:
                stats[split_name]["skip"] += 1
                continue

            # Full source path
            src_img = img_root / rel_path
            if not src_img.exists():
                stats[split_name]["skip"] += 1
                continue

            # Get image dimensions for normalization
            try:
                img = Image.open(src_img)
                W, H = img.size
                img.close()
            except Exception:
                stats[split_name]["skip"] += 1
                continue

            clothes_type, x1, y1, x2, y2 = bbox_map[rel_path]

            # Map class
            if single_class:
                cls_id = 0
            else:
                cls_id = CLOTHES_TYPE_TO_CLASS.get(clothes_type, 0)

            # Track class distribution
            cls_name = "clothing" if single_class else CLASS_NAMES[cls_id]
            stats[split_name]["classes"][cls_name] = (
                stats[split_name]["classes"].get(cls_name, 0) + 1
            )

            # Convert to YOLO format: center_x, center_y, width, height (normalized)
            cx = ((x1 + x2) / 2.0) / W
            cy = ((y1 + y2) / 2.0) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.001, min(1.0, bw))
            bh = max(0.001, min(1.0, bh))

            # Flatten filename to avoid nested dirs:
            # img/WOMEN/Dresses/id_00000002/02_1_front.jpg → WOMEN_Dresses_id_00000002_02_1_front
            flat_name = rel_path.replace("img/", "").replace("/", "_")
            stem = Path(flat_name).stem
            ext = Path(flat_name).suffix

            # Create symlink / copy for the image
            dst_img = img_dir / f"{stem}{ext}"
            if not dst_img.exists():
                try:
                    os.symlink(src_img.resolve(), dst_img)
                except (OSError, NotImplementedError):
                    # Symlinks may fail on Windows without dev mode; fall back to copy
                    shutil.copy2(src_img, dst_img)

            # Write YOLO label file
            lbl_file = lbl_dir / f"{stem}.txt"
            with open(lbl_file, "w") as f:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            stats[split_name]["ok"] += 1

    # Log stats
    for s in ("train", "val"):
        logger.info(
            f"  {s}: {stats[s]['ok']}/{stats[s]['total']} images "
            f"(skipped {stats[s]['skip']})"
        )
        for cn, cnt in sorted(stats[s]["classes"].items()):
            logger.info(f"    class '{cn}': {cnt}")

    return stats


# ------------------------------------------------------------------ #
#  Step 4: Write data.yaml for Ultralytics                             #
# ------------------------------------------------------------------ #

def write_data_yaml(dataset_dir: Path, single_class: bool):
    """Write the data.yaml config required by Ultralytics YOLO."""
    if single_class:
        names = {0: "clothing"}
        nc = 1
    else:
        names = {i: n for i, n in enumerate(CLASS_NAMES)}
        nc = len(CLASS_NAMES)

    data_cfg = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": names,
    }

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)

    return yaml_path


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("finetune_yolo")

    img_root = Path(cfg.paths.img_dir)
    bbox_file = cfg.paths.bbox_file
    partition_file = cfg.paths.partition_file
    dataset_dir = Path(args.dataset_dir or "data/deepfashion/yolo_dataset")
    output_dir = Path(args.output_dir or "checkpoints/yolo")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  YOLO Fine-tuning on DeepFashion In-Shop")
    logger.info("=" * 60)
    logger.info(f"  Base model   : {args.yolo_base}")
    logger.info(f"  Epochs       : {args.epochs}")
    logger.info(f"  Batch size   : {args.batch_size}")
    logger.info(f"  Image size   : {args.img_size}")
    logger.info(f"  Classes      : {'1 (clothing)' if args.single_class else '3 (upper/lower/full)'}")
    logger.info(f"  Dataset dir  : {dataset_dir}")
    logger.info(f"  Output dir   : {output_dir}")

    # ---- Step 1: Load annotations & partition ----
    logger.info("\n[1/4] Loading annotations and partition...")
    bbox_map = load_bbox_annotations(bbox_file, logger)
    partition = load_partition(partition_file)
    logger.info(
        f"  Partition: train={len(partition['train'])}, "
        f"query={len(partition['query'])}, gallery={len(partition['gallery'])}"
    )

    # ---- Step 2-3: Create YOLO dataset ----
    logger.info("\n[2/4] Creating YOLO dataset structure...")
    create_yolo_dataset(
        partition=partition,
        bbox_map=bbox_map,
        img_root=img_root,
        dataset_dir=dataset_dir,
        single_class=args.single_class,
        logger=logger,
    )

    # ---- Step 4: Write data.yaml ----
    logger.info("\n[3/4] Writing data.yaml...")
    data_yaml = write_data_yaml(dataset_dir, args.single_class)
    logger.info(f"  Config → {data_yaml}")

    # ---- Step 5: Fine-tune ----
    logger.info(f"\n[4/4] Starting YOLO fine-tuning for {args.epochs} epochs...")

    from ultralytics import YOLO
    model = YOLO(args.yolo_base)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        workers=args.workers,
        project=str(output_dir),
        name="deepfashion",
        exist_ok=True,
        resume=args.resume,
        # Training augmentation & hyperparameters
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=0.0005,
        # Augmentations suitable for fashion images
        hsv_h=0.015,      # color hue
        hsv_s=0.4,        # saturation
        hsv_v=0.3,        # value / brightness
        degrees=10.0,     # slight rotation (clothing is mostly upright)
        translate=0.1,
        scale=0.3,
        fliplr=0.5,       # horizontal flip
        flipud=0.0,       # no vertical flip (clothes don't appear upside down)
        mosaic=0.5,
        mixup=0.1,
        # Saving
        save=True,
        save_period=10,    # save every 10 epochs
        patience=15,       # early stopping patience
        verbose=True,
    )

    # ---- Step 6: Copy best model to a convenient location ----
    best_pt = output_dir / "deepfashion" / "weights" / "best.pt"
    final_pt = output_dir / "best.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, final_pt)
        logger.info(f"\nBest model copied to: {final_pt}")
        logger.info(
            "To use this model, update config.yaml:"
        )
        logger.info(f'  yolo.model_name: "{final_pt}"')
    else:
        logger.warning(f"Expected best.pt not found at {best_pt}")

    logger.info("\nYOLO fine-tuning complete!")


if __name__ == "__main__":
    main()
