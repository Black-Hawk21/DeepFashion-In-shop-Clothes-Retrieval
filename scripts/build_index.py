"""
scripts/build_index.py
-----------------------
Offline pipeline: embed the entire gallery split and build the HNSW index.

Steps:
  1. Load fine-tuned (or frozen) CLIP model.
  2. Optionally load BLIP-2 captioner.
  3. Load bounding-box annotations from Anno/list_bbox_inshop.txt.
  4. For each gallery image: bbox crop → BLIP-2 caption → CLIP fused embedding.
  5. Build and save HNSW index + metadata.

Run:
    # Condition A: vision-only (α=1)
    python scripts/build_index.py --config configs/config.yaml --alpha 1.0

    # Condition B: frozen CLIP + BLIP-2 caption (α=0.7)
    python scripts/build_index.py --config configs/config.yaml --alpha 0.7

    # Condition C: fine-tuned CLIP + BLIP-2 (α=0.7)
    python scripts/build_index.py --config configs/config.yaml --alpha 0.7 \\
        --clip_checkpoint checkpoints/best_model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import clip
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.models.clip_model import CLIPFineTuner
from src.models.blip2_model import BLIP2Captioner
from src.models.yolo_model import YOLODetector
from src.retrieval.embedder import FusedEmbedder
from src.retrieval.indexer import HNSWIndexer
from src.utils.dataset import load_partition
from src.utils.helpers import get_device, get_logger, load_checkpoint, load_config


# ------------------------------------------------------------------ #
#  Bounding-box loader                                                 #
# ------------------------------------------------------------------ #

def load_bbox_annotations(bbox_file: str, logger=None) -> dict:
    """
    Parse Anno/list_bbox_inshop.txt and return a dict:
        { relative_img_path: (x1, y1, x2, y2) }

    File format (first two lines are header):
        52712
        image_name  clothes_type  pose_type  x_1  y_1  x_2  y_2
        img/WOMEN/Dresses/id_00000002/02_1_front.jpg  3  1  065  045  233  252
        ...
    """
    bbox_map = {}
    with open(bbox_file) as f:
        lines = f.read().splitlines()

    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        img_name = parts[0]
        x1, y1, x2, y2 = int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
        bbox_map[img_name] = (x1, y1, x2, y2)

    if logger:
        logger.info(f"Loaded {len(bbox_map)} bounding-box annotations from {bbox_file}")
    return bbox_map


def crop_with_bbox(image: Image.Image, bbox: tuple, padding: float = 0.05) -> Image.Image:
    """
    Crop an image using the provided (x1, y1, x2, y2) bounding box,
    with optional padding around the crop.
    """
    W, H = image.size
    x1, y1, x2, y2 = bbox

    # Add padding
    bw, bh = x2 - x1, y2 - y1
    pw, ph = int(bw * padding), int(bh * padding)
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(W, x2 + pw)
    y2 = min(H, y2 + ph)

    return image.crop((x1, y1, x2, y2))


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(description="Build HNSW index for gallery split")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override config alpha (image embedding weight)")
    parser.add_argument("--clip_checkpoint", type=str, default=None,
                        help="Path to fine-tuned CLIP checkpoint. "
                             "If None, uses pretrained weights (condition A/B).")
    parser.add_argument("--no_blip2", action="store_true",
                        help="Disable BLIP-2 captioning (forces alpha=1.0, condition A)")
    parser.add_argument("--no_bbox", action="store_true",
                        help="Disable bbox cropping; fall back to YOLO or full images")
    parser.add_argument("--no_yolo", action="store_true",
                        help="Disable YOLO fallback (only relevant if --no_bbox is set)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix appended to saved index filename for ablation tracking")
    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()
    logger = get_logger("build_index")

    alpha = args.alpha if args.alpha is not None else cfg.embedding.alpha

    # ---- CLIP ----
    clip_model = CLIPFineTuner(
        model_name=cfg.clip.model_name,
        unfreeze_vision_blocks=0,   # No training at index time
        freeze_text_encoder=True,
        device=device,
    ).to(device)

    if args.clip_checkpoint:
        load_checkpoint(clip_model, args.clip_checkpoint, device=device)
        logger.info(f"Loaded fine-tuned CLIP from: {args.clip_checkpoint}")
    else:
        logger.info("Using pretrained CLIP weights (no fine-tuning checkpoint).")

    clip_model.eval()

    # ---- BLIP-2 ----
    blip2 = None
    if not args.no_blip2 and alpha < 1.0:
        blip2 = BLIP2Captioner(
            model_name=cfg.blip2.model_name,
            device_map=cfg.blip2.device_map,
            max_new_tokens=cfg.blip2.max_new_tokens,
        )
    else:
        alpha = 1.0
        logger.info("BLIP-2 disabled → vision-only mode (α=1.0)")

    # ---- Bounding-box annotations ----
    bbox_map = {}
    if not args.no_bbox:
        bbox_file = cfg.paths.bbox_file
        bbox_map = load_bbox_annotations(bbox_file, logger)
    else:
        logger.info("Bbox cropping disabled.")

    # ---- YOLO fallback (used only when bbox is unavailable for an image) ----
    yolo = None
    if not args.no_yolo:
        yolo = YOLODetector(
            model_name=cfg.yolo.model_name,
            conf_threshold=cfg.yolo.conf_threshold,
            iou_threshold=cfg.yolo.iou_threshold,
        )
        logger.info("YOLO loaded as fallback for images missing bbox annotations.")
    else:
        logger.info("YOLO fallback disabled.")

    # ---- Load gallery partition ----
    partition = load_partition(cfg.paths.partition_file)
    gallery_samples = partition["gallery"]   # list of (rel_img_path, item_id)
    logger.info(f"Gallery size: {len(gallery_samples)}")

    img_dir = Path(cfg.paths.img_dir)

    # ---- Embed gallery ----
    all_embeddings = []
    all_item_ids = []
    all_img_paths = []
    all_captions = []
    _, clip_preprocess = clip.load(cfg.clip.model_name, device="cpu")

    bbox_used, yolo_used, full_used = 0, 0, 0

    for rel_path, item_id in tqdm(gallery_samples, desc="Embedding gallery"):
        full_path = img_dir / rel_path
        try:
            img = Image.open(full_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {full_path}: {e}")
            continue

        # --- Crop the image ---
        # Priority: bbox annotation → YOLO detection → full image
        if rel_path in bbox_map:
            cropped = crop_with_bbox(img, bbox_map[rel_path])
            bbox_used += 1
        elif yolo is not None:
            cropped, _ = yolo.detect_and_crop(img)
            yolo_used += 1
        else:
            cropped = img
            full_used += 1

        # --- CLIP image embedding ---
        img_tensor = clip_preprocess(cropped).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = clip_model.encode_image(img_tensor, normalize=True)  # (1, D)

        if alpha == 1.0 or blip2 is None:
            emb = img_emb.squeeze(0).cpu().float().numpy()
            caption = ""
        else:
            # --- BLIP-2 caption on the cropped image ---
            captions = blip2.caption([cropped])
            caption = captions[0]

            # --- CLIP text embedding ---
            tokens = clip.tokenize([caption], truncate=True).to(device)
            with torch.no_grad():
                txt_emb = clip_model.encode_text(tokens, normalize=True)  # (1, D)

            # --- Fuse & normalize ---
            fused = alpha * img_emb + (1 - alpha) * txt_emb
            fused = F.normalize(fused, dim=-1)
            emb = fused.squeeze(0).cpu().float().numpy()

        all_embeddings.append(emb)
        all_item_ids.append(item_id)
        all_img_paths.append(rel_path)
        all_captions.append(caption)

    logger.info(
        f"Cropping stats: bbox={bbox_used}, yolo_fallback={yolo_used}, "
        f"full_image={full_used}"
    )

    embeddings_array = np.stack(all_embeddings, axis=0)
    logger.info(f"Embeddings shape: {embeddings_array.shape}")

    # ---- Build & save index ----
    dim = embeddings_array.shape[1]
    indexer = HNSWIndexer(dim=dim, space=cfg.index.space)
    indexer.build(
        embeddings=embeddings_array,
        item_ids=all_item_ids,
        img_paths=all_img_paths,
        captions=all_captions,
        ef_construction=cfg.index.ef_construction,
        M=cfg.index.M,
    )

    suffix = args.suffix or f"alpha{alpha}"
    index_path = f"{cfg.paths.index_dir}/hnsw_{suffix}.bin"
    meta_path  = f"{cfg.paths.index_dir}/metadata_{suffix}.json"
    indexer.save(index_path, meta_path)

    logger.info("Index build complete.")


if __name__ == "__main__":
    main()
