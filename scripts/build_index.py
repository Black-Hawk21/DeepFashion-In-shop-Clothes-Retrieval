"""
scripts/build_index.py
-----------------------
Offline pipeline: embed the entire gallery split and build the HNSW index.

Steps:
  1. Load fine-tuned (or frozen) CLIP model.
  2. Optionally load BLIP-2 captioner.
  3. Load YOLO detector.
  4. For each gallery image: YOLO crop → BLIP-2 caption → CLIP fused embedding.
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

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from src.models.clip_model import CLIPFineTuner
from src.models.blip2_model import BLIP2Captioner
from src.models.yolo_model import YOLODetector
from src.retrieval.embedder import FusedEmbedder
from src.retrieval.indexer import HNSWIndexer
from src.utils.dataset import load_partition
from src.utils.helpers import get_device, get_logger, load_checkpoint, load_config


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
    parser.add_argument("--no_yolo", action="store_true",
                        help="Disable YOLO cropping (use full images)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix appended to saved index filename for ablation tracking")
    return parser.parse_args()


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

    # ---- YOLO ----
    yolo = None
    if not args.no_yolo:
        yolo = YOLODetector(
            model_name=cfg.yolo.model_name,
            conf_threshold=cfg.yolo.conf_threshold,
            iou_threshold=cfg.yolo.iou_threshold,
        )
    else:
        logger.info("YOLO disabled → using full images.")

    # ---- Embedder ----
    embedder = FusedEmbedder(
        clip_model=clip_model,
        blip2_captioner=blip2,
        yolo_detector=yolo,
        alpha=alpha,
        device=device,
    )
    logger.info(f"Embedder ready  α={embedder.alpha}")

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

    for rel_path, item_id in tqdm(gallery_samples, desc="Embedding gallery"):
        full_path = img_dir / rel_path
        try:
            img = Image.open(full_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {full_path}: {e}")
            continue

        # Embed (includes YOLO crop + optional caption)
        emb = embedder.embed_image(img)
        all_embeddings.append(emb)
        all_item_ids.append(item_id)
        all_img_paths.append(rel_path)

        # Store caption if BLIP-2 is active
        if blip2 is not None:
            if yolo:
                crop, _ = yolo.detect_and_crop(img)
            else:
                crop = img
            captions = blip2.caption([crop])
            all_captions.append(captions[0])
        else:
            all_captions.append("")

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
