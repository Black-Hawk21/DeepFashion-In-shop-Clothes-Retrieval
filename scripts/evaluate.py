"""
scripts/evaluate.py
--------------------
Deliverable 3 — Batch evaluation script.

Given a folder of query images (or the official query split), runs the
full retrieval pipeline end-to-end and reports:
    Recall@K, NDCG@K, mAP@K  for K ∈ {5, 10, 15}

Supports all three ablation conditions (A / B / C) via command-line flags.

Run:
    # Evaluate condition C (best model) on the official query split
    python scripts/evaluate.py \\
        --config configs/config.yaml \\
        --index_path index/hnsw_condC.bin \\
        --meta_path index/metadata_condC.json \\
        --clip_checkpoint checkpoints/best_model.pt \\
        --alpha 0.7

    # Evaluate condition A (vision-only baseline)
    python scripts/evaluate.py \\
        --config configs/config.yaml \\
        --index_path index/hnsw_condA.bin \\
        --meta_path index/metadata_condA.json \\
        --alpha 1.0 --no_rerank

    # Evaluate on a custom folder of query images
    python scripts/evaluate.py \\
        --config configs/config.yaml \\
        --query_folder /path/to/query/images \\
        --index_path index/hnsw_condC.bin \\
        --meta_path index/metadata_condC.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import clip

from src.models.clip_model import CLIPFineTuner
from src.models.blip2_model import BLIP2Captioner, BLIP2Reranker
from src.models.yolo_model import YOLODetector
from src.retrieval.embedder import FusedEmbedder
from src.retrieval.indexer import HNSWIndexer
from src.retrieval.retriever import Retriever
from src.utils.dataset import load_partition, DeepFashionDataset
from src.utils.helpers import (
    get_device, get_logger, load_checkpoint, load_config, save_results
)
from src.utils.metrics import evaluate_retrieval, format_metrics
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate visual product search")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--index_path", required=True,
                        help="Path to saved HNSW .bin file")
    parser.add_argument("--meta_path", required=True,
                        help="Path to saved metadata .json file")
    parser.add_argument("--clip_checkpoint", default=None,
                        help="Fine-tuned CLIP checkpoint (condition C). "
                             "Omit for pretrained (conditions A/B).")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--query_folder", type=str, default=None,
                        help="Custom folder of query images. If omitted, uses "
                             "the official DeepFashion query split.")
    parser.add_argument("--no_rerank", action="store_true",
                        help="Skip BLIP-2 ITM re-ranking")
    parser.add_argument("--no_yolo", action="store_true")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix for output JSON filename")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Embed query split                                                   #
# ------------------------------------------------------------------ #

@torch.no_grad()
def embed_query_split(embedder, cfg, device, logger):
    """Embed official DeepFashion query split. Returns (embs, ids)."""
    _, preprocess = clip.load(cfg.clip.model_name, device="cpu")
    query_ds = DeepFashionDataset(cfg, split="query", transform=preprocess)
    loader = DataLoader(query_ds, batch_size=cfg.eval.batch_size,
                        shuffle=False, num_workers=cfg.dataset.num_workers)

    all_embs, all_ids, all_paths = [], [], []
    embedder.clip.eval()

    for imgs, item_ids, img_paths in tqdm(loader, desc="Embedding queries"):
        imgs = imgs.to(device)
        embs = embedder.clip.encode_image(imgs)           # (B, D)
        all_embs.append(embs.cpu().float().numpy())
        all_ids.extend(list(item_ids))
        all_paths.extend(list(img_paths))

    return np.concatenate(all_embs, axis=0), all_ids, all_paths


def embed_custom_folder(embedder, folder: str, logger):
    """Embed all images in a custom folder. Returns (embs, paths)."""
    folder = Path(folder)
    image_files = sorted(list(folder.glob("*.jpg")) +
                         list(folder.glob("*.jpeg")) +
                         list(folder.glob("*.png")))
    logger.info(f"Found {len(image_files)} images in {folder}")

    all_embs, all_paths = [], []
    for img_path in tqdm(image_files, desc="Embedding custom queries"):
        try:
            img = Image.open(img_path).convert("RGB")
            emb = embedder.embed_image(img)
            all_embs.append(emb)
            all_paths.append(str(img_path))
        except Exception as e:
            logger.warning(f"Skipping {img_path}: {e}")

    return np.stack(all_embs, axis=0), all_paths


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()
    logger = get_logger("evaluate")

    alpha = args.alpha if args.alpha is not None else cfg.embedding.alpha

    # ---- CLIP ----
    clip_model = CLIPFineTuner(
        model_name=cfg.clip.model_name,
        unfreeze_vision_blocks=0,
        freeze_text_encoder=True,
        device=device,
    ).to(device)

    if args.clip_checkpoint:
        load_checkpoint(clip_model, args.clip_checkpoint, device=device)
        logger.info(f"Loaded fine-tuned CLIP: {args.clip_checkpoint}")

    # ---- YOLO ----
    yolo = None
    if not args.no_yolo:
        yolo = YOLODetector(model_name=cfg.yolo.model_name)

    # ---- BLIP-2 (for captioning; only needed if alpha < 1) ----
    blip2 = None
    if alpha < 1.0:
        blip2 = BLIP2Captioner(model_name=cfg.blip2.model_name,
                                device_map=cfg.blip2.device_map)

    embedder = FusedEmbedder(clip_model, blip2, yolo, alpha=alpha, device=device)

    # ---- Load index ----
    indexer = HNSWIndexer(dim=cfg.embedding.embedding_dim)
    indexer.load(args.index_path, args.meta_path)
    gallery_ids = indexer.item_ids

    # ---- Embed queries ----
    if args.query_folder:
        query_embs, query_paths = embed_custom_folder(embedder, args.query_folder, logger)
        query_ids = None  # No ground-truth available for custom folders
    else:
        query_embs, query_ids, query_paths = embed_query_split(
            embedder, cfg, device, logger
        )

    logger.info(f"Query embeddings: {query_embs.shape}")

    # ---- Retrieval ----
    retriever = Retriever(embedder, indexer, reranker=None,
                          ef_search=cfg.index.ef_search)

    max_k = max(cfg.eval.K_values)
    ranked_indices = retriever.batch_query(query_embs, top_k=max_k)  # (Q, max_k)

    # ---- Metrics (only when ground truth is available) ----
    if query_ids is not None:
        metrics = evaluate_retrieval(
            query_ids, gallery_ids, ranked_indices, cfg.eval.K_values
        )
        logger.info("\n" + format_metrics(metrics, cfg.eval.K_values))

        # Save metrics
        run_label = args.output_suffix or f"eval_{Path(args.index_path).stem}"
        out_path = f"{cfg.paths.results_dir}/{run_label}_metrics.json"
        save_results({"metrics": metrics, "alpha": alpha,
                      "clip_checkpoint": args.clip_checkpoint,
                      "n_queries": len(query_ids)}, out_path)

    else:
        # Custom folder: just print top-5 results for each query
        logger.info("No ground truth — printing top-5 results per query image.")
        for q_idx, q_path in enumerate(query_paths):
            top_meta = [indexer.get_metadata(int(ranked_indices[q_idx, k]))
                        for k in range(min(5, ranked_indices.shape[1]))]
            logger.info(f"\nQuery: {q_path}")
            for rank, meta in enumerate(top_meta, 1):
                logger.info(f"  [{rank}] item_id={meta['item_id']}  "
                             f"img={meta['img_path']}")


if __name__ == "__main__":
    main()
