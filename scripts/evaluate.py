"""
scripts/evaluate.py
--------------------
Evaluation script — load cached embeddings + saved HNSW indices
and compute Recall@K, NDCG@K, mAP@K for K in {5, 10, 15}.

Supports all three conditions (A / B / C).
For Condition C, optionally applies BLIP-2 ITM re-ranking (Step 4 of
the online query pipeline) when --rerank is passed.

Run:
    # Evaluate all conditions (no re-ranking)
    python scripts/evaluate.py --results_root results --index_root index

    # Evaluate Cond C with BLIP-2 ITM re-ranking (needs CUDA)
    python scripts/evaluate.py --condition C --rerank --rerank_top_n 50

    # Evaluate a single index
    python scripts/evaluate.py --index_path index/Cond_C/hnsw_alpha0.7_seed510.bin \\
        --meta_path index/Cond_C/metadata_alpha0.7_seed510.json \\
        --query_emb_path results/Cond_C/query_embs_ft.npy \\
        --query_ids_path results/Cond_C/query_ids.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import hnswlib


# ------------------------------------------------------------------ #
#  Metrics                                                             #
# ------------------------------------------------------------------ #

def _recall_at_k(relevant, k):
    return float(relevant[:k].any())

def _ndcg_at_k(relevant, k):
    gains = relevant[:k].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
    dcg = float((gains * discounts).sum())
    n_relevant = int(relevant.sum())
    ideal_k = min(n_relevant, k)
    ideal_dcg = float((np.ones(ideal_k) * discounts[:ideal_k]).sum()) if ideal_k > 0 else 0.0
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def _ap_at_k(relevant, k):
    gains = relevant[:k].astype(float)
    if gains.sum() == 0:
        return 0.0
    cumsum = np.cumsum(gains)
    positions = np.arange(1, k + 1, dtype=float)
    precisions = cumsum / positions
    return float((precisions * gains).sum() / min(int(relevant.sum()), k))

def evaluate_retrieval(query_ids, gallery_ids, ranked_indices, K_values=(5, 10, 15)):
    gallery_arr = np.array(gallery_ids)
    max_k = max(K_values)
    recalls = {k: [] for k in K_values}
    ndcgs   = {k: [] for k in K_values}
    aps     = {k: [] for k in K_values}

    for q_idx, q_id in enumerate(query_ids):
        top_ranked = ranked_indices[q_idx, :max_k]
        relevant = (gallery_arr[top_ranked] == q_id)
        for k in K_values:
            recalls[k].append(_recall_at_k(relevant, k))
            ndcgs[k].append(_ndcg_at_k(relevant, k))
            aps[k].append(_ap_at_k(relevant, k))

    results = {}
    for k in K_values:
        results[f"recall@{k}"] = float(np.mean(recalls[k]))
        results[f"ndcg@{k}"]   = float(np.mean(ndcgs[k]))
        results[f"map@{k}"]    = float(np.mean(aps[k]))
    return results

def format_metrics(metrics, K_values=(5, 10, 15)):
    header = f"{'Metric':<15}" + "".join(f"K={k:<8}" for k in K_values)
    rows = [header, "-" * len(header)]
    for prefix in ("recall", "ndcg", "map"):
        row = f"{prefix.upper():<15}"
        for k in K_values:
            row += f"{metrics.get(f'{prefix}@{k}', 0):<8.4f}"
        rows.append(row)
    return "\n".join(rows)


# ------------------------------------------------------------------ #
#  Index + embedding loaders                                           #
# ------------------------------------------------------------------ #

def load_hnsw_index(index_path, meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    if "gallery_ids" in meta:
        gallery_ids = meta["gallery_ids"]
    elif "item_ids" in meta:
        gallery_ids = meta["item_ids"]
    else:
        raise KeyError(f"No 'gallery_ids' or 'item_ids' in {meta_path}")
    dim = meta.get("dim", 512)
    n_items = meta.get("n_items", len(gallery_ids))
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(index_path, max_elements=n_items)
    return index, gallery_ids, meta

def load_cached_embeddings(emb_path, ids_path):
    embs = np.load(emb_path)
    with open(ids_path) as f:
        ids = json.load(f)
    return embs, ids

def search_hnsw(index, query_embs, top_k=15, ef_search=100):
    index.set_ef(max(ef_search, top_k))
    indices, _ = index.knn_query(query_embs, k=top_k)
    return indices


# ------------------------------------------------------------------ #
#  BLIP-2 ITM Re-ranking                                              #
# ------------------------------------------------------------------ #

def load_blip2_reranker(model_name="Salesforce/blip2-opt-2.7b"):
    """Load BLIP-2 for ITM re-ranking. Requires CUDA."""
    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    print(f"\n[BLIP-2] Loading re-ranker: {model_name}")
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("[BLIP-2] Re-ranker ready")
    return processor, model


def blip2_itm_score(query_image, caption, processor, model):
    """
    Compute ITM score for a (query_image, caption) pair.
    Uses negative generation loss as a proxy for image-text matching.
    Higher score = better match.
    """
    import torch

    inputs = processor(
        images=query_image,
        text=caption if caption else "a clothing item",
        return_tensors="pt",
    ).to(model.device, dtype=torch.float16)

    labels = inputs["input_ids"].clone()
    with torch.no_grad():
        out = model(**inputs, labels=labels)
    return -float(out.loss.item())  # negate loss: higher = better match


def rerank_with_blip2(ranked_indices, query_images, gallery_captions,
                      processor, model, top_n=50, max_k=15):
    """
    Re-rank HNSW candidates using BLIP-2 ITM scores.

    For each query:
      1. Take top-N candidates from HNSW
      2. Score each (query_image, candidate_caption) with BLIP-2
      3. Sort by ITM score descending
      4. Return re-ranked indices (top max_k)

    Args:
        ranked_indices:  (Q, top_n) HNSW results
        query_images:    list of PIL Images for each query
        gallery_captions: list of captions for all gallery items
        processor, model: BLIP-2 processor and model
        top_n:           number of HNSW candidates to re-rank
        max_k:           final number of results to keep

    Returns:
        reranked: (Q, max_k) re-ranked indices
    """
    from tqdm import tqdm

    Q = len(query_images)
    reranked = np.zeros((Q, max_k), dtype=np.int64)

    for qi in tqdm(range(Q), desc="BLIP-2 re-ranking"):
        candidates = ranked_indices[qi, :top_n]
        query_img = query_images[qi]

        # Score each candidate
        scores = []
        for ci in candidates:
            caption = gallery_captions[ci] if ci < len(gallery_captions) else ""
            score = blip2_itm_score(query_img, caption, processor, model)
            scores.append(score)

        # Sort by score descending
        sorted_order = np.argsort(scores)[::-1]
        reranked_candidates = candidates[sorted_order[:max_k]]
        reranked[qi, :len(reranked_candidates)] = reranked_candidates

    return reranked


def load_query_images(data_root, partition_file, bbox_file=None, padding=0.05):
    """Load query images (with optional bbox cropping)."""
    from PIL import Image

    # Load partition
    query_items = []
    with open(partition_file) as f:
        lines = f.read().splitlines()
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[2] == "query":
            query_items.append((parts[0], parts[1]))

    # Load bbox annotations
    bbox_map = {}
    if bbox_file and os.path.exists(bbox_file):
        with open(bbox_file) as f:
            lines = f.read().splitlines()
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) >= 7:
                bbox_map[parts[0]] = (int(parts[3]), int(parts[4]),
                                      int(parts[5]), int(parts[6]))

    # Load images
    images = []
    for rel_path, item_id in query_items:
        full_path = os.path.join(data_root, rel_path)
        try:
            img = Image.open(full_path).convert("RGB")
            if rel_path in bbox_map:
                x1, y1, x2, y2 = bbox_map[rel_path]
                W, H = img.size
                pw = int((x2 - x1) * padding)
                ph = int((y2 - y1) * padding)
                x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
                x2, y2 = min(W, x2 + pw), min(H, y2 + ph)
                img = img.crop((x1, y1, x2, y2))
            images.append(img)
        except Exception:
            # Placeholder for failed loads
            images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

    print(f"  Loaded {len(images)} query images")
    return images


def load_gallery_captions(results_dir):
    """Load cached gallery captions from results directory."""
    # Try Cond_C first, then Cond_B (they share captions)
    for cond_dir in [results_dir, results_dir.replace("Cond_C", "Cond_B")]:
        caption_path = os.path.join(cond_dir, "gallery_captions.json")
        if os.path.exists(caption_path):
            with open(caption_path) as f:
                data = json.load(f)
            captions = data.get("captions", [])
            print(f"  Loaded {len(captions)} gallery captions from {caption_path}")
            return captions

    print("  [Warning] No gallery captions found")
    return []


# ------------------------------------------------------------------ #
#  Condition evaluators                                                #
# ------------------------------------------------------------------ #

def evaluate_single_index(index_path, meta_path, query_emb_path, query_ids_path,
                          K_values=(5, 10, 15), ef_search=100, label="",
                          reranker=None, query_images=None, gallery_captions=None,
                          rerank_top_n=50):
    """Evaluate a single (index, query) pair, with optional BLIP-2 re-ranking."""
    print(f"\n{'-'*60}")
    print(f"  {label}")
    print(f"{'-'*60}")
    print(f"  Index: {index_path}")
    print(f"  Query: {query_emb_path}")

    index, gallery_ids, meta = load_hnsw_index(index_path, meta_path)
    query_embs, query_ids = load_cached_embeddings(query_emb_path, query_ids_path)

    print(f"  Gallery: {len(gallery_ids)} items  |  Queries: {query_embs.shape}")

    max_k = max(K_values)

    # Step 1: HNSW retrieval
    if reranker is not None:
        # Retrieve more candidates for re-ranking
        retrieve_n = max(rerank_top_n, max_k)
        ranked = search_hnsw(index, query_embs, top_k=retrieve_n, ef_search=ef_search)

        # Step 2: BLIP-2 ITM re-ranking
        print(f"  Re-ranking top-{rerank_top_n} candidates with BLIP-2 ITM...")
        processor, model = reranker
        ranked = rerank_with_blip2(
            ranked, query_images, gallery_captions,
            processor, model, top_n=rerank_top_n, max_k=max_k
        )
        print(f"  Re-ranking complete")
    else:
        ranked = search_hnsw(index, query_embs, top_k=max_k, ef_search=ef_search)

    metrics = evaluate_retrieval(query_ids, gallery_ids, ranked, K_values)
    print(f"\n{format_metrics(metrics, K_values)}")
    return metrics


def discover_and_evaluate(results_dir, index_dir, condition, K_values, ef_search,
                          reranker=None, query_images=None, gallery_captions=None,
                          rerank_top_n=50):
    """
    Auto-discover index files and query embeddings for a condition folder
    and evaluate all (alpha, seed) combinations.
    """
    results_path = Path(results_dir)
    index_path = Path(index_dir)

    if not index_path.exists():
        print(f"  [Skip] No index folder: {index_path}")
        return {}

    # Find query embeddings
    query_emb = None
    query_ids = None
    for candidate in ["query_embs_ft.npy", "query_embs_frozen.npy", "query_embs.npy"]:
        ep = results_path / candidate
        ip = results_path / "query_ids.json"
        if ep.exists() and ip.exists():
            query_emb = str(ep)
            query_ids = str(ip)
            break

    if query_emb is None:
        print(f"  [Skip] No query embeddings found in {results_path}")
        return {}

    bin_files = sorted(index_path.glob("hnsw_*.bin"))
    if not bin_files:
        print(f"  [Skip] No .bin index files in {index_path}")
        return {}

    all_metrics = {}
    for bf in bin_files:
        meta_name = bf.name.replace("hnsw_", "metadata_").replace(".bin", ".json")
        mf = bf.parent / meta_name
        if not mf.exists():
            print(f"  [Skip] No metadata for {bf.name}")
            continue

        label = f"Cond {condition} | {bf.stem.replace('hnsw_', '')}"
        if reranker is not None:
            label += " + BLIP-2 rerank"

        metrics = evaluate_single_index(
            str(bf), str(mf), query_emb, query_ids, K_values, ef_search, label,
            reranker=reranker, query_images=query_images,
            gallery_captions=gallery_captions, rerank_top_n=rerank_top_n
        )
        key = bf.stem
        if reranker is not None:
            key += "_reranked"
        all_metrics[key] = metrics

    return all_metrics


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate visual product search")
    parser.add_argument("--condition", type=str, default=None, choices=["A", "B", "C"],
                        help="Evaluate a single condition. If omitted, evaluates all.")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--index_root", default="index")
    parser.add_argument("--data_root", default="data/deepfashion")
    parser.add_argument("--K", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--ef_search", type=int, default=100)
    parser.add_argument("--output", type=str, default="results/evaluation_report.json")
    # Re-ranking options (Condition C)
    parser.add_argument("--rerank", action="store_true",
                        help="Apply BLIP-2 ITM re-ranking (Cond C only, needs CUDA)")
    parser.add_argument("--rerank_top_n", type=int, default=50,
                        help="Number of HNSW candidates to re-rank per query")
    parser.add_argument("--blip2_model", type=str, default="Salesforce/blip2-opt-2.7b")
    # Single-index mode
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--meta_path", type=str, default=None)
    parser.add_argument("--query_emb_path", type=str, default=None)
    parser.add_argument("--query_ids_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    K_values = tuple(args.K)

    print("=" * 60)
    print("  VISUAL PRODUCT SEARCH - EVALUATION")
    print(f"  K = {list(K_values)}")
    if args.rerank:
        print(f"  BLIP-2 Re-ranking: ON (top-{args.rerank_top_n})")
    print("=" * 60)

    # Prepare BLIP-2 re-ranker if requested
    reranker = None
    query_images = None
    gallery_captions = None

    if args.rerank:
        processor, model = load_blip2_reranker(args.blip2_model)
        reranker = (processor, model)

    # Mode 1: Single index evaluation
    if args.index_path and args.meta_path and args.query_emb_path and args.query_ids_path:
        if args.rerank:
            print("\n[Loading] Query images for re-ranking...")
            partition_file = f"{args.data_root}/Eval/list_eval_partition.txt"
            bbox_file = f"{args.data_root}/Anno/list_bbox_inshop.txt"
            query_images = load_query_images(args.data_root, partition_file, bbox_file)
            gallery_captions = load_gallery_captions(f"{args.results_root}/Cond_C")

        metrics = evaluate_single_index(
            args.index_path, args.meta_path,
            args.query_emb_path, args.query_ids_path,
            K_values, args.ef_search, "Single Index",
            reranker=reranker, query_images=query_images,
            gallery_captions=gallery_captions, rerank_top_n=args.rerank_top_n
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"mode": "single", "reranked": args.rerank, "metrics": metrics}, f, indent=2)
        print(f"\nSaved -> {args.output}")
        return

    # Mode 2: Auto-discover all conditions
    conditions = [args.condition] if args.condition else ["A", "B", "C"]
    full_report = {}

    for cond in conditions:
        print(f"\n{'='*60}")
        print(f"  CONDITION {cond}")
        if cond == "C" and args.rerank:
            print(f"  + BLIP-2 ITM Re-ranking (top-{args.rerank_top_n})")
        print(f"{'='*60}")

        results_dir = f"{args.results_root}/Cond_{cond}"
        index_dir   = f"{args.index_root}/Cond_{cond}"

        # Only apply re-ranking to Condition C
        cond_reranker = None
        cond_query_images = None
        cond_gallery_captions = None

        if cond == "C" and args.rerank:
            cond_reranker = reranker

            if query_images is None:
                print("\n[Loading] Query images for re-ranking...")
                partition_file = f"{args.data_root}/Eval/list_eval_partition.txt"
                bbox_file = f"{args.data_root}/Anno/list_bbox_inshop.txt"
                query_images = load_query_images(args.data_root, partition_file, bbox_file)

            if gallery_captions is None:
                gallery_captions = load_gallery_captions(results_dir)

            cond_query_images = query_images
            cond_gallery_captions = gallery_captions

        cond_metrics = discover_and_evaluate(
            results_dir, index_dir, cond, K_values, args.ef_search,
            reranker=cond_reranker, query_images=cond_query_images,
            gallery_captions=cond_gallery_captions, rerank_top_n=args.rerank_top_n
        )
        full_report[f"Cond_{cond}"] = cond_metrics

    # Summary table
    print(f"\n\n{'='*60}")
    print("  SUMMARY TABLE")
    print(f"{'='*60}")

    for cond in conditions:
        cond_key = f"Cond_{cond}"
        if cond_key not in full_report or not full_report[cond_key]:
            continue
        print(f"\n  Condition {cond}:")
        for idx_name, metrics in full_report[cond_key].items():
            label = idx_name.replace("hnsw_", "")
            r5  = metrics.get("recall@5", 0)
            r10 = metrics.get("recall@10", 0)
            r15 = metrics.get("recall@15", 0)
            m10 = metrics.get("map@10", 0)
            n10 = metrics.get("ndcg@10", 0)
            print(f"    {label:<40}  R@5={r5:.4f}  R@10={r10:.4f}  R@15={r15:.4f}  "
                  f"mAP@10={m10:.4f}  NDCG@10={n10:.4f}")

    # Save report
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n\nFull report saved -> {args.output}")


if __name__ == "__main__":
    main()
