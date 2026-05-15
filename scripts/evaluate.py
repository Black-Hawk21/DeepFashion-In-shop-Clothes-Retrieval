"""
scripts/evaluate.py
--------------------
Evaluation script — load cached embeddings + saved HNSW indices
and compute Recall@K, NDCG@K, mAP@K for K in {5, 10, 15}.

Condition structure
-------------------
  Cond A : alpha=1.0,             3 seeds  → 3  index files  (no BLIP-2)
  Cond B : alpha in {0.5, 0.7},  3 seeds  → 6  index files  (with BLIP-2 re-ranking)
  Cond C : alpha in {0.5, 0.7},  3 seeds  → 6  index files  (with BLIP-2 re-ranking)

Summary reporting
-----------------
  • Per (condition, alpha): mean ± std across seeds for every metric/K
  • Final table printed to stdout and saved to JSON

Run:
    # Evaluate all conditions (no re-ranking)
    python scripts/evaluate.py --results_root results --index_root index

    # Evaluate Cond B & C with BLIP-2 ITM re-ranking (needs CUDA)
    python scripts/evaluate.py --rerank --rerank_top_n 50

    # Evaluate a single index
    python scripts/evaluate.py \\
        --index_path index/Cond_C/hnsw_alpha0.5_seed51.bin \\
        --meta_path  index/Cond_C/metadata_alpha0.5_seed51.json \\
        --query_emb_path  results/Cond_C/query_embs_ft.npy \\
        --query_ids_path  results/Cond_C/query_ids.json
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import hnswlib


# ------------------------------------------------------------------ #
#  Condition configuration                                             #
# ------------------------------------------------------------------ #

# For A: only alpha=1.0, no re-ranking, 3 seeds → 3 indices
# For B & C: alpha in {0.5, 0.7}, re-ranking applied, 3 seeds each → 6 indices each
COND_CONFIG = {
    "A": {"alphas": [1.0],      "rerank": False},
    "B": {"alphas": [0.5, 0.7], "rerank": True},
    "C": {"alphas": [0.5, 0.7], "rerank": True},
}

K_VALUES_DEFAULT = (5, 10, 15)


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


def evaluate_retrieval(query_ids, gallery_ids, ranked_indices, K_values=K_VALUES_DEFAULT):
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


# ------------------------------------------------------------------ #
#  Formatting helpers                                                  #
# ------------------------------------------------------------------ #

def format_metrics_table(metrics, K_values=K_VALUES_DEFAULT):
    """Single-run metrics table (no std)."""
    header = f"{'Metric':<12}" + "".join(f"  K={k:<8}" for k in K_values)
    sep    = "-" * len(header)
    rows   = [header, sep]
    for prefix in ("recall", "ndcg", "map"):
        row = f"{prefix.upper():<12}"
        for k in K_values:
            row += f"  {metrics.get(f'{prefix}@{k}', 0.0):<10.4f}"
        rows.append(row)
    return "\n".join(rows)


def format_mean_std_table(mean_metrics, std_metrics, K_values=K_VALUES_DEFAULT):
    """Mean ± std across seeds."""
    col_w = 18
    header = f"{'Metric':<12}" + "".join(f"  {'K='+str(k):<{col_w}}" for k in K_values)
    sep    = "-" * len(header)
    rows   = [header, sep]
    for prefix in ("recall", "ndcg", "map"):
        row = f"{prefix.upper():<12}"
        for k in K_values:
            key = f"{prefix}@{k}"
            mu  = mean_metrics.get(key, 0.0)
            sd  = std_metrics.get(key, 0.0)
            cell = f"{mu:.4f} ± {sd:.4f}"
            row += f"  {cell:<{col_w}}"
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
    dim     = meta.get("dim", 512)
    n_items = meta.get("n_items", len(gallery_ids))
    index   = hnswlib.Index(space="cosine", dim=dim)
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
    return -float(out.loss.item())  # negate: higher = better match


def rerank_with_blip2(ranked_indices, query_images, gallery_captions,
                      processor, model, k):
    """
    Re-rank the top-K HNSW candidates using BLIP-2 ITM scores for a single K.

    For each query:
      1. Take exactly the top-K candidates from HNSW
      2. Score each (query_image, candidate_caption) with BLIP-2
      3. Sort by ITM score descending → return re-ranked top-K

    Args:
        ranked_indices   : (Q, >=k) array of HNSW candidate indices
        query_images     : list of PIL Images, one per query
        gallery_captions : list of str captions, indexed by gallery position
        processor, model : BLIP-2 processor and model
        k                : number of candidates to re-rank (== the K being evaluated)

    Returns:
        reranked : (Q, k) array of re-ranked gallery indices
    """
    from tqdm import tqdm

    Q = len(query_images)
    reranked = np.zeros((Q, k), dtype=np.int64)

    for qi in tqdm(range(Q), desc=f"BLIP-2 re-ranking K={k}"):
        candidates = ranked_indices[qi, :k]
        query_img  = query_images[qi]

        scores = [
            blip2_itm_score(query_img,
                            gallery_captions[ci] if ci < len(gallery_captions) else "",
                            processor, model)
            for ci in candidates
        ]

        sorted_order        = np.argsort(scores)[::-1]
        reranked[qi]        = candidates[sorted_order]

    return reranked


# ------------------------------------------------------------------ #
#  Image / caption loaders (for re-ranking)                            #
# ------------------------------------------------------------------ #

def load_query_images(data_root, partition_file, bbox_file=None, padding=0.05):
    """Load query images (with optional bbox cropping)."""
    from PIL import Image

    query_items = []
    with open(partition_file) as f:
        lines = f.read().splitlines()
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[2] == "query":
            query_items.append((parts[0], parts[1]))

    bbox_map = {}
    if bbox_file and os.path.exists(bbox_file):
        with open(bbox_file) as f:
            lines = f.read().splitlines()
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) >= 7:
                bbox_map[parts[0]] = (int(parts[3]), int(parts[4]),
                                      int(parts[5]), int(parts[6]))

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
            images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

    print(f"  Loaded {len(images)} query images")
    return images


def load_gallery_captions(results_dir):
    """Load cached gallery captions from results directory."""
    for cond_dir in [results_dir,
                     results_dir.replace("Cond_C", "Cond_B"),
                     results_dir.replace("Cond_B", "Cond_C")]:
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
#  Single-index evaluation                                             #
# ------------------------------------------------------------------ #

def evaluate_single_index(index_path, meta_path, query_emb_path, query_ids_path,
                          K_values=K_VALUES_DEFAULT, ef_search=100, label="",
                          reranker=None, query_images=None, gallery_captions=None):
    """
    Evaluate one (index, query-embedding) pair, with optional BLIP-2 re-ranking.

    When re-ranking is enabled, HNSW retrieves exactly K candidates for each K
    and BLIP-2 re-ranks those K items. This is done independently per K so that
    each metric@K is computed on candidates that were re-ranked at that exact depth.
    """
    print(f"\n{'-'*60}")
    print(f"  {label}")
    print(f"{'-'*60}")
    print(f"  Index : {index_path}")
    print(f"  Query : {query_emb_path}")

    index, gallery_ids, meta = load_hnsw_index(index_path, meta_path)
    query_embs, query_ids   = load_cached_embeddings(query_emb_path, query_ids_path)
    print(f"  Gallery: {len(gallery_ids)} items  |  Queries: {query_embs.shape}")

    if reranker is not None:
        processor, model = reranker
        # Re-rank independently for each K; metrics collected per-K then merged
        gallery_arr = np.array(gallery_ids)
        max_k       = max(K_values)
        all_recalls = {k: [] for k in K_values}
        all_ndcgs   = {k: [] for k in K_values}
        all_aps     = {k: [] for k in K_values}

        # Fetch the maximum K once from HNSW (superset), then slice per K
        hnsw_ranked = search_hnsw(index, query_embs, top_k=max_k, ef_search=ef_search)

        for k in K_values:
            print(f"  Re-ranking top-{k} candidates with BLIP-2 ITM  (K={k})...")
            reranked_k = rerank_with_blip2(
                hnsw_ranked, query_images, gallery_captions,
                processor, model, k=k,
            )
            for q_idx, q_id in enumerate(query_ids):
                relevant = (gallery_arr[reranked_k[q_idx]] == q_id)
                all_recalls[k].append(_recall_at_k(relevant, k))
                all_ndcgs[k].append(_ndcg_at_k(relevant, k))
                all_aps[k].append(_ap_at_k(relevant, k))

        metrics = {}
        for k in K_values:
            metrics[f"recall@{k}"] = float(np.mean(all_recalls[k]))
            metrics[f"ndcg@{k}"]   = float(np.mean(all_ndcgs[k]))
            metrics[f"map@{k}"]    = float(np.mean(all_aps[k]))
        print("  Re-ranking complete")
    else:
        max_k  = max(K_values)
        ranked = search_hnsw(index, query_embs, top_k=max_k, ef_search=ef_search)
        metrics = evaluate_retrieval(query_ids, gallery_ids, ranked, K_values)

    print(f"\n{format_metrics_table(metrics, K_values)}")
    return metrics


# ------------------------------------------------------------------ #
#  Index filename parsing                                              #
# ------------------------------------------------------------------ #

def parse_index_filename(stem):
    """
    Extract (alpha, seed) from filenames like:
      hnsw_alpha0.5_seed51
      hnsw_alpha1.0_seed42
      hnsw_alpha0.7_seed510
    Returns (alpha: float, seed: str) or (None, None) if not parseable.
    """
    m = re.search(r"alpha([\d.]+)_seed(\w+)", stem)
    if m:
        return float(m.group(1)), m.group(2)
    return None, None


# ------------------------------------------------------------------ #
#  Condition-level evaluation with seed aggregation                    #
# ------------------------------------------------------------------ #

def discover_and_evaluate(results_dir, index_dir, condition,
                          K_values, ef_search,
                          reranker=None, query_images=None,
                          gallery_captions=None):
    """
    Auto-discover all index files for a condition and evaluate them.

    Returns
    -------
    per_run   : dict  {stem: metrics}   — raw per-index results
    aggregated: dict  {alpha: {"mean": {...}, "std": {...}, "n_seeds": int}}
    """
    results_path = Path(results_dir)
    index_path   = Path(index_dir)

    if not index_path.exists():
        print(f"  [Skip] No index folder: {index_path}")
        return {}, {}

    # Locate query embeddings
    query_emb = query_ids_path = None
    for candidate in ["query_embs_ft.npy", "query_embs_frozen.npy", "query_embs.npy"]:
        ep = results_path / candidate
        ip = results_path / "query_ids.json"
        if ep.exists() and ip.exists():
            query_emb      = str(ep)
            query_ids_path = str(ip)
            break

    if query_emb is None:
        print(f"  [Skip] No query embeddings found in {results_path}")
        return {}, {}

    bin_files = sorted(index_path.glob("hnsw_*.bin"))
    if not bin_files:
        print(f"  [Skip] No .bin index files in {index_path}")
        return {}, {}

    per_run   = {}                          # stem → metrics
    by_alpha  = defaultdict(list)           # alpha → [metrics, ...]

    for bf in bin_files:
        meta_name = bf.name.replace("hnsw_", "metadata_").replace(".bin", ".json")
        mf = bf.parent / meta_name
        if not mf.exists():
            print(f"  [Skip] No metadata for {bf.name}")
            continue

        alpha, seed = parse_index_filename(bf.stem)
        label = f"Cond {condition} | {bf.stem.replace('hnsw_', '')}"
        if reranker is not None:
            label += " + BLIP-2 rerank"

        metrics = evaluate_single_index(
            str(bf), str(mf), query_emb, query_ids_path,
            K_values, ef_search, label,
            reranker=reranker, query_images=query_images,
            gallery_captions=gallery_captions,
        )

        key = bf.stem + ("_reranked" if reranker is not None else "")
        per_run[key] = {"alpha": alpha, "seed": seed, "metrics": metrics}

        if alpha is not None:
            by_alpha[alpha].append(metrics)

    # Compute mean ± std across seeds for each alpha
    aggregated = {}
    for alpha, metric_list in sorted(by_alpha.items()):
        all_keys = list(metric_list[0].keys())
        mean_m = {k: float(np.mean([m[k] for m in metric_list])) for k in all_keys}
        std_m  = {k: float(np.std( [m[k] for m in metric_list], ddof=0)) for k in all_keys}
        aggregated[str(alpha)] = {
            "n_seeds":  len(metric_list),
            "mean":     mean_m,
            "std":      std_m,
            "per_seed": metric_list,
        }

    return per_run, aggregated


# ------------------------------------------------------------------ #
#  Summary printing                                                    #
# ------------------------------------------------------------------ #

def print_aggregated_summary(full_report, K_values=K_VALUES_DEFAULT):
    """Print mean ± std table for every (condition, alpha) group."""
    width = 70
    print(f"\n\n{'='*width}")
    print("  AGGREGATED SUMMARY  (mean ± std across seeds)")
    print(f"{'='*width}")

    for cond_key, cond_data in sorted(full_report.items()):
        agg = cond_data.get("aggregated", {})
        if not agg:
            continue
        cond = cond_key.replace("Cond_", "")
        print(f"\n  Condition {cond}:")
        for alpha_str, agg_vals in sorted(agg.items(), key=lambda x: float(x[0])):
            n = agg_vals["n_seeds"]
            tag = f"alpha={alpha_str}  [{n} seed{'s' if n != 1 else ''}]"
            rerank_note = ""
            if COND_CONFIG.get(cond, {}).get("rerank"):
                rerank_note = " + BLIP-2 rerank"
            print(f"\n    {tag}{rerank_note}")
            print("    " + format_mean_std_table(
                agg_vals["mean"], agg_vals["std"], K_values
            ).replace("\n", "\n    "))

    print(f"\n{'='*width}")


def print_compact_summary(full_report, K_values=K_VALUES_DEFAULT):
    """Single-line-per-(cond, alpha) compact table."""
    width = 110
    print(f"\n\n{'='*width}")
    print("  COMPACT SUMMARY  (mean ± std across seeds)")
    print(f"{'='*width}")

    # Build header
    k_headers = "".join(
        f"  R@{k}          NDCG@{k}       mAP@{k}      " for k in K_values
    )
    print(f"  {'Condition/Alpha':<28}{k_headers}")
    print(f"  {'-'*106}")

    for cond_key, cond_data in sorted(full_report.items()):
        agg = cond_data.get("aggregated", {})
        if not agg:
            continue
        cond = cond_key.replace("Cond_", "")
        uses_rerank = COND_CONFIG.get(cond, {}).get("rerank", False)

        for alpha_str, agg_vals in sorted(agg.items(), key=lambda x: float(x[0])):
            n    = agg_vals["n_seeds"]
            mu   = agg_vals["mean"]
            sd   = agg_vals["std"]
            tag  = f"Cond {cond} | α={alpha_str} (n={n})"
            if uses_rerank:
                tag += " ✓rerank"

            cells = ""
            for k in K_values:
                for prefix in ("recall", "ndcg", "map"):
                    key = f"{prefix}@{k}"
                    cells += f"  {mu[key]:.4f}±{sd[key]:.4f}   "

            print(f"  {tag:<28}{cells}")

    print(f"{'='*width}\n")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate visual product search")
    parser.add_argument("--condition", type=str, default=None, choices=["A", "B", "C"],
                        help="Evaluate a single condition. If omitted, evaluates all.")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--index_root",   default="index")
    parser.add_argument("--data_root",    default="data/deepfashion")
    parser.add_argument("--K",            type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--ef_search",    type=int, default=100)
    parser.add_argument("--output",       type=str, default="results/evaluation_report.json")

    # Re-ranking options (Cond B & C)
    parser.add_argument("--rerank",       action="store_true",
                        help="Apply BLIP-2 ITM re-ranking for Cond B & C (needs CUDA)")

    parser.add_argument("--blip2_model",  type=str, default="Salesforce/blip2-opt-2.7b")

    # Single-index mode
    parser.add_argument("--index_path",     type=str, default=None)
    parser.add_argument("--meta_path",      type=str, default=None)
    parser.add_argument("--query_emb_path", type=str, default=None)
    parser.add_argument("--query_ids_path", type=str, default=None)

    return parser.parse_args()


def main():
    args      = parse_args()
    K_values  = tuple(args.K)

    print("=" * 60)
    print("  VISUAL PRODUCT SEARCH — EVALUATION")
    print(f"  K = {list(K_values)}")
    if args.rerank:
        print(f"  BLIP-2 Re-ranking : ON  (re-ranks exactly top-K per K value)")
    print("=" * 60)

    # ---------------------------------------------------------------- #
    #  Mode 1 : Single-index evaluation                                 #
    # ---------------------------------------------------------------- #
    if args.index_path and args.meta_path and args.query_emb_path and args.query_ids_path:
        reranker = None
        query_images = gallery_captions = None

        if args.rerank:
            proc, mdl = load_blip2_reranker(args.blip2_model)
            reranker   = (proc, mdl)
            print("\n[Loading] Query images for re-ranking...")
            partition_file = f"{args.data_root}/Eval/list_eval_partition.txt"
            bbox_file      = f"{args.data_root}/Anno/list_bbox_inshop.txt"
            query_images      = load_query_images(args.data_root, partition_file, bbox_file)
            gallery_captions  = load_gallery_captions(str(Path(args.index_path).parent.parent
                                                         / "results"))

        metrics = evaluate_single_index(
            args.index_path, args.meta_path,
            args.query_emb_path, args.query_ids_path,
            K_values, args.ef_search, "Single Index",
            reranker=reranker, query_images=query_images,
            gallery_captions=gallery_captions,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"mode": "single", "reranked": args.rerank, "metrics": metrics}, f, indent=2)
        print(f"\nSaved → {args.output}")
        return

    # ---------------------------------------------------------------- #
    #  Mode 2 : Auto-discover all conditions                            #
    # ---------------------------------------------------------------- #

    # Load BLIP-2 once if needed (shared by Cond B & C)
    reranker         = None
    query_images     = None
    gallery_captions = None

    conditions_to_run = [args.condition] if args.condition else ["A", "B", "C"]
    needs_blip = args.rerank and any(
        COND_CONFIG.get(c, {}).get("rerank") for c in conditions_to_run
    )

    if needs_blip:
        proc, mdl = load_blip2_reranker(args.blip2_model)
        reranker   = (proc, mdl)

    full_report = {}   # Cond_X → {"per_run": ..., "aggregated": ...}

    for cond in conditions_to_run:
        print(f"\n{'='*60}")
        print(f"  CONDITION {cond}")
        uses_rerank = args.rerank and COND_CONFIG.get(cond, {}).get("rerank", False)
        if uses_rerank:
            print(f"  + BLIP-2 ITM Re-ranking (per K)")
        print(f"{'='*60}")

        results_dir = f"{args.results_root}/Cond_{cond}"
        index_dir   = f"{args.index_root}/Cond_{cond}"

        cond_reranker        = reranker if uses_rerank else None
        cond_query_images    = None
        cond_gallery_captions= None

        if uses_rerank:
            if query_images is None:
                print("\n[Loading] Query images for re-ranking...")
                partition_file = f"{args.data_root}/Eval/list_eval_partition.txt"
                bbox_file      = f"{args.data_root}/Anno/list_bbox_inshop.txt"
                query_images   = load_query_images(args.data_root, partition_file, bbox_file)

            if gallery_captions is None:
                # B and C share captions; try both dirs
                gallery_captions = load_gallery_captions(results_dir)

            cond_query_images     = query_images
            cond_gallery_captions = gallery_captions

        per_run, aggregated = discover_and_evaluate(
            results_dir, index_dir, cond, K_values, args.ef_search,
            reranker=cond_reranker, query_images=cond_query_images,
            gallery_captions=cond_gallery_captions,
        )

        full_report[f"Cond_{cond}"] = {
            "per_run":    per_run,
            "aggregated": aggregated,
        }

    # ---------------------------------------------------------------- #
    #  Print summary tables                                             #
    # ---------------------------------------------------------------- #
    print_aggregated_summary(full_report, K_values)
    print_compact_summary(full_report, K_values)

    # ---------------------------------------------------------------- #
    #  Save full JSON report                                            #
    # ---------------------------------------------------------------- #
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"Full report saved → {args.output}")


if __name__ == "__main__":
    main()
