"""
src/utils/metrics.py
---------------------
Retrieval evaluation metrics:
    - Recall@K
    - NDCG@K
    - mAP@K

All functions accept:
    query_ids   : list[str]  — item_id for each query image
    gallery_ids : list[str]  — item_id for each gallery image
    ranked_indices : np.ndarray shape (n_queries, n_gallery) — indices into
                     gallery_ids, sorted by descending similarity for each query

Usage:
    results = evaluate_retrieval(query_ids, gallery_ids, ranked_indices, K_values=[5,10,15])
    print(results)
    # {'recall@5': 0.72, 'ndcg@5': 0.61, 'map@5': 0.58, ...}
"""

from typing import Dict, List

import numpy as np


# ------------------------------------------------------------------ #
#  Per-query helpers                                                   #
# ------------------------------------------------------------------ #

def _recall_at_k(relevant: np.ndarray, k: int) -> float:
    """relevant: boolean mask over top-K gallery items."""
    return float(relevant[:k].any())


def _ndcg_at_k(relevant: np.ndarray, k: int) -> float:
    """Normalised Discounted Cumulative Gain at K."""
    gains = relevant[:k].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
    dcg = float((gains * discounts).sum())

    # Ideal DCG: all relevant docs at the top
    n_relevant = int(relevant.sum())
    ideal_k = min(n_relevant, k)
    ideal_dcg = float((np.ones(ideal_k) * discounts[:ideal_k]).sum())

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def _ap_at_k(relevant: np.ndarray, k: int) -> float:
    """Average Precision at K."""
    gains = relevant[:k].astype(float)
    if gains.sum() == 0:
        return 0.0
    cumsum = np.cumsum(gains)
    positions = np.arange(1, k + 1, dtype=float)
    precisions = cumsum / positions
    return float((precisions * gains).sum() / min(int(relevant.sum()), k))


# ------------------------------------------------------------------ #
#  Main evaluation function                                            #
# ------------------------------------------------------------------ #

def evaluate_retrieval(
    query_ids: List[str],
    gallery_ids: List[str],
    ranked_indices: np.ndarray,  # shape (n_queries, n_gallery)
    K_values: List[int] = (5, 10, 15),
) -> Dict[str, float]:
    """
    Compute Recall@K, NDCG@K, mAP@K for each K in K_values.

    Args:
        query_ids:      item_id for each query image (length n_queries)
        gallery_ids:    item_id for each gallery image (length n_gallery)
        ranked_indices: for each query, gallery indices sorted best-first
        K_values:       list of K values to evaluate at

    Returns:
        dict mapping metric names to float values, e.g.
        {'recall@5': 0.72, 'ndcg@5': 0.65, 'map@5': 0.60, 'recall@10': ...}
    """
    gallery_ids_arr = np.array(gallery_ids)
    results: Dict[str, float] = {}

    max_k = max(K_values)
    recalls = {k: [] for k in K_values}
    ndcgs   = {k: [] for k in K_values}
    aps     = {k: [] for k in K_values}

    for q_idx, q_id in enumerate(query_ids):
        top_ranked = ranked_indices[q_idx, :max_k]           # top max_k gallery indices
        retrieved_ids = gallery_ids_arr[top_ranked]
        relevant = (retrieved_ids == q_id)                    # bool mask

        for k in K_values:
            recalls[k].append(_recall_at_k(relevant, k))
            ndcgs[k].append(_ndcg_at_k(relevant, k))
            aps[k].append(_ap_at_k(relevant, k))

    for k in K_values:
        results[f"recall@{k}"] = float(np.mean(recalls[k]))
        results[f"ndcg@{k}"]   = float(np.mean(ndcgs[k]))
        results[f"map@{k}"]    = float(np.mean(aps[k]))

    return results


def format_metrics(metrics: Dict[str, float], K_values: List[int] = (5, 10, 15)) -> str:
    """Pretty-print metrics as a formatted table string."""
    header = f"{'Metric':<15}" + "".join(f"K={k:<8}" for k in K_values)
    rows = []
    for prefix in ("recall", "ndcg", "map"):
        row = f"{prefix.upper():<15}"
        for k in K_values:
            key = f"{prefix}@{k}"
            row += f"{metrics.get(key, float('nan')):<8.4f}"
        rows.append(row)
    return "\n".join([header, "-" * len(header)] + rows)
