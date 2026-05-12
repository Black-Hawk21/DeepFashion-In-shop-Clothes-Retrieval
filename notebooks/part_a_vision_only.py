# ============================================================
# Condition A — Vision-Only CLIP Baseline  (α = 1.0)
# ============================================================
# Frozen pretrained CLIP, no BLIP-2, no fine-tuning.
# YOLO crops the clothing region, CLIP encodes it.
# Self-contained: does NOT import from src/.
# Run each "# %%" section as a separate Kaggle cell.
#
# Cached artifacts (auto-saved to OUTPUT_DIR):
#   - gallery_img_embs.npy    Gallery CLIP vision embeddings
#   - gallery_ids.json        Gallery item IDs
#   - query_embs.npy          Query CLIP vision embeddings
#   - query_ids.json          Query item IDs
#   - condA_seed{s}.json      Per-seed metrics
#   - condA_summary.json      Aggregated mean±std
# ============================================================

# %% Install dependencies (run once)
# !pip install -q openai-clip hnswlib ultralytics tqdm Pillow

# %% Imports and configuration
import os, json, random, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
import clip
import hnswlib
from ultralytics import YOLO

# ===== EDIT THESE PATHS FOR YOUR KAGGLE SETUP =====
DATA_ROOT       = "/kaggle/input/deepfashion-inshop"
PARTITION_FILE  = f"{DATA_ROOT}/Eval/list_eval_partition.txt"
OUTPUT_DIR      = "/kaggle/working/results_condA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model settings
CLIP_MODEL_NAME = "ViT-B/32"
YOLO_MODEL_NAME = "yolov8m.pt"
USE_YOLO        = True          # Set False to skip YOLO cropping

# Evaluation
K_VALUES = [5, 10, 15]
SEEDS    = [510, 51]            # Team roll numbers

# HNSW
HNSW_EF_CONSTRUCTION = 200
HNSW_M               = 16
HNSW_EF_SEARCH       = 100

# Batching
EMBED_BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# %% Utility functions
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] All seeds set to {seed}")


def load_partition(path: str) -> Dict[str, List[Tuple[str, str]]]:
    """Parse list_eval_partition.txt → {split: [(rel_path, item_id), ...]}"""
    splits = {"train": [], "query": [], "gallery": []}
    with open(path) as f:
        lines = f.read().splitlines()
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        img_name, item_id, status = parts[0], parts[1], parts[2]
        if status in splits:
            splits[status].append((img_name, item_id))
    for k, v in splits.items():
        print(f"  {k}: {len(v)} images")
    return splits


# ---- Cache helpers ----
def save_embeddings(embs: np.ndarray, ids: list, emb_path: str, ids_path: str):
    np.save(emb_path, embs)
    with open(ids_path, "w") as f:
        json.dump(ids, f)
    print(f"  Saved embeddings → {emb_path}  ({embs.shape})")
    print(f"  Saved IDs        → {ids_path}  ({len(ids)} items)")

def load_embeddings(emb_path: str, ids_path: str):
    embs = np.load(emb_path)
    with open(ids_path) as f:
        ids = json.load(f)
    print(f"  Loaded embeddings ← {emb_path}  ({embs.shape})")
    return embs, ids

def cache_exists(*paths) -> bool:
    return all(os.path.exists(p) for p in paths)


# %% Metrics
def _recall_at_k(relevant: np.ndarray, k: int) -> float:
    return float(relevant[:k].any())

def _ndcg_at_k(relevant: np.ndarray, k: int) -> float:
    gains = relevant[:k].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
    dcg = float((gains * discounts).sum())
    n_relevant = int(relevant.sum())
    ideal_k = min(n_relevant, k)
    ideal_dcg = float((np.ones(ideal_k) * discounts[:ideal_k]).sum()) if ideal_k > 0 else 0.0
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def _ap_at_k(relevant: np.ndarray, k: int) -> float:
    gains = relevant[:k].astype(float)
    if gains.sum() == 0:
        return 0.0
    cumsum = np.cumsum(gains)
    positions = np.arange(1, k + 1, dtype=float)
    precisions = cumsum / positions
    return float((precisions * gains).sum() / min(int(relevant.sum()), k))

def evaluate_retrieval(query_ids, gallery_ids, ranked_indices, K_values=(5, 10, 15)):
    gallery_ids_arr = np.array(gallery_ids)
    results = {}
    max_k = max(K_values)
    recalls = {k: [] for k in K_values}
    ndcgs   = {k: [] for k in K_values}
    aps     = {k: [] for k in K_values}
    for q_idx, q_id in enumerate(query_ids):
        top_ranked = ranked_indices[q_idx, :max_k]
        retrieved_ids = gallery_ids_arr[top_ranked]
        relevant = (retrieved_ids == q_id)
        for k in K_values:
            recalls[k].append(_recall_at_k(relevant, k))
            ndcgs[k].append(_ndcg_at_k(relevant, k))
            aps[k].append(_ap_at_k(relevant, k))
    for k in K_values:
        results[f"recall@{k}"] = float(np.mean(recalls[k]))
        results[f"ndcg@{k}"]   = float(np.mean(ndcgs[k]))
        results[f"map@{k}"]    = float(np.mean(aps[k]))
    return results

def print_metrics(metrics, K_values=(5, 10, 15)):
    header = f"{'Metric':<15}" + "".join(f"K={k:<8}" for k in K_values)
    print(header)
    print("-" * len(header))
    for prefix in ("recall", "ndcg", "map"):
        row = f"{prefix.upper():<15}"
        for k in K_values:
            row += f"{metrics.get(f'{prefix}@{k}', 0):<8.4f}"
        print(row)

# %% YOLO detector
class YOLODetector:
    def __init__(self, model_name="yolov8m.pt", conf=0.25, iou=0.45):
        print(f"[YOLO] Loading {model_name}")
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def detect_and_crop(self, image: Image.Image, padding=0.05):
        W, H = image.size
        results = self.model.predict(
            source=np.array(image), conf=self.conf, iou=self.iou,
            device=self.device, verbose=False,
        )
        best_conf, best_box = -1, None
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf > best_conf:
                    best_conf = conf
                    xyxy = box.xyxy[0].cpu().numpy()
                    best_box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
        if best_box is None:
            return image
        x1, y1, x2, y2 = best_box
        pw, ph = int((x2 - x1) * padding), int((y2 - y1) * padding)
        x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
        x2, y2 = min(W, x2 + pw), min(H, y2 + ph)
        return image.crop((x1, y1, x2, y2))

# %% Embedding helpers
def load_and_crop(rel_path: str, yolo: Optional[YOLODetector] = None) -> Optional[Image.Image]:
    full_path = os.path.join(DATA_ROOT, rel_path)
    try:
        img = Image.open(full_path).convert("RGB")
    except Exception:
        return None
    if yolo is not None:
        img = yolo.detect_and_crop(img)
    return img


@torch.no_grad()
def embed_split(samples, clip_model, preprocess, yolo=None, desc="Embedding"):
    """Embed a list of (rel_path, item_id) samples with CLIP vision encoder."""
    all_embs, all_ids = [], []
    batch_tensors, batch_ids = [], []

    for rel_path, item_id in tqdm(samples, desc=desc):
        img = load_and_crop(rel_path, yolo)
        if img is None:
            continue
        tensor = preprocess(img)
        batch_tensors.append(tensor)
        batch_ids.append(item_id)

        if len(batch_tensors) >= EMBED_BATCH_SIZE:
            batch = torch.stack(batch_tensors).to(DEVICE)
            embs = clip_model.encode_image(batch)
            embs = F.normalize(embs, dim=-1)
            all_embs.append(embs.cpu().float().numpy())
            all_ids.extend(batch_ids)
            batch_tensors, batch_ids = [], []

    if batch_tensors:
        batch = torch.stack(batch_tensors).to(DEVICE)
        embs = clip_model.encode_image(batch)
        embs = F.normalize(embs, dim=-1)
        all_embs.append(embs.cpu().float().numpy())
        all_ids.extend(batch_ids)

    return np.concatenate(all_embs, axis=0), all_ids

# %% HNSW index helpers
def build_hnsw(embeddings: np.ndarray, ef_c=200, M=16):
    N, D = embeddings.shape
    index = hnswlib.Index(space="cosine", dim=D)
    index.init_index(max_elements=N, ef_construction=ef_c, M=M)
    index.add_items(embeddings, np.arange(N))
    print(f"[HNSW] Built index with {N} items, dim={D}")
    return index

def search_hnsw(index, queries: np.ndarray, top_k=15, ef_search=100):
    index.set_ef(max(ef_search, top_k))
    indices, distances = index.knn_query(queries, k=top_k)
    return indices

# %% Main: Load data and models
print("=" * 60)
print("  CONDITION A — Vision-Only CLIP Baseline (α = 1.0)")
print("=" * 60)

partition = load_partition(PARTITION_FILE)
gallery_samples = partition["gallery"]
query_samples   = partition["query"]

# Paths for cached artifacts
GALLERY_EMB_PATH = f"{OUTPUT_DIR}/gallery_img_embs.npy"
GALLERY_IDS_PATH = f"{OUTPUT_DIR}/gallery_ids.json"
QUERY_EMB_PATH   = f"{OUTPUT_DIR}/query_embs.npy"
QUERY_IDS_PATH   = f"{OUTPUT_DIR}/query_ids.json"

# %% Compute or load gallery embeddings
if cache_exists(GALLERY_EMB_PATH, GALLERY_IDS_PATH):
    print("\n[Cache] Loading gallery embeddings from disk...")
    gallery_embs, gallery_ids = load_embeddings(GALLERY_EMB_PATH, GALLERY_IDS_PATH)
else:
    print("\n[Compute] Embedding gallery images (CLIP vision)...")
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    clip_model = clip_model.float().eval()
    yolo = YOLODetector(YOLO_MODEL_NAME) if USE_YOLO else None

    gallery_embs, gallery_ids = embed_split(
        gallery_samples, clip_model, preprocess, yolo, desc="Gallery"
    )
    save_embeddings(gallery_embs, gallery_ids, GALLERY_EMB_PATH, GALLERY_IDS_PATH)

# %% Compute or load query embeddings
if cache_exists(QUERY_EMB_PATH, QUERY_IDS_PATH):
    print("\n[Cache] Loading query embeddings from disk...")
    query_embs, query_ids = load_embeddings(QUERY_EMB_PATH, QUERY_IDS_PATH)
else:
    print("\n[Compute] Embedding query images (CLIP vision)...")
    # Load models only if not already loaded above
    if 'clip_model' not in dir():
        clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
        clip_model = clip_model.float().eval()
        yolo = YOLODetector(YOLO_MODEL_NAME) if USE_YOLO else None

    query_embs, query_ids = embed_split(
        query_samples, clip_model, preprocess, yolo, desc="Queries"
    )
    save_embeddings(query_embs, query_ids, QUERY_EMB_PATH, QUERY_IDS_PATH)

# Free models if loaded (not needed further)
for v in ('clip_model', 'yolo'):
    if v in dir():
        exec(f"del {v}")
torch.cuda.empty_cache()

# %% Evaluate across seeds
print("\n" + "=" * 60)
print(f"  Gallery: {gallery_embs.shape}  |  Queries: {query_embs.shape}")
print("=" * 60)

all_seed_results = []

for seed in SEEDS:
    print(f"\n--- seed={seed} ---")
    set_seed(seed)
    t0 = time.time()

    # Build HNSW index
    index = build_hnsw(gallery_embs, HNSW_EF_CONSTRUCTION, HNSW_M)

    # Retrieve & evaluate
    max_k = max(K_VALUES)
    ranked = search_hnsw(index, query_embs, top_k=max_k, ef_search=HNSW_EF_SEARCH)
    metrics = evaluate_retrieval(query_ids, gallery_ids, ranked, K_VALUES)

    elapsed = time.time() - t0
    print(f"Results for seed={seed} (took {elapsed:.1f}s):")
    print_metrics(metrics, K_VALUES)
    all_seed_results.append(metrics)

    # Save per-seed results
    out = {"condition": "A", "alpha": 1.0, "seed": seed, "metrics": metrics}
    path = f"{OUTPUT_DIR}/condA_seed{seed}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {path}")

    del index
    torch.cuda.empty_cache()

# %% Aggregate results (mean ± std over seeds)
print("\n" + "=" * 60)
print("  CONDITION A — AGGREGATED RESULTS")
print("=" * 60)

metric_keys = [f"{m}@{k}" for m in ("recall", "ndcg", "map") for k in K_VALUES]
summary = {}
for mk in metric_keys:
    vals = [r[mk] for r in all_seed_results]
    summary[mk] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    print(f"  {mk:<12}  {summary[mk]['mean']:.4f} ± {summary[mk]['std']:.4f}")

with open(f"{OUTPUT_DIR}/condA_summary.json", "w") as f:
    json.dump({"condition": "A", "alpha": 1.0, "seeds": SEEDS, "summary": summary}, f, indent=2)
print(f"\nSummary saved → {OUTPUT_DIR}/condA_summary.json")
