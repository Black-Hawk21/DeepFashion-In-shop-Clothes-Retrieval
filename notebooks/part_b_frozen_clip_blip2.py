# ============================================================
# Condition B — Frozen CLIP + Frozen BLIP-2  (α ∈ {0.5, 0.7})
# ============================================================
# Pretrained CLIP (frozen) + BLIP-2 captions.
# Gallery: v = α·φ_V(crop) + (1-α)·φ_T(caption),  ‖v‖=1
# Query:   CLIP vision embedding only.
# Self-contained: does NOT import from src/.
# Run each "# %%" section as a separate Kaggle cell.
#
# Cached artifacts (auto-saved to OUTPUT_DIR):
#   - gallery_img_embs_frozen.npy   Gallery CLIP vision embeddings (frozen)
#   - gallery_ids.json              Gallery item IDs
#   - query_embs_frozen.npy         Query CLIP vision embeddings (frozen)
#   - query_ids.json                Query item IDs
#   - gallery_captions.json         BLIP-2 captions (reusable by Part C)
#   - gallery_txt_embs_frozen.npy   Gallery CLIP text embeddings (frozen)
#   - condB_alpha{a}_seed{s}.json   Per-(α,seed) metrics
#   - condB_summary.json            Aggregated mean±std
# ============================================================

# %% Install dependencies (run once)
# !pip install -q openai-clip hnswlib ultralytics tqdm Pillow transformers accelerate

# %% Imports and configuration
import os, json, random, time, gc
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
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ===== EDIT THESE PATHS FOR YOUR KAGGLE SETUP =====
DATA_ROOT       = "/kaggle/input/deepfashion-inshop"
PARTITION_FILE  = f"{DATA_ROOT}/Eval/list_eval_partition.txt"
OUTPUT_DIR      = "/kaggle/working/results_condB"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model settings
CLIP_MODEL_NAME = "ViT-B/32"
BLIP2_MODEL     = "Salesforce/blip2-opt-2.7b"
YOLO_MODEL_NAME = "yolov8m.pt"
USE_YOLO        = True

# Evaluation
K_VALUES     = [5, 10, 15]
SEEDS        = [510, 51]
ALPHA_VALUES = [0.5, 0.7]

# HNSW
HNSW_EF_CONSTRUCTION = 200
HNSW_M               = 16
HNSW_EF_SEARCH       = 100

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
def save_embeddings(embs, ids, emb_path, ids_path):
    np.save(emb_path, embs)
    with open(ids_path, "w") as f:
        json.dump(ids, f)
    print(f"  Saved embeddings → {emb_path}  ({embs.shape})")
    print(f"  Saved IDs        → {ids_path}  ({len(ids)} items)")

def load_embeddings(emb_path, ids_path):
    embs = np.load(emb_path)
    with open(ids_path) as f:
        ids = json.load(f)
    print(f"  Loaded embeddings ← {emb_path}  ({embs.shape})")
    return embs, ids

def save_np(arr, path):
    np.save(path, arr)
    print(f"  Saved → {path}  ({arr.shape})")

def load_np(path):
    arr = np.load(path)
    print(f"  Loaded ← {path}  ({arr.shape})")
    return arr

def cache_exists(*paths) -> bool:
    return all(os.path.exists(p) for p in paths)


# %% Metrics
def _recall_at_k(relevant, k):
    return float(relevant[:k].any())

def _ndcg_at_k(relevant, k):
    gains = relevant[:k].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
    dcg = float((gains * discounts).sum())
    n_rel = int(relevant.sum())
    ideal_k = min(n_rel, k)
    ideal_dcg = float((np.ones(ideal_k) * discounts[:ideal_k]).sum()) if ideal_k > 0 else 0.0
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def _ap_at_k(relevant, k):
    gains = relevant[:k].astype(float)
    if gains.sum() == 0:
        return 0.0
    cumsum = np.cumsum(gains)
    precisions = cumsum / np.arange(1, k + 1, dtype=float)
    return float((precisions * gains).sum() / min(int(relevant.sum()), k))

def evaluate_retrieval(query_ids, gallery_ids, ranked_indices, K_values=(5, 10, 15)):
    gallery_arr = np.array(gallery_ids)
    results = {}
    max_k = max(K_values)
    recalls = {k: [] for k in K_values}
    ndcgs   = {k: [] for k in K_values}
    aps     = {k: [] for k in K_values}
    for q_idx, q_id in enumerate(query_ids):
        top = ranked_indices[q_idx, :max_k]
        rel = (gallery_arr[top] == q_id)
        for k in K_values:
            recalls[k].append(_recall_at_k(rel, k))
            ndcgs[k].append(_ndcg_at_k(rel, k))
            aps[k].append(_ap_at_k(rel, k))
    for k in K_values:
        results[f"recall@{k}"] = float(np.mean(recalls[k]))
        results[f"ndcg@{k}"]   = float(np.mean(ndcgs[k]))
        results[f"map@{k}"]    = float(np.mean(aps[k]))
    return results

def print_metrics(metrics, K_values=(5, 10, 15)):
    header = f"{'Metric':<15}" + "".join(f"K={k:<8}" for k in K_values)
    print(header); print("-" * len(header))
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
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                c = float(box.conf.item())
                if c > best_conf:
                    best_conf = c
                    xyxy = box.xyxy[0].cpu().numpy()
                    best_box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
        if best_box is None:
            return image
        x1, y1, x2, y2 = best_box
        pw, ph = int((x2 - x1) * padding), int((y2 - y1) * padding)
        x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
        x2, y2 = min(W, x2 + pw), min(H, y2 + ph)
        return image.crop((x1, y1, x2, y2))

# %% Image loading & embedding helpers
def load_and_crop(rel_path, yolo=None):
    full_path = os.path.join(DATA_ROOT, rel_path)
    try:
        img = Image.open(full_path).convert("RGB")
    except Exception:
        return None
    if yolo is not None:
        img = yolo.detect_and_crop(img)
    return img


@torch.no_grad()
def embed_images(samples, clip_m, preproc, yolo_det=None, desc="Embedding"):
    all_embs, all_ids = [], []
    batch_t, batch_ids = [], []
    for rel_path, item_id in tqdm(samples, desc=desc):
        img = load_and_crop(rel_path, yolo_det)
        if img is None:
            continue
        batch_t.append(preproc(img))
        batch_ids.append(item_id)
        if len(batch_t) >= EMBED_BATCH_SIZE:
            b = torch.stack(batch_t).to(DEVICE)
            e = F.normalize(clip_m.encode_image(b), dim=-1)
            all_embs.append(e.cpu().float().numpy())
            all_ids.extend(batch_ids)
            batch_t, batch_ids = [], []
    if batch_t:
        b = torch.stack(batch_t).to(DEVICE)
        e = F.normalize(clip_m.encode_image(b), dim=-1)
        all_embs.append(e.cpu().float().numpy())
        all_ids.extend(batch_ids)
    return np.concatenate(all_embs, axis=0), all_ids


@torch.no_grad()
def encode_captions(captions, clip_m, batch_size=128):
    all_embs = []
    for i in tqdm(range(0, len(captions), batch_size), desc="Text encoding"):
        batch_caps = [c if c else "a clothing item" for c in captions[i:i+batch_size]]
        tokens = clip.tokenize(batch_caps, truncate=True).to(DEVICE)
        embs = clip_m.encode_text(tokens)
        embs = F.normalize(embs, dim=-1)
        all_embs.append(embs.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


# %% HNSW helpers
def build_hnsw(embeddings, ef_c=200, M=16):
    N, D = embeddings.shape
    index = hnswlib.Index(space="cosine", dim=D)
    index.init_index(max_elements=N, ef_construction=ef_c, M=M)
    index.add_items(embeddings, np.arange(N))
    print(f"[HNSW] Built index: {N} items, dim={D}")
    return index

def search_hnsw(index, queries, top_k=15, ef_search=100):
    index.set_ef(max(ef_search, top_k))
    indices, _ = index.knn_query(queries, k=top_k)
    return indices

# %% Load partition
print("=" * 60)
print("  CONDITION B — Frozen CLIP + Frozen BLIP-2")
print("=" * 60)

partition = load_partition(PARTITION_FILE)
gallery_samples = partition["gallery"]
query_samples   = partition["query"]

# Cache file paths
GALLERY_IMG_EMB_PATH = f"{OUTPUT_DIR}/gallery_img_embs_frozen.npy"
GALLERY_IDS_PATH     = f"{OUTPUT_DIR}/gallery_ids.json"
QUERY_EMB_PATH       = f"{OUTPUT_DIR}/query_embs_frozen.npy"
QUERY_IDS_PATH       = f"{OUTPUT_DIR}/query_ids.json"
CAPTION_CACHE_PATH   = f"{OUTPUT_DIR}/gallery_captions.json"
GALLERY_TXT_EMB_PATH = f"{OUTPUT_DIR}/gallery_txt_embs_frozen.npy"

# %% Step 1: Gallery image embeddings (CLIP vision, frozen)
if cache_exists(GALLERY_IMG_EMB_PATH, GALLERY_IDS_PATH):
    print("\n[Cache] Loading gallery image embeddings...")
    gallery_img_embs, gallery_ids = load_embeddings(GALLERY_IMG_EMB_PATH, GALLERY_IDS_PATH)
else:
    print("\n[Compute] Embedding gallery images (CLIP vision, frozen)...")
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    clip_model = clip_model.float().eval()
    yolo = YOLODetector(YOLO_MODEL_NAME) if USE_YOLO else None

    gallery_img_embs, gallery_ids = embed_images(
        gallery_samples, clip_model, preprocess, yolo, "Gallery images"
    )
    save_embeddings(gallery_img_embs, gallery_ids, GALLERY_IMG_EMB_PATH, GALLERY_IDS_PATH)

# %% Step 2: Query embeddings (CLIP vision, frozen)
if cache_exists(QUERY_EMB_PATH, QUERY_IDS_PATH):
    print("\n[Cache] Loading query embeddings...")
    query_embs, query_ids = load_embeddings(QUERY_EMB_PATH, QUERY_IDS_PATH)
else:
    print("\n[Compute] Embedding query images (CLIP vision, frozen)...")
    if 'clip_model' not in dir():
        clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
        clip_model = clip_model.float().eval()
        yolo = YOLODetector(YOLO_MODEL_NAME) if USE_YOLO else None

    query_embs, query_ids = embed_images(
        query_samples, clip_model, preprocess, yolo, "Query images"
    )
    save_embeddings(query_embs, query_ids, QUERY_EMB_PATH, QUERY_IDS_PATH)

# %% Step 3: BLIP-2 captions for gallery
if cache_exists(CAPTION_CACHE_PATH):
    print(f"\n[Cache] Loading BLIP-2 captions...")
    with open(CAPTION_CACHE_PATH) as f:
        caption_data = json.load(f)
    gallery_captions = caption_data["captions"]
    print(f"  Loaded {len(gallery_captions)} captions")
else:
    print("\n[Compute] Generating BLIP-2 captions for gallery...")
    print("  Loading BLIP-2 model (takes ~1-2 min)...")
    blip_processor = Blip2Processor.from_pretrained(BLIP2_MODEL)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_MODEL, device_map="auto", torch_dtype=torch.float16,
    )
    blip_model.eval()

    if 'yolo' not in dir() or yolo is None:
        yolo = YOLODetector(YOLO_MODEL_NAME) if USE_YOLO else None

    gallery_captions = []
    prompt = "A photo of a clothing item:"

    for i, (rel_path, item_id) in enumerate(tqdm(gallery_samples, desc="Captioning gallery")):
        img = load_and_crop(rel_path, yolo)
        if img is None:
            gallery_captions.append("")
            continue
        try:
            inputs = blip_processor(
                images=img, text=prompt, return_tensors="pt"
            ).to(blip_model.device, dtype=torch.float16)
            with torch.no_grad():
                out = blip_model.generate(**inputs, max_new_tokens=50)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            caption = caption.replace(prompt, "").strip()
        except Exception:
            caption = ""
        gallery_captions.append(caption)

        # Save progress every 1000 images
        if (i + 1) % 1000 == 0:
            with open(CAPTION_CACHE_PATH, "w") as f:
                json.dump({"captions": gallery_captions, "progress": i + 1}, f)
            print(f"  Checkpoint saved at {i + 1}/{len(gallery_samples)}")

    # Final save
    with open(CAPTION_CACHE_PATH, "w") as f:
        json.dump({"captions": gallery_captions, "progress": len(gallery_samples)}, f)
    print(f"  Saved {len(gallery_captions)} captions → {CAPTION_CACHE_PATH}")

    del blip_model, blip_processor
    gc.collect(); torch.cuda.empty_cache()
    print("  BLIP-2 model unloaded.")

# %% Step 4: Gallery text embeddings (CLIP text encoder, frozen)
if cache_exists(GALLERY_TXT_EMB_PATH):
    print("\n[Cache] Loading gallery text embeddings...")
    gallery_txt_embs = load_np(GALLERY_TXT_EMB_PATH)
else:
    print("\n[Compute] Encoding gallery captions with CLIP text encoder...")
    if 'clip_model' not in dir():
        clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
        clip_model = clip_model.float().eval()

    gallery_txt_embs = encode_captions(gallery_captions, clip_model)
    save_np(gallery_txt_embs, GALLERY_TXT_EMB_PATH)

# Free models
for v in ['clip_model', 'yolo', 'preprocess']:
    if v in dir():
        exec(f"del {v}")
gc.collect(); torch.cuda.empty_cache()

# %% Step 5: Evaluate across all (α, seed) combinations
print("\n" + "=" * 60)
print(f"  Gallery img: {gallery_img_embs.shape}  |  Gallery txt: {gallery_txt_embs.shape}")
print(f"  Queries: {query_embs.shape}")
print("=" * 60)

all_results = {}

for alpha in ALPHA_VALUES:
    print(f"\n{'='*50}")
    print(f"  α = {alpha}")
    print(f"{'='*50}")

    # Fuse: v = α·img + (1-α)·txt, then L2 normalize
    fused = alpha * gallery_img_embs + (1 - alpha) * gallery_txt_embs
    fused = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)

    # Save fused embeddings
    fused_path = f"{OUTPUT_DIR}/gallery_fused_alpha{alpha}.npy"
    save_np(fused, fused_path)

    for seed in SEEDS:
        print(f"\n--- seed={seed}, α={alpha} ---")
        set_seed(seed)
        t0 = time.time()

        index = build_hnsw(fused, HNSW_EF_CONSTRUCTION, HNSW_M)
        max_k = max(K_VALUES)
        ranked = search_hnsw(index, query_embs, top_k=max_k, ef_search=HNSW_EF_SEARCH)
        metrics = evaluate_retrieval(query_ids, gallery_ids, ranked, K_VALUES)

        print(f"Results (took {time.time()-t0:.1f}s):")
        print_metrics(metrics, K_VALUES)
        all_results[(alpha, seed)] = metrics

        out = {"condition": "B", "alpha": alpha, "seed": seed, "metrics": metrics}
        path = f"{OUTPUT_DIR}/condB_alpha{alpha}_seed{seed}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved → {path}")

        del index
        torch.cuda.empty_cache()

# %% Aggregate results
print("\n" + "=" * 60)
print("  CONDITION B — AGGREGATED RESULTS")
print("=" * 60)

metric_keys = [f"{m}@{k}" for m in ("recall", "ndcg", "map") for k in K_VALUES]
summary = {}

for alpha in ALPHA_VALUES:
    key = f"alpha={alpha}"
    seed_metrics = [all_results.get((alpha, s), {}) for s in SEEDS]
    agg = {}
    for mk in metric_keys:
        vals = [m[mk] for m in seed_metrics if mk in m]
        if vals:
            agg[mk] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary[key] = agg
    print(f"\n  α = {alpha}:")
    for mk in metric_keys:
        if mk in agg:
            print(f"    {mk:<12}  {agg[mk]['mean']:.4f} ± {agg[mk]['std']:.4f}")

with open(f"{OUTPUT_DIR}/condB_summary.json", "w") as f:
    json.dump({"condition": "B", "alphas": ALPHA_VALUES, "seeds": SEEDS, "summary": summary}, f, indent=2)
print(f"\nSummary saved → {OUTPUT_DIR}/condB_summary.json")
