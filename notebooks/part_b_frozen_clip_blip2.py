# ============================================================
# Condition B — Frozen CLIP + Frozen BLIP-2  (α ∈ {0.5, 0.7})
# ============================================================
# Pretrained CLIP (frozen) + BLIP-2 captions.
# Uses ground-truth bbox annotations to crop clothing regions.
# Gallery: v = α·φ_V(crop) + (1-α)·φ_T(caption),  ‖v‖=1
# Query:   CLIP vision embedding only.
# Self-contained. Run each "# %%" section as a separate Kaggle cell.
#
# Cached artifacts (auto-saved to OUTPUT_DIR):
#   - gallery_img_embs_frozen.npy / gallery_ids.json
#   - query_embs_frozen.npy / query_ids.json
#   - gallery_captions.json          (reusable by Part C)
#   - gallery_txt_embs_frozen.npy
#   - gallery_fused_alpha{a}.npy
#   - condB_alpha{a}_seed{s}.json / condB_summary.json
# ============================================================

# %% Install dependencies (run once)
# !pip install -q openai-clip hnswlib tqdm Pillow transformers accelerate

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
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ===== PATHS (local project) =====
DATA_ROOT      = "data/deepfashion"
PARTITION_FILE = f"{DATA_ROOT}/Eval/list_eval_partition.txt"
BBOX_FILE      = f"{DATA_ROOT}/Anno/list_bbox_inshop.txt"
OUTPUT_DIR     = "results/condB"
INDEX_DIR      = "index/condB"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

CLIP_MODEL_NAME = "ViT-B/32"
BLIP2_MODEL     = "Salesforce/blip2-opt-2.7b"

K_VALUES     = [5, 10, 15]
SEEDS        = [510, 51]
ALPHA_VALUES = [0.5, 0.7]

HNSW_EF_CONSTRUCTION = 200
HNSW_M               = 16
HNSW_EF_SEARCH       = 100
EMBED_BATCH_SIZE     = 128
BBOX_PADDING         = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# %% Utility functions
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] {seed}")

def load_partition(path):
    splits = {"train": [], "query": [], "gallery": []}
    with open(path) as f:
        lines = f.read().splitlines()
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 3: continue
        if parts[2] in splits:
            splits[parts[2]].append((parts[0], parts[1]))
    for k, v in splits.items(): print(f"  {k}: {len(v)} images")
    return splits

def load_bbox_annotations(path):
    bbox = {}
    with open(path) as f:
        lines = f.read().splitlines()
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 7: continue
        bbox[parts[0]] = (int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6]))
    print(f"  Loaded {len(bbox)} bounding boxes")
    return bbox

def crop_with_bbox(image, bbox, padding=0.05):
    W, H = image.size
    x1, y1, x2, y2 = bbox
    pw, ph = int((x2 - x1) * padding), int((y2 - y1) * padding)
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(W, x2 + pw), min(H, y2 + ph)
    return image.crop((x1, y1, x2, y2))

def load_and_crop(rel_path, bbox_map, padding=0.05):
    full_path = os.path.join(DATA_ROOT, rel_path)
    try:
        img = Image.open(full_path).convert("RGB")
    except Exception:
        return None
    if rel_path in bbox_map:
        img = crop_with_bbox(img, bbox_map[rel_path], padding)
    return img

def save_embeddings(embs, ids, ep, ip):
    np.save(ep, embs)
    with open(ip, "w") as f: json.dump(ids, f)
    print(f"  Saved → {ep} ({embs.shape}), {ip} ({len(ids)})")

def load_embeddings(ep, ip):
    embs = np.load(ep)
    with open(ip) as f: ids = json.load(f)
    print(f"  Loaded ← {ep} ({embs.shape})")
    return embs, ids

def save_np(arr, path):
    np.save(path, arr); print(f"  Saved → {path} ({arr.shape})")

def load_np(path):
    arr = np.load(path); print(f"  Loaded ← {path} ({arr.shape})"); return arr

def cache_exists(*paths): return all(os.path.exists(p) for p in paths)

# %% Metrics
def _recall_at_k(r, k): return float(r[:k].any())
def _ndcg_at_k(r, k):
    g = r[:k].astype(float)
    d = 1.0 / np.log2(np.arange(2, k+2, dtype=float))
    dcg = float((g * d).sum())
    n = min(int(r.sum()), k)
    idcg = float((np.ones(n) * d[:n]).sum()) if n > 0 else 0.0
    return dcg / idcg if idcg > 0 else 0.0
def _ap_at_k(r, k):
    g = r[:k].astype(float)
    if g.sum() == 0: return 0.0
    cs = np.cumsum(g)
    return float((cs / np.arange(1, k+1, dtype=float) * g).sum() / min(int(r.sum()), k))

def evaluate_retrieval(qids, gids, ranked, Ks=(5, 10, 15)):
    ga = np.array(gids); res = {}; mk = max(Ks)
    recs = {k: [] for k in Ks}; nds = {k: [] for k in Ks}; aps = {k: [] for k in Ks}
    for qi, qid in enumerate(qids):
        rel = (ga[ranked[qi, :mk]] == qid)
        for k in Ks:
            recs[k].append(_recall_at_k(rel, k))
            nds[k].append(_ndcg_at_k(rel, k))
            aps[k].append(_ap_at_k(rel, k))
    for k in Ks:
        res[f"recall@{k}"] = float(np.mean(recs[k]))
        res[f"ndcg@{k}"] = float(np.mean(nds[k]))
        res[f"map@{k}"] = float(np.mean(aps[k]))
    return res

def print_metrics(m, Ks=(5, 10, 15)):
    h = f"{'Metric':<15}" + "".join(f"K={k:<8}" for k in Ks)
    print(h); print("-" * len(h))
    for p in ("recall", "ndcg", "map"):
        r = f"{p.upper():<15}"
        for k in Ks: r += f"{m.get(f'{p}@{k}', 0):<8.4f}"
        print(r)

# %% Embedding helpers
@torch.no_grad()
def embed_images(samples, clip_m, preproc, bbox_map, desc="Emb"):
    ae, ai, bt, bi = [], [], [], []
    for rp, iid in tqdm(samples, desc=desc):
        img = load_and_crop(rp, bbox_map, BBOX_PADDING)
        if img is None: continue
        bt.append(preproc(img)); bi.append(iid)
        if len(bt) >= EMBED_BATCH_SIZE:
            b = torch.stack(bt).to(DEVICE)
            ae.append(F.normalize(clip_m.encode_image(b), dim=-1).cpu().float().numpy())
            ai.extend(bi); bt, bi = [], []
    if bt:
        b = torch.stack(bt).to(DEVICE)
        ae.append(F.normalize(clip_m.encode_image(b), dim=-1).cpu().float().numpy())
        ai.extend(bi)
    return np.concatenate(ae, axis=0), ai

@torch.no_grad()
def encode_captions(captions, clip_m, bs=128):
    ae = []
    for i in tqdm(range(0, len(captions), bs), desc="TxtEnc"):
        bc = [c if c else "a clothing item" for c in captions[i:i+bs]]
        t = clip.tokenize(bc, truncate=True).to(DEVICE)
        ae.append(F.normalize(clip_m.encode_text(t), dim=-1).cpu().float().numpy())
    return np.concatenate(ae, axis=0)

# %% HNSW helpers
def build_hnsw(embs, ef_c=200, M=16):
    N, D = embs.shape
    idx = hnswlib.Index(space="cosine", dim=D)
    idx.init_index(max_elements=N, ef_construction=ef_c, M=M)
    idx.add_items(embs, np.arange(N))
    print(f"[HNSW] {N} items, dim={D}"); return idx

def search_hnsw(idx, q, top_k=15, ef=100):
    idx.set_ef(max(ef, top_k))
    i, _ = idx.knn_query(q, k=top_k); return i

# %% Load data
print("=" * 60)
print("  CONDITION B — Frozen CLIP + Frozen BLIP-2")
print("=" * 60)

partition = load_partition(PARTITION_FILE)
bbox_map = load_bbox_annotations(BBOX_FILE)
gallery_samples = partition["gallery"]
query_samples   = partition["query"]

# Cache paths
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
    print("\n[Compute] Embedding gallery images (bbox-cropped, CLIP vision)...")
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    clip_model = clip_model.float().eval()
    gallery_img_embs, gallery_ids = embed_images(
        gallery_samples, clip_model, preprocess, bbox_map, "Gallery images"
    )
    save_embeddings(gallery_img_embs, gallery_ids, GALLERY_IMG_EMB_PATH, GALLERY_IDS_PATH)

# %% Step 2: Query embeddings (CLIP vision, frozen)
if cache_exists(QUERY_EMB_PATH, QUERY_IDS_PATH):
    print("\n[Cache] Loading query embeddings...")
    query_embs, query_ids = load_embeddings(QUERY_EMB_PATH, QUERY_IDS_PATH)
else:
    print("\n[Compute] Embedding query images (bbox-cropped, CLIP vision)...")
    if 'clip_model' not in dir():
        clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
        clip_model = clip_model.float().eval()
    query_embs, query_ids = embed_images(
        query_samples, clip_model, preprocess, bbox_map, "Query images"
    )
    save_embeddings(query_embs, query_ids, QUERY_EMB_PATH, QUERY_IDS_PATH)

# %% Step 3: BLIP-2 captions for gallery (on bbox-cropped images)
if cache_exists(CAPTION_CACHE_PATH):
    print(f"\n[Cache] Loading BLIP-2 captions...")
    with open(CAPTION_CACHE_PATH) as f:
        gallery_captions = json.load(f)["captions"]
    print(f"  {len(gallery_captions)} captions loaded")
else:
    print("\n[Compute] Generating BLIP-2 captions on bbox-cropped gallery images...")
    print("  Loading BLIP-2 model (takes ~1-2 min)...")
    blip_processor = Blip2Processor.from_pretrained(BLIP2_MODEL)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_MODEL, device_map="auto", torch_dtype=torch.float16,
    )
    blip_model.eval()

    gallery_captions = []
    prompt = "A photo of a clothing item:"

    for i, (rel_path, item_id) in enumerate(tqdm(gallery_samples, desc="Captioning gallery")):
        img = load_and_crop(rel_path, bbox_map, BBOX_PADDING)
        if img is None:
            gallery_captions.append(""); continue
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

        # Checkpoint every 1000 images
        if (i + 1) % 1000 == 0:
            with open(CAPTION_CACHE_PATH, "w") as f:
                json.dump({"captions": gallery_captions, "progress": i + 1}, f)
            print(f"  Checkpoint at {i + 1}/{len(gallery_samples)}")

    with open(CAPTION_CACHE_PATH, "w") as f:
        json.dump({"captions": gallery_captions, "progress": len(gallery_samples)}, f)
    print(f"  Saved {len(gallery_captions)} captions → {CAPTION_CACHE_PATH}")

    del blip_model, blip_processor
    gc.collect(); torch.cuda.empty_cache()

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
for v in ['clip_model', 'preprocess']:
    if v in dir(): exec(f"del {v}")
gc.collect(); torch.cuda.empty_cache()

# %% Step 5: Evaluate across (α, seed) combinations
print(f"\n  Gallery img: {gallery_img_embs.shape}  |  Gallery txt: {gallery_txt_embs.shape}")
print(f"  Queries: {query_embs.shape}")

all_results = {}

for alpha in ALPHA_VALUES:
    print(f"\n{'='*50}\n  α = {alpha}\n{'='*50}")
    fused = alpha * gallery_img_embs + (1 - alpha) * gallery_txt_embs
    fused = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)
    save_np(fused, f"{OUTPUT_DIR}/gallery_fused_alpha{alpha}.npy")

    for seed in SEEDS:
        print(f"\n--- seed={seed}, α={alpha} ---")
        set_seed(seed); t0 = time.time()
        index = build_hnsw(fused, HNSW_EF_CONSTRUCTION, HNSW_M)
        ranked = search_hnsw(index, query_embs, top_k=max(K_VALUES), ef=HNSW_EF_SEARCH)
        metrics = evaluate_retrieval(query_ids, gallery_ids, ranked, K_VALUES)
        print(f"Results (took {time.time()-t0:.1f}s):")
        print_metrics(metrics, K_VALUES)
        all_results[(alpha, seed)] = metrics

        out = {"condition": "B", "alpha": alpha, "seed": seed, "metrics": metrics}
        with open(f"{OUTPUT_DIR}/condB_alpha{alpha}_seed{seed}.json", "w") as f:
            json.dump(out, f, indent=2)

        # Save HNSW index to disk
        idx_path = f"{INDEX_DIR}/hnsw_alpha{alpha}_seed{seed}.bin"
        index.save_index(idx_path)
        print(f"  Index saved → {idx_path}")

        # Save metadata
        meta_path = f"{INDEX_DIR}/metadata_alpha{alpha}_seed{seed}.json"
        with open(meta_path, "w") as f:
            json.dump({"gallery_ids": gallery_ids, "alpha": alpha, "seed": seed,
                       "dim": int(gallery_img_embs.shape[1]), "n_items": len(gallery_ids)}, f)
        print(f"  Metadata  → {meta_path}")

        del index; torch.cuda.empty_cache()

# %% Aggregate results
print("\n" + "=" * 60)
print("  CONDITION B — AGGREGATED RESULTS")
print("=" * 60)
metric_keys = [f"{m}@{k}" for m in ("recall", "ndcg", "map") for k in K_VALUES]
summary = {}
for alpha in ALPHA_VALUES:
    key = f"alpha={alpha}"
    sm = [all_results.get((alpha, s), {}) for s in SEEDS]
    agg = {}
    for mk in metric_keys:
        vals = [m[mk] for m in sm if mk in m]
        if vals: agg[mk] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary[key] = agg
    print(f"\n  α = {alpha}:")
    for mk in metric_keys:
        if mk in agg: print(f"    {mk:<12}  {agg[mk]['mean']:.4f} ± {agg[mk]['std']:.4f}")

with open(f"{OUTPUT_DIR}/condB_summary.json", "w") as f:
    json.dump({"condition": "B", "alphas": ALPHA_VALUES, "seeds": SEEDS, "summary": summary}, f, indent=2)
print(f"\nSummary saved → {OUTPUT_DIR}/condB_summary.json")
