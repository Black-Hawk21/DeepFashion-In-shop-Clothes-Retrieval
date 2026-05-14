# ============================================================
# Condition C — Fine-tuned CLIP + Frozen BLIP-2  (α ∈ {0.5, 0.7})
# ============================================================
# Loads a pre-trained CLIP checkpoint (.pt from train_clip.py),
# uses bbox annotations to crop images, generates fused embeddings
# with BLIP-2 captions, builds HNSW index, and evaluates.
#
# NO TRAINING — the CLIP model is already fine-tuned locally.
# Self-contained. Run each "# %%" section as a separate Kaggle cell.
#
# Cached artifacts (auto-saved to OUTPUT_DIR):
#   - gallery_img_embs_ft.npy / gallery_ids.json
#   - query_embs_ft.npy / query_ids.json
#   - gallery_captions.json  (shared with Part B)
#   - gallery_txt_embs_ft.npy
#   - gallery_fused_alpha{a}.npy
#   - condC_alpha{a}_seed{s}.json / condC_summary.json
# ============================================================

# %% Install dependencies (run once)
# !pip install -q openai-clip hnswlib tqdm Pillow transformers accelerate

# %% Imports and configuration
import os, json, random, time, gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
OUTPUT_DIR     = "results/condC"
INDEX_DIR      = "index/condC"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

CLIP_CHECKPOINT = "checkpoints/best_model.pt"  # <-- EDIT THIS

CAPTION_CACHE = "results/condB/gallery_captions.json"

# ===== MODEL CONFIG =====
CLIP_MODEL_NAME        = "ViT-B/32"
BLIP2_MODEL            = "Salesforce/blip2-opt-2.7b"
UNFREEZE_VISION_BLOCKS = 4    # Must match the value used during training

# ===== EVALUATION CONFIG =====
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
print(f"CLIP checkpoint: {CLIP_CHECKPOINT}")

# %% CLIPFineTuner class (must match the architecture used in training)
class CLIPFineTuner(nn.Module):
    """
    Wrapper around CLIP that unfreezes the last N vision transformer blocks.
    Architecture must match what was used in scripts/train_clip.py.
    """
    def __init__(self, model_name="ViT-B/32", unfreeze_blocks=4, device=None):
        super().__init__()
        self.device = device or DEVICE
        self.clip_model, self.preprocess = clip.load(model_name, device=self.device)
        self.clip_model = self.clip_model.float()

        # Freeze everything first
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Unfreeze last N vision blocks (to match training setup)
        vis = self.clip_model.visual
        rb = list(vis.transformer.resblocks)
        n = len(rb) if unfreeze_blocks == -1 else min(unfreeze_blocks, len(rb))
        for blk in rb[-n:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in vis.ln_post.parameters():
            p.requires_grad = True
        if vis.proj is not None:
            vis.proj.requires_grad = True

        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        print(f"[CLIP] Unfroze {n}/{len(rb)} blocks. Params: {tr:,}/{tot:,} ({100*tr/tot:.1f}%)")

    def encode_image(self, imgs, normalize=True):
        f = self.clip_model.encode_image(imgs)
        return F.normalize(f, dim=-1) if normalize else f

    def encode_text(self, tokens, normalize=True):
        with torch.no_grad():
            f = self.clip_model.encode_text(tokens)
        return F.normalize(f, dim=-1) if normalize else f

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
    try:
        img = Image.open(os.path.join(DATA_ROOT, rel_path)).convert("RGB")
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
    clip_m.eval()
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
    ae = []; clip_m.eval()
    for i in tqdm(range(0, len(captions), bs), desc="TxtEnc"):
        bc = [c if c else "a clothing item" for c in captions[i:i+bs]]
        t = clip.tokenize(bc, truncate=True).to(DEVICE)
        ae.append(F.normalize(clip_m.encode_text(t), dim=-1).cpu().float().numpy())
    return np.concatenate(ae, axis=0)

def build_hnsw(embs, ef_c=200, M=16):
    N, D = embs.shape
    idx = hnswlib.Index(space="cosine", dim=D)
    idx.init_index(max_elements=N, ef_construction=ef_c, M=M)
    idx.add_items(embs, np.arange(N))
    print(f"[HNSW] {N} items, dim={D}"); return idx

def search_hnsw(idx, q, top_k=15, ef=100):
    idx.set_ef(max(ef, top_k))
    i, _ = idx.knn_query(q, k=top_k); return i

# %% Load data and fine-tuned CLIP model
print("=" * 60)
print("  CONDITION C — Fine-tuned CLIP + Frozen BLIP-2")
print("  (No training — loading pre-trained checkpoint)")
print("=" * 60)

partition = load_partition(PARTITION_FILE)
bbox_map = load_bbox_annotations(BBOX_FILE)
gallery_samples = partition["gallery"]
query_samples   = partition["query"]

# Build the CLIPFineTuner and load checkpoint
print(f"\n[Model] Building CLIPFineTuner (unfreeze_blocks={UNFREEZE_VISION_BLOCKS})...")
model = CLIPFineTuner(CLIP_MODEL_NAME, UNFREEZE_VISION_BLOCKS, DEVICE).to(DEVICE)

print(f"[Model] Loading checkpoint: {CLIP_CHECKPOINT}")
checkpoint = torch.load(CLIP_CHECKPOINT, map_location=DEVICE)

# Handle both checkpoint formats:
#   train_clip.py saves: {"epoch": ..., "model_state_dict": {...}, ...}
#   direct state_dict:   {"visual.conv1.weight": ..., ...}
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint.get("epoch", "?")
    metrics = checkpoint.get("metrics", {})
    print(f"  Loaded checkpoint from epoch {epoch}")
    if metrics:
        print(f"  Training metrics: {json.dumps({k: f'{v:.4f}' for k, v in metrics.items() if isinstance(v, float)}, indent=2)}")
else:
    # Assume it's a raw state_dict
    model.load_state_dict(checkpoint)
    print(f"  Loaded raw state_dict")

model.eval()
preprocess = model.preprocess
print("[Model] Fine-tuned CLIP ready ✓")

# Cache paths
GALLERY_IMG_EMB_PATH = f"{OUTPUT_DIR}/gallery_img_embs_ft.npy"
GALLERY_IDS_PATH     = f"{OUTPUT_DIR}/gallery_ids.json"
QUERY_EMB_PATH       = f"{OUTPUT_DIR}/query_embs_ft.npy"
QUERY_IDS_PATH       = f"{OUTPUT_DIR}/query_ids.json"
GALLERY_TXT_EMB_PATH = f"{OUTPUT_DIR}/gallery_txt_embs_ft.npy"
LOCAL_CAPTION_PATH   = f"{OUTPUT_DIR}/gallery_captions.json"

# %% Step 1: Gallery image embeddings (fine-tuned CLIP vision)
if cache_exists(GALLERY_IMG_EMB_PATH, GALLERY_IDS_PATH):
    print("\n[Cache] Loading gallery image embeddings...")
    gallery_img_embs, gallery_ids = load_embeddings(GALLERY_IMG_EMB_PATH, GALLERY_IDS_PATH)
else:
    print("\n[Compute] Embedding gallery images (bbox-cropped, fine-tuned CLIP)...")
    gallery_img_embs, gallery_ids = embed_images(
        gallery_samples, model, preprocess, bbox_map, "Gallery"
    )
    save_embeddings(gallery_img_embs, gallery_ids, GALLERY_IMG_EMB_PATH, GALLERY_IDS_PATH)

# %% Step 2: Query embeddings (fine-tuned CLIP vision)
if cache_exists(QUERY_EMB_PATH, QUERY_IDS_PATH):
    print("\n[Cache] Loading query embeddings...")
    query_embs, query_ids = load_embeddings(QUERY_EMB_PATH, QUERY_IDS_PATH)
else:
    print("\n[Compute] Embedding query images (bbox-cropped, fine-tuned CLIP)...")
    query_embs, query_ids = embed_images(
        query_samples, model, preprocess, bbox_map, "Queries"
    )
    save_embeddings(query_embs, query_ids, QUERY_EMB_PATH, QUERY_IDS_PATH)

# %% Step 3: BLIP-2 captions (on bbox-cropped gallery images)
caption_source = CAPTION_CACHE if os.path.exists(CAPTION_CACHE) else LOCAL_CAPTION_PATH

if os.path.exists(caption_source):
    print(f"\n[Cache] Loading captions from {caption_source}")
    with open(caption_source) as f:
        gallery_captions = json.load(f)["captions"]
    print(f"  {len(gallery_captions)} captions loaded")
else:
    print("\n[Compute] Generating BLIP-2 captions on bbox-cropped gallery images...")
    print("  Loading BLIP-2 model (~1-2 min)...")
    blip_processor = Blip2Processor.from_pretrained(BLIP2_MODEL)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_MODEL, device_map="auto", torch_dtype=torch.float16,
    )
    blip_model.eval()

    gallery_captions = []
    prompt = "A photo of a clothing item:"

    for i, (rel_path, item_id) in enumerate(tqdm(gallery_samples, desc="Captioning")):
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
            with open(LOCAL_CAPTION_PATH, "w") as f:
                json.dump({"captions": gallery_captions, "progress": i + 1}, f)
            print(f"  Checkpoint at {i + 1}/{len(gallery_samples)}")

    with open(LOCAL_CAPTION_PATH, "w") as f:
        json.dump({"captions": gallery_captions, "progress": len(gallery_samples)}, f)
    print(f"  Saved {len(gallery_captions)} captions → {LOCAL_CAPTION_PATH}")

    del blip_model, blip_processor
    gc.collect(); torch.cuda.empty_cache()

# %% Step 4: Gallery text embeddings (frozen CLIP text encoder)
if cache_exists(GALLERY_TXT_EMB_PATH):
    print("\n[Cache] Loading gallery text embeddings...")
    gallery_txt_embs = load_np(GALLERY_TXT_EMB_PATH)
else:
    print("\n[Compute] Encoding captions with CLIP text encoder (frozen)...")
    gallery_txt_embs = encode_captions(gallery_captions, model)
    save_np(gallery_txt_embs, GALLERY_TXT_EMB_PATH)

# Free CLIP model from GPU (not needed for eval)
del model
gc.collect(); torch.cuda.empty_cache()
print("\n[Memory] CLIP model unloaded")

# %% Step 5: Evaluate across (α, seed) combinations
print("\n" + "=" * 60)
print(f"  Gallery img: {gallery_img_embs.shape}  |  Gallery txt: {gallery_txt_embs.shape}")
print(f"  Queries: {query_embs.shape}")
print("=" * 60)

all_results = {}

for alpha in ALPHA_VALUES:
    print(f"\n{'='*50}\n  α = {alpha}\n{'='*50}")

    # Fuse: v = α·img + (1-α)·txt, then L2 normalize
    fused = alpha * gallery_img_embs + (1 - alpha) * gallery_txt_embs
    fused = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)
    save_np(fused, f"{OUTPUT_DIR}/gallery_fused_alpha{alpha}.npy")

    for seed in SEEDS:
        print(f"\n--- seed={seed}, α={alpha} ---")
        set_seed(seed)
        t0 = time.time()

        index = build_hnsw(fused, HNSW_EF_CONSTRUCTION, HNSW_M)
        ranked = search_hnsw(index, query_embs, top_k=max(K_VALUES), ef=HNSW_EF_SEARCH)
        metrics = evaluate_retrieval(query_ids, gallery_ids, ranked, K_VALUES)

        print(f"Results (took {time.time()-t0:.1f}s):")
        print_metrics(metrics, K_VALUES)
        all_results[(alpha, seed)] = metrics

        out = {"condition": "C", "alpha": alpha, "seed": seed, "metrics": metrics}
        with open(f"{OUTPUT_DIR}/condC_alpha{alpha}_seed{seed}.json", "w") as f:
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
print("  CONDITION C — AGGREGATED RESULTS")
print("=" * 60)

metric_keys = [f"{m}@{k}" for m in ("recall", "ndcg", "map") for k in K_VALUES]
summary = {}
for alpha in ALPHA_VALUES:
    key = f"alpha={alpha}"
    sm = [all_results.get((alpha, s), {}) for s in SEEDS]
    agg = {}
    for mk in metric_keys:
        vals = [m[mk] for m in sm if mk in m]
        if vals:
            agg[mk] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary[key] = agg
    print(f"\n  α = {alpha}:")
    for mk in metric_keys:
        if mk in agg:
            print(f"    {mk:<12}  {agg[mk]['mean']:.4f} ± {agg[mk]['std']:.4f}")

with open(f"{OUTPUT_DIR}/condC_summary.json", "w") as f:
    json.dump({
        "condition": "C",
        "checkpoint": CLIP_CHECKPOINT,
        "alphas": ALPHA_VALUES,
        "seeds": SEEDS,
        "summary": summary,
    }, f, indent=2)
print(f"\nSummary saved → {OUTPUT_DIR}/condC_summary.json")
