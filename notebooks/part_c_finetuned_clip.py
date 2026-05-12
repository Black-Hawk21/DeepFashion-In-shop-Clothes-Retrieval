# ============================================================
# Condition C — Fine-tuned CLIP + Frozen BLIP-2  (α ∈ {0.5, 0.7})
# ============================================================
# Fine-tune CLIP vision encoder (last 4 blocks) with InfoNCE,
# then evaluate with fused CLIP+BLIP-2 embeddings.
# Reuses BLIP-2 caption cache from Part B if available.
# Self-contained. Run each "# %%" section as a separate Kaggle cell.
#
# Cached artifacts per seed (auto-saved to OUTPUT_DIR):
#   - clip_seed{s}_final.pt              Trained CLIP weights
#   - gallery_img_embs_seed{s}.npy       Gallery vision embeddings (fine-tuned)
#   - gallery_ids.json                   Gallery item IDs
#   - query_embs_seed{s}.npy             Query vision embeddings (fine-tuned)
#   - query_ids.json                     Query item IDs
#   - gallery_captions.json              BLIP-2 captions (shared with Part B)
#   - gallery_txt_embs_seed{s}.npy       Gallery text embeddings
# ============================================================

# %% Install dependencies (run once)
# !pip install -q openai-clip hnswlib ultralytics tqdm Pillow transformers accelerate

# %% Imports and configuration
import os, json, random, time, gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from PIL import Image
from tqdm.auto import tqdm
import clip
import hnswlib
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ===== EDIT THESE PATHS FOR YOUR KAGGLE SETUP =====
DATA_ROOT      = "/kaggle/input/deepfashion-inshop"
PARTITION_FILE = f"{DATA_ROOT}/Eval/list_eval_partition.txt"
OUTPUT_DIR     = "/kaggle/working/results_condC"
CKPT_DIR       = "/kaggle/working/checkpoints"
# If Part B was run first, captions are here; otherwise generated fresh
CAPTION_CACHE  = "/kaggle/working/results_condB/gallery_captions.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

CLIP_MODEL_NAME = "ViT-B/32"
BLIP2_MODEL     = "Salesforce/blip2-opt-2.7b"
YOLO_MODEL_NAME = "yolov8m.pt"
USE_YOLO        = True

# Training
TRAIN_EPOCHS           = 20
TRAIN_BATCH_SIZE       = 64
LEARNING_RATE          = 1e-5
WEIGHT_DECAY           = 1e-4
WARMUP_STEPS           = 200
TEMPERATURE            = 0.07
GRAD_CLIP              = 1.0
USE_AMP                = True
UNFREEZE_VISION_BLOCKS = 4
SAVE_EVERY_N_EPOCHS    = 5
NUM_WORKERS            = 2

K_VALUES     = [5, 10, 15]
SEEDS        = [510, 51]
ALPHA_VALUES = [0.5, 0.7]
HNSW_EF_CONSTRUCTION = 200
HNSW_M               = 16
HNSW_EF_SEARCH       = 100
EMBED_BATCH_SIZE     = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# %% Utilities, metrics, YOLO, HNSW — identical to Parts A/B
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

def _recall_at_k(r, k): return float(r[:k].any())
def _ndcg_at_k(r, k):
    g = r[:k].astype(float)
    d = 1.0 / np.log2(np.arange(2, k+2, dtype=float))
    dcg = float((g*d).sum())
    n = min(int(r.sum()), k)
    idcg = float((np.ones(n)*d[:n]).sum()) if n > 0 else 0.0
    return dcg/idcg if idcg > 0 else 0.0
def _ap_at_k(r, k):
    g = r[:k].astype(float)
    if g.sum() == 0: return 0.0
    cs = np.cumsum(g)
    p = cs / np.arange(1, k+1, dtype=float)
    return float((p*g).sum() / min(int(r.sum()), k))

def evaluate_retrieval(qids, gids, ranked, Ks=(5,10,15)):
    ga = np.array(gids); res = {}
    mk = max(Ks)
    recs = {k:[] for k in Ks}; nds = {k:[] for k in Ks}; aps = {k:[] for k in Ks}
    for qi, qid in enumerate(qids):
        rel = (ga[ranked[qi,:mk]] == qid)
        for k in Ks:
            recs[k].append(_recall_at_k(rel, k))
            nds[k].append(_ndcg_at_k(rel, k))
            aps[k].append(_ap_at_k(rel, k))
    for k in Ks:
        res[f"recall@{k}"] = float(np.mean(recs[k]))
        res[f"ndcg@{k}"] = float(np.mean(nds[k]))
        res[f"map@{k}"] = float(np.mean(aps[k]))
    return res

def print_metrics(m, Ks=(5,10,15)):
    h = f"{'Metric':<15}" + "".join(f"K={k:<8}" for k in Ks)
    print(h); print("-"*len(h))
    for p in ("recall","ndcg","map"):
        r = f"{p.upper():<15}"
        for k in Ks: r += f"{m.get(f'{p}@{k}',0):<8.4f}"
        print(r)

class YOLODetector:
    def __init__(self, mn="yolov8m.pt", conf=0.25, iou=0.45):
        print(f"[YOLO] Loading {mn}")
        self.model = YOLO(mn); self.conf = conf; self.iou = iou
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    def detect_and_crop(self, image, padding=0.05):
        W, H = image.size
        results = self.model.predict(source=np.array(image), conf=self.conf,
                                     iou=self.iou, device=self.device, verbose=False)
        bc, bb = -1, None
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                c = float(box.conf.item())
                if c > bc:
                    bc = c; xy = box.xyxy[0].cpu().numpy()
                    bb = (int(xy[0]),int(xy[1]),int(xy[2]),int(xy[3]))
        if bb is None: return image
        x1,y1,x2,y2 = bb
        pw,ph = int((x2-x1)*padding), int((y2-y1)*padding)
        return image.crop((max(0,x1-pw),max(0,y1-ph),min(W,x2+pw),min(H,y2+ph)))

def load_and_crop(rp, yolo=None):
    try: img = Image.open(os.path.join(DATA_ROOT, rp)).convert("RGB")
    except: return None
    return yolo.detect_and_crop(img) if yolo else img

@torch.no_grad()
def embed_images(samples, clip_m, preproc, yolo_det=None, desc="Emb"):
    ae, ai, bt, bi = [], [], [], []
    clip_m.eval()
    for rp, iid in tqdm(samples, desc=desc):
        img = load_and_crop(rp, yolo_det)
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

# %% InfoNCE loss & CLIP fine-tuner
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__(); self.t = temperature; self.ce = nn.CrossEntropyLoss()
    def forward(self, a, p):
        B = a.size(0); logits = a @ p.T / self.t
        labels = torch.arange(B, device=a.device)
        return (self.ce(logits, labels) + self.ce(logits.T, labels)) / 2.0

class CLIPFineTuner(nn.Module):
    def __init__(self, model_name="ViT-B/32", unfreeze_blocks=4, device=None):
        super().__init__()
        self.device = device or DEVICE
        self.clip_model, self.preprocess = clip.load(model_name, device=self.device)
        self.clip_model = self.clip_model.float()
        for p in self.clip_model.parameters(): p.requires_grad = False
        vis = self.clip_model.visual
        rb = list(vis.transformer.resblocks)
        n = len(rb) if unfreeze_blocks == -1 else min(unfreeze_blocks, len(rb))
        for blk in rb[-n:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in vis.ln_post.parameters(): p.requires_grad = True
        if vis.proj is not None: vis.proj.requires_grad = True
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        print(f"[CLIP] Unfroze {n}/{len(rb)} blocks. Trainable: {tr:,}/{tot:,} ({100*tr/tot:.1f}%)")

    def encode_image(self, imgs, normalize=True):
        f = self.clip_model.encode_image(imgs)
        return F.normalize(f, dim=-1) if normalize else f
    def encode_text(self, tokens, normalize=True):
        with torch.no_grad(): f = self.clip_model.encode_text(tokens)
        return F.normalize(f, dim=-1) if normalize else f

# %% Training dataset & loop
class DeepFashionTrainDataset(Dataset):
    def __init__(self, samples, data_root, transform):
        self.samples = samples; self.data_root = data_root; self.transform = transform
        self.item_to_idx = {}
        for i, (_, iid) in enumerate(samples):
            self.item_to_idx.setdefault(iid, []).append(i)
        print(f"[Dataset] {len(samples)} imgs, {len(self.item_to_idx)} items")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, iid = self.samples[idx]
        anchor = Image.open(os.path.join(self.data_root, path)).convert("RGB")
        pi = self.item_to_idx[iid]
        px = idx
        if len(pi) > 1:
            while px == idx: px = random.choice(pi)
        pos = Image.open(os.path.join(self.data_root, self.samples[px][0])).convert("RGB")
        return self.transform(anchor), self.transform(pos), iid

def train_clip(model, train_samples, preprocess, seed):
    set_seed(seed)
    ds = DeepFashionTrainDataset(train_samples, DATA_ROOT, preprocess)
    loader = DataLoader(ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    criterion = InfoNCELoss(TEMPERATURE)
    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    w = LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=WARMUP_STEPS)
    c = CosineAnnealingLR(opt, T_max=TRAIN_EPOCHS, eta_min=1e-7)
    sched = SequentialLR(opt, [w, c], milestones=[WARMUP_STEPS])
    scaler = GradScaler(enabled=USE_AMP)
    model.train()
    for epoch in range(1, TRAIN_EPOCHS + 1):
        tl, nb = 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{TRAIN_EPOCHS}", leave=False)
        for a_img, p_img, _ in pbar:
            a_img, p_img = a_img.to(DEVICE), p_img.to(DEVICE)
            opt.zero_grad()
            with autocast(enabled=USE_AMP):
                loss = criterion(model.encode_image(a_img), model.encode_image(p_img))
            if USE_AMP:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            tl += loss.item(); nb += 1
            pbar.set_postfix(loss=f"{tl/nb:.4f}")
        sched.step()
        print(f"  Epoch {epoch:>2}  loss={tl/max(nb,1):.4f}  lr={opt.param_groups[0]['lr']:.2e}")
        if epoch % SAVE_EVERY_N_EPOCHS == 0 or epoch == TRAIN_EPOCHS:
            cp = f"{CKPT_DIR}/clip_seed{seed}_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), cp); print(f"  Checkpoint → {cp}")
    final = f"{CKPT_DIR}/clip_seed{seed}_final.pt"
    torch.save(model.state_dict(), final); print(f"  Final → {final}")
    return model

# %% Load partition & captions
print("=" * 60)
print("  CONDITION C — Fine-tuned CLIP + Frozen BLIP-2")
print("=" * 60)

partition = load_partition(PARTITION_FILE)
train_samples   = partition["train"]
gallery_samples = partition["gallery"]
query_samples   = partition["query"]

yolo = YOLODetector(YOLO_MODEL_NAME) if USE_YOLO else None

# Load or generate captions
LOCAL_CAPTION = f"{OUTPUT_DIR}/gallery_captions.json"
caption_source = CAPTION_CACHE if os.path.exists(CAPTION_CACHE) else LOCAL_CAPTION

if os.path.exists(caption_source):
    print(f"\n[Cache] Loading captions from {caption_source}")
    with open(caption_source) as f:
        gallery_captions = json.load(f)["captions"]
    print(f"  {len(gallery_captions)} captions loaded")
else:
    print("\n[Compute] Generating BLIP-2 captions...")
    bp = Blip2Processor.from_pretrained(BLIP2_MODEL)
    bm = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_MODEL, device_map="auto", torch_dtype=torch.float16)
    bm.eval()
    gallery_captions = []; prompt = "A photo of a clothing item:"
    for i, (rp, _) in enumerate(tqdm(gallery_samples, desc="Captioning")):
        img = load_and_crop(rp, yolo)
        if img is None: gallery_captions.append(""); continue
        try:
            inp = bp(images=img, text=prompt, return_tensors="pt").to(bm.device, dtype=torch.float16)
            with torch.no_grad(): out = bm.generate(**inp, max_new_tokens=50)
            cap = bp.decode(out[0], skip_special_tokens=True).replace(prompt,"").strip()
        except: cap = ""
        gallery_captions.append(cap)
        if (i+1) % 1000 == 0:
            with open(LOCAL_CAPTION, "w") as f:
                json.dump({"captions": gallery_captions, "progress": i+1}, f)
    with open(LOCAL_CAPTION, "w") as f:
        json.dump({"captions": gallery_captions, "progress": len(gallery_samples)}, f)
    del bm, bp; gc.collect(); torch.cuda.empty_cache()

# %% Main loop: for each seed → train → embed → evaluate
all_results = {}
GIDS_PATH = f"{OUTPUT_DIR}/gallery_ids.json"
QIDS_PATH = f"{OUTPUT_DIR}/query_ids.json"

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")

    # Cache paths for this seed
    ckpt_path     = f"{CKPT_DIR}/clip_seed{seed}_final.pt"
    g_img_path    = f"{OUTPUT_DIR}/gallery_img_embs_seed{seed}.npy"
    q_emb_path    = f"{OUTPUT_DIR}/query_embs_seed{seed}.npy"
    g_txt_path    = f"{OUTPUT_DIR}/gallery_txt_embs_seed{seed}.npy"

    # --- Step 1: Train or load checkpoint ---
    model = CLIPFineTuner(CLIP_MODEL_NAME, UNFREEZE_VISION_BLOCKS, DEVICE).to(DEVICE)

    if os.path.exists(ckpt_path):
        print(f"\n[Cache] Loading trained CLIP from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    else:
        print(f"\n[Train] Fine-tuning CLIP (seed={seed})...")
        t0 = time.time()
        model = train_clip(model, train_samples, model.preprocess, seed)
        print(f"  Training took {(time.time()-t0)/60:.1f} min")

    model.eval()

    # --- Step 2: Gallery image embeddings ---
    if cache_exists(g_img_path, GIDS_PATH):
        print(f"\n[Cache] Loading gallery image embeddings (seed={seed})...")
        gallery_img_embs, gallery_ids = load_embeddings(g_img_path, GIDS_PATH)
    else:
        print(f"\n[Compute] Gallery image embeddings (seed={seed})...")
        gallery_img_embs, gallery_ids = embed_images(
            gallery_samples, model, model.preprocess, yolo, "Gallery"
        )
        save_embeddings(gallery_img_embs, gallery_ids, g_img_path, GIDS_PATH)

    # --- Step 3: Query embeddings ---
    if cache_exists(q_emb_path, QIDS_PATH):
        print(f"\n[Cache] Loading query embeddings (seed={seed})...")
        query_embs, query_ids = load_embeddings(q_emb_path, QIDS_PATH)
    else:
        print(f"\n[Compute] Query embeddings (seed={seed})...")
        query_embs, query_ids = embed_images(
            query_samples, model, model.preprocess, yolo, "Queries"
        )
        save_embeddings(query_embs, query_ids, q_emb_path, QIDS_PATH)

    # --- Step 4: Gallery text embeddings ---
    if cache_exists(g_txt_path):
        print(f"\n[Cache] Loading gallery text embeddings (seed={seed})...")
        gallery_txt_embs = load_np(g_txt_path)
    else:
        print(f"\n[Compute] Gallery text embeddings (seed={seed})...")
        gallery_txt_embs = encode_captions(gallery_captions, model)
        save_np(gallery_txt_embs, g_txt_path)

    # --- Step 5: Evaluate for each α ---
    for alpha in ALPHA_VALUES:
        print(f"\n--- seed={seed}, α={alpha} ---")
        fused = alpha * gallery_img_embs + (1 - alpha) * gallery_txt_embs
        fused = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)

        fused_path = f"{OUTPUT_DIR}/gallery_fused_seed{seed}_alpha{alpha}.npy"
        save_np(fused, fused_path)

        set_seed(seed)
        index = build_hnsw(fused, HNSW_EF_CONSTRUCTION, HNSW_M)
        ranked = search_hnsw(index, query_embs, top_k=max(K_VALUES), ef=HNSW_EF_SEARCH)
        metrics = evaluate_retrieval(query_ids, gallery_ids, ranked, K_VALUES)
        print_metrics(metrics, K_VALUES)
        all_results[(alpha, seed)] = metrics

        out = {"condition": "C", "alpha": alpha, "seed": seed, "metrics": metrics}
        p = f"{OUTPUT_DIR}/condC_alpha{alpha}_seed{seed}.json"
        with open(p, "w") as f: json.dump(out, f, indent=2)
        print(f"Saved → {p}")
        del index

    del model; gc.collect(); torch.cuda.empty_cache()

# %% Aggregate results
print("\n" + "=" * 60)
print("  CONDITION C — AGGREGATED RESULTS")
print("=" * 60)

metric_keys = [f"{m}@{k}" for m in ("recall","ndcg","map") for k in K_VALUES]
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

with open(f"{OUTPUT_DIR}/condC_summary.json", "w") as f:
    json.dump({"condition": "C", "alphas": ALPHA_VALUES, "seeds": SEEDS, "summary": summary}, f, indent=2)
print(f"\nSummary saved → {OUTPUT_DIR}/condC_summary.json")
