"""
app/streamlit_app_full.py
--------------------------
Full-featured Streamlit demo following the PDF pipeline exactly.

Offline index: fused CLIP (image + BLIP-2 caption) embeddings per gallery item.
Online query pipeline (per PDF):
    Step 1. YOLO crops the main product region from the query image.
    Step 2. Cropped query -> CLIP visual embedding ONLY (no caption generated for query).
    Step 3. HNSW ANN search returns a configurable candidate pool (cosine similarity).
    Step 4. BLIP-2 ITM re-ranking: score(query_image, candidate_caption) for each candidate,
            then re-rank to produce final top-K result list.

Requires CUDA GPU with sufficient VRAM (~6GB for BLIP-2 + CLIP + YOLO).

Run:
    streamlit run app/streamlit_app_full.py
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import hnswlib

# ------------------------------------------------------------------ #
#  Page config                                                         #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Visual Product Search (Full Pipeline)",
    page_icon="👗",
    layout="wide",
)

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

DATA_ROOT       = "data"
IMG_ROOT        = "data/Img/img"
CLIP_CHECKPOINT = "checkpoints/best_model.pt"
INDEX_BIN       = "index/Cond_C/hnsw_alpha0.5_seed51.bin"
INDEX_META      = "index/Cond_C/metadata_alpha0.5_seed51.json"

# BLIP-2: same model used offline for captioning, reused here for ITM re-ranking.
# No new download — uses your existing HuggingFace cache.
BLIP2_MODEL_ID  = "Salesforce/blip2-opt-2.7b"

YOLO_MODEL      = "checkpoints/clothing yolo.pt"
YOLO_CONF       = 0.25
YOLO_IOU        = 0.45
CROP_PADDING    = 0.05
CLOTHING_LABELS = {"clothing"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ #
#  Build CLIP from checkpoint                                          #
# ------------------------------------------------------------------ #

def _build_clip_from_checkpoint(checkpoint_path, device):
    from clip.model import build_model
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    clean_sd = {}
    for k, v in state_dict.items():
        clean_sd[k.removeprefix("clip_model.")] = v

    model = build_model(clean_sd).to(device).float().eval()

    preprocess = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])

    return model, preprocess


# ------------------------------------------------------------------ #
#  Cached model loaders                                                #
# ------------------------------------------------------------------ #

@st.cache_resource(show_spinner="Loading fine-tuned CLIP...")
def load_clip():
    ckpt_path = Path(CLIP_CHECKPOINT)
    if not ckpt_path.exists():
        st.error(f"Checkpoint not found: {ckpt_path}")
        st.stop()
    model, preprocess = _build_clip_from_checkpoint(str(ckpt_path), DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    return model, preprocess, n_params


@st.cache_resource(show_spinner="Loading BLIP-2 for ITM re-ranking (uses existing cache)...")
def load_blip2():
    """
    Load the same BLIP-2 model used during offline indexing.
    Already on disk in your HuggingFace cache — no re-download.
    Used here for ITM re-ranking via language-model probability scoring:
    prompt the model with the candidate caption and measure P("yes").
    """
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained(BLIP2_MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="Loading YOLO detector...")
def load_yolo():
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL)
    return model


@st.cache_resource(show_spinner="Loading HNSW index...")
def load_index():
    idx_path = INDEX_BIN
    meta_path = INDEX_META

    if not Path(idx_path).exists():
        st.error(f"Index not found: {idx_path}")
        st.stop()
    if not Path(meta_path).exists():
        st.error(f"Metadata not found: {meta_path}")
        st.stop()

    with open(meta_path) as f:
        meta = json.load(f)

    gallery_ids = meta.get("gallery_ids", meta.get("item_ids", []))
    # Captions pre-computed during offline indexing and stored in metadata
    gallery_captions = meta.get("gallery_captions", {})
    dim = meta.get("dim", 512)
    n_items = meta.get("n_items", len(gallery_ids))

    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(idx_path, max_elements=n_items)
    index.set_ef(100)

    return index, gallery_ids, gallery_captions, idx_path


@st.cache_data(show_spinner=False)
def load_gallery_items():
    """Build mapping from sequential gallery index to (image_path, item_id)."""
    partition_file = f"{DATA_ROOT}/Eval/list_eval_partition.txt"
    gallery_items = []
    try:
        with open(partition_file) as f:
            lines = f.read().splitlines()
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] == "gallery":
                gallery_items.append((parts[0], parts[1]))
    except FileNotFoundError:
        pass
    return gallery_items


@st.cache_data(show_spinner=False)
def find_first_image_path(img_root):
    root = Path(img_root)
    if not root.exists():
        return None
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for path in root.rglob(ext):
            return str(path)
    return None


# ------------------------------------------------------------------ #
#  YOLO detection                                                      #
# ------------------------------------------------------------------ #

def _get_class_name_map(result, model):
    def _normalize(names):
        if isinstance(names, dict):
            return names
        if isinstance(names, list):
            return {i: name for i, name in enumerate(names)}
        return {}

    if getattr(result, "names", None):
        return _normalize(result.names)
    if getattr(model, "names", None):
        return _normalize(model.names)
    if getattr(model, "model", None) is not None and getattr(model.model, "names", None):
        return _normalize(model.model.names)
    return {}


def _pad_bbox(bbox, image_size, padding=0.05):
    W, H = image_size
    x1, y1, x2, y2 = bbox
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    x1_p = max(0, x1 - pad_w)
    y1_p = max(0, y1 - pad_h)
    x2_p = min(W, x2 + pad_w)
    y2_p = min(H, y2 + pad_h)
    return (x1_p, y1_p, x2_p, y2_p)


def yolo_detect_clothing(image, yolo_model, conf=0.25, iou=0.45, padding=0.05):
    """
    Step 1 (PDF): Run YOLO detection, return only clothing detections.
    Each detection: {"bbox": (x1,y1,x2,y2), "conf": float, "label": str}
    """
    results = yolo_model.predict(
        source=np.array(image),
        conf=conf,
        iou=iou,
        device=str(DEVICE),
        verbose=False,
    )

    if not results:
        return []

    result = results[0]
    class_names = _get_class_name_map(result, yolo_model)
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    detections = []
    for box in boxes:
        cls_id = int(box.cls.item()) if getattr(box, "cls", None) is not None else None
        if cls_id is not None:
            cls_name = class_names.get(cls_id, class_names.get(str(cls_id), str(cls_id)))
        else:
            cls_name = ""
        if cls_name.lower().strip() not in CLOTHING_LABELS:
            continue
        conf_val = float(box.conf.item()) if getattr(box, "conf", None) is not None else 0.0
        xyxy = box.xyxy[0].cpu().numpy()
        raw_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
        padded_bbox = _pad_bbox(raw_bbox, image.size, padding=padding)
        detections.append({
            "bbox": padded_bbox,
            "conf": conf_val,
            "label": cls_name,
        })

    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections


def draw_bboxes_on_image(image, detections):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['label']} {det['conf']:.2f}" if det.get("label") else None
        for i in range(3):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline="red")
        if label:
            draw.text((x1 + 4, y1 + 4), label, fill="red")
    return annotated


# ------------------------------------------------------------------ #
#  Pipeline functions                                                  #
# ------------------------------------------------------------------ #

@torch.no_grad()
def encode_query_image(image, clip_model, preprocess):
    """
    Step 2 (PDF): Map cropped query to CLIP visual embedding ONLY.
    No caption is generated for the query — that is only done offline for gallery items.
    """
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    emb = clip_model.encode_image(img_tensor)
    return F.normalize(emb.float(), dim=-1).cpu().numpy()


@torch.no_grad()
def compute_itm_score(query_image, caption, blip2_processor, blip2_model):
    """
    Step 4 (PDF): BLIP-2 ITM score between query image and a candidate caption.

    BLIP-2 has no dedicated ITM head, so we use its language model:
      prompt = "Question: Does this image show \'<caption>\'? Answer:"
      ITM score = P(first token == \'yes\') / (P(\'yes\') + P(\'no\'))

    Uses the already-cached blip2-opt-2.7b — zero new downloads.
    """
    if not caption:
        return 0.0

    prompt = f"Question: Does this image show \"{caption}\"? Answer:"
    inputs = blip2_processor(
        images=query_image,
        text=prompt,
        return_tensors="pt",
    ).to(blip2_model.device, torch.float16)

    # Get logits for the very first generated token
    outputs = blip2_model(**inputs)
    next_token_logits = outputs.logits[:, -1, :]  # (1, vocab_size)

    # Compare "yes" vs "no" token probabilities
    yes_id = blip2_processor.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = blip2_processor.tokenizer.encode("no",  add_special_tokens=False)[0]

    yes_logit = next_token_logits[0, yes_id].float()
    no_logit  = next_token_logits[0, no_id].float()
    score = torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)[0].item()
    return score


def rerank_candidates(
    query_image,
    candidate_indices,
    candidate_similarities,
    gallery_ids,
    gallery_captions,
    blip2_processor,
    blip2_model,
    top_k,
    progress_bar,
):
    """
    Step 4 (PDF): Re-rank candidates by BLIP-2 ITM score(query_image, candidate_caption).
    Returns top_k results sorted by ITM score descending.
    """
    scored = []
    for step, (idx, sim) in enumerate(zip(candidate_indices, candidate_similarities)):
        item_id = gallery_ids[idx] if idx < len(gallery_ids) else "unknown"
        # Look up the pre-computed caption for this gallery item
        caption = gallery_captions.get(str(idx), gallery_captions.get(item_id, ""))
        itm = compute_itm_score(query_image, caption, blip2_processor, blip2_model)
        scored.append({
            "gallery_idx": idx,
            "item_id": item_id,
            "cosine_sim": float(sim),
            "itm_score": itm,
            "caption": caption,
        })
        progress_bar.progress((step + 1) / len(candidate_indices),
                              text=f"Re-ranking {step + 1}/{len(candidate_indices)}...")

    # Sort by ITM score descending, return top_k
    scored.sort(key=lambda x: x["itm_score"], reverse=True)
    return scored[:top_k]


# ------------------------------------------------------------------ #
#  UI                                                                  #
# ------------------------------------------------------------------ #

st.title("👗 Visual Product Search")
st.markdown(
    """
    **Pipeline (per project spec):**
    1. **YOLO** — detect and crop the clothing item from the query image
    2. **CLIP (visual only)** — encode the cropped query image *(no caption generated for query)*
    3. **HNSW ANN** — retrieve top-K×{rerank} candidates by cosine similarity
    4. **BLIP ITM re-ranking** — score each candidate's pre-computed caption against the query image, re-rank to final top-K
    """
)

# Sidebar
with st.sidebar:
    st.header("Search Settings")
    top_k = st.slider("Top-K results", min_value=1, max_value=20, value=10)
    rerank_factor = st.slider(
        "Re-rank pool multiplier (fetch top-K × N before re-ranking)",
        min_value=1, max_value=10, value=5,
        help="Higher = better re-ranking quality but slower. 1 = no expansion (re-rank only what you show)."
    )
    show_scores = st.checkbox("Show similarity & ITM scores", value=True)
    show_caption = st.checkbox("Show candidate captions", value=True)
    st.markdown("---")

    st.header("Index Info")
    st.markdown(f"**Index:** `{Path(INDEX_BIN).name}`")
    st.markdown(f"**Re-rank pool:** top-K × {rerank_factor} = top-{top_k * rerank_factor}")
    st.markdown("---")

    st.markdown("**Pipeline steps:**")
    st.markdown("1. Upload image")
    st.markdown("2. YOLO detects clothing boxes")
    st.markdown("3. User selects / confirms crop")
    st.markdown("4. Fine-tuned CLIP encodes crop (image only)")
    st.markdown("5. HNSW retrieves candidates")
    st.markdown("6. BLIP ITM re-ranks by caption match")
    st.markdown("---")
    st.markdown(f"**Device:** `{DEVICE}`")
    st.markdown("---")

    st.markdown("**Sample Gallery Image**")
    sample_path = find_first_image_path(IMG_ROOT)
    if sample_path:
        st.image(Image.open(sample_path).convert("RGB"), use_container_width=True)
        st.caption(Path(sample_path).relative_to(Path(IMG_ROOT)).as_posix())
    else:
        st.warning(f"No images found under {IMG_ROOT}")

# ---- File uploader ----
uploaded = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

# ---- Load all models ----
clip_model, clip_preprocess, n_clip_params = load_clip()
blip2_processor, blip2_model = load_blip2()
yolo_model = load_yolo()
index, gallery_ids, gallery_captions, idx_path = load_index()
gallery_items = load_gallery_items()

with st.sidebar:
    st.success(f"CLIP loaded ({n_clip_params/1e6:.0f}M params)")
    st.success("BLIP-2 loaded (ITM re-ranking)")
    st.success("YOLO loaded")
    st.success(f"Index: {Path(idx_path).name} ({len(gallery_ids)} items)")
    has_captions = len(gallery_captions) > 0
    if has_captions:
        st.success(f"Gallery captions: {len(gallery_captions)} loaded")
    else:
        st.warning("No gallery captions in metadata — ITM scores will be 0.")

# ---- Display uploaded image + YOLO detection ----
query_image = Image.open(uploaded).convert("RGB")

# Step 1: YOLO crop
with st.spinner("Step 1 — Running YOLO clothing detection..."):
    t_yolo_start = time.time()
    detections = yolo_detect_clothing(
        query_image,
        yolo_model,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        padding=CROP_PADDING,
    )
    t_yolo = time.time() - t_yolo_start

col_orig, col_det = st.columns(2)
with col_orig:
    st.subheader("Original Image")
    st.image(query_image, use_container_width=True)

with col_det:
    st.subheader("YOLO Detection")
    if detections:
        annotated = draw_bboxes_on_image(query_image, detections)
        st.image(annotated,
                 caption=f"{len(detections)} clothing box(es) found — {t_yolo:.2f}s",
                 use_container_width=True)
    else:
        st.image(query_image, caption="No clothing detected", use_container_width=True)

# ---- Crop selection (user confirmation) ----
st.subheader("Select / Confirm Crop for Search")

if detections:
    crop_cards = [
        {"crop": query_image.crop(d["bbox"]), "bbox": d["bbox"], "conf": d["conf"]}
        for d in detections
    ]

    n_cols = min(3, len(crop_cards))
    grid_cols = st.columns(n_cols)
    for i, card in enumerate(crop_cards):
        with grid_cols[i % n_cols]:
            w, h = card["crop"].size
            st.image(
                card["crop"],
                caption=f"Crop {i + 1} (conf={card['conf']:.2f}, {w}×{h})",
                use_container_width=True,
            )

    selected_idx = st.radio(
        "Confirm crop to search with:",
        options=list(range(len(crop_cards))),
        format_func=lambda i: f"Crop {i + 1} (conf={crop_cards[i]['conf']:.2f})",
        horizontal=True,
        key="selected_crop_idx",
    )
    cropped = crop_cards[selected_idx]["crop"]
else:
    st.warning("No clothing detections found — using full image as crop.")
    cropped = query_image

st.subheader("Final Crop")
st.image(cropped, width=300, caption=f"{cropped.size[0]}×{cropped.size[1]}")

# ---- Search button ----
search = st.button("🔍 Search with this crop", type="primary", use_container_width=True)
if not search:
    st.stop()

# ======================================================== #
#  Online Query Pipeline (PDF spec)                        #
# ======================================================== #

st.markdown("---")
st.subheader("Running Query Pipeline")
timings = {}

# ---------- Step 2: CLIP visual embedding (query image only) ----------
with st.spinner("Step 2 — Encoding cropped query with fine-tuned CLIP (visual only)..."):
    t0 = time.time()
    query_emb = encode_query_image(cropped, clip_model, clip_preprocess)
    timings["clip_visual"] = time.time() - t0
st.success(f"Step 2 done — CLIP visual embedding ({timings['clip_visual']:.2f}s)")

# ---------- Step 3: ANN candidate retrieval ----------
rerank_pool = top_k * rerank_factor
with st.spinner(f"Step 3 — HNSW retrieval of top-{rerank_pool} candidates..."):
    t0 = time.time()
    indices, distances = index.knn_query(query_emb, k=rerank_pool)
    indices = indices[0]
    distances = distances[0]
    cosine_sims = 1.0 - distances
    timings["hnsw"] = time.time() - t0
st.success(f"Step 3 done — Retrieved {len(indices)} candidates ({timings['hnsw']:.2f}s)")

# ---------- Step 4: BLIP ITM re-ranking ----------
st.info(
    f"Step 4 — Re-ranking {len(indices)} candidates with BLIP ITM "
    f"(query image ↔ candidate caption)..."
)
rerank_progress = st.progress(0, text="Starting re-ranking...")
t0 = time.time()
reranked = rerank_candidates(
    query_image=cropped,
    candidate_indices=indices,
    candidate_similarities=cosine_sims,
    gallery_ids=gallery_ids,
    gallery_captions=gallery_captions,
    blip2_processor=blip2_processor,
    blip2_model=blip2_model,
    top_k=top_k,
    progress_bar=rerank_progress,
)
timings["itm_rerank"] = time.time() - t0
rerank_progress.progress(1.0, text=f"Re-ranking done ({timings['itm_rerank']:.2f}s)")
st.success(f"Step 4 done — Re-ranked to top-{top_k} ({timings['itm_rerank']:.2f}s)")

# ---- Timing summary ----
total_time = sum(timings.values())
st.markdown("#### Pipeline Timings")
tcols = st.columns(4)
labels_vals = [
    ("Step 1 — YOLO", t_yolo),
    ("Step 2 — CLIP Visual", timings["clip_visual"]),
    ("Step 3 — HNSW", timings["hnsw"]),
    ("Step 4 — ITM Re-rank", timings["itm_rerank"]),
]
for col, (label, val) in zip(tcols, labels_vals):
    col.metric(label, f"{val:.2f}s")
st.metric("Total Pipeline Time", f"{t_yolo + total_time:.2f}s")

# ---- Display re-ranked results ----
st.markdown("---")
st.subheader(f"Top-{top_k} Results (after ITM Re-ranking)")
st.caption(
    f"Pool: top-{rerank_pool} by cosine similarity → re-ranked by BLIP ITM score"
)

img_dir = Path(IMG_ROOT)
n_display_cols = 5
rows_data = [
    list(range(i, min(i + n_display_cols, len(reranked))))
    for i in range(0, len(reranked), n_display_cols)
]

for row_indices in rows_data:
    cols = st.columns(n_display_cols)
    for col, ri in zip(cols, row_indices):
        result = reranked[ri]
        idx = result["gallery_idx"]

        with col:
            img_shown = False
            if idx < len(gallery_items):
                rel_path, _ = gallery_items[idx]
                full_path = img_dir / rel_path
                if full_path.exists():
                    try:
                        result_img = Image.open(full_path).convert("RGB")
                        st.image(result_img, use_container_width=True)
                        img_shown = True
                    except Exception:
                        pass

            if not img_shown:
                st.warning("Image unavailable")

            st.markdown(f"**Rank {ri + 1}**")
            st.markdown(f"Item: `{result['item_id']}`")
            if show_scores:
                st.markdown(f"ITM: `{result['itm_score']:.3f}`")
                st.markdown(f"Cosine: `{result['cosine_sim']:.3f}`")
            if show_caption and result["caption"]:
                st.caption(result["caption"])