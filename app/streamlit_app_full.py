"""
app/streamlit_app_full.py
--------------------------
Full-featured Streamlit demo with clothing-specific YOLOv8n cropping + BLIP-2 captioning + fused retrieval.

Pipeline:
    Upload -> YOLOv8n clothing detector (Clothing boxes only)
                 -> User selects the crop
                 -> Fine-tuned CLIP (image embed on crop)
                 -> BLIP-2 (caption generation on crop)
                 -> CLIP text encoder (caption embed)
                 -> Fused query (alpha=0.5)
                 -> HNSW search on Cond_C (alpha0.5/seed51) index
                 -> Display top-K results

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
    page_icon="clothes",
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
BLIP2_MODEL_ID  = "Salesforce/blip2-opt-2.7b"
YOLO_MODEL      = "checkpoints/clothing yolo.pt"
YOLO_CONF       = 0.25
YOLO_IOU        = 0.45
CROP_PADDING    = 0.05
FUSION_ALPHA    = 0.5
CLOTHING_LABELS = {"clothing"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ #
#  Build CLIP from checkpoint (no download needed)                     #
# ------------------------------------------------------------------ #

def _build_clip_from_checkpoint(checkpoint_path, device):
    """
    Build ViT-B/32 architecture from the checkpoint's state_dict
    using clip's internal build_model(). Zero internet download.
    """
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


@st.cache_resource(show_spinner="Loading BLIP-2 captioning model (~1-2 min)...")
def load_blip2():
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
    dim = meta.get("dim", 512)
    n_items = meta.get("n_items", len(gallery_ids))

    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(idx_path, max_elements=n_items)
    index.set_ef(100)

    return index, gallery_ids, idx_path


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
    Run YOLO detection and return only clothing detections.
    Each detection is {"bbox": (x1, y1, x2, y2), "conf": float, "label": str}.
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
    """Draw bounding boxes with labels on an image copy."""
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
def encode_image(image, clip_model, preprocess):
    """Encode an image with fine-tuned CLIP vision encoder."""
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    emb = clip_model.encode_image(img_tensor)
    return F.normalize(emb.float(), dim=-1)


@torch.no_grad()
def generate_caption(image, blip_processor, blip_model):
    """Generate a caption for the image using BLIP-2."""
    prompt = "A photo of a clothing item:"
    inputs = blip_processor(
        images=image, text=prompt, return_tensors="pt"
    ).to(blip_model.device, dtype=torch.float16)
    output_ids = blip_model.generate(**inputs, max_new_tokens=50)
    caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)
    caption = caption.replace(prompt, "").strip()
    return caption


@torch.no_grad()
def encode_caption(caption, clip_model):
    """Encode a caption with CLIP text encoder."""
    import clip
    tokens = clip.tokenize([caption], truncate=True).to(DEVICE)
    emb = clip_model.encode_text(tokens)
    return F.normalize(emb.float(), dim=-1)


def fuse_embeddings(img_emb, txt_emb, alpha):
    """Fuse image and text embeddings: v = alpha*img + (1-alpha)*txt, normalized."""
    fused = alpha * img_emb + (1 - alpha) * txt_emb
    fused = F.normalize(fused, dim=-1)
    return fused.cpu().numpy()


# ------------------------------------------------------------------ #
#  UI                                                                  #
# ------------------------------------------------------------------ #

st.title("Visual Product Search")
st.markdown(
    "Upload a clothing image. **YOLOv8n (clothing detector)** finds clothing boxes, "
    "**BLIP-2** generates a caption, **fine-tuned CLIP** creates a fused embedding, "
    "and **HNSW** retrieves the most similar products."
)

# Sidebar
with st.sidebar:
    st.header("Search Settings")
    top_k = st.slider("Top-K results", min_value=1, max_value=20, value=10)
    show_scores = st.checkbox("Show similarity scores", value=True)
    show_caption = st.checkbox("Show generated caption", value=True)
    st.markdown("---")

    st.header("Fusion Settings")
    st.markdown(f"**Alpha (fixed):** `{FUSION_ALPHA}`")
    st.markdown(f"**Index:** `{Path(INDEX_BIN).name}`")
    st.markdown("---")

    st.markdown("**Full Pipeline:**")
    st.markdown("1. Upload image")
    st.markdown("2. YOLOv8n detects Clothing boxes")
    st.markdown("3. Select crop")
    st.markdown("4. BLIP-2 generates caption on crop")
    st.markdown("5. Fine-tuned CLIP encodes crop + caption")
    st.markdown(f"6. Fuse: `{FUSION_ALPHA}*img + {1-FUSION_ALPHA:.1f}*txt`")
    st.markdown("7. HNSW nearest-neighbor search")
    st.markdown("---")
    st.markdown(f"**Device:** `{DEVICE}`")
    st.markdown("---")
    st.markdown("**Sample Gallery Image (first found)**")
    sample_path = find_first_image_path(IMG_ROOT)
    if sample_path:
        st.image(Image.open(sample_path).convert("RGB"), use_container_width=True)
        sample_rel = Path(sample_path).relative_to(Path(IMG_ROOT)).as_posix()
        st.caption(sample_rel)
    else:
        st.warning(f"No images found under {IMG_ROOT}")

# File uploader
uploaded = st.file_uploader(
    "Upload a query image", type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

# ---- Load all models ----
clip_model, clip_preprocess, n_clip_params = load_clip()
blip_processor, blip_model = load_blip2()
yolo_model = load_yolo()
index, gallery_ids, idx_path = load_index()
gallery_items = load_gallery_items()

with st.sidebar:
    st.success(f"CLIP loaded ({n_clip_params/1e6:.0f}M params)")
    st.success("BLIP-2 loaded")
    st.success("YOLO loaded")
    st.success(f"Index: {Path(idx_path).name} ({len(gallery_ids)} items)")

# ---- Display uploaded image + YOLO detection ----
query_image = Image.open(uploaded).convert("RGB")

# Run YOLO detection
with st.spinner("Running clothing detection..."):
    detections = yolo_detect_clothing(
        query_image,
        yolo_model,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        padding=CROP_PADDING,
    )

col_orig, col_det = st.columns(2)

with col_orig:
    st.subheader("Original Image")
    st.image(query_image, use_container_width=True)

with col_det:
    st.subheader("YOLO Detection")
    if detections:
        annotated = draw_bboxes_on_image(query_image, detections)
        st.image(annotated, caption=f"{len(detections)} clothing boxes found",
                 use_container_width=True)
    else:
        st.image(query_image, caption="No clothing detected",
                 use_container_width=True)

# ---- Crop selection ----
st.subheader("Select Crop for Search")

if detections:
    crop_cards = []
    for det in detections:
        crop = query_image.crop(det["bbox"])
        crop_cards.append({
            "crop": crop,
            "bbox": det["bbox"],
            "conf": det["conf"],
        })

    n_cols = min(3, len(crop_cards))
    grid_cols = st.columns(n_cols)
    for i, card in enumerate(crop_cards):
        with grid_cols[i % n_cols]:
            w, h = card["crop"].size
            st.image(
                card["crop"],
                caption=f"Crop {i + 1} (conf={card['conf']:.2f}, {w}x{h})",
                use_container_width=True,
            )

    selected_idx = st.radio(
        "Choose a crop",
        options=list(range(len(crop_cards))),
        format_func=lambda i: f"Crop {i + 1} (conf={crop_cards[i]['conf']:.2f})",
        horizontal=True,
        key="selected_crop_idx",
    )
    cropped = crop_cards[selected_idx]["crop"]
else:
    st.warning("No clothing detections found. Using full image.")
    cropped = query_image

# Show the final crop
st.subheader("Final Crop for Search")
st.image(cropped, width=300, caption=f"Size: {cropped.size[0]}x{cropped.size[1]}")

# ---- Search button ----
search = st.button("Search with this crop", type="primary",
                   use_container_width=True)
if not search:
    st.stop()

# ---- Run the full pipeline ----
st.markdown("---")
timings = {}

# Step 1: YOLO (already done)
timings["yolo"] = 0.0  # Already ran above

# Step 2: CLIP image embedding on crop
with st.spinner("Step 1/4: Encoding cropped image with fine-tuned CLIP..."):
    t0 = time.time()
    img_emb = encode_image(cropped, clip_model, clip_preprocess)
    timings["clip_image"] = time.time() - t0

# Step 3: BLIP-2 caption on crop
with st.spinner("Step 2/4: Generating caption with BLIP-2..."):
    t0 = time.time()
    caption = generate_caption(cropped, blip_processor, blip_model)
    timings["blip2_caption"] = time.time() - t0

# Step 4: CLIP text embedding of caption
with st.spinner("Step 3/4: Encoding caption with CLIP text encoder..."):
    t0 = time.time()
    txt_emb = encode_caption(caption, clip_model)
    timings["clip_text"] = time.time() - t0

# Step 5: Fuse + search
with st.spinner("Step 4/4: Fusing embeddings and searching index..."):
    t0 = time.time()
    query_emb = fuse_embeddings(img_emb, txt_emb, FUSION_ALPHA)

    indices, distances = index.knn_query(query_emb, k=top_k)
    indices = indices[0]
    distances = distances[0]
    similarities = 1.0 - distances
    timings["search"] = time.time() - t0

# ---- Show pipeline results ----

# Caption display
if show_caption:
    st.subheader("Generated Caption (BLIP-2)")
    st.info(f'"{caption}"')

# Timing display
total_time = sum(timings.values())
timing_cols = st.columns(5)
timing_labels = ["CLIP Image", "BLIP-2 Caption", "CLIP Text", "HNSW Search", "Total"]
timing_values = [timings["clip_image"], timings["blip2_caption"],
                 timings["clip_text"], timings["search"], total_time]
for col, label, val in zip(timing_cols, timing_labels, timing_values):
    col.metric(label, f"{val:.2f}s")

# ---- Display results ----
st.subheader(f"Top-{top_k} Search Results")
st.caption(f"Fused query: {FUSION_ALPHA}*image + {1-FUSION_ALPHA:.1f}*caption  |  "
           f"Total time: {total_time:.2f}s")

img_dir = Path(IMG_ROOT)
n_cols = 5
rows_data = [list(range(i, min(i + n_cols, len(indices))))
             for i in range(0, len(indices), n_cols)]

for row_indices in rows_data:
    cols = st.columns(n_cols)
    for col, ri in zip(cols, row_indices):
        idx = int(indices[ri])
        item_id = gallery_ids[idx] if idx < len(gallery_ids) else "unknown"
        sim = float(similarities[ri])

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
            st.markdown(f"Item: `{item_id}`")
            if show_scores:
                st.markdown(f"Similarity: `{sim:.3f}`")
