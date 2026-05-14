"""
app/streamlit_app_full.py
--------------------------
Full-featured Streamlit demo with YOLO cropping + BLIP-2 captioning + fused retrieval.

Pipeline:
  Upload -> YOLO (detect & crop clothing region)
         -> User confirms / adjusts crop
         -> Fine-tuned CLIP (image embed on crop)
         -> BLIP-2 (caption generation on crop)
         -> CLIP text encoder (caption embed)
         -> Fused query: alpha * img_emb + (1-alpha) * txt_emb
         -> HNSW search on Cond_C index
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
from PIL import Image, ImageDraw, ImageFont
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

DATA_ROOT       = "data/deepfashion"
BBOX_FILE       = f"{DATA_ROOT}/Anno/list_bbox_inshop.txt"
CLIP_CHECKPOINT = "checkpoints/best_model.pt"
INDEX_DIR       = "index/Cond_C"
BLIP2_MODEL_ID  = "Salesforce/blip2-opt-2.7b"
YOLO_MODEL      = "yolov8m.pt"       # Base YOLOv8 or path to fine-tuned model
YOLO_CONF       = 0.25
YOLO_IOU        = 0.45

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
def load_index(alpha, seed):
    idx_path = f"{INDEX_DIR}/hnsw_alpha{alpha}_seed{seed}.bin"
    meta_path = f"{INDEX_DIR}/metadata_alpha{alpha}_seed{seed}.json"

    if not Path(idx_path).exists():
        available = sorted(Path(INDEX_DIR).glob("hnsw_*.bin"))
        if not available:
            st.error(f"No HNSW index found in {INDEX_DIR}.")
            st.stop()
        idx_path = str(available[0])
        meta_path = idx_path.replace("hnsw_", "metadata_").replace(".bin", ".json")

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


# ------------------------------------------------------------------ #
#  YOLO detection                                                      #
# ------------------------------------------------------------------ #

def yolo_detect(image, yolo_model, conf=0.25, iou=0.45, padding=0.05):
    """
    Run YOLO detection on the image.
    Returns (cropped_image, bbox, confidence) or (original_image, None, None).
    bbox is (x1, y1, x2, y2) in pixel coordinates.
    """
    W, H = image.size
    results = yolo_model.predict(
        source=np.array(image),
        conf=conf,
        iou=iou,
        device=str(DEVICE),
        verbose=False,
    )

    # Find the best (highest confidence) detection
    best_conf = -1
    best_box = None

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for box in result.boxes:
            c = float(box.conf.item())
            if c > best_conf:
                best_conf = c
                xyxy = box.xyxy[0].cpu().numpy()
                best_box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))

    if best_box is None:
        return image, None, None

    x1, y1, x2, y2 = best_box
    # Add padding
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    x1_p = max(0, x1 - pad_w)
    y1_p = max(0, y1 - pad_h)
    x2_p = min(W, x2 + pad_w)
    y2_p = min(H, y2 + pad_h)

    crop = image.crop((x1_p, y1_p, x2_p, y2_p))
    return crop, (x1_p, y1_p, x2_p, y2_p), best_conf


def draw_bbox_on_image(image, bbox, label=None):
    """Draw a bounding box with label on an image copy."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    x1, y1, x2, y2 = bbox
    # Draw a thick red rectangle
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
    "Upload a clothing image. **YOLO** detects the product region, "
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
    alpha = st.select_slider(
        "Alpha (image vs text weight)",
        options=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        value=0.7,
        help="alpha=1.0 is vision-only, alpha=0.5 is equal image+text"
    )
    seed = st.selectbox("Index seed", [510, 51, 105], index=0)
    st.markdown("---")

    st.markdown("**Full Pipeline:**")
    st.markdown("1. Upload image")
    st.markdown("2. YOLO detects clothing region")
    st.markdown("3. BLIP-2 generates caption on crop")
    st.markdown("4. Fine-tuned CLIP encodes crop + caption")
    st.markdown(f"5. Fuse: `{alpha}*img + {1-alpha:.1f}*txt`")
    st.markdown("6. HNSW nearest-neighbor search")
    st.markdown("---")
    st.markdown(f"**Device:** `{DEVICE}`")

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
index, gallery_ids, idx_path = load_index(alpha, seed)
gallery_items = load_gallery_items()

with st.sidebar:
    st.success(f"CLIP loaded ({n_clip_params/1e6:.0f}M params)")
    st.success("BLIP-2 loaded")
    st.success("YOLO loaded")
    st.success(f"Index: {Path(idx_path).name} ({len(gallery_ids)} items)")

# ---- Display uploaded image + YOLO detection ----
query_image = Image.open(uploaded).convert("RGB")

# Run YOLO detection
with st.spinner("Running YOLO detection..."):
    cropped, bbox, det_conf = yolo_detect(query_image, yolo_model,
                                           conf=YOLO_CONF, iou=YOLO_IOU)

col_orig, col_det = st.columns(2)

with col_orig:
    st.subheader("Original Image")
    st.image(query_image, use_container_width=True)

with col_det:
    st.subheader("YOLO Detection")
    if bbox is not None:
        annotated = draw_bbox_on_image(query_image, bbox,
                                        label=f"conf={det_conf:.2f}")
        st.image(annotated, caption=f"BBox: {bbox} (conf={det_conf:.2f})",
                 use_container_width=True)
    else:
        st.image(query_image, caption="No detection - using full image",
                 use_container_width=True)

# ---- Crop confirmation ----
st.subheader("Crop Confirmation")

# Initialize session state for crop
if "crop_mode" not in st.session_state:
    st.session_state.crop_mode = "auto"

col_a, col_b, col_c = st.columns(3)
with col_a:
    if st.button("Confirm YOLO crop", type="primary", use_container_width=True):
        st.session_state.crop_mode = "auto"
        st.session_state.confirmed = True
with col_b:
    if st.button("Re-crop manually", use_container_width=True):
        st.session_state.crop_mode = "manual"
        st.session_state.confirmed = False
with col_c:
    if st.button("Use full image", use_container_width=True):
        st.session_state.crop_mode = "full"
        st.session_state.confirmed = True

# Manual re-crop
if st.session_state.crop_mode == "manual":
    st.markdown("**Adjust the crop region:**")
    W, H = query_image.size

    # Default to YOLO bbox if available
    default_x1 = bbox[0] if bbox else 0
    default_y1 = bbox[1] if bbox else 0
    default_x2 = bbox[2] if bbox else W
    default_y2 = bbox[3] if bbox else H

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        x1 = st.slider("x1 (left)", 0, W - 1, default_x1, key="manual_x1")
        y1 = st.slider("y1 (top)", 0, H - 1, default_y1, key="manual_y1")
    with mcol2:
        x2 = st.slider("x2 (right)", 1, W, default_x2, key="manual_x2")
        y2 = st.slider("y2 (bottom)", 1, H, default_y2, key="manual_y2")

    if x2 > x1 and y2 > y1:
        cropped = query_image.crop((x1, y1, x2, y2))
        manual_annotated = draw_bbox_on_image(query_image, (x1, y1, x2, y2),
                                               label="manual")
        crop_col1, crop_col2 = st.columns(2)
        with crop_col1:
            st.image(manual_annotated, caption="Manual selection",
                     use_container_width=True)
        with crop_col2:
            st.image(cropped, caption=f"Crop: {x2-x1}x{y2-y1}",
                     use_container_width=True)
    else:
        st.warning("Invalid crop region (x2 must be > x1, y2 must be > y1)")

    if st.button("Use this crop - search now", type="primary",
                 use_container_width=True, key="confirm_manual"):
        st.session_state.confirmed = True

elif st.session_state.crop_mode == "full":
    cropped = query_image
    st.info("Using full image (no crop)")

else:  # auto (YOLO crop)
    if bbox is not None:
        st.info(f"Using YOLO crop: {bbox}")
    else:
        st.info("No YOLO detection - using full image")
        cropped = query_image

# Show the final crop
st.subheader("Final Crop for Search")
st.image(cropped, width=300, caption=f"Size: {cropped.size[0]}x{cropped.size[1]}")

# ---- Search button ----
if not st.session_state.get("confirmed", False):
    search = st.button("Search with this crop", type="primary",
                       use_container_width=True)
    if not search:
        st.stop()
else:
    # Auto-confirmed modes proceed directly
    pass

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
    if alpha < 1.0:
        query_emb = fuse_embeddings(img_emb, txt_emb, alpha)
    else:
        query_emb = img_emb.cpu().numpy()

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
st.caption(f"Fused query: {alpha}*image + {1-alpha:.1f}*caption  |  "
           f"Total time: {total_time:.2f}s")

img_dir = Path(DATA_ROOT)
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
