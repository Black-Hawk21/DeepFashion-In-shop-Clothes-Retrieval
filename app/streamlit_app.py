"""
app/streamlit_app.py
---------------------
Streamlit demo - Visual Product Search.

Uses the fine-tuned CLIP checkpoint directly (NO base model download needed)
and the Cond_C HNSW index for retrieval.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import hnswlib

# ------------------------------------------------------------------ #
#  Page config                                                         #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Visual Product Search",
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ #
#  Build CLIP architecture from scratch + load checkpoint weights      #
#  This avoids downloading the base ViT-B/32 model entirely           #
# ------------------------------------------------------------------ #

def build_clip_from_checkpoint(checkpoint_path, device):
    """
    Build the CLIP ViT-B/32 model architecture using clip's internal
    model builder, then load all weights from the fine-tuned checkpoint.
    No internet download needed.
    """
    import clip
    from clip.model import build_model
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Strip "clip_model." prefix from all keys
    clean_sd = {}
    for k, v in state_dict.items():
        if k.startswith("clip_model."):
            clean_sd[k[len("clip_model."):]] = v
        else:
            clean_sd[k] = v

    # Build model architecture from state_dict (clip's internal function)
    model = build_model(clean_sd).to(device)
    model = model.float()
    model.eval()

    # Build the standard CLIP preprocessing pipeline
    # (ViT-B/32 uses 224x224 input)
    preprocess = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])

    return model, preprocess


# ------------------------------------------------------------------ #
#  Cached loaders                                                      #
# ------------------------------------------------------------------ #

@st.cache_data(show_spinner=False)
def load_bbox_map():
    """Load bbox annotations for gallery images."""
    bbox = {}
    try:
        with open(BBOX_FILE) as f:
            lines = f.read().splitlines()
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            bbox[parts[0]] = (int(parts[3]), int(parts[4]),
                              int(parts[5]), int(parts[6]))
    except FileNotFoundError:
        pass
    return bbox


@st.cache_resource(show_spinner="Loading fine-tuned CLIP model...")
def load_clip_model():
    """Load fine-tuned CLIP from checkpoint without any download."""
    ckpt_path = Path(CLIP_CHECKPOINT)
    if not ckpt_path.exists():
        st.error(f"Checkpoint not found: {ckpt_path}")
        st.stop()

    model, preprocess = build_clip_from_checkpoint(str(ckpt_path), DEVICE)
    return model, preprocess


@st.cache_resource(show_spinner="Loading HNSW index...")
def load_index(alpha, seed):
    """Load HNSW index and metadata."""
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
def build_gallery_index_map():
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
#  Query embedding                                                     #
# ------------------------------------------------------------------ #

def embed_query(image, model, preprocess):
    """Embed a single query image with the fine-tuned CLIP."""
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_image(img_tensor)
        emb = F.normalize(emb.float(), dim=-1)
    return emb.cpu().numpy()


# ------------------------------------------------------------------ #
#  UI                                                                  #
# ------------------------------------------------------------------ #

st.title("Visual Product Search")
st.markdown("Upload a clothing image and find visually similar products from the catalog.")

# Sidebar
with st.sidebar:
    st.header("Search Settings")
    top_k = st.slider("Top-K results", min_value=1, max_value=20, value=10)
    show_scores = st.checkbox("Show similarity scores", value=True)
    st.markdown("---")

    st.header("Index Settings")
    alpha = st.selectbox("Alpha (image weight)", [0.7, 0.5], index=0)
    seed = st.selectbox("Seed", [510, 51, 105], index=0)
    st.markdown("---")

    st.markdown("**Pipeline:** Upload -> CLIP (fine-tuned) -> HNSW -> Results")
    st.markdown(f"**Device:** `{DEVICE}`")

# File uploader
uploaded = st.file_uploader(
    "Upload a query image", type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

# Display uploaded image
query_image = Image.open(uploaded).convert("RGB")
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Your query image")
    st.image(query_image, use_container_width=True)

# ---- Load models and index ----
model, preprocess = load_clip_model()
index, gallery_ids, idx_path = load_index(alpha, seed)
gallery_items = build_gallery_index_map()

with st.sidebar:
    st.success("CLIP checkpoint loaded")
    st.success(f"Index: {Path(idx_path).name} ({len(gallery_ids)} items)")

with col2:
    st.subheader("Ready to search")
    st.write(f"Image size: {query_image.size[0]}x{query_image.size[1]}")
    st.write(f"Gallery: {len(gallery_ids)} items | Alpha: {alpha}")

# ---- Search button ----
search = st.button("Search now", type="primary", use_container_width=True)

if not search:
    st.stop()

# ---- Retrieval ----
st.subheader(f"Top-{top_k} Search Results")
with st.spinner("Searching the catalog..."):
    query_emb = embed_query(query_image, model, preprocess)
    indices, distances = index.knn_query(query_emb, k=top_k)
    indices = indices[0]
    distances = distances[0]
    similarities = 1.0 - distances

# ---- Display results ----
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
