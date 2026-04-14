"""
app/streamlit_app.py
---------------------
Streamlit demo — Deliverable 2.

Features:
  - Upload a query image
  - Run YOLO detection & display the detected crop
  - Confirm crop / request re-crop
  - Run the full retrieval pipeline
  - Display top-K results with metadata and similarity scores

Run:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import torch

# ------------------------------------------------------------------ #
#  Page config                                                         #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Visual Product Search",
    page_icon="👗",
    layout="wide",
)

# ------------------------------------------------------------------ #
#  Lazy model loading (cached so they load only once)                  #
# ------------------------------------------------------------------ #

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    from omegaconf import OmegaConf
    from src.models.clip_model import CLIPFineTuner
    from src.models.blip2_model import BLIP2Captioner, BLIP2Reranker
    from src.models.yolo_model import YOLODetector
    from src.retrieval.embedder import FusedEmbedder
    from src.retrieval.indexer import HNSWIndexer
    from src.retrieval.retriever import Retriever
    from src.utils.helpers import get_device, load_config, load_checkpoint

    cfg = load_config("configs/config.yaml")
    device = get_device()

    # CLIP
    clip_model = CLIPFineTuner(
        model_name=cfg.clip.model_name,
        unfreeze_vision_blocks=0,
        freeze_text_encoder=True,
        device=device,
    ).to(device)

    best_ckpt = Path(cfg.paths.checkpoint_dir) / "best_model.pt"
    if best_ckpt.exists():
        load_checkpoint(clip_model, str(best_ckpt), device=device)
        st.sidebar.success(f"Fine-tuned CLIP loaded ✓")
    else:
        st.sidebar.warning("No fine-tuned checkpoint found. Using pretrained CLIP.")

    # BLIP-2
    blip2, reranker = None, None
    if cfg.embedding.alpha < 1.0:
        try:
            blip2 = BLIP2Captioner(model_name=cfg.blip2.model_name,
                                    device_map=cfg.blip2.device_map)
            reranker = BLIP2Reranker(model_name=cfg.blip2.model_name,
                                      device_map=cfg.blip2.device_map)
        except Exception as e:
            st.sidebar.warning(f"BLIP-2 unavailable: {e}")

    # YOLO
    try:
        yolo = YOLODetector(model_name=cfg.yolo.model_name)
    except Exception as e:
        yolo = None
        st.sidebar.warning(f"YOLO unavailable: {e}")

    # Embedder
    embedder = FusedEmbedder(clip_model, blip2, yolo,
                              alpha=cfg.embedding.alpha, device=device)

    # Index
    index_files = sorted(Path(cfg.paths.index_dir).glob("hnsw_*.bin"))
    if not index_files:
        st.error("No HNSW index found. Run `scripts/build_index.py` first.")
        st.stop()

    # Use the most recently modified index
    index_path = str(index_files[-1])
    meta_path  = index_path.replace("hnsw_", "metadata_").replace(".bin", ".json")

    indexer = HNSWIndexer(dim=cfg.embedding.embedding_dim)
    indexer.load(index_path, meta_path)

    retriever = Retriever(
        embedder, indexer,
        reranker=reranker if cfg.eval.rerank else None,
        rerank_top_n=cfg.eval.rerank_top_n,
        ef_search=cfg.index.ef_search,
    )

    return retriever, cfg


# ------------------------------------------------------------------ #
#  UI                                                                  #
# ------------------------------------------------------------------ #

st.title("👗 Visual Product Search")
st.markdown("Upload a clothing image and find visually similar products from the catalog.")

# Sidebar
with st.sidebar:
    st.header("Search Settings")
    top_k = st.slider("Top-K results", min_value=1, max_value=20, value=10)
    show_scores = st.checkbox("Show similarity scores", value=True)
    st.markdown("---")
    st.markdown("**Pipeline:** YOLO → BLIP-2 → CLIP → HNSW")

# File uploader
uploaded = st.file_uploader(
    "Upload a query image", type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("👆 Upload an image to begin.")
    st.stop()

# Display uploaded image
query_image = Image.open(uploaded).convert("RGB")
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Your query image")
    st.image(query_image, use_column_width=True)

# ---- Load models ----
retriever, cfg = load_models()
img_dir = Path(cfg.paths.img_dir)

# ---- YOLO detection ----
with st.spinner("Running YOLO detection…"):
    if retriever.embedder.yolo is not None:
        cropped, bbox = retriever.embedder.yolo.detect_and_crop(query_image)
    else:
        cropped, bbox = query_image, None

with col2:
    st.subheader("Detected product region (YOLO)")
    if bbox:
        # Draw bbox on original
        annotated = query_image.copy()
        draw = ImageDraw.Draw(annotated)
        draw.rectangle(bbox, outline="red", width=3)
        st.image(annotated, caption=f"BBox: {bbox}", use_column_width=True)
    else:
        st.image(cropped, caption="No detection — using full image",
                 use_column_width=True)

# ---- User confirmation ----
st.subheader("Crop confirmation")
col_a, col_b = st.columns(2)
with col_a:
    confirm = st.button("✅ Confirm crop — search now", type="primary")
with col_b:
    recrop = st.button("🔄 Re-crop (adjust manually)")

if recrop:
    st.info("Manual re-cropping: use the sliders below to set the crop region.")
    W, H = query_image.size
    x1 = st.slider("x1 (left)", 0, W - 1, 0)
    y1 = st.slider("y1 (top)", 0, H - 1, 0)
    x2 = st.slider("x2 (right)", 0, W, W)
    y2 = st.slider("y2 (bottom)", 0, H, H)
    cropped = query_image.crop((x1, y1, x2, y2))
    st.image(cropped, caption="Custom crop", width=200)
    confirm = st.button("✅ Use this crop — search now", type="primary",
                         key="confirm_manual")

if not confirm:
    st.stop()

# ---- Retrieval ----
st.subheader(f"🔍 Top-{top_k} Search Results")
with st.spinner("Searching the catalog…"):
    results = retriever.query(cropped, top_k=top_k)

if not results:
    st.error("No results returned. Check that the index is populated.")
    st.stop()

# ---- Display results ----
n_cols = 5
rows = [results[i: i + n_cols] for i in range(0, len(results), n_cols)]

for row in rows:
    cols = st.columns(n_cols)
    for col, res in zip(cols, row):
        with col:
            try:
                img_path = img_dir / res.img_path
                result_img = Image.open(img_path).convert("RGB")
                st.image(result_img, use_column_width=True)
            except Exception:
                st.warning("Image unavailable")

            st.markdown(f"**Rank {res.rank}**")
            st.markdown(f"Item: `{res.item_id}`")
            if show_scores:
                st.markdown(f"Cosine sim: `{res.similarity:.3f}`")
                if res.itm_score is not None:
                    st.markdown(f"ITM score: `{res.itm_score:.3f}`")
            if res.caption:
                with st.expander("Caption"):
                    st.write(res.caption)
