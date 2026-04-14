# Visual Product Search Engine

A query-by-image product search system built on the **DeepFashion In-Shop Clothes Retrieval** dataset.

## Project Overview

The system allows users to upload a clothing image and retrieve visually and semantically similar products from a catalog using a cross-modal embedding pipeline.

### Pipeline Summary

| Stage | Module | Role |
|---|---|---|
| Detection | YOLOv8 | Crop the primary clothing item |
| Captioning | BLIP-2 | Generate semantic descriptions |
| Embedding | CLIP (fine-tuned) | Fused image+text vector |
| Indexing | HNSW (Milvus/Pinecone) | ANN retrieval |
| Re-ranking | BLIP-2 ITM | Semantic re-ranking |

---

## File Structure

```
visual_product_search/
├── configs/
│   └── config.yaml               # All hyperparameters and paths
├── data/
│   └── deepfashion/              # Place dataset here after download
│       ├── Img/img/
│       ├── Anno/
│       └── Eval/list_eval_partition.txt
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clip_model.py         # CLIP wrapper + fine-tuning logic
│   │   ├── blip2_model.py        # BLIP-2 captioning + ITM
│   │   └── yolo_model.py         # YOLO detection + cropping
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embedder.py           # Fused embedding computation
│   │   ├── indexer.py            # HNSW index build & save
│   │   └── retriever.py          # ANN search + re-ranking
│   └── utils/
│       ├── __init__.py
│       ├── dataset.py            # DeepFashion dataset & dataloader
│       ├── metrics.py            # Recall@K, NDCG@K, mAP@K
│       └── helpers.py            # Image I/O, logging, seed utils
├── scripts/
│   ├── build_index.py            # Offline: embed catalog & build HNSW
│   ├── train_clip.py             # Fine-tune CLIP on DeepFashion
│   └── evaluate.py               # Batch evaluation script (deliverable 3)
├── app/
│   └── streamlit_app.py          # Streamlit demo (deliverable 2)
├── notebooks/
│   └── exploration.ipynb         # EDA and quick experiments
├── checkpoints/                  # Saved fine-tuned CLIP weights
├── index/                        # Saved HNSW index files
├── results/                      # Evaluation outputs (JSON/CSV)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place the DeepFashion dataset under `data/deepfashion/` following the structure in the dataset README.

---

## Usage

### 1. Fine-tune CLIP
```bash
python scripts/train_clip.py --config configs/config.yaml
```

### 2. Build the offline index
```bash
python scripts/build_index.py --config configs/config.yaml
```

### 3. Batch evaluation
```bash
python scripts/evaluate.py --config configs/config.yaml --split query
```

### 4. Launch Streamlit demo
```bash
streamlit run app/streamlit_app.py
```

---

## Ablation Conditions

| ID | Configuration | α |
|---|---|---|
| A | Vision-only CLIP (baseline) | 1.0 |
| B | Frozen CLIP + frozen BLIP-2 | {0.5, 0.7} |
| C | Fine-tuned CLIP + frozen BLIP-2 | {0.5, 0.7} |

Results are reported as mean ± std over 3–4 seeds.

---

## Metrics

- **Recall@K** – fraction of queries with ≥1 correct match in top-K
- **NDCG@K** – position-aware ranking gain
- **mAP@K** – mean average precision up to rank K

Reported at K ∈ {5, 10, 15}.
