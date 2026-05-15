# Visual Product Search Engine

A query-by-image product search system built on the DeepFashion In-Shop Clothes Retrieval dataset.

## Project Overview

The system lets users upload a clothing image and retrieve visually and semantically similar products. It combines YOLO-based cropping, fine-tuned CLIP embeddings, HNSW nearest-neighbor search, and BLIP-2 image-text matching (ITM) re-ranking.

## Models and Checkpoints

- YOLO detector: checkpoints/clothing yolo.pt comes from the kesimeg/yolov8n-clothing-detection model.
- CLIP: checkpoints/best_model.pt is the fine-tuned CLIP checkpoint used for embedding.
- BLIP-2: Salesforce/blip2-opt-2.7b via the HuggingFace cache for captions and ITM scoring.

## Pipeline

### Offline indexing

1. Generate a BLIP-2 caption for each gallery image.
2. Encode image and caption with CLIP, fuse them with an alpha weight, and build an HNSW index (hnswlib).
3. Persist gallery captions and metadata under index/.

### Online retrieval (shared steps)

1. YOLO detects clothing and crops the main region.
2. CLIP encodes the crop (image-only or fused with a BLIP-2 caption depending on the app).
3. HNSW retrieves a candidate pool by cosine similarity.
4. BLIP-2 ITM scores query image vs candidate captions and re-ranks.

### Streamlit variants

- app/streamlit_rerank_fullbody.py: uses CLIP image embeddings only before searching the index, then BLIP-2 ITM re-ranking. Includes a full-body union crop option.
- app/streamlit_best_demo.py: generates a BLIP-2 caption for the query crop, fuses caption and image with CLIP before retrieval, then applies BLIP-2 ITM re-ranking. This gave the best results.
- app/streamlit_app_full.py, app/streamlit_app.py, app/streamlit_rerank.py: earlier demos and ablations.

## Repository Structure

```
DeepFashion-In-shop-Clothes-Retrieval_/
├── app/
│   ├── streamlit_best_demo.py        # Best demo: BLIP-2 caption + CLIP fusion + ITM re-rank
│   ├── streamlit_rerank_fullbody.py  # CLIP image-only retrieval + ITM re-rank
│   ├── streamlit_app_full.py         # Full pipeline demo
│   ├── streamlit_app.py              # Basic demo
│   └── streamlit_rerank.py           # Rerank demo
├── checkpoints/
│   ├── best_model.pt                 # Fine-tuned CLIP weights
│   └── clothing yolo.pt              # YOLOv8n clothing detector
├── configs/
│   └── config.yaml                   # Hyperparameters and paths
├── data/
│   ├── README.txt
│   ├── Anno/
│   ├── Eval/
│   └── Img/img/
├── index/
│   ├── gallery_captions.json
│   ├── Cond_A/
│   ├── Cond_B/
│   └── Cond_C/
├── notebooks/
│   ├── clip_fine_tune.ipynb
│   ├── exploration.ipynb
│   ├── part_a_vision_only.py
│   ├── part_b_frozen_clip_blip2.py
│   └── part_c_finetuned_clip.py
├── results/
│   ├── evaluation_report.json
│   ├── Cond_A/
│   ├── Cond_B/
│   └── Cond_C/
├── scripts/
│   ├── build_index.py
│   ├── evaluate.py
│   ├── finetune_yolo.py
│   ├── run_ablation.py
│   └── train_clip.py
├── src/
│   ├── models/
│   ├── retrieval/
│   └── utils/
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Place the DeepFashion dataset under data/ following the structure in data/README.txt.

## Dependency Versions (tested)

requirements.txt is pinned to the reverse_image_search conda environment. The HF BLIP-2 model and YOLO detector were validated with these key versions:

- torch==2.5.1 and torchvision==0.20.1
- transformers==5.8.1, accelerate==1.13.0, tokenizers==0.22.2, huggingface_hub==1.14.0
- ultralytics==8.2.71 and opencv-python==4.11.0.86
- hnswlib==0.8.0
- openai-clip==1.0.1
- streamlit==1.57.0
- numpy==1.26.4 and pillow==12.2.0

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

### 4. Launch Streamlit demos

```bash
streamlit run app/streamlit_best_demo.py
```

```bash
streamlit run app/streamlit_rerank_fullbody.py
```

## Ablation Conditions

| ID | Configuration | alpha |
|---|---|---|
| A | Vision-only CLIP (baseline) | 1.0 |
| B | Frozen CLIP + frozen BLIP-2 | {0.5, 0.7} |
| C | Fine-tuned CLIP + frozen BLIP-2 | {0.5, 0.7} |

Notes:

- A uses only CLIP image embeddings (no BLIP-2 caption fusion).
- B uses BLIP-2 captions fused with a frozen CLIP image encoder.
- C uses BLIP-2 captions fused with a fine-tuned CLIP image encoder.

Results are reported as mean +/- std over 3-4 seeds.

## Metrics

- Recall@K - fraction of queries with at least one correct match in top-K
- NDCG@K - position-aware ranking gain
- mAP@K - mean average precision up to rank K

Reported at K in {5, 10, 15}.
