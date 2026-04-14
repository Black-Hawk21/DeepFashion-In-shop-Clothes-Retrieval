"""
src/retrieval/indexer.py
-------------------------
Build, save, and load an HNSW approximate nearest-neighbor index
using hnswlib (local) or optionally Milvus / Pinecone.

The index stores:
  - embedding vectors (float32, cosine space)
  - item_ids and image_paths as parallel metadata lists

Usage:
    indexer = HNSWIndexer(dim=512)
    indexer.build(embeddings, item_ids, img_paths)
    indexer.save("index/hnsw.bin", "index/metadata.json")

    # Later:
    indexer = HNSWIndexer(dim=512)
    indexer.load("index/hnsw.bin", "index/metadata.json")
    distances, indices = indexer.search(query_emb, top_k=20)
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import hnswlib


class HNSWIndexer:
    """
    Wraps hnswlib for cosine-similarity ANN search.

    Space is 'cosine', which is equivalent to inner-product on L2-normalized
    vectors. Distances returned are in [0, 2] (lower = more similar).
    Convert to similarity: sim = 1 - dist.
    """

    def __init__(self, dim: int = 512, space: str = "cosine"):
        self.dim = dim
        self.space = space
        self.index: Optional[hnswlib.Index] = None
        self.item_ids: List[str] = []
        self.img_paths: List[str] = []
        self.captions: List[str] = []
        self.n_items: int = 0

    # -------------------------------------------------------------- #
    #  Build                                                           #
    # -------------------------------------------------------------- #

    def build(
        self,
        embeddings: np.ndarray,       # (N, D) float32, L2-normalized
        item_ids: List[str],
        img_paths: List[str],
        captions: Optional[List[str]] = None,
        ef_construction: int = 200,
        M: int = 16,
    ) -> None:
        """
        Construct the HNSW index from embeddings.

        Args:
            embeddings:      (N, D) float32 embeddings
            item_ids:        item_id string for each embedding
            img_paths:       relative image path for each embedding
            captions:        BLIP-2 caption for each embedding (optional)
            ef_construction: HNSW build-time accuracy/speed trade-off
            M:               number of bidirectional links per element
        """
        assert embeddings.shape[0] == len(item_ids) == len(img_paths), \
            "embeddings, item_ids, and img_paths must have the same length"

        N, D = embeddings.shape
        assert D == self.dim, f"Expected dim={self.dim}, got {D}"

        print(f"[Indexer] Building HNSW index for {N} items (dim={D}, space={self.space})")

        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(
            max_elements=N,
            ef_construction=ef_construction,
            M=M,
        )
        # hnswlib uses integer labels; we use sequential 0..N-1
        self.index.add_items(embeddings, np.arange(N))

        self.item_ids = list(item_ids)
        self.img_paths = list(img_paths)
        self.captions = list(captions) if captions else [""] * N
        self.n_items = N

        print(f"[Indexer] Index built with {N} items.")

    # -------------------------------------------------------------- #
    #  Search                                                          #
    # -------------------------------------------------------------- #

    def search(
        self,
        query: np.ndarray,   # (D,) or (B, D) float32
        top_k: int = 20,
        ef_search: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top_k nearest neighbors.

        Args:
            query:   (D,) single query or (B, D) batch
            top_k:   number of results per query
            ef_search: search-time accuracy parameter (≥ top_k)

        Returns:
            indices:   (B, top_k) integer array into stored items
            distances: (B, top_k) float32 cosine distances [0, 2]
        """
        assert self.index is not None, "Index not built. Call build() or load() first."

        self.index.set_ef(max(ef_search, top_k))

        if query.ndim == 1:
            query = query.reshape(1, -1)

        indices, distances = self.index.knn_query(query, k=top_k)
        return indices, distances  # both (B, top_k)

    def get_metadata(self, idx: int) -> dict:
        """Return metadata dict for a stored item by integer index."""
        return {
            "item_id": self.item_ids[idx],
            "img_path": self.img_paths[idx],
            "caption": self.captions[idx],
        }

    # -------------------------------------------------------------- #
    #  Persistence                                                     #
    # -------------------------------------------------------------- #

    def save(self, index_path: str, meta_path: str) -> None:
        """Save HNSW index binary and metadata JSON."""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(meta_path).parent.mkdir(parents=True, exist_ok=True)

        self.index.save_index(index_path)

        meta = {
            "dim": self.dim,
            "space": self.space,
            "n_items": self.n_items,
            "item_ids": self.item_ids,
            "img_paths": self.img_paths,
            "captions": self.captions,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        print(f"[Indexer] Saved index → {index_path}")
        print(f"[Indexer] Saved metadata → {meta_path}")

    def load(self, index_path: str, meta_path: str) -> None:
        """Load a previously saved index."""
        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.dim = meta["dim"]
        self.space = meta["space"]
        self.n_items = meta["n_items"]
        self.item_ids = meta["item_ids"]
        self.img_paths = meta["img_paths"]
        self.captions = meta["captions"]

        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.load_index(index_path, max_elements=self.n_items)

        print(f"[Indexer] Loaded index from {index_path}  ({self.n_items} items)")
