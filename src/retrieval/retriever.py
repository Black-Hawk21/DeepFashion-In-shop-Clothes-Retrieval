"""
src/retrieval/retriever.py
---------------------------
Online retrieval pipeline:
  query image → YOLO crop → CLIP embedding → ANN search → BLIP-2 re-rank

Returns top-K (item_id, img_path, score, caption) tuples.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from src.retrieval.embedder import FusedEmbedder
from src.retrieval.indexer import HNSWIndexer
from src.models.blip2_model import BLIP2Reranker


@dataclass
class RetrievalResult:
    rank: int
    item_id: str
    img_path: str
    caption: str
    similarity: float          # cosine similarity ∈ [-1, 1]
    itm_score: Optional[float] # BLIP-2 ITM score (post re-rank)


class Retriever:
    """
    End-to-end online retrieval.

    Typical usage:
        retriever = Retriever(embedder, indexer, reranker)
        results = retriever.query(pil_image, top_k=10)
    """

    def __init__(
        self,
        embedder: FusedEmbedder,
        indexer: HNSWIndexer,
        reranker: Optional[BLIP2Reranker] = None,
        rerank_top_n: int = 50,
        ef_search: int = 100,
    ):
        self.embedder = embedder
        self.indexer = indexer
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n
        self.ef_search = ef_search

    def query(
        self,
        image: Image.Image,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Full retrieval pipeline for a single query image.

        Args:
            image:  PIL Image (user-uploaded query)
            top_k:  number of final results to return

        Returns:
            Sorted list of RetrievalResult (rank 1 = most similar)
        """
        # 1. Embed the query
        query_emb = self.embedder.embed_query(image)          # (D,)

        # 2. ANN search — fetch more than top_k to allow re-ranking
        n_candidates = self.rerank_top_n if self.reranker else top_k
        indices, distances = self.indexer.search(
            query_emb, top_k=n_candidates, ef_search=self.ef_search
        )
        # indices, distances: (1, n_candidates)
        indices = indices[0]
        distances = distances[0]

        # Convert cosine distance → similarity
        similarities = 1.0 - distances  # distance ∈ [0,2] → sim ∈ [-1,1]

        # 3. Gather metadata for candidates
        candidates = [self.indexer.get_metadata(int(i)) for i in indices]

        # 4. Optional BLIP-2 re-ranking
        itm_scores = [None] * len(candidates)
        if self.reranker is not None:
            captions = [c["caption"] for c in candidates]
            # get cropped query image for ITM (re-use YOLO crop from embedder)
            if self.embedder.yolo:
                query_crop, _ = self.embedder.yolo.detect_and_crop(image)
            else:
                query_crop = image

            reranked_order = self.reranker.rerank(query_crop, captions, list(range(len(candidates))))
            itm_score_list = self.reranker.score(query_crop, captions)

            # Apply reranked order
            candidates = [candidates[i] for i in reranked_order]
            similarities = [similarities[i] for i in reranked_order]
            itm_scores = [itm_score_list[i] for i in reranked_order]

        # 5. Build result list (top-K)
        results = []
        for rank, (meta, sim, itm) in enumerate(
            zip(candidates[:top_k], similarities[:top_k], itm_scores[:top_k]), start=1
        ):
            results.append(RetrievalResult(
                rank=rank,
                item_id=meta["item_id"],
                img_path=meta["img_path"],
                caption=meta["caption"],
                similarity=float(sim),
                itm_score=float(itm) if itm is not None else None,
            ))

        return results

    # -------------------------------------------------------------- #
    #  Batch retrieval (for evaluation script)                         #
    # -------------------------------------------------------------- #

    def batch_query(
        self,
        embeddings: np.ndarray,   # (Q, D) precomputed query embeddings
        top_k: int = 15,
    ) -> np.ndarray:
        """
        Fast batch retrieval without re-ranking.
        Returns ranked_indices: (Q, top_k) integer array into gallery.
        Used by evaluate.py for metric computation.
        """
        n_candidates = top_k
        indices, _ = self.indexer.search(
            embeddings, top_k=n_candidates, ef_search=self.ef_search
        )
        return indices   # (Q, top_k)
