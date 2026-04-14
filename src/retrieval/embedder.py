"""
src/retrieval/embedder.py
--------------------------
Compute fused image+text embeddings for the retrieval pipeline.

Fused vector (Eq. 1 from report):
    v_i = α * φ_V(x̂_i) + (1 - α) * φ_T(c_i),  ‖v_i‖ = 1

φ_V : CLIP visual encoder
φ_T : CLIP text encoder
c_i : BLIP-2 generated caption for item i
α   : image-text weighting hyper-parameter
"""

from typing import List, Optional, Tuple

import clip
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.models.clip_model import CLIPFineTuner
from src.models.blip2_model import BLIP2Captioner
from src.models.yolo_model import YOLODetector


class FusedEmbedder:
    """
    Orchestrates YOLO → BLIP-2 → CLIP to produce a single normalized
    embedding per image.

    Used both during offline indexing and online query processing.
    """

    def __init__(
        self,
        clip_model: CLIPFineTuner,
        blip2_captioner: Optional[BLIP2Captioner],
        yolo_detector: Optional[YOLODetector],
        alpha: float = 0.7,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            clip_model:       fine-tuned (or frozen) CLIP wrapper
            blip2_captioner:  BLIP-2 captioner; if None, α is forced to 1.0
            yolo_detector:    YOLO detector; if None, images are used as-is
            alpha:            weight for the image embedding (0 ≤ α ≤ 1)
            device:           torch device
        """
        self.clip = clip_model
        self.blip2 = blip2_captioner
        self.yolo = yolo_detector
        self.alpha = alpha if blip2_captioner is not None else 1.0
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if blip2_captioner is None:
            print("[Embedder] No BLIP-2 → vision-only mode (α=1.0)")

    # -------------------------------------------------------------- #
    #  Single image embedding                                          #
    # -------------------------------------------------------------- #

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Full pipeline for a single image:
        YOLO crop → BLIP-2 caption → CLIP fused embedding.

        Returns:
            (D,) float32 numpy array, L2-normalized
        """
        # Step 1: YOLO crop
        if self.yolo is not None:
            cropped, _ = self.yolo.detect_and_crop(image)
        else:
            cropped = image

        # Step 2: CLIP image embedding
        img_tensor = self.clip.preprocess(cropped).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_emb = self.clip.encode_image(img_tensor, normalize=True)  # (1, D)

        if self.alpha == 1.0 or self.blip2 is None:
            return img_emb.squeeze(0).cpu().float().numpy()

        # Step 3: BLIP-2 caption
        captions = self.blip2.caption([cropped])  # list of 1 string

        # Step 4: CLIP text embedding
        tokens = clip.tokenize(captions, truncate=True).to(self.device)
        with torch.no_grad():
            txt_emb = self.clip.encode_text(tokens, normalize=True)  # (1, D)

        # Step 5: Fuse & normalize
        fused = self.alpha * img_emb + (1 - self.alpha) * txt_emb
        fused = F.normalize(fused, dim=-1)
        return fused.squeeze(0).cpu().float().numpy()

    # -------------------------------------------------------------- #
    #  Batch embedding (catalog / gallery indexing)                    #
    # -------------------------------------------------------------- #

    def embed_catalog(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Compute fused embeddings for a list of images.

        Returns:
            (N, D) float32 numpy array
        """
        all_embeddings = []
        it = range(0, len(images), batch_size)
        if show_progress:
            it = tqdm(it, desc="Embedding catalog", unit="batch")

        for start in it:
            batch_imgs = images[start: start + batch_size]
            batch_embs = [self.embed_image(img) for img in batch_imgs]
            all_embeddings.extend(batch_embs)

        return np.stack(all_embeddings, axis=0).astype(np.float32)

    def embed_query(self, image: Image.Image) -> np.ndarray:
        """Convenience alias for online query embedding."""
        return self.embed_image(image)
