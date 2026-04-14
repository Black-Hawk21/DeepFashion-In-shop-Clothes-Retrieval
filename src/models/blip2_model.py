"""
src/models/blip2_model.py
--------------------------
BLIP-2 wrapper for:
  1. Semantic captioning (OPT / FlanT5 backbone)
  2. Image-Text Matching (ITM) re-ranking

Both captioning and ITM use the frozen pretrained model;
no fine-tuning is performed (per assignment spec).

Hardware note:
  BLIP-2 with OPT-2.7B fits in ~12 GB VRAM in fp16.
  Use blip2-opt-2.7b for faster inference.
  Use blip2-flan-t5-xl for better caption quality if VRAM allows.
"""

from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


# ------------------------------------------------------------------ #
#  BLIP-2 Captioner                                                    #
# ------------------------------------------------------------------ #

class BLIP2Captioner:
    """
    Generates natural-language captions for product images.
    Model is always loaded in inference mode (frozen).
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device_map: str = "auto",
        max_new_tokens: int = 50,
        torch_dtype: torch.dtype = torch.float16,
    ):
        print(f"[BLIP2] Loading captioner: {model_name}")
        self.max_new_tokens = max_new_tokens

        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        print("[BLIP2] Captioner ready (frozen).")

    @torch.no_grad()
    def caption(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompt: str = "A photo of a clothing item:",
    ) -> List[str]:
        """
        Generate captions for one or a batch of PIL images.

        Args:
            images: single PIL image or list of PIL images
            prompt: conditioning text prompt

        Returns:
            List of caption strings, one per image
        """
        if isinstance(images, Image.Image):
            images = [images]

        inputs = self.processor(
            images=images,
            text=[prompt] * len(images),
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.float16)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )
        captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
        # Strip the prompt from the output if echoed
        captions = [c.replace(prompt, "").strip() for c in captions]
        return captions


# ------------------------------------------------------------------ #
#  BLIP-2 Image-Text Matching (for re-ranking)                         #
# ------------------------------------------------------------------ #

class BLIP2Reranker:
    """
    Uses BLIP-2 Image-Text Matching (ITM) scores to re-rank
    candidate retrieval results.

    For each (query_image, candidate_caption) pair, computes an ITM
    relevance score. Candidates are then sorted by this score.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
    ):
        print(f"[BLIP2] Loading reranker: {model_name}")

        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print("[BLIP2] Reranker ready (frozen).")

    @torch.no_grad()
    def score(
        self,
        query_image: Image.Image,
        candidate_captions: List[str],
    ) -> List[float]:
        """
        Compute ITM scores for (query_image, caption) pairs.

        Strategy: We condition the model on each caption and compute the
        average log-probability of generating the caption given the image.
        Higher log-probability → better image-text alignment.

        Args:
            query_image: PIL image of the query product
            candidate_captions: list of captions for candidate products

        Returns:
            List of float scores, one per caption (higher = more relevant)
        """
        scores = []
        for caption in candidate_captions:
            inputs = self.processor(
                images=query_image,
                text=caption,
                return_tensors="pt",
            ).to(self.model.device, dtype=torch.float16)

            # Compute generation loss (negative log-likelihood) as proxy for ITM
            labels = inputs["input_ids"].clone()
            out = self.model(**inputs, labels=labels)
            # Lower loss = better match → negate for ranking
            score = -float(out.loss.item())
            scores.append(score)

        return scores

    def rerank(
        self,
        query_image: Image.Image,
        candidate_captions: List[str],
        candidate_indices: List[int],
    ) -> List[int]:
        """
        Re-rank candidate_indices by ITM score.

        Returns:
            candidate_indices sorted by descending ITM score
        """
        scores = self.score(query_image, candidate_captions)
        sorted_pairs = sorted(
            zip(scores, candidate_indices), key=lambda x: x[0], reverse=True
        )
        return [idx for _, idx in sorted_pairs]
