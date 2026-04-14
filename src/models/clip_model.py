"""
src/models/clip_model.py
-------------------------
CLIP wrapper with fine-tuning support.

Key responsibilities:
  1. Load pretrained CLIP (ViT-B/32 or others).
  2. Selectively unfreeze the last N vision transformer blocks.
  3. Keep the text encoder frozen (per assignment spec).
  4. Expose encode_image() and encode_text() with L2 normalization.
  5. InfoNCE / NT-Xent contrastive loss for item-level metric learning.

Fine-tuning goal: pull embeddings of the same item_id together,
push different item_ids apart in the shared embedding space.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip  # openai/CLIP


# ------------------------------------------------------------------ #
#  CLIP wrapper                                                        #
# ------------------------------------------------------------------ #

class CLIPFineTuner(nn.Module):
    """
    Thin wrapper around openai/CLIP that supports selective fine-tuning
    of the vision encoder's last N transformer blocks.

    Args:
        model_name: CLIP model string, e.g. 'ViT-B/32'
        unfreeze_vision_blocks: number of trailing transformer blocks to
            unfreeze. -1 means unfreeze the entire vision encoder.
        freeze_text_encoder: always True per the project spec.
        device: torch device
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        unfreeze_vision_blocks: int = 4,
        freeze_text_encoder: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP
        self.clip_model, self.preprocess = clip.load(model_name, device=self.device)
        # Convert to float32 for stable fine-tuning
        self.clip_model = self.clip_model.float()

        # Freeze everything first
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Unfreeze selected vision encoder blocks
        self._unfreeze_vision_blocks(unfreeze_vision_blocks)

        # Text encoder stays frozen (freeze_text_encoder is always True)
        if not freeze_text_encoder:
            # This branch is kept for flexibility but not used in the project
            for param in self.clip_model.transformer.parameters():
                param.requires_grad = True

        self._log_trainable_params()

    # -------------------------------------------------------------- #
    #  Selective unfreezing                                            #
    # -------------------------------------------------------------- #

    def _unfreeze_vision_blocks(self, n: int) -> None:
        """
        Unfreeze the last `n` transformer blocks of the vision encoder.
        Also unfreezes the final layer norm (ln_post) and projection.
        Pass n=-1 to unfreeze everything.
        """
        vision_enc = self.clip_model.visual

        # The ViT residual blocks are in visual.transformer.resblocks
        resblocks = list(vision_enc.transformer.resblocks)
        total_blocks = len(resblocks)

        if n == -1:
            blocks_to_unfreeze = resblocks
        else:
            n = min(n, total_blocks)
            blocks_to_unfreeze = resblocks[-n:]

        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True

        # Always unfreeze the output projection & layer norm
        for param in vision_enc.ln_post.parameters():
            param.requires_grad = True
        if vision_enc.proj is not None:
            vision_enc.proj.requires_grad = True

        print(f"[CLIP] Unfroze last {n} vision blocks "
              f"(total={total_blocks}) + ln_post + proj")

    def _log_trainable_params(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[CLIP] Trainable: {trainable:,} / {total:,} params "
              f"({100 * trainable / total:.2f}%)")

    # -------------------------------------------------------------- #
    #  Encoding                                                        #
    # -------------------------------------------------------------- #

    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) float tensor preprocessed by CLIP
            normalize: L2-normalize the output embedding
        Returns:
            (B, D) float tensor
        """
        features = self.clip_model.encode_image(images)
        if normalize:
            features = F.normalize(features, dim=-1)
        return features

    def encode_text(self, texts: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Args:
            texts: tokenized text tensor from clip.tokenize(...)
            normalize: L2-normalize the output embedding
        Returns:
            (B, D) float tensor
        """
        with torch.no_grad():   # text encoder always frozen
            features = self.clip_model.encode_text(texts)
        if normalize:
            features = F.normalize(features, dim=-1)
        return features

    def forward(
        self, images: torch.Tensor, texts: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(texts) if texts is not None else None
        return img_emb, txt_emb


# ------------------------------------------------------------------ #
#  Contrastive losses                                                  #
# ------------------------------------------------------------------ #

class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (NT-Xent) loss for (anchor, positive) pairs.

    Given a batch of (anchor_emb, positive_emb) pairs, each pair forms
    one positive; all other cross-pair combinations are negatives.

    temperature: logit scale (lower = harder decision boundary)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor:   (B, D) L2-normalized embeddings
            positive: (B, D) L2-normalized embeddings (same item_id)
        Returns:
            scalar loss
        """
        B = anchor.size(0)
        # Similarity matrix: (B, B)
        logits = torch.matmul(anchor, positive.T) / self.temperature
        labels = torch.arange(B, device=anchor.device)

        # Symmetric: anchor→positive and positive→anchor
        loss_a2p = self.cross_entropy(logits, labels)
        loss_p2a = self.cross_entropy(logits.T, labels)
        return (loss_a2p + loss_p2a) / 2.0


class SupervisedContrastiveLoss(nn.Module):
    """
    SupCon loss: all samples with the same item_id in a batch are positives.
    Useful when a batch contains multiple images per item_id.

    Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalized
            labels:     (B,) integer class labels (hashed item_ids)
        Returns:
            scalar loss
        """
        B = embeddings.size(0)
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature  # (B, B)

        # Mask: same label, exclude self
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0.0)

        # Log-softmax over all non-self pairs
        exp_sim = torch.exp(sim)
        exp_sim_sum = exp_sim.sum(dim=1, keepdim=True) - torch.diag(exp_sim).unsqueeze(1)
        log_prob = sim - torch.log(exp_sim_sum + 1e-8)

        # Mean log-prob over positives
        n_positives = pos_mask.sum(dim=1)
        # Avoid division by zero for samples with no positives in batch
        valid = n_positives > 0
        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        loss = -(pos_mask * log_prob).sum(dim=1)
        loss = loss[valid] / n_positives[valid]
        return loss.mean()


def build_loss(cfg) -> nn.Module:
    """Factory: return the configured loss module."""
    loss_name = cfg.train.loss.lower()
    temperature = cfg.train.temperature

    if loss_name == "infonce":
        return InfoNCELoss(temperature=temperature)
    elif loss_name == "supcon":
        return SupervisedContrastiveLoss(temperature=temperature)
    elif loss_name == "triplet":
        return nn.TripletMarginLoss(margin=cfg.train.triplet_margin, p=2)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
