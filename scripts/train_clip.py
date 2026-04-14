"""
scripts/train_clip.py
----------------------
Fine-tune the CLIP vision encoder on DeepFashion In-Shop using
contrastive metric learning (InfoNCE by default).

Run:
    python scripts/train_clip.py --config configs/config.yaml [--seed 42]

What is trained:
    - CLIP vision encoder (last N blocks + ln_post + proj)
    - CLIP text encoder stays FROZEN

Training objective:
    InfoNCE loss on (anchor, positive) image pairs sharing the same item_id.
    The loss pulls same-item embeddings together and pushes others apart.
"""

import argparse
import os
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.models.clip_model import CLIPFineTuner, build_loss
from src.utils.dataset import build_dataloader
from src.utils.helpers import (
    get_device, get_logger, load_config,
    save_checkpoint, save_results, set_seed,
)
from src.utils.metrics import evaluate_retrieval, format_metrics


# ------------------------------------------------------------------ #
#  Argument parsing                                                    #
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on DeepFashion")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed (use team roll number)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Training loop                                                       #
# ------------------------------------------------------------------ #

def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, cfg, logger, epoch
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", leave=False)):
        anchor_imgs, positive_imgs, item_ids = batch
        anchor_imgs = anchor_imgs.to(device)
        positive_imgs = positive_imgs.to(device)

        optimizer.zero_grad()

        with autocast(enabled=cfg.train.use_amp):
            anchor_emb = model.encode_image(anchor_imgs)     # (B, D)
            positive_emb = model.encode_image(positive_imgs) # (B, D)

            if cfg.train.loss == "infonce":
                loss = criterion(anchor_emb, positive_emb)
            elif cfg.train.loss == "supcon":
                # Stack both views; build integer label tensor
                import clip
                all_embs = torch.cat([anchor_emb, positive_emb], dim=0)
                # Map item_id strings → integer labels
                unique_ids = {iid: i for i, iid in enumerate(set(item_ids))}
                labels_list = [unique_ids[iid] for iid in item_ids] * 2
                labels = torch.tensor(labels_list, device=device)
                loss = criterion(all_embs, labels)
            else:
                raise ValueError(f"Unsupported loss: {cfg.train.loss}")

        if cfg.train.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.grad_clip
            )
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (step + 1) % cfg.logging.log_every_n_steps == 0:
            logger.info(
                f"Epoch {epoch}  step {step+1}/{len(loader)}  "
                f"loss={total_loss / n_batches:.4f}"
            )

    return total_loss / max(n_batches, 1)


# ------------------------------------------------------------------ #
#  Quick retrieval eval on query/gallery split                         #
# ------------------------------------------------------------------ #

@torch.no_grad()
def quick_eval(model, cfg, device, logger):
    """
    Encode query and gallery splits with the current model weights
    and compute Recall@10 as an early-stopping signal.
    """
    import numpy as np
    from src.utils.dataset import DeepFashionDataset
    import clip
    from torch.utils.data import DataLoader

    _, preprocess = clip.load(cfg.clip.model_name, device="cpu")

    query_ds = DeepFashionDataset(cfg, split="query", transform=preprocess)
    gallery_ds = DeepFashionDataset(cfg, split="gallery", transform=preprocess)

    def encode_split(ds):
        dl = DataLoader(ds, batch_size=cfg.eval.batch_size, shuffle=False,
                        num_workers=cfg.dataset.num_workers)
        embeddings, ids = [], []
        model.eval()
        for imgs, item_ids, _ in tqdm(dl, desc="Encoding", leave=False):
            imgs = imgs.to(device)
            embs = model.encode_image(imgs)  # (B, D)
            embeddings.append(embs.cpu().numpy())
            ids.extend(list(item_ids))
        return np.concatenate(embeddings, axis=0), ids

    query_embs, query_ids = encode_split(query_ds)
    gallery_embs, gallery_ids = encode_split(gallery_ds)

    # Cosine similarity matrix: (Q, G)
    sim_matrix = query_embs @ gallery_embs.T
    ranked = np.argsort(-sim_matrix, axis=1)  # (Q, G) descending

    metrics = evaluate_retrieval(query_ids, gallery_ids, ranked, cfg.eval.K_values)
    logger.info("\n" + format_metrics(metrics, cfg.eval.K_values))
    return metrics


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else cfg.ablation.seeds[0]
    set_seed(seed)

    device = get_device()
    logger = get_logger("train", log_file=f"{cfg.paths.results_dir}/train.log")
    logger.info(f"Config: {args.config}  Seed: {seed}")

    # Model
    model = CLIPFineTuner(
        model_name=cfg.clip.model_name,
        unfreeze_vision_blocks=cfg.clip.unfreeze_vision_blocks,
        freeze_text_encoder=cfg.clip.freeze_text_encoder,
        device=device,
    ).to(device)

    # Loss
    criterion = build_loss(cfg)

    # Optimizer (only update trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params,
                      lr=cfg.train.learning_rate,
                      weight_decay=cfg.train.weight_decay)

    # LR schedule: linear warmup → cosine decay
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=cfg.train.warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epochs,
        eta_min=1e-7,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.train.warmup_steps],
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=cfg.train.use_amp)

    # Resume from checkpoint
    start_epoch = 1
    best_metric = 0.0
    if args.resume:
        from src.utils.helpers import load_checkpoint
        state = load_checkpoint(model, args.resume, optimizer, device)
        start_epoch = state.get("epoch", 0) + 1

    # DataLoader
    train_loader = build_dataloader(cfg, split="train",
                                    clip_model_name=cfg.clip.model_name)

    logger.info(f"Starting training for {cfg.train.epochs} epochs "
                f"with loss={cfg.train.loss}, α_temp={cfg.train.temperature}")

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, cfg, logger, epoch
        )
        scheduler.step()
        logger.info(f"Epoch {epoch} done. avg_loss={avg_loss:.4f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Evaluate periodically
        metrics = {}
        if epoch % cfg.train.save_every_n_epochs == 0 or epoch == cfg.train.epochs:
            metrics = quick_eval(model, cfg, device, logger)
            current_best_metric = metrics.get(cfg.train.best_metric, 0.0)
            is_best = current_best_metric > best_metric
            if is_best:
                best_metric = current_best_metric
                logger.info(f"New best {cfg.train.best_metric}: {best_metric:.4f}")
        else:
            is_best = False

        # Save checkpoint
        ckpt_path = (f"{cfg.paths.checkpoint_dir}/epoch_{epoch:03d}_"
                     f"seed{seed}.pt")
        save_checkpoint(model, optimizer, epoch, metrics, ckpt_path, is_best)

    logger.info(f"Training complete. Best {cfg.train.best_metric}: {best_metric:.4f}")
    save_results({"best_metric": best_metric, "seed": seed},
                 f"{cfg.paths.results_dir}/train_summary_seed{seed}.json")


if __name__ == "__main__":
    main()
