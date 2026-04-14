"""
src/utils/dataset.py
--------------------
DeepFashion In-Shop Clothes Retrieval dataset loader.

Partition file format (list_eval_partition.txt):
    Row 1: number of images
    Row 2: column headers  (image_name  item_id  evaluation_status)
    Rows 3+: <image_name> <item_id> <train|query|gallery>

Usage:
    ds = DeepFashionDataset(cfg, split="train")
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import clip  # openai/CLIP


# ------------------------------------------------------------------ #
#  Partition reader                                                    #
# ------------------------------------------------------------------ #

def load_partition(partition_file: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns a dict with keys 'train', 'query', 'gallery'.
    Each value is a list of (image_path_relative, item_id) tuples.
    """
    splits: Dict[str, List[Tuple[str, str]]] = {"train": [], "query": [], "gallery": []}

    with open(partition_file, "r") as f:
        lines = f.read().splitlines()

    # Skip first two rows (count + header)
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        img_name, item_id, status = parts[0], parts[1], parts[2]
        if status in splits:
            splits[status].append((img_name, item_id))

    return splits


# ------------------------------------------------------------------ #
#  Core Dataset                                                        #
# ------------------------------------------------------------------ #

class DeepFashionDataset(Dataset):
    """
    Loads images for a given split (train / query / gallery).

    For training, returns (anchor_img, positive_img, item_id) triplets
    sampled from the same item_id group.

    For query/gallery, returns (image_tensor, item_id, img_path).
    """

    def __init__(self, cfg, split: str = "train", transform=None):
        """
        Args:
            cfg: OmegaConf config object.
            split: one of 'train', 'query', 'gallery'.
            transform: torchvision / CLIP preprocess transform.
        """
        assert split in ("train", "query", "gallery"), \
            f"split must be 'train', 'query', or 'gallery', got '{split}'"

        self.split = split
        self.img_dir = Path(cfg.paths.img_dir)
        self.transform = transform

        # Load partition
        partition = load_partition(cfg.paths.partition_file)
        self.samples: List[Tuple[str, str]] = partition[split]  # (img_rel_path, item_id)

        # Build item_id -> [indices] map (needed for positive sampling in train)
        self.item_to_indices: Dict[str, List[int]] = {}
        for idx, (_, item_id) in enumerate(self.samples):
            self.item_to_indices.setdefault(item_id, []).append(idx)

        self.item_ids = [s[1] for s in self.samples]
        self.img_paths = [s[0] for s in self.samples]

        print(f"[DeepFashionDataset] split={split}  samples={len(self.samples)}  "
              f"unique_items={len(self.item_to_indices)}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, rel_path: str) -> Image.Image:
        full_path = self.img_dir / rel_path
        img = Image.open(full_path).convert("RGB")
        return img

    def __getitem__(self, idx: int):
        img_path, item_id = self.samples[idx]
        img = self._load_image(img_path)

        if self.split == "train":
            # Sample a positive (different image, same item_id)
            pos_indices = self.item_to_indices[item_id]
            if len(pos_indices) > 1:
                pos_idx = idx
                while pos_idx == idx:
                    pos_idx = np.random.choice(pos_indices)
            else:
                pos_idx = idx  # degenerate; single image per item

            pos_path, _ = self.samples[pos_idx]
            pos_img = self._load_image(pos_path)

            if self.transform:
                img = self.transform(img)
                pos_img = self.transform(pos_img)

            return img, pos_img, item_id

        else:
            if self.transform:
                img = self.transform(img)
            return img, item_id, img_path


# ------------------------------------------------------------------ #
#  Dataloader factory                                                  #
# ------------------------------------------------------------------ #

def build_dataloader(cfg, split: str, clip_model_name: Optional[str] = None) -> DataLoader:
    """
    Build a DataLoader for the given split.
    If clip_model_name is provided, uses the CLIP preprocessing transform.
    """
    if clip_model_name:
        _, preprocess = clip.load(clip_model_name, device="cpu")
        transform = preprocess
    else:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    dataset = DeepFashionDataset(cfg, split=split, transform=transform)

    shuffle = (split == "train")
    batch_size = cfg.train.batch_size if split == "train" else cfg.eval.batch_size

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=(split == "train"),
    )
    return loader
