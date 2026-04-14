"""
src/utils/helpers.py
---------------------
Miscellaneous helpers: seeding, image I/O, config loading, logging.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import OmegaConf


# ------------------------------------------------------------------ #
#  Seeding                                                             #
# ------------------------------------------------------------------ #

def set_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] All seeds set to {seed}")


# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #

def load_config(config_path: str) -> Any:
    """Load YAML config with OmegaConf."""
    cfg = OmegaConf.load(config_path)
    return cfg


def save_config(cfg: Any, save_path: str) -> None:
    """Save config to YAML."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_path)


# ------------------------------------------------------------------ #
#  Logging                                                             #
# ------------------------------------------------------------------ #

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ------------------------------------------------------------------ #
#  Device                                                              #
# ------------------------------------------------------------------ #

def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


# ------------------------------------------------------------------ #
#  Checkpoint helpers                                                  #
# ------------------------------------------------------------------ #

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    is_best: bool = False,
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, save_path)
    if is_best:
        best_path = str(Path(save_path).parent / "best_model.pt")
        torch.save(state, best_path)
        print(f"[Checkpoint] Best model saved → {best_path}")
    print(f"[Checkpoint] Saved epoch {epoch} → {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    device = device or get_device()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    print(f"[Checkpoint] Loaded from {checkpoint_path}  (epoch={state.get('epoch')})")
    return state


# ------------------------------------------------------------------ #
#  Results I/O                                                         #
# ------------------------------------------------------------------ #

def save_results(results: Dict, save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved → {save_path}")


def load_results(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)
