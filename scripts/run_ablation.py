"""
scripts/run_ablation.py
------------------------
Automates the full ablation study across conditions A, B, C and the
configured seeds (team roll numbers).

Condition definitions:
    A — Vision-only CLIP, frozen, α=1.0
    B — Frozen CLIP + frozen BLIP-2, α ∈ {0.5, 0.7}
    C — Fine-tuned CLIP + frozen BLIP-2, α ∈ {0.5, 0.7}

For each (condition, alpha, seed) triplet this script:
    1. (Condition C only) Trains CLIP via train_clip.py
    2. Builds the HNSW index via build_index.py
    3. Evaluates via evaluate.py
    4. Aggregates results and writes a summary table

Run:
    python scripts/run_ablation.py --config configs/config.yaml
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.utils.helpers import load_config, get_logger, save_results


def parse_args():
    p = argparse.ArgumentParser(description="Run full ablation study")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--conditions", nargs="+", default=["A", "B", "C"],
                   choices=["A", "B", "C"],
                   help="Which ablation conditions to run")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing them")
    return p.parse_args()


def run_cmd(cmd: List[str], dry_run: bool, logger) -> int:
    logger.info("CMD: " + " ".join(cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, check=False)
    return result.returncode


def load_metrics(path: str) -> Dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("metrics", {})
    except Exception:
        return {}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("ablation")

    seeds = cfg.ablation.seeds
    alphas = cfg.ablation.alpha_values   # e.g. [0.5, 0.7]

    all_results = {}   # key: (condition, alpha, seed) → metrics dict

    for condition in args.conditions:
        logger.info(f"\n{'='*60}")
        logger.info(f"  CONDITION {condition}")
        logger.info(f"{'='*60}")

        # For condition A, alpha is always 1.0 and no training is needed
        alpha_list = [1.0] if condition == "A" else alphas

        for alpha in alpha_list:
            for seed in seeds:
                label = f"cond{condition}_alpha{alpha}_seed{seed}"
                logger.info(f"\n--- {label} ---")

                index_path = f"{cfg.paths.index_dir}/hnsw_{label}.bin"
                meta_path  = f"{cfg.paths.index_dir}/metadata_{label}.json"
                result_path = f"{cfg.paths.results_dir}/{label}_metrics.json"
                ckpt_path  = f"{cfg.paths.checkpoint_dir}/best_model_seed{seed}.pt"

                # ---- Step 1: Train (Condition C only) ----
                if condition == "C":
                    train_cmd = [
                        "python", "scripts/train_clip.py",
                        "--config", args.config,
                        "--seed", str(seed),
                    ]
                    rc = run_cmd(train_cmd, args.dry_run, logger)
                    if rc != 0:
                        logger.error(f"Training failed for {label}")
                        continue

                # ---- Step 2: Build index ----
                build_cmd = [
                    "python", "scripts/build_index.py",
                    "--config", args.config,
                    "--alpha", str(alpha),
                    "--suffix", label,
                ]
                if condition == "A":
                    build_cmd += ["--no_blip2"]
                if condition == "C" and Path(ckpt_path).exists():
                    build_cmd += ["--clip_checkpoint", ckpt_path]

                rc = run_cmd(build_cmd, args.dry_run, logger)
                if rc != 0:
                    logger.error(f"Index build failed for {label}")
                    continue

                # ---- Step 3: Evaluate ----
                eval_cmd = [
                    "python", "scripts/evaluate.py",
                    "--config", args.config,
                    "--index_path", index_path,
                    "--meta_path", meta_path,
                    "--alpha", str(alpha),
                    "--output_suffix", label,
                ]
                if condition == "C" and Path(ckpt_path).exists():
                    eval_cmd += ["--clip_checkpoint", ckpt_path]
                if condition == "A":
                    eval_cmd += ["--no_rerank"]

                rc = run_cmd(eval_cmd, args.dry_run, logger)
                if rc != 0:
                    logger.error(f"Evaluation failed for {label}")
                    continue

                # ---- Collect metrics ----
                if not args.dry_run:
                    metrics = load_metrics(result_path)
                    all_results[(condition, alpha, seed)] = metrics
                    logger.info(f"  Metrics: {metrics}")

    # ---- Aggregate: mean ± std across seeds ----
    if not args.dry_run:
        logger.info("\n" + "=" * 70)
        logger.info("ABLATION SUMMARY (mean ± std over seeds)")
        logger.info("=" * 70)

        metric_keys = ["recall@5", "recall@10", "recall@15",
                       "ndcg@5",   "ndcg@10",   "ndcg@15",
                       "map@5",    "map@10",     "map@15"]

        summary = {}
        for condition in args.conditions:
            alpha_list = [1.0] if condition == "A" else alphas
            for alpha in alpha_list:
                key = f"cond{condition}_alpha{alpha}"
                seed_metrics = [
                    all_results.get((condition, alpha, s), {})
                    for s in seeds
                ]
                agg = {}
                for mk in metric_keys:
                    vals = [m[mk] for m in seed_metrics if mk in m]
                    if vals:
                        agg[mk] = {
                            "mean": float(np.mean(vals)),
                            "std":  float(np.std(vals)),
                        }
                summary[key] = agg

                logger.info(f"\n{key}:")
                for mk in metric_keys:
                    if mk in agg:
                        logger.info(
                            f"  {mk:<12}  {agg[mk]['mean']:.4f} ± {agg[mk]['std']:.4f}"
                        )

        save_results(summary, f"{cfg.paths.results_dir}/ablation_summary.json")
        logger.info(f"\nSummary saved → {cfg.paths.results_dir}/ablation_summary.json")


if __name__ == "__main__":
    main()
