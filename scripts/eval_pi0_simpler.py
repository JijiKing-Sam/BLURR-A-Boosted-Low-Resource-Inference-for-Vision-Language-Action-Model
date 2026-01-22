#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blurr.imports import ensure_open_pi_zero_on_path
from blurr.paths import open_pi_zero_root, repo_root


def _apply_preset(cfg, preset: str) -> None:
    preset = preset.lower().strip()
    with open_dict(cfg):
        cfg.use_prefix_kv_cache = cfg.get("use_prefix_kv_cache", True)
        if preset in {"vanilla", "baseline"}:
            cfg.use_prefix_kv_cache = False
            cfg.use_bf16 = False
            cfg.use_torch_compile = False
            cfg.num_inference_steps = 10
        elif preset in {"prefix_cache", "cached"}:
            cfg.use_prefix_kv_cache = True
            cfg.use_bf16 = False
            cfg.use_torch_compile = False
            cfg.num_inference_steps = 10
        elif preset in {"blurr", "blurr_step1", "step1"}:
            cfg.use_prefix_kv_cache = True
            cfg.use_bf16 = True
            cfg.use_torch_compile = True
            cfg.num_inference_steps = 1
        else:
            raise ValueError(f"Unknown preset: {preset}")


def _default_log_dir(*, preset: str | None, task: str, seed: int) -> Path:
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    name = preset or "custom"
    return repo_root() / "runs" / "eval_bridge" / f"{name}_{seed}" / f"{task}_{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BLURR Pi0 evaluation in SimplerEnv (Bridge/Fractal tasks)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/eval/bridge.yaml",
        help="Path to an open-pi-zero eval config, relative to open_pi_zero_root.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="blurr",
        choices=["baseline", "vanilla", "prefix_cache", "blurr", "blurr_step1", "step1"],
        help="Named preset for toggles (prefix KV cache / BF16 / compile / steps).",
    )
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--n-eval-episode", type=int, default=240)
    parser.add_argument("--n-video", type=int, default=0)
    parser.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="Override output directory. Default: runs/eval_bridge/<preset>_<seed>/<task>_<timestamp>/",
    )

    # manual overrides (optional; preset applies first)
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--no-torch-compile", action="store_true")
    parser.add_argument("--num-inference-steps", type=int, default=0)
    parser.add_argument("--act-steps", type=int, default=0)
    parser.add_argument("--no-prefix-kv-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_open_pi_zero_on_path()
    from src.agent.eval import EvalAgent  # noqa: E402

    root = open_pi_zero_root()
    os.chdir(root)

    cfg = OmegaConf.load(args.config)
    _apply_preset(cfg, args.preset)

    log_dir = (
        Path(args.log_dir).expanduser()
        if args.log_dir
        else _default_log_dir(preset=args.preset, task=args.task, seed=args.seed)
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "run.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    with open_dict(cfg):
        cfg.env.task = args.task
        cfg.checkpoint_path = str(Path(args.checkpoint).expanduser())
        cfg.seed = args.seed
        cfg.gpu_id = args.gpu_id
        cfg.n_eval_episode = args.n_eval_episode
        cfg.n_video = args.n_video
        cfg.log_dir = str(log_dir)
        cfg.use_prefix_kv_cache = cfg.get("use_prefix_kv_cache", True)

        # manual overrides
        if args.use_bf16:
            cfg.use_bf16 = True
        if args.no_torch_compile:
            cfg.use_torch_compile = False
        if args.num_inference_steps > 0:
            cfg.num_inference_steps = args.num_inference_steps
        if args.act_steps > 0:
            cfg.act_steps = args.act_steps
        if args.no_prefix_kv_cache:
            cfg.use_prefix_kv_cache = False

    agent = EvalAgent(cfg)
    agent.run()

    print(f"\nDone. Logs written to: {log_dir}\n")


if __name__ == "__main__":
    main()
