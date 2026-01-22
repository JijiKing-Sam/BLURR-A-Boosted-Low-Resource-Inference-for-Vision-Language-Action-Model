#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blurr.imports import ensure_open_pi_zero_on_path
from blurr.paths import open_pi_zero_root, repo_root


def _default_log_dir(*, tag: str, seed: int, task: Optional[str] = None) -> Path:
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    base = repo_root() / "runs" / "eval_bridge" / f"{tag}_{seed}"
    if task:
        return base / f"{task}_{stamp}"
    return base / stamp


def _load_dataset_stats(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "dataset_statistics.json"
    if not path.is_file():
        return {}
    try:
        return json.load(open(path))
    except Exception:
        return {}


def _infer_unnorm_key(*, norm_stats: Dict[str, Any], dataset_stats: Dict[str, Any], override: Optional[str]) -> str:
    if override is not None:
        return override

    norm_keys = list(norm_stats.keys())
    if dataset_stats:
        intersection = [k for k in dataset_stats.keys() if k in norm_keys]
        if intersection:
            return intersection[0]
    if "bridge_dataset" in norm_keys:
        return "bridge_dataset"
    if norm_keys:
        return norm_keys[0]
    return "bridge_dataset"


def _set_cuda_fastpaths() -> None:
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _binarize_gripper_action(value: float, action_stats: Optional[Dict[str, Any]] = None) -> float:
    mask = None
    if action_stats:
        mask = action_stats.get("mask")

    if isinstance(mask, (list, tuple)) and len(mask) >= 7:
        if not bool(mask[6]):
            return -1.0 if value <= 0.0 else 1.0
        sign = float(np.sign(2.0 * value - 1.0))
        return -1.0 if sign == 0.0 else sign

    if 0.0 <= value <= 1.0:
        return -1.0 if value <= 0.5 else 1.0
    return -1.0 if value <= 0.0 else 1.0


def _bridge_action_to_simpler(action: np.ndarray, *, action_stats: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Convert a Bridge-style 7-DoF action into SimplerEnv's 7D control action:
      [dx, dy, dz, rotvec(xyz), gripper]

    Bridge action format assumed:
      [dx, dy, dz, roll, pitch, yaw, gripper_open]
    """
    ensure_open_pi_zero_on_path()
    from src.utils.geometry import euler2axangle  # noqa: E402

    if torch.is_tensor(action):
        action = action.detach().to("cpu").float().numpy()
    try:
        a = np.asarray(action, dtype=np.float32)
    except TypeError:
        if hasattr(action, "detach"):
            action = action.detach().to("cpu").float().numpy()
            a = np.asarray(action, dtype=np.float32)
        else:
            raise
    if a.ndim == 2 and a.shape[1] == 7:
        a = a[0]
    elif a.ndim == 2 and a.shape[0] == 7:
        a = a[:, 0]
    a = a.reshape(-1)
    if a.shape[0] != 7:
        raise ValueError(f"Expected 7D action, got shape {a.shape}")

    dx, dy, dz = a[:3]
    roll, pitch, yaw = a[3:6]
    axis, angle = euler2axangle(float(roll), float(pitch), float(yaw), axes="sxyz")
    rotvec = np.asarray(axis, dtype=np.float32) * float(angle)

    gripper = _binarize_gripper_action(float(a[6]), action_stats)

    out = np.zeros((7,), dtype=np.float32)
    out[:3] = [dx, dy, dz]
    out[3:6] = rotvec
    out[6] = gripper
    return out


def _extract_rgb(env: Any, obs: Dict[str, Any]) -> Image.Image:
    from simpler_env.utils.env.observation_utils import (  # noqa: E402
        get_image_from_maniskill2_obs_dict,
    )

    rgb = get_image_from_maniskill2_obs_dict(env, obs)
    if not isinstance(rgb, np.ndarray):
        rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)

    image = Image.fromarray(rgb)

    # Match the OpenVLA/Bridge preprocessing used in common evaluation scripts:
    # JPEG round-trip + resize to 128 then to 224 (LANCZOS).
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")

    try:
        resample = Image.Resampling.LANCZOS  # Pillow>=9
    except AttributeError:  # pragma: no cover
        resample = Image.LANCZOS

    image = image.resize((128, 128), resample=resample)
    image = image.resize((224, 224), resample=resample)
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Prismatic/OpenVLA-format .pt checkpoint (e.g., MiniVLA prismatic) on SimplerEnv."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint under run_dir/checkpoints/")
    parser.add_argument(
        "--openvla-mini-root",
        type=str,
        default="",
        help="Path to a cloned openvla-mini repo that provides `prismatic.models.load.load_vla`.",
    )
    parser.add_argument("--task", type=str, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--n-eval-episode", type=int, default=100)
    parser.add_argument(
        "--initial-states",
        type=str,
        default="eval",
        choices=["eval", "train", "episode_id"],
        help="eval: reset(seed=1000+episode_id); train: reset(seed=episode_id); episode_id: reset(options.obj_init_options.episode_id).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Hard cap on control steps per episode (episodes may end earlier due to gymnasium TimeLimit).",
    )
    parser.add_argument(
        "--num-steps-wait",
        type=int,
        default=0,
        help="Optional number of no-op steps at episode start to let objects settle.",
    )
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--unnorm-key", type=str, default=None)
    parser.add_argument(
        "--instruction-template",
        type=str,
        default="{instruction}",
        help="Instruction template passed to the prismatic model; must contain '{instruction}'.",
    )
    return parser.parse_args()


def _resolve_openvla_mini_root(arg: str) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    # Default: sibling of workspace root (repo root's parent).
    candidate = repo_root().parent / "openvla-mini"
    return candidate.resolve()


def main() -> None:
    args = parse_args()

    if "{instruction}" not in args.instruction_template:
        raise ValueError("--instruction-template must contain '{instruction}'")

    ensure_open_pi_zero_on_path()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. Run this on a GPU node.")

    openvla_mini_root = _resolve_openvla_mini_root(args.openvla_mini_root)
    if not openvla_mini_root.is_dir():
        raise FileNotFoundError(
            f"openvla-mini repo not found at: {openvla_mini_root}. "
            "Clone it or pass --openvla-mini-root."
        )

    # Prismatic MiniVLA checkpoints (VQ action tokenizer) expect a `vq/` folder relative to the
    # working directory. Use the openvla-mini repo root as the working directory.
    os.chdir(openvla_mini_root)

    # openvla-mini's prismatic package expects this env var during import (even for inference).
    os.environ.setdefault("PRISMATIC_DATA_ROOT", str(repo_root().parent))

    # Ensure openvla-mini's `prismatic/` is imported (not the workspace placeholder one).
    sys.path.insert(0, str(openvla_mini_root))

    from prismatic.models.load import load_vla  # noqa: E402

    # Keep the user-provided path (do not resolve symlinks): HF caches often symlink
    # `.../checkpoints/*.pt` to blob files without a `.pt` suffix.
    checkpoint_pt = Path(args.checkpoint).expanduser()
    if not checkpoint_pt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_pt}")
    if checkpoint_pt.suffix != ".pt":
        raise ValueError(f"--checkpoint must be a .pt file, got: {checkpoint_pt}")

    run_dir = checkpoint_pt.parents[1] if checkpoint_pt.parent.name == "checkpoints" else checkpoint_pt.parent
    dataset_stats = _load_dataset_stats(run_dir)

    tasks = list(args.task)
    log_dir = (
        Path(args.log_dir).expanduser()
        if args.log_dir
        else _default_log_dir(tag="prismatic", seed=args.seed, task=tasks[0] if len(tasks) == 1 else None)
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
    log = logging.getLogger("eval_prismatic_vla_simpler")

    device = torch.device(f"cuda:{args.gpu_id}")
    _set_cuda_fastpaths()

    log.info("Loading prismatic checkpoint: %s", checkpoint_pt)
    vla = load_vla(str(checkpoint_pt), hf_token=None, load_for_training=False)
    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(device)
    vla.eval()

    norm_stats = getattr(vla, "norm_stats", {}) or {}
    unnorm_key = _infer_unnorm_key(norm_stats=norm_stats, dataset_stats=dataset_stats, override=args.unnorm_key)
    log.info("Using device=%s unnorm_key=%s", device, unnorm_key)
    action_stats = norm_stats.get(unnorm_key, {}).get("action", {}) if unnorm_key else {}

    import simpler_env  # noqa: E402

    np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    dummy_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)

    per_task: Dict[str, float] = {}
    for task in tasks:
        log.info("Creating SimplerEnv env with task='%s'...", task)
        env = simpler_env.make(task)

        successes: list[bool] = []
        for episode_id in range(args.n_eval_episode):
            if args.initial_states == "episode_id":
                reset_options = {"obj_init_options": {"episode_id": episode_id}}
                obs, reset_info = env.reset(options=reset_options)
            else:
                seed_base = 1000 if args.initial_states == "eval" else 0
                obs, reset_info = env.reset(seed=seed_base + episode_id)
            instruction = env.get_language_instruction()
            log.info(
                "[%s] Episode %d reset: instruction='%s' max_steps=%s reset_info=%s",
                task,
                episode_id,
                instruction,
                getattr(env.spec, "max_episode_steps", None),
                reset_info,
            )

            step_in_episode = 0
            terminated = False
            truncated = False

            while step_in_episode < args.max_steps + args.num_steps_wait and not (terminated or truncated):
                if step_in_episode < args.num_steps_wait:
                    obs, reward, terminated, truncated, info = env.step(dummy_action)
                    step_in_episode += 1
                    continue

                image = _extract_rgb(env, obs)
                prompt = args.instruction_template.format(instruction=instruction)
                with torch.inference_mode():
                    action = vla.predict_action(image, prompt, unnorm_key=unnorm_key, do_sample=False)
                env_action = _bridge_action_to_simpler(action, action_stats=action_stats)

                obs, reward, terminated, truncated, info = env.step(env_action)
                step_in_episode += 1

                new_instruction = env.get_language_instruction()
                if new_instruction != instruction:
                    instruction = new_instruction

                if terminated or truncated:
                    break

            successes.append(bool(terminated))
            log.info(
                "[%s] Episode %d finished: success=%s steps=%d truncated=%s",
                task,
                episode_id,
                terminated,
                step_in_episode,
                truncated,
            )

        rate = float(np.mean(successes)) if successes else 0.0
        per_task[task] = rate
        log.info("[%s] Success rate: %.4f over %d episodes", task, rate, len(successes))

    avg_success = float(np.mean(list(per_task.values()))) if per_task else 0.0
    summary = {
        "checkpoint": str(checkpoint_pt),
        "unnorm_key": unnorm_key,
        "seed": int(args.seed),
        "initial_states": args.initial_states,
        "max_steps": int(args.max_steps),
        "num_steps_wait": int(args.num_steps_wait),
        "episodes_per_task": int(args.n_eval_episode),
        "per_task_success": per_task,
        "avg_success": avg_success,
    }
    with open(log_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    log.info("============ Evaluation Summary ============")
    log.info("Tasks: %s", ", ".join(tasks))
    log.info("Episodes per task: %d", args.n_eval_episode)
    log.info("Avg success: %.4f", avg_success)
    log.info("Wrote: %s", log_dir / "summary.json")
    log.info("============================================")

    print(f"\nDone. Logs written to: {log_dir}\n")


if __name__ == "__main__":
    main()
