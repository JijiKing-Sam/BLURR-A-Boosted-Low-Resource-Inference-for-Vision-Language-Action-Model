#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from blurr.imports import ensure_open_pi_zero_on_path
from blurr.paths import open_pi_zero_root, repo_root


class _OFTResNetBlock(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ffn = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)


class _OFTActionHead(torch.nn.Module):
    def __init__(self, *, in_dim: int, hidden_dim: int, out_dim: int, n_blocks: int) -> None:
        super().__init__()
        self.layer_norm1 = torch.nn.LayerNorm(in_dim)
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.mlp_resnet_blocks = torch.nn.ModuleList([_OFTResNetBlock(hidden_dim) for _ in range(n_blocks)])
        self.layer_norm2 = torch.nn.LayerNorm(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm1(x)
        x = self.fc1(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        return self.fc2(x)

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(actions_hidden_states):
            actions_hidden_states = torch.as_tensor(actions_hidden_states)
        flat = actions_hidden_states.reshape(actions_hidden_states.shape[0], -1)
        out = self.forward(flat)
        return out.reshape(-1)


class _OFTProprioProjector(torch.nn.Module):
    def __init__(self, *, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, out_dim)
        self.act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _load_oft_aux_modules(
    model_id: str, *, device: torch.device, dtype: torch.dtype
) -> tuple[Optional[_OFTActionHead], Optional[_OFTProprioProjector]]:
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None, None

    try:
        snapshot_dir = Path(hf_hub_download(model_id, "config.json", repo_type="model")).parent
    except Exception:
        return None, None

    action_head_files = sorted(snapshot_dir.glob("action_head--*_checkpoint.pt"))
    proprio_files = sorted(snapshot_dir.glob("proprio_projector--*_checkpoint.pt"))
    if not action_head_files and not proprio_files:
        return None, None

    cast_dtype = dtype if dtype in {torch.float16, torch.bfloat16} else torch.float32

    action_head: Optional[_OFTActionHead] = None
    proprio_projector: Optional[_OFTProprioProjector] = None

    if action_head_files:
        state = torch.load(action_head_files[-1], map_location="cpu")
        fc1_key = "module.model.fc1.weight"
        fc2_key = "module.model.fc2.weight"
        if fc1_key in state and fc2_key in state:
            hidden_dim, in_dim = state[fc1_key].shape
            out_dim = state[fc2_key].shape[0]
            block_indices: set[int] = set()
            for key in state.keys():
                if key.startswith("module.model.mlp_resnet_blocks."):
                    parts = key.split(".")
                    if len(parts) > 3 and parts[3].isdigit():
                        block_indices.add(int(parts[3]))
            n_blocks = len(block_indices)
            action_head = _OFTActionHead(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_blocks=n_blocks)
            stripped = {k.removeprefix("module.model."): v for k, v in state.items() if k.startswith("module.model.")}
            action_head.load_state_dict(stripped, strict=True)
            action_head.to(device=device, dtype=cast_dtype)
            action_head.eval()

    if proprio_files:
        state = torch.load(proprio_files[-1], map_location="cpu")
        fc1_key = "module.fc1.weight"
        fc2_key = "module.fc2.weight"
        if fc1_key in state and fc2_key in state:
            hidden_dim, in_dim = state[fc1_key].shape
            out_dim = state[fc2_key].shape[0]
            proprio_projector = _OFTProprioProjector(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
            stripped = {k.removeprefix("module."): v for k, v in state.items() if k.startswith("module.")}
            proprio_projector.load_state_dict(stripped, strict=True)
            proprio_projector.to(device=device, dtype=cast_dtype)
            proprio_projector.eval()

    return action_head, proprio_projector


def _maybe_load_lora_adapter(model: Any, model_id: str, *, device: torch.device) -> Any:
    """
    If the HF repo contains a PEFT LoRA adapter in `lora_adapter/`, load it and (if possible) merge it.
    """
    log = logging.getLogger("eval_hf_vla_simpler")

    try:
        from huggingface_hub import hf_hub_download
        from peft import PeftModel
    except Exception:
        return model

    try:
        adapter_cfg = hf_hub_download(model_id, "lora_adapter/adapter_config.json", repo_type="model")
    except Exception:
        return model

    adapter_dir = Path(adapter_cfg).parent

    # Ensure adapter weights are present locally. Some environments default to "partial" snapshots where
    # only requested files exist in the cache, so we explicitly pull the common weight filenames.
    for weight_relpath in ("lora_adapter/adapter_model.safetensors", "lora_adapter/adapter_model.bin"):
        try:
            hf_hub_download(model_id, weight_relpath, repo_type="model")
            break
        except Exception:
            continue
    try:
        hf_hub_download(model_id, "lora_adapter/README.md", repo_type="model")
    except Exception:
        pass
    try:
        log.info("Loading PEFT LoRA adapter from: %s", adapter_dir)
        peft_model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        peft_model = peft_model.to(device=device)
        try:
            merged = peft_model.merge_and_unload()
            log.info("Merged LoRA adapter into base model.")
            return merged
        except Exception:
            log.warning("LoRA merge_and_unload failed; keeping adapter attached.")
            return peft_model
    except Exception:
        log.warning("Failed to load PEFT LoRA adapter; running base model without adapter.", exc_info=True)
        return model


def _default_log_dir(*, tag: str, seed: int, task: Optional[str] = None) -> Path:
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    base = repo_root() / "runs" / "eval_bridge" / f"{tag}_{seed}"
    if task:
        return base / f"{task}_{stamp}"
    return base / stamp


def _load_dataset_stats(model_id_or_path: str) -> Dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_id_or_path, "dataset_statistics.json", repo_type="model")
        return json.load(open(path))
    except Exception:
        # Local path / missing stats: ignore.
        return {}


def _inject_dataset_statistics_into_norm_stats(model: Any, model_id_or_path: str) -> None:
    """
    Some community checkpoints ship correct per-dataset stats in `dataset_statistics.json` but have incomplete
    `config.norm_stats` (e.g., missing the dataset key used for unnormalization). We merge the dataset stats into
    `model.norm_stats` so `predict_action(..., unnorm_key=...)` can unnormalize with the intended ranges.
    """
    dataset_stats = _load_dataset_stats(model_id_or_path)
    if not dataset_stats:
        return

    norm_stats = getattr(model, "norm_stats", None)
    if not isinstance(norm_stats, dict):
        return

    def _is_all_zeros(stats: Any) -> bool:
        if not isinstance(stats, dict):
            return True
        for field in ("min", "max", "mean", "std", "q01", "q99"):
            vals = stats.get(field, None)
            if isinstance(vals, (list, tuple)) and any(float(x) != 0.0 for x in vals):
                return False
        return True

    updated = False

    # Special-case: some OFT Bridge checkpoints ship `bridge_orig` in config.norm_stats (with action mask)
    # but put the correct proprio stats under `bridge_dataset` in dataset_statistics.json. Merge them so we
    # preserve the action mask while enabling in-distribution proprio normalization.
    if "bridge_dataset" in dataset_stats and "bridge_orig" in norm_stats:
        ds_entry = dataset_stats.get("bridge_dataset", None)
        base_entry = norm_stats.get("bridge_orig", None)
        if isinstance(ds_entry, dict) and isinstance(base_entry, dict):
            ds_action = ds_entry.get("action", None)
            base_action = base_entry.get("action", None)
            if isinstance(ds_action, dict) and isinstance(base_action, dict):
                merged_action = dict(ds_action)
                merged_action.update(base_action)  # keep `mask` and any extra metadata
                base_entry["action"] = merged_action

            ds_proprio = ds_entry.get("proprio", None)
            base_proprio = base_entry.get("proprio", None)
            if isinstance(ds_proprio, dict) and _is_all_zeros(base_proprio):
                base_entry["proprio"] = ds_proprio
                updated = True
            elif isinstance(ds_proprio, dict) and isinstance(base_proprio, dict):
                for k, v in ds_proprio.items():
                    if k not in base_proprio:
                        base_proprio[k] = v
                        updated = True

            # Provide an alias key so users can pass `--unnorm-key bridge_dataset` as in upstream scripts.
            if "bridge_dataset" not in norm_stats:
                norm_stats["bridge_dataset"] = base_entry
                updated = True

    for key, value in dataset_stats.items():
        if key in norm_stats:
            continue
        if not isinstance(value, dict):
            continue
        if "action" not in value:
            continue
        norm_stats[key] = value
        updated = True

    if updated:
        setattr(model, "norm_stats", norm_stats)


def _infer_unnorm_key(*, model: Any, model_id: str, override: Optional[str]) -> Optional[str]:
    if override is not None:
        return override

    dataset_stats = _load_dataset_stats(model_id)
    norm_stats = getattr(model, "norm_stats", {}) or {}
    norm_keys = list(norm_stats.keys())

    # Prefer Bridge key when present; it typically includes the correct action mask.
    if "bridge_orig" in norm_keys:
        return "bridge_orig"

    if dataset_stats:
        intersection = [k for k in dataset_stats.keys() if k in norm_keys]
        if intersection:
            return intersection[0]
    if norm_keys:
        return norm_keys[0]
    return None


def _infer_center_crop_default(model_id_or_path: str) -> bool:
    """
    Upstream OpenVLA eval enables center-crop when the training run used image augmentations.
    On HuggingFace, `config._name_or_path` is often overwritten with the model id, so we inspect
    the raw `config.json` payload for a hint (e.g., contains 'image_aug').
    """
    cfg_path = None
    candidate = Path(model_id_or_path) / "config.json"
    if candidate.is_file():
        cfg_path = str(candidate)
    else:
        try:
            from huggingface_hub import hf_hub_download

            cfg_path = hf_hub_download(model_id_or_path, "config.json", repo_type="model")
        except Exception:
            cfg_path = None
    if not cfg_path:
        return False
    try:
        raw = json.load(open(cfg_path))
    except Exception:
        return False
    raw_name = str(raw.get("_name_or_path", "")).lower()
    return "image_aug" in raw_name


def _set_cuda_fastpaths() -> None:
    try:
        major, _minor = torch.cuda.get_device_capability()
        if major >= 8:
            # Prefer FlashAttention on Ampere+; keep other kernels disabled for determinism.
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
        else:
            # Volta/Turing do not support FlashAttention; allow math fallback.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
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
            # Many Bridge-style checkpoints use a binary gripper stored as {0,1} (open prob).
            # Use a 0.5 threshold when the value is in [0,1]; otherwise fall back to sign.
            if 0.0 <= value <= 1.0:
                return -1.0 if value <= 0.5 else 1.0
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

    return Image.fromarray(rgb).convert("RGB")


def _infer_resize_size_from_model(model: Any, processor: Any, *, fallback: int = 224) -> int:
    cfg = getattr(model, "config", None)
    sizes = getattr(cfg, "image_sizes", None)
    if isinstance(sizes, (list, tuple)) and len(sizes) >= 1:
        try:
            val = int(sizes[0])
            if val > 0:
                return val
        except Exception:
            pass

    img_proc = getattr(processor, "image_processor", None)
    input_sizes = getattr(img_proc, "input_sizes", None)
    if isinstance(input_sizes, (list, tuple)) and input_sizes:
        first = input_sizes[0]
        if isinstance(first, (list, tuple)) and len(first) >= 3:
            try:
                val = int(first[-1])
                if val > 0:
                    return val
            except Exception:
                pass
    return int(fallback)


def _octo_style_preprocess(image: Image.Image, *, resize_size: int, base_size: int = 128) -> Image.Image:
    """
    Approximate the OpenVLA/SimplerEnv preprocessing used in openvla-mini:
      JPEG round-trip + resize -> 128 + resize -> resize_size (Lanczos).
    TensorFlow is not available in our env, so we implement a close PIL equivalent.
    """
    try:
        resample_lanczos = Image.Resampling.LANCZOS  # Pillow>=9
    except AttributeError:  # pragma: no cover
        resample_lanczos = Image.LANCZOS

    # JPEG round-trip (RLDS builder artifact)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")

    # Resize to base_size then to resize_size
    image = image.resize((base_size, base_size), resample=resample_lanczos)
    image = image.resize((resize_size, resize_size), resample=resample_lanczos)
    return image


def _normalize_proprio(
    proprio: np.ndarray,
    *,
    norm_stats: Optional[Dict[str, Any]],
    unnorm_key: Optional[str],
    clip: bool = True,
) -> np.ndarray:
    if norm_stats is None or unnorm_key is None:
        return proprio
    key_stats = norm_stats.get(unnorm_key, None)
    if not isinstance(key_stats, dict):
        return proprio
    proprio_stats = key_stats.get("proprio", None)
    if not isinstance(proprio_stats, dict):
        return proprio

    from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType  # noqa: E402

    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        low = np.asarray(proprio_stats.get("q01", None), dtype=np.float32)
        high = np.asarray(proprio_stats.get("q99", None), dtype=np.float32)
    else:
        low = np.asarray(proprio_stats.get("min", None), dtype=np.float32)
        high = np.asarray(proprio_stats.get("max", None), dtype=np.float32)

    if low.size == 0 or high.size == 0:
        return proprio
    if low.shape != high.shape or low.shape[0] != proprio.shape[0]:
        return proprio

    if np.allclose(low, 0.0) and np.allclose(high, 0.0):
        return proprio

    eps = 1e-8
    denom = (high - low).astype(np.float32)
    if np.all(np.abs(denom) < eps):
        return proprio
    denom = np.where(np.abs(denom) < eps, 1.0, denom)
    out = (2.0 * (proprio.astype(np.float32) - low) / denom) - 1.0
    if clip:
        out = np.clip(out, -1.0, 1.0)
    return out.astype(np.float32)


def _center_crop_and_resize_back(image: Image.Image, *, crop_scale: float) -> Image.Image:
    """
    Center-crop with area fraction `crop_scale` and resize back to original resolution.
    Matches the OpenVLA helper note that crop side length scales with sqrt(crop_scale).
    """
    if not (0.0 < crop_scale <= 1.0):
        raise ValueError(f"crop_scale must be in (0, 1], got {crop_scale}")

    if crop_scale >= 1.0:
        return image

    w, h = image.size
    side_scale = math.sqrt(crop_scale)
    new_w = max(1, int(round(w * side_scale)))
    new_h = max(1, int(round(h * side_scale)))

    left = max(0, (w - new_w) // 2)
    top = max(0, (h - new_h) // 2)
    cropped = image.crop((left, top, left + new_w, top + new_h))

    try:
        resample = Image.Resampling.BILINEAR  # Pillow>=9
    except AttributeError:  # pragma: no cover
        resample = Image.BILINEAR
    return cropped.resize((w, h), resample=resample)


def _convert_simpler_eef_pos_to_bridge_proprio(eef_pos: np.ndarray) -> np.ndarray:
    """
    SimplerEnv `obs["agent"]["eef_pos"]` packs:
      [x, y, z, quat_w, quat_x, quat_y, quat_z, gripper]

    Bridge/OpenVLA-OFT proprio stats suggest a 7D representation:
      [x, y, z, roll, pitch, yaw, gripper]
    """
    eef_pos = np.asarray(eef_pos, dtype=np.float32).reshape(-1)
    if eef_pos.shape[0] != 8:
        raise ValueError(f"Expected 8D eef_pos, got shape {eef_pos.shape}")

    pos = eef_pos[:3]
    quat_wxyz = eef_pos[3:7]
    gripper = eef_pos[7]

    ensure_open_pi_zero_on_path()
    from src.utils.geometry import mat2euler, quat2mat  # noqa: E402

    roll, pitch, yaw = mat2euler(quat2mat(quat_wxyz), axes="sxyz")
    return np.asarray([pos[0], pos[1], pos[2], roll, pitch, yaw, gripper], dtype=np.float32)


def _convert_simpler_eef_pos_to_bridge_proprio_padded(eef_pos: np.ndarray) -> np.ndarray:
    """
    Bridge/Open-X POS_EULER uses an 8D state:
      [x, y, z, roll, pitch, yaw, <PAD>, gripper]
    """
    base = _convert_simpler_eef_pos_to_bridge_proprio(eef_pos)
    pad = np.asarray([0.0], dtype=np.float32)
    return np.concatenate([base[:6], pad, base[6:7]]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a HuggingFace VLA (e.g., OpenVLA) on SimplerEnv Bridge tasks."
    )
    parser.add_argument("--model-id", type=str, required=True, help="HF model id or local path.")
    parser.add_argument(
        "--preset",
        type=str,
        default="baseline",
        choices=["baseline", "blurr"],
        help="baseline=FP32 eager; blurr=BF16+torch.compile (BLURR-style).",
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
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="In: What action should the robot take to {instruction}?\nOut:",
        help="Prompt template; must contain '{instruction}'.",
    )
    parser.add_argument("--unnorm-key", type=str, default=None)
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="",
        help="Optional transformers attention implementation hint passed to from_pretrained, e.g. 'flash_attention_2'.",
    )
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--use-torch-compile", action="store_true")
    parser.add_argument("--no-torch-compile", action="store_true")
    parser.add_argument(
        "--use-lora",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Load & merge PEFT LoRA adapter from `lora_adapter/` when present.",
    )
    parser.add_argument(
        "--center-crop",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Center crop (area fraction) + resize back before processor; auto enables when model config suggests image aug.",
    )
    parser.add_argument("--center-crop-scale", type=float, default=0.9)
    parser.add_argument(
        "--image-preproc",
        type=str,
        default="octo128",
        choices=["octo128", "raw"],
        help="Image preprocessing before HF processor: octo128 matches OpenVLA eval (JPEG+128->resize); raw uses env RGB directly.",
    )
    parser.add_argument(
        "--normalize-proprio",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Normalize proprio using model norm_stats (q01/q99) before proprio projector; auto enables when projector is used.",
    )
    parser.add_argument(
        "--use-oft-action-head",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Use OFT regression action_head--*.pt when present; auto enables when available.",
    )
    parser.add_argument(
        "--use-oft-proprio",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Use OFT proprio_projector--*.pt and feed proprio when present; auto enables when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_open_pi_zero_on_path()
    os.chdir(open_pi_zero_root())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. Run this on a GPU node.")

    tasks = list(args.task)
    log_dir = (
        Path(args.log_dir).expanduser()
        if args.log_dir
        else _default_log_dir(
            tag=f"hf_{args.preset}",
            seed=args.seed,
            task=tasks[0] if len(tasks) == 1 else None,
        )
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
    log = logging.getLogger("eval_hf_vla_simpler")
    log.info("Args: %s", json.dumps(vars(args), sort_keys=True))

    device = torch.device(f"cuda:{args.gpu_id}")
    _set_cuda_fastpaths()

    use_bf16 = args.use_bf16
    use_fp16 = args.use_fp16
    use_compile = args.use_torch_compile
    if args.no_torch_compile:
        use_compile = False

    explicit_dtype = bool(args.use_bf16 or args.use_fp16)
    explicit_compile = bool(args.use_torch_compile or args.no_torch_compile)
    if args.preset == "baseline":
        if not explicit_dtype:
            use_bf16 = False
            use_fp16 = False
        if not explicit_compile:
            use_compile = False
    elif args.preset == "blurr":
        if not explicit_dtype:
            use_bf16 = True
            use_fp16 = False
        if not explicit_compile:
            use_compile = True

    if use_bf16 and use_fp16:
        raise ValueError("Specify at most one of --use-bf16 or --use-fp16.")

    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    torch_dtype = dtype if dtype in {torch.float16, torch.bfloat16} else torch.float32

    log.info("Loading model: %s", args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForVision2Seq.from_pretrained(args.model_id, **load_kwargs).to(device)
    use_lora_mode = str(args.use_lora).lower()
    # NOTE: Many community "OpenVLA-OFT" repos already ship fully materialized (merged) weights *and* keep a
    # `lora_adapter/` directory for reference. Auto-merging that adapter again can double-apply deltas and
    # invalidate evaluation results. Users can still force it with `--use-lora on`.
    if use_lora_mode == "auto" and "openvla-oft" in str(args.model_id).lower():
        log.info("Skipping LoRA auto-merge for OpenVLA-OFT model_id=%s (set --use-lora on to force).", args.model_id)
    elif use_lora_mode != "off":
        model = _maybe_load_lora_adapter(model, args.model_id, device=device)
    model.eval()

    _inject_dataset_statistics_into_norm_stats(model, args.model_id)

    center_crop_mode = str(args.center_crop).lower()
    if center_crop_mode == "on":
        do_center_crop = True
    elif center_crop_mode == "off":
        do_center_crop = False
    else:
        do_center_crop = _infer_center_crop_default(args.model_id)
    if do_center_crop:
        log.info("Center-crop enabled (mode=%s scale=%.3f).", args.center_crop, args.center_crop_scale)

    resize_size = _infer_resize_size_from_model(model, processor, fallback=224)
    image_preproc = str(args.image_preproc).lower()
    log.info("Image preprocess: %s (resize_size=%d).", image_preproc, resize_size)

    unnorm_key = _infer_unnorm_key(model=model, model_id=args.model_id, override=args.unnorm_key)
    action_stats = None
    get_action_stats = getattr(model, "get_action_stats", None)
    if callable(get_action_stats) and unnorm_key is not None:
        try:
            action_stats = get_action_stats(unnorm_key)
        except Exception:
            action_stats = None
    raw_oft_action_head, raw_oft_proprio_projector = _load_oft_aux_modules(
        args.model_id, device=device, dtype=torch_dtype
    )
    use_oft_action_head_mode = str(args.use_oft_action_head).lower()
    use_oft_proprio_mode = str(args.use_oft_proprio).lower()
    oft_action_head = (
        raw_oft_action_head
        if raw_oft_action_head is not None and use_oft_action_head_mode in {"auto", "on"}
        else None
    )
    oft_proprio_projector = (
        raw_oft_proprio_projector
        if raw_oft_proprio_projector is not None and use_oft_proprio_mode in {"auto", "on"}
        else None
    )
    log.info("Using dtype=%s torch.compile=%s unnorm_key=%s", dtype, use_compile, unnorm_key)
    log.info(
        "OFT extras: action_head=%s proprio_projector=%s",
        "on" if oft_action_head is not None else "off",
        "on" if oft_proprio_projector is not None else "off",
    )
    proprio_dim_warned = False
    proprio_format_warned = False
    invalid_action_warned = False
    allow_oft_kwargs = True

    if use_compile:
        model = torch.compile(model, mode="default")

    if "{instruction}" not in args.prompt_template:
        raise ValueError("--prompt-template must contain '{instruction}'")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    import simpler_env  # noqa: E402

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

                prompt = args.prompt_template.format(instruction=instruction)
                image = _extract_rgb(env, obs)
                if image_preproc == "octo128":
                    image = _octo_style_preprocess(image, resize_size=resize_size, base_size=128)
                if do_center_crop:
                    image = _center_crop_and_resize_back(image, crop_scale=args.center_crop_scale)
                batch = processor(prompt, image, return_tensors="pt")
                inputs = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        if v.dtype.is_floating_point:
                            inputs[k] = v.to(device=device, dtype=dtype, non_blocking=True)
                        else:
                            inputs[k] = v.to(device=device, non_blocking=True)
                    else:
                        inputs[k] = v

                with torch.inference_mode():
                    predict_kwargs: Dict[str, Any] = {"unnorm_key": unnorm_key, "do_sample": False}
                    if allow_oft_kwargs and oft_action_head is not None:
                        predict_kwargs["action_head"] = oft_action_head
                    if allow_oft_kwargs and oft_proprio_projector is not None:
                        proprio = None
                        expected_dim = int(getattr(oft_proprio_projector.fc1, "in_features", 0) or 0)
                        raw_source = None
                        try:
                            agent_obs = obs.get("agent", None)  # type: ignore[union-attr]
                            if isinstance(agent_obs, dict) and "eef_pos" in agent_obs:
                                proprio = agent_obs["eef_pos"]
                                raw_source = "agent.eef_pos"
                            else:
                                proprio = obs.get("extra", {}).get("tcp_pose", None)  # type: ignore[union-attr]
                                raw_source = "extra.tcp_pose"
                        except Exception:
                            proprio = None
                        if proprio is not None:
                            proprio_arr = np.asarray(proprio, dtype=np.float32).reshape(-1)
                            if (
                                expected_dim == 7
                                and proprio_arr.shape[0] == 8
                                and raw_source == "agent.eef_pos"
                            ):
                                try:
                                    proprio_arr = _convert_simpler_eef_pos_to_bridge_proprio(proprio_arr)
                                    if not proprio_format_warned:
                                        log.info(
                                            "Converted SimplerEnv eef_pos (8D pos+quat+gripper) -> Bridge proprio (7D pos+rpy+gripper)."
                                        )
                                        proprio_format_warned = True
                                except Exception:
                                    log.warning("Failed to convert eef_pos to 7D Bridge proprio.", exc_info=True)
                            if (
                                expected_dim == 8
                                and proprio_arr.shape[0] == 8
                                and raw_source == "agent.eef_pos"
                            ):
                                try:
                                    proprio_arr = _convert_simpler_eef_pos_to_bridge_proprio_padded(proprio_arr)
                                    if not proprio_format_warned:
                                        log.info(
                                            "Converted SimplerEnv eef_pos (8D pos+quat+gripper) -> Bridge proprio (8D pos+rpy+pad+gripper)."
                                        )
                                        proprio_format_warned = True
                                except Exception:
                                    log.warning("Failed to convert eef_pos to 8D Bridge proprio.", exc_info=True)
                            if expected_dim and proprio_arr.shape[0] != expected_dim:
                                if not proprio_dim_warned:
                                    log.warning(
                                        "Proprio dim mismatch (got %d, expected %d); slicing or skipping proprio.",
                                        proprio_arr.shape[0],
                                        expected_dim,
                                    )
                                    proprio_dim_warned = True
                                if proprio_arr.shape[0] > expected_dim:
                                    proprio_arr = proprio_arr[:expected_dim]
                                else:
                                    proprio_arr = None
                            if proprio_arr is not None:
                                normalize_mode = str(args.normalize_proprio).lower()
                                do_norm = normalize_mode == "on" or (
                                    normalize_mode == "auto" and hasattr(model, "norm_stats")
                                )
                                if do_norm:
                                    try:
                                        proprio_arr = _normalize_proprio(
                                            proprio_arr, norm_stats=getattr(model, "norm_stats", None), unnorm_key=unnorm_key
                                        )
                                    except Exception:
                                        log.warning("Failed to normalize proprio; using raw proprio.", exc_info=True)
                                predict_kwargs["proprio"] = proprio_arr
                                predict_kwargs["proprio_projector"] = oft_proprio_projector

                    try:
                        action = model.predict_action(**inputs, **predict_kwargs)
                    except ValueError as exc:
                        msg = str(exc)
                        is_unused_model_kwargs = "model_kwargs" in msg and "are not used by the model" in msg
                        has_oft_kwargs = any(k in predict_kwargs for k in ("action_head", "proprio", "proprio_projector"))
                        if allow_oft_kwargs and is_unused_model_kwargs and has_oft_kwargs:
                            dropped = [k for k in ("action_head", "proprio", "proprio_projector") if k in predict_kwargs]
                            log.warning(
                                "Model does not accept OFT kwargs %s (likely older remote-code); disabling OFT extras.",
                                dropped,
                            )
                            allow_oft_kwargs = False
                            for k in dropped:
                                predict_kwargs.pop(k, None)
                            action = model.predict_action(**inputs, **predict_kwargs)
                        else:
                            raise
                if isinstance(action, (tuple, list)):
                    action = action[0]
                if torch.is_tensor(action):
                    action = action.detach().to("cpu").float().numpy()
                action_seq = np.asarray(action, dtype=np.float32)
                if action_seq.ndim == 1:
                    action_seq = action_seq.reshape(1, -1)
                if action_seq.ndim != 2 or action_seq.shape[1] != 7:
                    raise ValueError(f"Expected action shape (7,) or (N,7), got {action_seq.shape}")

                for sub_action in action_seq:
                    if step_in_episode >= args.max_steps + args.num_steps_wait:
                        break

                    env_action = _bridge_action_to_simpler(sub_action, action_stats=action_stats)
                    if not np.isfinite(env_action).all():
                        if not invalid_action_warned:
                            log.warning(
                                "Non-finite env action detected (nan/inf); replacing with zeros. action=%s",
                                env_action,
                            )
                            invalid_action_warned = True
                        env_action = np.nan_to_num(env_action, nan=0.0, posinf=0.0, neginf=0.0)
                    obs, reward, terminated, truncated, info = env.step(env_action)
                    step_in_episode += 1

                    new_instruction = env.get_language_instruction()
                    if new_instruction != instruction:
                        instruction = new_instruction

                    if terminated or truncated:
                        break

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
        "model_id": args.model_id,
        "preset": args.preset,
        "dtype": str(dtype),
        "torch_compile": bool(use_compile),
        "attn_implementation": args.attn_implementation,
        "image_preproc": args.image_preproc,
        "center_crop": args.center_crop,
        "center_crop_scale": float(args.center_crop_scale),
        "normalize_proprio": args.normalize_proprio,
        "use_oft_action_head": args.use_oft_action_head,
        "use_oft_proprio": args.use_oft_proprio,
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
