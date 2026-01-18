#!/usr/bin/env python3
"""
Benchmark latency / VRAM (and optional GFLOPS) for local PiZero checkpoints.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

from blurr.imports import ensure_open_pi_zero_on_path
from blurr.paths import open_pi_zero_root

ensure_open_pi_zero_on_path()
# Keep relative paths (e.g., config/, media/) consistent with the vendored open-pi-zero layout.
os.chdir(open_pi_zero_root())

from src.model.vla.pizero import PiZeroInference
from src.model.vla.processing import VLAProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark latency / VRAM / GFLOPS for a PiZero checkpoint."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the Pi0 config yaml (relative to third_party/open_pi_zero/ or absolute).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint (.pt).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="In: What action should the robot take?\nOut:",
        help="Text prompt fed to the tokenizer.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="media/maniskill_pp.png",
        help="Path to an RGB image used as visual context.",
    )
    parser.add_argument(
        "--use-bf16",
        action="store_true",
        help="Evaluate using bfloat16.",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Evaluate using float16.",
    )
    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        help="Enable torch.compile for the PiZero model.",
    )
    parser.add_argument(
        "--no-prefix-kv-cache",
        action="store_true",
        help="Disable prefix KV cache (runs the VLM+proprio every flow step).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--skip-flops",
        action="store_true",
        help="Skip FLOP profiling (GFLOPS will be unavailable).",
    )
    parser.add_argument(
        "--proprio-mode",
        type=str,
        default="zeros",
        choices=["zeros", "random"],
        help="How to populate the proprio input for benchmarking.",
    )
    return parser.parse_args()


def _select_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.use_bf16 and args.use_fp16:
        raise ValueError("Specify at most one of --use-bf16 or --use-fp16.")
    if args.use_bf16:
        return torch.bfloat16
    if args.use_fp16:
        return torch.float16
    return torch.float32


def _load_model(
    cfg_path: str,
    checkpoint_path: str,
    dtype: torch.dtype,
    device: torch.device,
    use_torch_compile: bool,
):
    cfg = OmegaConf.load(str(Path(cfg_path).expanduser()))
    model = PiZeroInference(cfg, use_ddp=False)
    try:
        data = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "If you are on a compute node, ensure the path is accessible."
        ) from exc
    data["model"] = {
        k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
    }
    model.load_state_dict(data["model"], strict=True)
    model.freeze_all_weights()
    model.to(dtype)
    model.to(device)
    model.enable_action_quantization()
    if use_torch_compile:
        model = torch.compile(model, mode="reduce-overhead")
    model.eval()
    return model, cfg


def _load_image(path: str, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.array(image, dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _prepare_inputs(
    cfg,
    model,
    prompt: str,
    image_path: str,
    dtype: torch.dtype,
    device,
    proprio_mode: str,
    use_prefix_kv_cache: bool,
):
    tokenizer_padding_side = getattr(cfg, "tokenizer_padding_side", "right")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_model_path,
        padding_side=tokenizer_padding_side,
    )
    tokenizer.padding_side = tokenizer_padding_side
    processor = VLAProcessor(
        tokenizer,
        cfg.vision.config.num_image_tokens,
        cfg.max_seq_len,
        tokenizer_padding=getattr(cfg, "tokenizer_padding", "max_length"),
    )

    image_tensor = _load_image(
        image_path,
        cfg.vision.config.image_size,
    )
    processed = processor(text=[prompt], images=image_tensor)
    attention_mask = processed["attention_mask"]
    (
        causal_mask,
        vlm_position_ids,
        proprio_position_ids,
        action_position_ids,
    ) = model.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
    (
        image_text_proprio_mask,
        action_mask,
    ) = model.split_full_mask_into_submasks(causal_mask)

    if getattr(cfg, "proprio_dim", None) is None:
        raise ValueError("Config missing `proprio_dim`, required for benchmarking.")
    proprio_shape = (1, cfg.cond_steps, cfg.proprio_dim)

    if proprio_mode == "random":
        proprios = torch.rand(proprio_shape, dtype=dtype)
    else:
        proprios = torch.zeros(proprio_shape, dtype=dtype)

    if use_prefix_kv_cache:
        inputs = {
            "input_ids": processed["input_ids"],
            "pixel_values": processed["pixel_values"],
            "image_text_proprio_mask": image_text_proprio_mask,
            "action_mask": action_mask,
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": proprios,
        }
    else:
        inputs = {
            "input_ids": processed["input_ids"],
            "pixel_values": processed["pixel_values"],
            "causal_mask": causal_mask,
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": proprios,
        }

    tensors = {}
    float_keys = {"pixel_values", "image_text_proprio_mask", "action_mask", "proprios"}
    for key, value in inputs.items():
        if key in float_keys:
            tensors[key] = value.to(device=device, dtype=dtype)
        else:
            tensors[key] = value.to(device=device)
    return tensors


def _single_forward(model, inputs):
    with torch.inference_mode():
        if "image_text_proprio_mask" in inputs:
            model(**inputs)
            return

        infer_action_naive = getattr(model, "infer_action_naive", None)
        if infer_action_naive is None and hasattr(model, "_orig_mod"):
            infer_action_naive = getattr(model._orig_mod, "infer_action_naive", None)
        if infer_action_naive is None:
            raise AttributeError(
                "infer_action_naive not found on model; disable torch.compile or "
                "use prefix KV cache path."
            )
        infer_action_naive(**inputs)


def benchmark(model, inputs, warmup: int, iters: int, skip_flops: bool):
    device = next(model.parameters()).device

    _single_forward(model, inputs)
    torch.cuda.synchronize(device)

    total_flops = None
    flops_note = None
    if not skip_flops:
        try:
            from torch.profiler import ProfilerActivity, profile

            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            with profile(activities=activities, with_flops=True) as prof:
                _single_forward(model, inputs)
            accumulated = 0
            for evt in prof.key_averages():
                flops = getattr(evt, "flops", None)
                if flops in (None, 0):
                    flops = getattr(evt, "self_flops", 0)
                accumulated += flops or 0
            if accumulated > 0:
                total_flops = accumulated
            else:
                flops_note = "Profiler reported zero FLOPs."
        except Exception as exc:  # noqa: BLE001
            flops_note = f"FLOP profiling failed: {exc}"

    torch.cuda.reset_peak_memory_stats(device=device)

    for _ in range(warmup):
        _single_forward(model, inputs)
    torch.cuda.synchronize(device)

    start = time.time()
    for _ in range(iters):
        _single_forward(model, inputs)
    torch.cuda.synchronize(device)
    avg_latency = (time.time() - start) / max(iters, 1)
    peak_mem = torch.cuda.max_memory_reserved(device=device) / 1024**3

    gflops = None
    if total_flops is not None and avg_latency > 0:
        gflops = (total_flops / avg_latency) / 1e9

    return avg_latency, peak_mem, gflops, flops_note


def main():
    args = parse_args()
    dtype = _select_dtype(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device not available. Please run on a GPU node.")

    model, cfg = _load_model(
        args.config,
        args.checkpoint,
        dtype,
        device,
        args.use_torch_compile,
    )
    inputs = _prepare_inputs(
        cfg,
        model,
        args.prompt,
        args.image,
        dtype,
        device,
        args.proprio_mode,
        use_prefix_kv_cache=not args.no_prefix_kv_cache,
    )

    avg_latency, peak_mem, gflops, flops_note = benchmark(
        model,
        inputs,
        args.warmup,
        args.iters,
        args.skip_flops,
    )

    dtype_name = (
        "bfloat16"
        if dtype == torch.bfloat16
        else "float16"
        if dtype == torch.float16
        else "float32"
    )
    print("========== Benchmark Summary ==========")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Prompt: {args.prompt}")
    print(f"Image: {Path(args.image).resolve()}")
    print(f"Device: {device}")
    print(f"Dtype: torch.{dtype_name}")
    print(f"bfloat16: {dtype == torch.bfloat16}")
    print(f"torch.compile: {args.use_torch_compile}")
    print(f"prefix KV cache: {not args.no_prefix_kv_cache}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Timed iterations: {args.iters}")
    print(f"Average latency: {avg_latency * 1000:.2f} ms")
    print(f"Peak reserved GPU memory: {peak_mem:.2f} GB")
    if gflops is not None:
        print(f"Approximate GFLOPS: {gflops:,.2f}")
    elif args.skip_flops:
        print("GFLOPS skipped (--skip-flops).")
    else:
        print(f"GFLOPS unavailable ({flops_note})")
    print("=======================================")


if __name__ == "__main__":
    main()
