#!/usr/bin/env python3
"""
Benchmark latency / memory / GFLOPS for Hugging Face hosted Vision-Language-Action
models such as OpenVLA or OpenVLA-OFT.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import transformers.modeling_utils as modeling_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GFLOPS / latency for a Hugging Face VLA checkpoint."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Hugging Face model id (e.g. Kaipengm2/openvla-oft-64-130000).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="In: What action should the robot take?\nOut:",
        help="Text prompt fed to the processor.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="media/maniskill_pp.png",
        help="Path to an RGB image used as visual context.",
    )
    parser.add_argument(
        "--unnorm-key",
        type=str,
        default=None,
        help="Key used to unnormalize actions. If omitted we try to infer it from dataset_statistics.json.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations to discard.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--use-bf16",
        action="store_true",
        help="Evaluate model using bfloat16 weights/activations.",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Evaluate model using float16 weights/activations.",
    )
    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        help="Enable torch.compile for the model.",
    )
    parser.add_argument(
        "--skip-flops",
        action="store_true",
        help="Skip FLOP profiling (GFLOPS will be unavailable).",
    )
    return parser.parse_args()


def _ensure_sdpa_flag() -> None:
    """Inject conservative default for `_supports_sdpa` for older transformers versions."""
    base_cls = getattr(modeling_utils, "PreTrainedModel", None)
    if base_cls is not None and not hasattr(base_cls, "_supports_sdpa"):
        base_cls._supports_sdpa = False  # type: ignore[attr-defined]


def _select_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.use_bf16 and args.use_fp16:
        raise ValueError("Specify at most one of --use-bf16 or --use-fp16.")
    if args.use_bf16:
        return torch.bfloat16
    if args.use_fp16:
        return torch.float16
    return torch.float32


def _load_model_and_processor(model_id: str, dtype: torch.dtype):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    torch_dtype = dtype if dtype in {torch.float16, torch.bfloat16} else torch.float32
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    return model, processor


def _load_dataset_stats(model_id: str) -> Dict[str, Dict]:
    from huggingface_hub import hf_hub_download
    import json

    try:
        path = hf_hub_download(model_id, "dataset_statistics.json", repo_type="model")
    except Exception:
        return {}
    try:
        return json.load(open(path))
    except Exception:
        return {}


def _prepare_inputs(processor, prompt: str, image_path: str, device: str, dtype: torch.dtype):
    image = Image.open(image_path).convert("RGB")
    batch = processor(prompt, image, return_tensors="pt")
    prepared = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if v.dtype.is_floating_point:
                prepared[k] = v.to(device=device, dtype=dtype)
            else:
                prepared[k] = v.to(device=device)
        else:
            prepared[k] = v
    return prepared


def benchmark(
    model,
    inputs,
    unnorm_key: str | None,
    warmup: int,
    iters: int,
    skip_flops: bool,
):
    device = next(model.parameters()).device

    with torch.inference_mode():
        model.predict_action(**inputs, unnorm_key=unnorm_key)
    torch.cuda.synchronize(device)

    total_flops = None
    flops_note = None
    if not skip_flops:
        try:
            from torch.profiler import ProfilerActivity, profile

            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            with profile(activities=activities, with_flops=True) as prof:
                with torch.inference_mode():
                    model.predict_action(**inputs, unnorm_key=unnorm_key)
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

    with torch.inference_mode():
        for _ in range(warmup):
            model.predict_action(**inputs, unnorm_key=unnorm_key)
    torch.cuda.synchronize(device)

    start = time.time()
    with torch.inference_mode():
        for _ in range(iters):
            model.predict_action(**inputs, unnorm_key=unnorm_key)
    torch.cuda.synchronize(device)
    avg_latency = (time.time() - start) / max(iters, 1)
    peak_mem = torch.cuda.max_memory_reserved(device=device) / 1024**3

    gflops = None
    if total_flops is not None and avg_latency > 0:
        gflops = (total_flops / avg_latency) / 1e9

    return avg_latency, peak_mem, gflops, flops_note


def main():
    args = parse_args()
    _ensure_sdpa_flag()

    dtype = _select_dtype(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA device not available. Please run on a GPU node.")

    model, processor = _load_model_and_processor(args.model_id, dtype)
    model = model.to(device)
    model.eval()

    dataset_stats = _load_dataset_stats(args.model_id)
    norm_stats = getattr(model, "norm_stats", {}) or {}
    norm_keys = list(norm_stats.keys())

    unnorm_key = args.unnorm_key
    if unnorm_key is None:
        if dataset_stats:
            intersection = [k for k in dataset_stats.keys() if k in norm_keys]
            if intersection:
                unnorm_key = intersection[0]
        if unnorm_key is None and "bridge_orig" in norm_keys:
            unnorm_key = "bridge_orig"
        if unnorm_key is None and norm_keys:
            unnorm_key = norm_keys[0]

    if unnorm_key and norm_keys and unnorm_key not in norm_keys:
        raise ValueError(
            f"--unnorm-key '{unnorm_key}' not among available statistics keys: {norm_keys}"
        )

    inputs = _prepare_inputs(processor, args.prompt, args.image, device, dtype)

    if args.use_torch_compile:
        model = torch.compile(model, mode="default")

    avg_latency, peak_mem, gflops, flops_note = benchmark(
        model,
        inputs,
        unnorm_key,
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
    print(f"Model id: {args.model_id}")
    print(f"Prompt: {args.prompt}")
    print(f"Image: {Path(args.image).resolve()}")
    print(f"Device: {device}")
    print(f"Dtype: torch.{dtype_name}")
    print(f"torch.compile: {args.use_torch_compile}")
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
