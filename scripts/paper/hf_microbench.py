#!/usr/bin/env python3
"""
Paper-facing HuggingFace VLA microbenchmarks (latency / VRAM / GFLOPS).
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import transformers.modeling_utils as modeling_utils


def _set_default_caches(root: Path) -> None:
    hf_home = os.environ.get("HF_HOME") or str(root / "hf_cache")
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _ensure_sdpa_flag() -> None:
    base_cls = getattr(modeling_utils, "PreTrainedModel", None)
    if base_cls is not None and not hasattr(base_cls, "_supports_sdpa"):
        base_cls._supports_sdpa = False  # type: ignore[attr-defined]


def _ensure_rich_logging_fallback() -> None:
    """
    Some HF VLA repos (e.g., prismatic/OpenVLA implementations) configure logging with
    `rich.logging.RichHandler`. The pi0 conda env used for our paper does not always
    include `rich`, so provide a minimal stub to avoid hard failures in benchmarks.
    """

    try:
        import rich.logging  # noqa: F401
        return
    except Exception:
        pass

    import logging

    class RichHandler(logging.StreamHandler):
        def __init__(self, *args, **kwargs):  # noqa: ANN001
            super().__init__()

    rich_mod = types.ModuleType("rich")
    rich_logging_mod = types.ModuleType("rich.logging")
    rich_logging_mod.RichHandler = RichHandler
    rich_mod.logging = rich_logging_mod

    sys.modules.setdefault("rich", rich_mod)
    sys.modules.setdefault("rich.logging", rich_logging_mod)


def _prefer_workspace_prismatic(repo_root: Path) -> None:
    """
    Prefer a local/workspace `prismatic/` package if present.

    Some OpenVLA derivative checkpoints expect newer `prismatic` modules
    (e.g., `prismatic.vla.constants`, `prismatic.training.train_utils`).
    Our environment may also include an older `prismatic` on site-packages.
    """

    workspace_root = repo_root.parent
    candidate = workspace_root / "prismatic" / "__init__.py"
    if not candidate.is_file():
        return

    # Ensure workspace root takes precedence over any older installed versions.
    try:
        sys.path.remove(str(workspace_root))
    except ValueError:
        pass
    sys.path.insert(0, str(workspace_root))

    # If an incompatible `prismatic` was already imported, force a reload from the
    # workspace version for each model load.
    for name in list(sys.modules.keys()):
        if name == "prismatic" or name.startswith("prismatic."):
            del sys.modules[name]


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. Run this on a GPU node.")


def _select_dtype(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        # BF16 requires Ampere+ (sm80+) for efficient execution. For older GPUs
        # (e.g., V100), fall back to FP16 so the microbench can still run.
        try:
            is_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        except Exception:
            is_supported = False
        if torch.cuda.is_available() and not is_supported:
            return torch.float16
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unknown dtype: {name}")


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


def _infer_unnorm_key(model, model_id: str, override: str | None) -> str | None:
    if override is not None:
        return override

    dataset_stats = _load_dataset_stats(model_id)
    norm_stats = getattr(model, "norm_stats", {}) or {}
    norm_keys = list(norm_stats.keys())

    if dataset_stats:
        intersection = [k for k in dataset_stats.keys() if k in norm_keys]
        if intersection:
            return intersection[0]
    if "bridge_orig" in norm_keys:
        return "bridge_orig"
    if norm_keys:
        return norm_keys[0]
    return None


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


def _profile_flops_once(fn):
    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities, with_flops=True) as prof:
            fn()
        accumulated = 0
        for evt in prof.key_averages():
            flops = getattr(evt, "flops", None)
            if flops in (None, 0):
                flops = getattr(evt, "self_flops", 0)
            accumulated += flops or 0
        if accumulated > 0:
            return accumulated, None
        return None, "Profiler reported zero FLOPs."
    except Exception as exc:  # noqa: BLE001
        return None, f"FLOP profiling failed: {exc}"


def benchmark(model, inputs, unnorm_key: str | None, warmup: int, iters: int, skip_flops: bool):
    device = next(model.parameters()).device

    def fn():
        with torch.inference_mode():
            model.predict_action(**inputs, unnorm_key=unnorm_key)

    fn()
    torch.cuda.synchronize(device)

    total_flops, flops_note = (None, None)
    if not skip_flops:
        total_flops, flops_note = _profile_flops_once(fn)

    torch.cuda.reset_peak_memory_stats(device=device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    latency_s = (time.perf_counter() - start) / max(iters, 1)
    peak_mem = torch.cuda.max_memory_reserved(device=device) / 1024**3

    gflops = None
    if total_flops is not None and latency_s > 0:
        gflops = (total_flops / latency_s) / 1e9

    return latency_s, float(peak_mem), (float(gflops) if gflops is not None else None), flops_note


def _profiles(spec: List[str]) -> List[Dict]:
    out = []
    for item in spec:
        item = item.strip().lower()
        if item == "fp32_eager":
            out.append({"name": item, "dtype": "fp32", "compile": False})
        elif item == "bf16_eager":
            out.append({"name": item, "dtype": "bf16", "compile": False})
        elif item == "bf16_compile":
            out.append({"name": item, "dtype": "bf16", "compile": True})
        elif item == "fp16_compile":
            out.append({"name": item, "dtype": "fp16", "compile": True})
        else:
            raise ValueError(f"Unknown profile: {item}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--model-id", type=str, nargs="+", required=True)
    parser.add_argument("--prompt", type=str, default="In: What action should the robot take?\nOut:")
    parser.add_argument(
        "--image",
        type=str,
        default="third_party/open_pi_zero/media/maniskill_pp.png",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="",
        help=(
            "Optional transformers attention implementation hint passed to from_pretrained, "
            "e.g. 'flash_attention_2'."
        ),
    )
    parser.add_argument("--unnorm-key", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument(
        "--profiles",
        type=str,
        nargs="+",
        default=["fp32_eager", "bf16_compile"],
        help="Benchmark presets: fp32_eager, bf16_eager, bf16_compile, fp16_compile.",
    )
    args = parser.parse_args()

    _ensure_sdpa_flag()
    _ensure_rich_logging_fallback()
    _require_cuda()

    repo = Path(__file__).resolve().parents[2]
    _prefer_workspace_prismatic(repo)
    _set_default_caches(repo)

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    rows = []
    for model_id in args.model_id:
        for profile in _profiles(args.profiles):
            _prefer_workspace_prismatic(repo)
            dtype = _select_dtype(profile["dtype"])
            torch_dtype = dtype if dtype in {torch.float16, torch.bfloat16} else torch.float32

            latency_s = None
            peak_gb = None
            gflops = None
            note = ""
            if profile["dtype"] == "bf16" and dtype == torch.float16:
                note = "bf16->fp16 (no bf16 support)"
            try:
                # On smaller GPUs, FP32 weights may OOM for large VLAs; fail gracefully.
                if dtype == torch.float32:
                    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    if total_mem_gb < 40:
                        raise torch.cuda.OutOfMemoryError(
                            f"Skipping FP32 on {total_mem_gb:.1f}GB GPU (likely OOM)."
                        )

                processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                load_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch_dtype,
                    "low_cpu_mem_usage": True,
                }
                if args.attn_implementation:
                    load_kwargs["attn_implementation"] = args.attn_implementation
                model = AutoModelForVision2Seq.from_pretrained(model_id, **load_kwargs).to(device)
                model.eval()

                unnorm_key = _infer_unnorm_key(model, model_id, args.unnorm_key)
                inputs = _prepare_inputs(processor, args.prompt, args.image, device, dtype)

                if profile["compile"]:
                    model = torch.compile(model, mode="default")

                latency_s, peak_gb, gflops, flops_note = benchmark(
                    model,
                    inputs,
                    unnorm_key,
                    args.warmup,
                    args.iters,
                    args.skip_flops,
                )
                if flops_note:
                    note = (note + " | " if note else "") + flops_note
            except torch.cuda.OutOfMemoryError as exc:
                note = (note + " | " if note else "") + f"OOM: {exc}"
            except Exception as exc:  # noqa: BLE001
                note = (note + " | " if note else "") + f"ERR[{type(exc).__name__}]: {exc}"
            finally:
                # Try to free VRAM between runs.
                try:
                    del model  # type: ignore[name-defined]
                except Exception:
                    pass
                try:
                    del processor  # type: ignore[name-defined]
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

            rows.append(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "model_id": model_id,
                    "profile": profile["name"],
                    "dtype": (
                        "bf16"
                        if dtype == torch.bfloat16
                        else "fp16"
                        if dtype == torch.float16
                        else "fp32"
                    ),
                    "torch_compile": bool(profile["compile"]),
                    "latency_ms": None if latency_s is None else round(latency_s * 1000.0, 4),
                    "peak_reserved_gb": None if peak_gb is None else round(float(peak_gb), 4),
                    "gflops": None if gflops is None else round(float(gflops), 4),
                    "note": note,
                }
            )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
