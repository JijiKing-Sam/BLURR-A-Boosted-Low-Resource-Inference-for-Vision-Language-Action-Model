#!/usr/bin/env python3
"""
Paper-facing Pi-0 microbenchmarks for BLURR.

Outputs machine-readable CSV/JSON so results can be pasted into the paper with
minimal manual work.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from PIL import Image
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blurr.imports import ensure_open_pi_zero_on_path
from blurr.paths import open_pi_zero_root, repo_root


def _set_default_caches() -> None:
    root = repo_root()
    hf_home = os.environ.get("HF_HOME") or str(root / "hf_cache")
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. Run this on a GPU node.")


def _select_dtype(*, use_bf16: bool, use_fp16: bool) -> torch.dtype:
    if use_bf16 and use_fp16:
        raise ValueError("Specify at most one of --use-bf16 or --use-fp16.")
    if use_bf16:
        # BF16 requires Ampere+ (sm80+) in practice. For older GPUs (e.g., V100),
        # fall back to FP16 so cross-hardware benchmarks still run.
        try:
            is_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        except Exception:
            is_supported = False
        if torch.cuda.is_available() and not is_supported:
            return torch.float16
        return torch.bfloat16
    if use_fp16:
        return torch.float16
    return torch.float32


def _load_image(path: str, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.array(image, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _count_non_image_tokens(*, input_ids: torch.Tensor, image_token_id: int, pad_token_id: int) -> int:
    ids = input_ids[0].tolist()
    return sum(1 for t in ids if t not in (image_token_id, pad_token_id))


def _count_total_tokens(*, attention_mask: torch.Tensor) -> int:
    return int(attention_mask[0].sum().item())


def _load_cfg(cfg_path: str) -> Any:
    path = Path(cfg_path).expanduser()
    if not path.is_absolute():
        path = open_pi_zero_root() / path
    return OmegaConf.load(str(path))


def _load_model(
    *,
    cfg,
    checkpoint_path: str,
    dtype: torch.dtype,
    device: torch.device,
    use_torch_compile: bool,
):
    ensure_open_pi_zero_on_path()
    os.chdir(open_pi_zero_root())

    from src.model.vla.pizero import PiZeroInference  # noqa: E402

    model = PiZeroInference(cfg, use_ddp=False)
    data = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu")
    data["model"] = {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}
    model.load_state_dict(data["model"], strict=True)
    model.freeze_all_weights()
    model.to(dtype=dtype, device=device)
    model.enable_action_quantization()
    if use_torch_compile:
        model = torch.compile(model, mode="reduce-overhead")
    model.eval()
    return model


def _prepare_inputs(
    *,
    cfg,
    model,
    prompt: str,
    image_path: str,
    dtype: torch.dtype,
    device: torch.device,
    proprio_mode: str,
    use_prefix_kv_cache: bool,
    max_seq_len_override: int,
) -> Dict[str, torch.Tensor]:
    ensure_open_pi_zero_on_path()
    os.chdir(open_pi_zero_root())

    from src.model.vla.processing import VLAProcessor  # noqa: E402

    tokenizer_padding_side = getattr(cfg, "tokenizer_padding_side", "right")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_model_path,
        padding_side=tokenizer_padding_side,
    )
    tokenizer.padding_side = tokenizer_padding_side

    max_seq_len = int(max_seq_len_override) if max_seq_len_override > 0 else int(cfg.max_seq_len)
    processor = VLAProcessor(
        tokenizer,
        int(cfg.vision.config.num_image_tokens),
        max_seq_len,
        tokenizer_padding=getattr(cfg, "tokenizer_padding", "max_length"),
    )

    image_tensor = _load_image(image_path, int(cfg.vision.config.image_size))
    processed = processor(text=[prompt], images=image_tensor)
    attention_mask = processed["attention_mask"]
    (
        causal_mask,
        vlm_position_ids,
        proprio_position_ids,
        action_position_ids,
    ) = model.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
    image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(causal_mask)

    proprio_shape = (1, int(cfg.cond_steps), int(cfg.proprio_dim))
    if proprio_mode == "random":
        proprios = torch.rand(proprio_shape, dtype=dtype)
    elif proprio_mode == "zeros":
        proprios = torch.zeros(proprio_shape, dtype=dtype)
    else:
        raise ValueError(f"Unknown proprio_mode: {proprio_mode}")

    if use_prefix_kv_cache:
        inputs: Dict[str, torch.Tensor] = {
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

    float_keys = {
        "pixel_values",
        "image_text_proprio_mask",
        "action_mask",
        "causal_mask",
        "proprios",
    }
    prepared: Dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if key in float_keys:
            prepared[key] = value.to(device=device, dtype=dtype)
        else:
            prepared[key] = value.to(device=device)
    return prepared


def _single_forward(*, model, inputs: Dict[str, torch.Tensor]) -> None:
    with torch.inference_mode():
        if "image_text_proprio_mask" in inputs:
            model(**inputs)
            return

        infer_action_naive = getattr(model, "infer_action_naive", None)
        if infer_action_naive is None and hasattr(model, "_orig_mod"):
            infer_action_naive = getattr(model._orig_mod, "infer_action_naive", None)
        if infer_action_naive is None:
            raise AttributeError(
                "infer_action_naive not found on model; disable torch.compile or use prefix KV cache path."
            )
        infer_action_naive(**inputs)


@dataclass(frozen=True)
class BenchResult:
    latency_s: float
    peak_reserved_gb: float
    gflops: float | None
    flops_note: str | None

    @property
    def latency_ms(self) -> float:
        return self.latency_s * 1000.0


def _profile_flops_once(*, fn) -> tuple[int | None, str | None]:
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


def bench(
    *,
    model,
    inputs: Dict[str, torch.Tensor],
    warmup: int,
    iters: int,
    skip_flops: bool,
) -> BenchResult:
    device = next(model.parameters()).device

    def fn() -> None:
        _single_forward(model=model, inputs=inputs)

    fn()
    torch.cuda.synchronize(device)

    total_flops, flops_note = (None, None)
    if not skip_flops:
        total_flops, flops_note = _profile_flops_once(fn=fn)

    torch.cuda.reset_peak_memory_stats(device=device)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    latency_s = (time.perf_counter() - start) / max(iters, 1)
    peak_reserved_gb = torch.cuda.max_memory_reserved(device=device) / 1024**3

    gflops = None
    if total_flops is not None and latency_s > 0:
        gflops = (total_flops / latency_s) / 1e9

    return BenchResult(
        latency_s=latency_s,
        peak_reserved_gb=float(peak_reserved_gb),
        gflops=float(gflops) if gflops is not None else None,
        flops_note=flops_note,
    )


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _gpu_name() -> str:
    if not torch.cuda.is_available():
        return "unknown"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return "unknown"


def _apply_overrides(cfg, *, num_inference_steps: int, max_seq_len: int) -> None:
    with open_dict(cfg):
        if num_inference_steps > 0:
            cfg.num_inference_steps = int(num_inference_steps)
        if max_seq_len > 0:
            cfg.max_seq_len = int(max_seq_len)


def _preset(name: str) -> Dict[str, Any]:
    name = name.strip().lower()
    if name in {"baseline", "vanilla"}:
        return {
            "use_prefix_kv_cache": False,
            "use_bf16": False,
            "use_fp16": False,
            "use_torch_compile": False,
            "num_inference_steps": 10,
        }
    if name in {"blurr", "step1", "blurr_step1"}:
        return {
            "use_prefix_kv_cache": True,
            "use_bf16": True,
            "use_fp16": False,
            "use_torch_compile": True,
            "num_inference_steps": 1,
        }
    raise ValueError(f"Unknown preset: {name}")


def cmd_prompt_sweep(args: argparse.Namespace) -> None:
    _set_default_caches()
    _require_cuda()

    out_csv = Path(args.out_csv).expanduser().resolve()

    dtype = _select_dtype(use_bf16=args.use_bf16, use_fp16=args.use_fp16)
    device = torch.device("cuda")

    cfg = _load_cfg(args.config)
    _apply_overrides(cfg, num_inference_steps=args.num_inference_steps, max_seq_len=0)

    # Keep prompt sweep simple and comparable: avoid compile by default.
    model = _load_model(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        dtype=dtype,
        device=device,
        use_torch_compile=args.use_torch_compile,
    )

    ensure_open_pi_zero_on_path()
    os.chdir(open_pi_zero_root())
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_path, padding_side="right")
    from src.model.vla.processing import VLAProcessor  # noqa: E402

    num_image_tokens = int(cfg.vision.config.num_image_tokens)
    image_size = int(cfg.vision.config.image_size)
    image_tensor = _load_image(args.image, image_size)

    rows: List[Dict[str, Any]] = []
    for target_text_tokens in args.text_tokens:
        # Keep max sequence length fixed to the checkpoint/config, otherwise rotary
        # embeddings will mismatch and inference will error.
        max_seq_len = int(getattr(cfg, "max_seq_len", 0) or 0)
        if max_seq_len <= 0:
            raise ValueError("Config missing `max_seq_len` (required for prompt sweep).")
        processor = VLAProcessor(
            tokenizer,
            num_image_tokens=num_image_tokens,
            max_seq_len=max_seq_len,
            tokenizer_padding=getattr(cfg, "tokenizer_padding", "max_length"),
        )

        # Build a simple repeated-token prompt (record actual token count after processing).
        prompt = " ".join([args.token] * int(target_text_tokens))
        processed = processor(text=[prompt], images=image_tensor)
        actual_non_image = _count_non_image_tokens(
            input_ids=processed["input_ids"],
            image_token_id=processor.image_token_id,
            pad_token_id=int(cfg.pad_token_id),
        )
        total_tokens = _count_total_tokens(attention_mask=processed["attention_mask"])

        for use_prefix_kv_cache in (False, True):
            inputs = _prepare_inputs(
                cfg=cfg,
                model=model,
                prompt=prompt,
                image_path=args.image,
                dtype=dtype,
                device=device,
                proprio_mode=args.proprio_mode,
                use_prefix_kv_cache=use_prefix_kv_cache,
                max_seq_len_override=0,
            )
            result = bench(
                model=model,
                inputs=inputs,
                warmup=args.warmup,
                iters=args.iters,
                skip_flops=args.skip_flops,
            )
            rows.append(
                {
                    "gpu": _gpu_name(),
                    "checkpoint": str(Path(args.checkpoint).expanduser()),
                    "config": str(Path(args.config)),
                    "dtype": str(dtype).replace("torch.", ""),
                    "torch_compile": bool(args.use_torch_compile),
                    "num_inference_steps": int(getattr(cfg, "num_inference_steps", -1)),
                    "max_seq_len": int(max_seq_len),
                    "target_text_tokens": int(target_text_tokens),
                    "actual_non_image_tokens": int(actual_non_image),
                    "total_tokens(attn_mask_sum)": int(total_tokens),
                    "use_prefix_kv_cache": bool(use_prefix_kv_cache),
                    "latency_ms": round(result.latency_ms, 4),
                    "peak_reserved_gb": round(result.peak_reserved_gb, 4),
                    "gflops": None if result.gflops is None else round(result.gflops, 4),
                    "note": result.flops_note or "",
                }
            )
            # Free per-run tensors before building the next input dict.
            del inputs
            torch.cuda.empty_cache()

    _write_csv(out_csv, rows)
    print(f"Wrote: {out_csv}")


def cmd_first_vs_steady(args: argparse.Namespace) -> None:
    _set_default_caches()
    _require_cuda()

    out_csv = Path(args.out_csv).expanduser().resolve()
    device = torch.device("cuda")

    rows: List[Dict[str, Any]] = []
    for preset_name in args.presets:
        preset = _preset(preset_name)
        dtype = _select_dtype(use_bf16=preset["use_bf16"], use_fp16=preset["use_fp16"])

        cfg = _load_cfg(args.config)
        _apply_overrides(cfg, num_inference_steps=preset["num_inference_steps"], max_seq_len=0)

        model = _load_model(
            cfg=cfg,
            checkpoint_path=args.checkpoint,
            dtype=dtype,
            device=device,
            use_torch_compile=preset["use_torch_compile"],
        )

        inputs = _prepare_inputs(
            cfg=cfg,
            model=model,
            prompt=args.prompt,
            image_path=args.image,
            dtype=dtype,
            device=device,
            proprio_mode=args.proprio_mode,
            use_prefix_kv_cache=preset["use_prefix_kv_cache"],
            max_seq_len_override=0,
        )

        # First call (captures torch.compile warmup if enabled).
        start = time.perf_counter()
        _single_forward(model=model, inputs=inputs)
        torch.cuda.synchronize(device)
        first_s = time.perf_counter() - start

        # Steady-state (warmup + avg latency).
        result = bench(
            model=model,
            inputs=inputs,
            warmup=args.warmup,
            iters=args.iters,
            skip_flops=True,
        )

        rows.append(
            {
                "gpu": _gpu_name(),
                "preset": preset_name,
                "dtype": str(dtype).replace("torch.", ""),
                "torch_compile": bool(preset["use_torch_compile"]),
                "use_prefix_kv_cache": bool(preset["use_prefix_kv_cache"]),
                "num_inference_steps": int(getattr(cfg, "num_inference_steps", -1)),
                "first_call_ms": round(first_s * 1000.0, 3),
                "steady_latency_ms": round(result.latency_ms, 3),
            }
        )
        del inputs
        del model
        torch.cuda.empty_cache()

    _write_csv(out_csv, rows)
    print(f"Wrote: {out_csv}")


def cmd_steps_sweep(args: argparse.Namespace) -> None:
    _set_default_caches()
    _require_cuda()

    out_csv = Path(args.out_csv).expanduser().resolve()
    device = torch.device("cuda")

    dtype = _select_dtype(use_bf16=args.use_bf16, use_fp16=args.use_fp16)

    rows: List[Dict[str, Any]] = []
    for steps in args.steps:
        cfg = _load_cfg(args.config)
        _apply_overrides(cfg, num_inference_steps=int(steps), max_seq_len=0)

        model = _load_model(
            cfg=cfg,
            checkpoint_path=args.checkpoint,
            dtype=dtype,
            device=device,
            use_torch_compile=args.use_torch_compile,
        )

        inputs = _prepare_inputs(
            cfg=cfg,
            model=model,
            prompt=args.prompt,
            image_path=args.image,
            dtype=dtype,
            device=device,
            proprio_mode=args.proprio_mode,
            use_prefix_kv_cache=not args.no_prefix_kv_cache,
            max_seq_len_override=0,
        )

        result = bench(
            model=model,
            inputs=inputs,
            warmup=args.warmup,
            iters=args.iters,
            skip_flops=args.skip_flops,
        )
        rows.append(
            {
                "gpu": _gpu_name(),
                "steps": int(steps),
                "dtype": str(dtype).replace("torch.", ""),
                "torch_compile": bool(args.use_torch_compile),
                "use_prefix_kv_cache": bool(not args.no_prefix_kv_cache),
                "latency_ms": round(result.latency_ms, 4),
                "peak_reserved_gb": round(result.peak_reserved_gb, 4),
                "gflops": None if result.gflops is None else round(result.gflops, 4),
                "note": result.flops_note or "",
            }
        )
        # Ensure input tensors are freed before loading the next model variant,
        # otherwise VRAM can accumulate across steps and OOM on smaller GPUs.
        del inputs
        del model
        torch.cuda.empty_cache()

    _write_csv(out_csv, rows)
    print(f"Wrote: {out_csv}")


def cmd_compare_presets(args: argparse.Namespace) -> None:
    _set_default_caches()
    _require_cuda()

    out_json = Path(args.out_json).expanduser().resolve()
    device = torch.device("cuda")

    results: Dict[str, Any] = {
        "gpu": _gpu_name(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rows": [],
    }

    for preset_name in args.presets:
        preset = _preset(preset_name)
        dtype = _select_dtype(use_bf16=preset["use_bf16"], use_fp16=preset["use_fp16"])

        cfg = _load_cfg(args.config)
        _apply_overrides(cfg, num_inference_steps=preset["num_inference_steps"], max_seq_len=0)

        model = _load_model(
            cfg=cfg,
            checkpoint_path=args.checkpoint,
            dtype=dtype,
            device=device,
            use_torch_compile=preset["use_torch_compile"],
        )
        inputs = _prepare_inputs(
            cfg=cfg,
            model=model,
            prompt=args.prompt,
            image_path=args.image,
            dtype=dtype,
            device=device,
            proprio_mode=args.proprio_mode,
            use_prefix_kv_cache=preset["use_prefix_kv_cache"],
            max_seq_len_override=0,
        )
        result = bench(
            model=model,
            inputs=inputs,
            warmup=args.warmup,
            iters=args.iters,
            skip_flops=args.skip_flops,
        )
        results["rows"].append(
            {
                "preset": preset_name,
                "dtype": str(dtype).replace("torch.", ""),
                "torch_compile": bool(preset["use_torch_compile"]),
                "use_prefix_kv_cache": bool(preset["use_prefix_kv_cache"]),
                "num_inference_steps": int(getattr(cfg, "num_inference_steps", -1)),
                "latency_ms": round(result.latency_ms, 4),
                "peak_reserved_gb": round(result.peak_reserved_gb, 4),
                "gflops": None if result.gflops is None else round(result.gflops, 4),
                "note": result.flops_note or "",
            }
        )
        del inputs
        del model
        torch.cuda.empty_cache()

    # Add a speedup helper if there are exactly 2 presets.
    if len(results["rows"]) == 2:
        a, b = results["rows"]
        if a["latency_ms"] and b["latency_ms"]:
            results["speedup"] = round(a["latency_ms"] / b["latency_ms"], 4)

    _write_json(out_json, results)
    print(f"Wrote: {out_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common_io(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=str, default="config/eval/bridge.yaml")
        p.add_argument("--checkpoint", type=str, required=True)
        p.add_argument(
            "--image",
            type=str,
            default=str(open_pi_zero_root() / "media" / "maniskill_pp.png"),
        )
        p.add_argument(
            "--prompt",
            type=str,
            default="In: What action should the robot take?\nOut:",
        )
        p.add_argument("--proprio-mode", type=str, default="zeros", choices=["zeros", "random"])
        p.add_argument("--warmup", type=int, default=5)
        p.add_argument("--iters", type=int, default=50)

    p_prompt = sub.add_parser("prompt-sweep", help="Sweep prompt length vs latency (KV cache on/off).")
    add_common_io(p_prompt)
    p_prompt.add_argument("--out-csv", type=str, required=True)
    p_prompt.add_argument("--text-tokens", type=int, nargs="+", required=True)
    p_prompt.add_argument("--token", type=str, default="move")
    p_prompt.add_argument("--num-inference-steps", type=int, default=10)
    p_prompt.add_argument("--use-bf16", action="store_true")
    p_prompt.add_argument("--use-fp16", action="store_true")
    p_prompt.add_argument("--use-torch-compile", action="store_true")
    p_prompt.add_argument("--skip-flops", action="store_true")
    p_prompt.set_defaults(func=cmd_prompt_sweep)

    p_fvs = sub.add_parser("first-vs-steady", help="Measure 1st call vs steady-state latency.")
    add_common_io(p_fvs)
    p_fvs.add_argument("--out-csv", type=str, required=True)
    p_fvs.add_argument("--presets", type=str, nargs="+", default=["baseline", "blurr"])
    p_fvs.set_defaults(func=cmd_first_vs_steady)

    p_steps = sub.add_parser("steps-sweep", help="Sweep num_inference_steps vs latency.")
    add_common_io(p_steps)
    p_steps.add_argument("--out-csv", type=str, required=True)
    p_steps.add_argument("--steps", type=int, nargs="+", required=True)
    p_steps.add_argument("--use-bf16", action="store_true")
    p_steps.add_argument("--use-fp16", action="store_true")
    p_steps.add_argument("--use-torch-compile", action="store_true")
    p_steps.add_argument("--no-prefix-kv-cache", action="store_true")
    p_steps.add_argument("--skip-flops", action="store_true")
    p_steps.set_defaults(func=cmd_steps_sweep)

    p_cmp = sub.add_parser("compare-presets", help="Benchmark named presets and write JSON.")
    add_common_io(p_cmp)
    p_cmp.add_argument("--out-json", type=str, required=True)
    p_cmp.add_argument("--presets", type=str, nargs="+", default=["baseline", "blurr"])
    p_cmp.add_argument("--skip-flops", action="store_true")
    p_cmp.set_defaults(func=cmd_compare_presets)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
