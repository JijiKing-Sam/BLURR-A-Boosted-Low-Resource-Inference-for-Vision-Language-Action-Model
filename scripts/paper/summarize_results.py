#!/usr/bin/env python3
"""
Summarize paper experiment CSV/JSON outputs into compact LaTeX tables.

This is intended to keep the camera-ready edits low-effort: run the paper
experiments, then run this script and paste the generated LaTeX into BLURR.tex.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _fmt(x: Any, digits: int = 2) -> str:
    if x is None:
        return "--"
    try:
        if isinstance(x, str):
            s = x.strip()
            if s == "" or s.lower() == "none":
                return "--"
            return f"{float(s):.{digits}f}"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _latex_table(rows: List[List[str]], headers: List[str], caption: str, label: str) -> str:
    colspec = "l" + "c" * (len(headers) - 1)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def summarize_prompt_sweep(path: Path) -> str:
    data = _read_csv(path)
    by_len: Dict[int, Dict[bool, Dict[str, str]]] = {}
    for row in data:
        k = int(row["target_text_tokens"])
        use_cache = row["use_prefix_kv_cache"].lower() == "true"
        by_len.setdefault(k, {})[use_cache] = row

    rows = []
    for k in sorted(by_len.keys()):
        no_cache = by_len[k].get(False)
        yes_cache = by_len[k].get(True)
        lat0 = float(no_cache["latency_ms"]) if no_cache and no_cache["latency_ms"] else None
        lat1 = float(yes_cache["latency_ms"]) if yes_cache and yes_cache["latency_ms"] else None
        speedup = (lat0 / lat1) if (lat0 and lat1) else None
        rows.append(
            [
                str(k),
                _fmt(lat0, 1),
                _fmt(lat1, 1),
                _fmt(speedup, 2) + r"$\times$" if speedup is not None else "--",
            ]
        )

    caption = (
        "Prompt-length scaling on Pi-0 microbench (same image, $\\mathrm{steps}=10$). "
        "Prefix KV caching reduces prompt-dependent recomputation."
    )
    return _latex_table(
        rows,
        headers=["Text tokens", "No cache (ms)", "KV cache (ms)", "Speedup"],
        caption=caption,
        label="tab:prompt-sweep",
    )


def summarize_first_vs_steady(path: Path) -> str:
    data = _read_csv(path)
    # Expect one row per preset (baseline/blurr)
    rows = []
    for row in data:
        rows.append(
            [
                row["preset"],
                row["dtype"],
                "Y" if row["torch_compile"].lower() == "true" else "N",
                _fmt(row.get("first_call_ms"), 1),
                _fmt(row.get("steady_latency_ms"), 1),
            ]
        )
    caption = "First-call vs. steady-state latency (first call includes compile/warmup overhead when enabled)."
    return _latex_table(
        rows,
        headers=["Preset", "Dtype", "Compile", "First (ms)", "Steady (ms)"],
        caption=caption,
        label="tab:first-vs-steady",
    )


def summarize_horizon_sweep(path: Path) -> str:
    data = _read_csv(path)
    by_steps: Dict[int, List[float]] = {}
    for row in data:
        steps = int(row["steps"])
        by_steps.setdefault(steps, []).append(float(row["success_rate"]))
    rows = []
    for steps in sorted(by_steps.keys()):
        rows.append([str(steps), _fmt(mean(by_steps[steps]), 2)])
    caption = "Closed-loop success vs. flow steps (average over tasks; higher steps increase compute)."
    return _latex_table(
        rows,
        headers=["Steps", "Avg. success"],
        caption=caption,
        label="tab:horizon-sweep",
    )


def summarize_steps_tradeoff(*, horizon_csv: Path, steps_latency_csv: Path) -> str:
    horizon = _read_csv(horizon_csv)
    by_steps: Dict[int, List[float]] = {}
    for row in horizon:
        steps = int(row["steps"])
        by_steps.setdefault(steps, []).append(float(row["success_rate"]))

    steps_lat = _read_csv(steps_latency_csv)
    lat_by_steps: Dict[int, Dict[str, Any]] = {}
    for row in steps_lat:
        lat_by_steps[int(row["steps"])] = row

    rows = []
    for steps in sorted(by_steps.keys()):
        lat_row = lat_by_steps.get(steps, {})
        rows.append(
            [
                str(steps),
                _fmt(lat_row.get("latency_ms"), 1),
                _fmt(lat_row.get("peak_reserved_gb"), 2),
                _fmt(mean(by_steps[steps]), 2),
            ]
        )

    caption = (
        "Trade-off between flow steps and performance. Latency/VRAM are microbench numbers; "
        "success is averaged over tasks in SimplerEnv."
    )
    return _latex_table(
        rows,
        headers=["Steps", "Latency (ms)", "VRAM (GB)", "Avg. success"],
        caption=caption,
        label="tab:steps-tradeoff",
    )


def summarize_hf_microbench(path: Path) -> str:
    data = _read_csv(path)
    gpus = sorted({row.get("gpu", "").strip() for row in data if row.get("gpu")})
    gpu_note = f" (GPU: {gpus[0].replace('_', r'\\_')})" if len(gpus) == 1 else ""
    # Group by model_id and profile
    grouped: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in data:
        grouped.setdefault(row["model_id"], {})[row["profile"]] = row

    def _first_present_profile(candidates: List[str]) -> str | None:
        for name in candidates:
            for model_rows in grouped.values():
                row = model_rows.get(name)
                if not row:
                    continue
                val = row.get("latency_ms")
                if val not in (None, "", "None"):
                    return name
        return None

    base_profile = _first_present_profile(["fp32_eager", "bf16_eager", "fp16_eager"])
    comp_profile = _first_present_profile(["bf16_compile", "fp16_compile"])

    if base_profile is None or comp_profile is None:
        raise ValueError(
            "HF microbench CSV missing required profiles. "
            "Need an eager profile (fp32_eager/bf16_eager/fp16_eager) and a compile profile "
            "(bf16_compile/fp16_compile)."
        )

    def _label_from_row(profile: str) -> str:
        for model_rows in grouped.values():
            row = model_rows.get(profile)
            if not row:
                continue
            if row.get("latency_ms") in (None, "", "None"):
                continue
            dtype = (row.get("dtype") or "").upper()
            compile_flag = (row.get("torch_compile") or "").lower() == "true"
            if compile_flag:
                return f"{dtype}+compile (ms)" if dtype else "Compile (ms)"
            return f"{dtype} eager (ms)" if dtype else "Eager (ms)"
        return profile

    rows = []
    for model_id in sorted(grouped.keys()):
        a = grouped[model_id].get(base_profile)
        b = grouped[model_id].get(comp_profile)
        lat_a = float(a["latency_ms"]) if a and a.get("latency_ms") not in ("", "None", None) else None
        lat_b = float(b["latency_ms"]) if b and b.get("latency_ms") not in ("", "None", None) else None
        speed = (lat_a / lat_b) if (lat_a and lat_b) else None
        rows.append(
            [
                model_id.replace("_", r"\_"),
                _fmt(lat_a, 1),
                _fmt(lat_b, 1),
                _fmt(speed, 2) + r"$\times$" if speed is not None else "--",
            ]
        )

    caption = f"Cross-model microbench on HuggingFace VLAs (same prompt/image){gpu_note}."
    return _latex_table(
        rows,
        headers=[
            "Model",
            _label_from_row(base_profile),
            _label_from_row(comp_profile),
            "Speedup",
        ],
        caption=caption,
        label="tab:hf-microbench",
    )


def summarize_cross_hardware(paths: List[Path]) -> str:
    rows = []
    for path in paths:
        payload = json.load(path.open("r", encoding="utf-8"))
        gpu = payload.get("gpu", "unknown")
        rows_payload = payload.get("rows", [])
        by_preset = {r["preset"]: r for r in rows_payload}
        base = by_preset.get("baseline")
        blurr = by_preset.get("blurr")
        lat0 = float(base["latency_ms"]) if base else None
        lat1 = float(blurr["latency_ms"]) if blurr else None
        speed = (lat0 / lat1) if (lat0 and lat1) else None
        rows.append(
            [
                gpu.replace("_", r"\_"),
                _fmt(lat0, 1),
                _fmt(lat1, 1),
                _fmt(speed, 2) + r"$\times$" if speed is not None else "--",
            ]
        )

    caption = "Cross-hardware Pi-0 microbench (baseline vs. BLURR preset; same script/config)."
    return _latex_table(
        rows,
        headers=["GPU", "Baseline (ms)", "BLURR (ms)", "Speedup"],
        caption=caption,
        label="tab:cross-hardware",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=str, required=True, help="Directory with CSV outputs.")
    parser.add_argument(
        "--cross-hardware-json",
        type=str,
        nargs="*",
        default=[],
        help="Optional: one or more pi0_*.json produced by run_cross_hardware_pi0.sbatch.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    parts = []

    prompt_csv = results_dir / "prompt_length_sweep_pi0.csv"
    if prompt_csv.is_file():
        parts.append(summarize_prompt_sweep(prompt_csv))

    first_csv = results_dir / "first_vs_steady_pi0.csv"
    if first_csv.is_file():
        parts.append(summarize_first_vs_steady(first_csv))

    horizon_csv = results_dir / "horizon_sweep_success.csv"
    steps_latency_csv = results_dir / "steps_latency_pi0_bf16_compile.csv"
    if horizon_csv.is_file() and steps_latency_csv.is_file():
        parts.append(summarize_steps_tradeoff(horizon_csv=horizon_csv, steps_latency_csv=steps_latency_csv))
    elif horizon_csv.is_file():
        parts.append(summarize_horizon_sweep(horizon_csv))

    hf_csv = results_dir / "hf_microbench.csv"
    if hf_csv.is_file():
        parts.append(summarize_hf_microbench(hf_csv))

    if args.cross_hardware_json:
        hw_paths = [Path(p).expanduser().resolve() for p in args.cross_hardware_json]
        hw_paths = [p for p in hw_paths if p.is_file()]
        if hw_paths:
            parts.append(summarize_cross_hardware(hw_paths))

    if not parts:
        raise SystemExit(f"No known result files found under: {results_dir}")

    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
