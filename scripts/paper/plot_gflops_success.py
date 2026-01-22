#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _read_avg_success(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["avg_success"])


def _workspace_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent


def _default_success_paths() -> Tuple[Path, Path]:
    ws = _workspace_root()
    openvla = ws / "blurr_paper_results/bridge_success_l40s_final_42_10_v2/openvla_bf16_eager/summary.json"
    # Default OpenVLA-OFT point matches the paper's HF microbench model_id
    # (Kaipengm2/openvla-oft-64-130000).
    openvla_oft = ws / "blurr_paper_results/bridge_success_l40s_final_42_10_v2/openvla_oft_bf16_eager/summary.json"
    return openvla, openvla_oft


def main() -> None:
    ws = _workspace_root()
    default_openvla, default_openvla_oft = _default_success_paths()

    parser = argparse.ArgumentParser(description="Generate compute-vs-success plot for BLURR paper.")
    parser.add_argument("--out", type=str, default=str(ws / "gflops_success.png"))
    parser.add_argument("--openvla-summary", type=str, default=str(default_openvla))
    parser.add_argument("--openvla-oft-summary", type=str, default=str(default_openvla_oft))
    parser.add_argument(
        "--openvla-success",
        type=float,
        default=None,
        help="Optional override for OpenVLA avg success (0-1).",
    )
    parser.add_argument(
        "--openvla-oft-success",
        type=float,
        default=None,
        help="Optional override for OpenVLA-OFT avg success (0-1).",
    )
    args = parser.parse_args()

    openvla_success = args.openvla_success
    if openvla_success is None:
        openvla_success = _read_avg_success(Path(args.openvla_summary))

    openvla_oft_success = args.openvla_oft_success
    if openvla_oft_success is None:
        openvla_oft_success = _read_avg_success(Path(args.openvla_oft_summary))

    # GFLOPS points match Table~\ref{tab:latency1} in the paper (H100, 224x224, 256 visual tokens).
    # Success for Pi-0 variants matches Table~\ref{tab:bridge-success}.
    points: Dict[str, Tuple[float, float]] = {
        "OpenVLA": (5835.0, float(openvla_success)),
        "OpenVLA-OFT": (49886.0, float(openvla_oft_success)),
        r"$\pi_0$ baseline": (39038.0, 0.70),
        r"Interleave-$\pi_0$": (7989.0, 0.70),
        r"BLURR-$\pi_0$": (73525.0, 0.71),
    }

    x_max = max(x for x, _ in points.values())
    y_scale = x_max  # scale success to match GFLOPS magnitude for geometry overlays

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)

    # Guide circles (quarter-circles in first quadrant) with scaled y-axis.
    theta = np.linspace(0.0, math.pi / 2.0, 256)
    for _name, (x, success) in points.items():
        y = success * y_scale
        r = math.sqrt(x * x + y * y)
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="#c7c7c7", linewidth=1.0, linestyle="--", zorder=0)

    # Ray through BLURR-pi0.
    blurr_x, blurr_success = points[r"BLURR-$\pi_0$"]
    blurr_y = blurr_success * y_scale
    slope = blurr_y / blurr_x if blurr_x > 0 else 0.0
    ray_x = np.linspace(0.0, x_max * 1.08, 128)
    ax.plot(ray_x, slope * ray_x, color="#f39c12", linewidth=1.5, linestyle="--", zorder=1)

    # Scatter points.
    colors = {
        "OpenVLA": "#1f77b4",
        "OpenVLA-OFT": "#9467bd",
        r"$\pi_0$ baseline": "#7f7f7f",
        r"Interleave-$\pi_0$": "#2ca02c",
        r"BLURR-$\pi_0$": "#d62728",
    }
    for name, (x, success) in points.items():
        y = success * y_scale
        is_blurr = name == r"BLURR-$\pi_0$"
        ax.scatter(
            [x],
            [y],
            s=80 if is_blurr else 50,
            color=colors.get(name, "#333333"),
            edgecolor="black",
            linewidths=0.6,
            zorder=3 if is_blurr else 2,
        )

        dx = 0.012 * x_max
        dy = 0.03 * y_scale
        ax.text(x + dx, y + dy, name, fontsize=9, ha="left", va="bottom")

    ax.set_xlim(0.0, x_max * 1.12)
    ax.set_ylim(0.0, y_scale * 1.05)
    ax.set_xlabel("GFLOPS")
    ax.set_ylabel("Avg. success")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    # Display success ticks in [0,1] even though we plot scaled values.
    success_ticks = np.linspace(0.0, 1.0, 6)
    ax.set_yticks((success_ticks * y_scale).tolist())
    ax.set_yticklabels([f"{t:.1f}" for t in success_ticks])

    # Format x ticks with thousands separators for readability.
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _pos: f"{int(v):,}" if v >= 1 else "0"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
