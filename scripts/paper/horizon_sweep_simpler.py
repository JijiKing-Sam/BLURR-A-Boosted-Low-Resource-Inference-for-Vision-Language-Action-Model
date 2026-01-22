#!/usr/bin/env python3
"""
Run a control-horizon sweep (num_inference_steps) in SimplerEnv and write a CSV.

This script orchestrates multiple calls to `scripts/eval_pi0_simpler.py` so it
works with the existing open-pi-zero evaluation stack.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


SUCCESS_RE = re.compile(r"Success rate:\s*([0-9.]+)")
EPISODES_RE = re.compile(r"Number of episodes:\s*([0-9]+)")


DEFAULT_TASKS = [
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]


def _parse_run_log(run_log: Path) -> Dict:
    success = None
    episodes = None
    with run_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_s = SUCCESS_RE.search(line)
            if m_s:
                success = float(m_s.group(1))
            m_e = EPISODES_RE.search(line)
            if m_e:
                episodes = int(m_e.group(1))
    return {"success_rate": success, "episodes": episodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/eval/bridge.yaml")
    parser.add_argument("--preset", type=str, default="blurr")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 6, 10])
    parser.add_argument("--tasks", type=str, nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument(
        "--disable-torch-compile",
        action="store_true",
        help="Disable torch.compile during eval to reduce orchestration overhead (does not affect success).",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    for steps in args.steps:
        for task in args.tasks:
            log_dir = out_root / f"steps{steps}" / task
            log_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve().parents[1] / "eval_pi0_simpler.py"),
                "--preset",
                args.preset,
                "--config",
                args.config,
                "--task",
                task,
                "--checkpoint",
                args.checkpoint,
                "--seed",
                str(args.seed),
                "--gpu-id",
                str(args.gpu_id),
                "--n-eval-episode",
                str(args.episodes),
                "--n-video",
                "0",
                "--log-dir",
                str(log_dir),
                "--num-inference-steps",
                str(steps),
            ]
            if args.disable_torch_compile:
                cmd.append("--no-torch-compile")

            print("\n==> Running:", " ".join(cmd))
            subprocess.run(cmd, check=True, env=os.environ.copy())

            run_log = log_dir / "run.log"
            parsed = _parse_run_log(run_log)
            if parsed["success_rate"] is None:
                raise RuntimeError(f"Failed to parse success rate from {run_log}")

            rows.append(
                {
                    "steps": int(steps),
                    "task": task,
                    "episodes": parsed["episodes"],
                    "success_rate": parsed["success_rate"],
                    "log_dir": str(log_dir),
                }
            )

    # Write CSV
    import csv

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["steps", "task", "episodes", "success_rate", "log_dir"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote: {out_csv}\n")


if __name__ == "__main__":
    main()

