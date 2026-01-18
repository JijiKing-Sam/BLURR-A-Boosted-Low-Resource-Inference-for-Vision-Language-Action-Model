#!/usr/bin/env python3
import csv
import re
from pathlib import Path

from blurr.paths import repo_root

ROOT = repo_root()
RUNS_DIR = ROOT / "runs" / "eval_bridge"
OUT_CSV = ROOT / "runs" / "bridge_eval_summary.csv"

success_re = re.compile(r"Success rate:\s*([0-9.]+)")
episodes_re = re.compile(r"Number of episodes:\s*([0-9]+)")

rows = []

if not RUNS_DIR.is_dir():
    print(f"Runs directory not found: {RUNS_DIR}")
else:
    for model_dir in RUNS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name  # e.g. accel_step1_ta1_42 or interleave_default_ta4_42

        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run_log = run_dir / "run.log"
            if not run_log.is_file():
                continue

            base = run_dir.name
            parts = base.split("_")
            if len(parts) >= 4:
                task_name = "_".join(parts[:-2])
            else:
                task_name = base

            success = None
            episodes = None

            with run_log.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m_s = success_re.search(line)
                    if m_s:
                        success = float(m_s.group(1))
                    m_e = episodes_re.search(line)
                    if m_e:
                        episodes = int(m_e.group(1))

            if success is None:
                continue

            rows.append(
                {
                    "model": model_name,
                    "task": task_name,
                    "success_rate": success,
                    "episodes": episodes,
                    "run_dir": str(run_dir),
                }
            )

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["model", "task", "success_rate", "episodes", "run_dir"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT_CSV}")
