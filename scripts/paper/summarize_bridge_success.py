#!/usr/bin/env python3
"""
Summarize SimplerEnv Bridge success `summary.json` files into a compact table.

This is a small helper for camera-ready updates: pass one or more `summary.json`
paths produced by `scripts/eval_hf_vla_simpler.py` or `scripts/eval_prismatic_vla_simpler.py`,
and paste the printed LaTeX (or Markdown) into the paper appendix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


TASK_ORDER = [
    ("widowx_carrot_on_plate", "Carrot"),
    ("widowx_spoon_on_towel", "Spoon"),
    ("widowx_stack_cube", "Blocks"),
    ("widowx_put_eggplant_in_basket", "Eggplant"),
]


def _fmt(x: Any, digits: int = 2) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "--"


def _infer_label(payload: Dict[str, Any], path: Path) -> str:
    model_id = payload.get("model_id")
    if isinstance(model_id, str) and model_id.strip():
        preset = payload.get("preset")
        if isinstance(preset, str) and preset.strip():
            return f"{model_id} ({preset})"
        return model_id

    ckpt = payload.get("checkpoint")
    if isinstance(ckpt, str) and ckpt.strip():
        if "minivla" in ckpt.lower():
            return "MiniVLA (prismatic)"
        return Path(ckpt).name

    # Fallback: use parent directory name (usually includes model tag).
    return path.parent.name


def _read_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(payload: Dict[str, Any], *, label: str) -> List[str]:
    episodes = payload.get("episodes_per_task") or payload.get("n_eval_episode") or payload.get("episodes")
    episodes_str = str(int(episodes)) if isinstance(episodes, (int, float)) else "--"

    per_task: Dict[str, Any] = payload.get("per_task_success", {}) or {}
    avg = payload.get("avg_success", None)

    cells = [label, episodes_str]
    for key, _short in TASK_ORDER:
        cells.append(_fmt(per_task.get(key, None), 2))
    cells.append(_fmt(avg, 2))
    return cells


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


def _markdown_table(rows: List[List[str]], headers: List[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=str, nargs="+", help="One or more summary.json paths.")
    parser.add_argument("--format", type=str, default="latex", choices=["latex", "md"])
    parser.add_argument("--caption", type=str, default="Bridge task success rates on SimplerEnv (higher is better).")
    parser.add_argument("--label", type=str, default="tab:bridge-success-crossmodel")
    parser.add_argument(
        "--label-override",
        type=str,
        nargs="*",
        default=[],
        help="Optional per-summary label override (same length as summary list).",
    )
    args = parser.parse_args()

    paths = [Path(p).expanduser().resolve() for p in args.summary]
    overrides: List[Optional[str]] = list(args.label_override)
    if overrides and len(overrides) != len(paths):
        raise SystemExit("--label-override must be the same length as summary paths (or omitted).")
    overrides = overrides + [None] * (len(paths) - len(overrides))

    headers = ["Model", "Eps/task"] + [short for _key, short in TASK_ORDER] + ["Avg."]
    rows = []
    for path, override in zip(paths, overrides):
        payload = _read_summary(path)
        label = override or _infer_label(payload, path)
        rows.append(_row(payload, label=label))

    if args.format == "md":
        print(_markdown_table(rows, headers))
    else:
        print(_latex_table(rows, headers, caption=args.caption, label=args.label))


if __name__ == "__main__":
    main()

