from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def open_pi_zero_root() -> Path:
    return repo_root() / "third_party" / "open_pi_zero"

