from __future__ import annotations

import sys

from .paths import open_pi_zero_root


def ensure_open_pi_zero_on_path() -> None:
    root = open_pi_zero_root()
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)

