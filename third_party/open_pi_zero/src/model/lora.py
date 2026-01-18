"""
Minimal `get_layer()` helper.

This repository vendors a small subset of the open-pi-zero project to reproduce
the BLURR inference stack. The original open-pi-zero implementation includes
LoRA and 4-bit quantization layers backed by bitsandbytes.

For the BLURR demo + benchmark release we keep the default `nn.Linear` path and
avoid a hard dependency on bitsandbytes.
"""

from __future__ import annotations

from torch import nn


def get_layer(
    quantize: bool = False,
    lora: bool = False,
    r: int = 32,
    dropout: float = 0.05,
):
    if quantize or lora:
        raise RuntimeError(
            "This BLURR minimal release does not ship LoRA/4-bit layers. "
            "Set `quantize=False` and `lora=False` in configs, or use the full "
            "open-pi-zero implementation."
        )
    _ = (r, dropout)
    return nn.Linear

