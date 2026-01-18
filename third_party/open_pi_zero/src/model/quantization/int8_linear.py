"""Lightweight per-channel INT8 linear quantization utilities for inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Int8QuantConfig:
    activation_clip: Optional[float] = None
    cache_fp_weight: bool = False
    fp_dtype: str = "bfloat16"


class QuantizedLinear(nn.Module):
    """Wraps an nn.Linear with simple per-channel symmetric INT8 weights."""

    def __init__(
        self,
        weight_i8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation_clip: Optional[float] = None,
        weight_fp: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("weight_i8", weight_i8)
        self.register_buffer("weight_scale", weight_scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.activation_clip = activation_clip
        if weight_fp is not None:
            self.register_buffer("weight_fp", weight_fp)
        else:
            self.weight_fp = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: Int8QuantConfig,
    ) -> "QuantizedLinear":
        with torch.no_grad():
            weight = linear.weight.detach().to(torch.float32)
            max_val = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
            scale = max_val / 127.0
            quantized = torch.clamp((weight / scale).round(), -128, 127).to(torch.int8)
            bias = (
                linear.bias.detach().to(torch.float32)
                if linear.bias is not None
                else None
            )
        weight_fp = None
        if config.cache_fp_weight:
            fp_dtype = getattr(torch, config.fp_dtype, torch.bfloat16)
            weight_fp = (quantized.to(torch.float32) * scale).to(fp_dtype)
        return cls(
            quantized,
            scale.squeeze(1),
            bias,
            config.activation_clip,
            weight_fp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_clip is not None:
            x = torch.clamp(x, -self.activation_clip, self.activation_clip)
        if self.weight_fp is not None:
            weight = self.weight_fp.to(x.dtype)
        else:
            weight = (
                self.weight_i8.to(torch.float32)
                * self.weight_scale.view(-1, 1)
            ).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


def quantize_module_int8(
    module: nn.Module,
    config: Optional[Int8QuantConfig] = None,
) -> None:
    """In-place swap of nn.Linear layers under module with QuantizedLinear."""

    if config is None:
        config = Int8QuantConfig()

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            quant_child = QuantizedLinear.from_linear(child, config)
            setattr(module, name, quant_child)
        else:
            quantize_module_int8(child, config)
