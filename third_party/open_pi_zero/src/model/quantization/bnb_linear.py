"""bitsandbytes-based INT8 linear layers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:  # pragma: no cover - optional dependency
    from bitsandbytes.nn import Linear8bitLt
except Exception:  # pragma: no cover
    Linear8bitLt = None  # type: ignore[assignment]


class BNBLinearINT8(nn.Module):
    """Wrap an nn.Linear with bitsandbytes' fused Linear8bitLt."""

    def __init__(
        self,
        linear: nn.Linear,
        threshold: float = 0.0,
        bias_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if Linear8bitLt is None:
            raise RuntimeError(
                "bitsandbytes is required for BNBLinearINT8 but is not installed."
            )
        self.impl = Linear8bitLt(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            False,
            threshold,
        )
        with torch.no_grad():
            self.impl.weight.data.copy_(linear.weight.data)
            if linear.bias is not None:
                target_dtype = bias_dtype or linear.bias.dtype
                self.impl.bias.data.copy_(linear.bias.data.to(target_dtype))
        self.impl.weight.requires_grad = False
        if self.impl.bias is not None:
            self.impl.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.bfloat16:
            return self.impl(x.to(torch.float16)).to(torch.bfloat16)
        return self.impl(x)


def quantize_module_bnb_int8(
    module: nn.Module,
    threshold: float = 0.0,
    bias_dtype: Optional[torch.dtype] = None,
) -> None:
    """In-place swap of nn.Linear layers to BNBLinearINT8."""

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                BNBLinearINT8(child, threshold=threshold, bias_dtype=bias_dtype),
            )
        else:
            quantize_module_bnb_int8(child, threshold=threshold, bias_dtype=bias_dtype)
