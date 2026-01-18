"""Quantization utility modules."""

from .int8_linear import Int8QuantConfig, quantize_module_int8

try:  # pragma: no cover - optional dependency
    from .bnb_linear import BNBLinearINT8, quantize_module_bnb_int8
except Exception:  # pragma: no cover - gracefully handle missing bitsandbytes
    BNBLinearINT8 = None

    def quantize_module_bnb_int8(*args, **kwargs):
        raise RuntimeError(
            "bitsandbytes is required for `bnb_int8` quantization but is not available in "
            "this environment. Install bitsandbytes with a compatible libstdc++ or "
            "switch `action_quantization.mode` to `int8`."
        )

__all__ = [
    "Int8QuantConfig",
    "quantize_module_int8",
    "BNBLinearINT8",
    "quantize_module_bnb_int8",
]
