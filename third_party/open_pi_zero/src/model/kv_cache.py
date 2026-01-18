from typing import List, Optional, Tuple

import torch


class KVCache:
    def __init__(self, quant_config: Optional[dict] = None) -> None:
        """list for layers"""
        self.quant_mode: Optional[str] = None
        self.quant_clip: Optional[float] = None
        self.quant_dtype = torch.bfloat16
        if quant_config:
            mode = str(quant_config.get("mode", "")).lower()
            if mode in {"int8"}:
                self.quant_mode = mode
                self.quant_clip = quant_config.get("activation_clip")
                dtype_str = quant_config.get("dtype", "bfloat16")
                self.quant_dtype = getattr(torch, dtype_str, torch.bfloat16)
        self.key_cache: List = []
        self.value_cache: List = []
        if self.quant_mode == "int8":
            self.key_scale: List = []
            self.value_scale: List = []

    def has_item(self, layer_idx) -> bool:
        if len(self.key_cache) <= layer_idx:
            return False
        if self.quant_mode == "int8":
            return len(self.key_cache[layer_idx]) > 0
        return True

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            if self.quant_mode == "int8":
                return sum(chunk.shape[-2] for chunk in self.key_cache[0])
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def get(self, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.quant_mode == "int8":
            key = self._dequantize_chunks(
                self.key_cache[layer_idx], self.key_scale[layer_idx]
            )
            value = self._dequantize_chunks(
                self.value_cache[layer_idx], self.value_scale[layer_idx]
            )
            return key, value
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.quant_mode == "int8":
            quant_key, scale_key = self._quantize_chunk(key_states)
            quant_value, scale_value = self._quantize_chunk(value_states)
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append([quant_key])
                self.value_cache.append([quant_value])
                self.key_scale.append([scale_key])
                self.value_scale.append([scale_value])
            else:
                self.key_cache[layer_idx].append(quant_key)
                self.value_cache[layer_idx].append(quant_value)
                self.key_scale[layer_idx].append(scale_key)
                self.value_scale[layer_idx].append(scale_value)
            return self.get(layer_idx)

        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _quantize_chunk(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        orig = tensor
        if self.quant_clip is not None:
            tensor = torch.clamp(tensor, -self.quant_clip, self.quant_clip)
        scale = tensor.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-6) / 127.0
        quant = torch.clamp((tensor / scale).round(), -128, 127).to(torch.int8)
        # store scale with shape [B, H, 1, 1]
        scale = scale.squeeze(-1).squeeze(-1).to(orig.dtype)
        return quant, scale

    def _dequantize_chunks(self, chunks: List[torch.Tensor], scales: List[torch.Tensor]):
        restored = []
        for quant, scale in zip(chunks, scales):
            scale = scale.unsqueeze(-1).unsqueeze(-1).to(quant.device, torch.float32)
            chunk = quant.to(torch.float32) * scale
            restored.append(chunk.to(self.quant_dtype))
        return torch.cat(restored, dim=-2) if restored else None
