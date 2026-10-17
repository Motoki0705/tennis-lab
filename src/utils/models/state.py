"""Content identity of dense CPU model tensors, independent of pickle/storage IDs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import torch
from torch import Tensor

__all__ = ["state_dict_sha256"]


def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash names, dtypes, shapes and bytes; callers explicitly move tensors to CPU.

    Strides and shared-storage layout do not change the logical weight identity.
    Sparse, quantized and non-tensor state require a separate declared format.
    """
    if any(not isinstance(name, str) for name in state):
        raise ValueError("model state must map string names to tensors")
    digest = hashlib.sha256(b"dense_cpu_state_dict.v1\n")
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise ValueError("model state must map string names to tensors")
        if tensor.device.type != "cpu" or tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError("model state hashing requires dense, unquantized CPU tensors")
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.detach().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(raw.nbytes.to_bytes(8, "big"))
        digest.update(memoryview(raw))
    return digest.hexdigest()
