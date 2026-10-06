"""Exercise the CUDA/compiler/attention runtime before loading VidMap weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import xformers
    from xformers.ops import memory_efficient_attention

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the VidMap runtime")
    torch.manual_seed(42)
    value = torch.randn(64, device="cuda")
    compiled = torch.compile(lambda tensor: tensor.sin() + tensor, fullgraph=True)
    result = compiled(value)
    torch.cuda.synchronize()
    if not torch.allclose(result, value.sin() + value, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Compiled CUDA output differs from eager output")
    queries = torch.randn(1, 16, 1, 64, device="cuda", dtype=torch.float16)
    attention = memory_efficient_attention(queries, queries, queries)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(attention).all()):
        raise RuntimeError("Attention output is not finite")
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "xformers": xformers.__version__,
        "device": torch.cuda.get_device_name(),
        "compiled_matches_eager": True,
        "attention_finite": True,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "scope": "runtime compatibility only; not an SfM quality measurement",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
