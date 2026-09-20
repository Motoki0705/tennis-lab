# CPython built-in SHA256 implementation; typeshed does not provide its stub.
import _sha256  # type: ignore[import-not-found]
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import torch

p = Path(sys.argv[1])
out = p.parent / "verification.json"
result = {
    "path": str(p),
    "expected_size": 2549075546,
    "expected_sha256": "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc",
    "hash_passes": 1,
}
try:
    a, b = hashlib.sha256(), _sha256.sha256()
    size = 0
    with p.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            a.update(chunk)
            b.update(chunk)
            size += len(chunk)
    result.update(size=size, sha256_openssl=a.hexdigest(), sha256_builtin=b.hexdigest())
    assert size == result["expected_size"], "size mismatch"
    assert a.hexdigest() == b.hexdigest() == result["expected_sha256"], (
        "digest mismatch"
    )
    with zipfile.ZipFile(p) as z:
        result["zip_members"] = len(z.infolist())
        result["zip_bad_member"] = z.testzip()
        assert result["zip_bad_member"] is None
    ckpt = torch.load(p, weights_only=True, map_location="cpu")
    result["top_level_keys"] = sorted(ckpt)
    state = ckpt["state_dict"]
    assert isinstance(state, dict)
    result["state_dict_entries"] = len(state)
    result["tensor_entries"] = sum(isinstance(x, torch.Tensor) for x in state.values())
    result["tensor_numel"] = sum(
        x.numel() for x in state.values() if isinstance(x, torch.Tensor)
    )
    result["tensor_shapes"] = {
        k: list(v.shape) for k, v in state.items() if isinstance(v, torch.Tensor)
    }
    result["nonfinite_tensors"] = [
        k
        for k, v in state.items()
        if isinstance(v, torch.Tensor)
        and v.is_floating_point()
        and not torch.isfinite(v).all().item()
    ]
    assert not result["nonfinite_tensors"]
    assert result["tensor_entries"] == len(state)
    assert state["backbone.patch_embed.proj.weight"].shape == (1280, 3, 16, 16)
    assert state["keypoint_head.final_layer.weight"].shape[0] == 17
    result["status"] = "passed"
except Exception as exc:
    result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    raise
finally:
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k != "tensor_shapes"}, indent=2),
        flush=True,
    )
