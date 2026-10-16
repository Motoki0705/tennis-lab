"""Verify saved BLCS predictions, including CUDA bf16 metric arithmetic, on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.dataset_audit.read_text())
    path = args.predictions_dir / "pred_test.npz"
    with np.load(path, allow_pickle=False) as saved:
        pred = saved["pred_position"]
        target = saved["target_position"]
        mask = saved["mask"]
        ids = saved["scene_ids"].tolist()
    assert ids == audit["splits"]["test"]["eligible_t128"]
    assert pred.shape == target.shape == (88, 128, 3)
    assert mask.shape == (88, 128) and mask.dtype == np.bool_
    assert np.isfinite(pred).all() and np.isfinite(target).all()
    p = torch.from_numpy(pred).to(torch.bfloat16)
    assert np.array_equal(p.float().numpy(), pred)
    t = torch.from_numpy(target)
    valid = torch.from_numpy(mask)
    # Original CUDA autocast: dtype-preserving denormalization, identity
    # court-frame matmul casts the target to bf16, subtraction stays bf16,
    # then pow/sum/sqrt run in float32. Preserve the original batch-2 reduction.
    p_m = p * p.new_tensor([11.885] * 3)
    t_m = (t * t.new_tensor([11.885] * 3)).to(torch.bfloat16)
    error = (p_m - t_m).float()
    norm = torch.sqrt((error**2).sum(-1) + 1e-8)
    endpoints = [float(norm[i, torch.nonzero(valid[i])[-1, 0]]) for i in range(88)]
    computed = {
        "position_error_m": sum(float(norm[i:i+2][valid[i:i+2]].sum()) for i in range(0, 88, 2)) / int(valid.sum()),
        "position_accuracy_0.3m": int((norm[valid] < .3).sum()) / int(valid.sum()),
        "endpoint_error_m": sum(endpoints) / len(endpoints),
    }
    reported = json.loads((args.predictions_dir / "metrics.json").read_text())
    for name, value in computed.items():
        assert np.isclose(value, reported[name], rtol=1e-6, atol=1e-7), (name, value, reported[name])
    physical_error = np.linalg.norm((pred.astype(np.float64) - target.astype(np.float64)) * 11.885, axis=-1)
    float64_metrics = {
        "position_error_m": float(physical_error[mask].mean()),
        "position_accuracy_0.3m": float((physical_error[mask] < .3).mean()),
        "endpoint_error_m": float(np.mean([physical_error[i, np.flatnonzero(mask[i])[-1]] for i in range(88)])),
    }
    result = {
        "scene_count": len(ids), "valid_frames": int(mask.sum()), "all_frames": int(mask.size),
        "scene_ids_equal_frozen_test_order": True, "all_predictions_finite": True,
        "reported": reported, "recomputed_cuda_bf16_arithmetic_on_cpu": computed,
        "float64_saved_coordinate_diagnostic": float64_metrics,
        "diagnostic_note": "Float64 recomputation of stored normalized coordinates omits the original bf16 metric rounding; it is not a second inference or a checkpoint selection criterion.",
        "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
