"""Reproduce the read-only scout probe on commit 4671f9ba (CPU, no optimizer).

Run from that checkout with PYTHONPATH=. and OMP/MKL/OPENBLAS_NUM_THREADS=2.
The checkpoint/config inputs are the two completed GPU smoke runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.triangulation_residual.configuration import validate_config
from src.tasks.base.triangulation_residual.data import ResidualDataModule
from src.tasks.base.triangulation_residual.training import ResidualLightningModule

torch.set_num_threads(2)
ROOT = Path("/home/kamimura/projects/tennis-lab/outputs")


def quant(value: Any) -> dict[str, float | None]:
    values = np.asarray(value, dtype=np.float64)
    return {
        str(q): float(np.quantile(values, q)) if values.size else None
        for q in (0.5, 0.95, 1.0)
    }


def delta_stats(
    y: torch.Tensor,
    y0: torch.Tensor,
    reference: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, Any]:
    delta = (y.float() - y0.float()).detach().numpy()
    norm = np.linalg.norm(delta, axis=-1).reshape(-1)
    embed_norm = np.linalg.norm(y.float().detach().numpy(), axis=-1).reshape(-1)
    result: dict[str, Any] = {
        "token_norm": quant(norm),
        "mean_token_norm": float(norm.mean()),
        "mean_relative_to_embed": float((norm / np.maximum(embed_norm, 1e-12)).mean()),
        "exact_equal_entry_frac": float(np.mean(delta == 0)),
    }
    if reference is not None:
        ref = (reference[0].float() - reference[1].float()).detach().numpy()
        nonzero = ref.reshape(-1) != 0
        result["fp32_nonzero_but_this_zero_frac"] = float(
            np.mean(delta.reshape(-1)[nonzero] == 0)
        )
        result["delta_rmse_vs_fp32"] = float(np.sqrt(np.mean((delta - ref) ** 2)))
        cosine = np.sum(delta * ref, axis=-1) / (
            np.linalg.norm(delta, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-20
        )
        valid = (np.linalg.norm(delta, axis=-1) > 0) & (
            np.linalg.norm(ref, axis=-1) > 0
        )
        result["cosine_median_vs_fp32"] = float(np.median(cosine[valid]))
    return result


for task in ("plcs", "blcs"):
    run = ROOT / task / "triangulation_residual_v2_gpu_smoke_20260921"
    config = cast(DictConfig, OmegaConf.load(run / "config.yaml"))
    checkpoint = torch.load(
        json.loads((run / "evaluation.json").read_text())["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    module = ResidualLightningModule(config)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    model = module.model.cpu().eval()
    data_config = cast(DictConfig, OmegaConf.load(run / "config.yaml"))
    for split in ("train", "val", "test"):
        data_config.data[f"{split}_limit"] = 0
    data_config.data.min_views = 2
    data_config.data.max_views = 6
    data_config.data.num_workers = 0
    data_config.v2.evaluation_views = 3
    datamodule = ResidualDataModule(validate_config(data_config))
    datamodule.setup()
    selected: list[tuple[int, str, dict[str, Any]]] = []
    counts = {"calibration": 0, "observation": 0, "hard_combined": 0}
    for index in range(min(len(datamodule.val_dataset), 120)):
        sample = datamodule.val_dataset[index]
        family, severity = int(sample["corruption_family"]), float(sample["severity"])
        label = None
        if family == 1 and severity == 1 and counts["calibration"] < 2:
            label = "calibration"
        elif family == 2 and severity == 1 and counts["observation"] < 2:
            label = "observation"
        elif family == 5 and severity > 1 and counts["hard_combined"] < 4:
            label = "hard_combined"
        if label:
            counts[label] += 1
            selected.append((index, label, sample))
        if len(selected) == 8:
            break
    assert [row[0] for row in selected] == [1, 3, 4, 12, 16, 22, 25, 48]
    joints = 17 if task == "plcs" else 1
    sl = slice(4 * joints, 6 * joints)
    residual_values: list[float] = []
    underflow: list[bool] = []
    rows: list[dict[str, Any]] = []
    modes: dict[str, list[dict[str, Any]]] = {
        key: [] for key in ("raw_fp32", "raw_bf16", "asinh_fp32", "asinh_bf16")
    }
    for index, label, sample in selected:
        features = sample["features"].unsqueeze(0).float()
        zeroed = features.clone()
        zeroed[..., sl] = 0
        scaled = features.clone()
        scaled[..., sl] = torch.asinh(100 * scaled[..., sl])
        scaled_zero = scaled.clone()
        scaled_zero[..., sl] = 0
        residual = features[..., sl].numpy()
        nonzero = residual != 0
        residual_values.extend(np.abs(residual[nonzero]).tolist())
        underflow.extend(
            (
                features[..., sl].to(torch.bfloat16).float().numpy()[nonzero] == 0
            ).tolist()
        )
        with torch.inference_mode():
            raw32, zero32 = model.embed(features), model.embed(zeroed)
            scaled32, scaled_zero32 = model.embed(scaled), model.embed(scaled_zero)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                raw16, zero16 = model.embed(features), model.embed(zeroed)
                scaled16, scaled_zero16 = model.embed(scaled), model.embed(scaled_zero)
        stats = {
            "raw_fp32": delta_stats(raw32, zero32),
            "raw_bf16": delta_stats(raw16, zero16, (raw32, zero32)),
            "asinh_fp32": delta_stats(scaled32, scaled_zero32),
            "asinh_bf16": delta_stats(
                scaled16, scaled_zero16, (scaled32, scaled_zero32)
            ),
        }
        for name, values in stats.items():
            modes[name].append(values)
        rows.append(
            {
                "index": index,
                "family": label,
                "severity": float(sample["severity"]),
                "statistics": stats,
            }
        )
    aggregate = {}
    for mode, statistics in modes.items():
        keys = ["mean_token_norm", "mean_relative_to_embed", "exact_equal_entry_frac"]
        if mode.endswith("bf16"):
            keys += [
                "fp32_nonzero_but_this_zero_frac",
                "delta_rmse_vs_fp32",
                "cosine_median_vs_fp32",
            ]
        aggregate[mode] = {
            key: float(np.mean([entry[key] for entry in statistics])) for key in keys
        }
    print(
        json.dumps(
            {
                "task": task,
                "selected_counts": counts,
                "selected_indices": [x[0] for x in selected],
                "residual_nonzero_abs": quant(residual_values),
                "residual_input_bf16_underflow_zero_frac": float(np.mean(underflow)),
                "embed_aggregate": aggregate,
                "samples": rows,
            },
            indent=2,
        )
    )
