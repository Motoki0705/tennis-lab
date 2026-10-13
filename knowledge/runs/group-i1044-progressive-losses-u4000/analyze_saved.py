"""Post-hoc CPU diagnostics of frozen arrays; never reselect or run a model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.base.generate_dataset import build_physical_court_provenance
from src.tasks.blcs.experiments.issue1044 import ARMS, experiment_config
from src.tasks.blcs.model_io.factory import compose_blcs_model_io
from src.tasks.blcs.training.losses import projection_errors
from src.utils.schema.court_normalization import denormalize_court_position

BUNDLE = Path(__file__).resolve().parents[1] / "run-i1044-final-evaluation"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def population(error: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    squared = error[mask] ** 2
    return {
        "count": int(mask.sum()),
        "rmse": float(np.sqrt(squared.mean())) if len(squared) else None,
        "squared_error_sum": float(squared.sum()),
    }


def boundary_diagnostics(arm: str) -> dict[str, Any]:
    directory = BUNDLE / "evaluations" / f"{arm}-s42-u4000"
    with np.load(directory / "pred_test.npz", allow_pickle=False) as data:
        position = denormalize_court_position(data["pred_position"].astype(np.float64))
        target = denormalize_court_position(data["position_3d"].astype(np.float64))
        frame, segment = data["frame_index"], data["physics_segment"]
        valid = frame >= 0
        slot = np.broadcast_to(np.arange(frame.shape[1]), frame.shape)
        lengths = valid.sum(axis=1)
        boundary = np.zeros_like(valid)
        for index in range(128, frame.shape[1], 128):
            for near in (index - 1, index, index + 1):
                if near < frame.shape[1]:
                    boundary[:, near] |= (lengths > index) & valid[:, near]
        dt = (data["physics_clock"][:, 0] * data["physics_clock"][:, 3])[:, None, None]
        error = position - target
        velocity = np.linalg.norm(np.diff(error, axis=1) / dt, axis=-1)
        acceleration = np.linalg.norm(np.diff(error, n=2, axis=1) / dt**2, axis=-1)
        support1 = valid[:, :-1] & valid[:, 1:] & (segment[:, :-1] == segment[:, 1:])
        support2 = (
            valid[:, :-2]
            & valid[:, 1:-1]
            & valid[:, 2:]
            & (segment[:, :-2] == segment[:, 1:-1])
            & (segment[:, :-2] == segment[:, 2:])
        )
        cross1 = slot[:, :-1] // 128 != slot[:, 1:] // 128
        cross2 = slot[:, :-2] // 128 != slot[:, 2:] // 128
        result = {}
        saved = read_json(directory / "evaluation.json")["metrics"]
        for name, values, support, mask, metric in (
            ("position", np.linalg.norm(error, axis=-1), valid, boundary, "trajectory"),
            ("velocity", velocity, support1, cross1, "position_difference_velocity"),
            (
                "acceleration",
                acceleration,
                support2,
                cross2,
                "position_difference_acceleration",
            ),
        ):
            total = population(values, support)
            inside = population(values, support & mask)
            outside = population(values, support & ~mask)
            unit = {"position": "m", "velocity": "mps", "acceleration": "mps2"}[name]
            assert total["count"] == saved[f"{metric}_count"]
            assert np.isclose(total["rmse"], saved[f"{metric}_rmse_{unit}"], rtol=1e-5)
            result[name] = {
                "all": total,
                "boundary": inside,
                "rest": outside,
                "boundary_sse_fraction": inside["squared_error_sum"]
                / total["squared_error_sum"],
            }
        if arm == "reprojection":
            scene = list(data["scene_ids"]).index("scene_000459")
            result["example"] = {
                "scene_id": "scene_000459",
                "frames": [127, 128],
                "predicted_position_m": position[scene, 127:129].tolist(),
                "target_position_m": target[scene, 127:129].tolist(),
            }
        return result


def projection_outliers() -> dict[str, Any]:
    directory = BUNDLE / "evaluations/reprojection-s42-u4000"
    with np.load(directory / "pred_test.npz", allow_pickle=False) as data:
        scenes = [str(scene) for scene in data["scene_ids"]]
        batch = {
            key: torch.from_numpy(data[key].copy())
            for key in data.files
            if key != "scene_ids"
        }
    batch["court_reference_provenance"] = tuple(
        build_physical_court_provenance() for _ in scenes
    )
    config = experiment_config(
        data_root=Path("/tmp"), output="unused", arm="reprojection", seed=42
    )
    target = compose_blcs_model_io(config).adapter.build_training_batch(batch)
    _, pixels, supported, _ = projection_errors(batch["pred_position"], target)
    squared = pixels.double().square().sum(dim=-1).numpy()
    mask = supported.numpy()
    squared = np.where(mask, squared, 0)
    total = float(squared.sum())
    count = int(mask.sum())
    metrics = read_json(directory / "evaluation.json")["metrics"]
    assert count == metrics["reprojection_count"]
    assert np.isclose(
        np.sqrt(total / count), metrics["reprojection_rmse_px"], rtol=1e-6
    )
    clips: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes):
        frame_count = mask[index].sum(axis=0)
        frame_error = np.divide(
            squared[index].sum(axis=0),
            frame_count,
            out=np.full(frame_count.shape, np.nan),
            where=frame_count > 0,
        )
        worst = int(np.nanargmax(frame_error))
        clips.append(
            {
                "scene_id": scene,
                "observations": int(mask[index].sum()),
                "rmse_px": float(np.sqrt(squared[index].sum() / mask[index].sum())),
                "sse_fraction": float(squared[index].sum() / total),
                "worst_frame": worst,
                "worst_frame_rmse_px": float(np.sqrt(frame_error[worst])),
                "worst_view_at_that_frame_px": float(
                    np.sqrt(squared[index, :, worst].max())
                ),
            }
        )
    return {
        "observations": count,
        "rmse_px": float(np.sqrt(total / count)),
        "largest_sse_contributions": sorted(
            clips, key=lambda row: row["sse_fraction"], reverse=True
        )[:10],
        "largest_clip_rmse": sorted(
            clips, key=lambda row: row["rmse_px"], reverse=True
        )[:10],
    }


def main() -> None:
    torch.set_num_threads(1)
    reports = {
        path.parent.name: read_json(path)
        for path in (BUNDLE / "evaluations").glob("*/evaluation.json")
    }
    repeated = read_json(BUNDLE / "report/repeated_seeds.json")
    confirmation = {}
    for key in repeated["metrics"]:
        values = np.array(
            [
                [
                    reports[f"{arm}-s{seed}-u4000"]["metrics"][key]
                    for arm in ("reprojection", "parameters")
                ]
                for seed in (43, 44)
            ]
        )
        confirmation[key] = {
            "seeds": [43, 44],
            "control_mean": float(values[:, 0].mean()),
            "candidate_mean": float(values[:, 1].mean()),
            "control_sample_sd": float(values[:, 0].std(ddof=1)),
            "candidate_sample_sd": float(values[:, 1].std(ddof=1)),
            "paired_candidate_minus_control": (values[:, 1] - values[:, 0]).tolist(),
        }
    curves = read_json(BUNDLE / "report/training_curves.json")
    gradients: dict[str, dict[str, Any]] = {}
    for name, tags in curves.items():
        gradients[name] = {}
        for tag, entries in tags.items():
            if tag.startswith("gradient/") and tag.endswith("_weighted_l2"):
                values = [entry["value"] for entry in entries if entry["step"] >= 600]
                gradients[name][tag] = {
                    "recorded_steps_after_warmup": len(values),
                    "all_finite": all(
                        value is not None and np.isfinite(value) for value in values
                    ),
                    "all_positive": all(
                        value is not None and value > 0 for value in values
                    ),
                    "min": min(values),
                    "max": max(values),
                }
    result = {
        "schema": "blcs.issue1044.posthoc.v1",
        "scope": "Saved-array diagnostics after frozen test; no model/arm/checkpoint reselection",
        "boundary_definition": "Position: internal k=128n and k-1,k,k+1; differences: valid same-flight closed stencils whose endpoint window IDs differ",
        "boundary_dtype": "float64 arithmetic; all-population supports and RMSE checked against original float32-derived metrics",
        "boundaries_seed42": {arm: boundary_diagnostics(arm) for arm in ARMS},
        "control_test_reprojection_outliers": projection_outliers(),
        "confirmation_seeds_only": confirmation,
        "weighted_output_gradients_after_warmup": gradients,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
