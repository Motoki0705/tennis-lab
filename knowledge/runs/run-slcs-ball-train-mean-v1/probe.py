"""CPU train-only weighted-mean ball diagnostic; never an L2-optimal constant."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.tasks.slcs.configuration import SLCSDataRuntimeConfig  # noqa: E402
from src.tasks.slcs.data.annotation import (  # noqa: E402
    SLCSDataIndex,
    slcs_annotation_dir,
)
from src.tasks.slcs.data.dataset import SLCSWindowDataset  # noqa: E402
from src.tasks.slcs.data.splits import load_split_assignments  # noqa: E402
from src.utils.checksum import dual_sha256  # noqa: E402
from src.utils.configuration import PathResolver, RuntimePathRoots  # noqa: E402
from src.utils.schema.court import COURT_COORD_SCALE_XYZ  # noqa: E402

Arrays = dict[str, NDArray[Any]]
SCALE = np.asarray(COURT_COORD_SCALE_XYZ, dtype=np.float64)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def dataset_arrays(dataset: SLCSWindowDataset) -> Arrays:
    rows: dict[str, list[Any]] = {
        key: []
        for key in (
            "target_ball_position",
            "ball_mask",
            "ball_weight",
            "padding_mask",
            "frame_idx",
        )
    }
    for index in range(len(dataset)):
        sample = dataset[index]
        padding = sample["padding_mask"].numpy()
        rows["target_ball_position"].append(sample["target_ball_position"].numpy())
        rows["ball_mask"].append(sample["target_ball_valid"].numpy() & ~padding)
        rows["ball_weight"].append(sample["target_ball_weight"].numpy())
        rows["padding_mask"].append(padding)
        frames = sample["frame_idx"].numpy().copy()
        frames[padding] = -1
        rows["frame_idx"].append(frames)
    arrays = {key: np.stack(value) for key, value in rows.items()}
    arrays["scene_ids"] = np.asarray(dataset.prediction_ids)
    for key, attr in [
        ("video_ids", "video_id"),
        ("clip_ids", "clip_id"),
        ("camera_ids", "camera_id"),
        ("window_start", "window_start"),
        ("window_length", "window_length"),
    ]:
        arrays[key] = np.asarray([getattr(meta, attr) for meta in dataset.metas])
    return arrays


def fit_mean(arrays: Arrays) -> tuple[NDArray[np.float64], Arrays, dict[str, Any]]:
    """Deduplicate nonpadding identities, checking even invalid/zero-weight duplicates."""
    seen: dict[tuple[str, str, str, int], tuple[NDArray[Any], bool, float]] = {}
    for i, j in zip(*np.where(~arrays["padding_mask"]), strict=True):
        key = (
            str(arrays["video_ids"][i]),
            str(arrays["clip_ids"][i]),
            str(arrays["camera_ids"][i]),
            int(arrays["frame_idx"][i, j]),
        )
        target = arrays["target_ball_position"][i, j]
        valid, weight = (
            bool(arrays["ball_mask"][i, j]),
            float(arrays["ball_weight"][i, j]),
        )
        if (
            not np.isfinite(target).all()
            or not np.isfinite(weight)
            or weight < 0
            or weight > 1
            or (not valid and weight != 0)
        ):
            raise ValueError(f"Invalid train target/weight: {key}")
        if key in seen:
            old = seen[key]
            if not np.array_equal(old[0], target) or old[1:] != (valid, weight):
                raise ValueError(f"Inconsistent duplicate target/mask/weight: {key}")
        else:
            seen[key] = (target.copy(), valid, weight)
    selected = [
        (key, value) for key, value in seen.items() if value[1] and value[2] > 0
    ]
    if not selected:
        raise ValueError("No positive-weight train targets")
    targets = np.asarray([value[0] for _, value in selected], dtype=np.float64)
    weights = np.asarray([value[2] for _, value in selected], dtype=np.float64)
    mean = (targets * weights[:, None]).sum(axis=0) / weights.sum()
    evidence = {
        "identities": np.asarray([key[:3] for key, _ in selected]),
        "frame_idx": np.asarray([key[3] for key, _ in selected]),
        "target_normalized": targets,
        "weight": weights,
    }
    report = {
        "constant_normalized": mean.tolist(),
        "constant_m": (mean * SCALE).tolist(),
        "unique_nonpadding_count": len(seen),
        "unique_valid_count": sum(value[1] for value in seen.values()),
        "positive_weight_count": len(selected),
        "weight_sum": float(weights.sum()),
        "positive_weight_counts_by_video": dict(Counter(key[0] for key, _ in selected)),
        "positive_weight_counts_by_clip": dict(Counter(key[1] for key, _ in selected)),
    }
    return mean, evidence, report


def validate_evaluation(
    arrays: Arrays, expected: Arrays, train_videos: set[str]
) -> None:
    if train_videos.intersection(arrays["video_ids"].tolist()):
        raise ValueError("Train/evaluation video leakage")
    for key, value in expected.items():
        if (
            key not in arrays
            or arrays[key].dtype != value.dtype
            or not np.array_equal(arrays[key], value)
        ):
            raise ValueError(
                f"Evaluation disagrees with production split dataset: {key}"
            )


def summarize(values: NDArray[Any]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def score(
    arrays: Arrays, mean: NDArray[np.float64], recorded_mean: float
) -> dict[str, Any]:
    target, prediction, mask = (
        arrays["target_ball_position"],
        arrays["pred_ball_position"],
        arrays["ball_mask"],
    )
    if (
        prediction.shape != target.shape
        or not np.isfinite(prediction).all()
        or not mask.any()
    ):
        raise ValueError("Invalid/empty ball predictions")
    model_error = np.linalg.norm(
        (prediction.astype(np.float64) - target) * SCALE, axis=-1
    )
    stored = arrays["ball_pos_error_m"]
    if stored.shape != model_error.shape or not np.allclose(
        stored, model_error, rtol=2e-6, atol=2e-6
    ):
        raise ValueError("Saved model ball error disagrees with recomputation")
    if not np.isclose(model_error[mask].mean(), recorded_mean, rtol=2e-6, atol=2e-6):
        raise ValueError("Saved headline model metric disagrees with recomputation")
    baseline = np.linalg.norm((target[mask] - mean) * SCALE, axis=-1)
    pred_m = prediction[mask].astype(np.float64) * SCALE
    return {
        "valid_window_occurrences": int(mask.sum()),
        "model_error_m": summarize(model_error[mask]),
        "fixed_train_mean_error_m": summarize(baseline),
        "prediction_std_xyz_m": pred_m.std(axis=0).tolist(),
        "prediction_mean_minus_constant_xyz_m": (
            pred_m.mean(axis=0) - mean * SCALE
        ).tolist(),
        "prediction_distance_to_constant_m": summarize(
            np.linalg.norm(pred_m - mean * SCALE, axis=-1)
        ),
    }


def run(training_config: Path, evaluation_dirs: list[Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}

    def record(path: Path) -> None:
        name = str(path.absolute())
        digest = dual_sha256(path)
        if name in hashes and hashes[name] != digest:
            raise ValueError(f"Input changed while reading: {name}")
        hashes[name] = digest

    try:
        atomic_json(output / "status.json", {"status": "running"})
        record(Path(__file__))
        record(training_config)
        raw = cast(
            dict[str, Any],
            OmegaConf.to_container(OmegaConf.load(training_config), resolve=True),
        )
        resolver = PathResolver(
            RuntimePathRoots.from_mapping(raw["paths"], repository_root=ROOT)
        )
        runtime = SLCSDataRuntimeConfig.from_mapping(raw["data"], resolver)
        if runtime.overfit or runtime.pipeline.on_incomplete != "error":
            raise ValueError("Requires non-overfit config and on_incomplete=error")
        record(runtime.dataset_root / "dataset.json")
        record(runtime.split_file)
        index = SLCSDataIndex.load(runtime.dataset_root)
        assignments = load_split_assignments(runtime.split_file, index)
        for ref in index.clips:
            directory = index.clip_dir(ref)
            record(directory / "clip.json")
            for name in ("annotation.json", "scene.npz", "scene.metadata.json"):
                record(slcs_annotation_dir(directory) / name)
        # Record production code used for parsing, normalization, quality and metrics.
        for directory in (
            ROOT / "src/tasks/slcs",
            ROOT / "src/tennis_scene",
            ROOT / "src/utils",
        ):
            for path in sorted(directory.rglob("*.py")):
                record(path)
        config = replace(runtime.pipeline, require_dino=False)

        def load(split: str) -> Arrays:
            dataset = SLCSWindowDataset(
                dataset_root=runtime.dataset_root,
                split_file=runtime.split_file,
                split=split,
                config=config,
                stride=config.train_stride if split == "train" else config.eval_stride,
                augment=False,
            )
            return dataset_arrays(dataset)

        train = load("train")
        mean, evidence, fit = fit_mean(train)
        train_videos = {
            video for video, split in assignments.items() if split == "train"
        }
        results = []
        for directory in evaluation_dirs:
            record(directory / "metrics.json")
            record(directory / "eval_arrays.npz")
            payload = json.loads((directory / "metrics.json").read_text())
            context = payload["context"]
            split = context["split"]
            if split not in ("val", "test") or context["input_mode"] != "full":
                raise ValueError("Requires declared val/test full-input evaluation")
            if (
                Path(context["dataset_root"]).resolve()
                != runtime.dataset_root.resolve()
            ):
                raise ValueError("Evaluation dataset_root mismatch")
            eval_runtime = SLCSDataRuntimeConfig.from_mapping(
                context["data_config"], resolver
            )
            # Augmentation is never active for evaluation; other data semantics must match.
            if (
                eval_runtime.overfit
                or eval_runtime.dataset_root.resolve() != runtime.dataset_root.resolve()
                or eval_runtime.split_file.resolve() != runtime.split_file.resolve()
                or replace(eval_runtime.pipeline, augmentation=None)
                != replace(runtime.pipeline, augmentation=None)
            ):
                raise ValueError("Evaluation data configuration mismatch")
            expected_context = {
                "position_scale_xyz_m": list(COURT_COORD_SCALE_XYZ),
                "position_representation": "normalized court coordinates (same as training payload)",
                "frame_idx_reference": "zero-based absolute frame in clip camera video; padding=-1",
                "padding_mask": "true indicates padding",
            }
            if any(context[key] != value for key, value in expected_context.items()):
                raise ValueError("Evaluation coordinate context mismatch")
            with np.load(directory / "eval_arrays.npz", allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            validate_evaluation(arrays, load(split), train_videos)
            results.append(
                {
                    "evaluation_dir": str(directory.absolute()),
                    "split": split,
                    "context": context,
                    **score(arrays, mean, payload["ball_position_error_m"]),
                }
            )
        for name, digest in hashes.items():
            if dual_sha256(Path(name)) != digest:
                raise ValueError(f"Input changed during diagnostic: {name}")
        with (output / "fit_evidence.tmp").open("wb") as stream:
            np.savez_compressed(
                stream,
                identities=evidence["identities"],
                frame_idx=evidence["frame_idx"],
                target_normalized=evidence["target_normalized"],
                weight=evidence["weight"],
            )
        (output / "fit_evidence.tmp").replace(output / "fit_evidence.npz")
        report = {
            "fit": fit,
            "evaluations": results,
            "diagnostic_config": asdict(config),
            "diagnostic_overrides": {
                "require_dino": {
                    "original": runtime.pipeline.require_dino,
                    "diagnostic": False,
                },
                "augment": False,
            },
            "interpretation": "Pseudo-teacher agreement, not measured 3D accuracy. Train-only confidence-weighted arithmetic mean, not the L2-distance-optimal constant. Fit deduplicates (video, clip, camera, frame); cameras remain distinct. Scores use all saved window occurrences, unweighted on ball_mask. DINO existence/features are not validated.",
            "input_dual_sha256": hashes,
            "fit_evidence_dual_sha256": dual_sha256(output / "fit_evidence.npz"),
        }
        atomic_json(output / "results.json", report)
        atomic_json(output / "status.json", {"status": "completed"})
    except BaseException as error:
        atomic_json(
            output / "status.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "input_dual_sha256": hashes,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--evaluation-dir", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.training_config, args.evaluation_dir, args.output_dir)


if __name__ == "__main__":
    main()
