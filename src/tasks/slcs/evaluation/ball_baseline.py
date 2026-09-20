"""Train-only ball constant, fitted and scored on production label windows.

The confidence-weighted arithmetic mean minimizes weighted squared distance;
it is not the optimal constant for the headline mean Euclidean distance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from src.tasks.slcs.configuration import SLCSDataRuntimeConfig
from src.tasks.slcs.data.annotation import SLCSDataIndex
from src.tasks.slcs.data.dataset import SLCSWindowDataset
from src.tasks.slcs.data.splits import load_split_assignments
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def _validate_labels(arrays: dict[str, np.ndarray]) -> None:
    mask = arrays["ball_mask"]
    if mask.ndim != 2 or not mask.shape[0]:
        raise ValueError("Ball labels require nonempty (windows, frames) masks")
    n, t = mask.shape
    shapes: dict[str, tuple[int, ...]] = {
        key: (n,)
        for key in (
            "scene_ids",
            "video_ids",
            "clip_ids",
            "camera_ids",
            "window_start",
            "window_length",
        )
    }
    shapes.update(
        {
            "ball_mask": (n, t),
            "ball_weight": (n, t),
            "padding_mask": (n, t),
            "frame_idx": (n, t),
            "target_ball_position": (n, t, 3),
        }
    )
    for key, shape in shapes.items():
        value = arrays[key]
        if value.shape != shape:
            raise ValueError(f"Invalid ball label shape: {key}")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite ball label: {key}")
    for key in ("scene_ids", "video_ids", "clip_ids", "camera_ids"):
        if arrays[key].dtype.kind != "U" or any(not value for value in arrays[key]):
            raise ValueError(f"Expected nonempty string identifiers: {key}")
    if len(set(arrays["scene_ids"].tolist())) != n:
        raise ValueError("Window scene_ids must be unique")
    for key in ("padding_mask", "ball_mask"):
        if arrays[key].dtype != np.bool_:
            raise ValueError(f"Expected boolean mask: {key}")
    for key in ("frame_idx", "window_start", "window_length"):
        if arrays[key].dtype.kind not in "iu":
            raise ValueError(f"Expected integer indices: {key}")
    lengths, starts = arrays["window_length"], arrays["window_start"]
    if ((lengths < 1) | (lengths > t)).any() or (starts < 0).any():
        raise ValueError("Invalid window start/length")
    padding = np.arange(t)[None] >= lengths[:, None]
    frames = starts[:, None] + np.arange(t)[None]
    frames[padding] = -1
    if not np.array_equal(padding, arrays["padding_mask"]) or not np.array_equal(
        frames, arrays["frame_idx"]
    ):
        raise ValueError("Frame indices/padding disagree with window metadata")
    weight = arrays["ball_weight"]
    if (
        (mask & padding).any()
        or ((weight < 0) | (weight > 1)).any()
        or (weight[~mask] != 0).any()
    ):
        raise ValueError(
            "Invalid ball mask/weight; weights must be in [0, 1] and zero on invalid frames"
        )


def _label_windows(runtime: SLCSDataRuntimeConfig, split: str) -> dict[str, np.ndarray]:
    """CPU label view: same window/quality settings, no DINO cache validation."""
    config = replace(runtime.pipeline, require_dino=False)
    dataset = SLCSWindowDataset(
        dataset_root=runtime.dataset_root,
        split_file=runtime.split_file,
        split=split,
        config=config,
        augment=False,
        stride=config.train_stride if split == "train" else config.eval_stride,
    )
    columns: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "target_ball_position",
            "ball_mask",
            "ball_weight",
            "padding_mask",
            "frame_idx",
        )
    }
    for i in range(len(dataset)):
        sample = dataset[i]
        for key, value in {
            "target_ball_position": sample["target_ball_position"],
            "ball_mask": sample["target_ball_valid"],
            "ball_weight": sample["target_ball_weight"],
            "padding_mask": sample["padding_mask"],
            "frame_idx": sample["frame_idx"],
        }.items():
            columns[key].append(value.numpy())
    arrays = {key: np.stack(values) for key, values in columns.items()}
    arrays["frame_idx"][arrays["padding_mask"]] = -1
    arrays["scene_ids"] = np.asarray(dataset.prediction_ids)
    for key, attribute in (
        ("video_ids", "video_id"),
        ("clip_ids", "clip_id"),
        ("camera_ids", "camera_id"),
        ("window_start", "window_start"),
        ("window_length", "window_length"),
    ):
        arrays[key] = np.asarray([getattr(meta, attribute) for meta in dataset.metas])
    _validate_labels(arrays)
    return arrays


def _fit_mean(arrays: dict[str, np.ndarray], train_videos: set[str]) -> dict[str, Any]:
    _validate_labels(arrays)
    if not set(arrays["video_ids"].tolist()) <= train_videos:
        raise ValueError("Train mean cannot fit videos outside the train split")
    seen: dict[tuple[str, str, str, int], tuple[np.ndarray, bool, float]] = {}
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
        if key in seen:
            old = seen[key]
            if not np.array_equal(old[0], target) or old[1:] != (valid, weight):
                raise ValueError(f"Inconsistent duplicate target/mask/weight: {key}")
        else:
            seen[key] = (target.copy(), valid, weight)
    selected = [value for value in seen.values() if value[1] and value[2] > 0]
    if not selected:
        raise ValueError("No positive-weight train ball targets")
    targets = np.asarray([value[0] for value in selected], dtype=np.float64)
    weights = np.asarray([value[2] for value in selected], dtype=np.float64)
    mean = np.sum(targets * weights[:, None], axis=0) / weights.sum()
    return {
        "constant_normalized": mean.tolist(),
        "constant_m": (mean * np.asarray(COURT_COORD_SCALE_XYZ)).tolist(),
        "unique_nonpadding_count": len(seen),
        "unique_valid_count": sum(value[1] for value in seen.values()),
        "positive_weight_count": len(selected),
        "weight_sum": float(weights.sum()),
        "train_video_ids": sorted(train_videos),
        "train_windows": len(arrays["video_ids"]),
    }


@dataclass
class TrainBallMean:
    """A constant fitted exclusively from the saved configuration's train split."""

    runtime: SLCSDataRuntimeConfig
    assignments: dict[str, str]
    fit_report: dict[str, Any]

    @classmethod
    def fit(cls, runtime: SLCSDataRuntimeConfig) -> TrainBallMean:
        if runtime.overfit:
            raise ValueError(
                "Train ball baseline rejects overfit mode: no held-out split"
            )
        if runtime.pipeline.on_incomplete != "error":
            raise ValueError("Train ball baseline requires on_incomplete=error")
        index = SLCSDataIndex.load(runtime.dataset_root)
        assignments = load_split_assignments(runtime.split_file, index)
        for record in index.clips:
            manifest = ClipManifest.load(index.clip_dir(record))
            if (manifest.video_id, manifest.clip_id, manifest.fps) != (
                record.video_id,
                record.clip_id,
                record.fps,
            ):
                raise ValueError("Dataset/clip manifest mismatch")
        train = _label_windows(runtime, "train")
        report = _fit_mean(
            train, {video for video, split in assignments.items() if split == "train"}
        )
        report.update(
            {
                "schema_version": 1,
                "fit_split": "train",
                "dataset_root": str(runtime.dataset_root),
                "split_file": str(runtime.split_file),
                "assignments": assignments,
                "data_pipeline": asdict(runtime.pipeline),
                "label_view_overrides": {"require_dino": False, "augment": False},
                "interpretation": "Pseudo-teacher agreement, not measured 3D accuracy. Confidence-weighted arithmetic mean minimizes weighted squared distance, not mean Euclidean distance. Fit deduplicates (video, clip, camera, frame); cameras remain distinct. CPU label view does not read or validate DINO caches.",
            }
        )
        return cls(runtime, assignments, report)

    def expected_labels(self, split: str) -> dict[str, np.ndarray]:
        if split not in {"train", "val", "test"}:
            raise ValueError("Expected train, val or test split")
        assignments = load_split_assignments(
            self.runtime.split_file, SLCSDataIndex.load(self.runtime.dataset_root)
        )
        if assignments != self.assignments:
            raise ValueError("Split assignments changed after fitting train mean")
        labels = _label_windows(self.runtime, split)
        if any(
            self.assignments.get(video) != split
            for video in labels["video_ids"].tolist()
        ):
            raise ValueError("Evaluation split/video leakage")
        return labels

    def compare(
        self,
        arrays: dict[str, np.ndarray],
        *,
        expected: dict[str, np.ndarray],
        split: str,
        domains: dict[str, str],
        headline_error_m: float | None,
    ) -> dict[str, Any]:
        """Score unweighted valid window occurrences, including zero-weight labels."""
        _validate_labels(arrays)
        if split not in {"train", "val", "test"} or any(
            self.assignments.get(video) != split
            for video in arrays["video_ids"].tolist()
        ):
            raise ValueError("Evaluation split/video leakage")
        for key, value in expected.items():
            if (
                key not in arrays
                or arrays[key].dtype != value.dtype
                or not np.array_equal(arrays[key], value)
            ):
                raise ValueError(
                    f"Evaluation disagrees with production split labels: {key}"
                )
        videos = arrays["video_ids"]
        if set(domains) != set(videos.tolist()):
            raise ValueError("Domain mapping must cover evaluated videos exactly")
        prediction, target = (
            arrays["pred_ball_position"],
            arrays["target_ball_position"],
        )
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError("Invalid ball predictions")
        scale = np.asarray(COURT_COORD_SCALE_XYZ, dtype=np.float64)
        model_error = np.linalg.norm(
            (prediction.astype(np.float64) - target) * scale, axis=-1
        )
        baseline_error = np.linalg.norm(
            (target.astype(np.float64) - self.fit_report["constant_normalized"])
            * scale,
            axis=-1,
        )
        mask = arrays["ball_mask"]
        if mask.any() and (
            headline_error_m is None
            or not np.isclose(
                model_error[mask].mean(), headline_error_m, rtol=2e-6, atol=2e-6
            )
        ):
            raise ValueError("Model ball error disagrees with headline metric")
        groups: list[tuple[str, str, np.ndarray]] = [
            ("all", "all", np.ones(len(videos), dtype=bool))
        ]
        groups += [
            ("domain", domain, np.asarray([domains[v] == domain for v in videos]))
            for domain in sorted(set(domains.values()))
        ]
        groups += [
            ("video", str(video), videos == video) for video in np.unique(videos)
        ]
        rows = []
        for kind, name, selection in groups:
            keep = mask & selection[:, None]
            model = float(model_error[keep].mean()) if keep.any() else None
            baseline = float(baseline_error[keep].mean()) if keep.any() else None
            rows.append(
                {
                    "group_type": kind,
                    "group": name,
                    "num_windows": int(selection.sum()),
                    "valid_window_occurrences": int(keep.sum()),
                    "model_error_m": model,
                    "train_mean_error_m": baseline,
                    "model_minus_train_mean_error_m": model - baseline
                    if model is not None and baseline is not None
                    else None,
                }
            )
        return {
            "schema_version": 1,
            "split": split,
            "in_sample": split == "train",
            "aggregation": "Unweighted valid window occurrences; overlapping windows counted separately, including valid zero-confidence labels.",
            "interpretation": "Pseudo-teacher agreement; negative model-minus-baseline is better agreement. Train split scores are in-sample.",
            "position_scale_xyz_m": scale.tolist(),
            "constant_m": self.fit_report["constant_m"],
            "domain_mapping": domains,
            "rows": rows,
        }
