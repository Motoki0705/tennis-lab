"""Count nonpadding target/prediction degeneracy over explicit threshold candidates.

Inputs are saved target scenes (COCO17 world metres) or NPZ exports containing
``pose_m`` and bool ``padding_mask``. This CPU survey never masks predictions
in a training loss; it measures them separately for threshold qualification.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.geometry.skeleton import (
    MIN_BONE_LENGTH_M,
    MIN_PLANE_SINE,
    skeleton_domain_metrics,
)
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.player import (
    COCO17_BONE_LENGTH_EDGES,
    COCO17_TORSION_QUADRUPLETS,
    COCO_KP_NAMES,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="plcs.skeleton_validity",
    fields=(
        BoundaryPathField(
            "target",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            many=True,
            required=False,
        ),
        BoundaryPathField(
            "prediction",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            many=True,
            required=False,
        ),
        BoundaryPathField(
            "target_scenes",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
            many=True,
            required=False,
        ),
        BoundaryPathField(
            "output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.FILE
        ),
    ),
)


class FrequencySurvey:
    """Streaming counts; each feature's denominator is the nonpadding frame count."""

    def __init__(
        self, length_thresholds_m: Sequence[float], plane_thresholds: Sequence[float]
    ) -> None:
        for values in (length_thresholds_m, plane_thresholds):
            if (
                not values
                or len(set(values)) != len(values)
                or any(not math.isfinite(v) or v < 0 for v in values)
            ):
                raise ValueError(
                    "Threshold candidates must be unique, finite, nonnegative and nonempty."
                )
        self.total_frames = 0
        self.padding_frames = 0
        self.thresholds = {
            "bone_length_m": tuple(length_thresholds_m),
            "torsion_plane_sine": tuple(plane_thresholds),
            "torso_axis_length_m": tuple(length_thresholds_m),
            "torso_plane_sine": tuple(plane_thresholds),
        }
        shapes = {
            "bone_length_m": (12,),
            "torsion_plane_sine": (4, 2),
            "torso_axis_length_m": (3,),
            "torso_plane_sine": (2,),
        }
        self.counts: dict[str, list[np.ndarray]] = {
            key: [np.zeros(shapes[key], dtype=np.int64) for _ in candidates]
            for key, candidates in self.thresholds.items()
        }
        self.frame_counts = {
            key: [0 for _ in candidates] for key, candidates in self.thresholds.items()
        }

    def update(
        self, pose_m: np.ndarray, padding_mask: np.ndarray, *, chunk_size: int = 4096
    ) -> None:
        """Accept ``(T,17,3)``/``(B,T,17,3)`` and frame or multiview padding."""
        if (
            pose_m.ndim not in (3, 4)
            or pose_m.shape[-2:] != (17, 3)
            or pose_m.dtype not in (np.float32, np.float64)
        ):
            raise ValueError(
                "pose_m must be float32/64 COCO17 metres, (T,17,3) or (B,T,17,3)."
            )
        if padding_mask.dtype != np.bool_:
            raise ValueError("padding_mask must be bool, True=padding.")
        if pose_m.ndim == 4 and padding_mask.ndim == 3:
            padding_mask = padding_mask.all(axis=1)
        if padding_mask.shape != pose_m.shape[:-2]:
            raise ValueError(
                "padding_mask must match pose frame axes, optionally (B,V,T)."
            )
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        flat_pose, flat_padding = pose_m.reshape(-1, 17, 3), padding_mask.reshape(-1)
        self.total_frames += len(flat_pose)
        self.padding_frames += int(flat_padding.sum())
        for start in range(0, len(flat_pose), chunk_size):
            observed = flat_pose[start : start + chunk_size][
                ~flat_padding[start : start + chunk_size]
            ]
            if len(observed) == 0:
                continue
            metrics = skeleton_domain_metrics(torch.from_numpy(observed.copy()))
            for key, metric in metrics.items():
                for index, threshold in enumerate(self.thresholds[key]):
                    invalid = (metric <= threshold).cpu().numpy()
                    self.counts[key][index] += invalid.sum(axis=0)
                    self.frame_counts[key][index] += int(
                        invalid.reshape(len(observed), -1).any(axis=1).sum()
                    )

    def result(self) -> dict[str, Any]:
        n_frames = self.total_frames - self.padding_frames
        if n_frames == 0:
            raise ValueError(
                "No valid nonpadding frames supplied for the frequency survey."
            )
        return {
            "total_frames": self.total_frames,
            "padding_frames": self.padding_frames,
            "nonpadding_frames": n_frames,
            "feature_order": {
                "bone_length_m": [
                    [COCO_KP_NAMES[a], COCO_KP_NAMES[b]]
                    for a, b in COCO17_BONE_LENGTH_EDGES
                ],
                "torsion_plane_sine": [
                    [
                        [COCO_KP_NAMES[i] for i in q[:3]],
                        [COCO_KP_NAMES[i] for i in q[1:]],
                    ]
                    for q in COCO17_TORSION_QUADRUPLETS
                ],
                "torso_axis_length_m": ["hip", "shoulder", "torso"],
                "torso_plane_sine": ["hip_torso", "shoulder_torso"],
            },
            "metrics": {
                key: [
                    {
                        "threshold": threshold,
                        "invalid_count": counts.tolist(),
                        "invalid_fraction": (counts / n_frames).tolist(),
                        "any_invalid_frames": self.frame_counts[key][index],
                        "any_invalid_fraction": self.frame_counts[key][index]
                        / n_frames,
                    }
                    for index, (threshold, counts) in enumerate(
                        zip(candidates, self.counts[key], strict=True)
                    )
                ]
                for key, candidates in self.thresholds.items()
            },
        }


def _update_archive(survey: FrequencySurvey, path: Path) -> None:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"pose_m", "padding_mask"}:
            raise ValueError(
                f"{path}: expected exactly pose_m and padding_mask arrays."
            )
        survey.update(data["pose_m"], data["padding_mask"])


def main(argv: Sequence[str] | None = None) -> int:
    """Validate absolute paths and survey each supplied source independently."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, action="append", help="Target NPZ; repeat for shards."
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        action="append",
        help="Trained prediction NPZ; repeat for shards.",
    )
    parser.add_argument(
        "--target-scenes",
        type=Path,
        action="append",
        help="Recursively read saved human_kp_3d.npy targets.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--length-thresholds-m",
        type=float,
        nargs="+",
        default=[0.0, MIN_BONE_LENGTH_M, 1e-4],
    )
    parser.add_argument(
        "--plane-thresholds", type=float, nargs="+", default=[0.0, MIN_PLANE_SINE, 1e-3]
    )
    args = parser.parse_args(argv)
    if not any((args.target, args.prediction, args.target_scenes)):
        parser.error("Supply --target, --prediction or --target-scenes.")
    if not args.output.is_absolute():
        parser.error("--output must be absolute.")
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT,
        checkpoint_root=PROJECT_ROOT,
        artifact_root=PROJECT_ROOT,
        output_root=args.output.parent.resolve(),
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    arguments: dict[str, object] = {"output": args.output}
    for name in ("target", "prediction", "target_scenes"):
        supplied = getattr(args, name)
        if supplied is not None:
            arguments[name] = supplied
    paths = PATH_BOUNDARY.validate(
        arguments, resolver=PathResolver(roots), independent_artifact_inputs=True
    )
    results: dict[str, Any] = {
        "length_unit": "metre",
        "plane_metric": "normalized_cross_magnitude",
        "api_thresholds": {
            "min_bone_length_m": MIN_BONE_LENGTH_M,
            "min_plane_sine": MIN_PLANE_SINE,
        },
        "sources": {},
    }
    for name in ("target", "prediction", "target_scenes"):
        if name not in paths:
            continue
        survey = FrequencySurvey(args.length_thresholds_m, args.plane_thresholds)
        started = time.perf_counter()
        files: list[str] = []
        for entry in paths.declared_many(name):
            if name == "target_scenes":
                pose_files = sorted(entry.path.rglob("human_kp_3d.npy"))
                if not pose_files:
                    raise ValueError(
                        f"No human_kp_3d.npy targets found under {entry.path}."
                    )
                for pose_file in pose_files:
                    poses = np.load(pose_file, mmap_mode="r", allow_pickle=False)
                    # Stored scenes have no collator padding. For augmented or
                    # collated targets, export an NPZ with the actual mask.
                    survey.update(poses, np.zeros(poses.shape[:-2], dtype=bool))
                    files.append(str(pose_file))
            else:
                _update_archive(survey, entry.path)
                files.append(str(entry.path))
        result = survey.result()
        result.update(files=files, elapsed_seconds=time.perf_counter() - started)
        results["sources"][name] = result
    output = paths.declared("output").path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
