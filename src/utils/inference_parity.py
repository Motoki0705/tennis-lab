"""Shared output parity rules for precision changes within one model."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias, TypeVar

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from src.utils.geometry.bbox import match_iou, pairwise_iou
from src.utils.geometry.rotation_conversions import axis_angle_to_matrix
from src.utils.inference_benchmark import ParityResult

__all__ = [
    "BodyOutput",
    "Coco17Output",
    "DetectionOutput",
    "PARITY_RULES",
    "compare_bodies",
    "compare_coco17",
    "compare_detections",
]

_FloatArray: TypeAlias = NDArray[np.float64]
_Mask: TypeAlias = NDArray[np.bool_]
_Parts: TypeAlias = dict[str, list[_FloatArray]]
_Output = TypeVar("_Output")


@dataclass(frozen=True)
class _Rule:
    statistic: Literal["p05", "p95", "max", "ratio"]
    limit: float | None = None
    comparison: Literal[">=", "<="] = "<="


PARITY_RULES: Mapping[str, _Rule] = MappingProxyType(
    {
        "iou_p05": _Rule("p05", 0.98, ">="),
        "xyxy_abs_px_p95": _Rule("p95"),
        "score_abs_p95": _Rule("p95", 0.01),
        "missing_ratio": _Rule("ratio", 0.01),
        "extra_ratio": _Rule("ratio", 0.01),
        "pose_xy_px_p95": _Rule("p95", 1.0),
        "pose_xy_px_max": _Rule("max", 5.0),
        "pose_confidence_abs_p95": _Rule("p95", 0.02),
        "feature_cosine_p05": _Rule("p05", 0.995, ">="),
        "feature_relative_l2_p95": _Rule("p95", 0.05),
        "rotation_deg_p95": _Rule("p95", 1.0),
        "joints_mm_p95": _Rule("p95", 10.0),
        "translation_mm_p95": _Rule("p95"),
        "betas_abs_p95": _Rule("p95"),
    }
)
_MIN_MATCH_IOU = 0.5
_DETECTION = (
    "iou_p05",
    "xyxy_abs_px_p95",
    "score_abs_p95",
    "missing_ratio",
    "extra_ratio",
)
_POSE = ("pose_xy_px_p95", "pose_xy_px_max", "pose_confidence_abs_p95")
_BODY = (
    "rotation_deg_p95",
    "joints_mm_p95",
    "translation_mm_p95",
    "betas_abs_p95",
    "feature_cosine_p05",
    "feature_relative_l2_p95",
)


@dataclass(frozen=True)
class Coco17Output:
    """Original-pixel x, y, confidence per joint, with an optional boolean mask."""

    keypoints: ArrayLike
    valid: ArrayLike | None = None


@dataclass(frozen=True)
class DetectionOutput:
    boxes_xyxy: ArrayLike
    scores: ArrayLike
    poses: Coco17Output | None = None


@dataclass(frozen=True)
class BodyOutput:
    """Camera-space joints/translation in metres and axis-angle rotations in radians."""

    joints: ArrayLike
    rotations: ArrayLike
    translation: ArrayLike
    valid_joints: ArrayLike | None = None
    frame_ids: tuple[int, ...] | None = None
    features: ArrayLike | None = None
    betas: ArrayLike | None = None


def _array(value: ArrayLike, shape: Sequence[int | None], label: str) -> _FloatArray:
    array = np.asarray(value)
    if array.ndim != len(shape) or any(
        size is not None and size != actual
        for size, actual in zip(shape, array.shape, strict=True)
    ):
        raise ValueError(f"{label} must have shape {tuple(shape)}, got {array.shape}")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{label} must contain real numbers")
    return array.astype(np.float64, copy=False)


def _finite(array: _FloatArray, label: str) -> _FloatArray:
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains nonfinite values at compared positions")
    return array


def _mask(value: ArrayLike | None, shape: tuple[int, ...], label: str) -> _Mask:
    mask = np.ones(shape, dtype=bool) if value is None else np.asarray(value)
    if mask.dtype != np.bool_ or mask.shape != shape:
        raise ValueError(f"{label} must be a boolean mask with shape {shape}")
    return mask


def _same_mask(first: _Mask, second: _Mask) -> None:
    if not np.array_equal(first, second):
        raise ValueError("Compared joint masks differ")


def _pose_output(
    pose: Coco17Output, shape: tuple[int, ...]
) -> tuple[_FloatArray, _Mask]:
    coordinates = _array(pose.keypoints, shape, "COCO17 keypoints")
    valid = _mask(pose.valid, shape[:-1], "COCO17 validity")
    _finite(coordinates[valid], "COCO17 keypoints")
    return coordinates, valid


def _pose_parts(
    ref: tuple[_FloatArray, _Mask], other: tuple[_FloatArray, _Mask], samples: _Parts
) -> None:
    first, valid = ref
    second, other_valid = other
    _same_mask(valid, other_valid)
    difference = first[valid] - second[valid]
    xy = np.linalg.norm(difference[:, :2], axis=-1)
    samples["pose_xy_px_p95"].append(xy)
    samples["pose_xy_px_max"].append(xy)
    samples["pose_confidence_abs_p95"].append(np.abs(difference[:, 2]))


def _aggregate(parts: list[_FloatArray], rule: _Rule) -> float | None:
    values = np.concatenate([part.ravel() for part in parts])
    if not values.size:
        return None
    _finite(values, "Parity statistic")
    if rule.statistic == "max":
        return float(values.max())
    if rule.statistic == "ratio":
        return float(values.item())
    return float(
        np.percentile(values, 5 if rule.statistic == "p05" else 95, method="linear")
    )


def _compare(
    reference: Mapping[str, _Output],
    candidate: Mapping[str, _Output],
    names: Sequence[str],
    collect: Callable[
        [Mapping[str, _Output], Mapping[str, _Output], _Parts, dict[str, object]], None
    ],
    *,
    extra_criteria: Mapping[str, object] | None = None,
) -> ParityResult:
    rules = {name: PARITY_RULES[name] for name in names}
    criteria = {
        "reference": "same model PT32",
        "quantile_method": "linear",
        "rules": {key: asdict(rule) for key, rule in rules.items()},
        **(extra_criteria or {}),
    }
    metrics: dict[str, object] = {
        "reference_items": len(reference),
        "candidate_items": len(candidate),
    }
    if not reference or reference.keys() != candidate.keys():
        return ParityResult(
            False,
            {**metrics, "invalid": True},
            "Input ID coverage differs or is empty",
            criteria,
        )
    metrics["checked_ids"] = len(reference)
    samples: _Parts = {name: [] for name in names}
    violations = []
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            collect(reference, candidate, samples, metrics)
            for name, rule in rules.items():
                if not samples[name]:  # Optional body features or shape parameters.
                    continue
                value = _aggregate(samples[name], rule)
                metrics[name] = value
                if rule.limit is None:
                    continue
                if value is None:
                    violations.append(f"{name} has no eligible observations")
                elif (
                    value < rule.limit
                    if rule.comparison == ">="
                    else value > rule.limit
                ):
                    violations.append(
                        f"{name}={value:.8g} requires {rule.comparison} {rule.limit:g}"
                    )
    except (ValueError, TypeError, FloatingPointError, OverflowError) as error:
        return ParityResult(False, {**metrics, "invalid": True}, str(error), criteria)
    return ParityResult(
        not violations, metrics, ", ".join(violations) or None, criteria
    )


def _detector_arrays(value: DetectionOutput) -> tuple[_FloatArray, _FloatArray]:
    boxes = _finite(
        _array(value.boxes_xyxy, (None, 4), "Detection boxes"), "Detection boxes"
    )
    scores = _finite(
        _array(value.scores, (len(boxes),), "Detection scores"), "Detection scores"
    )
    return boxes, scores


def compare_detections(
    reference: Mapping[str, DetectionOutput],
    candidate: Mapping[str, DetectionOutput],
    *,
    with_pose: bool = False,
) -> ParityResult:
    """Match boxes by IoU within each input, with optional matched COCO17 poses."""

    def collect(
        ref: Mapping[str, DetectionOutput],
        actual: Mapping[str, DetectionOutput],
        samples: _Parts,
        metrics: dict[str, object],
    ) -> None:
        reference_boxes = candidate_boxes = matches = 0
        for key, first in ref.items():
            second = actual[key]
            a, scores_a = _detector_arrays(first)
            b, scores_b = _detector_arrays(second)
            iou = pairwise_iou(a, b)
            left, right = match_iou(iou, _MIN_MATCH_IOU)
            reference_boxes += len(a)
            candidate_boxes += len(b)
            matches += len(left)
            samples["iou_p05"].append(iou[left, right])
            samples["xyxy_abs_px_p95"].append(np.abs(a[left] - b[right]))
            samples["score_abs_p95"].append(np.abs(scores_a[left] - scores_b[right]))
            if with_pose:
                if first.poses is None or second.poses is None:
                    raise ValueError(
                        "Matched detection parity requires COCO17 outputs for all boxes"
                    )
                x, valid_x = _pose_output(first.poses, (len(a), 17, 3))
                y, valid_y = _pose_output(second.poses, (len(b), 17, 3))
                _pose_parts(
                    (x[left], valid_x[left]), (y[right], valid_y[right]), samples
                )
        missing, extra = reference_boxes - matches, candidate_boxes - matches
        metrics.update(
            reference_boxes=reference_boxes,
            candidate_boxes=candidate_boxes,
            matched_boxes=matches,
            missing=missing,
            extra=extra,
        )
        samples["missing_ratio"].append(np.array([missing / max(reference_boxes, 1)]))
        samples["extra_ratio"].append(np.array([extra / max(candidate_boxes, 1)]))

    return _compare(
        reference,
        candidate,
        _DETECTION + (_POSE if with_pose else ()),
        collect,
        extra_criteria={
            "matching_min_iou": _MIN_MATCH_IOU,
            "matching": "maximum pairs then total IoU",
            "ratio_denominator": {
                "missing_ratio": "reference_boxes",
                "extra_ratio": "candidate_boxes",
            },
        },
    )


def compare_coco17(
    reference: Mapping[str, Coco17Output], candidate: Mapping[str, Coco17Output]
) -> ParityResult:
    """Compare one COCO17 prediction per pinned input crop."""

    def collect(
        ref: Mapping[str, Coco17Output],
        actual: Mapping[str, Coco17Output],
        samples: _Parts,
        metrics: dict[str, object],
    ) -> None:
        for key, expected in ref.items():
            _pose_parts(
                _pose_output(expected, (17, 3)),
                _pose_output(actual[key], (17, 3)),
                samples,
            )
        metrics["compared_joints"] = sum(
            part.size for part in samples["pose_xy_px_p95"]
        )

    return _compare(reference, candidate, _POSE, collect)


def _body_array(
    first: ArrayLike, second: ArrayLike, shape: tuple[int | None, ...], label: str
) -> tuple[_FloatArray, _FloatArray]:
    expected = _array(first, shape, label)
    observed = _array(second, expected.shape, label)
    if not expected.size:
        raise ValueError(f"{label} has no values")
    return expected, observed


def _optional_body_array(
    first: ArrayLike | None, second: ArrayLike | None, frames: int, name: str
) -> tuple[_FloatArray, _FloatArray] | None:
    if first is None or second is None:
        if first is not None or second is not None:
            raise ValueError(f"{name} availability differs")
        return None
    a, b = _body_array(first, second, (frames, None), name)
    return _finite(a, name), _finite(b, name)


def _body_parts(a: BodyOutput, b: BodyOutput, samples: _Parts) -> None:
    joints_a, joints_b = _body_array(a.joints, b.joints, (None, None, 3), "Body joints")
    frames = len(joints_a)
    if a.frame_ids != b.frame_ids or (
        a.frame_ids is not None
        and (len(a.frame_ids) != frames or len(set(a.frame_ids)) != frames)
    ):
        raise ValueError("Body frame IDs differ or do not match the time axis")
    valid = _mask(a.valid_joints, joints_a.shape[:-1], "Body joint validity")
    _same_mask(valid, _mask(b.valid_joints, joints_a.shape[:-1], "Body joint validity"))
    distance = _finite(joints_a[valid], "Body joints") - _finite(
        joints_b[valid], "Body joints"
    )
    samples["joints_mm_p95"].append(np.linalg.norm(distance, axis=-1) * 1000)

    rot_a, rot_b = _body_array(
        a.rotations, b.rotations, (frames, None, 3), "Body rotations"
    )
    first, second = (
        axis_angle_to_matrix(torch.from_numpy(_finite(value, "Body rotations")))
        for value in (rot_a, rot_b)
    )
    trace = (first * second).sum(dim=(-1, -2))
    samples["rotation_deg_p95"].append(
        torch.rad2deg(torch.acos(((trace - 1) / 2).clamp(-1, 1))).numpy()
    )

    transl_a, transl_b = _body_array(
        a.translation, b.translation, (frames, 3), "Body translation"
    )
    translation = _finite(transl_a, "Body translation") - _finite(
        transl_b, "Body translation"
    )
    samples["translation_mm_p95"].append(np.linalg.norm(translation, axis=-1) * 1000)

    features = _optional_body_array(a.features, b.features, frames, "Body features")
    if features is not None:
        f, g = features
        norm_f, norm_g = np.linalg.norm(f, axis=-1), np.linalg.norm(g, axis=-1)
        samples["feature_cosine_p05"].append(
            np.sum(f * g, axis=-1) / np.maximum(norm_f * norm_g, 1e-12)
        )
        samples["feature_relative_l2_p95"].append(
            np.linalg.norm(f - g, axis=-1) / np.maximum(norm_f, 1e-12)
        )
    betas = _optional_body_array(a.betas, b.betas, frames, "Body shape parameters")
    if betas is not None:
        samples["betas_abs_p95"].append(np.abs(betas[0] - betas[1]))


def compare_bodies(
    reference: Mapping[str, BodyOutput], candidate: Mapping[str, BodyOutput]
) -> ParityResult:
    """Compare matched time series in the camera coordinate system."""

    def collect(
        ref: Mapping[str, BodyOutput],
        actual: Mapping[str, BodyOutput],
        samples: _Parts,
        metrics: dict[str, object],
    ) -> None:
        for name in ("features", "betas"):
            if len({getattr(row, name) is None for row in ref.values()}) != 1:
                raise ValueError(
                    f"Body {name} must be supplied for every input or for none"
                )
        for key, expected in ref.items():
            _body_parts(expected, actual[key], samples)
        metrics["compared_joints"] = sum(part.size for part in samples["joints_mm_p95"])

    return _compare(
        reference,
        candidate,
        _BODY,
        collect,
        extra_criteria={
            "joint_coordinates": "camera",
            "feature_metrics": "when supplied by the model",
        },
    )
