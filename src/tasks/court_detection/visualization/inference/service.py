"""Image-space Court detection dataset review and inference backend.

This is the Court implementation of the shared detection ``DetectionBackend``
protocol.  It resolves opaque server-side scene IDs into canonical ``CourtInput``
records, renders provenance-verified ground truth in original image pixels, and
runs exactly one strictly loaded Lightning checkpoint per inference request.
Nothing here invents 3D geometry from 2D labels: the review surface stays in
image space.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from src.tasks.court_detection.data.contracts import (
    CourtInputSpec,
    CourtKeypointChannels,
    CourtRawSample,
    CourtSampleRecord,
    CourtSourceKind,
)
from src.tasks.court_detection.data.inputs.contract import CourtInput
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtLinePrediction,
    CourtSegmentationPrediction,
)
from src.tasks.court_detection.target_schemas import SEMANTIC_LINE_CHANNEL_NAMES
from src.tasks.court_detection.visualization.inference.checkpoints import (
    COURT_TASK,
    CourtCheckpointInfo,
    describe_checkpoint,
    scan_checkpoints,
)
from src.tasks.court_detection.visualization.inference.metrics import (
    categorical_metrics,
    keypoint_metrics,
    line_metrics,
    resize_note,
)
from src.tasks.court_detection.visualization.inference.runner import CourtHeadRunner
from src.tasks.court_detection.visualization.review.datasets import (
    CourtDatasetCatalog,
    CourtDatasetEntry,
    CourtDatasetLayers,
    GroundTruthMasks,
    keypoint_points,
    layer_identity,
    load_ground_truth_masks,
)
from src.tasks.court_detection.visualization.review.rasters import (
    heatmap_raster,
    line_probability_raster,
    segmentation_raster,
    semantic_line_raster,
)

TITLE = "Court Detection"
DEFAULT_THRESHOLD = 0.5
_JPEG_QUALITY = 90
_THRESHOLD_NOTE = "しきい値は予測 keypoint の visible 判定と、line head の二値化（採点用）に適用します。"


class DetectionService:
    """Court backend for the shared image review and inference UI."""

    def __init__(
        self,
        *,
        project_root: Path,
        data_root: Path | None = None,
        outputs_root: Path | None = None,
        checkpoints_root: Path | None = None,
    ) -> None:
        project = Path(project_root).expanduser().resolve(strict=False)
        self.project_root = project
        self.data_root = (
            Path(data_root).expanduser().resolve(strict=False)
            if data_root is not None
            else project / "data"
        )
        self.output_root = (
            Path(outputs_root).expanduser().resolve(strict=False)
            if outputs_root is not None
            else project / "outputs" / COURT_TASK
        )
        self.checkpoints_root = (
            Path(checkpoints_root).expanduser().resolve(strict=False)
            if checkpoints_root is not None
            else project / "ckpt" / COURT_TASK
        )
        self.datasets = CourtDatasetCatalog(
            project_root=project,
            data_root=self.data_root,
            checkpoint_root=self.checkpoints_root,
            output_root=self.output_root,
        )

    # ----------------------------------------------------------------- catalog
    def catalog(self) -> dict[str, Any]:
        """Rebuild the catalog from disk; refresh must surface new files.

        Dataset inputs and file stamps are invalidated on every explicit refresh.
        Checkpoints are re-listed, while parsed manifests and checkpoint metadata
        stay cached by size+mtime.
        """
        datasets = self.datasets.catalog_entries()
        available = {
            cast(str, entry["id"]): entry
            for entry in datasets
            if bool(entry["available"])
        }
        checkpoints: list[dict[str, Any]] = []
        for info in self._scan():
            payload = info.to_dict()
            payload["compatible_datasets"] = (
                self._compatible_dataset_ids(info, available) if info.supported else []
            )
            checkpoints.append(payload)
        return {
            "task": COURT_TASK,
            "title": TITLE,
            "datasets": datasets,
            "checkpoints": checkpoints,
            "warnings": self.datasets.warnings(),
        }

    def scenes(
        self,
        dataset: str,
        search: str = "",
        offset: int = 0,
        limit: int = 100,
        checkpoint: str | None = None,
    ) -> dict[str, Any]:
        if offset < 0:
            raise ValueError("Court scene の offset は 0 以上である必要があります。")
        if limit <= 0:
            raise ValueError("Court scene の limit は正の値である必要があります。")
        entry = self.datasets.entry(dataset)
        if checkpoint:
            self._require_compatible(self._describe(checkpoint), entry)
        needle = search.strip().lower()
        identifiers = [
            record.sample_id
            for record in self.datasets.records(dataset)
            if needle in record.sample_id.lower()
        ]
        window = identifiers[offset : offset + limit]
        return {
            "total": len(identifiers),
            "items": [
                {
                    "id": self.datasets.scene_id(dataset, sample_id),
                    "label": sample_id,
                    "frames": 1,
                }
                for sample_id in window
            ],
        }

    def preview(self, scene: str, start: int = 0, count: int = 1) -> dict[str, Any]:
        self._require_single_frame(start=start, count=count)
        entry, record = self.datasets.sample(scene)
        raw = self._input(entry).load(record)
        masks, warnings = load_ground_truth_masks(
            record,
            input_spec=self._input(entry).spec,
            raw=raw,
        )
        points = keypoint_points(raw)
        return {
            "scene": scene,
            "label": record.sample_id,
            "frames": 1,
            "start": 0,
            "width": int(raw.image.width),
            "height": int(raw.image.height),
            "items": [
                {
                    "index": 0,
                    "name": record.sample_id,
                    "gt": {
                        "points": points,
                        "segments": _keypoint_segments(points),
                        "rasters": masks.raster_payloads(),
                    },
                }
            ],
            "warnings": warnings,
        }

    def image(self, scene: str, frame: int) -> bytes:
        self._require_single_frame(start=frame, count=1)
        buffer = io.BytesIO()
        self.datasets.image(scene).convert("RGB").save(
            buffer, format="JPEG", quality=_JPEG_QUALITY
        )
        return buffer.getvalue()

    def validate(
        self,
        checkpoint: str,
        scene: str,
        start: int = 0,
        count: int = 1,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "cuda",
    ) -> None:
        """Reject any request the worker could not complete before GPU enqueue."""
        self._prepare(
            checkpoint=checkpoint,
            scene=scene,
            start=start,
            count=count,
            threshold=threshold,
            device=device,
        )

    def infer(
        self,
        checkpoint: str,
        scene: str,
        start: int = 0,
        count: int = 1,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "cuda",
    ) -> dict[str, Any]:
        info, entry, record, raw = self._prepare(
            checkpoint=checkpoint,
            scene=scene,
            start=start,
            count=count,
            threshold=threshold,
            device=device,
        )
        capacity = _keypoint_capacity(raw.keypoint_channels)
        runner = CourtHeadRunner(
            info.path,
            device=torch.device(device),
            subpixel_refine=True,
            max_peaks=capacity,
        )
        try:
            predictions = runner.predict(raw.image)
        finally:
            runner.close()
        masks, gt_warnings = load_ground_truth_masks(
            record,
            input_spec=self._input(entry).spec,
            raw=raw,
        )
        pred_payload, metrics, metric_warnings = self._render_prediction(
            predictions,
            masks=masks,
            raw=raw,
            threshold=float(threshold),
            channel_names=_keypoint_channel_names(raw.keypoint_channels),
        )
        return {
            "scene": scene,
            "start": 0,
            "items": [{"index": 0, "pred": pred_payload}],
            "metrics": metrics,
            "warnings": [*gt_warnings, *metric_warnings],
        }

    # ---------------------------------------------------------------- internals
    def _scan(self) -> tuple[CourtCheckpointInfo, ...]:
        return tuple(scan_checkpoints(self.output_root, self.checkpoints_root))

    def _describe(self, checkpoint: str) -> CourtCheckpointInfo:
        info = describe_checkpoint(self.output_root, self.checkpoints_root, checkpoint)
        if not info.supported:
            raise ValueError(
                f"Court checkpoint {info.id!r} は非対応です: {info.reason}"
            )
        return info

    def _input(self, entry: CourtDatasetEntry) -> CourtInput:
        """Return the canonical, cached input layer for one dataset entry."""
        return self.datasets.input_for(entry)

    def _compatible_dataset_ids(
        self,
        info: CourtCheckpointInfo,
        available: Mapping[str, Mapping[str, object]],
    ) -> list[str]:
        compatible: list[str] = []
        for dataset_id, entry in available.items():
            identity = layer_identity(
                source_kind=cast("CourtSourceKind", entry["source_kind"]),
                published_schema=cast("str | None", entry.get("schema")),
            )
            if _compatibility_reason(info, identity) is None:
                compatible.append(dataset_id)
        return compatible

    def _require_compatible(
        self, info: CourtCheckpointInfo, entry: CourtDatasetEntry
    ) -> None:
        identity = layer_identity(
            source_kind=entry.source_kind,
            published_schema=entry.published_schema,
        )
        reason = _compatibility_reason(info, identity)
        if reason is not None:
            raise ValueError(
                f"Court checkpoint {info.id!r} は dataset {entry.id!r} と比較できません: "
                f"{reason}"
            )

    def _prepare(
        self,
        *,
        checkpoint: str,
        scene: str,
        start: int,
        count: int,
        threshold: float,
        device: str,
    ) -> tuple[
        CourtCheckpointInfo, CourtDatasetEntry, CourtSampleRecord, CourtRawSample
    ]:
        self._require_single_frame(start=start, count=count)
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("Court のしきい値は [0, 1] の範囲で指定してください。")
        if device not in {"cuda", "cpu"}:
            raise ValueError(f"Court の device は 'cuda' か 'cpu' です: {device!r}")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA が要求されましたが、この process では利用できません。"
            )
        info = self._describe(checkpoint)
        entry, record = self.datasets.sample(scene)
        self._require_compatible(info, entry)
        _require_unchanged_checkpoint(info)
        layer = self._input(entry)
        self._require_spec_matches_bundle(info, layer.spec)
        raw = layer.load(record)
        return info, entry, record, raw

    @staticmethod
    def _require_spec_matches_bundle(
        info: CourtCheckpointInfo, spec: CourtInputSpec
    ) -> None:
        declared = tuple(spec.keypoint_channel_names)
        expected = info.keypoint_channel_names()
        if expected and declared != expected:
            raise ValueError(
                "dataset 契約の keypoint channel 順 "
                f"{declared} と checkpoint bundle の {expected} が一致しません。"
            )

    @staticmethod
    def _require_single_frame(*, start: int, count: int) -> None:
        if int(count) != 1:
            raise ValueError(
                "Court dataset は 1 sample = 1 枚の静止画です。count は 1 のみです: "
                f"{count}"
            )
        if int(start) != 0:
            raise ValueError(
                "Court dataset は 1 sample = 1 枚の静止画です。start は 0 のみです: "
                f"{start}"
            )

    def _render_prediction(
        self,
        predictions: Mapping[str, Any],
        *,
        masks: GroundTruthMasks,
        raw: CourtRawSample,
        threshold: float,
        channel_names: Sequence[str],
    ) -> tuple[dict[str, object], dict[str, object], list[str]]:
        rasters: list[dict[str, object]] = []
        metrics: dict[str, object] = {}
        warnings: list[str] = [_THRESHOLD_NOTE]
        points: list[dict[str, object]] = []
        kp = predictions.get("kp")
        if isinstance(kp, CourtKeypointPrediction):
            points, kp_raster = _keypoint_prediction_payload(
                kp,
                channel_names=channel_names,
                threshold=threshold,
            )
            rasters.append(kp_raster)
            if raw.keypoint_channels is not None:
                metrics.update(_keypoint_metrics(kp, raw))
        seg = predictions.get("seg")
        if isinstance(seg, CourtSegmentationPrediction):
            rasters.append(segmentation_raster(seg.mask.numpy()).to_dict())
        line = predictions.get("line")
        if isinstance(line, CourtLinePrediction):
            rasters.append(line_probability_raster(line.probability.numpy()).to_dict())
        semantic = predictions.get("semantic_line")
        if isinstance(semantic, CourtSegmentationPrediction):
            rasters.append(
                semantic_line_raster(
                    semantic.mask.numpy(), SEMANTIC_LINE_CHANNEL_NAMES
                ).to_dict()
            )
        sampled, sampled_warnings = _sample_metrics(
            predictions, masks=masks, threshold=threshold
        )
        metrics.update(sampled)
        warnings.extend(sampled_warnings)
        return (
            {
                "points": points,
                "segments": _keypoint_segments(points),
                "rasters": rasters,
            },
            metrics,
            warnings,
        )


def _keypoint_channel_names(channels: CourtKeypointChannels | None) -> tuple[str, ...]:
    if channels is None:
        return ()
    return tuple(channels.channel_names)


def _require_unchanged_checkpoint(info: CourtCheckpointInfo) -> None:
    """Reject a checkpoint whose bytes changed after the catalog was read."""
    try:
        stat = info.path.stat()
    except OSError as error:
        raise ValueError(
            f"Court checkpoint {info.id!r} が見つかりません。catalog を更新してください。"
        ) from error
    if (stat.st_size, stat.st_mtime_ns) != (info.size_bytes, info.modified_ns):
        raise ValueError(
            f"Court checkpoint {info.id!r} は catalog 読み込み後に disk 上で変化しました。"
            "catalog を更新してから推論してください。"
        )


def _keypoint_capacity(channels: CourtKeypointChannels | None) -> int:
    if channels is None:
        return 1
    return int(channels.points_xy.shape[1])


def _keypoint_prediction_payload(
    prediction: CourtKeypointPrediction,
    *,
    channel_names: Sequence[str],
    threshold: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Flatten decoded KP peaks into JSON points plus the mean heatmap raster."""
    keypoints = prediction.keypoints.detach().cpu().numpy()
    scores = prediction.scores.detach().cpu().numpy()
    valid = prediction.valid.detach().cpu().numpy()
    points: list[dict[str, object]] = []
    for channel in range(keypoints.shape[0]):
        label = channel_names[channel] if channel < len(channel_names) else str(channel)
        for peak in range(keypoints.shape[1]):
            if not bool(valid[channel, peak]):
                continue
            score = float(scores[channel, peak])
            points.append(
                {
                    "x": float(keypoints[channel, peak, 0]),
                    "y": float(keypoints[channel, peak, 1]),
                    "label": label,
                    "score": score,
                    "visible": score >= float(threshold),
                }
            )
    heatmaps = torch.sigmoid(prediction.heatmaps).amax(dim=0).numpy()
    return points, heatmap_raster(heatmaps).to_dict()


def _keypoint_metrics(
    prediction: CourtKeypointPrediction,
    raw: CourtRawSample,
) -> dict[str, object]:
    channels = raw.keypoint_channels
    if channels is None:
        return {}
    if int(channels.points_xy.shape[1]) != 1:
        return {"kp_scored_points": 0}
    # Assign through a typed local: the pre-commit hook runs mypy with
    # ``--follow-imports=skip``, so sibling modules resolve to ``Any`` there.
    metrics: dict[str, object] = keypoint_metrics(
        prediction,
        ground_truth_xy=channels.points_xy[:, 0, :].numpy().astype(np.float64),
        ground_truth_visible=channels.point_visible[:, 0].numpy(),
        physical_indices=channels.physical_indices[:, 0].numpy(),
    )
    return metrics


def _sample_metrics(
    predictions: Mapping[str, Any],
    *,
    masks: GroundTruthMasks,
    threshold: float,
) -> tuple[dict[str, object], list[str]]:
    """Score exactly the layers whose supervision schema matched the dataset."""
    metrics: dict[str, object] = {}
    warnings: list[str] = []
    seg = predictions.get("seg")
    if isinstance(seg, CourtSegmentationPrediction) and masks.seg is not None:
        metrics.update(categorical_metrics(seg, ground_truth=masks.seg, prefix="seg"))
        note = resize_note(seg.mask.numpy(), masks.seg)
        if note is not None:
            warnings.append(f"seg: {note}")
    line = predictions.get("line")
    if isinstance(line, CourtLinePrediction) and masks.line is not None:
        metrics.update(line_metrics(line, ground_truth=masks.line, threshold=threshold))
        note = resize_note(line.probability.numpy(), masks.line)
        if note is not None:
            warnings.append(f"line: {note}")
    semantic = predictions.get("semantic_line")
    if (
        isinstance(semantic, CourtSegmentationPrediction)
        and masks.semantic_line is not None
    ):
        metrics.update(
            categorical_metrics(
                semantic,
                ground_truth=masks.semantic_line,
                prefix="semantic_line",
            )
        )
        note = resize_note(semantic.mask.numpy(), masks.semantic_line)
        if note is not None:
            warnings.append(f"semantic_line: {note}")
    return metrics, warnings


def _keypoint_segments(
    points: Sequence[Mapping[str, object]],
) -> list[dict[str, float]]:
    lookup = {
        cast(str, point["label"]): (
            float(cast(float, point["x"])),
            float(cast(float, point["y"])),
        )
        for point in points
        if point.get("visible") is not False
    }
    pairs = (
        ("far_doubles_left", "far_doubles_right"),
        ("near_doubles_left", "near_doubles_right"),
    )
    if any(left not in lookup or right not in lookup for left, right in pairs):
        return []
    return [
        {
            "x1": lookup[left][0],
            "y1": lookup[left][1],
            "x2": lookup[right][0],
            "y2": lookup[right][1],
        }
        for left, right in pairs
    ]


def _compatibility_reason(
    info: CourtCheckpointInfo, identity: CourtDatasetLayers
) -> str | None:
    """Return why a checkpoint cannot be compared with a dataset, or ``None``."""
    if not info.kinds:
        return "checkpoint bundle に head がありません。"
    expected_kp = info.keypoint_channel_names()
    if expected_kp and tuple(identity.keypoint_channel_names) != expected_kp:
        return (
            "keypoint channel semantics が異なります: dataset "
            f"{tuple(identity.keypoint_channel_names)} と checkpoint {expected_kp}"
        )
    for kind, schema in info.dense_schemas().items():
        dataset_schema = identity.dense_schemas.get(kind)
        if dataset_schema != schema:
            return (
                f"{kind} の supervision schema が異なります: dataset {dataset_schema!r} と "
                f"checkpoint {schema!r}"
            )
    return None


__all__ = ["DEFAULT_THRESHOLD", "DetectionService", "TITLE"]
