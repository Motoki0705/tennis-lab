"""Read-only dataset review and bounded ball inference for the web UI.

The service is the single backend the shared detection app
(``src.tasks.base.visualization.detection``) talks to.  It owns two things:

* the dataset/scene/frame catalog built from the ball-detection sources
  (TrackNet, annotated YouTube frames, and the optional unified web store), and
* one bounded inference window per request, executed on CPU in-process or on
  CUDA through the shared GPU queue owned by the HTTP layer.

Ground truth is exactly what the datasets store: ``FrameLabel`` points in
original-image pixels.  Nothing is interpolated, padded, or invented -- a
window that does not fit the selected scene is rejected, and a static web frame
is only ever expanded through the dataset's own canonical static sampling mode,
which is labelled in the response.

Scene ids are opaque (``<dataset>::<scene>``) and are resolved through the
catalog, so the API never accepts a caller-supplied filesystem path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_detection.training.metrics import BallDetectionMetrics
from src.tasks.ball_detection.visualization.inference.loader import (
    LoadedBallModel,
    load_ball_model,
)
from src.tasks.ball_detection.visualization.inference.peaks import (
    decode_frame_peaks,
    peaks_to_points,
)
from src.tasks.ball_detection.visualization.inference.rasters import probability_raster
from src.tasks.ball_detection.visualization.review.checkpoints import (
    TASK_NAME,
    BallCheckpointInfo,
    checkpoint_roots,
    describe_checkpoint,
    file_state,
    rejected_checkpoint,
    scan_checkpoints,
)
from src.tasks.ball_detection.visualization.review.datasets import (
    DATASET_SPECS,
    BallDatasetCatalog,
    BallDatasetCatalogError,
    BallDatasetSpec,
    DatasetEntry,
    SceneFrames,
    SceneRef,
    split_scene_id,
)
from src.utils.device import DeviceSelectionError, resolve_device

TASK: Final = TASK_NAME
TITLE: Final = "Ball Detection"

#: The shared web layer caps previews at this many frames per request.
PREVIEW_FRAME_LIMIT: Final = 64

WindowMode = Literal["temporal", "static_repeat"]


class DetectionRequestError(ValueError):
    """Raised when a review or inference request is invalid."""


@dataclass(frozen=True, slots=True)
class WindowPlan:
    """One resolved inference window over scene frame positions."""

    positions: tuple[int, ...]
    mode: WindowMode
    repeat: int

    @property
    def start(self) -> int:
        """Return the first scene frame position in the window."""
        return self.positions[0]

    @property
    def end(self) -> int:
        """Return the last scene frame position in the window."""
        return self.positions[-1]

    def to_dict(self, *, checkpoint_frames: int) -> dict[str, Any]:
        """Return the window description embedded in the metrics payload."""
        return {
            "mode": self.mode,
            "start": self.start,
            "end": self.end,
            "count": len(self.positions),
            "repeat": self.repeat,
            "checkpoint_frames": checkpoint_frames,
        }


class DetectionService:
    """Catalog, frame access, and bounded inference for ball detection."""

    def __init__(
        self,
        project_root: str | Path,
        data_root: str | Path | None = None,
        outputs_root: str | Path | None = None,
        checkpoints_root: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve(strict=False)
        self.data_root = (
            Path(data_root).expanduser().resolve(strict=False)
            if data_root is not None
            else self.project_root / "data"
        )
        self.outputs_root = (
            Path(outputs_root).expanduser().resolve(strict=False)
            if outputs_root is not None
            else self.project_root / "outputs" / TASK
        )
        self.checkpoints_root = (
            Path(checkpoints_root).expanduser().resolve(strict=False)
            if checkpoints_root is not None
            else self.project_root / "ckpt" / TASK
        )
        self.dataset_catalog = BallDatasetCatalog(self.data_root)
        self._checkpoint_cache: dict[str, BallCheckpointInfo] | None = None

    # ------------------------------------------------------------- catalog

    def checkpoints(self) -> dict[str, BallCheckpointInfo]:
        """Return every discovered checkpoint keyed by its catalog id."""
        if self._checkpoint_cache is None:
            self._checkpoint_cache = self._scan_checkpoints()
        return self._checkpoint_cache

    def _scan_checkpoints(self) -> dict[str, BallCheckpointInfo]:
        roots = checkpoint_roots(self.outputs_root, self.checkpoints_root)
        return {info.id: info for info in scan_checkpoints(roots)}

    def catalog(self) -> dict[str, Any]:
        """Rescan disk and return datasets, checkpoints, and their warnings.

        The UI ships a catalog refresh control, so every call re-enumerates the
        data and checkpoint roots instead of serving a process-lifetime cache: a
        new clip or a newly copied ``.ckpt`` must show up without a restart, and
        a dataset that became unavailable must keep reporting its reason.
        """
        self.dataset_catalog.refresh()
        self._checkpoint_cache = self._scan_checkpoints()
        entries = self.dataset_catalog.entries()
        warnings: list[str] = []
        datasets = [entry.to_dict() for entry in entries]
        for entry in entries:
            if not entry.available and entry.reason:
                warnings.append(f"{entry.spec.id}: {entry.reason}")
            warnings.extend(f"{entry.spec.id}: {text}" for text in entry.warnings)

        checkpoints: list[dict[str, Any]] = []
        for info in self.checkpoints().values():
            compatible = self._compatible_datasets(info, entries)
            checkpoints.append(info.to_dict(compatible_datasets=compatible))
            if info.error:
                warnings.append(f"{info.id}: {info.error}")
            warnings.extend(f"{info.id}: {text}" for text in info.warnings)
        return {
            "task": TASK,
            "title": TITLE,
            "datasets": datasets,
            "checkpoints": checkpoints,
            "warnings": warnings,
        }

    @staticmethod
    def _compatible_datasets(
        info: BallCheckpointInfo, entries: Sequence[DatasetEntry]
    ) -> list[str]:
        """Return the datasets a checkpoint can actually run on.

        The compatibility rule is architectural, not a re-reading of the
        training ``data.source``:

        * a checkpoint whose saved config cannot be read, or whose model is
          unsupported, runs nowhere;
        * a static dataset is always runnable because the shared web store
          defines a canonical "repeat one labelled frame to the model window"
          sampling mode, which the window plan labels explicitly;
        * a temporal dataset needs at least one scene long enough to hold the
          checkpoint's smallest honest window, which is the architecture
          minimum (plus the MDD two-frame requirement for multi-frame MDD
          checkpoints).
        """
        if not info.usable:
            return []
        compatible: list[str] = []
        for entry in entries:
            if entry.count == 0:
                continue
            if entry.spec.mode == "static":
                compatible.append(entry.spec.id)
                continue
            if entry.max_scene_frames >= info.minimum_window:
                compatible.append(entry.spec.id)
        return compatible

    # --------------------------------------------------------------- scenes

    def scenes(
        self,
        dataset: str,
        search: str = "",
        offset: int = 0,
        limit: int = 100,
        checkpoint: str | None = None,
    ) -> dict[str, Any]:
        """Return one page of scenes for a dataset, optionally filtered."""
        spec = self._spec(dataset)
        if offset < 0:
            raise DetectionRequestError("offset must be non-negative.")
        if limit <= 0:
            raise DetectionRequestError("limit must be positive.")
        if checkpoint is not None:
            info = self._checkpoint(checkpoint)
            if dataset not in self._compatible_datasets(
                info, self.dataset_catalog.entries()
            ):
                raise DetectionRequestError(
                    f"Checkpoint {checkpoint!r} cannot run on dataset {dataset!r}."
                )
        needle = search.strip().lower()
        refs = [
            ref
            for ref in self.dataset_catalog.refs(spec.id)
            if not needle or needle in ref.label.lower()
        ]
        if checkpoint is not None:
            # Dataset-level compatibility only says the source is usable at all;
            # a short clip inside a compatible dataset still cannot hold the
            # checkpoint's window, so each scene is filtered on its own length.
            info = self._checkpoint(checkpoint)
            refs = [
                ref for ref in refs if self._scene_supports_checkpoint(ref, info, spec)
            ]
        window = refs[offset : offset + limit]
        return {"items": [ref.to_dict() for ref in window], "total": len(refs)}

    @staticmethod
    def _scene_supports_checkpoint(
        ref: SceneRef, info: BallCheckpointInfo, spec: BallDatasetSpec
    ) -> bool:
        """Return whether one scene can hold this checkpoint's smallest window."""
        if spec.mode == "static":
            # A static scene is one labelled frame expanded by the store's own
            # static sampling mode, so its length never limits the window.
            return True
        return bool(ref.frames >= info.minimum_window)

    # -------------------------------------------------------------- frames

    def preview(self, scene: str, start: int = 0, count: int = 1) -> dict[str, Any]:
        """Return original-size ground truth for a bounded frame range."""
        if count < 1 or count > PREVIEW_FRAME_LIMIT:
            raise DetectionRequestError(
                f"count must be within [1, {PREVIEW_FRAME_LIMIT}], got {count}."
            )
        resolved = self._resolve_scene(scene)
        frames = resolved.frames.frames
        self._check_range(start, count, frames=frames, label="preview")
        warnings: list[str] = []
        width, height = resolved.frames.original_size(start)
        items: list[dict[str, Any]] = []
        for index in range(start, start + count):
            size = resolved.frames.original_size(index)
            if size != (width, height):
                warnings.append(
                    f"frame {index} is {size[0]}x{size[1]}, not "
                    f"{width}x{height}; the viewer uses one image geometry."
                )
            if not resolved.frames.annotated(index):
                warnings.append(
                    f"frame {index} has no annotation row; it is unlabelled, not "
                    "an annotated negative."
                )
            items.append(
                {
                    "index": index,
                    "name": resolved.frames.name(index),
                    "gt": {
                        "points": self._label_points(resolved.frames, index),
                        "rasters": [],
                    },
                    "annotated": resolved.frames.annotated(index),
                }
            )
        return {
            "scene": resolved.ref.id,
            "label": resolved.ref.label,
            "frames": frames,
            "start": start,
            "width": width,
            "height": height,
            "items": items,
            "warnings": warnings,
        }

    @staticmethod
    def _label_points(scene: SceneFrames, index: int) -> list[dict[str, Any]]:
        """Return every labelled instance as a pixel-space point."""
        points: list[dict[str, Any]] = []
        for label in scene.labels(index):
            points.append(
                {
                    "x": float(label.x),
                    "y": float(label.y),
                    "label": label.instance_id or label.role,
                    "score": float(label.visibility),
                    "visible": bool(label.visibility > 0),
                }
            )
        return points

    def image(self, scene: str, frame: int) -> bytes:
        """Return one frame as JPEG bytes in original resolution."""
        resolved = self._resolve_scene(scene)
        frames = resolved.frames.frames
        if frame < 0 or frame >= frames:
            raise DetectionRequestError(
                f"frame {frame} is out of range [0, {frames - 1}]."
            )
        rgb = resolved.frames.read_rgb(frame)
        ok, buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        if not ok:
            raise RuntimeError(f"Failed to encode frame {frame} of scene {scene!r}.")
        return bytes(buffer.tobytes())

    # ----------------------------------------------------------- inference

    def validate(
        self,
        checkpoint: str,
        scene: str,
        start: int = 0,
        count: int = 1,
        threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        """Validate a request without loading weights or touching the GPU."""
        resolved = self._resolve_scene(scene)
        # The web layer calls this immediately before inference, so it re-reads a
        # replaced checkpoint's metadata here too: a request must never be
        # validated against one body and executed against another.
        info = self._checkpoint(checkpoint, refresh_metadata=True)
        self._plan_window(info, resolved, start=start, count=count)
        self._check_threshold(threshold)
        self._resolve_device(device)

    def infer(
        self,
        checkpoint: str,
        scene: str,
        start: int = 0,
        count: int = 1,
        threshold: float = 0.5,
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Run one bounded window and return prediction points and metrics."""
        resolved = self._resolve_scene(scene)
        # Re-read the checkpoint's own metadata when its file changed since the
        # catalog listed it, so a replaced body is never inferred with the old
        # architecture, window, or thresholds.
        info = self._checkpoint(checkpoint, refresh_metadata=True)
        plan = self._plan_window(info, resolved, start=start, count=count)
        self._check_threshold(threshold)
        resolved_device = self._resolve_device(device)
        loaded = load_ball_model(info.path, device=resolved_device)
        return self._run(info, loaded, resolved, plan, threshold=threshold)

    def _run(
        self,
        info: BallCheckpointInfo,
        loaded: LoadedBallModel,
        resolved: _ResolvedScene,
        plan: WindowPlan,
        *,
        threshold: float,
    ) -> dict[str, Any]:
        size = loaded.image_size_hw
        if size is None:
            raise DetectionRequestError(
                f"Checkpoint {info.id!r} does not declare an input image size."
            )
        warnings: list[str] = []
        if plan.mode == "static_repeat":
            warnings.append(
                "Static scene: the labelled frame was repeated to the checkpoint "
                f"window ({plan.repeat} frames) through the store's canonical "
                "static sampling mode."
            )
        images = self._window_tensor(resolved.frames, plan, size=size)
        with torch.no_grad():
            call = loaded.adapter.prepare_model_call(
                images.to(loaded.device), image_normalization=loaded.image_normalization,
            )
            logits = loaded.model(*call.model_args)
            heatmaps = loaded.adapter.probability_heatmaps(logits, call)
        heatmaps = heatmaps.detach().cpu()
        if heatmaps.shape[0] != 1:
            raise RuntimeError(
                f"Ball inference expected one window, got {heatmaps.shape[0]}."
            )
        window_heatmaps = heatmaps[0]
        original_size = resolved.frames.original_size(plan.start)
        if plan.mode == "static_repeat":
            # The repeated frames are one original frame, so they collapse back
            # to one query.  Averaging the window's probabilities is the explicit
            # aggregation; emitting one item per repeat would let the viewer's
            # frame map keep an arbitrary duplicate and would score the same
            # ground truth once per repeat.
            emitted = [(plan.start, window_heatmaps.mean(dim=0, keepdim=True)[0])]
            warnings.append(
                f"Averaged {plan.repeat} repeated-window heatmaps into one "
                "prediction for the single labelled frame."
            )
        else:
            emitted = [
                (position, window_heatmaps[offset])
                for offset, position in enumerate(plan.positions)
            ]
        peaks = decode_frame_peaks(
            torch.stack([heatmap for _, heatmap in emitted]),
            original_size=original_size,
            threshold=threshold,
            nms_kernel=info.metrics.nms_kernel,
            max_peaks=info.metrics.max_predictions_per_frame,
            subpixel_refine=info.metrics.subpixel_refine,
        )
        items = [
            {
                "index": position,
                "pred": {
                    "points": peaks_to_points(frame_peaks),
                    "rasters": [probability_raster(heatmap.numpy()).to_dict()],
                },
            }
            for (position, heatmap), frame_peaks in zip(emitted, peaks, strict=True)
        ]
        metrics = self._metrics(
            info=info,
            resolved=resolved,
            plan=plan,
            emitted=emitted,
            threshold=threshold,
        )
        for position in metrics["excluded_frames"]:
            warnings.append(
                f"frame {position} has no annotation row; it is unlabelled, not an "
                "annotated negative, and was excluded from the metrics."
            )
        return {
            "scene": resolved.ref.id,
            "start": plan.start,
            "items": items,
            "metrics": metrics,
            "warnings": warnings,
        }

    def _metrics(
        self,
        *,
        info: BallCheckpointInfo,
        resolved: _ResolvedScene,
        plan: WindowPlan,
        emitted: Sequence[tuple[int, torch.Tensor]],
        threshold: float,
    ) -> dict[str, Any]:
        """Score the emitted frames with the repository's canonical metrics.

        Only frames that carry an annotation row are scored.  A frame whose row is
        missing is *not* an annotated negative, so counting it would both invent a
        negative and (for a repeated static frame) multiply one frame's ground
        truth by the repeat count.  When nothing in the window is annotated the
        metric is reported as unavailable with the reason, never as a zero.
        """
        excluded = [
            position
            for position, _ in emitted
            if not resolved.frames.annotated(position)
        ]
        scored = [
            (position, heatmap)
            for position, heatmap in emitted
            if resolved.frames.annotated(position)
        ]
        base: dict[str, Any] = {
            "window": plan.to_dict(checkpoint_frames=info.num_frames),
            "ball_distance_threshold_px": info.metrics.ball_distance_threshold,
            "scored_frames": [position for position, _ in scored],
            "excluded_frames": excluded,
        }
        if not scored:
            base.update(
                {
                    "available": False,
                    "reason": (
                        "no frame in this window has an annotation row, so the "
                        "metric cannot be computed."
                    ),
                }
            )
            return base

        frame_count = len(scored)
        max_instances = max(
            len(resolved.frames.labels(position)) for position, _ in scored
        )
        coords: NDArray[np.float32] = np.zeros(
            (1, frame_count, max_instances, 2), dtype=np.float32
        )
        visibility: NDArray[np.float32] = np.zeros(
            (1, frame_count, max_instances), dtype=np.float32
        )
        for offset, (position, _) in enumerate(scored):
            for slot, label in enumerate(resolved.frames.labels(position)):
                coords[0, offset, slot] = (float(label.x), float(label.y))
                visibility[0, offset, slot] = 1.0 if label.visibility > 0 else 0.0
        width, height = resolved.frames.original_size(plan.start)
        # The built-in metric cannot accept an empty instance axis; one unused
        # slot keeps the tensor shape valid and contributes nothing because its
        # visibility is zero.
        if max_instances == 0:
            coords = np.zeros((1, frame_count, 1, 2), dtype=np.float32)
            visibility = np.zeros((1, frame_count, 1), dtype=np.float32)
        tracker = BallDetectionMetrics(
            peak_threshold=threshold,
            ball_distance_threshold=info.metrics.ball_distance_threshold,
            nms_kernel=info.metrics.nms_kernel,
            max_predictions_per_frame=info.metrics.max_predictions_per_frame,
            subpixel_refine=info.metrics.subpixel_refine,
        )
        scored_heatmaps = torch.stack([heatmap for _, heatmap in scored])
        tracker.update(
            scored_heatmaps.unsqueeze(0),
            torch.from_numpy(coords),
            torch.from_numpy(visibility),
            torch.tensor([[float(width), float(height)]], dtype=torch.float32),
        )
        values: dict[str, Any] = {
            name: float(value.detach().cpu().item())
            for name, value in tracker.compute().items()
        }
        values["matched_detections"] = int(tracker.distance_count.item())
        if values["matched_detections"] == 0:
            values["mean_distance_px"] = None
        values.update(base)
        values["available"] = True
        values["note"] = (
            "precision/recall/f1 use Hungarian matching at "
            "metrics.ball_distance_threshold in original image pixels."
        )
        return values

    @staticmethod
    def _window_tensor(
        scene: SceneFrames,
        plan: WindowPlan,
        *,
        size: tuple[int, int],
    ) -> torch.Tensor:
        """Build the ``(1, T, 3, H, W)`` float32 ``[0, 1]`` model window."""
        height, width = size
        original_sizes = {scene.original_size(position) for position in plan.positions}
        if len(original_sizes) != 1:
            raise DetectionRequestError(
                "All frames in one inference window must share one original size; "
                f"got {sorted(original_sizes)}. Prediction pixels would be "
                "ambiguous otherwise."
            )
        frames: list[NDArray[np.float32]] = []
        for position in plan.positions:
            rgb = scene.read_rgb(position)
            resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            frames.append(resized.astype(np.float32) / 255.0)
        stacked = np.stack(frames)
        tensor = torch.from_numpy(np.transpose(stacked, (0, 3, 1, 2)))
        return tensor.unsqueeze(0).contiguous()

    # ------------------------------------------------------------ helpers

    def _spec(self, dataset: str) -> BallDatasetSpec:
        try:
            return self.dataset_catalog.spec(dataset)
        except BallDatasetCatalogError as error:
            raise DetectionRequestError(str(error)) from error

    def _checkpoint(
        self, checkpoint: str, *, refresh_metadata: bool = False
    ) -> BallCheckpointInfo:
        """Return one usable checkpoint description.

        ``refresh_metadata`` re-reads the checkpoint's own config when the file's
        ``(size, mtime_ns)`` no longer matches the cached description, which is
        what keeps a request from running a replaced body with the previous
        metadata.
        """
        info = self.checkpoints().get(checkpoint)
        if info is None:
            known = ", ".join(sorted(self.checkpoints())) or "none"
            raise DetectionRequestError(
                f"Unknown checkpoint {checkpoint!r}; available: {known}."
            )
        if refresh_metadata:
            info = self._refresh_checkpoint(info)
        if info.error is not None:
            raise DetectionRequestError(
                f"Checkpoint {checkpoint!r} is unusable: {info.error}"
            )
        return info

    def _refresh_checkpoint(self, info: BallCheckpointInfo) -> BallCheckpointInfo:
        """Re-read a checkpoint description whose file changed on disk."""
        if info.error is not None:
            # A checkpoint the catalog already rejected stays rejected: refreshing
            # metadata must not resurrect a policy rejection or a corrupt body.
            # Picking up a repaired body is what ``catalog()`` rescanning is for.
            return info
        try:
            state = file_state(info.path)
        except FileNotFoundError as error:
            raise DetectionRequestError(
                f"Checkpoint {info.id!r} disappeared from {info.path}."
            ) from error
        if state == info.state:
            return info
        if not self._within_checkpoint_roots(info.path):
            # The file may have been swapped for a symlink that escapes the
            # configured roots between the catalog call and this request.
            rejected = rejected_checkpoint(
                checkpoint_id=info.id,
                path=info.path,
                label=info.label,
                error=(
                    "checkpoint resolves outside the configured checkpoint roots; "
                    "refusing to load it."
                ),
            )
            if self._checkpoint_cache is not None:
                self._checkpoint_cache[info.id] = rejected
            return rejected
        refreshed = describe_checkpoint(
            checkpoint_id=info.id, path=info.path, label=info.label
        )
        if self._checkpoint_cache is not None:
            self._checkpoint_cache[info.id] = refreshed
        return refreshed

    def _within_checkpoint_roots(self, path: Path) -> bool:
        """Return whether ``path`` resolves inside one configured checkpoint root."""
        try:
            resolved = path.resolve()
        except OSError:
            return False
        return any(
            resolved.is_relative_to(root)
            for root, _ in checkpoint_roots(self.outputs_root, self.checkpoints_root)
        )

    def _resolve_scene(self, scene: str) -> _ResolvedScene:
        try:
            dataset_id, local_id = split_scene_id(scene)
            ref = self.dataset_catalog.scene_ref(dataset_id, local_id)
            frames = self.dataset_catalog.resolve(dataset_id, local_id)
        except BallDatasetCatalogError as error:
            raise DetectionRequestError(str(error)) from error
        return _ResolvedScene(ref=ref, frames=frames)

    @staticmethod
    def _check_threshold(threshold: float) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, int | float):
            raise DetectionRequestError("threshold must be a number.")
        if not 0.0 <= float(threshold) <= 1.0:
            raise DetectionRequestError(
                f"threshold must be within [0, 1], got {threshold}."
            )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        try:
            return resolve_device(device)
        except DeviceSelectionError as error:
            raise DetectionRequestError(str(error)) from error

    @staticmethod
    def _check_range(start: int, count: int, *, frames: int, label: str) -> None:
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise DetectionRequestError(f"{label} start must be a non-negative int.")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise DetectionRequestError(f"{label} count must be a positive int.")
        if start + count > frames:
            raise DetectionRequestError(
                f"{label} range [{start}, {start + count}) exceeds the scene's "
                f"{frames} frame(s)."
            )

    def _plan_window(
        self,
        info: BallCheckpointInfo,
        resolved: _ResolvedScene,
        *,
        start: int,
        count: int,
    ) -> WindowPlan:
        """Resolve the exact frame positions a request will feed the model."""
        frames = resolved.frames.frames
        mode = resolved.ref_mode
        self._check_range(start, 1, frames=frames, label="inference")
        if mode == "static":
            if count != info.num_frames:
                raise DetectionRequestError(
                    f"{info.id!r} consumes {info.num_frames} frames; a static scene "
                    "is expanded by repeating one labelled frame, so count must "
                    f"equal {info.num_frames}, got {count}."
                )
            return WindowPlan(
                positions=(start,) * info.num_frames,
                mode="static_repeat",
                repeat=info.num_frames,
            )
        self._check_range(start, count, frames=frames, label="inference")
        if count < info.minimum_window or count > info.maximum_window:
            raise DetectionRequestError(
                f"{info.id!r} accepts a temporal window between "
                f"{info.minimum_window} and {info.maximum_window} frames, "
                f"got count={count}."
            )
        return WindowPlan(
            positions=tuple(range(start, start + count)),
            mode="temporal",
            repeat=1,
        )


@dataclass(frozen=True, slots=True)
class _ResolvedScene:
    """One catalogued scene plus its materialised frame accessor."""

    ref: SceneRef
    frames: SceneFrames

    @property
    def ref_mode(self) -> Literal["static", "temporal"]:
        """Return the dataset's sampling mode, not the frame accessor's."""
        return cast(Literal["static", "temporal"], self.frames.mode)


def dataset_ids() -> tuple[str, ...]:
    """Return the catalog dataset ids in display order."""
    return tuple(spec.id for spec in DATASET_SPECS)


__all__ = [
    "PREVIEW_FRAME_LIMIT",
    "TASK",
    "TITLE",
    "DetectionRequestError",
    "DetectionService",
    "WindowPlan",
    "WindowMode",
    "dataset_ids",
]
