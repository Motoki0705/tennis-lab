"""Review immutable caches without requiring their retired RGB/label store.

Saved evaluation teachers stay authoritative. A separately supplied RGB store
is only an image provider after exact JPEG, frame/PTS and geometry checks.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, fields
from fractions import Fraction
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord, shard_name
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tasks.ball_refiner.data.context_arrays import ContextArrays, GeneratedContext
from src.tasks.ball_refiner.data.evidence import ClipEvidence
from src.tasks.ball_refiner.data.targets import TargetReason
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.visualization.overlay import component_geometry
from src.utils.checksum import dual_sha256

Array: TypeAlias = NDArray[Any]


def read_document(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return value


def read_archive(root: Path, relative: str, digest: str) -> dict[str, Array]:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its declared directory")
    if dual_sha256(path) != digest:
        raise ValueError(f"Artifact checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    if dual_sha256(path) != digest:
        raise ValueError(f"Artifact changed while reading: {path}")
    return result


def clip_record(value: dict[str, Any]) -> ClipRecord:
    record = {field.name: value[field.name] for field in fields(ClipRecord)}
    record["track_ids"] = tuple(record["track_ids"])
    result = ClipRecord(**record)
    if (min(result.source_width, result.source_height, result.width, result.height) <= 1
            or result.width * result.source_height != result.height * result.source_width
            or result.frame_count < 1 or Fraction(result.time_base) <= 0):
        raise ValueError("Invalid cached clip geometry/timeline")
    return result


def assert_timeline(arrays: dict[str, Array], evidence: ClipEvidence) -> None:
    for name in ("frame_index", "pts"):
        value = arrays[name]
        expected = getattr(evidence, name)
        if value.dtype != expected.dtype or not np.array_equal(value, expected):
            raise ValueError(f"Artifact {name} differs from detector evidence")


@dataclass(frozen=True)
class SavedPrediction:
    arrays: dict[str, Array]
    distribution: BallGMM2D

    @classmethod
    def load(cls, arrays: dict[str, Array], evidence: ClipEvidence, condition: str) -> SavedPrediction:
        assert_timeline(arrays, evidence)
        n = len(evidence.frame_index)
        for name, shape, dtype in (
            ("target_uv", (n, 2), np.float32), ("target_reason", (n,), np.uint8),
            ("presence", (n,), np.bool_), ("presence_valid", (n,), np.bool_),
            ("gap_mask", (n,), np.bool_),
        ):
            if arrays[name].shape != shape or arrays[name].dtype != dtype:
                raise ValueError(f"Invalid saved teacher array: {name}")
        reasons = arrays["target_reason"]
        if not np.isin(reasons, list(TargetReason)).all():
            raise ValueError("Unknown saved teacher reason")
        observed = reasons == TargetReason.OBSERVED
        known = observed | (reasons == TargetReason.OUT_OF_FRAME)
        if not np.array_equal(arrays["presence"], observed) or not np.array_equal(arrays["presence_valid"], known):
            raise ValueError("Saved presence masks disagree with teacher reasons")
        located = np.isin(reasons, [TargetReason.OBSERVED, TargetReason.INTERPOLATED, TargetReason.OCCLUSION_ESTIMATED])
        uv = arrays["target_uv"][located]
        if not np.isfinite(uv).all() or ((uv < 0) | (uv > 1)).any():
            raise ValueError("Saved located teachers must use source endpoint UV")
        if condition == "observed" and arrays["gap_mask"].any():
            raise ValueError("Observed condition cannot hide detector evidence")
        distribution = BallGMM2D(**{
            field.name: torch.from_numpy(arrays[field.name])[None]
            for field in fields(BallGMM2D)
        })
        if distribution.presence_logits.shape != (1, n):
            raise ValueError("GMM frame count differs from evidence")
        return cls(arrays, distribution)


class ReviewArtifacts:
    """Explicit sources, lazy checksum validation, one loaded clip at a time."""

    def __init__(self, evidence_root: Path, *, predictions_root: Path | None = None,
                 context_root: Path | None = None, rgb_store: Path | None = None,
                 pose_threshold: float = 0.5) -> None:
        if not np.isfinite(pose_threshold) or not 0 <= pose_threshold <= 1:
            raise ValueError("pose_threshold must lie in [0,1]")
        self.evidence_root = evidence_root
        self.predictions_root = predictions_root
        self.context_root = context_root
        self.pose_threshold = pose_threshold
        manifest_path = evidence_root / "manifest.json"
        self.evidence_manifest = read_document(manifest_path)
        self.evidence_hash = dual_sha256(manifest_path)
        m = self.evidence_manifest
        if (m["schema"] != "ball_refiner_detector_evidence.v1" or m["status"] != "complete"
                or m["coordinate_system"] != "source_xy_div_size_minus_one"):
            raise ValueError("Incomplete or unsupported evidence cache")
        if (m["detector"]["window_selection"] != "nearest_window_centre_then_earlier_start"
                or m["detector"]["tail_policy"] != "backfill_real_frames_no_padding"):
            raise ValueError("Unsupported detector window policy")
        self.candidate_config = BallCandidateConfig(**m["detector"]["candidates"])
        self.records = {item["clip"]["clip_id"]: item for item in m["clips"]}
        if list(self.records) != m["selection"]["clip_ids"] or len(self.records) != len(m["clips"]):
            raise ValueError("Evidence selection contains missing/duplicate clips")
        self.clips = {cid: clip_record(item["clip"]) for cid, item in self.records.items()}
        self.prediction_manifest: dict[str, Any] | None = None
        self.predictions: dict[tuple[str, str], dict[str, Any]] = {}
        if predictions_root is not None:
            self._index_predictions(predictions_root)
        self.contexts: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
        if context_root is not None:
            self._index_contexts(context_root)
        self.rgb = None if rgb_store is None else BallFrameStore(rgb_store)
        self.rgb_status: dict[str, dict[str, Any]] = {}
        self._loaded_id: str | None = None
        self._evidence: ClipEvidence | None = None
        self._context: GeneratedContext | None = None
        self._saved: dict[str, SavedPrediction] = {}

    def _index_predictions(self, root: Path) -> None:
        m = read_document(root / "manifest.json")
        state = read_document(root / "run_state.json")
        if m["schema"] != "ball_refiner_cached_comparison.v1" or state["status"] != "complete":
            raise ValueError("Only complete cached-comparison GMM outputs are supported")
        inputs = m["input_sha256"]
        if inputs[str(self.evidence_root / "manifest.json")] != self.evidence_hash:
            raise ValueError("Saved GMM belongs to different detector evidence")
        original_store = Path(self.evidence_manifest["store"]["directory"])
        for name, digest in self.evidence_manifest["store"]["sha256"].items():
            if inputs[str(original_store / name)] != digest:
                raise ValueError("Saved teachers belong to a different original label store")
        for record in m["artifacts"]:
            cid, condition = record["clip_id"], record["condition"]
            key = (cid, condition)
            clip = self.clips[cid]
            if (key in self.predictions or condition not in {"observed", "evidence_gap"}
                    or record["frames"] != clip.frame_count or record["camera"] != clip.camera_id
                    or record["source"] != clip.source):
                raise ValueError("Saved GMM identity/condition is ambiguous")
            self.predictions[key] = record
        if set(self.predictions) != {(cid, condition) for cid, _ in self.predictions for condition in ("observed", "evidence_gap")}:
            raise ValueError("Saved predictions require paired observed/evidence_gap conditions")
        self.prediction_manifest = m

    def _index_contexts(self, root: Path) -> None:
        paths = [root / "manifest.json"] if (root / "manifest.json").is_file() else sorted(root.glob("*/manifest.json"))
        if not paths:
            raise ValueError("No context manifests found in explicitly supplied root")
        for path in paths:
            m = read_document(path)
            if (m["schema"] != "ball_refiner_context.v1" or m["status"] != "complete"
                    or m["rgb_condition"] != "unmodified_stored_jpeg_bgr.v1"
                    or m["coordinate_system"] != "stored_jpeg_pixels; source_xy=stored_xy/clip.scale"
                    or m["store"]["sha256"] != self.evidence_manifest["store"]["sha256"]):
                raise ValueError("Context is incomplete or has different labels/coordinates")
            selection = m["selection"]
            if selection["clip_ids"] != [x["clip"]["clip_id"] for x in m["clips"]]:
                raise ValueError("Context selection differs from its records")
            for item in m["clips"]:
                cid = item["clip"]["clip_id"]
                expected = self.records[cid]
                if (cid in self.contexts or item["clip"] != expected["clip"]
                        or item["jpeg_shard_sha256"] != expected["jpeg_shard_sha256"]):
                    raise ValueError("Context clip/JPEG identity differs from detector evidence")
                # Context may come from another detector run. This is declared,
                # never described as context used by the detector-only GMM.
                self.contexts[cid] = (path.parent, item, m)

    def catalog(self) -> dict[str, Any]:
        groups = Counter((c.source, c.split) for c in self.clips.values())
        return {
            "evidence": self.evidence_root.name,
            "predictions": None if self.predictions_root is None else self.predictions_root.name,
            "context": None if self.context_root is None else self.context_root.name,
            "rgb_provider": None if self.rgb is None else str(self.rgb.directory),
            "original_store": self.evidence_manifest["store"]["directory"],
            "original_store_present": Path(self.evidence_manifest["store"]["directory"]).is_dir(),
            "counts": {"clips": len(self.clips), "frames": sum(c.frame_count for c in self.clips.values()),
                       "context_clips": len(self.contexts), "teacher_gmm_clips": len(self.predictions) // 2},
            "groups": [{"source": source, "split": split, "clips": count,
                        "frames": sum(c.frame_count for c in self.clips.values() if c.source == source and c.split == split)}
                       for (source, split), count in sorted(groups.items())],
            "clips": [{"id": cid, "source": c.source, "split": c.split, "camera": c.camera_id,
                       "frames": c.frame_count, "has_context": cid in self.contexts,
                       "has_teachers": (cid, "observed") in self.predictions} for cid, c in self.clips.items()],
        }

    def load_clip(self, cid: str) -> ClipEvidence:
        if self._loaded_id == cid:
            assert self._evidence is not None
            return self._evidence
        record = self.records[cid]
        arrays = read_archive(self.evidence_root, record["file"], record["sha256"])
        evidence = ClipEvidence.from_arrays(arrays, config=self.candidate_config,
                                           heatmap_size_hw=tuple(record["heatmap_size_hw"]),
                                           window_length=self.evidence_manifest["detector"]["window_length"])
        clip = self.clips[cid]
        seconds = ((evidence.pts - evidence.pts[0]).astype(np.float64) * float(Fraction(clip.time_base))).astype(np.float32)
        if len(evidence.pts) != clip.frame_count or not np.array_equal(seconds, evidence.timestamps_seconds):
            raise ValueError("Evidence clip count/seconds differ from cached PTS/time_base")
        saved: dict[str, SavedPrediction] = {}
        if self.predictions_root is not None and (cid, "observed") in self.predictions:
            for condition in ("observed", "evidence_gap"):
                item = self.predictions[(cid, condition)]
                saved[condition] = SavedPrediction.load(read_archive(self.predictions_root, item["path"], item["sha256"]), evidence, condition)
            for name in ("target_uv", "target_reason", "presence", "presence_valid"):
                if not np.array_equal(saved["observed"].arrays[name], saved["evidence_gap"].arrays[name], equal_nan=True):
                    raise ValueError("Paired saved teacher arrays differ between conditions")
        context = None
        if cid in self.contexts:
            root, item, _ = self.contexts[cid]
            values = read_archive(root, item["file"], item["sha256"])
            assert_timeline(values, evidence)
            context = GeneratedContext(ContextArrays.from_arrays(values), item["execution"])
            court = context.arrays.court_points[context.arrays.court_valid]
            if ((court < 0) | (court > np.asarray([clip.width - 1, clip.height - 1]))).any():
                raise ValueError("Court context exceeds stored JPEG bounds")
        self._evidence, self._saved, self._context, self._loaded_id = evidence, saved, context, cid
        self.rgb_status[cid] = self._verify_rgb(cid, evidence)
        return evidence

    def _verify_rgb(self, cid: str, evidence: ClipEvidence) -> dict[str, Any]:
        if self.rgb is None:
            return {"available": False, "reason": "RGB provider was not supplied"}
        matches = [c for c in self.rgb.clips if c.clip_id == cid]
        if len(matches) != 1:
            return {"available": False, "reason": "Exact clip ID missing from supplied RGB provider"}
        actual, expected = matches[0], self.clips[cid]
        names = ("source", "group_id", "camera_id", "width", "height", "source_width", "source_height",
                 "frame_count", "time_base", "fps", "media_sha256")
        if any(getattr(actual, name) != getattr(expected, name) for name in names):
            return {"available": False, "reason": "RGB source identity/geometry differs from cached clip"}
        rows = self.rgb.clip_rows(actual)
        if (not np.array_equal(self.rgb.frames["frame_index"][rows], evidence.frame_index)
                or not np.array_equal(self.rgb.frames["pts"][rows], evidence.pts)):
            return {"available": False, "reason": "RGB frame/PTS differs from cached evidence"}
        digest = dual_sha256(self.rgb.directory / "shards" / shard_name(actual.index))
        if digest != self.records[cid]["jpeg_shard_sha256"]:
            return {"available": False, "reason": "RGB JPEG shard checksum differs from detector input"}
        return {"available": True, "reason": "Exact JPEG SHA256 + dense frame/PTS + source/stored size verified",
                "jpeg_sha256": digest, "provider": str(self.rgb.directory), "provider_clip_index": actual.index,
                "labels": "Saved evaluation NPZ only; RGB-provider labels are never substituted"}

    def image(self, cid: str, frame: int) -> bytes:
        self.load_clip(cid)
        if not self.rgb_status[cid]["available"] or self.rgb is None:
            raise FileNotFoundError(self.rgb_status[cid]["reason"])
        clip = self.rgb.clip_by_id(cid)
        row = int(self.rgb.clip_rows(clip)[frame])
        self.rgb.read_bgr(row)  # Verify the actual JPEG dimensions before serving.
        return bytes(self.rgb.read_jpeg(row))

    def timeline(self, cid: str) -> dict[str, Any]:
        evidence = self.load_clip(cid)
        saved = self._saved
        prediction = saved["observed"] if saved else None
        return {
            "frame_count": len(evidence.pts), "pts": [str(x) for x in evidence.pts],
            "timestamps": evidence.timestamps_seconds.tolist(),
            "candidate_count": evidence.candidates.valid[0].sum(-1).tolist(),
            "reason": None if prediction is None else prediction.arrays["target_reason"].tolist(),
            "position_valid": None if prediction is None else (prediction.arrays["target_reason"] == TargetReason.OBSERVED).tolist(),
            "presence_valid": None if prediction is None else prediction.arrays["presence_valid"].tolist(),
            "presence_target": None if prediction is None else prediction.arrays["presence"].tolist(),
            "gap_mask": None if not saved else saved["evidence_gap"].arrays["gap_mask"].tolist(),
            "pose_count": None if self._context is None else self._context.arrays.track_observed.sum(-1).tolist(),
            "rgb": self.rgb_status[cid],
        }

    def frame(self, cid: str, frame: int, condition: str) -> dict[str, Any]:
        if condition not in {"observed", "evidence_gap"}:
            raise ValueError("Unknown evaluation condition")
        evidence = self.load_clip(cid)
        clip = self.clips[cid]
        if not 0 <= frame < clip.frame_count:
            raise IndexError("Frame is outside the real clip timeline")
        if condition == "evidence_gap" and not self._saved:
            raise ValueError("No saved evidence_gap output for this clip")
        prediction = self._saved[condition] if self._saved else None
        gap = prediction is not None and bool(prediction.arrays["gap_mask"][frame])
        c = evidence.candidates
        candidates = [{"slot": slot, "uv": c.coords[0, frame, slot].tolist(),
                       "score": float(c.scores[0, frame, slot]), "cell": c.cells[0, frame, slot].tolist(),
                       "patch": c.patches[0, frame, slot].tolist(), "patch_valid": c.patch_valid[0, frame, slot].tolist()}
                      for slot in range(c.config.max_candidates) if bool(c.valid[0, frame, slot])]
        target, gmm = None, None
        if prediction is not None:
            a, d = prediction.arrays, prediction.distribution
            reason = TargetReason(int(a["target_reason"][frame]))
            located = reason in {TargetReason.OBSERVED, TargetReason.INTERPOLATED, TargetReason.OCCLUSION_ESTIMATED}
            target = {"reason": reason.name.lower(), "reason_code": int(reason),
                      "uv": a["target_uv"][frame].tolist() if located else None,
                      "position_valid": reason == TargetReason.OBSERVED,
                      "presence_valid": bool(a["presence_valid"][frame]),
                      "presence": bool(a["presence"][frame]) if bool(a["presence_valid"][frame]) else None}
            _, ellipses = component_geometry(a["means"][frame], a["scale_tril"][frame], a["mixture_logits"][frame],
                                             (clip.source_width - 1, clip.source_height - 1))
            gmm = {"presence_probability": float(d.presence_probability[0, frame]), "calibration": "uncalibrated",
                   "components": [{"uv": a["means"][frame, i].tolist(), "weight": e.weight,
                                   "center_px": list(e.center), "semiaxes_px": list(e.semiaxes),
                                   "angle_degrees": e.angle_degrees} for i, e in enumerate(ellipses)]}
        context = None
        if self._context is not None:
            a = self._context.arrays
            factor = clip.scale * np.asarray([clip.source_width - 1, clip.source_height - 1], np.float32)
            people = []
            for i in np.flatnonzero(a.track_observed[frame]):
                raw = a.keypoints[frame, i]
                confidence = np.clip(raw[:, 2], 0, 1)
                people.append({"id": int(a.track_ids[i]), "uv": (raw[:, :2] / factor).tolist(),
                               "raw_peak": raw[:, 2].tolist(),
                               "valid": ((confidence > 0) & (confidence >= self.pose_threshold)).tolist()})
            _, _, manifest = self.contexts[cid]
            context = {"people": people, "detection_count": int(a.detection_count[frame]),
                       "court_uv": (a.court_points / factor).tolist(), "court_valid": a.court_valid.tolist(),
                       "court_frame": 0, "pose_threshold": self.pose_threshold,
                       "detector_manifest_sha256": manifest["evidence"]["manifest_sha256"],
                       "binding": "same store + exact clip/JPEG hash + frame/PTS; separate comparison context",
                       "used_by_saved_gmm": False}
        return {"clip_id": cid, "source": clip.source, "camera": clip.camera_id, "split": clip.split,
                "frame": frame, "frame_count": clip.frame_count, "pts": str(evidence.pts[frame]),
                "seconds": float(evidence.timestamps_seconds[frame]), "condition": condition,
                "source_size_wh": [clip.source_width, clip.source_height], "stored_size_wh": [clip.width, clip.height],
                "stored_scale": clip.scale, "target": target, "gmm": gmm, "context": context,
                "candidates": candidates, "effective_candidates": [] if gap else candidates,
                "gap_active": gap, "heatmap_size_hw": list(evidence.heatmap_size_hw),
                "detector_rgb_support": [int(evidence.window_start[frame]), int(evidence.window_start[frame]) + evidence.window_length],
                "rgb": self.rgb_status[cid],
                "provenance": {"evidence_manifest": str(self.evidence_root / "manifest.json"),
                               "evidence_manifest_sha256": self.evidence_hash,
                               "evidence_npz_sha256": self.records[cid]["sha256"],
                               "teacher_gmm_npz_sha256": None if prediction is None else self.predictions[(cid, condition)]["sha256"],
                               "context_npz_sha256": None if cid not in self.contexts else self.contexts[cid][1]["sha256"],
                               "coordinate_system": "source_xy_div_size_minus_one",
                               "teacher_source": "saved evaluation NPZ from original store; not RGB-provider annotations"}}
