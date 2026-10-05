"""Display saved production diagnostics without scoring or judging hypotheses."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.court_side.review.store import StoreCase
from src.utils.checksum import dual_sha256


class SavedDiagnostics:
    """Bind the published diagnosis CSV/NPZ to the exact failed execution and points."""

    def __init__(self, directory: Path, case: StoreCase) -> None:
        report_path = directory / "production.json"
        report = json.loads(report_path.read_text())
        receipts = report["inputs"]
        for receipt in receipts:
            path = Path(receipt["path"])
            if not path.is_file() or dual_sha256(path) != receipt["sha256"]:
                raise ValueError("Saved diagnostic input checksum mismatch")
        executions = [
            Path(receipt["path"])
            for receipt in receipts
            if Path(receipt["path"]).name == "execute.json"
        ]
        if len(executions) != 1:
            raise ValueError("Diagnostics require one saved execute.json receipt")
        execution = json.loads(executions[0].read_text())["run"]
        if (
            Path(execution["scene_index"]).resolve() != case.index
            or execution["source"] != case.source
        ):
            raise ValueError("Diagnostic source/store mismatch")
        for node in case.descriptors:
            if execution["artifacts"].get(node) != case.references[node]:
                raise ValueError(
                    "Diagnostic artifact lineage differs from saved execution"
                )
        original = execution["error_diagnostics"]
        for key in ("frames", "hypotheses", "pair_frames", "sampled_frames"):
            if original[key] != report[key]:
                raise ValueError(f"Diagnostic and execution disagree: {key}")
        if (
            execution["error_reason"] != f"court_side_{report['reason']}"
            or report["decided"]
        ):
            raise ValueError("Expected a saved production stop")
        if case.decision is not None:
            raise ValueError(
                "A stopped diagnostic cannot replace a published side decision"
            )
        if (
            case.calibration is None
            or [v["camera_id"] for v in case.calibration["views"]] != case.camera_ids
        ):
            raise ValueError(
                "Saved diagnosis requires its complete calibrated camera order"
            )
        if any(case.points[camera] is None for camera in case.camera_ids):
            raise ValueError("Diagnostic needs every saved ball stream")
        observations_path = directory / "production-observations.npz"
        with np.load(observations_path, allow_pickle=False) as arrays:
            uv, visible = arrays["uv_px"], arrays["visible"]
            self.sampled = arrays["sampled_frame_indices"].copy()
            self.distinct = arrays["distinct_mask"].copy()
            self.scored = arrays["scored_mask"].copy()
        points = [case.points[camera] for camera in case.camera_ids]
        if not np.array_equal(
            uv, np.stack([p["uv"] for p in points if p is not None])
        ) or not np.array_equal(
            visible, np.stack([p["observed"] for p in points if p is not None])
        ):
            raise ValueError("Diagnostic observations differ from the published store")
        if (
            self.sampled.ndim != 1
            or self.sampled.dtype != np.int64
            or (np.diff(self.sampled) <= 0).any()
            or (self.sampled < 0).any()
            or (self.sampled >= case.frames).any()
        ):
            raise ValueError("Invalid saved sampling axis")
        if (
            self.distinct.dtype != np.bool_
            or self.scored.dtype != np.bool_
            or self.distinct.shape != self.sampled.shape
            or self.scored.shape != self.sampled.shape
        ):
            raise ValueError("Invalid saved diagnostic masks")
        if (
            (self.scored & ~self.distinct).any()
            or len(self.sampled) != report["sampled_frames"]
            or int(self.scored.sum()) != report["frames"]
        ):
            raise ValueError("Saved frame counts/masks disagree")
        self.sample_positions = {int(frame): i for i, frame in enumerate(self.sampled)}
        csv_path = directory / "production-frames.csv"
        hypotheses = report["hypotheses"]
        with csv_path.open() as stream:
            reader = csv.DictReader(stream)
            expected_fields = [
                "frame",
                "view_mask",
                *[f"cost_{i}" for i in range(len(hypotheses))],
                *[f"support_{i}" for i in range(len(hypotheses))],
            ]
            if reader.fieldnames != expected_fields:
                raise ValueError("Unsupported saved frame score columns")
            rows = list(reader)
        if [int(row["frame"]) for row in rows] != self.sampled[self.scored].tolist():
            raise ValueError("CSV frame axis differs from the saved scored mask")
        self.rows: dict[int, dict[str, Any]] = {}
        for row in rows:
            frame = int(row["frame"])
            mask = int(row["view_mask"])
            if (
                mask
                != sum(int(visible[i, frame]) << i for i in range(len(case.camera_ids)))
                or visible[:, frame].sum() < 2
            ):
                raise ValueError("CSV cameras differ from the saved observations")
            costs = [float(row[f"cost_{i}"]) for i in range(len(hypotheses))]
            supports = [int(row[f"support_{i}"]) for i in range(len(hypotheses))]
            if (
                not np.isfinite(costs).all()
                or any(not 0 <= cost <= 1 for cost in costs)
                or any(support not in (0, 1) for support in supports)
            ):
                raise ValueError("Invalid saved per-frame cost/support")
            self.rows[frame] = {
                "costs": costs,
                "supports": supports,
                "view_mask": mask,
                "cameras": [
                    camera
                    for i, camera in enumerate(case.camera_ids)
                    if mask & (1 << i)
                ],
            }
        for i, hypothesis in enumerate(hypotheses):
            if len(hypothesis["view_half_turns"]) != len(case.camera_ids) or hypothesis[
                "frames"
            ] != len(rows):
                raise ValueError("Saved hypothesis axis mismatch")
            if not np.isclose(
                np.mean([row["costs"][i] for row in self.rows.values()]),
                hypothesis["cost"],
                atol=1e-12,
                rtol=0,
            ) or not np.isclose(
                np.mean([row["supports"][i] for row in self.rows.values()]),
                hypothesis["support"],
                atol=1e-12,
                rtol=0,
            ):
                raise ValueError("CSV scores disagree with the saved aggregate")
        self.sources = [
            {"path": str(path), "sha256": dual_sha256(path)}
            for path in (report_path, observations_path, csv_path)
        ]
        self.sources.extend(receipts)
        self.decision = {
            key: report[key]
            for key in (
                "decided",
                "reason",
                "hypotheses",
                "frames",
                "margin",
                "pair_frames",
                "thresholds",
                "view_half_turns",
            )
        }
        self.decision.update(
            camera_ids=case.camera_ids,
            reference_camera=case.calibration["reference_camera"],
            record_source="saved production diagnosis",
            schema_version=None,
        )

    def frame(self, frame: int) -> dict[str, Any]:
        if frame not in self.sample_positions:
            return {"state": "not_sampled", "scores": None}
        position = self.sample_positions[frame]
        if not self.distinct[position]:
            return {"state": "duplicate_removed", "scores": None}
        if not self.scored[position]:
            return {"state": "fewer_than_two_views", "scores": None}
        return {"state": "scored", "scores": self.rows[frame]}
