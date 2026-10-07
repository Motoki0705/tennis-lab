"""Read explicitly published artifacts, including labelled historical schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.storage.codec import unpack_value
from src.tennis_scene.pipeline.storage.scene_index import (
    assert_current_component_lineage,
    read_component_descriptor,
)
from src.utils.checksum import dual_sha256


class StoreCase:
    """A snapshot of a store's source timeline, points, calibration and decision."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.index = self.root / "scene.json"
        self.document = json.loads(self.index.read_text())
        if self.document["schema"] != "tennis_scene_index_v1":
            raise ValueError("Expected tennis_scene_index_v1")
        self.index_sha256 = dual_sha256(self.index)
        self.source = self.document["source"]
        self.videos = self.source["videos"]
        self.camera_ids = [video["camera_id"] for video in self.videos]
        if len(set(self.camera_ids)) != len(self.camera_ids) or len(self.videos) < 2:
            raise ValueError("Review needs unique multi-view cameras")
        self.frames = int(self.videos[0]["num_frames"])
        self.fps = float(self.videos[0]["fps"])
        if self.frames < 1 or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Invalid source timeline")
        for video in self.videos:
            if video["num_frames"] != self.frames or video["fps"] != self.fps:
                raise ValueError("Review requires a saved synchronized source timeline")
        self.references = self.document["artifacts"]
        self.descriptors: dict[str, dict[str, Any]] = {}
        self.points: dict[str, dict[str, Any] | None] = {}
        for video in self.videos:
            camera = video["camera_id"]
            point_node, detector_node = (
                f"ball_points/{camera}",
                f"ball_detection/{camera}",
            )
            if point_node in self.references:
                self.points[camera] = self.read_points(point_node, video)
            elif detector_node in self.references:
                # Explicit historical reader; the UI names the schema and origin.
                self.points[camera] = self.read_points(detector_node, video)
            else:
                self.points[camera] = None
        self.calibration: dict[str, Any] | None = None
        if "court_calibration" in self.references:
            payload = self.read(
                "court_calibration",
                "local_court_calibration",
                (1,),
                ("calibration", "reference_camera"),
            )
            self.calibration = {
                "reference_camera": payload["reference_camera"],
                "excluded": payload["calibration"]["excluded"],
                "views": [],
            }
            for view in payload["calibration"]["views"]:
                camera = view["camera"]
                rotation = np.asarray(camera["rotation"], np.float64)
                center = -rotation.T @ np.asarray(camera["translation"], np.float64)
                direction = rotation.T @ np.array([0.0, 0.0, 1.0])
                self.calibration["views"].append(
                    {
                        "camera_id": camera["camera_id"],
                        "center_m": center.tolist(),
                        "direction": direction.tolist(),
                        "rmse_px": view["rmse_px"],
                        "frame": view["frame_index"],
                    }
                )
        self.decision: dict[str, Any] | None = None
        if "court_side" in self.references:
            version = self.references["court_side"]["version"]
            payload = self.read("court_side", "court_side", (2, 3))
            self.decision = {
                **payload,
                "decided": True,
                "reason": None,
                "record_source": "court_side artifact",
                "schema_version": version,
            }
            if version == 2:
                self.decision.update(hypotheses=None, frames=None, margin=None)
            for port, dependency in self.descriptors["court_side"][
                "dependencies"
            ].items():
                if port.startswith("ball_"):
                    camera = port.removeprefix("ball_")
                    points = self.points[camera]
                    if points is None or self.references[points["node"]] != dependency:
                        raise ValueError(
                            "Displayed points differ from side artifact lineage"
                        )
        self.diagnostics: Any = None

    def read(
        self,
        node: str,
        schema: str,
        versions: tuple[int, ...],
        fields: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        reference = self.references[node]
        if reference["schema"] != schema or reference["version"] not in versions:
            raise ValueError(
                f"Unsupported {node}: {reference['schema']} v{reference['version']}"
            )
        assert_current_component_lineage(self.document, self.root, {node: reference})
        descriptor = read_component_descriptor(
            self.root,
            reference,
            node=node,
            source_sha256=self.document["source_sha256"],
        )
        self.descriptors[node] = descriptor
        packed = descriptor["payload"]
        if fields is not None:
            packed = {field: packed[field] for field in fields}
        payload: dict[str, Any] = unpack_value(
            packed, (self.root / reference["path"]).parent, descriptor["arrays"]
        )
        return payload

    def read_points(self, node: str, video: dict[str, Any]) -> dict[str, Any]:
        reference = self.references[node]
        schema = reference["schema"]
        if schema == "ball_points":
            payload = self.read(
                node,
                schema,
                (2,),
                (
                    "camera_id",
                    "source_size_wh",
                    "frame_indices",
                    "uv_px",
                    "presence_probability",
                ),
            )
            if payload["source_size_wh"] != [video["width"], video["height"]]:
                raise ValueError("Point source size mismatch")
            observed: NDArray[np.bool_] = np.ones(self.frames, dtype=bool)
            kinds: NDArray[np.uint8] = np.ones(self.frames, dtype=np.uint8)
            confidence = payload["presence_probability"]
            semantics = (
                "refiner maximum-weight mean; presence diagnostic only, no filtering"
            )
        elif schema == "ball_detections":
            payload = self.read(
                node,
                schema,
                (1, 2),
                (
                    "camera_id",
                    "frame_indices",
                    "uv_px",
                    "observed",
                    "point_kind",
                    "confidence",
                    "score_semantics",
                ),
            )
            observed, kinds, confidence = (
                payload["observed"],
                payload["point_kind"],
                payload["confidence"],
            )
            semantics = payload["score_semantics"]
        else:
            raise ValueError(f"Unsupported ball observation schema: {schema}")
        uv = np.asarray(payload["uv_px"])
        if payload["camera_id"] != video["camera_id"] or not np.array_equal(
            payload["frame_indices"], np.arange(self.frames)
        ):
            raise ValueError("Ball camera/frame axis mismatch")
        if uv.shape != (self.frames, 2) or not np.isfinite(uv).all():
            raise ValueError("Invalid ball coordinates")
        if (
            observed.shape != (self.frames,)
            or observed.dtype != np.bool_
            or kinds.shape != observed.shape
            or confidence.shape != observed.shape
        ):
            raise ValueError("Invalid ball observation masks")
        if not np.isfinite(confidence).all() or not np.isin(kinds, [0, 1, 2, 3]).all():
            raise ValueError("Invalid point kinds/confidence")
        return {
            "uv": uv,
            "observed": observed,
            "kinds": kinds,
            "confidence": confidence,
            "schema": f"{schema} v{reference['version']}",
            "semantics": semantics,
            "provenance": self.descriptors[node]["provenance"],
            "node": node,
        }

    def assert_unchanged(self) -> None:
        if dual_sha256(self.index) != self.index_sha256:
            raise ValueError(
                "scene.json changed; restart review to load the new snapshot"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "clip": self.source["clip_id"],
            "store": str(self.root),
            "source_sha256": self.document["source_sha256"],
            "index_sha256": self.index_sha256,
            "frames": self.frames,
            "fps": self.fps,
            "cameras": self.camera_ids,
            "calibration": self.calibration,
            "decision": self.decision,
            "frame_scores_saved": self.diagnostics is not None,
            "point_sources": {
                camera: None
                if points is None
                else {
                    key: points[key]
                    for key in ("schema", "semantics", "provenance", "node")
                }
                for camera, points in self.points.items()
            },
            "diagnostic_sources": None
            if self.diagnostics is None
            else self.diagnostics.sources,
        }
