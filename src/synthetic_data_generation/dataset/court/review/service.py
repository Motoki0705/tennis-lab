"""Manifest-backed scene summaries and bounded on-demand overlay rendering."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from threading import RLock, Semaphore
from typing import Any

import cv2
import numpy as np

from src.synthetic_data_generation.alignment.validation import load_alignment_result
from src.synthetic_data_generation.dataset.court.sample_store import (
    open_court_store,
    read_court_labels,
    read_court_manifest,
    read_court_rgb,
)
from src.synthetic_data_generation.dataset.court.schema import (
    court_schema_from_dataset_schema,
)
from src.synthetic_data_generation.visualization.overlays import render_court_overlay
from src.synthetic_data_generation.visualization.sources import CourtSourceFrame
from src.utils.schema.court import (
    COURT_SKELETON,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)

from .records import (
    rejection_summary,
    sample_summary,
    split_target_counts,
    target_projection,
    visibility_summary,
)


def contained_file(root: Path, relative: str) -> Path:
    """Only serve files belonging to this dataset, including after symlink resolution."""
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Dataset file escapes its owner.")
    return path


def read_object(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return result


class ReviewService:
    def __init__(self, scenes_root: Path) -> None:
        self.root = scenes_root.resolve(strict=True)
        self.lock = RLock()
        self.render_slots = Semaphore(2)
        self._cached_load = lru_cache(maxsize=2)(self._load)
        self._cached_overlay = lru_cache(maxsize=128)(self._overlay)

    def scene_root(self, scene: str) -> Path:
        if Path(scene).name != scene or scene in {".", ".."}:
            raise ValueError("Invalid scene ID.")
        path = (self.root / scene).resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_dir():
            raise ValueError("Scene escapes configured root.")
        return path

    def revision(self, scene: str) -> str:
        root = self.scene_root(scene)
        identities = []
        files = ["datasets/court/dataset.json", "alignment/alignment.json"]
        files.extend(
            name
            for name in ("run.json", "resolved-config.yaml")
            if (root / name).exists()
        )
        for relative in files:
            p = contained_file(root, relative)
            st = p.stat()
            identities.append((relative, st.st_ino, st.st_size, st.st_mtime_ns))
        index = root / "datasets/court/samples/index.npz"
        if index.is_file():
            st = index.stat()
            identities.append(
                ("samples/index.npz", st.st_ino, st.st_size, st.st_mtime_ns)
            )
        return hashlib.sha256(repr(identities).encode()).hexdigest()[:20]

    def scenes(self) -> list[dict[str, str]]:
        result = []
        for p in sorted(self.root.iterdir()):
            if not (p / "datasets/court/dataset.json").is_file():
                continue
            try:
                result.append({"id": p.name, "revision": self.revision(p.name)})
            except (ValueError, OSError) as error:
                result.append({"id": p.name, "error": str(error)})
        return result

    def publication(self, scene: str, dataset: dict[str, Any]) -> dict[str, Any]:
        root = self.scene_root(scene)
        descriptor = read_object(contained_file(root, "datasets/court/dataset.json"))
        run = (
            read_object(contained_file(root, "run.json"))
            if (root / "run.json").exists()
            else None
        )
        if run is not None and run.get("scene_id") != scene:
            raise ValueError("Source run and dataset scene identities disagree.")
        stages = run.get("stages", {}) if run is not None else {}
        generation = stages.get("court_dataset", {})
        storage = descriptor.get("storage")
        count = storage["count"] if storage is not None else len(dataset["samples"])
        if count != dataset["metrics"]["accepted_frame_count"]:
            raise ValueError("Published sample count and acceptance metrics disagree.")
        rejected = rejection_summary(dataset)
        config = root / "resolved-config.yaml"
        return {
            "id": scene,
            "schema": dataset["schema"],
            "status": dataset["status"],
            "profile": dataset.get("profile"),
            "seed": dataset.get("seed"),
            "storage_schema": descriptor["schema"]
            if storage is not None
            else "legacy_float32",
            "storage_format": storage["format"]
            if storage is not None
            else "float32 / sparse JSON",
            "renderer_rasters_retained": descriptor.get("renderer_rasters_retained"),
            "visibility_authority": descriptor.get("visibility_authority"),
            "source_video": run.get("source_video") if run is not None else None,
            "targets": run.get("targets") if run is not None else None,
            "generation_state": generation.get("status"),
            "generated_at": generation.get("updated_at"),
            "captured_camera_count": stages.get("reconstruction", {})
            .get("summary", {})
            .get("camera_count")
            if stages.get("reconstruction", {}).get("summary") is not None
            else None,
            "sample_count": count,
            "group_count": len(dataset["trajectory_groups"]),
            "split_frame_counts": dataset["metrics"]["split_frame_counts"],
            "rejected_count": rejected["count"],
            "rejected_record_count": rejected["record_count"],
            "manifest_path": str(root / "datasets/court/dataset.json"),
            "manifest_sha256": hashlib.sha256(
                (root / "datasets/court/dataset.json").read_bytes()
            ).hexdigest(),
            "index_sha256": storage["index_sha256"] if storage is not None else None,
            "config_sha256": hashlib.sha256(
                contained_file(root, "resolved-config.yaml").read_bytes()
            ).hexdigest()
            if config.exists()
            else None,
        }

    def catalog(self) -> list[dict[str, Any]]:
        """Inspect publication metadata only; do not decode every image or sample."""
        result = []
        for scene in self.scenes():
            if "error" in scene:
                result.append(scene)
                continue
            try:
                root = self.scene_root(scene["id"]) / "datasets/court"
                descriptor = read_object(root / "dataset.json")
                dataset = (
                    open_court_store(root).metadata
                    if "storage" in descriptor
                    else read_court_manifest(root)
                )
                if (
                    dataset["scene_id"] != scene["id"]
                    or dataset["status"] != "completed"
                ):
                    raise ValueError(
                        "Only completed matching publications are supported."
                    )
                entry = self.publication(scene["id"], dataset)
                if self.revision(scene["id"]) != scene["revision"]:
                    raise RuntimeError("Publication changed during catalog inspection.")
                result.append({**entry, "revision": scene["revision"]})
            except (ValueError, RuntimeError, OSError) as error:
                result.append({"id": scene["id"], "error": str(error)})
        return result

    def load(self, scene: str, revision: str) -> dict[str, Any]:
        if revision != self.revision(scene):
            raise RuntimeError("Dataset changed. Reload this scene.")
        with self.lock:
            return self._cached_load(scene, revision)

    def _load(self, scene: str, revision: str) -> dict[str, Any]:
        root = self.scene_root(scene)
        dataset = read_court_manifest(root / "datasets/court")
        court_schema_from_dataset_schema(dataset["schema"])
        if dataset["status"] != "completed" or dataset["scene_id"] != scene:
            raise ValueError(
                "Only completed datasets with matching scene identity can be reviewed."
            )
        alignment = load_alignment_result(
            contained_file(root, "alignment/alignment.json")
        )
        ref = alignment.layout.courts[0].court_from_scene
        groups = []
        sample_index = {s["sample_id"]: s for s in dataset["samples"]}
        if len(sample_index) != len(dataset["samples"]):
            raise ValueError("Duplicate sample IDs.")
        for group in dataset["trajectory_groups"]:
            trajectory = group["trajectory"]
            gid = trajectory["trajectory_group_id"]
            samples = sorted(
                (s for s in dataset["samples"] if s["trajectory_group_id"] == gid),
                key=lambda s: (s["trajectory_frame_index"], s["view_id"]),
            )
            if not samples:
                raise ValueError(f"Trajectory has no published samples: {gid}")
            if any(s["split"] != group["split"] for s in samples):
                raise ValueError(f"Sample split disagrees with trajectory group: {gid}")
            points = []
            forwards = []
            for sample in samples:
                pose = np.asarray(
                    sample["camera"]["camera_to_scene"], dtype=float
                ).reshape(4, 4)
                points.append(ref.apply(pose[:3, 3][None])[0].tolist())
                forwards.append(
                    ref.apply((pose[:3, 3] + pose[:3, 2] * 2)[None])[0].tolist()
                )
            groups.append(
                {
                    "id": gid,
                    "trajectory": trajectory,
                    "split": group["split"],
                    "points": points,
                    "forwards": forwards,
                    "samples": [sample_summary(s) for s in samples],
                }
            )
        geometry = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()
        courts = [
            {
                "id": c.court_instance_id,
                "points": ref.apply(c.scene_from_court.apply(geometry)).tolist(),
            }
            for c in alignment.layout.courts
        ]
        summary = {
            "id": scene,
            "revision": revision,
            "groups": groups,
            "courts": courts,
            "edges": [list(e) for e in COURT_SKELETON],
            "metrics": dataset["metrics"],
            "shapes": dict(Counter(g["trajectory"]["shape"] for g in groups)),
            "schema": dataset["schema"],
            "publication": self.publication(scene, dataset),
            "rejections": rejection_summary(dataset),
            "split_targets": split_target_counts(
                dataset["samples"], [c["id"] for c in courts]
            ),
        }
        if revision != self.revision(scene):
            raise RuntimeError("Dataset changed during loading. Reload this scene.")
        return {
            "summary": summary,
            "samples": sample_index,
            "rejected_samples": {
                s["sample_id"]: s for s in dataset.get("rejected_samples", [])
            },
            "schema": dataset["schema"],
            "image_store": open_court_store(root / "datasets/court")
            if "storage" in dataset
            else None,
        }

    def sample_detail(self, scene: str, revision: str, sample: str) -> dict[str, Any]:
        data = self.load(scene, revision)
        entry = data["samples"][sample]
        return {
            **sample_summary(entry),
            "split": entry["split"],
            "trajectory_group": entry["trajectory_group_id"],
            "visibility": visibility_summary(entry["projection"]),
            "intrinsics": entry["camera"]["intrinsics"],
            "camera_to_scene": entry["camera"]["camera_to_scene"],
            "target_binding": entry.get("target_court"),
        }

    def rejection_detail(
        self, scene: str, revision: str, sample: str
    ) -> dict[str, Any]:
        entry = self.load(scene, revision)["rejected_samples"][sample]
        return {
            **sample_summary(entry),
            "split": entry["split"],
            "trajectory_group": entry["trajectory_group_id"],
            "reasons": entry["reasons"],
            "visibility": visibility_summary(entry.get("projection")),
            "intrinsics": entry["camera"]["intrinsics"],
            "camera_to_scene": entry["camera"]["camera_to_scene"],
            "target_binding": entry.get("target_court"),
            "image_state": "not_retained",
        }

    def overlay(
        self,
        scene: str,
        revision: str,
        sample: str,
        width: int,
        mode: str = "overlay",
        label_scope: str = "all",
    ) -> bytes:
        if mode not in {"overlay", "raw"}:
            raise ValueError("Unknown image display mode.")
        if label_scope not in {"target", "all"}:
            raise ValueError("Unknown label court scope.")
        data = self.load(scene, revision)
        if sample not in data["samples"]:
            raise KeyError(sample)
        with self.render_slots:
            return self._cached_overlay(
                scene, revision, sample, width, mode, label_scope
            )

    def _overlay(
        self,
        scene: str,
        revision: str,
        sample: str,
        width: int,
        mode: str = "overlay",
        label_scope: str = "all",
    ) -> bytes:
        data = self.load(scene, revision)
        entry = data["samples"][sample]
        root = self.scene_root(scene) / "datasets/court"
        rgb = (
            read_court_rgb(root, entry, store=data["image_store"]).astype(np.float32)
            / 255.0
        )
        if (
            rgb.dtype != np.float32
            or rgb.shape != (entry["height"], entry["width"], 3)
            or not np.isfinite(rgb).all()
            or np.any((rgb < 0) | (rgb > 1))
        ):
            raise ValueError("Invalid RGB array.")
        label = read_court_labels(root, entry, dataset_schema=data["schema"])
        if (
            label["sample_id"] != sample
            or label["projection"] != entry["projection"]
            or label["camera"] != entry["camera"]
        ):
            raise ValueError("Sample image/label binding mismatch.")
        frame = CourtSourceFrame(
            rgb=rgb,
            sample_id=sample,
            view_id=entry["view_id"],
            trajectory_frame_index=entry["trajectory_frame_index"],
            projection=target_projection(entry)
            if mode == "overlay" and label_scope == "target"
            else label["projection"],
            schema_version=court_schema_from_dataset_schema(data["schema"]).version,
        )
        image = (
            render_court_overlay(
                frame, trajectory_id=entry["trajectory_id"], show_metadata=False
            )
            if mode == "overlay"
            else np.round(rgb * 255).astype(np.uint8)
        )
        resized = (
            cv2.resize(
                image,
                (width, round(image.shape[0] * width / image.shape[1])),
                interpolation=cv2.INTER_AREA,
            )
            if width and image.shape[1] > width
            else image
        )
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        if not ok:
            raise ValueError("Image encoding failed.")
        if revision != self.revision(scene):
            raise RuntimeError("Dataset changed while rendering. Reload this scene.")
        return encoded.tobytes()
