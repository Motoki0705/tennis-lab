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
        for relative in ("datasets/court/dataset.json", "alignment/alignment.json"):
            p = contained_file(root, relative)
            st = p.stat()
            identities.append((relative, st.st_ino, st.st_size, st.st_mtime_ns))
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

    def load(self, scene: str, revision: str) -> dict[str, Any]:
        if revision != self.revision(scene):
            raise RuntimeError("Dataset changed. Reload this scene.")
        with self.lock:
            return self._cached_load(scene, revision)

    def _load(self, scene: str, revision: str) -> dict[str, Any]:
        root = self.scene_root(scene)
        dataset = read_object(contained_file(root, "datasets/court/dataset.json"))
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
                    "samples": [
                        {
                            "id": s["sample_id"],
                            "frame": s["trajectory_frame_index"],
                            "view": s["view_id"],
                        }
                        for s in samples
                    ],
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
        }
        if revision != self.revision(scene):
            raise RuntimeError("Dataset changed during loading. Reload this scene.")
        return {
            "summary": summary,
            "samples": sample_index,
            "schema": dataset["schema"],
        }

    def overlay(self, scene: str, revision: str, sample: str, width: int) -> bytes:
        data = self.load(scene, revision)
        if sample not in data["samples"]:
            raise KeyError(sample)
        with self.render_slots:
            return self._cached_overlay(scene, revision, sample, width)

    def _overlay(self, scene: str, revision: str, sample: str, width: int) -> bytes:
        data = self.load(scene, revision)
        entry = data["samples"][sample]
        root = self.scene_root(scene) / "datasets/court"
        rgb = np.load(contained_file(root, entry["rgb"]), allow_pickle=False)
        if (
            rgb.dtype != np.float32
            or rgb.shape != (entry["height"], entry["width"], 3)
            or not np.isfinite(rgb).all()
            or np.any((rgb < 0) | (rgb > 1))
        ):
            raise ValueError("Invalid RGB array.")
        label = read_object(contained_file(root, entry["labels"]))
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
            projection=label["projection"],
            schema_version=court_schema_from_dataset_schema(data["schema"]).version,
        )
        image = render_court_overlay(
            frame, trajectory_id=entry["trajectory_id"], show_metadata=False
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
