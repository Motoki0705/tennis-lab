"""Explicit dataset-version catalog on the existing bounded Court Review UI."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.synthetic_data_generation.alignment.contracts import AlignmentResult
from src.synthetic_data_generation.dataset.court.sample_store import (
    open_court_store,
    read_court_manifest,
)
from src.synthetic_data_generation.dataset.court.scale.core import digest, stable_id
from src.synthetic_data_generation.dataset.court.scale.publication import (
    contained,
    iter_plan_groups,
    load_dataset_version,
    read_json,
)
from src.synthetic_data_generation.dataset.court.scale.validation import arc_fraction
from src.utils.schema.court import (
    COURT_SKELETON,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)

from .records import rejection_summary, sample_summary, split_target_counts
from .service import ReviewService


class ScaleReviewService(ReviewService):
    def __init__(self, versions: list[Path]) -> None:
        if not versions:
            raise ValueError("Review requires explicit dataset-version paths.")
        super().__init__(versions[0].resolve(strict=True).parent)
        self.entries: dict[str, dict[str, Any]] = {}
        for path in versions:
            path = path.resolve(strict=True)
            version = load_dataset_version(path)
            for scene in version["scenes"]:
                key = stable_id(
                    "review_scene", version["dataset_version"], scene["scene_id"]
                )
                if key in self.entries:
                    raise ValueError("Duplicate scene/dataset-version catalog entry.")
                self.entries[key] = {
                    "path": path,
                    "root": path.parent,
                    "version": version,
                    "scene": scene,
                }

    def _entry(self, key: str) -> dict[str, Any]:
        if key not in self.entries:
            raise ValueError("Unknown dataset-version/scene selection key.")
        return self.entries[key]

    def revision(self, scene: str) -> str:
        entry = self._entry(scene)
        root, version, selected = entry["root"], entry["version"], entry["scene"]
        paths = [entry["path"], contained(root, version["request_path"])]
        paths += [
            contained(root, selected[k])
            for k in ("plan_path", "asset_manifest_path", "accepted_ids_path")
        ]
        for cache in selected["render_caches"]:
            completion = contained(root, cache["path"])
            paths += [completion, completion.parent / "qa.json"]
        plan = read_json(contained(root, selected["plan_path"]))
        paths += [contained(root, r["path"]) for r in plan["trajectory_group_refs"]]
        for store in selected["stores"]:
            store_root = contained(root, store["path"], directory=True)
            paths += [store_root / "dataset.json", store_root / "samples/index.npz"]
            paths += list((store_root / "samples/shards").glob("*.bin"))
        return str(
            digest(
                [
                    (str(p), p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)
                    for p in paths
                ]
            )
        )

    def scenes(self) -> list[dict[str, Any]]:
        return [
            {
                "id": key,
                "scene_id": e["scene"]["scene_id"],
                "method": e["version"]["method"],
                "dataset_version": e["version"]["dataset_version"],
                "label": e["scene"]["scene_id"],
                "method_label": f"{e['version']['method']} · seed {e['version']['generation_seed']} · {e['version']['dataset_version'][:10]}",
                "revision": self.revision(key),
            }
            for key, e in self.entries.items()
        ]

    def catalog(self) -> list[dict[str, Any]]:
        return [
            self._publication(key) | {"revision": self.revision(key)}
            for key in self.entries
        ]

    def _publication(self, key: str) -> dict[str, Any]:
        entry = self._entry(key)
        version, selected = entry["version"], entry["scene"]
        counts = selected["counts"]
        return {
            "id": key,
            "scene_id": selected["scene_id"],
            "method": version["method"],
            "dataset_version": version["dataset_version"],
            "schema": "canonical_court_dataset_v3",
            "status": "completed",
            "profile": "scale_train_v1",
            "seed": version["generation_seed"],
            "storage_schema": "court_image_store_v1",
            "storage_format": "jpeg_sparse_records_v1",
            "renderer_rasters_retained": False,
            "visibility_authority": "generation_alpha_depth_gate",
            "source_video": None,
            "sample_count": counts["qa_accepted"],
            "group_count": selected["trajectory_count"],
            "split_frame_counts": {
                "train": counts["qa_accepted"],
                "validation": 0,
                "test": 0,
            },
            "rejected_count": counts["geometry_rejected"] + counts["qa_rejected"],
            "manifest_path": str(entry["path"]),
            "request_sha256": version["request_sha256"],
            "recipe_sha256": version["recipe_sha256"],
            "density_level": selected["density_level"],
            "counts": counts,
            "empty_trajectory_count": selected["empty_trajectory_count"],
        }

    def _load(self, scene: str, revision: str) -> dict[str, Any]:
        entry = self._entry(scene)
        version = load_dataset_version(entry["path"])
        if version != entry["version"]:
            raise RuntimeError(
                "Pinned dataset version changed. Restart Review with the explicit version."
            )
        root, selected = entry["root"], entry["scene"]
        plan = read_json(contained(root, selected["plan_path"]))
        asset = read_json(contained(root, selected["asset_manifest_path"]))
        layout = AlignmentResult.from_dict(asset["alignment"]).layout
        ref = layout.courts[0].court_from_scene
        samples, source_by_sample = {}, {}
        for store_entry in selected["stores"]:
            store_root = contained(root, store_entry["path"], directory=True)
            store = open_court_store(store_root)
            for sample in read_court_manifest(store_root, store=store)["samples"]:
                samples[sample["sample_id"]] = sample
                source_by_sample[sample["sample_id"]] = (store_root, store)
        rejected, ledger = {}, []
        for cache in selected["render_caches"]:
            completion = read_json(contained(root, cache["path"]))
            records = read_json(contained(root, completion["ledger_path"]))["records"]
            ledger.extend(records)
            rejected.update(
                {
                    r["sample_id"]: r["record"]
                    for r in records
                    if r["state"] != "accepted"
                }
            )
        groups = []
        for group in iter_plan_groups(root, plan):
            gid = group["trajectory_id"]
            accepted = sorted(
                (s for s in samples.values() if s["trajectory_id"] == gid),
                key=arc_fraction,
            )
            points, forwards = [], []
            for sample in accepted:
                pose = np.asarray(sample["camera"]["camera_to_scene"]).reshape(4, 4)
                points.append(ref.apply(pose[:3, 3][None])[0].tolist())
                forwards.append(
                    ref.apply((pose[:3, 3] + pose[:3, 2] * 2)[None])[0].tolist()
                )
            curve = group["curve"]
            shape = (
                curve["parameters"].get("shape", curve["kind"])
                if curve
                else "rejected_curve"
            )
            planned_points = group.get("planned_points_scene_m", [])
            rejects = [s for s in rejected.values() if s["trajectory_id"] == gid]
            groups.append(
                {
                    "id": gid,
                    "label": f"{group['ordinal'] + 1:02d} · {shape}",
                    "trajectory": {
                        "schema": "court_scale_review_trajectory_v1",
                        "trajectory_id": gid,
                        "shape": shape,
                    },
                    "split": "train",
                    "points": points,
                    "forwards": forwards,
                    "samples": [
                        {
                            **sample_summary(s),
                            "arc_fraction": s["scale"]["arc_fraction"],
                            "qa": s["qa"],
                        }
                        for s in accepted
                    ],
                    "planned_points": ref.apply(np.asarray(planned_points)).tolist()
                    if planned_points
                    else [],
                    "rejected_points": [
                        ref.apply(
                            np.asarray(s["camera"]["camera_to_scene"]).reshape(4, 4)[
                                :3, 3
                            ][None]
                        )[0].tolist()
                        for s in rejects
                    ],
                    "closed": curve["kind"] == "orbit" if curve else False,
                    "diagnostics": {
                        k: v
                        for k, v in group.items()
                        if k not in {"samples", "arc_map", "planned_points_scene_m"}
                    },
                    "candidate_count": len(group["samples"]),
                    "rejected_count": len(rejects),
                }
            )
        geometry = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()
        courts = [
            {
                "id": c.court_instance_id,
                "points": ref.apply(c.scene_from_court.apply(geometry)).tolist(),
            }
            for c in layout.courts
        ]
        counts = selected["counts"]
        coverage: Counter[str] = Counter()
        visible: Counter[str] = Counter()
        for s in samples.values():
            for court in s["projection"]["courts"]:
                coverage[court["coverage_mode"]] += 1
                for cls in court["classes"]:
                    visible[cls["class_name"]] += sum(
                        p["renderer_visible"] for p in cls["points"]
                    )
        metrics = {
            "accepted_frame_count": counts["qa_accepted"],
            "rejected_frame_count": counts["geometry_rejected"] + counts["qa_rejected"],
            "accepted_fraction": counts["qa_accepted"] / counts["candidate_unique"],
            "split_frame_counts": {
                "train": counts["qa_accepted"],
                "validation": 0,
                "test": 0,
            },
            "coverage_counts": dict(coverage),
            "renderer_visible_points_by_class": dict(visible),
        }
        request = read_json(contained(root, version["request_path"]))
        summary = {
            "id": scene,
            "scene_id": selected["scene_id"],
            "revision": revision,
            "groups": groups,
            "courts": courts,
            "edges": [list(e) for e in COURT_SKELETON],
            "metrics": metrics,
            "shapes": dict(Counter(g["trajectory"]["shape"] for g in groups)),
            "schema": "canonical_court_dataset_v3",
            "publication": self._publication(scene),
            "rejections": rejection_summary(
                {"metrics": metrics, "rejected_samples": list(rejected.values())}
            ),
            "split_targets": split_target_counts(
                list(samples.values()), [c["id"] for c in courts]
            ),
            "scale": {
                "method": version["method"],
                "recipe": plan["recipe"],
                "quality": request["quality"],
                "seed": version["generation_seed"],
                "density_level": plan["density_level"],
                "counts": counts,
                "empty_trajectory_count": selected["empty_trajectory_count"],
                "sensitivity": {
                    "target_kp_ge_4": sum(
                        bool(r.get("qa", {}).get("sensitivity_target_kp_ge_4"))
                        for r in ledger
                        if r["qa"] is not None
                    ),
                    "roi_holes_le_0_01": sum(
                        r["qa"]["sensitivity_roi_holes_le_0_01"] is True
                        for r in ledger
                        if r["qa"] is not None
                    ),
                },
            },
        }
        if self.revision(scene) != revision:
            raise RuntimeError("Dataset changed while loading. Reload this scene.")
        return {
            "summary": summary,
            "samples": samples,
            "rejected_samples": rejected,
            "schema": "canonical_court_dataset_v3",
            "source_by_sample": source_by_sample,
        }

    def sample_source(
        self, scene: str, data: dict[str, Any], sample: str
    ) -> tuple[Path, Any]:
        root, store = data["source_by_sample"][sample]
        return Path(root), store

    def sample_detail(self, scene: str, revision: str, sample: str) -> dict[str, Any]:
        result = super().sample_detail(scene, revision, sample)
        record = self.load(scene, revision)["samples"][sample]
        return {**result, "qa": record["qa"], "scale": record["scale"]}
