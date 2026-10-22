"""Frozen requests, immutable shared JPEG stores, QA ledger and dataset versions."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.synthetic_data_generation.alignment.contracts import AlignmentResult
from src.synthetic_data_generation.dataset.court.sample_store import (
    finish_court_store,
    open_court_store,
)
from src.synthetic_data_generation.rendering.nht import NHTRenderArrays
from src.synthetic_data_generation.scene_contract import MultiCourtLayout
from src.utils.data.image_record_store import (
    ImageRecordWriter,
    encode_rgb_jpeg,
    file_sha256,
)

from .core import GeometryRecipe, QualityPolicy, canonical_bytes, digest, stable_id
from .planning import PLAN_SCHEMA, candidate_plan_header, iter_trajectory_plans
from .quality import (
    evaluate_render,
    quality_rejection_reasons,
    validate_quality_binding,
)
from .recipes import ProductionRecipe
from .validation import validate_plan_geometry, validate_record_geometry

REQUEST_SCHEMA = "court_scale_condition_request_v1"
VERSION_SCHEMA = "court_scale_dataset_version_v1"
PRODUCTION_REQUEST_SCHEMA = "court_scale_condition_request_v2"
PRODUCTION_VERSION_SCHEMA = "court_scale_dataset_version_v2"
STORE_PROFILE = "court_scale_train_store_v1"
Render = Callable[
    [dict[str, Any], list[dict[str, Any]]], Iterable[tuple[str, NHTRenderArrays]]
]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def contained(root: Path, relative: str, *, directory: bool = False) -> Path:
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("Dataset-version reference escapes its explicit root.")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve(strict=True)) or not (
        path.is_dir() if directory else path.is_file()
    ):
        raise ValueError("Dataset-version reference escapes its explicit root.")
    return path


def write_content(root: Path, category: str, value: Any) -> tuple[str, str]:
    payload = canonical_bytes(value)
    sha = digest(value)
    path = root / category / f"{sha}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Content-addressed file has conflicting bytes.")
    else:
        with path.open("xb") as stream:
            stream.write(payload)
    return path.relative_to(root).as_posix(), sha


def persist_candidate_plan(root: Path, **kwargs: Any) -> dict[str, Any]:
    """Stream one curve/table at a time; retain only content references."""
    references = []
    ids = set()
    for group in iter_trajectory_plans(**kwargs):
        for sample in group["samples"]:
            if sample["sample_id"] in ids:
                raise ValueError("Stable point ID collision.")
            ids.add(sample["sample_id"])
        path, sha = write_content(root, "curves", group)
        references.append(
            {"trajectory_id": group["trajectory_id"], "path": path, "sha256": sha}
        )
    return {
        **candidate_plan_header(**kwargs),
        "trajectory_storage": "curve_files_v1",
        "trajectory_group_refs": references,
    }


def iter_plan_groups(root: Path, plan: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if plan["trajectory_storage"] != "curve_files_v1":
        raise ValueError(
            "Published candidate plans require explicit curve-file storage."
        )
    references = plan["trajectory_group_refs"]
    if len(references) != plan["trajectory_count"] or len(
        {r["trajectory_id"] for r in references}
    ) != len(references):
        raise ValueError("Candidate trajectory inventory/count mismatch.")
    for reference in references:
        path = contained(root, reference["path"])
        if file_sha256(path) != reference["sha256"]:
            raise ValueError("Frozen trajectory plan hash mismatch.")
        group = read_json(path)
        if group["trajectory_id"] != reference["trajectory_id"]:
            raise ValueError("Frozen trajectory identity mismatch.")
        yield group


def freeze_request(
    root: Path,
    *,
    plans: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    quality: QualityPolicy,
    renderer: dict[str, Any],
    production_recipe: ProductionRecipe | None = None,
) -> Path:
    """Write every condition input before any render. The caller pins asset hashes."""
    root = root.resolve()
    if not plans or len(plans) != len(assets):
        raise ValueError("Plans and assets must be nonempty and aligned.")
    root.mkdir(parents=True, exist_ok=True)
    scenes = []
    shared = [(p["method"], p["recipe_sha256"], p["generation_seed"]) for p in plans]
    if len(set(shared)) != 1 or len({p["scene_id"] for p in plans}) != len(plans):
        raise ValueError(
            "One request requires one method/recipe/seed and unique scene IDs."
        )
    if production_recipe is not None:
        if quality != production_recipe.quality:
            raise ValueError("Quality policy disagrees with the production recipe.")
        for plan in plans:
            production_recipe.validate_plan(plan)
    for plan, asset in sorted(
        zip(plans, assets, strict=True), key=lambda pa: pa[0]["scene_id"]
    ):
        if (
            plan["schema"] != PLAN_SCHEMA
            or plan["split"] != "train"
            or asset["scene_id"] != plan["scene_id"]
        ):
            raise ValueError("Invalid train-only plan/asset identity.")
        if plan["trajectory_storage"] == "inline_v1":
            references = []
            for group in plan["trajectory_groups"]:
                path, sha = write_content(root, "curves", group)
                references.append(
                    {
                        "trajectory_id": group["trajectory_id"],
                        "path": path,
                        "sha256": sha,
                    }
                )
            plan = {
                **{k: v for k, v in plan.items() if k != "trajectory_groups"},
                "trajectory_storage": "curve_files_v1",
                "trajectory_group_refs": references,
            }
        elif plan["trajectory_storage"] != "curve_files_v1":
            raise ValueError("Unknown candidate trajectory storage schema.")
        plan_path, plan_hash = write_content(root, "plans", plan)
        asset_path, asset_hash = write_content(root, "assets", asset)
        scenes.append(
            {
                "scene_id": plan["scene_id"],
                "plan_path": plan_path,
                "plan_sha256": plan_hash,
                "asset_manifest_path": asset_path,
                "asset_manifest_sha256": asset_hash,
                "trajectory_count": plan["trajectory_count"],
                "density_level": plan["density_level"],
            }
        )
    method, recipe, seed = shared[0]
    request = {
        "schema": REQUEST_SCHEMA,
        "method": method,
        "recipe_sha256": recipe,
        "generation_seed": seed,
        "split": "train",
        "quality": quality.to_dict(),
        "renderer": renderer,
        "scenes": scenes,
    }
    if production_recipe is not None:
        request.update(
            schema=PRODUCTION_REQUEST_SCHEMA,
            production_recipe=production_recipe.to_dict(),
            recipe_sha256=digest(production_recipe.to_dict()),
            geometry_recipe_sha256=recipe,
            **_quality_authority(quality),
        )
    path, _ = write_content(root, "requests", request)
    return root / path


def _quality_authority(quality: QualityPolicy) -> dict[str, Any]:
    if quality.schema == "court_scale_quality_policy_v1":
        return {}
    return {
        "quality_policy_sha256": digest(quality.to_dict()),
        "quality_rule_version": quality.rule_version,
    }


def _request_authority(request: dict[str, Any]) -> dict[str, Any]:
    """Dispatch explicitly; old comparison requests never become production inputs."""
    keys = {"method", "recipe_sha256", "generation_seed", "split"}
    if request["schema"] == PRODUCTION_REQUEST_SCHEMA:
        recipe = ProductionRecipe.from_dict(request["production_recipe"])
        if (
            request["method"] != recipe.method.value
            or request["recipe_sha256"] != digest(recipe.to_dict())
            or request["geometry_recipe_sha256"] != digest(recipe.geometry.to_dict())
            or request["quality"] != recipe.quality.to_dict()
            or any(
                request[k] != v for k, v in _quality_authority(recipe.quality).items()
            )
        ):
            raise ValueError("Request disagrees with its complete production recipe.")
        keys.update(
            {
                "production_recipe",
                "geometry_recipe_sha256",
                "quality_policy_sha256",
                "quality_rule_version",
            }
        )
    elif request["schema"] != REQUEST_SCHEMA:
        raise ValueError("Unknown train-only condition request schema.")
    if request["split"] != "train":
        raise ValueError("Condition requests must be train-only.")
    return {key: request[key] for key in keys}


def validate_counts(counts: dict[str, Any], *, complete: bool = True) -> None:
    names = {
        "logical_records",
        "candidate_unique",
        "geometry_pass",
        "geometry_rejected",
        "rendered_unique",
        "render_cache_reused",
        "qa_accepted",
        "qa_rejected",
        "pending",
        "failed",
    }
    if set(counts) != names or any(
        type(v) is not int or v < 0 for v in counts.values()
    ):
        raise ValueError("Invalid candidate/render/QA counts.")
    if (
        counts["logical_records"] != counts["candidate_unique"]
        or counts["candidate_unique"]
        != counts["geometry_pass"] + counts["geometry_rejected"]
    ):
        raise ValueError("Candidate geometry counts disagree.")
    if complete and (
        counts["pending"]
        or counts["failed"]
        or counts["geometry_pass"] != counts["qa_accepted"] + counts["qa_rejected"]
        or counts["geometry_pass"]
        != counts["rendered_unique"] + counts["render_cache_reused"]
    ):
        raise ValueError(
            "Dataset version has incomplete or inconsistent render/QA counts."
        )


def _count(records: list[dict[str, Any]], *, reused: bool) -> dict[str, int]:
    geometry_rejected = sum(r["state"] == "geometry_rejected" for r in records)
    eligible = len(records) - geometry_rejected
    accepted = sum(r["state"] == "accepted" for r in records)
    return {
        "logical_records": len(records),
        "candidate_unique": len(records),
        "geometry_pass": eligible,
        "geometry_rejected": geometry_rejected,
        "rendered_unique": 0 if reused else eligible,
        "render_cache_reused": eligible if reused else 0,
        "qa_accepted": accepted,
        "qa_rejected": eligible - accepted,
        "pending": 0,
        "failed": 0,
    }


def _read_cache(
    root: Path,
    path: Path,
    key: str,
    members: list[str],
    *,
    quality: QualityPolicy,
    verify_images: bool = False,
) -> dict[str, Any]:
    completion = read_json(path / "completion.json")
    version = 2 if quality.schema == "court_scale_quality_policy_v2" else 1
    authority = _quality_authority(quality)
    if (
        completion["schema"] != f"court_scale_render_cache_v{version}"
        or completion["cache_key"] != key
        or completion["member_ids"] != members
        or any(completion.get(k) != v for k, v in authority.items())
    ):
        raise ValueError("Render cache has a conflicting plan or renderer/QA version.")
    ledger = contained(root, completion["ledger_path"])
    if file_sha256(ledger) != completion["ledger_sha256"]:
        raise ValueError("Render cache QA ledger hash mismatch.")
    ledger_value = read_json(ledger)
    if ledger_value["schema"] != f"court_scale_qa_ledger_v{version}" or any(
        ledger_value.get(k) != v for k, v in authority.items()
    ):
        raise ValueError("QA ledger disagrees with the versioned quality policy.")
    records = ledger_value["records"]
    for item in records:
        if item["qa"] is not None:
            validate_quality_binding(item["qa"], quality)
    if [r["sample_id"] for r in records] != members:
        raise ValueError("Render cache QA ledger inventory mismatch.")
    if any(
        r["state"] not in {"geometry_rejected", "qa_rejected", "accepted"}
        or (r["state"] == "accepted") != (not r["reasons"])
        for r in records
    ):
        raise ValueError("Render cache has pending, failed or inconsistent QA states.")
    validate_counts(_count(records, reused=True))
    store = completion["store"]
    if store is not None:
        store_root = contained(root, store["path"], directory=True)
        if (
            file_sha256(store_root / "dataset.json") != store["dataset_sha256"]
            or file_sha256(store_root / "samples/index.npz") != store["index_sha256"]
        ):
            raise ValueError("Render cache store hash mismatch.")
        opened = open_court_store(store_root)
        opened.validate(decode=verify_images)
    elif any(r["state"] == "accepted" for r in records):
        raise ValueError("Accepted cache records have no image store.")
    return {**completion, "records": records}


def _render_cache(
    root: Path,
    scene: dict[str, Any],
    group: dict[str, Any],
    members: list[dict[str, Any]],
    *,
    renderer: dict[str, Any],
    quality: QualityPolicy,
    recipe: GeometryRecipe,
    layout: MultiCourtLayout,
    render: Render,
) -> tuple[dict[str, Any], bool]:
    key = stable_id(
        "render_cache",
        scene["asset_manifest_sha256"],
        renderer,
        quality.to_dict(),
        members,
    )
    destination = root / "stores" / key
    member_ids = [s["sample_id"] for s in members]
    if destination.exists():
        return _read_cache(
            root, destination, key, member_ids, quality=quality, verify_images=True
        ), True
    staging = root / ".staging" / key
    staging.mkdir(parents=True, exist_ok=False)
    eligible = [s for s in members if not s["reasons"]]
    by_id = {s["sample_id"]: s for s in eligible}
    evaluated: dict[str, dict[str, Any]] = {}
    try:
        for sample_id, arrays in render(scene, eligible) if eligible else ():
            if sample_id not in by_id or sample_id in evaluated:
                raise ValueError("Renderer returned an unknown or duplicate point ID.")
            record = evaluate_render(
                by_id[sample_id],
                arrays,
                layout=layout,
                policy=quality,
                near_plane_m=recipe.near_plane_m,
            )
            evaluated[sample_id] = record
            if not record["reasons"]:
                (staging / f"{sample_id}.jpg").write_bytes(
                    encode_rgb_jpeg(np.round(arrays.rgb * 255).astype(np.uint8))
                )
            del arrays
        if set(evaluated) != set(by_id):
            raise RuntimeError("Renderer left pending camera results.")
    except Exception as error:
        (staging / "failure.json").write_bytes(
            canonical_bytes(
                {
                    "schema": "court_scale_render_failure_v1",
                    "failed": 1,
                    "pending_ids": sorted(set(by_id) - set(evaluated)),
                    "error": str(error),
                }
            )
        )
        raise
    records = []
    ledger = []
    for planned in members:
        record = evaluated.get(planned["sample_id"], planned)
        state = (
            "geometry_rejected"
            if planned["reasons"]
            else "qa_rejected"
            if record["reasons"]
            else "accepted"
        )
        ledger.append(
            {
                "sample_id": record["sample_id"],
                "state": state,
                "reasons": record["reasons"],
                "qa": record.get("qa"),
                "record": record,
            }
        )
        if state == "accepted":
            records.append({k: v for k, v in record.items() if k != "reasons"})
    counts = _count(ledger, reused=False)
    validate_counts(counts)
    store_entry = None
    if records:
        fields = list(records[0]["metadata"])
        manifest = {
            "schema": "canonical_court_dataset_v3",
            "status": "completed",
            "scene_id": scene["scene_id"],
            "profile": "scale_train_v1",
            "scale_schema": STORE_PROFILE,
            "seed": records[0]["metadata"]["seed"],
            "metadata_fields": fields,
            "trajectory_groups": [
                {
                    k: v
                    for k, v in group.items()
                    if k not in {"samples", "arc_map", "planned_points_scene_m"}
                }
            ],
            "rejected_samples": [
                r["record"] for r in ledger if r["state"] != "accepted"
            ],
            "metrics": {
                "accepted_frame_count": len(records),
                "rejected_frame_count": len(members) - len(records),
                "split_frame_counts": {
                    "train": len(records),
                    "validation": 0,
                    "test": 0,
                },
            },
        }
        with ImageRecordWriter(staging / "samples") as writer:
            for record in records:
                jpeg = staging / f"{record['sample_id']}.jpg"
                writer.append(jpeg.read_bytes(), record)
                jpeg.unlink()
            finish_court_store(staging, writer, manifest)
        opened = open_court_store(staging)
        opened.validate()
        store_entry = {
            "path": destination.relative_to(root).as_posix(),
            "dataset_sha256": file_sha256(staging / "dataset.json"),
            "index_sha256": file_sha256(staging / "samples/index.npz"),
        }
    qa_version = 2 if quality.schema == "court_scale_quality_policy_v2" else 1
    authority = _quality_authority(quality)
    ledger_value = {
        "schema": f"court_scale_qa_ledger_v{qa_version}",
        **authority,
        "cache_key": key,
        "records": ledger,
    }
    (staging / "qa.json").write_bytes(canonical_bytes(ledger_value))
    completion = {
        "schema": f"court_scale_render_cache_v{qa_version}",
        **authority,
        "cache_key": key,
        "member_ids": member_ids,
        "store": store_entry,
        "ledger_path": (destination / "qa.json").relative_to(root).as_posix(),
        "ledger_sha256": file_sha256(staging / "qa.json"),
    }
    (staging / "completion.json").write_bytes(canonical_bytes(completion))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)
    return {**completion, "records": ledger}, False


def execute_request(root: Path, request_path: Path, render: Render) -> Path:
    """Local publication; Drive durability/retry orchestration belongs to the caller."""
    root = root.resolve(strict=True)
    request_path = request_path.resolve(strict=True)
    if not request_path.is_relative_to(root):
        raise ValueError("Condition request is outside the explicit output root.")
    request = read_json(request_path)
    authority = _request_authority(request)
    receipt = root / "completed-requests" / f"{file_sha256(request_path)}.json"
    if receipt.exists():
        previous = contained(root, read_json(receipt)["dataset_version_path"])
        version = load_dataset_version(previous, verify_images=True)
        if version["request_sha256"] != file_sha256(request_path):
            raise ValueError("Completed request receipt refers to another request.")
        return previous
    quality = QualityPolicy(**request["quality"])
    scenes = []
    for scene in request["scenes"]:
        plan_path = contained(root, scene["plan_path"])
        asset_path = contained(root, scene["asset_manifest_path"])
        if (
            file_sha256(plan_path) != scene["plan_sha256"]
            or file_sha256(asset_path) != scene["asset_manifest_sha256"]
        ):
            raise ValueError("Frozen plan/asset hash mismatch before rendering.")
        plan, asset = read_json(plan_path), read_json(asset_path)
        if request["schema"] == PRODUCTION_REQUEST_SCHEMA:
            ProductionRecipe.from_dict(request["production_recipe"]).validate_plan(plan)
        alignment = AlignmentResult.from_dict(asset["alignment"])
        recipe = GeometryRecipe.from_dict(plan["recipe"])
        accepted_ids: list[str] = []
        accepted_groups: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        stores, cache_entries = [], []
        counts: Counter[str] = Counter(_count([], reused=False))
        for group in iter_plan_groups(root, plan):
            for birth in range(plan["density_level"] + 1):
                born = [
                    s for s in group["samples"] if s["scale"]["birth_level"] == birth
                ]
                # Chunking is fixed by birth order, independent of workers and requested density.
                for offset in range(0, len(born), 64):
                    completed, reused = _render_cache(
                        root,
                        scene,
                        group,
                        born[offset : offset + 64],
                        renderer=request["renderer"],
                        quality=quality,
                        recipe=recipe,
                        layout=alignment.layout,
                        render=render,
                    )
                    for item in completed["records"]:
                        reasons.update(item["reasons"])
                        if item["state"] == "accepted":
                            accepted_ids.append(item["sample_id"])
                            accepted_groups[item["record"]["trajectory_id"]] += 1
                    counts.update(_count(completed["records"], reused=reused))
                    if completed["store"] is not None:
                        stores.append(completed["store"])
                    cache_entries.append(
                        {
                            "path": f"stores/{completed['cache_key']}/completion.json",
                            "sha256": file_sha256(
                                root
                                / f"stores/{completed['cache_key']}/completion.json"
                            ),
                        }
                    )
        validate_counts(dict(counts))
        accepted = sorted(accepted_ids)
        if not accepted:
            raise ValueError(
                f"Empty accepted condition for scene {scene['scene_id']}; counts={dict(counts)}."
            )
        if len(set(accepted)) != len(accepted):
            raise ValueError("Duplicate accepted point IDs.")
        membership_path, membership_hash = write_content(
            root,
            "memberships",
            {"schema": "court_scale_membership_v1", "sample_ids": accepted},
        )
        scenes.append(
            {
                **scene,
                "stores": stores,
                "accepted_ids_path": membership_path,
                "accepted_ids_sha256": membership_hash,
                "counts": dict(counts),
                "rejection_reasons": dict(reasons),
                "render_caches": cache_entries,
                "empty_trajectory_count": plan["trajectory_count"]
                - len(accepted_groups),
            }
        )
    result = {
        "schema": PRODUCTION_VERSION_SCHEMA
        if request["schema"] == PRODUCTION_REQUEST_SCHEMA
        else VERSION_SCHEMA,
        "request_path": request_path.relative_to(root).as_posix(),
        "request_sha256": file_sha256(request_path),
        "plan_sha256": digest([s["plan_sha256"] for s in scenes]),
        **authority,
        "scenes": scenes,
    }
    result["dataset_version"] = digest(result)
    path = root / f"{result['dataset_version']}.dataset.json"
    path.write_bytes(canonical_bytes(result))
    load_dataset_version(path, verify_images=True)
    receipt.parent.mkdir(exist_ok=True)
    receipt.write_bytes(canonical_bytes({"dataset_version_path": path.name}))
    return path


def load_dataset_version(path: Path, *, verify_images: bool = False) -> dict[str, Any]:
    """Strict shared entrypoint for training/review. No latest/store auto-discovery."""
    path = path.resolve(strict=True)
    root = path.parent
    version = read_json(path)
    expected = {
        "schema",
        "dataset_version",
        "request_path",
        "request_sha256",
        "plan_sha256",
        "method",
        "recipe_sha256",
        "generation_seed",
        "split",
        "scenes",
    }
    if version["schema"] == PRODUCTION_VERSION_SCHEMA:
        expected.update(
            {
                "production_recipe",
                "geometry_recipe_sha256",
                "quality_policy_sha256",
                "quality_rule_version",
            }
        )
    if (
        set(version) != expected
        or version["schema"] not in {VERSION_SCHEMA, PRODUCTION_VERSION_SCHEMA}
        or version["split"] != "train"
        or version["dataset_version"]
        != digest({k: v for k, v in version.items() if k != "dataset_version"})
    ):
        raise ValueError("Invalid dataset-version schema, train split or content hash.")
    request_path = contained(root, version["request_path"])
    if file_sha256(request_path) != version["request_sha256"]:
        raise ValueError("Dataset-version request hash mismatch.")
    request = read_json(request_path)
    authority = _request_authority(request)
    expected_schema = (
        PRODUCTION_VERSION_SCHEMA
        if request["schema"] == PRODUCTION_REQUEST_SCHEMA
        else VERSION_SCHEMA
    )
    if version["schema"] != expected_schema or any(
        version[k] != value for k, value in authority.items()
    ):
        raise ValueError("Dataset version disagrees with its frozen request.")
    quality = QualityPolicy(**request["quality"])
    if not version["scenes"] or len({s["scene_id"] for s in version["scenes"]}) != len(
        version["scenes"]
    ):
        raise ValueError("Dataset version requires unique nonempty scenes.")
    if [s["scene_id"] for s in version["scenes"]] != [
        s["scene_id"] for s in request["scenes"]
    ]:
        raise ValueError("Dataset-version scenes disagree with request.")
    for scene, requested in zip(version["scenes"], request["scenes"], strict=True):
        if any(scene[k] != requested[k] for k in requested):
            raise ValueError("Dataset-version scene changed its frozen inputs.")
        validate_counts(scene["counts"])
        if scene["counts"]["qa_accepted"] <= 0:
            raise ValueError("An empty scene cannot be published.")
        for field, hash_field in (
            ("plan_path", "plan_sha256"),
            ("asset_manifest_path", "asset_manifest_sha256"),
            ("accepted_ids_path", "accepted_ids_sha256"),
        ):
            if file_sha256(contained(root, scene[field])) != scene[hash_field]:
                raise ValueError(f"Dataset-version {field} hash mismatch.")
        plan = read_json(contained(root, scene["plan_path"]))
        if (
            plan["schema"] != PLAN_SCHEMA
            or plan["split"] != "train"
            or plan["scene_id"] != scene["scene_id"]
        ):
            raise ValueError("Invalid candidate plan identity.")
        if request["schema"] == PRODUCTION_REQUEST_SCHEMA:
            ProductionRecipe.from_dict(request["production_recipe"]).validate_plan(plan)
        if (
            plan["method"] != request["method"]
            or plan["generation_seed"] != request["generation_seed"]
            or plan["recipe_sha256"]
            != (
                request["geometry_recipe_sha256"]
                if request["schema"] == PRODUCTION_REQUEST_SCHEMA
                else request["recipe_sha256"]
            )
        ):
            raise ValueError("Candidate plan disagrees with its frozen request.")
        asset = read_json(contained(root, scene["asset_manifest_path"]))
        layout = AlignmentResult.from_dict(asset["alignment"]).layout
        planned: dict[str, str] = {}
        references_by_id = {
            r["trajectory_id"]: r for r in plan["trajectory_group_refs"]
        }
        for group in iter_plan_groups(root, plan):
            records = validate_plan_geometry(
                {**plan, "trajectory_groups": [group]}, layout
            )
            for sample_id in records:
                if sample_id in planned:
                    raise ValueError("Duplicate candidate point ID.")
                planned[sample_id] = group["trajectory_id"]

        @lru_cache(maxsize=2)
        def planned_group(
            gid: str,
            refs: dict[str, Any] = references_by_id,
            frozen_plan: dict[str, Any] = plan,
            frozen_layout: MultiCourtLayout = layout,
        ) -> dict[str, dict[str, Any]]:
            reference = refs[gid]
            group = read_json(contained(root, reference["path"]))
            records: dict[str, dict[str, Any]] = validate_plan_geometry(
                {**frozen_plan, "trajectory_groups": [group]}, frozen_layout
            )
            return records

        membership = read_json(contained(root, scene["accepted_ids_path"]))
        accepted = membership["sample_ids"]
        if (
            membership["schema"] != "court_scale_membership_v1"
            or len(accepted) != len(set(accepted))
            or len(accepted) != scene["counts"]["qa_accepted"]
        ):
            raise ValueError("Dataset-version accepted membership/counts disagree.")
        stored_ids = []
        stored_record_hashes: dict[str, str] = {}
        for store_entry in scene["stores"]:
            store_root = contained(root, store_entry["path"], directory=True)
            if (
                file_sha256(store_root / "dataset.json")
                != store_entry["dataset_sha256"]
                or file_sha256(store_root / "samples/index.npz")
                != store_entry["index_sha256"]
            ):
                raise ValueError(
                    "Dataset-version store descriptor/index hash mismatch."
                )
            store = open_court_store(store_root)
            if (
                store.metadata.get("scale_schema") != STORE_PROFILE
                or store.metadata["schema"] != "canonical_court_dataset_v3"
                or store.metadata["scene_id"] != scene["scene_id"]
            ):
                raise ValueError(
                    "Dataset-version store geometry/profile/scene mismatch."
                )
            if verify_images:
                store.validate()
            for i in range(len(store)):
                record = store.record(i)
                if record["split"] != "train":
                    raise ValueError("Scale store contains a non-train sample.")
                if record["sample_id"] not in planned:
                    raise ValueError("Store contains an unplanned point ID.")
                validate_record_geometry(
                    record,
                    planned_group(planned[record["sample_id"]])[record["sample_id"]],
                )
                stored_ids.append(record["sample_id"])
                stored_record_hashes[record["sample_id"]] = digest(record)
        if len(stored_ids) != len(set(stored_ids)) or set(stored_ids) != set(accepted):
            raise ValueError("Unknown/duplicate membership or store point ID.")
        ledger_ids: list[str] = []
        ledger_accepted: list[str] = []
        ledger_counts: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        store_paths = []
        for cache in scene["render_caches"]:
            completion_path = contained(root, cache["path"])
            if file_sha256(completion_path) != cache["sha256"]:
                raise ValueError("Render-cache completion hash mismatch.")
            completion = read_json(completion_path)
            records = _read_cache(
                root,
                completion_path.parent,
                completion["cache_key"],
                completion["member_ids"],
                quality=quality,
                verify_images=verify_images,
            )["records"]
            ledger_ids.extend(r["sample_id"] for r in records)
            ledger_accepted.extend(
                r["sample_id"] for r in records if r["state"] == "accepted"
            )
            for item in records:
                if item["sample_id"] not in planned:
                    raise ValueError("QA ledger contains an unplanned point ID.")
                validate_record_geometry(
                    item["record"],
                    planned_group(planned[item["sample_id"]])[item["sample_id"]],
                )
                qa = item["qa"]
                if qa is None:
                    if item["state"] != "geometry_rejected":
                        raise ValueError(
                            "Rendered QA ledger record has no diagnostics."
                        )
                elif (
                    item["state"] == "geometry_rejected"
                    or item["reasons"] != quality_rejection_reasons(qa, quality)
                    or item["record"].get("qa") != qa
                ):
                    raise ValueError(
                        "QA ledger disagrees with the frozen quality policy."
                    )
                if item["state"] == "accepted" and stored_record_hashes.get(
                    item["sample_id"]
                ) != digest(
                    {k: v for k, v in item["record"].items() if k != "reasons"}
                ):
                    raise ValueError(
                        "Stored sample and QA ledger sparse records disagree."
                    )
            ledger_counts.update(_count(records, reused=False))
            reasons.update(reason for r in records for reason in r["reasons"])
            if completion["store"] is not None:
                store_paths.append(completion["store"]["path"])
        candidate_ids = list(planned)
        if (
            len(ledger_ids) != len(set(ledger_ids))
            or set(ledger_ids) != set(candidate_ids)
            or store_paths != [s["path"] for s in scene["stores"]]
        ):
            raise ValueError("QA ledger/store inventory disagrees with candidate plan.")
        for field in (
            "candidate_unique",
            "geometry_pass",
            "geometry_rejected",
            "qa_accepted",
            "qa_rejected",
        ):
            if scene["counts"][field] != ledger_counts[field]:
                raise ValueError("Dataset-version counts disagree with QA ledger.")
        if set(ledger_accepted) != set(accepted):
            raise ValueError("QA ledger accepted IDs disagree with membership.")
        if scene["rejection_reasons"] != dict(reasons):
            raise ValueError(
                "Dataset-version rejection reasons disagree with QA ledger."
            )
    if version["plan_sha256"] != digest([s["plan_sha256"] for s in version["scenes"]]):
        raise ValueError("Dataset-version aggregate plan hash mismatch.")
    return version
