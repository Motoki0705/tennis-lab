"""Synchronous cache persistence, explicit failed-staging retry and VM recovery."""

from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.synthetic_data_generation.dataset.court.sample_store import open_court_store
from src.utils.data.image_record_store import file_sha256

from .core import canonical_bytes
from .durable_storage import BundleStore, RemoteFiles, tree_files
from .publication import (
    Render,
    RenderCacheContext,
    contained,
    execute_request,
    iter_plan_groups,
    load_dataset_version,
    read_json,
)
from .telemetry import CacheTelemetry

INPUT_RECEIPT = "receipts/request.json"
FINAL_RECEIPT = "receipts/completed.json"


class GenerationInterrupted(RuntimeError):
    """An explicitly requested stop after a durably acknowledged cache."""


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(canonical_bytes(value))
    temporary.replace(path)


def _input_files(root: Path, request_path: Path) -> list[str]:
    request = read_json(request_path)
    files = [request_path.relative_to(root).as_posix()]
    for scene in request["scenes"]:
        for field in ("plan_path", "asset_manifest_path"):
            path = contained(root, scene[field])
            if file_sha256(path) != scene[field.replace("_path", "_sha256")]:
                raise ValueError("Frozen input hash mismatch before Drive persistence.")
            files.append(scene[field])
        plan = read_json(root / scene["plan_path"])
        # Verify the complete curve inventory before the first renderer call.
        for _ in iter_plan_groups(root, plan):
            pass
        files.extend(r["path"] for r in plan["trajectory_group_refs"])
    return files


def _check_receipt(receipt: dict[str, Any], **identity: Any) -> None:
    if any(receipt.get(key) != value for key, value in identity.items()):
        raise ValueError(
            "Durable receipt identity disagrees with the frozen request/cache."
        )


def restore_inputs(root: Path, remote: RemoteFiles, *, request_sha256: str) -> Path:
    store = BundleStore(root, remote)
    receipt = store.receipt(INPUT_RECEIPT)
    if receipt is None:
        raise ValueError("Drive has no committed condition-input receipt.")
    _check_receipt(receipt, kind="inputs", request_sha256=request_sha256)
    request_relative = f"requests/{request_sha256}.json"
    if receipt.get("request_path") != request_relative or any(
        Path(name).parts[0] not in {"requests", "assets", "plans", "curves"}
        for name in receipt["files"]
    ):
        raise ValueError("Invalid durable input inventory.")
    store.restore(receipt)
    request_path: Path = contained(root, request_relative)
    if (
        file_sha256(request_path) != request_sha256
        or sorted(_input_files(root, request_path)) != receipt["files"]
    ):
        raise ValueError("Restored inputs disagree with the pinned request.")
    return request_path


def _cache_receipt(key: str) -> str:
    return f"receipts/caches/{key}.json"


def _restore_cache(
    store: BundleStore,
    key: str,
    request_sha: str,
    *,
    receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if receipt is None:
        receipt = store.receipt(_cache_receipt(key))
    if receipt is None:
        raise ValueError(f"Drive has no acknowledged cache: {key}.")
    _check_receipt(receipt, kind="cache", cache_key=key, request_sha256=request_sha)
    if any(
        not name.startswith(f"stores/{key}/") and name != f"cache-metrics/{key}.json"
        for name in receipt["files"]
    ):
        raise ValueError("Durable cache inventory escapes its cache unit.")
    store.restore(receipt)
    return receipt


def fetch_completed_dataset(
    root: Path, remote: RemoteFiles, *, request_sha256: str
) -> Path:
    """Recover only acknowledged bundles, then strictly decode all JPEG stores."""
    restore_inputs(root, remote, request_sha256=request_sha256)
    return _fetch_completed_publication(root, remote, request_sha256=request_sha256)


def _fetch_completed_publication(
    root: Path, remote: RemoteFiles, *, request_sha256: str
) -> Path:
    """Recover a publication after this attempt has verified its frozen inputs."""
    store = BundleStore(root, remote)
    receipt = store.receipt(FINAL_RECEIPT)
    if receipt is None:
        raise ValueError(
            "Drive has no completed dataset receipt; resume generation instead."
        )
    _check_receipt(receipt, kind="publication", request_sha256=request_sha256)
    version_name = receipt["dataset_version_path"]
    if Path(version_name).name != version_name or not version_name.endswith(
        ".dataset.json"
    ):
        raise ValueError("Invalid durable dataset-version path.")
    allowed = {
        version_name,
        f"completed-requests/{request_sha256}.json",
        f"reports/{request_sha256}.json",
    }
    if any(
        name not in allowed and not name.startswith("memberships/")
        for name in receipt["files"]
    ):
        raise ValueError("Invalid durable publication inventory.")
    store.restore(receipt)
    version_path: Path = contained(root, version_name)
    if file_sha256(version_path) != receipt["dataset_version_sha256"]:
        raise ValueError("Drive dataset-version hash mismatch.")
    version = read_json(version_path)
    for scene in version["scenes"]:
        for cache in scene["render_caches"]:
            _restore_cache(store, Path(cache["path"]).parent.name, request_sha256)
    checked = load_dataset_version(version_path, verify_images=True)
    completed = read_json(contained(root, f"completed-requests/{request_sha256}.json"))
    if completed != {"dataset_version_path": version_name}:
        raise ValueError(
            "Completed-request receipt disagrees with the durable version."
        )
    # Exercise the consumer boundary explicitly, rather than relying on tar hashes.
    for scene in checked["scenes"]:
        for entry in scene["stores"]:
            open_court_store(contained(root, entry["path"], directory=True)).validate(
                decode=True
            )
    return version_path


def _storage_bytes(root: Path, completion: dict[str, Any]) -> dict[str, int]:
    entry = completion["store"]
    if entry is None:
        return {"jpeg_shard_bytes": 0, "index_bytes": 0}
    store = contained(root, entry["path"], directory=True)
    return {
        "jpeg_shard_bytes": sum(
            p.stat().st_size for p in (store / "samples/shards").glob("*.bin")
        ),
        "index_bytes": (store / "samples/index.npz").stat().st_size,
    }


def _metrics(root: Path, context: RenderCacheContext) -> dict[str, Any]:
    value: dict[str, Any] = read_json(
        contained(root, f"cache-metrics/{context.cache_key}.json")
    )
    if (
        value.get("schema") != "court_scale_cache_metrics_v1"
        or value.get("cache") != asdict(context)
        or value["completion_sha256"]
        != file_sha256(root / f"stores/{context.cache_key}/completion.json")
        or not isinstance(value["generation_seconds"], (int, float))
        or not math.isfinite(value["generation_seconds"])
        or value["generation_seconds"] < 0
    ):
        raise ValueError("Cache generation metrics disagree with their completed unit.")
    return value


def generation_report(
    root: Path, version_path: Path, release: dict[str, Any]
) -> dict[str, Any]:
    """Derived costs/rates; counts and decision reasons stay in version/QA ledgers."""
    version = load_dataset_version(version_path, verify_images=True)
    scenes = []
    for index, scene in enumerate(version["scenes"]):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for cache in scene["render_caches"]:
            key = Path(cache["path"]).parent.name
            metric = read_json(contained(root, f"cache-metrics/{key}.json"))
            _metrics(root, RenderCacheContext(**metric["cache"]))
            groups[metric["cache"]["trajectory_id"]].append(metric)
        costs = []
        for trajectory_id, metrics in sorted(groups.items()):
            costs.append(
                {
                    "trajectory_id": trajectory_id,
                    "generation_seconds": sum(m["generation_seconds"] for m in metrics),
                    "gpu_sampled_peak_bytes": max(
                        m["gpu_memory"]["sampled_peak_bytes"] for m in metrics
                    ),
                    "jpeg_shard_bytes": sum(
                        m["storage"]["jpeg_shard_bytes"] for m in metrics
                    ),
                    "index_bytes": sum(m["storage"]["index_bytes"] for m in metrics),
                    "cache_metrics_paths": [
                        f"cache-metrics/{m['cache']['cache_key']}.json" for m in metrics
                    ],
                }
            )
        counts = scene["counts"]
        scenes.append(
            {
                "scene_id": scene["scene_id"],
                "counts_ref": {
                    "path": version_path.name,
                    "json_pointer": f"/scenes/{index}/counts",
                },
                "rejection_reasons_ref": {
                    "path": version_path.name,
                    "json_pointer": f"/scenes/{index}/rejection_reasons",
                },
                "decision_authority": [
                    str(Path(c["path"]).with_name("qa.json"))
                    for c in scene["render_caches"]
                ],
                "candidate_acceptance_rate": counts["qa_accepted"]
                / counts["candidate_unique"],
                "rendered_acceptance_rate": counts["qa_accepted"]
                / counts["geometry_pass"],
                "generation_seconds": sum(g["generation_seconds"] for g in costs),
                "gpu_sampled_peak_bytes": max(
                    g["gpu_sampled_peak_bytes"] for g in costs
                ),
                "jpeg_shard_bytes": sum(g["jpeg_shard_bytes"] for g in costs),
                "index_bytes": sum(g["index_bytes"] for g in costs),
                "groups": costs,
            }
        )
    return {
        "schema": "court_scale_generation_report_v1",
        "dataset_version_path": version_path.name,
        "dataset_version_sha256": file_sha256(version_path),
        "decision_counts": "qa_accepted and geometry_rejected/qa_rejected in counts_ref; reason frequencies in rejection_reasons_ref",
        "generation_timing": "per-cache render, QA and JPEG/index publication; excludes Drive transfer and cache restoration",
        "gpu_memory": "sampled device-wide peak is a lower bound; allocator peaks are unavailable at the public subprocess boundary",
        "renderer_release": release,
        "scenes": scenes,
    }


class DurableGeneration:
    def __init__(
        self,
        root: Path,
        request_path: Path,
        remote: RemoteFiles,
        telemetry: CacheTelemetry,
        *,
        retry_staging: bool = False,
        stop_after_caches: int | None = None,
    ) -> None:
        self.root = root.resolve(strict=True)
        self.request_path = request_path.resolve(strict=True)
        if not self.request_path.is_relative_to(self.root):
            raise ValueError("Request must be inside the explicit output root.")
        self.sha = file_sha256(self.request_path)
        if (
            self.request_path.relative_to(self.root).as_posix()
            != f"requests/{self.sha}.json"
        ):
            raise ValueError(
                "Durable generation requires the frozen request's canonical path."
            )
        if stop_after_caches is not None and (
            type(stop_after_caches) is not int or stop_after_caches < 1
        ):
            raise ValueError("stop_after_caches must be a positive integer.")
        self.store = BundleStore(self.root, remote)
        self.telemetry, self.retry_staging, self.stop_after = (
            telemetry,
            retry_staging,
            stop_after_caches,
        )
        self.attempt_id = uuid.uuid4().hex
        self.events: list[dict[str, Any]] = []
        self.active = False
        self.started = 0.0
        self.restored = False
        self.restored_receipt: dict[str, Any] | None = None
        self.acknowledged: set[str] = set()

    def _save_failed_staging(self, directory: Path, key: str) -> None:
        self.store.publish(
            f"receipts/failures/{directory.parent.name}-{key}.json",
            tree_files(self.root, directory),
            identity={
                "kind": "failed_staging",
                "request_sha256": self.sha,
                "cache_key": key,
            },
        )

    def prepare(self, context: RenderCacheContext) -> None:
        key = context.cache_key
        staging = self.root / ".staging" / key
        if staging.exists():
            if not self.retry_staging:
                raise RuntimeError(
                    f"Incomplete staging for {key}; inspect failure.json and explicitly pass --retry-staging to archive and retry."
                )
            archive_path = self.root / "failed-attempts" / self.attempt_id / key
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(archive_path)
            _write(
                archive_path / "retry.json",
                {
                    "schema": "court_scale_staging_retry_v1",
                    "cache": asdict(context),
                    "request_sha256": self.sha,
                    "original_path": f".staging/{key}",
                    "action": "archived_before_explicit_retry",
                },
            )
            self._save_failed_staging(archive_path, key)
        receipt = self.store.receipt(_cache_receipt(key))
        self.restored = receipt is not None
        self.restored_receipt = None
        if self.restored:
            receipt = _restore_cache(self.store, key, self.sha, receipt=receipt)
            _check_receipt(receipt, cache=asdict(context))
            _metrics(self.root, context)
            self.restored_receipt = receipt
        elif (self.root / "stores" / key).exists():
            # A local completion after an interrupted upload still needs its original metrics.
            _metrics(self.root, context)
        self.telemetry.begin()
        self.active = True
        self.started = time.perf_counter()

    def complete(
        self, context: RenderCacheContext, completion: dict[str, Any], reused: bool
    ) -> None:
        seconds = time.perf_counter() - self.started
        self.active = False
        gpu_memory = self.telemetry.end()
        key = context.cache_key
        if not reused:
            _write(
                self.root / f"cache-metrics/{key}.json",
                {
                    "schema": "court_scale_cache_metrics_v1",
                    "cache": asdict(context),
                    "completion_sha256": file_sha256(
                        self.root / f"stores/{key}/completion.json"
                    ),
                    "generation_seconds": seconds,
                    "gpu_memory": gpu_memory,
                    "storage": _storage_bytes(self.root, completion),
                },
            )
        _metrics(self.root, context)
        saved = self.restored_receipt or self.store.publish(
            _cache_receipt(key),
            [
                *tree_files(self.root, self.root / "stores" / key),
                f"cache-metrics/{key}.json",
            ],
            identity={
                "kind": "cache",
                "request_sha256": self.sha,
                "cache_key": key,
                "cache": asdict(context),
            },
        )
        _write(self.root / f".durability/acknowledged/{key}.json", saved)
        self.acknowledged.add(key)
        self.events.append(
            {
                "cache_key": key,
                "reused": reused,
                "restored_from_drive": self.restored,
                "drive_acknowledged": True,
            }
        )
        if self.stop_after is not None and len(self.events) == self.stop_after:
            raise GenerationInterrupted(
                "Requested interruption after a verified Drive cache acknowledgement."
            )

    def run(self, render: Render) -> Path | None:
        outcome, version_path = "failed", None
        error_text = None
        release = None
        try:
            self.store.publish(
                INPUT_RECEIPT,
                _input_files(self.root, self.request_path),
                identity={
                    "kind": "inputs",
                    "request_sha256": self.sha,
                    "request_path": self.request_path.relative_to(self.root).as_posix(),
                },
            )
            # A prior explicit retry may have stopped while saving its quarantine.
            # Finish that exact save before any more rendering; do not lose evidence.
            for retry_path in sorted(self.root.glob("failed-attempts/*/*/retry.json")):
                retry = read_json(retry_path)
                if retry.get("request_sha256") == self.sha:
                    self._save_failed_staging(retry_path.parent, retry_path.parent.name)
            if self.store.remote.exists(FINAL_RECEIPT):
                version_path = _fetch_completed_publication(
                    self.root, self.store.remote, request_sha256=self.sha
                )
                release = self.telemetry.released()
            else:
                version_path = execute_request(
                    self.root,
                    self.request_path,
                    render,
                    prepare_cache=self.prepare,
                    complete_cache=self.complete,
                )
                release = self.telemetry.released()
                self._publish_version(version_path, release)
            outcome = "completed"
            return version_path
        except GenerationInterrupted as error:
            outcome, error_text = "interrupted", str(error)
            return None
        except BaseException as error:
            error_text = f"{type(error).__name__}: {error}"
            raise
        finally:
            cleanup_error = None
            if self.active:
                try:
                    self.telemetry.end()
                except Exception as error:
                    cleanup_error = error
                finally:
                    self.active = False
            if release is None:
                try:
                    release = self.telemetry.released()
                except Exception as error:
                    cleanup_error = error
                    release = {"renderer_context_released": False, "error": str(error)}
            if cleanup_error is not None:
                outcome = "failed"
                error_text = f"{error_text or ''}; GPU cleanup: {cleanup_error}"
            status = {
                "schema": "court_scale_generation_attempt_v1",
                "attempt_id": self.attempt_id,
                "request_sha256": self.sha,
                "outcome": outcome,
                "error": error_text,
                "caches": self.events,
                "retry_staging": self.retry_staging,
                "renderer_release": release,
            }
            path = self.root / f"attempts/{self.attempt_id}.json"
            _write(path, status)
            self.store.remote.put(path, f"attempts/{self.attempt_id}.json")
            if cleanup_error is not None:
                raise RuntimeError(
                    "GPU cleanup/release could not be verified; see the attempt record."
                ) from cleanup_error

    def _publish_version(self, version_path: Path, release: dict[str, Any]) -> None:
        version = load_dataset_version(version_path, verify_images=True)
        for scene in version["scenes"]:
            for cache in scene["render_caches"]:
                key = Path(cache["path"]).parent.name
                if key not in self.acknowledged:
                    _restore_cache(self.store, key, self.sha)
        report_path = self.root / f"reports/{self.sha}.json"
        if report_path.exists():
            value = read_json(report_path)
            if value["dataset_version_sha256"] != file_sha256(version_path):
                raise ValueError(
                    "Existing generation report belongs to another dataset version."
                )
        else:
            _write(report_path, generation_report(self.root, version_path, release))
        files = [
            version_path.name,
            f"completed-requests/{self.sha}.json",
            report_path.relative_to(self.root).as_posix(),
        ]
        files.extend(scene["accepted_ids_path"] for scene in version["scenes"])
        self.store.publish(
            FINAL_RECEIPT,
            files,
            identity={
                "kind": "publication",
                "request_sha256": self.sha,
                "dataset_version_path": version_path.name,
                "dataset_version_sha256": file_sha256(version_path),
            },
        )
