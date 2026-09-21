"""Bounded parallel acquisition and sequential preparation for URL batches."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import preparation
from .configuration import PrepareConfig, youtube_id
from .kit import build_kit
from .runtime.contracts import SourceInfo, write_json


class BatchPreparationError(RuntimeError):
    """Raised after every batch item has been attempted and indexed."""

    def __init__(self, index_path: Path, failures: int) -> None:
        self.index_path = index_path
        self.failures = failures
        super().__init__(
            f"batch preparation failed for {failures} item(s); results: {index_path}"
        )


def _batch_index_path(config: PrepareConfig, kit_id: str) -> Path:
    identity = {
        "kit_id": kit_id,
        "urls": config.urls,
        "format_selector": config.format_selector,
        "js_runtimes": config.js_runtimes,
        "remote_components": config.remote_components,
        "duration_seconds": config.duration_seconds,
        "context_seconds": config.context_seconds,
        "max_bytes": config.max_bytes,
        "crf": config.crf,
        "preset": config.preset,
        "max_clips_per_video": config.max_clips_per_video,
        "sampling_strategy": config.sampling_strategy,
        "policies": config.policies.model_dump(mode="json"),
        "download_workers": config.download_workers,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()[:16]
    return config.output / "batches" / digest / "batch.json"


def _error_text(error: BaseException) -> str:
    message = " ".join(str(error).splitlines()).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def prepare_batch(config: PrepareConfig) -> Path:
    """Acquire concurrently and encode ready inputs one at a time."""
    if not config.urls or config.url is not None or config.local_video is not None:
        raise ValueError("prepare_batch requires source.urls as the only input")
    config.output.mkdir(parents=True, exist_ok=True)
    kit_directory, kit_id = build_kit(config.output / "project_kits")
    index_path = _batch_index_path(config, kit_id)
    results: list[dict[str, Any]] = [
        {
            "index": position,
            "url": url,
            "youtube_id": youtube_id(url),
            "status": "pending",
            "result": None,
            "error": None,
        }
        for position, url in enumerate(config.urls)
    ]

    def publish(status: str) -> None:
        write_json(
            index_path,
            {
                "schema_version": "tennis_chat_batch.v1",
                "status": status,
                "download_workers": config.download_workers,
                "results": results,
            },
        )

    publish("running")
    with ThreadPoolExecutor(max_workers=config.download_workers) as executor:
        acquisitions: dict[
            Future[tuple[Path, SourceInfo]], tuple[int, PrepareConfig]
        ] = {}
        for position, url in enumerate(config.urls):
            item_config = replace(config, url=url, urls=())
            acquisitions[executor.submit(preparation._acquire, item_config)] = (
                position,
                item_config,
            )
        for future in as_completed(acquisitions):
            position, item_config = acquisitions[future]
            item = results[position]
            item["status"] = "running"
            publish("running")
            try:
                source, source_info = future.result()
                destination = preparation._prepare_acquired(
                    item_config, source, source_info, kit_directory, kit_id
                )
            except Exception as error:
                item["status"] = "failed"
                item["error"] = _error_text(error)
            else:
                item["status"] = "completed"
                item["result"] = str(destination.resolve())
            publish("running")
    failures = sum(item["status"] == "failed" for item in results)
    status = (
        "completed"
        if failures == 0
        else "failed"
        if failures == len(results)
        else "partial"
    )
    publish(status)
    if failures:
        raise BatchPreparationError(index_path, failures)
    return index_path
