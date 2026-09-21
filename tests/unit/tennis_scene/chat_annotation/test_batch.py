from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tennis_scene.chat_annotation import batch, configuration, preparation
from src.tennis_scene.chat_annotation.batch import BatchPreparationError, prepare_batch
from src.tennis_scene.chat_annotation.configuration import PrepareConfig
from src.tennis_scene.chat_annotation.runtime.contracts import SourceInfo, read_json

URLS = (
    "https://www.youtube.com/watch?v=aaaaaaaaaaa",
    "https://www.youtube.com/watch?v=bbbbbbbbbbb",
    "https://www.youtube.com/watch?v=ccccccccccc",
)


@pytest.fixture
def cfg(tmp_path: Path) -> DictConfig:
    config_dir = Path(configuration.__file__).parent / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        result = compose(
            config_name="prepare",
            overrides=[
                f"paths.data_root={tmp_path}",
                f"paths.output_root={tmp_path / 'output'}",
            ],
        )
    result.source.urls = list(URLS)
    return result


def _source(url: str, root: Path) -> tuple[Path, SourceInfo]:
    video_id = url.split("v=")[1]
    source = root / f"{video_id}.mp4"
    source.write_bytes(video_id.encode())
    return source, SourceInfo(
        source_id=video_id,
        youtube_id=video_id,
        url=url,
        title=video_id,
        filename=source.name,
        sha256="a" * 64,
        bytes=source.stat().st_size,
        acquired_at="2026-09-21T00:00:00+00:00",
    )


def _patch_kit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    kit = tmp_path / "kit"
    kit.mkdir()
    monkeypatch.setattr(batch, "build_kit", lambda _: (kit, "b" * 64))


def test_batch_bounds_parallel_downloads_and_overlaps_sequential_encoding(
    tmp_path: Path, cfg: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PrepareConfig.from_config(cfg)
    _patch_kit(monkeypatch, tmp_path)
    lock = threading.Lock()
    release_downloads = threading.Event()
    blocked_downloads = threading.Event()
    active = 0
    maximum = 0
    blocked = 0

    def acquire(item_config: PrepareConfig) -> tuple[Path, SourceInfo]:
        nonlocal active, maximum, blocked
        assert item_config.url is not None
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            if item_config.url != URLS[1]:
                with lock:
                    blocked += 1
                    if blocked == 2:
                        blocked_downloads.set()
                assert release_downloads.wait(timeout=5)
            return _source(item_config.url, tmp_path)
        finally:
            with lock:
                active -= 1

    prepared: list[str] = []

    def prepare_acquired(
        item_config: PrepareConfig,
        source: Path,
        source_info: SourceInfo,
        kit_directory: Path,
        kit_id: str,
    ) -> Path:
        assert kit_directory.name == "kit" and kit_id == "b" * 64
        assert source.is_file()
        if not prepared:
            assert blocked_downloads.wait(timeout=5)
            with lock:
                assert active == 2
            release_downloads.set()
        prepared.append(source_info.source_id)
        return tmp_path / "results" / str(source_info.source_id)

    monkeypatch.setattr(preparation, "_acquire", acquire)
    monkeypatch.setattr(
        preparation, "_prepare_acquired", prepare_acquired, raising=False
    )
    index = prepare_batch(config)
    payload = read_json(index)
    assert maximum == 2
    assert prepared[0] == "bbbbbbbbbbb"
    assert set(prepared) == {"aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"}
    assert [item["url"] for item in payload["results"]] == list(URLS)
    assert payload["status"] == "completed"


def test_batch_atomically_publishes_running_progress(
    tmp_path: Path, cfg: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PrepareConfig.from_config(cfg)
    _patch_kit(monkeypatch, tmp_path)
    snapshots: list[dict[str, Any]] = []
    real_write_json = batch.write_json

    def capture(path: Path, value: Any) -> None:
        snapshots.append(deepcopy(value))
        real_write_json(path, value)

    def acquire(item_config: PrepareConfig) -> tuple[Path, SourceInfo]:
        assert item_config.url is not None
        return _source(item_config.url, tmp_path)

    def prepare_acquired(
        item_config: PrepareConfig,
        source: Path,
        source_info: SourceInfo,
        kit_directory: Path,
        kit_id: str,
    ) -> Path:
        del item_config, source, kit_directory, kit_id
        return tmp_path / "results" / str(source_info.source_id)

    monkeypatch.setattr(batch, "write_json", capture)
    monkeypatch.setattr(preparation, "_acquire", acquire)
    monkeypatch.setattr(
        preparation, "_prepare_acquired", prepare_acquired, raising=False
    )
    prepare_batch(config)
    assert snapshots[0]["status"] == "running"
    assert [item["status"] for item in snapshots[0]["results"]] == [
        "pending",
        "pending",
        "pending",
    ]
    assert any(
        "running" in [item["status"] for item in snapshot["results"]]
        for snapshot in snapshots[1:-1]
    )
    assert snapshots[-1]["status"] == "completed"


def test_batch_isolates_failures_and_writes_input_order(
    tmp_path: Path, cfg: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PrepareConfig.from_config(cfg)
    _patch_kit(monkeypatch, tmp_path)

    def acquire(item_config: PrepareConfig) -> tuple[Path, SourceInfo]:
        assert item_config.url is not None
        if item_config.url == URLS[1]:
            time.sleep(0.01)
            raise RuntimeError("download unavailable")
        if item_config.url == URLS[0]:
            time.sleep(0.02)
        return _source(item_config.url, tmp_path)

    prepared: list[str] = []

    def prepare_acquired(
        item_config: PrepareConfig,
        source: Path,
        source_info: SourceInfo,
        kit_directory: Path,
        kit_id: str,
    ) -> Path:
        del item_config, source, kit_directory, kit_id
        prepared.append(source_info.source_id)
        return tmp_path / "results" / str(source_info.source_id)

    monkeypatch.setattr(preparation, "_acquire", acquire)
    monkeypatch.setattr(
        preparation, "_prepare_acquired", prepare_acquired, raising=False
    )
    with pytest.raises(BatchPreparationError) as raised:
        prepare_batch(config)
    payload = read_json(raised.value.index_path)
    assert payload["status"] == "partial"
    assert [item["url"] for item in payload["results"]] == list(URLS)
    assert [item["status"] for item in payload["results"]] == [
        "completed",
        "failed",
        "completed",
    ]
    assert "download unavailable" in payload["results"][1]["error"]
    assert set(prepared) == {"aaaaaaaaaaa", "ccccccccccc"}


@pytest.mark.parametrize("mode", ["duplicate", "mixed", "workers", "strategy"])
def test_batch_configuration_rejects_ambiguous_inputs(
    cfg: DictConfig, mode: str
) -> None:
    if mode == "duplicate":
        cfg.source.urls = [URLS[0], "https://youtu.be/aaaaaaaaaaa"]
    elif mode == "mixed":
        cfg.source.url = URLS[0]
    elif mode == "workers":
        cfg.batch.download_workers = 0
    else:
        cfg.sampling.strategy = "random"
    with pytest.raises(ValueError):
        PrepareConfig.from_config(cfg)


def test_batch_configuration_preserves_url_order(cfg: DictConfig) -> None:
    config = PrepareConfig.from_config(cfg)
    assert config.urls == URLS
    assert config.download_workers == 2
    assert config.max_clips_per_video == 5
    assert config.sampling_strategy == "uniform_midpoints"


def test_hydra_accepts_quoted_url_list() -> None:
    config_dir = Path(configuration.__file__).parent / "configs"
    override = 'source.urls=["' + '","'.join(URLS) + '"]'
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        composed = compose(config_name="prepare", overrides=[override])
    assert tuple(composed.source.urls) == URLS


def test_batch_identity_covers_kit_and_download_settings(cfg: DictConfig) -> None:
    config = PrepareConfig.from_config(cfg)
    baseline = batch._batch_index_path(config, "a" * 64)
    assert batch._batch_index_path(config, "b" * 64) != baseline
    assert (
        batch._batch_index_path(replace(config, js_runtimes="node"), "a" * 64)
        != baseline
    )
    assert (
        batch._batch_index_path(
            replace(
                config,
                policies=config.policies.model_copy(
                    update={"ball_max_gap_seconds": 0.2}
                ),
            ),
            "a" * 64,
        )
        != baseline
    )
