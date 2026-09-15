"""Tests for catalog refresh, root confinement, and strict checkpoint metadata."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.tasks.ball_detection.visualization.inference.service import (
    DetectionRequestError,
    DetectionService,
)
from src.tasks.ball_detection.visualization.review.checkpoints import (
    describe_config,
)
from src.tasks.ball_detection.visualization.review.datasets import (
    BallDatasetCatalog,
    BallDatasetCatalogError,
)
from tests.unit.tasks.ball_detection.visualization.conftest import (
    tiny_model_config,
    write_clip,
)


def service_for(tmp_path: Path) -> DetectionService:
    return DetectionService(
        tmp_path,
        data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs" / "ball_detection",
        checkpoints_root=tmp_path / "ckpt" / "ball_detection",
    )


def clip_row(frame_index: int, *, x: float = 10.0, y: float = 20.0) -> dict[str, object]:
    return {
        "file name": f"{frame_index:04d}.jpg",
        "instance id": "b001",
        "visibility": 1,
        "x-coordinate": x,
        "y-coordinate": y,
        "ball state": "visible",
        "role": "target",
    }


def write_tracknet_clip(
    tmp_path: Path,
    name: str,
    *,
    frames: int,
    labelled: tuple[int, ...] | None = None,
) -> Path:
    """Write one clip whose rows may cover only some of its frames."""
    indices = tuple(range(frames)) if labelled is None else labelled
    return write_clip(
        tmp_path / "data" / "tennis" / "tracknet" / "game1" / name,
        frames=frames,
        rows=[clip_row(index) for index in indices],
    )


# --------------------------------------------------------------- 1) refresh


def test_catalog_rescan_adds_new_checkpoint_and_clip(tmp_path: Path) -> None:
    service = service_for(tmp_path)
    write_tracknet_clip(tmp_path, "Clip1", frames=4)
    assert service.catalog()["checkpoints"] == []

    # A newly copied checkpoint and a newly added clip appear without a restart.
    from tests.unit.tasks.ball_detection.visualization.conftest import (
        tiny_checkpoint_from_config,
    )

    tiny_checkpoint_from_config(
        tmp_path / "outputs" / "ball_detection" / "run-new.ckpt"
    )
    write_tracknet_clip(tmp_path, "Clip2", frames=4)

    catalog = service.catalog()
    assert [entry["id"] for entry in catalog["checkpoints"]] == ["run-new.ckpt"]
    assert service.scenes("tracknet", limit=10)["total"] == 2


def test_unavailable_reason_survives_repeated_catalog_calls(tmp_path: Path) -> None:
    service = service_for(tmp_path)
    write_tracknet_clip(tmp_path, "Clip1", frames=4)
    first = {entry["id"]: entry for entry in service.catalog()["datasets"]}
    assert first["web_static"]["available"] is False
    assert "index.npz" in first["web_static"]["reason"]

    # A second call must not lose the reason when it hits the cached discovery.
    cached = {
        entry.spec.id: entry for entry in service.dataset_catalog.entries()
    }
    again = {entry.spec.id: entry for entry in service.dataset_catalog.entries()}
    second = {entry["id"]: entry for entry in service.catalog()["datasets"]}
    for dataset_id in ("web_static", "web_temporal"):
        assert cached[dataset_id].reason is not None
        assert again[dataset_id].reason == cached[dataset_id].reason
        assert second[dataset_id]["reason"] == first[dataset_id]["reason"]
    assert any("web_static" in text for text in service.catalog()["warnings"])


def test_infer_rereads_replaced_checkpoint_metadata(
    tmp_path: Path,
    make_tiny_checkpoint: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = service_for(tmp_path)
    write_tracknet_clip(tmp_path, "Clip1", frames=6)
    path = make_tiny_checkpoint(
        tmp_path / "outputs" / "ball_detection" / "run.ckpt", num_frames=2
    )
    scene = "tracknet::game1/Clip1"
    service.catalog()
    service.validate("run.ckpt", scene, start=0, count=2, device="cpu")
    with pytest.raises(DetectionRequestError, match="between 2 and 2"):
        service.validate("run.ckpt", scene, start=0, count=3, device="cpu")

    # Replace the body with a wider window.  The next request must read the new
    # config instead of reusing the description the catalog cached.
    make_tiny_checkpoint(path, num_frames=3)
    assert service.checkpoints()["run.ckpt"].num_frames == 2  # cache is still old
    service.validate("run.ckpt", scene, start=0, count=3, device="cpu")
    assert service.checkpoints()["run.ckpt"].num_frames == 3
    # The inference path performs the same stat check before loading weights.
    make_tiny_checkpoint(path, num_frames=4)
    result = service.infer("run.ckpt", scene, start=0, count=4, device="cpu")
    assert result["metrics"]["window"]["checkpoint_frames"] == 4
    assert service.checkpoints()["run.ckpt"].num_frames == 4

    # A file that disappears between catalog and inference fails explicitly.
    os.remove(path)
    with pytest.raises(DetectionRequestError, match="disappeared"):
        service.validate("run.ckpt", scene, start=0, count=4, device="cpu")


# ------------------------------------------------------- 2) root confinement


def test_checkpoint_symlink_outside_root_is_rejected(
    tmp_path: Path, make_tiny_checkpoint: Callable[..., Path]
) -> None:
    outside = tmp_path / "outside" / "run.ckpt"
    make_tiny_checkpoint(outside, num_frames=2)
    outputs = tmp_path / "outputs" / "ball_detection"
    outputs.mkdir(parents=True)
    link = outputs / "linked.ckpt"
    link.symlink_to(outside)
    service = service_for(tmp_path)
    write_tracknet_clip(tmp_path, "Clip1", frames=4)

    catalog = service.catalog()
    assert [entry["id"] for entry in catalog["checkpoints"]] == ["linked.ckpt"]
    entry = catalog["checkpoints"][0]
    assert "error" in entry and "resolves outside" in entry["error"]
    assert any("linked.ckpt" in text for text in catalog["warnings"])
    with pytest.raises(DetectionRequestError, match="unusable"):
        service.validate("linked.ckpt", "tracknet::game1/Clip1", count=2, device="cpu")


def test_clip_directory_symlink_outside_root_is_skipped(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside" / "gameX"
    write_clip(outside / "Clip1", frames=4, rows=[clip_row(index) for index in range(4)])
    write_tracknet_clip(tmp_path, "Clip1", frames=4)
    root = tmp_path / "data" / "tennis" / "tracknet"
    (root / "game_escaped").symlink_to(outside, target_is_directory=True)

    catalog = BallDatasetCatalog(tmp_path / "data")
    entries = {entry.spec.id: entry for entry in catalog.entries()}
    tracknet = entries["tracknet"]
    # The good scene is still served, and the escaping link is reported.
    assert [ref.local_id for ref in catalog.refs("tracknet")] == ["game1/Clip1"]
    assert any("resolves outside" in text for text in tracknet.warnings)
    assert any(
        "resolves outside" in text for text in tracknet.to_dict()["warnings"]
    )


def test_frame_symlink_outside_root_disqualifies_the_scene(tmp_path: Path) -> None:
    clip = write_tracknet_clip(tmp_path, "Clip1", frames=3)
    secret = tmp_path / "secret.jpg"
    secret.write_bytes((clip / "0000.jpg").read_bytes())
    (clip / "0002.jpg").unlink()
    (clip / "0002.jpg").symlink_to(secret)

    catalog = BallDatasetCatalog(tmp_path / "data")
    entries = {entry.spec.id: entry for entry in catalog.entries()}
    assert not entries["tracknet"].available
    assert entries["tracknet"].reason is not None
    assert any(
        "resolve outside the dataset root" in text
        for text in entries["tracknet"].warnings
    )
    with pytest.raises(BallDatasetCatalogError):
        catalog.resolve("tracknet", "game1/Clip1")


def test_web_store_with_escaping_reference_is_rejected(tmp_path: Path) -> None:
    from tests.unit.tasks.ball_detection.visualization.conftest import (
        write_unified_store,
    )

    store = write_unified_store(tmp_path / "data")
    import json

    strings = json.loads((store / "index_strings.json").read_text(encoding="utf-8"))
    strings["paths"] = ["../../../../etc/passwd"]
    (store / "index_strings.json").write_text(json.dumps(strings), encoding="utf-8")

    catalog = BallDatasetCatalog(tmp_path / "data")
    entries = {entry.spec.id: entry for entry in catalog.entries()}
    for dataset_id in ("web_static", "web_temporal"):
        assert not entries[dataset_id].available
        reason = entries[dataset_id].reason
        assert reason is not None
        assert "non-escaping" in reason


# ------------------------------------------------- 3) strict early validation


def test_catalog_marks_missing_input_mode_unusable() -> None:
    config = tiny_model_config()
    del config["model"]["input_mode"]
    info = describe_config(
        checkpoint_id="run.ckpt",
        path=Path("/tmp/run.ckpt"),
        label="run",
        config=config,
    )
    assert not info.usable
    assert info.error is not None and "input_mode" in info.error


def test_catalog_marks_missing_image_size_unusable() -> None:
    config = tiny_model_config()
    del config["data"]["image_size"]
    info = describe_config(
        checkpoint_id="run.ckpt",
        path=Path("/tmp/run.ckpt"),
        label="run",
        config=config,
    )
    assert not info.usable
    assert info.error is not None and "image_size" in info.error


def test_catalog_marks_invalid_metrics_unusable() -> None:
    config = OmegaConf.create(
        {
            "model": {
                "name": "conv_next_unet",
                "input_mode": "mdd",
                "in_channels": 2,
                "num_classes": 1,
                "num_frames": 8,
                "input_layout": "bcthw",
                "dims": [4, 8, 16, 32],
                "depth": 1,
                "drop_path_prob": 0.0,
                "mdd_a": 0.2,
                "mdd_b": 0.15,
            },
            "data": {"image_size": [64, 128]},
            "metrics": {"peak_threshold": float("nan")},
        }
    )
    info = describe_config(
        checkpoint_id="run.ckpt",
        path=Path("/tmp/run.ckpt"),
        label="run",
        config=config,
    )
    assert not info.usable
    assert info.error is not None and "peak_threshold" in info.error


def test_catalog_surfaces_missing_metrics_as_warning_but_keeps_it_usable() -> None:
    config = tiny_model_config()
    config["metrics"] = {"peak_threshold": 0.4}
    info = describe_config(
        checkpoint_id="run.ckpt",
        path=Path("/tmp/run.ckpt"),
        label="run",
        config=config,
    )
    assert info.usable
    assert info.metrics.peak_threshold == 0.4
    assert info.warnings
    assert "warnings" in info.to_dict(compatible_datasets=[])
