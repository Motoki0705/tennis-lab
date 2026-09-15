"""Checkpoint compatibility and inference contract for the Court UI backend."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.tasks.court_detection.data.bundle_state import serialize_target_bundle
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.visualization.inference import service as service_module
from src.tasks.court_detection.visualization.inference.checkpoints import (
    COURT_TASK,
    CourtCheckpointInfo,
    CourtHeadSpec,
    describe_checkpoint,
    scan_checkpoints,
)
from src.tasks.court_detection.visualization.inference.metrics import (
    resize_probability_bilinear,
)
from src.tasks.court_detection.visualization.inference.service import (
    DetectionService,
    _compatibility_reason,
)
from src.tasks.court_detection.visualization.review.datasets import (
    DENSE_TARGET_SCHEMAS,
    CourtDatasetLayers,
)
from src.utils.schema.court import COURT_KP_NAMES, GROUND_COURT_KP_NAMES

pytestmark = pytest.mark.unit


def _heads(
    *,
    kp_channels: tuple[str, ...] = GROUND_COURT_KP_NAMES,
    seg_schema: str = DENSE_TARGET_SCHEMAS["seg"],
) -> tuple[CourtHeadSpec, ...]:
    return (
        CourtHeadSpec(
            kind="kp",
            schema="fixture_kp14",
            output_channels=len(kp_channels),
            channel_names=kp_channels,
        ),
        CourtHeadSpec(
            kind="seg",
            schema=seg_schema,
            output_channels=7,
            channel_names=tuple(f"class_{index}" for index in range(7)),
        ),
    )


def _info(
    heads: tuple[CourtHeadSpec, ...],
    *,
    path: Path = Path("/tmp/epoch.ckpt"),
) -> CourtCheckpointInfo:
    try:
        stat = path.stat()
        size_bytes, modified_ns = stat.st_size, stat.st_mtime_ns
    except OSError:
        size_bytes, modified_ns = 1, 1
    return CourtCheckpointInfo(
        id="run/checkpoints/epoch.ckpt",
        path=path,
        label="run/checkpoints/epoch",
        model="court_hierarchical",
        metadata_source="checkpoint",
        supported=True,
        reason=None,
        heads=heads,
        size_bytes=size_bytes,
        modified_ns=modified_ns,
    )


def _layers() -> CourtDatasetLayers:
    return CourtDatasetLayers(
        keypoint_schema="fixture_kp14",
        keypoint_channel_names=GROUND_COURT_KP_NAMES,
        dense_schemas=DENSE_TARGET_SCHEMAS,
    )


def test_compatibility_requires_matching_channel_and_dense_schemas() -> None:
    assert _compatibility_reason(_info(_heads()), _layers()) is None

    legacy_seg = _info(_heads(seg_schema="court_cell_segmentation_v1"))
    reason = _compatibility_reason(legacy_seg, _layers())
    assert reason is not None and "supervision schema が異なります" in reason

    other_channels = _info(
        _heads(kp_channels=COURT_KP_NAMES[:7]),
    )
    reason = _compatibility_reason(other_channels, _layers())
    assert reason is not None and "channel semantics が異なります" in reason

    assert _compatibility_reason(replace(_info(()), heads=()), _layers()) is not None


def test_synthetic_v1_scene_is_excluded_from_checkpoint_comparison() -> None:
    """A v1 all-courts scene has different KP semantics and must be excluded."""
    from src.tasks.court_detection.visualization.review.datasets import (
        layer_identity,
    )

    identity = layer_identity(
        source_kind="synthetic_court",
        published_schema="canonical_court_dataset_v1",
    )
    reason = _compatibility_reason(_info(_heads()), identity)

    assert len(identity.keypoint_channel_names) == 7
    assert reason is not None and "channel semantics が異なります" in reason


def _write_checkpoint(
    path: Path,
    *,
    bundle: dict[str, object] | None,
    config: dict[str, object] | None = None,
) -> None:
    hyper_parameters: dict[str, object] = {
        "config": config if config is not None else {}
    }
    if bundle is not None:
        hyper_parameters["target_bundle_state"] = bundle
    torch.save({"hyper_parameters": hyper_parameters, "state_dict": {}}, path)


def _kp_bundle() -> dict[str, object]:
    return dict(
        serialize_target_bundle(
            CourtTargetBundleSpec(
                {
                    "kp": CourtTargetSpec(
                        kind="kp",
                        schema="fixture_kp14",
                        output_channels=14,
                        channel_names=GROUND_COURT_KP_NAMES,
                        target_dtype=torch.float32,
                        precomputed=False,
                    )
                }
            )
        )
    )


def test_legacy_checkpoint_without_bundle_is_unsupported(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / COURT_TASK
    checkpoint = (
        output_root / "run" / "logs" / "version_0" / "checkpoints" / "legacy.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    _write_checkpoint(checkpoint, bundle=None)

    described = describe_checkpoint(
        output_root,
        tmp_path / "ckpt" / COURT_TASK,
        "run/logs/version_0/checkpoints/legacy.ckpt",
    )

    assert described.supported is False
    assert described.reason is not None and "target_bundle_state" in described.reason
    assert described.heads == ()


def test_invalid_bundle_and_stale_sidecar_are_rejected(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / COURT_TASK
    checkpoint = (
        output_root / "run" / "logs" / "version_0" / "checkpoints" / "broken.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    _write_checkpoint(
        checkpoint, bundle={"schema": "court_target_bundle_v0", "targets": []}
    )

    described = describe_checkpoint(
        output_root,
        tmp_path / "ckpt" / COURT_TASK,
        "run/logs/version_0/checkpoints/broken.ckpt",
    )
    assert described.supported is False
    assert (
        described.reason is not None
        and "target_bundle_state が不正" in described.reason
    )

    # A sibling hparams.yaml that disagrees with the body is never trusted.
    checkpoint2 = checkpoint.with_name("stale.ckpt")
    _write_checkpoint(
        checkpoint2,
        bundle=_kp_bundle(),
        config={"data": {"source": {"kind": "synthetic_court", "schema": "v3"}}},
    )
    (checkpoint2.parent / "hparams.yaml").write_text(
        "config:\n  data:\n    source:\n      kind: synthetic_court\n      schema: v1\n",
        encoding="utf-8",
    )
    described2 = describe_checkpoint(
        output_root,
        tmp_path / "ckpt" / COURT_TASK,
        "run/logs/version_0/checkpoints/stale.ckpt",
    )
    assert described2.supported is False
    assert described2.reason is not None and "hparams.yaml" in described2.reason


def test_scan_skips_symlinked_checkpoints(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / COURT_TASK
    checkpoint = output_root / "run" / "checkpoints" / "epoch.ckpt"
    checkpoint.parent.mkdir(parents=True)
    _write_checkpoint(checkpoint, bundle=_kp_bundle())
    checkpoint.with_name("best.ckpt").symlink_to(checkpoint.name)

    scanned = scan_checkpoints(output_root, tmp_path / "ckpt" / COURT_TASK)

    assert [item.id for item in scanned] == ["run/checkpoints/epoch.ckpt"]


def test_checkpoint_ids_cannot_escape_the_configured_roots(tmp_path: Path) -> None:
    from src.tasks.court_detection.visualization.inference.checkpoints import (
        resolve_checkpoint_path,
    )

    output_root = tmp_path / "outputs" / COURT_TASK
    checkpoint_root = tmp_path / "ckpt" / COURT_TASK
    checkpoint_root.mkdir(parents=True)
    outside = tmp_path / "outside.ckpt"
    _write_checkpoint(outside, bundle=_kp_bundle())
    (checkpoint_root / "linked.ckpt").symlink_to(outside)
    (checkpoint_root / "link-dir").symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(ValueError, match="絶対 path"):
        resolve_checkpoint_path(checkpoint_root, str(outside))
    with pytest.raises(ValueError, match="安全な相対 path"):
        resolve_checkpoint_path(checkpoint_root, "../outside.ckpt")
    with pytest.raises(ValueError, match="許可 root の外"):
        resolve_checkpoint_path(checkpoint_root, "link-dir/outside.ckpt")
    # A symlink whose target escapes the root is refused as an escape, and a
    # symlink that stays inside the root is never executed directly.
    with pytest.raises(ValueError, match="許可 root の外"):
        resolve_checkpoint_path(checkpoint_root, "linked.ckpt")
    _write_checkpoint(checkpoint_root / "inside.ckpt", bundle=_kp_bundle())
    (checkpoint_root / "inside-link.ckpt").symlink_to("inside.ckpt")
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_path(checkpoint_root, "inside-link.ckpt")
    assert all(
        item.id.startswith("ckpt/")
        for item in scan_checkpoints(output_root, checkpoint_root)
    )


def test_unreadable_body_is_unusable_even_with_a_complete_sidecar(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs" / COURT_TASK
    checkpoint = output_root / "run" / "checkpoints" / "corrupt.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not a pickle")
    (checkpoint.parent / "hparams.yaml").write_text(
        "config:\n"
        "  data:\n"
        "    source:\n"
        "      kind: synthetic_court\n"
        "      schema: v3\n"
        "  model:\n"
        "    name: court_hierarchical\n",
        encoding="utf-8",
    )

    described = describe_checkpoint(
        output_root, tmp_path / "ckpt" / COURT_TASK, "run/checkpoints/corrupt.ckpt"
    )

    assert described.supported is False
    assert described.reason is not None and "使用できません" in described.reason
    assert described.heads == ()


def test_catalog_rescans_new_datasets_and_checkpoints(review_project: Path) -> None:
    service = DetectionService(project_root=review_project)
    before = {entry["id"] for entry in service.catalog()["datasets"]}
    assert "synthetic_court/B07/train" not in before

    scene_root = (
        review_project / "data" / "synthetic_data_generation" / "scenes" / "B07"
    )
    scene_root.mkdir(parents=True)
    (scene_root / "datasets").mkdir()
    source = (
        review_project
        / "data"
        / "synthetic_data_generation"
        / "scenes"
        / "B00"
        / "datasets"
    )
    (scene_root / "datasets" / "court").symlink_to(
        source / "court", target_is_directory=True
    )
    checkpoint = (
        review_project / "outputs" / COURT_TASK / "fresh" / "checkpoints" / "epoch.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    _write_checkpoint(checkpoint, bundle=_kp_bundle())

    refreshed = service.catalog()
    dataset_ids = {entry["id"] for entry in refreshed["datasets"]}
    checkpoint_ids = {entry["id"] for entry in refreshed["checkpoints"]}

    assert "synthetic_court/B07/train" in dataset_ids
    assert "fresh/checkpoints/epoch.ckpt" in checkpoint_ids


def test_checkpoint_changing_after_catalog_is_refused(
    materialized_catalog, review_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = (
        review_project / "outputs" / COURT_TASK / "run" / "checkpoints" / "epoch.ckpt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _write_checkpoint(checkpoint, bundle=_kp_bundle())
    info = _info(
        (
            CourtHeadSpec(
                kind="kp",
                schema="fixture_kp14",
                output_channels=14,
                channel_names=GROUND_COURT_KP_NAMES,
            ),
        ),
        path=checkpoint,
    )
    monkeypatch.setattr(service_module, "describe_checkpoint", lambda *_: info)
    with checkpoint.open("ab") as handle:
        handle.write(b"trailing-bytes")
    service = DetectionService(project_root=review_project)
    scene = service.scenes("tennis_court_detector/val", limit=1)["items"][0]["id"]

    with pytest.raises(ValueError, match="disk 上で変化"):
        service.validate("run/checkpoints/epoch.ckpt", scene, device="cpu")


def _legacy_checkpoint(project: Path, name: str = "legacy.ckpt") -> str:
    checkpoint = project / "outputs" / COURT_TASK / "run" / "checkpoints" / name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _write_checkpoint(checkpoint, bundle=None)
    return f"run/checkpoints/{name}"


def test_validation_rejects_unsupported_requests(review_project: Path) -> None:
    service = DetectionService(project_root=review_project)
    identifier = _legacy_checkpoint(review_project)
    scene = service.scenes("tennis_court_detector/train", limit=1)["items"][0]["id"]

    with pytest.raises(ValueError, match="非対応"):
        service.validate(identifier, scene, device="cpu")
    with pytest.raises(ValueError, match="count は 1"):
        service.validate(identifier, scene, count=2, device="cpu")
    with pytest.raises(ValueError, match="start は 0"):
        service.validate(identifier, scene, start=3, device="cpu")
    with pytest.raises(ValueError, match="しきい値"):
        service.validate(identifier, scene, threshold=1.5, device="cpu")
    with pytest.raises(ValueError, match="device"):
        service.validate(identifier, scene, device="tpu")
    with pytest.raises(FileNotFoundError):
        service.validate("run/checkpoints/missing.ckpt", scene, device="cpu")


def test_unsupported_checkpoint_does_not_list_scenes(review_project: Path) -> None:
    service = DetectionService(project_root=review_project)
    identifier = _legacy_checkpoint(review_project, "legacy-scenes.ckpt")

    with pytest.raises(ValueError, match="非対応"):
        service.scenes("tennis_court_detector/val", checkpoint=identifier)


def test_schema_mismatch_message_is_explicit(review_project: Path) -> None:
    service = DetectionService(project_root=review_project)
    entry = service.datasets.entry("tennis_court_detector/val")
    info = _info(_heads(seg_schema="court_cell_segmentation_v1"))

    with pytest.raises(ValueError, match="supervision schema が異なります"):
        service._require_compatible(info, entry)  # noqa: SLF001


def test_infer_renders_all_heads_from_one_mocked_forward(
    materialized_catalog, review_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The response contract is exercised with a stub runner (no model load)."""
    checkpoint = (
        review_project / "outputs" / COURT_TASK / "run" / "checkpoints" / "epoch.ckpt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _write_checkpoint(
        checkpoint,
        bundle=serialize_target_bundle(
            CourtTargetBundleSpec(
                {
                    kind: CourtTargetSpec(
                        kind=kind,
                        schema=DENSE_TARGET_SCHEMAS[kind],
                        output_channels=channels,
                        channel_names=names,
                        target_dtype=torch.float32,
                        precomputed=True,
                    )
                    for kind, channels, names in (
                        ("seg", 7, tuple(f"class_{index}" for index in range(7))),
                        ("line", 1, ("court_line",)),
                    )
                }
            )
        ),
    )

    class _StubRunner:
        closed = False

        def __init__(self, path: Path, *, device: torch.device, **_: object) -> None:
            assert path.resolve(strict=False) == checkpoint.resolve(strict=False)
            assert device == torch.device("cpu")

        def predict(self, image) -> dict[str, object]:
            height, width = image.height, image.width
            return {
                "kp": service_module.CourtKeypointPrediction(
                    keypoints=torch.zeros(14, 1, 2),
                    scores=torch.full((14, 1), 0.9),
                    valid=torch.ones(14, 1, dtype=torch.bool),
                    heatmaps=torch.zeros(14, 4, 4),
                ),
                "seg": service_module.CourtSegmentationPrediction(
                    mask=torch.zeros(height, width, dtype=torch.long),
                    logits=torch.zeros(7, height, width),
                ),
                "line": service_module.CourtLinePrediction(
                    probability=torch.full((height, width), 0.01),
                    logits=torch.zeros(1, height, width),
                ),
            }

        def close(self) -> None:
            _StubRunner.closed = True

    monkeypatch.setattr(service_module, "CourtHeadRunner", _StubRunner)
    # The strict config gate is covered by the describe/validate tests; this test
    # targets the response contract, so discovery is stubbed to a supported
    # bundle that matches the heads the stub runner returns.
    monkeypatch.setattr(
        service_module,
        "describe_checkpoint",
        lambda *_: _info(
            (
                CourtHeadSpec(
                    kind="kp",
                    schema="fixture_kp14",
                    output_channels=14,
                    channel_names=GROUND_COURT_KP_NAMES,
                ),
                CourtHeadSpec(
                    kind="seg",
                    schema=DENSE_TARGET_SCHEMAS["seg"],
                    output_channels=7,
                    channel_names=tuple(f"class_{index}" for index in range(7)),
                ),
                CourtHeadSpec(
                    kind="line",
                    schema=DENSE_TARGET_SCHEMAS["line"],
                    output_channels=1,
                    channel_names=("court_line",),
                ),
            ),
            path=checkpoint,
        ),
    )
    service = DetectionService(project_root=review_project)
    scene = service.scenes("tennis_court_detector/val", limit=1)["items"][0]["id"]

    result = service.infer(
        "run/checkpoints/epoch.ckpt", scene, count=1, threshold=0.5, device="cpu"
    )

    assert json.dumps(result, allow_nan=False)
    assert _StubRunner.closed is True
    assert result["scene"] == scene and result["start"] == 0
    prediction = result["items"][0]["pred"]
    assert len(prediction["points"]) == 14
    assert [raster["name"] for raster in prediction["rasters"]] == [
        "heatmap",
        "seg",
        "line",
    ]
    assert "kp_mean_error_px" in result["metrics"]
    assert "seg_pixel_accuracy" in result["metrics"]
    assert "line_iou" in result["metrics"]
    for value in result["metrics"].values():
        assert value is None or isinstance(value, (int, float))
    assert result["warnings"]


def test_checkpoint_catalog_reports_settings_and_compatibility(
    materialized_catalog, review_project: Path
) -> None:
    identifier = _legacy_checkpoint(review_project)
    service = DetectionService(project_root=review_project)

    catalog = service.catalog()
    entries = {entry["id"]: entry for entry in catalog["checkpoints"]}

    assert catalog["task"] == COURT_TASK
    assert entries[identifier]["settings"] == {"count": 1, "threshold": 0.5}
    assert entries[identifier]["error"]
    assert entries[identifier]["compatible_datasets"] == []


def test_resize_helper_preserves_requested_grid() -> None:
    resized = resize_probability_bilinear(torch.zeros(2, 2).numpy(), (4, 6))

    assert resized.shape == (4, 6)
