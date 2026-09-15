"""Unit tests for PLCS inference-UI checkpoint discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from src.tasks.plcs.visualization.inference.checkpoints import (
    OBJECTNESS_MULTI,
    OBJECTNESS_SINGLE,
    allowed_scene_families,
    describe_checkpoint,
    is_reference_model,
    load_checkpoint_config,
    objectness_for_model,
    resolve_checkpoint,
    resolve_checkpoint_path,
    scan_checkpoints,
    scene_family_of,
)


def _config(
    *,
    model_name: str,
    selector: str,
    input_profile: str | None,
    scene_dir: str,
    max_views: int | None = None,
    max_seq_len: int | None = None,
) -> dict[str, object]:
    model: dict[str, object] = {"name": model_name}
    if input_profile is not None:
        model["io"] = {"input_profile": input_profile}
    if max_views is not None:
        model["max_views"] = max_views
    if max_seq_len is not None:
        model["max_seq_len"] = max_seq_len
    return {
        "model": model,
        "court_keypoints": {"selector": selector},
        "data": {"scene_dir": scene_dir},
    }


def _write_sidecar(checkpoint: Path, config: dict[str, object]) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"")
    checkpoints_dir = checkpoint.parent
    log_dir = checkpoints_dir.parent
    (log_dir / "hparams.yaml").write_text(
        yaml.safe_dump({"config": config}), encoding="utf-8"
    )


def test_allowed_families_follow_selector_and_object_count() -> None:
    assert allowed_scene_families("plcs_multiview_axial_split", "physical_v1") == (
        "single_object",
        "single_object_broadcast",
    )
    assert allowed_scene_families("plcs_multiview_axial_split", "camera_view_v2") == (
        "single_object_camera_view_v2",
    )
    assert allowed_scene_families("plcs_track_query", "physical_v1") == (
        "multi_object",
        "multi_object_broadcast",
    )
    assert allowed_scene_families("plcs_track_query_reference", "camera_view_v2") == (
        "multi_object_camera_view_v2",
    )
    assert allowed_scene_families("unknown_model", "physical_v1") == ()


def test_model_classification_helpers() -> None:
    assert objectness_for_model("plcs_track_query") == OBJECTNESS_MULTI
    assert objectness_for_model("plcs_multiview_axial") == OBJECTNESS_SINGLE
    assert objectness_for_model("nope") is None
    assert is_reference_model("plcs_multiview_axial_reference")
    assert not is_reference_model("plcs_multiview_axial_split")
    assert scene_family_of("plcs/single_object") == "single_object"
    assert scene_family_of(None) is None


def test_sidecar_metadata_narrows_families_without_reading_weights(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run_a" / "logs" / "version_0" / "checkpoints" / "best.ckpt"
    _write_sidecar(
        checkpoint,
        _config(
            model_name="plcs_multiview_axial_split",
            selector="camera_view_v2",
            input_profile="multiview",
            scene_dir="plcs/single_object_camera_view_v2",
            max_views=4,
            max_seq_len=256,
        ),
    )
    info = describe_checkpoint(tmp_path, "run_a/logs/version_0/checkpoints/best.ckpt")
    assert info.supported
    assert info.metadata_source == "hparams_yaml"
    assert info.model_name == "plcs_multiview_axial_split"
    assert info.selector == "camera_view_v2"
    assert info.objects == OBJECTNESS_SINGLE
    assert not info.reference
    assert info.max_views == 4
    assert info.max_seq_len == 256
    assert info.families == ("single_object_camera_view_v2",)
    assert info.trained_scene_dir == "plcs/single_object_camera_view_v2"
    assert info.id == "run_a/logs/version_0/checkpoints/best.ckpt"


def test_trained_family_is_preferred_first(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run_b" / "logs" / "version_0" / "checkpoints" / "b.ckpt"
    _write_sidecar(
        checkpoint,
        _config(
            model_name="plcs_multiview_axial",
            selector="physical_v1",
            input_profile="multiview",
            scene_dir="plcs/single_object_broadcast",
        ),
    )
    info = describe_checkpoint(tmp_path, "run_b/logs/version_0/checkpoints/b.ckpt")
    assert info.families == ("single_object_broadcast", "single_object")


def test_track_query_checkpoint_is_supported_as_multi_object(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run_c" / "logs" / "version_0" / "checkpoints" / "c.ckpt"
    _write_sidecar(
        checkpoint,
        _config(
            model_name="plcs_track_query",
            selector="physical_v1",
            input_profile=None,
            scene_dir="plcs/multi_object",
        ),
    )
    info = describe_checkpoint(tmp_path, "run_c/logs/version_0/checkpoints/c.ckpt")
    assert info.supported
    assert info.unsupported_reason is None
    assert info.input_profile == "track_query"
    assert info.objects == OBJECTNESS_MULTI
    assert info.families == ("multi_object", "multi_object_broadcast")


def test_archive_fallback_reads_saved_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run_d" / "checkpoints" / "d.ckpt"
    checkpoint.parent.mkdir(parents=True)
    config = _config(
        model_name="plcs_multiview_axial_reference",
        selector="camera_view_v2",
        input_profile="multiview",
        scene_dir="plcs/single_object_camera_view_v2",
        max_views=4,
        max_seq_len=256,
    )
    torch.save(
        {
            "hyper_parameters": {"config": config},
            "court_keypoints": {"selector": "camera_view_v2"},
            "state_dict": {"w": torch.zeros(2)},
        },
        checkpoint,
    )
    info = describe_checkpoint(tmp_path, "run_d/checkpoints/d.ckpt")
    assert info.metadata_source == "checkpoint"
    assert info.supported
    assert info.reference
    assert info.model_name == "plcs_multiview_axial_reference"
    assert info.families == ("single_object_camera_view_v2",)


def test_reference_model_with_physical_selector_is_unsupported(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run_e" / "logs" / "version_0" / "checkpoints" / "e.ckpt"
    _write_sidecar(
        checkpoint,
        _config(
            model_name="plcs_multiview_axial_reference",
            selector="physical_v1",
            input_profile="multiview",
            scene_dir="plcs/single_object",
        ),
    )
    info = describe_checkpoint(tmp_path, "run_e/logs/version_0/checkpoints/e.ckpt")
    assert not info.supported
    assert info.unsupported_reason is not None
    assert "camera_view_v2" in info.unsupported_reason


def test_scan_returns_every_checkpoint_sorted(tmp_path: Path) -> None:
    for name in ("zeta", "alpha"):
        _write_sidecar(
            tmp_path / name / "logs" / "version_0" / "checkpoints" / "best.ckpt",
            _config(
                model_name="plcs_multiview_axial",
                selector="physical_v1",
                input_profile="multiview",
                scene_dir="plcs/single_object",
            ),
        )
    items = scan_checkpoints(tmp_path)
    assert [item.relative for item in items] == [
        "alpha/logs/version_0/checkpoints/best.ckpt",
        "zeta/logs/version_0/checkpoints/best.ckpt",
    ]
    assert all(item.supported for item in items)


def test_resolve_checkpoint_path_rejects_non_checkpoints(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_path(tmp_path, "missing.ckpt")
    plain = tmp_path / "not_a_checkpoint.txt"
    plain.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="must end with"):
        resolve_checkpoint_path(tmp_path, plain)


def test_checkpoint_payload_is_json_serializable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run_f" / "logs" / "version_0" / "checkpoints" / "f.ckpt"
    _write_sidecar(
        checkpoint,
        _config(
            model_name="plcs",
            selector="physical_v1",
            input_profile="frame",
            scene_dir="plcs/single_object",
        ),
    )
    info = describe_checkpoint(tmp_path, "run_f/logs/version_0/checkpoints/f.ckpt")
    document = json.loads(json.dumps(info.to_dict()))
    assert document["model_name"] == "plcs"
    assert document["families"] == ["single_object", "single_object_broadcast"]


def test_checkpoint_body_is_canonical_over_a_stale_sidecar(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run_g" / "checkpoints" / "g.ckpt"
    checkpoint.parent.mkdir(parents=True)
    body = _config(
        model_name="plcs_multiview_axial",
        selector="physical_v1",
        input_profile="multiview",
        scene_dir="plcs/single_object",
    )
    torch.save(
        {"hyper_parameters": {"config": body}, "state_dict": {"w": torch.zeros(1)}},
        checkpoint,
    )
    # A stale neighbour that disagrees on the model name must not be trusted.
    (checkpoint.parent / "hparams.yaml").write_text(
        yaml.safe_dump(
            {
                "config": _config(
                    model_name="plcs",
                    selector="physical_v1",
                    input_profile="frame",
                    scene_dir="plcs/single_object",
                )
            }
        ),
        encoding="utf-8",
    )
    info = describe_checkpoint(tmp_path, "run_g/checkpoints/g.ckpt")
    assert not info.supported
    assert info.unsupported_reason is not None
    assert "一致しません" in info.unsupported_reason


def test_matching_sidecar_keeps_checkpoint_as_the_metadata_source(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run_h" / "checkpoints" / "h.ckpt"
    checkpoint.parent.mkdir(parents=True)
    config = _config(
        model_name="plcs_multiview_axial_split",
        selector="camera_view_v2",
        input_profile="multiview",
        scene_dir="plcs/single_object_camera_view_v2",
        max_views=4,
        max_seq_len=256,
    )
    torch.save(
        {"hyper_parameters": {"config": config}, "state_dict": {"w": torch.zeros(1)}},
        checkpoint,
    )
    (checkpoint.parent / "hparams.yaml").write_text(
        yaml.safe_dump({"config": config}), encoding="utf-8"
    )
    info = describe_checkpoint(tmp_path, "run_h/checkpoints/h.ckpt")
    assert info.supported
    assert info.metadata_source == "checkpoint"


def test_extra_checkpoint_root_is_prefixed_and_describable(tmp_path: Path) -> None:
    primary = tmp_path / "outputs" / "plcs"
    primary.mkdir(parents=True)
    extra = tmp_path / "ckpt" / "plcs"
    _write_sidecar(
        extra / "axial" / "logs" / "version_0" / "checkpoints" / "best.ckpt",
        _config(
            model_name="plcs_multiview_axial",
            selector="physical_v1",
            input_profile="multiview",
            scene_dir="plcs/single_object",
        ),
    )
    items = scan_checkpoints(primary, extra_roots=[extra])
    assert [item.id for item in items] == [
        "ckpt/plcs/axial/logs/version_0/checkpoints/best.ckpt"
    ]
    info = describe_checkpoint(
        primary,
        "ckpt/plcs/axial/logs/version_0/checkpoints/best.ckpt",
        extra_roots=[extra],
    )
    assert info.supported
    assert info.families == ("single_object", "single_object_broadcast")
    owning_root, path = resolve_checkpoint(
        primary,
        "ckpt/plcs/axial/logs/version_0/checkpoints/best.ckpt",
        extra_roots=[extra],
    )
    assert owning_root == extra.resolve()
    assert (
        path
        == (
            extra / "axial" / "logs" / "version_0" / "checkpoints" / "best.ckpt"
        ).resolve()
    )


def test_load_checkpoint_config_rejects_a_contradicted_sidecar(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run_i" / "checkpoints" / "i.ckpt"
    checkpoint.parent.mkdir(parents=True)
    body = _config(
        model_name="plcs_track_query",
        selector="physical_v1",
        input_profile=None,
        scene_dir="plcs/multi_object",
    )
    torch.save(
        {"hyper_parameters": {"config": body}, "state_dict": {"w": torch.zeros(1)}},
        checkpoint,
    )
    (checkpoint.parent / "hparams.yaml").write_text(
        yaml.safe_dump(
            {
                "config": _config(
                    model_name="plcs_track_query_reference",
                    selector="camera_view_v2",
                    input_profile=None,
                    scene_dir="plcs/multi_object_camera_view_v2",
                )
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="一致しません"):
        load_checkpoint_config(tmp_path, "run_i/checkpoints/i.ckpt")
