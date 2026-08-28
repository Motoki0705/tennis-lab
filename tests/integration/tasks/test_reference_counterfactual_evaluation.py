"""Task-adapter integration for strict paired reference evaluation payloads."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tasks.base.evaluation import (
    ReferenceCounterfactualError,
    ReferenceCounterfactualManifest,
    ReferenceCounterfactualReportPaths,
    ReferenceCounterfactualRunIdentity,
    build_reference_counterfactual_manifest_from_documents,
    evaluate_reference_counterfactual,
    read_reference_counterfactual_report,
    write_reference_counterfactual_report,
)
from src.tasks.base.generate_dataset import (
    CAMERA_VIEW_V2_SELECTOR,
    CourtKeypointArtifactMetadata,
    build_court_view_record,
    court_headings_physical_to_target,
    court_points_physical_to_target,
    court_world_joints_physical_to_target,
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.evaluation import (
    build_blcs_counterfactual_pass,
    run_blcs_reference_counterfactual,
)
from src.tasks.blcs.model_io.training import compose_blcs_training
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)
from src.tasks.plcs.evaluation import (
    build_plcs_counterfactual_pass,
    run_plcs_reference_counterfactual,
)
from src.tasks.plcs.training.composition import build_plcs_lightning_module
from src.tasks.plcs.training.tracking_lightning_module import (
    PLCSTrackingLightningModule,
)
from src.utils.configuration import PathContractError, SemanticConfigurationError
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court_normalization import (
    normalize_court_position,
    normalize_court_velocity,
)

_SCHEMA = "synthetic_task_adapter_v2"
_Runner = Callable[[DictConfig], ReferenceCounterfactualReportPaths]


def _evaluation_config(task: str, checkpoint: Path) -> DictConfig:
    config_dir = PROJECT_ROOT / "src" / "tasks" / task / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(
            config_name="evaluate_reference_counterfactual",
            overrides=[
                f"paths.checkpoint_root={checkpoint.parent}",
                f"evaluation.checkpoint_path={checkpoint}",
            ],
        )


def _manifest() -> ReferenceCounterfactualManifest:
    contract = resolve_court_keypoint_contract(CAMERA_VIEW_V2_SELECTOR)
    views = tuple(
        build_court_view_record(
            camera_id=f"camera_{index}",
            camera_center_court_m=(
                float(index),
                -4.0 if index < 3 else 4.0,
                2.0,
            ),
            contract=contract,
        )
        for index in range(6)
    )
    metadata = CourtKeypointArtifactMetadata.from_contract(
        contract,
        dataset_schema_id=_SCHEMA,
    ).to_dict()
    return build_reference_counterfactual_manifest_from_documents(
        root_metadata={
            "court_keypoints": metadata,
            "scenes": [{"file": "scene_a", "scene_id": "scene_a", "num_cameras": 6}],
        },
        scene_metadata={
            "scene_a": {
                "scene_id": "scene_a",
                "num_cameras": 6,
                "court_keypoints": metadata,
                "court_keypoint_views": [view.to_dict() for view in views],
            }
        },
        expected_dataset_schema_id=_SCHEMA,
    )


def _two_scene_manifest() -> ReferenceCounterfactualManifest:
    single = _manifest()
    return replace(
        single,
        scenes=(single.scenes[0], replace(single.scenes[0], scene_id="scene_b")),
    )


def _identity(
    manifest: ReferenceCounterfactualManifest,
    task: str,
) -> ReferenceCounterfactualRunIdentity:
    return ReferenceCounterfactualRunIdentity.create(
        task=cast("Any", task),
        seed=42,
        selector_mode="reference",
        resolved_config={"task": task, "seed": 42, "views": 6, "length": 128},
        checkpoint_sha256="a" * 64,
        manifest_digest=manifest.digest,
        court_keypoint_contract="camera_view_courtkp20_rzpi_v1",
        target_frame_contract="reference_camera_court_rzpi_v1",
        track_query_rope_contract="track_query_reference_selector_v2",
    )


def _metadata(
    manifest: ReferenceCounterfactualManifest,
    side: str,
) -> dict[str, np.ndarray[Any, Any]]:
    scene = manifest.scenes[0]
    selection = scene.selection(cast("Any", side))
    ordering = scene.local_ordering
    code_by_id = {camera_id: index for index, camera_id in enumerate(sorted(ordering))}
    view_codes = np.asarray([[code_by_id[camera_id] for camera_id in ordering]])
    return {
        "scene_ids": np.asarray([scene.scene_id]),
        "view_camera_id_strings": np.asarray([ordering]),
        "reference_camera_id_string": np.asarray([selection.camera_id]),
        "reference_view_index": np.asarray([selection.local_index], dtype=np.int64),
        "reference_camera_id": np.asarray(
            [code_by_id[selection.camera_id]], dtype=np.int64
        ),
        "view_camera_ids": view_codes,
        "reference_from_physical": np.asarray(
            [selection.provenance.reference_from_physical], dtype=np.float32
        ),
        "physical_from_reference": np.asarray(
            [selection.provenance.physical_from_reference], dtype=np.float32
        ),
        "court_keypoint_contract": np.asarray(["camera_view_courtkp20_rzpi_v1"]),
        "target_frame_contract": np.asarray(["reference_camera_court_rzpi_v1"]),
        "track_query_rope_contract": np.asarray(["track_query_reference_selector_v2"]),
        "reference_selector_mode": np.asarray(["reference"]),
    }


def _blcs_payload(
    manifest: ReferenceCounterfactualManifest,
    side: str,
) -> dict[str, np.ndarray[Any, Any]]:
    selection = manifest.scenes[0].selection(cast("Any", side))
    physical = np.asarray([[[[1.0, 2.0, 0.5]], [[-1.0, -2.0, 1.0]]]])
    target = np.asarray(court_points_physical_to_target(physical, selection.provenance))
    target_norm = np.asarray(normalize_court_position(target), dtype=np.float32)
    velocity_physical = np.asarray([[[[0.2, 0.4, 0.1]], [[0.1, -0.2, 0.0]]]])
    velocity = np.asarray(
        court_points_physical_to_target(velocity_physical, selection.provenance)
    )
    velocity_norm = np.asarray(normalize_court_velocity(velocity), dtype=np.float32)
    uv: np.ndarray[Any, Any] = np.arange(24, dtype=np.float32).reshape(1, 6, 2, 1, 2)
    visible: np.ndarray[Any, Any] = np.ones((1, 6, 2, 1), dtype=np.bool_)
    payload = {
        **_metadata(manifest, side),
        "pred_position": target_norm.copy(),
        "pred_presence_logits": np.zeros((1, 2, 1), dtype=np.float32),
        "ball_uv": uv,
        "ball_vis": visible,
        "court_kp": np.zeros((1, 6, 2, 14, 2), dtype=np.float32),
        "court_vis": np.ones((1, 6, 2, 14), dtype=np.bool_),
        "padding_mask": np.zeros((1, 6, 2), dtype=np.bool_),
        "target_position": target_norm,
        "target_velocity": velocity_norm,
        "target_presence": np.ones((1, 2, 1), dtype=np.bool_),
        "target_instance_id": np.zeros((1, 2, 1), dtype=np.int64),
        "target_slot_mask": np.ones((1, 1), dtype=np.bool_),
        "frame_valid": np.ones((1, 2), dtype=np.bool_),
        "clean_ball_uv": uv.copy(),
        "clean_ball_vis": visible.copy(),
        "candidate_gt_index": np.zeros((1, 6, 2, 1), dtype=np.int64),
    }
    return payload


def _plcs_payload(
    manifest: ReferenceCounterfactualManifest,
    side: str,
) -> dict[str, np.ndarray[Any, Any]]:
    selection = manifest.scenes[0].selection(cast("Any", side))
    physical = np.asarray([[[[1.0, 2.0, 0.5]], [[-1.0, -2.0, 1.0]]]])
    target = np.asarray(court_points_physical_to_target(physical, selection.provenance))
    target_norm = np.asarray(normalize_court_position(target), dtype=np.float32)
    physical_heading = np.asarray([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    heading = np.asarray(
        court_headings_physical_to_target(physical_heading, selection.provenance),
        dtype=np.float32,
    )
    physical_world = np.stack((physical, physical + 0.25), axis=-2)
    world = np.asarray(
        court_world_joints_physical_to_target(physical_world, selection.provenance),
        dtype=np.float32,
    )
    human: np.ndarray[Any, Any] = np.arange(408, dtype=np.float32).reshape(
        1, 6, 2, 1, 17, 2
    )
    visible: np.ndarray[Any, Any] = np.ones((1, 6, 2, 1, 17), dtype=np.bool_)
    return {
        **_metadata(manifest, side),
        "pred_position": target_norm.copy(),
        "pred_rotation": heading.copy(),
        "pred_presence_logits": np.zeros((1, 2, 1), dtype=np.float32),
        "human_kp": human,
        "human_vis": visible,
        "court_kp": np.zeros((1, 6, 2, 14, 2), dtype=np.float32),
        "court_vis": np.ones((1, 6, 2, 14), dtype=np.bool_),
        "target_position": target_norm,
        "target_rotation": heading,
        "target_canonical_pose_3d": np.zeros((1, 2, 1, 2, 3), dtype=np.float32),
        "target_human_kp_3d": world,
        "target_presence": np.ones((1, 2, 1), dtype=np.bool_),
        "target_instance_id": np.zeros((1, 2, 1), dtype=np.int64),
        "padding_mask": np.zeros((1, 6, 2), dtype=np.bool_),
        "clean_human_kp": human.copy(),
        "clean_human_vis": visible.copy(),
        "detection_gt_index": np.zeros((1, 6, 2, 1), dtype=np.int64),
    }


def _write(path: Path, payload: dict[str, np.ndarray[Any, Any]]) -> None:
    cast("Any", np.savez_compressed)(path, **payload)


def _two_scene_payload(
    manifest: ReferenceCounterfactualManifest,
    side: str,
    task: str,
) -> dict[str, np.ndarray[Any, Any]]:
    single = (
        _blcs_payload(manifest, side)
        if task == "blcs"
        else _plcs_payload(manifest, side)
    )
    payload = {
        key: np.concatenate((value, value.copy()), axis=0)
        for key, value in single.items()
    }
    payload["scene_ids"] = np.asarray(["scene_a", "scene_b"])
    payload["pred_position"][1] += np.float32(0.25)
    if task == "plcs":
        payload["pred_rotation"][1] *= np.float32(2.0)
    return payload


@pytest.mark.parametrize("task", ["blcs", "plcs"])
def test_task_adapters_build_and_join_reference_only_passes(
    tmp_path: Path,
    task: str,
) -> None:
    manifest = _manifest()
    identity = _identity(manifest, task)
    passes = []
    for side in ("same_side", "opposite_side"):
        path = tmp_path / f"{task}_{side}.npz"
        payload = (
            _blcs_payload(manifest, side)
            if task == "blcs"
            else _plcs_payload(manifest, side)
        )
        _write(path, payload)
        adapter = (
            build_blcs_counterfactual_pass
            if task == "blcs"
            else build_plcs_counterfactual_pass
        )
        passes.append(
            adapter(
                path,
                side=cast("Any", side),
                identity=identity,
                manifest=manifest,
                window_bounds={"scene_a": (5, 7)},
            )
        )
    report = evaluate_reference_counterfactual(manifest, passes[0], passes[1])
    repro_dir = tmp_path / f"{task}_queue_repro"
    bundle_dir = repro_dir / "predictions"
    paths = write_reference_counterfactual_report(report, bundle_dir)
    recomputed = read_reference_counterfactual_report(bundle_dir)
    assert report.metrics.same_side.position.axis_wise_position_error.to_dict() == {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
    }
    assert report.metrics.physical_consistency.position_error_m == 0.0
    assert recomputed.metrics.to_dict() == report.metrics.to_dict()
    assert paths.npz_path == bundle_dir / "pred_test.npz"
    assert paths.metrics_path == bundle_dir / "metrics.json"
    assert paths.json_path == bundle_dir / "reference_counterfactual.json"
    flat_metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
    # This is the exact conversion performed by kg_register.load_metrics().
    registerable_metrics = {
        key: round(float(value), 6) for key, value in flat_metrics.items()
    }
    assert registerable_metrics
    assert set(registerable_metrics) == set(report.metrics.flat_dict())

    (repro_dir / "run.json").write_text(
        json.dumps(
            {
                "name": f"i801_{task}_counterfactual_test",
                "provider": "codex",
                "issue": "801",
                "command": "python -m evaluator",
            }
        ),
        encoding="utf-8",
    )
    knowledge_dir = tmp_path / f"{task}_knowledge"
    node_id = f"run-i801-{task}-counterfactual-test"
    completed = subprocess.run(
        [
            sys.executable,
            ".agents/skills/knowledge-control/scripts/kg_register.py",
            "--repro-dir",
            str(repro_dir),
            "--id",
            node_id,
            "--issue",
            "801",
            "--provider",
            "codex",
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "KNOWLEDGE_DIR": str(knowledge_dir)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "metrics:" in completed.stdout
    promoted = knowledge_dir / "runs" / node_id
    assert (promoted / "pred_test.npz").read_bytes() == paths.npz_path.read_bytes()
    assert (promoted / "metrics.json").read_bytes() == paths.metrics_path.read_bytes()
    node_text = (knowledge_dir / "nodes" / f"{node_id}.md").read_text(
        encoding="utf-8"
    )
    assert "reference_target_same_side_y_sign_accuracy" in node_text
    if task == "plcs":
        assert report.metrics.same_side.heading_error_deg == 0.0


@pytest.mark.parametrize("task", ["blcs", "plcs"])
def test_task_adapter_raw_scene_permutation_is_canonicalized_with_arrays(
    tmp_path: Path,
    task: str,
) -> None:
    manifest = _two_scene_manifest()
    identity = _identity(manifest, task)
    adapter = (
        build_blcs_counterfactual_pass
        if task == "blcs"
        else build_plcs_counterfactual_pass
    )
    canonical_passes = []
    permuted_passes = []
    for side in ("same_side", "opposite_side"):
        payload = _two_scene_payload(manifest, side, task)
        canonical_path = tmp_path / f"{task}_{side}_canonical.npz"
        permuted_path = tmp_path / f"{task}_{side}_permuted.npz"
        _write(canonical_path, payload)
        _write(
            permuted_path,
            {key: value[np.asarray([1, 0])] for key, value in payload.items()},
        )
        arguments = {
            "side": cast("Any", side),
            "identity": identity,
            "manifest": manifest,
            "window_bounds": {"scene_a": (5, 7), "scene_b": (11, 13)},
        }
        canonical_passes.append(adapter(canonical_path, **cast("Any", arguments)))
        permuted_passes.append(adapter(permuted_path, **cast("Any", arguments)))

    expected = evaluate_reference_counterfactual(
        manifest,
        canonical_passes[0],
        canonical_passes[1],
    )
    report = evaluate_reference_counterfactual(
        manifest,
        permuted_passes[0],
        permuted_passes[1],
    )

    assert report.report_digest == expected.report_digest
    assert [row.key.scene_id for row in report.same_side_pass.rows] == [
        "scene_a",
        "scene_b",
    ]
    assert np.array_equal(
        report.same_side_pass.position.prediction,
        expected.same_side_pass.position.prediction,
    )


def test_task_adapter_digest_makes_changed_uv_fail_strict_join(tmp_path: Path) -> None:
    manifest = _manifest()
    identity = _identity(manifest, "blcs")
    built = []
    for side in ("same_side", "opposite_side"):
        payload = _blcs_payload(manifest, side)
        if side == "opposite_side":
            payload["ball_uv"][0, 0, 0, 0, 0] += 1.0
        path = tmp_path / f"{side}.npz"
        _write(path, payload)
        built.append(
            build_blcs_counterfactual_pass(
                path,
                side=cast("Any", side),
                identity=identity,
                manifest=manifest,
                window_bounds={"scene_a": (5, 7)},
            )
        )
    with pytest.raises(ReferenceCounterfactualError, match="observation_digest"):
        evaluate_reference_counterfactual(manifest, built[0], built[1])


def test_plcs_inactive_heading_fill_is_excluded_from_physical_target_parity(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    identity = _identity(manifest, "plcs")
    passes = []
    for side in ("same_side", "opposite_side"):
        payload = _plcs_payload(manifest, side)
        payload["target_presence"][:, 1] = False
        # Inactive headings use the target-frame identity fill. Restoring that
        # fill to physical legitimately differs by side and is unsupervised.
        payload["target_rotation"][:, 1] = np.asarray([1.0, 0.0])
        payload["pred_rotation"][:, 1] = np.asarray([1.0, 0.0])
        path = tmp_path / f"plcs_inactive_{side}.npz"
        _write(path, payload)
        passes.append(
            build_plcs_counterfactual_pass(
                path,
                side=cast("Any", side),
                identity=identity,
                manifest=manifest,
                window_bounds={"scene_a": (5, 7)},
            )
        )

    report = evaluate_reference_counterfactual(manifest, passes[0], passes[1])

    assert report.metrics.same_side.heading_error_deg == 0.0
    assert report.metrics.opposite_side.heading_error_deg == 0.0


def test_task_adapter_rejects_metadata_free_raw_payload(tmp_path: Path) -> None:
    manifest = _manifest()
    payload = _blcs_payload(manifest, "same_side")
    del payload["track_query_rope_contract"]
    path = tmp_path / "missing_contract.npz"
    _write(path, payload)
    with pytest.raises(ReferenceCounterfactualError, match="track_query_rope_contract"):
        build_blcs_counterfactual_pass(
            path,
            side="same_side",
            identity=_identity(manifest, "blcs"),
            manifest=manifest,
            window_bounds={"scene_a": (5, 7)},
        )


@pytest.mark.parametrize("task", ["blcs", "plcs"])
@pytest.mark.parametrize(
    ("field", "mutation", "error_pattern"),
    [
        (
            "reference_view_index",
            lambda value: value.reshape(1, 1),
            "reference_view_index must have exact shape",
        ),
        (
            "reference_camera_id",
            lambda value: value.astype(np.int32),
            "reference_camera_id must use exact int64",
        ),
        (
            "court_kp",
            lambda value: np.full_like(value, np.inf),
            "cannot contain non-finite",
        ),
        (
            "view_camera_ids",
            lambda value: np.roll(value, 1, axis=1),
            "complete-scene lexicographic ranks",
        ),
    ],
)
def test_task_adapter_rejects_malformed_or_nonfinite_raw_queue_payload(
    tmp_path: Path,
    task: str,
    field: str,
    mutation: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
    error_pattern: str,
) -> None:
    manifest = _manifest()
    payload = (
        _blcs_payload(manifest, "same_side")
        if task == "blcs"
        else _plcs_payload(manifest, "same_side")
    )
    payload[field] = mutation(payload[field])
    path = tmp_path / f"{task}_{field}.npz"
    _write(path, payload)
    adapter = (
        build_blcs_counterfactual_pass
        if task == "blcs"
        else build_plcs_counterfactual_pass
    )

    with pytest.raises(ReferenceCounterfactualError, match=error_pattern):
        adapter(
            path,
            side="same_side",
            identity=_identity(manifest, task),
            manifest=manifest,
            window_bounds={"scene_a": (5, 7)},
        )


@pytest.mark.parametrize("task", ["blcs", "plcs"])
def test_minimal_reference_v2_checkpoint_boundary_rejects_metadata_free_v1(
    task: str,
) -> None:
    config_dir = PROJECT_ROOT / "src" / "tasks" / task / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                "court_keypoints=camera_view_v2",
                "model=track_query_ablation_d_v2_selector",
                "model.cswa.backend=reference",
            ],
        )
    if task == "blcs":
        module = cast(
            BLCSTrackingLightningModule,
            compose_blcs_training(config, generator_config=None).lightning_module,
        )
    else:
        module = cast(
            PLCSTrackingLightningModule,
            build_plcs_lightning_module(config),
        )
    metadata_free_checkpoint: dict[str, Any] = {"state_dict": module.state_dict()}
    with pytest.raises(ValueError):
        module.on_load_checkpoint(metadata_free_checkpoint)

    strict_checkpoint: dict[str, Any] = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(strict_checkpoint)
    module.on_load_checkpoint(strict_checkpoint)


@pytest.mark.parametrize(
    ("task", "runner"),
    [
        ("blcs", run_blcs_reference_counterfactual),
        ("plcs", run_plcs_reference_counterfactual),
    ],
)
def test_counterfactual_evaluator_validates_checkpoint_before_writing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    runner: _Runner,
) -> None:
    repro_dir = tmp_path / task
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro_dir))
    checkpoint = tmp_path / f"{task}.ckpt"
    torch.save({"state_dict": {}}, checkpoint)

    with pytest.raises(ValueError, match="court_coordinate_normalization"):
        runner(_evaluation_config(task, checkpoint))

    assert not (repro_dir / "predictions").exists()


def test_blcs_evaluator_passes_typed_court_contract_to_checkpoint_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repro_dir = tmp_path / "blcs_typed_contract"
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro_dir))
    checkpoint = tmp_path / "blcs.ckpt"
    checkpoint.touch()
    captured: dict[str, object] = {}

    class _BoundaryReached(RuntimeError):
        pass

    def _capture_checkpoint_boundary(
        path: Path,
        runtime_court_keypoints: object,
        runtime_track_query_reference: object,
    ) -> None:
        captured.update(
            {
                "path": path,
                "court": runtime_court_keypoints,
                "reference": runtime_track_query_reference,
            }
        )
        raise _BoundaryReached

    monkeypatch.setattr(
        "src.tasks.blcs.evaluation.reference_counterfactual.validate_checkpoint_path",
        _capture_checkpoint_boundary,
    )

    with pytest.raises(_BoundaryReached):
        run_blcs_reference_counterfactual(_evaluation_config("blcs", checkpoint))

    court = cast("Any", captured["court"])
    assert court.selector == "camera_view_v2"
    assert court.contract_id == "camera_view_courtkp20_rzpi_v1"
    assert captured["path"] == checkpoint
    assert not (repro_dir / "predictions").exists()


@pytest.mark.parametrize(
    ("task", "runner"),
    [
        ("blcs", run_blcs_reference_counterfactual),
        ("plcs", run_plcs_reference_counterfactual),
    ],
)
def test_counterfactual_evaluator_rejects_any_existing_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    runner: _Runner,
) -> None:
    repro_dir = tmp_path / task
    output_dir = repro_dir / "predictions"
    output_dir.mkdir(parents=True)
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro_dir))
    checkpoint = tmp_path / f"{task}.ckpt"
    torch.save({"state_dict": {}}, checkpoint)

    with pytest.raises(ReferenceCounterfactualError, match="refuses overwrite"):
        runner(_evaluation_config(task, checkpoint))


@pytest.mark.parametrize(
    ("task", "runner"),
    [
        ("blcs", run_blcs_reference_counterfactual),
        ("plcs", run_plcs_reference_counterfactual),
    ],
)
def test_counterfactual_evaluator_rejects_checkpoint_outside_declared_role_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    runner: _Runner,
) -> None:
    repro_dir = tmp_path / f"{task}_repro"
    checkpoint_root = tmp_path / f"{task}_declared_checkpoints"
    checkpoint_root.mkdir()
    checkpoint = tmp_path / f"{task}_outside.ckpt"
    checkpoint.touch()
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro_dir))
    config_dir = PROJECT_ROOT / "src" / "tasks" / task / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="evaluate_reference_counterfactual",
            overrides=[
                f"paths.checkpoint_root={checkpoint_root}",
                f"evaluation.checkpoint_path={checkpoint}",
            ],
        )

    with pytest.raises(PathContractError, match="outside its root"):
        runner(config)

    assert not (repro_dir / "predictions").exists()


@pytest.mark.parametrize(
    ("task", "runner"),
    [
        ("blcs", run_blcs_reference_counterfactual),
        ("plcs", run_plcs_reference_counterfactual),
    ],
)
def test_counterfactual_evaluator_requires_dedicated_predictions_output_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    runner: _Runner,
) -> None:
    repro_dir = tmp_path / f"{task}_repro"
    checkpoint = tmp_path / f"{task}.ckpt"
    checkpoint.touch()
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro_dir))
    config_dir = PROJECT_ROOT / "src" / "tasks" / task / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="evaluate_reference_counterfactual",
            overrides=[
                f"paths.checkpoint_root={checkpoint.parent}",
                f"evaluation.checkpoint_path={checkpoint}",
                f"evaluation.output_dir={repro_dir / 'adjacent'}",
            ],
        )

    with pytest.raises(
        SemanticConfigurationError,
        match=r"exactly \$TENNIS_REPRO_DIR/predictions",
    ):
        runner(config)

    assert not repro_dir.exists()
