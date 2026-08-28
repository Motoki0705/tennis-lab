"""Durable Attempt-2 learned evidence for Issue #801.

The checks in this module intentionally consume only versioned knowledge
bundles.  Queue logs, raw side-pass archives, datasets, and checkpoints are
not part of the evidence oracle.
"""

from __future__ import annotations

import json
import math
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import yaml

from src.tasks.base.evaluation import (
    compute_heading_error_radians,
    compute_paired_reference_position_metrics,
    file_sha256,
    read_reference_counterfactual_report,
)

_RUN_ROOT = Path("knowledge/runs")
_NODE_ROOT = Path("knowledge/nodes")
_GROUP_ID = "group-i801-reference-counterfactual-attempt2"
_TRAINING_RUNS = {
    "reference": "run-i801-a2-plcs-d-reference",
    "selector_zero": "run-i801-a2-plcs-d-selector-zero",
}
_PAIRED_RUNS = {
    "blcs": {
        "reference": "run-i801-a2-blcs-reference-paired",
        "selector_zero": "run-i801-a2-blcs-selector-zero-paired",
    },
    "plcs": {
        "reference": "run-i801-a2-plcs-reference-paired",
        "selector_zero": "run-i801-a2-plcs-selector-zero-paired",
    },
}
_FORMAL_RUN_IDS = frozenset(
    set(_TRAINING_RUNS.values())
    | {
        run_id
        for task_runs in _PAIRED_RUNS.values()
        for run_id in task_runs.values()
    }
)
_SELECTOR_PROFILE = {
    "reference": "track_query_ablation_d_v2_selector",
    "selector_zero": "track_query_ablation_d_v2_selector_zero",
}
_CONTRACTS = {
    "court_keypoint_contract": "camera_view_courtkp20_rzpi_v1",
    "target_frame_contract": "reference_camera_court_rzpi_v1",
    "track_query_rope_contract": "time_camera_reference_selector_v1",
}
_TRAINING_QUEUE_IDS = {
    "reference": "1787644646345366163_3910851_i801_attempt2_plcs_d_reference_t128_v6_eb32_e100_final",
    "selector_zero": "1787644646445719253_3910910_i801_attempt2_plcs_d_selector_zero_t128_v6_eb32_e100_final",
}
_TRAINING_PREDICTION_SHA256 = {
    "reference": "a178f5fb62549aed5d6094187cd80695bdc5eb29ff4e3c407a02ad2161b51544",
    "selector_zero": "aebb636f193b14b9ce0c1621d52b6e1c9e66b72f8d8ac0f085255d0d4c57cb97",
}
_PAIRED_EVIDENCE = {
    ("blcs", "reference"): {
        "queue_id": "1787643447709299464_3887229_i801_attempt2_blcs_reference_paired_v6_final",
        "commit": "40c21fbad59e040f52c040ae354637c5f3c8975a",
        "checkpoint": "51e49749b9389157c0e975729a329ca2aced003fe0fa4830daf5d1aa334569fc",
        "config": "a3cf9370c9e2c04aab22c31a93e3e3ba9faf9ae11e197e0f26e8d9446703fa21",
        "manifest": "61e7ce3b9aca1db325f6399aa1ac7c4f4d10e4caf70feb02d3b16e6ec11faf8d",
        "parity": "c489cbb4f294a2cfdcc1ebaeaec609e811e01c9b451b5faeb2524e1804468bf8",
        "arrays": "a635e7f888ef04c6349365c8afbd9719b6cda5085abd2b37fb61381f24601334",
        "report": "df5f558763f422e35ed63815a0c0f7e72708def179240a32da2229d551312363",
    },
    ("blcs", "selector_zero"): {
        "queue_id": "1787643617793137915_3889485_i801_attempt2_blcs_selector_zero_paired_v6_final",
        "commit": "40c21fbad59e040f52c040ae354637c5f3c8975a",
        "checkpoint": "bae09f3628fba881dd8fee11c2a8b82f9be988eb5858eb0636494fcd8e8af25b",
        "config": "d6abafc7e550697cde8eb9aee5989546718b28c712b37230b2fcdce0cb84532c",
        "manifest": "61e7ce3b9aca1db325f6399aa1ac7c4f4d10e4caf70feb02d3b16e6ec11faf8d",
        "parity": "c50b76728c44b970303c7178ff3fd4aef0aaad69c0be571afdf6d3facb0fd801",
        "arrays": "2c51ccc57ddf70d64a42a918b8207b46d69c366f4fa033511b3c8a6971abdab0",
        "report": "cab62459a0d671c60286770e8b10159d4913f9ea311ccfcb13d850edd05b443c",
    },
    ("plcs", "reference"): {
        "queue_id": "1787652981842495298_4030041_i801_attempt2_plcs_reference_paired_v6_final",
        "commit": "71426cb84519d4fa716ac9b221d90b00d26b4e63",
        "checkpoint": "7a79b61b39ccbf75dbfc1dfcbaac9c010e2315d2038a6e721092039de3d49779",
        "config": "a9796b17985daa972affa0dd6f2a3ed34e7a8271b186f7d8db9fc828962aceca",
        "manifest": "45698da77ae4e666c2976a062e4b0a58809e518a01d090e9de1f479d9a21cae7",
        "parity": "04fa5cafbe2d12099babfec7bbf5f83edfde8f52c209910b503e35b10a6334ab",
        "arrays": "5e2403496b64a86010363b820548553604f50f7476c9839a56dc3bd75923a4ac",
        "report": "a12d29a4d444b24161b27a3b26a266d4469e29001e4c257fdb05fca9a75c0f98",
    },
    ("plcs", "selector_zero"): {
        "queue_id": "1787652981925032454_4030076_i801_attempt2_plcs_selector_zero_paired_v6_final",
        "commit": "71426cb84519d4fa716ac9b221d90b00d26b4e63",
        "checkpoint": "dfc340f8b45ba10c716b474824f3be4a0f2cbfd6788d1fdb352a30bb7a77abd0",
        "config": "68cef94993c27f9a90ce7bf2901f7f35c1b002d890d36238ad1827ecbbf6589a",
        "manifest": "45698da77ae4e666c2976a062e4b0a58809e518a01d090e9de1f479d9a21cae7",
        "parity": "1b35805f20ff66a70d3f99eb22425fa29b4e9060f808c46a2987fc9578d2d1df",
        "arrays": "6a6d82cb35c4ff5a8e09848793b1ed5ccf2231be28019ce1413cd2d9dda97bd2",
        "report": "a4206c755c657ac28aa0e2aea6726789613d7db4765a7748796d7aba93c3bdb1",
    },
}
_TRAINING_FILES = {
    "curves.png",
    "git_status.txt",
    "metrics.json",
    "output_dir.txt",
    "pred_test.npz",
    "repro.sh",
    "run.json",
    "uncommitted.patch",
}
_PAIRED_FILES = {
    "git_status.txt",
    "metrics.json",
    "pred_test.npz",
    "reference_counterfactual.json",
    "repro.sh",
    "run.json",
    "uncommitted.patch",
}


def _frontmatter(node_id: str) -> dict[str, Any]:
    text = (_NODE_ROOT / f"{node_id}.md").read_text(encoding="utf-8")
    parts = text.split("---", 2)
    assert len(parts) == 3 and not parts[0]
    value = yaml.safe_load(parts[1])
    assert isinstance(value, dict)
    return cast("dict[str, Any]", value)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast("dict[str, Any]", value)


def _npz(run_id: str) -> dict[str, np.ndarray[Any, Any]]:
    with np.load(_RUN_ROOT / run_id / "pred_test.npz", allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _normalize_command(command: str, *, training: bool) -> list[str]:
    replacements = (
        ("model=", "model=<MATCHED_VARIANT>"),
        (
            "run.output_dir=" if training else "evaluation.checkpoint_path=",
            "run.output_dir=<MATCHED_VARIANT>"
            if training
            else "evaluation.checkpoint_path=<MATCHED_CHECKPOINT>",
        ),
    )
    normalized: list[str] = []
    for token in shlex.split(command):
        replacement = next(
            (value for prefix, value in replacements if token.startswith(prefix)),
            None,
        )
        normalized.append(token if replacement is None else replacement)
    return normalized


def _normalized_report_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    model = cast("dict[str, Any]", normalized["model"])
    evaluation = cast("dict[str, Any]", normalized["evaluation"])
    model["reference_selector_mode"] = "<MATCHED_VARIANT>"
    evaluation["checkpoint_path"] = "<MATCHED_CHECKPOINT>"
    evaluation["output_dir"] = "<MATCHED_OUTPUT>"
    return normalized


def _assert_node_metrics(
    node: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    node_metrics = cast("dict[str, Any]", node["metrics"])
    assert set(node_metrics) == set(metrics)
    for name, value in metrics.items():
        assert isinstance(value, (int, float)) and math.isfinite(float(value))
        assert float(node_metrics[name]) == pytest.approx(
            round(float(value), 6), abs=5e-7
        )


def test_attempt2_group_freezes_only_the_six_formal_runs_and_excludes_retry3() -> None:
    group = _frontmatter(_GROUP_ID)
    assert group["members"] == [
        "run-i801-a2-blcs-reference-paired",
        "run-i801-a2-blcs-selector-zero-paired",
        "run-i801-a2-plcs-d-reference",
        "run-i801-a2-plcs-d-selector-zero",
        "run-i801-a2-plcs-reference-paired",
        "run-i801-a2-plcs-selector-zero-paired",
    ]
    assert set(group["members"]) == _FORMAL_RUN_IDS
    assert group["parents"] == ["group-i801-reference-selector-ablation"]
    assert {
        path.stem for path in _NODE_ROOT.glob("run-i801-a2-*.md")
    } == _FORMAL_RUN_IDS
    assert {
        path.name
        for path in _RUN_ROOT.glob("run-i801-a2-*")
        if path.is_dir()
    } == _FORMAL_RUN_IDS

    for run_id in sorted(_FORMAL_RUN_IDS):
        node = _frontmatter(run_id)
        bundle = _RUN_ROOT / run_id
        metadata = _json(bundle / "run.json")
        assert node["id"] == run_id
        assert node["issue"] == 801
        assert node["status"] == "done"
        assert node["provider"] == metadata["provider"] == "codex"
        assert node["session"] == metadata["session"]
        assert node["repro"]["commit"] == metadata["commit"]
        assert node["repro"]["command"] == metadata["command"]
        assert node["artifacts"]["run_dir"] == bundle.as_posix()
        assert metadata["branch"] == "feat/issue-801-reference-camera-rope"
        assert metadata["issue"] == "801"
        assert "retry3" not in str(metadata["run_id"]).lower()
        assert "retry3" not in str(metadata["name"]).lower()
        assert "retry3" not in str(metadata["command"]).lower()
        assert metadata["command"] in (bundle / "repro.sh").read_text(
            encoding="utf-8"
        )
        assert (bundle / "uncommitted.patch").read_bytes() == b""
        assert (bundle / "git_status.txt").read_text(encoding="utf-8") == (
            "?? .codex/tasks/issue-801/\n"
        )
        assert not tuple(bundle.glob("*.ckpt"))


def test_plcs_training_bundles_are_complete_and_matched_except_selector() -> None:
    metadata: dict[str, dict[str, Any]] = {}
    nodes: dict[str, dict[str, Any]] = {}
    archives: dict[str, dict[str, np.ndarray[Any, Any]]] = {}
    for selector_mode, run_id in _TRAINING_RUNS.items():
        bundle = _RUN_ROOT / run_id
        assert {path.name for path in bundle.iterdir()} == _TRAINING_FILES
        assert (bundle / "curves.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (bundle / "curves.png").stat().st_size > 100_000
        assert (bundle / "output_dir.txt").read_text(encoding="utf-8").strip().endswith(
            "/checkpoints"
        )
        assert file_sha256(bundle / "pred_test.npz") == (
            _TRAINING_PREDICTION_SHA256[selector_mode]
        )

        metadata[selector_mode] = _json(bundle / "run.json")
        nodes[selector_mode] = _frontmatter(run_id)
        archives[selector_mode] = _npz(run_id)
        metrics = _json(bundle / "metrics.json")
        _assert_node_metrics(nodes[selector_mode], metrics)
        assert metadata[selector_mode]["run_id"] == _TRAINING_QUEUE_IDS[selector_mode]
        assert metadata[selector_mode]["commit"] == (
            "71426cb84519d4fa716ac9b221d90b00d26b4e63"
        )

        config = cast("dict[str, Any]", nodes[selector_mode]["config"])
        assert config == {
            "model": _SELECTOR_PROFILE[selector_mode],
            "architecture": "track_query_ablation_d",
            "task": "plcs",
            "ffn_mode": "shared",
            "mhc_writeback": "layer_end",
            "reference_selector_mode": selector_mode,
            **_CONTRACTS,
            "loss": "tracking",
            "data": "plcs/multi_object_camera_view_norm-v2",
            "seed": 42,
            "seq_len": 128,
            "num_views": 6,
            "batch_size": 8,
            "accumulate_grad_batches": 4,
            "effective_batch_size": 32,
            "epochs": 100,
            "precision": "bf16-mixed",
            "cswa_backend": "cuda",
        }
        argv = shlex.split(cast("str", metadata[selector_mode]["command"]))
        for invariant in (
            "CUDA_VISIBLE_DEVICES=0",
            "court_keypoints=camera_view_v2",
            "model.cswa.backend=cuda",
            "data.scene_dir=plcs/multi_object_camera_view_norm-v2",
            "data.seq_len_range=[128,128]",
            "data.num_views_range=[6,6]",
            "data.batch_size=8",
            "data.num_workers=16",
            "training.trainer.precision=bf16-mixed",
            "training.trainer.accumulate_grad_batches=4",
            "training.trainer.max_epochs=100",
            "training.early_stopping.enabled=false",
            "run.seed=42",
            "run.fast_dev_run=false",
            "run.test_after_fit=true",
        ):
            assert invariant in argv
        assert f"model={_SELECTOR_PROFILE[selector_mode]}" in argv

    assert _normalize_command(
        cast("str", metadata["reference"]["command"]), training=True
    ) == _normalize_command(
        cast("str", metadata["selector_zero"]["command"]), training=True
    )

    reference = archives["reference"]
    selector_zero = archives["selector_zero"]
    assert set(reference) == set(selector_zero)
    permitted_differences = {
        "pred_position",
        "pred_rotation",
        "pred_presence_logits",
        "reference_selector_mode",
    }
    for name in sorted(set(reference) - permitted_differences):
        np.testing.assert_array_equal(reference[name], selector_zero[name])
    for name in permitted_differences - {"reference_selector_mode"}:
        assert not np.array_equal(reference[name], selector_zero[name])
    np.testing.assert_array_equal(
        np.unique(reference["reference_selector_mode"]), np.asarray(["reference"])
    )
    np.testing.assert_array_equal(
        np.unique(selector_zero["reference_selector_mode"]),
        np.asarray(["selector_zero"]),
    )
    assert set(np.unique(reference["reference_view_index"]).tolist()) == set(range(6))
    rows = np.arange(reference["reference_view_index"].shape[0])
    np.testing.assert_array_equal(
        reference["view_camera_id_strings"][rows, reference["reference_view_index"]],
        reference["reference_camera_id_string"],
    )
    assert set(reference["scene_ids"].tolist()) == {
        scene_id
        for scene_id in _npz(_PAIRED_RUNS["plcs"]["reference"])[
            "scene_ids"
        ].tolist()
    }


@pytest.mark.parametrize("task", ["blcs", "plcs"])
def test_paired_bundles_have_exact_identity_config_and_causal_parity(task: str) -> None:
    documents: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    arrays: dict[str, dict[str, np.ndarray[Any, Any]]] = {}
    for selector_mode, run_id in _PAIRED_RUNS[task].items():
        bundle = _RUN_ROOT / run_id
        assert {path.name for path in bundle.iterdir()} == _PAIRED_FILES
        node = _frontmatter(run_id)
        metadata[selector_mode] = _json(bundle / "run.json")
        documents[selector_mode] = _json(bundle / "reference_counterfactual.json")
        arrays[selector_mode] = _npz(run_id)
        evidence = _PAIRED_EVIDENCE[(task, selector_mode)]
        document = documents[selector_mode]
        identity = cast("dict[str, Any]", document["identity"])

        assert metadata[selector_mode]["run_id"] == evidence["queue_id"]
        assert metadata[selector_mode]["commit"] == evidence["commit"]
        assert identity["task"] == task
        assert identity["selector_mode"] == selector_mode
        assert identity["seed"] == 42
        assert identity["checkpoint_sha256"] == evidence["checkpoint"]
        assert identity["resolved_config_digest"] == evidence["config"]
        assert identity["manifest_digest"] == evidence["manifest"]
        assert document["parity_digest"] == evidence["parity"]
        assert document["arrays_digest"] == evidence["arrays"]
        assert document["report_digest"] == evidence["report"]
        for field, marker in _CONTRACTS.items():
            assert identity[field] == marker

        config = cast("dict[str, Any]", identity["resolved_config"])
        data = cast("dict[str, Any]", config["data"])
        model = cast("dict[str, Any]", config["model"])
        evaluation = cast("dict[str, Any]", config["evaluation"])
        assert data["scene_dir"] == f"{task}/multi_object_camera_view_norm-v2"
        assert data["seq_len_range"] == [128, 128]
        assert data["num_views_range"] == [6, 6]
        assert data["evaluation_reference_camera_id"] == "manifest_resolved_per_side"
        assert model["reference_selector_mode"] == selector_mode
        assert model["ffn_mode"] == "shared"
        assert model["mhc_writeback"] == "layer_end"
        assert config["run"]["seed"] == 42
        assert config["court_keypoints"] == {"selector": "camera_view_v2"}
        assert config["court_coordinate_normalization"] == {"version": "v2"}
        assert evaluation["passes"] == ["same_side", "opposite_side"]
        assert evaluation["trainer"]["precision"] == "bf16-mixed"
        assert evaluation["trainer"]["deterministic"] is True

        report = read_reference_counterfactual_report(bundle)
        assert report.identity.to_dict() == identity
        assert report.report_digest == evidence["report"]
        assert report.arrays_digest == evidence["arrays"]
        flat_metrics = _json(bundle / "metrics.json")
        assert report.metrics.flat_dict() == flat_metrics
        _assert_node_metrics(node, flat_metrics)
        assert node["config"]["checkpoint_sha256"] == evidence["checkpoint"]
        assert node["config"]["manifest_digest"] == evidence["manifest"]
        assert node["config"]["report_digest"] == evidence["report"]
        if task == "plcs":
            assert node["parents"] == [_TRAINING_RUNS[selector_mode]]
        else:
            assert node["parents"] == [
                f"run-i801-d-{'reference' if selector_mode == 'reference' else 'selector-zero'}-seeded"
            ]

    assert _normalize_command(
        cast("str", metadata["reference"]["command"]), training=False
    ) == _normalize_command(
        cast("str", metadata["selector_zero"]["command"]), training=False
    )
    assert documents["reference"]["manifest"] == documents["selector_zero"][
        "manifest"
    ]
    assert _normalized_report_config(
        cast("dict[str, Any]", documents["reference"]["identity"])[
            "resolved_config"
        ]
    ) == _normalized_report_config(
        cast("dict[str, Any]", documents["selector_zero"]["identity"])[
            "resolved_config"
        ]
    )

    reference = arrays["reference"]
    selector_zero = arrays["selector_zero"]
    assert set(reference) == set(selector_zero)
    for name in sorted(reference):
        if "prediction" not in name:
            np.testing.assert_array_equal(reference[name], selector_zero[name])
    assert not np.array_equal(
        reference["same_side_position_prediction"],
        selector_zero["same_side_position_prediction"],
    )
    assert not np.array_equal(
        reference["opposite_side_position_prediction"],
        selector_zero["opposite_side_position_prediction"],
    )

    if task == "plcs":
        for selector_mode, paired_metadata in metadata.items():
            training_metadata = _json(
                _RUN_ROOT / _TRAINING_RUNS[selector_mode] / "run.json"
            )
            training_output = next(
                token.removeprefix("run.output_dir=")
                for token in shlex.split(training_metadata["command"])
                if token.startswith("run.output_dir=")
            )
            expected_checkpoint = (
                Path(training_metadata["repo_root"])
                / "outputs"
                / training_output
                / "logs/version_0/checkpoints/last.ckpt"
            )
            paired_checkpoint = next(
                token.removeprefix("evaluation.checkpoint_path=")
                for token in shlex.split(paired_metadata["command"])
                if token.startswith("evaluation.checkpoint_path=")
            )
            assert Path(paired_checkpoint) == expected_checkpoint


@pytest.mark.parametrize("task", ["blcs", "plcs"])
@pytest.mark.parametrize("selector_mode", ["reference", "selector_zero"])
def test_paired_saved_metrics_recompute_from_npz_with_transform_and_strata(
    task: str,
    selector_mode: str,
) -> None:
    run_id = _PAIRED_RUNS[task][selector_mode]
    bundle = _RUN_ROOT / run_id
    arrays = _npz(run_id)
    saved = _json(bundle / "metrics.json")
    valid_mask = torch.from_numpy(arrays["valid_mask"])

    for side, expected_index in (("same_side", 2), ("opposite_side", 0)):
        position = compute_paired_reference_position_metrics(
            torch.from_numpy(arrays[f"{side}_position_prediction"]),
            torch.from_numpy(arrays[f"{side}_position_target"]),
            torch.from_numpy(arrays[f"{side}_reference_view_index"]),
            valid_mask=valid_mask,
        )
        prefix = f"reference_target_{side}"
        assert position.y_sign_accuracy == pytest.approx(
            saved[f"{prefix}_y_sign_accuracy"], abs=1e-7
        )
        for axis, value in zip(
            ("x", "y", "z"),
            (
                position.axis_wise_position_error.x,
                position.axis_wise_position_error.y,
                position.axis_wise_position_error.z,
            ),
            strict=True,
        ):
            assert value == pytest.approx(
                saved[f"{prefix}_position_error_{axis}_m"], abs=1e-7
            )
        assert position.local_reference_index_error == {
            expected_index: pytest.approx(
                saved[
                    f"{prefix}_reference_index_{expected_index}_position_error_m"
                ],
                abs=1e-7,
            )
        }
        if task == "plcs":
            heading_deg = math.degrees(
                compute_heading_error_radians(
                    torch.from_numpy(arrays[f"{side}_heading_prediction"]),
                    torch.from_numpy(arrays[f"{side}_heading_target"]),
                    valid_mask=valid_mask,
                )
            )
            assert heading_deg == pytest.approx(
                saved[f"{prefix}_heading_error_deg"], abs=1e-7
            )

    identity = np.eye(3, dtype=np.float64)
    opposite = np.diag([-1.0, -1.0, 1.0])
    count = arrays["scene_ids"].shape[0]
    np.testing.assert_array_equal(
        arrays["same_side_reference_from_physical"],
        np.broadcast_to(identity, (count, 3, 3)),
    )
    np.testing.assert_array_equal(
        arrays["opposite_side_reference_from_physical"],
        np.broadcast_to(opposite, (count, 3, 3)),
    )
    for side in ("same_side", "opposite_side"):
        forward = arrays[f"{side}_reference_from_physical"]
        inverse = arrays[f"{side}_physical_from_reference"]
        np.testing.assert_array_equal(forward @ inverse, np.broadcast_to(identity, forward.shape))
        np.testing.assert_array_equal(np.linalg.det(forward), np.ones(count))

    same_physical = np.einsum(
        "bij,btqj->btqi",
        arrays["same_side_physical_from_reference"],
        arrays["same_side_position_prediction"],
    )
    opposite_physical = np.einsum(
        "bij,btqj->btqi",
        arrays["opposite_side_physical_from_reference"],
        arrays["opposite_side_position_prediction"],
    )
    position_consistency = np.linalg.norm(
        same_physical - opposite_physical, axis=-1
    )[arrays["valid_mask"]].mean()
    assert float(position_consistency) == pytest.approx(
        saved["physical_restored_position_consistency_error_m"], abs=1e-6
    )

    same_target_physical = np.einsum(
        "bij,btqj->btqi",
        arrays["same_side_physical_from_reference"],
        arrays["same_side_position_target"],
    )
    opposite_target_physical = np.einsum(
        "bij,btqj->btqi",
        arrays["opposite_side_physical_from_reference"],
        arrays["opposite_side_position_target"],
    )
    np.testing.assert_allclose(
        same_target_physical[arrays["valid_mask"]],
        opposite_target_physical[arrays["valid_mask"]],
        rtol=1e-5,
        atol=1e-6,
    )

    if task == "plcs":
        same_heading_physical = np.einsum(
            "bij,btqj->btqi",
            arrays["same_side_physical_from_reference"][:, :2, :2],
            arrays["same_side_heading_prediction"],
        )
        opposite_heading_physical = np.einsum(
            "bij,btqj->btqi",
            arrays["opposite_side_physical_from_reference"][:, :2, :2],
            arrays["opposite_side_heading_prediction"],
        )
        heading_consistency = np.linalg.norm(
            same_heading_physical - opposite_heading_physical,
            axis=-1,
        )[arrays["valid_mask"]].mean()
        assert float(heading_consistency) == pytest.approx(
            saved["physical_restored_heading_consistency_l2"], abs=1e-6
        )


@pytest.mark.parametrize("task", ["blcs", "plcs"])
def test_manifest_rows_fix_scene_window_order_and_reference_identity(task: str) -> None:
    run_id = _PAIRED_RUNS[task]["reference"]
    arrays = _npz(run_id)
    document = _json(_RUN_ROOT / run_id / "reference_counterfactual.json")
    scenes = cast("list[dict[str, Any]]", document["manifest"]["scenes"])
    camera_prefix = "cam" if task == "blcs" else "camera"
    expected_order = np.asarray(
        [[f"{camera_prefix}_{index}" for index in range(6)]] * 100
    )
    assert len(scenes) == 100
    assert len(set(arrays["scene_ids"].tolist())) == 100
    np.testing.assert_array_equal(arrays["view_camera_ids"], expected_order)
    np.testing.assert_array_equal(arrays["local_ordering"], expected_order)
    np.testing.assert_array_equal(arrays["window_start"], np.full(100, 448))
    np.testing.assert_array_equal(arrays["window_stop"], np.full(100, 576))
    np.testing.assert_array_equal(
        arrays["same_side_reference_camera_id"],
        np.full(100, f"{camera_prefix}_2"),
    )
    np.testing.assert_array_equal(
        arrays["opposite_side_reference_camera_id"],
        np.full(100, f"{camera_prefix}_0"),
    )
    np.testing.assert_array_equal(
        arrays["same_side_reference_view_index"], np.full(100, 2)
    )
    np.testing.assert_array_equal(
        arrays["opposite_side_reference_view_index"], np.zeros(100, dtype=np.int64)
    )
    for digest_name in (
        "frame_digest",
        "lifecycle_digest",
        "observation_digest",
        "target_digest",
    ):
        values = arrays[digest_name]
        assert values.shape == (100,)
        assert all(len(str(value)) == 64 for value in values)
    for index, scene in enumerate(scenes):
        assert scene["scene_id"] == arrays["scene_ids"][index]
        assert scene["view_camera_ids"] == expected_order[index].tolist()
        assert scene["local_ordering"] == expected_order[index].tolist()
        assert scene["same_side"]["camera_id"] == f"{camera_prefix}_2"
        assert scene["same_side"]["local_index"] == 2
        assert scene["opposite_side"]["camera_id"] == f"{camera_prefix}_0"
        assert scene["opposite_side"]["local_index"] == 0
