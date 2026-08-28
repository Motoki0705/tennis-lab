"""Matched PLCS selector/selector-zero training evidence for Issue #801."""

from __future__ import annotations

import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml

_RUN_ROOT = Path("knowledge/runs")
_NODE_ROOT = Path("knowledge/nodes")
_TRAINING_RUNS = {
    "reference": "run-i801-a2-plcs-d-reference",
    "selector_zero": "run-i801-a2-plcs-d-selector-zero",
}
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_training_command(command: str) -> list[str]:
    normalized: list[str] = []
    for token in shlex.split(command):
        if token.startswith("model="):
            normalized.append("model=<MATCHED_VARIANT>")
        elif token.startswith("run.output_dir="):
            normalized.append("run.output_dir=<MATCHED_VARIANT>")
        else:
            normalized.append(token)
    return normalized


def _assert_node_metrics(node: dict[str, Any], metrics: dict[str, Any]) -> None:
    node_metrics = cast("dict[str, Any]", node["metrics"])
    assert set(node_metrics) == set(metrics)
    for name, value in metrics.items():
        assert isinstance(value, (int, float)) and math.isfinite(float(value))
        assert float(node_metrics[name]) == pytest.approx(
            round(float(value), 6), abs=5e-7
        )


def test_plcs_training_bundles_are_complete_and_matched_except_selector() -> None:
    metadata: dict[str, dict[str, Any]] = {}
    archives: dict[str, dict[str, np.ndarray[Any, Any]]] = {}
    for selector_mode, run_id in _TRAINING_RUNS.items():
        bundle = _RUN_ROOT / run_id
        assert {path.name for path in bundle.iterdir()} == _TRAINING_FILES
        assert (bundle / "curves.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (bundle / "curves.png").stat().st_size > 100_000
        assert (
            (bundle / "output_dir.txt")
            .read_text(encoding="utf-8")
            .strip()
            .endswith("/checkpoints")
        )
        assert (
            _file_sha256(bundle / "pred_test.npz")
            == (_TRAINING_PREDICTION_SHA256[selector_mode])
        )

        metadata[selector_mode] = _json(bundle / "run.json")
        node = _frontmatter(run_id)
        archives[selector_mode] = _npz(run_id)
        _assert_node_metrics(node, _json(bundle / "metrics.json"))

        assert node["id"] == run_id
        assert node["issue"] == 801
        assert node["status"] == "done"
        assert node["provider"] == metadata[selector_mode]["provider"] == "codex"
        assert node["repro"]["commit"] == metadata[selector_mode]["commit"]
        assert node["repro"]["command"] == metadata[selector_mode]["command"]
        assert node["artifacts"]["run_dir"] == bundle.as_posix()
        assert metadata[selector_mode]["run_id"] == _TRAINING_QUEUE_IDS[selector_mode]
        assert metadata[selector_mode]["commit"] == (
            "71426cb84519d4fa716ac9b221d90b00d26b4e63"
        )
        assert metadata[selector_mode]["branch"] == (
            "feat/issue-801-reference-camera-rope"
        )
        assert metadata[selector_mode]["issue"] == "801"
        assert metadata[selector_mode]["command"] in (bundle / "repro.sh").read_text(
            encoding="utf-8"
        )
        assert (bundle / "uncommitted.patch").read_bytes() == b""
        assert not tuple(bundle.glob("*.ckpt"))

        config = cast("dict[str, Any]", node["config"])
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

    assert _normalize_training_command(metadata["reference"]["command"]) == (
        _normalize_training_command(metadata["selector_zero"]["command"])
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
