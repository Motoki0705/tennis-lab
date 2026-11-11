"""CPU contracts for the new campaign; expectations come from the agreed experiment."""

import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "query_controls_under_test", Path(__file__).with_name("control.py")
)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def report():
    return dict(
        identities={
            "target_identity_sha256": "target",
            "teacher_identity_sha256": "teacher",
            "scored_frame_sha256": "scored",
        },
        snapshot={"directory": "fixed"},
        metrics=dict(
            mean_error_px=28.0,
            median_error_px=7.0,
            p95_error_px=116.0,
            within_distance_rate=0.76,
            distance_threshold_source_px=20.0,
            support=dict(frames=128668, observed=85473),
        ),
    )


def test_first_t64_stage_starts_at_zero_and_preserves_common_transformer():
    model = dict(
        encoder_variant="convnext_v2",
        temporal_mixing="factorized",
        stem_channels=[16, 32, 64, 128],
        mixed_channels=[192, 256],
        residual_blocks=[0, 1, 2, 2, 1, 1],
        dim=256,
        heads=8,
        layers=4,
        ffn_dim=704,
        frames=32,
    )
    conv = dict(model=model, parent={"checkpoint": "conv", "update": 10000})
    x3d = dict(
        model=dict(
            model,
            encoder_variant="x3d_csn",
            temporal_mixing="depthwise3d",
            mixed_channels=[256, 512],
            residual_blocks=[0, 1, 2, 3, 3, 2],
        ),
        parent={"checkpoint": "x3d", "update": 10000},
    )
    jobs = control.definitions(conv, x3d, dict(updates=6000, schedule_offset=6000))
    assert [s["budget"]["schedule_offset"] for j in jobs for s in j["stages"]] == [
        0,
        6000,
    ] * 3
    assert [s["model"]["frames"] for j in jobs for s in j["stages"]] == [
        32,
        64,
        32,
        64,
        64,
        64,
    ]
    assert sum(j["updates"] for j in jobs) == 36000
    assert all(
        (
            s["model"]["dim"],
            s["model"]["layers"],
            s["model"]["heads"],
            s["model"]["ffn_dim"],
        )
        == (256, 4, 8, 704)
        for j in jobs
        for s in j["stages"]
    )
    assert [j["stages"][0]["parent"]["checkpoint"] for j in jobs] == [
        "conv",
        "x3d",
        "conv",
    ]
    assert jobs[0]["stages"][0]["model"]["num_streams"] == 4
    assert (
        "num_streams" not in jobs[1]["stages"][0]["model"]
        and "num_streams" not in jobs[2]["stages"][0]["model"]
    )


def test_newest_committed_snapshot_never_silently_rolls_back(tmp_path):
    for name in (
        "checkpoint-00000500-11111111",
        "checkpoint-00001000-22222222",
        "completed-00001000-33333333",
    ):
        control.write(
            tmp_path / (name + ".json"), dict(status="verified", generation=name)
        )
    path, _ = control.select_receipt(tmp_path)
    assert path.stem == "completed-00001000-33333333"
    control.write(path, dict(status="failed", generation=path.stem))
    with pytest.raises(ValueError, match="Invalid newest"):
        control.select_receipt(tmp_path)


def test_validation_rejects_teacher_id_mismatch_even_at_identical_counts(monkeypatch):
    reference = report()
    monkeypatch.setattr(control, "reference_report", lambda: reference)
    changed = copy.deepcopy(reference)
    changed["identities"]["teacher_identity_sha256"] = "changed"
    with pytest.raises(ValueError, match="frame/teacher IDs"):
        control.validate_report(changed)
    control.validate_report(copy.deepcopy(reference))


def test_comparison_writes_px_and_percentage_point_deltas_with_real_store(
    tmp_path, monkeypatch
):
    from src.utils.artifact_store import RcloneArtifactStore

    if not Path("/tmp/ball-tools/rclone").is_file():
        pytest.skip("rclone binary is required for the publication contract")
    monkeypatch.setenv("PATH", "/tmp/ball-tools:" + os.environ["PATH"])
    root = tmp_path / "local"
    remote = tmp_path / "drive"
    remote.mkdir()
    config = tmp_path / "rclone.conf"
    config.write_text(f"[test]\ntype = alias\nremote = {remote}\n")
    config.chmod(0o600)
    artifact_store = RcloneArtifactStore(
        root, remote="test", remote_root="experiment", config_path=config
    )
    monkeypatch.setattr(control, "ROOT", root)
    monkeypatch.setattr(control, "reference_report", lambda: report())
    monkeypatch.setattr(control, "store", lambda: artifact_store)
    for condition in control.CONDITIONS:
        value = report()
        value["metrics"].update(
            mean_error_px=30.0,
            median_error_px=6.0,
            p95_error_px=110.0,
            within_distance_rate=0.8,
        )
        directory = root / f"{condition}-s2-T64"
        control.write(directory / "validation-update-00006000.json", value)
        control.write(directory / "acceptance.json", {"complete": True})
        control.write(root / f"job-status/{condition}.json", {"complete": True})
    result = control.compare()
    delta = result["rows"][1]["delta"]
    assert (
        delta["mean_error_px"] == 2.0
        and delta["median_error_px"] == -1.0
        and delta["p95_error_px"] == -6.0
    )
    assert delta["within_distance_rate"] == pytest.approx(4.0)
    assert json.loads((remote / "experiment/comparison.json").read_text()) == result
    assert (remote / "experiment/COMPLETED.json").exists()


def test_inputs_code_gate_detects_a_single_changed_source(tmp_path, monkeypatch):
    import torch

    source = tmp_path / "model.py"
    source.write_text("original")
    lock = tmp_path / "uv.lock"
    lock.write_text("locked")
    monkeypatch.chdir(tmp_path)
    plan = dict(
        code={"source_sha256": {str(source): control.sha(source)}},
        controller_sha256={},
        lock_sha256=control.sha(lock),
        torch_version=str(torch.__version__),
    )
    control.verify_plan_sources(plan)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Pinned training source"):
        control.verify_plan_sources(plan)


@pytest.mark.parametrize("corrupt", [False, True])
def test_fresh_vm_restore_verifies_pinned_payload_before_admission(
    tmp_path, monkeypatch, corrupt
):
    import hashlib

    bootstrap_spec = importlib.util.spec_from_file_location(
        "query_controls_bootstrap", Path(__file__).with_name("bootstrap.py")
    )
    bootstrap = importlib.util.module_from_spec(bootstrap_spec)
    bootstrap_spec.loader.exec_module(bootstrap)
    monkeypatch.setattr(bootstrap, "DRIVE", tmp_path / "drive")
    source = tmp_path / "drive/outputs/run/one"
    source.mkdir(parents=True)
    payload = b"full-state-example"
    (source / "state.pt").write_bytes(payload)
    manifest = dict(
        generation="one",
        files={
            "state.pt": dict(
                size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
            )
        },
    )
    control.write(source / "_SNAPSHOT.json", manifest)
    receipt = dict(
        status="verified",
        schema="verified_artifact_snapshot.v1",
        generation="one",
        remote="mounted_drive:outputs/run/one",
        manifest_sha256=control.sha(source / "_SNAPSHOT.json"),
    )
    if corrupt:
        (source / "state.pt").write_bytes(b"corrupt-state")
        with pytest.raises(ValueError, match="Snapshot payload differs"):
            bootstrap.restore(receipt, tmp_path / "restored")
    else:
        bootstrap.restore(receipt, tmp_path / "restored")
        assert (tmp_path / "restored/state.pt").read_bytes() == payload
        with pytest.raises(ValueError, match="overwrite an existing"):
            bootstrap.restore(receipt, tmp_path / "restored")
