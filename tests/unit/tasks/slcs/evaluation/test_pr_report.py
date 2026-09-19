"""CPU artifact and failure tests for the validation figure report."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from torch.utils.tensorboard import SummaryWriter

from src.tasks.slcs.evaluation.pr_report import (
    REPORT_CONDITIONS,
    TAGS,
    generate_report,
    read_curves,
)
from src.utils.paths import PROJECT_ROOT
from tests.unit.tasks.slcs.evaluation.test_comparison import DOMAINS, _bundles


def _run(root: Path) -> Path:
    root.mkdir()
    (root / "val").mkdir()
    bundles = _bundles(root / "val", REPORT_CONDITIONS)
    context = None
    for directory in bundles.values():
        path = directory / "metrics.json"
        value = json.loads(path.read_text())
        context = value["context"]
        context["split"] = "val"
        path.write_text(json.dumps(value))
    assert context is not None
    (root / "selection.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": context["checkpoint_sha256"],
                "monitor": "val/scene_position_error_m_epoch",
                "mode": "min",
                "selected": {
                    "epoch_zero_based": 1,
                    "path": str(root / "train/model.ckpt"),
                },
            }
        )
    )
    comparison = root / "val/comparison"
    comparison.mkdir()
    (comparison / "comparison.json").write_text(json.dumps({"domain_mapping": DOMAINS}))
    return root


def test_real_figures_statistics_hashes_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unrelated = tmp_path / "unrelated-cwd"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    baseline = _run(tmp_path / "baseline")
    candidate = _run(tmp_path / "candidate")
    destination = generate_report(
        evaluations={"Baseline": baseline, "Candidate": candidate},
        output_root=tmp_path,
        output="slcs/visualize/test/run1",
    )
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["git_commit"] == expected_commit
    row = next(
        row
        for row in manifest["runs"]["Baseline"]["summaries"]
        if row["entity"] == "ball"
        and row["condition"] == "full"
        and row["group"] == "all"
    )
    assert row["mean"] == pytest.approx(5)
    assert row["p95"] == pytest.approx(7.7)
    assert row["count"] == 4
    assert "not measured 3D accuracy" in manifest["interpretation"]
    assert set(manifest["artifacts"]) == {"conditions.png", "distribution.png"}
    for name, digest in manifest["artifacts"].items():
        assert (destination / name).read_bytes().startswith(b"\x89PNG")
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    repeated = generate_report(
        evaluations={"Baseline": baseline, "Candidate": candidate},
        output_root=tmp_path,
        output="slcs/visualize/test/run2",
    )
    assert (
        json.loads((repeated / "manifest.json").read_text())["artifacts"]
        == manifest["artifacts"]
    )
    with pytest.raises(FileExistsError):
        generate_report(
            evaluations={"Baseline": baseline},
            output_root=tmp_path,
            output="slcs/visualize/test/run1",
        )


@pytest.mark.parametrize(
    "change", ["nan", "shape", "empty", "missing", "alignment", "receipt", "split"]
)
def test_invalid_artifacts_fail_before_output(tmp_path: Path, change: str) -> None:
    baseline = _run(tmp_path / "baseline")
    candidate = _run(tmp_path / "candidate")
    path = candidate / "val/full/eval_arrays.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if change == "nan":
        arrays["pred_ball_position"][0, 0, 0] = np.nan
    elif change == "shape":
        arrays["pred_ball_position"] = arrays["pred_ball_position"][:, :1]
    elif change == "empty":
        arrays = {key: value[:0] for key, value in arrays.items()}
    elif change == "missing":
        path.unlink()
    elif change == "alignment":
        # All conditions agree within a run, but targets differ across runs.
        for condition in REPORT_CONDITIONS:
            target = candidate / "val" / condition / "eval_arrays.npz"
            with np.load(target) as archive:
                modified = {key: archive[key] for key in archive.files}
            modified["target_ball_position"] += 0.1
            np.savez(target, **modified)
    elif change == "receipt":
        receipt = json.loads((candidate / "selection.json").read_text())
        receipt["checkpoint_sha256"] = "f" * 64
        (candidate / "selection.json").write_text(json.dumps(receipt))
    elif change == "split":
        metrics = candidate / "val/full/metrics.json"
        payload = json.loads(metrics.read_text())
        payload["context"]["split"] = "test"
        metrics.write_text(json.dumps(payload))
    if change in {"nan", "shape", "empty"}:
        np.savez(path, **arrays)
    with pytest.raises((ValueError, FileNotFoundError)):
        generate_report(
            evaluations={"Baseline": baseline, "Candidate": candidate},
            output_root=tmp_path,
            output="slcs/visualize/test/invalid",
        )
    assert not (tmp_path / "slcs/visualize/test/invalid").exists()


def test_learning_curve_artifact(tmp_path: Path) -> None:
    baseline = _run(tmp_path / "baseline")
    train = baseline / "train"
    with SummaryWriter(str(train / "logs/version_0")) as writer:
        for epoch in range(3):
            writer.add_scalar("epoch", epoch, epoch * 10)
            for tag in TAGS:
                writer.add_scalar(tag, 3 - epoch, epoch * 10)
    receipt_path = baseline / "selection.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["selected"]["validation_score"] = 2
    receipt_path.write_text(json.dumps(receipt))
    output = generate_report(
        evaluations={"Baseline": baseline},
        training={"Baseline": train},
        output_root=tmp_path,
        output="slcs/visualize/test/curves",
    )
    assert (output / "learning_curves.png").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert any("tfevents" in name for name in manifest["source_sha256"])
    receipt["selected"]["validation_score"] = 9
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="score differs"):
        generate_report(
            evaluations={"Baseline": baseline},
            training={"Baseline": train},
            output_root=tmp_path,
            output="slcs/visualize/test/bad-score",
        )


def test_tensorboard_epoch_alignment_and_conflicting_resume(tmp_path: Path) -> None:
    log = tmp_path / "logs/version_0"
    with SummaryWriter(str(log)) as writer:
        for epoch in range(3):
            writer.add_scalar("epoch", epoch, epoch * 10)
            for tag in TAGS:
                writer.add_scalar(tag, 3 - epoch, epoch * 10)
    curves = read_curves(tmp_path, 1)
    assert curves[TAGS[0]] == [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0]]
    with pytest.raises(ValueError, match="selected epoch"):
        read_curves(tmp_path, 4)
    with SummaryWriter(str(tmp_path / "logs/version_1")) as writer:
        writer.add_scalar("epoch", 1, 10)
        writer.add_scalar(TAGS[0], 99, 10)
    with pytest.raises(ValueError, match="Ambiguous repeated"):
        read_curves(tmp_path, 1)


def test_invalid_output_and_requested_training(tmp_path: Path) -> None:
    baseline = _run(tmp_path / "baseline")
    with pytest.raises(ValueError, match="output_root"):
        generate_report(
            evaluations={"A": baseline}, output_root=tmp_path, output="../other"
        )
    with pytest.raises(ValueError, match="Training labels"):
        generate_report(
            evaluations={"A": baseline},
            training={},
            output_root=tmp_path,
            output="slcs/visualize/test/new",
        )
    with pytest.raises(ValueError, match="No TensorBoard"):
        read_curves(baseline, 1)
