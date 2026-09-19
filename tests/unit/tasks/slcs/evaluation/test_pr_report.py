"""CPU artifact and failure tests for the validation figure report."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torch.utils.tensorboard import SummaryWriter

from src.tasks.slcs.evaluation import pr_report
from src.tasks.slcs.evaluation.pr_report import (
    REPORT_CONDITIONS,
    TAGS,
    TRAIN_TAGS,
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
                writer.add_scalar(
                    tag, (30 if tag in TRAIN_TAGS else 3) - epoch, epoch * 10
                )
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
    assert manifest["runs"]["Baseline"]["curves"][TRAIN_TAGS[0]] == [
        [0.0, 30.0],
        [1.0, 29.0],
        [2.0, 28.0],
    ]
    receipt["selected"]["validation_score"] = 9
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="score differs"):
        generate_report(
            evaluations={"Baseline": baseline},
            training={"Baseline": train},
            output_root=tmp_path,
            output="slcs/visualize/test/bad-score",
        )


@pytest.mark.parametrize("conflicting_tag", [TAGS[0], *TRAIN_TAGS[:3]])
def test_tensorboard_epoch_alignment_and_conflicting_resume(
    tmp_path: Path, conflicting_tag: str
) -> None:
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
        writer.add_scalar(conflicting_tag, 99, 10)
    with pytest.raises(ValueError, match="Ambiguous repeated"):
        read_curves(tmp_path, 1)


@pytest.mark.parametrize(
    "problem", ["missing_tag", "missing_epoch", "nan", "inf", "unaligned"]
)
def test_required_train_curves_fail_with_label_and_tag(
    tmp_path: Path, problem: str
) -> None:
    baseline = _run(tmp_path / "baseline")
    train = baseline / "train"
    bad_tag = TRAIN_TAGS[1]
    with SummaryWriter(str(train / "logs/version_0")) as writer:
        for epoch in range(3):
            writer.add_scalar("epoch", epoch, epoch * 10)
            for tag in TAGS:
                if tag == bad_tag and (
                    problem == "missing_tag"
                    or (problem == "missing_epoch" and epoch == 2)
                ):
                    continue
                value = (
                    float(problem)
                    if tag == bad_tag and epoch == 2 and problem in {"nan", "inf"}
                    else 3 - epoch
                )
                step = epoch * 10 + int(
                    tag == bad_tag and epoch == 2 and problem == "unaligned"
                )
                writer.add_scalar(tag, value, step)
    with pytest.raises(ValueError, match=f"Training curves for Baseline: .*{bad_tag}"):
        generate_report(
            evaluations={"Baseline": baseline},
            training={"Baseline": train},
            output_root=tmp_path,
            output="slcs/visualize/test/invalid-curves",
        )
    assert not (tmp_path / "slcs/visualize/test/invalid-curves").exists()


def test_identical_resume_is_joined_without_duplicate_train_points(
    tmp_path: Path,
) -> None:
    for version, epochs in ((0, (0, 1)), (1, (1, 2))):
        with SummaryWriter(str(tmp_path / f"logs/version_{version}")) as writer:
            for epoch in epochs:
                writer.add_scalar("epoch", epoch, epoch * 10)
                for index, tag in enumerate(TAGS):
                    writer.add_scalar(tag, 20 + index - epoch, epoch * 10)
    curves = read_curves(tmp_path, 1)
    for index, tag in enumerate(TAGS):
        assert curves[tag] == [
            [float(epoch), float(20 + index - epoch)] for epoch in range(3)
        ]


def test_every_panel_draws_raw_train_and_val_without_clipping_or_caption_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(("Baseline", "Candidate")):
        curves = {
            tag: [
                [float(epoch), (1 + index * 1e6) * (20 + tag_index - epoch)]
                for epoch in range(3)
            ]
            for tag_index, tag in enumerate(TAGS)
        }
        runs[label] = {
            "curves": curves,
            "selection": {"selected": {"epoch_zero_based": 1}},
        }
    saved: list[Any] = []
    original_save = pr_report._save

    def capture(fig: Any, path: Path, subtitle: str) -> None:
        original_save(fig, path, subtitle)
        saved.append(fig)

    monkeypatch.setattr(pr_report, "_save", capture)
    pr_report._curves_plot(runs, tmp_path)
    fig = saved[0]
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    captions = [text for text in fig.texts if text.get_position()[1] < 0.1]
    assert any("augmentation" in text.get_text() for text in captions)
    for ax, val_tag, train_tag in zip(fig.axes, TAGS[:4], TRAIN_TAGS, strict=True):
        lines = {line.get_label(): line for line in ax.lines}
        for label, run in runs.items():
            for phase, tag, style in (
                ("val", val_tag, "-"),
                ("train", train_tag, "--"),
            ):
                line = lines[f"{label} · {phase}"]
                expected = np.array(run["curves"][tag])
                np.testing.assert_array_equal(line.get_xdata(), expected[:, 0])
                np.testing.assert_array_equal(line.get_ydata(), expected[:, 1])
                assert line.get_linestyle() == style
                assert ax.get_ylim()[0] <= expected[:, 1].min()
                assert ax.get_ylim()[1] > expected[:, 1].max()
        assert len(ax.collections) == 2
        for collection, run in zip(ax.collections, runs.values(), strict=True):
            np.testing.assert_array_equal(
                collection.get_offsets(), [run["curves"][val_tag][1]]
            )
        axis_bounds = ax.get_tightbbox(renderer)
        assert all(
            axis_bounds.y0 > caption.get_window_extent(renderer).y1
            for caption in captions
        )
        assert fig.legends[0].get_window_extent(renderer).y0 > axis_bounds.y1
    bounds = [text.get_window_extent(renderer) for text in captions]
    for index, first in enumerate(bounds):
        assert all(not first.overlaps(second) for second in bounds[index + 1 :])


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
