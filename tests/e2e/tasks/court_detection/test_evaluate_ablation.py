"""Exercise synthetic selection arguments and DATA path validation at the CLI."""

from __future__ import annotations

import sys

import pytest

from src.tasks.court_detection.scripts import evaluate_ablation


def test_cli_rejects_conflicting_synthetic_selection(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_ablation",
            "--checkpoint",
            "unused.ckpt",
            "--output",
            str(tmp_path / "out"),
            "--include-synthetic-val",
            "--synthetic-eval-manifest",
            "unused.json",
        ],
    )
    with pytest.raises(SystemExit) as error:
        evaluate_ablation.main()
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_binds_external_manifest_and_keeps_root_overrides_separate(
    tmp_path, monkeypatch, capsys
):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    data_root = tmp_path / "data"
    data_root.mkdir()
    manifest = data_root / "shared.subset.json"
    manifest.write_text("{}")
    output = tmp_path / "out"
    calls = []
    monkeypatch.setattr(
        evaluate_ablation,
        "evaluate_checkpoint",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_ablation",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--data-root",
            str(data_root),
            "--synthetic-eval-manifest",
            str(manifest),
            "--device",
            "cpu",
            "--expected-step",
            "33160",
            "--split",
            "val",
            "--series",
            "pairs",
        ],
    )
    evaluate_ablation.main()
    arguments, options = calls[0]
    assert arguments == (checkpoint, output)
    assert options["synthetic_eval_manifest"] == manifest
    assert options["include_synthetic_val"] is False
    assert options["path_overrides"] == {"data_root": str(data_root)}
    assert options["expected_step"] == 33160
    assert str(output / "evaluation.json") in capsys.readouterr().out


def test_cli_rejects_missing_evaluation_manifest_before_model_loading(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_ablation",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "out"),
            "--data-root",
            str(tmp_path),
            "--synthetic-eval-manifest",
            str(tmp_path / "absent.json"),
        ],
    )
    with pytest.raises(ValueError, match="does not exist"):
        evaluate_ablation.main()
    assert not (tmp_path / "out").exists()
