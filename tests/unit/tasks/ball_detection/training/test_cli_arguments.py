"""Reject omitted recipe and execution choices at the public training entry."""

import sys
from pathlib import Path

import pytest

from src.tasks.ball_detection.scripts.train import main

RUNTIME_CHOICES = ("precision", "jpeg-decoder", "compile-mode", "input-verification")
RECIPE_CHOICES = ("epochs", "windows-per-epoch", "learning-rate", "seed", "warmup-updates")
REQUIRED_CHOICES = [
    (stage, option)
    for stage in ("scratch", "dpt_pretrain", "encoder")
    for option in (
        *RUNTIME_CHOICES,
        *(RECIPE_CHOICES if stage != "scratch" else ("epochs", "learning-rate", "seed")),
    )
]


@pytest.mark.parametrize(("stage", "omitted"), REQUIRED_CHOICES)
def test_omitted_choice_fails_before_reading_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    stage: str, omitted: str,
) -> None:
    # Input paths do not exist: argument rejection must precede any file access.
    values = dict(manifest=str(tmp_path / "manifest.json"), output=str(tmp_path / "run"),
                  epochs="2", **{"windows-per-epoch": "6", "learning-rate": ".001",
                  "seed": "42", "device": "cpu", "precision": "fp32", "jpeg-decoder": "opencv",
                  "compile-mode": "off", "input-verification": "lazy"})
    if stage == "encoder":
        selector = ["--stage", "coordinates", "--initialize", "encoder"]
        values.update({"pretraining-run": str(tmp_path / "pretrained"),
                       "augmentation-config": str(tmp_path / "augmentation.yaml"), "warmup-updates": "0"})
    else:
        selector = (["--stage", "coordinates", "--initialize", "scratch"] if stage == "scratch"
                    else ["--stage", "dpt_pretrain"])
        values["model-config"] = str(tmp_path / "model.yaml")
        if stage == "dpt_pretrain":
            values["warmup-updates"] = "0"
    del values[omitted]
    argv = ["train", *selector, *(arg for key, value in values.items() for arg in (f"--{key}", value))]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert f"the following arguments are required: --{omitted}" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()
