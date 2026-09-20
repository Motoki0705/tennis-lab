"""CPU contracts for reconstruction-review dispatch and held-out BLCS evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tennis_scene.scripts import render_reconstruction_review as review
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT


@pytest.mark.parametrize(
    "arguments",
    [
        ["--dataset-root", "/datasets", "--run-root", "/runs", "--frames", "0"],
        ["--dataset-root", "/datasets", "--run-root", "/runs", "--mode", "frames"],
        [
            "--data-root",
            "/datasets",
            "--dataset",
            "sample",
            "--video",
            "v",
            "--frames",
            "0",
            "--run-root",
            "/runs",
        ],
        [
            "--data-root",
            "/datasets",
            "--dataset",
            "sample",
            "--video",
            "v",
            "--frames",
            "-1",
        ],
        ["--dataset-root", "relative", "--run-root", "/runs"],
    ],
)
def test_review_rejects_conflicting_or_invalid_modes(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        review.main(
            [
                *arguments,
                "--output-root",
                "/results",
                "--output",
                "tennis_scene/visualize/exp/run",
                "--clip",
                "v/c",
            ]
        )
    assert error.value.code == 2


def test_blcs_fixed_seed_strict_load_and_test_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.blcs.evaluation import real
    from src.tasks.blcs.evaluation.configuration import RealEvaluationConfig

    calls: list[object] = []
    roots = RuntimePathRoots(
        **{
            f"{role}_root": tmp_path
            for role in (
                "project",
                "data",
                "checkpoint",
                "artifact",
                "output",
                "cache",
                "external_asset",
            )
        }
    )
    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    provenance = tmp_path / "dataset/provenance.json"
    provenance.parent.mkdir()
    provenance.write_text("{}")
    module = SimpleNamespace(
        on_load_checkpoint=lambda value: calls.append("contract"),
        load_state_dict=lambda state, strict: calls.append((state, strict)),
    )
    monkeypatch.setattr(
        real,
        "compose_blcs_training",
        lambda cfg, generator_config: SimpleNamespace(
            lightning_module=module, datamodule="fixed-split"
        ),
    )
    monkeypatch.setattr(
        real.torch, "load", lambda *args, **kwargs: {"state_dict": "weights"}
    )
    monkeypatch.setattr(
        real.pl, "seed_everything", lambda seed, workers: calls.append((seed, workers))
    )

    def trainer(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(test=lambda module, datamodule: [{"test/position": 1.0}])

    monkeypatch.setattr(real.pl, "Trainer", trainer)
    cfg = OmegaConf.create({"run": {"seed": 42}, "data": {"scene_dir": "dataset"}})
    request = RealEvaluationConfig(
        cfg, PathResolver(roots), checkpoint, tmp_path / "evaluation", "cpu"
    )
    real.evaluate(request)
    assert calls[:3] == [(42, True), "contract", ("weights", True)]
    trainer_options = calls[3]
    assert isinstance(trainer_options, dict)
    assert trainer_options["enable_checkpointing"] is False
    assert trainer_options["precision"] == "32-true"
    report = json.loads((request.output / "evaluation.json").read_text())
    assert report["metrics"] == {"test/position": 1.0}
    with pytest.raises(FileExistsError):
        real.evaluate(request)


def test_blcs_evaluation_configuration_is_training_recipe_with_explicit_checkpoint(
    tmp_path: Path,
) -> None:
    from src.tasks.blcs.evaluation.configuration import RealEvaluationConfig

    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tasks/blcs/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="evaluate_real",
            overrides=["evaluation.checkpoint=blcs/selected.ckpt"],
        )
    cfg.paths.output_root = str(tmp_path / "results")
    request = RealEvaluationConfig.from_config(cfg)
    assert request.training.run.seed == 42
    assert request.training.data.scene_dir == "blcs/meiji_geometry_replay_v1"
    assert request.training.data.num_workers == 0
    assert "evaluation" not in request.training
    assert request.output.is_relative_to(tmp_path / "results/blcs/evaluate")
