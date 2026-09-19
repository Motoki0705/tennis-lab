"""Validation selection and paired CPU evaluation preserve training contracts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.tasks.slcs.data.splits import save_split_file
from src.tasks.slcs.evaluation.run_evaluation import (
    MONITOR,
    evaluate_training_run,
    evaluation_config,
    select_checkpoint,
)
from src.tasks.slcs.training.lightning_module import SLCSLightningModule
from src.utils.schema.court import COURT_COORD_SCALE_XYZ
from tests.support.tasks.slcs.dataset import (
    DEFAULT_FIXTURE_DINO_SPEC,
    SLCSFixtureDatasetConfig,
    build_slcs_dataset_fixture,
)


def _checkpoints(root: Path) -> Path:
    directory = root / "logs/version_0/checkpoints"
    directory.mkdir(parents=True)
    best, other = directory / "epoch=2.ckpt", directory / "epoch=9.ckpt"
    torch.save({"epoch": 2, "test_error": 99}, best)
    torch.save({"epoch": 9, "test_error": 0}, other)
    os.utime(best, (1, 1))
    torch.save(
        {
            "epoch": 59,
            "callbacks": {
                "validation": {
                    "monitor": MONITOR,
                    "best_k_models": {
                        str(other): torch.tensor(4.0),
                        str(best): torch.tensor(2.0),
                    },
                    "best_model_path": str(
                        other
                    ),  # Stale convenience field is not a score.
                },
                "test": {"monitor": "test/error", "best_model_path": str(other)},
            },
        },
        directory / "last.ckpt",
    )
    return best


def _config(tmp_path: Path) -> DictConfig:
    config_dir = Path(__file__).resolve().parents[5] / "src/tasks/slcs/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="train_real_rgb_pilot")
    cfg.paths = {
        "project_root": str(tmp_path),
        "data_root": str(tmp_path / "data"),
        "checkpoint_root": str(tmp_path / "ckpt"),
        "artifact_root": str(tmp_path / "outputs"),
        "output_root": str(tmp_path / "outputs"),
        "cache_root": str(tmp_path / "cache"),
        "external_asset_root": str(tmp_path / "external"),
    }
    cfg.data.dataset_root = "scene"
    cfg.data.split_file = "scene/splits.json"
    cfg.data.window_size = 8
    cfg.data.train_stride = 3
    cfg.data.eval_stride = 8
    cfg.data.quality.min_window_label_ratio = 0.37
    cfg.data.dino = asdict(DEFAULT_FIXTURE_DINO_SPEC)
    cfg.model.hidden_dim = 32
    cfg.model.num_shared_layers = 1
    cfg.model.ffn_dim = 64
    cfg.model.rope_dim = 8
    cfg.model.dino_patch_downsample_factor = 1
    cfg.run.output_dir = "slcs/train/example/run-001"
    return cfg


def test_selection_ignores_test_scores_mtime_and_stale_best_path(
    tmp_path: Path,
) -> None:
    best = _checkpoints(tmp_path)
    chosen, receipt = select_checkpoint(tmp_path)
    assert chosen == best
    assert receipt["selected"]["epoch_zero_based"] == 2
    assert receipt["selected"]["validation_score"] == 2


@pytest.mark.parametrize("failure", ["outside", "nan", "missing", "ambiguous"])
def test_invalid_retained_selection_is_rejected(tmp_path: Path, failure: str) -> None:
    run = tmp_path / "run"
    best = _checkpoints(run)
    last = best.parent / "last.ckpt"
    state = torch.load(last, weights_only=False)
    if failure == "outside":
        outside = tmp_path / "outside.ckpt"
        torch.save({"epoch": 1}, outside)
        state["callbacks"]["validation"]["best_k_models"] = {str(outside): 0.0}
    elif failure == "nan":
        state["callbacks"]["validation"]["best_k_models"] = {str(best): float("nan")}
    elif failure == "missing":
        state["callbacks"]["validation"]["best_k_models"] = {}
    else:
        state["callbacks"]["other"] = state["callbacks"]["validation"]
    torch.save(state, last)
    with pytest.raises(ValueError):
        select_checkpoint(run)


def test_config_keeps_quality_windows_and_tokens_without_mutating_training(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    before = OmegaConf.to_container(cfg.data, resolve=True)
    run = tmp_path / "outputs/slcs/train/example/run-001"
    best = _checkpoints(run)
    result = evaluation_config(
        cfg,
        training_run=run,
        checkpoint=best,
        output_root=tmp_path / "outputs",
        output="slcs/evaluate/example/run-001",
        split="val",
        device="cpu",
        batch_size=2,
    )
    expected = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(expected, dict)
    expected["augmentation"]["enabled"] = False
    assert OmegaConf.to_container(result.data, resolve=True) == expected
    assert OmegaConf.to_container(cfg.data, resolve=True) == before


def test_existing_output_rejected_before_checkpoint_or_model_loading(
    tmp_path: Path,
) -> None:
    (tmp_path / "slcs/evaluate/example/run-001").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        evaluate_training_run(
            training_run=Path("does-not-exist"),
            output_root=tmp_path,
            output="slcs/evaluate/example/run-001",
            domain_prefixes=[],
            default_domain="all",
        )


def test_cuda_requires_queue_before_any_model_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TENNIS_RUN_ID", raising=False)
    monkeypatch.delenv("TENNIS_REPRO_DIR", raising=False)
    with pytest.raises(ValueError, match="training queue"):
        evaluate_training_run(
            training_run=Path("missing"),
            output_root=tmp_path,
            output="slcs/evaluate/example/run-001",
            device="cuda:0",
            domain_prefixes=[],
            default_domain="all",
        )


@pytest.mark.parametrize("requested", [[], ["--splits", "val", "test"]])
def test_cli_test_evaluation_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requested: list[str]
) -> None:
    from scripts.analysis import evaluate_slcs_run

    calls = []

    def evaluate(**kwargs: Any) -> Path:
        calls.append(kwargs)
        return tmp_path

    monkeypatch.setattr(evaluate_slcs_run, "evaluate_training_run", evaluate)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_slcs_run",
            "--output-root",
            str(tmp_path),
            "--training-run",
            "slcs/train/example/run-001",
            "--output",
            "slcs/evaluate/example/run-001",
            "--domain-prefix",
            "video_=meiji",
            "--default-domain",
            "broadcast",
            *requested,
        ],
    )
    evaluate_slcs_run.main()
    assert calls[0]["splits"] == (["val", "test"] if requested else ["val"])
    assert calls[0]["domain_prefixes"] == [("video_", "meiji")]


def test_paired_cpu_run_exports_mixed_fps_and_defaults_to_val(tmp_path: Path) -> None:
    dataset = tmp_path / "data/scene"
    for video, fps in (("video_000", 25.0), ("broadcast", 50.0)):
        build_slcs_dataset_fixture(
            dataset, SLCSFixtureDatasetConfig(videos=(video,), num_frames=8, fps=fps)
        )
    save_split_file(
        dataset / "splits.json",
        {"video_000": "val", "broadcast": "val"},
        seed=0,
        val_ratio=1.0,
        test_ratio=0.0,
    )
    cfg = _config(tmp_path)
    output_root = tmp_path / "outputs"
    run = output_root / "slcs/train/example/run-001"
    best = _checkpoints(run)
    OmegaConf.save(cfg, run / "config.yaml", resolve=True)
    module = SLCSLightningModule(cfg)
    torch.save(
        {
            "epoch": 2,
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        best,
    )
    out = evaluate_training_run(
        training_run=run.relative_to(output_root),
        output_root=output_root,
        output="slcs/evaluate/example/run-001",
        domain_prefixes=[("video_", "meiji")],
        default_domain="broadcast",
        batch_size=2,
    )
    receipt = json.loads((out / "selection.json").read_text())
    assert receipt["selected"]["path"] == str(best)
    assert not (out / "test").exists()
    comparison = json.loads((out / "val/comparison/comparison.json").read_text())
    assert comparison["domain_mapping"] == {
        "video_000": "meiji",
        "broadcast": "broadcast",
    }
    for mode in ("full", "no_rgb", "rgb_only", "detector_gap"):
        folder = out / "val" / mode
        metrics = json.loads((folder / "metrics.json").read_text())
        assert metrics["num_windows"] == 2
        assert metrics["context"]["selection"]["selected"]["epoch_zero_based"] == 2
        assert (
            metrics["context"]["evaluation_config"]["data"]["quality"][
                "min_window_label_ratio"
            ]
            == 0.37
        )
        motion = json.loads((folder / "motion.json").read_text())
        assert {item["fps"] for item in motion["fps_by_clip"]} == {25, 50}
        with np.load(folder / "eval_arrays.npz") as arrays:
            pred = arrays["pred_ball_position"] * np.asarray(COURT_COORD_SCALE_XYZ)
            for row in motion["rows"]:
                if row["group_type"] != "video":
                    continue
                index = arrays["video_ids"].tolist().index(row["video_id"])
                keep = arrays["ball_mask"][index, 1:] & arrays["ball_mask"][index, :-1]
                fps = 25 if row["video_id"] == "video_000" else 50
                expected = np.linalg.norm(
                    np.diff(pred[index], axis=0)[keep] * fps, axis=-1
                ).mean()
                assert row["entities"]["ball"]["velocity"]["pred_norm"][
                    "mean"
                ] == pytest.approx(expected)
