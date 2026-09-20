"""Check the immutable comparison recipe and fail-closed resumption."""

from __future__ import annotations

import copy
import tarfile
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.colab.train.court_vit_ablation.checkpoints import validate_checkpoint
from scripts.colab.train.court_vit_ablation.prepare import validate_member
from scripts.colab.train.court_vit_ablation.run import HERE, VARIANTS, variant_config
from scripts.colab.workflow.jobs import load_registry


def test_suite_inputs_and_durable_outputs() -> None:
    root = Path(__file__).resolve().parents[3]
    job = load_registry(root / "scripts/colab/workflows/jobs")["court_vit_ablation"]
    assert job.output_storage == "drive"
    assert len(job.inputs) == 12
    assert any(i.destination == "ckpt/court_vit_ablation/b.ckpt" for i in job.inputs)


@pytest.mark.parametrize("size", VARIANTS)
def test_variant_only_changes_backbone_dependent_dimensions(
    size: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ckpt/court_vit_ablation").mkdir(parents=True)
    (tmp_path / "ckpt/court_vit_ablation/b.ckpt").touch()
    cfg = variant_config(
        size,
        [
            "run.artifact_store.mode=rclone",
            "run.artifact_store.remote=gdrive",
            "run.artifact_store.sync_interval_seconds=60",
            "run.artifact_store.remote_root=tennis_lab/colab-live/test/training",
        ],
    )
    assert cfg.model.transformer_encoder.dim == 768
    assert cfg.model.transformer_encoder.num_heads == 12
    assert ("feature_adapter" in cfg.model) == (size != "b")
    if size != "b":
        assert cfg.model.feature_adapter.output_channels == 768
    assert list(cfg.model.encoder.out_indices) == VARIANTS[size][3]
    assert cfg.model.transformer_encoder.depth == 8
    assert cfg.model.transformer_encoder.ffn_dim == 2048
    assert cfg.training.trainer.max_epochs == 20
    assert cfg.data.batch_size == 8
    assert cfg.mixed.train_batch_counts == {
        "synthetic_court": 4,
        "tennis_court_detector": 4,
    }
    assert cfg.run.artifact_store.remote_root.endswith(f"/{size}")
    assert (cfg.run.resume is not None) == (size == "b")
    from src.tasks.court_detection.training.runner_mixed import (
        resolve_mixed_training_config,
    )

    resolve_mixed_training_config(cfg)


def checkpoint_and_config() -> tuple[dict, dict]:
    cfg = OmegaConf.to_container(OmegaConf.load(HERE / "baseline.yaml"), resolve=True)
    assert isinstance(cfg, dict)
    saved = copy.deepcopy(cfg)
    saved.pop("mixed")
    saved["training"]["compile"]["enabled"] = True
    saved["training"]["early_stopping"]["enabled"] = True
    ckpt = {
        "epoch": 17,
        "global_step": 29844,
        "state_dict": {"w": 1},
        "optimizer_states": [{"state": 1}],
        "lr_schedulers": [{"last_epoch": 18}],
        "loops": {"fit_loop": 1},
        "hyper_parameters": {"config": saved},
    }
    return ckpt, cfg


def test_resume_allows_execution_and_output_changes_only() -> None:
    ckpt, cfg = checkpoint_and_config()
    cfg["paths"]["output_root"] = "outputs/colab"
    cfg["run"]["output_dir"] = "new_run"
    validate_checkpoint(ckpt, cfg, initial=True)


@pytest.mark.parametrize("key", ["optimizer_states", "lr_schedulers", "loops"])
def test_resume_rejects_weight_only_checkpoint(key: str) -> None:
    ckpt, cfg = checkpoint_and_config()
    ckpt.pop(key)
    with pytest.raises(ValueError, match="full training state"):
        validate_checkpoint(ckpt, cfg, initial=True)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("data", "batch_size", 4),
        ("training", "learning_rate", 0.01),
        ("run", "seed", 1),
    ],
)
def test_resume_rejects_changed_experiment(
    section: str, key: str, value: object
) -> None:
    ckpt, cfg = checkpoint_and_config()
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_checkpoint(ckpt, cfg, initial=True)


def test_resume_rejects_wrong_initial_checkpoint() -> None:
    ckpt, cfg = checkpoint_and_config()
    ckpt["epoch"] = 5
    with pytest.raises(ValueError, match="Expected baseline"):
        validate_checkpoint(ckpt, cfg, initial=True)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../bad", tarfile.REGTYPE),
        ("/bad", tarfile.REGTYPE),
        ("court/link", tarfile.SYMTYPE),
    ],
)
def test_archive_rejects_escaping_paths_and_links(name: str, kind: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.type = kind
    with pytest.raises(ValueError, match="Unsafe"):
        validate_member(member)


def test_continuation_stages_both_durable_candidates() -> None:
    from scripts.colab.train.court_vit_ablation.launch import resume_inputs

    files: list[dict[str, object]] = [
        {"Path": "b/logs/version_0/checkpoints/last.ckpt"},
        {"Path": "b/logs/version_0/checkpoints/recovery.ckpt"},
        {"Path": "s/smoke/logs/version_0/checkpoints/last.ckpt"},
        {"Path": "b/logs/version_0/checkpoints/last.ckpt.uploading-partial"},
    ]
    inputs = resume_inputs(files, "prior-run")
    assert len(inputs) == 2
    assert all(
        str(i["source"]).startswith("colab-live/prior-run/training/b/") for i in inputs
    )
    with pytest.raises(ValueError, match="duplicate"):
        resume_inputs([files[0], files[0]], "prior-run")
    with pytest.raises(ValueError, match="No durable"):
        resume_inputs([], "prior-run")


def test_job_encoding_preserves_manifest() -> None:
    import tomllib

    from scripts.colab.train.court_vit_ablation.launch import REPO, encode_job

    path = REPO / "scripts/colab/workflows/jobs/court_vit_ablation.toml"
    job = tomllib.loads(path.read_text())
    assert tomllib.loads(encode_job(job)) == job


def test_required_recipe_resources_are_in_git_source() -> None:
    import json
    import subprocess

    from scripts.colab.train.court_vit_ablation.launch import REPO

    for name in ("archives.json", "baseline.yaml"):
        path = HERE / name
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(REPO))],
            cwd=REPO,
            check=True,
            capture_output=True,
        )
    manifest = json.loads((HERE / "archives.json").read_text())
    assert len(manifest["archives"]) == 6
    assert sum(item["size_bytes"] for item in manifest["archives"]) == 42905595821


def test_continuation_keeps_completed_variants_from_earlier_sessions() -> None:
    from scripts.colab.train.court_vit_ablation.launch import resume_inputs

    result = resume_inputs(
        [
            {"Path": "b/resume/last.ckpt"},
            {"Path": "s/logs/version_0/checkpoints/recovery.ckpt"},
            {"Path": "splus/smoke/resume/last.ckpt"},
        ],
        "second-run",
    )
    assert len(result) == 2
    assert result[0]["destination"] == "ckpt/court_vit_ablation/b/resume/last.ckpt"
