"""Executable negative validation matrix for ball-detection configuration."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from src.tasks.ball_detection.configuration import (
    BallRuntimePaths,
    validate_eval,
    validate_training,
    validate_visualization,
    validate_youtube_boundary,
)
from src.utils.configuration import ConfigurationError, PathContractError


@contextmanager
def _composition() -> Iterator[None]:
    config_dir = str((Path(__file__).parent / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        yield


def _train() -> DictConfig:
    return compose(config_name="train")


def _web() -> DictConfig:
    return compose(config_name="train", overrides=["data=web_frames"])


def _unknown_key() -> None:
    config = _train()
    with open_dict(config):
        config.model["num_frmaes"] = 8
    validate_training(config)


def _missing_key() -> None:
    config = _train()
    with open_dict(config):
        del config.model.num_frames
    validate_training(config)


def _wrong_type() -> None:
    config = _train()
    config.model.num_frames = "8"
    validate_training(config)


def _run_typo() -> None:
    config = _train()
    with open_dict(config):
        config.run["num_frmaes"] = 8
    validate_training(config)


def _eval_missing_max_batches() -> None:
    config = compose(config_name="eval")
    with open_dict(config):
        del config.evaluation.max_batches_per_split
    validate_eval(config)


def _conflicting_run_paths() -> None:
    config = _train()
    config.run.resume = "resume.ckpt"
    config.run.init_weights = "init.ckpt"
    validate_training(config)


def _gan_trainer_clip_conflict() -> None:
    config = compose(config_name="train", overrides=["training=gan"])
    config.training.trainer.gradient_clip_val = 1.0
    validate_training(config)


def _gan_early_stopping_conflict() -> None:
    config = compose(config_name="train", overrides=["training=gan"])
    config.training.early_stopping.enabled = True
    validate_training(config)


def _removed_temporal_only() -> None:
    config = _web()
    with open_dict(config):
        config.data.temporal_only = True
    validate_training(config)


def _escaping_derived_path() -> None:
    paths = BallRuntimePaths.from_config(_train())
    paths.output("../escape")


def _absolute_derived_path() -> None:
    config = compose(config_name="visualize")
    config.visualization.clip_dir = "/tmp/clip"
    validate_visualization(config)


def _empty_youtube_source_field(field: str) -> None:
    config = compose(config_name="prepare_youtube_dataset")
    config.workflow.sources[0][field] = ""
    validate_youtube_boundary(config)


def _invalid_dino_value(path: str, value: object) -> None:
    config = compose(config_name="prepare_dinov3_ssl_images")
    with open_dict(config):
        OmegaConf.update(config, path, value, merge=False)
    validate_youtube_boundary(config)


def run_negative_matrix() -> tuple[str, ...]:
    """Run every required failure class and return passed case names."""
    cases: tuple[tuple[str, Callable[[], None]], ...] = (
        ("unknown-key", _unknown_key),
        ("missing-key", _missing_key),
        ("wrong-type", _wrong_type),
        ("run-typo", _run_typo),
        ("eval-missing-max-batches", _eval_missing_max_batches),
        ("conflicting-values", _conflicting_run_paths),
        ("gan-trainer-clip-conflict", _gan_trainer_clip_conflict),
        ("gan-early-stopping-conflict", _gan_early_stopping_conflict),
        ("removed-temporal-only", _removed_temporal_only),
        ("escaping-derived-path", _escaping_derived_path),
        ("absolute-derived-path", _absolute_derived_path),
        ("empty-youtube-source-id", lambda: _empty_youtube_source_field("source_id")),
        ("empty-youtube-url", lambda: _empty_youtube_source_field("url")),
        ("empty-youtube-split", lambda: _empty_youtube_source_field("split")),
        (
            "dino-zero-discovery-results",
            lambda: _invalid_dino_value(
                "workflow.discovery.max_results_per_query", 0
            ),
        ),
        (
            "dino-negative-min-duration",
            lambda: _invalid_dino_value("workflow.discovery.min_duration_sec", -1),
        ),
        (
            "dino-negative-max-duration",
            lambda: _invalid_dino_value("workflow.discovery.max_duration_sec", -1),
        ),
        (
            "dino-inverted-duration-range",
            lambda: _invalid_dino_value("workflow.discovery.min_duration_sec", 4000),
        ),
        (
            "dino-zero-processing-limit",
            lambda: _invalid_dino_value("workflow.processing.max_new_videos", 0),
        ),
        (
            "dino-negative-storage-limit",
            lambda: _invalid_dino_value("workflow.storage.max_root_gb", -1),
        ),
        (
            "dino-zero-frame-limit",
            lambda: _invalid_dino_value("workflow.frames.frames_per_video", 0),
        ),
        (
            "dino-empty-download-format",
            lambda: _invalid_dino_value("workflow.download.format", ""),
        ),
        (
            "dino-empty-download-container",
            lambda: _invalid_dino_value("workflow.download.merge_output_format", ""),
        ),
        (
            "dino-empty-js-runtime",
            lambda: _invalid_dino_value("workflow.download.js_runtimes", ""),
        ),
        (
            "dino-invalid-transcode-encoder",
            lambda: _invalid_dino_value("workflow.transcode.encoder", ""),
        ),
        (
            "dino-empty-transcode-pixel-format",
            lambda: _invalid_dino_value("workflow.transcode.pix_fmt", ""),
        ),
        (
            "dino-negative-transcode-crf",
            lambda: _invalid_dino_value("workflow.transcode.crf", -1),
        ),
        (
            "dino-negative-transcode-cq",
            lambda: _invalid_dino_value("workflow.transcode.cq", -1),
        ),
        (
            "dino-invalid-gate-backend",
            lambda: _invalid_dino_value("workflow.gate.backend", "legacy"),
        ),
        (
            "dino-invalid-vllm-label",
            lambda: _invalid_dino_value(
                "workflow.gate.vllm.accept_labels", ["other"]
            ),
        ),
        (
            "dino-negative-vllm-timeout",
            lambda: _invalid_dino_value("workflow.gate.vllm.timeout_sec", -1),
        ),
        (
            "dino-zero-vllm-tokens",
            lambda: _invalid_dino_value("workflow.gate.vllm.max_tokens", 0),
        ),
        (
            "dino-negative-startup-timeout",
            lambda: _invalid_dino_value(
                "workflow.gate.vllm.server.startup_timeout_sec", -1
            ),
        ),
        (
            "dino-zero-poll-interval",
            lambda: _invalid_dino_value(
                "workflow.gate.vllm.server.poll_interval_sec", 0
            ),
        ),
    )
    passed: list[str] = []
    with _composition():
        for name, case in cases:
            try:
                case()
            except (ConfigurationError, PathContractError, ValueError, TypeError):
                passed.append(name)
            else:
                raise AssertionError(f"Negative validation case was accepted: {name}")
    return tuple(passed)


def main() -> int:
    """Run the deterministic matrix as ``python -m ...validation``."""
    passed = run_negative_matrix()
    print(f"ball configuration negative matrix: PASS ({len(passed)} cases)")
    for name in passed:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_negative_matrix"]
