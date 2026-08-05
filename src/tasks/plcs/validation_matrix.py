"""Executable negative validation matrix for PLCS configuration boundaries."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import as_config_mapping
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.generate_dataset.config import PLCSGenerationConfig
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    PathContractError,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.hydra import validate_boundary


def _copy(config: DictConfig) -> dict[str, object]:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("Composed PLCS config must be a mapping.")
    return dict(as_config_mapping(value, path="configuration"))


def _expect_failure(
    name: str,
    action: Callable[[], object],
    *,
    error_type: type[BaseException],
    message_fragment: str,
) -> None:
    try:
        action()
    except error_type as error:
        if message_fragment not in str(error):
            raise AssertionError(
                f"PLCS negative case {name!r} raised the right class but not the "
                f"required classification {message_fragment!r}: {error}"
            ) from error
        return
    raise AssertionError(f"PLCS negative validation case {name!r} was accepted.")


def run_negative_matrix() -> None:
    """Reject missing, unknown, mistyped, conflicting, legacy, and invalid paths."""
    config_dir = str(Path(__file__).parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        valid_train = compose(config_name="train")
        valid_tracking = compose(config_name="train_tracking")
        valid_gan = compose(config_name="train_chunked_gan")
        valid_generation = compose(config_name="generate_dataset")
        valid_multi_generation = compose(
            config_name="generate_dataset", overrides=["generation=multi_object"]
        )
        valid_visualization = compose(config_name="visualize")

    missing = _copy(valid_train)
    missing_model = missing["model"]
    if not isinstance(missing_model, dict):
        raise TypeError("model must be a mapping")
    del missing_model["hidden_dim"]
    _expect_failure(
        "missing",
        lambda: PLCSTrainingConfig.from_config(OmegaConf.create(missing)),
        error_type=MissingConfigurationKeyError,
        message_fragment="model.hidden_dim",
    )

    unknown = _copy(valid_train)
    unknown["modle"] = unknown["model"]
    _expect_failure(
        "unknown",
        lambda: PLCSTrainingConfig.from_config(OmegaConf.create(unknown)),
        error_type=UnknownConfigurationKeyError,
        message_fragment="configuration.modle",
    )

    wrong_type = _copy(valid_train)
    wrong_type["data"]["batch_size"] = "four"  # type: ignore[index]
    _expect_failure(
        "wrong-type",
        lambda: PLCSTrainingConfig.from_config(OmegaConf.create(wrong_type)),
        error_type=ConfigurationTypeError,
        message_fragment="data.batch_size",
    )

    conflict = _copy(valid_train)
    conflict["run"]["resume"] = "a.ckpt"  # type: ignore[index]
    conflict["run"]["init_weights"] = "b.ckpt"  # type: ignore[index]
    _expect_failure(
        "exclusive-conflict",
        lambda: PLCSTrainingConfig.from_config(OmegaConf.create(conflict)),
        error_type=SemanticConfigurationError,
        message_fragment="mutually exclusive",
    )

    legacy = _copy(valid_train)
    legacy["data"]["augmentation"]["keypoint_noise_std"] = 0.1  # type: ignore[index]
    _expect_failure(
        "legacy-augmentation-key",
        lambda: PLCSTrainingConfig.from_config(OmegaConf.create(legacy)),
        error_type=UnknownConfigurationKeyError,
        message_fragment="keypoint_noise_std",
    )

    invalid_root = _copy(valid_train)
    invalid_root["paths"]["data_root"] = 5  # type: ignore[index]
    _expect_failure(
        "invalid-root-type",
        lambda: PLCSTrainingConfig.from_config(OmegaConf.create(invalid_root)),
        error_type=ConfigurationTypeError,
        message_fragment="paths.data_root",
    )

    escaping_path = _copy(valid_train)
    escaping_path["run"]["output_dir"] = "../escape"  # type: ignore[index]
    _expect_failure(
        "escaping-derived-path",
        lambda: PLCSTrainingConfig.from_config(OmegaConf.create(escaping_path)),
        error_type=PathContractError,
        message_fragment="escapes its declared parent",
    )

    generation_conflict = _copy(valid_generation)
    generation_conflict["camera"]["broadcast_court_width_frac_range"] = [0.5, 0.9]  # type: ignore[index]
    generation_conflict["camera"]["broadcast_hfov_jitter_deg"] = 1.0  # type: ignore[index]
    _expect_failure(
        "camera-exclusive-conflict",
        lambda: PLCSGenerationConfig.from_config(OmegaConf.create(generation_conflict)),
        error_type=SemanticConfigurationError,
        message_fragment="mutually exclusive",
    )

    unsupported_generation_device = _copy(valid_generation)
    unsupported_generation_device["run"]["device"] = "cuda"  # type: ignore[index]
    _expect_failure(
        "unsupported-generation-device",
        lambda: PLCSGenerationConfig.from_config(
            OmegaConf.create(unsupported_generation_device)
        ),
        error_type=SemanticConfigurationError,
        message_fragment="run.device",
    )

    nested_trainer_typo = _copy(valid_train)
    nested_trainer_typo["training"]["trainer"]["__unknown__"] = 1  # type: ignore[index]
    _expect_failure(
        "nested-trainer-typo",
        lambda: PLCSTrainingConfig.from_config(OmegaConf.create(nested_trainer_typo)),
        error_type=UnknownConfigurationKeyError,
        message_fragment="training.trainer.__unknown__",
    )

    null_model_choice = _copy(valid_train)
    null_model_choice["model"]["ffn_dim"] = None  # type: ignore[index]
    _expect_failure(
        "null-model-choice",
        lambda: PLCSTrainingConfig.from_config(OmegaConf.create(null_model_choice)),
        error_type=ConfigurationTypeError,
        message_fragment="model.ffn_dim",
    )

    invalid_model_dimensions = _copy(valid_train)
    invalid_model_dimensions["model"]["hidden_dim"] = 257  # type: ignore[index]
    _expect_failure(
        "invalid-model-head-dimensions",
        lambda: PLCSTrainingConfig.from_config(
            OmegaConf.create(invalid_model_dimensions)
        ),
        error_type=SemanticConfigurationError,
        message_fragment="positive and divisible",
    )

    invalid_augmentation_probability = _copy(valid_train)
    invalid_augmentation_probability["data"]["augmentation"][  # type: ignore[index]
        "gaussian_noise"
    ]["prob"] = 1.1
    _expect_failure(
        "invalid-augmentation-probability",
        lambda: PLCSTrainingConfig.from_config(
            OmegaConf.create(invalid_augmentation_probability)
        ),
        error_type=SemanticConfigurationError,
        message_fragment="within [0, 1]",
    )

    gan_global_clip = _copy(valid_gan)
    gan_global_clip["training"]["trainer"]["gradient_clip_val"] = 1.0  # type: ignore[index]
    _expect_failure(
        "gan-global-gradient-clip-conflict",
        lambda: PLCSTrainingConfig.from_config(OmegaConf.create(gan_global_clip)),
        error_type=SemanticConfigurationError,
        message_fragment="must be null when GAN is enabled",
    )

    gan_early_stop = _copy(valid_gan)
    gan_early_stop["training"]["early_stopping"]["enabled"] = True  # type: ignore[index]
    _expect_failure(
        "gan-early-stopping-conflict",
        lambda: PLCSTrainingConfig.from_config(OmegaConf.create(gan_early_stop)),
        error_type=SemanticConfigurationError,
        message_fragment="must be false when GAN is enabled",
    )

    old_path_key = _copy(valid_generation)
    old_path_key["paths"]["smplh_model_path"] = "smplx/smplh"  # type: ignore[index]
    _expect_failure(
        "old-path-key",
        lambda: PLCSGenerationConfig.from_config(OmegaConf.create(old_path_key)),
        error_type=UnknownConfigurationKeyError,
        message_fragment="paths.smplh_model_path",
    )

    doubled_prefix = _copy(valid_generation)
    doubled_prefix["motion_sources"]["running"]["paths"] = [  # type: ignore[index]
        "data/ACCAD/Female1Running_c3d"
    ]
    _expect_failure(
        "doubled-data-prefix",
        lambda: PLCSGenerationConfig.from_config(OmegaConf.create(doubled_prefix)),
        error_type=PathContractError,
        message_fragment="root-prefixed or legacy fragment",
    )

    external_escape = _copy(valid_generation)
    external_escape["external_assets"]["smplh_model_path"] = "../smplh"  # type: ignore[index]
    _expect_failure(
        "external-root-escape",
        lambda: PLCSGenerationConfig.from_config(OmegaConf.create(external_escape)),
        error_type=PathContractError,
        message_fragment="escapes its declared parent",
    )

    nested_timeline_typo = _copy(valid_multi_generation)
    nested_timeline_typo["generation"]["timeline"]["max_concurent"] = 4  # type: ignore[index]
    _expect_failure(
        "nested-generation-typo",
        lambda: PLCSGenerationConfig.from_config(
            OmegaConf.create(nested_timeline_typo)
        ),
        error_type=UnknownConfigurationKeyError,
        message_fragment="generation.timeline.max_concurent",
    )

    missing_tracking_metrics = _copy(valid_tracking)
    del missing_tracking_metrics["tracking_metrics"]
    _expect_failure(
        "missing-tracking-metrics",
        lambda: PLCSTrainingConfig.from_config(
            OmegaConf.create(missing_tracking_metrics)
        ),
        error_type=MissingConfigurationKeyError,
        message_fragment="configuration.tracking_metrics",
    )

    visualization_typo = _copy(valid_visualization)
    visualization_typo["visualization"]["style"]["show_shdaow"] = True  # type: ignore[index]
    _expect_failure(
        "nested-visualization-typo",
        lambda: validate_boundary(
            "plcs.visualize", OmegaConf.create(visualization_typo)
        ),
        error_type=UnknownConfigurationKeyError,
        message_fragment="visualization.style.show_shdaow",
    )

    null_canonical_source = _copy(valid_visualization)
    null_canonical_source["visualization"]["canonical_pose_source"] = None  # type: ignore[index]
    _expect_failure(
        "null-canonical-pose-source",
        lambda: validate_boundary(
            "plcs.visualize", OmegaConf.create(null_canonical_source)
        ),
        error_type=ConfigurationTypeError,
        message_fragment="visualization.canonical_pose_source",
    )


def main() -> int:
    """Run every PLCS negative case without starting a runtime workload."""
    run_negative_matrix()
    print("PLCS negative validation matrix: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
