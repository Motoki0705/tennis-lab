from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from src.tasks.blcs.configuration import (
    AxialModelConfig,
    TrackQueryAblationModelConfig,
    TrackQueryModelConfig,
    TrackQueryReferenceAblationModelConfig,
    TrackQueryReferenceModelConfig,
    parse_model_config,
    validate_training_boundary,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


@pytest.mark.parametrize(
    ("model_name", "num_layers"),
    [
        ("multiview_axial_small", 8),
        ("multiview_axial_base", 8),
        ("multiview_axial_large", 12),
        ("multiview_axial_xlarge", 12),
    ],
)
def test_axial_configs_preserve_local_temporal_attention_contract(
    model_name: str,
    num_layers: int,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train", overrides=[f"model={model_name}"])

    parsed = parse_model_config(config)

    assert isinstance(parsed, AxialModelConfig)
    assert parsed.time_window_radius == 16
    assert parsed.time_global_stage_mask == (False,) * num_layers


def test_legacy_non_track_training_config_needs_no_reference_only_data_key() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train")

    assert "evaluation_reference_camera_id" not in config.data
    validate_training_boundary(config)


@pytest.mark.parametrize(
    "config_name", ("train_tracking", "train_tracking_chunked")
)
def test_tracking_training_roots_preserve_effective_batch_at_maximum_length(
    config_name: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name=config_name)

    physical_batch_size = int(config.data.batch_size)
    accumulate_grad_batches = int(
        config.training.trainer.accumulate_grad_batches
    )
    assert physical_batch_size == 1
    assert accumulate_grad_batches == 8
    assert physical_batch_size * accumulate_grad_batches == 8
    assert list(config.data.num_views_range) == [3, 5]
    assert list(config.data.seq_len_range) == [512, 1024]
    assert config.training.trainer.precision == "bf16-mixed"
    assert isinstance(validate_training_boundary(config), TrackQueryModelConfig)


@pytest.mark.parametrize(
    "config_name", ("train_tracking", "train_tracking_chunked")
)
def test_tracking_training_roots_require_query_slot_lifecycle_packing(
    config_name: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name=config_name)

    with open_dict(config.data.lifecycle):
        config.data.lifecycle.pack_to_query_slots = False

    with pytest.raises(
        SemanticConfigurationError,
        match=r"data\.lifecycle\.pack_to_query_slots=true",
    ):
        validate_training_boundary(config)


@pytest.mark.parametrize(
    (
        "model_name",
        "hidden_dim",
        "num_heads",
        "num_stages",
        "ffn_dim",
        "rope_dim",
    ),
    [
        ("track_query_small", 256, 4, 8, 704, 64),
        ("track_query_base", 512, 8, 8, 1408, 64),
        ("track_query_large", 512, 8, 12, 1408, 64),
        ("track_query_xlarge", 1024, 8, 12, 2752, 128),
    ],
)
def test_track_query_size_configs_compose(
    model_name: str,
    hidden_dim: int,
    num_heads: int,
    num_stages: int,
    ffn_dim: int,
    rope_dim: int,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train_tracking", overrides=[f"model={model_name}"])

    assert config.model.name == "blcs_track_query"
    assert config.model.hidden_dim == hidden_dim
    assert config.model.num_heads == num_heads
    assert config.model.num_stages == num_stages
    assert config.model.ffn_dim == ffn_dim
    assert config.model.rope_dim == rope_dim
    assert config.model.mhc.coefficient_dim == 64
    assert config.model.cswa.compression_ratio == 4
    parsed = parse_model_config(config)
    assert isinstance(parsed, TrackQueryModelConfig)


def test_default_track_query_config_completes_one_cccg_cycle() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train_tracking")

    parsed = parse_model_config(config)
    assert isinstance(parsed, TrackQueryModelConfig)
    assert parsed.num_stages == 4
    assert parsed.mhc.sinkhorn_iters == 20
    assert parsed.cswa.backend == "reference"


@pytest.mark.parametrize(
    ("condition", "ffn_mode", "mhc_writeback", "query_ffn_after_spatial"),
    [
        ("a", "per_attention", "after_object_temporal", False),
        ("b", "shared", "after_object_temporal", False),
        ("c", "per_attention", "layer_end", False),
        ("d", "shared", "layer_end", False),
        ("e", "shared", "layer_end", True),
    ],
)
def test_all_five_track_query_ablation_configs_compose_and_validate(
    condition: str,
    ffn_mode: str,
    mhc_writeback: str,
    query_ffn_after_spatial: bool,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[f"model=track_query_ablation_{condition}"],
        )

    parsed = validate_training_boundary(config)

    assert isinstance(parsed, TrackQueryAblationModelConfig)
    assert parsed.name == "blcs_track_query_ablation"
    assert parsed.ffn_mode == ffn_mode
    assert parsed.mhc_writeback == mhc_writeback
    assert parsed.query_ffn_after_spatial is query_ffn_after_spatial
    assert parsed.hidden_dim == 512
    assert parsed.num_heads == 8
    assert parsed.num_stages == 12
    assert parsed.ffn_dim == 1408
    assert parsed.num_queries == 4
    assert parsed.rope_dim == 64
    assert parsed.dropout == 0.0
    assert parsed.cswa.compression_ratio == 4
    assert parsed.cswa.window_radius == 4
    assert parsed.cswa.backend == "cuda"


@pytest.mark.parametrize(
    ("violation", "error"),
    [
        ("missing_ffn", MissingConfigurationKeyError),
        ("missing_writeback", MissingConfigurationKeyError),
        ("missing_query_ffn", MissingConfigurationKeyError),
        ("unknown", UnknownConfigurationKeyError),
        ("invalid_ffn", SemanticConfigurationError),
        ("invalid_writeback", SemanticConfigurationError),
        ("invalid_query_ffn_type", ConfigurationTypeError),
        ("invalid_query_ffn_combination", SemanticConfigurationError),
    ],
)
def test_ablation_axes_reject_missing_unknown_and_invalid_values(
    violation: str,
    error: type[Exception],
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=["model=track_query_ablation_a"],
        )

    with open_dict(config.model):
        if violation == "missing_ffn":
            del config.model["ffn_mode"]
        elif violation == "missing_writeback":
            del config.model["mhc_writeback"]
        elif violation == "missing_query_ffn":
            del config.model["query_ffn_after_spatial"]
        elif violation == "unknown":
            config.model["legacy_ablation"] = True
        elif violation == "invalid_ffn":
            config.model.ffn_mode = "legacy"
        elif violation == "invalid_writeback":
            config.model.mhc_writeback = "before_spatial"
        elif violation == "invalid_query_ffn_type":
            config.model.query_ffn_after_spatial = "yes"
        else:
            config.model.query_ffn_after_spatial = True

    with pytest.raises(error):
        parse_model_config(config)


@pytest.mark.parametrize(
    ("violation", "error"),
    [
        ("missing", MissingConfigurationKeyError),
        ("unknown", UnknownConfigurationKeyError),
        ("stage_cycle", SemanticConfigurationError),
        ("compression", SemanticConfigurationError),
        ("backend", SemanticConfigurationError),
    ],
)
def test_track_query_nested_contract_rejects_missing_unknown_and_invalid_values(
    violation: str,
    error: type[Exception],
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train_tracking")

    if violation == "missing":
        with open_dict(config.model.mhc):
            del config.model.mhc["eps"]
    elif violation == "unknown":
        with open_dict(config.model.cswa):
            config.model.cswa["legacy_fallback"] = True
    elif violation == "stage_cycle":
        config.model.num_stages = 3
    elif violation == "compression":
        config.model.cswa.compression_ratio = 1
    else:
        config.model.cswa.backend = "auto"

    with pytest.raises(error):
        parse_model_config(config)


@pytest.mark.parametrize(
    ("profile", "expected_type", "selector_mode"),
    [
        ("track_query_reference", TrackQueryReferenceModelConfig, "reference"),
        (
            "track_query_ablation_d_v2_selector",
            TrackQueryReferenceAblationModelConfig,
            "reference",
        ),
        (
            "track_query_ablation_d_v2_selector_zero",
            TrackQueryReferenceAblationModelConfig,
            "selector_zero",
        ),
    ],
)
def test_reference_v2_profiles_compose_with_explicit_independent_contracts(
    profile: str,
    expected_type: type[object],
    selector_mode: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                f"model={profile}",
                "court_keypoints=camera_view_v2",
            ],
        )

    parsed = validate_training_boundary(config)
    assert isinstance(parsed, expected_type)
    assert isinstance(
        parsed,
        (TrackQueryReferenceModelConfig, TrackQueryReferenceAblationModelConfig),
    )
    assert parsed.target_frame_contract == "reference_camera_court_rzpi_v1"
    assert parsed.track_query_rope_contract == "time_camera_reference_selector_v1"
    assert parsed.reference_selector_mode == selector_mode
    assert "role_rope_enabled" not in config.model


@pytest.mark.parametrize(
    "profile",
    ["track_query_reference", "track_query_ablation_d_v2_selector"],
)
def test_reference_v2_rejects_dim4_role_reinterpretation_and_physical_court(
    profile: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[f"model={profile}", "court_keypoints=camera_view_v2"],
        )
    config.model.rope_dim = 4
    with pytest.raises(SemanticConfigurationError, match="at least 6"):
        parse_model_config(config)

    config.model.rope_dim = 6
    with open_dict(config.model):
        config.model.role_rope_enabled = True
    with pytest.raises(UnknownConfigurationKeyError, match="role_rope_enabled"):
        parse_model_config(config)

    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        physical = compose(
            config_name="train_tracking",
            overrides=[f"model={profile}"],
        )
    with pytest.raises(SemanticConfigurationError, match="camera_view_v2"):
        validate_training_boundary(physical)
