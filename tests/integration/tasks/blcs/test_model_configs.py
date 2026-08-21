from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from src.tasks.blcs.configuration import (
    TrackQueryModelConfig,
    parse_model_config,
    validate_training_boundary,
)
from src.utils.configuration import (
    MissingConfigurationKeyError,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


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
