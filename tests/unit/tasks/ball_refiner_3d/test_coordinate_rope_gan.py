"""RoPE architecture, trajectory-only GAN and zero-noise experiment contracts."""

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.ball_refiner_3d.configuration.core import parse_section
from src.tasks.ball_refiner_3d.configuration.data import CorruptionConfig
from src.tasks.ball_refiner_3d.configuration.model import ModelConfig
from src.tasks.ball_refiner_3d.configuration.training import parse_gan, training_config
from src.tasks.ball_refiner_3d.data.augmentation.noise import coordinate_noise
from src.tasks.ball_refiner_3d.model_io.checkpoint import (
    checkpoint_metadata,
    load_checkpoint,
)
from src.tasks.ball_refiner_3d.model_io.factory import build_refiner
from src.tasks.ball_refiner_3d.models.discriminators import (
    build_refiner_discriminator,
)
from src.tasks.ball_refiner_3d.visualization.dataset_review.contracts import (
    Augmentation,
)
from src.tasks.base.training.gan_loss import LSGANLoss
from src.utils.models.components import TransformerBlock, default_ffn_dim
from src.utils.paths import PROJECT_ROOT


def configuration(overrides=()):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(PROJECT_ROOT / "src/tasks/ball_refiner_3d/configs"),
    ):
        return compose(config_name="train", overrides=list(overrides))


@pytest.mark.parametrize("dimensions", [3])
def test_requested_architecture_scores_only_complete_output_and_backpropagates(
    dimensions,
):
    torch.set_num_threads(1)
    config = configuration()
    config.model.dimensions = dimensions
    raw = training_config(config).raw
    generator = build_refiner(parse_section(ModelConfig, raw["model"])).eval()
    _, disc_config = parse_gan(raw["training"]["gan"])
    discriminator = build_refiner_discriminator(dimensions, disc_config).eval()
    assert len(generator.blocks) == 8 and len(discriminator.network.blocks) == 4
    assert generator.config.width == discriminator.network.hidden_dim == 256
    assert (
        generator.config.ffn_dim == disc_config.ffn_dim == default_ffn_dim(256) == 704
    )
    assert all(isinstance(block, TransformerBlock) for block in generator.blocks)
    coords = torch.randn(2, 128, dimensions)
    missing = torch.zeros(2, 128, dtype=torch.bool)
    missing[:, 48:65] = True
    coords[missing] = float("nan")
    output = generator(coords, missing).coordinates
    output.retain_grad()
    received = []
    discriminator.network.input_projection.register_forward_pre_hook(
        lambda _, args: received.append(args[0].detach().clone())
    )
    score = discriminator(output)
    assert score.shape == (2,)
    torch.testing.assert_close(received[0], output.detach())
    LSGANLoss().generator_loss(score).backward()
    assert torch.isfinite(output).all() and output.shape == coords.shape
    assert (
        output.grad[missing].abs().sum() > 0 and output.grad[~missing].abs().sum() > 0
    )
    assert generator.input.weight.grad.abs().sum() > 0
    # Neither an observed-input sequence nor a missing mask is part of D's API.
    with pytest.raises(TypeError):
        discriminator(output, coords, missing)


def test_zero_noise_is_exact_and_mixed_invalid_configuration_is_rejected():
    config = CorruptionConfig(0.5, 0.0, 3, 10, 0.0, 0.0, 0.0, 3)
    noise = coordinate_noise((4, 500), config, np.random.default_rng(4))
    assert noise.shape == (4, 500, 2) and np.count_nonzero(noise) == 0
    assert Augmentation(**asdict(config)).config() == config
    for change in (
        {"jitter_sigma_px": 3.0},
        {"outlier_probability": 0.1},
        {"noise_p95_px": -1.0},
    ):
        with pytest.raises(ValueError):
            CorruptionConfig(**(asdict(config) | change))


def test_schedule_must_reach_target_and_defaults_have_only_event_corruption():
    config = configuration()
    raw = OmegaConf.to_container(config, resolve=True)
    corruption = parse_section(CorruptionConfig, raw["augmentation"])
    assert (
        corruption.noise_p95_px
        == corruption.jitter_sigma_px
        == corruption.outlier_probability
        == corruption.isolated_probability
        == 0
    )
    config.training.gan.enabled = True
    config.training.gan.schedule_enabled = True
    config.training.gan.warmup_steps = config.training.steps
    with pytest.raises(ValueError, match="must finish within"):
        training_config(config)


@pytest.mark.parametrize("ffn_type", ["swiglu", "mlp"])
def test_event_checkpoint_configured_ffn_roundtrips_with_checkpoint(tmp_path, ffn_type):
    config = ModelConfig(
        3, "regression", 16, 1, 2, 0.0, 32, 3, 64, 8, 10000.0, ffn_type
    )
    model = build_refiner(config).eval()
    path = tmp_path / "v3.ckpt"
    metadata = checkpoint_metadata(model, event_sigma_frames=2.0)
    assert metadata["schema"] == "ball_refiner_3d.events.v1"
    torch.save({**metadata, "model": model.state_dict()}, path)
    loaded, _ = load_checkpoint(path, torch.device("cpu"))
    assert loaded.config == config
    coords, missing = torch.randn(1, 16, 3), torch.zeros(1, 16, dtype=torch.bool)
    torch.testing.assert_close(
        model(coords, missing), loaded(coords, missing), atol=0, rtol=0
    )
    with pytest.raises(ValueError, match="Unsupported FFN"):
        replace(config, ffn_type="invalid")


@pytest.mark.parametrize("architecture", ["regression", "flow"])
def test_forward_can_compile_without_moving_validation_into_tensor_graph(architecture):
    from src.tasks.ball_refiner_3d.model_io.factory import bind_refiner

    torch.set_num_threads(1)
    model = build_refiner(
        ModelConfig(3, architecture, 16, 1, 2, 0.0, 16, 3, 64, 8, 10000.0, "swiglu")
    ).eval()
    binding = bind_refiner(model)
    batch = {
        "coordinates": torch.randn(2, 16, 3),
        "missing": torch.zeros(2, 16, dtype=torch.bool),
    }
    if architecture == "flow":
        batch.update(state=torch.randn(2, 16, 3), time=torch.full((2,), 0.5))
    with torch.no_grad():
        before = binding.run(batch)
        model.compile(backend="eager", fullgraph=True)
        after = binding.run(batch)
    torch.testing.assert_close(after, before, atol=0, rtol=0)
    batch["coordinates"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        binding.run(batch)


@pytest.mark.parametrize(
    "override",
    [
        "training.steps_per_epoch=10",
        "training.gan.warmup_epochs=3",
        "training.gan.generator_gradient_clip_val=0.5",
        "run.gpus=2",
    ],
)
def test_unused_derived_training_overrides_are_rejected(override):
    with pytest.raises(ValueError):
        training_config(configuration([override]))
