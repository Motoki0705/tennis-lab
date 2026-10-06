"""RoPE architecture, trajectory-only GAN and zero-noise experiment contracts."""

from dataclasses import asdict

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.ball_refiner.coordinates.config import (
    CorruptionConfig,
    LegacyModelConfig,
    ModelConfig,
    parse_gan,
    parse_section,
    training_config,
)
from src.tasks.ball_refiner.coordinates.corruption import coordinate_noise
from src.tasks.ball_refiner.coordinates.inference import (
    checkpoint_metadata,
    load_checkpoint,
)
from src.tasks.ball_refiner.coordinates.model import CoordinateRefiner
from src.tasks.ball_refiner.coordinates.models.discriminators import (
    build_refiner_discriminator,
)
from src.tasks.ball_refiner.coordinates.models.legacy import LegacyCoordinateRefiner
from src.tasks.ball_refiner.coordinates.review.contracts import Augmentation
from src.tasks.base.training.gan_loss import LSGANLoss
from src.utils.models.components import TransformerBlock, default_ffn_dim
from src.utils.paths import PROJECT_ROOT


def configuration():
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "src/tasks/ball_refiner/configs")):
        return compose(config_name="train_coordinates")


@pytest.mark.parametrize("dimensions", [2, 3])
def test_requested_architecture_scores_only_complete_output_and_backpropagates(dimensions):
    torch.set_num_threads(1)
    config = configuration()
    config.model.dimensions = dimensions
    raw, _, _ = training_config(config)
    generator = CoordinateRefiner(parse_section(ModelConfig, raw["model"])).eval()
    _, disc_config = parse_gan(raw["training"]["gan"])
    discriminator = build_refiner_discriminator(dimensions, disc_config).eval()
    assert len(generator.blocks) == 8 and len(discriminator.network.blocks) == 4
    assert generator.config.width == discriminator.network.hidden_dim == 256
    assert generator.config.ffn_dim == disc_config.ffn_dim == default_ffn_dim(256) == 704
    assert all(isinstance(block, TransformerBlock) for block in generator.blocks)
    coords = torch.randn(2, 128, dimensions)
    missing = torch.zeros(2, 128, dtype=torch.bool)
    missing[:, 48:65] = True
    coords[missing] = float("nan")
    output = generator(coords, missing)
    output.retain_grad()
    received = []
    discriminator.network.input_projection.register_forward_pre_hook(lambda _, args: received.append(args[0].detach().clone()))
    score = discriminator(output)
    assert score.shape == (2,)
    torch.testing.assert_close(received[0], output.detach())
    LSGANLoss().generator_loss(score).backward()
    assert torch.isfinite(output).all() and output.shape == coords.shape
    assert output.grad[missing].abs().sum() > 0 and output.grad[~missing].abs().sum() > 0
    assert generator.input.weight.grad.abs().sum() > 0
    # Neither an observed-input sequence nor a missing mask is part of D's API.
    with pytest.raises(TypeError):
        discriminator(output, coords, missing)


def test_zero_noise_is_exact_and_mixed_invalid_configuration_is_rejected():
    config = CorruptionConfig(0.5, 0.0, 3, 10, 0.0, 0.0, 0.0, 3)
    noise = coordinate_noise((4, 500), config, np.random.default_rng(4))
    assert noise.shape == (4, 500, 2) and np.count_nonzero(noise) == 0
    assert Augmentation(**asdict(config)).config() == config
    for change in ({"jitter_sigma_px": 3.0}, {"outlier_probability": 0.1}, {"noise_p95_px": -1.0}):
        with pytest.raises(ValueError):
            CorruptionConfig(**(asdict(config) | change))


@pytest.mark.parametrize("dimensions,architecture", [(2, "regression"), (3, "regression"), (3, "flow")])
def test_v1_checkpoints_restore_explicit_legacy_architecture(tmp_path, dimensions, architecture):
    model = LegacyCoordinateRefiner(LegacyModelConfig(dimensions, architecture, 16, 1, 2, 0.0, 32, 3)).eval()
    path = tmp_path / "v1.ckpt"
    payload = {**checkpoint_metadata(model), "model": model.state_dict()}
    torch.save(payload, path)
    loaded, _ = load_checkpoint(path, torch.device("cpu"))
    assert isinstance(loaded, LegacyCoordinateRefiner)
    coords = torch.randn(1, 16, dimensions)
    missing = torch.rand(1, 16) < 0.5
    first = model.predict(coords, missing, generator=torch.Generator().manual_seed(8))
    second = loaded.predict(coords, missing, generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    payload["schema"] = "ball_refiner.coordinates.v2"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="missing"):
        load_checkpoint(path, torch.device("cpu"))


def test_schedule_must_reach_target_and_defaults_have_only_event_corruption():
    config = configuration()
    raw = OmegaConf.to_container(config, resolve=True)
    corruption = parse_section(CorruptionConfig, raw["corruption"])
    assert corruption.noise_p95_px == corruption.jitter_sigma_px == corruption.outlier_probability == corruption.isolated_probability == 0
    config.training.gan.warmup_steps = config.training.steps
    with pytest.raises(ValueError, match="within training.steps"):
        training_config(config)
