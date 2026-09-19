"""CPU Lightning checks for epoch notification, loader reload and resume."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    default_collate,
)

from src.tasks.slcs.configuration import (
    SLCSDataRuntimeConfig,
    SLCSTrainingRuntimeConfig,
)
from src.tasks.slcs.data.datamodule import SLCSDataModule
from src.tasks.slcs.data.dataset import SLCSWindowDataset
from src.tasks.slcs.data.sampling import DomainBalancedSampler, DomainSamplingConfig
from src.utils.configuration.contracts import inspect_typed_adapter

_CONFIG_DIR = Path(__file__).parents[4] / "src/tasks/slcs/configs"


class _Windows(Dataset[int]):
    def __init__(self) -> None:
        self.metas = [SimpleNamespace(video_id=v) for v in ["a"] * 18 + ["b"] * 2]

    def __len__(self) -> int:
        return len(self.metas)

    def __getitem__(self, index: int) -> int:
        return index


@pytest.fixture
def runtime() -> SLCSDataRuntimeConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train")
    return replace(
        SLCSTrainingRuntimeConfig.from_config(config).data,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        domain_sampling=DomainSamplingConfig(True, {"a": "major", "b": "minor"}),
    )


def _module(
    runtime: SLCSDataRuntimeConfig, monkeypatch: pytest.MonkeyPatch
) -> SLCSDataModule:
    monkeypatch.setattr("src.tasks.slcs.data.datamodule.collate_slcs", default_collate)
    module = SLCSDataModule(runtime, seed=42)
    dataset = cast(SLCSWindowDataset, _Windows())
    module.train_dataset = dataset
    module.val_dataset = dataset
    module.test_dataset = dataset
    return module


def test_validation_sequential_disabled_legacy_rng_and_ddp_guard(
    runtime: SLCSDataRuntimeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module(runtime, monkeypatch)
    for loader in (module.val_dataloader(), module.test_dataloader()):
        assert isinstance(loader.sampler, SequentialSampler)
        assert list(loader.sampler) == list(range(20))
    for disabled in (None, DomainSamplingConfig(False, {})):
        module = _module(replace(runtime, domain_sampling=disabled), monkeypatch)
        torch.manual_seed(91)
        loader = module.train_dataloader()
        assert isinstance(loader.sampler, RandomSampler)
        actual = list(loader)
        state = torch.random.get_rng_state()
        torch.manual_seed(91)
        legacy = list(DataLoader(_Windows(), batch_size=4, shuffle=True))
        assert all(torch.equal(a, b) for a, b in zip(actual, legacy, strict=True))
        assert torch.equal(state, torch.random.get_rng_state())
    module = _module(runtime, monkeypatch)
    module.trainer = cast(pl.Trainer, SimpleNamespace(world_size=2))
    with pytest.raises(RuntimeError, match="distributed"):
        module.train_dataloader()
    module.trainer = None
    module.seed = None
    with pytest.raises(ValueError, match="run.seed"):
        module.train_dataloader()


class _Recorder(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.draws: dict[int, list[int]] = {}

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        self.draws.setdefault(self.current_epoch, []).extend(batch.tolist())
        return self.weight.square()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _trainer(epochs: int, reload: int) -> pl.Trainer:
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=epochs,
        reload_dataloaders_every_n_epochs=reload,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )


@pytest.mark.parametrize("reload", [0, 1])
def test_actual_trainer_epochs_and_checkpoint_resume(
    runtime: SLCSDataRuntimeConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reload: int,
) -> None:
    module = _module(runtime, monkeypatch)
    model = _Recorder()
    trainer = _trainer(2, reload)
    trainer.fit(model, datamodule=module)
    assert trainer.global_step == 10
    assert len(module.train_dataloader()) == 5
    path = tmp_path / "epoch.ckpt"
    trainer.save_checkpoint(path)
    resumed = _Recorder()
    resumed_trainer = _trainer(3, reload)
    resumed_trainer.fit(
        resumed, datamodule=_module(runtime, monkeypatch), ckpt_path=path
    )
    assert resumed_trainer.global_step == 15
    assert set(resumed.draws) == {2}
    sampler = DomainBalancedSampler(
        ["a"] * 18 + ["b"] * 2, {"a": "major", "b": "minor"}, seed=42
    )
    for epoch, draws in {**model.draws, **resumed.draws}.items():
        sampler.set_epoch(epoch)
        assert draws == list(sampler)
    assert model.draws[0] != model.draws[1]


def test_configuration_defaults_legacy_migration_and_strict_keys() -> None:
    inspect_typed_adapter(DomainSamplingConfig)
    inspect_typed_adapter(SLCSDataRuntimeConfig)
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train")
    assert SLCSTrainingRuntimeConfig.from_config(
        config
    ).data.domain_sampling == DomainSamplingConfig(False, {})
    with open_dict(config.data.domain_sampling):
        config.data.domain_sampling.typo = True
    with pytest.raises(ValueError, match="typo"):
        SLCSTrainingRuntimeConfig.from_config(config)
    with open_dict(config.data):
        del config.data.domain_sampling
    assert SLCSTrainingRuntimeConfig.from_config(config).data.domain_sampling is None
