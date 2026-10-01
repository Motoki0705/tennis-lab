"""Learned appearance weights and external model source have distinct roots."""

from pathlib import Path
from typing import Any

import pytest

from src.tasks.player_association.appearance import encoders


@pytest.mark.parametrize('name', encoders.ENCODER_CANDIDATES)
def test_every_encoder_weight_uses_the_checkpoint_root(tmp_path: Path, name: str) -> None:
    checkpoint_root = tmp_path / 'weights'
    first = encoders.encoder_weights(name, checkpoint_root=checkpoint_root, external_root=tmp_path / 'source')
    other_source = encoders.encoder_weights(name, checkpoint_root=checkpoint_root, external_root=tmp_path / 'other-source')
    assert first == other_source
    assert first.is_relative_to(checkpoint_root)


def test_dinov3_keeps_source_external_and_weights_in_ckpt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_root, external_root = tmp_path / 'weights', tmp_path / 'source'
    path = encoders.encoder_weights('dinov3_vitb16', checkpoint_root=checkpoint_root, external_root=external_root)
    path.parent.mkdir(parents=True)
    path.touch()
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(encoders, 'DINOv3Encoder', lambda *args: calls.append(args))
    encoders.build_encoder('dinov3_vitb16', checkpoint_root=checkpoint_root, external_root=external_root, device='cpu')
    assert calls == [('dinov3_vitb16', 'dinov3_vitb16', external_root / 'dinov3', path, 'cpu')]


def test_missing_checkpoint_never_uses_an_external_copy(tmp_path: Path) -> None:
    external_root = tmp_path / 'source'
    previous = external_root / 'dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    previous.parent.mkdir(parents=True)
    previous.touch()
    with pytest.raises(FileNotFoundError, match='Required appearance checkpoint'):
        encoders.build_encoder('dinov3_vitb16', checkpoint_root=tmp_path / 'weights', external_root=external_root, device='cpu')


def test_unknown_encoder_fails_before_model_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='Unknown appearance encoder'):
        encoders.build_encoder('typo', checkpoint_root=tmp_path, external_root=tmp_path, device='cpu')
