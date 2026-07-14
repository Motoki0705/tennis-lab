"""Tests for root setup.py CUDA source preparation."""

from pathlib import Path

import pytest

from src.utils.models.components.ops.build import _prepare_dino_ops_sources

_LEGACY_DISPATCH = "AT_DISPATCH_FLOATING_TYPES(value.type(),"
_MODERN_DISPATCH = "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),"


def test_prepare_dino_ops_sources_patches_copy_only(tmp_path: Path) -> None:
    source = tmp_path / "third_party_src"
    cuda_source = source / "cuda/ms_deform_attn_cuda.cu"
    cuda_source.parent.mkdir(parents=True)
    original = f"{_LEGACY_DISPATCH}\n{_LEGACY_DISPATCH}\n"
    cuda_source.write_text(original)
    destination = tmp_path / "build_src"

    result = _prepare_dino_ops_sources(source, destination)

    assert result == destination
    assert cuda_source.read_text() == original
    generated = (destination / "cuda/ms_deform_attn_cuda.cu").read_text()
    assert _LEGACY_DISPATCH not in generated
    assert generated.count(_MODERN_DISPATCH) == 2


def test_prepare_dino_ops_sources_requires_initialized_submodule(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="git submodule update --init"):
        _prepare_dino_ops_sources(tmp_path / "missing", tmp_path / "build")
