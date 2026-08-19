"""CPU-only CUDA dispatch tests for compressed time-local attention."""

from __future__ import annotations

import importlib

import pytest

from src.utils.models.components.ops import loader
from src.utils.models.components.ops.compressed_time_local import api


def test_cuda_resolver_requires_extension_before_importing_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = False

    def unavailable() -> object:
        raise RuntimeError("compressed extension missing")

    def unexpected_import(name: str) -> object:
        nonlocal imported
        imported = True
        return importlib.import_module(name)

    monkeypatch.setattr(
        api, "require_compressed_time_local_cuda_extension", unavailable
    )
    monkeypatch.setattr(api, "import_module", unexpected_import)

    with pytest.raises(RuntimeError, match="compressed extension missing"):
        api.resolve_compressed_time_local_attention(
            "cuda", compression_ratio=4, window_radius=2
        )
    assert imported is False


def test_compressed_extension_loader_never_substitutes_another_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader.get_compressed_time_local_cuda_extension.cache_clear()

    def missing(name: str) -> object:
        assert name == loader.COMPRESSED_TIME_LOCAL_EXTENSION_NAME
        raise ImportError("missing test extension")

    monkeypatch.setattr(loader.importlib, "import_module", missing)
    try:
        assert loader.get_compressed_time_local_cuda_extension() is None
        with pytest.raises(RuntimeError, match="Compressed time-local CUDA"):
            loader.require_compressed_time_local_cuda_extension()
    finally:
        loader.get_compressed_time_local_cuda_extension.cache_clear()


def test_cuda_resolver_rejects_unsupported_window_before_extension_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = False

    def unexpected_load() -> object:
        nonlocal loaded
        loaded = True
        return object()

    monkeypatch.setattr(
        api, "require_compressed_time_local_cuda_extension", unexpected_load
    )

    with pytest.raises(ValueError, match="window_radius <= 64"):
        api.resolve_compressed_time_local_attention(
            "cuda", compression_ratio=4, window_radius=65
        )
    assert loaded is False
