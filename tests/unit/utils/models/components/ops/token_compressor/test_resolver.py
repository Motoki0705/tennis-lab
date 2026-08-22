"""CPU-only dispatch tests for token-compressor pooling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from src.utils.models.components.ops.token_compressor import api


def test_reference_resolution_does_not_import_triton_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = False

    def unexpected_import(_name: str) -> object:
        nonlocal imported
        imported = True
        raise AssertionError("reference resolution must remain Triton-lazy")

    monkeypatch.setattr(api, "import_module", unexpected_import)
    pool = api.resolve_token_compressor_pool(
        "reference", compression_ratio=3, head_dim=5
    )
    pooled, pooled_valid = pool(
        torch.randn(1, 2, 2, 5),
        torch.randn(1, 2, 2, 5),
        torch.ones(1, 2, dtype=torch.bool),
    )

    assert imported is False
    assert pooled.shape == (1, 1, 5)
    assert pooled_valid.shape == (1, 1)


def test_cuda_resolution_is_explicit_and_loads_only_cuda_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = torch.empty(0)

    def cuda_pool(
        raw_kv: Tensor,
        raw_gate: Tensor,
        state_valid: Tensor,
        *,
        compression_ratio: int,
    ) -> tuple[Tensor, Tensor]:
        del raw_kv, raw_gate, state_valid
        assert compression_ratio == 4
        return sentinel, sentinel

    imported: list[str] = []

    def fake_import(name: str) -> object:
        imported.append(name)
        return SimpleNamespace(cuda_token_compressor_pool=cuda_pool)

    monkeypatch.setattr(api, "import_module", fake_import)
    pool = api.resolve_token_compressor_pool("cuda", compression_ratio=4, head_dim=64)
    result = pool(torch.empty(0), torch.empty(0), torch.empty(0))

    assert result == (sentinel, sentinel)
    assert imported == ["src.utils.models.components.ops.token_compressor._triton"]


def test_cuda_unavailable_fails_without_reference_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_name: str) -> object:
        raise ImportError("missing Triton")

    monkeypatch.setattr(api, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="Triton is unavailable"):
        api.resolve_token_compressor_pool("cuda", compression_ratio=4, head_dim=64)


@pytest.mark.parametrize(
    ("backend", "ratio", "head_dim", "message"),
    [
        ("automatic", 4, 64, "Unsupported"),
        ("cuda", 3, 64, "compression_ratio=4"),
        ("cuda", 4, 32, "head_dim=64"),
    ],
)
def test_resolver_rejects_unsupported_configuration_before_import(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    ratio: int,
    head_dim: int,
    message: str,
) -> None:
    imported = False

    def unexpected_import(_name: str) -> object:
        nonlocal imported
        imported = True
        return object()

    monkeypatch.setattr(api, "import_module", unexpected_import)
    with pytest.raises(ValueError, match=message):
        api.resolve_token_compressor_pool(
            backend,  # type: ignore[arg-type]
            compression_ratio=ratio,
            head_dim=head_dim,
        )
    assert imported is False
