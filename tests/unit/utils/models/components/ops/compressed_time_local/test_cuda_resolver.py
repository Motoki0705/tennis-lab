"""CPU-only CUDA dispatch tests for compressed time-local attention."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import NoReturn

import pytest
import torch
from torch import Tensor

from src.utils.models.components.ops import loader
from src.utils.models.components.ops.compressed_time_local import api

_CUDA_AUTOGRAD_MODULE = (
    "src.utils.models.components.ops.compressed_time_local._autograd"
)
_INVALID_ROW_MESSAGE = "valid query has no valid compressed key in its local window"


def _is_alias_of(left: Tensor, right: Tensor) -> bool:
    return bool(torch._C._is_alias_of(left, right))  # type: ignore[attr-defined]


@pytest.fixture
def cuda_autograd_module(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[ModuleType]:
    previous_module = sys.modules.pop(_CUDA_AUTOGRAD_MODULE, None)

    def unexpected_extension_load() -> object:
        raise AssertionError("importing the CUDA executor must not load the extension")

    monkeypatch.setattr(
        loader,
        "require_compressed_time_local_cuda_extension",
        unexpected_extension_load,
    )
    try:
        yield importlib.import_module(_CUDA_AUTOGRAD_MODULE)
    finally:
        sys.modules.pop(_CUDA_AUTOGRAD_MODULE, None)
        if previous_module is not None:
            sys.modules[_CUDA_AUTOGRAD_MODULE] = previous_module


def _stub_cuda_forward(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    *,
    invalid_row: Tensor,
) -> tuple[Tensor, tuple[Tensor, ...], dict[str, object]]:
    output = torch.randn(1, 1, 1, 1)

    def accept_inputs(*_args: object, **_kwargs: object) -> None:
        return None

    def fake_forward(*args: object) -> tuple[Tensor, Tensor, Tensor]:
        assert all(
            not isinstance(argument, Tensor) or argument.is_contiguous()
            for argument in args[1:5]
        )
        logsumexp = torch.empty(1, 1, 1, dtype=torch.float32)
        return output, logsumexp, invalid_row

    monkeypatch.setattr(module, "_validate_inputs", accept_inputs)
    monkeypatch.setattr(module, "compressed_time_local_forward", fake_forward)
    inputs = (torch.randn(1), torch.randn(1), torch.randn(1))
    kwargs: dict[str, object] = {
        "query_valid": torch.ones(1, dtype=torch.bool),
        "key_valid": torch.ones(1, dtype=torch.bool),
        "compression_ratio": 2,
        "window_radius": 0,
    }
    return output, inputs, kwargs


def _forbid_tensor_scalar_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_item(_tensor: Tensor) -> NoReturn:
        raise AssertionError("Tensor.item() must not inspect the device flag")

    def fail_int(_tensor: Tensor) -> NoReturn:
        raise AssertionError("int(Tensor) must not inspect the device flag")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)
    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)


def test_cuda_executor_import_is_extension_lazy(
    cuda_autograd_module: ModuleType,
) -> None:
    assert callable(cuda_autograd_module.cuda_compressed_time_local_attention)


@pytest.mark.parametrize("rank", [3, 4])
def test_rope_phasors_normalize_as_non_materialized_float32_stride_views(
    cuda_autograd_module: ModuleType,
    rank: int,
) -> None:
    angles = torch.randn(2, 9, 3, 16)
    frequencies = torch.polar(torch.ones_like(angles), angles)[..., ::2]
    if rank == 3:
        frequencies = frequencies[0, :, :1]
        batch_size = 2
        heads = 3
    else:
        batch_size = 2
        heads = 3

    normalized = cuda_autograd_module._normalize_rope_phasors(
        frequencies,
        name="freqs",
        batch_size=batch_size,
        heads=heads,
        sequence_length=9,
        head_dim=16,
        device=torch.device("cpu"),
    )

    assert normalized.shape == ((1, 9, 1, 8, 2) if rank == 3 else (2, 9, 3, 8, 2))
    assert normalized.dtype == torch.float32
    assert normalized.stride(-1) == 1
    assert normalized.stride(-2) == frequencies.stride(-1) * 2
    assert _is_alias_of(normalized, frequencies)


@pytest.mark.parametrize(
    ("make_frequencies", "error_type", "message"),
    [
        (
            lambda: torch.ones(9, 1, 8, dtype=torch.complex128),
            TypeError,
            "dtype complex64",
        ),
        (
            lambda: torch.ones(9, 1, 8, dtype=torch.complex64).requires_grad_(),
            ValueError,
            "must not require gradients",
        ),
        (
            lambda: torch.ones(9, 2, 8, dtype=torch.complex64),
            ValueError,
            "head dimension",
        ),
        (
            lambda: torch.ones(8, 1, 8, dtype=torch.complex64),
            ValueError,
            "sequence dimension",
        ),
        (
            lambda: torch.ones(9, 1, 7, dtype=torch.complex64),
            ValueError,
            "pair dimension",
        ),
        (
            lambda: torch.ones(3, 9, 1, 8, dtype=torch.complex64),
            ValueError,
            "batch dimension",
        ),
        (
            lambda: torch.ones(9, 1, 8, dtype=torch.complex64, device="meta"),
            ValueError,
            "on device cpu",
        ),
        (
            lambda: torch.ones(2, 2, 9, 1, 8, dtype=torch.complex64),
            ValueError,
            "rank 3 or 4",
        ),
    ],
)
def test_rope_phasor_normalization_fails_closed(
    cuda_autograd_module: ModuleType,
    make_frequencies: Callable[[], Tensor],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        cuda_autograd_module._normalize_rope_phasors(
            make_frequencies(),
            name="freqs",
            batch_size=2,
            heads=3,
            sequence_length=9,
            head_dim=16,
            device=torch.device("cpu"),
        )


def test_rope_phasor_pair_must_be_complete(
    cuda_autograd_module: ModuleType,
) -> None:
    query = torch.empty(2, 3, 9, 16)
    key = torch.empty(2, 1, 3, 16)

    with pytest.raises(ValueError, match="both be provided"):
        cuda_autograd_module._normalize_rope_pair(
            query,
            key,
            torch.ones(9, 1, 8, dtype=torch.complex64),
            None,
        )


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


def test_cuda_wrapper_schedules_device_assert_without_scalar_conversion(
    monkeypatch: pytest.MonkeyPatch,
    cuda_autograd_module: ModuleType,
) -> None:
    invalid_row = torch.zeros(1, dtype=torch.int32)
    output, inputs, kwargs = _stub_cuda_forward(
        cuda_autograd_module, monkeypatch, invalid_row=invalid_row
    )
    scheduled: list[tuple[Tensor, str]] = []

    def record_assert(condition: Tensor, message: str) -> None:
        scheduled.append((condition, message))

    _forbid_tensor_scalar_conversion(monkeypatch)
    monkeypatch.setattr(torch, "_assert_async", record_assert)

    actual = cuda_autograd_module.cuda_compressed_time_local_attention(
        *inputs, **kwargs
    )

    assert actual is output
    assert len(scheduled) == 1
    condition, message = scheduled[0]
    assert torch.equal(condition, torch.ones(1, dtype=torch.bool))
    assert condition.device == invalid_row.device
    assert message == _INVALID_ROW_MESSAGE


def test_cuda_wrapper_preserves_dense_query_layout_while_normalizing_other_inputs(
    monkeypatch: pytest.MonkeyPatch,
    cuda_autograd_module: ModuleType,
) -> None:
    query = torch.randn(2, 9, 3, 4).transpose(1, 2)
    key = torch.randn(2, 1, 3, 8)[..., ::2]
    value = torch.randn(2, 1, 3, 8)[..., ::2]
    query_valid = torch.ones(2, 18, dtype=torch.bool)[:, ::2]
    key_valid = torch.ones(2, 6, dtype=torch.bool)[:, ::2]
    observed: list[tuple[Tensor, ...]] = []

    def accept_inputs(*_args: object, **_kwargs: object) -> None:
        return None

    def fake_forward(*args: object) -> tuple[Tensor, Tensor, Tensor]:
        tensors = tuple(argument for argument in args if isinstance(argument, Tensor))
        observed.append(tensors)
        return (
            torch.empty_like(query),
            torch.empty(query.shape[:3], dtype=torch.float32),
            torch.zeros(1, dtype=torch.int32),
        )

    monkeypatch.setattr(cuda_autograd_module, "_validate_inputs", accept_inputs)
    monkeypatch.setattr(
        cuda_autograd_module, "compressed_time_local_forward", fake_forward
    )

    cuda_autograd_module.cuda_compressed_time_local_attention(
        query,
        key,
        value,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=4,
        window_radius=1,
    )

    assert len(observed) == 1
    (
        forwarded_query,
        forwarded_key,
        forwarded_value,
        forwarded_q_mask,
        forwarded_k_mask,
    ) = observed[0]
    assert forwarded_query is query
    assert forwarded_query.stride() == (108, 4, 12, 1)
    assert all(
        tensor.is_contiguous()
        for tensor in (
            forwarded_key,
            forwarded_value,
            forwarded_q_mask,
            forwarded_k_mask,
        )
    )


def test_cuda_wrapper_forwards_noncontiguous_phasors_as_read_only_real_views(
    monkeypatch: pytest.MonkeyPatch,
    cuda_autograd_module: ModuleType,
) -> None:
    query = torch.randn(2, 3, 9, 16)
    key = torch.randn(2, 1, 3, 16)
    value = torch.randn_like(key)
    query_valid = torch.ones(2, 9, dtype=torch.bool)
    key_valid = torch.ones(2, 3, dtype=torch.bool)
    query_storage = torch.polar(torch.ones(2, 3, 9, 16), torch.randn(2, 3, 9, 16))
    key_storage = torch.polar(torch.ones(2, 1, 3, 16), torch.randn(2, 1, 3, 16))
    query_freqs_cis = query_storage.transpose(1, 2)[..., ::2]
    key_freqs_cis = key_storage.transpose(1, 2)[..., ::2]
    observed: list[tuple[object, ...]] = []

    def accept_inputs(*_args: object, **_kwargs: object) -> None:
        return None

    def fake_forward(*args: object) -> tuple[Tensor, Tensor, Tensor]:
        observed.append(args)
        return (
            torch.empty_like(query),
            torch.empty(query.shape[:3], dtype=torch.float32),
            torch.zeros(1, dtype=torch.int32),
        )

    monkeypatch.setattr(cuda_autograd_module, "_validate_inputs", accept_inputs)
    monkeypatch.setattr(
        cuda_autograd_module, "compressed_time_local_forward", fake_forward
    )

    cuda_autograd_module.cuda_compressed_time_local_attention(
        query,
        key,
        value,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=4,
        window_radius=1,
        query_freqs_cis=query_freqs_cis,
        key_freqs_cis=key_freqs_cis,
    )

    assert len(observed) == 1
    query_phasors_real = observed[0][5]
    key_phasors_real = observed[0][6]
    assert isinstance(query_phasors_real, Tensor)
    assert isinstance(key_phasors_real, Tensor)
    assert query_phasors_real.shape == (2, 9, 3, 8, 2)
    assert key_phasors_real.shape == (2, 3, 1, 8, 2)
    assert query_phasors_real.dtype == key_phasors_real.dtype == torch.float32
    assert _is_alias_of(query_phasors_real, query_freqs_cis)
    assert _is_alias_of(key_phasors_real, key_freqs_cis)
    assert not query_phasors_real.is_contiguous()
    assert not key_phasors_real.is_contiguous()


def test_cuda_wrapper_cpu_probe_raises_invalid_row_contract_without_scalar_conversion(
    monkeypatch: pytest.MonkeyPatch,
    cuda_autograd_module: ModuleType,
) -> None:
    invalid_row = torch.ones(1, dtype=torch.int32)
    _, inputs, kwargs = _stub_cuda_forward(
        cuda_autograd_module, monkeypatch, invalid_row=invalid_row
    )
    _forbid_tensor_scalar_conversion(monkeypatch)

    with pytest.raises(RuntimeError, match=_INVALID_ROW_MESSAGE):
        cuda_autograd_module.cuda_compressed_time_local_attention(*inputs, **kwargs)
