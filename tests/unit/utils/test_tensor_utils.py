"""Unit tests for :mod:`src.utils.tensor_utils`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.utils.tensor_utils import (
    clone_tensor_dict,
    masked_mean,
    normalize_padding_mask,
    to_numpy,
)


class TestCloneTensorDict:
    def test_tensors_are_cloned(self) -> None:
        original = {"x": torch.ones(3), "name": "ball"}
        cloned = clone_tensor_dict(original)
        cloned["x"][0] = 99.0
        assert original["x"][0] == 1.0  # original untouched
        assert cloned["x"] is not original["x"]

    def test_non_tensors_passed_by_reference(self) -> None:
        payload = [1, 2, 3]
        original = {"meta": payload}
        cloned = clone_tensor_dict(original)
        assert cloned["meta"] is payload

    def test_returns_plain_dict(self) -> None:
        assert isinstance(clone_tensor_dict({"x": torch.zeros(1)}), dict)


class TestToNumpy:
    def test_detaches_and_converts(self) -> None:
        t = torch.ones(2, 2, requires_grad=True) * 2
        arr = to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, np.full((2, 2), 2.0))

    def test_bfloat16_upcast(self) -> None:
        t = torch.ones(3, dtype=torch.bfloat16)
        arr = to_numpy(t)
        assert arr.dtype == np.float32

    def test_dtype_override(self) -> None:
        arr = to_numpy(torch.ones(3), dtype=np.int64)
        assert arr.dtype == np.int64

    def test_array_like_passthrough(self) -> None:
        np.testing.assert_array_equal(to_numpy([1, 2, 3]), np.array([1, 2, 3]))


class TestMaskedMean:
    def test_none_mask_is_plain_mean(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert masked_mean(values).item() == pytest.approx(2.5)

    def test_basic_masked_mean(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert masked_mean(values, mask).item() == pytest.approx(2.0)

    def test_binarize_treats_mask_as_boolean(self) -> None:
        values = torch.tensor([10.0, 20.0])
        mask = torch.tensor([0.5, 0.0])
        assert masked_mean(values, mask, binarize=True).item() == pytest.approx(10.0)

    def test_broadcast_lower_rank_mask(self) -> None:
        values = torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        mask = torch.tensor([1.0, 0.0])  # keep only the first row
        assert masked_mean(values, mask, broadcast=True).item() == pytest.approx(1.0)

    def test_denom_min_prevents_div_by_zero(self) -> None:
        values = torch.tensor([5.0, 5.0])
        mask = torch.zeros(2)
        out = masked_mean(values, mask, denom_min=1.0)
        assert out.item() == pytest.approx(0.0)
        assert torch.isfinite(out)


class TestNormalizePaddingMask:
    def test_none_returns_none(self) -> None:
        assert normalize_padding_mask(None) is None

    def test_1d_and_2d_passthrough(self) -> None:
        mask = torch.tensor([[1, 0], [0, 1]])
        out = normalize_padding_mask(mask)
        assert out.dtype == torch.bool
        assert out.tolist() == [[True, False], [False, True]]

    def test_3d_reduces_over_n(self) -> None:
        # (B=1, N=2, T=3) -> any over N
        mask = torch.tensor([[[1, 0, 0], [0, 0, 1]]])
        out = normalize_padding_mask(mask)
        assert out.shape == (1, 3)
        assert out.tolist() == [[True, False, True]]

    def test_4d_reduces_over_n_and_j(self) -> None:
        mask = torch.zeros(2, 3, 4, 5)
        mask[0, 0, 1, 0] = 1
        out = normalize_padding_mask(mask)
        assert out.shape == (2, 4)
        assert bool(out[0, 1])

    def test_flatten(self) -> None:
        mask = torch.tensor([[1, 0], [1, 1]])
        out = normalize_padding_mask(mask, flatten=True)
        assert out.shape == (4,)

    def test_invalid_rank_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_padding_mask(torch.zeros(2, 3, 4, 5, 6))
