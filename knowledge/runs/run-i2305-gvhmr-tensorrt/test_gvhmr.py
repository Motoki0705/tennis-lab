"""Check sequence geometry, locked input identity and error units without CUDA."""

import json

import numpy as np
import pytest
import torch
from accuracy_gvhmr import rotation_degrees
from locked_inputs import HERE, load_segments, tensor_addresses
from model import DenoiserIO, attention_mask, dynamic_attention, input_shapes

from src.submodules.vendor.gvhmr.network.relative_transformer import NetworkEncoderRoPE


@pytest.mark.parametrize(
    ("length", "allowed"),
    [
        (117, {0: (0, 117), 116: (0, 117)}),
        (120, {0: (0, 120), 119: (0, 120)}),
        (121, {0: (0, 120), 60: (0, 120), 61: (1, 121), 120: (1, 121)}),
        (177, {0: (0, 120), 78: (18, 138), 176: (57, 177)}),
    ],
)
def test_local_attention_keeps_the_production_boundary_windows(length, allowed):
    mask = attention_mask(length, torch.device("cpu")).numpy()
    for row, (start, end) in allowed.items():
        np.testing.assert_array_equal(np.flatnonzero(~mask[row]), np.arange(start, end))


def test_dynamic_attention_runs_the_original_blocks_with_length_derived_padding():
    torch.manual_seed(21)
    model = DenoiserIO(
        NetworkEncoderRoPE(latent_dim=32, num_heads=2, num_layers=1)
    ).eval()
    # A fresh block initializes its gates to zero, which would hide an incorrect mask.
    with torch.no_grad():
        model.network.blocks[0].gate_msa.fill_(1)
    inputs = [
        torch.randn(shape)
        for name, shape in input_shapes(121).items()
        if name != "length"
    ]
    length = torch.tensor([118])
    with torch.inference_mode():
        native = model(length, *inputs)
        with dynamic_attention(model):
            adjusted = model(length, *inputs)
    for left, right in zip(native, adjusted, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert adjusted[-1].tolist() == [118]
    assert adjusted[-2][0, -4:].tolist() == [True, False, False, False]
    assert model.network.max_len == 120


def test_profile_contains_exact_locked_lengths_and_distinct_representatives():
    profile = json.loads((HERE / "shape-profile.json").read_text())
    lock = json.loads((HERE / "tensor-lock.json").read_text())
    lengths = list(profile["profile_lengths"].values())
    assert lengths == [117, 145, 177]
    for name in profile["inputs"]:
        assert profile["inputs"][name] == [
            list(input_shapes(length)[name]) for length in lengths
        ]
    addressed = {row["id"]: row for row in lock["segments"]}
    for length, address in profile["representatives"].items():
        assert addressed[address]["length"] == int(length)
        assert addressed[address]["split"] == "eval"


def test_input_lock_rejects_same_shape_tensor_content_changes(tmp_path):
    arrays = {
        key: np.ones(shape, dtype=np.int64 if key == "length" else np.float32)
        for key, shape in input_shapes(2).items()
    }
    arrays["length"][0] = 2
    arrays["sequence_mask"] = np.ones((1, 2), dtype=bool)
    lock = {"segments": [{"id": "s", "length": 2, "tensors": tensor_addresses(arrays)}]}
    np.savez(tmp_path / "s.npz", **arrays)
    assert len(list(load_segments(tmp_path, lock))) == 1
    arrays["obs"][0, 1, 0, 2] = 0
    np.savez(tmp_path / "s.npz", **arrays)
    with pytest.raises(ValueError, match="Tensor lock mismatch"):
        list(load_segments(tmp_path, lock))


def test_so3_metric_respects_degrees_and_axis_angle_wrap():
    reference = np.array([[[0, 0, np.pi, 0, 0, 0]]])
    actual = np.array([[[0, 0, -np.pi, 0, np.pi / 3, 0]]])
    np.testing.assert_allclose(
        rotation_degrees(actual, reference), [[[0, 60]]], atol=1e-10
    )
