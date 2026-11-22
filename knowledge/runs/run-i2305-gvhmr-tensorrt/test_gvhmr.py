"""Check sequence geometry, locked input identity and error units without CUDA."""

import json
from types import SimpleNamespace

import experiment
import numpy as np
import pytest
import torch
from accuracy_gvhmr import checked_measurement, rotation_degrees
from convert_gvhmr import OPSET, calibration_inputs, export_index_put
from inputs import sha256_file
from locked_inputs import HERE, load_segments, sweep_profile, tensor_addresses
from model import (
    DenoiserIO,
    attention_mask,
    dynamic_attention,
    input_shapes,
    pack_sequences,
)
from run_gvhmr import seq_stats
from torch.onnx._internal.torchscript_exporter.jit_utils import GraphContext

from src.submodules.vendor.gvhmr.network.relative_transformer import NetworkEncoderRoPE


def test_gvhmr_batch_profile_keeps_the_two_evaluation_lengths():
    profile = sweep_profile(4, 4)
    assert profile["profile_batches"] == {"min": 1, "opt": 4, "max": 4}
    assert profile["inputs"]["length"] == [[1], [4], [4]]
    assert profile["inputs"]["obs"] == [
        [1, 145, 17, 3],
        [4, 145, 17, 3],
        [4, 177, 17, 3],
    ]


def test_gvhmr_fixed_length_profiles_keep_only_the_batch_axis_dynamic():
    for frames in (145, 177):
        profile = sweep_profile(4, 4, fixed_length=frames)
        assert profile["profile_lengths"] == dict.fromkeys(
            ("min", "opt", "max"), frames
        )
        assert profile["inputs"]["obs"] == [
            [batch, frames, 17, 3] for batch in (1, 4, 4)
        ]
        assert profile["outside_lengths"] == [frames - 1, frames + 1]
    with pytest.raises(ValueError, match="No locked"):
        sweep_profile(4, 4, fixed_length=146)


def test_gvhmr_person_batches_preserve_frames_and_refuse_mixed_lengths():
    def person(frames, marker):
        arrays = {
            name: np.full(shape, marker, dtype=np.float32)
            for name, shape in input_shapes(frames).items()
        }
        arrays["length"] = np.array([frames], dtype=np.int64)
        return arrays

    people = [person(145, index) for index in range(4)]
    batch = pack_sequences(people)
    assert batch["length"].tolist() == [145] * 4
    np.testing.assert_array_equal(
        batch["obs"], np.repeat(np.arange(4), 145 * 17 * 3).reshape(4, 145, 17, 3)
    )
    with pytest.raises(ValueError, match="same complete frame length"):
        pack_sequences([people[0], person(177, 1)])
    result = seq_stats([2.0, 6.0], frames=145, batch_size=4)
    assert result["sequences_per_second"] == 1000
    assert result["frames_per_second"] == 145_000


def test_gvhmr_reference_cannot_be_replaced_with_batched_fp32(tmp_path):
    rows = [{"id": "person0"}, {"id": "person1"}]
    (tmp_path / "measurement.json").write_text(
        json.dumps(
            {
                "condition": "0",
                "batch_size": 2,
                "length": 145,
                "segments": [row["id"] for row in rows],
                "tensor_lock_sha256": sha256_file(HERE / "tensor-lock.json"),
            }
        )
    )
    assert (
        checked_measurement(tmp_path, "0", 145, rows, batch_size=2)["batch_size"] == 2
    )
    with pytest.raises(ValueError, match="input identity"):
        checked_measurement(tmp_path, "0", 145, rows)


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


def test_traced_gvhmr_padding_mask_uses_the_runtime_batch():
    model = DenoiserIO(
        NetworkEncoderRoPE(latent_dim=32, num_heads=2, num_layers=1)
    ).eval()
    inputs = tuple(
        torch.randn(shape)
        for name, shape in input_shapes(121, 4).items()
        if name != "length"
    )
    lengths = torch.tensor([118, 119, 120, 121])
    two_people = (lengths[:2], *(tensor[:2] for tensor in inputs))
    with torch.inference_mode():
        expected = model(*two_people)
        with dynamic_attention(model, dynamic_batch=True):
            traced = torch.jit.trace(model, (lengths, *inputs))
            actual = traced(*two_people)
    assert actual[-1].tolist() == [118, 119]
    assert actual[-2][:, -3:].tolist() == [[False, False, False], [True, False, False]]
    torch.testing.assert_close(actual, expected)


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


def test_ptq_provider_arrays_preserve_each_sequence_and_use_batch_one():
    first, second = [
        {
            name: np.array([2], dtype=np.int64)
            if name == "length"
            else np.full(shape, value, dtype=np.float32)
            for name, shape in input_shapes(2).items()
        }
        for value in (1, 3)
    ]
    data, shapes = calibration_inputs([first, second])
    assert (
        shapes
        == "length:1,obs:1x2x17x3,f_cliffcam:1x2x3,f_cam_angvel:1x2x6,f_imgseq:1x2x1024"
    )
    np.testing.assert_array_equal(data["length"], [2, 2])
    for index, sample in enumerate((first, second)):
        np.testing.assert_array_equal(data["obs"][index : index + 1], sample["obs"])
    second["length"] = np.array([1], dtype=np.int64)
    with pytest.raises(ValueError, match="one locked sequence length"):
        calibration_inputs([first, second])


def test_obs_boolean_index_export_keeps_coordinate_axis():
    graph = torch._C.Graph()
    obs = graph.addInput().setType(
        torch._C.TensorType.create_from_tensor(torch.zeros(1, 145, 17, 3))
    )
    mask = graph.addInput().setType(
        torch._C.TensorType.create_from_tensor(
            torch.zeros(1, 145, 17, dtype=torch.bool)
        )
    )
    indices = graph.insertNode(graph.create("prim::ListConstruct", [mask], 1))
    context = GraphContext(graph, graph.block(), OPSET, indices, {}, {}, set())
    value = context.op("Constant", value_t=torch.tensor(0.0))

    where = export_index_put(context, obs, indices.output(), value).node()
    assert where.kind() == "onnx::Where"
    broadcast_mask, _, unmasked = list(where.inputs())
    assert unmasked is obs
    unsqueeze = next(broadcast_mask.node().inputs()).node()
    assert unsqueeze.kind() == "onnx::Unsqueeze"
    original, axis = list(unsqueeze.inputs())
    assert original is mask
    assert axis.node().t("value").tolist() == [-1]


@pytest.mark.parametrize("replayed", [867, 866])
def test_ncu_retry_uses_saved_nsys_coverage(replayed, tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "retry"
    output.mkdir()
    timing = source / "measurement.json"
    timing.write_text(
        json.dumps(
            {
                "condition": "0",
                "length": 177,
                "tensor_lock_sha256": sha256_file(HERE / "tensor-lock.json"),
            }
        )
    )
    (source / "profiles.json").write_text(
        json.dumps(
            {
                "nsys": {"status": "measured", "gpu_kernels": 867},
                "measurement_sha256": sha256_file(timing),
            }
        )
    )
    args = SimpleNamespace(
        output=output,
        profile_from=source,
        condition="0",
        conversion=None,
        length=177,
        profiles="ncu",
        ncu_timeout=2400,
        inputs=tmp_path,
        checkpoints=tmp_path,
    )

    def capture(tool, executable, command, directory, *, ncu_timeout):
        assert (tool, executable, directory, ncu_timeout) == (
            "ncu",
            "/ncu",
            output,
            2400,
        )
        assert command[command.index("--mode") + 1] == "profile"
        return {"status": "measured", "profiled_kernels": replayed}

    monkeypatch.setattr(experiment.shutil, "which", lambda _: "/ncu")
    monkeypatch.setattr(experiment, "capture", capture)
    monkeypatch.setenv("TENNIS_RUN_ID", "retry")
    matches = replayed == 867
    assert experiment.collect_profiles(args) is matches
    recorded = json.loads((output / "profiles.json").read_text())
    assert recorded["nsys_reused"]
    assert recorded["ncu"]["status"] == ("measured" if matches else "coverage_mismatch")
