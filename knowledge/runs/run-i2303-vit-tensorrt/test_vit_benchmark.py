"""CPU contracts for the Phase 1-B adapters."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from convert import constant_tensors, linear_names, quantized_matrix_ops
from fixed_crops import body_rows
from runtime import torch_dispatch_precisions
from torch import nn
from vit_models import FlipRunner, feature_accuracy, pose_accuracy


def test_body_view_rows_use_exact_frame_and_its_box():
    payload = {
        "selections": [
            {
                "camera_id": "cam2",
                "person_id": 7,
                "requests": [
                    {
                        "source_frames": "frames",
                        "boxes_xys": "boxes",
                        "keypoints": "poses",
                    }
                ],
            }
        ]
    }
    poses = np.zeros((3, 17, 3))
    poses[2, 5, 2] = 0.75
    arrays = {
        "frames": np.array([2, 4, 8]),
        "boxes": np.array([[11, 12, 13], [21, 23, 24], [41, 43, 46]], dtype=np.float32),
        "poses": poses,
    }
    assert body_rows(
        payload, arrays.__getitem__, {"cam2": {"sampled_frames": [3, 8]}}
    ) == [
        {
            "camera": "cam2",
            "frame": 8,
            "person": 7,
            "segment": 0,
            "boxes_xys": [41, 43, 46],
            "any_positive_joint": True,
        }
    ]


def test_flip_runner_constructs_two_views_and_restores_left_joint():
    graph_input = torch.zeros((2, 3, 2, 3))
    graph_output = torch.zeros((2, 17, 2, 3))
    graph_output[0, 1, 1, 0] = 8
    graph_output[1, 2, 1, 2] = 4
    graph_output[1, 1, 0, 0] = 10
    runner = FlipRunner(SimpleNamespace(pixels=graph_input, run=lambda: graph_output))
    runner.pixels.copy_(torch.arange(18).reshape(1, 3, 2, 3))
    merged = runner.run()
    torch.testing.assert_close(graph_input[0, 0, 0], torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(graph_input[1, 0, 0], torch.tensor([2.0, 1.0, 0.0]))
    assert merged[0, 1, 1, 0].item() == 6
    assert merged[0, 2, 0, 2].item() == 5


def test_udp_pixel_error_confidence_and_mask_use_the_same_reference():
    y, x = np.mgrid[:64, :48]
    base = np.zeros((1, 17, 64, 48), dtype=np.float32)
    base[0, :2] = np.exp(-((x - 12) ** 2 + (y - 18) ** 2) / 18)
    actual = base.copy()
    actual[0, 0] = 1.1 * np.exp(-((x - 13) ** 2 + (y - 18) ** 2) / 18)
    actual[0, 1] = 0
    boxes = np.array([[510, 220, 188]], dtype=np.float32)
    metrics, _ = pose_accuracy(actual, base, boxes)
    assert metrics["common_valid_joints"] == 1
    assert metrics["mask_mismatches"] == 1
    assert metrics["xy_l2_px"]["p95"] == pytest.approx(3, abs=0.015)
    assert metrics["confidence_abs"]["p95"] == pytest.approx(0.1)
    same, _ = pose_accuracy(base, base, boxes)
    assert same["xy_l2_px"]["max"] == 0
    assert base[0, 0, 18, 12] == 1


def test_hmr2_scale_error_remains_visible_with_unit_cosine():
    gold = np.array([[3, 4], [-5, 12]], dtype=np.float32)
    scaled = gold * np.array([[2], [0.5]], dtype=np.float32)
    metrics, by_crop = feature_accuracy(scaled, gold)
    np.testing.assert_allclose(by_crop, [1, 0.5])
    assert metrics["cosine"]["p05"] == pytest.approx(1)
    assert metrics["max_abs_error_by_dimension"] == [3, 6]


def test_linear_scope_keeps_transposed_and_dequantized_weights_without_attention():
    nodes = [
        ("transpose", "Transpose", ["weights"], ["weights_t"]),
        ("dequantize", "DequantizeLinear", ["q_weights", "scale"], ["weight_f"]),
        ("linear1", "MatMul", ["x", "weights_t"], ["q"]),
        ("linear2", "Gemm", ["x", "weight_f"], ["k"]),
        ("attention", "MatMul", ["q", "k"], ["att"]),
        ("conv", "Conv", ["x", "weights"], ["patch"]),
    ]
    graph = SimpleNamespace(
        initializer=[
            SimpleNamespace(name=name) for name in ("weights", "q_weights", "scale")
        ],
        node=[
            SimpleNamespace(name=name, op_type=op, input=inputs, output=outputs)
            for name, op, inputs, outputs in nodes
        ],
    )
    assert linear_names(graph) == {"linear1", "linear2"}


def test_qdq_scope_requires_dequantization_of_both_operands():
    nodes = [
        ("activation_dq", "DequantizeLinear", ["qx", "scale"], ["x_dq"]),
        ("weight_dq", "DequantizeLinear", ["qw", "scale"], ["w_dq"]),
        ("explicit_int8", "MatMul", ["x_dq", "w_dq"], ["y"]),
        ("weight_only", "MatMul", ["x", "w_dq"], ["z"]),
        ("unselected_add", "Add", ["x_dq", "w_dq"], ["skip"]),
    ]
    graph = SimpleNamespace(
        node=[
            SimpleNamespace(name=name, op_type=op, input=inputs, output=outputs)
            for name, op, inputs, outputs in nodes
        ]
    )
    assert quantized_matrix_ops(graph) == {"explicit_int8"}


@pytest.mark.parametrize("precision", [torch.float32, torch.float16])
def test_torch_dispatch_tracks_composite_matrix_and_convolution(precision):
    model = nn.Sequential(
        nn.Conv2d(3, 5, 1),
        nn.ConvTranspose2d(5, 6, 2),
        nn.Flatten(2),
        nn.Linear(25, 9),
        nn.LayerNorm(9),
    ).eval()
    pixels = torch.linspace(-2, 3, 48).reshape(1, 3, 4, 4)

    def infer():
        with torch.autocast(
            "cpu", dtype=torch.float16, enabled=precision == torch.float16
        ):
            features = model(pixels)
            return features @ features.transpose(-1, -2)

    with torch.inference_mode():
        measured = torch_dispatch_precisions(infer, precision)
    assert (
        sum(
            measured.get(k, {}).get(str(precision), 0)
            for k in ("linear", "addmm", "mm")
        )
        >= 1
    )
    assert (
        sum(
            measured.get(k, {}).get(str(precision), 0)
            for k in ("conv2d", "conv_transpose2d", "convolution", "_convolution")
        )
        >= 2
    )


def test_torch_dispatch_detects_fp32_during_fp16_probe():
    values = torch.ones(2, 3)
    with (
        torch.inference_mode(),
        pytest.raises(RuntimeError, match="did not execute at torch.float16"),
    ):
        torch_dispatch_precisions(lambda: values @ values.T, torch.float16)


def test_quantized_initializer_is_static_and_quantized_input_is_dynamic():
    nodes = [
        ("Transpose", ["weight"], ["wt"]),
        ("QuantizeLinear", ["wt", "scale"], ["qw"]),
        ("QuantizeLinear", ["input", "scale"], ["qx"]),
        ("DequantizeLinear", ["qw", "scale"], ["weight_dq"]),
        ("DequantizeLinear", ["qx", "scale"], ["input_dq"]),
    ]
    graph = SimpleNamespace(
        initializer=[SimpleNamespace(name=name) for name in ("weight", "scale")],
        node=[
            SimpleNamespace(op_type=op, input=ins, output=outs)
            for op, ins, outs in nodes
        ],
    )
    assert constant_tensors(graph) == {"weight", "scale", "wt", "qw", "weight_dq"}
