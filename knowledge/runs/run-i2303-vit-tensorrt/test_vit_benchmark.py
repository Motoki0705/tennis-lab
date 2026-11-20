"""CPU contracts for the Phase 1-B adapters."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from benchmark import accuracy
from clip_ptq import accuracy_excluding_selection
from convert import (
    constant_tensors,
    linear_names,
    quantized_matrix_ops,
    restore_shape_inputs,
    verify_compute_precision,
    verify_qdq_scope,
)
from fixed_crops import body_rows
from runtime import torch_dispatch_precisions
from torch import nn
from vit_models import (
    HMR2,
    MODEL,
    VITPOSE,
    FlipRunner,
    feature_accuracy,
    pose_accuracy,
    ptq_operations,
)


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


@pytest.mark.parametrize(
    ("operation", "norm_input", "bias", "unexpected"),
    [
        ("MatMul", "pixels", [], None),
        ("MatMul", "pixels_dq", [], r"layer_norm \(LayerNormalization\) input\[0\]"),
        ("Gemm", "pixels", ["bias_dq"], r"projection \(Gemm\) input\[2\]"),
    ],
)
def test_explicit_qdq_scope_checks_every_consumer_input(
    operation, norm_input, bias, unexpected
):
    nodes = [
        ("activation_q", "QuantizeLinear", ["pixels", "scale"], ["pixels_q"]),
        ("activation_dq", "DequantizeLinear", ["pixels_q", "scale"], ["pixels_dq"]),
        ("weight_dq", "DequantizeLinear", ["weight_q", "scale"], ["weight_dq"]),
        ("bias_dq", "DequantizeLinear", ["bias_q", "scale"], ["bias_dq"]),
        ("projection", operation, ["pixels_dq", "weight_dq", *bias], ["projected"]),
        ("layer_norm", "LayerNormalization", [norm_input, "gamma"], ["normalized"]),
    ]
    graph = SimpleNamespace(
        node=[
            SimpleNamespace(name=name, op_type=op, input=inputs, output=outputs)
            for name, op, inputs, outputs in nodes
        ],
        output=[SimpleNamespace(name="projected"), SimpleNamespace(name="normalized")],
    )
    # The matrix set alone misses both the shared DQ and a quantized bias.
    assert quantized_matrix_ops(graph) == {"projection"}
    if unexpected is None:
        assert verify_qdq_scope(graph, {"projection"}) == {"projection"}
    else:
        with pytest.raises(RuntimeError, match=unexpected):
            verify_qdq_scope(graph, {"projection"})


def test_ptq_shape_reads_are_restored_without_removing_the_projection_qdq():
    nodes = [
        ("q", "QuantizeLinear", ["pixels", "scale"], ["pixels_q"]),
        ("dq", "DequantizeLinear", ["pixels_q", "scale"], ["pixels_dq"]),
        ("dq_weight", "DequantizeLinear", ["q_weight", "scale"], ["weight_dq"]),
        ("size", "Shape", ["pixels_dq"], ["sizes"]),
        ("projection", "MatMul", ["pixels_dq", "weight_dq"], ["projected"]),
    ]
    graph = SimpleNamespace(
        node=[
            SimpleNamespace(
                name=name, op_type=op, domain="", input=inputs, output=outputs
            )
            for name, op, inputs, outputs in nodes
        ],
        output=[SimpleNamespace(name="projected")],
    )
    with pytest.raises(RuntimeError, match=r"size \(Shape\) input\[0\]"):
        verify_qdq_scope(graph, {"projection"})
    assert restore_shape_inputs(graph) == 1
    assert graph.node[3].input == ["pixels"]
    assert graph.node[4].input == ["pixels_dq", "weight_dq"]
    assert verify_qdq_scope(graph, {"projection"}) == {"projection"}


def test_clip_selection_exclusion_rebuilds_gallery_and_queries():
    addresses = [
        ("clip_000", "cam0", 10, 0),
        ("clip_000", "cam0", 10, 1),
        ("clip_000", "cam1", 10, 0),
        ("clip_000", "cam1", 10, 1),
        ("clip_000", "cam1", 20, 0),
    ]
    baseline = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float32)
    actual = baseline.copy()
    actual[2] = [0, 1]
    full = accuracy(actual, baseline, addresses)
    assert any(row["address"] == addresses[1] for row in full["id_changed_examples"])
    retained = accuracy_excluding_selection(actual, baseline, addresses, {addresses[2]})
    assert retained["crops"] == retained["id_matches_fp32"] == 4
    assert retained["cosine_p05"] == pytest.approx(1)


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


@pytest.mark.parametrize(
    ("model", "prefix", "suffix", "blocks"),
    [
        (MODEL, "/visual/transformer/resblocks/resblocks", "/attn/Gemm", 12),
        (VITPOSE, "/backbone/blocks", "/attn/proj/MatMul", 32),
        (HMR2, "/hmr/backbone/blocks", "/attn/proj/MatMul", 32),
    ],
)
def test_attention_out_ptq_selects_one_weighted_projection_per_block(
    model, prefix, suffix, blocks
):
    outputs = [f"{prefix}.{block}{suffix}" for block in range(blocks)]
    graph = SimpleNamespace(
        initializer=[SimpleNamespace(name="weight")],
        node=[
            SimpleNamespace(
                name=name,
                op_type="MatMul",
                input=["pixels", "weight"],
                output=[name + ":0"],
            )
            for name in [*outputs, f"{prefix}.0/mlp/fc2/MatMul", "/smpl_head/MatMul"]
        ],
    )
    assert ptq_operations(model, graph, "attention-out") == set(outputs)
    graph.node.pop(0)
    with pytest.raises(ValueError, match="expected 1 operations"):
        ptq_operations(model, graph, "attention-out")


@pytest.mark.parametrize(
    ("tactic", "shapes", "names"),
    [
        (
            "_gemv_mha_v1_test",
            [[8, 1, 64], [8, 192, 64], [8, 192, 64]],
            ("scores", "values"),
        ),
        ("__myl_ReshMoveMul_test", [[8, 1, 64], [1, 8, 1, 1]], ("values",)),
    ],
)
def test_hmr2_attention_tactics_require_fp16_operands(tactic, shapes, names):
    layer = {
        "Name": "attention",
        "LayerType": "kgen",
        "TacticName": tactic,
        "Inputs": [
            {"Dimensions": shape, "Format/Datatype": "Half"} for shape in shapes
        ],
        "Metadata": "".join(f"[ONNX Layer: {name}]" for name in names),
    }
    expected = dict.fromkeys(names, "fp16")
    result = verify_compute_precision(
        {"Layers": [layer]}, "fp16", expected_operations=expected
    )
    assert result["onnx_operations_by_precision"] == {"fp16": len(names)}
    layer["Inputs"][-1]["Format/Datatype"] = "FP32"
    with pytest.raises(RuntimeError, match="expected fp16 operands"):
        verify_compute_precision(
            {"Layers": [layer]}, "fp16", expected_operations=expected
        )


def test_static_onnx_ops_are_checked_if_executed_and_can_be_folded():
    dynamic_layer = {
        "LayerType": "gemm",
        "Name": "dynamic",
        "TacticName": "sm80_xmma_gemm_f16f16_f16f32_f32_tn",
        "Inputs": [{"Format/Datatype": "Half"}],
        "Metadata": "[ONNX Layer: dynamic]",
    }
    requirements = {
        "expected_operations": {"dynamic": "fp16", "constant": "fp16"},
        "constant_operations": ["constant"],
    }
    details = {"Layers": [dynamic_layer]}
    result = verify_compute_precision(details, "fp16", **requirements)
    assert result["onnx_operations_by_precision"] == {"fp16": 1}
    assert result["unobserved_constant_onnx_operations"] == ["constant"]
    details["Layers"].append(
        {
            **dynamic_layer,
            "Name": "constant",
            "Metadata": "[ONNX Layer: constant]",
            "TacticName": "sm80_xmma_gemm_f32f32_f32f32_f32_tn",
        }
    )
    with pytest.raises(RuntimeError, match="Compute layer constant expected fp16"):
        verify_compute_precision(details, "fp16", **requirements)


def test_cublas_gemvx_uses_adjacent_operand_codes_before_accumulator():
    layer = {
        "Name": "decoder projection",
        "LayerType": "gemm",
        "TacticName": "sm50_xmma_cublas_gemvx_f16f16_f32_f32_tn_n_int32_unit_n",
        "Inputs": [{"Format/Datatype": "Half"}],
        "Metadata": "[ONNX Layer: projection]",
    }
    expected = {"projection": "fp16"}
    assert verify_compute_precision(
        {"Layers": [layer]}, "fp16", expected_operations=expected
    )["onnx_operations_by_precision"] == {"fp16": 1}
    layer["TacticName"] = "sm50_xmma_cublas_gemvx_f16f32_f16_f16_tn_n_int32_unit_n"
    with pytest.raises(RuntimeError, match="expected fp16 operands"):
        verify_compute_precision(
            {"Layers": [layer]}, "fp16", expected_operations=expected
        )


def test_vitpose_int8_projection_fusion_keeps_fp16_residual():
    name = "/backbone/blocks.0/attn/proj/MatMul"
    layer = {
        "Name": "__myl_FcCastAddAdd_myl0_8",
        "LayerType": "fusion",
        "TacticName": "sm80_xmma_gemm_i8f32_i8i32_f32_tn_by_fusion_tactic",
        "Inputs": [
            {"Format/Datatype": "Half", "Dimensions": [384, 1280]},
            {"Format/Datatype": "Int8", "Dimensions": [384, 1280]},
        ],
        "Metadata": (
            f"[ONNX Layer: {name}]"
            "[ONNX Layer: activation_DequantizeLinear]"
            "[ONNX Layer: weight_DequantizeLinear]"
            "[ONNX Layer: /backbone/blocks.0/Add]"
        ),
    }
    expected = {name: "int8"}
    assert verify_compute_precision(
        {"Layers": [layer]}, "int8", expected_operations=expected
    )["onnx_operations_by_precision"] == {"int8": 1}
    layer["Metadata"] = layer["Metadata"].replace(
        "[ONNX Layer: activation_DequantizeLinear]", ""
    )
    with pytest.raises(RuntimeError, match="expected int8 operands"):
        verify_compute_precision(
            {"Layers": [layer]}, "int8", expected_operations=expected
        )


def test_vitpose_fused_int8_input_cannot_mask_a_fp16_gemm():
    name = "/backbone/blocks.0/attn/proj/MatMul"
    layer = {
        "Name": "__myl_FcCastAddAdd_myl0_8",
        "LayerType": "fusion",
        "TacticName": "sm80_xmma_gemm_f16f16_f16f32_f32_tn_by_fusion_tactic",
        "Inputs": [
            {"Format/Datatype": "Half", "Dimensions": [384, 1280]},
            {"Format/Datatype": "Int8", "Dimensions": [384, 1280]},
        ],
        "Metadata": (
            f"[ONNX Layer: {name}]"
            "[ONNX Layer: activation_DequantizeLinear]"
            "[ONNX Layer: weight_DequantizeLinear]"
            "[ONNX Layer: /backbone/blocks.0/Add]"
        ),
    }
    with pytest.raises(RuntimeError, match="expected int8 operands"):
        verify_compute_precision(
            {"Layers": [layer]}, "int8", expected_operations={name: "int8"}
        )
