"""Check the GVHMR production export and TensorRT evidence rules on CPU."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.submodules.models.gvhmr import onnx_export
from src.submodules.models.gvhmr.tensorrt_validation import (
    probe_inputs,
    verify_bit_equal,
    verify_fp16_layers,
    verify_outputs,
)
from src.submodules.vendor.gvhmr.network.relative_transformer import NetworkEncoderRoPE


def test_mask_export_keeps_dynamic_people_and_finite_additive_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(871)
    network = NetworkEncoderRoPE(num_layers=1, latent_dim=32, num_heads=2)
    # Activate the block so changes in attention reach observable outputs.
    network.blocks[0].gate_msa.data.fill_(1)
    checkpoint, path = tmp_path / "weights.pt", tmp_path / "model.onnx"
    torch.save({"state_dict": {f"pipeline.denoiser3d.{k}": v for k, v in network.state_dict().items()}}, checkpoint)
    monkeypatch.setattr(onnx_export, "NetworkEncoderRoPE", lambda: network)
    onnx_export.export_gvhmr_onnx(checkpoint, path, keep_mask_reference=True)
    reference = path.with_name(onnx_export.MASK_REFERENCE_FILE)
    before, graph = onnx.load(reference).graph, onnx.load(path).graph
    predecessors = {out: node for node in before.node for out in node.output}
    softmax = next(n for n in before.node if n.op_type == "Softmax")
    padding = predecessors[softmax.input[0]]
    local = predecessors[padding.input[2]]
    original_masks = {padding.name, local.name}
    assert {n.name for n in graph.node if n.op_type == "Add"} >= original_masks
    biases = [n for n in graph.node if n.name.endswith("/Bias") and n.op_type == "Where"]
    assert len(biases) == 2
    fills = {item.name: onnx.numpy_helper.to_array(item) for item in graph.initializer
             if item.name.startswith("__attention_mask")}
    assert all(fills[n.input[1]].dtype == np.float16 and fills[n.input[1]].item() == -32768.
               and fills[n.input[2]].item() == 0. for n in biases)
    assert {n.name for n in graph.node if n.op_type == "Where" and n not in biases} == {
        n.name for n in before.node if n.op_type == "Where"} - original_masks

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    sessions = [ort.InferenceSession(str(file), sess_options=options, providers=providers) for file in (reference, path)]
    for batch, frames in ((1, 2), (2, 121), (3, 177)):
        values = []
        for session in sessions:
            actual = dict(zip((out.name for out in session.get_outputs()),
                              session.run(None, probe_inputs(batch, frames)), strict=True))
            verify_outputs(actual, batch=batch, frames=frames)
            values.append(actual)
        verify_bit_equal(*values)


def _layers() -> tuple[dict[str, Any], set[str]]:
    def half(*shape: int) -> dict[str, Any]:
        return {"Dimensions": shape, "Format/Datatype": "Half"}

    scope = "/network/blocks.0/attn/"
    qk, pv, small_k, dense = (scope + "MatMul", scope + "MatMul_1",
                               "/network/learned_pos_linear/MatMul", "/network/final_layer/MatMul")
    return {"Layers": [
        {"LayerType": "kgen", "Name": "attention", "TacticName": "_gemm_mha_v2_f16",
         "Metadata": f"[ONNX Layer: {qk}][ONNX Layer: {pv}]",
         "Inputs": [half(-1, 177, 64), half(-1, 64, 177), half(1, 1, -1, -1), half(-1, 1, 1, -1), half(-1, 177, 64)]},
        {"LayerType": "kgen", "Name": "projection", "TacticName": "__myl_SlicSlicMulMulAdd_x",
         "Metadata": f"[ONNX Layer: {small_k}]", "Inputs": [half(-1, 17, 2), half(32, 2)], "Outputs": [half(-1, 17, 32)]},
        {"LayerType": "gemm", "Name": "dense", "TacticName": "sm120_xmma_gemm_f16f16_f16f16_f32",
         "Metadata": f"[ONNX Layer: {dense}]", "Inputs": [half(-1, 512)]},
    ]}, {qk, pv, small_k, dense}


def test_layer_verifier_accounts_for_both_attention_matrix_products_and_k2() -> None:
    layers, operations = _layers()
    report = verify_fp16_layers(layers, operations, attention_blocks=1)
    assert report["fused_additive_attention"] == ["attention"]
    assert report["elementwise_small_k"] == ["projection"]
    assert set(report["fp16_operations"]) == operations


@pytest.mark.parametrize("fault", ["mask_dtype", "mask_shape", "projection_width", "fp32_gemm", "missing_layer"])
def test_layer_verifier_rejects_invalid_or_missing_fp16_evidence(fault: str) -> None:
    layers, expected = _layers()
    attention, projection, dense = layers["Layers"]
    if fault == "mask_dtype":
        attention["Inputs"][2]["Format/Datatype"] = "Bool"
    elif fault == "mask_shape":
        attention["Inputs"][3]["Dimensions"] = (-1, 2, 1, 177)
    elif fault == "projection_width":
        projection["Inputs"][0]["Dimensions"] = (-1, 17, 3)
    elif fault == "fp32_gemm":
        dense["TacticName"] = "sm120_xmma_gemm_f32f32_f32f32_f32"
    else:
        layers["Layers"].pop()
    with pytest.raises(ValueError, match="FP16"):
        verify_fp16_layers(layers, expected, attention_blocks=1)


def test_boundary_probe_rejects_a_single_nonfinite_prediction() -> None:
    outputs = {
        "pred_x": np.ones((2, 177, 151), dtype=np.float16),
        "pred_cam": np.ones((2, 177, 3), dtype=np.float16),
        "static_conf_logits": np.ones((2, 177, 6), dtype=np.float16),
        "mask_out": np.ones((2, 177), dtype=np.bool_),
        "length_out": np.array([177, 177], dtype=np.int64),
    }
    verify_outputs(outputs, batch=2, frames=177)
    outputs["pred_x"][1, -1, 1] = np.nan
    with pytest.raises(ValueError, match="pred_x at B=2, L=177.*non-finite"):
        verify_outputs(outputs, batch=2, frames=177)
