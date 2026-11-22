"""CPU contracts for the additive-mask attention precision check."""

import pytest
from additive_precision import elementwise_small_k, strip_additive_masks


def attention_layer(mask_type: str = "Half", padding=(2, 1, 1, 145)) -> dict:
    shapes = [(16, 145, 64), (16, 64, 145), (1, 1, 145, 145), padding, (16, 145, 64)]
    types = ["Half", "Half", mask_type, mask_type, "Half"]
    return {
        "Name": "_gemm_mha_v2_myl0_0",
        "LayerType": "kgen",
        "TacticName": "_gemm_mha_v2_0x0",
        "Inputs": [
            {"Dimensions": list(shape), "Format/Datatype": dtype}
            for shape, dtype in zip(shapes, types, strict=True)
        ],
    }


def test_strip_keeps_query_key_value_of_each_fused_attention() -> None:
    details = {"Layers": [attention_layer(), {"LayerType": "gemm", "Inputs": []}]}

    stripped, count = strip_additive_masks(details)

    assert count == 1
    dims = [tensor["Dimensions"] for tensor in stripped["Layers"][0]["Inputs"]]
    assert dims == [[16, 145, 64], [16, 64, 145], [16, 145, 64]]
    assert len(details["Layers"][0]["Inputs"]) == 5


def test_strip_accepts_dynamic_profile_dimensions() -> None:
    layer = attention_layer(padding=(-1, 1, 1, -1))
    for tensor, shape in zip(
        layer["Inputs"],
        [(-1, -1, 64), (-1, 64, -1), (1, 1, -1, -1), (-1, 1, 1, -1), (-1, -1, 64)],
        strict=True,
    ):
        tensor["Dimensions"] = list(shape)

    _stripped, count = strip_additive_masks({"Layers": [layer]})

    assert count == 1


@pytest.mark.parametrize(
    "layer",
    [
        attention_layer(mask_type="Bool"),
        attention_layer(padding=(3, 1, 1, 145)),
        attention_layer(padding=(2, 1, 1, 144)),
        attention_layer(padding=(2, 1, 145)),
    ],
)
def test_strip_rejects_other_mask_inputs(layer) -> None:
    with pytest.raises(RuntimeError, match="two FP16 mask biases"):
        strip_additive_masks({"Layers": [layer]})


def test_strip_requires_a_fused_attention_layer() -> None:
    with pytest.raises(RuntimeError, match="No fused attention"):
        strip_additive_masks({"Layers": [{"LayerType": "gemm", "Inputs": []}]})


def small_k_layer(output_width: int = 32, dtype: str = "Half") -> dict:
    return {
        "Name": "__myl_SlicSlicMulMulAdd_myl0_2",
        "LayerType": "kgen",
        "TacticName": "__myl_SlicSlicMulMulAdd_0x0",
        "Metadata": "[ONNX Layer: /network/learned_pos_linear/MatMul]",
        "Inputs": [
            {"Dimensions": [-1, 2], "Format/Datatype": dtype},
            {"Dimensions": [-1, 2], "Format/Datatype": "Half"},
        ],
        "Outputs": [{"Dimensions": [-1, output_width], "Format/Datatype": "Half"}],
    }


def test_small_k_layer_is_named_as_fp16_evidence() -> None:
    found = elementwise_small_k({"Layers": [small_k_layer()]})

    assert found == {
        "/network/learned_pos_linear/MatMul": "__myl_SlicSlicMulMulAdd_myl0_2"
    }


@pytest.mark.parametrize(
    "layer", [small_k_layer(output_width=16), small_k_layer(dtype="Float")]
)
def test_small_k_layer_rejects_other_shapes_and_types(layer) -> None:
    with pytest.raises(RuntimeError, match="K=2 learned_pos_linear"):
        elementwise_small_k({"Layers": [layer]})
