"""CPU contracts for the additive-mask attention precision check."""

import pytest
from additive_precision import strip_additive_masks


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


@pytest.mark.parametrize(
    "layer",
    [attention_layer(mask_type="Bool"), attention_layer(padding=(3, 1, 1, 145))],
)
def test_strip_rejects_other_mask_inputs(layer) -> None:
    with pytest.raises(RuntimeError, match="two FP16 mask biases"):
        strip_additive_masks({"Layers": [layer]})


def test_strip_requires_a_fused_attention_layer() -> None:
    with pytest.raises(RuntimeError, match="No fused attention"):
        strip_additive_masks({"Layers": [{"LayerType": "gemm", "Inputs": []}]})
