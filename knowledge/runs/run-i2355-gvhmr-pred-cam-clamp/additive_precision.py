"""Verify FP16 compute precision for GVHMR engines with additive attention masks.

``convert.verify_compute_precision`` recognises fused masked attention whose two
masks enter the kernel as Bool tensors.  With the additive-mask rewrite they
enter as FP16 bias tensors instead.  This wrapper removes exactly those two
bias inputs from each fused attention layer after checking their shapes, so
the shared verifier still requires FP16 Q, K and V operands.
"""

from __future__ import annotations

import copy

from convert import verify_compute_precision

FUSED_ATTENTION = ("_gemm_mha_v2_", "_gemv_mha_v1_")


def strip_additive_masks(details: dict) -> tuple[dict, int]:
    stripped = copy.deepcopy(details)
    count = 0
    for layer in stripped["Layers"]:
        tactic = layer.get("TacticName") or ""
        if layer["LayerType"].lower() != "kgen" or not tactic.startswith(
            FUSED_ATTENTION
        ):
            continue
        inputs = layer.get("Inputs", [])
        if len(inputs) != 5:
            raise RuntimeError(
                f"Fused attention {layer['Name']} has {len(inputs)} inputs"
            )
        query, key, local, padding, value = (
            tuple(tensor["Dimensions"]) for tensor in inputs
        )
        heads_batch, tokens, _width = query
        batch = padding[0]
        if (
            key[0] != heads_batch
            or value != query
            or local != (1, 1, tokens, tokens)
            or padding != (batch, 1, 1, tokens)
            or batch <= 0
            or heads_batch % batch
            or any(inputs[i]["Format/Datatype"] != "Half" for i in (2, 3))
        ):
            raise RuntimeError(
                f"Fused attention {layer['Name']} does not take two FP16 mask biases"
            )
        layer["Inputs"] = [inputs[0], inputs[1], inputs[4]]
        count += 1
    if not count:
        raise RuntimeError("No fused attention layer with additive masks was found")
    return stripped, count


def verify_additive_attention(details, required_precision, *, expected_operations):
    stripped, count = strip_additive_masks(details)
    evidence = verify_compute_precision(
        stripped, required_precision, expected_operations=expected_operations
    )
    evidence["additive_mask_attention_layers"] = count
    return evidence
