"""CPU contracts for the exact attention-mask rewrites."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from minimal_repro import run_onnx
from onnx import TensorProto, helper, numpy_helper
from rewrite_attention_mask import rewrite_attention_mask


def write_masked_softmax(
    path: Path, *, fill: float = -np.inf, dtype=np.float32
) -> None:
    """Mirror the exporter's masked_fill lowering for a padding mask."""
    constants = {
        "mask_shape": np.array([2, 1, 1, 3], dtype=np.int64),
        "sentinel_shape": np.array([2, 2, -1, -1], dtype=np.int64),
        "ones_shape": np.array([4], dtype=np.int64),
        "minus_one": np.array(-1, dtype=np.int64),
        "fill": np.array(fill, dtype=dtype),
    }
    nodes = [
        helper.make_node("Reshape", ["mask", "mask_shape"], ["mask4"], name="reshape"),
        helper.make_node("ConstantOfShape", ["ones_shape"], ["ones"], name="ones"),
        helper.make_node("Mul", ["ones", "minus_one"], ["minus_ones"], name="mul"),
        helper.make_node(
            "Equal", ["sentinel_shape", "minus_ones"], ["is_sentinel"], name="eq"
        ),
        helper.make_node(
            "Where", ["is_sentinel", "ones", "sentinel_shape"], ["shape"], name="w0"
        ),
        helper.make_node("Expand", ["mask4", "shape"], ["expanded"], name="expand"),
        helper.make_node("Cast", ["expanded"], ["condition"], name="cast", to=9),
        helper.make_node(
            "Where", ["condition", "fill", "scores"], ["masked"], name="masked_fill"
        ),
        helper.make_node("Softmax", ["masked"], ["probabilities"], name="softmax"),
    ]
    nodes[1].attribute.append(
        helper.make_attribute(
            "value", numpy_helper.from_array(np.ones((1,), dtype=np.int64))
        )
    )
    graph = helper.make_graph(
        nodes,
        "masked-softmax",
        [
            helper.make_tensor_value_info(
                "scores",
                TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT,
                [2, 2, 3, 3],
            ),
            helper.make_tensor_value_info("mask", TensorProto.BOOL, [2, 3]),
        ],
        [
            helper.make_tensor_value_info(
                "probabilities",
                TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT,
                [2, 2, 3, 3],
            )
        ],
        initializer=[
            numpy_helper.from_array(value, name=name)
            for name, value in constants.items()
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


@pytest.mark.parametrize("mode", ["drop-expand", "additive", "additive-finite"])
def test_mask_rewrite_is_bit_exact_and_removes_expand(tmp_path: Path, mode: str):
    source = tmp_path / "source.onnx"
    target = tmp_path / "target.onnx"
    write_masked_softmax(source)
    rng = np.random.default_rng(0)
    feeds = {
        "scores": rng.normal(size=(2, 2, 3, 3)).astype(np.float32),
        "mask": np.array([[False, False, True], [False, True, True]]),
    }

    report = rewrite_attention_mask(source, target, mode=mode, expected=1)

    operations = [node.op_type for node in onnx.load(target).graph.node]
    assert report["rewritten_masks"] == 1
    assert "Expand" not in operations
    assert operations.count("Add") == mode.startswith("additive")
    expected = run_onnx(source, feeds)["probabilities"]
    actual = run_onnx(target, feeds)["probabilities"]
    np.testing.assert_array_equal(actual, expected)
    assert np.all(expected[1, :, :, 1:] == 0)


def test_finite_additive_mask_is_bit_exact_in_fp16(tmp_path: Path):
    source = tmp_path / "source.onnx"
    target = tmp_path / "target.onnx"
    write_masked_softmax(source, dtype=np.float16)
    rng = np.random.default_rng(1)
    feeds = {
        "scores": (rng.normal(size=(2, 2, 3, 3)) * 30).astype(np.float16),
        "mask": np.array([[False, False, True], [False, True, True]]),
    }

    rewrite_attention_mask(source, target, mode="additive-finite", expected=1)

    expected = run_onnx(source, feeds)["probabilities"]
    actual = run_onnx(target, feeds)["probabilities"]
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()


def test_mask_rewrite_rejects_a_finite_fill(tmp_path: Path):
    source = tmp_path / "source.onnx"
    write_masked_softmax(source, fill=-1e4)

    with pytest.raises(ValueError, match="does not fill with -inf"):
        rewrite_attention_mask(
            source, tmp_path / "target.onnx", mode="drop-expand", expected=1
        )


def test_mask_rewrite_requires_the_expected_count(tmp_path: Path):
    source = tmp_path / "source.onnx"
    write_masked_softmax(source)

    with pytest.raises(ValueError, match="Expected 2 masked_fill"):
        rewrite_attention_mask(
            source, tmp_path / "target.onnx", mode="additive", expected=2
        )


def test_diagnostic_mask_removal_returns_unmasked_softmax(tmp_path: Path):
    from rewrite_attention_mask import remove_masks_for_diagnosis

    source = tmp_path / "source.onnx"
    target = tmp_path / "target.onnx"
    write_masked_softmax(source)
    scores = np.random.default_rng(1).normal(size=(2, 2, 3, 3)).astype(np.float32)

    report = remove_masks_for_diagnosis(
        source, target, fills=("masked_fill",), expected=1
    )

    assert report["equivalent"] is False
    operations = [node.op_type for node in onnx.load(target).graph.node]
    assert operations == ["Softmax"]
    actual = run_onnx(target, {"scores": scores, "mask": np.ones((2, 3), dtype=bool)})[
        "probabilities"
    ]
    shifted = np.exp(scores - scores.max(axis=-1, keepdims=True))
    np.testing.assert_allclose(actual, shifted / shifted.sum(-1, keepdims=True), 1e-6)
