"""CPU regressions for the frozen DINO model boundary and evidence checks."""

import json
from types import SimpleNamespace

import benchmark_dino
import convert_dino
import fixed_inputs
import measure_dino
import numpy as np
import pytest
import torch
from export_dino import batched_window_reverse
from measure_dino import decode_batch
from msda import PLUGIN, PluginCore, sampling_layers

from src.submodules.models.dino.person_detector import PersonDetectionRequest


def test_recorder_captures_the_production_rgb_stack(monkeypatch):
    monkeypatch.setattr(fixed_inputs, "SHORT_SIDE", 2)
    monkeypatch.setattr(fixed_inputs, "MAX_LONG_SIDE", 3)
    detector, recorder = fixed_inputs.input_recorder()
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[1, 2] = [0, 51, 255]
    detector._predict_batch_impl([PersonDetectionRequest(frame)], batch_size=1)
    batch = recorder.batch.numpy()
    assert batch.shape == (1, 3, 2, 3)
    assert batch.dtype == np.float32
    np.testing.assert_allclose(
        batch[0, :, 1, 2],
        [(1 - 0.485) / 0.229, (0.2 - 0.456) / 0.224, -0.406 / 0.225],
        rtol=1e-6,
    )


def test_tensor_digest_rejects_a_value_change_at_the_same_shape(tmp_path):
    path = tmp_path / "image.npy"
    data = np.zeros((1, 3, 2, 3), dtype=np.float32)
    np.save(path, data)
    rows = [{"file": path.name, "sha256": fixed_inputs.sha256_file(path)}]
    data[0, 0, 1, 2] = 2
    np.save(path, data)
    with pytest.raises(ValueError, match="Fixed tensor digest changed"):
        fixed_inputs.tensor_digest(tmp_path, rows)


def test_window_reverse_trace_at_batch_one_keeps_two_images_distinct():
    class ReverseWindows(torch.nn.Module):
        def forward(self, windows):
            return batched_window_reverse(windows, 2, 2, 4)

    windows = torch.arange(16.0).reshape(4, 2, 2, 1)
    traced = torch.jit.trace(ReverseWindows(), (windows[:2],))
    expected = torch.tensor(
        [[[0, 1, 4, 5], [2, 3, 6, 7]], [[8, 9, 12, 13], [10, 11, 14, 15]]]
    )
    torch.testing.assert_close(traced(windows)[..., 0], expected.float())


def test_batched_decode_pairs_each_image_with_its_own_geometry():
    logits = torch.tensor(
        [[[-10.0, 5.0], [-10.0, -15.0]], [[-10.0, -15.0], [-10.0, 8.0]]]
    )
    boxes = torch.tensor(
        [
            [[0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.25, 0.5]],
        ]
    )
    rows = [
        {"file": "frame-a", "original_hw": (20, 40)},
        {"file": "frame-b", "original_hw": (40, 80)},
    ]
    results = decode_batch({"pred_logits": logits, "pred_boxes": boxes}, rows)
    assert [row["file"] for row in results] == ["frame-a", "frame-b"]
    assert [row["boxes"] for row in results] == [[[10, 5, 30, 15]], [[30, 10, 50, 30]]]


@pytest.fixture
def dino_benchmark_case(tmp_path, monkeypatch):
    args = SimpleNamespace(
        inputs=tmp_path / "inputs",
        bucket="2x3",
        checkpoint=tmp_path / "weights",
        condition="2",
        output=tmp_path / "output",
        engine_dir=tmp_path,
        profile_from=None,
    )
    shape = [1, 3, 2, 3]
    lock = {
        "tensor_set_sha256": "input-id",
        "buckets": {
            args.bucket: {
                "profile": dict.fromkeys(("min", "opt", "max"), shape),
                "representative": "frame",
            }
        },
    }
    monkeypatch.setattr(benchmark_dino, "verify_inputs", lambda *_: lock)
    plan = tmp_path / "dino.plan"
    plan.write_bytes(b"first engine")
    receipt = {
        "condition": "2",
        "input_shape": shape,
        "tensor_set_sha256": "input-id",
        "checkpoint_sha256": benchmark_dino.CHECKPOINT_SHA256,
        "plan_sha256": fixed_inputs.sha256_file(plan),
        "method": "grid",
    }
    return args, plan, receipt


def test_tensor_rt_plan_from_different_bytes_cannot_run(dino_benchmark_case):
    args, plan, receipt = dino_benchmark_case
    (plan.parent / "conversion.json").write_text(json.dumps(receipt))
    plan.write_bytes(b"other engine")
    with pytest.raises(ValueError, match="plan receipt"):
        benchmark_dino.run_case(args)
    assert not args.output.exists()


@pytest.mark.parametrize(
    ("method", "load_plugins"), [("grid", False), ("plugin", True)]
)
def test_verified_receipt_controls_plugin_initialization(
    method, load_plugins, dino_benchmark_case, monkeypatch
):
    args, plan, receipt = dino_benchmark_case
    receipt["method"] = method
    (plan.parent / "conversion.json").write_text(json.dumps(receipt))

    class MeasurementStarted(Exception):
        pass

    def inspect_context(
        self, actual_plan, *, input_shapes, output_shapes, init_plugins
    ):
        assert actual_plan == plan
        assert init_plugins is load_plugins
        raise MeasurementStarted

    def start_measurement(command, _log, *, timeout):
        actual_plan = command[command.index("--plan") + 1]
        assert actual_plan == str(plan)
        measure_dino.TensorRTDino(
            plan,
            tuple(receipt["input_shape"]),
            method=command[command.index("--method") + 1],
        )

    monkeypatch.setattr(measure_dino.DinoTRTContext, "__init__", inspect_context)
    monkeypatch.setattr(benchmark_dino, "run_stage", start_measurement)
    with pytest.raises(MeasurementStarted):
        benchmark_dino.run_case(args)


def test_fused_sampler_requires_its_onnx_operation_in_engine_metadata():
    inspection = {
        "Layers": [
            {
                "Name": "__myl_Grid",
                "LayerType": "kgen",
                "Metadata": "[ONNX Layer: /Transpose]\x1e[ONNX Layer: /GridSample]",
                "Inputs": [{"Format/Datatype": "Half"}, {"Format/Datatype": "Half"}],
                "Outputs": [{"Format/Datatype": "Half"}],
            }
        ]
    }
    (observed,) = sampling_layers(inspection, {"/GridSample"}, method="grid")
    assert observed["operations"] == ["/GridSample"]
    assert observed["inputs"] == ["Half", "Half"]
    with pytest.raises(
        RuntimeError, match="Missing TensorRT layer evidence.*GridSample_1"
    ):
        sampling_layers(inspection, {"/GridSample", "/GridSample_1"}, method="grid")


def test_dino_build_rejects_qdq_feeding_unselected_residual(tmp_path, monkeypatch):
    graph = SimpleNamespace(
        node=[
            SimpleNamespace(name=name, op_type=op, input=inputs, output=[output])
            for name, op, inputs, output in [
                ("input_dq", "DequantizeLinear", ["x_quant", "scale"], "x_dq"),
                ("weight_dq", "DequantizeLinear", ["w_quant", "scale"], "w_dq"),
                ("projection", "MatMul", ["x_dq", "w_dq"], "projected"),
                ("encoder_residual", "Add", ["x", "projected"], "result"),
            ]
        ],
        output=[SimpleNamespace(name="result")],
    )
    required = {"projection": "int8"}
    convert_dino.verify_qdq(graph, required, "int8")
    graph.node[-1].input[0] = "x_dq"
    monkeypatch.setattr(
        convert_dino,
        "build_checked_plan",
        lambda *_a, **_kw: pytest.fail("reached TRT builder"),
    )
    with pytest.raises(RuntimeError, match=r"encoder_residual \(Add\) input\[0\]"):
        convert_dino.build(
            SimpleNamespace(condition="3a", output=tmp_path),
            tmp_path / "typed.onnx",
            {"method": "grid", "precision_requirements": required},
            graph,
        )
    assert not (tmp_path / "dino.plan").exists()


@pytest.mark.parametrize(
    ("kind", "input_dtype", "output_dtype"),
    [
        pytest.param("cast", "Float", "Float", id="same-name-fp32-cast"),
        pytest.param("custom_layer", "Float", "Half", id="wrong-input-dtype"),
        pytest.param("custom_layer", "Half", "Float", id="wrong-output-dtype"),
    ],
)
def test_dino_plugin_build_rejects_wrong_layer_or_formats(
    kind, input_dtype, output_dtype, tmp_path, monkeypatch
):
    names = {f"/encoder/{i}/{PLUGIN}" for i in range(12)}
    graph = SimpleNamespace(
        node=[SimpleNamespace(name=name, op_type=PLUGIN) for name in sorted(names)]
    )
    details = {
        "Layers": [
            {
                "Name": "input projection",
                "LayerType": "gemm",
                "TacticName": "sm80_xmma_gemm_f16f16_f16f32_f32_tn",
                "Metadata": "[ONNX Layer: /projection]",
                "Inputs": [{"Format/Datatype": "Half"}],
            },
            *[
                {
                    "Name": f"{name}_myl0",
                    "LayerType": "custom_layer",
                    "Metadata": f"[ONNX Layer: {name}]",
                    "Inputs": [{"Format/Datatype": "Half"} for _ in range(3)],
                    "Outputs": [{"Format/Datatype": "Half"}],
                }
                for name in sorted(names)
            ],
        ]
    }
    assert len(sampling_layers(details, names, method="plugin")) == 12
    for layer in details["Layers"][1:]:
        layer["LayerType"] = kind
        if kind == "cast":
            layer["Inputs"] = layer["Inputs"][:1]
        layer["Inputs"][0]["Format/Datatype"] = input_dtype
        layer["Outputs"][0]["Format/Datatype"] = output_dtype

    def validate_before_writing(_source, plan, *, verify, **_options):
        verify(details)
        plan.write_bytes(b"plan")
        pytest.fail("unexpectedly serialized an invalid plugin plan")

    monkeypatch.setattr(convert_dino, "build_checked_plan", validate_before_writing)
    with pytest.raises(RuntimeError, match="Unexpected plugin sampling layer"):
        convert_dino.build(
            SimpleNamespace(condition="2", output=tmp_path),
            tmp_path / "typed.onnx",
            {"method": "plugin", "precision_requirements": {"/projection": "fp16"}},
            graph,
        )
    assert not (tmp_path / "dino.plan").exists()


def test_plugin_export_uses_traced_shape_when_symbolic_shape_is_unavailable():
    unknown = torch._C.TensorType.get()
    assert unknown.sizes() is None
    value = SimpleNamespace(type=lambda: unknown)

    def op(name, *_inputs, plugin_version_s):
        assert name == f"trt::{PLUGIN}"
        assert plugin_version_s == "2"
        return SimpleNamespace(setType=lambda type_info: type_info)

    output_type = PluginCore.symbolic(
        SimpleNamespace(op=op), value, None, None, value, value, 1, 900, 8, 32, False
    )
    assert output_type.sizes() == [1, 900, 8, 32]
    assert output_type.dtype() == torch.float32
