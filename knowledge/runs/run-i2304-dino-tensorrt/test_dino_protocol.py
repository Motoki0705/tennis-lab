"""CPU regressions for the frozen DINO model boundary and evidence checks."""

import json
from types import SimpleNamespace

import benchmark_dino
import fixed_inputs
import numpy as np
import pytest
import torch
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


def test_tensor_rt_plan_from_different_bytes_cannot_run(tmp_path, monkeypatch):
    args = SimpleNamespace(
        inputs=tmp_path / "inputs",
        bucket="2x3",
        checkpoint=tmp_path / "weights",
        condition="2",
        output=tmp_path / "output",
        engine_dir=tmp_path,
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
    (tmp_path / "conversion.json").write_text(
        json.dumps(
            {
                "condition": "2",
                "input_shape": shape,
                "tensor_set_sha256": "input-id",
                "checkpoint_sha256": benchmark_dino.CHECKPOINT_SHA256,
                "plan_sha256": fixed_inputs.sha256_file(plan),
            }
        )
    )
    plan.write_bytes(b"other engine")
    with pytest.raises(ValueError, match="plan receipt"):
        benchmark_dino.run_case(args)
    assert not args.output.exists()


def test_fused_sampler_requires_its_onnx_operation_in_engine_metadata():
    inspection = {
        "Layers": [
            {
                "Name": "__myl_Grid",
                "LayerType": "kgen",
                "Metadata": "[ONNX Layer: /Transpose]\x1e[ONNX Layer: /GridSample]",
                "Inputs": [{"Format/Datatype": "Half"}, {"Format/Datatype": "Half"}],
            }
        ]
    }
    (observed,) = sampling_layers(inspection, {"/GridSample"})
    assert observed["operations"] == ["/GridSample"]
    assert observed["inputs"] == ["Half", "Half"]
    with pytest.raises(
        RuntimeError, match="Missing TensorRT layer evidence.*GridSample_1"
    ):
        sampling_layers(inspection, {"/GridSample", "/GridSample_1"})


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
