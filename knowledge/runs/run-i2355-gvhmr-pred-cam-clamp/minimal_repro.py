"""Generate and compare the minimal in-place camera-clamp ONNX graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class InplaceCameraClamp(nn.Module):
    def forward(self, raw_cam):
        pred_cam = raw_cam.clone()
        torch.clamp_min_(pred_cam[..., 0], 0.25)
        return pred_cam


def asymmetric_camera_input(batch: int, frames: int, dtype: np.dtype) -> np.ndarray:
    values = np.arange(batch * frames * 3, dtype=np.float32).reshape(batch, frames, 3)
    values[..., 0] = np.linspace(-1.0, 1.0, batch * frames).reshape(batch, frames)
    values[..., 1] = values[..., 1] * -0.125 - 3.0
    values[..., 2] = values[..., 2] * 0.0625 + 7.0
    return values.astype(dtype)


def export_minimal(path: Path, *, batch: int, frames: int, dtype: str) -> np.ndarray:
    if dtype not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported minimal dtype {dtype!r}")
    numpy_dtype = np.float16 if dtype == "fp16" else np.float32
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    inputs = asymmetric_camera_input(batch, frames, numpy_dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        InplaceCameraClamp().eval(),
        (torch.from_numpy(inputs).to(torch_dtype),),
        str(path),
        input_names=["raw_cam"],
        output_names=["pred_cam"],
        opset_version=19,
        dynamo=False,
        external_data=False,
    )
    return inputs


def run_onnx(path: Path, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    values = session.run(None, inputs)
    return {
        output.name: value
        for output, value in zip(session.get_outputs(), values, strict=True)
    }


def compare_minimal(
    original: Path, rewritten: Path, raw_cam: np.ndarray
) -> dict[str, object]:
    expected = raw_cam.copy()
    expected[..., 0] = np.maximum(expected[..., 0], np.array(0.25, dtype=raw_cam.dtype))
    source = run_onnx(original, {"raw_cam": raw_cam})["pred_cam"]
    actual = run_onnx(rewritten, {"raw_cam": raw_cam})["pred_cam"]
    np.testing.assert_array_equal(source, expected)
    np.testing.assert_array_equal(actual, expected)
    return {
        "shape": list(raw_cam.shape),
        "dtype": str(raw_cam.dtype),
        "source_matches_pytorch": True,
        "rewrite_matches_pytorch": True,
        "source_rewrite_max_abs": float(
            np.max(np.abs(source.astype(np.float64) - actual.astype(np.float64)))
        ),
    }
