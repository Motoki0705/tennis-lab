"""Compare three TRT boundaries against the pinned CUDA op on a small DINO-shaped block."""

import argparse
import os
import time
import traceback
from pathlib import Path

import numpy as np
import onnx
import torch
from benchmark import write_json
from benchmark_dino import check_extension, distribution
from convert import build_checked_plan, require_queue, verify_compute_precision
from measure_dino import DinoTRTContext, setup_cuda
from msda import (
    export_onnx,
    grid_core,
    plugin_core,
    sampling_layers,
    sampling_names,
    upstream_modules,
)
from runtime import DeviceMemory, stats

GEOMETRY = ((25, 42), (13, 21), (7, 11), (4, 6))
QUERY_COUNT = 900
HEADS = 8
CHANNELS = 32
POINTS = 4
SEED = 2304


class SampleBlock(torch.nn.Module):
    def __init__(self, method, projection, geometry, starts, reference):
        super().__init__()
        self.method = method
        self.projection = projection
        self.register_buffer("geometry", geometry.int())
        self.register_buffer("starts", starts.int())
        self.reference = reference

    def forward(self, pixels, locations=None, weights=None):
        value = self.projection(pixels).unflatten(-1, (HEADS, CHANNELS))
        if self.method == "partial":
            return value
        if self.method == "grid":
            return grid_core(self.reference, value, self.geometry, locations, weights)
        return plugin_core(value, self.geometry, self.starts, locations, weights)


def evidence(details, names, method):
    return {
        **verify_compute_precision(
            details, "fp16", expected_operations={"/projection/MatMul": "fp16"}
        ),
        "sampling_layers": sampling_layers(details, names, method=method),
    }


def measure(fn):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(100):
        start = time.perf_counter_ns()
        result = fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - start) / 1e6)
    return result, stats(times)


def run_method(
    method,
    output,
    projection,
    geometry,
    starts,
    reference,
    tensors,
    native,
    pixels,
    locations,
    weights,
):
    graph = output / f"{method}.onnx"
    plan = output / f"{method}.plan"
    tensors = tensors[:1] if method == "partial" else tensors
    names = ["pixels"] if method == "partial" else ["pixels", "locations", "weights"]
    block = SampleBlock(method, projection, geometry, starts, reference)
    export_onnx(block, tensors, graph, inputs=names, outputs=["result"])
    onnx_graph = onnx.load(graph, load_external_data=False).graph
    sampling = sampling_names(
        onnx_graph, method, {"grid": 4, "plugin": 1, "partial": 0}[method]
    )
    checked = build_checked_plan(
        graph,
        plan,
        verify=lambda details: evidence(details, sampling, method),
        init_plugins=method == "plugin",
        min_free_gib=11,
        allow_tf32=False,
    )
    flat_value_shape = (*pixels.shape[:-1], HEADS, CHANNELS)
    runner = DinoTRTContext(
        plan,
        input_shapes={
            name: tuple(t.shape) for name, t in zip(names, tensors, strict=True)
        },
        output_shapes={
            "result": flat_value_shape
            if method == "partial"
            else (1, QUERY_COUNT, HEADS * CHANNELS)
        },
        init_plugins=method == "plugin",
    )
    output_buffer = runner.outputs["result"]
    for name, tensor in zip(
        names, (pixels, locations, weights)[: len(names)], strict=True
    ):
        runner.inputs[name].copy_(tensor)

    def forward():
        runner.execute()
        if method == "partial":
            return native(output_buffer.float(), locations, weights)
        return output_buffer

    value, timing = measure(forward)
    return value.float().cpu().numpy(), timing, checked


@torch.no_grad()
def run(output):
    require_queue()
    check_extension()
    output.mkdir(parents=True)
    _, upstream = upstream_modules()
    generator = torch.Generator().manual_seed(SEED)
    geometry = torch.tensor(GEOMETRY, dtype=torch.long)
    areas = geometry.prod(1)
    starts = torch.cat((torch.zeros(1, dtype=torch.long), areas.cumsum(0)[:-1]))
    tokens = int(areas.sum())
    dims = HEADS * CHANNELS
    projection = torch.nn.Linear(dims, dims).eval()
    projection.weight.copy_(torch.randn((dims, dims), generator=generator) * dims**-0.5)
    projection.bias.zero_()
    pixels = torch.randn((1, tokens, dims), generator=generator)
    locations = (
        torch.rand(
            (1, QUERY_COUNT, HEADS, len(GEOMETRY), POINTS, 2), generator=generator
        )
        * 1.4
        - 0.2
    )
    weights = torch.rand(
        (1, QUERY_COUNT, HEADS, len(GEOMETRY) * POINTS), generator=generator
    )
    weights = weights.softmax(-1).unflatten(-1, (len(GEOMETRY), POINTS))
    summary = {
        "queue_job": os.environ["TENNIS_RUN_ID"],
        "geometry": GEOMETRY,
        "queries": QUERY_COUNT,
        "tokens": tokens,
        "seed": SEED,
        "heads": HEADS,
        "channels_per_head": CHANNELS,
        "points": POINTS,
        "methods": {},
    }
    mem = DeviceMemory()
    try:
        setup_cuda(mem)
        torch.cuda.set_stream(torch.cuda.Stream())
        gpu_geometry, gpu_starts = geometry.cuda(), starts.cuda()
        cpu_inputs = (pixels.half(), locations.half(), weights.half())
        gpu_pixels, gpu_locations, gpu_weights = (
            t.cuda() for t in (pixels, locations, weights)
        )

        def native(value, positions, coeffs):
            return upstream.MSDA.ms_deform_attn_forward(
                value, gpu_geometry, gpu_starts, positions, coeffs, 64
            )

        gpu_projection = torch.nn.Linear(dims, dims, device="cuda").eval()
        gpu_projection.load_state_dict(projection.state_dict())
        actual, timing = measure(
            lambda: native(
                gpu_projection(gpu_pixels).unflatten(-1, (HEADS, CHANNELS)),
                gpu_locations,
                gpu_weights,
            )
        )
        expected = actual.cpu().numpy()
        summary["fp32_torch_reference"] = timing
        summary["methods"] = {}
        projection.half()
        for method in ("grid", "plugin", "partial"):
            try:
                value, timing, checked = run_method(
                    method,
                    output,
                    projection,
                    geometry,
                    starts,
                    upstream.ms_deform_attn_core_pytorch,
                    cpu_inputs,
                    native,
                    gpu_pixels,
                    gpu_locations,
                    gpu_weights,
                )
                if not np.isfinite(value).all():
                    raise ValueError("Nonfinite MSDA output")
                summary["methods"][method] = {
                    "status": "measured",
                    "timing": timing,
                    "inspection": checked,
                    "absolute_error": distribution(np.abs(expected - value)),
                    "relative_l2": float(
                        np.linalg.norm(expected - value) / np.linalg.norm(expected)
                    ),
                }
            except (RuntimeError, ValueError, TypeError):
                error = traceback.format_exc()
                print(error, flush=True)
                (output / f"{method}.txt").write_text(error)
                summary["methods"][method] = {
                    "status": "failed",
                    "error": error.splitlines()[-1],
                }
            write_json(output / "comparison.json", summary)
    finally:
        mem.close()
    return int(
        any(result["status"] != "measured" for result in summary["methods"].values())
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.output))


if __name__ == "__main__":
    main()
