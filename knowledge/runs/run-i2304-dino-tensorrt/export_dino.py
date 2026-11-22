"""Export the real checkpoint at one pinned DINO input shape."""

import argparse
import os
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import torch
from batching import require_dynamic_axis
from benchmark import write_json
from benchmark_dino import (
    CHECKPOINT_SHA256,
    EXTENSION_SHA256,
    check_extension,
    verify_inputs,
)
from convert import require_queue
from fixed_inputs import (
    CHECKPOINT,
    REPOSITORY,
    load_sample,
    require_bucket,
    rows_for,
    sha256_file,
)
from measure_dino import BOXES, INPUT_NAME, LOGITS
from msda import export_onnx, export_sampling, upstream_modules

from src.submodules.models.dino.architecture import (
    build_dino,
    dino_4scale_swin_args,
    load_dino_state_dict,
)


class ExportModel(torch.nn.Module):
    def __init__(self, model, nested_tensor):
        super().__init__()
        self.model = model
        self.nested_tensor = nested_tensor

    def forward(self, pixels):
        # The production stack contains same-size images, so there is no padding.
        mask = torch.zeros(
            (pixels.shape[0], *pixels.shape[-2:]),
            dtype=torch.bool,
            device=pixels.device,
        )
        outputs = self.model(self.nested_tensor(pixels, mask))
        return outputs[LOGITS], outputs[BOXES]


def batched_window_reverse(windows, window_size, height, width):
    # The upstream int() captures the example batch in the legacy ONNX trace.
    channels = windows.shape[-1]
    tiles = windows.reshape(
        -1,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    return tiles.permute(0, 1, 3, 2, 4, 5).reshape(-1, height, width, channels)


def graph_files(path):
    import onnx

    model = onnx.load(path, load_external_data=False)
    files = {path.name}
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            attributes = {item.key: item.value for item in tensor.external_data}
            files.add(attributes["location"])
    return {file: sha256_file(path.parent / file) for file in sorted(files)}


def run(args):
    import subprocess

    import onnx

    lock = verify_inputs(args.inputs, args.bucket, args.checkpoint)
    check_extension()
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    shape = require_bucket(lock, args.bucket)
    rows = rows_for(lock, args.bucket, "eval")
    address = lock["buckets"][args.bucket]["representative"]
    sample = torch.from_numpy(
        load_sample(args.inputs, next(row for row in rows if row["file"] == address))
    )
    model, criterion = build_dino(
        REPOSITORY.resolve(), dino_4scale_swin_args("cpu", use_checkpoint=False)
    )
    del criterion
    model.load_state_dict(load_dino_state_dict(args.checkpoint), strict=True)
    model.eval()
    from util.misc import NestedTensor

    wrapper = ExportModel(model, NestedTensor).eval()
    graph = args.output / "export.onnx"
    print(f"Exporting {args.method} at {shape} to {graph}", flush=True)
    window_reverse = nullcontext()
    if args.dynamic_batch:
        layer, _ = upstream_modules()
        prefix = layer.__name__.split(".ops.")[0]
        swin = import_module(f"{prefix}.swin_transformer")
        window_reverse = patch.object(swin, "window_reverse", batched_window_reverse)
    with export_sampling(args.method, dynamic_batch=args.dynamic_batch), window_reverse:
        export_onnx(
            wrapper,
            (sample,),
            graph,
            inputs=[INPUT_NAME],
            outputs=[LOGITS, BOXES],
            dynamic_batch=args.dynamic_batch,
        )
    del wrapper, model
    onnx.checker.check_model(str(graph))
    if args.dynamic_batch:
        metadata = onnx.load(graph, load_external_data=False).graph
        for value in (*metadata.input, *metadata.output):
            require_dynamic_axis(value, 0, "batch")
    write_json(
        args.output / "export.json",
        {
            "method": args.method,
            "input_shape": shape,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "tensor_set_sha256": lock["tensor_set_sha256"],
            "extension_sha256": EXTENSION_SHA256,
            "dino_revision": subprocess.check_output(
                ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True
            ).strip(),
            "onnx": onnx.__version__,
            "opset": 19,
            "files": graph_files(graph),
            "queue_job": os.environ["TENNIS_RUN_ID"],
            **({"dynamic_batch": True} if args.dynamic_batch else {}),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("grid", "plugin"))
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", default=CHECKPOINT, type=Path)
    parser.add_argument("--dynamic-batch", action="store_true")
    args = parser.parse_args()
    require_queue()
    run(args)


if __name__ == "__main__":
    main()
