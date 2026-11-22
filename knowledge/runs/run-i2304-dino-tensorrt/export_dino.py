"""Export the real checkpoint at one pinned DINO input shape."""

import argparse
import os
from pathlib import Path

import onnx
import torch
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
from msda import export_onnx, export_sampling

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


def graph_files(path):
    model = onnx.load(path, load_external_data=False)
    files = {path.name}
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            attributes = {item.key: item.value for item in tensor.external_data}
            files.add(attributes["location"])
    return {file: sha256_file(path.parent / file) for file in sorted(files)}


def run(args):
    import subprocess

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
    with export_sampling(args.method):
        export_onnx(
            wrapper, (sample,), graph, inputs=[INPUT_NAME], outputs=[LOGITS, BOXES]
        )
    del wrapper, model
    onnx.checker.check_model(str(graph))
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
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("grid", "plugin"))
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", default=CHECKPOINT, type=Path)
    args = parser.parse_args()
    require_queue()
    run(args)


if __name__ == "__main__":
    main()
