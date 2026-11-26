"""Export the pinned RF-DETR-L graph with the vendor FP16 cast and FP32 I/O."""

from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.submodules.models.rfdetr.person_detector import (
    ONNX_OPSET,
    RESOLUTION,
    load_rf_detr,
)


def export_rfdetr_onnx(checkpoint: Path, output: Path, batch_size: int) -> None:
    model = load_rf_detr(checkpoint, torch.device("cpu"))
    cast_graph = import_module("rfdetr.export._tensorrt.exporter").fp16_source_graph
    with TemporaryDirectory(dir=output.parent, prefix=".rfdetr-export-") as root:
        fp32 = model.export(output_dir=root, batch_size=batch_size, shape=(RESOLUTION, RESOLUTION),
                            opset_version=ONNX_OPSET, verbose=False)
        with cast_graph(str(fp32), dynamic_batch=False) as graph:
            Path(graph).rename(output)
