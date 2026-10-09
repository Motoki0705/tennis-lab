"""CPU/meta parameter and Conv/Linear MAC counts; not a latency/FLOPs benchmark."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.tasks.ball_detection.models.mdd_pretrain import (
    DeepMDDQueryDetector,
    MDDDPTDetector,
    MDDPretrainConfig,
)
from src.tasks.ball_detection.training.posttraining.paths import resolver
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(name="ball_detection.profile_mdd_cnn", fields=(
    BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.FILE),
))


def profile() -> dict[str, Any]:
    rows = []
    for variant, filename in (("residual", "mdd_dpt_pretrain.yaml"), ("convnext_v2", "mdd_dpt_convnext_v2.yaml"),
                              ("fasternet", "mdd_dpt_fasternet.yaml")):
        config = MDDPretrainConfig.load(PROJECT_ROOT / "src/tasks/ball_detection/configs/model" / filename)
        with torch.device("meta"):
            model = MDDDPTDetector(config)
            query = DeepMDDQueryDetector(config)
        operations = [0]

        def count(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor, counter: list[int] = operations) -> None:
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                counter[0] += output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
            elif isinstance(module, nn.Linear):
                counter[0] += output.numel() * module.in_features

        hooks = [module.register_forward_hook(count) for module in model.modules()
                 if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear))]
        with torch.no_grad():
            # Fixed RGB->MDD has no Conv/Linear operations, and autocast does not support meta.
            features = model.encoder(torch.empty(1, 2, 32, 720, 1280, device="meta"))
            fused = model.decoder([feature.flatten(0, 1) for feature in features])
            output = model.head(F.interpolate(fused, size=(180, 320), mode="bilinear", align_corners=False))
            assert output.shape == (32, 1, 180, 320)
        for hook in hooks:
            hook.remove()
        rows.append(dict(variant=variant, pretraining_parameters=sum(p.numel() for p in model.parameters()),
            encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
            posttraining_parameters=sum(p.numel() for p in query.parameters()),
            conv_linear_macs_per_window=operations[0]))
    return dict(input_shape=[1, 32, 3, 720, 1280], rows=rows,
                limitation="Conv/Linear MACs only; excludes normalization, activation, MDD, memory traffic and backward. Not measured speed.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = PATH_BOUNDARY.validate({"output": args.output}, resolver=resolver(PROJECT_ROOT,
        (PROJECT_ROOT,), (args.output,), args.output.parent))
    args.output = paths.declared("output").path
    args.output.write_text(json.dumps(profile(), indent=2))
    print(args.output.read_text())


if __name__ == "__main__":
    main()
