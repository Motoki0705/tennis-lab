"""Regenerate peaked heatmaps using only the pinned, pre-replacement model."""

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torch import nn

from src.submodules.configuration import ViTPoseHeadConfig

LEGACY_COMMIT = "3ad2bb3806e8e266d51878c7f27950512f93744f"
LEGACY_DIRECTORY = "src/submodules/vendor/gvhmr/vitpose"
FIXTURES = Path(__file__).resolve().parent


def _legacy_modules() -> dict[str, ModuleType]:
    # Execute the historical sources in memory; never copy GPL-derived code
    # into the fixture directory or import the replacement head/builder.
    modules = {}
    package_name = "_legacy_vitpose_reference"
    package = ModuleType(package_name)
    package.__path__ = []  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    for name in ("heatmap_head", "flip_utils", "kp2d_utils", "__init__"):
        filename = f"{LEGACY_DIRECTORY}/{name}.py"
        source = subprocess.check_output(
            ["git", "show", f"{LEGACY_COMMIT}:{filename}"], text=True,
        )
        module_name = package_name if name == "__init__" else f"{package_name}.{name}"
        module = package if name == "__init__" else ModuleType(module_name)
        module.__package__ = package_name
        sys.modules[module_name] = module
        exec(compile(source, filename, "exec"), module.__dict__)
        modules[name] = module
    return modules


def main() -> None:
    torch.set_num_threads(1)
    legacy = _legacy_modules()
    rng = np.random.default_rng(20261020)
    ys, xs = np.mgrid[:13, :11]
    # Off-grid centers avoid tied maxima, including for an Identity head.
    peak = np.exp(-((xs - 4.3) ** 2 + (ys - 5.1) ** 2) / (2 * 1.2**2))
    image = np.stack((peak, peak * 0.8, peak * 0.6)).astype(np.float32)[None]
    images = np.concatenate((image, image[..., ::-1]), axis=0)
    arrays = {}
    with np.load(FIXTURES / "heatmap_reference.npz", allow_pickle=False) as original:
        for config_key in original.files:
            if not config_key.endswith("/config"):
                continue
            prefix = config_key.removesuffix("config")
            values = json.loads(str(original[config_key]))
            for key in ("num_deconv_filters", "num_deconv_kernels", "num_conv_kernels"):
                values[key] = tuple(values[key])
            config = ViTPoseHeadConfig(**values)
            model = legacy["__init__"].VitPoseModel(
                nn.Conv2d(3, config.in_channels, 1),
                legacy["heatmap_head"].TopdownHeatmapSimpleHead(config),
            ).eval()
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                        weights = rng.uniform(0.5, 1.0, module.weight.shape)
                        # Positive, normalized kernels preserve the input peak.
                        weights /= module.in_channels * np.prod(module.kernel_size)
                        if isinstance(module, nn.ConvTranspose2d):
                            weights *= 4  # compensate for stride-2 upsampling
                        module.weight.copy_(torch.from_numpy(weights.astype(np.float32)))
                        if module.bias is not None:
                            module.bias.zero_()
                    elif isinstance(module, nn.BatchNorm2d):
                        module.weight.fill_(1)
                        module.bias.zero_()
                        module.running_mean.zero_()
                        module.running_var.fill_(1)
                        module.num_batches_tracked.zero_()
            arrays[prefix + "config"] = original[config_key]
            arrays[prefix + "images"] = images
            for key, tensor in model.state_dict().items():
                arrays[prefix + "state/" + key] = tensor.numpy()
            with torch.inference_mode():
                heatmaps = model(torch.from_numpy(images))
                averaged = (heatmaps[:1] + legacy["flip_utils"].flip_heatmap_coco17(heatmaps[1:])) * 0.5
            for name, tensor in (("heatmaps", heatmaps), ("flip_average", averaged)):
                value = tensor.numpy()
                count = len(value)
                coordinates, confidence = legacy["kp2d_utils"].keypoints_from_heatmaps(
                    value, center=np.tile([[100., 200.]], (count, 1)),
                    scale=np.tile([[0.9, 1.2]], (count, 1)), use_udp=True, kernel=3,
                )
                arrays[prefix + name + "_coordinates"] = coordinates
                arrays[prefix + name + "_confidence"] = confidence
    np.savez_compressed(FIXTURES / "udp_reference.npz", **arrays)


if __name__ == "__main__":
    main()
