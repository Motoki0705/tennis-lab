"""Record both CPU dispatch results for one explicitly selected DINO binary."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

from src.utils.checksum import dual_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.extension_directory.resolve(strict=True)
    sys.path.insert(0, str(directory))
    extension = importlib.import_module("MultiScaleDeformableAttention")
    path = Path(str(extension.__file__)).resolve(strict=True)
    if path.parent != directory:
        raise ValueError(f"Imported another extension: {path}")
    value = torch.ones((1, 1, 1, 1), device="cpu")
    shapes = torch.tensor([[1, 1]], dtype=torch.int64, device="cpu")
    starts = torch.tensor([0], dtype=torch.int64, device="cpu")
    locations = torch.full((1, 1, 1, 1, 1, 2), .5, device="cpu")
    weights = torch.ones((1, 1, 1, 1, 1), device="cpu")
    gradient = torch.ones((1, 1, 1), device="cpu")
    results = {}
    for entry, tail in (("forward", ()), ("backward", (gradient,))):
        function = getattr(extension, f"ms_deform_attn_{entry}")
        try:
            function(value, shapes, starts, locations, weights, *tail, 1)
        except RuntimeError as error:
            results[entry] = str(error)
        else:
            results[entry] = "unexpected_cpu_success"
    print(json.dumps({
        "torch": torch.__version__, "extension": str(path), "sha256": dual_sha256(path),
        "cpu_dispatch": results, "cuda_initialized": torch.cuda.is_initialized(),
    }, indent=2))


if __name__ == "__main__":
    main()
