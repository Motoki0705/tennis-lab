"""Shared explicit runtime arguments for update-budgeted heatmap operations."""

import argparse

from src.tasks.ball_detection.training.runtime.compilation import COMPILE_MODES
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--jpeg-decoder", choices=("opencv", "nvjpeg"), required=True)
    parser.add_argument("--compile-mode", choices=COMPILE_MODES, required=True)
    parser.add_argument("--input-verification", choices=("upfront", "lazy"), required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--compile-recompile-limit", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--image-prefetch", action="store_true")


def runtime_from_arguments(args: argparse.Namespace) -> CoordinateRuntime:
    return CoordinateRuntime(**{key: getattr(args, key) for key in CoordinateRuntime.__dataclass_fields__})
