"""Re-extract the frozen pilot validation bank into a new directory."""
import argparse
from pathlib import Path

from src.tasks.ball_refiner.refiner_3d.synthetic.calibration import build_calibration

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if not args.source.is_absolute() or not args.output.is_absolute():
    parser.error('Use absolute paths')
build_calibration(args.source, args.output)
