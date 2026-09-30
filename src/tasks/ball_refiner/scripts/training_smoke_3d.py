"""Run the explicitly CPU-only, saved-rally flow matching diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tasks.ball_refiner.refiner_3d.diffusion.training_smoke import (
    run_training_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if not all(p.is_absolute() for p in (args.dataset,args.config,args.output)):
        parser.error('All paths must be absolute')
    print(json.dumps(run_training_smoke(args.dataset,args.config,args.output),indent=2))


if __name__=='__main__':
    main()
