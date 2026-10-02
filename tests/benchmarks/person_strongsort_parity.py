"""CPU interoperability check with a separately obtained GPL reference checkout.

Only synthetic trajectories are used; no Meiji labels or data are read. Upstream
source and checkpoint are external inputs and are not redistributed by this repo.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.tasks.person_tracking.strongsort_offline import AFLink, aflink_inputs
from src.utils.checksum import dual_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'weight', 'report'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.upstream))
    dataset = importlib.import_module('AFLink.dataset').LinkData('', mode='test')
    reference = importlib.import_module('AFLink.model').PostLinker().eval()
    reference.load_state_dict(torch.load(args.weight, map_location='cpu', weights_only=True), strict=True)
    candidate = AFLink(args.weight).model
    equal_keys = set(reference.state_dict()) == set(candidate.state_dict())
    if not equal_keys:
        raise ValueError('AFLink state keys differ')
    rng = np.random.default_rng(964)
    trials = []
    for former, later in ((1, 1), (2, 2), (30, 30), (40, 10), (10, 50)):
        a = np.column_stack((np.arange(former), rng.normal(150, 4, (former, 2))))
        b = np.column_stack((np.arange(former + 2, former + 2 + later), rng.normal(150, 4, (later, 2))))
        x, y = dataset.transform(a.copy(), b.copy())
        xx, yy = aflink_inputs(a, b)
        with torch.inference_mode():
            p, q = reference(x[None], y[None]), candidate(xx, yy)
        error = float((p - q).abs().max())
        equal = torch.equal(x[None], xx) and torch.equal(y[None], yy)
        if not equal or error > 1e-6:
            raise ValueError(f'AFLink synthetic parity failed: {former}/{later}, input={equal}, output={error}')
        trials.append({'former_length': former, 'later_length': later, 'inputs_equal': equal, 'max_abs_diff': error})
    result = {'status': 'ok', 'state_keys': len(candidate.state_dict()), 'state_keys_equal': equal_keys,
              'weight_sha256': dual_sha256(args.weight), 'trials': trials,
              'upstream': {str(p.relative_to(args.upstream)): dual_sha256(p)
                           for p in sorted(args.upstream.rglob('*.py'))},
              'implementation_sha256': dual_sha256(Path(__file__).resolve().parents[2] / 'src/tasks/person_tracking/strongsort_offline.py'),
              'scope': 'AFLink input preprocessing and learned inference only; no full tracker parity claim'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
