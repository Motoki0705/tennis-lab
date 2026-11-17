"""Reuse frozen RGB measurements; enforce service order and explicit drift scope."""
from dataclasses import asdict
from pathlib import Path
import argparse
from collections import Counter
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from src.tasks.court_detection.geometry.video_shared.consensus import merge_measurements
from src.tasks.court_detection.geometry.video_shared.fit import PaintGroup, SharedPaintConfig, fit_common_court
from probe import save, sha


def main(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU only')
    started = time.monotonic()
    protocol = json.loads((args.source / 'protocol.json').read_text())
    files = sorted((args.source / 'measurements').glob('*.json'))
    inventory = [{'path': str(p.relative_to(args.source)), 'sha256': sha(p)} for p in files]
    frozen = {'schema': 'court_service_order_replay_v1', 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_hashes': {str(p): sha(p) for p in sorted(Path('src/tasks/court_detection/geometry/video_shared').glob('*.py'))},
              'source_protocol_sha256': sha(args.source / 'protocol.json'), 'driver_sha256': sha(Path(__file__)),
              'settings': asdict(SharedPaintConfig()), 'reference_h_used': False, 'evaluation_frames_used': False}
    save(args.output / 'input-inventory.json', inventory)
    frozen['input_inventory_sha256'] = sha(args.output / 'input-inventory.json')
    save(args.output / 'protocol.json', frozen)
    results = [json.loads(p.read_text()) for p in files]
    summary = {'cameras': {}}
    for camera, seeds in protocol['candidates'].items():
        hypotheses = []
        individual = []
        for index, seed in enumerate(seeds):
            selected = [r for r in results if r['camera'] == camera and r['initializer'] == index]
            groups = tuple(PaintGroup(g['clip_id'], g['frame_index'], g['line_index'], np.array(g['points']), tuple(g['endings']))
                           for r in selected for g in r['groups'])
            fit = fit_common_court(np.array(seed['matrix']), groups, (1920, 1080), SharedPaintConfig())
            individual.append({'initializer': index, 'matrix': None if fit.matrix is None else fit.matrix.tolist(),
                               'fitted_matrix': None if fit.fitted_matrix is None else fit.fitted_matrix.tolist(),
                               'reasons': fit.reasons, 'diagnostic': fit.diagnostic,
                               'frame_reasons': dict(Counter(r['diagnostic']['reason'] for r in selected))})
            if fit.matrix is not None:
                hypotheses.append(groups)
        merged = None if not hypotheses else merge_measurements(np.array(seeds[0]['matrix']), tuple(hypotheses), (1920, 1080), SharedPaintConfig())
        summary['cameras'][camera] = {'individual': individual, 'accepted_individual': len(hypotheses),
                                      'matrix': None if merged is None or merged.matrix is None else merged.matrix.tolist(),
                                      'reasons': ['no_acceptable_shared_court'] if merged is None else merged.reasons,
                                      'diagnostic': None if merged is None else merged.diagnostic}
        print(camera, 'qualified initializers',len(hypotheses),'reasons',summary['cameras'][camera]['reasons'],flush=True)
    summary['seconds'] = time.monotonic() - started
    for path, digest in frozen['source_hashes'].items():
        assert sha(Path(path)) == digest
    assert all(sha(args.source / row['path']) == row['sha256'] for row in inventory)
    save(args.output / 'summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
