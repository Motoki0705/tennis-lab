"""Summarize all feasibility variants without changing their frozen evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from src.tasks.court_detection.geometry.video_shared.fit import SharedFit, SharedPaintConfig, choose_common_solution


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for label, subdirectory in [('R1', ''), ('R2', 'endpoints'), ('R3', 'all-fit'), ('R4', 'service-order')]:
        folder = args.source / subdirectory
        summary = json.loads((folder / 'summary.json').read_text())
        protocol = json.loads((folder / 'protocol.json').read_text())
        result[label] = {'head': protocol['head'], 'seconds': summary['seconds'],
                         'summary_sha256': sha(folder / 'summary.json'), 'protocol_sha256': sha(folder / 'protocol.json'),
                         'source_hashes': protocol['source_hashes'], 'settings': protocol['settings'],
                         'cameras': {c: {k: v[k] for k in ('initializers', 'accepted_candidates', 'accepted_individual', 'reasons') if k in v}
                                     for c, v in summary['cameras'].items()}}
        if label == 'R4':
            result[label]['fixed_selection_instead_of_merging'] = {}
            for camera, value in summary['cameras'].items():
                solutions = tuple(SharedFit(None if row['matrix'] is None else np.array(row['matrix']),
                                           None if row['fitted_matrix'] is None else np.array(row['fitted_matrix']),
                                           tuple(row['reasons']), row['diagnostic']) for row in value['individual'])
                selected = choose_common_solution(solutions, SharedPaintConfig())
                result[label]['fixed_selection_instead_of_merging'][camera] = {
                    'admissible': selected.matrix is not None, 'reasons': selected.reasons,
                    'max_group_p95_px': selected.diagnostic.get('max_group_p95_px'),
                    'selected_matrix_sha256': None if selected.matrix is None else hashlib.sha256(selected.matrix.tobytes()).hexdigest()}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
