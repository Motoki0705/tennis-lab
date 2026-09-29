"""Check all refined posteriors in the generator's exported float32 precision."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--audit', type=Path, required=True)
parser.add_argument('--summary', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
summary = json.loads(args.summary.read_text())
rule = summary['rule']
count, mass_error, minimum_depth = 0, 0., float('inf')
for source in summary['source_rallies']:
    with np.load(args.source / (source['rally_id'] + '.npz'), allow_pickle=False) as z:
        subsets = z['gmm3d_camera_subsets'][0]
        p = torch.sigmoid(torch.from_numpy(z['gmm2d_presence_logits'])).numpy().T.astype(np.float64)
        r, t = z['camera_estimated_R'], z['camera_estimated_t']
    for path in sorted(args.audit.glob(source['rally_id'] + '-*.npz')):
        with np.load(path, allow_pickle=False) as z:
            frames = z['frames']
            means, cov, weights = (z[key].astype(np.float32) for key in ('means', 'covariance', 'weights'))
            assert means.shape == (len(frames), 64, 3) and cov.shape == (len(frames), 64, 3, 3) and weights.shape == (len(frames), 64)
            assert all(np.isfinite(value).all() for value in (means, cov, weights))
            np.linalg.cholesky(cov)
            np.testing.assert_allclose(weights.sum(-1), 1, atol=1e-6)
            assert (weights >= 0).all()
            for camera in range(3):
                depth = means.astype(np.float64) @ r[camera, 2] + t[camera, 2]
                active = depth[:, subsets[:, camera]]
                assert (active > 0).all()
                minimum_depth = min(minimum_depth, float(active.min()))
            for mask in np.unique(subsets, axis=0):
                selected = (subsets == mask).all(-1)
                assert selected.sum() == 3 ** int(mask.sum())
                expected = np.prod(np.where(mask, p[frames], 1 - p[frames]), axis=-1)
                delta = np.abs(weights[:, selected].sum(-1) - expected)
                assert (delta < 1.e-6).all()
                mass_error = max(mass_error, float(delta.max()))
            expected_flags = (z['component_changes'] <= [rule['log_evidence_tolerance_nat'], rule['mean_tolerance'], rule['covariance_relative_tolerance']]).all(-1)
            np.testing.assert_array_equal(expected_flags, z['component_converged'])
            count += len(frames)
assert count == summary['frames'] == 4809
report = {'frames_verified': count, 'components_per_frame': 64, 'float32_spd': True, 'positive_active_camera_means': True,
          'minimum_active_depth_after_float32': minimum_depth, 'maximum_camera_subset_mass_error_after_float32': mass_error,
          'component_flags_match_achieved_tolerances': True}
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
