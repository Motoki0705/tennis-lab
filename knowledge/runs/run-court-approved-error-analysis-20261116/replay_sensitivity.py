"""Check whether remaining frozen/replay differences come from instrumentation."""
from pathlib import Path
import hashlib
import json
import time
import cv2
import numpy as np
from src.tasks.court_detection.geometry import hybrid_homography as production
from regression_factorial import module_for
from common import BASE, OLD, WORLD, EDGES, write_json


def main():
    cv2.setNumThreads(1)
    manifest = json.loads((BASE / 'regression-factorial/manifest.json').read_text())['cases']
    records = {r['key']: r for r in json.loads((OLD / 'test_raw/manifest.json').read_text())['records']}
    replays = {d['key']: d for f in (BASE / 'regression-factorial/test').glob('*/o1s1v1.json') if (d := json.loads(f.read_text()))}
    cases = [r for r in manifest if r['dataset'] == 'test' and r['frozen_status']['current'] != replays[r['key']]['status']]
    original_descriptor = production.LineEvidence.__dict__['from_probability']
    original = production.LineEvidence.from_probability
    evidence_hashes = {}
    def capture(cls, *args, **kwargs):
        e = original(*args, **kwargs)
        evidence_hashes.clear()
        evidence_hashes.update({name: hashlib.sha256(np.asarray(getattr(e, name)).tobytes()).hexdigest() for name in ('distance', 'points', 'weights', 'tangents', 'coherence')})
        return e
    production.LineEvidence.from_probability = classmethod(capture)
    rows = []
    try:
        for case in cases:
            r = records[case['key']]; raw = np.load(OLD / 'test_raw' / r['key'] / 'raw_00.npz')
            observed = raw['raw_xy'].astype(float); observed[~raw['valid']] = np.nan
            for label, function in [('production_no_rng_reset', production.estimate_hybrid_homography),
                                    ('instrumented_rng2190', module_for(1, 1, 1).estimate_hybrid_homography),
                                    ('production_rng2190', production.estimate_hybrid_homography)]:
                if label != 'production_no_rng_reset': cv2.setRNGSeed(2190)
                start = time.monotonic(); result = function(WORLD, observed, raw['scores'], raw['line_probability'], edges=EDGES, image_size_hw=tuple(raw['image_size_hw']))
                rows.append({'key': r['key'], 'trial': label, 'frozen_status': case['frozen_status']['current'],
                    'factorial_status': replays[r['key']]['status'], 'status': result.status, 'seconds': time.monotonic() - start,
                    'evidence_hashes': evidence_hashes.copy(), 'candidate_count': result.candidate_count,
                    'refined_count': result.refined_count, 'score_gap': result.alternative_score_gap})
                print(r['key'], label, result.status, flush=True)
                write_json(BASE / 'replay-sensitivity.json', rows)
    finally:
        production.LineEvidence.from_probability = original_descriptor


if __name__ == '__main__': main()
