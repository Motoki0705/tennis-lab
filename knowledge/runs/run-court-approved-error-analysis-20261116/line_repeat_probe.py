"""Repeated identical binary masks: record the installed OpenCV distance-field variation."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from src.tasks.court_detection.geometry.line_evidence import LineEvidence

BASE = Path(__file__).resolve().parent


def main():
    path = BASE.parent / 'court-acceptance-implementation-20261116/test_raw/synthetic_court_test/B00_court-sample-001348/raw_00.npz'
    raw = np.load(path); cv2.setNumThreads(1)
    original = cv2.distanceTransform; calls = []
    def tracked(array, *args, **kwargs):
        result = original(array, *args, **kwargs)
        calls.append({'input': hashlib.sha256(array.tobytes()).hexdigest(), 'output': hashlib.sha256(result.tobytes()).hexdigest(), 'nonzero': int((array > 0).sum())})
        return result
    cv2.distanceTransform = tracked
    rows = []; first = None
    try:
        for repeat in range(12):
            calls.clear(); cv2.setRNGSeed(2190)
            evidence = LineEvidence.from_probability(raw['line_probability'], tuple(raw['image_size_hw']))
            hashes = {k: hashlib.sha256(np.asarray(getattr(evidence, k)).tobytes()).hexdigest() for k in ['distance', 'points', 'tangents', 'weights', 'coherence']}
            if first is None: first = evidence
            rows.append({'repeat': repeat, 'hashes': hashes, 'distance_transform_calls': calls.copy(), 'point_count': len(evidence.points),
                'distance_max_delta': float(np.max(abs(evidence.distance - first.distance))), 'same_points': np.array_equal(evidence.points, first.points)})
    finally:
        cv2.distanceTransform = original
    (BASE / 'line-repeat-probe.json').write_text(json.dumps({'case': str(path), 'opencv': cv2.__version__, 'threads': cv2.getNumThreads(), 'records': rows}, indent=2) + '\n')


if __name__ == '__main__': main()
