"""Fixed inputs and units for the approved-reference analysis."""
from pathlib import Path
import hashlib
import json
import numpy as np
from src.tasks.court_detection.geometry.homography import court_template_xy, COURT_HOMOGRAPHY_EDGES

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / 'court-acceptance-implementation-20261116'
WHITE = BASE.parent / 'court-reference-white-lines-20261116'
WORLD = court_template_xy().astype(float)
EDGES = np.array(COURT_HOMOGRAPHY_EDGES)
CONDITIONS = ('legacy', 'acceptance_only', 'postprocess_and_acceptance')
REFS = json.loads((WHITE / 'reference/references.json').read_text())
assert REFS['human_reviewed'] and REFS['approved_for_evaluation']


def project(matrix):
    a = np.c_[WORLD, np.ones(len(WORLD))] @ np.asarray(matrix).T
    return a[:, :2] / a[:, 2, None]


def truth(camera):
    reference = next(c for c in REFS['cameras'] if c['camera'] == camera)
    xy = project(reference['matrix_court_metres_to_image_pixels'])
    mask = (xy[:, 0] >= 0) & (xy[:, 0] <= 1919) & (xy[:, 1] >= 0) & (xy[:, 1] <= 1079)
    return xy, mask


def errors(points, camera):
    if points is None: return np.array([])
    gt, mask = truth(camera)
    return np.linalg.norm(np.asarray(points)[mask] - gt[mask], axis=1)


def correct(points, camera):
    e = errors(points, camera)
    return bool(len(e) and e.mean() <= 10 and e.max() <= 25)


def distribution(values):
    x = np.asarray(values, float); x = x[np.isfinite(x)]
    return {'n': len(x), 'mean': float(x.mean()) if len(x) else None,
        'median': float(np.median(x)) if len(x) else None,
        'p90': float(np.percentile(x, 90)) if len(x) else None,
        'p95': float(np.percentile(x, 95)) if len(x) else None,
        'max': float(x.max()) if len(x) else None}


def source_rows():
    paths = sorted((OLD / 'meiji4-evaluation/meiji_4cam').glob('**/evaluation.json'))
    assert len(paths) == 276
    return [(path, json.loads(path.read_text())) for path in paths]


def source_hash():
    digest = hashlib.sha256()
    for path, _ in source_rows():
        digest.update(str(path.relative_to(OLD)).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
