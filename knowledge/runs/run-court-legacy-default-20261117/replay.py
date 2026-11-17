"""Paired CPU replay: original source versus default public inference boundaries.

Never regenerate LINE fields or model outputs. No reference / GT is read.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
BASE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/tennis_scene/evaluate/court-legacy-default-20261117'
SAVED = ROOT / 'outputs/tennis_scene/evaluate/court-fixed-support-rgb-20261117'
RAW = ROOT / 'outputs/tennis_scene/evaluate/court-acceptance-implementation-20261116'
sys.path.insert(0, str(ROOT / 'knowledge/runs/run-court-acceptance-implementation-20261116'))
from legacy_geometry import hybrid_homography as legacy
from legacy_geometry.line_evidence import LineEvidence as LegacyEvidence
from src.tasks.court_detection.geometry import hybrid_homography as current
from src.tasks.court_detection.geometry.homography import court_template_xy, COURT_HOMOGRAPHY_EDGES
from src.tasks.court_detection.geometry.line_evidence import LineEvidence
from src.tasks.court_detection.inference import regions
from src.tasks.court_detection.inference.contracts import CourtPrediction
from src.tasks.court_detection.geometry.paint_observations import PaintConfig
from src.tasks.court_detection.model_io.contracts import CourtKeypointPrediction, CourtLinePrediction

spec = importlib.util.spec_from_file_location('baseline_regions', BASE / 'legacy_regions.py')
old_regions = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old_regions
spec.loader.exec_module(old_regions)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration():
    mapping = yaml.safe_load((ROOT / 'src/tennis_scene/configs/pipeline.yaml').read_text())['court_kp']
    support = current.HybridHomographyConfig(**mapping['postprocess'])
    region = dict(mapping['region_search'])
    region['paint'] = PaintConfig(**region['paint'])
    gate = regions.CourtRegionSearchConfig(**region)
    assert support == current.HybridHomographyConfig()
    assert asdict(gate) | {'enabled': False} == asdict(regions.CourtRegionSearchConfig())
    for name, value in asdict(legacy.HybridHomographyConfig()).items():
        assert getattr(support, name) == value
    return support, gate


def typed(raw, result):
    return CourtPrediction({
        'kp': CourtKeypointPrediction(torch.from_numpy(raw['raw_xy'][:, None]),
            torch.from_numpy(raw['scores'][:, None]), torch.from_numpy(raw['valid'][:, None]), torch.empty((14, 0, 0))),
        'line': CourtLinePrediction(torch.from_numpy(raw['line_probability']), torch.empty((0, 0)))},
        tuple(map(int, raw['image_size_hw'])), tuple(map(int, raw['native_size_hw'])), result,
        'synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1')


def equal(a, b):
    if a is None or b is None:
        return a is b
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def clean(v):
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def selection(api, predictions, image, config):
    queue = iter(predictions)
    def predict(crop, **kwargs):
        value = next(queue)
        assert tuple(crop.shape[:2]) == value.original_size_hw
        assert kwargs == {'postprocess': 'hybrid', 'heads': ('kp', 'line')}
        return value
    try:
        choice = api.select_court_region(SimpleNamespace(predict=predict), image, config)
        selected, candidates = choice.region, choice.candidates
    except api.CourtRegionUnavailable as error:
        selected, candidates = None, error.candidates
    assert next(queue, None) is None
    return selected, candidates


def process(record):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    support, gate = configuration()
    start = time.monotonic()
    image = cv2.cvtColor(cv2.imread(str(SAVED / 'rgb' / record['key'] / 'frame.png')), cv2.COLOR_BGR2RGB)
    assert hashlib.sha256(image.tobytes()).hexdigest() == record['rgb_sha256']
    assert tuple(image.shape[:2]) == tuple(record['source_size_hw'])
    assert regions.region_proposals(image) == old_regions.region_proposals(image)
    # Regression datasets evaluate full frame only; scene datasets search all proposals.
    proposals = tuple(tuple(r) for r in record['regions'])
    assert proposals == regions.region_proposals(image) or proposals == ((0, 0, image.shape[1], image.shape[0]),)
    before, after, cases, mismatches = [], [], [], []
    for index, region in enumerate(proposals):
        with np.load(RAW / record['old_raw_group'] / record['key'] / f'raw_{index:02d}.npz') as data:
            raw = dict(data)
        path = SAVED / 'line-evidence' / record['key'] / f'evidence_{index:02d}.npz'
        assert path.is_file(), f'Frozen LINE evidence missing: {path}'
        with np.load(path) as data:
            h, w = map(int, raw['image_size_hw'])
            evidence = LineEvidence(w, h, float(np.hypot(w, h)), data['distance'], data['points'], data['weights'],
                data['tangents'], data['coherence'], cKDTree(data['all_points']))
        observed = raw['raw_xy'].astype(np.float64)
        observed[~raw['valid']] = np.nan
        kwargs = dict(edges=np.asarray(COURT_HOMOGRAPHY_EDGES), image_size_hw=(h, w))
        with patch.object(LineEvidence, 'from_probability', return_value=evidence), patch.object(LegacyEvidence, 'from_probability', return_value=evidence):
            old = legacy.estimate_hybrid_homography(court_template_xy(), observed, raw['scores'], raw['line_probability'], **kwargs)
            new = current.estimate_hybrid_homography(court_template_xy(), observed, raw['scores'], raw['line_probability'], config=support, **kwargs)
        failures = [name for name in ('matrix', 'projected', 'selected', 'residuals_px') if not equal(getattr(old, name), getattr(new, name))]
        for name in ('status', 'candidate_count', 'refined_count', 'alternative_score_gap'):
            if getattr(old, name) != getattr(new, name): failures.append(name)
        old_typed, new_typed = typed(raw, old), typed(raw, new)
        before.append(old_typed); after.append(new_typed)
        old_region = old_regions.CourtRegionPrediction(old_typed, region, record['source_size_hw'])
        new_region = regions.assess_region_prediction(new_typed, image, region, gate)
        assert new_region.paint is None
        if not all(equal(a,b) for a,b in zip(old_region.downstream_keypoints(), new_region.downstream_keypoints())):
            failures.append('region_output')
        # A default no-search full-frame path must match CourtPrediction as well.
        if index == 0 and not all(equal(a,b) for a,b in zip(old_typed.downstream_keypoints(), new_region.downstream_keypoints())):
            failures.append('full_frame_output')
        if failures: mismatches.append({'region': index, 'fields': failures})
        cases.append({'region_index': index, 'region': region, 'status': old.status,
                      'old_h': old.matrix, 'new_h': new.matrix, 'exact': not failures})
    # Only the manifest's evaluated regions are supplied, identical on both sides.
    with patch.object(regions, 'region_proposals', return_value=proposals), patch.object(old_regions, 'region_proposals', return_value=proposals):
        old_choice, old_candidates = selection(old_regions, before, image, old_regions.CourtRegionSearchConfig())
        new_choice, new_candidates = selection(regions, after, image, gate)
    if old_choice != new_choice: mismatches.append({'fields':['selected_region']})
    if old_candidates != new_candidates: mismatches.append({'fields':['acceptance_diagnostics']})
    old_report = json.loads((SAVED / 'evaluation' / record['key'] / 'evaluation.json').read_text())
    historical = old_report['conditions']['legacy']
    historical_match = bool(old_choice is not None) == historical['accepted']
    if old_choice is not None: historical_match &= list(old_choice) == historical['region']
    result = clean({'key': record['key'], 'dataset': record['dataset'], 'regions': len(cases),
        'mismatches': mismatches, 'all_exact': not mismatches, 'old_selected_region': old_choice,
        'new_selected_region': new_choice, 'accepted': old_choice is not None,
        'any_h': any(v.homography.matrix is not None for v in before),
        'historical_run73_choice_match': historical_match,
        'old_candidates': old_candidates, 'new_candidates': new_candidates, 'geometry': cases,
        'seconds':time.monotonic()-start})
    dest = OUT / 'evaluation' / record['key'] / 'comparison.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, allow_nan=False))
    return {k:v for k,v in result.items() if k not in ('geometry','old_candidates','new_candidates')}


def freeze():
    source = json.loads((BASE / 'baseline-source.json').read_text())
    for path, metadata in source['files'].items(): assert digest(Path(path)) == metadata['sha256']
    paths = sorted(set(Path('src/tasks/court_detection/geometry').glob('*.py')) |
        set(Path('src/tasks/court_detection/inference').glob('*.py')) |
        {Path('src/tennis_scene/configs/pipeline.yaml'), Path('src/tennis_scene/pipeline/components/court_kp.py'),
         Path('src/tennis_scene/configuration.py'), BASE/'replay.py'})
    return {'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'files':{str(p):digest(p) for p in paths}, 'manifest_sha256':digest(SAVED/'manifest.json'),
        'baseline_commit':source['git_commit'], 'support':asdict(configuration()[0]), 'region':asdict(configuration()[1]),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu_used':False,
        'comparison':'same frozen LINE arrays; exact dtype/shape/bytes for H and projected/mask/residual arrays'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=10);p.add_argument('--limit',type=int)
    args=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    protocol=freeze();(BASE/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    records=json.loads((SAVED/'manifest.json').read_text())['records']
    if args.limit:records=records[:args.limit]
    start=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(process,r) for r in records]):
            row=future.result();rows.append(row);print(json.dumps(row),flush=True)
    assert protocol['files']==freeze()['files'], 'Source changed during replay'
    summary={'images':len(rows),'regions':sum(r['regions'] for r in rows),'seconds':time.monotonic()-start,
        'workers':args.workers,'limit':args.limit,'datasets':{},'mismatches':[r for r in rows if not r['all_exact']],
        'historical_run73_choice_mismatches':[r['key'] for r in rows if not r['historical_run73_choice_match']]}
    for dataset in sorted({r['dataset'] for r in rows}):
        values=[r for r in rows if r['dataset']==dataset]
        summary['datasets'][dataset]={k:sum(int(r[k]) for r in values) for k in ('regions','accepted','any_h','all_exact')}
        summary['datasets'][dataset]['images']=len(values)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary));assert not summary['mismatches']


if __name__=='__main__': main()
