"""Measure frozen model evidence without running a neural model or changing H."""
from collections import defaultdict
import csv
import json
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

from src.tasks.court_detection.geometry.confidence_homography import estimate_confidence_homography
from src.tasks.court_detection.geometry.line_evidence import LineEvidence, clip_segments, segment_distances
from src.tasks.court_detection.geometry.observations import keypoint_support
from common import BASE, OLD, WHITE, WORLD, EDGES, CONDITIONS, correct, distribution, source_rows, truth, write_json

FAR = {0, 1, 4, 6, 8, 9, 12}
LINE_NAMES = ['far_baseline', 'near_baseline', 'left_doubles', 'right_doubles', 'left_singles', 'right_singles', 'far_service', 'near_service', 'centre_service']


def line_measure(probability, region, reference):
    x, y, right, bottom = region; h, w = bottom - y, right - x
    native = probability.shape
    try:
        evidence = LineEvidence.from_probability(probability, native, threshold=.5, max_observations=100000)
    except ValueError as error:
        if str(error) != 'insufficient_or_diffuse_line_evidence': raise
        return {'status': str(error)}, []
    points = evidence.points * [(w - 1) / (native[1] - 1), (h - 1) / (native[0] - 1)]
    tree = cKDTree(points)
    segments, visible = clip_segments(reference[EDGES] - [x, y], w, h)
    rows = []; all_distances = []
    for i, segment in enumerate(segments):
        if not visible[i] or np.linalg.norm(segment[1] - segment[0]) < 1: continue
        # Ignore the first/last 10% to reduce intersection ambiguity; no nearest-semantic-line claim.
        query = segment[0] + np.linspace(.1, .9, 81)[:, None] * (segment[1] - segment[0])
        distance, indices = tree.query(query)
        direction = segment[1] - segment[0]; normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        signed = (points[indices] - query) @ normal
        rows.append({'line': LINE_NAMES[i], 'distance': distribution(distance), 'signed_median_px': float(np.median(signed))})
        all_distances.extend(distance)
    reverse = segment_distances(points, segments); reverse[:, ~visible] = np.inf
    return {'status': 'measured', 'native_size_hw': list(native), 'ridge_points': len(points),
        'gt_to_nearest_unlabelled_ridge_px': distribution(all_distances),
        'ridge_to_nearest_gt_segment_px': distribution(reverse.min(axis=1))}, rows


def main():
    cv2.setNumThreads(1); started = time.monotonic()
    out = BASE / 'raw-analysis'; out.mkdir(exist_ok=True)
    kp_rows = []; line_rows = []; case_rows = []
    masks = json.loads((WHITE / 'net-exclusions.json').read_text())
    for _, record in source_rows():
        cam = record['camera']; gt, visible = truth(cam)
        regions = record['legacy_candidates']; selected = {name: record['conditions'][name]['region_index'] for name in CONDITIONS}
        cached = {}
        for index, candidate in enumerate(regions):
            raw_path = OLD / 'raw' / record['key'] / f'raw_{index:02d}.npz'; raw = np.load(raw_path)
            offset = np.array(candidate['region'][:2]); observed = raw['raw_xy'].astype(float); global_xy = observed + offset
            h, w = raw['image_size_hw']; nh, nw = raw['native_size_hw']
            support = keypoint_support(observed, raw['scores'], image_size_hw=(h, w), native_size_hw=(nh, nw),
                min_score=.05, boundary_margin_cells=1., duplicate_distance_cells=1.)
            truth_local = gt - offset; grid = truth_local * [(nw - 1) / (w - 1), (nh - 1) / (h - 1)]
            inside = (grid[:, 0] >= 0) & (grid[:, 0] <= nw - 1) & (grid[:, 1] >= 0) & (grid[:, 1] <= nh - 1)
            edge_distance = np.minimum.reduce([grid[:, 0], grid[:, 1], nw - 1 - grid[:, 0], nh - 1 - grid[:, 1]])
            ends = np.array(masks['cameras'][cam]['endpoints_xy'], float); vector = ends[1] - ends[0]
            t = np.clip((gt - ends[0]) @ vector / (vector @ vector), 0, 1)
            net_overlap = np.linalg.norm(gt - (ends[0] + t[:, None] * vector), axis=1) <= masks['half_width_px']
            raw_error = np.linalg.norm(global_xy - gt, axis=1)
            scopes = ['all_regions'] + (['full_frame'] if index == 0 else []) + [name for name, chosen in selected.items() if chosen == index]
            for i in range(14):
                kp_rows.append({'key': record['key'], 'camera': cam, 'region': index, 'kp': i,
                    'scopes': scopes, 'raw_valid': bool(raw['valid'][i]), 'reference_visible': bool(visible[i]),
                    'reference_inside_crop': bool(inside[i]), 'reference_near_crop_border': bool(inside[i] and edge_distance[i] < 1),
                    'decoded_boundary': bool(support.boundary[i]), 'decoded_duplicate': bool(support.duplicate[i]),
                    'eligible': bool(support.eligible[i]), 'far_half': i in FAR, 'net_band_overlap': bool(net_overlap[i]),
                    'score': float(raw['scores'][i]), 'error_px': float(raw_error[i]), 'delta_x': float(global_xy[i, 0] - gt[i, 0]), 'delta_y': float(global_xy[i, 1] - gt[i, 1])})
            if index in {0, selected['postprocess_and_acceptance']}:
                lm, lines = line_measure(raw['line_probability'], candidate['region'], gt)
                for line in lines: line_rows.append({'key': record['key'], 'camera': cam, 'region': index, 'scope': 'full_frame' if index == 0 else 'selected_C', **line})
                observed_valid = observed.copy(); observed_valid[~raw['valid']] = np.nan
                start_fit = time.monotonic()
                baseline = estimate_confidence_homography(WORLD, observed_valid, raw['scores'], reprojection_threshold_px=.005 * np.hypot(w, h), min_score=.05)
                global_projection = baseline.projected + offset if baseline.matrix is not None else None
                kp_errors = np.linalg.norm(global_projection[visible] - gt[visible], axis=1) if global_projection is not None else np.array([])
                frame_raw = raw_error[visible & raw['valid']]
                case_rows.append({'key': record['key'], 'camera': cam, 'region': index, 'is_full_frame': index == 0,
                    'is_selected_C': index == selected['postprocess_and_acceptance'], 'raw_visible_points': int((visible & raw['valid']).sum()),
                    'raw_visible_expected': int(visible.sum()), 'raw_visible_error': distribution(frame_raw), 'line': lm,
                    'existing_KP_only_status': baseline.status, 'existing_KP_only_correct': correct(global_projection, cam),
                    'existing_KP_only_error': distribution(kp_errors), 'existing_KP_only_seconds': time.monotonic() - start_fit})
            cached[index] = {'raw_correct': bool(raw['valid'][visible].all() and raw_error[visible].mean() <= 10 and raw_error[visible].max() <= 25)}
        print(record['key'], flush=True)
    groups = defaultdict(list)
    for row in kp_rows:
        for scope in row['scopes']:
            for camera in ('all', row['camera']):
                prefix = (scope, camera)
                for name, flag in [('visible_in_crop', row['reference_visible'] and row['reference_inside_crop']),
                    ('outside_crop', not row['reference_inside_crop']), ('decoded_boundary', row['decoded_boundary']),
                    ('eligible_visible', row['eligible'] and row['reference_visible'] and row['reference_inside_crop']),
                    ('far_visible', row['far_half'] and row['reference_visible'] and row['reference_inside_crop']),
                    ('near_visible', not row['far_half'] and row['reference_visible'] and row['reference_inside_crop']),
                    ('net_band_overlap', row['net_band_overlap'] and row['reference_visible'] and row['reference_inside_crop'])]:
                    if flag: groups[prefix + (name,)].append(row)
    summaries = []
    for key, rows in groups.items():
        valid = [r for r in rows if r['raw_valid']]
        summaries.append({'scope': key[0], 'camera': key[1], 'stratum': key[2], 'points': len(rows), 'valid_points': len(valid),
            'error_px': distribution([r['error_px'] for r in valid]), 'score': distribution([r['score'] for r in rows]),
            'within10px': sum(r['error_px'] <= 10 for r in valid), 'within25px': sum(r['error_px'] <= 25 for r in valid)})
    per_point = []
    for camera in ('cam0', 'cam1', 'cam2', 'cam3'):
        for i in range(14):
            rows = [r for r in kp_rows if r['camera'] == camera and r['region'] == 0 and r['kp'] == i]
            per_point.append({'camera': camera, 'kp': i, 'reference_visible': rows[0]['reference_visible'],
                'error_px': distribution([r['error_px'] for r in rows if r['raw_valid']]),
                'median_delta_xy': [float(np.median([r[k] for r in rows if r['raw_valid']])) for k in ('delta_x', 'delta_y')]})
    write_json(out / 'summary.json', {'reference_approved': True, 'units': 'original 1920x1080 source pixels',
        'scopes_are_overlapping': True, 'net_band_is_not_person_occlusion': True,
        'line_precision_limit': 'nearest unlabelled native-grid ridge, not semantic LINE correctness; <=0.5 native cell quantization and intersection ambiguity remain',
        'strata': summaries, 'full_frame_per_keypoint': per_point, 'seconds': time.monotonic() - started})
    write_json(out / 'cases.json', case_rows); write_json(out / 'line-cases.json', line_rows)
    with (out / 'keypoint-observations.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(kp_rows[0]), lineterminator='\n'); writer.writeheader()
        writer.writerows({**r, 'scopes': '|'.join(r['scopes'])} for r in kp_rows)
    print('complete', len(kp_rows), len(case_rows), time.monotonic() - started, flush=True)


if __name__ == '__main__': main()
