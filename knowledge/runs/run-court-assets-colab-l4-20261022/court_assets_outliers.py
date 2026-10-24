"""Audit cross-GPU errors without suppressing invalid/background depth pixels."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from court_assets_compare import stats


def clip_polygon(vertices, plane):
    result = []
    for prev, curr in zip(vertices[-1:] + vertices[:-1], vertices):
        pa, pb = plane(prev), plane(curr)
        if (pa >= 0) != (pb >= 0):
            result.append(prev + (curr - prev) * pa / (pa - pb))
        if pb >= 0:
            result.append(curr)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--visibility', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    comparison = json.loads(a.comparison.read_text())
    visibility = json.loads(a.visibility.read_text())['views']
    report = {'schema': 'court_render_outlier_audit_v1', 'court_mask': 'physical corners 0,1,3,2, clipped at camera depth 0.01m and viewport in homogeneous coordinates, then PIL rasterized; diagnostic only', 'scenes': {}}
    for sid, info in comparison['scenes'].items():
        scale = info['nht_units_per_metre']
        groups = {}
        views = []
        for view in info['views']:
            cid = view['camera_id']
            arrays = [{f: np.load(root / 'repeat-0' / sid / cid / f'{f}.npy').astype(np.float64) for f in ('rgb', 'alpha', 'depth')} for root in (a.reference, a.candidate)]
            ref, cand = arrays
            for arrays_item in arrays:
                arrays_item['depth'] /= scale
            h, w, _ = ref['alpha'].shape
            geometry = next(v['points'] for v in visibility if (v['scene_id'], v['repeat'], v['camera_id']) == (sid, 'repeat-0', cid))
            image = Image.new('1', (w, h))
            draw = ImageDraw.Draw(image)
            for court in sorted({pt['court_id'] for pt in geometry}):
                points = [pt for pt in geometry if pt['court_id'] == court]
                by_index = {pt['physical_index']: pt for pt in points}
                vertices = [np.array([*by_index[i]['uv'], 1.]) * by_index[i]['depth_m'] for i in (0, 1, 3, 2)]
                for plane in (lambda v: v[2] - .01, lambda v: v[0], lambda v: (w - 1) * v[2] - v[0], lambda v: v[1], lambda v: (h - 1) * v[2] - v[1]):
                    if vertices:
                        vertices = clip_polygon(vertices, plane)
                if len(vertices) >= 3:
                    draw.polygon([tuple(v[:2] / v[2]) for v in vertices], fill=1)
            masks = {'all': np.ones((h, w), dtype=bool), 'projected_courts': np.asarray(image, dtype=bool)}
            for threshold in (.01, .5, .99):
                masks[f'both_alpha_ge_{threshold}'] = (ref['alpha'][..., 0] >= threshold) & (cand['alpha'][..., 0] >= threshold)
            item = {'camera_id': cid, 'regions': {}}
            for name, mask in masks.items():
                measured = {'pixels': int(mask.sum())}
                for field in ('rgb', 'alpha', 'depth'):
                    values = stats(ref[field][mask], cand[field][mask])
                    if field == 'depth':
                        values.pop('psnr_db')
                    measured[field] = values
                    groups.setdefault(name, {}).setdefault(field, [[], []])[0].append(ref[field][mask].reshape(-1))
                    groups[name][field][1].append(cand[field][mask].reshape(-1))
                measured['rgb_error_gt_one_8bit_step_pixels'] = int(np.count_nonzero(np.max(np.abs(ref['rgb'] - cand['rgb']), axis=-1)[mask] > 1 / 255))
                measured['depth_error_gt_1cm_pixels'] = int(np.count_nonzero(np.abs(ref['depth'] - cand['depth'])[..., 0][mask] > .01))
                item['regions'][name] = measured
            for field in ('rgb', 'alpha', 'depth'):
                error = np.max(np.abs(ref[field] - cand[field]), axis=-1)
                y, x = np.unravel_index(np.argmax(error), error.shape)
                item[f'max_{field}_error_pixel'] = {'x': int(x), 'y': int(y), 'error': float(error[y, x]), 'in_projected_court': bool(masks['projected_courts'][y, x]), 'reference': {f: ref[f][y, x].tolist() for f in ref}, 'candidate': {f: cand[f][y, x].tolist() for f in cand}}
            switches = np.argwhere((ref['alpha'][..., 0] >= .01) != (cand['alpha'][..., 0] >= .01))
            item['alpha_threshold_switches'] = [{'x': int(x), 'y': int(y), 'in_projected_court': bool(masks['projected_courts'][y, x]), 'reference_alpha': float(ref['alpha'][y, x, 0]), 'candidate_alpha': float(cand['alpha'][y, x, 0])} for y, x in switches]
            views.append(item)
        summary = {}
        for region, fields in groups.items():
            summary[region] = {field: stats(np.concatenate(pair[0]), np.concatenate(pair[1])) for field, pair in fields.items()}
            summary[region]['depth'].pop('psnr_db')
            for key in ('pixels', 'rgb_error_gt_one_8bit_step_pixels', 'depth_error_gt_1cm_pixels'):
                summary[region][key] = sum(v['regions'][region][key] for v in views)
        report['scenes'][sid] = {'summary': summary, 'views': views}
        print(sid, {region: {'pixels': d['pixels'], 'rgb_mae': d['rgb']['mae'], 'rgb_max': d['rgb']['abs_quantiles']['1.0'], 'depth_mae_m': d['depth']['mae'], 'depth_max_m': d['depth']['abs_quantiles']['1.0']} for region, d in summary.items()}, flush=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
