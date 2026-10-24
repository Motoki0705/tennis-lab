"""Summarize float32 NHT differences, decoded JPEGs and exact KP inventories."""
from __future__ import annotations
import argparse
import io
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path.cwd()))
from src.synthetic_data_generation.alignment.validation import load_alignment_result


def stats(a, b):
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64)).reshape(-1)
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    return {'mae': float(delta.mean()), 'rmse': rmse,
            'abs_quantiles': {str(q): float(np.quantile(delta, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)},
            'nonzero_fraction': float(np.mean(delta != 0)),
            'psnr_db': float(-20 * np.log10(rmse)) if rmse else None}


def jpeg(rgb):
    stream = io.BytesIO()
    Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(stream, format='JPEG', quality=95)
    stream.seek(0)
    return np.asarray(Image.open(stream).convert('RGB')).astype(np.float64) / 255


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--reference-visibility', type=Path, required=True)
    p.add_argument('--candidate-visibility', type=Path, required=True)
    p.add_argument('--reference-repeat', default='repeat-0')
    p.add_argument('--candidate-repeat', default='repeat-0')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rv = json.loads(args.reference_visibility.read_text())['views']
    cv = json.loads(args.candidate_visibility.read_text())['views']
    report = {'schema': 'court_render_comparison_v1', 'reference_repeat': args.reference_repeat, 'candidate_repeat': args.candidate_repeat, 'scenes': {}}
    for sid in ('B00', 'B01', 'B02', 'B03'):
        root_a, root_b = args.reference / args.reference_repeat / sid, args.candidate / args.candidate_repeat / sid
        ma, mb = [json.loads((root / 'render.json').read_text()) for root in (root_a, root_b)]
        assert ma == mb, 'Public render manifests differ'
        alignment = load_alignment_result(args.scenes / sid / 'alignment/alignment.json')
        scale = alignment.metric_adapter.nht_scene_units_per_metre
        collected = {name: ([], []) for name in ('rgb', 'alpha', 'depth_m', 'jpeg_rgb')}
        views = []
        for ra, rb in zip(ma['renders'], mb['renders'], strict=True):
            camera_id = ra['camera_id']
            va = next(v for v in rv if (v['scene_id'], v['repeat'], v['camera_id']) == (sid, args.reference_repeat, camera_id))
            vb = next(v for v in cv if (v['scene_id'], v['repeat'], v['camera_id']) == (sid, args.candidate_repeat, camera_id))
            assert len(va['points']) == len(vb['points'])
            visibility = {'points': len(va['points']), 'geometry_disagreements': 0, 'renderer_disagreements': 0, 'max_uv_delta_px': 0., 'max_kp_depth_delta_m': 0., 'in_frame_points': 0, 'renderer_visible_points': 0}
            for pa, pb in zip(va['points'], vb['points'], strict=True):
                visibility['geometry_disagreements'] += any(pa[k] != pb[k] for k in ('court_id', 'class_id', 'physical_index', 'in_front', 'in_frame'))
                visibility['renderer_disagreements'] += pa['renderer_visible'] != pb['renderer_visible']
                visibility['max_uv_delta_px'] = max(visibility['max_uv_delta_px'], max(abs(a-b) for a,b in zip(pa['uv'], pb['uv'], strict=True)))
                visibility['max_kp_depth_delta_m'] = max(visibility['max_kp_depth_delta_m'], abs(pa['depth_m']-pb['depth_m']))
                visibility['in_frame_points'] += pa['in_frame']
                visibility['renderer_visible_points'] += pa['renderer_visible']
            metrics = {}
            arrays_a, arrays_b = {}, {}
            for field in ('rgb', 'alpha', 'depth'):
                a, b = np.load(root_a / ra[field]), np.load(root_b / rb[field])
                assert a.shape == b.shape and a.dtype == b.dtype == np.float32
                assert np.isfinite(a).all() and np.isfinite(b).all()
                arrays_a[field], arrays_b[field] = a, b
                if field == 'depth':
                    name, a, b = 'depth_m', a.astype(np.float64) / scale, b.astype(np.float64) / scale
                else:
                    name = field
                metrics[name] = stats(a, b)
                if name == 'depth_m':
                    metrics[name].pop('psnr_db')
                    positive = (a > 0) & (b > 0)
                    metrics['depth_relative'] = stats(np.zeros_like(a[positive]), np.abs(a[positive] - b[positive]) / a[positive])
                    metrics['depth_relative'].pop('psnr_db')
                collected[name][0].append(a.reshape(-1))
                collected[name][1].append(b.reshape(-1))
            ja, jb = jpeg(arrays_a['rgb']), jpeg(arrays_b['rgb'])
            metrics['jpeg_rgb'] = stats(ja, jb)
            metrics['reference_jpeg_encoding'] = stats(arrays_a['rgb'], ja)
            collected['jpeg_rgb'][0].append(ja.reshape(-1))
            collected['jpeg_rgb'][1].append(jb.reshape(-1))
            metrics['alpha_threshold_disagreements'] = int(np.count_nonzero((arrays_a['alpha'] >= .01) != (arrays_b['alpha'] >= .01)))
            metrics['depth_positive_disagreements'] = int(np.count_nonzero((arrays_a['depth'] > 0) != (arrays_b['depth'] > 0)))
            views.append({'camera_id': camera_id, 'metrics': metrics, 'visibility': visibility})
        summary = {name: stats(np.concatenate(a), np.concatenate(b)) for name, (a,b) in collected.items()}
        summary['depth_m'].pop('psnr_db')
        summary['visibility'] = {key: sum(v['visibility'][key] for v in views) for key in ('points', 'geometry_disagreements', 'renderer_disagreements', 'in_frame_points', 'renderer_visible_points')}
        summary['max_uv_delta_px'] = max(v['visibility']['max_uv_delta_px'] for v in views)
        summary['max_kp_depth_delta_m'] = max(v['visibility']['max_kp_depth_delta_m'] for v in views)
        report['scenes'][sid] = {'summary': summary, 'views': views, 'nht_units_per_metre': scale}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({sid: value['summary'] for sid, value in report['scenes'].items()}, indent=2))


if __name__ == '__main__':
    main()
