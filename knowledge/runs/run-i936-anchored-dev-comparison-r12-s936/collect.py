"""Rebuild the paired tables and verify archived score/baseline NPZ hashes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.tasks.ball_refiner.refiner_3d.condition_audit import compare_condition_reports
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.generator import write_json


def collect(root: Path) -> dict[str, Any]:
    reports = {name: json.loads((root / name / 'manifest.json').read_text()) for name in ('control', 'anchored')}
    verified = {}
    for name, report in reports.items():
        for filename, expected in report['artifacts'].items():
            path = root / name / filename
            if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
                raise ValueError('Archived NPZ changed: ' + str(path))
        verified[name] = len(report['artifacts'])
    paired = compare_condition_reports(reports['control'], reports['anchored'])
    calibration = reports['anchored']['audit']['calibration']
    for filename, key in (('bank.npz', 'bank_sha256'), ('calibration.json', 'report_sha256')):
        if sha256(root / 'calibration' / filename) != calibration[key]:
            raise ValueError('Archived calibration identity mismatch')
    write_json(root / 'comparison.json', paired)
    lines = ['# 旧#959 bank → anchored bank：同一軌道の条件比較', '',
        '品質は全80train+val。全96件はmetadata/hash監査のみ、test配列読込0。',
        '全125成分、HDR閾値512＋独立体積512標本。可視camera数はpresenceと別。', '']
    for split in ('train_val', 'train', 'val'):
        lines.extend([f'## {split}', '', '| bank | 可視camera | frames | GT NLL nat | RMSE m | HDR50 % | HDR90 % | HDR95 % |',
            '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for camera in ('all', '0', '1', '2', '3'):
            for name, report in reports.items():
                entry = report['condition_metrics'][split]
                row = entry['overall'] if camera == 'all' else entry['by_visible_cameras'][camera]
                levels = ' | '.join(f'{v*100:.3f}' for v in row['hdr_coverage'])
                lines.append(f"| {name} | {camera} | {row['frames']} | {row['nll_nat']:.6f} | {row['mixture_mean_rmse_m']:.6f} | {levels} |")
        lines.extend(['', '| bank | 可視camera | 平均V50 m³ | 平均V90 m³ | 平均V95 m³ | 中央値V95 m³ | frame MC SE平均V95 m³ |',
            '|---|---:|---:|---:|---:|---:|---:|'])
        for camera in ('all', '0', '1', '2', '3'):
            for name, report in reports.items():
                entry = report['condition_metrics'][split]
                row = entry['overall'] if camera == 'all' else entry['by_visible_cameras'][camera]
                volume = ' | '.join(f'{v:.6f}' for v in row['hdr_volume_mean_m3'])
                lines.append(f"| {name} | {camera} | {volume} | {row['hdr_volume_median_m3'][2]:.6f} | {row['hdr_volume_mean_frame_mc_se_m3'][2]:.6f} |")
        lines.append('')
    lines.extend(['## 同じ16 valの無学習baseline', '',
        '| bank | method | RMSE m | gap m | event±5 m | 加速度p95 m/s² | jerk p95 m/s³ | 再投影p95 px | behind |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|'])
    for name, report in reports.items():
        for method in ('mixture_mean', 'mixture_mean_rts'):
            m = report['baselines']['methods'][method]['metrics']
            lines.append(f"| {name} | {method} | {m['rmse_m_overall']['value']:.6f} | {m['rmse_m_gap']['value']:.6f} | {m['rmse_m_event_pm5']['value']:.6f} | {m['acceleration_all']['p95']:.3f} | {m['jerk_all']['p95']:.3f} | {m['reprojection_px_all']['p95']:.3f} | {m['behind_all']['invalid_count']}/{m['behind_all']['count']} |")
    lines.extend(['', '| bank | method | 可視camera | frames | RMSE m |', '|---|---|---:|---:|---:|'])
    for name, report in reports.items():
        for method in ('mixture_mean', 'mixture_mean_rts'):
            for camera, entry in report['baselines']['methods'][method]['by_visible_cameras'].items():
                m = entry['rmse_m_overall']
                lines.append(f"| {name} | {method} | {camera} | {m['count']} | {m['value']:.6f} |")
    lines.extend(['', '体積MC SEは推定閾値に条件付きで、平均体積のSEではない。再投影は正depth条件付き。',
        '独立Meiji/OOF性能、test品質、学習済みrefinerの新bank性能は測っていない。', ''])
    (root / 'comparison.md').write_text('\n'.join(lines))
    collection = {'status': 'complete', 'verified_npz_files': verified,
        'quality_frame_count': reports['control']['condition_metrics']['train_val']['overall']['frames'],
        'paired_metadata_rallies': paired['paired_metadata_rallies'], 'paired_quality_rallies': paired['paired_quality_rallies'],
        'resources': {name: report['resources'] for name, report in reports.items()},
        'artifact_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}
    write_json(root / 'collection.json', collection)
    return collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-root', type=Path, required=True)
    args = parser.parse_args()
    if not args.bundle_root.is_absolute():
        parser.error('Need an absolute bundle root')
    print(collect(args.bundle_root))


if __name__ == '__main__':
    main()
