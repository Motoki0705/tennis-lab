"""Write r19-compatible tables with one column per declared variant/reference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render(plan: dict[str, Any], destination: Path) -> None:
    results = {v['name']: json.loads((Path(v['evaluation_output']) / 'metrics.json').read_text()) for v in plan['variants']}
    first = next(iter(results.values()))
    combined = {'r18_refiner': {k.removeprefix('new_refiner/'): v for k, v in first.items() if k.startswith('new_refiner/')},
                'e9_detector': {k.removeprefix('new_detector/'): v for k, v in first.items() if k.startswith('new_detector/')}}
    for name, metrics in results.items():
        for ref in ('new_refiner', 'new_detector'):
            if {k: v for k, v in metrics.items() if k.startswith(ref + '/')} != {k: v for k, v in first.items() if k.startswith(ref + '/')}:
                raise ValueError('References differ between variants')
        combined[name] = {k.removeprefix('variant/'): v for k, v in metrics.items() if k.startswith('variant/')}
    keys = set(next(iter(combined.values())))
    if any(set(metrics) != keys for metrics in combined.values()):
        raise ValueError('Variant/reference strata differ')
    for key in keys:
        if len({metrics[key]['frames'] for metrics in combined.values()}) != 1:
            raise ValueError('Variant/reference denominators differ')
    (destination / 'comparison.json').write_text(json.dumps(combined, indent=2, allow_nan=False) + '\n')
    names = list(combined)
    groups = ['meiji', 'meiji/cam0', 'meiji/cam1', 'meiji/cam2', 'meiji/selection', 'meiji/calibration',
              *[f'meiji/{half}/cam{i}' for half in ('selection', 'calibration') for i in range(3)], 'tracknet', 'chat_annotation']
    lines = ['# 同一validationの精度比較', '',
             '点要約は最大weight成分の平均。r18/e9は保存済み同一frameの参照値で、検出器は再推論しない。',
             '位置NLLはsource px²密度。HDRは全GMMのR²領域、面積は平均source px²。MC設定はr19と同じ。',
             'e9のgap密度は一様で、50/90/95%ともcoverage1・全画面の自明な領域。較正成功と扱わない。',
             '推定位置は参考値、unknownはN/A、不在は位置N/Aで存在NLLだけ。すべての母数はJSONを参照。', '']

    def number(value: Any) -> str:
        return 'N/A' if value is None else f'{value:.4g}'

    def table_header(title: str) -> None:
        lines.extend([title, '', '| group/condition | n | ' + ' | '.join(names) + ' |',
                      '|---|---:|' + '---|' * len(names)])

    def table_row(key: str, columns: tuple[str, ...], denominator: str) -> None:
        rows = [combined[name][key] for name in names]
        counts = sorted({r[denominator] for r in rows})
        lines.append('| ' + key + ' | ' + '/'.join(str(n) for n in counts) + ' | ' +
                     ' | '.join(' / '.join(number(row[column]) for column in columns) for row in rows) + ' |')

    table_header('## 観測位置誤差 p50 / p90 / p95 (px)')
    for group in groups:
        table_row(f'{group}/observed/observed', ('median_error_px', 'p90_error_px', 'p95_error_px'), 'error_px_frames')
    table_header('## 位置NLL observed / artificial gaps')
    for group in groups:
        for condition in ('observed', 'evidence_gap'):
            table_row(f'{group}/{condition}/observed', ('mean_nll_px',), 'nll_px_frames')
    for level in (.5, .9, .95):
        table_header(f'## HDR {level:.0%}: coverage / mean area px²')
        for group in groups:
            for condition in ('observed', 'evidence_gap'):
                table_row(f'{group}/{condition}/observed', (f'coverage_{level:g}', f'area_px2_{level:g}'), f'coverage_{level:g}_frames')
    table_header('## 存在NLL（detectorはamodal存在を出さないためN/A）')
    for group in groups:
        for condition in ('observed', 'evidence_gap'):
            for label in ('observed', 'absent'):
                table_row(f'{group}/{condition}/{label}', ('mean_presence_nll',), 'presence_nll_frames')
    (destination / 'comparison.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    bundle = Path(__file__).resolve().parent
    run_plan = json.loads((bundle / 'plan.json').read_text())
    render(run_plan, Path(run_plan['report']))
