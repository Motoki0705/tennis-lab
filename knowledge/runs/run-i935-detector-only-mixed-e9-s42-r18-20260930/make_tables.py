"""Render audited frame-weighted results; never report absent/unknown as scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BUNDLE = Path(__file__).resolve().parent
METHODS = ('new_refiner', 'old_refiner', 'new_detector', 'old_detector')
NAMES = dict(zip(METHODS, ('新pilot', '旧r4 pilot', '新detector e9', '旧detector e13'), strict=True))
GROUPS = ('meiji', 'meiji/cam0', 'meiji/cam1', 'meiji/cam2', 'meiji/selection', 'meiji/calibration',
          'meiji/selection/cam0', 'meiji/selection/cam1', 'meiji/selection/cam2',
          'meiji/calibration/cam0', 'meiji/calibration/cam1', 'meiji/calibration/cam2', 'tracknet', 'chat_annotation')


def fmt(value: Any, digits: int = 3) -> str:
    return 'N/A' if value is None else f'{value:.{digits}f}'


def main() -> None:
    metrics = json.loads((BUNDLE / 'paired_recomputed.json').read_text())
    audit = json.loads((BUNDLE / 'collection.json').read_text())
    assert audit['status'] == 'verified'
    lines = ['# 同一validationフレーム比較（run 19回収）', '',
             '全70 clip / 40,144 frameを4手法×2条件で保存。数値はframe加重、単位はsource px。',
             'Meijiはvideo_000のみ。selection/calibrationはcheckpoint選択の既存分割で、testではない。',
             '新旧pilotの点評価は最大weight成分の平均、detectorは無閾値argmax。動画の混合平均とは異なる。',
             'detectorとの公平な性能比較には位置誤差を用いる。', '',
             '## Observedの位置誤差', '', '| source / camera / half | 手法 | n | p50 px | p90 px | p95 px |',
             '|---|---|---:|---:|---:|---:|']
    for group in GROUPS:
        for method in METHODS:
            row = metrics[f'{method}/{group}/observed/observed']
            lines.append(f"| {group} | {NAMES[method]} | {row['error_px_frames']} | {fmt(row['median_error_px'], 2)} | {fmt(row['p90_error_px'], 2)} | {fmt(row['p95_error_px'], 2)} |")
    lines += ['', '## 新旧の差（新−旧、負なら誤差が改善）', '',
              '| source / camera / half | 系列 | n | Δp50 px | Δp90 px | Δp95 px |', '|---|---|---:|---:|---:|---:|']
    for group in GROUPS:
        for series in ('refiner', 'detector'):
            new, old = [metrics[f'{prefix}_{series}/{group}/observed/observed'] for prefix in ('new', 'old')]
            delta = [fmt(new[k] - old[k], 2) for k in ('median_error_px', 'p90_error_px', 'p95_error_px')]
            lines.append(f"| {group} | {series} | {new['error_px_frames']} | " + ' | '.join(delta) + ' |')
    lines += ['', '## 存在NLLと位置NLL・HDR（未較正）', '',
              '位置NLLはsource px²に対する密度の負の自然対数。HDR列は **coverage / 平均面積px²**。',
              '存在NLLには独立した分母n_existを併記。detectorにamodal存在確率はなく常にN/A。',
              'occlusion_estimatedとinterpolatedは位置の参考値だけで、独立したamodal GTでも存在教師でもない。',
              '人工gapは固定された証拠dropout内だけを採点（RGB遮蔽ではない）。',
              'detectorのgap密度は一様なので、MeijiではHDR50/90/95がすべて全画面2,070,601 px²、coverage=1になる。',
              'この自明なcoverageは成功や較正改善を意味しない。通常heatmapでも高coverageと巨大面積が共存しうる。',
              'GMMは画面外tailを持つR²のHDR、detectorは有限画面内。NLL/HDRの絶対値だけで優劣を決めない。',
              'HDRは保存済み2048 sample/seed1729のMC診断。表の差に有意性を主張しない。', '']
    for condition, label, title in [
        ('observed', 'observed', 'Observed（無人工gap）'),
        ('evidence_gap', 'observed', 'Observed位置既知の人工gap'),
        ('observed', 'occlusion_estimated_reference', '実遮蔽の推定位置（無人工gap、参考）'),
        ('observed', 'interpolated_reference', '補間位置（無人工gap、参考）'),
        ('evidence_gap', 'occlusion_estimated_reference', '実遮蔽の推定位置＋人工gap（参考）'),
        ('evidence_gap', 'interpolated_reference', '補間位置＋人工gap（参考）'),
    ]:
        lines += [f'### {title}', '', '| source / camera / half | 手法 | n_position | n_exist | 存在NLL | 位置NLL | HDR50 cov / px² | HDR90 cov / px² | HDR95 cov / px² |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for group in GROUPS:
            for method in METHODS:
                row = metrics[f'{method}/{group}/{condition}/{label}']
                hdr = [f"{fmt(row[f'coverage_{q}'])} / {fmt(row[f'area_px2_{q}'], 1)}" for q in ('0.5', '0.9', '0.95')]
                lines.append(f"| {group} | {NAMES[method]} | {row['nll_px_frames']} | {row['presence_nll_frames']} | {fmt(row['mean_presence_nll'], 6)} | {fmt(row['mean_nll_px'])} | " + ' | '.join(hdr) + ' |')
        lines.append('')
    lines += ['## 不在・unknown（すべてN/A）', '',
              '指示に従い不存在も全指標N/Aとして扱う。元のr18成果物にはrefinerの不在BCEがあるが、今回の表では採用しない。',
              '未レビュー、instanceなし、複数球、unresolvedはunknown。不在への読み替えを行わない。', '',
              '| source / camera / half | 条件 | absent n | unknown n | 全指標 |', '|---|---|---:|---:|---|']
    for group in GROUPS:
        for condition in ('observed', 'evidence_gap'):
            absent = metrics[f'new_refiner/{group}/{condition}/absent']['frames']
            unknown = sum(metrics[f'new_refiner/{group}/{condition}/{key}']['frames'] for key in ('unresolved', 'no_instance_unknown', 'unreviewed_unknown', 'multiple_instances_unknown'))
            lines.append(f'| {group} | {condition} | {absent} | {unknown} | N/A |')
    (BUNDLE / 'comparison.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
