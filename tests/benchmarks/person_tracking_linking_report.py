"""Tables and correspondence for the fixed run-9 comparison; no inference/tuning."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from person_tracking_linking import (  # type: ignore[import-not-found]
    CLIP,
    KPR,
    SOLIDER,
    checked,
)
from person_tracking_matrix_video import (  # type: ignore[import-not-found]
    recommendation,
)

from src.tasks.player_detection.evaluation.person_sources import write_csv
from src.tennis_scene.pipeline.artifacts import write_json_atomic


def short(name: str) -> str:
    return name.replace(CLIP, 'CLIP').replace(SOLIDER, 'SOLIDER').replace(KPR, 'KPR')


def report(root: Path) -> None:
    comparison = json.loads((root / 'comparison.json').read_text())
    selected_method = json.loads((root / 'kpr_method.json').read_text())['chosen']
    chosen = recommendation(comparison['table'])
    overall = [row for row in comparison['table'] if row['camera'] == row['near_far'] == 'all']
    pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    availability, correspondence = [], []
    for record in comparison['records']:
        result = json.loads(checked(record['result']).read_text())
        variant, clip = record['variant'], record['clip']
        for cam, value in result['cameras'].items():
            checked(value['arrays'])
            availability.append({'variant': variant, 'clip': clip, 'camera': cam,
                                 'status': value['tracking']['status'], 'reason': value['tracking'].get('reason', ''),
                                 'raw_ids': value['tracking']['track_count'],
                                 'selected_groups': sum(g['selected'] for g in value['selection']['groups']),
                                 'unlabelled_raw': value['unlabelled_selected_boxes'],
                                 'unlabelled_group': value['group_unlabelled_selected_boxes'],
                                 'gsi_interpolated': value['tracking'].get('interpolated_boxes', 0)})
        for encoder, value in result['association'].items():
            pairs[variant, encoder].append(value)
            if value['status'] != 'ok':
                correspondence.append({'variant': variant, 'clip': clip, 'encoder': encoder, 'status': value['status'],
                                       'reason': value['reason'], 'camera': '', 'raw_id': '', 'reference_to_prediction': '', 'boxes': ''})
                continue
            checked(value['arrays'])
            for camera, tracks in value['metrics']['track_confusion'].items():
                for track, counts in tracks.items():
                    for label, count in counts.items():
                        correspondence.append({'variant': variant, 'clip': clip, 'encoder': encoder, 'status': 'ok',
                                               'reason': '', 'camera': camera, 'raw_id': track,
                                               'reference_to_prediction': label, 'boxes': count})
    association = []
    for (variant, encoder), values in pairs.items():
        valid = [v['metrics'] for v in values if v['status'] == 'ok']
        totals = {k: sum(v['pairs'][k] for v in valid) for k in ('tp', 'fp', 'fn')}
        denominator = 2 * totals['tp'] + totals['fp'] + totals['fn']
        correct = sum(v['group_accuracy']['frames_correct'] for v in valid)
        frames = sum(v['group_accuracy']['frames_scored'] for v in valid)
        labels = sum(c['label_boxes'] for v in valid for c in v['coverage'].values())
        matched = sum(c['matched_label_boxes'] for v in valid for c in v['coverage'].values())
        association.append({'variant': variant, 'encoder': encoder, 'decided_clips': len(valid),
                            'total_clips': len(values), 'pair_f1': 2 * totals['tp'] / denominator if denominator else None,
                            **totals, 'group_accuracy': correct / frames if frames else None,
                            'matched_label_boxes': matched, 'label_boxes': labels,
                            'undecided': '; '.join(v['reason'] for v in values if v['status'] != 'ok')})
    write_csv(root / 'availability.csv', availability)
    write_csv(root / 'association.csv', association)
    write_csv(root / 'correspondence.csv', correspondence)
    chosen_row = next(r for r in overall if r['variant'] == chosen)
    write_json_atomic(root / 'recommendation.json', {'chosen': chosen, 'metrics': chosen_row,
        'association': [r for r in association if r['variant'] == chosen], 'pipeline_default_changed': False,
        'kpr_selected_method': json.loads((root / 'kpr_method.json').read_text())['chosen']})
    lines = ['# Run 9 固定比較', '', f'主指標による候補内推薦: **{short(chosen)}**。pipeline既定は変更しない。', '',
             f'表のbest_kprは、事前規則で選んだ **{short(selected_method)} のtracker encoderをKPRへ置換**した条件。', '',
             'raw IDF1はrun 8の主指標。group IDF1はrun 8結果後に追加した副指標で、下流のcamera-local連結groupを測る。',
             '全条件とも同じ固定コート選別。GSI補間は実観測へ昇格せず、主表は実観測のみ。',
             '旧6条件のraw指標は12cameraごとに保存済みrun 8と一致を検証した。未見性能/完全GT MOTとは呼ばない。', '',
             '|条件|完走|raw IDF1|group IDF1|raw switch/frag|group switch/frag|raw/group選手保持|非選手raw/group|人物50%|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in overall:
        complete = sum(a['status'] == 'ok' for a in availability if a['variant'] == r['variant'])
        lines.append(f'|{short(r["variant"])}|{complete}/12|{r["idf1"]:.6f}|{r["group_idf1"]:.6f}|'
                     f'{r["id_switches"]}/{r["fragments"]}|{r["group_id_switches"]}/{r["group_fragments"]}|'
                     f'{r["player_units_kept"]}/{r["group_player_units_kept"]} of {r["player_units"]}|'
                     f'{r["nonplayer_units_kept"]}/{r["group_nonplayer_units_kept"]}|{r["player_identities_kept50"]}/8|')
    lines += ['', '## camera × near/far（raw / group IDF1）', '',
              '|条件|cam0 near|cam0 far|cam1 near|cam1 far|cam2 near|cam2 far|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in overall:
        cells = []
        for camera in ('cam0', 'cam1', 'cam2'):
            for side in ('near', 'far'):
                value = next(x for x in comparison['table'] if x['variant'] == r['variant'] and x['camera'] == camera and x['near_far'] == side)
                cells.append(f'{value["idf1"]:.4f} / {value["group_idf1"]:.4f} (N={value["player_units"]})')
        lines.append('|' + '|'.join([short(r['variant']), *cells]) + '|')
    lines += ['', 'unknownを含む全144層とIDTP/FP/FN・保持数は`comparison.csv`。停止/未照合数は`availability.csv`。', '',
              '## camera間対応', '', 'pair F1はdecided clipのcountsをpool。4/4未満は条件付きの値で、coverageが異なる。',
              'KPR/SOLIDERはCLIPの固定calibrationを転用し、encoder固有の再校正はしていない。', '',
              '|Tracker|camera間encoder|decided|pair F1|TP/FP/FN|group accuracy|label box coverage|',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in association:
        f1 = f'{row["pair_f1"]:.6f}' if row['pair_f1'] is not None else '—'
        accuracy = f'{row["group_accuracy"]:.6f}' if row['group_accuracy'] is not None else '—'
        lines.append(f'|{short(row["variant"])}|{short(row["encoder"])}|{row["decided_clips"]}/{row["total_clips"]}|{f1}|'
                     f'{row["tp"]}/{row["fp"]}/{row["fn"]}|{accuracy}|{row["matched_label_boxes"]}/{row["label_boxes"]}|')
    lines += ['', '人物→予測IDの全対応は`correspondence.csv`、未決定理由は`association.csv`。', '',
              '旧box・旧track由来のラベルによるbaseline有利の偏り、選択後の部分参照指標、既知clipへの設計上の依存は残る。',
              'StrongSORT++のGSI再構成とmaskは各tracking結果の`gsi`参照で監査できる。', '']
    (root / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    report(parser.parse_args().report)
