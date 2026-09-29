"""Human-readable report and identity/camera tables from the CPU diagnostic."""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.tasks.person_tracking.selection_metrics import aggregate_units
from src.tasks.player_detection.evaluation.person_sources import write_csv
from src.utils.checksum import dual_sha256


def _pct(numerator: int, denominator: int) -> str:
    return f'{numerator}/{denominator} ({numerator / denominator:.2%})' if denominator else 'N/A (0 units)'


def write_review_report(report: Path) -> None:
    sources = json.loads((report / 'sources.json').read_text())
    selection = json.loads((report / 'selection.json').read_text())
    video = json.loads((report / 'video.json').read_text())
    if selection['sources_sha256'] != dual_sha256(report / 'sources.json') or video['selection_sha256'] != dual_sha256(report / 'selection.json'):
        raise ValueError('Report inputs are not the evaluated versions')
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    identities: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(report / 'selection_units.jsonl.gz', 'rt') as handle:
        for line in handle:
            unit = json.loads(line)
            grouped[(unit['source'], unit['stage'], unit['clip'], unit['camera'])].append(unit)
            identities[(unit['source'], unit['stage'], unit['clip'], unit['person'])].append(unit)
    camera_rows = [{'source': s, 'stage': stage, 'clip': clip, 'camera': camera, **aggregate_units(units)}
                   for (s, stage, clip, camera), units in grouped.items()]
    identity_rows = []
    for (source, stage, clip, person), units in identities.items():
        row = {'source': source, 'stage': stage, 'clip': clip, 'person': person, 'role': units[0]['role'],
               'kind': units[0]['kind'], 'units': len(units)}
        for iou in ('03', '05'):
            row[f'tracked_{iou}'] = sum(u[f'tracked_{iou}'] for u in units)
            row[f'kept_{iou}'] = sum(u[f'selected_{iou}'] for u in units)
        identity_rows.append(row)
    write_csv(report / 'selection_per_camera.csv', camera_rows)
    write_csv(report / 'selection_identities.csv', identity_rows)
    lines = ['# #964 run 4: 保存済みデータによる人物候補とコート選別のCPU比較', '',
        '既存の4開発clipのみ。未見clip・GPU再推論・pipeline既定変更は使っていない。',
        '元GPU jobはユーザーの方針変更でcancelled。ft_base 12/12、ft_1080 11/12の全23 NPZのSHA-256をprogress.jsonと照合した。',
        'ft_1080のvideo_002/clip_013/cam2は欠損のまま。高解像度/tileの本評価は中止した。', '',
        '## ソース比較', '',
        '参照はCOCO由来のレビュー済み旧boxでありCOCOに有利。数値は旧box一致率で、検出recallではない。',
        'FTはROI前の全保存box、COCOはROI後のみ。COCOのROI外人数は未保存であり不明。unionはFT>=0.3に全COCO boxを加え、IoU>=0.5の重複ではFTを優先。',
        '各personは検出box候補で、重複・誤検出を含む。近遠は2選手の旧box下端y順位proxy。unknownもCSVに含めた。',
        'FTのms/frameはGPU forwardのみ。COCOは旧component全体の壁時計時間（load/decode/ROI/save込み）。unionは両者の和の参考値で、union自体の実測ではない。', '',
        '|source|対象camera-clip|IoU .5一致|IoU .3一致|候補/frame|ROI内/frame|ROI外/frame|既知非選手hit .3|ms/frame・範囲|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in sources['table']:
        if r['coverage'] != 'available' or r['camera'] != 'all' or r['near_far'] != 'all':
            continue
        outside = '未保存' if r['outside_per_frame'] is None else f"{r['outside_per_frame']:.3f}"
        lines.append(f"|{r['source']}|{r['camera_clips']}|{_pct(r['player_hit_05'], r['player_units'])}|{_pct(r['player_hit_03'], r['player_units'])}|"
                     f"{r['persons_per_frame']:.3f}|{r['inside_per_frame']:.3f}|{outside}|{_pct(r['non_player_hit_03'], r['non_player_units'])}|{r['ms_per_frame']:.1f} {r['runtime_kind']}|")
    lines += ['', '共通11 camera-clipの閾値0.01（1080版の欠損分を両方から除く）:', '',
              '|source|IoU .5一致|IoU .3一致|候補/frame|ROI内/frame|', '|---|---:|---:|---:|---:|']
    for r in sources['table']:
        if r['coverage'] == 'common11' and r['camera'] == 'all' and r['near_far'] == 'all' and r['source'] in ('ft_base_0.01', 'ft_1080_0.01'):
            lines.append(f"|{r['source']}|{_pct(r['player_hit_05'], r['player_units'])}|{_pct(r['player_hit_03'], r['player_units'])}|{r['persons_per_frame']:.3f}|{r['inside_per_frame']:.3f}|")
    lines += ['', '## コート選別', '',
        '有望候補はft_base_0.01 / COCO / union。1080は共通11件でIoU .3一致の改善がなく候補数・forward時間が増えたため、選別試行に採らなかった。',
        'Ultralytics BoT-SORTは元scoreを保存し、追加score gate=0 / fuse_score=False。全人物には上限をかけない。raw検出boxの足元を既存校正で写す。',
        'dwellはclipの25%以上の実観測frameが既存プレー領域（sideline +2.5m / baseline +5m）内にあるcamera-local track。その後、滞在frame数で上位6を候補とする。',
        '断片化したtrackの滞在時間はこの段階で合算しないため、短い断片を失う可能性がある。これは測定した未調整の診断ルールで、採用済み既定ではない。',
        '第2確認は既存CLIP-ReID＋コート幾何のcamera間対応。1秒未満の曖昧区間除外・同camera最大0.2秒重複handoffを含む既存ルールを維持。',
        'associatedは地面上の足元整合を含む対応結果であり、完全な3D人体再構成の検証ではない。未決定clipに最終IDを補わない。',
        'old_pipelineは保存済み旧COCO＋旧BoT-SORT/Lab再連結のbaseline。derivativeは今回の全人物入力のpose/外観が未保存のため未評価。最終追跡方式比較ではない。',
        'identity保持はその人物のラベル単位の50%以上が残ること、非選手identity除外は選択単位がゼロであること。単位は人物×camera×frameで旧重複boxをまとめる。',
        'IoU .3/.5両方を記録。以下はラケットを含むFT boxを考慮した .3。全非選手の除外には未検出を含むため、追跡でhitした単位のうち選別で除外した数も別に示す。', '',
        '|source|段|成功clip|選手identity保持|選手単位保持|非選手identity全除外|非選手単位除外|追跡hit非選手の選別除外|隣コート除外|コート外除外|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in selection['table']:
        lines.append(f"|{r['source']}|{r['stage']}|{r['clips']}|{_pct(r['player_identities_kept50_03'], r['player_identities'])}|{_pct(r['player_kept_03'], r['player_units'])}|"
                     f"{_pct(r['non_player_identities_rejected_all_03'], r['non_player_identities'])}|{_pct(r['non_player_excluded_03'], r['non_player_units'])}|"
                     f"{_pct(r['non_player_rejected_among_tracked_03'], r['non_player_tracked_03'])}|{_pct(r['adjacent_court_excluded_03'], r['adjacent_court_units'])}|{_pct(r['off_court_excluded_03'], r['off_court_units'])}|")
    lines += ['', '|source|段|cam0 far保持frame|cam1 far保持frame|cam2 far保持frame|', '|---|---|---:|---:|---:|']
    for r in selection['table']:
        lines.append(f"|{r['source']}|{r['stage']}|" + '|'.join(_pct(r[f'{c}_far_covered_03'], r[f'{c}_far_reference_frames']) for c in ('cam0', 'cam1', 'cam2')) + '|')
    lines += ['', '第2確認の状態:', '']
    for variant, clips in selection['records'].items():
        for clip, r in clips.items():
            lines.append(f"- {variant} / {clip}: {r['association']['status']} {r['association'].get('reason', '')}; 候補数 " + str({c: v['candidates'] for c, v in r['cameras'].items()}))
    lines += ['', '## 3camera動画', '', f"`{video['path']}`", '',
        f"SHA-256 `{video['sha256']}`。{video['frames']}frame / {video['fps']}fps、全frameのdecode読戻しを検証。灰色=利用可能な全人物box、色=コート滞在＋CLIP対応の予測identity。GTを色付けに使わない。", '',
        '## camera×近遠の全ソース表', '', '候補人数は当該側のbox数を全camera-frameで割った値。ms/frameはcamera全体の値。', '',
        '|source|camera|近遠|frame|旧box一致 .5|旧box一致 .3|候補/frame|ROI内/frame|ROI外/frame|非選手hit .3|ms/frame|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in sources['table']:
        if r['coverage'] != 'available' or r['camera'] == 'all' or r['near_far'] == 'all':
            continue
        outside = '未保存' if r['outside_per_frame'] is None else f"{r['outside_per_frame']:.3f}"
        lines.append(f"|{r['source']}|{r['camera']}|{r['near_far']}|{r['frames']}|{_pct(r.get('player_hit_05', 0), r.get('player_units', 0))}|{_pct(r.get('player_hit_03', 0), r.get('player_units', 0))}|"
                     f"{r['persons_per_frame']:.3f}|{r['inside_per_frame']:.3f}|{outside}|{_pct(r.get('non_player_hit_03', 0), r.get('non_player_units', 0))}|{r['ms_per_frame']:.1f}|")
    lines += ['', '全数値: sources.csv、selection.csv、selection_per_clip.csv、selection_per_camera.csv、selection_identities.csv。',
              '行別根拠: selection_units.jsonl.gz。hash/校正/全重み/各追跡・候補の出自: sources.json、tracks.json、selection.json。', '']
    (report / 'report.md').write_text('\n'.join(lines))
