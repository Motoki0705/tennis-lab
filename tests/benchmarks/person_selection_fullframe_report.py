"""Tables and integrity audit of all seven fixed-dev full-frame source variants."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.person_tracking.selection_metrics import aggregate_units
from src.tasks.player_detection.evaluation.fullframe_sources import SOURCES
from src.tasks.player_detection.evaluation.person_sources import DEV_CLIPS, write_csv
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256


def table(headers: list[str], rows: list[list[Any]]) -> str:
    return '\n'.join(['|' + '|'.join(headers) + '|', '|' + '|'.join('---' for _ in headers) + '|',
                      *('|' + '|'.join(str(v) for v in row) + '|' for row in rows)])


def ratio(row: dict[str, Any], numerator: str, denominator: str) -> str:
    return f"{row[numerator]}/{row[denominator]}"


def summarize(report: Path) -> None:
    sources = json.loads((report / 'sources.json').read_text())
    protocol_path = report / 'selection_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    totals, cameras, identities, clips, associations = [], [], [], [], []
    burden: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    artifact_hashes = []
    for name in SOURCES:
        all_units: list[dict[str, Any]] = []
        for clip in DEV_CLIPS:
            path = report / 'selection' / name / clip / 'result.json'
            value = json.loads(path.read_text())
            if value['protocol_sha256'] != dual_sha256(protocol_path):
                raise ValueError('Protocol changed')
            results.setdefault(name, {})[clip] = value
            records = [value['units']]
            for camera in value['cameras'].values():
                records.extend([camera, camera['groups']])
                if 'identities' in camera:
                    records.append(camera['identities'])
                with np.load(camera['path'], allow_pickle=False) as a:
                    if (a['selected'] & ~a['observed']).any() or camera['linked']['selected_groups'] > 6:
                        raise ValueError('Selection lost observed-subset/cap invariants')
                    # Every selected fragment retains all of its actual observations.
                    for group in camera['linked']['groups']:
                        if group['selected']:
                            for index in group['fragments']:
                                fragment = camera['linked']['fragments'][index]
                                row, start, end = fragment['row'], fragment['start'], fragment['end']
                                if not np.array_equal(a['selected'][row, start:end], a['observed'][row, start:end]):
                                    raise ValueError('Selected fragment was spatially trimmed')
            for record in records:
                if dual_sha256(Path(record['path'])) != record['sha256']:
                    raise ValueError('Result artifact changed')
                artifact_hashes.append({'path': record['path'], 'sha256': record['sha256']})
            with gzip.open(value['units']['path'], 'rt') as handle:
                units = [json.loads(line) for line in handle]
            all_units.extend(units)
            for stage in sorted({u['stage'] for u in units}):
                clips.append({'source': name, 'clip': clip, 'stage': stage, **aggregate_units([u for u in units if u['stage'] == stage])})
            burden.extend({'source': name, 'clip': clip, **row} for row in value['burden'])
            associations.append({'source': name, 'clip': clip, 'status': value['association']['status'], 'reason': value['association'].get('reason', ''),
                'camera_groups': {cam: record['linked']['selected_groups'] for cam, record in value['cameras'].items()}})
        for stage in sorted({u['stage'] for u in all_units}):
            rows = [u for u in all_units if u['stage'] == stage]
            totals.append({'source': name, 'stage': stage, 'clips': len({u['clip'] for u in rows}), **aggregate_units(rows)})
            for cam in ('cam0', 'cam1', 'cam2'):
                for side in ('all', 'near', 'far', 'unknown'):
                    cameras.append({'source': name, 'stage': stage, 'camera': cam, 'near_far': side,
                        **aggregate_units([u for u in rows if u['camera'] == cam and (side == 'all' or u['near_far'] == side)])})
            for clip, person in sorted({(u['clip'], u['person']) for u in rows}):
                identities.append({'source': name, 'stage': stage, 'clip': clip, 'person': person,
                    **aggregate_units([u for u in rows if (u['clip'], u['person']) == (clip, person)])})
    population = []
    for source_row in sources['table']:
        name, cam, side = (source_row[k] for k in ('source', 'camera', 'near_far'))
        rows = [r for r in burden if r['source'] == name and r['near_far'] == side and (cam == 'all' or r['camera'] == cam)]
        frame_count = sum(r['frames'] for r in rows)
        row = {'source': name, 'camera': cam, 'near_far': side, 'frames': frame_count,
            'persons': source_row['persons'], 'persons_per_frame': source_row['persons_per_frame']}
        for field in ('track_observations', 'selected_observations', 'tracks_total', 'selected_tracks_total'):
            row[field] = sum(r[field] for r in rows)
        row['tracks_per_frame'] = row['track_observations'] / frame_count
        row['selected_per_frame'] = row['selected_observations'] / frame_count
        for field in ('max_tracks_per_frame', 'max_selected_per_frame'):
            row[field] = max(r[field] for r in rows)
        population.append(row)
    result = {'protocol': protocol, 'sources_sha256': dual_sha256(report / 'sources.json'),
              'results': results, 'totals': totals, 'camera': cameras, 'identity': identities,
              'clip': clips, 'population': population, 'association': associations}
    write_json_atomic(report / 'comparison.json', result)
    for filename, rows in (('totals', totals), ('camera', cameras), ('identity', identities), ('clip', clips), ('population', population), ('association', associations)):
        write_csv(report / f'{filename}.csv', rows)
    write_json_atomic(report / 'verified_artifacts.json', {'artifacts': artifact_hashes, 'selection_invariants': 'all passed: observed subset, <=6 groups, no spatial trimming of selected fragments'})
    primary = [r for r in totals if r['stage'] == 'selected']
    lines = ['# #964 run 6 — ROI前の人物source比較（固定4 dev clip）', '',
        '参照boxは旧COCO/旧trackerから作った部分ラベルで、**COCOに有利**。旧boxとの一致を検出recallとは呼ばない。未ラベル予測をFPと数えない。全sourceは800/1333・全画面・ROI前。同じmotion/IoU BoT-SORTを使うsource比較で、最終tracking方式比較ではない。', '',
        '## 固定した選別規則', '',
        'box下端中心を校正cameraのz=0平面へ投影。シングルス幅 |x|≤4.115m、|y|≤16.885mのcoreにおけるdistinctな実観測frameがclipの25%以上となる連結groupを選手候補とし、その後に上限6。ダブルス幅 |x|≤5.485mは診断値で、出力を切らない。**選択済み断片の全実観測を保持する**（wide run・無効足元も含む）。', '',
        'gap>1秒または既存0.25秒窓/3m jumpで区間を切る。両方coreへ入る断片を位置連続性＋CLIP cosine≥.8のvetoで連結。mutual nearestと次点差.2を要求、1秒未満の曖昧断片は除外、同camera handoff重複は≤.2秒。滞在の重複は1回だけ数える。選別maskは重複観測も保持し、既存v3対応に渡すgroup timelineだけ早いraw track IDで1box/frameへ正規化する。検出sourceの閾値以降にscore gate/fusionは無い。ラベルはこの判定へ渡さない。', '',
        '全raw trackでcropの遮蔽を検査し、coreに入るtrackへ既定CLIPをCPUで適用。同じ動画/hash・重み/hash・frame・resized RGBが完全一致するcropだけ再利用する。crop不足・短いsegmentに埋め込みが無い場合はmissingと記録し、幾何だけで照合する既存挙動。camera間対応は同じ既存CLIP+幾何の第2確認で、undecidedを補完しない。', '',
        '主表は選別直後、IoU≥.3。IoU≥.5、clip別、identity別、対応成功clipだけのassociated表はCSV/JSONに保存。選手ID保持は当該集計範囲でラベルunitの50%以上、非選手ID完全除外は残存0。unit=(clip,camera,frame,person)で重複boxをまとめる。near/farは同frameの2選手のbox下端順位、順位が定まらなければunknown。', '',
        '## 全体', '', table(['source','選手ID保持','選手unit保持','非選手ID全除外','非選手unit除外（全体）','非選手unit除外（追跡hit内）','隣コート除外','コート外除外','wide保持','対応決定clip'],
        [[r['source'], ratio(r,'player_identities_kept50_03','player_identities'), ratio(r,'player_kept_03','player_units'),
          ratio(r,'non_player_identities_rejected_all_03','non_player_identities'), ratio(r,'non_player_excluded_03','non_player_units'),
          ratio(r,'non_player_rejected_among_tracked_03','non_player_tracked_03'), ratio(r,'adjacent_court_excluded_03','adjacent_court_units'),
          ratio(r,'off_court_excluded_03','off_court_units'), ratio(r,'player_wide_kept_03','player_wide_units'),
          str(sum(a['status']=='ok' for a in associations if a['source']==r['source']))+'/4'] for r in primary]), '',
        '## camera × near/far', '', table(['source','camera','近遠','選手ID保持','選手unit保持','非選手unit除外（全体）','非選手unit除外（追跡hit内）','隣コート除外','コート外除外','wide保持'],
        [[r['source'], r['camera'], r['near_far'], ratio(r,'player_identities_kept50_03','player_identities'), ratio(r,'player_kept_03','player_units'),
          ratio(r,'non_player_excluded_03','non_player_units'), ratio(r,'non_player_rejected_among_tracked_03','non_player_tracked_03'),
          ratio(r,'adjacent_court_excluded_03','adjacent_court_units'), ratio(r,'off_court_excluded_03','off_court_units'), ratio(r,'player_wide_kept_03','player_wide_units')]
         for r in cameras if r['stage']=='selected' and r['near_far'] in ('near','far','unknown')]), '',
        '## 人物候補・trackの負荷', '',
        'personsは検出box件数（ユニークな実人物数ではない）。tracks_totalはcamera/clip内raw IDの累計。near/farを跨ぐIDは各層に現れるため層別累計を足し合わせない。frame平均の分母はcamera-frame。選別raw ID数と上限6の連結group数は別。', '',
        table(['source','camera','近遠','person box総数','person/frame','raw track累計','track/frame','最大track/frame','選別box/frame'],
        [[r['source'], r['camera'], r['near_far'], r['persons'], f"{r['persons_per_frame']:.3f}", r['tracks_total'],
          f"{r['tracks_per_frame']:.3f}", r['max_tracks_per_frame'], f"{r['selected_per_frame']:.3f}"]
         for r in population if (r['camera']=='all' and r['near_far']=='all') or (r['camera']!='all' and r['near_far'] in ('near','far','unknown'))]), '',
        '## 既存CLIPのcamera間対応', '', table(['source','clip','状態','理由','camera別候補group'],
        [[r['source'], r['clip'], r['status'], r['reason'], r['camera_groups']] for r in associations]), '',
        '成功clipだけの対応指標を全4clipの主表と同一視しない。sideは既存の注釈ballによる判定を固定した。最終方式・encoder比較、未見、全pipeline完走は未実施。pipeline既定・#937重み/閾値は変更していない。', '']
    (report / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(primary, indent=2), flush=True)
