"""Format already-scored JSON/CSV/unit receipts. Never imports a scorer or predictions."""
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
DEV = BUNDLE.parent / 'run-i964-recalibration-fit-r14-20261001/dev'
SCORE = BUNDLE / 'scoring-r18'
unseen = json.loads((SCORE / 'summary.json').read_text())
dev = json.loads((DEV / 'comparison.json').read_text())
lines = ['# run18: 凍結未見3clip・一回採点結果', '',
'ラベルcommit **43da331cc**をpush後、変更のない`person_unseen_score.py`を一度だけ実行し正常終了した。採点crash・再試行は0。元の予測/ラベル/指標/設定を変更していない。以下は保存JSON/CSV/countの転記・加算で、再照合や再採点ではない。', '',
'## 全体と開発4clip（参考のみ）', '',
'rawは「コート選別後のraw ID」であり、選別前の全画面tracking精度ではない。全停止も母数へ含む。devは旧観測box参照、今回は同じ推論のraw box参照なので、数値差を改善・劣化や調整の根拠にしない。', '',
'|集合/段階|IDF1|IDTP/FP/FN|switch/fragment|選手保持unit|非選手残留unit|50%人物保持|',
'|---|---:|---:|---:|---:|---:|---:|']
def fmt(x):
    return 'NA' if x is None else f'{float(x):.6f}'
def show(name, m):
    lines.append(f'|{name}|{fmt(m["idf1"])}|{m["idtp"]}/{m["idfp"]}/{m["idfn"]}|{m["id_switches"]}/{m["fragments"]}|{m["player_units_kept"]}/{m["player_units"]}|{m["nonplayer_units_kept"]}/{m["nonplayer_units"]}|{m.get("player_identities_kept50","—")}/{m.get("player_identities","—")}|')
for name,m in unseen['tracking'].items(): show('未見 '+name,m)
for name,m in dev['tracking'].items(): show('dev '+name,m)
dev_final=list(csv.DictReader((DEV/'fitted-camera-near-far.csv').open()))[0]
show('dev associated',dev_final)
lines += ['', '|集合|決定clip|pair TP/FP/FN|pair F1|group accuracy|exclusion TP/FP/FN|#933 switch真/予測/一致|','|---|---:|---:|---:|---:|---:|---:|']
for name,m in [('未見',unseen['association']),('dev参考',dev['summaries']['fitted'])]:
    p,g,e,s=m['pairs'],m['group_accuracy'],m['exclusion'],m['id_switch']
    lines.append(f'|{name}|{m["decided_clips"]}/{m["total_clips"]}|{p["tp"]}/{p["fp"]}/{p["fn"]}|{fmt(p["f1"])}|{g["frames_correct"]}/{g["frames_scored"]}={fmt(g["accuracy"])}|{e["tp"]}/{e["fp"]}/{e["fn"]}|{s["true"]}/{s["predicted"]}/{s["matched"]}|')
lines += ['', '## clip別（全3clip）', '', '|clip|状態|pair TP/FP/FN|pair F1|group accuracy|#933 switch真/予測/一致|', '|---|---|---:|---:|---:|---:|']
for r in unseen['clips']:
    p,g,s=r['metrics']['pairs'],r['metrics']['group_accuracy'],r['metrics']['id_switch']
    lines.append(f'|{r["clip"]}|{r["status"]}|{p["tp"]}/{p["fp"]}/{p["fn"]}|{fmt(p["f1"])}|{fmt(g["accuracy"])}|{s["true"]}/{s["predicted"]}/{s["matched"]}|')
lines += ['', '**video_001/clip_003**は`annotation_side_missing_after_court_failure`により対応全-1。現courtでもcam0校正が得られずraw/group選別が空、cam1/2の局所選別は保持。全9,541選手unitを対応後IDFNへ、9,452 pairをFNへ残す。既存関数は予測正例0のclip単独pair F1をnull（precision未定義、recall 0）にする。これを0へ書き換えず、pooled TP/FP/FNの分母に含めた。', '',
'全clipとも冒頭f0/1はraw観測が無くラベルも無い。動画全長3,708frameに対しgroup採点frameは3,702。検出されなかった人物/frameを新たな負例にしない。', '',
'|clip/camera/段階|IDF1|IDTP/FP/FN|switch/fragment|選手保持unit|非選手残留unit|50%人物保持|', '|---|---:|---:|---:|---:|---:|---:|']
for r in unseen['clips']:
    for camera,t in r['tracking'].items():
        for stage in ['raw','group','associated']: show(f'{r["clip"]} {camera} {stage}',t[stage])
lines += ['', '## camera×near/far（事前scorerのCSV）', '',
'近遠はcamera/frame内のラベル選手box下端順位。下表は各集合をpoolした既存CSVで、unknownを省略しない。段階順はraw / group / associated。devは参考のみ。', '',
'|camera/層|未見IDF1 r/g/a|dev IDF1 r/g/a|未見保持 r/g/a / 選手unit|dev保持 r/g/a / 選手unit|', '|---|---:|---:|---:|---:|']
cs={stage:{(r['camera'],r['near_far']):r for r in csv.DictReader((SCORE/f'{stage}-camera-near-far.csv').open())} for stage in ['raw','group','associated']}
ds={stage:{(r['camera'],r['near_far']):r for r in csv.DictReader((DEV/f'{name}-camera-near-far.csv').open())} for stage,name in [('raw','raw'),('group','group'),('associated','fitted')]}
for camera in ['all','cam0','cam1','cam2']:
    for nf in ['all','near','far','unknown']:
        a=[cs[s][camera,nf] for s in cs]; d=[ds[s][camera,nf] for s in ds]
        idf=lambda rows:' / '.join(fmt(r['idf1'] or None) for r in rows)
        kept=lambda rows:' / '.join(r['player_units_kept'] for r in rows)+' / '+rows[0]['player_units']
        lines.append(f'|{camera}/{nf}|{idf(a)}|{idf(d)}|{kept(a)}|{kept(d)}|')
lines += ['', '## clip×camera×near/farの保持（保存unitのクロス表）', '',
'既に採点済みunitのラベル層と保持フラグだけを数えた表示用の表。新しい層別ID割当や指標関数の再実行はしていない。unknownの非選手数も併記する。', '',
'|clip/camera/層|raw/group/associated 保持 / 選手unit|非選手保持 r/g/a / 非選手unit|', '|---|---:|---:|']
counts={}
for stage in ['raw','group','associated']:
    values=Counter()
    with gzip.open(SCORE/f'{stage}-units.jsonl.gz','rt') as stream:
        for line in stream:
            u=json.loads(line); key=(u['clip'],u['camera'],u['near_far'],u['role'])
            values[key+('total',)]+=1;values[key+('kept',)]+=u['track_id'] is not None
    counts[stage]=values
for clip in [r['clip'] for r in unseen['clips']]:
    for camera in ['cam0','cam1','cam2']:
        for nf in ['near','far','unknown']:
            def totals(role):
                k=(clip,camera,nf,role)
                return '/'.join(str(counts[s][k+('kept',)]) for s in counts)+' / '+str(counts['raw'][k+('total',)])
            lines.append(f'|{clip} {camera} {nf}|{totals("player")}|{totals("non_player")}|')
lines += ['', '## unknown・coverage・停止', '', '|集合/clip/camera|照合label box / 全label box|未照合予測 / ID付き未照合|', '|---|---:|---:|']
for name,coverage in [('dev参考',dev['summaries']['fitted']['coverage'])]+[(r['clip'],r['metrics']['coverage']) for r in unseen['clips']]:
    for camera,c in coverage.items():
        lines.append(f'|{name} {camera}|{c["matched_label_boxes"]}/{c["label_boxes"]}|{c["unmatched_predicted_boxes"]}/{c["unmatched_predicted_boxes_with_id"]}|')
lines += ['', '未見raw_unlabelled_selected/group_unlabelledはいずれも全9cameraで0。曖昧box71は別途保持し人物評価から除く。label box照合100%は同じraw検出由来の部分参照であることの帰結であり、完全な検出recallではない。devの未照合15,114（ID付き67）は誤検出とはみなさない。', '',
'## 解釈と資源', '',
'設定・モデル・threshold・規則は凍結から不変。完了2clipの高い対応値と1clipの明示停止を両方報告し、成功clipだけの値へ差し替えない。この開封結果を用いた再fit/モデル選択/再推論は行わない。残るclip_000全pipeline検証はユーザー判断により#935 stackで行う。', '',
'見積り150–180分に対してinner111.49分、guard118.82分。NVML peak4.785GB（見積り4.3–8GB）、torch allocated3.021GB、最低RAM空き17.418GB、出力0.597GB（見積り1–3GB）。CPUのラベル/採点/動画分はreview receiptで別記する。学習曲線は該当しない。', '',
'正本: [single score summary](scoring-r18/summary.json)、[一回claim](scoring-r18/attempt.json)、[label hash](label-receipt-r18.json)、[目視手順と限界](labeling-r18.md)、[trial=1のdataset receipt](unseen-protocol-after-scoring-r18.json)。']
(BUNDLE/'report-r18.md').write_text('\n'.join(lines)+'\n')
