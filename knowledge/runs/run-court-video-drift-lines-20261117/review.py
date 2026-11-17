"""Native-resolution review and per-frame diagnostic tables, no estimator changes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path.cwd()
PREVIOUS = ROOT / 'knowledge/runs/run-court-video-shared-acceptance-20261117'
sys.path[:0] = [str(ROOT), str(PREVIOUS)]
spec = importlib.util.spec_from_file_location('run77_evaluate', PREVIOUS / 'evaluate.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
from src.tennis_scene.pipeline.video_court.store import VideoCourtStore
from src.tennis_scene.pipeline.video_court.diagnostics import DriftReference, load_video_drift
from src.utils.video import read_video_frame
from build import save, sha


def render(job):
    output, dataset, camera, clip, base, report, accuracy, reference_lines, previous = job
    cv2.setNumThreads(1)
    name = clip.split('/')[1]
    media = dataset / f'videos/video_000/clips/{name}/media/{camera}.mp4'
    court = legacy.segments(base['matrix'])
    boxes = legacy.crop_boxes(base['matrix'], *base['size_wh'])
    frames = []
    for index in [0, *[f['frame_index'] for f in report['frames']]]:
        measured = next((f for f in report['frames'] if f['frame_index'] == index), None)
        image = read_video_frame(media, index).frame
        if measured:
            assert hashlib.sha256(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()).hexdigest() == measured['rgb_sha256']
        path = Path(camera) / name / f'frame-{index:06d}'
        destination = output / 'review' / path
        overlay = legacy.draw(image, court, reference_lines)
        verdict = f'{report["status"]}: {report["reason"] or "ok"}' if measured is None else f'{measured["status"]}: {measured["reason"] or "ok"}'
        title = [f'{camera} {clip} frame {index} / {verdict}',
                 'visual review only (frame 0)' if measured is None else f'mean={measured["mean_displacement_px"]} max={measured["max_displacement_px"]} px']
        legacy.imwrite(destination / 'source.jpg', image)
        legacy.imwrite(destination / 'overlay.jpg', legacy.label(overlay, title, border=True))
        crops = {}
        for label, box in boxes.items():
            if box is None:
                crops[label] = None
                continue
            x0,y0,x1,y1 = box
            legacy.imwrite(destination / f'{label}-raw.png', image[y0:y1,x0:x1])
            legacy.imwrite(destination / f'{label}-overlay.png', overlay[y0:y1,x0:x1])
            crops[label] = {'raw': str(path / f'{label}-raw.png'), 'overlay': str(path / f'{label}-overlay.png')}
        frames.append({'index': index, 'source': str(path / 'source.jpg'), 'overlay': str(path / 'overlay.jpg'),
                       'report': measured, 'crops': crops})
    return {'camera': camera, 'clip_id': clip, 'status': report['status'], 'reason': report['reason'],
            'worst_frame_index': report['worst_frame_index'], 'frames': frames, 'court': court, 'reference': reference_lines,
            'accuracy': accuracy, 'previous': previous}


HTML = '''<!doctype html><html lang="ja"><meta charset="utf-8"><title>共有コート・frame別drift確認</title>
<style>body{font-family:system-ui;margin:16px;background:#17191e;color:#eee}label{display:inline-block;margin:8px}a{color:#8cf}#viewport{overflow:auto;max-height:72vh;border:1px solid #555}canvas{display:block}#crops{display:flex;flex-wrap:wrap;gap:12px}#crops img{width:300px;max-height:220px;object-fit:contain}pre{white-space:pre-wrap}button,select{font:inherit}td,th{border:1px solid #555;padding:5px}table{border-collapse:collapse}.failed{color:#ff9090}.warning{color:#ffd166}.selected{background:#34404f}</style>
<h1>共有コートとframe別の変位</h1>
<p>全69clip × 4camera。共通Hはrun 77と同一です。<span style="color:#0f0">緑: 共通H</span>、<span style="color:#f0f">紫: 承認済み参照</span>。人間の品質確認は未完了です。</p>
<p>各clipの1/10・3/10・7/10・8/10・9/10付近を独立に検査し、最も悪いframeでclipを判定します。時間平均は使いません。検査不能は失敗です。未測定frameの安全を保証しません。先頭frameは目視用です。</p>
<p>参照と推定器は白線抽出部品を共有しています。同じ共通Hと参照Hの距離を69回表示するため、69回の独立した精度検証ではありません。</p>
<label>camera <select id="camera"></select></label><label>clip <select id="clip"></select></label><label>frame <select id="frame"></select></label>
<button id="prev">前</button><button id="next">次</button><button id="worst">最も悪いframe</button>
<label>拡大 <select id="zoom"><option value="0.5">50%</option><option value="1" selected>100%</option><option value="2">200%</option><option value="4">400%</option></select></label>
<label><input type="checkbox" id="shared" checked>共通H</label><label><input type="checkbox" id="reference">参照H</label><label><input type="checkbox" id="motion" checked>測定したKP変位（橙）</label>
<p id="status"></p><table><thead><tr><th>frame</th><th>判定</th><th>mean px</th><th>max px</th><th>最大std px</th><th>理由</th></tr></thead><tbody id="table"></tbody></table>
<p><a id="download">選択frameの重ね描き</a></p><div id="viewport"><canvas id="canvas"></canvas></div><h2>選択frameの原寸crop</h2><div id="crops"></div>
<h2>来歴・frame別診断</h2><pre id="details"></pre><p><a href="../frames.csv">全frameの表</a> · <a href="../clips.csv">全clipの表</a> · <a href="../evaluation-summary.json">集計</a></p>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
for(const c of [...new Set(data.map(r=>r.camera))])$('camera').add(new Option(c,c));for(const c of [...new Set(data.map(r=>r.clip_id))])$('clip').add(new Option(c,c));
let current,currentFrame,currentImage,version=0;
function paint(){if(!currentImage)return;const cv=$('canvas');cv.width=currentImage.width;cv.height=currentImage.height;const ctx=cv.getContext('2d');ctx.drawImage(currentImage,0,0);
for(const [id,lines,col] of [['reference',current.reference,'#f0f'],['shared',current.court,'#0f0']])if($(id).checked){ctx.strokeStyle=col;ctx.lineWidth=1;for(const [a,b] of lines){ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}}
const f=currentFrame.report;if($('motion').checked&&f&&f.keypoint_displacements_px){const points=__POINTS__[current.camera];ctx.strokeStyle='#ffaa00';ctx.lineWidth=2;for(const i of f.visible_keypoint_indices){const a=points[i],d=f.keypoint_displacements_px[i];ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(a[0]+d[0],a[1]+d[1]);ctx.stroke();ctx.beginPath();ctx.arc(a[0]+d[0],a[1]+d[1],3,0,Math.PI*2);ctx.stroke()}}
const zoom=Number($('zoom').value);cv.style.width=cv.width*zoom+'px';cv.style.height=cv.height*zoom+'px';}
function frame(){const token=++version;currentImage=null;currentFrame=current.frames.find(f=>String(f.index)===$('frame').value);const f=currentFrame.report;$('status').className=current.status;$('status').textContent=`${current.camera} ${current.clip_id}: clip=${current.status} / frame ${currentFrame.index}=${f?f.status:'目視のみ'}`;$('download').href=currentFrame.overlay;
$('details').textContent=JSON.stringify({clip_status:current.status,clip_reason:current.reason,worst_frame:current.worst_frame_index,reference_accuracy:current.accuracy,old_clip_comparison:current.previous,selected_frame:f},null,2);
$('crops').replaceChildren();for(const [name,paths] of Object.entries(currentFrame.crops)){const div=document.createElement('div');const p=document.createElement('p');p.textContent=name;div.append(p);if(paths){const a=document.createElement('a');a.href=paths.overlay;const img=new Image();img.src=paths.overlay;img.loading='lazy';a.append(img);div.append(a);const raw=document.createElement('a');raw.href=paths.raw;raw.textContent='原画像crop';div.append(raw)}else div.append('画像外');$('crops').append(div)}
for(const row of $('table').rows)row.className=row.dataset.frame===$('frame').value?'selected':'';
const image=new Image();image.onload=()=>{if(token!==version)return;currentImage=image;paint()};image.onerror=()=>{if(token===version)$('status').textContent='画像を読み込めません: '+image.src};image.src=currentFrame.source;}
function load(){current=data.find(r=>r.camera===$('camera').value&&r.clip_id===$('clip').value);$('frame').replaceChildren();$('table').replaceChildren();for(const item of current.frames){$('frame').add(new Option(item.index===0?'先頭（目視のみ）':String(item.index),item.index));if(!item.report)continue;const f=item.report,row=$('table').insertRow();row.dataset.frame=String(item.index);for(const value of [item.index,f.status,f.mean_displacement_px,f.max_displacement_px,f.max_visible_projection_std_px,f.geometry_reasons.join(', ')||f.reason||'']){const td=row.insertCell();td.textContent=typeof value==='number'?Number.isInteger(value)?value:value.toFixed(2):value??'—'}row.onclick=()=>{$('frame').value=String(item.index);frame()}}
$('frame').value=String(current.worst_frame_index);frame();}
for(const id of ['camera','clip'])$(id).onchange=load;$('frame').onchange=frame;for(const id of ['zoom','shared','reference','motion'])$(id).onchange=paint;
$('prev').onclick=()=>{$('clip').selectedIndex=Math.max(0,$('clip').selectedIndex-1);load()};$('next').onclick=()=>{$('clip').selectedIndex=Math.min(68,$('clip').selectedIndex+1);load()};$('worst').onclick=()=>{$('frame').value=String(current.worst_frame_index);frame()};load();
</script></html>'''


def main(args):
    started = time.monotonic()
    summary = json.loads((args.output / 'evaluation-summary.json').read_text())
    base = VideoCourtStore(Path('outputs/tennis_scene/evaluate/court-video-shared-acceptance-20261117/store')).load(summary['base_bundle_digest'])
    artifact = load_video_drift(base, DriftReference(args.output / 'drift-store', summary['drift_bundle_digest']))
    reference_path = Path('knowledge/runs/run-court-white-line-reference-20261116/reference/references.json')
    ref = json.loads(reference_path.read_text())
    assert ref['approved_for_evaluation'] and ref['human_reviewed']
    refs = {r['camera']:r for r in ref['cameras']}
    old = {(r['camera'],r['clip_id']):r for r in csv.DictReader((PREVIOUS / 'cases.csv').open())}
    jobs, clip_rows, frame_rows = [], [], []
    points = {}
    for camera, doc in artifact.drift_cameras.items():
        r = refs[camera]
        value = base.cameras[camera]
        accuracy = legacy.difference(value['matrix'], r['matrix_court_metres_to_image_pixels'], r['visible_reference_indices'])
        reference_lines = legacy.segments(r['matrix_court_metres_to_image_pixels'])
        points[camera] = legacy.project_candidate(np.asarray(value['matrix']), legacy.court_template_xy().astype(float)).tolist()
        for clip, report in doc['clips'].items():
            previous = old[camera,clip]
            retained = {k:v for k,v in previous.items() if k.startswith('legacy_') or k=='selected_initializer'}
            clip_rows.append({'camera':camera,'clip_id':clip,'status':report['status'],'reason':report['reason'],
                'worst_frame':report['worst_frame_index'], 'reference_mean_px':accuracy['mean_px'], 'reference_max_px':accuracy['max_px'], **retained})
            for f in report['frames']:
                frame_rows.append({'camera':camera,'clip_id':clip,'frame_index':f['frame_index'],'status':f['status'],'reason':f['reason'],
                    'geometry_reasons':'|'.join(f['geometry_reasons']),'mean_px':f['mean_displacement_px'],'max_px':f['max_displacement_px'],
                    'max_projection_std_px':f['max_visible_projection_std_px'], 'supported_lines':'|'.join(f.get('supported_lines',[])),
                    'unsupported_observed_lines':'|'.join(f.get('unsupported_observed_lines',[])), 'rgb_sha256':f['rgb_sha256']})
            jobs.append((args.output,args.dataset,camera,clip,value,report,accuracy,reference_lines,retained))
    for name, rows in [('frames',frame_rows),('clips',clip_rows)]:
        with (args.output / f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        gallery=list(pool.map(render,jobs))
    page=HTML.replace('__DATA__',json.dumps(gallery,ensure_ascii=False).replace('<','\\u003c')).replace('__POINTS__',json.dumps(points))
    (args.output/'review/index.html').write_text(page)
    artifact.verify()
    save(args.output/'review-summary.json',{'camera_clips':len(gallery),'diagnostic_frames':len(frame_rows),'first_overlays':len(gallery),
        'diagnostic_overlays':len(frame_rows),'native_crop_pairs':sum(sum(bool(c) for c in f['crops'].values()) for row in gallery for f in row['frames']),
        'seconds':time.monotonic()-started,'human_reviewed':False,'reference_sha256':sha(reference_path),
        'source_rgb_hashes_verified':len(frame_rows)})
    print((args.output/'review-summary.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('data/tennis_multiview/processed/meiji_4cam/dataset'))
    p.add_argument('--workers',type=int,default=4);main(p.parse_args())
