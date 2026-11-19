"""Evaluate a finished immutable build, then render every camera/clip on CPU.

The evaluator owns the reference; no estimate is recomputed or changed here.
Native crops and a canvas viewer preserve access to the source beneath lines.
"""
from collections import Counter
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src.tasks.court_detection.geometry.homography import COURT_HOMOGRAPHY_EDGES, court_template_xy
from src.tasks.court_detection.geometry.hybrid_homography import project_candidate
from src.tennis_scene.pipeline.video_court.store import VideoCourtStore
from src.tennis_scene.pipeline.video_court.component import SharedCourtInput, SharedCourtModule
from src.tennis_scene.pipeline.clip_inputs import prepare_clip_source
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.utils.video import read_video_frame
from build import save, sha


def difference(matrix, reference, visible):
    if matrix is None:
        return {'mean_px': None, 'max_px': None, 'mean_all_px': None, 'max_all_px': None, 'within_tolerance': False}
    actual = project_candidate(np.asarray(matrix, float), court_template_xy().astype(float))
    expected = project_candidate(np.asarray(reference, float), court_template_xy().astype(float))
    if actual is None or expected is None:
        raise ValueError('Invalid evaluation homography')
    distances = np.linalg.norm(actual - expected, axis=1)
    primary = distances[visible]
    return {'mean_px': float(primary.mean()), 'max_px': float(primary.max()),
            'mean_all_px': float(distances.mean()), 'max_all_px': float(distances.max()),
            'within_tolerance': bool(primary.mean() <= 10 and primary.max() <= 25)}


def segments(matrix):
    if matrix is None:
        return []
    xy = court_template_xy().astype(float)[np.asarray(COURT_HOMOGRAPHY_EDGES)].reshape(-1, 2)
    result = project_candidate(np.asarray(matrix, float), xy)
    assert result is not None
    return result.reshape(-1, 2, 2).tolist()


def draw(image, court, reference):
    result = image.copy()
    for lines, colour in ((reference, (255, 0, 255)), (court, (0, 255, 0))):
        for a, b in lines:
            ok, first, last = cv2.clipLine((0, 0, image.shape[1], image.shape[0]),
                                         tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)))
            if ok:
                cv2.line(result, first, last, colour, 1, cv2.LINE_AA)
    return result


def label(image, text, *, border=False):
    result = cv2.copyMakeBorder(image, 54, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)) if border else image.copy()
    if not border:
        cv2.rectangle(result, (0, 0), (result.shape[1], 54), (25, 25, 25), -1)
    for index, line in enumerate(text):
        width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .6, 1)[0][0]
        size = .6 * min(1., (result.shape[1] - 16) / max(width, 1))
        cv2.putText(result, line, (8, 20 + 23 * index), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def crop_boxes(matrix, width, height):
    # Evaluation-only display regions, derived from the estimated court itself.
    # A failed camera uses the reference only to locate review crops; it stays failed.
    lines = segments(matrix)
    net = project_candidate(np.asarray(matrix, float), np.array([[-5.485, 0], [5.485, 0]]))
    candidates = {'near-baseline': [lines[1]], 'net-side': [net],
                  'left-sideline': [lines[2], lines[4]], 'right-sideline': [lines[3], lines[5]]}
    boxes = {}
    for name, selected in candidates.items():
        points = []
        for line in selected:
            a, b = np.asarray(line, float)
            valid, p, q = cv2.clipLine((0, 0, width, height), tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)))
            if valid:
                points.extend((p, q))
        if not points:
            boxes[name] = None
            continue
        lower = np.maximum(np.min(points, axis=0) - [64, 80], 0)
        upper = np.minimum(np.max(points, axis=0) + [65, 81], [width, height])
        boxes[name] = [int(lower[0]), int(lower[1]), int(upper[0]), int(upper[1])]
    return boxes


def imwrite(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    options = [cv2.IMWRITE_JPEG_QUALITY, 98] if path.suffix == '.jpg' else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if not cv2.imwrite(str(path), image, options):
        raise OSError(f'Could not save {path}')


HTML = '''<!doctype html><html lang="ja"><meta charset="utf-8"><title>動画共有コートの確認</title>
<style>body{font-family:system-ui;margin:16px;background:#17191e;color:#eee}label{margin:10px}a{color:#8cf}#viewport{overflow:auto;max-height:80vh;border:1px solid #555}canvas{display:block}#crops{display:flex;flex-wrap:wrap;gap:10px}#crops img{width:300px;max-height:240px;object-fit:contain;object-position:left}pre{white-space:pre-wrap}button,select{font:inherit}</style>
<h1>動画共有コート: meiji_4cam / video_000</h1>
<p>全69clip × 4camera。<span style="color:#0f0">緑: 共通H</span> / <span style="color:#f0f">紫: 承認済み参照</span>。線を消して原画像を確認できます。画像とcropは元解像度です。人間の承認は未完了です。</p>
<p>参照との距離はcameraごとに同一Hの比較を繰り返すため、69回の独立な精度評価ではありません。heldout RGBの線残差とdriftはclipごとに別途測定しました。参照と推定器は白線抽出部品を共有しており、独立性に限界があります。</p>
<p>driftは各clipのheldout 1frameで検査しています。先頭frameは目視用で、ここではdriftを数値検査していません。frame間の移動を保証する評価ではありません。</p>
<label>camera <select id="camera"></select></label><label>clip <select id="clip"></select></label>
<label>frame <select id="frame"><option value="first">先頭frame</option><option value="heldout">heldout 4/5 frame</option></select></label>
<button id="prev">前</button><button id="next">次</button>
<label>拡大 <select id="zoom"><option value="0.5">50%</option><option value="1" selected>100%</option><option value="2">200%</option><option value="4">400%</option></select></label>
<label><input type="checkbox" id="shared" checked>共通H</label><label><input type="checkbox" id="reference">参照H</label>
<p id="status"></p><div id="viewport"><canvas id="canvas"></canvas></div><h2>先頭frameの元解像度crop</h2><div id="crops"></div>
<h2>診断</h2><pre id="details"></pre><p><a href="../cases.csv">全camera/clipの表</a> · <a href="../evaluation-summary.json">評価集計</a></p>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
for(const c of [...new Set(data.map(r=>r.camera))])$('camera').add(new Option(c,c));
for(const c of [...new Set(data.map(r=>r.clip_id))])$('clip').add(new Option(c,c));
let currentImage, current, imageVersion=0;
function paint(){if(!currentImage)return;const cv=$('canvas');cv.width=currentImage.width;cv.height=currentImage.height;const ctx=cv.getContext('2d');ctx.drawImage(currentImage,0,0);for(const [id,lines,col] of [['reference',current.reference,'#ff00ff'],['shared',current.court,'#00ff00']])if($(id).checked){ctx.strokeStyle=col;ctx.lineWidth=1;for(const [a,b] of lines){ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}}const scale=Number($('zoom').value);cv.style.width=cv.width*scale+'px';cv.style.height=cv.height*scale+'px';}
function load(){const version=++imageVersion;currentImage=null;$('canvas').getContext('2d').clearRect(0,0,$('canvas').width,$('canvas').height);current=data.find(r=>r.camera===$('camera').value&&r.clip_id===$('clip').value);$('status').textContent=`${current.clip_id} / ${current.camera} / ${current.status} / ${current.reason??'drift警告なし'}`;$('details').textContent=JSON.stringify(current.metrics,null,2);$('crops').replaceChildren();for(const [name,paths] of Object.entries(current.crops)){let div=document.createElement('div');let title=document.createElement('p');title.textContent=name;div.append(title);if(paths){let a=document.createElement('a');a.href=paths.overlay;let im=document.createElement('img');im.src=paths.overlay;im.loading='lazy';a.append(im);div.append(a);let p=document.createElement('p');let raw=document.createElement('a');raw.href=paths.raw;raw.textContent='原画像crop';p.append(raw);div.append(p)}else div.append(document.createTextNode('画像外'));$('crops').append(div)}const image=new Image();image.onload=()=>{if(version!==imageVersion)return;currentImage=image;paint()};image.onerror=()=>{if(version===imageVersion)$('status').textContent='画像を読み込めません: '+image.src};image.src=current[$('frame').value];}
for(const id of ['camera','clip','frame'])$(id).onchange=load;for(const id of ['zoom','shared','reference'])$(id).onchange=paint;
$('prev').onclick=()=>{$('clip').selectedIndex=Math.max(0,$('clip').selectedIndex-1);load()};$('next').onclick=()=>{$('clip').selectedIndex=Math.min($('clip').options.length-1,$('clip').selectedIndex+1);load()};load();
</script></html>'''


def main(args):
    cv2.setNumThreads(1)
    started = time.monotonic()
    build_summary = json.loads((args.output / 'build-summary.json').read_text())
    artifact = VideoCourtStore(args.output / 'store').load(build_summary['artifact_digest'])
    reference = json.loads(args.reference.read_text())
    assert reference['approved_for_evaluation'] and reference['human_reviewed']
    refs = {r['camera']: r for r in reference['cameras']}
    reference_frames = {(camera['camera'], row['clip_id'], row['frame_index'])
                        for camera in json.loads((args.reference.parent / 'frame-stability.json').read_text()) for row in camera['frames']}
    split = json.loads(args.split.read_text())
    for key in ('seed', 'fit', 'evaluation'):
        assert not reference_frames & {(r['camera_id'], r['clip_id'], r['frame_index']) for r in split[key]}
    expected_fit = {(r['camera_id'], r['clip_id'], r['frame_index']) for r in split['fit']}
    expected_eval = {(r['camera_id'], r['clip_id'], r['frame_index']) for r in split['evaluation']}
    inventory = {r['path']: r for r in map(json.loads, (args.legacy / 'inventories/comparisons.jsonl').read_text().splitlines())}
    review = args.output / 'review'
    rows, gallery, camera_metrics, legacy_provenance, line_rows = [], [], {}, [], []
    clip_sources = {}
    component_points = {}
    component = SharedCourtModule()
    for camera, value in artifact.cameras.items():
        assert {(camera, clip, frame) for clip, frame in value['diagnostics']['requested_fit_frames']} == {v for v in expected_fit if v[0] == camera}
        ref = refs[camera]
        primary = ref['visible_reference_indices']
        accuracy = difference(value['matrix'], ref['matrix_court_metres_to_image_pixels'], primary)
        camera_metrics[camera] = {'common_status': value['status'], 'accuracy': accuracy,
            'selected_initializer': value['diagnostics']['selected_initializer'],
            'model_accepted_seed_clips': value['diagnostics']['seed_model_accepted_clip_count'],
            'used_fit_frames': len(value['diagnostics']['fit_frames']), 'clip_counts': value['diagnostics']['clip_status_counts']}
        court, reference_lines = segments(value['matrix']), segments(ref['matrix_court_metres_to_image_pixels'])
        boxes = crop_boxes(value['matrix'] or ref['matrix_court_metres_to_image_pixels'], *value['size_wh'])
        for clip, status in value['clips'].items():
            diagnostic, = status['frames']
            frame = diagnostic['frame_index']
            assert (camera, clip, frame) in expected_eval
            media = args.dataset / f'videos/video_000/clips/{clip.split("/")[1]}/media/{camera}.mp4'
            assert sha(media) == value['sources'][clip]['sha256']
            if clip not in clip_sources:
                ids = tuple(sorted(artifact.cameras))
                paths = tuple((media.parent / f'{c}.mp4').resolve() for c in ids)
                clip_sources[clip] = prepare_clip_source(paths, ids, clip_id=clip).source
                artifact.bind(clip_sources[clip])
            try:
                result = component.process(SharedCourtInput(clip_sources[clip].video(camera), clip, value,
                                                           artifact.manifest['cameras'][camera]))
            except ReconstructionUnavailable as error:
                assert status['status'] == 'failed' and error.reason == 'video_shared_court_failed'
                component_status = 'explicit_failure'
            else:
                assert status['status'] != 'failed'
                assert result.diagnostics['temporal_policy'] == 'static_video_shared'
                if camera in component_points:
                    np.testing.assert_array_equal(result.keypoints, component_points[camera])
                component_points[camera] = result.keypoints
                component_status = 'shared_output'
            legacy_relative = f'evaluation/meiji_4cam/{clip}/{camera}/comparison.json'
            legacy_path = args.legacy / legacy_relative
            assert sha(legacy_path) == inventory[legacy_relative]['sha256']
            old = json.loads(legacy_path.read_text())
            legacy_provenance.append({'path': legacy_relative, 'sha256': sha(legacy_path)})
            old_h = None
            excluded_legacy_frame = (camera, clip, 0) in reference_frames
            if old['old_selected_region'] is not None and not excluded_legacy_frame:
                region = next(g for g in old['geometry'] if g['region'] == old['old_selected_region'])
                x, y = region['region'][:2]
                old_h = np.array([[1, 0, x], [0, 1, y], [0, 0, 1.]]) @ np.array(region['old_h'])
            old_delta = difference(None if value['matrix'] is None else old_h, value['matrix'], primary)
            common_residuals = [r['p95_px'] for r in diagnostic.get('common_line_residuals', []) if r['used_by_local_fit']]
            for measured_line in diagnostic.get('common_line_residuals', []):
                line_rows.append({'camera': camera, 'clip_id': clip, 'frame_index': frame,
                                  'status': status['status'], 'reason': diagnostic['reason'], **measured_line})
            row = {'camera': camera, 'clip_id': clip, 'evaluation_frame': frame, 'common_status': value['status'],
                   'clip_status': status['status'], 'component_status': component_status, 'reason': diagnostic['reason'],
                   'heldout_rgb_diagnosed': value['status'] == 'accepted',
                   'crop_geometry': 'shared' if value['matrix'] is not None else 'reference_for_display_only',
                   'measurement_reason': diagnostic.get('measurement', {}).get('reason'),
                   'geometry_reasons': '|'.join(diagnostic.get('geometry_reasons', [])),
                   'reference_mean_px': accuracy['mean_px'], 'reference_max_px': accuracy['max_px'],
                   'reference_within_tolerance': accuracy['within_tolerance'],
                   'drift_mean_px': diagnostic['mean_displacement_px'], 'drift_max_px': diagnostic['max_displacement_px'],
                   'drift_max_projection_std_px': diagnostic.get('max_visible_projection_std_px'),
                   'common_line_p95_max_px': max(common_residuals, default=None),
                   'legacy_clip_accepted': None if excluded_legacy_frame else old['accepted'],
                   'legacy_comparison_status': 'excluded_reference_frame' if excluded_legacy_frame else 'measured' if old_h is not None else 'legacy_not_accepted',
                   'legacy_to_shared_mean_px': old_delta['mean_px'],
                   'legacy_to_shared_max_px': old_delta['max_px'], 'selected_initializer': value['diagnostics']['selected_initializer'],
                   'camera_digest': artifact.manifest['cameras'][camera]}
            rows.append(row)
            key = f'{camera}/{clip.split("/")[1]}'
            paths = {}
            crops = {}
            for kind, index in [('first', 0), ('heldout', frame)]:
                image = read_video_frame(media, index).frame
                rgb_digest = hashlib.sha256(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
                if kind == 'heldout' and value['status'] == 'accepted':
                    assert rgb_digest == value['diagnostics']['decoded_rgb_sha256'][f'{clip}/{index}']
                overlay = draw(image, court, reference_lines)
                labels = [f'{camera} {clip} / {kind} frame {index}',
                          f'{status["status"]}: {diagnostic["reason"] or "ok"} (drift inspected at heldout frame {frame})']
                for name, pixels in [('source.jpg', image), ('overlay.jpg', label(overlay, labels))]:
                    imwrite(review / key / kind / name, pixels)
                paths[kind] = f'{key}/{kind}/source.jpg'
                if kind == 'first':
                    for name, box in boxes.items():
                        if box is None:
                            crops[name] = None
                            continue
                        x1, y1, x2, y2 = box
                        crops[name] = {'box_xyxy': box}
                        for style, pixels in [('raw', image), ('overlay', overlay)]:
                            relative = f'{key}/crops/{name}-{style}.png'
                            crop = pixels[y1:y2, x1:x2]
                            if style == 'overlay':
                                crop = label(crop, [f'{camera} {clip.split("/")[-1]} / {status["status"]}',
                                                    diagnostic['reason'] or 'drift: ok'], border=True)
                            imwrite(review / relative, crop)
                            crops[name][style] = relative
            gallery.append({'camera': camera, 'clip_id': clip, 'status': status['status'], 'reason': diagnostic['reason'],
                            **paths, 'crops': crops, 'court': court, 'reference': reference_lines, 'metrics': row})
        print(camera, camera_metrics[camera], flush=True)
    with (args.output / 'cases.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    if line_rows:
        with (args.output / 'drift-lines.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(line_rows[0]))
            writer.writeheader(); writer.writerows(line_rows)
    (review / 'index.html').write_text(HTML.replace('__DATA__', json.dumps(gallery).replace('<', '\\u003c')))
    summary = {'artifact_digest': artifact.digest, 'reference_sha256': sha(args.reference), 'split_sha256': sha(args.split),
               'reference_excluded_from_seed_fit_evaluation': True, 'cameras': camera_metrics, 'camera_clips': len(rows),
               'successful_camera_clips': sum(r['clip_status'] != 'failed' for r in rows),
               'complete_four_camera_clips': sum(all(r['clip_status'] != 'failed' for r in rows if r['clip_id'] == clip) for clip in {r['clip_id'] for r in rows}),
               'false_accepted_cameras': sum(m['common_status'] == 'accepted' and not m['accuracy']['within_tolerance'] for m in camera_metrics.values()),
               'false_accepted_camera_clips': sum(r['clip_status'] != 'failed' and not r['reference_within_tolerance'] for r in rows),
               'drift_reason_counts': dict(Counter(str(r['reason']) for r in rows)),
               'real_component_counts': dict(Counter(r['component_status'] for r in rows)),
               'camera_clips_with_reference_error': sum(not r['reference_within_tolerance'] for r in rows),
               'frame0_overlays': len(gallery), 'heldout_overlays': len(gallery),
               'measured_line_rows': len(line_rows),
               'native_crop_pairs': sum(p is not None for row in gallery for p in row['crops'].values()),
               'human_reviewed': False, 'independent_reference_extractor': False, 'seconds': time.monotonic() - started,
               'driver_sha256': sha(Path(__file__)), 'legacy_comparisons': legacy_provenance}
    artifact.verify()
    save(args.output / 'evaluation-summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'legacy_comparisons'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    main(parser.parse_args())
