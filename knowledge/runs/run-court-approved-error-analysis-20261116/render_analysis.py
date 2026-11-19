"""Native-pixel evidence and compact figures for error patterns and RGB failures."""
import hashlib
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from common import BASE, OLD, EDGES, project, source_rows, truth, write_json


def image_for(key):
    meta = json.loads((OLD / 'raw' / key / 'record.json').read_text())
    capture = cv2.VideoCapture(meta['source_video']); ok, image = capture.read(); capture.release(); assert ok
    assert hashlib.sha256(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()).hexdigest() == meta['rgb_sha256']
    return image


def draw(image, points, color):
    output = image.copy()
    for a, b in EDGES:
        if np.isfinite(points[[a, b]]).all() and np.abs(points[[a, b]]).max() < 1e5:
            cv2.line(output, tuple(np.rint(points[a] * 16).astype(int)), tuple(np.rint(points[b] * 16).astype(int)), color, 1, cv2.LINE_AA, shift=4)
    return output


def panel(image, reference, prediction, label):
    visible = (reference[:, 0] >= 0) & (reference[:, 0] <= 1919) & (reference[:, 1] >= 0) & (reference[:, 1] <= 1079)
    distance = np.linalg.norm(reference - prediction, axis=1); indices = np.flatnonzero(visible); index = int(indices[np.argmax(distance[visible])])
    x, y = reference[index]; left = int(np.clip(round(x - 256), 0, 1408)); top = int(np.clip(round(y - 96), 0, 888))
    reference_image = draw(image, reference, (0, 220, 0))
    overlays = [image, reference_image, draw(reference_image, prediction, (255, 0, 255))]
    result = np.zeros((242, 1536, 3), np.uint8)
    cv2.putText(result, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, f'RAW | APPROVED GREEN | GREEN + ESTIMATE MAGENTA; KP{index}; bbox={left},{top},512,192', (8, 41), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
    result[50:] = np.hstack([im[top:top + 192, left:left + 512] for im in overlays])
    return result, {'kp': index, 'bbox_xywh': [left, top, 512, 192], 'max_error_px': float(distance[index])}


def main():
    cv2.setNumThreads(1); out = BASE / 'figures'; native = out / 'native'; native.mkdir(parents=True, exist_ok=True)
    rows = {r['key']: r for _, r in source_rows()}; examples = json.loads((BASE / 'candidate-analysis/representatives.json').read_text())
    records = []
    for camera in ('cam0', 'cam1', 'cam2', 'cam3'):
        chosen = [r for r in examples if r['camera'] == camera and r['condition'] == 'postprocess_and_acceptance']
        panels = []
        for r in chosen:
            predicted = np.array(rows[r['key']]['conditions'][r['condition']]['points']); reference, _ = truth(camera)
            label = f"{r['key']} C {r['pattern']}; mean={r['mean_error_px']:.2f}px max={r['max_error_px']:.2f}px"
            figure, location = panel(image_for(r['key']), reference, predicted, label)
            filename = f"{camera}-{r['pattern']}.png"; cv2.imwrite(str(native / filename), figure)
            panels.append(figure); records.append({**r, **location, 'native_image': 'native/' + filename})
        Image.fromarray(cv2.cvtColor(np.vstack(panels), cv2.COLOR_BGR2RGB)).quantize(colors=128).save(out / f'{camera}-patterns.png', optimize=True)
    pilot = json.loads((BASE / 'rgb-pilot/results.json').read_text())
    failures = [r for r in pilot if not r['correct']]
    pilot_figures = []
    for r in failures:
        reference, _ = truth(r['camera']); predicted = project(r['matrix'])
        figure, location = panel(image_for(r['key']), reference, predicted,
            f"RGB PILOT {r['key']}; mean {r['old_mean_px']:.2f} -> {r['mean_error_px']:.2f}px; max {r['max_error_px']:.2f}px")
        name = r['key'].replace('/', '-') + '-rgb.png'; cv2.imwrite(str(native / name), figure); pilot_figures.append(figure)
        records.append({'key': r['key'], 'type': 'RGB_pilot_failure', 'native_image': 'native/' + name, **location})
    Image.fromarray(cv2.cvtColor(np.vstack(pilot_figures), cv2.COLOR_BGR2RGB)).quantize(colors=128).save(out / 'rgb-pilot-failures.png', optimize=True)
    raw = json.loads((BASE / 'raw-analysis/summary.json').read_text())
    values = np.array([[next(r['error_px']['median'] for r in raw['full_frame_per_keypoint'] if r['camera'] == camera and r['kp'] == i) for i in range(14)] for camera in ('cam0', 'cam1', 'cam2', 'cam3')])
    fig, ax = plt.subplots(figsize=(12, 3.4)); im = ax.imshow(np.log10(np.maximum(values, 1)), cmap='magma', vmin=0, vmax=3)
    for y in range(4):
        for x in range(14): ax.text(x, y, f'{values[y,x]:.0f}', color='cyan' if values[y,x] > 100 else 'white', ha='center', va='center', fontsize=9)
    ax.set_xticks(range(14), [f'KP{i}' for i in range(14)]); ax.set_yticks(range(4), ['cam0', 'cam1', 'cam2', 'cam3'])
    ax.set_title('Frozen full-frame raw KP: per-point median error (original px)'); fig.colorbar(im, ax=ax, label='log10(pixel error)')
    fig.tight_layout(); fig.savefig(out / 'raw-kp-errors.png', dpi=140); plt.close(fig)
    write_json(out / 'manifest.json', {'reference_status': 'human_approved', 'native_crops_unscaled': True,
        'knowledge_previews_color_quantized': True, 'records': records})
    links = ''.join(f'<li>{r["key"]}: <a href="{r["native_image"]}">{r.get("pattern",r.get("type"))}</a></li>' for r in records)
    (out / 'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><h1>承認済み参照による誤差分析</h1><p>緑＝承認参照、マゼンタ＝推定。原画／参照／比較の元pixel crop。遠方点の外挿と時間変化は承認済みの前提。分類は誤差形状の診断で、画像だけで原因を断定したものではありません。</p><ul>' + links + '</ul></html>')


if __name__ == '__main__': main()
