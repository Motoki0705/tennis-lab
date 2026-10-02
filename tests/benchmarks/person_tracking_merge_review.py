"""All merge-event context crops, independent of labels and the selected video windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from person_tracking_matrix import (  # type: ignore[import-not-found]
    checked,
    record_file,
)

from src.tennis_scene.pipeline.artifacts import write_json_atomic


def contacts(root: Path) -> None:
    audit = json.loads((root / 'merge_audit.json').read_text())
    identity = json.loads((root / 'identity.json').read_text())
    plan = json.loads(checked(identity['plan']).read_text())
    source = json.loads(checked(plan['sources']).read_text())
    sources = {(r['clip'], r['camera']): r for r in source['inputs']}
    pages, index = [], []
    directory = root / 'merge_contacts'
    directory.mkdir(exist_ok=True)
    for offset in range(0, len(audit['records']), 5):
        canvas: np.ndarray = np.zeros((5 * 300, 1200, 3), np.uint8)
        for entry, merge in enumerate(audit['records'][offset:offset + 5]):
            record = sources[merge['clip'], merge['camera']]
            capture = cv2.VideoCapture(str(checked(record['video'])))
            boxes = np.asarray([merge['kept_box'], merge['dropped_box']])
            center = (boxes[:, :2].min(0) + boxes[:, 2:].max(0)) / 2
            size = np.maximum(boxes[:, 2:].max(0) - boxes[:, :2].min(0), [40, 60])
            extent = np.maximum(size * 2, [160, 120])
            left = np.maximum(0, np.floor(center - extent / 2).astype(int))
            right = np.minimum([1920, 1080], np.ceil(center + extent / 2).astype(int))
            title = f'{offset+entry}: {merge["clip"]}/{merge["camera"]} f{merge["frame"]} keep={merge["kept_row"]} drop={merge["dropped_row"]} IoU={merge["iou"]:.3f}'
            cv2.putText(canvas, title, (8, entry * 300 + 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
            try:
                for column, shift in enumerate((-6, 0, 6)):
                    frame = max(0, min(record['video']['num_frames'] - 1, merge['frame'] + shift))
                    if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, frame):
                        raise ValueError('Cannot seek merge context')
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError('Cannot read merge context')
                    crop = image[left[1]:right[1], left[0]:right[0]]
                    scale = min(390 / crop.shape[1], 250 / crop.shape[0])
                    tile = cv2.resize(crop, None, fx=scale, fy=scale)
                    # Exact boxes only at the merged frame, never propagated as observations.
                    if shift == 0:
                        for box, color, tag in zip(boxes, ((255, 255, 0), (0, 0, 255)), ('KEEP', 'DROP'), strict=True):
                            p = np.rint((box.reshape(2, 2) - left) * scale).astype(int)
                            cv2.rectangle(tile, tuple(p[0]), tuple(p[1]), color, 2)
                            cv2.putText(tile, tag, tuple(p[0] + [0, 12]), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
                    cv2.putText(tile, f'f{frame}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                    canvas[entry * 300 + 35:entry * 300 + 35 + tile.shape[0], column * 400:column * 400 + tile.shape[1]] = tile
            finally:
                capture.release()
            index.append({'entry': offset + entry, 'page': offset // 5, **merge})
        path = directory / f'page_{offset//5:02d}.jpg'
        if path.exists():
            raise FileExistsError(path)
        if not cv2.imwrite(str(path), canvas):
            raise ValueError('Cannot save contact page')
        pages.append(record_file(path))
    write_json_atomic(root / 'merge_contacts.json', {'pages': pages, 'records': index,
        'scope': 'Every dropped box, merge frame and +/-6 source frames, enlarged context. Only merge-frame boxes overlaid.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    contacts(args.report)
