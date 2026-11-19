"""RGB-only visibility contact sheets on a declared 20-image subset."""
import cv2
import numpy as np
from common import BASE, WHITE, truth, write_json


def main():
    out = BASE / 'occlusion-review'; out.mkdir(exist_ok=True)
    manifest = []
    for camera in ('cam0', 'cam1', 'cam2', 'cam3'):
        gt, visible = truth(camera); sheet = np.zeros((5 * 102 + 34, 14 * 84, 3), np.uint8)
        for column in range(14): cv2.putText(sheet, f'KP{column}', (column * 84 + 5, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        for row, clip in enumerate((0, 17, 34, 51, 68)):
            source = WHITE / 'frames' / camera / f'clip_{clip:03d}_frame_000000.png'; image = cv2.imread(str(source)); assert image is not None
            for column, (x, y) in enumerate(gt):
                ox, oy = column * 84, 34 + row * 102
                if visible[column]:
                    # Original 80x80 pixels, no scaling. Small red centre mark remains removable via saved raw crop.
                    crop = cv2.getRectSubPix(image, (80, 80), (float(x), float(y)))
                    name = f'{camera}-clip{clip:03d}-kp{column:02d}.png'; cv2.imwrite(str(out / name), crop)
                    marked = crop.copy(); cv2.drawMarker(marked, (40, 40), (0, 0, 255), cv2.MARKER_CROSS, 5, 1)
                    sheet[oy:oy + 80, ox:ox + 80] = marked
                else: name = None
                cv2.putText(sheet, f'{clip:03d}', (ox + 5, oy + 96), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                manifest.append({'camera': camera, 'clip': clip, 'kp': column, 'reference_visible': bool(visible[column]), 'source': str(source), 'crop': name})
        cv2.imwrite(str(out / f'{camera}-sheet.png'), sheet)
    write_json(out / 'manifest.json', {'sample_clips': [0, 17, 34, 51, 68], 'method': 'agent inspection of original RGB, no confidence/heatmap labels', 'points': manifest})


if __name__ == '__main__': main()
