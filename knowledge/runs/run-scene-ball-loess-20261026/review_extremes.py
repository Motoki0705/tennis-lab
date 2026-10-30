"""Inspect extrema and lost-valid frames without changing filters or intervals."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.tennis_scene.pipeline.storage.codec import unpack_value
from src.tennis_scene.visualization.review.trajectory import masked_trajectory
from src.utils.geometry.triangulation import PointRejection
from src.utils.video.reader import RandomAccessVideoReader

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--replay', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
manifest = json.loads((args.replay / 'replay.json').read_text())
meta = json.loads((args.replay / 'scene.metadata.json').read_text())
with np.load(manifest['provenance']['source_scene']) as data:
    before = {k: data[k] for k in ('ball_3d', 'ball_3d_valid', 'ball_uv', 'ball_vis')}
with np.load(args.replay / 'scene.npz') as data:
    after = {k: data[k] for k in ('ball_3d', 'ball_3d_valid', 'ball_uv', 'ball_vis', 'ball_rejection_code')}
rows = []
for camera in meta['camera_ids']:
    directory = args.replay / f'ball_{camera}'
    doc = json.loads((directory / 'artifact.json').read_text())
    rows.append(unpack_value(doc['payload'], directory, doc['arrays']))
lost = np.flatnonzero(before['ball_3d_valid'] & ~after['ball_3d_valid'])
maximum = {}
for name, scene in (('before', before), ('after', after)):
    valid = scene['ball_3d_valid']
    step = np.where(valid[1:] & valid[:-1], np.linalg.norm(np.diff(scene['ball_3d'], axis=0), axis=1), -1)
    i = int(step.argmax())
    maximum[name] = {'frames': [i, i + 1], 'step_m': float(step[i]),
        'before_valid': before['ball_3d_valid'][i:i + 2].tolist(),
        'after_valid': after['ball_3d_valid'][i:i + 2].tolist(),
        'before_real_views': before['ball_vis'][:, i:i + 2].sum(0).tolist(),
        'after_real_views': after['ball_vis'][:, i:i + 2].sum(0).tolist()}
review = {'maximum_adjacent_step': maximum, 'lost_valid_frames': [
    {'frame': int(i), 'reason': PointRejection(int(after['ball_rejection_code'][i])).name,
     'real_views_before': before['ball_vis'][:, i].sum().item(), 'real_views_after': after['ball_vis'][:, i].sum().item(),
     'rejected_cameras': [camera for camera, row in zip(meta['camera_ids'], rows, strict=True) if row['smoothing']['rejected'][i]]}
    for i in lost], 'review_frame_groups': {'bounce_gap': [56, 57, 58, 59, 60], 'recovered_max_step': [968, 969, 970, 971],
                                         'rejected_detector_excursion': [761, 762, 764, 765]},
    'interpretation': 'No 3D ground truth. Remaining/newly exposed pair-change discontinuity at 969->970 and speed-gate gap at 57..59 are retained, not hidden by manual intervals.'}
fig, axes = plt.subplots(3, 2, figsize=(10, 6), constrained_layout=True)
for col, bounds in enumerate(((50, 66), (960, 980))):
    for scene, label, color in ((before, 'Before', '#de8c29'), (after, 'LOESS replay', '#168666')):
        xyz = masked_trajectory(scene['ball_3d'], scene['ball_3d_valid'])
        for dim in range(3):
            axes[dim, col].plot(np.arange(*bounds), xyz[slice(*bounds), dim], '.-', ms=3, label=label, color=color)
            axes[dim, col].set(xlabel='Source frame', ylabel=f'{"XYZ"[dim]} (m)')
            axes[dim, col].grid(alpha=.2)
axes[0, 0].set_title('Bounce region: speed-gate gap at 57..59')
axes[0, 1].set_title('Recovered region: max step at 969 -> 970')
axes[0, 0].legend(fontsize=8)
fig.savefig(args.output / 'extremes.png', dpi=100)
plt.close(fig)
cv2.setNumThreads(2)
for group, frames in review['review_frame_groups'].items():
    panels = [[] for _ in frames]
    for view, camera in enumerate(meta['camera_ids']):
        with RandomAccessVideoReader(Path(meta['video_paths'][view])) as reader:
            for row, frame in enumerate(frames):
                image = reader.read(frame)
                center = None
                for scene, color in ((before, (0, 150, 255)), (after, (50, 255, 60))):
                    if scene['ball_vis'][view, frame]:
                        pixel = np.rint(scene['ball_uv'][view, frame] * [image.shape[1], image.shape[0]]).astype(int)
                        cv2.circle(image, tuple(pixel), 10, color, 2, cv2.LINE_AA)
                        center = pixel if center is None else center
                if center is None:
                    center = np.array([image.shape[1] // 2, image.shape[0] // 2])
                left = max(0, min(image.shape[1] - 440, int(center[0]) - 220))
                top = max(0, min(image.shape[0] - 260, int(center[1]) - 130))
                panel = np.zeros((300, 440, 3), np.uint8)
                panel[40:] = image[top:top + 260, left:left + 440]
                text = f'{camera} f{frame} raw={int(before["ball_vis"][view, frame])} accepted={int(after["ball_vis"][view, frame])}'
                cv2.putText(panel, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
                panels[row].append(panel)
    cv2.imwrite(str(args.output / f'{group}.png'), np.concatenate([np.concatenate(row, axis=1) for row in panels], axis=0))
(args.output / 'extremes.json').write_text(json.dumps(review, indent=2) + '\n')
print(json.dumps(review, indent=2))
