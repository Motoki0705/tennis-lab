"""Verify final CPU media, bundle masks, source-plate correspondence and decoding."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.utils.checksum import dual_sha256
from src.utils.video.reader import RandomAccessVideoReader

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--render', type=Path, required=True)
parser.add_argument('--replay', type=Path, required=True)
args = parser.parse_args()
root = args.render
meta = json.loads((args.replay / 'scene.metadata.json').read_text())
scene = np.load(args.replay / 'scene.npz')
manifest = json.loads((root / 'blender_bundle/manifest.json').read_text())
ball_valid = np.load(root / 'blender_bundle/ball_valid.npy')
ball_positions = np.load(root / 'blender_bundle/ball_positions.npy')
np.testing.assert_array_equal(ball_valid, scene['ball_3d_valid'])
np.testing.assert_array_equal(ball_positions[ball_valid], scene['ball_3d'][ball_valid])
assert manifest['ball']['max_gap_frames'] == 0 and manifest['ball']['filled_frames'] == []
result = {'ball_display_masks_equal_replayed_3d': True, 'display_ball_filled_frames': 0,
          'valid_ball_frames': int(ball_valid.sum()), 'gifs': {}}
expected = {'multiview_reconstruction.gif': ((960, 620), 168, 16.75),
            'rich_scene_rendering.gif': ((400, 338), 168, 16.75),
            'blender_reveal_meiji_cam1.gif': ((400, 225), None, 16.9169)}
for filename, (size, count, duration) in expected.items():
    path = root / filename
    with Image.open(path) as gif:
        assert gif.size == size, (filename, gif.size)
        assert gif.info['loop'] == 0
        if count is not None:
            assert gif.n_frames == count
        milliseconds = 0
        for i in range(gif.n_frames):
            gif.seek(i)
            gif.load()
            milliseconds += gif.info['duration']
        assert abs(milliseconds / 1000 - duration) < .15
        result['gifs'][filename] = {'size': size, 'frames': gif.n_frames, 'duration_s': milliseconds / 1000,
                                   'bytes': path.stat().st_size, 'sha256': dual_sha256(path), 'fully_decoded': True}
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], check=True)
subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(root / 'reveal.mp4'), '-f', 'null', '-'], check=True)
result['mp4_fully_decoded'] = True
video = Path(meta['video_paths'][meta['camera_ids'].index('cam1')])
frame = 96
rendered = cv2.imread(str(root / f'blender/frames/{frame:05d}.png'))
with RandomAccessVideoReader(video) as reader:
    expected_frame = cv2.resize(reader.read(frame), (960, 540), interpolation=cv2.INTER_AREA)
psnr = float(cv2.PSNR(expected_frame, rendered))
assert psnr > 38, psnr
result['blender_plate_source_frame'] = frame
result['blender_plate_psnr_db'] = psnr
samples = [0, 180, 360, 540, 720, 1008]
images = [cv2.resize(cv2.imread(str(root / f'blender/frames/{frame:05d}.png')), (400, 225)) for frame in samples]
cv2.imwrite(str(root / 'blender_contact_sheet.png'), np.concatenate([np.concatenate(images[:3], axis=1), np.concatenate(images[3:], axis=1)], axis=0))
(root / 'media-validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
