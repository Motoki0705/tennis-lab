"""Queue-only CUDA capture, exact source RGB checks, no geometric decisions."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from omegaconf import OmegaConf

from src.tasks.court_detection.configuration import TennisCourtDetectorSourceConfig, SyntheticCourtSourceConfig
from src.tasks.court_detection.data.inputs.tennis_court_detector import TennisCourtDetectorInput
from src.tasks.court_detection.data.inputs.synthetic_court import SyntheticCourtInput
from src.tasks.court_detection.inference.predictor import CourtPredictor
from src.tennis_scene.configuration import PROJECT_ROOT, _roots
from src.utils.configuration import PathRole
from src.utils.video import read_video_frame

OLD = Path('outputs/tennis_scene/evaluate/court-acceptance-implementation-20261116')
OUT = Path('outputs/tennis_scene/evaluate/court-fixed-support-rgb-20261117')


def main():
    cv2.setNumThreads(1)
    paths = OmegaConf.load(PROJECT_ROOT/'src/tennis_scene/configs/paths/default.yaml')
    paths.external_asset_root = '/home/kamimura/projects/tennis-lab/third_party'
    _, resolver = _roots(OmegaConf.to_container(paths, resolve=True))
    predictor = CourtPredictor.load_from_checkpoint(resolver.resolve(PathRole.CHECKPOINT,
        'court_detection/multiscale_depth3/b863df1f01f0.ckpt'), resolver=resolver, device='cuda', subpixel_refine=True)
    real = TennisCourtDetectorInput(TennisCourtDetectorSourceConfig('tennis_court_detector',
        Path('data/court_detection/tennis_court_detector-v1'), {'train':'train','val':'val','test':None}, ('QszoUKyCOHo_600',)))
    synthetic = SyntheticCourtInput(SyntheticCourtSourceConfig('synthetic_court','v3','target_court',
        Path('data/synthetic_data_generation/scenes'), ('B00','B01','B02','B03')))
    real_records = {r.sample_id:r for r in real.records('val')}
    synthetic_records = {r.sample_id:r for r in synthetic.records('test')}
    all_records = []
    start = time.monotonic()
    for group in ('raw','val_raw','test_raw'):
        old = json.loads((OLD/group/'manifest.json').read_text())
        records = []
        for index, record in enumerate(old['records']):
            destination = OUT/'raw'/record['key']
            destination.mkdir(parents=True, exist_ok=True)
            if (destination/'capture.json').exists():
                records.append(record); all_records.append({**record, 'old_raw_group':group}); continue
            if group == 'raw':
                rgb = cv2.cvtColor(read_video_frame(Path(record['source_video']),0).frame,cv2.COLOR_BGR2RGB)
            elif group == 'val_raw':
                rgb = np.asarray(real.load(real_records[record['clip_id']]).image).copy()
            else:
                rgb = np.asarray(synthetic.load(synthetic_records[record['clip_id']]).image).copy()
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == record['rgb_sha256'], record['key']
            assert cv2.imwrite(str(destination/'frame.png'), cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION,3])
            differences = []
            for region_index, (x0,y0,x1,y1) in enumerate(record['regions']):
                prediction = predictor.predict(np.ascontiguousarray(rgb[y0:y1,x0:x1]), heads=('kp','line'), postprocess='none')
                kp, line = prediction.raw_heads['kp'], prediction.raw_heads['line']
                values = dict(raw_xy=kp.keypoints[:,0].numpy(),scores=kp.scores[:,0].numpy(),valid=kp.valid[:,0].numpy(),
                    line_probability=line.probability.numpy(),image_size_hw=np.array(prediction.original_size_hw),
                    native_size_hw=np.array(prediction.native_size_hw))
                np.savez_compressed(destination/f'raw_{region_index:02d}.npz',**values)
                frozen = np.load(OLD/group/record['key']/f'raw_{region_index:02d}.npz')
                deltas = {}
                for name, value in values.items():
                    previous = frozen[name]
                    equal = bool(np.array_equal(value, previous, equal_nan=True))
                    if not equal:
                        assert value.shape == previous.shape, (record['key'], name)
                        delta = np.abs(value.astype(float)-previous.astype(float))
                        deltas[name] = {'max_abs':float(np.nanmax(delta)), 'nan_pattern_equal':bool(np.array_equal(np.isnan(value), np.isnan(previous)))}
                differences.append(deltas)
            (destination/'capture.json').write_text(json.dumps({'key':record['key'],'original_rgb_exact':True,
                'raw_differences':differences,'checkpoint':predictor.checkpoint_identity},allow_nan=False))
            (destination/'record.json').write_text(json.dumps(record))
            records.append(record);all_records.append({**record,'old_raw_group':group})
            print(json.dumps({'captured':record['key'],'regions':len(record['regions']),
                'different_regions':sum(bool(d) for d in differences),'elapsed':time.monotonic()-start}),flush=True)
        (OUT/f'{group}-manifest.json').write_text(json.dumps({'header':old['header'],'records':records}))
    (OUT/'manifest.json').write_text(json.dumps({'header':{'checkpoint':predictor.checkpoint_identity,
        'source':'exact original RGB; fresh CUDA raw; historical inputs unchanged'},'records':all_records}))
    (OUT/'capture-complete.json').write_text(json.dumps({'images':len(all_records),
        'regions':sum(len(r['regions']) for r in all_records),'seconds':time.monotonic()-start}))


if __name__ == '__main__':
    main()
