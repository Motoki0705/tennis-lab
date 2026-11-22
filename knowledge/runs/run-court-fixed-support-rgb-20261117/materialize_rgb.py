"""Original RGB only, CPU decoding, frozen neural evidence stays untouched."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from src.tasks.court_detection.configuration import TennisCourtDetectorSourceConfig, SyntheticCourtSourceConfig
from src.tasks.court_detection.data.inputs.tennis_court_detector import TennisCourtDetectorInput
from src.tasks.court_detection.data.inputs.synthetic_court import SyntheticCourtInput
from src.utils.video import read_video_frame

OLD = Path('outputs/tennis_scene/evaluate/court-acceptance-implementation-20261116')
OUT = Path('outputs/tennis_scene/evaluate/court-fixed-support-rgb-20261117')
SOURCES = {}


def materialize(record):
    cv2.setNumThreads(1)
    destination = OUT/'rgb'/record['key']/'frame.png'
    if destination.exists():
        rgb = cv2.cvtColor(cv2.imread(str(destination)),cv2.COLOR_BGR2RGB)
    else:
        group = record['old_raw_group']
        if group == 'raw':
            rgb = cv2.cvtColor(read_video_frame(Path(record['source_video']),0).frame,cv2.COLOR_BGR2RGB)
        else:
            if group not in SOURCES:
                if group == 'val_raw':
                    source = TennisCourtDetectorInput(TennisCourtDetectorSourceConfig('tennis_court_detector',
                        Path('data/court_detection/tennis_court_detector-v1'), {'train':'train','val':'val','test':None}, ('QszoUKyCOHo_600',)))
                    records = source.records('val')
                else:
                    source = SyntheticCourtInput(SyntheticCourtSourceConfig('synthetic_court','v3','target_court',
                        Path('data/synthetic_data_generation/scenes'), ('B00','B01','B02','B03')))
                    records = source.records('test')
                SOURCES[group] = source, {r.sample_id:r for r in records}
            source, records = SOURCES[group]
            rgb = np.asarray(source.load(records[record['clip_id']]).image).copy()
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(destination),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3])
    assert hashlib.sha256(rgb.tobytes()).hexdigest() == record['rgb_sha256'], record['key']
    return record['key']


def main():
    records = [dict(record,old_raw_group=group) for group in ('raw','val_raw','test_raw')
               for record in json.loads((OLD/group/'manifest.json').read_text())['records']]
    (OUT/'manifest.json').write_text(json.dumps({'source':'frozen neural raw; exact original RGB SHA256 checked','records':records}))
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(materialize,record) for record in records]
        for index, future in enumerate(as_completed(futures)):
            key = future.result()
            if index % 100 == 0: print(index+1,key,flush=True)
    (OUT/'rgb-complete.json').write_text(json.dumps({'images':len(records),'all_original_rgb_exact':True,'seconds':time.monotonic()-start}))


if __name__ == '__main__':main()
