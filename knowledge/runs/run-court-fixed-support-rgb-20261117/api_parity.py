"""Frozen heads through the public region API versus offline acceptance choices."""
import hashlib
import argparse
import json
import time
from unittest.mock import patch

import cv2
import numpy as np

from evaluate import OUT, OLD, LineEvidence, fixed, typed, prepare, court_template_xy, COURT_HOMOGRAPHY_EDGES
from src.tasks.court_detection.inference.regions import CourtRegionSearchConfig, CourtRegionUnavailable, region_proposals, select_court_region


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--accepted-only',action='store_true');args=parser.parse_args()
    cv2.setNumThreads(1)
    records=[r for r in json.loads((OUT/'manifest.json').read_text())['records']
             if r['dataset']=='meiji_4cam' and (r['clip_id']=='video_000/clip_068' and r['camera']=='cam0'
                 if args.accepted_only else r['clip_id']=='video_000/clip_000')]
    assert len(records)==(1 if args.accepted_only else 4)
    report=[];start=time.monotonic()
    for record in records:
        image=cv2.cvtColor(cv2.imread(str(OUT/'rgb'/record['key']/'frame.png')),cv2.COLOR_BGR2RGB)
        assert [list(r) for r in region_proposals(image)]==record['regions']
        prepared={};predictions={};raws={}
        for index,region in enumerate(record['regions']):
            raw=dict(np.load(OLD/record['old_raw_group']/record['key']/f'raw_{index:02d}.npz'))
            key=hashlib.sha256(raw['line_probability'].tobytes()).hexdigest(),tuple(raw['image_size_hw'])
            prepared[key]=prepare(raw,OUT/'line-evidence'/record['key']/f'evidence_{index:02d}.npz')
            raws[tuple(region)]=raw
        def evidence(probability,size,**_):return prepared[hashlib.sha256(probability.tobytes()).hexdigest(),tuple(size)]
        with patch.object(LineEvidence,'from_probability',side_effect=evidence):
            for region,raw in raws.items():
                observed=raw['raw_xy'].astype(float);observed[~raw['valid']]=np.nan
                geometry=fixed.estimate_hybrid_homography(court_template_xy().astype(float),observed,raw['scores'],raw['line_probability'],
                    edges=np.asarray(COURT_HOMOGRAPHY_EDGES),image_size_hw=tuple(raw['image_size_hw']))
                x0,y0,x1,y1=region;crop=np.ascontiguousarray(image[y0:y1,x0:x1])
                predictions[hashlib.sha256(crop.tobytes()).hexdigest(),crop.shape]=typed(raw,geometry)
            class FrozenPredictor:
                def predict(self,crop,*,postprocess,heads):
                    assert postprocess=='hybrid' and heads==('kp','line')
                    return predictions[hashlib.sha256(crop.tobytes()).hexdigest(),crop.shape]
            expected=json.loads((OUT/'evaluation'/record['key']/'evaluation.json').read_text())['conditions']
            for mode,method in (('fixed_support','model'),('fixed_support_rgb','rgb_paint')):
                chosen=None;status='rejected'
                try:
                    result=select_court_region(FrozenPredictor(),image,CourtRegionSearchConfig(enabled=True,refinement_method=method))
                    chosen=list(result.region);status='accepted'
                except CourtRegionUnavailable:pass  # expected declared outcome; never substitute a H
                agreed=(status=='accepted')==expected[mode]['accepted'] and (chosen is None or chosen==expected[mode]['region'])
                report.append({'key':record['key'],'condition':mode,'status':status,'region':chosen,'matches_offline':agreed})
                print(report[-1],flush=True)
    output='api-parity-accepted.json' if args.accepted_only else 'api-parity.json'
    (OUT/output).write_text(json.dumps({'cases':report,'seconds':time.monotonic()-start,'model_forward':'frozen decoded heads, no CUDA'},indent=2))
    assert all(r['matches_offline'] for r in report)


if __name__=='__main__':main()
