"""CPU frozen-head public region selection, independent of the scoring harness."""
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
import cv2
import numpy as np
from evaluate import OUT,PREVIOUS,prior,estimate_hybrid_homography,LineEvidence,court_template_xy,COURT_HOMOGRAPHY_EDGES
from src.tasks.court_detection.inference.regions import CourtRegionSearchConfig,CourtRegionUnavailable,region_proposals,select_court_region


def main():
 cv2.setNumThreads(1)
 records=[r for r in json.loads((PREVIOUS/'manifest.json').read_text())['records'] if r['dataset']=='meiji_3cam' and r['clip_id']=='video_000/clip_000']
 report=[];start=time.monotonic()
 for record in records:
  image=cv2.cvtColor(cv2.imread(str(PREVIOUS/'rgb'/record['key']/'frame.png')),cv2.COLOR_BGR2RGB)
  assert [list(r) for r in region_proposals(image)]==record['regions']
  prepared={};raws={};predictions={}
  for index,region in enumerate(record['regions']):
   raw=dict(np.load(prior.OLD/record['old_raw_group']/record['key']/f'raw_{index:02d}.npz'))
   key=hashlib.sha256(raw['line_probability'].tobytes()).hexdigest(),tuple(raw['image_size_hw'])
   path=PREVIOUS/'line-evidence'/record['key']/f'evidence_{index:02d}.npz'
   assert path.is_file();prepared[key]=prior.prepare(raw,path);raws[tuple(region)]=raw
  def evidence(probability,size,**_):
   value=prepared[hashlib.sha256(probability.tobytes()).hexdigest(),tuple(size)]
   if value is None:raise ValueError('insufficient_or_diffuse_line_evidence')
   return value
  with patch.object(LineEvidence,'from_probability',side_effect=evidence):
   for region,raw in raws.items():
    observed=raw['raw_xy'].astype(float);observed[~raw['valid']]=np.nan
    geom=estimate_hybrid_homography(court_template_xy().astype(float),observed,raw['scores'],raw['line_probability'],edges=np.array(COURT_HOMOGRAPHY_EDGES),image_size_hw=tuple(raw['image_size_hw']))
    x0,y0,x1,y1=region;crop=np.ascontiguousarray(image[y0:y1,x0:x1]);predictions[hashlib.sha256(crop.tobytes()).hexdigest(),crop.shape]=prior.typed(raw,geom)
   class FrozenPredictor:
    def predict(self,crop,*,postprocess,heads):
     assert postprocess=='hybrid' and heads==('kp','line')
     return predictions[hashlib.sha256(crop.tobytes()).hexdigest(),crop.shape]
   expected=json.loads((OUT/'regression'/record['key']/'evaluation.json').read_text())['conditions']
   for mode,method in [('new_support','model'),('new_rgb','rgb_paint')]:
    chosen=None;accepted=False
    try:
     selection=select_court_region(FrozenPredictor(),image,CourtRegionSearchConfig(enabled=True,refinement_method=method));chosen=list(selection.region);accepted=True
    except CourtRegionUnavailable:pass
    match=accepted==expected[mode]['accepted'] and (chosen is None or chosen==expected[mode]['region'])
    report.append({'key':record['key'],'mode':mode,'accepted':accepted,'region':chosen,'matches_offline':match})
 result={'cases':report,'seconds':time.monotonic()-start,'neural_forward':False,'CUDA_used':False}
 (OUT/'api-parity.json').write_text(json.dumps(result,indent=2));print(result);assert all(r['matches_offline'] for r in report)
if __name__=='__main__':main()
