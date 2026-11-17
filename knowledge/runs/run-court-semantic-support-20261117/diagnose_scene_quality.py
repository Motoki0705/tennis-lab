"""Factor model fit vs acceptance on the existing meiji3 scene, CPU only."""
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np

s=importlib.util.spec_from_file_location('replay',Path(__file__).with_name('evaluate.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
from src.tasks.court_detection.geometry.homography import court_template_xy
from src.tasks.court_detection.geometry.line_evidence import LineEvidence
from src.tasks.court_detection.geometry.quality import assess_court_quality,CourtQualityConfig


def process(record):
 cv2.setNumThreads(1)
 row=json.loads((m.PREVIOUS/'evaluation'/record['key']/'evaluation.json').read_text());candidates={}
 for mode in ('legacy','fixed_support'):
  values=[]
  for value in row['candidates'][mode]:
   index=value['region_index'];region=value['region'];raw=dict(np.load(m.prior.OLD/'raw'/record['key']/f'raw_{index:02d}.npz'))
   points=None if value['points'] is None else np.asarray(value['points']);quality=None;old_accept=False
   if points is not None:
    local=points-np.array(region[:2]);matrix=cv2.findHomography(court_template_xy().astype(float),local,method=0)[0]
    evidence=m.prior.prepare(raw,m.PREVIOUS/'line-evidence'/record['key']/f'evidence_{index:02d}.npz')
    with patch.object(LineEvidence,'from_probability',return_value=evidence):
     quality=assess_court_quality(matrix,raw['raw_xy'],raw['valid']&np.asarray(value['selected'],bool),raw['line_probability'],
      image_size_hw=tuple(raw['image_size_hw']),evaluation_size_hw=tuple(record['source_size_hw']),config=CourtQualityConfig()).diagnostics()
    left,top,right,bottom=record['content_region'];inside=((points[:,0]>=left)&(points[:,0]<right)&(points[:,1]>=top)&(points[:,1]<bottom)).all()
    support=raw['valid']&(np.linalg.norm(local-raw['raw_xy'],axis=1)<=.005*np.hypot(*record['source_size_hw']))
    old_accept=bool(inside and support.sum()>=8 and m.prior.area_ok(points,record))
   values.append({'region_index':index,'old_gate':old_accept,'new_gate':bool(quality and quality['accepted'] and m.prior.area_ok(points,record)),
      'quality':quality,'gt_mean_px':value['mean_error_px'],'gt_max_px':value['max_error_px']})
  candidates[mode]=values
 return {'key':record['key'],'candidates':candidates,'accepted':{mode:{gate:any(c[gate] for c in values) for gate in ['old_gate','new_gate']} for mode,values in candidates.items()}}

if __name__=='__main__':
 records=[r for r in json.loads((m.PREVIOUS/'manifest.json').read_text())['records'] if r['dataset']=='meiji_3cam']
 with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(process,records))
 counts={mode:{gate:sum(r['accepted'][mode][gate] for r in rows) for gate in ['old_gate','new_gate']} for mode in ['legacy','fixed_support']}
 (m.OUT/'scene-gate-factorial.json').write_text(json.dumps({'counts':counts,'rows':rows}));print(counts)
