"""Replay frozen RGB measurements for false accepts, with native inspection crops."""
import json
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from evaluate import OUT,PREVIOUS,prior
from render import draw,text
from src.tasks.court_detection.geometry import paint_refinement as paint
from src.tasks.court_detection.geometry.paint_observations import PAINT_NAMES,image_lines
from src.tasks.court_detection.geometry.homography import court_template_xy


def main():
 cv2.setNumThreads(1);refs=json.loads(prior.REF.read_text());references={r['camera']:np.asarray(r['matrix_court_metres_to_image_pixels']) for r in refs['cameras']}
 records=[r for r in json.loads((OUT/'manifest.json').read_text())['records'] if r['dataset']=='meiji_4cam'];chosen={}
 for record in records:
  row=json.loads((OUT/'evaluation'/record['key']/'evaluation.json').read_text());value=row['conditions']['new_rgb']
  if value['accepted'] and not value['correct']:chosen.setdefault(record['camera'],(record,row,value))
 summary=[]
 for camera,(record,row,value) in chosen.items():
  candidate=next(c for c in row['candidates']['new_rgb'] if c['region_index']==value['region_index']);diag=candidate['diagnostics']
  initial=np.asarray(diag['model_homography']);bgr=cv2.imread(str(PREVIOUS/'rgb'/record['key']/'frame.png'));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);captured=[];extract=paint.extract_paint_lines
  def capture(*args,**kwargs):
   lines=extract(*args,**kwargs);captured.append(lines);return lines
  with patch.object(paint,'extract_paint_lines',side_effect=capture):result=paint.refine_paint(rgb,initial)
  assert result.quality.accepted and len(captured)==3
  assert abs(result.quality.score-value['quality']['score'])<1e-8
  np.testing.assert_allclose(result.fitted_matrix,diag['rgb_fitted_homography'],atol=1e-9,rtol=1e-9)
  reference_lines=image_lines(references[camera]);reports=[]
  for line in captured[-1]:
   if line.status!='ok':continue
   error=np.abs(np.c_[line.points,np.ones(len(line.points))]@reference_lines[line.index])
   reports.append({'line':PAINT_NAMES[line.index],'index':line.index,'locations':len(line.points),'coverage':line.coverage,
     'distance_to_reference_line_median_px':float(np.median(error)),'distance_to_reference_line_p95_px':float(np.percentile(error,95)),
     'points':line.points.tolist()})
  worst=max(reports,key=lambda x:x['distance_to_reference_line_median_px']);points=np.asarray(worst['points'])
  minxy=np.maximum(np.floor(points.min(axis=0)).astype(int)-[30,45],0);maxxy=np.minimum(np.ceil(points.max(axis=0)).astype(int)+[30,45],bgr.shape[1::-1])
  x0,y0=minxy;x1,y1=maxxy;annotated=bgr.copy();truth=cv2.perspectiveTransform(court_template_xy()[None].astype(float),references[camera])[0]
  draw(annotated,truth,(255,0,255),1);draw(annotated,result.projected,(40,220,40),1)
  for point in points:cv2.circle(annotated,tuple(np.rint(point).astype(int)),3,(255,255,0),-1,cv2.LINE_AA)
  top=bgr[y0:y1,x0:x1];bottom=annotated[y0:y1,x0:x1];banner=np.full((45,top.shape[1],3),24,np.uint8)
  text(banner,f'{camera} {record["clip_id"]} / {worst["line"]}',27,scale=.6)
  panel=np.vstack([banner,top,bottom]);folder=OUT/'review/measurements';folder.mkdir(exist_ok=True)
  filename=f'{camera}_{record["clip_id"].split("/")[-1]}_{worst["line"]}.png';cv2.imwrite(str(folder/filename),panel,[cv2.IMWRITE_PNG_COMPRESSION,9])
  summary.append({'key':record['key'],'native_crop':str(folder/filename),'crop_xyxy':[int(x0),int(y0),int(x1),int(y1)],'quality_replay_matches':True,
    'mean_error_px':value['mean_error_px'],'max_error_px':value['max_error_px'],'lines':reports,'worst_reference_line':worst['line']})
 (OUT/'rgb-measurement-inspection.json').write_text(json.dumps(summary,indent=2));print([{k:v for k,v in s.items() if k!='lines'} for s in summary])
if __name__=='__main__':main()
