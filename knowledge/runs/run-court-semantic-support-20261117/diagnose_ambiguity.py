"""Diagnose remaining synthetic ambiguity using saved solves; never change results."""
import json
from pathlib import Path
import cv2
import numpy as np
from evaluate import OUT,PREVIOUS,prior
from src.tasks.court_detection.geometry.hybrid_homography import HybridHomographyConfig,project_candidate,selected_kp_residuals
from src.tasks.court_detection.geometry.homography import court_template_xy,COURT_HOMOGRAPHY_EDGES
from src.tasks.court_detection.geometry.line_evidence import line_residuals
from src.tasks.court_detection.geometry.observations import keypoint_support


def main():
 cv2.setNumThreads(1);world=court_template_xy().astype(float);edges=np.array(COURT_HOMOGRAPHY_EDGES);config=HybridHomographyConfig();reports=[]
 transverse=np.abs(world[edges[:,1],0]-world[edges[:,0],0])>np.abs(world[edges[:,1],1]-world[edges[:,0],1])
 for path in sorted((OUT/'regression/synthetic_court_test').rglob('evaluation.json')):
  row=json.loads(path.read_text());value=row['conditions']['new_support'];legacy=row['conditions']['legacy']
  if not legacy['geometry_valid'] or value['status']!='ambiguous_candidates':continue
  raw=dict(np.load(prior.OLD/'test_raw'/row['key']/'raw_00.npz'));observed=raw['raw_xy'].astype(float);observed[~raw['valid']]=np.nan
  size=tuple(map(int,raw['image_size_hw']));diagonal=float(np.hypot(*size));threshold=config.threshold_diagonal_ratio*diagonal
  support=keypoint_support(observed,raw['scores'],image_size_hw=size,native_size_hw=tuple(raw['line_probability'].shape),
    min_score=config.min_score,boundary_margin_cells=config.boundary_margin_cells,duplicate_distance_cells=config.duplicate_distance_cells)
  evidence=prior.prepare(raw,PREVIOUS/'line-evidence'/row['key']/'evidence_00.npz');hypotheses=[]
  for solve in row['candidates']['new_support'][0]['diagnostics']['support_solves']:
   if solve['stop_reason']!='converged':continue
   points=project_candidate(np.asarray(solve['fitted_homography']),world);assert points is not None
   selected=np.zeros(14,bool);selected[solve['selected_indices']]=True
   residual,line=line_residuals(points,edges,evidence,threshold,samples_per_line=config.samples_per_line)
   kp=selected_kp_residuals(points,observed,raw['scores'].astype(float),selected,threshold,support)
   score=float(residual@residual+config.kp_weight*(kp@kp));supported=np.asarray(line['segment_support'])>=.5
   admissible=bool(min(line['forward_support'],line['reverse_support'])>=config.min_line_support and line['supported_segments']>=4 and
                   sum(supported&transverse)>=2 and sum(supported&~transverse)>=2)
   hypotheses.append({'candidate_id':solve['candidate_id'],'score':score,'points':points,'passes_line_gate':admissible,'line':line})
  hypotheses.sort(key=lambda h:h['score']);best=hypotheses[0];assert abs(best['score']-value['model_score'])<1e-10
  alternatives=[h for h in hypotheses[1:] if np.sqrt(np.mean((h['points']-best['points'])**2))>config.ambiguity_displacement_ratio*diagonal and h['score']-best['score']<config.ambiguity_gap]
  assert alternatives
  reports.append({'key':row['key'],'legacy_correct':legacy['correct'],'best_passes_line_gate':best['passes_line_gate'],
   'competitive_alternatives':len(alternatives),'admissible_alternatives':sum(h['passes_line_gate'] for h in alternatives),
   'all_competitors_fail_line_gate':not any(h['passes_line_gate'] for h in alternatives),
   'alternatives':[{k:v for k,v in h.items() if k!='points'} for h in alternatives]})
 result={'cases':len(reports),'only_inadmissible_competitors':sum(r['all_competitors_fail_line_gate'] for r in reports),
         'old_correct_and_only_inadmissible':sum(r['legacy_correct'] and r['all_competitors_fail_line_gate'] for r in reports),
         'note':'Post-freeze diagnosis, no implementation or acceptance change. Valid alternate courts still require disambiguation.','records':reports}
 (OUT/'ambiguity-diagnosis.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='records'})
if __name__=='__main__':main()
