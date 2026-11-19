"""Paired losses/gains and rejection causes; run only after the frozen evaluation."""
from collections import Counter
import json
from pathlib import Path
import numpy as np

OUT=Path('outputs/tennis_scene/evaluate/court-semantic-support-20261117')
PREVIOUS=Path('outputs/tennis_scene/evaluate/court-fixed-support-rgb-20261117')
OLD=Path('outputs/tennis_scene/evaluate/court-acceptance-implementation-20261116')


def main():
 records=json.loads((OUT/'manifest.json').read_text())['records'];rows={r['key']:json.loads((OUT/'evaluation'/r['key']/'evaluation.json').read_text()) for r in records}
 before=json.loads((OUT/'diagnosis.json').read_text());history=json.loads((PREVIOUS/'historical-transitions.json').read_text());report={}
 for dataset in ('tennis_court_detector_val','synthetic_court_test'):
  keys=before[dataset]['geometry_loss'];items=[]
  for key in keys:
   row=rows[key];new=row['conditions']['new_support'];diagnostic=row['candidates']['new_support'][0]['diagnostics']
   items.append({'key':key,'recovered':new['geometry_valid'],'status':new['status'],'old_mean_error_px':row['conditions']['legacy']['mean_error_px'],
    'new_mean_error_px':new['mean_error_px'],'solve_stops':dict(Counter(s['stop_reason'] for s in diagnostic['support_solves']))})
  old_losses=history[dataset]['losses'];old_gains=history[dataset]['gains']
  report[dataset]={'previous_a_losses':len(keys),'previous_a_losses_recovered':sum(x['recovered'] for x in items),
   'remaining_statuses':dict(Counter(x['status'] for x in items if not x['recovered'])),
   'historical_losses':len(old_losses),'historical_losses_recovered':sum(rows[x['key']]['conditions']['new_support']['geometry_valid'] for x in old_losses),
   'historical_gains':len(old_gains),'historical_gains_retained':sum(rows[x['key']]['conditions']['new_support']['geometry_valid'] for x in old_gains),
   'paired_previous_losses':items}
  subset=[row for row in rows.values() if row['dataset']==dataset]
  latest_losses=[r for r in subset if r['conditions']['legacy']['geometry_valid'] and not r['conditions']['new_support']['geometry_valid']]
  report[dataset]['new_a_losses']=[{'key':r['key'],'legacy_correct':r['conditions']['legacy']['correct'], 'legacy_mean_gt_px':r['conditions']['legacy']['mean_error_px'], 'previous_a_valid':r['conditions']['fixed_support']['geometry_valid'], 'new_status':r['conditions']['new_support']['status']} for r in latest_losses]
  report[dataset]['geometry_correct_counts']={m:sum(r['conditions'][m]['correct'] for r in subset) for m in ('legacy','fixed_support','fixed_support_rgb','new_support','new_rgb')}
  for new in ('new_support','new_rgb'):
   old='fixed_support' if new=='new_support' else 'fixed_support_rgb'
   report[dataset][new+'_vs_previous']={'geometry_lost':sum(r['conditions'][old]['geometry_valid'] and not r['conditions'][new]['geometry_valid'] for r in subset),
    'geometry_gained':sum(not r['conditions'][old]['geometry_valid'] and r['conditions'][new]['geometry_valid'] for r in subset),
    'acceptance_lost':sum(r['conditions'][old]['accepted'] and not r['conditions'][new]['accepted'] for r in subset),
    'acceptance_gained':sum(not r['conditions'][old]['accepted'] and r['conditions'][new]['accepted'] for r in subset)}
 new_old_gate=[]
 for r in records:
  if r['dataset']!='meiji_3cam':continue
  row=rows[r['key']];accepted=False
  for candidate in row['candidates']['new_support']:
   if candidate['points'] is None:continue
   p=np.asarray(candidate['points']);region=candidate['region'];raw=np.load(OLD/'raw'/r['key']/f'raw_{candidate["region_index"]:02d}.npz')
   left,top,right,bottom=r['content_region'];inside=((p[:,0]>=left)&(p[:,0]<right)&(p[:,1]>=top)&(p[:,1]<bottom)).all()
   inliers=raw['valid']&(np.linalg.norm(p-np.array(region[:2])-raw['raw_xy'],axis=1)<=.005*np.hypot(*r['source_size_hw']))
   # Existing geometry-valid H already has a convex noncollapsed polygon; use
   # the same explicit image-intersection helper as the original comparison.
   from evaluate import prior
   accepted|=bool(inside and inliers.sum()>=8 and prior.area_ok(p,r))
  new_old_gate.append({'key':r['key'],'old_gate':accepted,'new_gate':row['conditions']['new_support']['accepted']})
 report['meiji3_new_a_gate_factorial']={'old_gate':sum(x['old_gate'] for x in new_old_gate),'new_gate':sum(x['new_gate'] for x in new_old_gate),'cases':new_old_gate}
 report['meiji4']={}
 for camera in ('cam0','cam1','cam2','cam3'):
  subset=[r for r in rows.values() if r['dataset']=='meiji_4cam' and r['camera']==camera];false=[];reject=Counter();lost=Counter();retained=Counter()
  for row in subset:
   c=row['conditions']['new_rgb'];p=c['quality']['rgb_paint']
   if c['accepted'] and not c['correct']:false.append({'key':row['key'],'region_index':c['region_index'],'mean_error_px':c['mean_error_px'],'max_error_px':c['max_error_px']})
   if c['correct'] and not c['accepted']:reject.update(c['quality']['reasons']);lost.update(p.get('lost_initial_lines',[]));retained.update(p.get('shifted_confirmed_lines',[]))
  report['meiji4'][camera]={'false_accepts':false,'correct_rejected_reasons':dict(reject),'confirmed_lines_lost':dict(lost),'confirmed_lines_shifted':dict(retained)}
 (OUT/'transitions.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
