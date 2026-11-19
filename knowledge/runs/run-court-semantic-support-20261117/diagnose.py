"""Saved run73 regressions; no meiji_4cam reads or neural execution."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

OLD = Path('outputs/tennis_scene/evaluate/court-acceptance-implementation-20261116')
PREVIOUS = Path('outputs/tennis_scene/evaluate/court-fixed-support-rgb-20261117')
OUT = Path('outputs/tennis_scene/evaluate/court-semantic-support-20261117')

def main():
    result={}
    for dataset in ('synthetic_court_test','tennis_court_detector_val','meiji_3cam','tennis_clip'):
        records=[json.loads(f.read_text()) for f in sorted((PREVIOUS/'evaluation'/dataset).rglob('evaluation.json'))]
        lost=[r for r in records if r['conditions']['legacy']['geometry_valid'] and not r['conditions']['fixed_support']['geometry_valid']]
        adoption=[r for r in records if r['conditions']['legacy']['accepted'] and not r['conditions']['fixed_support']['accepted']]
        entries=[]
        for row in lost:
            raw=np.load(OLD/'test_raw'/row['key']/'raw_00.npz')
            threshold=.005*np.hypot(*raw['image_size_hw'])
            traces=row['candidates']['fixed_support'][0]['diagnostics']['support_solves']
            entries.append({'key':row['key'],'status':row['conditions']['fixed_support']['status'],
                'old_mean_gt_px':row['conditions']['legacy']['mean_error_px'],
                'threshold_px':threshold,'solves':[{'stop':s['stop_reason'],'selected':s['selected_indices'],
                'remaining':int(sum(x<=threshold for x in s['selected_residuals_px']))} for s in traces if 'selected_residuals_px' in s]})
        result[dataset]={'geometry_loss':[r['key'] for r in lost], 'acceptance_loss':[r['key'] for r in adoption],
            'geometry_status':dict(Counter(r['conditions']['fixed_support']['status'] for r in lost)),
            'adoption_reasons':dict(Counter(x for r in adoption for x in r['conditions']['fixed_support']['quality']['reasons'])),
            'losses':entries}
    (OUT/'diagnosis.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
