"""Saved-output diagnostics only: line loss and historical gain/loss retention."""
from collections import Counter
import json
from pathlib import Path

OUT=Path('outputs/tennis_scene/evaluate/court-fixed-support-rgb-20261117')
OLD=Path('outputs/tennis_scene/evaluate/court-acceptance-implementation-20261116')


def main():
    records=json.loads((OUT/'manifest.json').read_text())['records'];transitions={};paint={}
    for dataset,prefix in [('tennis_court_detector_val','val'),('synthetic_court_test','test')]:
        gains=[];losses=[];statuses=Counter();stops=Counter();old_ok=0;previous_ok=0
        for record in records:
            if record['dataset']!=dataset:continue
            old=json.loads((OLD/(prefix+'_legacy')/record['key']/'geometry.json').read_text())['candidates'][0]
            previous=json.loads((OLD/(prefix+'_current')/record['key']/'geometry.json').read_text())['candidates'][0]
            old_valid=old['matrix'] is not None;previous_valid=previous['matrix'] is not None
            old_ok+=old_valid;previous_ok+=previous_valid
            row=json.loads((OUT/'evaluation'/record['key']/'evaluation.json').read_text());value=row['conditions']['fixed_support']
            if previous_valid and not old_valid:gains.append({'key':record['key'],'retained':value['geometry_valid'],'new_status':value['status']})
            if old_valid and not previous_valid:losses.append({'key':record['key'],'recovered':value['geometry_valid'],'new_status':value['status']})
            if row['conditions']['legacy']['geometry_valid'] and not value['geometry_valid']:
                statuses[value['status']]+=1
                diagnostic=row['candidates']['fixed_support'][0]['diagnostics'];stops.update(s['stop_reason'] for s in diagnostic['support_solves'])
        transitions[dataset]={'historical_legacy_success':old_ok,'historical_previous_success':previous_ok,
            'historical_gains':len(gains),'historical_gains_retained':sum(x['retained'] for x in gains),
            'historical_losses':len(losses),'historical_losses_recovered':sum(x['recovered'] for x in losses),
            'paired_lost_statuses':dict(statuses),'paired_lost_solver_stops':dict(stops),'gains':gains,'losses':losses}
    rows=[json.loads((OUT/'evaluation'/r['key']/'evaluation.json').read_text()) for r in records if r['dataset']=='meiji_4cam']
    for camera in ('cam0','cam1','cam2','cam3'):
        lost=Counter();gated=Counter();fn_lost=Counter();accepted=[]
        for row in rows:
            if row['camera']!=camera:continue
            value=row['conditions']['fixed_support_rgb'];quality=value['quality'];rgb=quality['rgb_paint']
            lost.update(rgb.get('lost_initial_lines',[]))
            if value['correct'] and not value['accepted']:
                gated.update(quality['reasons']);fn_lost.update(rgb.get('lost_initial_lines',[]))
            if value['accepted']:accepted.append(row['key'])
        paint[camera]={'lost_initial_lines':dict(lost),'correct_rejected_reasons':dict(gated),
            'correct_rejected_lost_lines':dict(fn_lost),'accepted':accepted}
    for filename,result in [('historical-transitions.json',transitions),('rgb-rejection-analysis.json',paint)]:
        (OUT/filename).write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
