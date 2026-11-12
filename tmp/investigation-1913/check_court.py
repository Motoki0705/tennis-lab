import json,hashlib,csv,statistics
from pathlib import Path
r=Path('tmp/investigation-1913');idx=json.load(open(r/'evidence/index.json'))['files'];by={x['path']:x for x in idx};sources=json.load(open(r/'prs/1831/knowledge/runs/run-court-scene-pair-comparison-s42/sources.json'));rows={};checks=[]
keys=['kp_mean_distance_px','kp_median_distance_px','seg_miou','line_dice','semantic_line_miou']
for name,pin in sources['conditions'].items():
 x=by[pin['evaluation']];b=(r/x['local_path']).read_bytes();d=json.loads(b);mrow=by[pin['original_marker']];marker=json.loads((r/mrow['local_path']).read_text())
 assert x['sha256']==pin['evaluation_sha256']==marker['result']['evaluation_sha256']==marker['result']['evaluation_files'][pin['evaluation']]
 assert mrow['sha256']==pin['original_marker_sha256']
 assert d['checkpoint_sha256']==pin['checkpoint_sha256']==marker['result']['checkpoint_sha256']
 assert d['global_step']==33160==marker['result']['global_step'] and d['config']['run']['seed']==42 and d['config']['data']['batch_size']==8
 split=d['splits'];ids={}
 for s,count,sha in [('tennis_court_detector-val',2211,sources['common']['real_ids_sha256']),('synthetic_court-val',2439,sources['common']['synthetic_ids_sha256'])]:
  v=split[s];assert v['count']==count and len(v['sample_ids'])==count
  assert hashlib.sha256(json.dumps(v['sample_ids'],separators=(',',':')).encode()).hexdigest()==v['ordered_sample_ids_sha256']==sha
  ids[s]=sha
 real={k:split['tennis_court_detector-val']['metrics'][k] for k in keys};syn={scene:{k:v['metrics'][k] for k in keys} for scene,v in split['synthetic_court-val']['scenes'].items()}
 rows['P01' if name=='scene-2' else name]=dict(condition=name,real=real,synthetic=syn,seed=42,updates=33160,checkpoint=pin['checkpoint'],checkpoint_sha256=pin['checkpoint_sha256'],evaluation_path=x['path'],evaluation_drive_id=x['drive_id'],evaluation_sha256=x['sha256'],marker_drive_id=mrow['drive_id'],marker_sha256=mrow['sha256'],identities=ids)
 checks.append(dict(condition=name,evaluation_match=True,marker_match=True,checkpoint_identity_match=True,ids_and_order_match=True))
contrast={}
for k in keys:
 same=statistics.mean(rows[n]['real'][k] for n in ['P01','P23']);cross=statistics.mean(rows[n]['real'][k] for n in ['P02','P03','P12','P13']);contrast[k]=dict(same_group_mean=same,cross_group_mean=cross,delta=same-cross,additivity=rows['P02']['real'][k]+rows['P13']['real'][k]-rows['P03']['real'][k]-rows['P12']['real'][k])
(r/'analysis/court-recomputed.json').write_text(json.dumps(dict(rows=rows,contrasts=contrast,checks=checks),ensure_ascii=False,indent=2)+'\n')
with (r/'analysis/court-real-metrics.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['condition']+keys+['drive_id','sha256']);
 for n,v in rows.items():w.writerow([n]+[v['real'][k] for k in keys]+[v['evaluation_drive_id'],v['evaluation_sha256']])
print('9/9 evaluation hashes, markers, updates, seed, ID order: PASS');print(json.dumps(contrast,indent=2))
