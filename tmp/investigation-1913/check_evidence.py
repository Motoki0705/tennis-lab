import hashlib,json
from pathlib import Path
r=Path('tmp/investigation-1913');idx=json.loads((r/'evidence/index.json').read_text())['files'];by={x['path']:x for x in idx};checks=[]
for x in idx:
 if not x['path'].endswith('/_SNAPSHOT.json'):continue
 marker=json.loads((r/x['local_path']).read_text());root=x['path'].rsplit('/',1)[0]
 for n,d in marker['files'].items():
  if root+'/'+n not in by:continue
  y=by[root+'/'+n];ok=y['sha256']==d['sha256'] and y['size']==d['size_bytes'] and y['md5']==d['md5'];assert ok,(root,n)
  checks.append(dict(path=y['path'],drive_id=y['drive_id'],sha256=y['sha256'],marker=x['path'],match=ok))
(r/'analysis/snapshot-hash-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print('snapshot checks',len(checks),'ALL PASS')
cap=[];controls=[]
for x in idx:
 if '/validation-update-' not in x['path']:continue
 d=json.loads((r/x['local_path']).read_text());m=d['metrics'];row={k:x[k] for k in ['path','drive_id','size','sha256']}
 row.update(split=d['split'],identities=d['identities'],manifest_sha256=d['manifest_sha256'],metrics={k:m[k] for k in ['mean_error_px','median_error_px','p95_error_px','within_distance_rate','distance_threshold_source_px','support']},model_config=d['model_config'],artifact_sha256=d['artifact_sha256'])
 dirname=x['path'].rsplit('/',1)[0];runrow=by.get(dirname+'/run.json');markerrow=by.get(dirname+'/_SNAPSHOT.json')
 if runrow:
  run=json.loads((r/runrow['local_path']).read_text());row['recipe']={k:v for k,v in run['recipe'].items() if k in ['budget','architecture','phase','initialization','model','runtime','lineage']};row['code']=run['code']
 if markerrow:
  marker=json.loads((r/markerrow['local_path']).read_text());ck='checkpoint-update-'+x['path'].split('validation-update-')[-1].replace('.json','.pt');row['checkpoint_path']=dirname+'/'+ck;row['checkpoint_record']=marker['files'][ck]
  listings=[json.loads(p.read_text()) for p in (r/'evidence/listings').glob('*.json')];lf=next((q for q in listings if q['path']==dirname),None)
  if lf:row['checkpoint_drive_id']=next((f['id'] for f in lf['files'] if f['title']==ck),None)
 if '/capacity_scaling/' in x['path']:cap.append(row)
 else:controls.append(row)
for name,rows in [('capacity-metrics',cap),('query-control-metrics',controls)]:
 (r/f'analysis/{name}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');print(name,len(rows))
base=next(x for x in controls if '/sequential_length/' in x['path'])
for x in cap+controls:
 assert x['identities']==base['identities'],x['path']
 print(x['path'].split('/')[-3],round(x['metrics']['mean_error_px'],6),'id identical',x['recipe']['budget']['seed'] if 'recipe' in x else None)
