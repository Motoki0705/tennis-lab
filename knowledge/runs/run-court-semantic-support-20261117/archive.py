"""Version small summaries; inventory every large evaluation and review artifact."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

BASE=Path(__file__).resolve().parent
OUT=Path('outputs/tennis_scene/evaluate/court-semantic-support-20261117')
PREVIOUS=Path('outputs/tennis_scene/evaluate/court-fixed-support-rgb-20261117')


def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


def main():
 target=BASE/'report';target.mkdir(exist_ok=True)
 for name in ['report/summary.json','transitions.json','api-parity.json','mutations.json','regression/complete.json','meiji4/complete.json','input-integrity.json','verification.json','ambiguity-diagnosis.json']:
  destination=target/(name.replace('/','-'))
  shutil.copyfile(OUT/name,destination)
 inspection=json.loads((OUT/'rgb-measurement-inspection.json').read_text())
 compact=[{**case,'lines':[{k:v for k,v in line.items() if k!='points'} for line in case['lines']]} for case in inspection]
 (target/'rgb-measurement-inspection.json').write_text(json.dumps({'cases':compact,'full_measurements':str(OUT/'rgb-measurement-inspection.json')},indent=2))
 factorial=json.loads((OUT/'scene-gate-factorial.json').read_text());(target/'scene-gate-factorial.json').write_text(json.dumps({'counts':factorial['counts'],'scope':171,'details':str(OUT/'scene-gate-factorial.json')},indent=2))
 with (OUT/'report/cases.csv').open() as f:cases=list(csv.DictReader(f))
 # Compact one-row-per-image paired values. No diagnostic arrays enter Git.
 by_key={}
 aliases={'legacy':'L','fixed_support':'A0','fixed_support_rgb':'B0','new_support':'A1','new_rgb':'B1'}
 for row in cases:
  item=by_key.setdefault(row['key'],{'key':row['key']})
  for key in ('geometry_valid','accepted','mean_error_px','max_error_px'):
   value=row[key]
   if value in ('True','False'):value=int(value=='True')
   elif value:value=f'{float(value):.5g}'
   item[f'{aliases[row["condition"]]}_{key}']=value
 rows=list(by_key.values())
 for index in range(0,len(rows),600):
  with (target/f'cases-{index//600:02d}.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows[index:index+600])
 logs=BASE/'verification';logs.mkdir(exist_ok=True)
 for name in ['related-tests.log','refinement-tests.log','paint-alternatives-tests.log','semantic-alternative-test.log','precommit-code.log','integration-tests.log','mutations.log','final-tests.log']:
  shutil.copyfile(OUT/name,logs/name)
 records=json.loads((PREVIOUS/'manifest.json').read_text())['records']
 groups={'evaluation':[OUT/'evaluation'/r['key']/'evaluation.json' for r in records],
         'model_geometry':[OUT/('meiji4' if r['dataset']=='meiji_4cam' else 'regression')/r['key']/f'geometry_{i:02d}.pkl' for r in records for i in range(len(r['regions']))],
         'review':sorted((OUT/'review').glob('*.jpg')),
         'measured-points':[OUT/'rgb-measurement-inspection.json'],
         'review-extra':[OUT/'review/index.html',OUT/'review/manifest.json',*sorted((OUT/'review/measurements').glob('*.png'))]}
 inventories=OUT/'inventories';inventories.mkdir(exist_ok=True);report={}
 for group,files in groups.items():
  path=inventories/f'{group}.jsonl';total=0
  with path.open('w') as f:
   for file in files:
    size=file.stat().st_size;total+=size
    f.write(json.dumps({'path':str(file),'bytes':size,'sha256':digest(file)},separators=(',',':'))+'\n')
  report[group]={'files':len(files),'bytes':total,'inventory':str(path),'sha256':digest(path)}
 # Inputs are the same inventoried bytes from run73; retain their authority.
 report['frozen_inputs']={'inventory':'knowledge/runs/run-court-fixed-support-rgb-20261117/artifact-inventories.json',
       'source':str(PREVIOUS),'rgb_sha_checked_per_image':True,'persisted_LINE_regions':8530,'missing_LINE_files':0}
 (BASE/'artifact-inventories.json').write_text(json.dumps(report,indent=2))
 print({k:v.get('files') for k,v in report.items()})
if __name__=='__main__':main()
