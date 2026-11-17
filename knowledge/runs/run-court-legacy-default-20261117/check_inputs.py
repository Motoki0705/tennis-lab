"""Read-only verification of all inherited frozen input bytes against run73."""
import hashlib
import json
from pathlib import Path
import time

BASE=Path(__file__).resolve().parent
OUT=Path('outputs/tennis_scene/evaluate/court-legacy-default-20261117')

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 start=time.monotonic();old=json.loads(Path('knowledge/runs/run-court-fixed-support-rgb-20261117/artifact-inventories.json').read_text());report={}
 for group in ['input-raw','input-val_raw','input-test_raw','line-evidence','rgb']:
  data=old[group];index=Path(data['inventory_path']);assert sha(index)==data['jsonl_sha256'];count=0;size=0
  for line in index.read_text().splitlines():
   value=json.loads(line);p=Path(data['root'])/value['path'];assert p.stat().st_size==value['bytes'] and sha(p)==value['sha256'],str(p)
   count+=1;size+=value['bytes']
  assert count==data['files'] and size==data['bytes'];report[group]={'files':count,'bytes':size,'all_hashes_match':True};print(group,count,flush=True)
 report['seconds']=time.monotonic()-start
 (OUT/'input-integrity.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
