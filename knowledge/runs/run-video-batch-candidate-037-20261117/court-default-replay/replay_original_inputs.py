"""Current default postprocess on saved neural inputs; no inference/cache bypass."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path.cwd()
OUT=REPO/'outputs/tennis_scene/evaluate/video-batch-current-main-court-20261117'
ROOT=Path('/home/kamimura/projects/tennis-lab/.worktree-outputs/court-acceptance-postprocess/tennis_scene/evaluate')
SAVED=ROOT/'court-fixed-support-rgb-20261117'
RAW=ROOT/'court-acceptance-implementation-20261116'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
module_path=REPO/'knowledge/runs/run-court-legacy-default-20261117/replay.py'
spec=importlib.util.spec_from_file_location('legacy_replay_for_c1',module_path)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
module.OUT=OUT
module.SAVED=SAVED
module.RAW=RAW
manifest=json.loads((SAVED/'manifest.json').read_text())
records=[r for r in manifest['records'] if r['dataset']=='meiji_4cam' and r['clip_id'] in {f'video_000/clip_{i:03d}' for i in range(3)}]
assert len(records)==12
paths=[SAVED/'manifest.json',module_path]+list((REPO/'src/tasks/court_detection/geometry').glob('*.py'))+list((REPO/'src/tasks/court_detection/inference').glob('*.py'))+[REPO/'src/tennis_scene/configs/pipeline.yaml']
for r in records:
 paths.append(SAVED/'rgb'/r['key']/'frame.png')
 for i in range(len(r['regions'])):
  paths.extend([RAW/r['old_raw_group']/r['key']/f'raw_{i:02d}.npz',SAVED/'line-evidence'/r['key']/f'evidence_{i:02d}.npz'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
frozen={str(p):sha(p) for p in paths}
(OUT/'inputs-and-source.json').write_text(json.dumps({'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'records':records,'file_sha256':frozen,'purpose':'CPU default-postprocess replay; saved neural outputs, no GPU forward, not baseline'},indent=2)+'\n')
start=time.perf_counter();rows=[]
for record in records:
 result=module.process(record)
 result.update(clip_id=record['clip_id'],camera_id=record['camera'])
 rows.append(result)
 print(json.dumps(result),flush=True)
assert frozen=={str(p):sha(p) for p in paths},'Inputs/source changed'
assert not module.torch.cuda.is_initialized()
result={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'seconds':time.perf_counter()-start,'all_exact':all(r['all_exact'] for r in rows),'cuda_initialized':False,'inputs_unchanged':True,'rows':rows,'all_cameras_accepted':{c:all(r['accepted'] for r in rows if r['clip_id']==c) for c in sorted({r['clip_id'] for r in rows})}}
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
