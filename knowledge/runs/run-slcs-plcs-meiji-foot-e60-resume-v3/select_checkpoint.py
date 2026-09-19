"""CPU-only selection receipt; requires successful completed queue job."""
import hashlib
import json
import shutil
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

MAIN=Path('/home/kamimura/projects/tennis-lab')
JOB='1789725679558922390_14156_slcs-plcs-meiji-foot-e60-resume-v3'
OUT=Path(__file__).resolve().parent
LOGS=MAIN/'outputs/plcs/train/meiji_foot_real_rgb/s42-001/logs'
assert (MAIN/'.training_queue/done'/f'{JOB}.job').exists(), 'Queue job not successful yet'
metric='val/position_error_m'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 return h.hexdigest()
rows=[];hist=[]
for version in range(4):
 d=LOGS/f'version_{version}'
 a=EventAccumulator(str(d),size_guidance={'scalars':0});a.Reload()
 events={e.step:e.value for e in a.Scalars(metric)}
 for p in sorted((d/'checkpoints').glob('*.ckpt')):
  c=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
  callback=next(v for v in c['callbacks'].values() if v.get('monitor')==metric)
  tb=events.get(c['global_step']-1)
  key=next((k for k in callback['best_k_models'] if Path(k).name==f"plcs-epoch={c['epoch']}.ckpt"),None)
  callback_score=float(callback['best_k_models'][key]) if key else None
  if tb is not None and callback_score is not None: assert abs(tb-callback_score)<1e-7,(p,tb,callback_score)
  score=tb if tb is not None else callback_score
  assert score is not None, (p,'No exact epoch score available')
  rows.append(dict(path=str(p),sha256=sha(p),epoch_index=c['epoch'],global_step=c['global_step'],validation_position_error_m=score,tensorboard_value=tb,tensorboard_status='matched' if tb is not None else 'missing_after_environment_interruption; checkpoint_callback_used',version=version))
  if p.name=='last.ckpt':
   hist.append(dict(version=version,last_epoch_index=c['epoch'],global_step=c['global_step'],status='done' if version==3 else 'interrupted',reason=['WSL restart','native exit 139','WSL restart','completed with data.num_workers=0'][version]))
   if version==3: assert c['epoch']==59
selected=min(rows,key=lambda x:(x['validation_position_error_m'],x['epoch_index'],x['path']))
dest=MAIN/'ckpt/plcs/real-rgb-meiji-foot-e60-v1.ckpt'
assert not dest.exists(), 'Refusing to replace existing final checkpoint'
shutil.copy2(selected['path'],dest)
assert sha(dest)==selected['sha256']
terminal=next(r for r in rows if r['version']==3 and Path(r['path']).name=='last.ckpt')
receipt=dict(selection_metric=metric,mode='min',test_used_for_selection=False,scope='All saved checkpoints in versions 0–3; pruned epochs unavailable',selected=selected,destination=str(dest),candidates=rows,resume_history=hist,queue_job_id=JOB,queue_status='done',terminal_test=dict(checkpoint=terminal,equals_selected_weights=terminal['epoch_index']==selected['epoch_index'],source='BaseTrainingRunner.run trainer.test(lightning_module) evaluates terminal in-memory weights'),interpretation='Synthetic source-motion-disjoint target agreement, not independently measured 3D accuracy')
(OUT/'selection.json').write_text(json.dumps(receipt,indent=2)+'\n')
(dest.with_suffix('.metadata.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(selected,indent=2))
