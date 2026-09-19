"""Evaluate validation-selected PLCS on original synthetic test split, CPU only."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from src.tasks.plcs.training.composition import build_plcs_datamodule,build_plcs_lightning_module
from src.utils.paths import PROJECT_ROOT
import src.utils.hydra  # Register repository Hydra path resolver

OUT=Path(__file__).resolve().parent
MAIN=Path('/home/kamimura/projects/tennis-lab')
JOB='1789725679558922390_14156_slcs-plcs-meiji-foot-e60-resume-v3'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU-only evaluation required'
os.environ.pop('TENNIS_REPRO_DIR',None)
with initialize_config_dir(config_dir=str(PROJECT_ROOT/'src/tasks/plcs/configs'),version_base='1.3'):
 cfg=compose(config_name='train_meiji_foot_real_rgb',overrides=['data.num_workers=0','run.init_weights=null','run.resume=null','run.output_dir=plcs/analyze/meiji_foot_final/s42-001/selected_test'])
pl.seed_everything(int(cfg.run.seed),workers=True)
dm=build_plcs_datamodule(cfg);module=build_plcs_lightning_module(cfg)
receipt=json.loads((OUT/'selection.json').read_text())
p=Path(receipt['destination'])
c=torch.load(p,map_location='cpu',weights_only=False)
module.on_load_checkpoint(c);module.load_state_dict(c['state_dict'],strict=True);del c
output=OUT/'selected_test';output.mkdir(exist_ok=False)
OmegaConf.save(cfg,output/'config.yaml',resolve=True)
trainer=pl.Trainer(accelerator='cpu',devices=1,precision='32-true',logger=False,enable_checkpointing=False,enable_progress_bar=False,default_root_dir=str(output))
report=trainer.test(module,datamodule=dm)[0]
terminal=MAIN/'.training_queue/repro'/JOB/'predictions/pred_test.npz'
selected=output/'predictions/pred_test.npz'
checks={}
with np.load(terminal,allow_pickle=False) as a,np.load(selected,allow_pickle=False) as b:
 for k in a.files:
  if k=='scene_ids' or 'target' in k or 'mask' in k:
   checks[k]={'terminal_shape':list(a[k].shape),'selected_shape':list(b[k].shape),'exact_match':bool(k in b.files and np.array_equal(a[k],b[k],equal_nan=True)) if a[k].dtype.kind not in 'US' else bool(k in b.files and np.array_equal(a[k],b[k]))}
result=dict(metrics=report,checkpoint=str(p),checkpoint_sha256=sha(p),precision='32-true',terminal_precision='bf16-mixed',bundle_alignment=checks,terminal_bundle_sha256=sha(terminal),selected_bundle_sha256=sha(selected),interpretation='Synthetic source-motion-disjoint target agreement; not independent measured 3D accuracy. Test not used for checkpoint selection.')
(output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
