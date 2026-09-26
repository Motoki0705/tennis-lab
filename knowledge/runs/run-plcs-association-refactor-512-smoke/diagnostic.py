import json,sys,time
from pathlib import Path
import torch
from hydra import initialize_config_dir,compose
from src.utils.hydra import hydra_main
from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.plcs.training.runner import PLCSTrainingRunner
from src.tasks.blcs.training.runner import BLCSTrainingRunner
from src.tasks.plcs.data.association_datamodule import PLCSAssociationDataModule
from src.tasks.blcs.data.association_datamodule import BLCSAssociationDataModule
from src.tasks.plcs.inference.association_predictor import PLCSAssociationPredictor
from src.tasks.blcs.inference.association_predictor import BLCSAssociationPredictor

task=sys.argv[1]
DM=PLCSAssociationDataModule if task=='plcs' else BLCSAssociationDataModule
Runner=PLCSTrainingRunner if task=='plcs' else BLCSTrainingRunner
Predictor=PLCSAssociationPredictor if task=='plcs' else BLCSAssociationPredictor
original=DM.setup

def small_setup(self,stage=None):
 original(self,stage)
 for name in ['train_dataset','val_dataset','test_dataset']:
  dataset=getattr(self,name)
  if dataset is not None:dataset.scenes=dataset.scenes[:8]
DM.setup=small_setup
with initialize_config_dir(config_dir=str(Path(f'src/tasks/{task}/configs').resolve()),version_base='1.3'):
 config=compose(config_name='train_association',overrides=[
 'paths.project_root=/home/kamimura/projects/tennis-lab','paths.data_root=/home/kamimura/projects/tennis-lab/data','paths.output_root=/home/kamimura/projects/tennis-lab/outputs',
 f'run.output_dir={task}/train/view_association_refactor/gpu_smoke_512_v1',
 'training.trainer.max_epochs=1','training.warmup_steps=0','training.trainer.accumulate_grad_batches=1','training.trainer.enable_progress_bar=false','training.trainer.log_every_n_steps=1',
 'data.num_workers=0','data.batch_size=2','training.compile.enabled=true'])
start=time.perf_counter();Runner().run(config);elapsed=time.perf_counter()-start
root=Path('/home/kamimura/projects/tennis-lab/outputs')/task/'train/view_association_refactor/gpu_smoke_512_v1'
checkpoints=list(root.glob('logs/version_*/checkpoints/last.ckpt'));assert len(checkpoints)==1,checkpoints
predictor=Predictor.load(checkpoints[0],device='cpu')
dm=DM(config);dm.setup('validate') if False else dm.setup('fit')
batch=dm.collate_fn([dm.val_dataset[0]])
from src.tasks.base.model_io.association_contracts import INPUT_KEYS
pred=predictor.predict({k:batch[k] for k in INPUT_KEYS})
assert pred['side_logits'].shape==(1,5)
assert all(torch.isfinite(x).all() for x in pred.values())
report={'task':task,'elapsed_seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'checkpoint':str(checkpoints[0]),'compile':True,'batch':2,'width':512,'stages':12,'heads':8,'output_shapes':{k:list(v.shape) for k,v in pred.items()}}
(root/'smoke_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
