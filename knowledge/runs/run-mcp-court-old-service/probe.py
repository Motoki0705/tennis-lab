import json
from pathlib import Path
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.training.lightning_module import CourtDetectionLightningModule

pl.seed_everything(42, workers=True)
torch.set_num_threads(2)
with initialize_config_dir(config_dir=str(Path('src/tasks/court_detection/configs').resolve()), version_base='1.3'):
    cfg=compose(config_name='train',overrides=['data.batch_size=1','data.num_workers=0','training.compile.enabled=false','training.qualitative_logging.enabled=false','training.trainer.precision=32-true'])
data=CourtDetectionDataModule(cfg)
module=CourtDetectionLightningModule(cfg,target_bundle=data.target_bundle_spec)
module.steps_per_epoch=3
class Verify(pl.Callback):
    def on_train_start(self,trainer,module):
        self.name,self.parameter=next((n,p) for n,p in module.named_parameters() if p.requires_grad)
        self.before=self.parameter.detach().clone()
        self.losses=[]
        self.gradient_steps=0
    def on_after_backward(self,trainer,module):
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        self.gradient_steps+=1
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        loss=outputs['loss']
        assert torch.isfinite(loss)
        self.losses.append(float(loss))
    def on_train_end(self,trainer,module):
        assert self.gradient_steps==3
        assert not torch.equal(self.before,self.parameter)
        print('COURT_SMOKE_RESULT '+json.dumps({'steps':trainer.global_step,'losses':self.losses,'gradient_steps':self.gradient_steps,'updated_parameter':self.name,'data':'court','encoder_weights':cfg.model.encoder.checkpoint_path}))
trainer=pl.Trainer(accelerator='gpu',devices=[0],precision='32-true',fast_dev_run=3,logger=False,enable_checkpointing=False,enable_model_summary=False,callbacks=[Verify()],default_root_dir='/tmp/mcp-court-smoke',enable_progress_bar=False)
trainer.fit(module,datamodule=data)
