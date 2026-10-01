import json,contextlib,time
from pathlib import Path
import torch
from src.tasks.plcs.training.association_lightning_module import PLCSAssociationLightningModule
from src.tasks.plcs.data.association_datamodule import PLCSAssociationDataModule
from src.tasks.plcs.training.reid_losses import reid_loss
checkpoint='/home/kamimura/projects/tennis-lab/outputs/plcs/train/fixed_track_reid_v1_s42/logs/version_0/checkpoints/last.ckpt'
torch.set_num_threads(4)
module=PLCSAssociationLightningModule.load_from_checkpoint(checkpoint,map_location='cpu',weights_only=False).eval()
cfg=module.config; cfg.data.num_workers=0
module_data=PLCSAssociationDataModule(cfg);module_data.setup()
batch=next(iter(module_data.val_dataloader()))
print('batch', {k:[str(v.dtype),list(v.shape),bool(torch.isfinite(v).all())] for k,v in batch.items()},flush=True)


m=PLCSAssociationLightningModule.load_from_checkpoint(checkpoint,map_location='cpu',weights_only=False).cuda()
m.model.compile(backend='inductor',mode='default',fullgraph=False,dynamic=False)
x={k:v.cuda() for k,v in batch.items()}
optimizer=torch.optim.AdamW(m.parameters(),lr=.0001)
for phase,grad,step in [('eval_initial',False,False),('train_nograd',False,False),('eval_after_forward',False,False),('train_grad',True,False),('eval_after_grad',False,False),('train_step',True,True),('eval_after_step',False,False)]:
    m.train(phase.startswith('train'))
    with torch.set_grad_enabled(grad),torch.autocast('cuda',dtype=torch.bfloat16):
        out=m.model_io.run(x)
        loss=reid_loss(out,x['track_person_id'],temperature=.1,margin=.5,player_weight=.1)['loss']
    print(phase,float(loss.detach().cpu()),flush=True)
    if grad:
        optimizer.zero_grad();loss.backward()
        if step:optimizer.step()
    def inspect(value):
        if isinstance(value,torch.Tensor):return dict(shape=list(value.shape),dtype=str(value.dtype),finite=bool(torch.isfinite(value).all()),pointer=value.data_ptr())
        if isinstance(value,dict):return {str(k):inspect(v) for k,v in value.items() if isinstance(v,(torch.Tensor,dict))}
        return type(value).__name__
    print('frequency',inspect(vars(m.model.frequency)),flush=True)
