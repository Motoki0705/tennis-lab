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

for compiled in (False,True):
    m=PLCSAssociationLightningModule.load_from_checkpoint(checkpoint,map_location='cpu',weights_only=False).to('cuda')
    if compiled:m.model.compile(backend='inductor',mode='default',fullgraph=False,dynamic=False)
    x={k:v.cuda() for k,v in batch.items()}
    optimizer=torch.optim.AdamW(m.parameters(),lr=.0001)
    for phase in ('sanity','train','val','train','val'):
        m.train(phase=='train')
        with torch.set_grad_enabled(phase=='train'),torch.autocast('cuda',dtype=torch.bfloat16):
            out=m.model_io.run(x)
            loss=reid_loss(out,x['track_person_id'],temperature=.1,margin=.5,player_weight=.1)['loss']
        print(json.dumps(dict(compiled=compiled,phase=phase,loss=float(loss.detach().cpu()),finite={k:bool(torch.isfinite(v).all()) for k,v in out.items()})),flush=True)
        if phase=='train':
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);optimizer.step()
    del m,optimizer,out,loss,x
    torch.cuda.empty_cache()
print('inductor',dict(freezing=torch._inductor.config.freezing,cudagraphs=torch._inductor.config.triton.cudagraphs),flush=True)
