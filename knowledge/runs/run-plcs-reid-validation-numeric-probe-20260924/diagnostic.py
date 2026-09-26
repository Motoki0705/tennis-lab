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
for device,compiled,amp,inference in [('cpu',False,False,False),('cuda',False,False,False),('cuda',False,True,False),('cuda',False,True,True),('cuda',True,True,False),('cuda',True,True,True)]:
    m=PLCSAssociationLightningModule.load_from_checkpoint(checkpoint,map_location='cpu',weights_only=False).eval().to(device)
    if compiled:m.model.compile(backend='inductor',mode='default',fullgraph=False,dynamic=False)
    x={k:v.to(device) for k,v in batch.items()}
    start=time.monotonic()
    with torch.inference_mode() if inference else torch.no_grad():
        with torch.autocast(device,dtype=torch.bfloat16,enabled=amp):
            out=m.model_io.run(x)
            loss=reid_loss(out,x['track_person_id'],temperature=.1,margin=.5,player_weight=.1)
    print(json.dumps(dict(device=device,compiled=compiled,amp=amp,inference=inference,seconds=time.monotonic()-start,loss=float(loss['loss'].cpu()),finite={k:bool(torch.isfinite(v).all()) for k,v in out.items()},valid=int(out['track_valid'].sum()),maxes={k:float(v.float().abs().max()) for k,v in out.items()})),flush=True)
    del m,x,out,loss
    if device=='cuda':torch.cuda.empty_cache()
