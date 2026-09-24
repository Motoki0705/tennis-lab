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


from torch.nn.attention import sdpa_kernel,SDPBackend
import src.tasks.plcs.models.components.track_query as track_query
original_keep=track_query.keep_mask
for variant in ('repeat_eval','dropout_zero','safe_mask','math_sdpa'):
    m=PLCSAssociationLightningModule.load_from_checkpoint(checkpoint,map_location='cpu',weights_only=False).cuda()
    if variant=='dropout_zero':
        for layer in m.modules():
            if hasattr(layer,'attn_dropout'):layer.attn_dropout=0.
    if variant=='safe_mask':
        def safe(valid):
            mask=valid[:,:,None]&valid[:,None,:]
            diagonal=torch.eye(valid.shape[-1],dtype=torch.bool,device=valid.device)[None]
            return mask | (diagonal & ~valid[:,:,None])
        track_query.keep_mask=safe
    else:track_query.keep_mask=original_keep
    m.model.compile(backend='inductor',mode='default',fullgraph=False,dynamic=False)
    x={k:v.cuda() for k,v in batch.items()}
    context=sdpa_kernel(SDPBackend.MATH) if variant=='math_sdpa' else contextlib.nullcontext()
    with context:
        for phase in ('eval','eval','train','eval','train','eval'):
            if variant=='repeat_eval' and phase=='train':continue
            m.train(phase=='train')
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                out=m.model_io.run(x)
                loss=reid_loss(out,x['track_person_id'],temperature=.1,margin=.5,player_weight=.1)['loss']
            print(json.dumps(dict(variant=variant,phase=phase,loss=float(loss.cpu()),finite={k:bool(torch.isfinite(v).all()) for k,v in out.items()})),flush=True)
    del m,x,out,loss
    torch.cuda.empty_cache()
