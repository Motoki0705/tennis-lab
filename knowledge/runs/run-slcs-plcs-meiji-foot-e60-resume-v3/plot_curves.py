from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root=Path(__file__).resolve().parent
logs=Path('/home/kamimura/projects/tennis-lab/outputs/plcs/train/meiji_foot_real_rgb/s42-001/logs')
fig,axs=plt.subplots(2,2,figsize=(12,8))
metrics=['loss','position_error_m','angular_error_deg','canonical_mpjpe_m']
series={}
for ax,metric in zip(axs.flat,metrics):
 for stage in ['train','val']:
  pts=[]
  for v in range(4):
   a=EventAccumulator(str(logs/f'version_{v}'),size_guidance={'scalars':0});a.Reload()
   tag=f'{stage}/{metric}'
   if tag in a.Tags()['scalars']:pts.extend((x.step,x.value) for x in a.Scalars(tag))
  series[f'{stage}/{metric}']=pts
  ax.plot([s/250 for s,y in pts],[y for s,y in pts],label=stage)
 for boundary in [30,38,47]:ax.axvline(boundary,color='gray',alpha=.3,linestyle=':')
 ax.set_title(metric);ax.set_xlabel('Completed epochs (global_step / 250)');ax.legend();ax.grid(alpha=.2)
fig.suptitle('Meiji PLCS: 60 epochs across versions 0–3; dotted lines = restarts')
fig.tight_layout();fig.savefig(root/'all_versions_curves.png',dpi=150)
(root/'curve_scalars.json').write_text(json.dumps(series,indent=2)+'\n')
