"""Plot recorded CPU diagnostics only; this script runs no reader or model."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
summary=json.loads((root/'profile-summary.json').read_text())
cases=summary['cases']
rows=cases['normal-workers8']['rounds']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'path','pdf.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#263445','text.color':'#162535'})
fig,(a,b)=plt.subplots(1,2,figsize=(12.8,7.2),gridspec_kw={'width_ratios':[1,1.15]})
fig.subplots_adjust(left=.11,right=.965,top=.77,bottom=.33,wspace=.48)
colors=['#237fa3','#e6a14a','#acb9c7','#55a59a']
keys=['mdd','verify','read_bgr','luminance_and_metadata']
labels=['CPU MDD','Integrity checks','JPEG read + decode','Luminance + metadata']
left=np.zeros(2)
for key,label,color in zip(keys,labels,colors):
 vals=np.array([r['mean_stage_seconds'][key]/r['mean_stage_seconds']['prepare']*100 for r in rows])
 a.barh([0,1],vals,left=left,color=color,height=.55,label=label,edgecolor='white',linewidth=1.5)
 for i,v in enumerate(vals):
  if v>=8: a.text(left[i]+v/2,i,f'{v:.1f}%',ha='center',va='center',color='white' if key=='mdd' else '#162535',fontweight='bold',fontsize=10)
 left+=vals
a.set_yticks([0,1],['First\nverification','Verified\nreplay']); a.invert_yaxis()
a.set_xlim(0,100); a.set_xticks([0,25,50,75,100]); a.set_xlabel('Share of worker input-preparation time (%)')
a.set_title('A   Where CPU preparation time goes',loc='left',fontweight='bold',pad=18)
a.legend(loc='upper left',bbox_to_anchor=(-.04,-.16),ncol=2,frameon=False,fontsize=9,columnspacing=1)
names=['Full reader\nfirst pass','Full reader\nverified replay','Preverified\nreader only','Cached tensor\ncollate + IPC']
values=[rows[0]['windows_per_second'],rows[1]['windows_per_second'],cases['preverified-workers8']['rounds'][0]['windows_per_second'],cases['cached-input-workers8']['rounds'][1]['windows_per_second']]
b.barh(range(4),values,height=.56,color=['#9dabb8','#63869c','#55a59a','#237fa3'])
b.set_yticks(range(4),names); b.invert_yaxis(); b.set_xlim(0,15.5); b.set_xlabel('Windows / second (32 frames per window)')
b.set_title('B   Isolating the input pipeline',loc='left',fontweight='bold',pad=18)
b.set_axisbelow(True); b.grid(axis='x',color='#e3e9ef',linewidth=.7)
for y,v in enumerate(values): b.text(v+.2,y,f'{v:.2f}',va='center',fontweight='bold')
fig.text(.085,.925,'CPU input bottleneck  |  Conv2d + query-only',fontsize=21,fontweight='bold')
fig.text(.085,.865,'1280 × 720  ·  32 frames  ·  8 workers  ·  real frozen train inputs  ·  training remains stopped',fontsize=10.5,color='#536273')
fig.text(.085,.145,'Main targets: CPU luminance/MDD generation and repeated full-clip integrity checks.',fontsize=11,fontweight='bold')
fig.text(.085,.055,'A: per-worker service time; workers run concurrently. B: CPU-only; no model, GPU, H2D or pinned-memory timing.\nPreverified reader excludes 122.7 s of serial setup (89 clips); it is not an end-to-end speedup claim.\nCached-input case reuses one real tensor to isolate collation/IPC. Other cases use the same 96-window sequence.',fontsize=9,color='#536273',linespacing=1.55)
for ext in ('svg','pdf','png'): fig.savefig(root/f'cpu-input-bottleneck.{ext}',dpi=180,facecolor='white')
