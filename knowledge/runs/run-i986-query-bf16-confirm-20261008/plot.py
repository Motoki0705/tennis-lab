"""Render the stored BF16 measurements; no training or network access."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

root = Path(__file__).resolve().parent
capacity = root.parent / 'run-i986-query-gpu-capacity-20261008' / 'measurements'
compute = [json.loads((capacity / f'bf16-bs{bs}.json').read_text()) for bs in (1, 2, 4, 6)]
rows = [json.loads(p.read_text()) for p in (root / 'measurements').glob('*.json')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':'#263445','text.color':'#162535','xtick.color':'#465464',
                     'ytick.color':'#465464','svg.fonttype':'path','pdf.fonttype':42})
fig = plt.figure(figsize=(12.8, 7.8), facecolor='white')
grid = fig.add_gridspec(2, 2, width_ratios=(1,1.2), left=.075,right=.94,bottom=.17,top=.80,hspace=.48,wspace=.32)
a = fig.add_subplot(grid[0,0]); b = fig.add_subplot(grid[1,0]); c = fig.add_subplot(grid[:,1])
labels = ['1','2','4','6']
for ax in (a,b):
 ax.set_axisbelow(True); ax.grid(axis='y',color='#e3e9ef',linewidth=.7)
 ax.set_xticks(range(4),labels); ax.set_xlabel('Batch size (32-frame windows)')
values = [r['windows_per_second'] for r in compute]
a.bar(range(4),values,color='#237fa3',width=.58)
a.set_ylim(0,8.5); a.set_ylabel('Windows / second')
a.set_title('A   GPU-resident training',loc='left',fontweight='bold',pad=14)
for x,y in enumerate(values): a.text(x,y+.13,f'{y:.2f}',ha='center',fontsize=10)
mem = [r['peak_reserved_bytes']/2**30 for r in compute]
b.bar(range(4),mem,color='#63869c',width=.58)
b.set_ylim(0,16.4); b.set_yticks((0,4,8,12,16)); b.set_ylabel('Peak reserved VRAM (GiB)')
b.set_title('B   GPU memory',loc='left',fontweight='bold',pad=14)
b.axhline(14.34,color='#b96929',linestyle=(0,(4,3)),linewidth=1.1)
b.text(-.4,14.6,'90% allocator limit',fontsize=9,color='#945422')
for x,y in enumerate(mem): b.text(x,y+.18,f'{y:.2f}',ha='center',fontsize=10)
values=np.full((3,3),np.nan)
status={}
for row in rows:
 i=(1,2,4).index(row['arguments']['batch_size']); j=(4,6,8).index(row['arguments']['workers'])
 status[(i,j)]=row['status']
 if row['status']=='ok': values[i,j]=row['windows_per_second']
c.set_facecolor('#eef1f4')
mesh=c.imshow(np.ma.masked_invalid(values),vmin=0,vmax=2,cmap='Blues',aspect='auto')
c.set_xticks(range(3),['4 workers','6 workers','8 workers'])
c.set_yticks(range(3),['BS = 1','BS = 2','BS = 4'])
c.tick_params(length=0); c.set_title('C   End-to-end training: same window sequence',loc='left',fontweight='bold',pad=22)
for i in range(3):
 for j in range(3):
  value=values[i,j]
  if np.isfinite(value): text=f'{value:.2f}\nwindows/s'; color='white' if value>1.2 else '#162535'
  elif status.get((i,j))=='error': text='CUDA error\nnot a speed result'; color='#9b3e32'
  else: text='Not measured'; color='#6b7785'
  c.text(j,i,text,ha='center',va='center',color=color,fontsize=11 if np.isfinite(value) else 9,linespacing=1.7)
c.add_patch(Rectangle((1.52,-.48),.96,.96,fill=False,edgecolor='#edac43',linewidth=3,zorder=5))
c.set_xticks(np.arange(-.5,3,1),minor=True); c.set_yticks(np.arange(-.5,3,1),minor=True)
c.grid(which='minor',color='white',linewidth=3); c.tick_params(which='minor',bottom=False,left=False)
for spine in c.spines.values(): spine.set_visible(False)
fig.text(.075,.94,'Conv2d + query-only  |  BF16 GPU validation',fontsize=21,fontweight='bold')
fig.text(.075,.892,'RTX 5060 Ti 16GB  ·  1280 × 720  ·  32 sampled frames  ·  PyTorch 2.13.0 / CUDA 13.0',fontsize=11,color='#536273')
fig.text(.075,.085,'Recommended: BS 1, 8 workers. Longer check: 1.33 windows/s over 240 measured updates; peak VRAM 2.40 GiB.',fontsize=10,fontweight='bold')
fig.text(.075,.045,'A/B: 4 warm-up + 16 measured updates. C: identical 12 warm-up + 96 measured windows; pin memory, prefetch 1.\nVRAM excludes display/context overhead. GPU-resident BS 8: OOM at the 90% allocator cap. Throughput checks, not accuracy results.',fontsize=9,color='#536273',linespacing=1.5)
for extension in ('svg','pdf','png'):
 fig.savefig(root / f'bf16-gpu-validation.{extension}',dpi=180,facecolor='white')
