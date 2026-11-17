"""Compact scientific review figures; full native images stay in outputs."""
from pathlib import Path
import csv,json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('outputs/tennis_scene/evaluate/court-video-drift-lines-20261117')
rows=list(csv.DictReader((ROOT/'frames.csv').open()))
fig,axes=plt.subplots(4,1,figsize=(12,7.6),sharex=True,constrained_layout=True)
for ax,camera in zip(axes,['cam0','cam1','cam2','cam3']):
    selected=[r for r in rows if r['camera']==camera]
    for status,color,marker in [('ok','#24805b','.'),('warning','#d48600','o'),('failed','#bd3656','x')]:
        values=[r for r in selected if r['status']==status]
        x=[int(r['clip_id'].split('_')[-1])+([v['frame_index'] for v in selected if v['clip_id']==r['clip_id']].index(r['frame_index'])-2)*.13 for r in values]
        y=[float(r['mean_px']) if r['mean_px'] else -1 for r in values]
        ax.scatter(x,y,color=color,marker=marker,s=12,label=status)
    ax.axhline(3,color='#d48600',ls='--',lw=.8)
    ax.set_ylabel(camera+'\nmean px');ax.set_yticks([-1,0,3,6],['unknown','0','3','6']);ax.grid(alpha=.2);ax.set_ylim(-1.7,7)
axes[0].legend(ncol=3,loc='upper right');axes[-1].set_xlabel('Clip index (five independently measured frames per clip)')
fig.suptitle('Immutable shared H: per-frame drift (no temporal averaging)')
fig.savefig(ROOT/'drift-timeline.svg');plt.close(fig)
examples=['cam0-clip_000-106-left_doubles','cam1-clip_001-448-right_singles','cam2-clip_004-409-left_singles','cam3-clip_000-35-right_doubles']
canvas=np.full((2*330,2*361,3),28,np.uint8)
for i,name in enumerate(examples):
    image=cv2.imread(str(ROOT/'causes'/f'{name}.png'))
    y,x=(i//2)*330,(i%2)*361
    canvas[y+29:y+29+image.shape[0],x:x+image.shape[1]]=image
    cv2.putText(canvas,name,(x+6,y+19),cv2.FONT_HERSHEY_SIMPLEX,.34,(245,245,245),1,cv2.LINE_AA)
cv2.imwrite(str(ROOT/'conflicting-lines.png'),canvas,[cv2.IMWRITE_PNG_COMPRESSION,9])
