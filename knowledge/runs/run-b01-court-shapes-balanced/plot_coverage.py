from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
root=Path(__file__).resolve().parent
baseline=root.parent/'b01-court-bounds-wide'
colors={'circle':'#2471a3','ellipse':'#1e8449','rectangle':'#c0392b','superellipse':'#8e44ad'}
fig,axes=plt.subplots(1,2,figsize=(12,6),sharex=True,sharey=True,layout='constrained')
stats={}
for ax,path,label in zip(axes,[baseline,root],['Circle + ellipse','Four shapes + spatial coverage']):
 plan=json.loads((path/'sfm_bounded-plan.json').read_text());d=np.load(path/'sfm_bounded-positions.npz');c=d['captured'];p=d['generated'];h=ConvexHull(c[:,:2]);border=c[np.r_[h.vertices,h.vertices[0]],:2]
 groups={g['trajectory']['trajectory_group_id']:g['trajectory']['shape'] for g in plan['groups']}
 shapes=np.array([groups[s['trajectory_group_id']] for s in plan['samples']])
 for shape,color in colors.items():
  points=p[shapes==shape]
  if len(points):ax.scatter(points[:,0],points[:,1],s=4,c=color,alpha=.55,label=shape)
 ax.plot(border[:,0],border[:,1],'k-',lw=1.2,label='SfM hull');ax.scatter(c[:,0],c[:,1],s=4,c='black',alpha=.4)
 cells=len(np.unique(np.floor(p[:,:2]).astype(int),axis=0));stats[label]={'proposals':len(p),'occupied_1m_xy_cells':cells}
 ax.set_title(f'{label}\n{cells} occupied 1m cells / {len(p)} views');ax.set_xlabel('Court-plane x (m)');ax.set_ylabel('Court-plane y (m)');ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(root/'coverage-comparison.png',dpi=150)
(root/'coverage-metrics.json').write_text(json.dumps(stats,indent=2))
