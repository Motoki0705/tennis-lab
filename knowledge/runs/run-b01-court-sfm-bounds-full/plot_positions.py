from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
root=Path(__file__).resolve().parent
fig,axes=plt.subplots(1,2,figsize=(12,6),sharex=True,sharey=True,layout='constrained')
for ax,selector,title in zip(axes,['v3','sfm_bounded'],['Previous: 3,769 / 4,800 outside','SfM bounded: 0 / 2,288 outside']):
 d=np.load(root/f'{selector}-positions.npz');c=d['captured'];p=d['generated'];h=ConvexHull(c[:,:2]);border=c[np.r_[h.vertices,h.vertices[0]],:2]
 ax.scatter(p[:,0],p[:,1],s=3,c=p[:,2],cmap='viridis',alpha=.6,label='Generated cameras')
 ax.plot(border[:,0],border[:,1],color='black',lw=1.3,label='Captured-camera hull')
 ax.scatter(c[:,0],c[:,1],s=3,color='orangered',label='SfM cameras')
 for x in [0,-14.84764638,-29.6395624]:
  xx=np.array([-5.485,5.485,5.485,-5.485,-5.485])+x;yy=np.array([-11.885,-11.885,11.885,11.885,-11.885]);ax.plot(xx,yy,color='gray',lw=1)
 ax.set_title(title);ax.set_xlabel('Court-plane x (m)');ax.set_ylabel('Court-plane y (m)');ax.set_aspect('equal');ax.grid(alpha=.2)
axes[0].legend(loc='lower left',fontsize=8)
fig.savefig(root/'position-comparison.png',dpi=140)
