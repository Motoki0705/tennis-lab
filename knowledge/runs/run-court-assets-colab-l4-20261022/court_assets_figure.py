"""Plot the worst RGB-error camera in each scene, preserving numeric evidence."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--comparison', type=Path, required=True)
p.add_argument('--visibility', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a=p.parse_args()
report=json.loads(a.comparison.read_text())
visibility=json.loads(a.visibility.read_text())['views']
fig, axes=plt.subplots(4,5,figsize=(13,7.6),dpi=75)
for row, (sid, info) in enumerate(report['scenes'].items()):
    worst=max(info['views'],key=lambda view:view['metrics']['rgb']['abs_quantiles']['1.0'])
    cid=worst['camera_id']
    left=a.reference/'repeat-0'/sid/cid
    right=a.candidate/'repeat-0'/sid/cid
    rgb_a,rgb_b=np.load(left/'rgb.npy'),np.load(right/'rgb.npy')
    for col,rgb in enumerate((rgb_a,rgb_b)):
        axes[row,col].imshow(rgb)
        points=next(v['points'] for v in visibility if (v['scene_id'],v['repeat'],v['camera_id'])==(sid,'repeat-0',cid))
        uv=np.array([pt['uv'] for pt in points if pt['renderer_visible']])
        if len(uv):
            axes[row,col].scatter(uv[:,0],uv[:,1],s=10,facecolors='none',edgecolors='#00ff80',linewidths=.65)
    delta=np.abs(rgb_a.astype(np.float64)-rgb_b)
    axes[row,2].imshow(np.clip(delta*32,0,1))
    depth=np.abs(np.load(left/'depth.npy').astype(np.float64)-np.load(right/'depth.npy'))[...,0]/info['nht_units_per_metre']*1000
    alpha=np.abs(np.load(left/'alpha.npy').astype(np.float64)-np.load(right/'alpha.npy'))[...,0]
    axes[row,3].imshow(np.clip(alpha*4,0,1),cmap='inferno',vmin=0,vmax=1)
    axes[row,3].text(.02,.03,f'max {alpha.max():.4g}',color='cyan',transform=axes[row,3].transAxes,fontsize=7)
    cmap=plt.get_cmap('magma').copy()
    cmap.set_bad('black')
    image=axes[row,4].imshow(np.ma.masked_less_equal(depth,0),cmap=cmap,norm=LogNorm(vmin=.001,vmax=max(.01,float(depth.max()))))
    axes[row,4].text(.02,.03,f'0.001 to {depth.max():.3g} mm',color='cyan',transform=axes[row,4].transAxes,fontsize=7)
    axes[row,0].set_ylabel(f'{sid}\n{cid}',fontsize=8)
    for col in range(5):
        axes[row,col].set_xticks([])
        axes[row,col].set_yticks([])
for ax,title in zip(axes[0],('Local RGB + visible KP','L4 RGB + visible KP','Absolute RGB error x32','Absolute alpha error x4','Depth error (mm, log scale)'),strict=True):
    ax.set_title(title,fontsize=9)
fig.suptitle('Pinned NHT assets: largest RGB error among four observed views per scene',fontsize=11)
fig.tight_layout(rect=(0,0,1,.96),w_pad=.4,h_pad=.5)
fig.savefig(a.output)
plt.close(fig)
