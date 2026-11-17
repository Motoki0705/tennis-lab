"""Small versioned figures; original-resolution evidence remains under outputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

BASE=Path(__file__).resolve().parent
OUT=Path('outputs/tennis_scene/evaluate/court-fixed-support-rgb-20261117')
MODES=('legacy','fixed_support','fixed_support_rgb')
LABELS=('Legacy','Fixed KP','Fixed KP + RGB')


def main():
    full=json.loads((OUT/'report/summary.json').read_text())
    report={'all':full['datasets']['meiji_4cam']['all'],'holdout':full['holdout']['all'],
            'cameras':{cam:values for cam,values in full['datasets']['meiji_4cam'].items() if cam!='all'}}
    figures=BASE/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    left=np.zeros(3)
    for name,color in [('TP','#2a9d55'),('FP','#d94b43'),('FN','#d7a63b'),('TN','#999999')]:
        values=np.array([report['all'][mode][name] for mode in MODES])
        axes[0].barh(LABELS,values,left=left,label=name,color=color)
        for i,value in enumerate(values):
            if value>=15:axes[0].text(left[i]+value/2,i,str(value),ha='center',va='center',fontsize=9)
        left+=values
    axes[0].set(title='276 images: acceptance confusion',xlabel='Images');axes[0].legend(ncols=2,fontsize=8);axes[0].invert_yaxis()
    x=np.arange(3)
    for offset,group,label in [(-.18,'all','All 276'),(.18,'holdout','No reference/pilot: 224')]:
        values=[100*report[group][mode]['correct_acceptance_rate'] for mode in MODES]
        bars=axes[1].bar(x+offset,values,.34,label=label)
        axes[1].bar_label(bars,fmt='%.1f',fontsize=8)
    axes[1].axhline(90,color='#bd3030',ls='--',lw=1,label='Target 90%')
    axes[1].set(xticks=x,xticklabels=['Legacy','Fixed KP','+ RGB'],ylim=(0,106),ylabel='Correct estimate acceptance (%)',title='False accepts: RGB 0; recall target missed')
    axes[1].legend(fontsize=8,loc='upper left')
    x=np.arange(4)
    for offset,mode,label in zip([-.25,0,.25],MODES,LABELS):
        axes[2].bar(x+offset,[report['cameras'][f'cam{i}'][mode]['frame_mean_error_px']['median'] for i in range(4)],.24,label=label)
    axes[2].set(xticks=x,xticklabels=['cam0','cam1','cam2','cam3'],ylabel='Median frame mean error (source px)',title='Diagnostic H, including rejected estimates')
    axes[2].legend(fontsize=8)
    fig.savefig(figures/'acceptance.png',dpi=120);plt.close(fig)
    tiles=[]
    for clip,camera,label in [(0,'cam0','Correct RGB fit, rejected'),(23,'cam0','Correct RGB fit, accepted'),(0,'cam3','Incorrect RGB fit, rejected')]:
        original=cv2.imread(str(OUT/'review'/f'clip_{clip:03d}_{camera}_fixed_support_rgb.jpg'))
        # Source image occupies the original 1920 px; the appended panel is not
        # resampled into the overview. Native evidence is linked from the node.
        crop=original[340:900,100:1880]
        tile=cv2.resize(crop,(780,246),interpolation=cv2.INTER_AREA)
        banner=np.full((52,780,3),24,np.uint8)
        cv2.putText(banner,f'clip {clip:03d} {camera}: {label}',(12,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
        tiles.append(np.vstack([banner,tile]))
    montage=np.vstack(tiles)
    montage=cv2.resize(montage,(720,round(montage.shape[0]*720/montage.shape[1])),interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(figures/'examples.png'),montage,[cv2.IMWRITE_PNG_COMPRESSION,9])


if __name__=='__main__':main()
