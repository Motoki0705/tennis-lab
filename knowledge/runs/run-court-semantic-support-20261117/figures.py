"""Four requested comparisons; every native overlay remains in the review index."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

BASE=Path(__file__).resolve().parent
OUT=Path('outputs/tennis_scene/evaluate/court-semantic-support-20261117')
MODES=('legacy','fixed_support','fixed_support_rgb','new_rgb')
LABELS=('Legacy','Previous (a)','Previous (a+b)','New (a+b)')


def main():
 report=json.loads((OUT/'report/summary.json').read_text());full=report['datasets']['meiji_4cam'];figures=BASE/'figures';figures.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');left=np.zeros(4)
 for name,color in [('TP','#2a9d55'),('FP','#d94b43'),('FN','#d7a63b'),('TN','#999999')]:
  values=np.array([full['all'][mode][name] for mode in MODES]);axes[0].barh(LABELS,values,left=left,label=name,color=color)
  for i,v in enumerate(values):
   if v>=10:axes[0].text(left[i]+v/2,i,str(v),ha='center',va='center',fontsize=8)
  left+=values
 axes[0].set(title='All 276: acceptance confusion',xlabel='Images');axes[0].invert_yaxis();axes[0].legend(ncols=2,fontsize=8)
 for offset,values,label in [(-.18,full['all'],'All 276'),(.18,report['holdout']['all'],'No reference/pilot: 224')]:
  bars=axes[1].bar(np.arange(4)+offset,[100*values[m]['correct_acceptance_rate'] if values[m]['correct_acceptance_rate'] is not None else 0 for m in MODES],.34,label=label);axes[1].bar_label(bars,fmt='%.1f',fontsize=8)
 axes[1].axhline(90,color='red',ls='--',lw=1);axes[1].set(xticks=np.arange(4),xticklabels=['Legacy','Prev A','Prev AB','New AB'],ylim=(0,106),title='Correct H acceptance (%)');axes[1].legend(fontsize=8)
 for offset,mode,label in zip([-.3,-.1,.1,.3],MODES,LABELS):
  axes[2].bar(np.arange(4)+offset,[full[f'cam{i}'][mode]['frame_mean_error_px']['median'] or 0 for i in range(4)],.19,label=label)
 axes[2].set(xticks=np.arange(4),xticklabels=['cam0','cam1','cam2','cam3'],ylabel='Median frame mean error (px)',title='Diagnostic H, including rejected');axes[2].legend(fontsize=7)
 fig.savefig(figures/'acceptance.png',dpi=120);plt.close(fig)
 # Native measured net stripe and the distinct wrong-baseline failure.
 inspections=json.loads((OUT/'rgb-measurement-inspection.json').read_text());tiles=[];manifest=[]
 for item in inspections:
  camera=item['key'].split('/')[-1];clip=int(item['key'].split('/')[-2].split('_')[-1])
  if camera=='cam1':
   original=cv2.imread(item['native_crop']);left=(480-original.shape[1])//2
   tile=cv2.copyMakeBorder(original,0,0,left,480-original.shape[1]-left,cv2.BORDER_CONSTANT,value=(24,24,24))
  else:
   original=cv2.imread(str(OUT/'review'/f'clip_{clip:03d}_{camera}_new_rgb.jpg'))[:,:1920]
   tile=cv2.resize(original,(480,270),interpolation=cv2.INTER_AREA)
  banner=np.full((42,480,3),24,np.uint8)
  label=f'clip{clip:03d} {camera}: mean/max {item["mean_error_px"]:.1f}/{item["max_error_px"]:.1f}px'
  cv2.putText(banner,label,(8,27),cv2.FONT_HERSHEY_SIMPLEX,.46,(245,245,245),1,cv2.LINE_AA)
  tiles.append(np.vstack([banner,tile]));manifest.append({k:v for k,v in item.items() if k!='lines'})
 if tiles:cv2.imwrite(str(figures/'examples.png'),np.vstack(tiles),[cv2.IMWRITE_PNG_COMPRESSION,9])
 (figures/'examples.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
