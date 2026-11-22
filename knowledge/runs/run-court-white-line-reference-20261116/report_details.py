"""Summarize reference-only label changes and native RGB evidence."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import fit_reference as fit
from render_remaining import draw

BASE=Path(__file__).resolve().parent
PREVIOUS=BASE.parent/'court-acceptance-implementation-20261116'


def main():
    out=BASE/'report';out.mkdir(exist_ok=True);refs=json.loads((BASE/'reference/references.json').read_text());summary=json.loads((BASE/'evaluation/summary.json').read_text());changes=json.loads((BASE/'evaluation/label-changes.json').read_text())
    details={'evaluation_status':'provisional / 未再承認','prediction_selection_source':'original primary meiji4-evaluation snapshot; later scene-evaluation has different selections in two conditions at clip_033/cam0 and was not used','label_changes':{},'reference_changes':{}}
    for condition in ('legacy','acceptance_only','postprocess_and_acceptance'):
        selected=[r for r in changes if r['condition']==condition]
        details['label_changes'][condition]={'accepted_wrong_to_correct':sum(r['accepted'] and r['new_correct'] for r in selected),'accepted_correct_to_wrong':sum(r['accepted'] and not r['new_correct'] for r in selected),'rejected_wrong_to_correct':sum(not r['accepted'] and r['new_correct'] for r in selected),'rejected_correct_to_wrong':sum(not r['accepted'] and not r['new_correct'] for r in selected)}
    for ref in refs['cameras']:
        old,new=fit.project(np.array(ref['initial_matrix'])),fit.project(np.array(ref['matrix_court_metres_to_image_pixels']));oldmask=(old[:,0]>=0)&(old[:,0]<=1919)&(old[:,1]>=0)&(old[:,1]<=1079);newmask=(new[:,0]>=0)&(new[:,0]<=1919)&(new[:,1]>=0)&(new[:,1]<=1079);difference=np.linalg.norm(new[oldmask&newmask]-old[oldmask&newmask],axis=1)
        details['reference_changes'][ref['camera']]={'mean_common_visible_displacement_px':float(difference.mean()),'max_common_visible_displacement_px':float(difference.max()),'old_visible_indices':np.flatnonzero(oldmask).tolist(),'new_visible_indices':np.flatnonzero(newmask).tolist(),'measurement_failures':[l['name'] for l in ref['lines'] if not l['supported_for_fit']],'failed_frames':ref['failed_frames']}
    # Same-frame independent RGB fits: a sensitivity check, not a replacement
    # for the required camera-level provisional reference or its 276-case table.
    sensitivity=[]
    for ref in refs['cameras']:
        e=json.loads((BASE/'fit-final'/f'{ref["camera"]}-evidence.json').read_text())
        for m in e['frame_models']:
            if m['frame']['frame_index']!=0 or m['matrix'] is None:continue
            row=json.loads((BASE/'evaluation/meiji_4cam'/m['frame']['clip_id']/ref['camera']/'evaluation.json').read_text());gt=fit.project(np.array(m['matrix']));mask=(gt[:,0]>=0)&(gt[:,0]<=1919)&(gt[:,1]>=0)&(gt[:,1]<=1079)
            for name,c in row['provisional'].items():
                err=np.linalg.norm(np.array(c['points'])[mask]-gt[mask],axis=1)
                sensitivity.append({'key':row['key'],'condition':name,'accepted':c['accepted'],'pooled_correct':c['correct'],'framewise_correct':bool(err.mean()<=10 and err.max()<=25),'pooled_mean_px':c['mean_error_px'],'framewise_mean_px':float(err.mean()),'framewise_max_px':float(err.max())})
    details['same_frame_sensitivity']={'images':len(sensitivity)//3,'changed_classifications':sum(r['pooled_correct']!=r['framewise_correct'] for r in sensitivity),'not_used_to_replace_primary_results':True}
    with (out/'framewise-sensitivity.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(sensitivity[0]),lineterminator='\n');writer.writeheader();writer.writerows(sensitivity)
    with (BASE/'reference/line-frame-residuals.csv').open() as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,2,figsize=(10,5.8),sharex=True,sharey=True)
    for cam,ax in zip(('cam0','cam1','cam2','cam3'),axes.flat):
        e=json.loads((BASE/'fit-final'/f'{cam}-evidence.json').read_text());lookup={(r['clip_id'],r['frame_index']):i for i,r in enumerate(e['frames'])}
        for line in ('near_baseline','left_singles','near_service'):
            matched=[r for r in rows if r['camera']==cam and r['line']==line];xx=[lookup[(r['clip_id'],int(r['frame_index']))] for r in matched];yy=[float(r['new_signed_median_px']) for r in matched]
            ax.plot(xx,yy,'.-',label=line,linewidth=1,markersize=4)
        ax.axhline(0,color='gray',linewidth=.5);ax.set_title(cam);ax.set_xticks([1,4,7,10,13],['000','017','034','051','068']);ax.set_ylabel('Signed line offset (source px)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.supxlabel('Clip (three sampled frames per clip)');fig.tight_layout();fig.savefig(out/'line-motion.png',dpi=140);plt.close(fig)
    # A reference-error example: the prediction and its acceptance are identical.
    cam='cam2';key='meiji_4cam/video_000/clip_000/cam2';row=json.loads((BASE/'evaluation'/key/'evaluation.json').read_text());value=row['provisional']['legacy'];assert row['old']['legacy']['accepted'] and not row['old']['legacy']['correct'] and value['correct']
    ref=next(c for c in refs['cameras'] if c['camera']==cam);old,new=fit.project(np.array(ref['initial_matrix'])),fit.project(np.array(ref['matrix_court_metres_to_image_pixels']));pred=np.array(value['points']);supported=np.zeros(14,int)
    for i,l in enumerate(ref['lines']):
        if l['supported_for_fit']:supported[fit.EDGES[i]]+=1
    indices=np.flatnonzero((supported>=2)&(new[:,0]>=0)&(new[:,0]<=1919)&(new[:,1]>=0)&(new[:,1]<=1079));improvement=np.linalg.norm(pred-old,axis=1)-np.linalg.norm(pred-new,axis=1);index=indices[np.argmax(improvement[indices])];cx,cy=new[index];x=int(np.clip(round(cx-256),0,1408));y=int(np.clip(round(cy-96),0,888))
    raw=cv2.imread(str(BASE/'frames/cam2/clip_000_frame_000000.png'));images=[raw,draw(draw(raw,old,(255,80,0)),pred,(255,0,255)),draw(draw(raw,new,(0,160,255)),pred,(255,0,255))]
    head=np.zeros((52,1536,3),np.uint8);cv2.putText(head,f"SAME frozen legacy prediction: clip_000/cam2; old mean={row['old']['legacy']['mean_error_px']:.2f}px -> provisional={value['mean_error_px']:.2f}px",(8,20),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA);cv2.putText(head,'RAW | OLD BLUE + PRED MAGENTA | NEW ORANGE + SAME PRED MAGENTA; native crop',(8,43),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    grid=np.vstack([head,np.hstack([im[y:y+192,x:x+512] for im in images])]);cv2.imwrite(str(out/'reference-error-native.png'),grid);Image.fromarray(cv2.cvtColor(grid,cv2.COLOR_BGR2RGB)).quantize(colors=128).save(out/'reference-error.png',optimize=True)
    details['reference_error_example']={'key':key,'condition':'legacy','point_index':int(index),'old_mean_px':row['old']['legacy']['mean_error_px'],'new_mean_px':value['mean_error_px'],'same_prediction_and_gate':True}
    (out/'details.json').write_text(json.dumps(details,indent=2,allow_nan=False));print(json.dumps(details,indent=2))


if __name__=='__main__':main()
