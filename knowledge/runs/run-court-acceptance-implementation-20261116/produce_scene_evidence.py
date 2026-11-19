"""Compact per-image evidence and measured overlays, without changing the gate."""
import csv
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from current_geometry.homography import COURT_HOMOGRAPHY_EDGES, court_template_xy

BASE=Path(__file__).resolve().parent
CONDITIONS=['legacy','acceptance_only','postprocess_and_acceptance']


def main():
    out=BASE/'scene-evidence';out.mkdir(exist_ok=True)
    rows=[json.loads(f.read_text()) for f in sorted((BASE/'meiji4-evaluation').glob('**/evaluation.json'))]
    assert len(rows)==276
    with (out/'meiji4-cases.csv').open('w',newline='') as f:
        fields=['key','condition','accepted','correct','classification','mean_error_px','max_error_px','any_correct_candidate','region_index','score','reasons']
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in rows:
            for name in CONDITIONS:
                v=row['conditions'][name]
                d={k:v[k] for k in ['accepted','correct','mean_error_px','max_error_px','any_correct_candidate','region_index']}
                d.update(key=row['key'],condition=name,classification='TP' if v['accepted'] and v['correct'] else 'FP' if v['accepted'] else 'FN' if v['correct'] else 'TN',
                    score=v['quality']['score'],reasons=';'.join(v['quality']['reasons']))
                writer.writerow(d)
    candidate_counts={}
    for camera in ['all','cam0','cam1','cam2','cam3']:
        subset=[r for r in rows if camera=='all' or r['camera']==camera];candidate_counts[camera]={}
        for name in CONDITIONS:
            candidates=[c for r in subset for c in r['current_candidates' if name=='postprocess_and_acceptance' else 'legacy_candidates']]
            gate='legacy_accepted' if name=='legacy' else 'new_accepted'
            counts={'TP':0,'FP':0,'FN':0,'TN':0}
            for c in candidates:counts['TP' if c[gate] and c['correct'] else 'FP' if c[gate] else 'FN' if c['correct'] else 'TN']+=1
            candidate_counts[camera][name]=counts
    (out/'all-candidate-confusion.json').write_text(json.dumps(candidate_counts,indent=2))
    accepted_metrics={}
    for camera in ['cam0','cam1','cam2','cam3','all']:
        accepted_metrics[camera]={}
        for name in CONDITIONS:
            chosen=[r['conditions'][name] for r in rows if (camera=='all' or r['camera']==camera) and r['conditions'][name]['accepted']]
            errors=[v['mean_error_px'] for v in chosen if v['mean_error_px'] is not None]
            accepted_metrics[camera][name]={'accepted':len(chosen),'correct':sum(v['correct'] for v in chosen),
                'mean_of_frame_mean_error_px':float(np.mean(errors)) if errors else None,
                'median_frame_mean_error_px':float(np.median(errors)) if errors else None}
    (out/'accepted-metrics.json').write_text(json.dumps(accepted_metrics,indent=2))
    rejected=[(r['camera'],c['quality']['reasons']) for r in rows for c in r['current_candidates'] if c['correct'] and not c['new_accepted']]
    rejection_reasons={'new_correct_rejected_candidates':len(rejected),
        'reason_counts':dict(Counter(reason for _,reasons in rejected for reason in reasons)),
        'cam3_reason_counts':dict(Counter(reason for camera,reasons in rejected if camera=='cam3' for reason in reasons))}
    (out/'correct-rejection-reasons.json').write_text(json.dumps(rejection_reasons,indent=2))
    # Reference unchanged; use the approved fixed H for every pictured condition.
    references=json.loads((BASE/'references.json').read_text())
    requested=[('clip_004','cam0'),('clip_000','cam1'),('clip_000','cam2'),('clip_013','cam3')]
    tiles=[];examples=[]
    for clip,cam in requested:
        row=next(r for r in rows if r['key']==f'meiji_4cam/video_000/{clip}/{cam}')
        frame=cv2.imread(str(BASE/'raw'/row['key']/'frame.jpg'))
        ref=next(r for r in references['cameras'] if r['camera']==cam)
        truth=cv2.perspectiveTransform(court_template_xy().astype(float)[None],np.array(ref['matrix_court_metres_to_image_pixels']))[0]
        columns=[]
        for name in CONDITIONS:
            v=row['conditions'][name];im=frame.copy()
            for points,color in [(truth,(255,255,0)),(v['points'],(40,220,40) if v['accepted'] else (40,140,255))]:
                if points is None:continue
                xy=np.asarray(points)
                for a,b in COURT_HOMOGRAPHY_EDGES:
                    if np.isfinite(xy[[a,b]]).all() and np.max(np.abs(xy[[a,b]]))<1e6:
                        cv2.line(im,tuple(np.rint(xy[a]).astype(int)),tuple(np.rint(xy[b]).astype(int)),color,3,cv2.LINE_AA)
            im=cv2.resize(im,(960,540));cv2.rectangle(im,(0,0),(960,64),(0,0,0),-1)
            err='no H' if v['mean_error_px'] is None else f"mean {v['mean_error_px']:.1f}px max {v['max_error_px']:.1f}px"
            cv2.putText(im,f"{clip}/{cam}  {name}",(10,24),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(im,f"{'ACCEPT' if v['accepted'] else 'REJECT'} / {'correct' if v['correct'] else 'wrong'}  {err}",(10,51),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),1,cv2.LINE_AA)
            columns.append(im)
        tiles.append(np.hstack(columns));examples.append({'key':row['key'],'conditions':row['conditions']})
    figure=np.vstack(tiles)
    cv2.imwrite(str(out/'representative-overlays.jpg'),figure,[cv2.IMWRITE_JPEG_QUALITY,65])
    for index,name in enumerate(['cam0-good','cam1-offset','cam2-offset','cam3-unstable']):
        tile=Image.fromarray(cv2.cvtColor(figure[540*index:540*(index+1)],cv2.COLOR_BGR2RGB))
        tile.resize((1920,360),Image.Resampling.LANCZOS).quantize(colors=128).save(out/f'{name}.png',optimize=True)
    (out/'representative-examples.json').write_text(json.dumps(examples,indent=2))
    print('rows',len(rows),'figure bytes',(out/'representative-overlays.jpg').stat().st_size)


if __name__=='__main__':main()
