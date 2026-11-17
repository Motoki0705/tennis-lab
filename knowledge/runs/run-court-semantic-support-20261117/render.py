"""All 69 x 4 original-resolution overlays, with acceptance and physical residuals."""
from collections import defaultdict
import html
import json
from pathlib import Path

import cv2
import numpy as np

from src.tasks.court_detection.geometry.homography import COURT_HOMOGRAPHY_EDGES, court_template_xy

OUT=Path('outputs/tennis_scene/evaluate/court-semantic-support-20261117')
RGB=Path('outputs/tennis_scene/evaluate/court-fixed-support-rgb-20261117/rgb')
REF=Path('knowledge/runs/run-court-white-line-reference-20261116/reference/references.json')
MODES=('legacy','fixed_support','fixed_support_rgb','new_support','new_rgb')


def draw(image,points,color,thickness=2):
    if points is None:return
    points=np.asarray(points)
    if not np.isfinite(points).all():return
    h,w=image.shape[:2]
    for a,b in COURT_HOMOGRAPHY_EDGES:
        okay,start,end=cv2.clipLine((0,0,w,h),tuple(np.clip(np.rint(points[a]),-100000,100000).astype(int)),
                                   tuple(np.clip(np.rint(points[b]),-100000,100000).astype(int)))
        if okay:cv2.line(image,start,end,color,thickness,cv2.LINE_AA)


def text(image,line,y,color=(240,240,240),scale=.55):
    cv2.putText(image,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),4,cv2.LINE_AA)
    cv2.putText(image,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def main():
    references={r['camera']:cv2.perspectiveTransform(court_template_xy()[None].astype(float),np.asarray(r['matrix_court_metres_to_image_pixels']))[0]
                for r in json.loads(REF.read_text())['cameras']}
    records=[r for r in json.loads((OUT/'manifest.json').read_text())['records'] if r['dataset']=='meiji_4cam']
    output=OUT/'review';output.mkdir(exist_ok=True)
    clips=defaultdict(dict);index=[];examples=[]
    for record in records:
        result=json.loads((OUT/'evaluation'/record['key']/'evaluation.json').read_text())
        original=cv2.imread(str(RGB/record['key']/'frame.png'))
        clip=int(record['clip_id'].split('_')[-1]);camera=record['camera']
        modes=[];descriptions=[]
        for mode in MODES:
            chosen=result['conditions'][mode];image=original.copy()
            color=(40,220,40) if chosen['accepted'] else (0,150,255)
            draw(image,references[camera],(255,0,255),1)
            draw(image,chosen['points'],color,2)
            clean_image=image.copy()
            panel=np.full((image.shape[0],720,3),24,np.uint8)
            image=np.hstack([image,panel]);panel=image[:,original.shape[1]:]
            text(panel,f'{clip:03d} {camera} {mode}',30,color,scale=.7)
            text(panel,f'{"ACCEPT" if chosen["accepted"] else "REJECT"} | reference=magenta',60,color,scale=.7)
            mean,maximum=chosen['mean_error_px'],chosen['max_error_px']
            text(panel,f'mean/max: {mean:.2f}/{maximum:.2f} px; correct={chosen["correct"]}' if mean is not None else 'No fitted H',95,scale=.7)
            descriptions.append(f'{"採用" if chosen["accepted"] else "拒否"} / {mean:.2f},{maximum:.2f}px' if mean is not None else '推定なし')
            if mode in ('fixed_support_rgb','new_rgb'):
                q=chosen['quality'];paint=q['rgb_paint']
                for n,reason in enumerate(q['reasons'] or ['All RGB gates passed']):text(panel,reason,135+26*n,scale=.65)
                for row,line in enumerate(paint.get('lines',[])):
                    p95=line.get('p95_residual_px');res='--' if p95 is None else f'{p95:.2f}'
                    text(panel,f'{line["name"]}: p95={res}px cov={line["coverage"]:.2f}',335+55*row,scale=.65)
                    loo=paint.get('stability',{}).get('leave_line_out_max_px',{}).get(line['name'])
                    loo_text='--' if loo is None else f'{loo:.1f}'
                    text(panel,f'{line["status"]} required={line.get("required_initially",False)} n={line.get("support_locations",0)} LOO={loo_text}',358+55*row,scale=.50)
                stability=paint.get('stability',{})
                text(panel,f'stability: {stability.get("stability_status","not_fitted")}',870,scale=.65)
                if 'max_visible_projection_std_px' in stability:
                    text(panel,f'std={stability["max_visible_projection_std_px"]:.2f}px LOO={stability["max_leave_line_out_px"]:.2f}px',900,scale=.65)
                text(panel,'Essential: '+', '.join(stability.get('essential_lines',[])),932,scale=.5)
            name=f'clip_{clip:03d}_{camera}_{mode}.jpg'
            cv2.imwrite(str(output/name),image,[cv2.IMWRITE_JPEG_QUALITY,96]);modes.append(name)
            if mode=='new_rgb':
                tile=np.vstack([np.full((60,960,3),24,np.uint8),cv2.resize(clean_image,(960,540),interpolation=cv2.INTER_AREA)])
                text(tile,f'clip {clip:03d} {camera}: {"ACCEPT" if chosen["accepted"] else "REJECT"}',23,color,scale=.65)
                text(tile,f'mean/max {mean:.2f}/{maximum:.2f}px; correct={chosen["correct"]}' if mean is not None else 'No fitted H',48,scale=.6)
                clips[clip][camera]=tile
                if chosen['accepted'] and not chosen['correct']:examples.append((clip,camera,name,mean,maximum))
        original_link=html.escape('../../court-fixed-support-rgb-20261117/rgb/'+record['key']+'/frame.png',quote=True)
        index.append(f'<tr><td>{clip:03d} {camera}<br><a href="{original_link}">原画像PNG</a></td>'+''.join(f'<td><a href="{name}">{mode}</a><br>{description}</td>' for mode,name,description in zip(MODES,modes,descriptions))+'</tr>')
    for clip,cameras in clips.items():
        assert len(cameras)==4
        grid=np.vstack([np.hstack([cameras['cam0'],cameras['cam1']]),np.hstack([cameras['cam2'],cameras['cam3']])])
        cv2.imwrite(str(output/f'clip_{clip:03d}.jpg'),grid,[cv2.IMWRITE_JPEG_QUALITY,96])
    body=''.join(f'<a href="clip_{clip:03d}.jpg"><img width="480" src="clip_{clip:03d}.jpg"></a>' for clip in sorted(clips))
    false='<ul>'+''.join(f'<li><a href="{name}">clip{clip:03d} {camera}</a>: mean/max {mean:.2f}/{maximum:.2f}px</li>' for clip,camera,name,mean,maximum in examples)+'</ul>'
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Court RGB acceptance</title><style>body{font-family:sans-serif;background:#17191e;color:#eee}a{color:#8bd}td{padding:5px}</style><h1>固定参照・固定閾値の受入評価</h1><p>緑=採用、橙=拒否した診断H、magenta=承認済み参照。元解像度画像には採否・線別残差・coverageを表示。RGB失敗時に旧Hを代用していません。</p><h2>RGB誤採用</h2>'+false+'<h2>全69clip</h2>'+body+'<h2>元解像度・4比較条件＋新(a)単独</h2><table>'+''.join(index)+'</table>')
    (output/'manifest.json').write_text(json.dumps({'clips':len(clips),'cameras':len(records),'conditions':len(MODES),'images':len(records)*len(MODES)+len(clips),'rgb_false_accepts':examples},indent=2))


if __name__=='__main__':main()
