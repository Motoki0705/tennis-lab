"""Native RGB crops for every provisional false acceptance and representative changes."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from PIL import Image

import fit_reference as fit

BASE=Path(__file__).resolve().parent
PREVIOUS=BASE.parent/'court-acceptance-implementation-20261116'
WIDTH,HEIGHT=512,192


def draw(image,points,color):
    im=image.copy()
    for a,b in fit.EDGES:
        if np.isfinite(points[[a,b]]).all() and np.abs(points[[a,b]]).max()<1e6:
            cv2.line(im,tuple(np.rint(points[a]*16).astype(int)),tuple(np.rint(points[b]*16).astype(int)),color,1,cv2.LINE_AA,shift=4)
    return im


def main():
    cv2.setNumThreads(1);out=BASE/'remaining';out.mkdir(exist_ok=True)
    refs={c['camera']:c for c in json.loads((BASE/'reference/references.json').read_text())['cameras']};records=[];examples={}
    for path in sorted((BASE/'evaluation/meiji_4cam').glob('**/evaluation.json')):
        row=json.loads(path.read_text());cam=row['camera'];ref=refs[cam];xy=fit.project(np.array(ref['matrix_court_metres_to_image_pixels']))
        visible=(xy[:,0]>=0)&(xy[:,0]<=1919)&(xy[:,1]>=0)&(xy[:,1]<=1079)
        support=np.zeros(14,int)
        for i,line in enumerate(ref['lines']):
            if line['supported_for_fit']:support[fit.EDGES[i]]+=1
        reliable=visible&(support>=2)
        needs=[(name,c) for name,c in row['provisional'].items() if c['accepted'] and not c['correct']]
        if not needs:continue
        raw_meta=json.loads((PREVIOUS/'raw'/row['key']/'record.json').read_text());capture=cv2.VideoCapture(raw_meta['source_video']);ok,bgr=capture.read();capture.release();assert ok and bgr.shape==(1080,1920,3)
        assert hashlib.sha256(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==raw_meta['rgb_sha256']
        for name,value in needs:
            predicted=np.array(value['points']);distance=np.linalg.norm(predicted-xy,axis=1);indices=np.flatnonzero(reliable)
            if not len(indices):raise ValueError((row['key'],'no independently supported visible intersection for image review'))
            index=int(indices[np.argmax(distance[indices])]);cx,cy=xy[index];x=int(np.clip(round(cx-WIDTH/2),0,1920-WIDTH));y=int(np.clip(round(cy-HEIGHT/2),0,1080-HEIGHT))
            images=[bgr,draw(bgr,xy,(0,160,255)),draw(draw(bgr,xy,(0,160,255)),predicted,(255,0,255))]
            panels=[]
            for im in images:panels.append(im[y:y+HEIGHT,x:x+WIDTH])
            head=np.zeros((52,WIDTH*3,3),np.uint8)
            cv2.putText(head,f"PROVISIONAL {row['key']} {name}",(8,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(head,f"RAW | ORANGE REF | ORANGE REF + MAGENTA PRED; mean={value['mean_error_px']:.2f}px max={value['max_error_px']:.2f}px; visible KP{index} delta={distance[index]:.2f}px; x={x} y={y}",(8,41),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
            figure=np.vstack([head,np.hstack(panels)]);filename=f"{row['key'].replace('/','-')}-{name}.png";cv2.imwrite(str(out/filename),figure)
            entry={'key':row['key'],'camera':cam,'condition':name,'mean_error_px':value['mean_error_px'],'max_error_px':value['max_error_px'],'max_supported_visible_point_index':index,'supported_point_error_px':float(distance[index]),'bbox_xyxy':[x,y,x+WIDTH,y+HEIGHT],'image':filename,'status':'provisional_reference_requires_human_reapproval'};records.append(entry)
            if name=='postprocess_and_acceptance' and (cam not in examples or distance[index]>examples[cam]['supported_point_error_px']):examples[cam]=entry
    for cam,e in examples.items():
        im=Image.open(out/e['image']);im.quantize(colors=128).save(out/f'{cam}-remaining-example.png',optimize=True)
    (out/'manifest.json').write_text(json.dumps({'status':'provisional','examples':examples,'records':records},indent=2))
    table=''.join(f'<tr><td>{r["key"]}</td><td>{r["condition"]}</td><td>{r["mean_error_px"]:.2f}</td><td>{r["max_error_px"]:.2f}</td><td>{r["supported_point_error_px"]:.2f}</td><td><a href="{r["image"]}">元pixel crop</a></td></tr>' for r in records)
    (out/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>仮参照で残る採用誤り候補</title><h1>仮参照・未再承認</h1><p>参照のframe間変化と未観測の遠方baselineによる不確かさを含みます。各画像で塗装の白線とのずれを確認してください。画像は元解像度pixelのままです。</p><table border="1"><tr><th>入力</th><th>条件</th><th>平均px</th><th>最大px</th><th>支持のある可視交点のずれpx</th><th>画像</th></tr>'+table+'</table></html>')
    print('provisional false accept crops',len(records));print(json.dumps(examples,indent=2))


if __name__=='__main__':main()
