"""Lossless native-pixel crops for human reapproval; no prediction inputs."""
from pathlib import Path
import json
import html
import cv2
import numpy as np
from PIL import Image

import fit_reference as fit

BASE=Path(__file__).resolve().parent
WIDTH,HEIGHT=512,160
COLORS={'old':(255,80,0),'new':(0,160,255)}


def overlay(image,h,color):
    result=image.copy();points=fit.project(h)
    for a,b in fit.EDGES:
        cv2.line(result,tuple(np.rint(points[a]*16).astype(int)),tuple(np.rint(points[b]*16).astype(int)),color,1,cv2.LINE_AA,shift=4)
    return result


def midpoint(points,edge,fraction):
    a,b=points[list(edge)];visible,a,b=cv2.clipLine((0,0,1920,1080),tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)))
    if not visible:raise ValueError(('no visible review segment',edge))
    return (1-fraction)*np.array(a)+fraction*np.array(b)


def main():
    references=json.loads((BASE/'reference/references.json').read_text());frames=json.loads((BASE/'frames.json').read_text())['records'];out=BASE/'review';out.mkdir(exist_ok=True)
    manifest={'status':'PROVISIONAL / 未再承認','size_hw':[1080,1920],'crop_pixels_unscaled':True,'panels':[]}
    pages=[]
    for camera in references['cameras']:
        cam=camera['camera'];old=np.array(camera['initial_matrix']);new=np.array(camera['matrix_court_metres_to_image_pixels']);points=fit.project(new)
        centres={'near_baseline':midpoint(points,(2,3),.35),'net_side':midpoint(points,(8,9),.5),
            'left_sideline':midpoint(points,(0,2),.60),'right_sideline':midpoint(points,(1,3),.60)}
        for clip in ('000','034','068'):
            record=next(r for r in frames if r['camera']==cam and r['clip_id']==f'video_000/clip_{clip}' and r['frame_index']==0)
            raw=cv2.imread(str(BASE/record['path']));images={'raw':raw,'old':overlay(raw,old,COLORS['old']),'new':overlay(raw,new,COLORS['new'])};images['both']=overlay(images['old'],new,COLORS['new'])
            tiles=[];panels=[]
            for zone,centre in centres.items():
                x=int(np.clip(round(centre[0]-WIDTH/2),0,1920-WIDTH));y=int(np.clip(round(centre[1]-HEIGHT/2),0,1080-HEIGHT));paths={}
                for mode,im in images.items():
                    target=out/f'{cam}-clip{clip}-{zone}-{mode}.png';cv2.imwrite(str(target),im[y:y+HEIGHT,x:x+WIDTH]);paths[mode]=target.name
                head=np.zeros((32,WIDTH*3,3),np.uint8)
                for j,label in enumerate(('RAW','OLD BLUE','PROVISIONAL ORANGE')):
                    cv2.putText(head,f'{zone} {label} x={x} y={y}',(j*WIDTH+8,21),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
                tiles.append(np.vstack([head,np.hstack([images[m][y:y+HEIGHT,x:x+WIDTH] for m in ('raw','old','new')])]))
                item={'camera':cam,'clip':clip,'zone':zone,'bbox_xyxy':[x,y,x+WIDTH,y+HEIGHT],'images':paths,'source':record['path']};manifest['panels'].append(item);panels.append(item)
            grid=np.vstack(tiles);cv2.imwrite(str(out/f'{cam}-clip{clip}-native-comparison.png'),grid)
            if clip=='000':
                Image.fromarray(cv2.cvtColor(grid,cv2.COLOR_BGR2RGB)).quantize(colors=96).save(out/f'{cam}-overview.png',optimize=True)
            title=f'{cam} / clip_{clip} / frame 0'
            pages.append(f'<section><h2>{title}</h2><div class="panels">'+''.join(f'<figure><figcaption>{html.escape(p["zone"])}: {p["bbox_xyxy"]}</figcaption><img width="512" height="160" data-images="{html.escape(json.dumps(p["images"]),quote=True)}" src="{p["images"]["raw"]}"></figure>' for p in panels)+'</div></section>')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    document='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>白線参照の再確認（仮）</title>
<style>body{background:#171717;color:#eee;font-family:sans-serif;margin:18px}header{position:sticky;top:0;background:#222;padding:12px;z-index:2;border:2px solid #e98}.panels{display:grid;grid-template-columns:repeat(2,max-content);gap:12px}figure{margin:0}img{display:block;image-rendering:pixelated}figcaption{font-size:14px;padding:6px}button,select{font-size:16px;margin:5px}section{overflow:auto;margin:24px 0}a{color:#acd}</style>
<header><strong>仮参照・未再承認。推論結果はこのページに含めていません。</strong><br>
元画像1920×1080から切り出したlossless PNG。旧参照=青、新参照=橙。倍率は表示だけを変えます。<br>
<button onclick="mode('raw')">原画像</button><button onclick="mode('old')">旧参照 青</button><button onclick="mode('new')">仮参照 橙</button><button onclick="mode('both')">両方</button>
倍率 <select onchange="zoom(this.value)"><option value="1">100%（元pixel）</option><option value="2" selected>200%</option><option value="4">400%</option></select>
明るさ <input type="range" min="100" max="250" value="100" oninput="document.querySelectorAll('img').forEach(i=>i.style.filter='brightness('+this.value+'%)')">（表示のみ）</header>
<p>遠方ベースラインはネット白帯との重なりのため直接測れておらず、他の地面線から外挿しています。cam1・cam3では数pxのframe間変化があり、1つのHは全frameへ完全には一致しません。cam3 clip_068 frame 0は線の幾何一致不足で参照fitへの使用を除外しました。</p>
'''+''.join(pages)+'''<script>function mode(m){document.querySelectorAll('img[data-images]').forEach(i=>i.src=JSON.parse(i.dataset.images)[m])}function zoom(s){document.querySelectorAll('img').forEach(i=>{i.style.width=(512*s)+'px';i.style.height=(160*s)+'px'})}zoom(2);</script></html>'''
    (out/'index.html').write_text(document)
    print('native review panels',len(manifest['panels']),'contact sheets',len(pages))


if __name__=='__main__':main()
