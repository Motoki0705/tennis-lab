"""Standalone evaluation figures from generated arrays and observed 2D references."""
from pathlib import Path
import json,sys
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
requests={p:json.loads((root/f'{p}.json').read_text()) for p in ('pipeline','dataset')}
clip=Path(requests['pipeline']['clip']);manifest=json.loads((clip/'clip.json').read_text())
manual=np.asarray(json.loads((clip/'annotations/manual_court_kp_result.json').read_text())['keypoints'])[:,0,:14]
out=root/'figures';out.mkdir(exist_ok=True)
paths={'pipeline':Path(requests['pipeline']['output'])/'scene.npz','dataset':clip/'annotations/tennis_scene/scene.npz'}
if len(sys.argv)>1:
 phase=sys.argv[1]
 if phase not in paths:raise ValueError(phase)
 paths={phase:paths[phase]}
fig,axes=plt.subplots(3,3,figsize=(17,10),sharex='col',layout='constrained')
for c,camera in enumerate(manifest['camera_ids']):
 annotation=json.loads((clip/'outsource'/f'{camera}_annotations.json').read_text())
 observed=np.array([f['status']=='observed' for f in annotation['frames']])
 target=np.array([[f['center_px']['x'],f['center_px']['y']] if f['status']=='observed' else [0,0] for f in annotation['frames']])
 for phase,path in paths.items():
  court=json.loads((Path(requests[phase]['output'])/'court_kp_result.json').read_text())
  points=np.asarray(court['keypoints'])[c,:,:14];valid=np.asarray(court['visibility'])[c,:,:14]>0
  distances=np.linalg.norm((points-manual[c])*[manifest['width']-1,manifest['height']-1],axis=-1)
  distances[~valid]=np.nan
  axes[0,c].plot(np.nanmean(distances,axis=-1),label=phase,alpha=.8,lw=1)
  with np.load(path) as a:
   visibility=a['ball_vis'][c].astype(bool);uv=a['ball_uv'][c]*[manifest['width']-1,manifest['height']-1]
   mask=observed&visibility
   ball_error=np.full(len(observed),np.nan);ball_error[mask]=np.linalg.norm(uv[mask]-target[mask],axis=-1)
   axes[1,c].plot(ball_error,'.',ms=2,label=phase,alpha=.7)
   axes[2,c].plot(np.flatnonzero(observed&~visibility),np.full((observed&~visibility).sum(),0 if phase=='pipeline' else 1),'|',label=phase)
   selected={0,505,1009}
   if mask.any():selected.add(int(np.nanargmax(ball_error)))
   missing=np.flatnonzero(observed&~visibility)
   if len(missing):selected.add(int(missing[0]))
   cap=cv2.VideoCapture(str(clip/'media'/f'{camera}.mp4'));panels=[]
   for frame_index in sorted(selected):
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index);ok,frame=cap.read()
    if not ok:raise RuntimeError(f'Cannot inspect {camera} source frame {frame_index}')
    for k,point in enumerate(manual[c]*[manifest['width']-1,manifest['height']-1]):
     cv2.circle(frame,tuple(np.rint(point).astype(int)),7,(255,0,255),2)
    for k,point in enumerate(points[frame_index]*[manifest['width']-1,manifest['height']-1]):
     if valid[frame_index,k]:cv2.drawMarker(frame,tuple(np.rint(point).astype(int)),(255,255,0),cv2.MARKER_CROSS,15,2)
    if observed[frame_index]:cv2.circle(frame,tuple(np.rint(target[frame_index]).astype(int)),13,(0,0,255),2)
    if visibility[frame_index]:cv2.drawMarker(frame,tuple(np.rint(uv[frame_index]).astype(int)),(0,255,0),cv2.MARKER_CROSS,26,3)
    if mask[frame_index]:cv2.line(frame,tuple(np.rint(target[frame_index]).astype(int)),tuple(np.rint(uv[frame_index]).astype(int)),(255,255,255),2)
    title=f'{camera} frame {frame_index} ball observed={observed[frame_index]} predicted={visibility[frame_index]}'
    cv2.putText(frame,title,(18,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
    cv2.putText(frame,'Court: magenta=manual cyan=model; ball: red=observed green=model',(18,65),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    panels.append(cv2.resize(frame,(640,360)))
   cap.release();cv2.imwrite(str(out/f'{phase}-{camera}-source-observations.jpg'),np.concatenate(panels,axis=1))
 axes[0,c].set_title(camera);axes[2,c].set_xlabel('Frame');axes[2,c].set_yticks([0,1],['pipeline','dataset'])
 for row in axes[:,c]:row.grid(alpha=.25)
axes[0,0].set_ylabel('Court mean distance (px)');axes[1,0].set_ylabel('Ball distance on observed frames (px)');axes[2,0].set_ylabel('Missing ball detections on observed frames')
axes[0,0].legend();fig.suptitle('2D reference consistency; manual court points are static, ball labels use observed frames only')
figure_name='observations.png' if len(paths)==2 else f'{next(iter(paths))}-observations.png'
fig.savefig(out/figure_name,dpi=140);plt.close(fig)

for phase,path in paths.items():
 with np.load(path) as a:
  positions=a['player_position'];aligned=a['gvhmr_aligned_player_position'];ball=a['ball_3d'];fps=float(a['fps'])
  fig,axes=plt.subplots(3,3,figsize=(17,10),layout='constrained')
  for p in range(2):
   for coordinate,label in enumerate(('X','Y','Z (height)')):
    axes[p,coordinate].plot(positions[p,:,coordinate],label='PLCS',lw=1)
    axes[p,coordinate].plot(aligned[p,:,coordinate],label='Aligned GVHMR',lw=1,alpha=.8)
    axes[p,coordinate].set_title(f'Player {p} - {label} (m)')
   axes[p,0].legend()
  for coordinate,label in enumerate(('X','Y','Z (height)')):
   axes[2,coordinate].plot(ball[:,coordinate],lw=1);axes[2,coordinate].set_title(f'Ball - {label} (m)')
  for ax in axes.flat:ax.grid(alpha=.25);ax.set_xlabel('Frame')
  fig.suptitle(f'{phase}: 3D predictions and alignment; no measured 3D ground truth')
  fig.savefig(out/f'{phase}-trajectories.png',dpi=140);plt.close(fig)
  fig,axes=plt.subplots(1,3,figsize=(17,4),layout='constrained')
  for p in range(2):
   axes[0].plot(np.linalg.norm(np.diff(positions[p],axis=0),axis=-1)*fps,label=f'PLCS {p}',lw=1)
   axes[1].plot(np.linalg.norm(np.diff(aligned[p],axis=0),axis=-1)*fps,label=f'GVHMR aligned {p}',lw=1)
  axes[2].plot(np.linalg.norm(np.diff(ball,axis=0),axis=-1)*fps,lw=1)
  for ax,title in zip(axes,('PLCS players','Aligned GVHMR players','BLCS ball'),strict=True):ax.set_title(title);ax.set_ylabel('Interframe speed (m/s)');ax.set_xlabel('Frame');ax.grid(alpha=.25)
  axes[0].legend();axes[1].legend();fig.suptitle(f'{phase}: discontinuity inspection, not 3D accuracy')
  fig.savefig(out/f'{phase}-speeds.png',dpi=140);plt.close(fig)
print(out)
