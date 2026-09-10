"""Reopen canonical owners and write compact evidence after each completed run."""
from pathlib import Path
import json,sys
from collections import Counter
import numpy as np
from scipy.spatial import ConvexHull
from PIL import Image,ImageDraw
from src.synthetic_data_generation.alignment.validation import validate_alignment_outputs
from src.synthetic_data_generation.dataset.court.assembler import validate_court_dataset,CourtArrayValidationMode
from src.synthetic_data_generation.reconstruction.scene_export import validate_standard_scene_export
out=Path(__file__).resolve().parent
for name in sys.argv[1:]:
 root=Path('/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes')/name
 a=validate_alignment_outputs(root/'alignment')
 validate_court_dataset(root/'datasets/court',array_validation=CourtArrayValidationMode.HEADERS_ONLY)
 d=json.loads((root/'datasets/court/dataset.json').read_text());scene=validate_standard_scene_export(root/'reconstruction/export/scene.json');ref=a.layout.courts[0].court_from_scene
 captured=ref.apply(np.array([a.metric_adapter.metric_from_nht_camera(c.camera_to_scene).matrix()[:3,3] for c in scene.cameras]))
 generated=ref.apply(np.array([np.array(s['camera']['camera_to_scene']).reshape(4,4)[:3,3] for s in d['samples']]))
 hull=ConvexHull(captured[:,:2]);origin=hull.points[hull.vertices].mean(0);expanded=ConvexHull(origin+(hull.points-origin)*1.05)
 slack=-(generated[:,:2]@expanded.equations[:,:2].T+expanded.equations[:,2]).max(1)
 assert slack.min()>=.5-1e-7,(name,slack.min())
 metric={'scene':name,'schema':d['schema'],'alignment_schema':json.loads((root/'alignment/alignment.json').read_text())['schema'],'courts':len(a.layout.courts),'accepted_frames':len(d['samples']),'rejected_frames':len(d['rejected_samples']),'minimum_expanded_hull_clearance_m':float(slack.min()),'metrics':d['metrics'],'trajectory_shapes':dict(Counter(g['trajectory']['shape'] for g in d['trajectory_groups'])),'hfov_degrees':sorted({v['hfov_degrees'] for g in d['trajectory_groups'] for v in g['views']})}
 (out/f'{name}-verification.json').write_text(json.dumps(metric,indent=2))
 chosen=[d['samples'][i] for i in np.linspace(0,len(d['samples'])-1,16,dtype=int)]
 sheet=Image.new('RGB',(1280,816),'white');draw=ImageDraw.Draw(sheet)
 for i,s in enumerate(chosen):
  im=Image.open(root/'datasets/court'/s['rgb_preview']);im.thumbnail((320,180));x=i%4*320;y=i//4*204;sheet.paste(im,(x,y));draw.text((x+4,y+181),s['sample_id'],fill='black')
 sheet.save(out/f'{name}-preview.jpg')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,8))
 for h,label,color in [(hull,'Captured SfM hull','black'),(expanded,'Allowed hull (+5%, before 0.5 m margin)','tab:red')]:
  vertices=np.append(h.vertices,h.vertices[0]);ax.plot(h.points[vertices,0],h.points[vertices,1],label=label,color=color)
 ax.scatter(captured[:,0],captured[:,1],s=8,c='gray',alpha=.5,label='Captured cameras')
 pts=ax.scatter(generated[:,0],generated[:,1],s=5,c=generated[:,2],cmap='viridis',label='Generated cameras')
 for court in a.layout.courts:
  corners=ref.apply(court.scene_from_court.apply(np.array([[-5.485,-11.885,0],[5.485,-11.885,0],[5.485,11.885,0],[-5.485,11.885,0],[-5.485,-11.885,0]])))
  ax.plot(corners[:,0],corners[:,1],color='tab:green')
 ax.set_aspect('equal');ax.set_xlabel('Court-plane X (m)');ax.set_ylabel('Court-plane Y (m)');ax.set_title(f'{name}: {len(generated)} accepted poses');ax.legend(fontsize=7)
 fig.colorbar(pts,ax=ax,label='Height in reference court frame (m)');fig.tight_layout();fig.savefig(out/f'{name}-camera-coverage.png',dpi=140);plt.close(fig)
 print(name,metric['accepted_frames'],'verified',flush=True)
