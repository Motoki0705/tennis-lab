"""Reopen canonical owners and write compact evidence after each completed run."""
from pathlib import Path
import json,sys
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
 metric={'scene':name,'schema':d['schema'],'alignment_schema':json.loads((root/'alignment/alignment.json').read_text())['schema'],'courts':len(a.layout.courts),'accepted_frames':len(d['samples']),'rejected_frames':len(d['rejected_samples']),'minimum_expanded_hull_clearance_m':float(slack.min()),'metrics':d['metrics']}
 (out/f'{name}-verification.json').write_text(json.dumps(metric,indent=2))
 chosen=[d['samples'][i] for i in np.linspace(0,len(d['samples'])-1,16,dtype=int)]
 sheet=Image.new('RGB',(1280,816),'white');draw=ImageDraw.Draw(sheet)
 for i,s in enumerate(chosen):
  im=Image.open(root/'datasets/court'/s['rgb_preview']);im.thumbnail((320,180));x=i%4*320;y=i//4*204;sheet.paste(im,(x,y));draw.text((x+4,y+181),s['sample_id'],fill='black')
 sheet.save(out/f'{name}-preview.jpg');print(name,metric['accepted_frames'],'verified',flush=True)
