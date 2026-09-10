"""B01 experiment: render deterministic stratified samples of saved plans."""
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw
from src.synthetic_data_generation.scene_contract import SceneCamera
from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.rendering.nht import NHTRenderClient, NHTRenderCamera, NHTRenderCommandRequest, NHTRenderRequest

root=Path(__file__).resolve().parent
scene_root=Path('/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes/B01')
g=json.loads((scene_root/'alignment/court-geometry.json').read_text())
adapter=MetricSceneAdapter.from_dict(g['metric_scene_adapter'])
client=NHTRenderClient()
metrics={}
for selector in ['v3','sfm_bounded']:
 plan=json.loads((root/f'{selector}-plan.json').read_text())
 groups=defaultdict(list)
 for sample in plan['samples']:
  groups[sample['trajectory_group_id']].append(sample)
 # Four angular positions in each group: complete group inventory, no quality-based selection.
 selected=[items[i] for items in groups.values() for i in np.linspace(0,len(items)-1,4,dtype=int)]
 cameras=[]
 for sample in selected:
  camera=SceneCamera.from_dict(sample['camera'])
  cameras.append(NHTRenderCamera(camera_id=camera.camera_id,width=camera.width,height=camera.height,intrinsics=camera.intrinsics,camera_to_scene=adapter.nht_from_metric_camera(camera.camera_to_scene)))
 request=NHTRenderCommandRequest(scene_path=scene_root/'reconstruction/export/scene.json',output_directory=root/f'{selector}-renders',arbitrary_cameras=NHTRenderRequest(tuple(cameras)),arbitrary_request_path=root/f'{selector}-request.json')
 result=client.render(request,timeout_seconds=1800)
 rows=[]
 for record in result.records:
  alpha=np.load(record.alpha_path);depth=np.load(record.depth_path)
  rows.append({'camera_id':record.camera_id,'alpha_below_0_5_fraction':float((alpha<0.5).mean()),'alpha_below_0_9_fraction':float((alpha<0.9).mean()),'zero_depth_fraction':float((depth<=0).mean())})
 metrics[selector]={'rendered_count':len(rows),'mean_alpha_below_0_5_fraction':float(np.mean([x['alpha_below_0_5_fraction'] for x in rows])),'mean_alpha_below_0_9_fraction':float(np.mean([x['alpha_below_0_9_fraction'] for x in rows])),'frames':rows}
 (root/'render-metrics.json').write_text(json.dumps(metrics,indent=2))
 # Contact sheets include every sampled view, labelled for follow-up inspection.
 for page,start in enumerate(range(0,len(result.records),32)):
  batch=result.records[start:start+32];sheet=Image.new('RGB',(320*4,204*8),'white');draw=ImageDraw.Draw(sheet)
  for j,record in enumerate(batch):
   im=Image.open(record.rgb_preview_path).convert('RGB');im.thumbnail((320,180));x=(j%4)*320;y=(j//4)*204;sheet.paste(im,(x,y));draw.text((x+4,y+181),f'{selector} {record.camera_id}',fill='black')
  sheet.save(root/f'{selector}-contact-{page}.jpg')
 print(selector,metrics[selector]['rendered_count'],metrics[selector]['mean_alpha_below_0_5_fraction'],flush=True)
