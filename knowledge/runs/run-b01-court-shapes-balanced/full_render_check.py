"""Render every geometrically valid B01 bounded view; retain compact RGB evidence.

The input is explicitly human-confirmed geometry, not a canonical alignment
owner. No source dataset or alignment artifact is mutated or published here.
"""
from pathlib import Path
import json
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
from src.synthetic_data_generation.scene_contract import SceneCamera, CourtInstance, MultiCourtLayout, RigidTransform
from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.dataset.court.components.labels import project_court_semantics_for_version, attach_renderer_visibility, AmbiguousCameraRelativeNearFarError
from src.synthetic_data_generation.dataset.court.schema import CourtDatasetSchemaVersion
from src.synthetic_data_generation.rendering.nht import NHTRenderClient, NHTRenderCamera, NHTRenderCommandRequest, NHTRenderRequest

root=Path(__file__).resolve().parent
scene_root=Path('/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes/B01')
g=json.loads((scene_root/'alignment/court-geometry.json').read_text())
assert g['schema']=='human_confirmed_court_geometry_v1'
courts=[]
for c in g['layout']['courts']:
 assert c['fit_status']==c['holdout_status']=='human_confirmed'
 courts.append(CourtInstance(court_instance_id=c['court_instance_id'],candidate_id=c['candidate_id'],scene_from_court=RigidTransform(tuple(c['scene_from_court'])),court_from_scene=RigidTransform(tuple(c['court_from_scene'])),fit_status='accepted',holdout_status='accepted',fit_metrics={'experiment_geometry_source':'human_confirmed'},holdout_metrics={'experiment_geometry_source':'human_confirmed'}))
layout=MultiCourtLayout(tuple(courts),tuple(g['layout']['complex_bounds_scene']),g['layout']['primary_court_instance_id'])
adapter=MetricSceneAdapter.from_dict(g['metric_scene_adapter'])
plan=json.loads((root/'sfm_bounded-plan.json').read_text())
client=NHTRenderClient();valid=[];rejected=[]
for s in plan['samples']:
 camera=SceneCamera.from_dict(s['camera'])
 try:
  projection=project_court_semantics_for_version(camera,layout,schema_version=CourtDatasetSchemaVersion.V3)
 except AmbiguousCameraRelativeNearFarError:
  rejected.append({'sample_id':s['sample_id'],'reason':'ambiguous_near_far'});continue
 if sum(c.in_frame_point_count for c in projection.courts)<4:
  rejected.append({'sample_id':s['sample_id'],'reason':'insufficient_semantic_coverage'});continue
 valid.append((s,camera,projection))
rows=[];accepted_by_group=Counter();coverage=Counter();classes=Counter()
for batch_index,start in enumerate(range(0,len(valid),256)):
 batch=valid[start:start+256]
 cameras=tuple(NHTRenderCamera(camera_id=c.camera_id,width=c.width,height=c.height,intrinsics=c.intrinsics,camera_to_scene=adapter.nht_from_metric_camera(c.camera_to_scene)) for _,c,_ in batch)
 result=client.render(NHTRenderCommandRequest(scene_path=scene_root/'reconstruction/export/scene.json',output_directory=root/f'renders-{batch_index:02d}',arbitrary_cameras=NHTRenderRequest(cameras),arbitrary_request_path=root/f'request-{batch_index:02d}.json'),environment={'CUDA_VISIBLE_DEVICES':'0'},timeout_seconds=1800)
 by_id={record.camera_id:record for record in result.records}
 for sample,camera,projection in batch:
  record=by_id[camera.camera_id];alpha=np.load(record.alpha_path);depth=np.load(record.depth_path)
  visible=attach_renderer_visibility(projection,alpha=alpha,depth=depth)
  accepted=visible.visible_point_count>0
  rows.append({'sample_id':sample['sample_id'],'group_id':sample['trajectory_group_id'],'accepted':accepted,'visible_point_count':visible.visible_point_count,'alpha_below_0_5_fraction':float((alpha<0.5).mean()),'rgb_preview':str(record.rgb_preview_path)})
  if accepted:
   accepted_by_group[sample['trajectory_group_id']]+=1
   coverage.update(c.coverage_mode for c in visible.courts)
   classes.update(visible.visible_class_names)
  # Verification keeps original-resolution RGB PNGs and numeric summary;
  # remove only this experiment's arrays after public-client validation/scoring.
  for path in (record.rgb_path,record.alpha_path,record.depth_path,record.alpha_preview_path):
   assert path.resolve().is_relative_to(root)
   path.unlink()
 progress={'rendered_count':len(rows),'pre_render_rejected_count':len(rejected),'accepted_count':sum(r['accepted'] for r in rows),'proposals':len(plan['samples']),'accepted_by_group':dict(accepted_by_group),'coverage':dict(coverage),'visible_classes':dict(classes),'frames':rows,'pre_render_rejected':rejected}
 (root/'full-render-metrics.json').write_text(json.dumps(progress,indent=2))
 print('PROGRESS',len(rows),len(valid),progress['accepted_count'],flush=True)
accepted=progress['accepted_count'];gates={'minimum_frames':accepted>=2000,'minimum_fraction':accepted/len(plan['samples'])>=0.9,'every_group':all(accepted_by_group[g['trajectory']['trajectory_group_id']]>0 for g in plan['groups']),'coverage_modes':{'full','near_full','partial'}<=set(coverage),'all_kp14_classes':len(classes)==14}
progress['acceptance_fraction']=accepted/len(plan['samples']);progress['gates']=gates
(root/'full-render-metrics.json').write_text(json.dumps(progress,indent=2))
# Representative all-group sheets: 4 accepted render positions per group.
selected=[]
for group_id in accepted_by_group:
 items=[r for r in rows if r['group_id']==group_id]
 selected.extend(items[i] for i in np.linspace(0,len(items)-1,4,dtype=int))
for page,start in enumerate(range(0,len(selected),32)):
 sheet=Image.new('RGB',(1280,1632),'white');draw=ImageDraw.Draw(sheet)
 for j,row in enumerate(selected[start:start+32]):
  im=Image.open(row['rgb_preview']);im.thumbnail((320,180));x=(j%4)*320;y=(j//4)*204;sheet.paste(im,(x,y));draw.text((x+4,y+181),row['sample_id'],fill='black')
 sheet.save(root/f'contact-{page}.jpg')
print('FINAL',accepted,len(plan['samples']),gates,flush=True)
assert all(gates.values()),gates
