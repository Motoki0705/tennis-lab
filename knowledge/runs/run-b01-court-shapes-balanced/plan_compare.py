from pathlib import Path
import json
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial import ConvexHull
from src.synthetic_data_generation.scene_contract import CourtInstance, MultiCourtLayout, RigidTransform
from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.configuration import CourtDatasetConfiguration
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import build_court_dataset_plan
from src.synthetic_data_generation.reconstruction.scene_export import validate_standard_scene_export
r=Path('/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes/B01')
out=Path('/home/kamimura/projects/tennis-lab-worktrees/court-sfm-bounded/outputs/b01-court-shapes-balanced');out.mkdir(parents=True,exist_ok=True)
g=json.loads((r/'alignment/court-geometry.json').read_text())
assert g['schema']=='human_confirmed_court_geometry_v1'
# Explicit experiment-only interpretation of human-confirmed geometry. No
# canonical alignment owner or acceptance evidence is written or upgraded.
cs=[]
for c in g['layout']['courts']:
 assert c['fit_status']==c['holdout_status']=='human_confirmed'
 cs.append(CourtInstance(court_instance_id=c['court_instance_id'],candidate_id=c['candidate_id'],scene_from_court=RigidTransform(tuple(c['scene_from_court'])),court_from_scene=RigidTransform(tuple(c['court_from_scene'])),fit_status='accepted',holdout_status='accepted',fit_metrics={'experiment_geometry_source':'human_confirmed'},holdout_metrics={'experiment_geometry_source':'human_confirmed'}))
l=MultiCourtLayout(tuple(cs),tuple(g['layout']['complex_bounds_scene']),g['layout']['primary_court_instance_id']);a=MetricSceneAdapter.from_dict(g['metric_scene_adapter']);scene=validate_standard_scene_export(r/'reconstruction/export/scene.json')
plans={}
for selector in ['sfm_bounded']:
 with initialize_config_dir(version_base='1.3',config_dir=str(Path('src/synthetic_data_generation/configs').resolve())):
  cfg=compose(config_name='run_scene_pipeline',overrides=[f'dataset/court={selector}'])
 config=CourtDatasetConfiguration.from_mapping(OmegaConf.to_container(cfg.dataset.court,resolve=True))
 plan=build_court_dataset_plan(scene_id='B01',profile='b01',cameras=scene.cameras,layout=l,configuration=config,metric_adapter=a)
 plans[selector]=plan
 (out/f'{selector}-plan.json').write_text(json.dumps(plan.to_dict(),indent=2))
 ref=l.courts[0].court_from_scene
 captured=ref.apply(np.stack([a.metric_from_nht_camera(c.camera_to_scene).matrix()[:3,3] for c in scene.cameras]))
 generated=ref.apply(np.stack([s.camera_center_scene_m for s in plan.samples]))
 h=ConvexHull(captured[:,:2]);slack=-(generated[:,:2]@h.equations[:,:2].T+h.equations[:,2]).max(1)
 print(selector,'groups',len(plan.groups),'samples',len(plan.samples),'outside',sum(slack<0),'minimum_slack',slack.min(),flush=True)
 np.savez(out/f'{selector}-positions.npz',captured=captured,generated=generated,slack=slack)
np.save(out/'metric_to_nht.npy',a.nht_matrix())
from collections import Counter
from src.synthetic_data_generation.dataset.court.components.labels import project_court_semantics_for_version, AmbiguousCameraRelativeNearFarError
summary={}
for selector,plan in plans.items():
 counts=Counter();coverage=Counter()
 for sample in plan.samples:
  try:
   projection=project_court_semantics_for_version(sample.camera,l,schema_version=plan.schema_version)
  except AmbiguousCameraRelativeNearFarError:
   counts['ambiguous']+=1
   continue
  counts['pre_render_accepted' if sum(c.in_frame_point_count for c in projection.courts)>=4 else 'pre_render_rejected']+=1
  coverage.update(c.coverage_mode for c in projection.courts)
 positions=np.load(out/f'{selector}-positions.npz')
 summary[selector]={'groups':len(plan.groups),'proposals':len(plan.samples),'outside_sfm_hull_count':int((positions['slack']<0).sum()),'minimum_boundary_clearance_m':float(positions['slack'].min()),'geometric_gates':dict(counts),'court_coverage':dict(coverage),'base_heights_m':sorted({g.trajectory.base_height_m for g in plan.groups}),'radius_scales':sorted({g.trajectory.radius_scale for g in plan.groups}),'axis_ratios':sorted({g.trajectory.axis_ratio for g in plan.groups})}
 print(selector,summary[selector],flush=True)
(out/'plan-metrics.json').write_text(json.dumps(summary,indent=2))
