from pathlib import Path
from collections import Counter
import json,sys
from dataclasses import replace
from omegaconf import OmegaConf
from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.synthetic_data_generation.alignment.validation import load_alignment_result
from src.synthetic_data_generation.reconstruction.scene_export import validate_standard_scene_export
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import build_court_dataset_plan
from src.synthetic_data_generation.dataset.court.components.labels import project_court_semantics_for_version,AmbiguousCameraRelativeNearFarError
out=Path(__file__).resolve().parent;name=sys.argv[1];r=Path('data/synthetic_data_generation/scenes')/name
runtime=ScenePipelineConfiguration.from_config(OmegaConf.load(out/f'{name}-requested.yaml'));a=load_alignment_result(r/'alignment/alignment.json');scene=validate_standard_scene_export(r/'reconstruction/export/scene.json')
results=[]
for fov in [(45.,90.),(60.,100.),(75.,110.),(90.,120.)]:
 cfg=replace(runtime.court,view=replace(runtime.court.view,hfov_degrees=fov))
 plan=build_court_dataset_plan(scene_id=name,profile=runtime.profile,cameras=scene.cameras,layout=a.layout,configuration=cfg,metric_adapter=a.metric_adapter)
 c=Counter()
 for sample in plan.samples:
  try:p=project_court_semantics_for_version(sample.camera,a.layout,schema_version=plan.schema_version)
  except AmbiguousCameraRelativeNearFarError:c['ambiguous']+=1;continue
  c['valid' if sum(x.in_frame_point_count for x in p.courts)>=4 else 'insufficient']+=1
 row={'hfov':fov,'proposals':len(plan.samples),'groups':len(plan.groups),**c};print(name,row,flush=True);results.append(row)
 (out/f'{name}-view-comparison.json').write_text(json.dumps(results,indent=2))
