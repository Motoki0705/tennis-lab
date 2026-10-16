import json,sys
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path.cwd()/'tests/benchmarks'))
from person_selection_cpu import calibration
from person_selection_refinement import load_saved
from src.tasks.person_tracking.court_linking import LinkingConfig,select_linked_candidates
from src.tasks.person_tracking.selection_metrics import selection_units,aggregate_units
from src.tasks.player_association.geometry.footpoints import FootpointConfig
from src.tasks.player_association.evaluation.labels import ClipLabels
p=Path('/home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/court_selection/i964-cpu-r4-20260929');r=json.loads((p/'selection.json').read_text());source=json.loads((p/'sources.json').read_text());sides=json.loads(Path(r['side_decisions']).read_text())
for rec in source['inputs']:
 if rec['camera']!='cam0':continue
 clip=rec['clip'];cam=rec['camera'];side=next(s for s in sides['clips'] if s['clip_id']==clip);turn=dict(zip(side['camera_ids'],side['annotation']['view_half_turns']))[cam]
 tr,prev=load_saved(r['records']['ft_base_0.01'][clip]['cameras'][cam],calibration(rec,turn));labels=ClipLabels.load(Path(rec['label_path']));fps=rec['video']['fps']
 for gap in [.5,1.,2.]:
  for margin in [.2, .05]:
   mask,d=select_linked_candidates(tr,fps,replace(LinkingConfig(),max_gap_s=gap,ambiguity_margin=margin),FootpointConfig());m=aggregate_units(selection_units(tr,mask,labels));print(clip,gap,margin,'player',m['player_kept_03'],'far',m['cam0_far_covered_03'],'nonplayer',m['non_player_kept_03'],'adj',m['adjacent_court_kept_03'],'links',len(d['links']), flush=True)
