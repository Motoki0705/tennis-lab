"""In-memory old-policy mutations; never modify the evaluated production files."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import time

OUT=Path('outputs/tennis_scene/evaluate/court-semantic-support-20261117')

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 paint=load('paint_contract','tests/unit/tasks/court_detection/geometry/test_paint_refinement.py')
 support=load('support_contract','tests/unit/tasks/court_detection/geometry/test_hybrid_homography.py')
 changes=[]
 original=paint.refine_paint
 def old_paint(image,matrix,*args,**kwargs):
  return original(image,matrix,replace(paint.PaintConfig(),version='paired_edges_v1'))
 original_support=support.estimate_hybrid_homography
 def old_support(*args,**kwargs):
  kwargs['config']=replace(kwargs['config'],support_method='fixed_subsets');return original_support(*args,**kwargs)
 for module,name,mutation,restore,test in [
  (paint,'refine_paint',old_paint,original,paint.test_unconfirmed_net_tape_is_diagnostic_not_required_paint),
  (support,'estimate_hybrid_homography',old_support,original_support,support.test_new_support_candidate_recovers_after_fixed_set_becomes_inconsistent)]:
  start=time.monotonic();setattr(module,name,mutation);caught=False
  try:test()
  except AssertionError:caught=True
  finally:setattr(module,name,restore)
  assert caught,'mutation escaped contract';test()
  changes.append({'test':test.__name__,'mutation':mutation.__name__,'caught_by_assertion':caught,'restored_passed':True,'seconds':time.monotonic()-start})
 (OUT/'mutations.json').write_text(json.dumps(changes,indent=2));print(changes)
if __name__=='__main__':main()
