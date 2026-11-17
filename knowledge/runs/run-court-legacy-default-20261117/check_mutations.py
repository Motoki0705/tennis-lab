"""In-memory compatibility mutations; production source stays frozen for replay."""
from dataclasses import replace
from functools import partial
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import ModuleType

sys.path.insert(0, str(Path.cwd()))
import pytest

OUT=Path('outputs/tennis_scene/evaluate/court-legacy-default-20261117')

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def main():
    tests = ModuleType('tests')
    tests.__path__ = [str(Path('tests').resolve())]
    sys.modules['tests'] = tests
    gate=load('gate_contract','tests/unit/tasks/court_detection/inference/test_regions.py')
    support=load('support_contract','tests/unit/tasks/court_detection/geometry/test_hybrid_homography.py')
    results=[]
    original=gate.CourtRegionSearchConfig
    selector=support.select_keypoints
    mutations=[(gate,'CourtRegionSearchConfig',lambda **kwargs: replace(original(**kwargs),acceptance_method='quality'),
                gate.test_default_search_preserves_visibility_inlier_gate_and_stable_confidence_ranking,True),
               (support,'select_keypoints',partial(selector,spatial=True),
                support.test_gate_excludes_geometric_and_line_outliers_and_never_uses_all_points,False)]
    for module,attribute,mutation,test,needs_patch in mutations:
        start=time.monotonic();restore=getattr(module,attribute);caught=None
        try:
            setattr(module,attribute,mutation)
            with pytest.MonkeyPatch.context() as patch:
                test(patch) if needs_patch else test()
        except AssertionError as error:caught=str(error)
        finally:setattr(module,attribute,restore)
        assert caught is not None,'Mutation escaped contract'
        with pytest.MonkeyPatch.context() as patch:
            test(patch) if needs_patch else test()
        results.append({'test':test.__name__,'mutation':attribute,'caught_assertion':caught,
                        'restored_passed':True,'seconds':time.monotonic()-start})
    (OUT/'mutations.json').write_text(json.dumps(results,indent=2)+'\n')
    print(results)

if __name__=='__main__':main()
