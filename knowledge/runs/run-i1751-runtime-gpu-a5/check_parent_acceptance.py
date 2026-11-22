"""CPU boundary and fault checks for the explicit parent acceptance policy."""
import copy
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from assess_parent_acceptance import assess, position_error

started = time.perf_counter()
root = Path('outputs/tennis_scene/evaluate/runtime-i1751-20261118-a5').resolve()
policy = json.loads(Path(__file__).with_name('parent-acceptance-policy.json').read_text())
assert np.isclose(position_error(np.array([[0., 0.]]), np.array([[.3, .4]]))['max_l2'], .5)
assert position_error(np.array([[0., 0.]]), np.array([[.4, .4]]))['max_l2'] > .5
assert np.isclose(position_error(np.array([[0., 0., 0.]]), np.array([[.0006, .0008, 0.]]))['max_l2'], .001)
assert assess(root, policy)['parent_acceptance']
with tempfile.TemporaryDirectory() as temporary:
    work = Path(temporary)
    for variant in ['main', 'runtime']:
        (work / variant).symlink_to(root / variant, target_is_directory=True)
    for name in ['inputs.json', 'acceptance-summary.json', 'terminal.json', 'comparison.json']:
        (work / name).write_bytes((root / name).read_bytes())
    original = json.loads((root / 'comparison-completed.json').read_text())
    for kind in ['non_ball', 'mask']:
        comparison = copy.deepcopy(original)
        name = 'person_detection/cam0' if kind == 'non_ball' else 'ball_detection/cam0'
        row = next(value for value in comparison['components'][name]['arrays'].values()
                   if np.dtype(value['dtype_main']).kind in ('fc' if kind == 'non_ball' else 'b'))
        row.update(exact=False, close=False)
        (work / 'comparison-completed.json').write_text(json.dumps(comparison))
        result = assess(work, policy)
        assert not result['parent_acceptance']
        criterion = 'non_ball_original_float_rule' if kind == 'non_ball' else 'all_discrete_observed_masks_exact'
        assert not result['checks'][criterion]
print(json.dumps({'status': 'passed', 'checks': 6, 'seconds': time.perf_counter() - started}))
