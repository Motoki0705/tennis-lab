"""CPU contracts for array and JSON comparison; no inference or GPU query."""
import json
from pathlib import Path

import numpy as np
from compare_completed import arrays, json_values

assert arrays(np.array('metadata'), np.array('metadata'))['exact']
assert not arrays(np.array('metadata'), np.array('different'))['close']
assert arrays(np.array([1., np.nan]), np.array([1., np.nan]))['exact']
assert not arrays(np.array([1, 2]), np.array([1, 3]))['close']
assert arrays(np.array([1., 2.]), np.array([1., 2. + 1e-6]))['close']
cases = [({'ids': [2, 7], 'ok': True}, {'ids': [2, 7], 'ok': True}, True, True),
         ({'value': 2.0}, {'value': 2.000001}, False, True),
         ({'value': 2.0}, {'value': 2.0000205}, False, True),
         ({'value': 2.0}, {'value': 2.1}, False, False),
         ({'ok': True}, {'ok': 1}, False, False),
         ({'ids': [2, 7]}, {'ids': [7, 2]}, False, False),
         ({'id': 2}, {'id': 2, 'new': 0}, False, False)]
for left, right, exact, close in cases:
    result = json_values(left, right)
    assert (result['exact'], result['close']) == (exact, close)
print(json.dumps({'status': 'passed', 'array_checks': 5, 'json_checks': len(cases)}))
