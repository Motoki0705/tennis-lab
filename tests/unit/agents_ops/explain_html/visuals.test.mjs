import assert from 'node:assert/strict';
import {test} from 'node:test';
import renderer from '../../../../.agents/ops/explain_html/renderers/visuals-v1.js';
const {series} = renderer;

const stereo = {
  type: 'observation_error',
  x: {label: 'Observation error', unit: 'px', domain: [-10, 10]},
  y: {label: 'Depth', unit: 'm', domain: [0, 26]},
  data: {focal_px: 600, baseline_m: 0.8, disparity_px: 30, sample_errors_px: [-10, 0, 10]},
  slider: {label: 'Observation error', unit: 'px', min: -10, max: 10, step: 1, initial: 0}
};

test('signed disparity error: hand-calculated depths at three slider values', () => {
  // f B = 480 px*m; denominators 20, 30, 40 px -> 24, 16, 12 m.
  for (const [input, expected] of [[-10, 24], [0, 16], [10, 12]]) {
    const rows = series(stereo, input);
    assert.equal(rows[0].points.find(p => p.x === input).y, expected);
    assert.deepEqual(rows[0].points.map(p => p.y), [24, 16, 12]);
  }
});

test('comparison preserves asymmetric x/y coordinates and series identity', () => {
  const rows = series({type: 'comparison', series: [
    {label: 'A', points: [{x: 1, y: 7}, {x: 8, y: 3}]},
    {label: 'B', points: [{x: 2, y: 4}, {x: 7, y: 11}]}
  ]});
  assert.deepEqual(rows, [
    {label: 'A', points: [{x: 1, y: 7}, {x: 8, y: 3}]},
    {label: 'B', points: [{x: 2, y: 4}, {x: 7, y: 11}]}
  ]);
});

test('unknown executable graph type, nonfinite values and out-of-range input fail', () => {
  assert.throws(() => series({type: 'javascript', formula: 'process.exit(0)'}));
  assert.throws(() => series(stereo, Infinity));
  assert.throws(() => series(stereo, -11));
  assert.throws(() => series(stereo, '0'));
});
