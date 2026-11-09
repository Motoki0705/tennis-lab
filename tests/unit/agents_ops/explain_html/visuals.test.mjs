import assert from 'node:assert/strict';
import {test} from 'node:test';
import renderer from '../../../../.agents/ops/explain_html/renderers/visuals-v2.js';
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
  class Element {
    constructor(tag, attributes = {}) { this.tagName = tag; this.attributes = attributes; this.children = []; this.listeners = {}; }
    getAttribute(name) { return this.attributes[name] ?? null; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    append(...nodes) { for (const node of nodes) { node.parentElement = this; this.children.push(node); } }
    replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
    querySelector(selector) { return this.parts?.[selector] ?? null; }
    addEventListener(name, handler) { this.listeners[name] = handler; }
  }
  const make = (id, definition) => {
    const figure = new Element('FIGURE', {'data-figure-id': id});
    const container = new Element('DIV', {'data-managed-graph': '2', 'data-definition': definition});
    container.parts = Object.fromEntries(['[data-plot]', '[data-values]', '[data-slider-value]', '[data-managed-slider]'].map(name => [name, new Element('DIV')]));
    figure.append(container);
    return {figure, container};
  };
  const goodDefinition = JSON.stringify(stereo);
  const bad = make('bad', '{}'), good = make('good', goodDefinition);
  const rogue = new Element('DIV', {'data-managed-graph': '1', 'data-definition': '{}'});
  const scope = {querySelectorAll(selector) { return selector.startsWith('figure') ? [bad.figure, good.figure] : [rogue, bad.container, good.container]; }};
  const previous = globalThis.document;
  globalThis.document = {createElement: tag => new Element(tag), createElementNS: (_, tag) => new Element(tag)};
  try {
    let errors;
    assert.doesNotThrow(() => { errors = renderer.mount(scope, [{figure_id: 'bad', definition: '{}'}, {figure_id: 'good', definition: goodDefinition}]); });
    assert.deepEqual(errors, ['bad']);
    assert.equal(bad.container.getAttribute('data-render-error'), 'invalid_graph');
    assert.equal(good.container.parts['[data-slider-value]'].textContent, '0 px; Depth = 16 m');
    assert.equal(typeof good.container.parts['[data-managed-slider]'].listeners.input, 'function');
    assert.equal(rogue.getAttribute('data-render-error'), null);
  } finally { if (previous === undefined) delete globalThis.document; else globalThis.document = previous; }
});
