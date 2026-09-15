import { test } from "node:test";
import assert from "node:assert/strict";
import {
  fitScale,
  zoomAt,
  visiblePoints,
} from "../../../../../../src/tasks/base/visualization/detection/static/viewer.mjs";
test("fit preserves aspect and margin", () => {
  assert.equal(fitScale(1000, 500, 500, 500), 0.47);
  assert.equal(fitScale(500, 1000, 500, 500), 0.47);
});
test("invalid canvas dimensions are finite", () => {
  assert.equal(fitScale(0, 500, 500, 500), 1);
});
test("zoom anchors the cursor pixel", () => {
  const t = { x: 20, y: 30, scale: 2 };
  const n = zoomAt(t, 2, 100, 100);
  assert.equal((100 - t.x) / t.scale, (100 - n.x) / n.scale);
  assert.equal((100 - t.y) / t.scale, (100 - n.y) / n.scale);
});
test("zoom is bounded", () => {
  assert.equal(zoomAt({ x: 0, y: 0, scale: 1 }, 100, 0, 0).scale, 32);
  assert.equal(zoomAt({ x: 0, y: 0, scale: 1 }, 0, 0, 0).scale, 0.02);
});
test("absent and invalid points never draw", () => {
  assert.deepEqual(
    visiblePoints({
      points: [
        { x: 1, y: 2 },
        { x: 3, y: 4, visible: false },
        { x: NaN, y: 0 },
      ],
    }),
    [{ x: 1, y: 2 }],
  );
  assert.deepEqual(visiblePoints(null), []);
});
