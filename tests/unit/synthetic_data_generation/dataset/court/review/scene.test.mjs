import test from "node:test";
import assert from "node:assert/strict";
import {
  project,
  distanceToSegment,
} from "../../../../../../src/synthetic_data_generation/dataset/court/review/static/scene.mjs";
test("3D projection uses height and camera orbit", () => {
  const v = {
    center: [0, 0, 0],
    yaw: 0,
    pitch: Math.PI / 4,
    scale: 10,
    distance: 100,
    width: 800,
    height: 600,
    pan: [0, 0],
  };
  assert.deepEqual(project([0, 0, 0], v), [400, 300, 0]);
  assert(project([0, 0, 3], v)[1] < 300);
  assert(project([5, 0, 0], v)[0] > 400);
  assert(
    Math.abs(project([5, 0, 0], { ...v, yaw: Math.PI / 2 })[0] - 400) < 1e-9,
  );
});
test("trajectory picking uses segments including degenerate samples", () => {
  assert.equal(distanceToSegment([5, 3], [0, 0], [10, 0]), 3);
  assert.equal(distanceToSegment([12, 0], [0, 0], [10, 0]), 2);
  assert.equal(distanceToSegment([3, 4], [0, 0], [0, 0]), 5);
});
