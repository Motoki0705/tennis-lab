import test from "node:test";
import assert from "node:assert/strict";
import {
  project,
  distanceToSegment,
  trajectoryPolyline,
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
test("planned open paths retain endpoints and rejected gaps without connecting accepted samples", () => {
  const planned = [[0,0,2], [1,1,2], [2,0,2]];
  const group = {planned_points: planned, points: [planned[0], planned[2]], samples: [{frame:0}, {frame:2}]};
  assert.equal(trajectoryPolyline(group), planned);
  assert.deepEqual(trajectoryPolyline({...group, points: [], samples: []}), planned);
  assert.notDeepEqual(planned[0], planned.at(-1));
  assert.deepEqual(trajectoryPolyline({points:[[0,0,0],[0,0,0],[1,0,0]], samples:[{frame:0},{frame:0},{frame:1}]}), [[0,0,0],[1,0,0]]);
});
test("trajectory picking uses segments including degenerate samples", () => {
  assert.equal(distanceToSegment([5, 3], [0, 0], [10, 0]), 3);
  assert.equal(distanceToSegment([12, 0], [0, 0], [10, 0]), 2);
  assert.equal(distanceToSegment([3, 4], [0, 0], [0, 0]), 5);
});
