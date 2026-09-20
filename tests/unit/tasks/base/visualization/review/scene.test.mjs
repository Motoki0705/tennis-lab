// Unit tests for the shared review renderer math (node --test).

import assert from "node:assert/strict";
import test from "node:test";

import {
  NEAR,
  cameraBasis,
  clipPolygonNear,
  clipSegment,
  dot,
  frustumEdges,
  length,
  projectPoint,
  projectPolygon,
  projectPolyline,
  sortFarToNear,
  strideFor,
  subtract,
  toCamera,
} from "../../../../../../src/tasks/base/visualization/review/static/scene.mjs";

const HALF_FOV = (36 * Math.PI) / 360;

function viewFor({ distance, target = [0, 0, 0], yaw = 0, pitch = 0, height = 852 }) {
  return {
    yaw,
    pitch,
    distance,
    target,
    halfFov: HALF_FOV,
    focal: height / 2 / Math.tan(HALF_FOV),
    cx: 0,
    cy: height / 2,
  };
}

test("cameraBasis is an orthonormal, right-handed frame", () => {
  const view = viewFor({ distance: 5, target: [1, 2, 0.9], yaw: -0.62, pitch: 0.32 });
  const camera = cameraBasis(view);
  assert.ok(Math.abs(length(camera.forward) - 1) < 1e-9);
  assert.ok(Math.abs(length(camera.right) - 1) < 1e-9);
  assert.ok(Math.abs(length(camera.up) - 1) < 1e-9);
  assert.ok(Math.abs(dot(camera.right, camera.up)) < 1e-9);
  assert.ok(Math.abs(dot(camera.right, camera.forward)) < 1e-9);
  assert.ok(Math.abs(dot(camera.up, camera.forward)) < 1e-9);
  const offset = subtract(camera.eye, view.target);
  assert.ok(Math.abs(dot(offset, camera.forward) + view.distance) < 1e-9);
});

test("projectPoint centres the target and keeps world +Z up", () => {
  const view = viewFor({ distance: 6, target: [2, -1, 0.9] });
  const camera = cameraBasis(view);
  const centre = projectPoint(view.target, camera, view);
  assert.ok(Math.abs(centre[0] - view.cx) < 1e-6);
  assert.ok(Math.abs(centre[1] - view.cy) < 1e-6);
  const above = projectPoint([2, -1, 1.9], camera, view);
  assert.ok(above[1] < centre[1], "higher world Z must project higher on screen");
  assert.equal(projectPoint([9, -1, 0.9], camera, view), null);
});

test("toCamera reports depth along the view axis", () => {
  const view = viewFor({ distance: 4, target: [0, 0, 0] });
  const camera = cameraBasis(view);
  const centre = toCamera([0, 0, 0], camera);
  assert.ok(Math.abs(centre[2] - 4) < 1e-9);
});

test("projectPolyline splits a path at the near plane", () => {
  const view = viewFor({ distance: 4, target: [0, 0, 0] });
  const camera = cameraBasis(view);
  const runs = projectPolyline(
    [
      [-3, 0, 0],
      [-2, 0, 0],
      [9, 0, 0],
      [-2, 0, 0],
      [-3, 0, 0],
    ],
    camera,
    view,
  );
  assert.equal(runs.length, 2);
  assert.equal(runs[0].length, 2);
  assert.equal(runs[1].length, 2);
});

test("clipSegment drops fully-behind segments and clips crossing ones", () => {
  const view = viewFor({ distance: 4, target: [0, 0, 0] });
  const camera = cameraBasis(view);
  assert.equal(clipSegment([9, 2, 0], [10, 2, 0], camera, view), null);
  // The eye sits at x = +4 looking toward -x, so y = 2 keeps the clipped span
  // off the view axis and makes the projected ordering observable.
  const clipped = clipSegment([-3, 2, 0], [9, 2, 0], camera, view);
  assert.ok(clipped !== null);
  assert.ok(clipped.a[0] < clipped.b[0]);
});

test("clipPolygonNear keeps front polygons and drops behind ones", () => {
  const front = clipPolygonNear(
    [
      [0, 0, 1],
      [1, 0, 1],
      [1, 1, 1],
      [0, 1, 1],
    ],
    NEAR,
  );
  assert.equal(front.length, 4);
  assert.equal(clipPolygonNear([[-1, 0, -1], [1, 0, -1], [1, 1, -1]], NEAR).length, 0);
  // The eye sits at x = +4 looking toward -x, so a world polygon at x > 4 is
  // fully behind the near plane and must project to null.
  const behind = [[5, 0, -1], [6, 0, -1], [6, 1, -1]];
  assert.equal(projectPolygon(behind, cameraBasis(viewFor({ distance: 4 })), viewFor({ distance: 4 })), null);
});

test("frustumEdges has eight pairs covering the five vertices", () => {
  const edges = frustumEdges();
  assert.equal(edges.length, 8);
  const used = new Set(edges.flat());
  assert.deepEqual([...used].sort(), [0, 1, 2, 3, 4]);
  assert.deepEqual(edges[0], [0, 1]);
  assert.deepEqual(edges[7], [4, 1]);
});

test("sortFarToNear orders items by descending depth", () => {
  const sorted = sortFarToNear([{ depth: 1 }, { depth: 5 }, { depth: 3 }]);
  assert.deepEqual(sorted.map((item) => item.depth), [5, 3, 1]);
});

test("strideFor keeps a trajectory within its sample budget", () => {
  assert.equal(strideFor(500, 500), 1);
  assert.equal(strideFor(1000, 500), 2);
  assert.equal(strideFor(10, 500), 1);
});
