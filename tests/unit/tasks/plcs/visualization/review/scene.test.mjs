// Unit tests for the ACCAD review projection math (node --test).

import assert from "node:assert/strict";
import test from "node:test";

import {
  FOCUS_RADIUS_M,
  cameraBasis,
  dot,
  fitDistance,
  jointBounds,
  jointSide,
  length,
  projectPoint,
  projectPolyline,
  subtract,
} from "../../../../../../src/tasks/plcs/visualization/review/static/scene.mjs";

const HALF_FOV = (38 * Math.PI) / 360;

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
  const view = viewFor({ distance: 4, target: [2, -1, 0.9] });
  const camera = cameraBasis(view);
  const centre = projectPoint(view.target, camera, view);
  assert.ok(Math.abs(centre[0] - view.cx) < 1e-6);
  assert.ok(Math.abs(centre[1] - view.cy) < 1e-6);
  const above = projectPoint([2, -1, 1.9], camera, view);
  assert.ok(above[1] < centre[1], "higher world Z must project higher on screen");
  // The eye sits at x = target.x + distance, so anything beyond it is clipped.
  assert.equal(projectPoint([7, -1, 0.9], camera, view), null);
});

test("an elevated camera sees higher world points as closer", () => {
  const view = viewFor({ distance: 4, target: [0, 0, 0.9], pitch: 0.32 });
  const camera = cameraBasis(view);
  const low = projectPoint([0, 0, 0], camera, view);
  const high = projectPoint([0, 0, 1.8], camera, view);
  assert.ok(high[2] < low[2], "an overhead view brings the head nearer");
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

test("jointBounds reports the world-space sphere of a motion", () => {
  const joints = new Float32Array([0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 3, 0]);
  const bounds = jointBounds(joints, 1, 4);
  assert.deepEqual(bounds.min, [0, 0, 0]);
  assert.deepEqual(bounds.max, [2, 3, 1]);
  assert.deepEqual(bounds.center, [1, 1.5, 0.5]);
  assert.ok(bounds.radius > 0);
});

test("fitDistance keeps a whole bounding sphere inside the viewport", () => {
  const height = 852;
  const radius = 0.787;
  const view = viewFor({ distance: fitDistance(radius, HALF_FOV), height });
  const camera = cameraBasis(view);
  const top = projectPoint([0, 0, radius], camera, view);
  const bottom = projectPoint([0, 0, -radius], camera, view);
  assert.ok(top[1] >= 0, `top edge ${top[1]} must stay on screen`);
  assert.ok(bottom[1] <= height, `bottom edge ${bottom[1]} must stay on screen`);
  // ...and the subject must still fill a useful share of the viewport.
  assert.ok(top[1] < height * 0.4, `top edge ${top[1]} is too loose`);
});

test("jointSide reads left/right from SMPL-H joint names", () => {
  assert.equal(jointSide("left_knee"), "left");
  assert.equal(jointSide("left_thumb3"), "left");
  assert.equal(jointSide("right_wrist"), "right");
  assert.equal(jointSide("right_index1"), "right");
  assert.equal(jointSide("pelvis"), "core");
  assert.equal(jointSide("head"), "core");
});

test("follow framing is tighter than whole-trajectory framing", () => {
  const height = 852;
  const radius = 3.6; // a walking clip that travels several metres
  const wide = fitDistance(radius, HALF_FOV);
  const focused = fitDistance(Math.min(radius, FOCUS_RADIUS_M), HALF_FOV);
  assert.ok(focused < wide * 0.6, "follow mode must dolly in on the subject");
  const view = viewFor({ distance: focused, height });
  const camera = cameraBasis(view);
  const top = projectPoint([0, 0, FOCUS_RADIUS_M], camera, view);
  const bottom = projectPoint([0, 0, -FOCUS_RADIUS_M], camera, view);
  assert.ok(top[1] >= 0 && bottom[1] <= height);
  assert.ok(top[1] > height * 0.08, "the subject must still be reasonably large");
});
