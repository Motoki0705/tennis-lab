// Unit tests for the PLCS inference 3D court-scene math (node --test).

import assert from "node:assert/strict";
import test from "node:test";

import {
  CourtScene,
  add,
  cameraBasis,
  clamp,
  cross,
  dot,
  fitDistance,
  headingVector,
  jointSide,
  length,
  normalize,
  projectPoint,
  projectPolyline,
  scale,
  subtract,
  trackBounds,
} from "../../../../../../src/tasks/plcs/visualization/inference/static/court_scene.mjs";

const HALF_FOV = (42 * Math.PI) / 360;

function viewFor({ distance, target = [0, 0, 0], yaw = 0, pitch = 0, height = 720 }) {
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

test("clamp keeps a value inside its window", () => {
  assert.equal(clamp(5, 0, 10), 5);
  assert.equal(clamp(-3, 0, 10), 0);
  assert.equal(clamp(42, 0, 10), 10);
  assert.equal(clamp(2.5, 2, 2), 2);
});

test("vector helpers compose a right-handed algebra", () => {
  const a = [1, 2, 3];
  const b = [4, 5, 6];
  assert.deepEqual(add(a, b), [5, 7, 9]);
  assert.deepEqual(subtract(a, b), [-3, -3, -3]);
  assert.deepEqual(scale(a, 2), [2, 4, 6]);
  assert.equal(dot(a, b), 32);
  assert.deepEqual(cross([1, 0, 0], [0, 1, 0]), [0, 0, 1]);
  assert.ok(Math.abs(length([3, 4, 12]) - 13) < 1e-12);
  const unit = normalize([0, 0, 5]);
  assert.ok(Math.abs(length(unit) - 1) < 1e-12);
  assert.deepEqual(normalize([0, 0, 0]), [0, 0, 0]);
});

test("cameraBasis is an orthonormal, right-handed frame", () => {
  const view = viewFor({ distance: 40, target: [0, 1, 0.5], yaw: -1.9, pitch: 0.6 });
  const camera = cameraBasis(view);
  assert.ok(Math.abs(length(camera.forward) - 1) < 1e-9);
  assert.ok(Math.abs(length(camera.right) - 1) < 1e-9);
  assert.ok(Math.abs(length(camera.up) - 1) < 1e-9);
  assert.ok(Math.abs(dot(camera.right, camera.up)) < 1e-9);
  assert.ok(Math.abs(dot(camera.right, camera.forward)) < 1e-9);
  assert.ok(Math.abs(dot(camera.up, camera.forward)) < 1e-9);
  // right = forward x up keeps the basis right-handed.
  const recomposed = cross(camera.forward, camera.up);
  assert.ok(Math.abs(recomposed[0] - camera.right[0]) < 1e-9);
  assert.ok(Math.abs(recomposed[2] - camera.right[2]) < 1e-9);
  const offset = subtract(camera.eye, view.target);
  assert.ok(Math.abs(dot(offset, camera.forward) + view.distance) < 1e-9);
});

test("projectPoint centres the target and clips behind the near plane", () => {
  const view = viewFor({ distance: 5, target: [0, 0, 0] });
  const camera = cameraBasis(view);
  const centre = projectPoint(view.target, camera, view);
  assert.ok(Math.abs(centre[0] - view.cx) < 1e-6);
  assert.ok(Math.abs(centre[1] - view.cy) < 1e-6);
  // The eye sits at x = +distance, so anything further along +x is behind it.
  assert.equal(projectPoint([8, 0, 0], camera, view), null);
  // ...and anything within the near plane of the eye is clipped too.
  assert.equal(projectPoint([4.99, 0, 0], camera, view), null);
});

test("projectPolyline splits a path at the near plane", () => {
  const view = viewFor({ distance: 5, target: [0, 0, 0] });
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

test("headingVector maps a (cos, sin) rotation to a world heading", () => {
  assert.deepEqual(headingVector([1, 0]), [1, 0, 0]);
  const north = headingVector([0, 1]);
  assert.ok(Math.abs(north[0]) < 1e-12);
  assert.equal(north[1], 1);
  assert.equal(north[2], 0);
  const diagonal = headingVector([3, 4]);
  assert.ok(Math.abs(diagonal[0] - 0.6) < 1e-12);
  assert.ok(Math.abs(diagonal[1] - 0.8) < 1e-12);
  const yaw = headingVector(Math.PI / 2);
  assert.ok(Math.abs(yaw[0]) < 1e-12);
  assert.equal(yaw[1], 1);
});

test("jointSide reads left/right from COCO-17 joint names", () => {
  assert.equal(jointSide("left_knee"), "left");
  assert.equal(jointSide("left_shoulder"), "left");
  assert.equal(jointSide("right_ankle"), "right");
  assert.equal(jointSide("right_ear"), "right");
  assert.equal(jointSide("nose"), "core");
  assert.equal(jointSide("left_hip"), "left");
});

test("trackBounds unions every position and joint of the tracks", () => {
  const tracks = [
    {
      position: new Float32Array([0, 0, 0, 2, 4, 0]),
      joints: new Float32Array([1, 1, 1, 3, 5, 2]),
    },
    {
      position: new Float32Array([-1, -2, 0, -1, -2, 0]),
      joints: null,
    },
  ];
  const bounds = trackBounds(tracks);
  assert.deepEqual(bounds.min, [-1, -2, 0]);
  assert.deepEqual(bounds.max, [3, 5, 2]);
  assert.deepEqual(bounds.center, [1, 1.5, 1]);
  assert.ok(bounds.radius > 0);
});

test("trackBounds tolerates tracks without joints or points", () => {
  const empty = trackBounds([{ position: new Float32Array(0), joints: null }]);
  assert.deepEqual(empty.min, [0, 0, 0]);
  assert.ok(empty.radius > 0);
});

test("fitDistance frames a whole bounding sphere on screen", () => {
  const radius = 12;
  const distance = fitDistance(radius, HALF_FOV);
  assert.ok(distance > radius);
  const view = viewFor({ distance, target: [0, 0, 0], height: 720 });
  const camera = cameraBasis(view);
  const top = projectPoint([0, 0, radius], camera, view);
  const bottom = projectPoint([0, 0, -radius], camera, view);
  assert.ok(top[1] >= 0, `top edge ${top[1]} must stay on screen`);
  assert.ok(bottom[1] <= 720, `bottom edge ${bottom[1]} must stay on screen`);
});

test("CourtScene.setData slices two packed tracks by float32 offset", () => {
  const data = new Float32Array(24);
  for (let index = 0; index < data.length; index += 1) {
    data[index] = index;
  }
  const payload = {
    header: {
      court: {
        keypoints: [
          [-9, -18, 0],
          [9, -18, 0],
          [-9, 18, 0],
          [9, 18, 0],
        ],
        edges: [[0, 1]],
      },
      skeleton: { names: ["nose", "left_hip"], edges: [[0, 1]] },
      tracks: [
        {
          kind: "gt",
          label: "GT",
          has_joints: true,
          position: { offset: 0, count: 6, shape: [2, 3] },
          joints: { offset: 6, count: 12, shape: [2, 2, 3] },
        },
        {
          kind: "pred",
          label: "推論",
          has_joints: false,
          position: { offset: 18, count: 6, shape: [2, 3] },
          joints: null,
        },
      ],
    },
    data,
  };

  const scene = new CourtScene(null);
  scene.setData(payload);

  assert.equal(scene.tracks.length, 2);
  assert.equal(scene.frameCount, 2);

  const [gt, pred] = scene.tracks;
  assert.equal(gt.kind, "gt");
  assert.equal(gt.label, "GT");
  assert.equal(gt.hasJoints, true);
  assert.equal(gt.frameCount, 2);
  assert.equal(gt.jointCount, 2);
  assert.equal(gt.position.length, 6);
  // Known value at a known element offset.
  assert.equal(gt.position[0], 0);
  assert.equal(gt.position[3], 3);
  assert.equal(gt.joints[0], 6);
  assert.equal(gt.joints[5], 11);

  assert.equal(pred.kind, "pred");
  assert.equal(pred.label, "推論");
  assert.equal(pred.hasJoints, false);
  assert.equal(pred.joints, null);
  assert.equal(pred.position[0], 18);
  assert.equal(pred.position[5], 23);

  // Slicing must not copy: the views point into the payload buffer.
  assert.equal(scene.tracks[0].position.buffer, data.buffer);
});

test("CourtScene.setFrame clamps to the installed track length", () => {
  const data = new Float32Array(6);
  const scene = new CourtScene(null);
  scene.setData({
    header: {
      court: { keypoints: [], edges: [] },
      skeleton: { names: [], edges: [] },
      tracks: [
        {
          kind: "gt",
          label: "GT",
          has_joints: false,
          position: { offset: 0, count: 6, shape: [2, 3] },
          joints: null,
        },
      ],
    },
    data,
  });
  scene.setFrame(9);
  assert.equal(scene.frame, 1);
  scene.setFrame(-4);
  assert.equal(scene.frame, 0);
});

test("CourtScene visibility toggles are independent per track and trails", () => {
  const scene = new CourtScene(null);
  assert.equal(scene.visible.gt, true);
  assert.equal(scene.visible.pred, true);
  assert.equal(scene.showTrails, true);
  scene.setKindVisible("pred", false);
  assert.equal(scene.visible.pred, false);
  assert.equal(scene.visible.gt, true);
  assert.equal(scene.toggleKind("pred"), true);
  scene.setTrailsVisible(false);
  assert.equal(scene.showTrails, false);
});
