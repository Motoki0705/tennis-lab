// Unit tests for the BLCS inference review projection math (node --test).

import assert from "node:assert/strict";
import test from "node:test";

import {
  activeAt,
  cameraBasis,
  clamp,
  courtBounds,
  dot,
  fitDistance,
  fitViewDistance,
  length,
  normalizeSeries,
  positionAt,
  projectPoint,
  projectPolyline,
  subtract,
} from "../../../../../../src/tasks/blcs/visualization/inference/static/scene.mjs";

const HALF_FOV = (38 * Math.PI) / 360;

function viewFor({
  distance,
  target = [0, 0, 0],
  yaw = 0,
  pitch = 0,
  width = 1280,
  height = 852,
}) {
  return {
    yaw,
    pitch,
    distance,
    target,
    halfFov: HALF_FOV,
    focal: height / 2 / Math.tan(HALF_FOV),
    cx: width / 2,
    cy: height / 2,
    width,
    height,
  };
}

// The CourtKP20 layout from the frozen API contract, in metres, +Z up.
const HALF_LENGTH = 11.885;
const HALF_DOUBLES = 5.485;
const HALF_SINGLES = 4.115;
const SERVICE_LINE = 6.4;
const NET_CENTER = 0.914;
const NET_POST = 1.07;
const POST_X = HALF_DOUBLES + 0.914;

const COURT_KEYPOINTS = [
  [-HALF_DOUBLES, HALF_LENGTH, 0],
  [HALF_DOUBLES, HALF_LENGTH, 0],
  [-HALF_DOUBLES, -HALF_LENGTH, 0],
  [HALF_DOUBLES, -HALF_LENGTH, 0],
  [-HALF_SINGLES, HALF_LENGTH, 0],
  [-HALF_SINGLES, -HALF_LENGTH, 0],
  [HALF_SINGLES, HALF_LENGTH, 0],
  [HALF_SINGLES, -HALF_LENGTH, 0],
  [-HALF_SINGLES, SERVICE_LINE, 0],
  [HALF_SINGLES, SERVICE_LINE, 0],
  [-HALF_SINGLES, -SERVICE_LINE, 0],
  [HALF_SINGLES, -SERVICE_LINE, 0],
  [0, SERVICE_LINE, 0],
  [0, -SERVICE_LINE, 0],
  [0, 0, 0],
  [-POST_X, 0, 0],
  [-POST_X, 0, NET_POST],
  [POST_X, 0, 0],
  [POST_X, 0, NET_POST],
  [0, 0, NET_CENTER],
];

test("cameraBasis is an orthonormal, right-handed frame", () => {
  const view = viewFor({ distance: 40, target: [0, 0, 0], yaw: -Math.PI / 2, pitch: 0.34 });
  const camera = cameraBasis(view);
  assert.ok(Math.abs(length(camera.forward) - 1) < 1e-9);
  assert.ok(Math.abs(length(camera.right) - 1) < 1e-9);
  assert.ok(Math.abs(length(camera.up) - 1) < 1e-9);
  assert.ok(Math.abs(dot(camera.right, camera.up)) < 1e-9);
  assert.ok(Math.abs(dot(camera.right, camera.forward)) < 1e-9);
  assert.ok(Math.abs(dot(camera.up, camera.forward)) < 1e-9);
  const offset = subtract(camera.eye, view.target);
  assert.ok(Math.abs(dot(offset, camera.forward) + view.distance) < 1e-9);
  // right x up must point back toward the eye for a right-handed frame.
  const handed = [
    camera.right[1] * camera.up[2] - camera.right[2] * camera.up[1],
    camera.right[2] * camera.up[0] - camera.right[0] * camera.up[2],
    camera.right[0] * camera.up[1] - camera.right[1] * camera.up[0],
  ];
  assert.ok(dot(handed, camera.forward) < 0);
});

test("projectPoint centres the target and keeps world +Z up", () => {
  const view = viewFor({ distance: 30, target: [0, 0, 0], yaw: -Math.PI / 2, pitch: 0.34 });
  const camera = cameraBasis(view);
  const centre = projectPoint(view.target, camera, view);
  assert.ok(Math.abs(centre[0] - view.cx) < 1e-6);
  assert.ok(Math.abs(centre[1] - view.cy) < 1e-6);
  const above = projectPoint([0, 0, 1.9], camera, view);
  assert.ok(above[1] < centre[1], "higher world Z must project higher on screen");
});

test("projectPoint returns null behind the near plane", () => {
  const view = viewFor({ distance: 4, target: [0, 0, 0] });
  const camera = cameraBasis(view);
  // The eye sits at x = +4 looking toward -x, so x = 9 is behind the camera.
  assert.equal(projectPoint([9, 0, 0], camera, view), null);
  // At x = 3.99 the depth is 0.01, inside the near plane.
  assert.equal(projectPoint([3.99, 0, 0], camera, view), null);
  assert.notEqual(projectPoint([-2, 0, 0], camera, view), null);
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

test("fitDistance grows monotonically with radius and the framed sphere fits", () => {
  const small = fitDistance(1, HALF_FOV);
  const large = fitDistance(10, HALF_FOV);
  assert.ok(large > small, "a bigger subject must require a larger distance");
  const height = 852;
  const radius = 1.5;
  const view = viewFor({ distance: fitDistance(radius, HALF_FOV), target: [0, 0, 0], height });
  const camera = cameraBasis(view);
  const top = projectPoint([0, 0, radius], camera, view);
  const bottom = projectPoint([0, 0, -radius], camera, view);
  assert.ok(top[1] >= 0 && bottom[1] <= height);
});

test("flat-buffer accessors read positions and the active mask by (frame, track)", () => {
  // frames = 2, tracks = 3, row-major (frame, track, xyz).
  const positions = new Float32Array([
    0, 1, 2, 10, 11, 12, 20, 21, 22,
    100, 101, 102, 110, 111, 112, 120, 121, 122,
  ]);
  assert.deepEqual(positionAt(positions, 3, 0, 0), [0, 1, 2]);
  assert.deepEqual(positionAt(positions, 3, 0, 2), [20, 21, 22]);
  assert.deepEqual(positionAt(positions, 3, 1, 0), [100, 101, 102]);
  assert.deepEqual(positionAt(positions, 3, 1, 2), [120, 121, 122]);

  const mask = new Uint8Array([1, 0, 1, 0, 1, 0]);
  assert.equal(activeAt(mask, 3, 0, 0), true);
  assert.equal(activeAt(mask, 3, 0, 1), false);
  assert.equal(activeAt(mask, 3, 1, 1), true);
  assert.equal(activeAt(mask, 3, 1, 2), false);
});

test("normalizeSeries keeps presence and defaults a null mask to all active", () => {
  const withPresence = normalizeSeries({
    frames: 2,
    tracks: 2,
    positions: new Float32Array(12),
    presence: [1, 0, 1, 1],
  });
  assert.equal(withPresence.frames, 2);
  assert.equal(withPresence.tracks, 2);
  assert.equal(activeAt(withPresence.mask, 2, 0, 1), false);
  assert.equal(activeAt(withPresence.mask, 2, 1, 0), true);

  const nullPresence = normalizeSeries({
    frames: 1,
    tracks: 2,
    positions: new Float32Array(6),
    presence: null,
  });
  assert.equal(activeAt(nullPresence.mask, 2, 0, 0), true);
  assert.equal(activeAt(nullPresence.mask, 2, 0, 1), true);

  assert.equal(normalizeSeries(null), null);
});

test("courtBounds centres the court on the ground plane", () => {
  const bounds = courtBounds(COURT_KEYPOINTS);
  assert.deepEqual(bounds.center, [0, 0, 0]);
  assert.ok(bounds.radius > HALF_LENGTH);
  assert.ok(bounds.radius < Math.hypot(HALF_DOUBLES + 0.914, HALF_LENGTH) + 0.5);
});

test("a fitted broadcast camera keeps the origin and a court corner on screen", () => {
  const view = viewFor({
    distance: 50,
    yaw: -Math.PI / 2,
    pitch: 0.34,
    width: 1280,
    height: 800,
  });
  view.target = courtBounds(COURT_KEYPOINTS).center;
  view.distance = fitViewDistance(COURT_KEYPOINTS, view);
  const camera = cameraBasis(view);
  const origin = projectPoint([0, 0, 0], camera, view);
  const corner = projectPoint(COURT_KEYPOINTS[0], camera, view);
  assert.notEqual(origin, null);
  assert.notEqual(corner, null);
  for (const point of [origin, corner]) {
    assert.ok(point[0] >= 0 && point[0] <= view.width, `x ${point[0]} outside frame`);
    assert.ok(point[1] >= 0 && point[1] <= view.height, `y ${point[1]} outside frame`);
  }
  // All keypoints must fit inside the framed viewport too.
  for (const keypoint of COURT_KEYPOINTS) {
    const projected = projectPoint(keypoint, camera, view);
    assert.notEqual(projected, null);
    assert.ok(projected[0] >= 0 && projected[0] <= view.width);
    assert.ok(projected[1] >= 0 && projected[1] <= view.height);
  }
});

test("fitViewDistance pulls back further as the requested margin grows", () => {
  const view = viewFor({ distance: 50, yaw: -Math.PI / 2, pitch: 0.34, width: 1024, height: 700 });
  view.target = courtBounds(COURT_KEYPOINTS).center;
  const tight = fitViewDistance(COURT_KEYPOINTS, view, 0.02);
  const loose = fitViewDistance(COURT_KEYPOINTS, view, 0.18);
  assert.ok(loose > tight, "a larger inset margin must push the camera back");
});

test("clamp bounds a value to the inclusive range", () => {
  assert.equal(clamp(5, 0, 10), 5);
  assert.equal(clamp(-3, 0, 10), 0);
  assert.equal(clamp(42, 0, 10), 10);
});
