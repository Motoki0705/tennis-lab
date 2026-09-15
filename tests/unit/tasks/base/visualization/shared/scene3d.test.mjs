// Unit tests for the shared Three.js scene engine's pure model helpers
// (node --test). Runtime rendering is covered by the browser regressions.

import assert from "node:assert/strict";
import test from "node:test";

import {
  PRESETS,
  cameraAxis,
  deriveApron,
  deriveNet,
  deriveSurface,
  frustumFromParams,
  normalizeCourt,
  presenceAt,
  rootAt,
  trailSegments,
} from "../../../../../../src/tasks/base/visualization/shared/scene3d.mjs";

function assertVector(actual, expected, tolerance = 1e-9) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    assert.ok(
      Math.abs(actual[index] - expected[index]) < tolerance,
      `component ${index}: ${actual[index]} !== ${expected[index]}`,
    );
  }
}

const HALF_LENGTH = 11.885;
const HALF_DOUBLES = 5.485;
const HALF_SINGLES = 4.115;
const SERVICE_LINE = 6.4;
const NET_CENTER = 0.914;
const NET_POST = 1.07;
const POST_X = HALF_DOUBLES + 0.914;

const KEYPOINTS = [
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

test("deriveApron centres itself on the doubles corners", () => {
  const apron = deriveApron(KEYPOINTS, 1.0);
  const xs = apron.map((point) => point[0]);
  const ys = apron.map((point) => point[1]);
  assert.equal(Math.min(...xs), -HALF_DOUBLES - 1);
  assert.equal(Math.max(...xs), HALF_DOUBLES + 1);
  assert.equal(Math.min(...ys), -HALF_LENGTH - 1);
  assert.equal(Math.max(...ys), HALF_LENGTH + 1);
});

test("deriveApron and deriveSurface reject short keypoint lists", () => {
  assert.equal(deriveApron([[0, 0, 0]]), null);
  assert.equal(deriveSurface([[0, 0, 0]]), null);
});

test("deriveNet reads CourtKP20 post and centre strap indices", () => {
  const net = deriveNet(KEYPOINTS);
  assert.deepEqual(net.posts[0][0], [-POST_X, 0, 0]);
  assert.deepEqual(net.posts[0][1], [-POST_X, 0, NET_POST]);
  assert.deepEqual(net.posts[1][1], [POST_X, 0, NET_POST]);
  // The top profile sags through the centre strap between both posts.
  assert.deepEqual(net.profile.x, [-POST_X, 0, POST_X]);
  assert.deepEqual(net.profile.z, [NET_POST, NET_CENTER, NET_POST]);
  assert.equal(deriveNet(KEYPOINTS.slice(0, 19)), null);
});

test("normalizeCourt derives apron, surface and net from keypoints + edges", () => {
  const court = normalizeCourt({ keypoints: KEYPOINTS, edges: [[0, 1]] });
  assert.equal(court.edges.length, 1);
  assert.equal(court.surface.length, 4);
  assert.ok(court.apron.length === 4);
  assert.ok(court.net.profile.x.length === 3);
  assert.equal(normalizeCourt({ keypoints: KEYPOINTS.slice(0, 2) }), null);
  assert.equal(normalizeCourt(null), null);
});

test("normalizeCourt accepts the BLCS lines alias", () => {
  const court = normalizeCourt({ keypoints: KEYPOINTS, lines: [[0, 1], [2, 3]] });
  assert.deepEqual(court.edges, [[0, 1], [2, 3]]);
});

test("cameraAxis uses rotation row 2 as the optical axis and negates row 1", () => {
  // A camera at +y = 20 looking toward -y with world +z up:
  // row 2 (view dir) = (0, -1, 0), row 1 (down) = (0, 0, -1).
  const axis = cameraAxis({
    center: [0, 20, 2],
    rotation: [
      [1, 0, 0],
      [0, 0, -1],
      [0, -1, 0],
    ],
  });
  assertVector(axis.direction, [0, -1, 0]);
  assertVector(axis.up, [0, 0, 1]);
});

test("cameraAxis normalises a non-unit rotation axis", () => {
  const axis = cameraAxis({
    center: [0, 0, 0],
    rotation: [
      [1, 0, 0],
      [0, 0, -2],
      [0, -3, 0],
    ],
  });
  assert.ok(Math.abs(axis.direction[1] + 1) < 1e-9);
  assert.ok(Math.abs(axis.up[2] - 1) < 1e-9);
});

test("cameraAxis falls back to the frustum centroid, not a single corner", () => {
  // Apex at the origin, four corners on the plane y = 10 spanning x and z.
  const frustum = [
    [0, 0, 0],
    [-2, 10, -1],
    [2, 10, -1],
    [2, 10, 1],
    [-2, 10, 1],
  ];
  const axis = cameraAxis({ center: [0, 0, 0], frustum });
  assert.ok(Math.abs(axis.direction[0]) < 1e-9);
  assert.ok(Math.abs(axis.direction[1] - 1) < 1e-9);
  assert.ok(Math.abs(axis.direction[2]) < 1e-9);
  assertVector(axis.up, [0, 0, 1]);
  // A corner-to-apex direction would be tilted; the centroid is not.
  const corner = frustum[1];
  const cornerDir = [corner[0], corner[1], corner[2]];
  const cornerLength = Math.hypot(...cornerDir);
  assert.ok(Math.abs(cornerDir[2] / cornerLength) > 1e-3);
});

test("cameraAxis prefers rotation over the frustum when both are present", () => {
  const axis = cameraAxis({
    center: [0, 0, 0],
    frustum: [
      [0, 0, 0],
      [-2, 10, -1],
      [2, 10, -1],
      [2, 10, 1],
      [-2, 10, 1],
    ],
    rotation: [
      [1, 0, 0],
      [0, 0, -1],
      [0, -1, 0],
    ],
  });
  assertVector(axis.direction, [0, -1, 0]);
});

test("cameraAxis rejects payloads without geometry", () => {
  assert.equal(cameraAxis(null), null);
  assert.equal(cameraAxis({ center: [0, 0, 0] }), null);
  assert.equal(cameraAxis({ center: [0, 0, 0], rotation: [[0, 0, 0], [0, 0, 0], [0, 0, 0]] }), null);
});

test("presenceAt defaults to present without a mask", () => {
  assert.equal(presenceAt({}, 3), true);
  assert.equal(presenceAt({ presence: new Uint8Array([1, 0, 1]) }, 1), false);
  assert.equal(presenceAt({ presence: new Uint8Array([1, 0, 1]) }, 2), true);
});

test("rootAt prefers an explicit root track and falls back to positions", () => {
  const out = { set(x, y, z) { this.v = [x, y, z]; return this; } };
  const entity = {
    frames: 2,
    joints: 2,
    positions: new Float32Array([0, 0, 0, 9, 9, 9, 1, 2, 3, 8, 8, 8]),
    roots: new Float32Array([10, 11, 12, 20, 21, 22]),
  };
  rootAt(entity, 1, out);
  assert.deepEqual(out.v, [20, 21, 22]);
  delete entity.roots;
  entity.rootIndex = 1;
  rootAt(entity, 0, out);
  assert.deepEqual(out.v, [9, 9, 9]);
});

test("PRESETS expose the four documented orbit framings", () => {
  assert.deepEqual(Object.keys(PRESETS).sort(), ["broadcast", "corner", "overhead", "side"]);
  for (const preset of Object.values(PRESETS)) {
    assert.equal(preset.target.length, 3);
    assert.ok(preset.distance > 0);
  }
});

// A real BLCS camera: world +z up, looking down and toward the court.
const CAMERA_PARAMS = {
  C: [-9.205181121826172, 18.056982040405273, 3.798030376434326],
  R: [
    [-0.8812909722328186, -0.4725739657878876, 0],
    [-0.08933991938829422, 0.16660770773887634, -0.9819675087928772],
    [0.46405231952667236, -0.865399181842804, -0.1890496015548706],
  ],
  f: 1108.5125168440816,
  cx: 640,
  cy: 360,
  w: 1280,
  h: 720,
};

/** Reproject a world point through an OpenCV camera; returns pixel (x, y). */
function reproject(world, params) {
  const c = [
    world[0] - params.C[0],
    world[1] - params.C[1],
    world[2] - params.C[2],
  ];
  const cam = [
    params.R[0][0] * c[0] + params.R[0][1] * c[1] + params.R[0][2] * c[2],
    params.R[1][0] * c[0] + params.R[1][1] * c[1] + params.R[1][2] * c[2],
    params.R[2][0] * c[0] + params.R[2][1] * c[1] + params.R[2][2] * c[2],
  ];
  return [
    params.cx + (params.f * cam[0]) / cam[2],
    params.cy + (params.f * cam[1]) / cam[2],
  ];
}

test("frustumFromParams returns centre plus four corners in image order", () => {
  const frustum = frustumFromParams(CAMERA_PARAMS, 6);
  assert.equal(frustum.length, 5);
  assert.equal(frustum[0][0], CAMERA_PARAMS.C[0]);
  assert.equal(frustum[0][1], CAMERA_PARAMS.C[1]);
  assert.equal(frustum[0][2], CAMERA_PARAMS.C[2]);
});

test("frustumFromParams corners reproject onto the image corners", () => {
  const depth = 6;
  const frustum = frustumFromParams(CAMERA_PARAMS, depth);
  const expected = [
    [0, 0],
    [CAMERA_PARAMS.w - 1, 0],
    [CAMERA_PARAMS.w - 1, CAMERA_PARAMS.h - 1],
    [0, CAMERA_PARAMS.h - 1],
  ];
  expected.forEach((pixel, index) => {
    const projected = reproject(frustum[index + 1], CAMERA_PARAMS);
    // The corners are recomputed as (ray * depth) / depth, so a sub-milli-pixel
    // float64 rounding residual is expected; anything larger is a real error.
    assert.ok(
      Math.abs(projected[0] - pixel[0]) < 1e-3,
      `corner ${index} x ${projected[0]} !== ${pixel[0]}`,
    );
    assert.ok(
      Math.abs(projected[1] - pixel[1]) < 1e-3,
      `corner ${index} y ${projected[1]} !== ${pixel[1]}`,
    );
  });
  // The corners sit exactly at the requested camera-Z depth.
  for (const corner of frustum.slice(1)) {
    const c = [
      corner[0] - CAMERA_PARAMS.C[0],
      corner[1] - CAMERA_PARAMS.C[1],
      corner[2] - CAMERA_PARAMS.C[2],
    ];
    const cameraZ =
      CAMERA_PARAMS.R[2][0] * c[0] +
      CAMERA_PARAMS.R[2][1] * c[1] +
      CAMERA_PARAMS.R[2][2] * c[2];
    assert.ok(Math.abs(cameraZ - depth) < 1e-6, `camera z ${cameraZ} !== ${depth}`);
  }
});

test("frustumFromParams rejects incomplete parameters", () => {
  assert.equal(frustumFromParams(null), null);
  assert.equal(frustumFromParams({ C: [0, 0, 0] }), null);
  assert.equal(frustumFromParams({ ...CAMERA_PARAMS, f: 0 }), null);
  assert.equal(frustumFromParams(CAMERA_PARAMS, 0), null);
});

test("trailSegments links consecutive present frames", () => {
  const { pairs, prefix, count } = trailSegments(5, null);
  assert.equal(count, 4);
  assert.deepEqual(pairs, [0, 1, 1, 2, 2, 3, 3, 4]);
  assert.deepEqual([...prefix], [0, 1, 2, 3, 4]);
});

test("trailSegments never bridges a missing frame", () => {
  // Frames 0,1 present, 2 missing, 3,4 present: the 1->3 gap is not a segment.
  const presence = new Uint8Array([1, 1, 0, 1, 1]);
  const { pairs, prefix, count } = trailSegments(5, presence);
  assert.equal(count, 2);
  assert.deepEqual(pairs, [0, 1, 3, 4]);
  // Only the first segment is complete at frame 2, none is added by the gap.
  assert.deepEqual([...prefix], [0, 1, 1, 1, 2]);
  const grouped = [];
  for (let index = 0; index < pairs.length; index += 2) {
    grouped.push([pairs[index], pairs[index + 1]]);
  }
  assert.deepEqual(grouped, [[0, 1], [3, 4]]);
  assert.ok(!grouped.some(([a, b]) => a === 1 && b === 3));
});

test("trailSegments handles an entirely absent entity", () => {
  const { pairs, count } = trailSegments(4, new Uint8Array([0, 0, 0, 0]));
  assert.equal(count, 0);
  assert.deepEqual(pairs, []);
});

test("presenceAt beyond an entity's own length is treated as absent", () => {
  // A shorter prediction than the model length must not read past its buffer:
  // masking is the engine's guard (frame >= entity.frames -> absent).
  const entity = { frames: 3, presence: new Uint8Array([1, 1, 1]) };
  assert.equal(presenceAt(entity, 2), true);
  // Out of range yields undefined, which never equals 1.
  assert.equal(presenceAt(entity, 5), false);
});

test("trailSegments prefix stays bounded for a shorter series", () => {
  const { prefix, count } = trailSegments(3, new Uint8Array([1, 1, 1]));
  assert.equal(count, 2);
  assert.equal(prefix[2], 2);
  // Indexing past the series yields undefined; callers fall back to 0.
  assert.equal(prefix[7], undefined);
});
