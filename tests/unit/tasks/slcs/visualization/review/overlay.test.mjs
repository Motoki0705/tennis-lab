import test from "node:test";
import assert from "node:assert/strict";
import { poseCrop, ballResidualPixels } from "../../../../../../src/tasks/slcs/visualization/review/static/slcs-overlay.mjs";
import { fitOrbit, framingPoints } from "../../../../../../src/tasks/slcs/visualization/review/static/slcs-framing.mjs";
import { cameraBasis, projectPoint } from "../../../../../../src/tasks/base/visualization/review/static/scene.mjs";

test("pose crop excludes unknown and out-of-frame joints and stays in the image", () => {
  assert.equal(poseCrop([null, [-0.1, 0.4], [1.2, 0.3]], 1920, 1080), null);
  const region = poseCrop([[0.99, 0.95], [0.98, 0.99], null], 1920, 1080);
  assert.ok(region[0] >= 0 && region[1] >= 0);
  assert.ok(region[0] + region[2] <= 1920 && region[1] + region[3] <= 1080);
  assert.ok(region[3] >= 150);
});

test("ball residual uses full image dimensions and remains unknown without both values", () => {
  assert.equal(ballResidualPixels({ uv: null, projection: { uv: [0, 0] } }, 1920, 1080), null);
  assert.equal(ballResidualPixels({ uv: [0.1, 0.2], projection: { uv: null } }, 1920, 1080), null);
  assert.equal(ballResidualPixels({ uv: [0.2, 0.5], projection: { uv: [0.1, 0.5] } }, 1000, 500), 100);
});

test("fit includes accepted roots and court but excludes zero placeholders in gaps", () => {
  const points = framingPoints({court:{keypoints:[[1, 2, 0]]},entities:[{frames:2,roots:new Float32Array([0,0,0,10,20,5]),presence:new Uint8Array([0,1])}]});
  assert.deepEqual(points, [[1,2,0],[10,20,5]]);
  const narrow = fitOrbit(points, {yaw:-2.35,pitch:0.5,fov:36,aspect:0.65});
  const wide = fitOrbit(points, {yaw:-2.35,pitch:0.5,fov:36,aspect:1.6});
  assert.ok(narrow.distance >= wide.distance);
  assert.deepEqual(narrow.target,[5.5,11,2.5]);
  // Check the projected bounds through the shared projection helpers.
  for (const point of points) {
    const camera = cameraBasis(narrow);
    const projected = projectPoint(point, camera, {focal:1000 / (2 * Math.tan(36 * Math.PI / 360)),cx:325,cy:500});
    assert.ok(projected[0] >= 0 && projected[0] <= 650);
    assert.ok(projected[1] >= 0 && projected[1] <= 1000);
  }
  assert.throws(()=>fitOrbit(points,{yaw:0,pitch:0.5,fov:36,aspect:0}),/Invalid/);
});
