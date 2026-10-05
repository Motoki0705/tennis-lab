import { test } from "node:test";
import assert from "node:assert/strict";
import { jumpTarget } from "../../../../../../src/tasks/base/visualization/detection/static/review.mjs";
import { ImageViewer } from "../../../../../../src/tasks/base/visualization/detection/static/viewer.mjs";

test("review jumps are strict, bounded, and do not skip the first matching frame", () => {
  const positions = [2, 10, 11, 40];
  assert.equal(jumpTarget(positions, 0, 1), 2);
  assert.equal(jumpTarget(positions, 10, 1), 11);
  assert.equal(jumpTarget(positions, 10, -1), 2);
  assert.equal(jumpTarget(positions, 9, -1), 2);
  assert.equal(jumpTarget(positions, 2, -1), null);
  assert.equal(jumpTarget(positions, 40, 1), null);
  assert.equal(jumpTarget([], 0, 1), null);
});

test("ball kinds use distinct colors while retaining small filled dots and hidden unknown positions", () => {
  const drawn = [];
  let arc;
  const ctx = {
    setTransform() {}, fillRect() {}, save() {}, translate() {}, scale() {},
    drawImage() {}, beginPath() {}, rect() {}, clip() {}, restore() {},
    arc(x, y, radius) { arc = {x, y, radius}; },
    fill() { drawn.push({...arc, color: this.fillStyle}); },
    stroke() { assert.fail("ball points must remain filled dots"); },
  };
  const viewer = Object.create(ImageViewer.prototype);
  Object.assign(viewer, {
    ctx, image: {width: 1280, height: 720}, width: 640, height: 360,
    transform: {x: 0, y: 0, scale: 0.5}, people: [],
    options: {gt: true, ballPoints: true, players: false}, rasters: new Map(),
    gt: {points: [
      {x: 10, y: 20, state: "observed"},
      {x: 20, y: 20, state: "interpolated"},
      {x: 30, y: 20, state: "occlusion_estimated"},
      {x: 0, y: 0, state: "unresolved", visible: false},
      {x: 0, y: 0, state: "out_of_frame", visible: false},
    ]},
  });
  viewer.draw();
  assert.equal(drawn.length, 3);
  assert.equal(new Set(drawn.map((p) => p.color)).size, 3);
  assert.ok(drawn.every((p) => p.radius * viewer.transform.scale === 1.25));
});
