import { test } from "node:test";
import assert from "node:assert/strict";
import {
  fitScale,
  zoomAt,
  visiblePoints,
  ImageViewer,
} from "../../../../../../src/tasks/base/visualization/detection/static/viewer.mjs";
test("fit preserves aspect and margin", () => {
  assert.equal(fitScale(1000, 500, 500, 500), 0.47);
  assert.equal(fitScale(500, 1000, 500, 500), 0.47);
});
test("Court invisible references are opt-in and remain visually distinct", () => {
  let arc, dash = [];
  const drawn = [];
  const ctx = {
    setTransform() {}, fillRect() {}, save() {}, translate() {}, scale() {},
    drawImage() {}, beginPath() {}, rect() {}, clip() {}, restore() {},
    setLineDash(value) { dash = value; },
    arc(x, y) { arc = { x, y }; },
    stroke() { drawn.push({ ...arc, color: this.strokeStyle, dash }); },
  };
  const viewer = Object.create(ImageViewer.prototype);
  Object.assign(viewer, {
    ctx, image: {width: 1280, height: 720}, width: 640, height: 360,
    transform: {x: 0, y: 0, scale: 0.5}, people: [], rasters: new Map(),
    options: {gt: true, players: false, referencePoints: false},
    gt: {points: [{x: 10, y: 20}, {x: 30, y: 40, visible: false}]},
  });
  viewer.draw();
  assert.equal(drawn.length, 1);
  drawn.length = 0;
  viewer.options.referencePoints = true;
  viewer.draw();
  assert.equal(drawn.length, 2);
  assert.equal(drawn[0].color, "#25db97");
  assert.deepEqual(drawn[0].dash, []);
  assert.equal(drawn[1].color, "#ffc857");
  assert.deepEqual(drawn[1].dash, [6, 4]);
  assert.equal(viewer.gt.points[1].visible, false);
});
test("point focus centers source pixels without changing annotation coordinates", () => {
  const viewer = Object.create(ImageViewer.prototype);
  Object.assign(viewer, {image: {width:1000, height:500}, width:600, height:400,
    transform:{x:0,y:0,scale:0.5}, options:{}, onZoom() {}, draw() {}});
  const point = {x:420,y:123};
  viewer.focusPoint(point);
  assert.equal(viewer.transform.x + point.x * viewer.transform.scale, 300);
  assert.equal(viewer.transform.y + point.y * viewer.transform.scale, 200);
  assert.deepEqual(point, {x:420,y:123});
  assert.equal(viewer.options.highlightPoint, point);
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
