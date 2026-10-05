import { test } from "node:test";
import assert from "node:assert/strict";
import { canFocus, pointStateLabel, targetDescription } from "../../../../../../src/tasks/court_detection/visualization/review/static/review.mjs";

test("image-bounds visibility is not declared observed or occlusion-free", () => {
  assert.equal(pointStateLabel({state: "in_frame_visibility_unknown"}), "画像内 · 遮蔽不明");
  assert.equal(pointStateLabel({state: "renderer_not_visible"}), "renderer不可視");
  assert.equal(pointStateLabel({state: "unknown"}), "状態未提供");
});
test("saved out-of-frame and behind-camera points cannot focus an image", () => {
  assert.equal(canFocus({in_frame:false,x:-2,y:4}), false);
  assert.equal(canFocus({in_frame:true,in_front:false,x:2,y:4}), false);
  assert.equal(canFocus({in_frame:true,in_front:null,x:2,y:4}), true);
  assert.equal(canFocus({in_frame:true,x:NaN,y:4}), false);
});
test("dense target names describe their real teacher meaning", () => {
  assert.match(targetDescription("line"), /7.5 cm.*15 cm.*被覆率/);
  assert.match(targetDescription("seg"), /7領域/);
  assert.match(targetDescription("semantic_line"), /12種類/);
});
