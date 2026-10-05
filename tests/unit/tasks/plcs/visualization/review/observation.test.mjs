import assert from "node:assert/strict";
import test from "node:test";
import { decodeObservations, frameStatistics, playerCrop } from "../../../../../../src/tasks/plcs/visualization/review/dataset_static/observation.mjs";

test("binary fields require the exact size and contiguous descriptors", () => {
  const buffer = new Float32Array([1, 2, 3]).buffer;
  const document = { byte_length: 12, buffer_fields: { root: { byte_offset: 0, shape: [1, 3], dtype: "float32" } } };
  assert.deepEqual([...decodeObservations(buffer, document).root], [1, 2, 3]);
  assert.throws(() => decodeObservations(new ArrayBuffer(8), document), /size mismatch/);
  document.buffer_fields.root.byte_offset = 4;
  assert.throws(() => decodeObservations(buffer, document), /Invalid observation field/);
});

test("out-of-frame and behind-camera teachers cannot become visible observations", () => {
  const observation = { count: 3, uv: [0.2, 0.5, 1.2, 0.5, 0, 0], visible: [1, 1, 0],
    projected: [0.21, 0.5, 1.2, 0.5, 0.5, 0.5], front: [1, 1, 0] };
  const stats = frameStatistics(observation, [1280, 720]);
  assert.equal(stats.visible, 2);
  assert.equal(stats.mismatch, 1);
  assert.equal(stats.outside, 1);
  assert.ok(Math.abs(stats.maxError - 12.8) < 1e-6);
});

test("no visible observations means no comparison or invented player crop", () => {
  const observation = { count: 1, uv: [0, 0], visible: [0], projected: [-0.2, 0.5], front: [1] };
  assert.equal(frameStatistics(observation, [1280, 720]).maxError, null);
  assert.equal(playerCrop(observation, [1280, 720]), null);
});
