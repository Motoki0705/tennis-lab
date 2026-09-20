import assert from "node:assert/strict";
import test from "node:test";

import {
  buildModel,
  decodeBuffers,
} from "../../../../../../src/tasks/base/visualization/review/static/model.mjs";

function descriptor(kind, slots, joints, orientation, presence) {
  return {
    kind, slots, joint_count: joints, frames: 3, orientation, presence,
    colors: ["#112233", "#445566"], skeleton: joints === 17 ? [[11, 12]] : null,
  };
}

function encode(entities) {
  let size = 0;
  const offsets = entities.map((entity) => {
    size = Math.ceil(size / 4) * 4;
    const offset = size;
    const values = entity.slots * entity.frames;
    size += values * entity.joint_count * 12 + (entity.orientation ? values * 8 : 0) + (entity.presence ? values : 0);
    return offset;
  });
  const buffer = new ArrayBuffer(size);
  for (const [index, entity] of entities.entries()) {
    let offset = offsets[index];
    const values = entity.slots * entity.frames;
    const positions = new Float32Array(buffer, offset, values * entity.joint_count * 3);
    positions.set(positions.map((_, i) => i + index * 1000));
    offset += positions.byteLength;
    if (entity.orientation) {
      new Float32Array(buffer, offset, values * 2).fill(0.5);
      offset += values * 8;
    }
    if (entity.presence) {
      const presence = new Uint8Array(buffer, offset, values);
      presence.fill(1);
      presence[1] = 0;
    }
  }
  return buffer;
}

function scene(fields) {
  return { frame_count: 3, court: {}, cameras: [], ...fields };
}

test("legacy PLCS skeletons retain hip roots, headings, slots and presence", () => {
  const entity = descriptor("player", 2, 17, true, true);
  const document = scene({ entity });
  const model = buildModel(document, decodeBuffers(encode([entity]), document));
  assert.equal(model.entities.length, 2);
  const [first, second] = model.entities;
  assert.deepEqual([...first.roots.slice(0, 3)], [34.5, 35.5, 36.5]);
  assert.deepEqual([...second.roots.slice(0, 3)], [187.5, 188.5, 189.5]);
  assert.deepEqual([...first.presence], [1, 0, 1]);
  assert.deepEqual([...second.presence], [1, 1, 1]);
  assert.equal(first.heading.length, 6);
  assert.deepEqual(first.edges, [[11, 12]]);
});

test("legacy BLCS balls retain position tracks without masks or headings", () => {
  const entity = descriptor("ball", 1, 1, false, false);
  const document = scene({ entity });
  const [ball] = buildModel(document, decodeBuffers(encode([entity]), document)).entities;
  assert.equal(ball.kind, "ball");
  assert.deepEqual([...ball.roots], [...ball.positions]);
  assert.equal(ball.heading, null);
  assert.equal(ball.presence, null);
});

test("mixed root players and ball decode across unaligned presence bytes", () => {
  const entities = [descriptor("player", 2, 1, true, true), descriptor("ball", 1, 1, false, true)];
  const document = scene({ entities });
  const model = buildModel(document, decodeBuffers(encode(entities), document));
  assert.deepEqual(model.entities.map((entity) => entity.kind), ["player", "player", "ball"]);
  assert.equal(new Set(model.entities.map((entity) => entity.id)).size, 3);
  assert.deepEqual([...model.entities[1].roots], [9, 10, 11, 12, 13, 14, 15, 16, 17]);
  assert.deepEqual([...model.entities[2].roots], [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]);
  assert.deepEqual([...model.entities[2].presence], [1, 0, 1]);
});

test("wrong timeline, truncated data and surplus bytes fail explicitly", () => {
  const entity = descriptor("ball", 1, 1, false, true);
  const document = scene({ entity });
  const buffer = encode([entity]);
  assert.throws(() => decodeBuffers(buffer, { ...document, frame_count: 2 }), /timeline/);
  assert.throws(() => decodeBuffers(buffer.slice(0, -1), document), /truncated/);
  assert.throws(() => decodeBuffers(new ArrayBuffer(buffer.byteLength + 1), document), /trailing/);
});
