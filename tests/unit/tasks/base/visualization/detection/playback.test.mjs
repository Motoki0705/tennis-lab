import { test } from "node:test";
import assert from "node:assert/strict";
import {
  FrameBuffer,
  SequentialClock,
} from "../../../../../../src/tasks/base/visualization/detection/static/playback.mjs";

const turn = () => new Promise((resolve) => setImmediate(resolve));
function fixture(extra = {}) {
  const calls = [];
  const images = [];
  const buffer = new FrameBuffer({
    scene: "a",
    total: 150,
    playerDataset: "poses",
    lookAhead: 4,
    imageLimit: 10,
    byteLimit: 4000,
    json: async (url) => {
      calls.push(url);
      const u = new URL(url, "http://local");
      const start = Number(u.searchParams.get("start")),
        count = Number(u.searchParams.get("count"));
      return {
        items: Array.from({ length: count }, (_, i) => ({
          index: start + i,
          name: String(start + i),
          people: [{ id: String(start + i) }],
        })),
      };
    },
    bitmap: async (url) => {
      const index = Number(
        new URL(url, "http://local").searchParams.get("frame"),
      );
      const image = {
        index,
        width: 10,
        height: 10,
        closed: false,
        close() {
          this.closed = true;
        },
      };
      images.push(image);
      return image;
    },
    ...extra,
  });
  return { buffer, calls, images };
}

test("sequential playback batches metadata and bounds decoded images", async () => {
  const { buffer, calls, images } = fixture();
  for (let frame = 0; frame < 100; frame++) {
    const value = await buffer.get(frame);
    assert.equal(value.image.index, frame);
    assert.equal(value.image.closed, false);
    assert.equal(value.item.index, frame);
    assert.equal(value.people[0].id, String(frame));
    buffer.pin(frame);
    await turn();
    assert.ok(buffer.images.size <= 10);
    assert.ok(buffer.bytes <= 4000);
    assert.ok(buffer.metadata.size <= 4);
    assert.ok(buffer.queue.jobs.size <= 15);
  }
  assert.ok(calls.filter((u) => u.includes("/preview?")).length <= 4);
  buffer.dispose();
  await turn();
  assert.ok(images.every((i) => i.closed));
  assert.equal(buffer.bytes, 0);
});

test("arbitrary seek keeps the displayed image alive until replacement", async () => {
  const { buffer } = fixture();
  const old = await buffer.get(2);
  buffer.pin(2);
  const next = await buffer.get(120);
  assert.equal(next.image.index, 120);
  assert.equal(old.image.closed, false);
  buffer.pin(120);
  assert.equal(old.image.closed, true);
  buffer.dispose();
});

test("late image after disposal is closed and rejected", async () => {
  let release;
  const image = {
    width: 10,
    height: 10,
    closed: false,
    close() {
      this.closed = true;
    },
  };
  const { buffer } = fixture({
    lookAhead: 0,
    bitmap: () =>
      new Promise((resolve) => {
        release = resolve;
      }),
  });
  const result = buffer.get(0);
  const rejected = assert.rejects(result, { name: "AbortError" });
  await turn();
  buffer.dispose();
  release(image);
  await rejected;
  await turn();
  assert.equal(image.closed, true);
  assert.equal(buffer.images.size, 0);
});

test("prefetch failure is surfaced on demand, not replaced with empty annotations", async () => {
  const { buffer } = fixture({
    json: async () => {
      throw new Error("broken artifact");
    },
  });
  buffer.prefetch(0);
  await turn();
  await assert.rejects(buffer.get(0), /broken artifact/);
  buffer.dispose();
});

test("mismatched annotation frame is rejected", async () => {
  const { buffer } = fixture({
    json: async () => ({ items: [{ index: 99 }] }),
  });
  await assert.rejects(buffer.get(0), /一致しません/);
  buffer.dispose();
});

test("clock preserves 25fps deadlines through display jitter and resets after stalls", () => {
  const clock = new SequentialClock(25, 0);
  assert.equal(clock.due(39), false);
  assert.equal(clock.due(50), true);
  clock.commit(50);
  assert.equal(clock.next, 80);
  clock.commit(800);
  assert.equal(clock.next, 840);
  assert.equal(clock.due(801), false);
});

test("memory pressure does not evict a requested frame or exceed the byte budget", async () => {
  const { buffer } = fixture({ byteLimit: 800 }); // exactly two decoded images
  for (const frame of [0, 1, 2, 80, 81]) {
    const value = await buffer.get(frame);
    assert.equal(value.image.closed, false);
    buffer.pin(frame);
    await turn();
    assert.ok(buffer.bytes <= 800);
  }
  buffer.dispose();
});

test("oversized frames fail explicitly instead of exhausting playback memory", async () => {
  const { buffer, images } = fixture({ byteLimit: 700 });
  await assert.rejects(buffer.get(0), /メモリ上限/);
  buffer.dispose();
  await turn();
  assert.ok(images.every((image) => image.closed));
});
