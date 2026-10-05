// Bounded, abortable look-ahead. RGB and annotations share one frame identity.
export class SequentialClock {
  constructor(fps, now) {
    this.period = 1000 / fps;
    this.next = now + this.period;
  }
  due(now) {
    return now >= this.next;
  }
  commit(now) {
    // Preserve the clock through ordinary display jitter. After a stall, wait
    // one full interval: never skip indices or burst through unseen frames.
    this.next =
      now - this.next >= this.period
        ? now + this.period
        : this.next + this.period;
  }
}

export async function loadBitmap(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok)
    throw new Error(`画像の取得に失敗しました (${response.status})`);
  return createImageBitmap(await response.blob());
}

class RequestQueue {
  constructor(limit = 4) {
    this.limit = limit;
    this.active = 0;
    this.jobs = new Map();
    this.pending = [];
  }
  get(key, load) {
    if (this.jobs.has(key)) return this.jobs.get(key).promise;
    const controller = new AbortController();
    const job = { key, load, controller };
    job.promise = new Promise((resolve, reject) =>
      Object.assign(job, { resolve, reject }),
    );
    // Prefetch errors are retained and rethrown when that frame is requested.
    job.promise.catch(() => {});
    this.jobs.set(key, job);
    this.pending.push(job);
    this.drain();
    return job.promise;
  }
  drain() {
    while (this.active < this.limit && this.pending.length) {
      const job = this.pending.shift();
      this.active++;
      Promise.resolve()
        .then(() => job.load(job.controller.signal))
        .then(job.resolve, job.reject)
        .finally(() => {
          this.active--;
          this.drain();
        });
    }
  }
  retain(keys) {
    for (const [key, job] of this.jobs) {
      if (keys.has(key)) continue;
      job.controller.abort();
      job.reject(new DOMException("Selection changed", "AbortError"));
      this.jobs.delete(key);
    }
    this.pending = this.pending.filter((job) => this.jobs.has(job.key));
  }
}

export class FrameBuffer {
  constructor({
    scene,
    total,
    playerDataset = "",
    mode = "reviewed",
    json,
    bitmap = loadBitmap,
    lookAhead = 12,
    imageLimit = 24,
    byteLimit = 96 * 1024 * 1024,
  }) {
    Object.assign(this, {
      scene,
      total,
      playerDataset,
      mode,
      json,
      bitmap,
      lookAhead,
      imageLimit,
      byteLimit,
    });
    this.images = new Map();
    this.metadata = new Map();
    this.queue = new RequestQueue();
    this.bytes = 0;
    this.pinned = -1;
    this.focus = -1;
    this.disposed = false;
    this.wanted = new Set();
  }
  url(endpoint, fields) {
    return `/api/${endpoint}?${new URLSearchParams({ scene: this.scene, ...fields })}`;
  }
  block(frame) {
    return Math.floor(frame / 32) * 32;
  }
  async meta(frame) {
    const block = this.block(frame);
    if (this.metadata.has(block)) return this.metadata.get(block);
    return this.queue.get(`m${block}`, async (signal) => {
      const fields = { start: block, count: Math.min(32, this.total - block) };
      const [preview, players] = await Promise.all([
        this.json(this.url("preview", fields), { signal }),
        this.playerDataset
          ? this.json(
              this.url("players", {
                ...fields,
                dataset: this.playerDataset,
                mode: this.mode,
              }),
              { signal },
            )
          : null,
      ]);
      if (signal.aborted || this.disposed)
        throw new DOMException("Selection changed", "AbortError");
      const result = { preview, players };
      this.metadata.set(block, result);
      return result;
    });
  }
  async image(frame) {
    if (this.images.has(frame)) return this.images.get(frame);
    return this.queue.get(`i${frame}`, async (signal) => {
      const image = await this.bitmap(this.url("image", { frame }), signal);
      if (signal.aborted || this.disposed) {
        image.close();
        throw new DOMException("Selection changed", "AbortError");
      }
      const size = image.width * image.height * 4;
      // A temporal clip needs room for both the visible and requested frame.
      if (size > this.byteLimit / (this.total > 1 ? 2 : 1)) {
        image.close();
        throw new Error("画像が再生キャッシュのメモリ上限を超えています。");
      }
      this.images.set(frame, image);
      this.bytes += size;
      this.trim();
      return image;
    });
  }
  drop(frame) {
    const image = this.images.get(frame);
    if (!image) return;
    this.bytes -= image.width * image.height * 4;
    image.close();
    this.images.delete(frame);
    // A completed promise also holds its bitmap: drop that reference too.
    const key = `i${frame}`;
    const job = this.queue.jobs.get(key);
    job?.controller.abort();
    this.queue.jobs.delete(key);
  }
  trim() {
    for (const frame of this.images.keys()) {
      if (frame === this.pinned || frame === this.focus) continue;
      if (
        !this.wanted.has(frame) ||
        this.images.size > this.imageLimit ||
        this.bytes > this.byteLimit
      )
        this.drop(frame);
    }
  }
  prefetch(frame) {
    if (this.disposed) return;
    this.focus = frame;
    const ahead = Array.from(
      { length: Math.min(this.total, this.lookAhead + 1) },
      (_, i) => (frame + i) % this.total,
    );
    this.wanted = new Set([
      ...ahead,
      ...Array.from(
        { length: Math.min(4, this.total) },
        (_, i) => (frame - i - 1 + this.total) % this.total,
      ),
    ]);
    const blocks = new Set([...this.wanted].map((i) => this.block(i)));
    const keys = new Set([...this.wanted].map((i) => `i${i}`));
    for (const block of blocks) keys.add(`m${block}`);
    this.queue.retain(keys);
    for (const block of this.metadata.keys())
      if (!blocks.has(block)) this.metadata.delete(block);
    this.trim();
    // An arbitrary seek takes precedence over speculative work.
    this.meta(frame).catch(() => {});
    this.image(frame).catch(() => {});
    for (const index of ahead.slice(1)) {
      this.meta(index).catch(() => {});
      this.image(index).catch(() => {});
    }
  }
  ready(frame) {
    return this.images.has(frame) && this.metadata.has(this.block(frame));
  }
  async get(frame) {
    if (this.disposed)
      throw new DOMException("Selection changed", "AbortError");
    this.focus = frame;
    if (!this.wanted.has(frame)) this.prefetch(frame);
    const [{ preview, players }, image] = await Promise.all([
      this.meta(frame),
      this.image(frame),
    ]);
    const item = preview.items.find((i) => i.index === frame);
    const pose = players?.items.find((i) => i.index === frame);
    if (!item || (players && !pose))
      throw new Error("選択フレームと注釈が一致しません。");
    return { image, item, preview, players, people: pose?.people || [] };
  }
  pin(frame) {
    this.pinned = frame;
    this.prefetch(frame);
  }
  dispose() {
    this.disposed = true;
    this.queue.retain(new Set());
    for (const frame of this.images.keys()) this.drop(frame);
    this.metadata.clear();
  }
  retire() {
    this.disposed = true;
    this.queue.retain(new Set());
    for (const frame of this.images.keys())
      if (frame !== this.pinned) this.drop(frame);
    this.metadata.clear();
  }
}
