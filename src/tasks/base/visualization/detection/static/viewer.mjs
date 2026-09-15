export function fitScale(width, height, viewportWidth, viewportHeight) {
  if (
    ![width, height, viewportWidth, viewportHeight].every(
      (v) => Number.isFinite(v) && v > 0,
    )
  )
    return 1;
  return Math.min(viewportWidth / width, viewportHeight / height) * 0.94;
}
export function zoomAt(transform, factor, x, y) {
  const scale = Math.max(0.02, Math.min(32, transform.scale * factor));
  const ratio = scale / transform.scale;
  return {
    scale,
    x: x - (x - transform.x) * ratio,
    y: y - (y - transform.y) * ratio,
  };
}
export function visiblePoints(layer) {
  return (layer?.points || []).filter(
    (p) => p.visible !== false && Number.isFinite(p.x) && Number.isFinite(p.y),
  );
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("画像の読み込みに失敗しました。"));
    image.src = url;
  });
}

export class ImageViewer {
  constructor(canvas, onZoom = () => {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onZoom = onZoom;
    this.transform = { x: 0, y: 0, scale: 1 };
    this.image = null;
    this.token = 0;
    this.gt = null;
    this.pred = null;
    this.rasters = new Map();
    this.pointers = new Map();
    this.options = {
      gt: true,
      pred: true,
      labels: false,
      raster: "",
      opacity: 0.55,
    };
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        this.zoom(
          Math.exp(-event.deltaY * 0.0015),
          event.clientX - rect.left,
          event.clientY - rect.top,
        );
      },
      { passive: false },
    );
    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      this.pointers.set(event.pointerId, {
        x: event.clientX,
        y: event.clientY,
      });
    });
    canvas.addEventListener("pointermove", (event) => {
      const before = this.pointers.get(event.pointerId);
      if (!before) return;
      const after = { x: event.clientX, y: event.clientY };
      if (this.pointers.size === 2) {
        const other = [...this.pointers.entries()].find(
          ([id]) => id !== event.pointerId,
        )[1];
        const distance = Math.hypot(before.x - other.x, before.y - other.y);
        const next = Math.hypot(after.x - other.x, after.y - other.y);
        const rect = canvas.getBoundingClientRect();
        if (distance > 0)
          this.zoom(
            next / distance,
            (before.x + other.x) / 2 - rect.left,
            (before.y + other.y) / 2 - rect.top,
          );
      } else {
        this.transform.x += after.x - before.x;
        this.transform.y += after.y - before.y;
      }
      this.pointers.set(event.pointerId, after);
      this.draw();
    });
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"])
      canvas.addEventListener(name, (event) =>
        this.pointers.delete(event.pointerId),
      );
    canvas.addEventListener("dblclick", () => this.fit());
  }
  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.width = rect.width;
    this.height = rect.height;
    this.dpr = dpr;
    this.fit();
  }
  clear() {
    this.token++;
    this.image = null;
    this.gt = null;
    this.pred = null;
    this.rasters.clear();
    this.draw();
  }
  async setFrame(url, gt, pred, reset = false) {
    const token = ++this.token;
    const image = await loadImage(url);
    const rasterList = [...(gt?.rasters || []), ...(pred?.rasters || [])];
    const loaded = await Promise.all(
      rasterList.map(async (layer) => [
        layer.data,
        await loadImage(layer.data),
      ]),
    );
    if (token !== this.token) return false;
    const changedSize =
      !this.image ||
      this.image.width !== image.width ||
      this.image.height !== image.height;
    this.image = image;
    this.gt = gt;
    this.pred = pred;
    this.rasters = new Map(loaded);
    if (reset || changedSize) this.fit();
    else this.draw();
    return true;
  }
  configure(options) {
    Object.assign(this.options, options);
    this.draw();
  }
  fit() {
    if (!this.image) return this.draw();
    const scale = fitScale(
      this.image.width,
      this.image.height,
      this.width,
      this.height,
    );
    this.transform = {
      scale,
      x: (this.width - this.image.width * scale) / 2,
      y: (this.height - this.image.height * scale) / 2,
    };
    this.onZoom(scale);
    this.draw();
  }
  zoom(factor, x = this.width / 2, y = this.height / 2) {
    this.transform = zoomAt(this.transform, factor, x, y);
    this.onZoom(this.transform.scale);
    this.draw();
  }
  draw() {
    const ctx = this.ctx;
    if (!ctx) return;
    ctx.setTransform(this.dpr || 1, 0, 0, this.dpr || 1, 0, 0);
    ctx.fillStyle = "#17221e";
    ctx.fillRect(0, 0, this.width || 1, this.height || 1);
    if (!this.image) return;
    const { x, y, scale } = this.transform;
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(scale, scale);
    ctx.drawImage(this.image, 0, 0);
    ctx.beginPath();
    ctx.rect(0, 0, this.image.width, this.image.height);
    ctx.clip();
    for (const [kind, color] of [
      ["gt", "#25db97"],
      ["pred", "#ff6285"],
    ]) {
      if (!this.options[kind] || !this[kind]) continue;
      const layer = this[kind];
      const raster = (layer.rasters || []).find(
        (r) => r.name === this.options.raster,
      );
      if (raster && this.rasters.has(raster.data)) {
        ctx.globalAlpha = this.options.opacity;
        ctx.drawImage(
          this.rasters.get(raster.data),
          0,
          0,
          this.image.width,
          this.image.height,
        );
        ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1.8 / scale;
      for (const line of layer.segments || []) {
        ctx.beginPath();
        ctx.moveTo(line.x1, line.y1);
        ctx.lineTo(line.x2, line.y2);
        ctx.stroke();
      }
      for (const point of visiblePoints(layer)) {
        const radius = (kind === "gt" ? 5 : 3.2) / scale;
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius, 0, 2 * Math.PI);
        if (kind === "gt") {
          ctx.stroke();
        } else {
          ctx.fill();
        }
        if (this.options.labels && point.label) {
          ctx.font = `${10 / scale}px system-ui`;
          const label = String(point.label);
          const tw = ctx.measureText(label).width;
          const tx = Math.min(
            Math.max(point.x + 7 / scale, 0),
            this.image.width - tw - 4 / scale,
          );
          const ty = Math.max(
            13 / scale,
            Math.min(point.y - 7 / scale, this.image.height - 3 / scale),
          );
          ctx.fillStyle = "#10251fe6";
          ctx.fillRect(
            tx - 2 / scale,
            ty - 11 / scale,
            tw + 4 / scale,
            14 / scale,
          );
          ctx.fillStyle = color;
          ctx.fillText(label, tx, ty);
        }
      }
    }
    ctx.restore();
  }
  download(name) {
    this.canvas.toBlob((blob) => {
      if (!blob) return;
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${name.replace(/[^a-zA-Z0-9_-]/g, "_")}.png`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    });
  }
}
