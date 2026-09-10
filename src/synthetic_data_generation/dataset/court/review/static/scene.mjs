export const COLORS = {
  train: "#63b3dd",
  validation: "#e0ac61",
  test: "#b597ee",
};
export function distanceToSegment(p, a, b) {
  const x = b[0] - a[0],
    y = b[1] - a[1],
    t = Math.max(
      0,
      Math.min(
        1,
        ((p[0] - a[0]) * x + (p[1] - a[1]) * y) / (x * x + y * y || 1),
      ),
    );
  return Math.hypot(p[0] - a[0] - t * x, p[1] - a[1] - t * y);
}
export function project(p, view) {
  const x = p[0] - view.center[0],
    y = p[1] - view.center[1],
    z = p[2] - view.center[2],
    u = x * Math.cos(view.yaw) - y * Math.sin(view.yaw),
    v = x * Math.sin(view.yaw) + y * Math.cos(view.yaw),
    depth = v * Math.cos(view.pitch) - z * Math.sin(view.pitch),
    vertical = v * Math.sin(view.pitch) + z * Math.cos(view.pitch),
    factor = view.scale / (1 + depth / view.distance);
  return [
    view.width / 2 + view.pan[0] + u * factor,
    view.height / 2 + view.pan[1] - vertical * factor,
    depth,
  ];
}
export class SceneView {
  constructor(canvas, onSelect) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onSelect = onSelect;
    this.groups = [];
    this.courts = [];
    this.edges = [];
    this.selected = null;
    this.sampleIndex = null;
    this.hits = [];
    this.pending = false;
    this.view = {
      center: [0, 0, 0],
      yaw: -0.55,
      pitch: 0.8,
      scale: 8,
      distance: 200,
      pan: [0, 0],
      width: 1,
      height: 1,
    };
    this.sizeObserver = new ResizeObserver(() => this.resize());
    this.sizeObserver.observe(canvas);
    this.bind();
  }
  setScene(data) {
    this.groups = data.groups;
    this.courts = data.courts;
    this.edges = data.edges;
    const pts = [
      ...this.groups.flatMap((g) => g.points),
      ...this.courts.flatMap((c) => c.points),
    ];
    const low = [0, 1, 2].map((i) => Math.min(...pts.map((p) => p[i]))),
      high = [0, 1, 2].map((i) => Math.max(...pts.map((p) => p[i])));
    this.view.center = low.map((x, i) => (x + high[i]) / 2);
    this.extent = Math.max(...high.map((x, i) => x - low[i]), 20);
    this.selected = null;
    this.sampleIndex = null;
    this.reset();
  }
  resize() {
    const r = this.canvas.getBoundingClientRect(),
      dpr = Math.min(devicePixelRatio || 1, 2);
    this.canvas.width = r.width * dpr;
    this.canvas.height = r.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.view.width = r.width;
    this.view.height = r.height;
    this.schedule();
  }
  reset() {
    this.view.yaw = -0.55;
    this.view.pitch = 0.85;
    this.view.pan = [0, 0];
    this.view.distance = (this.extent || 60) * 4;
    this.view.scale =
      (Math.min(this.view.width, this.view.height) / (this.extent || 60)) *
      0.75;
    this.schedule();
  }
  select(id, index = null) {
    this.selected = id;
    this.sampleIndex = index;
    this.schedule();
  }
  schedule() {
    if (!this.pending) {
      this.pending = true;
      requestAnimationFrame(() => {
        this.pending = false;
        this.draw();
      });
    }
  }
  line(points, color, width = 1, alpha = 1) {
    const c = this.ctx;
    c.strokeStyle = color;
    c.lineWidth = width;
    c.globalAlpha = alpha;
    c.beginPath();
    points.forEach((p, i) => {
      const q = project(p, this.view);
      i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]);
    });
    c.stroke();
    c.globalAlpha = 1;
  }
  draw() {
    const c = this.ctx,
      v = this.view;
    c.clearRect(0, 0, v.width, v.height);
    this.hits = [];
    const size = Math.ceil((this.extent || 60) / 10) * 10;
    for (let n = -size; n <= size; n += 5) {
      this.line(
        [
          [v.center[0] + n, v.center[1] - size, 0],
          [v.center[0] + n, v.center[1] + size, 0],
        ],
        "#354952",
        0.6,
        0.35,
      );
      this.line(
        [
          [v.center[0] - size, v.center[1] + n, 0],
          [v.center[0] + size, v.center[1] + n, 0],
        ],
        "#354952",
        0.6,
        0.35,
      );
    }
    for (const court of this.courts) {
      c.fillStyle = "#244438";
      c.globalAlpha = 0.4;
      c.beginPath();
      [0, 1, 3, 2].forEach((i, n) => {
        const q = project(court.points[i], v);
        n ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]);
      });
      c.closePath();
      c.fill();
      c.globalAlpha = 1;
      for (const [a, b] of this.edges) {
        if (court.points[a] && court.points[b])
          this.line([court.points[a], court.points[b]], "#b1cac3", 1.2, 0.8);
      }
      const q = project(court.points[0], v);
      c.fillStyle = "#98b4b0";
      c.font = "10px system-ui";
      c.fillText(court.id, q[0] + 5, q[1] - 8);
    }
    const ordered = [...this.groups].sort(
      (a, b) => (a.id === this.selected) - (b.id === this.selected),
    );
    for (const g of ordered) {
      const active = g.id === this.selected,
        color = COLORS[g.split] || "#fff",
        seen = new Set(),
        unique = [];
      g.points.forEach((p, i) => {
        const key = g.samples[i].frame;
        if (!seen.has(key)) {
          seen.add(key);
          unique.push(p);
        }
      });
      this.line(
        unique,
        color,
        active ? 3 : 1.3,
        active ? 1 : this.selected ? 0.22 : 0.5,
      );
      const points = unique.map((p) => project(p, v));
      for (let i = 1; i < points.length; i++)
        this.hits.push({ id: g.id, a: points[i - 1], b: points[i] });
      c.fillStyle = color;
      for (const q of points) {
        c.globalAlpha = active ? 0.9 : this.selected ? 0.22 : 0.45;
        c.beginPath();
        c.arc(q[0], q[1], active ? 2.2 : 1.3, 0, Math.PI * 2);
        c.fill();
      }
      c.globalAlpha = 1;
      if (active) {
        for (
          let i = 0;
          i < g.points.length;
          i += Math.max(1, Math.floor(g.points.length / 12))
        )
          this.line([g.points[i], g.forwards[i]], color, 1, 0.6);
        if (this.sampleIndex !== null) {
          const i = this.sampleIndex,
            q = project(g.points[i], v);
          this.line([g.points[i], g.forwards[i]], "#fff", 3);
          c.strokeStyle = "#fff";
          c.lineWidth = 2;
          c.beginPath();
          c.arc(q[0], q[1], 7, 0, Math.PI * 2);
          c.stroke();
        }
      }
    }
  }
  pick(x, y) {
    let best = null,
      d = 9;
    for (const h of this.hits) {
      const distance = distanceToSegment([x, y], h.a, h.b);
      if (distance < d) {
        best = h.id;
        d = distance;
      }
    }
    return best;
  }
  bind() {
    let drag = null;
    const canvas = this.canvas;
    canvas.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 && e.button !== 1) return;
      drag = {
        x: e.clientX,
        y: e.clientY,
        startX: e.clientX,
        startY: e.clientY,
        moved: 0,
      };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      const r = canvas.getBoundingClientRect();
      if (!drag) {
        canvas.style.cursor = this.pick(e.clientX - r.left, e.clientY - r.top)
          ? "pointer"
          : "grab";
        return;
      }
      const dx = e.clientX - drag.x,
        dy = e.clientY - drag.y;
      drag.moved = Math.max(
        drag.moved,
        Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY),
      );
      drag.x = e.clientX;
      drag.y = e.clientY;
      if (e.shiftKey || e.buttons === 4) {
        this.view.pan[0] += dx;
        this.view.pan[1] += dy;
      } else {
        this.view.yaw += dx * 0.008;
        this.view.pitch = Math.max(
          0.06,
          Math.min(1.52, this.view.pitch + dy * 0.008),
        );
      }
      this.schedule();
    });
    canvas.addEventListener("pointerup", (e) => {
      if (drag && drag.moved < 5) {
        const r = canvas.getBoundingClientRect(),
          id = this.pick(e.clientX - r.left, e.clientY - r.top);
        if (id) this.onSelect(id);
      }
      drag = null;
    });
    canvas.addEventListener("pointercancel", () => {
      drag = null;
    });
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.view.scale = Math.max(
          0.4,
          Math.min(120, this.view.scale * Math.exp(-e.deltaY * 0.001)),
        );
        this.schedule();
      },
      { passive: false },
    );
    canvas.addEventListener("keydown", (e) => {
      if (e.key === "Home") {
        e.preventDefault();
        this.reset();
      }
    });
  }
}
