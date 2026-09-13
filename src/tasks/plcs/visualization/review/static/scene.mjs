// Software 3D renderer for SMPL-H world joints on a 2D canvas.
//
// World frame: right-handed, +Z up, metres (PLCS "plcs_amass_smplh_z_up_v1").
// The camera orbits a target point; drag rotates, shift+drag pans, the wheel
// dollies. Bones and joints are painted far-to-near so the skeleton reads
// correctly without a depth buffer.

const WORLD_UP = [0, 0, 1];
const NEAR = 0.05;
const MAX_PITCH = 1.5;
const MIN_PITCH = -0.45;
// A person is roughly this tall in metres; follow mode frames that, not the
// whole trajectory, so long clips stay readable when tracking the root.
const FOCUS_RADIUS_M = 1.2;

export const COLORS = {
  skyTop: "#eef3f9",
  skyBottom: "#dde6ef",
  grid: "rgba(114, 132, 152, 0.28)",
  gridMajor: "rgba(84, 104, 126, 0.44)",
  axisX: "#cf4b5b",
  axisY: "#1f8a76",
  axisZ: "#3a6fb0",
  left: "#11827b",
  right: "#d1623f",
  core: "#33465a",
  shadow: "rgba(50, 70, 92, 0.15)",
  trail: "#b8801f",
};

export function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

export function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function scale(a, k) {
  return [a[0] * k, a[1] * k, a[2] * k];
}

export function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function length(a) {
  return Math.hypot(a[0], a[1], a[2]);
}

export function normalize(a) {
  const size = length(a) || 1;
  return [a[0] / size, a[1] / size, a[2] / size];
}

/** Return the orthonormal camera basis for one orbit view. */
export function cameraBasis(view) {
  const pitch = clamp(view.pitch, MIN_PITCH, MAX_PITCH);
  const cp = Math.cos(pitch);
  const direction = [cp * Math.cos(view.yaw), cp * Math.sin(view.yaw), Math.sin(pitch)];
  const eye = add(view.target, scale(direction, view.distance));
  const forward = scale(direction, -1);
  let right = cross(forward, WORLD_UP);
  if (length(right) < 1e-6) {
    right = [1, 0, 0];
  } else {
    right = normalize(right);
  }
  return { eye, forward, right, up: cross(right, forward) };
}

/** Project one world point. Returns ``[x, y, depth]`` or ``null`` when behind. */
export function projectPoint(point, camera, view) {
  const relative = subtract(point, camera.eye);
  const depth = dot(relative, camera.forward);
  if (depth <= NEAR) {
    return null;
  }
  const factor = view.focal / depth;
  return [
    view.cx + dot(relative, camera.right) * factor,
    view.cy - dot(relative, camera.up) * factor,
    depth,
  ];
}

/** Project a polyline, splitting it at the near plane. */
export function projectPolyline(points, camera, view) {
  const runs = [];
  let current = [];
  for (const point of points) {
    const projected = projectPoint(point, camera, view);
    if (projected === null) {
      if (current.length > 1) {
        runs.push(current);
      }
      current = [];
      continue;
    }
    current.push(projected);
  }
  if (current.length > 1) {
    runs.push(current);
  }
  return runs;
}

/** Return the world-space bounds of a joint buffer over all frames. */
export function jointBounds(joints, frameCount, jointCount) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let frame = 0; frame < frameCount; frame += 1) {
    const base = frame * jointCount * 3;
    for (let joint = 0; joint < jointCount; joint += 1) {
      const offset = base + joint * 3;
      for (let axis = 0; axis < 3; axis += 1) {
        const value = joints[offset + axis];
        if (value < min[axis]) min[axis] = value;
        if (value > max[axis]) max[axis] = value;
      }
    }
  }
  const center = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];
  const radius = Math.max(
    length(subtract(max, center)),
    0.75,
  );
  return { min, max, center, radius };
}

/** Distance that frames a bounding sphere of ``radius`` inside the vertical fov. */
export function fitDistance(radius, halfFov) {
  return Math.max(radius / (0.62 * Math.tan(halfFov)), 1.5);
}

/** Classify a joint as left, right, or core from its SMPL-H name. */
export function jointSide(name) {
  if (name.startsWith("left")) return "left";
  if (name.startsWith("right")) return "right";
  return "core";
}

export class MotionView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.view = {
      yaw: -0.62,
      pitch: 0.32,
      distance: 6,
      target: [0, 0, 0.9],
      halfFov: (38 * Math.PI) / 360,
      focal: 1,
      cx: 0,
      cy: 0,
    };
    this.motion = null;
    this.frame = 0;
    this.follow = false;
    this.showTrail = true;
    this.dirty = true;
    this.onFollowChange = null;
    this._radius = 1;
    this._wideDistance = null;
    this._pointer = null;
    this._resize = new ResizeObserver(() => this.resize());
    this._resize.observe(canvas);
    this._bindPointer();
    this.resize();
  }

  /** Install a decoded motion: joints, skeleton edges, and joint names. */
  setMotion({ joints, frameCount, jointCount, names, edges }) {
    this.motion = {
      joints,
      frameCount,
      jointCount,
      edges,
      sides: names.map(jointSide),
      bounds: jointBounds(joints, frameCount, jointCount),
    };
    this.frame = 0;
    this.resetView();
    this.draw();
  }

  clearMotion() {
    this.motion = null;
    this.draw();
  }

  setFrame(frame) {
    if (!this.motion) return;
    this.frame = clamp(Math.round(frame), 0, this.motion.frameCount - 1);
    this.dirty = true;
  }

  /** Toggle root tracking, zooming in while following and back out after. */
  setFollow(follow) {
    if (follow === this.follow) return;
    this.follow = follow;
    if (!this.motion) return;
    if (follow) {
      this._wideDistance = this.view.distance;
      this.view.distance = fitDistance(
        Math.min(this.motion.bounds.radius, FOCUS_RADIUS_M),
        this.view.halfFov,
      );
    } else if (this._wideDistance !== null) {
      this.view.distance = this._wideDistance;
      this._wideDistance = null;
    }
    this.dirty = true;
  }

  /** Glide the camera target toward the current root; called every render. */
  updateFollow() {
    if (!this.follow || !this.motion) return;
    const root = this._joint(this.frame, 0);
    const damping = 0.35;
    this.view.target = [
      this.view.target[0] + (root[0] - this.view.target[0]) * damping,
      this.view.target[1] + (root[1] - this.view.target[1]) * damping,
      this.view.target[2],
    ];
    this.dirty = true;
  }

  resetView() {
    if (!this.motion) return;
    const { bounds } = this.motion;
    this._radius = bounds.radius;
    this.view.target = [bounds.center[0], bounds.center[1], bounds.center[2]];
    const wide = fitDistance(bounds.radius, this.view.halfFov);
    if (this.follow) {
      this._wideDistance = wide;
      this.view.distance = fitDistance(
        Math.min(bounds.radius, FOCUS_RADIUS_M),
        this.view.halfFov,
      );
    } else {
      this._wideDistance = null;
      this.view.distance = wide;
    }
    this.view.yaw = -0.62;
    this.view.pitch = 0.32;
    this.dirty = true;
  }

  _joint(frame, joint) {
    const { joints, jointCount } = this.motion;
    const offset = (frame * jointCount + joint) * 3;
    return [joints[offset], joints[offset + 1], joints[offset + 2]];
  }

  resize() {
    const ratio = window.devicePixelRatio || 1;
    const width = this.canvas.clientWidth || 1;
    const height = this.canvas.clientHeight || 1;
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.view.cx = width / 2;
    this.view.cy = height / 2;
    this.view.focal = height / 2 / Math.tan(this.view.halfFov);
    this.dirty = true;
  }

  _bindPointer() {
    const canvas = this.canvas;
    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      this._pointer = {
        id: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        pan: event.shiftKey || event.button === 1 || event.button === 2,
      };
      canvas.style.cursor = this._pointer.pan ? "move" : "grabbing";
    });
    canvas.addEventListener("pointermove", (event) => {
      const pointer = this._pointer;
      if (!pointer || pointer.id !== event.pointerId) return;
      const dx = event.clientX - pointer.x;
      const dy = event.clientY - pointer.y;
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      if (pointer.pan) {
        const camera = cameraBasis(this.view);
        const perPixel = this.view.distance / this.view.focal;
        const shift = add(
          scale(camera.right, -dx * perPixel),
          scale(camera.up, dy * perPixel),
        );
        this.view.target = add(this.view.target, shift);
        if (this.follow) {
          this.setFollow(false);
          this.onFollowChange?.(false);
        }
      } else {
        this.view.yaw -= dx * 0.0075;
        this.view.pitch = clamp(this.view.pitch + dy * 0.0075, MIN_PITCH, MAX_PITCH);
      }
      this.dirty = true;
    });
    const release = (event) => {
      if (this._pointer?.id === event.pointerId) {
        this._pointer = null;
        canvas.style.cursor = "grab";
      }
    };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("dblclick", () => {
      this.resetView();
      this.draw();
    });
    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const factor = Math.exp(event.deltaY * 0.0012);
        this.view.distance = clamp(this.view.distance * factor, 0.6, 400);
        this.dirty = true;
      },
      { passive: false },
    );
    canvas.style.cursor = "grab";
  }

  draw() {
    if (!this.dirty) return;
    this.dirty = false;
    const ctx = this.ctx;
    const width = this.canvas.clientWidth || 1;
    const height = this.canvas.clientHeight || 1;
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, COLORS.skyTop);
    gradient.addColorStop(1, COLORS.skyBottom);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    if (!this.motion) return;
    const camera = cameraBasis(this.view);
    this._drawGrid(ctx, camera);
    this._drawTrail(ctx, camera);
    this._drawSkeleton(ctx, camera, true);
    this._drawSkeleton(ctx, camera, false);
  }

  _gridExtent() {
    const { bounds } = this.motion;
    const margin = 1.5;
    const min = [bounds.min[0] - margin, bounds.min[1] - margin];
    const max = [bounds.max[0] + margin, bounds.max[1] + margin];
    const span = Math.max(max[0] - min[0], max[1] - min[1]);
    const step = span > 40 ? 5 : 1;
    return { min, max, step };
  }

  _drawGrid(ctx, camera) {
    const { min, max, step } = this._gridExtent();
    ctx.lineWidth = 1;
    const startX = Math.floor(min[0] / step) * step;
    const startY = Math.floor(min[1] / step) * step;
    for (let x = startX; x <= max[0]; x += step) {
      const points = [
        [x, min[1], 0],
        [x, max[1], 0],
      ];
      const runs = projectPolyline(points, camera, this.view);
      ctx.strokeStyle = Math.abs(x % (step * 5)) < 1e-6 ? COLORS.gridMajor : COLORS.grid;
      for (const run of runs) {
        ctx.beginPath();
        ctx.moveTo(run[0][0], run[0][1]);
        for (let i = 1; i < run.length; i += 1) ctx.lineTo(run[i][0], run[i][1]);
        ctx.stroke();
      }
    }
    for (let y = startY; y <= max[1]; y += step) {
      const points = [
        [min[0], y, 0],
        [max[0], y, 0],
      ];
      const runs = projectPolyline(points, camera, this.view);
      ctx.strokeStyle = Math.abs(y % (step * 5)) < 1e-6 ? COLORS.gridMajor : COLORS.grid;
      for (const run of runs) {
        ctx.beginPath();
        ctx.moveTo(run[0][0], run[0][1]);
        for (let i = 1; i < run.length; i += 1) ctx.lineTo(run[i][0], run[i][1]);
        ctx.stroke();
      }
    }
    const origin = projectPoint([0, 0, 0], camera, this.view);
    if (origin) {
      const axes = [
        { tip: [1.5, 0, 0], color: COLORS.axisX, label: "X" },
        { tip: [0, 1.5, 0], color: COLORS.axisY, label: "Y" },
        { tip: [0, 0, 1.5], color: COLORS.axisZ, label: "Z" },
      ];
      ctx.lineWidth = 2;
      ctx.font = "600 12px ui-monospace, SFMono-Regular, Menlo, monospace";
      for (const axis of axes) {
        const tip = projectPoint(axis.tip, camera, this.view);
        if (!tip) continue;
        ctx.strokeStyle = axis.color;
        ctx.beginPath();
        ctx.moveTo(origin[0], origin[1]);
        ctx.lineTo(tip[0], tip[1]);
        ctx.stroke();
        ctx.fillStyle = axis.color;
        ctx.fillText(axis.label, tip[0] + 4, tip[1] - 3);
      }
    }
  }

  _trailPoint(frame) {
    const root = this._joint(frame, 0);
    return [root[0], root[1], 0.008];
  }

  _drawTrail(ctx, camera) {
    if (!this.showTrail) return;
    const { frameCount } = this.motion;
    const stride = Math.max(1, Math.floor(frameCount / 900));
    const full = [];
    for (let frame = 0; frame < frameCount; frame += stride) {
      full.push(this._trailPoint(frame));
    }
    full.push(this._trailPoint(frameCount - 1));
    const passed = full.slice(0, Math.floor(this.frame / stride) + 1);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.strokeStyle = COLORS.trail;
    for (const [alpha, width, points] of [
      [0.24, 2, full],
      [0.85, 3, passed],
    ]) {
      if (points.length < 2) continue;
      for (const run of projectPolyline(points, camera, this.view)) {
        ctx.globalAlpha = alpha;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(run[0][0], run[0][1]);
        for (let i = 1; i < run.length; i += 1) ctx.lineTo(run[i][0], run[i][1]);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    const marker = projectPoint(this._trailPoint(this.frame), camera, this.view);
    if (marker) {
      ctx.fillStyle = COLORS.trail;
      ctx.beginPath();
      ctx.arc(marker[0], marker[1], 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  _drawSkeleton(ctx, camera, shadow) {
    const { joints, jointCount, edges, sides } = this.motion;
    const frame = this.frame;
    const base = frame * jointCount * 3;
    const points = [];
    for (let joint = 0; joint < jointCount; joint += 1) {
      const offset = base + joint * 3;
      const point = [joints[offset], joints[offset + 1], joints[offset + 2]];
      if (shadow) {
        point[2] = 0.004;
      }
      points.push(point);
    }
    const segments = [];
    for (const [a, b] of edges) {
      const pa = points[a];
      const pb = points[b];
      const projectedA = projectPoint(pa, camera, this.view);
      const projectedB = projectPoint(pb, camera, this.view);
      if (!projectedA || !projectedB) continue;
      const side = sides[b] !== "core" ? sides[b] : sides[a];
      const handBone = sides[a] !== "core" && sides[b] !== "core" && handEdge(a, b);
      segments.push({
        depth: (projectedA[2] + projectedB[2]) / 2,
        a: projectedA,
        b: projectedB,
        side,
        width: handBone ? 0.014 : 0.038,
      });
    }
    segments.sort((left, right) => right.depth - left.depth);
    ctx.lineCap = "round";
    for (const segment of segments) {
      const factor =
        (this.view.focal / segment.depth) * segment.width * (shadow ? 0.7 : 1);
      const width = clamp(factor, shadow ? 1 : 1.4, 14);
      if (shadow) {
        ctx.strokeStyle = COLORS.shadow;
      } else {
        ctx.strokeStyle = COLORS[segment.side];
      }
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(segment.a[0], segment.a[1]);
      ctx.lineTo(segment.b[0], segment.b[1]);
      ctx.stroke();
    }
    const jointsToDraw = [];
    for (let joint = 0; joint < jointCount; joint += 1) {
      const projected = projectPoint(points[joint], camera, this.view);
      if (!projected) continue;
      jointsToDraw.push({
        depth: projected[2],
        at: projected,
        side: sides[joint],
        radius: sides[joint] === "core" ? 0.032 : 0.022,
      });
    }
    jointsToDraw.sort((left, right) => right.depth - left.depth);
    for (const joint of jointsToDraw) {
      const radius = clamp(
        (this.view.focal / joint.depth) * joint.radius * (shadow ? 0.7 : 1),
        shadow ? 0.8 : 1.2,
        16,
      );
      ctx.fillStyle = shadow ? COLORS.shadow : COLORS[joint.side];
      ctx.beginPath();
      ctx.arc(joint.at[0], joint.at[1], radius, 0, Math.PI * 2);
      ctx.fill();
      if (!shadow) {
        ctx.strokeStyle = COLORS.core;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }
}

function handEdge(a, b) {
  return a >= 22 && b >= 22;
}

export { WORLD_UP, NEAR, MIN_PITCH, MAX_PITCH, FOCUS_RADIUS_M };
