// Projection and series math for the BLCS inference review.
//
// World frame: right-handed, +Z up, metres (court contract "physical_v1").
// x is lateral across the court, y runs along the court, z is height. The WebGL
// scene is the shared Three.js engine at /shared/scene3d.mjs; this module keeps
// the pure, dependency-free helpers that the unit tests exercise.

const WORLD_UP = [0, 0, 1];
const NEAR = 0.05;
const MAX_PITCH = 1.5;
const MIN_PITCH = -0.45;
const GROUND_EPS = 0.02;

export const COLORS = {
  sky: "#eef2f7",
  floor: "rgba(120, 138, 158, 0.14)",
  floorMajor: "rgba(96, 114, 136, 0.24)",
  courtLine: "#4d6478",
  courtLineStrong: "#2f4256",
  net: "rgba(96, 114, 136, 0.20)",
  netLine: "#6a7f93",
  post: "#47596b",
  axisX: "#cf4b5b",
  axisY: "#1f8a76",
  axisZ: "#3a6fb0",
  shadow: "rgba(50, 70, 92, 0.18)",
};

// Four shades per family so multiple tracks stay distinguishable.
export const GT_SHADES = ["#2f8f5b", "#3fa96c", "#5cc084", "#7fd3a2"];
export const PRED_SHADES = ["#d0552f", "#e0724d", "#ec9172", "#f3b199"];

export const GT_PRIMARY = GT_SHADES[0];
export const PRED_PRIMARY = PRED_SHADES[0];

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

/** Distance that frames a bounding sphere of ``radius`` inside the vertical fov. */
export function fitDistance(radius, halfFov) {
  return Math.max(radius / (0.62 * Math.tan(halfFov)), 1.5);
}

/**
 * Search the smallest camera distance for ``view`` that keeps every ``points``
 * entry inside the framed viewport. ``view`` must carry width, height, focal,
 * cx, cy, yaw, pitch and target. Used so the court always fits on load.
 */
export function fitViewDistance(points, view, margin = 0.06) {
  const { width, height } = view;
  if (!points.length || !width || !height) {
    return fitDistance(1, view.halfFov);
  }
  const insetX = width * margin;
  const insetY = height * margin;
  const fits = (distance) => {
    const probe = { ...view, distance };
    const camera = cameraBasis(probe);
    for (const point of points) {
      const screen = projectPoint(point, camera, probe);
      if (screen === null) return false;
      if (screen[0] < insetX || screen[0] > width - insetX) return false;
      if (screen[1] < insetY || screen[1] > height - insetY) return false;
    }
    return true;
  };
  let hi = 1;
  while (!fits(hi) && hi < 1e6) {
    hi *= 1.7;
  }
  let lo = 0;
  for (let step = 0; step < 48; step += 1) {
    const mid = (lo + hi) / 2;
    if (fits(mid)) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  return hi;
}

/** World-space bounds of a court keypoint list, centred on the ground plane. */
export function courtBounds(keypoints) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const point of keypoints) {
    for (let axis = 0; axis < 3; axis += 1) {
      const value = point[axis];
      if (value < min[axis]) min[axis] = value;
      if (value > max[axis]) max[axis] = value;
    }
  }
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, 0];
  let radius = 0;
  for (const point of keypoints) {
    radius = Math.max(radius, length(subtract(point, center)));
  }
  return { min, max, center, radius };
}

/** Read one ``(frame, track)`` position from a flat ``(frames, tracks, 3)`` buffer. */
export function positionAt(positions, tracks, frame, track) {
  const offset = (frame * tracks + track) * 3;
  return [positions[offset], positions[offset + 1], positions[offset + 2]];
}

/** Read one ``(frame, track)`` active flag from a flat ``(frames, tracks)`` mask. */
export function activeAt(mask, tracks, frame, track) {
  return mask[frame * tracks + track] > 0;
}

/**
 * Normalize a flat ``(frames, tracks, positions)`` series into the shape the
 * renderer consumes. Prediction payloads carry ``presence`` instead of
 * ``active``; a ``null`` presence means every frame is present.
 */
export function normalizeSeries(series) {
  if (!series) return null;
  const frames = series.frames | 0;
  const tracks = series.tracks | 0;
  if (frames <= 0 || tracks <= 0) return null;
  const raw = series.active ?? series.presence ?? null;
  const mask = new Uint8Array(frames * tracks).fill(1);
  if (raw) {
    for (let i = 0; i < mask.length; i += 1) {
      mask[i] = raw[i] > 0 ? 1 : 0;
    }
  }
  return { frames, tracks, positions: series.positions, mask };
}

/** The five world points describing the net: two posts plus the center strap. */
export function netGeometry(keypoints) {
  if (!keypoints || keypoints.length < 20) return null;
  return {
    leftBase: keypoints[15],
    leftTop: keypoints[16],
    rightBase: keypoints[17],
    rightTop: keypoints[18],
    centerTop: keypoints[19],
  };
}

// Broadcast-style presets. ``fit`` scales the fitted distance to tune framing.
export const VIEW_PRESETS = {
  broadcast: { yaw: -Math.PI / 2, pitch: 0.34, fit: 0.92 },
  side: { yaw: 0, pitch: 0.16, fit: 0.9 },
  overhead: { yaw: -Math.PI / 2, pitch: 1.44, fit: 0.98 },
};
