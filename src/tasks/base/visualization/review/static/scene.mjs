// Projection and vector math shared by the dataset review tooling.
//
// World frame: right-handed, +Z up, metres, physical court frame. The WebGL
// scene itself lives in the shared Three.js engine at /shared/scene3d.mjs;
// this module keeps the pure, dependency-free helpers that the unit tests and
// the legend share.

export const WORLD_UP = [0, 0, 1];
export const NEAR = 0.05;
export const MIN_PITCH = -1.5;
export const MAX_PITCH = 1.5;
export const BALL_RADIUS_M = 0.055;

export const COLORS = {
  background: "#eef1f4",
  backgroundFar: "#dfe5ec",
  grid: "rgba(120, 134, 150, 0.20)",
  gridMajor: "rgba(90, 106, 124, 0.32)",
  apron: "#93a68f",
  court: "#3f7d64",
  line: "#f2f5f2",
  net: "rgba(240, 244, 248, 0.34)",
  netLine: "#eef2f6",
  post: "#c8d0d8",
  axisX: "#cf4b5b",
  axisY: "#1f8a76",
  axisZ: "#3a6fb0",
  camera: "#5b6b7f",
  cameraSelected: "#d9731f",
  cameraLabel: "#38485a",
  shadow: "rgba(38, 54, 72, 0.18)",
  trail: "rgba(224, 120, 40, 0.85)",
  trailFaint: "rgba(224, 120, 40, 0.20)",
};

export const PRESETS = {
  overhead: { yaw: -Math.PI / 2, pitch: 1.45, distance: 34, target: [0, 0, 0] },
  corner: { yaw: -2.35, pitch: 0.5, distance: 30, target: [0, 0, 1] },
  broadcast: { yaw: -Math.PI / 2, pitch: 0.24, distance: 44, target: [0, 0, 1.2] },
  side: { yaw: 0, pitch: 0.12, distance: 34, target: [0, 0, 1] },
};

// ---------------------------------------------------------------- vector math

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

/** Convert one world point into camera coordinates (x right, y up, z forward). */
export function toCamera(point, camera) {
  const relative = subtract(point, camera.eye);
  return [dot(relative, camera.right), dot(relative, camera.up), dot(relative, camera.forward)];
}

/** Project a camera-space point to [x, y] pixels. Caller clips z first. */
export function projectCamera(point, view) {
  const factor = view.focal / point[2];
  return [view.cx + point[0] * factor, view.cy - point[1] * factor];
}

/** Project one world point. Returns [x, y, depth] or null when behind. */
export function projectPoint(point, camera, view) {
  const cameraPoint = toCamera(point, camera);
  if (cameraPoint[2] <= NEAR) {
    return null;
  }
  const [x, y] = projectCamera(cameraPoint, view);
  return [x, y, cameraPoint[2]];
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

/** Intersect segment a-b (camera space) with the plane z = near. */
export function intersectNear(a, b, near) {
  const t = (near - a[2]) / (b[2] - a[2]);
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, near];
}

/** Clip one world segment at the near plane and project it, or return null. */
export function clipSegment(a, b, camera, view) {
  const ca = toCamera(a, camera);
  const cb = toCamera(b, camera);
  const aIn = ca[2] >= NEAR;
  const bIn = cb[2] >= NEAR;
  if (!aIn && !bIn) {
    return null;
  }
  const pa = aIn ? ca : intersectNear(ca, cb, NEAR);
  const pb = bIn ? cb : intersectNear(cb, ca, NEAR);
  return {
    a: projectCamera(pa, view),
    b: projectCamera(pb, view),
    depth: (ca[2] + cb[2]) / 2,
  };
}

/** Sutherland-Hodgman clip of a camera-space polygon against z >= near. */
export function clipPolygonNear(cameraPoints, near) {
  const out = [];
  const count = cameraPoints.length;
  for (let index = 0; index < count; index += 1) {
    const current = cameraPoints[index];
    const previous = cameraPoints[(index - 1 + count) % count];
    const currentIn = current[2] >= near;
    const previousIn = previous[2] >= near;
    if (currentIn) {
      if (!previousIn) {
        out.push(intersectNear(previous, current, near));
      }
      out.push(current);
    } else if (previousIn) {
      out.push(intersectNear(previous, current, near));
    }
  }
  return out;
}

/** Project a filled polygon clipped at the near plane, or null if empty. */
export function projectPolygon(points, camera, view) {
  const cameraPoints = points.map((point) => toCamera(point, camera));
  const clipped = clipPolygonNear(cameraPoints, NEAR);
  if (clipped.length < 3) {
    return null;
  }
  return clipped.map((point) => projectCamera(point, view));
}

/** Sort drawable items far-to-near so nearer geometry paints last. */
export function sortFarToNear(items) {
  return [...items].sort((left, right) => right.depth - left.depth);
}

/** Return the eight frustum index pairs (centre rays first, then perimeter). */
export function frustumEdges() {
  return [
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 1],
  ];
}

/** Return projected frustum edge segments for one scene camera. */
export function frustumSegments(frustum, camera, view) {
  return sortFarToNear(
    frustumEdges()
      .map(([a, b]) => clipSegment(frustum[a], frustum[b], camera, view))
      .filter((segment) => segment !== null),
  );
}

export function strideFor(frameCount, budget) {
  return Math.max(1, Math.ceil(frameCount / budget));
}
