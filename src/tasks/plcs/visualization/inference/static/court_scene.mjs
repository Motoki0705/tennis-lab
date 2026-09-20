// Court-scene data model for the PLCS inference UI.
//
// World frame: right-handed court metres, +Z up, X across the court width,
// Y along the court length, net at y = 0.  The court keypoints and both the GT
// and prediction tracks arrive from /api/predict (or the GPU-free preview)
// already in this frame.  The WebGL scene is the shared Three.js engine at
// /shared/scene3d.mjs; this module keeps the pure projection/series math that
// the unit tests exercise and the ``CourtScene`` container that slices one
// framed payload into per-track typed-array views.

const WORLD_UP = [0, 0, 1];
const NEAR = 0.05;
const MAX_PITCH = 1.45;
const MIN_PITCH = -0.3;
// CourtKP20: keypoints 14..19 describe the net (centre, posts, strap).
const NET_FIRST = 14;
// Curtain outline: left post base -> net centre ground -> right post base, then
// up the right post, across the top cable and back down the left post.
const NET_CURTAIN = [15, 14, 17, 18, 19, 16];
const CENTER_STRAP = [14, 19];

export const COLORS = {
  skyTop: "#eef3f9",
  skyBottom: "#dde6ef",
  courtFill: "rgba(120, 150, 130, 0.14)",
  courtLine: "#6a7b8a",
  net: "#7b3ff2",
  netFill: "rgba(123, 63, 242, 0.16)",
  gt: "#1f6fb0",
  pred: "#c25a0a",
  gtTrail: "rgba(31, 111, 176, 0.32)",
  predTrail: "rgba(194, 90, 10, 0.32)",
  gtLeft: "#1f6fb0",
  gtRight: "#63b3e8",
  gtCore: "#123a5c",
  predLeft: "#c25a0a",
  predRight: "#f0a24a",
  predCore: "#7a3405",
  jointOutline: "#ffffff",
  marker: "#101820",
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

/**
 * World-space bounds of the supplied tracks.
 *
 * Each track is ``{position: Float32Array, joints: Float32Array|null}`` with
 * flat ``[frame, axis]`` and ``[frame, joint, axis]`` layouts; both are walked
 * so a track with a skeleton frames as tightly as its joints, not just its
 * root path.
 */
export function trackBounds(tracks) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const visit = (values) => {
    if (!values) {
      return;
    }
    for (let index = 0; index + 3 <= values.length; index += 3) {
      for (let axis = 0; axis < 3; axis += 1) {
        const value = values[index + axis];
        if (value < min[axis]) min[axis] = value;
        if (value > max[axis]) max[axis] = value;
      }
    }
  };
  for (const track of tracks) {
    visit(track.position);
    visit(track.joints);
  }
  if (!Number.isFinite(min[0])) {
    return { min: [0, 0, 0], max: [0, 0, 0], center: [0, 0, 0.5], radius: 6 };
  }
  const center = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];
  const radius = Math.max(length(subtract(max, center)), 1);
  return { min, max, center, radius };
}

/** Distance (metres) that frames a bounding sphere of ``radius`` in the fov. */
export function fitDistance(radius, halfFov) {
  return Math.max(radius / (0.62 * Math.tan(halfFov)), 1.5);
}

/** Classify a COCO-17 joint as left, right, or core from its name. */
export function jointSide(name) {
  if (name.startsWith("left")) return "left";
  if (name.startsWith("right")) return "right";
  return "core";
}

function pickPayload(payload) {
  if (payload instanceof Float32Array) {
    return payload;
  }
  for (const key of ["data", "buffer", "payload", "floats"]) {
    const value = payload ? payload[key] : null;
    if (value instanceof Float32Array) {
      return value;
    }
  }
  return new Float32Array(0);
}

function slice(data, descriptor) {
  if (!data || !descriptor) {
    return null;
  }
  const offset = Number(descriptor.offset ?? 0);
  const count = Number(descriptor.count ?? 0);
  if (!Number.isFinite(offset) || !Number.isFinite(count) || count <= 0) {
    return null;
  }
  return data.subarray(offset, offset + count);
}

function shapeFrame(descriptor) {
  const shape = descriptor ? descriptor.shape : null;
  return Array.isArray(shape) && shape.length > 0 ? Number(shape[0]) : null;
}

function shapeJoint(descriptor) {
  const shape = descriptor ? descriptor.shape : null;
  return Array.isArray(shape) && shape.length > 1 ? Number(shape[1]) : null;
}

/**
 * One decoded inference payload as a data container.
 *
 * ``setData`` accepts the framed response (``{header, data}``) or a flat object
 * carrying ``court``/``skeleton``/``tracks`` plus a ``data`` Float32Array. Track
 * slices come straight from the float32-element ``offset``/``count`` and alias
 * the payload buffer, so nothing is copied and the numeric payload stays the
 * single source of truth. Drawing is delegated to the shared Three.js engine;
 * this class only models state (frame, visibility, tracks).
 */
export class CourtScene {
  constructor() {
    this.court = { keypoints: [], edges: [] };
    this.skeleton = { names: [], edges: [], sides: [] };
    this.tracks = [];
    this.frame = 0;
    this.frameCount = 0;
    this.visible = { gt: true, pred: true };
    this.showTrails = true;
    this.mode = null;
  }

  /**
   * Install one inference/preview payload.
   *
   * Each ``header.tracks[*]`` entry carries ``position``, ``rotation``,
   * ``presence`` and (optionally) ``joints`` descriptors, each
   * ``{offset, count, shape}`` in float32 elements. ``presence`` is a ``(T,)``
   * 0/1 mask honoured by the renderer; ``rotation`` is the ``(T, 2)``
   * ``(cos, sin)`` yaw. Multi-object predictions leave ``joints`` null.
   */
  setData(payload) {
    const source = payload || {};
    const header = source.header ? source.header : source;
    const data = pickPayload(source);

    const court = header.court || {};
    this.court = {
      keypoints: (court.keypoints || []).map((point) => [
        Number(point[0]),
        Number(point[1]),
        Number(point[2] ?? 0),
      ]),
      edges: (court.edges || []).map((edge) => [Number(edge[0]), Number(edge[1])]),
    };

    const skeleton = header.skeleton || {};
    const names = skeleton.names || [];
    this.skeleton = {
      names,
      edges: (skeleton.edges || []).map((edge) => [Number(edge[0]), Number(edge[1])]),
      sides: names.map(jointSide),
    };
    this.mode = header.mode ?? null;

    const tracks = [];
    for (const raw of header.tracks || []) {
      const position = slice(data, raw.position);
      const joints = raw.has_joints && raw.joints ? slice(data, raw.joints) : null;
      const rotation = slice(data, raw.rotation);
      const presence = slice(data, raw.presence);
      const frameCount = shapeFrame(raw.position) ?? shapeFrame(raw.joints) ?? 0;
      const jointCount = (joints && shapeJoint(raw.joints)) || names.length || 0;
      tracks.push({
        kind: raw.kind,
        label: raw.label ?? raw.kind,
        objectIndex: Number(raw.object_index ?? 0),
        hasJoints: Boolean(joints),
        frameCount,
        jointCount,
        position: position || new Float32Array(0),
        joints,
        rotation,
        presence,
      });
      if (!(raw.kind in this.visible)) {
        this.visible[raw.kind] = true;
      }
    }
    this.tracks = tracks;
    this.frameCount = tracks.reduce(
      (largest, track) => Math.max(largest, track.frameCount),
      0,
    );
    this.frame = clamp(this.frame, 0, Math.max(0, this.frameCount - 1));
  }

  clear() {
    this.tracks = [];
    this.frameCount = 0;
    this.frame = 0;
    this.mode = null;
  }

  setFrame(frame) {
    this.frame = clamp(Math.round(frame), 0, Math.max(0, this.frameCount - 1));
  }

  setKindVisible(kind, visible) {
    this.visible[kind] = Boolean(visible);
  }

  setTrackVisible(kind, visible) {
    this.setKindVisible(kind, visible);
  }

  setTrailsVisible(visible) {
    this.showTrails = Boolean(visible);
  }

  toggleKind(kind) {
    this.setKindVisible(kind, !this.visible[kind]);
    return this.visible[kind];
  }
}
