// Shared Three.js court scene for the BLCS/PLCS dataset-review and inference
// UIs. Consumers stay unaware of Three.js: each app normalises its own JSON or
// typed arrays into the model consumed by ``setModel`` and keeps its public
// view API thin.
//
// World frame: right-handed, +Z up, metres, physical court frame. The camera
// orbits ``controls.target``: drag rotates, right/middle drag pans, the wheel
// dollies, and clicking a camera apex selects that camera.

import * as THREE from "./three.module.js";
import { OrbitControls } from "./OrbitControls.js";

export const BACKGROUND = 0xe8ecf0;

export const PALETTE = {
  background: 0xe8ecf0,
  ground: 0xdfe4e9,
  grid: 0xc6cfd8,
  gridMajor: 0xadb9c5,
  apron: 0x9fae9c,
  court: 0x3f7d64,
  line: 0xf1f5f2,
  net: 0xd7dfe7,
  netLine: 0xeef2f6,
  post: 0x9aa6b2,
  axisX: 0xcf4b5b,
  axisY: 0x1f8a76,
  axisZ: 0x3a6fb0,
  camera: 0x5b6b7f,
  cameraSelected: 0xd9731f,
  cameraLabel: 0x38485a,
  trail: 0xe07828,
  heading: 0x2f3a46,
};

// Orbit presets in the physical court frame. ``corner`` is the default.
export const PRESETS = {
  overhead: { yaw: -Math.PI / 2, pitch: 1.45, distance: 34, target: [0, 0, 0] },
  corner: { yaw: -2.35, pitch: 0.5, distance: 30, target: [0, 0, 1] },
  broadcast: { yaw: -Math.PI / 2, pitch: 0.24, distance: 44, target: [0, 0, 1.2] },
  side: { yaw: 0, pitch: 0.12, distance: 34, target: [0, 0, 1] },
};

const DEFAULT_PRESET = "corner";
const MIN_PITCH = -1.5;
const MAX_PITCH = 1.5;
const BALL_RADIUS_M = 0.055;
const JOINT_RADIUS_M = 0.05;
const PLAYER_ROOT_RADIUS_M = 0.34;
const CAMERA_APEX_RADIUS_M = 0.13;
const GROUND_EXTENT_M = 30;

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function toVector(target, point) {
  return target.set(point[0], point[1], point[2]);
}

/** Ground-plane polygon (world x/y) from the four doubles corners, expanded. */
export function deriveApron(keypoints, margin = 1.1) {
  if (!keypoints || keypoints.length < 4) return null;
  const corners = [keypoints[0], keypoints[1], keypoints[3], keypoints[2]];
  const xs = corners.map((corner) => corner[0]);
  const ys = corners.map((corner) => corner[1]);
  const minX = Math.min(...xs) - margin;
  const maxX = Math.max(...xs) + margin;
  const minY = Math.min(...ys) - margin;
  const maxY = Math.max(...ys) + margin;
  return [
    [minX, minY],
    [maxX, minY],
    [maxX, maxY],
    [minX, maxY],
  ];
}

/** Court-surface polygon (world x/y) from the four doubles corners. */
export function deriveSurface(keypoints) {
  if (!keypoints || keypoints.length < 4) return null;
  return [keypoints[0], keypoints[1], keypoints[3], keypoints[2]].map((corner) => [
    corner[0],
    corner[1],
  ]);
}

/** Net point set derived from CourtKP20 indices 15..19 (base, top, centre). */
export function deriveNet(keypoints) {
  if (!keypoints || keypoints.length < 20) return null;
  return {
    posts: [
      [keypoints[15], keypoints[16]],
      [keypoints[17], keypoints[18]],
    ],
    profile: {
      x: [keypoints[15][0], keypoints[19][0], keypoints[17][0]],
      z: [keypoints[16][2], keypoints[19][2], keypoints[18][2]],
    },
  };
}

/**
 * Normalise the court payload every task ships into one shape. ``apron`` and
 * ``net`` default from the CourtKP20 keypoints so the inference apps only need
 * to send ``{ keypoints, edges }``.
 */
export function normalizeCourt(court) {
  if (!court || !court.keypoints || court.keypoints.length < 4) return null;
  return {
    keypoints: court.keypoints,
    edges: court.edges || court.lines || [],
    apron: court.apron || deriveApron(court.keypoints),
    surface: court.surface || deriveSurface(court.keypoints),
    net: court.net || deriveNet(court.keypoints),
  };
}

/**
 * Return the world-space viewing direction and up vector for one scene camera.
 *
 * ``rotation`` is the world-to-camera matrix, so its row 2 is the optical axis
 * and row 1 the camera's down axis, giving ``up = -rotation[1]``. When only the
 * frustum is present (legacy BLCS-style payloads) the direction is the image
 * plane centroid minus the apex, never a single corner.
 */
export function cameraAxis(camera) {
  if (!camera) return null;
  const center = camera.center || camera.frustum?.[0];
  if (!center) return null;
  if (camera.rotation && camera.rotation.length === 3) {
    const axis = camera.rotation[2];
    const down = camera.rotation[1];
    const length = Math.hypot(axis[0], axis[1], axis[2]);
    if (length < 1e-9) return null;
    const downLength = Math.hypot(down[0], down[1], down[2]) || 1;
    return {
      direction: [axis[0] / length, axis[1] / length, axis[2] / length],
      up: [-down[0] / downLength, -down[1] / downLength, -down[2] / downLength],
    };
  }
  if (!camera.frustum || camera.frustum.length < 5) return null;
  const corners = camera.frustum.slice(1, 5);
  const centroid = corners.reduce(
    (sum, point) => [
      sum[0] + point[0] / corners.length,
      sum[1] + point[1] / corners.length,
      sum[2] + point[2] / corners.length,
    ],
    [0, 0, 0],
  );
  const direction = [
    centroid[0] - center[0],
    centroid[1] - center[1],
    centroid[2] - center[2],
  ];
  const length = Math.hypot(direction[0], direction[1], direction[2]);
  if (length < 1e-9) return null;
  return {
    direction: [direction[0] / length, direction[1] / length, direction[2] / length],
    up: [0, 0, 1],
  };
}

/**
 * Derive one OpenCV pinhole camera's five frustum vertices from its stored
 * parameters, matching ``src.utils.rendering.camera_geometry``: the vertices
 * are the camera centre followed by the top-left, top-right, bottom-right and
 * bottom-left image-plane corners at camera-Z ``depth``. ``params`` carries
 * ``C`` (camera centre), ``R`` (world-to-camera rotation, rows are the camera
 * axes), ``f``, ``cx``, ``cy``, ``w`` and ``h``. Returns ``null`` when the
 * parameters are missing, so callers can fall back to a provided ``frustum``.
 */
export function frustumFromParams(params, depth = 6) {
  if (!params) return null;
  const { C, R, f, cx, cy, w, h } = params;
  if (!C || !R || !(f > 0) || !(w > 0) || !(h > 0) || !(depth > 0)) return null;
  const pixelCorners = [
    [0, 0],
    [w - 1, 0],
    [w - 1, h - 1],
    [0, h - 1],
  ];
  const toWorld = (cameraPoint) => [
    R[0][0] * cameraPoint[0] + R[1][0] * cameraPoint[1] + R[2][0] * cameraPoint[2] + C[0],
    R[0][1] * cameraPoint[0] + R[1][1] * cameraPoint[1] + R[2][1] * cameraPoint[2] + C[1],
    R[0][2] * cameraPoint[0] + R[1][2] * cameraPoint[1] + R[2][2] * cameraPoint[2] + C[2],
  ];
  const corners = pixelCorners.map(([px, py]) => {
    const ray = [(px - cx) / f, (py - cy) / f, 1];
    return toWorld([ray[0] * depth, ray[1] * depth, depth]);
  });
  return [[C[0], C[1], C[2]], ...corners];
}

function polygonGeometry(polygon, z) {
  const shape = new THREE.Shape();
  shape.moveTo(polygon[0][0], polygon[0][1]);
  for (let index = 1; index < polygon.length; index += 1) {
    shape.lineTo(polygon[index][0], polygon[index][1]);
  }
  shape.closePath();
  const geometry = new THREE.ShapeGeometry(shape);
  geometry.translate(0, 0, z);
  return geometry;
}

/** Vertical net surface (world y = 0) from a sagging ``{ x, z }`` profile. */
function netGeometry(profile) {
  const positions = [];
  for (let index = 0; index < profile.x.length - 1; index += 1) {
    const x0 = profile.x[index];
    const x1 = profile.x[index + 1];
    const z0 = profile.z[index];
    const z1 = profile.z[index + 1];
    positions.push(x0, 0, z0, x1, 0, z1, x1, 0, 0);
    positions.push(x0, 0, z0, x1, 0, 0, x0, 0, 0);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  return geometry;
}

function lineGeometry(segments) {
  const positions = [];
  for (const [a, b] of segments) {
    positions.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  return geometry;
}

function gridGeometry(extent, step) {
  const segments = [];
  for (let offset = -extent; offset <= extent + 1e-6; offset += step) {
    segments.push([
      [-extent, offset, 0],
      [extent, offset, 0],
    ]);
    segments.push([
      [offset, -extent, 0],
      [offset, extent, 0],
    ]);
  }
  return lineGeometry(segments);
}

function labelSprite(text, color) {
  const canvas = document.createElement("canvas");
  const scale = 3;
  const context = canvas.getContext("2d");
  context.font = `600 ${13 * scale}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  const width = Math.ceil(context.measureText(text).width) + 18 * scale;
  canvas.width = width;
  canvas.height = 34 * scale;
  const draw = canvas.getContext("2d");
  draw.font = `600 ${13 * scale}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  draw.fillStyle = "rgba(255, 255, 255, 0.86)";
  draw.strokeStyle = "rgba(60, 74, 90, 0.35)";
  draw.lineWidth = scale;
  draw.beginPath();
  draw.rect(scale, scale, canvas.width - 2 * scale, canvas.height - 2 * scale);
  draw.fill();
  draw.stroke();
  draw.fillStyle = `#${new THREE.Color(color).getHexString()}`;
  draw.textBaseline = "middle";
  draw.fillText(text, 9 * scale, canvas.height / 2 + scale);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, depthWrite: false });
  const sprite = new THREE.Sprite(material);
  const worldHeight = 0.9;
  sprite.scale.set((canvas.width / canvas.height) * worldHeight, worldHeight, 1);
  return sprite;
}

function presenceAt(entity, frame) {
  if (!entity.presence) return true;
  return entity.presence[frame] === 1;
}

/**
 * Convert PLCS root yaw (flat ``T*2`` or ``T`` cos/sin pairs) to forward XY.
 * SMPL body forward is -Y in the yaw-zero Z-up pose, so the arrow is
 * Rz(yaw) @ [0, -1, 0] = [sin, -cos, 0].
 */
function normalizeHeadings(heading, frames) {
  const out = new Array(frames).fill(null);
  const flat = typeof heading[0] === "number";
  for (let frame = 0; frame < frames; frame += 1) {
    let cos;
    let sin;
    if (flat) {
      cos = heading[frame * 2];
      sin = heading[frame * 2 + 1];
    } else {
      const pair = heading[frame];
      if (!pair) continue;
      [cos, sin] = pair;
    }
    const norm = Math.hypot(cos, sin);
    if (!norm) continue;
    out[frame] = [sin / norm, -cos / norm];
  }
  return out;
}

function rootAt(entity, frame, out) {
  if (entity.roots) {
    const base = frame * 3;
    return out.set(entity.roots[base], entity.roots[base + 1], entity.roots[base + 2]);
  }
  const index = entity.rootIndex ?? 0;
  const base = (frame * entity.joints + index) * 3;
  return out.set(
    entity.positions[base],
    entity.positions[base + 1],
    entity.positions[base + 2],
  );
}

/**
 * Build the trail segment list for one entity. Only pairs of *consecutive*
 * present frames become segments, so a frame gap is never bridged by a straight
 * line across missing data. ``prefix[frame]`` is how many segments are complete
 * at that frame, which drives the incremental draw range as the playhead moves.
 */
export function trailSegments(frames, presence) {
  const pairs = [];
  const prefix = new Int32Array(Math.max(frames, 1));
  let count = 0;
  const present = (frame) => !presence || presence[frame] === 1;
  for (let frame = 0; frame < frames - 1; frame += 1) {
    if (present(frame) && present(frame + 1)) {
      pairs.push(frame, frame + 1);
      count += 1;
    }
    prefix[frame + 1] = count;
  }
  return { pairs, prefix, count };
}

/**
 * One shared WebGL scene. The canvas must be sized by CSS; the renderer tracks
 * its client box through a ``ResizeObserver``.
 */
export class Scene3D {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.fov = options.fov ?? 38;
    this.follow = false;
    this.showCameras = true;
    this.showTrail = true;
    this.selectedCamera = null;
    this.dirty = true;
    this.onCameraPick = options.onCameraPick ?? null;
    this.onFollowChange = options.onFollowChange ?? null;

    this.model = null;
    this.frame = 0;
    this._entities = [];
    this._cameras = [];
    this._followOffset = new THREE.Vector3();
    this._wideOffset = null;
    this._pointer = null;
    this._raycaster = new THREE.Raycaster();
    this._pointerNdc = new THREE.Vector2();
    this._pickables = [];
    this._scratch = new THREE.Vector3();
    this._matrix = new THREE.Matrix4();
    this._hidden = new THREE.Matrix4().makeScale(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(PALETTE.background, 1);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(PALETTE.background);

    this.camera = new THREE.PerspectiveCamera(this.fov, 1, 0.1, 2000);
    this.camera.up.set(0, 0, 1);

    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.courtGroup = new THREE.Group();
    this.entityGroup = new THREE.Group();
    this.cameraGroup = new THREE.Group();
    this.axisGroup = new THREE.Group();
    this.root.add(this.courtGroup, this.entityGroup, this.cameraGroup, this.axisGroup);

    this._buildControls();

    this._buildStatic();
    this.resetView();
    this._bindPointer();
    this._resizeObserver = new ResizeObserver(() => this.resize());
    this._resizeObserver.observe(canvas);
    this.resize();
  }

  /**
   * (Re)create the orbit controls so their internal orbit axis matches the
   * camera up vector. ``OrbitControls`` derives that axis from ``object.up`` in
   * its constructor only, so a camera that changes its up must rebuild.
   */
  _buildControls() {
    this.controls?.dispose?.();
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 400;
    this.controls.maxPolarAngle = Math.PI - 0.02;
    this.controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN,
    };
  }

  _setUp(up) {
    if (this.camera.up.distanceToSquared(new THREE.Vector3(up[0], up[1], up[2])) < 1e-12) {
      return;
    }
    this.camera.up.set(up[0], up[1], up[2]);
    this._buildControls();
  }

  // -------------------------------------------------------------- static bits

  _buildStatic() {
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(GROUND_EXTENT_M * 2, GROUND_EXTENT_M * 2),
      new THREE.MeshBasicMaterial({ color: PALETTE.ground }),
    );
    ground.position.z = -0.01;
    this.courtGroup.add(ground);

    const grid = new THREE.LineSegments(
      gridGeometry(GROUND_EXTENT_M, 2),
      new THREE.LineBasicMaterial({ color: PALETTE.grid, transparent: true, opacity: 0.5 }),
    );
    grid.position.z = -0.005;
    this.courtGroup.add(grid);

    const axisSegments = [
      [[0, 0, 0.01], [2.4, 0, 0.01]],
      [[0, 0, 0.01], [0, 2.4, 0.01]],
      [[0, 0, 0.01], [0, 0, 2.4]],
    ];
    const axisColors = [PALETTE.axisX, PALETTE.axisY, PALETTE.axisZ];
    axisSegments.forEach((segment, index) => {
      const line = new THREE.LineSegments(
        lineGeometry([segment]),
        new THREE.LineBasicMaterial({ color: axisColors[index] }),
      );
      this.axisGroup.add(line);
    });
  }

  // ------------------------------------------------------------------ model

  /** Install one full model: ``{ court, frames, entities, cameras }``. */
  setModel(model) {
    this.clearModel();
    this.model = model;
    this.frame = 0;
    if (!model) return;
    this._buildCourt(normalizeCourt(model.court));
    this._buildEntities(model.entities || []);
    this._buildCameras(model.cameras || []);
    this._applyFrame();
  }

  clearModel() {
    this._disposeGroup(this.courtGroup);
    this._disposeGroup(this.entityGroup);
    this._disposeGroup(this.cameraGroup);
    this._disposeGroup(this.axisGroup);
    this._entities = [];
    this._cameras = [];
    this._pickables = [];
    this.model = null;
    this.frame = 0;
    this.selectedCamera = null;
    this._buildStatic();
  }

  _disposeGroup(group) {
    for (let index = group.children.length - 1; index >= 0; index -= 1) {
      const child = group.children[index];
      group.remove(child);
      child.traverse?.((node) => {
        node.geometry?.dispose?.();
        const material = node.material;
        if (Array.isArray(material)) material.forEach((entry) => entry.dispose?.());
        else {
          material?.map?.dispose?.();
          material?.dispose?.();
        }
      });
    }
  }

  _buildCourt(court) {
    if (!court) return;
    if (court.apron) {
      const apron = new THREE.Mesh(
        polygonGeometry(court.apron, 0.001),
        new THREE.MeshBasicMaterial({ color: PALETTE.apron }),
      );
      this.courtGroup.add(apron);
    }
    if (court.surface) {
      const surface = new THREE.Mesh(
        polygonGeometry(court.surface, 0.002),
        new THREE.MeshBasicMaterial({ color: PALETTE.court }),
      );
      this.courtGroup.add(surface);
    }
    const keypoints = court.keypoints;
    const segments = court.edges
      .filter(([a, b]) => keypoints[a] && keypoints[b])
      .map(([a, b]) => [
        [keypoints[a][0], keypoints[a][1], 0.004],
        [keypoints[b][0], keypoints[b][1], 0.004],
      ]);
    if (segments.length) {
      this.courtGroup.add(
        new THREE.LineSegments(
          lineGeometry(segments),
          new THREE.LineBasicMaterial({ color: PALETTE.line }),
        ),
      );
    }
    this._buildNet(court.net, keypoints);
  }

  _buildNet(net, keypoints) {
    if (!net) return;
    if (net.profile && net.profile.x && net.profile.x.length > 1) {
      const surface = new THREE.Mesh(
        netGeometry(net.profile),
        new THREE.MeshBasicMaterial({
          color: PALETTE.net,
          transparent: true,
          opacity: 0.34,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      );
      this.courtGroup.add(surface);
      const topSegments = [];
      for (let index = 0; index < net.profile.x.length - 1; index += 1) {
        topSegments.push([
          [net.profile.x[index], 0, net.profile.z[index]],
          [net.profile.x[index + 1], 0, net.profile.z[index + 1]],
        ]);
      }
      this.courtGroup.add(
        new THREE.LineSegments(
          lineGeometry(topSegments),
          new THREE.LineBasicMaterial({ color: PALETTE.netLine }),
        ),
      );
    }
    for (const [base, tip] of net.posts || []) {
      const height = Math.abs(tip[2] - base[2]) || 1;
      const post = new THREE.Mesh(
        new THREE.CylinderGeometry(0.06, 0.06, height, 12),
        new THREE.MeshBasicMaterial({ color: PALETTE.post }),
      );
      post.rotation.x = Math.PI / 2;
      post.position.set(base[0], base[1], (base[2] + tip[2]) / 2);
      this.courtGroup.add(post);
    }
  }

  _buildEntities(entities) {
    const sphere = new THREE.SphereGeometry(1, 16, 12);
    for (const entity of entities) {
      const presence = entity.presence
        ? entity.presence instanceof Uint8Array
          ? entity.presence
          : Uint8Array.from(entity.presence, (value) => (value ? 1 : 0))
        : null;
      const normalized = { ...entity, presence };
      const group = new THREE.Group();
      const color = new THREE.Color(entity.color ?? PALETTE.court);

      const joints = entity.joints || 1;
      // Root-only players (e.g. multi-object PLCS predictions that carry no
      // joints) still need a marker a person reads at court scale, so they get
      // a larger body sphere; jointed players keep the joint-sized dots.
      const hasBones = Boolean(entity.edges && entity.edges.length);
      const radius =
        entity.radius ??
        (entity.kind === "ball"
          ? BALL_RADIUS_M
          : hasBones
            ? JOINT_RADIUS_M
            : PLAYER_ROOT_RADIUS_M);
      const markers = new THREE.InstancedMesh(
        sphere,
        new THREE.MeshBasicMaterial({ color }),
        joints,
      );
      markers.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      markers.frustumCulled = false;
      group.add(markers);

      let bones = null;
      if (entity.edges && entity.edges.length) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(new Float32Array(entity.edges.length * 2 * 3), 3),
        );
        bones = new THREE.LineSegments(
          geometry,
          new THREE.LineBasicMaterial({ color }),
        );
        bones.frustumCulled = false;
        group.add(bones);
      }

      const headings = entity.heading
        ? normalizeHeadings(entity.heading, entity.frames)
        : null;
      let headingLine = null;
      if (headings) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(new Float32Array(6 * 3), 3),
        );
        headingLine = new THREE.LineSegments(
          geometry,
          new THREE.LineBasicMaterial({ color: PALETTE.heading }),
        );
        headingLine.frustumCulled = false;
        group.add(headingLine);
      }

      const trail = this._buildTrail(normalized, color);
      group.add(trail.line);

      this.entityGroup.add(group);
      this._entities.push({
        ...normalized,
        group,
        markers,
        bones,
        headingLine,
        headings,
        trail,
        radius,
        color,
      });
    }
  }

  /**
   * Trails break at missing frames: only consecutive present pairs become
   * segments, and ``drawRange`` grows with the playhead so a gap is never
   * bridged by a straight line.
   */
  _buildTrail(entity, color) {
    const frames = entity.frames;
    const { pairs, prefix, count } = trailSegments(frames, entity.presence);
    const positions = new Float32Array(count * 2 * 3);
    const scratch = new THREE.Vector3();
    for (let index = 0; index < count; index += 1) {
      const a = pairs[index * 2];
      const b = pairs[index * 2 + 1];
      rootAt(entity, a, scratch);
      positions.set([scratch.x, scratch.y, scratch.z + 0.01], index * 6);
      rootAt(entity, b, scratch);
      positions.set([scratch.x, scratch.y, scratch.z + 0.01], index * 6 + 3);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const line = new THREE.LineSegments(
      geometry,
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.75 }),
    );
    line.frustumCulled = false;
    line.visible = this.showTrail;
    return { line, prefix, count };
  }

  _buildCameras(cameras) {
    const sphere = new THREE.SphereGeometry(CAMERA_APEX_RADIUS_M, 12, 10);
    for (const camera of cameras) {
      const group = new THREE.Group();
      const color = camera.color ?? PALETTE.camera;
      const frustum = camera.frustum || [];
      if (frustum.length >= 5) {
        const edges = [
          [0, 1],
          [0, 2],
          [0, 3],
          [0, 4],
          [1, 2],
          [2, 3],
          [3, 4],
          [4, 1],
        ].map(([a, b]) => [frustum[a], frustum[b]]);
        group.add(
          new THREE.LineSegments(
            lineGeometry(edges),
            new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.6 }),
          ),
        );
      }
      const apex = new THREE.Mesh(
        sphere,
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.75 }),
      );
      const center = camera.center || (frustum.length ? frustum[0] : [0, 0, 0]);
      apex.position.set(center[0], center[1], center[2]);
      group.add(apex);
      const label = labelSprite(String(camera.label ?? camera.id ?? ""), color);
      label.position.set(center[0], center[1], center[2] + 0.6);
      group.add(label);
      this.cameraGroup.add(group);
      const record = { ...camera, group, apex, label, color, frustum };
      this._cameras.push(record);
      apex.userData.cameraId = camera.id;
      label.userData.cameraId = camera.id;
      // The apex sphere and its label sprite are both pick targets so a click
      // anywhere on the camera marker (not just its few-pixel centre) selects it.
      this._pickables.push(apex, label);
    }
    this.cameraGroup.visible = this.showCameras;
    this._refreshCameraColors();
  }

  // ------------------------------------------------------------------ frame

  get frameCount() {
    return this.model?.frames || 0;
  }

  setFrame(frame) {
    if (!this.model) return;
    this.frame = clamp(Math.round(frame), 0, Math.max(0, this.frameCount - 1));
    this._applyFrame();
  }

  setEntityVisible(id, visible) {
    for (const entity of this._entities) {
      if (entity.id !== id) continue;
      entity.group.visible = visible;
    }
  }

  setCamerasVisible(visible) {
    this.showCameras = visible;
    this.cameraGroup.visible = visible;
  }

  setTrailsVisible(visible) {
    this.showTrail = visible;
    for (const entity of this._entities) {
      entity.trail.line.visible = visible;
    }
  }

  _applyFrame() {
    const frame = this.frame;
    for (const entity of this._entities) {
      // Entities may be shorter than the model (e.g. a BLCS window prediction
      // over a full-length GT); a frame outside an entity's own range is absent.
      const present = frame < entity.frames && presenceAt(entity, frame);
      const color = entity.color;
      for (let joint = 0; joint < entity.markers.count; joint += 1) {
        if (!present) {
          entity.markers.setMatrixAt(joint, this._hidden);
          continue;
        }
        const base = (frame * entity.joints + joint) * 3;
        this._scratch.set(
          entity.positions[base],
          entity.positions[base + 1],
          entity.positions[base + 2],
        );
        this._matrix.makeScale(entity.radius, entity.radius, entity.radius);
        this._matrix.setPosition(this._scratch);
        entity.markers.setMatrixAt(joint, this._matrix);
      }
      entity.markers.instanceMatrix.needsUpdate = true;
      entity.markers.visible = present;

      if (entity.bones) {
        entity.bones.visible = present;
        if (present) {
          const attribute = entity.bones.geometry.getAttribute("position");
          entity.edges.forEach(([a, b], index) => {
            const baseA = (frame * entity.joints + a) * 3;
            const baseB = (frame * entity.joints + b) * 3;
            attribute.setXYZ(
              index * 2,
              entity.positions[baseA],
              entity.positions[baseA + 1],
              entity.positions[baseA + 2],
            );
            attribute.setXYZ(
              index * 2 + 1,
              entity.positions[baseB],
              entity.positions[baseB + 1],
              entity.positions[baseB + 2],
            );
          });
          attribute.needsUpdate = true;
        }
      }

      if (entity.headingLine) {
        const direction = entity.headings[frame];
        const show = present && Boolean(direction);
        entity.headingLine.visible = show;
        if (show) {
          rootAt(entity, frame, this._scratch);
          const length = 0.9;
          const x = direction[0];
          const y = direction[1];
          const tip = [this._scratch.x + x * length, this._scratch.y + y * length, 0.03];
          const left = [
            tip[0] - x * 0.28 + y * 0.14,
            tip[1] - y * 0.28 - x * 0.14,
            0.03,
          ];
          const right = [
            tip[0] - x * 0.28 - y * 0.14,
            tip[1] - y * 0.28 + x * 0.14,
            0.03,
          ];
          const attribute = entity.headingLine.geometry.getAttribute("position");
          attribute.setXYZ(0, this._scratch.x, this._scratch.y, 0.03);
          attribute.setXYZ(1, tip[0], tip[1], tip[2]);
          attribute.setXYZ(2, tip[0], tip[1], tip[2]);
          attribute.setXYZ(3, left[0], left[1], left[2]);
          attribute.setXYZ(4, tip[0], tip[1], tip[2]);
          attribute.setXYZ(5, right[0], right[1], right[2]);
          attribute.needsUpdate = true;
        }
      }

      const visible = entity.trail.prefix[frame] ?? 0;
      entity.trail.line.geometry.setDrawRange(0, Math.min(visible, entity.trail.count) * 2);
      entity.trail.line.visible = this.showTrail;
      entity.trail.line.material.color.copy(color);
    }
    if (this.follow) this._snapFollow();
  }

  // ------------------------------------------------------------------ follow

  setFollow(follow, notify = true) {
    if (follow === this.follow) return;
    this.follow = follow;
    if (follow) {
      this._wideOffset = this.camera.position.clone().sub(this.controls.target);
      this._followOffset.copy(this._wideOffset).multiplyScalar(0.42);
      this._snapFollow(true);
    } else if (this._wideOffset) {
      this.camera.position.copy(this.controls.target).add(this._wideOffset);
      this._wideOffset = null;
    }
    if (notify) this.onFollowChange?.(follow);
    this.dirty = true;
  }

  _followTarget(out) {
    if (!this.model) return null;
    const entity = this._entities.find((entry) => entry.trail.count > 0) || this._entities[0];
    if (!entity || !presenceAt(entity, this.frame)) return null;
    return rootAt(entity, this.frame, out);
  }

  _snapFollow(immediate = false) {
    const target = this._followTarget(this._scratch);
    if (!target) return;
    this.controls.target.lerp(target, immediate ? 1 : 0.25);
    this.camera.position.copy(this.controls.target).add(this._followOffset);
  }

  // ----------------------------------------------------------------- cameras

  selectCamera(id) {
    this.selectedCamera = id ?? null;
    this._refreshCameraColors();
    this.dirty = true;
  }

  _refreshCameraColors() {
    for (const camera of this._cameras) {
      const selected = camera.id === this.selectedCamera;
      const material = camera.apex.material;
      material.color.set(selected ? PALETTE.cameraSelected : camera.color);
      camera.group.children.forEach((child) => {
        if (child.isLineSegments) {
          child.material.color.set(selected ? PALETTE.cameraSelected : camera.color);
          child.material.opacity = selected ? 0.95 : 0.6;
        }
      });
    }
  }

  _pick(clientX, clientY) {
    if (!this._pickables.length) return null;
    const rect = this.canvas.getBoundingClientRect();
    this._pointerNdc.set(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    this._raycaster.setFromCamera(this._pointerNdc, this.camera);
    const hits = this._raycaster.intersectObjects(this._pickables, false);
    return hits.length ? hits[0].object.userData.cameraId : null;
  }

  // -------------------------------------------------------------------- view

  applyPreset(name) {
    const preset = PRESETS[name] || PRESETS[DEFAULT_PRESET];
    this._setUp([0, 0, 1]);
    this._setOrbit(preset.yaw, preset.pitch, preset.distance, preset.target);
    if (this.follow) this.setFollow(false);
  }

  /**
   * Place the orbit camera explicitly. Apps that fit the camera to their own
   * domain (e.g. point-cloud framing) drive the engine through this rather than
   * the fixed presets.
   */
  setOrbit({ yaw, pitch, distance, target, up = [0, 0, 1] }) {
    this._setOrbit(yaw, pitch, distance, target, up);
    if (this.follow) this.setFollow(false);
  }

  /** Frame the whole court through the given yaw/pitch/distance. */
  _setOrbit(yaw, pitch, distance, target, up = [0, 0, 1]) {
    this._setUp(up);
    this.controls.target.set(target[0], target[1], target[2]);
    const cp = Math.cos(clamp(pitch, MIN_PITCH, MAX_PITCH));
    this.camera.position.set(
      target[0] + distance * cp * Math.cos(yaw),
      target[1] + distance * cp * Math.sin(yaw),
      target[2] + distance * Math.sin(clamp(pitch, MIN_PITCH, MAX_PITCH)),
    );
    this.controls.update();
    this.dirty = true;
  }

  /**
   * Place the eye at a scene camera and aim it down that camera's optical axis.
   *
   * ``rotation`` is the world-to-camera matrix, so its row 2 is the camera's
   * viewing direction in world coordinates and row 1 the camera's down axis;
   * the camera up is therefore ``-rotation[1]``. When only the frustum is
   * available (legacy BLCS-style payloads) the optical axis is approximated by
   * the four image-plane corners' centroid, never by a single corner.
   */
  lookThroughCamera(id) {
    const camera = this._cameras.find((entry) => entry.id === id);
    if (!camera) return;
    const view = cameraAxis(camera);
    if (!view) return;
    const center = camera.center || camera.frustum[0];
    const direction = new THREE.Vector3(view.direction[0], view.direction[1], view.direction[2]);
    const distance = camera.frustum && camera.frustum.length >= 5
      ? Math.max(
          new THREE.Vector3(
            center[0] - camera.frustum[1][0],
            center[1] - camera.frustum[1][1],
            center[2] - camera.frustum[1][2],
          ).length(),
          6,
        )
      : 18;
    this._setUp(view.up);
    this.camera.position.set(center[0], center[1], center[2]);
    this.controls.target
      .set(center[0], center[1], center[2])
      .addScaledVector(direction, distance);
    if (this.follow) this.setFollow(false);
    this.selectedCamera = id;
    this._refreshCameraColors();
    this.controls.update();
    this.dirty = true;
  }

  resetView(preset = DEFAULT_PRESET) {
    this.applyPreset(preset);
  }

  // ------------------------------------------------------------------ render

  resize() {
    const width = this.canvas.clientWidth || 1;
    const height = this.canvas.clientHeight || 1;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  render() {
    if (this.follow) this._snapFollow();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.dirty = false;
  }

  draw() {
    this.render();
  }

  dispose() {
    this._resizeObserver?.disconnect();
    this.controls.dispose();
    this._disposeGroup(this.courtGroup);
    this._disposeGroup(this.entityGroup);
    this._disposeGroup(this.cameraGroup);
    this.renderer.dispose();
  }

  _bindPointer() {
    const canvas = this.canvas;
    canvas.addEventListener("pointerdown", (event) => {
      this._pointer = { x: event.clientX, y: event.clientY, moved: 0, button: event.button };
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!this._pointer) return;
      this._pointer.moved +=
        Math.abs(event.clientX - this._pointer.x) + Math.abs(event.clientY - this._pointer.y);
      this._pointer.x = event.clientX;
      this._pointer.y = event.clientY;
    });
    const release = (event) => {
      const pointer = this._pointer;
      this._pointer = null;
      if (!pointer || pointer.moved > 4 || pointer.button !== 0) return;
      const id = this._pick(event.clientX, event.clientY);
      this.selectedCamera = id;
      this._refreshCameraColors();
      this.onCameraPick?.(id);
      this.dirty = true;
    };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", () => {
      this._pointer = null;
    });
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("dblclick", () => this.resetView());
  }
}

export { clamp, presenceAt, rootAt };
