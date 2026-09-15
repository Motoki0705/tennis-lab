// Browser wiring for the PLCS scene-inference UI: checkpoint and scene
// selection on the left, a 3D court view in the middle, inference settings on
// the right. All geometry comes from the framed /api/predict (or preview)
// response and is drawn by the shared Three.js engine at /shared/scene3d.mjs;
// nothing is fabricated client-side.

import { Scene3D } from "/shared/scene3d.mjs";

import {
  CourtScene,
  fitDistance,
  trackBounds,
} from "./court_scene.mjs";

const PLAY_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z" fill="currentColor" stroke="none" /></svg>';
const PAUSE_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="14" y="3" width="5" height="18" rx="1" fill="currentColor" stroke="none" /><rect x="5" y="3" width="5" height="18" rx="1" fill="currentColor" stroke="none" /></svg>';

const SPLITS = ["train", "val", "test"];
const MAX_SCENE_LIMIT = 1000;

const byId = (id) => document.getElementById(id);

const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
if (!littleEndian) {
  throw new Error("This viewer requires a little-endian platform for float32 payloads.");
}

const dom = {
  checkpointSearch: byId("checkpoint-search"),
  checkpointList: byId("checkpoint-list"),
  pathInput: byId("path-input"),
  pathSubmit: byId("path-submit"),
  familyList: byId("family-list"),
  familyUnsupported: byId("family-unsupported"),
  familyUnsupportedList: byId("family-unsupported-list"),
  familyUnsupportedCount: byId("family-unsupported-count"),
  splitTabs: byId("split-tabs"),
  sceneSearch: byId("scene-search"),
  resultCount: byId("result-count"),
  sceneList: byId("scene-list"),
  sceneTitle: byId("scene-title"),
  sceneEyebrow: byId("scene-eyebrow"),
  toggleGt: byId("toggle-gt"),
  togglePred: byId("toggle-pred"),
  toggleTrails: byId("toggle-trails"),
  resetView: byId("reset-view"),
  court: byId("court"),
  legend: byId("legend"),
  hudFrame: byId("hud-frame"),
  hudTime: byId("hud-time"),
  hudGt: byId("hud-gt"),
  hudPred: byId("hud-pred"),
  hudModel: byId("hud-model"),
  status: byId("status"),
  transport: byId("transport"),
  toStart: byId("to-start"),
  prev: byId("prev"),
  play: byId("play"),
  next: byId("next"),
  toEnd: byId("to-end"),
  scrub: byId("scrub"),
  clock: byId("clock"),
  device: byId("device"),
  cameras: byId("cameras"),
  cameraCount: byId("camera-count"),
  referenceCamera: byId("reference-camera"),
  windowStart: byId("window-start"),
  windowLength: byId("window-length"),
  windowHint: byId("window-hint"),
  run: byId("run"),
  runHint: byId("run-hint"),
  metrics: byId("metrics"),
  warnings: byId("warnings"),
  speeds: Array.from(document.querySelectorAll(".speeds button")),
  poseSources: Array.from(document.querySelectorAll("#pose-source .pose-source")),
};

const state = {
  catalog: null,
  checkpointQuery: "",
  checkpoint: null,
  family: null,
  split: "val",
  sceneQuery: "",
  scenes: [],
  sceneTotal: 0,
  sceneReturned: 0,
  scene: null,
  detail: null,
  cameras: new Set(),
  reference: null,
  windowStart: 0,
  windowLength: 1,
  poseSource: "gt",
  fps: null,
  frame: 0,
  frameCount: 0,
  phase: 0,
  playing: false,
  speed: 1,
  lastTime: 0,
  token: 0,
  requestToken: 0,
  running: false,
  lastRequest: null,
  legendTracks: null,
  previewCameras: [],
  needsFraming: true,
};

const scene = new CourtScene();
const scene3d = new Scene3D(dom.court, { fov: 42 });

const GT_SHADES = ["#1f8a70", "#2f9d7f", "#4bb094", "#7cc4b0"];
const PRED_SHADES = ["#d1623f", "#dd7a55", "#e89376", "#f0ab97"];

/** One shared-engine entity for a single PLCS track (root + optional joints). */
function trackEntity(track, shades) {
  const rootOnly = !track.joints;
  const positions = rootOnly ? track.position : track.joints;
  const joints = rootOnly ? 1 : track.jointCount || 17;
  const root = track.position && track.position.length ? track.position : positions;
  const presence = track.presence
    ? Uint8Array.from(track.presence, (value) => (value ? 1 : 0))
    : null;
  const heading = [];
  for (let frame = 0; frame < track.frameCount; frame += 1) {
    if (!track.rotation) break;
    heading.push([track.rotation[frame * 2], track.rotation[frame * 2 + 1]]);
  }
  return {
    id: `${track.kind}-${track.objectIndex}`,
    color: shades[track.objectIndex % shades.length],
    kind: rootOnly ? "player-root" : "player",
    frames: track.frameCount,
    joints,
    positions,
    roots: root,
    presence,
    edges: rootOnly ? null : scene.skeleton.edges,
    heading: heading.length ? heading : null,
  };
}

/** Rebuild the shared model from the installed payload and retained cameras. */
function rebuildScene() {
  const entities = [];
  for (const track of scene.tracks) {
    if (!scene.visible[track.kind]) continue;
    entities.push(
      trackEntity(track, track.kind === "pred" ? PRED_SHADES : GT_SHADES),
    );
  }
  scene3d.setModel({
    court: scene.court,
    frames: scene.frameCount,
    entities,
    cameras: state.previewCameras,
  });
  scene3d.setTrailsVisible(scene.showTrails);
  scene3d.setFrame(scene.frame);
}

/** Frame the court and players with the tested point-fitter. */
function frameScene() {
  if (!scene.court.keypoints.length) return;
  const bounds = trackBounds(scene.tracks);
  const court = trackBounds([{ position: flattenPoints(scene.court.keypoints), joints: null }]);
  const center = court.center;
  const radius = Math.max(bounds.radius, court.radius);
  const halfFov = (scene3d.fov * Math.PI) / 360;
  scene3d.setOrbit({
    yaw: -Math.PI / 2 - 0.35,
    pitch: 0.6,
    distance: fitDistance(radius, halfFov) * 0.72,
    target: [center[0], center[1], 0.5],
  });
}

function flattenPoints(points) {
  const flat = new Float32Array(points.length * 3);
  points.forEach((point, index) => {
    flat[index * 3] = point[0];
    flat[index * 3 + 1] = point[1];
    flat[index * 3 + 2] = point[2] ?? 0;
  });
  return flat;
}

/** Bring the stacked 3D view back into sight after a scene is chosen. */
function revealViewport() {
  if (!window.matchMedia("(max-width: 720px)").matches) return;
  document.querySelector(".viewport")?.scrollIntoView({ block: "start" });
}

// ------------------------------------------------------------------ helpers

function encode(value) {
  return encodeURIComponent(value);
}

function span(className, text) {
  const element = document.createElement("span");
  element.className = className;
  element.textContent = text;
  return element;
}

function emptyNote(text) {
  const note = document.createElement("p");
  note.className = "empty";
  note.textContent = text;
  return note;
}

function clampInt(value, low, high) {
  const rounded = Number.isFinite(value) ? Math.round(value) : low;
  return Math.min(high, Math.max(low, rounded));
}

function matches(haystack, query) {
  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
  const text = haystack.toLowerCase();
  return tokens.every((token) => text.includes(token));
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "–";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 || value >= 10 ? 0 : 1)} ${units[unit]}`;
}

function formatDate(ns) {
  if (!Number.isFinite(ns) || ns <= 0) return "–";
  const date = new Date(ns / 1e6);
  if (Number.isNaN(date.getTime())) return "–";
  return date.toISOString().slice(0, 10);
}

function formatNumber(value, digits) {
  return Number.isFinite(value) ? value.toFixed(digits) : "–";
}

async function describeError(response) {
  try {
    const body = await response.json();
    if (body && typeof body.detail === "string") {
      return `${response.status} ${body.detail}`;
    }
  } catch (error) {
    // Body was not JSON; fall back to the status line.
  }
  return `${response.status} ${response.statusText}`;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(await describeError(response));
  }
  return response.json();
}

function setStatus(message, tone) {
  if (!message) {
    dom.status.hidden = true;
    dom.status.textContent = "";
    return;
  }
  dom.status.hidden = false;
  dom.status.dataset.tone = tone || "info";
  dom.status.textContent = message;
}

// --------------------------------------------------------- checkpoints view

function renderCheckpointList() {
  dom.checkpointList.textContent = "";
  const checkpoints = state.catalog ? state.catalog.checkpoints || [] : [];
  const visible = checkpoints.filter((checkpoint) =>
    matches(
      `${checkpoint.model_name || ""} ${checkpoint.label || ""} ${checkpoint.path || ""} ${checkpoint.id}`,
      state.checkpointQuery,
    ),
  );
  for (const checkpoint of visible) {
    dom.checkpointList.append(checkpointRow(checkpoint));
  }
  if (!visible.length) {
    dom.checkpointList.append(
      emptyNote(
        state.checkpointQuery
          ? "一致するチェックポイントがありません"
          : "チェックポイントが見つかりません",
      ),
    );
  }
}

function checkpointRow(checkpoint) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "checkpoint";
  button.dataset.checkpointId = checkpoint.id;
  if (state.checkpoint && state.checkpoint.id === checkpoint.id) {
    button.setAttribute("aria-current", "true");
  }
  button.append(span("name", checkpoint.label || checkpoint.model_name || checkpoint.id));

  const tags = document.createElement("span");
  tags.className = "meta";
  const badge = span(
    checkpoint.supported === false ? "badge muted" : "badge",
    checkpoint.selector || "?",
  );
  tags.append(badge, document.createTextNode(` ${checkpoint.model_name || ""}`));
  button.append(tags);

  const info = [];
  if (checkpoint.trained_scene_dir) info.push(checkpoint.trained_scene_dir);
  if (checkpoint.size_bytes) info.push(formatBytes(checkpoint.size_bytes));
  if (checkpoint.modified_ns) info.push(formatDate(checkpoint.modified_ns));
  if (info.length) {
    button.append(span("meta", info.join(" · ")));
  }
  if (checkpoint.supported === false) {
    button.disabled = true;
    button.title = checkpoint.unsupported_reason || "このチェックポイントは非対応です";
    button.append(span("meta", checkpoint.unsupported_reason || "非対応"));
  }
  return button;
}

function selectCheckpoint(checkpoint) {
  state.checkpoint = checkpoint;
  const offered = offeredFamilies();
  if (state.family && !offered.some((family) => family.id === state.family)) {
    state.family = null;
    state.scene = null;
    state.detail = null;
  }
  renderCheckpointList();
  renderFamilyList();
  renderSplitTabs();
  renderSceneList();
  clearSceneView();
  if (state.family) {
    void loadScenes().then(() => {
      if (state.scene) void loadSceneDetail();
    });
  }
  updateRunState();
}

// ------------------------------------------------------------- families view

function offeredFamilies() {
  const catalog = state.catalog;
  if (!catalog) return [];
  if (!state.checkpoint) return catalog.families.slice();
  const byId = new Map(catalog.families.map((family) => [family.id, family]));
  const offered = [];
  for (const id of state.checkpoint.families || []) {
    const family = byId.get(id);
    if (family) offered.push(family);
  }
  return offered;
}

function unsupportedFamilies() {
  if (!state.checkpoint) return [];
  const allowed = new Set(offeredFamilies().map((family) => family.id));
  return (state.catalog ? state.catalog.families : []).filter(
    (family) => !allowed.has(family.id),
  );
}

function renderFamilyList() {
  dom.familyList.textContent = "";
  const offered = offeredFamilies();
  for (const family of offered) {
    dom.familyList.append(familyRow(family, false));
  }
  if (!offered.length) {
    dom.familyList.append(
      emptyNote(state.catalog ? "選択できるファミリがありません" : "読み込み中…"),
    );
  }

  const unsupported = unsupportedFamilies();
  dom.familyUnsupportedList.textContent = "";
  if (!state.checkpoint || !unsupported.length) {
    dom.familyUnsupported.hidden = true;
  } else {
    dom.familyUnsupported.hidden = false;
    dom.familyUnsupportedCount.textContent = String(unsupported.length);
    for (const family of unsupported) {
      dom.familyUnsupportedList.append(familyRow(family, true));
    }
  }
}

function familyRow(family, unsupported) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "family";
  button.dataset.family = family.id;
  if (unsupported) {
    button.disabled = true;
    button.setAttribute("aria-disabled", "true");
    button.title = "このチェックポイントでは選択できません";
  } else if (state.family === family.id) {
    button.setAttribute("aria-current", "true");
  }
  if (!unsupported && family.available === false) {
    button.disabled = true;
    button.title = "利用できません";
  }
  const head = document.createElement("span");
  head.className = "head";
  head.append(span("name", family.id));
  const badge = span("badge" + (unsupported ? " muted" : ""), family.selector || "?");
  head.append(badge);
  button.append(head);
  const meta = [];
  if (family.scene_count != null) meta.push(`${family.scene_count} scenes`);
  if (family.num_cameras != null) meta.push(`${family.num_cameras} cam`);
  if (family.objects) meta.push(family.objects);
  if (meta.length) {
    button.append(span("meta", meta.join(" · ")));
  }
  return button;
}

function selectFamily(id) {
  if (state.family === id) return;
  state.family = id;
  state.scene = null;
  state.sceneQuery = "";
  dom.sceneSearch.value = "";
  clearSceneView();
  renderFamilyList();
  renderSplitTabs();
  renderSceneList();
  updateRunState();
  void loadScenes();
}

// ----------------------------------------------------------------- splits

function renderSplitTabs() {
  dom.splitTabs.textContent = "";
  for (const split of SPLITS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "split";
    button.dataset.split = split;
    button.textContent = split;
    button.setAttribute("aria-pressed", String(state.split === split));
    button.disabled = !state.family;
    dom.splitTabs.append(button);
  }
}

// ------------------------------------------------------------------ scenes

async function loadScenes() {
  if (!state.family) {
    state.scenes = [];
    renderSceneList();
    return;
  }
  const token = (state.token += 1);
  try {
    const params = new URLSearchParams({
      family: state.family,
      split: state.split,
      limit: String(MAX_SCENE_LIMIT),
    });
    if (state.sceneQuery) params.set("query", state.sceneQuery);
    const data = await fetchJson(`/api/scenes?${params.toString()}`);
    if (token !== state.token) return;
    state.scenes = data.scenes || [];
    state.sceneTotal = Number.isFinite(data.total) ? data.total : state.scenes.length;
    state.sceneReturned = Number.isFinite(data.returned)
      ? data.returned
      : state.scenes.length;
    renderSceneList();
  } catch (error) {
    if (token !== state.token) return;
    state.scenes = [];
    renderSceneList();
    setStatus(`シーン一覧の読み込みに失敗しました: ${error.message}`, "error");
  }
}

function renderSceneList() {
  dom.sceneList.textContent = "";
  if (!state.family) {
    dom.resultCount.textContent = "";
    return;
  }
  for (const sceneItem of state.scenes) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "scene";
    button.dataset.scene = sceneItem.id;
    if (state.scene === sceneItem.id) {
      button.setAttribute("aria-current", "true");
    }
    button.append(span("name", sceneItem.id));
    const meta = [];
    if (sceneItem.num_frames != null) meta.push(`${sceneItem.num_frames}f`);
    if (sceneItem.fps != null) meta.push(`${Math.round(sceneItem.fps)}fps`);
    if (sceneItem.num_cameras_sampled != null) {
      meta.push(`${sceneItem.num_cameras_sampled}cam`);
    }
    if (sceneItem.motion_category) meta.push(sceneItem.motion_category);
    if (meta.length) button.append(span("meta", meta.join(" · ")));
    dom.sceneList.append(button);
  }
  if (!state.scenes.length) {
    dom.sceneList.append(
      emptyNote(state.sceneQuery ? "一致するシーンがありません" : "シーンがありません"),
    );
  }
  dom.resultCount.textContent = `表示 ${state.sceneReturned} / 全 ${state.sceneTotal} 件`;
}

// ------------------------------------------------------- scene detail view

function clearSceneView() {
  state.requestToken += 1;
  state.detail = null;
  state.cameras = new Set();
  state.reference = null;
  state.frameCount = 0;
  state.frame = 0;
  state.phase = 0;
  state.playing = false;
  state.fps = null;
  state.windowStart = 0;
  state.windowLength = 1;
  scene.clear();
  scene.setKindVisible("gt", true);
  scene.setKindVisible("pred", true);
  state.previewCameras = [];
  state.needsFraming = true;
  rebuildScene();
  dom.toggleGt.setAttribute("aria-pressed", "true");
  dom.togglePred.setAttribute("aria-pressed", "true");
  dom.cameras.textContent = "";
  dom.cameraCount.textContent = "0";
  dom.referenceCamera.textContent = "";
  dom.referenceCamera.disabled = true;
  dom.windowHint.textContent = "–";
  dom.metrics.textContent = "";
  dom.warnings.textContent = "";
  dom.sceneTitle.textContent = "シーンを選択してください";
  dom.sceneEyebrow.textContent = "シーン未選択";
  dom.scrub.max = "0";
  dom.scrub.value = "0";
  dom.clock.textContent = "0 / 0";
  dom.hudFrame.textContent = "–";
  dom.hudTime.textContent = "–";
  dom.hudGt.textContent = "–";
  dom.hudPred.textContent = "–";
  dom.hudModel.textContent = "–";
  state.legendTracks = null;
  syncLegend();
  syncPlayButton();
}

async function selectScene(sceneId) {
  if (state.scene === sceneId && state.detail) return;
  state.scene = sceneId;
  renderSceneList();
  await loadSceneDetail();
}

async function loadSceneDetail() {
  if (!state.family || !state.scene) return;
  const token = (state.token += 1);
  state.requestToken += 1;
  state.detail = null;
  updateRunState();
  setStatus("シーン情報を読み込み中…", "info");
  try {
    const params = new URLSearchParams();
    if (state.checkpoint) params.set("checkpoint", state.checkpoint.id);
    const query = params.toString();
    const detail = await fetchJson(
      `/api/scenes/${encode(state.family)}/${encode(state.scene)}${query ? `?${query}` : ""}`,
    );
    if (token !== state.token) return;
    applySceneDetail(detail);
    setStatus("");
    void runPreview();
  } catch (error) {
    if (token !== state.token) return;
    setStatus(`シーン情報の読み込みに失敗しました: ${error.message}`, "error");
  }
}

function applySceneDetail(detail) {
  state.detail = detail;
  state.fps = Number.isFinite(detail.fps) ? detail.fps : null;
  if (detail.checkpoint && state.checkpoint && detail.checkpoint.id === state.checkpoint.id) {
    state.checkpoint = { ...state.checkpoint, ...detail.checkpoint };
    renderCheckpointList();
    renderFamilyList();
  }
  dom.sceneTitle.textContent = detail.id;
  dom.sceneEyebrow.textContent = `${detail.family} · ${detail.selector}`;

  const cameras = detail.cameras || [];
  const profile = state.checkpoint ? state.checkpoint.input_profile : null;
  const maxViews = state.checkpoint ? state.checkpoint.max_views : null;
  let wanted = 1;
  if (profile === "multiview") {
    wanted = maxViews ? Math.min(cameras.length, maxViews) : cameras.length;
    wanted = Math.max(wanted, Math.min(3, cameras.length));
  }
  state.cameras = new Set(
    cameras.slice(0, wanted).map((camera) => Number(camera.index)),
  );

  renderCameras(cameras);

  if (detail.reference_required) {
    state.reference = cameras.length ? cameras[0].id : null;
    dom.referenceCamera.disabled = false;
    dom.referenceCamera.value = state.reference || "";
  } else {
    state.reference = null;
    dom.referenceCamera.disabled = true;
  }

  const windowInfo = detail.window || {};
  const maxLength =
    Number.isFinite(windowInfo.max_length) && windowInfo.max_length > 0
      ? Math.min(windowInfo.max_length, detail.num_frames)
      : detail.num_frames;
  state.windowStart = 0;
  state.windowLength = clampInt(
    windowInfo.default_length ?? maxLength,
    1,
    maxLength,
  );
  applyWindowInputs();
  updateRunState();
}

function renderCameras(cameras) {
  dom.cameras.textContent = "";
  for (const camera of cameras) {
    const label = document.createElement("label");
    label.className = "camera-row";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.className = "camera";
    input.value = String(camera.index);
    input.checked = state.cameras.has(Number(camera.index));
    input.dataset.cameraId = camera.id || `camera_${camera.index}`;
    label.append(input, span("camera-name", camera.id || `camera_${camera.index}`));
    dom.cameras.append(label);
  }
  dom.cameraCount.textContent = String(state.cameras.size);

  dom.referenceCamera.textContent = "";
  for (const camera of cameras) {
    const option = document.createElement("option");
    option.value = camera.id || `camera_${camera.index}`;
    option.textContent = camera.id || `camera_${camera.index}`;
    dom.referenceCamera.append(option);
  }
  if (state.reference) {
    dom.referenceCamera.value = state.reference;
  }
}

function toggleCamera(index, checked) {
  const profile = state.checkpoint ? state.checkpoint.input_profile : null;
  if (profile === "multiview") {
    if (checked) state.cameras.add(index);
    else state.cameras.delete(index);
  } else {
    state.cameras = checked ? new Set([index]) : new Set();
  }
  for (const input of dom.cameras.querySelectorAll(".camera")) {
    input.checked = state.cameras.has(Number(input.value));
  }
  dom.cameraCount.textContent = String(state.cameras.size);
  updateRunState();
}

function selectedCameras() {
  return Array.from(state.cameras).sort((left, right) => left - right);
}

function requestCameras() {
  const selected = selectedCameras();
  const profile = state.checkpoint ? state.checkpoint.input_profile : null;
  return profile === "multiview" ? selected : selected.slice(0, 1);
}

// ------------------------------------------------------------------ window

function windowCeiling() {
  const detail = state.detail;
  if (!detail) return 1;
  const maxLength = detail.window ? detail.window.max_length : null;
  if (Number.isFinite(maxLength) && maxLength > 0) {
    return Math.max(1, Math.min(maxLength, detail.num_frames));
  }
  return Math.max(1, detail.num_frames);
}

function applyWindowInputs() {
  const detail = state.detail;
  if (!detail) {
    dom.windowHint.textContent = "–";
    return;
  }
  const numFrames = detail.num_frames;
  state.windowStart = clampInt(state.windowStart, 0, Math.max(0, numFrames - 1));
  const ceiling = Math.min(windowCeiling(), numFrames - state.windowStart);
  state.windowLength = clampInt(state.windowLength, 1, Math.max(1, ceiling));
  dom.windowStart.value = String(state.windowStart);
  dom.windowStart.max = String(Math.max(0, numFrames - 1));
  dom.windowLength.value = String(state.windowLength);
  dom.windowLength.max = String(Math.max(1, ceiling));
  dom.windowHint.textContent =
    `frame ${state.windowStart} .. ${state.windowStart + state.windowLength - 1}` +
    ` / 全 ${numFrames} frame · 最大 ${windowCeiling()}`;
}

// ------------------------------------------------------------------ validate

function settingsIssue() {
  if (!state.catalog) return "カタログを読み込み中です";
  if (!state.checkpoint) return "チェックポイントを選択してください";
  if (state.checkpoint.supported === false) {
    return state.checkpoint.unsupported_reason || "このチェックポイントは非対応です";
  }
  if (!state.detail) return "シーンを選択してください";
  if (state.detail.supported === false) {
    return state.detail.unsupported_reason || "このシーンでは推論できません";
  }
  if (!state.family || !state.scene) return "シーンを選択してください";
  const cameras = selectedCameras();
  if (!cameras.length) return "カメラを1つ以上選択してください";
  const profile = state.checkpoint.input_profile;
  const maxViews = state.checkpoint.max_views;
  // Only the model's camera capacity is a hard constraint here. The training
  // view-count range is advisory: the backend decides, and a mismatch surfaces
  // as its own error rather than a silent client-side block.
  if (maxViews && cameras.length > maxViews) {
    return `このモデルは最大 ${maxViews} カメラまでです`;
  }
  if (state.detail.reference_required) {
    if (!state.reference) return "参照カメラを選択してください";
    if (!cameras.includes(selectedIndexById(state.reference))) {
      return "参照カメラは選択中のカメラから選んでください";
    }
  }
  if (state.windowLength < 1) return "ウィンドウ長を1以上にしてください";
  if (state.windowLength > windowCeiling()) {
    return `ウィンドウ長は最大 ${windowCeiling()} frame です`;
  }
  if (state.windowStart + state.windowLength > state.detail.num_frames) {
    return "ウィンドウがシーンの範囲を超えています";
  }
  return null;
}

function selectedIndexById(cameraId) {
  const detail = state.detail;
  if (!detail) return -1;
  const found = (detail.cameras || []).find((camera) => camera.id === cameraId);
  return found ? Number(found.index) : -1;
}

function updateRunState() {
  const issue = settingsIssue();
  dom.run.disabled = Boolean(issue) || state.running;
  dom.run.textContent = state.running ? "推論を実行中…" : "推論を実行";
  dom.runHint.textContent = issue || cameraCountNote();
  dom.runHint.dataset.tone = issue ? "error" : "info";
}

/** Advisory note when the camera count is outside the checkpoint's training range. */
function cameraCountNote() {
  const range = state.checkpoint?.num_views_range;
  if (!Array.isArray(range) || range.length !== 2) return "";
  const count = selectedCameras().length;
  if (count >= range[0] && count <= range[1]) return "";
  return `学習時のカメラ数は ${range[0]}〜${range[1]} です（現在 ${count}）`;
}

// ------------------------------------------------------------------- predict

function decodePrediction(buffer) {
  const view = new DataView(buffer);
  const headerLength = view.getUint32(0, true);
  const headerBytes = new Uint8Array(buffer, 4, headerLength);
  const header = JSON.parse(new TextDecoder("utf-8").decode(headerBytes));
  const start = 4 + headerLength;
  const payloadLength = buffer.byteLength - start;
  if (payloadLength < 0 || payloadLength % 4 !== 0) {
    throw new Error(
      `推論レスポンスの float32 payload が不正です (${payloadLength} bytes)`,
    );
  }
  const data = new Float32Array(buffer.slice(start));
  return { header, data };
}

/**
 * GPU-free ground-truth preview so a scene shows up before any checkpoint is
 * chosen. Needs no checkpoint; only the family/scene and camera window.
 */
function previewIssue() {
  if (!state.catalog) return "カタログを読み込み中です";
  if (!state.family || !state.scene) return "シーンを選択してください";
  if (!state.detail) return "シーンを選択してください";
  if (!requestCameras().length) return "カメラを1つ以上選択してください";
  return null;
}

async function runPreview() {
  if (previewIssue()) return;
  const params = new URLSearchParams();
  for (const camera of requestCameras()) params.append("cameras", String(camera));
  params.set("window_start", String(state.windowStart));
  params.set("window_length", String(state.windowLength));
  const token = (state.requestToken += 1);
  try {
    const response = await fetch(
      `/api/scenes/${encode(state.family)}/${encode(state.scene)}/preview?${params}`,
    );
    if (!response.ok) throw new Error(await describeError(response));
    const buffer = await response.arrayBuffer();
    if (token !== state.requestToken) return;
    installPrediction(decodePrediction(buffer), { preview: true });
  } catch (error) {
    if (token !== state.requestToken) return;
    setStatus(`プレビューに失敗しました: ${error.message}`, "error");
  }
}

async function runPrediction() {
  if (state.running) return;
  const issue = settingsIssue();
  if (issue) {
    updateRunState();
    setStatus(issue, "error");
    return;
  }
  const body = {
    checkpoint: state.checkpoint.id,
    family: state.family,
    scene: state.scene,
    cameras: requestCameras(),
    reference_camera_id: state.detail.reference_required ? state.reference : null,
    window_start: state.windowStart,
    window_length: state.windowLength,
    canonical_pose_source: state.poseSource,
    device: state.device,
  };
  const key = JSON.stringify(body);
  if (state.lastRequest && state.lastRequest.key === key) {
    installPrediction(state.lastRequest.parsed, { preview: false });
    return;
  }

  const token = (state.requestToken += 1);
  state.running = true;
  updateRunState();
  setStatus("GPUキュー・推論実行中", "busy");
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      throw new Error(await describeError(response));
    }
    const buffer = await response.arrayBuffer();
    if (token !== state.requestToken) return;
    const parsed = decodePrediction(buffer);
    state.lastRequest = { key, parsed };
    installPrediction(parsed, { preview: false });
    setStatus("");
  } catch (error) {
    if (token !== state.requestToken) return;
    setStatus(`推論に失敗しました: ${error.message}`, "error");
  } finally {
    // Preview tokens determine which result is displayed, not job ownership.
    state.running = false;
    updateRunState();
  }
}

function installPrediction(parsed, { preview }) {
  const header = parsed.header || {};
  const tracks = header.tracks || [];
  scene.setData({ header, data: parsed.data });
  // ``header.cameras`` arrives only with the preview payload, so a later
  // predict response must keep the frustums the preview already installed.
  if (Array.isArray(header.cameras) && header.cameras.length) {
    state.previewCameras = header.cameras.map((camera) => ({
      id: camera.id || `camera_${camera.index}`,
      label: camera.id || `camera_${camera.index}`,
      center: camera.center,
      frustum: camera.frustum,
      rotation: camera.rotation,
    }));
  }
  state.frameCount = scene.frameCount;
  state.frame = 0;
  state.phase = 0;
  state.playing = false;
  const request = header.request || {};
  if (Number.isFinite(request.window_start)) state.windowStart = request.window_start;
  if (Number.isFinite(request.window_length)) state.windowLength = request.window_length;
  applyWindowInputs();

  const predVisible = !preview;
  scene.setKindVisible("pred", predVisible);
  dom.togglePred.setAttribute("aria-pressed", String(predVisible));
  dom.toggleGt.setAttribute("aria-pressed", "true");
  scene.setKindVisible("gt", true);

  const checkpoint = header.checkpoint || {};
  const cameraCount = Array.isArray(request.cameras) ? request.cameras.length : 0;
  const modelName = checkpoint.model_name || state.checkpoint?.model_name || "GT";
  dom.hudModel.textContent = `${modelName} · ${cameraCount} cam`;

  dom.scrub.max = String(Math.max(0, state.frameCount - 1));
  dom.transport.hidden = false;

  state.legendTracks = tracks.map((track) => ({
    kind: track.kind,
    label: track.label || track.kind,
    hasJoints: track.has_joints !== false,
  }));
  syncLegend();
  syncPlayButton();
  rebuildScene();
  if (state.needsFraming) {
    frameScene();
    state.needsFraming = false;
    revealViewport();
  }
  applyFrame(0);

  if (preview) {
    dom.metrics.textContent = "";
    dom.warnings.textContent = "";
  } else {
    renderMetrics(header.metrics || null);
    renderWarnings(header.warnings || []);
  }
}

// ------------------------------------------------------------------- legend

function syncLegend() {
  const list =
    state.legendTracks ||
    [
      { kind: "gt", label: "GT", hasJoints: true },
      { kind: "pred", label: "推論", hasJoints: true },
    ];
  // One entry per kind; multi-object payloads carry many tracks per kind.
  const groups = new Map();
  for (const track of list) {
    const group = groups.get(track.kind) || {
      kind: track.kind,
      count: 0,
      hasJoints: false,
    };
    group.count += 1;
    group.hasJoints = group.hasJoints || track.hasJoints !== false;
    groups.set(track.kind, group);
  }
  dom.legend.textContent = "";
  for (const group of groups.values()) {
    const item = document.createElement("span");
    item.dataset.kind = group.kind;
    const swatch = document.createElement("i");
    swatch.className = `swatch ${group.kind === "pred" ? "pred" : "gt"}`;
    const base = group.kind === "pred" ? "推論" : "GT";
    const suffix = group.count > 1 ? ` ×${group.count}` : "";
    item.append(swatch, document.createTextNode(`${base}${suffix}`));
    if (!scene.visible[group.kind]) item.hidden = true;
    dom.legend.append(item);
  }
}

// ------------------------------------------------------------------ metrics

function metricRow(label, value) {
  const row = document.createElement("div");
  row.className = "metric";
  const left = document.createElement("span");
  left.className = "label";
  left.textContent = label;
  const right = document.createElement("span");
  right.className = "value";
  right.textContent = value;
  row.append(left, right);
  return row;
}

function distributionRows(label, distribution, unit, digits) {
  if (!distribution || typeof distribution !== "object") return [];
  const rows = [];
  const group = document.createElement("div");
  group.className = "group";
  group.textContent = label;
  rows.push(group);
  for (const key of ["mean", "median", "max"]) {
    const value = distribution[key];
    const text = Number.isFinite(value) ? `${formatNumber(value, digits)} ${unit}` : "–";
    rows.push(metricRow(key, text));
  }
  return rows;
}

function renderMetrics(metrics) {
  dom.metrics.textContent = "";
  if (!metrics) return;
  const rows = [];
  rows.push(...distributionRows("位置誤差", metrics.position_error_m, "m", 3));
  rows.push(...distributionRows("yaw 誤差", metrics.yaw_error_deg, "deg", 2));
  rows.push(...distributionRows("関節誤差", metrics.joints_error_m, "m", 3));
  const context = [];
  if (Number.isFinite(metrics.window_start) && Number.isFinite(metrics.window_length)) {
    context.push(`window ${metrics.window_start}..${metrics.window_start + metrics.window_length - 1}`);
  }
  if (Number.isFinite(metrics.scene_frames)) context.push(`scene ${metrics.scene_frames}f`);
  if (Number.isFinite(metrics.cameras)) context.push(`${metrics.cameras} cam`);
  if (context.length) rows.push(metricRow("条件", context.join(" · ")));
  if (!rows.length) rows.push(emptyNote("指標はありません"));
  for (const row of rows) dom.metrics.append(row);
}

function renderWarnings(warnings) {
  dom.warnings.textContent = "";
  if (!warnings || !warnings.length) {
    const note = document.createElement("p");
    note.className = "empty-note";
    note.textContent = "なし";
    dom.warnings.append(note);
    return;
  }
  for (const warning of warnings) {
    const item = document.createElement("div");
    item.className = "warning";
    item.textContent = String(warning);
    dom.warnings.append(item);
  }
}

// -------------------------------------------------------------------- frame

function rootOf(track, frame) {
  if (!track || !track.position || frame >= track.frameCount) return null;
  const offset = frame * 3;
  if (offset + 2 >= track.position.length) return null;
  return [track.position[offset], track.position[offset + 1], track.position[offset + 2]];
}

function rootText(track, frame) {
  const root = rootOf(track, frame);
  if (!root) return "–";
  return `${root[0].toFixed(2)}, ${root[1].toFixed(2)}, ${root[2].toFixed(2)} m`;
}

/** First track of ``kind`` whose ``presence`` mask is set at ``frame``. */
function presentTrack(kind, frame) {
  for (const track of scene.tracks) {
    if (track.kind !== kind) continue;
    if (track.presence && !track.presence[frame]) continue;
    return track;
  }
  return null;
}

function applyFrame(local) {
  if (!state.frameCount) {
    dom.clock.textContent = "0 / 0";
    return;
  }
  state.frame = clampInt(local, 0, state.frameCount - 1);
  scene.setFrame(state.frame);
  scene3d.setFrame(state.frame);
  dom.scrub.value = String(state.frame);
  dom.clock.textContent = `${state.frame + 1} / ${state.frameCount}`;
  const sceneFrame = state.windowStart + state.frame;
  dom.hudFrame.textContent = String(sceneFrame);
  dom.hudTime.textContent = state.fps ? `${(sceneFrame / state.fps).toFixed(3)}s` : "–";
  dom.hudGt.textContent = rootText(presentTrack("gt", state.frame), state.frame);
  dom.hudPred.textContent = rootText(presentTrack("pred", state.frame), state.frame);
}

function setPlaying(playing) {
  state.playing = Boolean(playing) && Boolean(state.frameCount);
  if (state.playing) state.phase = state.frame;
  syncPlayButton();
}

function syncPlayButton() {
  dom.play.innerHTML = state.playing ? PAUSE_ICON : PLAY_ICON;
  dom.play.title = state.playing ? "一時停止" : "再生";
}

function loop(now) {
  const delta = state.lastTime ? (now - state.lastTime) / 1000 : 0;
  state.lastTime = now;
  if (state.playing && state.frameCount && state.fps) {
    state.phase += delta * state.fps * state.speed;
    const total = state.frameCount;
    if (state.phase >= total) state.phase -= total;
    const target = Math.floor(state.phase);
    if (target !== state.frame) applyFrame(target);
  }
  scene3d.render();
  requestAnimationFrame(loop);
}

// ------------------------------------------------------------------ startup

async function init() {
  try {
    state.catalog = await fetchJson("/api/catalog");
  } catch (error) {
    setStatus(`カタログの読み込みに失敗しました: ${error.message}`, "error");
    return;
  }
  if (state.catalog.device) {
    state.device = state.catalog.device;
    dom.device.value = state.catalog.device;
  }
  renderCheckpointList();
  renderFamilyList();
  renderSplitTabs();
  renderSceneList();
  updateRunState();
}

// --------------------------------------------------------------- listeners

dom.checkpointSearch.addEventListener("input", () => {
  state.checkpointQuery = dom.checkpointSearch.value.trim();
  renderCheckpointList();
});

dom.checkpointList.addEventListener("click", (event) => {
  const button = event.target.closest(".checkpoint");
  if (!button || button.disabled) return;
  const checkpoint = (state.catalog.checkpoints || []).find(
    (item) => item.id === button.dataset.checkpointId,
  );
  if (checkpoint) selectCheckpoint(checkpoint);
});

let sceneSearchTimer = null;
dom.sceneSearch.addEventListener("input", () => {
  state.sceneQuery = dom.sceneSearch.value.trim();
  if (sceneSearchTimer) clearTimeout(sceneSearchTimer);
  sceneSearchTimer = setTimeout(() => {
    void loadScenes();
  }, 250);
});

dom.splitTabs.addEventListener("click", (event) => {
  const button = event.target.closest(".split");
  if (!button || button.disabled) return;
  state.split = button.dataset.split;
  renderSplitTabs();
  void loadScenes();
});

dom.familyList.addEventListener("click", (event) => {
  const button = event.target.closest(".family");
  if (!button || button.disabled) return;
  selectFamily(button.dataset.family);
});

dom.sceneList.addEventListener("click", (event) => {
  const button = event.target.closest(".scene");
  if (!button || button.disabled) return;
  void selectScene(button.dataset.scene);
});

dom.cameras.addEventListener("change", (event) => {
  const input = event.target.closest(".camera");
  if (!input) return;
  toggleCamera(Number(input.value), input.checked);
});

dom.referenceCamera.addEventListener("change", () => {
  state.reference = dom.referenceCamera.value || null;
  updateRunState();
});

dom.windowStart.addEventListener("change", () => {
  state.windowStart = Number(dom.windowStart.value);
  applyWindowInputs();
  updateRunState();
});

dom.windowLength.addEventListener("change", () => {
  state.windowLength = Number(dom.windowLength.value);
  applyWindowInputs();
  updateRunState();
});

for (const radio of dom.poseSources) {
  radio.addEventListener("change", () => {
    if (radio.checked) {
      state.poseSource = radio.value;
    }
  });
}

dom.device.addEventListener("change", () => {
  state.device = dom.device.value;
});

dom.run.addEventListener("click", () => {
  void runPrediction();
});

dom.pathSubmit.addEventListener("click", () => {
  void submitPath();
});
dom.pathInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    void submitPath();
  }
});

dom.toggleGt.addEventListener("click", () => {
  const visible = scene.toggleKind("gt");
  dom.toggleGt.setAttribute("aria-pressed", String(visible));
  rebuildScene();
  syncLegend();
});
dom.togglePred.addEventListener("click", () => {
  const visible = scene.toggleKind("pred");
  dom.togglePred.setAttribute("aria-pressed", String(visible));
  rebuildScene();
  syncLegend();
});
dom.toggleTrails.addEventListener("click", () => {
  scene.setTrailsVisible(!scene.showTrails);
  scene3d.setTrailsVisible(scene.showTrails);
  dom.toggleTrails.setAttribute("aria-pressed", String(scene.showTrails));
});
dom.resetView.addEventListener("click", () => {
  frameScene();
  scene3d.render();
});

dom.play.addEventListener("click", () => setPlaying(!state.playing));
dom.prev.addEventListener("click", () => {
  setPlaying(false);
  applyFrame(state.frame - 1);
});
dom.next.addEventListener("click", () => {
  setPlaying(false);
  applyFrame(state.frame + 1);
});
dom.toStart.addEventListener("click", () => {
  setPlaying(false);
  applyFrame(0);
});
dom.toEnd.addEventListener("click", () => {
  setPlaying(false);
  applyFrame(state.frameCount - 1);
});
dom.scrub.addEventListener("input", () => {
  setPlaying(false);
  applyFrame(Number(dom.scrub.value));
});
for (const button of dom.speeds) {
  button.addEventListener("click", () => {
    state.speed = Number(button.dataset.speed);
    for (const other of dom.speeds) {
      other.setAttribute("aria-pressed", String(other === button));
    }
  });
}

window.addEventListener("keydown", (event) => {
  const target = event.target;
  if (
    target instanceof HTMLInputElement ||
    target instanceof HTMLSelectElement ||
    target instanceof HTMLTextAreaElement
  ) {
    return;
  }
  if (event.key === " ") {
    if (target instanceof HTMLButtonElement) {
      return; // A focused button activates itself; do not double-toggle play.
    }
    event.preventDefault();
    setPlaying(!state.playing);
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    setPlaying(false);
    applyFrame(state.frame - 1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    setPlaying(false);
    applyFrame(state.frame + 1);
  } else if (event.key === "Home") {
    event.preventDefault();
    setPlaying(false);
    applyFrame(0);
  } else if (event.key === "End") {
    event.preventDefault();
    setPlaying(false);
    applyFrame(state.frameCount - 1);
  }
});

// --------------------------------------------------------------- path entry

async function submitPath() {
  const value = dom.pathInput.value.trim();
  if (!value) return;
  const previous = state.checkpoint;
  const entered = {
    id: value,
    label: value,
    path: value,
    model_name: "",
    selector: "",
    input_profile: null,
    max_views: null,
    supported: null,
    unsupported_reason: null,
    families: [],
  };
  state.checkpoint = entered;
  renderCheckpointList();
  updateRunState();
  try {
    const checkpoint = await validateCheckpointPath(value);
    state.checkpoint = { ...entered, ...checkpoint };
    renderCheckpointList();
    renderFamilyList();
    updateRunState();
    setStatus("");
  } catch (error) {
    state.checkpoint = previous;
    renderCheckpointList();
    renderFamilyList();
    updateRunState();
    setStatus(`パスの読み込みに失敗しました: ${error.message}`, "error");
  }
}

async function validateCheckpointPath(path) {
  const familyId = state.family || (state.catalog.families[0] || {}).id;
  if (!familyId) {
    throw new Error("検証に使えるシーンファミリがありません");
  }
  let sceneId = state.scene;
  if (!sceneId) {
    const params = new URLSearchParams({
      family: familyId,
      split: state.split,
      limit: "1",
    });
    const list = await fetchJson(`/api/scenes?${params.toString()}`);
    sceneId = (list.scenes || [])[0] ? list.scenes[0].id : null;
  }
  if (!sceneId) {
    throw new Error("検証に使えるシーンがありません");
  }
  const params = new URLSearchParams({ checkpoint: path });
  const detail = await fetchJson(
    `/api/scenes/${encode(familyId)}/${encode(sceneId)}?${params.toString()}`,
  );
  if (!detail.checkpoint) {
    throw new Error("サーバーがチェックポイント情報を返しませんでした");
  }
  return detail.checkpoint;
}

// -------------------------------------------------------------------- boot

syncLegend();
syncPlayButton();
updateRunState();
requestAnimationFrame(loop);
void init();
