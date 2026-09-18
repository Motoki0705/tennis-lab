// Browser wiring for the BLCS inference review UI: checkpoint + scene
// selection on the left, court playback in the centre, inference settings and
// results on the right. All drawing is delegated to the shared Three.js engine
// at /shared/scene3d.mjs; the score and fit math stays in ./scene.mjs.

import { Scene3D, frustumFromParams } from "/shared/scene3d.mjs";

import {
  GT_SHADES,
  PRED_SHADES,
  VIEW_PRESETS,
  courtBounds,
  fitViewDistance,
  normalizeSeries,
} from "./scene.mjs";

const PLAY_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z" fill="currentColor" stroke="none" /></svg>';
const PAUSE_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="14" y="3" width="5" height="18" rx="1" fill="currentColor" stroke="none" /><rect x="5" y="3" width="5" height="18" rx="1" fill="currentColor" stroke="none" /></svg>';

const SCENE_PAGE = 50;
const DEFAULT_FPS = 30;

const byId = (id) => document.getElementById(id);

const dom = {
  ckptCount: byId("ckpt-count"),
  ckptQuery: byId("ckpt-query"),
  ckptQueryClear: byId("ckpt-query-clear"),
  ckptNone: byId("ckpt-none"),
  ckptList: byId("ckpt-list"),
  formCount: byId("form-count"),
  formList: byId("form-list"),
  sceneCount: byId("scene-count"),
  sceneQuery: byId("scene-query"),
  sceneQueryClear: byId("scene-query-clear"),
  sceneList: byId("scene-list"),
  sceneMore: byId("scene-more"),
  split: byId("split"),
  splitButtons: Array.from(document.querySelectorAll(".split button")),
  sceneFormLabel: byId("scene-form-label"),
  sceneTitle: byId("scene-title"),
  sceneChips: byId("scene-chips"),
  presetButtons: Array.from(document.querySelectorAll("[data-preset]")),
  viewReset: byId("view-reset"),
  toggleGt: byId("toggle-gt"),
  togglePred: byId("toggle-pred"),
  toggleTrails: byId("toggle-trails"),
  transport: byId("transport"),
  play: byId("play"),
  prev: byId("prev"),
  next: byId("next"),
  toStart: byId("to-start"),
  toEnd: byId("to-end"),
  scrub: byId("scrub"),
  clock: byId("clock"),
  speeds: Array.from(document.querySelectorAll(".speeds button")),
  hud: byId("hud"),
  hudFrame: byId("hud-frame"),
  hudTime: byId("hud-time"),
  hudDevice: byId("hud-device"),
  hudWindow: byId("hud-window"),
  hudError: byId("hud-error"),
  status: byId("status"),
  ckptSummary: byId("ckpt-summary"),
  ckptNotice: byId("ckpt-notice"),
  device: byId("device"),
  cameraList: byId("camera-list"),
  cameraWarn: byId("camera-warn"),
  refRow: byId("ref-row"),
  refCamera: byId("ref-camera"),
  window: byId("window"),
  windowHint: byId("window-hint"),
  run: byId("run"),
  runLabel: byId("run-label"),
  results: byId("results"),
};

const state = {
  catalog: null,
  ckptQuery: "",
  checkpoint: null,
  allowedForms: null,
  form: null,
  split: "test",
  sceneQuery: "",
  scenes: [],
  sceneTotal: 0,
  scene: null,
  token: 0,
  listToken: 0,
  cameras: new Set(),
  device: "cuda",
  referenceCameraId: null,
  window: null,
  running: false,
  prediction: null,
  metrics: null,
  warnings: [],
  elapsedMs: null,
  deviceUsed: null,
  frames: 0,
  fps: DEFAULT_FPS,
  frame: 0,
  phase: 0,
  playing: false,
  speed: 1,
  lastTime: 0,
  court: null,
  gtSeries: null,
  gtVisible: true,
  predVisible: true,
  preset: "broadcast",
};

const scene3d = new Scene3D(byId("view"), { fov: 38 });

/**
 * Split one flat ``(frames, tracks, xyz)`` series into per-ball entities so the
 * shared engine can hide, colour and trail each trajectory independently.
 */
function ballEntities(series, idPrefix, shades) {
  const entities = [];
  const { frames, tracks, positions, mask } = series;
  for (let track = 0; track < tracks; track += 1) {
    const points = new Float32Array(frames * 3);
    const presence = new Uint8Array(frames);
    for (let frame = 0; frame < frames; frame += 1) {
      const source = (frame * tracks + track) * 3;
      points[frame * 3] = positions[source];
      points[frame * 3 + 1] = positions[source + 1];
      points[frame * 3 + 2] = positions[source + 2];
      presence[frame] = mask[frame * tracks + track];
    }
    entities.push({
      id: `${idPrefix}-${track}`,
      color: shades[track % shades.length],
      kind: "ball",
      frames,
      joints: 1,
      positions: points,
      roots: points,
      presence,
      edges: null,
      heading: null,
    });
  }
  return entities;
}

/** Rebuild the shared model from the installed court and series. */
function rebuildModel() {
  const court = state.court || state.catalog?.court;
  const entities = [];
  if (state.gtSeries && state.gtVisible) {
    entities.push(...ballEntities(state.gtSeries, "gt", GT_SHADES));
  }
  if (state.prediction && state.predVisible) {
    entities.push(...ballEntities(normalizeSeries(state.prediction), "pred", PRED_SHADES));
  }
  const frames = Math.max(state.gtSeries?.frames || 0, state.prediction?.frames || 0);
  scene3d.setModel({ court, frames, entities, cameras: sceneCameras() });
}

/**
 * Scene cameras for the shared engine. The BLCS scene payload stores OpenCV
 * pinhole parameters (``params``) rather than a frustum, so the frustum is
 * derived from those real values with the shared helper; nothing is invented.
 */
function sceneCameras() {
  const cameras = state.scene?.cameras || [];
  const depth = state.catalog?.camera_depth ?? 6;
  return cameras.map((camera, index) => ({
    id: camera.camera_id || `cam_${index}`,
    label: camera.camera_id || `cam_${index}`,
    center: camera.params?.C,
    frustum: frustumFromParams(camera.params, depth),
    rotation: camera.params?.R,
  }));
}

/** Frame the whole court for one preset using the tested point-fitter. */
function framePreset(name = "broadcast") {
  const court = state.court || state.catalog?.court;
  if (!court?.keypoints) return;
  const preset = VIEW_PRESETS[name] || VIEW_PRESETS.broadcast;
  const bounds = courtBounds(court.keypoints);
  const canvas = byId("view");
  const width = canvas.clientWidth || 1;
  const height = canvas.clientHeight || 1;
  const halfFov = (scene3d.fov * Math.PI) / 360;
  const probe = {
    yaw: preset.yaw,
    pitch: preset.pitch,
    distance: 40,
    target: bounds.center,
    halfFov,
    focal: height / 2 / Math.tan(halfFov),
    cx: width / 2,
    cy: height / 2,
    width,
    height,
  };
  // Frame the court together with the camera frustums so the camera geometry
  // stays on screen at every preset, not just when it happens to fall inside a
  // court-only fit.
  const fitPoints = [
    ...court.keypoints,
    ...sceneCameras().flatMap((camera) => camera.frustum || []),
  ];
  const distance = fitViewDistance(fitPoints, probe) * preset.fit;
  scene3d.setOrbit({
    yaw: preset.yaw,
    pitch: preset.pitch,
    distance,
    target: bounds.center,
  });
  state.preset = name;
}

/** Bring the stacked 3D view back into sight after a scene is chosen. */
function revealViewport() {
  if (!window.matchMedia("(max-width: 720px)").matches) return;
  document.querySelector(".viewport")?.scrollIntoView({ block: "start" });
}

// A thin facade over the shared engine so the rest of this module keeps its
// original scene/playhead vocabulary.
const view = {
  setScene({ court, gt }) {
    if (court) state.court = court;
    state.gtSeries = gt ? normalizeSeries(gt) : null;
    state.prediction = null;
    rebuildModel();
    framePreset(state.preset);
  },
  setGroundTruth(gt) {
    state.gtSeries = gt ? normalizeSeries(gt) : null;
    rebuildModel();
  },
  setPrediction(prediction) {
    state.prediction = prediction || null;
    rebuildModel();
  },
  setFrame(frame) {
    scene3d.setFrame(frame);
  },
  resetView(preset = "broadcast") {
    framePreset(preset);
  },
  resetFrame() {
    framePreset(state.preset);
  },
  draw() {
    scene3d.render();
  },
};

// ------------------------------------------------------------------ helpers

function encode(value) {
  return encodeURIComponent(value);
}

function matches(haystack, query) {
  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
  const text = String(haystack).toLowerCase();
  return tokens.every((token) => text.includes(token));
}

function formatBytes(bytes) {
  if (bytes === null || bytes === undefined) return "–";
  if (bytes < 1024) return `${bytes}B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes;
  let index = -1;
  do {
    value /= 1024;
    index += 1;
  } while (value >= 1024 && index < units.length - 1);
  return `${value.toFixed(value >= 100 ? 0 : 1)}${units[index]}`;
}

function formatMetric(value, digits = 3) {
  if (typeof value !== "number" || Number.isNaN(value)) return "–";
  return value.toFixed(digits);
}

function formatSeconds(seconds) {
  return `${seconds.toFixed(2)}s`;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(await describeError(response));
  }
  return response.json();
}

async function describeError(response) {
  try {
    const body = await response.json();
    if (body && typeof body.detail === "string") {
      return `${response.status} ${body.detail}`;
    }
  } catch (error) {
    void error;
  }
  return `${response.status} ${response.statusText}`;
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

function createChip(label, value, title) {
  const chip = document.createElement("span");
  chip.className = "chip";
  const labelEl = document.createElement("span");
  labelEl.className = "chip-label";
  labelEl.textContent = label;
  chip.append(labelEl);
  const strong = document.createElement("b");
  strong.textContent = String(value);
  chip.append(strong);
  chip.title = title || `${label} ${value}`;
  return chip;
}

// ----------------------------------------------------------------- catalog

function renderCatalogMeta() {
  renderCheckpoints();
  renderForms();
}

function checkpointButton(ckpt) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ckpt";
  if (!ckpt.runnable) button.setAttribute("aria-disabled", "true");
  if (state.checkpoint?.id === ckpt.id) button.setAttribute("aria-current", "true");

  const name = document.createElement("span");
  name.className = "ckpt-name";
  name.textContent = ckpt.name || ckpt.id;
  name.title = ckpt.name || ckpt.id;

  const meta = document.createElement("span");
  meta.className = "ckpt-meta";
  meta.textContent = [
    ckpt.model_name || ckpt.model_family,
    ckpt.object_mode,
    ckpt.reference ? "reference" : "single-view",
    `${(ckpt.allowed_forms || []).length} forms`,
    formatBytes(ckpt.size_bytes),
  ]
    .filter(Boolean)
    .join(" · ");
  meta.title = meta.textContent;

  button.append(name, meta);

  const chips = document.createElement("span");
  chips.className = "ckpt-chips";
  if (ckpt.scene_dir) {
    const base = String(ckpt.scene_dir).split("/").filter(Boolean).pop();
    chips.append(createChip("dir", base || ckpt.scene_dir, ckpt.scene_dir));
  }
  if (ckpt.seq_len !== null && ckpt.seq_len !== undefined) {
    chips.append(createChip("seq", ckpt.seq_len));
  }
  if (ckpt.num_court_tokens !== null && ckpt.num_court_tokens !== undefined) {
    chips.append(createChip("tokens", ckpt.num_court_tokens));
  }
  if (chips.childElementCount > 0) button.append(chips);

  button.addEventListener("click", () => {
    selectCheckpoint(state.checkpoint?.id === ckpt.id ? null : ckpt);
  });
  return button;
}

function renderCheckpoints() {
  dom.ckptList.textContent = "";
  const catalog = state.catalog;
  if (!catalog) {
    dom.ckptCount.textContent = "";
    return;
  }
  const roots = new Map((catalog.roots || []).map((root) => [root.id, root]));
  const groups = new Map();
  let visible = 0;
  for (const ckpt of catalog.checkpoints || []) {
    if (
      state.ckptQuery &&
      !matches(
        `${ckpt.name} ${ckpt.id} ${ckpt.path} ${ckpt.model_name} ${ckpt.object_mode} ${ckpt.root}`,
        state.ckptQuery,
      )
    ) {
      continue;
    }
    if (!groups.has(ckpt.root)) groups.set(ckpt.root, []);
    groups.get(ckpt.root).push(ckpt);
    visible += 1;
  }
  for (const [rootId, list] of groups) {
    const root = roots.get(rootId);
    const header = document.createElement("p");
    header.className = "ckpt-root";
    const path = root ? root.path : rootId;
    header.textContent = path;
    header.title = path;
    if (root && !root.exists) header.dataset.exists = "false";
    dom.ckptList.append(header);
    for (const ckpt of list) dom.ckptList.append(checkpointButton(ckpt));
  }
  dom.ckptCount.textContent = `${visible} / ${(catalog.checkpoints || []).length}`;
  if (visible === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = state.ckptQuery
      ? "一致するチェックポイントがありません"
      : "チェックポイントが見つかりません";
    dom.ckptList.append(empty);
  }
  dom.ckptNone.setAttribute("aria-pressed", String(state.checkpoint === null));
}

function renderForms() {
  dom.formList.textContent = "";
  const catalog = state.catalog;
  if (!catalog) {
    dom.formCount.textContent = "";
    return;
  }
  const forms = catalog.scene_forms || [];
  for (const form of forms) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "form";
    button.dataset.form = form.id;
    if (state.form === form.id) button.setAttribute("aria-current", "true");
    const blocked = state.allowedForms !== null && !state.allowedForms.has(form.id);
    if (blocked) {
      button.setAttribute("aria-disabled", "true");
      button.disabled = true;
    }
    const name = document.createElement("span");
    name.className = "form-name";
    name.textContent = form.id;
    name.title = form.path || form.id;
    const count = document.createElement("span");
    count.className = "form-count";
    count.textContent = `${form.scene_count}`;
    button.append(name, count);
    button.addEventListener("click", () => selectForm(form.id));
    dom.formList.append(button);
  }
  dom.formCount.textContent = `${forms.length}`;
}

// -------------------------------------------------------------------- scenes

let sceneSearchTimer = 0;

async function loadScenes(reset) {
  if (!state.form) {
    state.scenes = [];
    state.sceneTotal = 0;
    renderScenes();
    return;
  }
  const token = (state.listToken += 1);
  const offset = reset ? 0 : state.scenes.length;
  try {
    const url =
      `/api/scenes?form=${encode(state.form)}&split=${encode(state.split)}` +
      `&query=${encode(state.sceneQuery)}&offset=${offset}&limit=${SCENE_PAGE}`;
    const data = await fetchJson(url);
    if (token !== state.listToken) return;
    state.sceneTotal = data.total;
    state.scenes = reset ? data.scenes : state.scenes.concat(data.scenes);
    renderScenes();
  } catch (error) {
    if (token !== state.listToken) return;
    state.scenes = reset ? [] : state.scenes;
    renderScenes();
    dom.sceneCount.textContent = "エラー";
    setStatus(`シーン一覧の読み込みに失敗しました: ${error.message}`, "error");
  }
}

function renderScenes() {
  dom.sceneList.textContent = "";
  for (const scene of state.scenes) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "scene";
    button.dataset.scene = scene.id;
    if (state.scene?.scene_id === scene.id && state.scene?.form === state.form) {
      button.setAttribute("aria-current", "true");
    }
    const id = document.createElement("span");
    id.className = "scene-id";
    id.textContent = scene.id;
    id.title = scene.id;
    const meta = document.createElement("span");
    meta.className = "scene-meta";
    meta.textContent =
      `${scene.frame_count}f · ${scene.num_cameras}cam · ${scene.num_balls}ball`;
    button.append(id, meta);
    button.addEventListener("click", () => selectScene(scene.id));
    dom.sceneList.append(button);
  }
  if (state.scenes.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = state.form
      ? "シーンが見つかりません"
      : "シーン形式を選択してください";
    dom.sceneList.append(empty);
  }
  dom.sceneCount.textContent = `${state.scenes.length} / ${state.sceneTotal}`;
  dom.sceneMore.hidden = state.scenes.length >= state.sceneTotal;
}

// --------------------------------------------------------------- selections

function clearPrediction() {
  state.prediction = null;
  state.metrics = null;
  state.warnings = [];
  state.elapsedMs = null;
  state.deviceUsed = null;
  view.setPrediction(null);
  renderResults();
  renderHud();
  view.draw();
}

function clearScene() {
  state.scene = null;
  state.cameras = new Set();
  state.referenceCameraId = null;
  state.frames = 0;
  state.frame = 0;
  state.phase = 0;
  state.playing = false;
  syncPlayButton();
  view.setScene({ court: state.catalog?.court, world: null, gt: null });
  clearPrediction();
  dom.transport.hidden = true;
  renderSceneHeader();
  renderInspector();
  updateRunButton();
  setStatus("左の一覧からシーンを選択してください", "info");
  view.draw();
}

async function selectForm(id) {
  if (state.allowedForms !== null && !state.allowedForms.has(id)) return;
  state.form = id;
  clearScene();
  renderForms();
  await loadScenes(true);
}

function selectCheckpoint(ckpt) {
  state.checkpoint = ckpt;
  state.allowedForms = ckpt ? new Set(ckpt.allowed_forms || []) : null;
  state.window = ckpt && ckpt.seq_len ? ckpt.seq_len : null;

  if (ckpt && state.form && !state.allowedForms.has(state.form)) {
    const first = (state.catalog.scene_forms || []).find((form) =>
      state.allowedForms.has(form.id),
    );
    state.form = first ? first.id : null;
    clearScene();
    renderForms();
    loadScenes(true);
  }

  clearPrediction();
  renderCheckpoints();
  renderForms();
  renderInspector();
  updateRunButton();
}

async function selectScene(sceneId) {
  const form = state.form;
  if (!form) return;
  const token = (state.token += 1);
  state.playing = false;
  syncPlayButton();
  setStatus("シーンを読み込み中…", "busy");
  try {
    const data = await fetchJson(`/api/scene?form=${encode(form)}&scene=${encode(sceneId)}`);
    if (token !== state.token) return;
    state.scene = data;
    state.fps = Number(data.fps) || DEFAULT_FPS;
    state.frames = data.gt?.frames ?? data.frame_count ?? 0;
    state.frame = 0;
    state.phase = 0;
    state.cameras = new Set((data.cameras || []).map((camera) => camera.index));
    state.referenceCameraId = null;
    state.prediction = null;
    state.metrics = null;
    state.warnings = [];
    state.elapsedMs = null;
    state.deviceUsed = null;
    view.setScene({ court: state.catalog.court, world: data.world, gt: data.gt });
    view.setPrediction(null);
    applyFrame(0);
    renderScenes();
    renderSceneHeader();
    renderInspector();
    updateRunButton();
    setStatus("");
    revealViewport();
  } catch (error) {
    if (token !== state.token) return;
    setStatus(`シーンの読み込みに失敗しました: ${error.message}`, "error");
  }
}

// -------------------------------------------------------------- scene header

function renderSceneHeader() {
  const scene = state.scene;
  if (!scene) {
    dom.sceneFormLabel.textContent = state.form ? `形式 ${state.form}` : "シーン未選択";
    dom.sceneTitle.textContent = "シーンを選択してください";
    dom.sceneChips.textContent = "";
    return;
  }
  dom.sceneFormLabel.textContent = `形式 ${scene.form}`;
  dom.sceneTitle.textContent = scene.scene_id;
  dom.sceneTitle.title = scene.scene_id;
  dom.sceneChips.textContent = "";
  const chips = [
    ["frames", scene.gt?.frames ?? scene.frame_count],
    ["fps", Math.round(state.fps)],
    ["cams", (scene.cameras || []).length],
    ["tracks", scene.gt?.tracks ?? "–"],
  ];
  for (const [label, value] of chips) {
    dom.sceneChips.append(createChip(label, value));
  }
}

// ---------------------------------------------------------------- inspector

function renderInspector() {
  const ckpt = state.checkpoint;
  dom.ckptSummary.textContent = "";
  if (!ckpt) {
    dom.ckptSummary.append(summaryRow("状態", "ckpt 指定なし"));
  } else {
    const rows = [
      ["name", ckpt.name],
      ["model", ckpt.model_name || ckpt.model_family],
      ["family", ckpt.model_family],
      ["object_mode", ckpt.object_mode],
      ["reference", ckpt.reference ? "true" : "false"],
      ["scene_dir", ckpt.scene_dir],
      ["seq_len", ckpt.seq_len],
      ["court_tokens", ckpt.num_court_tokens],
      ["device", state.deviceUsed || state.device],
    ];
    for (const [label, value] of rows) {
      if (value === null || value === undefined || value === "") continue;
      dom.ckptSummary.append(summaryRow(label, value));
    }
  }

  if (ckpt && !ckpt.runnable) {
    dom.ckptNotice.hidden = false;
    dom.ckptNotice.dataset.tone = "error";
    dom.ckptNotice.textContent = `このチェックポイントは実行できません: ${
      ckpt.unavailable_reason || "理由が提供されていません"
    }`;
  } else {
    dom.ckptNotice.hidden = true;
    dom.ckptNotice.textContent = "";
  }

  renderCameraList();
  renderReferenceRow();
  renderWindowField();
}

function summaryRow(label, value) {
  const row = document.createElement("div");
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = String(value);
  dd.title = String(value);
  row.append(dt, dd);
  return row;
}

function renderCameraList() {
  dom.cameraList.textContent = "";
  const cameras = state.scene?.cameras || [];
  if (cameras.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "シーンを選択してください";
    dom.cameraList.append(empty);
    updateCameraWarn();
    return;
  }
  for (const camera of cameras) {
    const label = document.createElement("label");
    label.className = "check";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = String(camera.index);
    input.checked = state.cameras.has(camera.index);
    input.disabled = state.running;
    input.addEventListener("change", () => {
      if (input.checked) state.cameras.add(camera.index);
      else state.cameras.delete(camera.index);
      onSettingsChanged();
    });
    const text = document.createElement("span");
    const id = camera.camera_id ?? `cam${camera.index}`;
    text.textContent = `${id} · #${camera.index}`;
    text.title = `${id} (#${camera.index})`;
    label.append(input, text);
    dom.cameraList.append(label);
  }
  updateCameraWarn();
}

function updateCameraWarn() {
  const reference = Boolean(state.checkpoint?.reference);
  const count = state.cameras.size;
  if (reference && (count < 3 || count > 4)) {
    dom.cameraWarn.hidden = false;
    dom.cameraWarn.textContent = `参照モデルは3〜4台のカメラを想定しています（現在 ${count} 台）。`;
  } else {
    dom.cameraWarn.hidden = true;
    dom.cameraWarn.textContent = "";
  }
}

function renderReferenceRow() {
  const reference = Boolean(state.checkpoint?.reference);
  dom.refRow.hidden = !reference;
  if (!reference) return;
  const candidates = state.scene?.reference_candidate_ids || [];
  const selectedIds = (state.scene?.cameras || [])
    .filter((camera) => state.cameras.has(camera.index))
    .map((camera) => camera.camera_id)
    .filter(Boolean);

  const options = candidates.filter((id) => id !== null && id !== undefined);
  const preferred = selectedIds.find((id) => options.includes(id)) ?? options[0] ?? null;
  if (!options.includes(state.referenceCameraId)) {
    state.referenceCameraId = preferred;
  }
  dom.refCamera.textContent = "";
  for (const id of options) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = id;
    if (id === state.referenceCameraId) option.selected = true;
    dom.refCamera.append(option);
  }
  dom.refCamera.disabled = state.running || options.length === 0;
}

function renderWindowField() {
  const seqLen = state.checkpoint?.seq_len ?? null;
  if (state.window !== null) {
    dom.window.value = String(state.window);
  }
  dom.windowHint.textContent = seqLen
    ? `チェックポイントの seq_len = ${seqLen}`
    : "–";
}

function onSettingsChanged() {
  clearPrediction();
  renderReferenceRow();
  updateCameraWarn();
  renderInspector();
  updateRunButton();
}

function canRun() {
  return Boolean(
    state.checkpoint &&
      state.checkpoint.runnable !== false &&
      state.scene &&
      !state.running,
  );
}

function updateRunButton() {
  dom.run.disabled = !canRun();
  dom.run.dataset.running = String(state.running);
  dom.runLabel.textContent = state.running ? "実行中…" : "推論実行";
}

function setRunning(running) {
  state.running = running;
  dom.device.disabled = running;
  dom.window.disabled = running;
  dom.refCamera.disabled = running;
  for (const input of dom.cameraList.querySelectorAll("input")) input.disabled = running;
  updateRunButton();
}

function syncPresetButtons(active) {
  for (const button of dom.presetButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.preset === active));
  }
}

// ------------------------------------------------------------------- results

function renderResults(errorMessage) {
  dom.results.textContent = "";
  if (errorMessage) {
    const error = document.createElement("p");
    error.className = "notice";
    error.dataset.tone = "error";
    error.textContent = errorMessage;
    dom.results.append(error);
  }
  if (!state.metrics && !state.prediction) {
    if (!errorMessage) {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "まだ推論を実行していません";
      dom.results.append(empty);
    }
    return;
  }
  if (state.metrics) {
    const metrics = document.createElement("dl");
    metrics.className = "metrics";
    const rows = [
      ["device", state.deviceUsed || state.device],
      ["elapsed", state.elapsedMs !== null ? `${Math.round(state.elapsedMs)} ms` : "–"],
      ["position_error_m", formatMetric(state.metrics.position_error_m)],
      ["endpoint_error_m", formatMetric(state.metrics.endpoint_error_m)],
      ["accuracy_0p3m", formatMetric(state.metrics.accuracy_0p3m, 3)],
      ["tracks", state.prediction?.tracks ?? "–"],
    ];
    for (const [label, value] of rows) {
      const row = document.createElement("div");
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = String(value);
      row.append(dt, dd);
      metrics.append(row);
    }
    dom.results.append(metrics);
  }
  if (state.warnings.length > 0) {
    const list = document.createElement("ul");
    list.className = "warnings";
    for (const warning of state.warnings) {
      const item = document.createElement("li");
      item.textContent = warning;
      list.append(item);
    }
    dom.results.append(list);
  }
}

async function runInference() {
  if (!canRun()) return;
  const token = (state.token += 1);
  setRunning(true);
  setStatus("GPUキュー・推論実行中", "busy");
  renderResults();
  const body = {
    checkpoint: state.checkpoint.id,
    form: state.form,
    scene: state.scene.scene_id,
    cameras: Array.from(state.cameras).sort((a, b) => a - b),
    reference_camera_id: state.checkpoint.reference ? state.referenceCameraId : null,
    device: state.device,
    window: state.window,
  };
  try {
    const response = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(await describeError(response));
    const data = await response.json();
    if (token !== state.token) return;
    state.prediction = data.prediction;
    state.metrics = data.metrics;
    state.warnings = data.warnings || [];
    state.elapsedMs = data.elapsed_ms;
    state.deviceUsed = data.device;
    if (data.gt && (!state.scene || !state.scene.gt)) {
      view.setGroundTruth(data.gt);
    }
    view.setPrediction(data.prediction);
    if (typeof data.frames === "number" && data.frames > 0) {
      state.frames = data.frames;
    }
    renderResults();
    renderInspector();
    renderHud();
    applyFrame(state.frame);
    setStatus("");
    view.draw();
  } catch (error) {
    if (token !== state.token) return;
    state.prediction = null;
    state.metrics = null;
    state.warnings = [];
    state.elapsedMs = null;
    view.setPrediction(null);
    renderResults(`推論に失敗しました: ${error.message}`);
    setStatus(`推論に失敗しました: ${error.message}`, "error");
    view.draw();
  } finally {
    if (token === state.token) setRunning(false);
  }
}

// ------------------------------------------------------------------ playback

function applyFrame(frame) {
  if (state.frames <= 0) {
    dom.transport.hidden = true;
    renderHud();
    return;
  }
  state.frame = Math.max(0, Math.min(state.frames - 1, Math.round(frame)));
  state.phase = state.frame;
  view.setFrame(state.frame);
  dom.transport.hidden = false;
  dom.scrub.max = String(state.frames - 1);
  dom.scrub.value = String(state.frame);
  dom.clock.textContent = `${state.frame + 1} / ${state.frames}`;
  renderHud();
}

function renderHud() {
  const hasScene = state.frames > 0;
  dom.hud.hidden = !hasScene;
  dom.hudFrame.textContent = hasScene ? String(state.frame) : "–";
  dom.hudTime.textContent = hasScene ? formatSeconds(state.frame / state.fps) : "–";
  dom.hudDevice.textContent = state.deviceUsed || state.device;
  dom.hudWindow.textContent = state.window !== null ? String(state.window) : "–";
  dom.hudError.textContent = state.metrics
    ? `${formatMetric(state.metrics.position_error_m)} m`
    : "–";
}

function setPlaying(playing) {
  state.playing = playing && state.frames > 0;
  syncPlayButton();
}

function syncPlayButton() {
  dom.play.innerHTML = state.playing ? PAUSE_ICON : PLAY_ICON;
  dom.play.title = state.playing ? "一時停止" : "再生";
  dom.play.setAttribute("aria-label", state.playing ? "一時停止" : "再生");
}

function loop(now) {
  const delta = state.lastTime ? (now - state.lastTime) / 1000 : 0;
  state.lastTime = now;
  if (state.playing && state.frames > 0) {
    state.phase += delta * state.fps * state.speed;
    if (state.phase >= state.frames) state.phase -= state.frames;
    const target = Math.floor(state.phase);
    if (target !== state.frame) applyFrame(target);
  }
  view.draw();
  requestAnimationFrame(loop);
}

// -------------------------------------------------------------------- events

function wireEvents() {
  dom.ckptQuery.addEventListener("input", () => {
    state.ckptQuery = dom.ckptQuery.value.trim();
    dom.ckptQueryClear.hidden = state.ckptQuery.length === 0;
    renderCheckpoints();
  });
  dom.ckptQueryClear.addEventListener("click", () => {
    dom.ckptQuery.value = "";
    state.ckptQuery = "";
    dom.ckptQueryClear.hidden = true;
    dom.ckptQuery.focus();
    renderCheckpoints();
  });
  dom.ckptNone.addEventListener("click", () => selectCheckpoint(null));

  for (const button of dom.splitButtons) {
    button.addEventListener("click", () => {
      state.split = button.dataset.split;
      for (const other of dom.splitButtons) {
        other.setAttribute("aria-pressed", String(other === button));
      }
      loadScenes(true);
    });
  }
  dom.sceneQuery.addEventListener("input", () => {
    state.sceneQuery = dom.sceneQuery.value.trim();
    dom.sceneQueryClear.hidden = state.sceneQuery.length === 0;
    clearTimeout(sceneSearchTimer);
    sceneSearchTimer = setTimeout(() => loadScenes(true), 200);
  });
  dom.sceneQueryClear.addEventListener("click", () => {
    dom.sceneQuery.value = "";
    state.sceneQuery = "";
    dom.sceneQueryClear.hidden = true;
    dom.sceneQuery.focus();
    loadScenes(true);
  });
  dom.sceneMore.addEventListener("click", () => loadScenes(false));

  for (const button of dom.presetButtons) {
    button.addEventListener("click", () => {
      view.resetView(button.dataset.preset);
      syncPresetButtons(button.dataset.preset);
      view.draw();
    });
  }
  dom.viewReset.addEventListener("click", () => {
    view.resetView("broadcast");
    syncPresetButtons("broadcast");
    view.draw();
  });
  dom.toggleGt.addEventListener("click", () => {
    state.gtVisible = !state.gtVisible;
    dom.toggleGt.setAttribute("aria-pressed", String(state.gtVisible));
    rebuildModel();
    view.draw();
  });
  dom.togglePred.addEventListener("click", () => {
    state.predVisible = !state.predVisible;
    dom.togglePred.setAttribute("aria-pressed", String(state.predVisible));
    rebuildModel();
    view.draw();
  });
  dom.toggleTrails.addEventListener("click", () => {
    const next = !scene3d.showTrail;
    scene3d.setTrailsVisible(next);
    dom.toggleTrails.setAttribute("aria-pressed", String(next));
    view.draw();
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
    applyFrame(state.frames - 1);
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

  dom.device.addEventListener("change", () => {
    state.device = dom.device.value;
    onSettingsChanged();
  });
  dom.window.addEventListener("change", () => {
    let value = Number.parseInt(dom.window.value, 10);
    if (!Number.isFinite(value) || value < 1) value = 1;
    const max = Number(state.checkpoint?.max_seq_len);
    if (Number.isFinite(max) && max >= 1) value = Math.min(value, max);
    state.window = value;
    dom.window.value = String(value);
    onSettingsChanged();
  });
  dom.refCamera.addEventListener("change", () => {
    state.referenceCameraId = dom.refCamera.value;
    onSettingsChanged();
  });
  dom.run.addEventListener("click", () => runInference());

  window.addEventListener("keydown", (event) => {
    const target = event.target;
    if (
      target instanceof HTMLInputElement ||
      target instanceof HTMLSelectElement ||
      target instanceof HTMLTextAreaElement ||
      target?.isContentEditable
    ) {
      return;
    }
    if (event.key === " ") {
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
      view.resetView();
      view.draw();
    }
  });
}

// ------------------------------------------------------------------- startup

async function init() {
  wireEvents();
  syncPlayButton();
  renderHud();
  try {
    state.catalog = await fetchJson("/api/catalog");
  } catch (error) {
    setStatus(`カタログの読み込みに失敗しました: ${error.message}`, "error");
    requestAnimationFrame(loop);
    return;
  }
  state.court = state.catalog.court;
  rebuildModel();
  framePreset("broadcast");
  syncPresetButtons("broadcast");
  view.draw();
  renderCatalogMeta();
  renderInspector();
  updateRunButton();
  const firstForm = (state.catalog.scene_forms || [])[0];
  if (firstForm) {
    state.form = firstForm.id;
    renderForms();
    await loadScenes(true);
  }
  setStatus("左の一覧からシーンを選択してください", "info");
  requestAnimationFrame(loop);
}

init();
