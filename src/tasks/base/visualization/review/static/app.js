// Browser wiring for the dataset scene review UI: form/scene browser on the
// left, live 3D playback of the selected scene in the centre. All
// drawing is delegated to the shared Three.js engine at /shared/scene3d.mjs.

import { Scene3D } from "/shared/scene3d.mjs";

import { COLORS } from "./scene.mjs";
import { buildModel, decodeBuffers, entityGroups } from "./model.mjs";

const PLAY_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z" fill="currentColor" stroke="none" /></svg>';
const PAUSE_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="14" y="3" width="5" height="18" rx="1" fill="currentColor" stroke="none" /><rect x="5" y="3" width="5" height="18" rx="1" fill="currentColor" stroke="none" /></svg>';

const LEGEND_AXES = [
  [COLORS.axisX, "X"],
  [COLORS.axisY, "Y"],
  [COLORS.axisZ, "Z"],
];

const byId = (id) => document.getElementById(id);
const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
if (!littleEndian) {
  throw new Error("This viewer requires a little-endian platform for entity buffers.");
}

const dom = {
  brandTask: byId("brand-task"),
  query: byId("query"),
  clear: byId("clear"),
  tree: byId("tree"),
  resultCount: byId("result-count"),
  dataRoot: byId("data-root"),
  dataset: byId("scene-dataset"),
  title: byId("scene-title"),
  note: byId("scene-note"),
  chips: byId("chips"),
  presets: Array.from(document.querySelectorAll("[data-preset]")),
  cameras: byId("cameras"),
  follow: byId("follow"),
  trail: byId("trail"),
  openCamera: byId("open-camera"),
  cameraReadout: byId("camera-readout"),
  reset: byId("reset"),
  legend: byId("legend"),
  hud: byId("hud"),
  hudFrame: byId("hud-frame"),
  hudTime: byId("hud-time"),
  hudPos: byId("hud-pos"),
  hudYaw: byId("hud-yaw"),
  status: byId("status"),
  transport: byId("transport"),
  play: byId("play"),
  prev: byId("prev"),
  next: byId("next"),
  toStart: byId("to-start"),
  toEnd: byId("to-end"),
  scrub: byId("scrub"),
  clock: byId("clock"),
  speeds: Array.from(document.querySelectorAll(".speeds button")),
};

const state = {
  catalog: null,
  scenes: new Map(),
  query: "",
  selection: null,
  scene: null,
  model: null,
  openForms: new Set(),
  frame: 0,
  phase: 0,
  playing: false,
  speed: 1,
  token: 0,
  lastTime: 0,
};

const view = new Scene3D(byId("view"), { fov: 36 });

// ------------------------------------------------------------------ helpers

function encode(value) {
  return encodeURIComponent(value);
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
    // fall through to the status text when the body is not JSON
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

function matches(haystack, query) {
  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
  const text = haystack.toLowerCase();
  return tokens.every((token) => text.includes(token));
}

// -------------------------------------------------------------------- tree

function filteredForms() {
  if (!state.query) {
    return state.catalog.forms.map((form) => ({ form, scenes: null }));
  }
  const result = [];
  for (const form of state.catalog.forms) {
    const formMatches = matches(form.name, state.query);
    const scenes = state.scenes.get(form.name) || null;
    const sceneMatches = scenes
      ? scenes.filter((id) => matches(id, state.query) || formMatches)
      : null;
    if (formMatches || (sceneMatches && sceneMatches.length > 0)) {
      result.push({ form, scenes: formMatches && scenes ? scenes : sceneMatches });
    }
  }
  return result;
}

function renderTree() {
  const catalog = state.catalog;
  dom.tree.textContent = "";
  if (!catalog) {
    dom.resultCount.textContent = "";
    return;
  }
  const groups = filteredForms();
  let visible = 0;
  for (const { form, scenes } of groups) {
    const group = document.createElement("details");
    group.className = "form";
    // Keep a group open across the re-render that follows its scene load, so
    // expanding a non-selected form does not snap shut under the user.
    group.open =
      Boolean(state.query) ||
      state.selection?.form === form.name ||
      state.openForms.has(form.name);
    const summary = document.createElement("summary");
    const name = document.createElement("span");
    name.className = "form-name";
    name.textContent = form.name;
    const tags = document.createElement("span");
    tags.className = "form-tags";
    for (const [text, title] of [
      [form.mode, "オブジェクト数"],
      [form.has_samples === undefined ? null : form.has_samples ? "samples" : "no-samples", "サンプル画像の有無"],
    ]) {
      if (text === null) continue;
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = text;
      tag.title = title;
      tags.append(tag);
    }
    const count = document.createElement("span");
    count.className = "form-count";
    count.textContent = String(scenes ? scenes.length : form.scene_count);
    summary.append(name, tags, count);
    group.append(summary);

    const list = document.createElement("ul");
    list.className = "scene-list";
    const names = scenes || state.scenes.get(form.name);
    if (names) {
      visible += names.length;
      for (const sceneId of names) {
        list.append(sceneButton(form.name, sceneId));
      }
      if (names.length === 0) {
        list.append(note("一致するシーンがありません"));
      }
    } else {
      list.append(note("展開すると読み込みます"));
    }
    group.append(list);
    group.addEventListener("toggle", () => {
      if (group.open) {
        state.openForms.add(form.name);
        ensureScenes(form.name);
      } else {
        state.openForms.delete(form.name);
      }
    });
    dom.tree.append(group);
  }
  dom.resultCount.textContent = state.query
    ? `${visible} シーンが「${state.query}」に一致`
    : `${catalog.forms.reduce((sum, form) => sum + form.scene_count, 0)} シーン / ${catalog.forms.length} 形式`;
  if (groups.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "一致する形式・シーンがありません";
    dom.tree.append(empty);
  }
}

function note(text) {
  const item = document.createElement("li");
  item.className = "tree-note";
  item.textContent = text;
  return item;
}

function sceneButton(formName, sceneId) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "scene";
  button.dataset.form = formName;
  button.dataset.scene = sceneId;
  button.textContent = sceneId;
  if (state.selection?.form === formName && state.selection?.scene === sceneId) {
    button.setAttribute("aria-current", "true");
  }
  return button;
}

async function ensureScenes(formName) {
  if (state.scenes.has(formName)) return;
  state.scenes.set(formName, null);
  try {
    const payload = await fetchJson(`/api/scenes?form=${encode(formName)}`);
    state.scenes.set(formName, payload.scenes);
    renderTree();
  } catch (error) {
    state.scenes.delete(formName);
    setStatus(`シーン一覧の読み込みに失敗しました: ${error.message}`, "error");
  }
}

async function ensureAllScenes() {
  await Promise.all(
    state.catalog.forms
      .filter((form) => !state.scenes.has(form.name))
      .map((form) => ensureScenes(form.name)),
  );
}

// -------------------------------------------------------------------- scene

async function selectedScene(formName, sceneId) {
  state.selection = { form: formName, scene: sceneId };
  state.playing = false;
  syncPlayButton();
  renderTree();
  const token = (state.token += 1);
  dom.dataset.textContent = `${state.catalog.task}/${formName}`;
  dom.title.textContent = sceneId;
  setStatus("読み込み中…", "info");
  try {
    const scene = await fetchJson(
      `/api/scene?form=${encode(formName)}&scene=${encode(sceneId)}`,
    );
    if (token !== state.token) return;
    const response = await fetch(
      `/api/scene/buffer?form=${encode(formName)}&scene=${encode(sceneId)}&revision=${encode(scene.revision)}`,
    );
    if (!response.ok) {
      throw new Error(await describeError(response));
    }
    const buffer = await response.arrayBuffer();
    if (token !== state.token) return;
    const buffers = decodeBuffers(buffer, scene);
    state.scene = scene;
    dom.dataset.textContent = scene.dataset;
    dom.note.textContent = scene.description ?? "";
    dom.note.hidden = !scene.description;
    state.model = buildModel(scene, buffers);
    state.frame = 0;
    state.phase = 0;
    view.setModel(state.model);
    dom.scrub.max = String(scene.frame_count - 1);
    dom.transport.hidden = false;
    dom.hud.hidden = false;
    selectCamera(null);
    dom.cameras.disabled = scene.cameras.length === 0;
    renderChips(scene);
    renderLegend(scene);
    applyFrame(0);
    setStatus("");
    state.playing = true;
    syncPlayButton();
    revealViewport();
  } catch (error) {
    if (token !== state.token) return;
    setStatus(`読み込みに失敗しました: ${error.message}`, "error");
    dom.transport.hidden = true;
    dom.hud.hidden = true;
  }
}

function renderChips(scene) {
  const chips = [
    ["mode", scene.mode],
    ["fps", Math.round(scene.fps)],
    ["frames", scene.frame_count],
    ["slots", entityGroups(scene).reduce((sum, entity) => sum + entity.slots, 0)],
    [scene.source_camera_ids ? "views" : "cams", scene.source_camera_ids?.length ?? scene.cameras.length],
  ];
  dom.chips.textContent = "";
  for (const [label, value] of chips) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.append(`${label}`);
    const strong = document.createElement("b");
    strong.textContent = String(value);
    chip.append(strong);
    dom.chips.append(chip);
  }
}

function renderLegend(scene) {
  dom.legend.textContent = "";
  for (const [color, label] of LEGEND_AXES) {
    const span = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.className = "swatch";
    swatch.style.background = color;
    span.append(swatch, label);
    dom.legend.append(span);
  }
  const camera = document.createElement("span");
  camera.textContent = scene.source_camera_ids
    ? `入力カメラ ${scene.source_camera_ids.join(", ")}`
    : `カメラ ${scene.cameras.length}`;
  dom.legend.append(camera);
  for (const group of entityGroups(scene)) {
    const entity = document.createElement("span");
    entity.textContent = group.kind === "ball"
      ? `ボール ${group.slots}`
      : `選手 ${group.slots}${group.joint_count > 1 ? ` / 関節 ${group.joint_count}` : " / ルート位置"}`;
    dom.legend.append(entity);
  }
}

function applyFrame(frame) {
  const scene = state.scene;
  if (!scene) return;
  const total = scene.frame_count;
  state.frame = Math.max(0, Math.min(total - 1, Math.round(frame)));
  state.phase = state.frame;
  view.setFrame(state.frame);
  dom.scrub.value = String(state.frame);
  dom.clock.textContent = `${state.frame + 1} / ${total}`;
  dom.hudFrame.textContent = String(state.frame);
  dom.hudTime.textContent = formatSeconds(state.frame / scene.fps);
  const entity = state.model.entities[0];
  const present = entity && (!entity.presence || entity.presence[state.frame] === 1);
  if (present) {
    const root = entity.roots.subarray(state.frame * 3, state.frame * 3 + 3);
    dom.hudPos.textContent =
      `${root[0].toFixed(2)}, ${root[1].toFixed(2)}, ${root[2].toFixed(2)} m`;
  } else {
    dom.hudPos.textContent = "–";
  }
  const orientation = entity?.heading;
  if (present && orientation) {
    const base = state.frame * 2;
    const yaw = Math.atan2(orientation[base + 1], orientation[base]);
    dom.hudYaw.textContent = `${((yaw * 180) / Math.PI).toFixed(1)}°`;
  } else {
    dom.hudYaw.textContent = "–";
  }
}

function setPlaying(playing) {
  state.playing = playing && Boolean(state.scene);
  syncPlayButton();
}

function syncPlayButton() {
  dom.play.innerHTML = state.playing ? PAUSE_ICON : PLAY_ICON;
  dom.play.title = state.playing ? "一時停止" : "再生";
}

function selectCamera(id) {
  view.selectCamera(id);
  if (id === null || id === undefined) {
    dom.cameraReadout.textContent = "カメラ";
    dom.openCamera.disabled = true;
  } else {
    dom.cameraReadout.textContent = String(id);
    dom.openCamera.disabled = false;
  }
}

/** Bring the stacked 3D view back into sight after a scene is chosen. */
function revealViewport() {
  if (!window.matchMedia("(max-width: 720px)").matches) return;
  document.querySelector(".viewport")?.scrollIntoView({ block: "start" });
}

// ------------------------------------------------------------------- frame

function loop(now) {
  const delta = state.lastTime ? (now - state.lastTime) / 1000 : 0;
  state.lastTime = now;
  if (state.playing && state.scene) {
    const total = state.scene.frame_count;
    state.phase += delta * state.scene.fps * state.speed;
    if (state.phase >= total) {
      state.phase -= total;
    }
    const target = Math.floor(state.phase);
    if (target !== state.frame) {
      state.frame = target;
      applyFrame(target);
    }
  }
  view.render();
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
  dom.brandTask.textContent = state.catalog.task.toUpperCase();
  document.title = `${state.catalog.task.toUpperCase()} Dataset Review`;
  dom.dataRoot.textContent = state.catalog.root;
  dom.dataRoot.title = state.catalog.root;
  renderTree();
  const first = state.catalog.forms[0];
  if (!first) return;
  await ensureScenes(first.name);
  const firstScene = state.scenes.get(first.name)?.[0];
  if (firstScene) {
    await selectedScene(first.name, firstScene);
  }
}

dom.query.addEventListener("input", async () => {
  state.query = dom.query.value.trim();
  dom.clear.hidden = state.query.length === 0;
  if (state.query) {
    await ensureAllScenes();
  }
  renderTree();
});
dom.clear.addEventListener("click", () => {
  dom.query.value = "";
  state.query = "";
  dom.clear.hidden = true;
  dom.query.focus();
  renderTree();
});
dom.tree.addEventListener("click", (event) => {
  const button = event.target.closest(".scene");
  if (!button) return;
  selectedScene(button.dataset.form, button.dataset.scene);
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
  applyFrame(state.scene ? state.scene.frame_count - 1 : 0);
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
for (const button of dom.presets) {
  button.addEventListener("click", () => {
    view.applyPreset(button.dataset.preset);
    view.render();
  });
}
dom.cameras.addEventListener("click", () => {
  const next = !view.showCameras;
  view.setCamerasVisible(next);
  dom.cameras.setAttribute("aria-pressed", String(next));
});
dom.follow.addEventListener("click", () => {
  view.setFollow(!view.follow);
  dom.follow.setAttribute("aria-pressed", String(view.follow));
});
dom.trail.addEventListener("click", () => {
  const next = !view.showTrail;
  view.setTrailsVisible(next);
  dom.trail.setAttribute("aria-pressed", String(next));
});
dom.reset.addEventListener("click", () => {
  view.resetView();
  view.render();
});
dom.openCamera.addEventListener("click", () => {
  if (view.selectedCamera === null) return;
  view.lookThroughCamera(view.selectedCamera);
  view.render();
});
view.onFollowChange = (on) => dom.follow.setAttribute("aria-pressed", String(on));
view.onCameraPick = (id) => selectCamera(id);

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) return;
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
    view.render();
  }
});

syncPlayButton();
requestAnimationFrame(loop);
init();
