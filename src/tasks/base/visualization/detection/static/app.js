import { ImageViewer } from "./viewer.mjs";
import { FrameBuffer, SequentialClock } from "./playback.mjs";
import { fillIcons, icon } from "./icons.mjs";
const $ = (id) => document.getElementById(id);
fillIcons();
const state = {
  catalog: null,
  dataset: "",
  scene: null,
  frame: 0,
  total: 0,
  page: 0,
  sceneTotal: 0,
  sceneToken: 0,
  listToken: 0,
  catalogToken: 0,
  frameToken: 0,
  checkpoint: "",
  prediction: new Map(),
  buffer: null,
  retiredBuffers: [],
  playerDataset: "",
  playToken: 0,
  busyToken: null,
  paintTimes: [],
  running: false,
  playing: false,
  timer: null,
};
const viewer = new ImageViewer(
  $("view"),
  (scale) => ($("zoom").textContent = `${Math.round(scale * 100)}%`),
);
const query = (values) =>
  new URLSearchParams(
    Object.entries(values).filter(([, v]) => v !== null && v !== undefined),
  ).toString();
function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}
async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail),
    );
  return data;
}
function showWarnings(messages) {
  $("warnings").replaceChildren(
    ...messages.map((text) => {
      const p = document.createElement("p");
      p.textContent = text;
      return p;
    }),
  );
}
function checkpoint() {
  return state.catalog?.checkpoints.find((c) => c.id === state.checkpoint);
}
function updateRun() {
  $("infer").disabled =
    state.running ||
    !state.scene ||
    !checkpoint() ||
    Boolean(checkpoint()?.error);
  $("infer-label").textContent = state.running ? "推論中..." : "推論を実行";
  for (const id of ["start", "count", "threshold", "device"])
    $(id).disabled = state.running;
}
function stop() {
  state.playing = false;
  cancelAnimationFrame(state.timer);
  state.playToken++;
  state.frameToken++;
  viewer.token++;
  $("buffer-status").textContent = "";
  $("play").innerHTML = icon("play");
}
function resetScene() {
  stop();
  state.sceneToken++;
  state.frameToken++;
  state.scene = null;
  state.total = 0;
  state.frame = 0;
  state.resultWarnings = [];
  viewer.clear();
  for (const key of ["scene", "frame", "people", "playerMode"])
    delete $("view").dataset[key];
  state.buffer?.dispose();
  for (const buffer of state.retiredBuffers) buffer.dispose();
  state.retiredBuffers = [];
  state.buffer = null;
  $("player-status").textContent = "";
  $("actual-fps").textContent = "実測 — fps";
  state.prediction.clear();
  $("empty").hidden = false;
  $("frame-name").textContent = "";
  $("scene-title").textContent = "未選択";
  $("resolution").textContent = "";
  $("seek").max = "0";
  $("seek").value = "0";
  $("frame-position").textContent = "0 / 0";
  $("metrics").replaceChildren();
  updateRun();
}
function renderCheckpoints() {
  const term = $("checkpoint-search").value.toLowerCase();
  const select = $("checkpoint");
  select.replaceChildren(new Option("未選択", ""));
  for (const item of state.catalog.checkpoints) {
    if (
      item.id !== state.checkpoint &&
      !`${item.label} ${item.path}`.toLowerCase().includes(term)
    )
      continue;
    const opt = new Option(
      `${item.error ? "[非対応] " : ""}${item.label}`,
      item.id,
    );
    opt.disabled = Boolean(item.error);
    opt.title = item.error || item.path;
    select.append(opt);
  }
  select.value = state.checkpoint;
  const rejected = state.catalog.checkpoints.filter(
    (c) => c.error && `${c.label} ${c.path}`.toLowerCase().includes(term),
  );
  $("checkpoint-errors").hidden = !rejected.length;
  $("checkpoint-errors")
    .querySelector("div")
    .replaceChildren(
      ...rejected.map((c) => {
        const p = document.createElement("p");
        p.className = "muted";
        p.textContent = `${c.label}: ${c.error}`;
        return p;
      }),
    );
}
function renderDatasets() {
  const cp = checkpoint();
  const items = state.catalog.datasets.filter(
    (d) => !cp || cp.compatible_datasets.includes(d.id),
  );
  const previous = state.dataset;
  if (!items.some((d) => d.id === state.dataset && d.available))
    state.dataset = items.find((d) => d.available)?.id || "";
  if (state.dataset !== previous) {
    state.page = 0;
    resetScene();
  }
  $("datasets").replaceChildren(
    ...items.map((d) => {
      const b = document.createElement("button");
      b.className = `dataset ${d.id === state.dataset ? "selected" : ""}`;
      b.disabled = !d.available;
      b.title = d.reason || d.path;
      b.innerHTML = icon("folder");
      const span = document.createElement("span");
      span.className = "name";
      span.textContent = d.label;
      const count = document.createElement("span");
      count.className = "count";
      count.textContent = d.available ? String(d.count ?? "") : "未配置";
      b.append(span, count);
      b.onclick = () => {
        state.dataset = d.id;
        state.page = 0;
        resetScene();
        renderDatasets();
        loadScenes();
      };
      return b;
    }),
  );
  renderPlayerDatasets();
  $("dataset-label").textContent =
    items.find((d) => d.id === state.dataset)?.label || "該当データセットなし";
}
function renderPlayerDatasets() {
  const items = (state.catalog.player_datasets || []).filter(
    (d) =>
      d.ball_version === state.dataset.replace(/^store\//, "") || !d.available,
  );
  $("player-settings").hidden = state.catalog.task !== "ball_detection";
  if (!items.some((d) => d.id === state.playerDataset && d.available))
    state.playerDataset = items.find((d) => d.available)?.id || "";
  $("player-dataset").replaceChildren(
    new Option("なし", ""),
    ...items.map((d) => {
      const option = new Option(
        `${d.available ? "" : "[利用不可] "}${d.label}`,
        d.id,
      );
      option.disabled = !d.available;
      option.title = d.error || d.label;
      return option;
    }),
  );
  $("player-dataset").value = state.playerDataset;
  const failures = items.filter((d) => !d.available);
  $("player-dataset-errors").hidden = !failures.length;
  $("player-dataset-errors").textContent = failures
    .map((d) => `${d.label}: ${d.error}`)
    .join("\n");
  $("player-status-filter").hidden = !state.playerDataset;
  if (!state.playerDataset) $("player-status-filter").value = "";
}
function makeBuffer() {
  return new FrameBuffer({
    scene: state.scene.id,
    total: state.total,
    playerDataset: state.playerDataset,
    mode: $("player-mode").value,
    json: api,
  });
}
function reportPlayers(payload, people) {
  if (!payload) {
    $("player-status").textContent = "Playerデータセット未選択";
    return;
  }
  const kind =
    payload.mode === "reviewed" ? "採用結果" : "生成結果（全人物・raw ID）";
  $("player-status").textContent =
    `${payload.label} · ${kind} · ${payload.available ? `${people.length}人観測` : "この結果は未提供"}`;
}
async function changePlayerSource() {
  if (!state.scene) return;
  stop();
  if (state.buffer) {
    state.buffer.retire();
    state.retiredBuffers.push(state.buffer);
  }
  state.buffer = makeBuffer();
  viewer.people = [];
  viewer.draw();
  await showFrame(state.frame);
}
async function loadCatalog() {
  const token = ++state.catalogToken;
  // カタログ更新中は古い一覧応答を破棄し、選択中sceneを一度リセットする。
  state.listToken++;
  clearTimeout(searchTimer);
  resetScene();
  status("カタログを読み込み中...");
  try {
    const catalog = await api("/api/catalog");
    if (token !== state.catalogToken) return;
    // Discard selections and requests made while this catalog was loading.
    state.listToken++;
    clearTimeout(searchTimer);
    resetScene();
    state.catalog = catalog;
    viewer.configure({ ballPoints: catalog.task === "ball_detection" });
    document.title = catalog.title;
    $("title").textContent = catalog.title;
    $("mode").textContent =
      catalog.mode === "review" ? "Dataset Review" : "Inference";
    const review = catalog.mode === "review";
    for (const id of [
      "checkpoint-section",
      "inference-settings",
      "pred-toggle",
      "pred-legend",
    ])
      $(id).hidden = review;
    $("device-status").textContent = catalog.cuda_available
      ? "CUDA available"
      : "CPU only";
    if (!catalog.checkpoints.some((c) => c.id === state.checkpoint && !c.error))
      state.checkpoint = "";
    renderCheckpoints();
    renderDatasets();
    showWarnings(catalog.warnings || []);
    await loadScenes();
    status("カタログ更新完了");
  } catch (error) {
    if (token === state.catalogToken) status(error.message, true);
  }
}
async function loadScenes() {
  const token = ++state.listToken;
  const catalogToken = state.catalogToken;
  const dataset = state.dataset;
  if (!dataset) {
    $("scenes").replaceChildren();
    $("scene-count").textContent = "0";
    return;
  }
  try {
    const result = await api(
      `/api/scenes?${query({ dataset, search: $("scene-search").value, offset: state.page * 100, limit: 100, checkpoint: state.checkpoint || null, player_dataset: state.playerDataset || null, player_status: $("player-status-filter").value || null })}`,
    );
    if (
      token !== state.listToken ||
      dataset !== state.dataset ||
      catalogToken !== state.catalogToken
    )
      return;
    state.sceneTotal = result.total;
    $("scene-count").textContent = result.total.toLocaleString();
    $("page-label").textContent =
      `${state.page + 1} / ${Math.max(1, Math.ceil(result.total / 100))}`;
    $("page-prev").disabled = state.page === 0;
    $("page-next").disabled = (state.page + 1) * 100 >= result.total;
    $("scenes").replaceChildren(
      ...result.items.map((item) => {
        const b = document.createElement("button");
        b.className = `scene ${item.id === state.scene?.id ? "selected" : ""}`;
        b.dataset.scene = item.id;
        const name = document.createElement("span");
        name.textContent = item.label;
        const meta = document.createElement("small");
        meta.textContent = `${item.frames} frame${item.frames === 1 ? "" : "s"}${item.player_status ? ` · ${item.player_status.label}` : ""}`;
        b.append(name, meta);
        b.onclick = () => selectScene(item);
        return b;
      }),
    );
    if (!state.scene && result.items.length) await selectScene(result.items[0]);
  } catch (error) {
    if (token === state.listToken) status(error.message, true);
  }
}
async function selectScene(item) {
  resetScene();
  state.scene = item;
  state.total = item.frames;
  state.frame = 0;
  state.buffer = makeBuffer();
  $("scene-title").textContent = item.label;
  $("start").value = "0";
  $("start").max = String(Math.max(0, item.frames - 1));
  $("seek").max = String(Math.max(0, item.frames - 1));
  configureWindow();
  for (const b of $("scenes").children)
    b.classList.toggle("selected", b.dataset.scene === item.id);
  updateRun();
  await showFrame(0, true);
}
function configureWindow() {
  const cp = checkpoint();
  if (!cp || !state.scene) return;
  const dataset = state.catalog.datasets.find((d) => d.id === state.dataset);
  const staticFrame = dataset?.mode === "static";
  const min = staticFrame ? cp.settings.count : (cp.window?.min ?? 1);
  const max = staticFrame
    ? cp.settings.count
    : Math.min(cp.window?.max ?? cp.settings.count, state.total);
  $("count").min = String(min);
  $("count").max = String(Math.min(64, max));
  $("count").value = String(
    Math.max(min, Math.min(max, Number($("count").value))),
  );
}
function renderLayers(gt, pred) {
  const previous = $("raster").value;
  const rasters = [...(gt?.rasters || []), ...(pred?.rasters || [])];
  const names = [...new Set(rasters.map((r) => r.name))];
  const signature = JSON.stringify(
    rasters.map((r) => [r.name, r.legend || []]),
  );
  if ($("raster").dataset.signature === signature) return;
  $("raster").dataset.signature = signature;
  $("raster").replaceChildren(
    new Option("なし", ""),
    ...names.map((name) => new Option(name, name)),
  );
  $("raster").value = names.includes(previous) ? previous : "";
  viewer.configure({ raster: $("raster").value });
  renderRasterLegend(rasters);
}
function renderRasterLegend(rasters) {
  const r = rasters.find((r) => r.name === $("raster").value);
  $("raster-legend").replaceChildren(
    ...(r?.legend || []).map((entry) => {
      const span = document.createElement("span");
      const color = document.createElement("i");
      color.style.background = entry.color;
      span.append(color, document.createTextNode(entry.label));
      return span;
    }),
  );
}
async function showFrame(frame, reset = false) {
  if (!state.scene || !state.buffer) return false;
  frame = Math.max(0, Math.min(state.total - 1, frame));
  const token = ++state.frameToken;
  viewer.token++;
  const selection = state.sceneToken;
  const buffer = state.buffer;
  if (!buffer.ready(frame))
    $("buffer-status").textContent = "読み込み待ち（全フレーム表示）";
  try {
    const { image, item, preview, players, people } = await buffer.get(frame);
    if (selection !== state.sceneToken || token !== state.frameToken)
      return false;
    const pred = state.prediction.get(frame)?.pred;
    const applied = await viewer.setFrame(image, item.gt, pred, reset, people);
    if (
      !applied ||
      selection !== state.sceneToken ||
      token !== state.frameToken
    )
      return false;
    state.frame = frame;
    buffer.pin(frame);
    for (const retired of state.retiredBuffers) retired.dispose();
    state.retiredBuffers = [];
    $("seek").value = String(frame);
    $("frame-position").textContent = `${frame + 1} / ${state.total}`;
    $("empty").hidden = true;
    $("frame-name").textContent = item.name;
    $("resolution").textContent = `${preview.width} × ${preview.height}`;
    $("buffer-status").textContent = "";
    renderLayers(item.gt, pred);
    reportPlayers(players, people);
    showWarnings([
      ...(item.warnings || preview.warnings || []),
      ...(state.resultWarnings || []),
    ]);
    status(
      pred
        ? "GT + Prediction"
        : state.prediction.size
          ? "GT / このフレームは推論範囲外"
          : "Ground Truth",
    );
    // Committed frame identity is also used by browser regression/performance tests.
    $("view").dataset.frame = String(frame);
    $("view").dataset.scene = state.scene.id;
    $("view").dataset.people = String(people.length);
    $("view").dataset.playerMode = players?.mode || "none";
    $("view").dispatchEvent(
      new CustomEvent("frame-presented", {
        detail: { scene: state.scene.id, frame, at: performance.now() },
      }),
    );
    return true;
  } catch (error) {
    if (token === state.frameToken) {
      status(error.message, true);
      stop();
    }
    return false;
  }
}
async function playTick(now) {
  if (!state.playing) return;
  state.timer = requestAnimationFrame(playTick);
  if (now - state.lastFpsUpdate >= 250) {
    const since = Math.max(state.playStarted, now - 2000);
    state.paintTimes = state.paintTimes.filter((time) => time >= since);
    $("actual-fps").textContent =
      `実測 ${((state.paintTimes.length * 1000) / (now - since)).toFixed(1)} fps`;
    state.lastFpsUpdate = now;
  }
  const token = state.playToken;
  if (state.busyToken === token || !state.clock.due(now)) return;
  state.busyToken = token;
  try {
    const applied = await showFrame((state.frame + 1) % state.total);
    if (!applied || !state.playing || token !== state.playToken) return;
    const painted = performance.now();
    state.clock.commit(painted);
    state.paintTimes.push(painted);
  } finally {
    if (state.busyToken === token) state.busyToken = null;
  }
}
function renderMetrics(metrics) {
  $("metrics").title = typeof metrics?.note === "string" ? metrics.note : "";
  $("metrics").replaceChildren(
    ...Object.entries(metrics || {})
      .filter(([name]) => name !== "note")
      .map(([name, value]) => {
        const row = document.createElement("div");
        const dt = document.createElement("dt");
        dt.textContent = name;
        const dd = document.createElement("dd");
        dd.textContent =
          name === "window" && value && typeof value === "object"
            ? `${value.start}..${value.end} / ${value.count} frames${value.repeat > 1 ? ` (repeat ${value.repeat})` : ""}`
            : value === null
              ? "N/A"
              : typeof value === "number"
                ? Number(value.toFixed(5)).toString()
                : typeof value === "object"
                  ? JSON.stringify(value)
                  : String(value);
        row.append(dt, dd);
        return row;
      }),
  );
}
async function infer() {
  if (state.running || !state.scene || !checkpoint()) return;
  const request = {
    checkpoint: state.checkpoint,
    scene: state.scene.id,
    start: Number($("start").value),
    count: Number($("count").value),
    threshold: Number($("threshold").value),
    device: $("device").value,
  };
  const selection = state.sceneToken;
  const selectedCheckpoint = state.checkpoint;
  state.running = true;
  updateRun();
  $("run-status").textContent =
    request.device === "cuda" ? "GPUキュー待機 / 実行中" : "CPU実行中";
  try {
    const result = await api("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (
      selection !== state.sceneToken ||
      selectedCheckpoint !== state.checkpoint
    ) {
      $("run-status").textContent = "推論完了（選択変更済み）";
      return;
    }
    state.prediction = new Map(result.items.map((i) => [i.index, i]));
    state.resultWarnings = result.warnings || [];
    renderMetrics(result.metrics);
    $("run-status").textContent = `完了 / ${result.items.length} frames`;
    stop();
    await showFrame(result.start);
  } catch (error) {
    $("run-status").textContent = error.message;
    if (selection === state.sceneToken) status(error.message, true);
  } finally {
    state.running = false;
    updateRun();
  }
}
$("checkpoint-search").oninput = renderCheckpoints;
$("checkpoint").onchange = () => {
  state.checkpoint = $("checkpoint").value;
  const cp = checkpoint();
  $("checkpoint-meta").textContent = cp ? `${cp.model} · ${cp.path}` : "";
  if (cp) {
    $("count").value = String(cp.settings.count);
    $("threshold").value = String(cp.settings.threshold);
    $("threshold-value").textContent = Number(cp.settings.threshold).toFixed(2);
  }
  resetScene();
  state.resultWarnings = [];
  state.page = 0;
  renderDatasets();
  loadScenes();
};
let searchTimer;
$("scene-search").oninput = () => {
  clearTimeout(searchTimer);
  state.listToken++;
  searchTimer = setTimeout(() => {
    state.page = 0;
    loadScenes();
  }, 200);
};
$("page-prev").onclick = () => {
  state.page--;
  loadScenes();
};
$("page-next").onclick = () => {
  state.page++;
  loadScenes();
};
// loadCatalog が冒頭で scene をリセットするため、ここでは再読込だけを行う。
$("refresh").onclick = loadCatalog;
$("infer").onclick = infer;
$("first").onclick = () => {
  stop();
  showFrame(0);
};
$("previous").onclick = () => {
  stop();
  showFrame(state.frame - 1);
};
$("next").onclick = () => {
  stop();
  showFrame(state.frame + 1);
};
$("seek").oninput = () => {
  stop();
  showFrame(Number($("seek").value));
};
$("play").onclick = () => {
  if (state.playing) {
    stop();
    return;
  }
  if (state.total < 2) return;
  state.playing = true;
  $("play").innerHTML = icon("pause");
  state.paintTimes = [];
  state.playStarted = state.lastFpsUpdate = performance.now();
  state.clock = new SequentialClock(Number($("fps").value), performance.now());
  state.buffer.prefetch(state.frame);
  state.timer = requestAnimationFrame(playTick);
};
$("zoom-in").onclick = () => viewer.zoom(1.25);
$("zoom-out").onclick = () => viewer.zoom(0.8);
$("fit").onclick = () => viewer.fit();
$("download").onclick = () => {
  if (state.scene) viewer.download(`${state.scene.label}_${state.frame}`);
};
for (const [id, key] of [
  ["show-gt", "gt"],
  ["show-pred", "pred"],
  ["show-labels", "labels"],
  ["show-players", "players"],
  ["show-pose", "pose"],
  ["show-boxes", "boxes"],
  ["show-identities", "identities"],
  ["show-trails", "trails"],
])
  $(id).onchange = () => viewer.configure({ [key]: $(id).checked });
$("raster").onchange = () => {
  viewer.configure({ raster: $("raster").value });
  renderRasterLegend([
    ...(viewer.gt?.rasters || []),
    ...(viewer.pred?.rasters || []),
  ]);
};
$("opacity").oninput = () =>
  viewer.configure({ opacity: Number($("opacity").value) });
$("threshold").oninput = () =>
  ($("threshold-value").textContent = Number($("threshold").value).toFixed(2));
$("player-mode").onchange = changePlayerSource;
$("player-dataset").onchange = () => {
  state.playerDataset = $("player-dataset").value;
  $("player-status-filter").value = "";
  $("player-status-filter").hidden = !state.playerDataset;
  state.page = 0;
  changePlayerSource();
  loadScenes();
};
$("player-status-filter").onchange = () => {
  state.page = 0;
  resetScene();
  loadScenes();
};
$("fps").onchange = () => {
  if (state.playing)
    state.clock = new SequentialClock(
      Number($("fps").value),
      performance.now(),
    );
};
window.addEventListener("pagehide", resetScene);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stop();
});
loadCatalog();
