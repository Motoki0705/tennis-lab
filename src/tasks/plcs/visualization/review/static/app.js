// Browser wiring for the ACCAD motion review UI: catalog search on the left,
// world-coordinate playback of the selected motion on the right.

import { MotionView } from "./scene.mjs";

const PLAY_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><polygon points="7,4 20,12 7,20" fill="currentColor" stroke="none" /></svg>';
const PAUSE_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="4" width="4" height="16" fill="currentColor" stroke="none" /><rect x="14" y="4" width="4" height="16" fill="currentColor" stroke="none" /></svg>';

const byId = (id) => document.getElementById(id);
const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
if (!littleEndian) {
  throw new Error("This viewer requires a little-endian platform for joint buffers.");
}

const dom = {
  query: byId("query"),
  clear: byId("clear"),
  tree: byId("tree"),
  resultCount: byId("result-count"),
  dataRoot: byId("data-root"),
  subject: byId("motion-subject"),
  title: byId("motion-title"),
  chips: byId("motion-chips"),
  status: byId("status"),
  hud: byId("hud"),
  hudFrame: byId("hud-frame"),
  hudTime: byId("hud-time"),
  hudX: byId("hud-x"),
  hudY: byId("hud-y"),
  hudZ: byId("hud-z"),
  transport: byId("transport"),
  play: byId("play"),
  prev: byId("prev"),
  next: byId("next"),
  toStart: byId("to-start"),
  toEnd: byId("to-end"),
  scrub: byId("scrub"),
  clock: byId("clock"),
  follow: byId("follow"),
  trail: byId("trail"),
  reset: byId("reset"),
  speeds: Array.from(document.querySelectorAll(".speeds button")),
};

const state = {
  catalog: null,
  query: "",
  selection: null,
  meta: null,
  points: null,
  frame: 0,
  phase: 0,
  playing: false,
  speed: 1,
  token: 0,
  lastTime: 0,
};

const view = new MotionView(byId("view"));

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

function renderTree() {
  const catalog = state.catalog;
  dom.tree.textContent = "";
  if (!catalog) {
    dom.resultCount.textContent = "";
    return;
  }
  let visible = 0;
  for (const subject of catalog.subjects) {
    const motions = subject.motions.filter((motion) =>
      matches(`${subject.id} ${motion.name} ${motion.gender}`, state.query),
    );
    if (motions.length === 0) {
      continue;
    }
    visible += motions.length;
    const group = document.createElement("details");
    group.className = "subject";
    group.open = true;
    const summary = document.createElement("summary");
    const name = document.createElement("span");
    name.className = "subject-name";
    name.textContent = subject.id;
    const count = document.createElement("span");
    count.className = "subject-count";
    count.textContent = String(motions.length);
    summary.append(name, count);
    group.append(summary);
    for (const motion of motions) {
      group.append(motionButton(subject.id, motion));
    }
    dom.tree.append(group);
  }
  dom.resultCount.textContent = state.query
    ? `${visible} 件が「${state.query}」に一致`
    : `${visible} モーション / ${catalog.subjects.length} ディレクトリ`;
  if (visible === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "一致するモーションがありません";
    dom.tree.append(empty);
  }
}

function motionButton(subjectId, motion) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "motion";
  button.dataset.subject = subjectId;
  button.dataset.motion = motion.id;
  const selected =
    state.selection?.subject === subjectId && state.selection?.motion === motion.id;
  if (selected) {
    button.setAttribute("aria-current", "true");
  }
  const name = document.createElement("span");
  name.className = "name";
  name.textContent = motion.name;
  const meta = document.createElement("span");
  meta.className = "meta";
  meta.textContent = `${motion.duration_s.toFixed(1)}s · ${Math.round(motion.fps)}fps · ${motion.frame_count}f`;
  button.append(name, meta);
  return button;
}

// ------------------------------------------------------------------ motion

async function selectMotion(subject, motion) {
  state.selection = { subject, motion };
  state.playing = false;
  syncPlayButton();
  renderTree();
  const token = (state.token += 1);
  dom.title.textContent = motion.replace(/_poses\.npz$/, "");
  dom.subject.textContent = subject;
  setStatus("読み込み中…", "info");
  try {
    const meta = await fetchJson(`/api/motions/${encode(subject)}/${encode(motion)}`);
    if (token !== state.token) return;
    const response = await fetch(
      `/api/motions/${encode(subject)}/${encode(motion)}/joints?revision=${encode(meta.revision)}`,
    );
    if (!response.ok) {
      throw new Error(await describeError(response));
    }
    const buffer = await response.arrayBuffer();
    if (token !== state.token) return;
    const jointCount = meta.joints.count;
    const frameCount = meta.sampled_frame_count;
    const expected = frameCount * jointCount * 3 * 4;
    if (buffer.byteLength !== expected) {
      throw new Error(
        `joint buffer is ${buffer.byteLength} bytes, expected ${expected}`,
      );
    }
    state.meta = meta;
    state.points = new Float32Array(buffer);
    state.frame = 0;
    state.phase = 0;
    view.setMotion({
      joints: state.points,
      frameCount,
      jointCount,
      names: meta.joints.names,
      edges: meta.joints.edges,
    });
    dom.scrub.max = String(frameCount - 1);
    dom.transport.hidden = false;
    dom.hud.hidden = false;
    renderChips(meta);
    applyFrame(0);
    setStatus("");
    state.playing = true;
    syncPlayButton();
  } catch (error) {
    if (token !== state.token) return;
    setStatus(`読み込みに失敗しました: ${error.message}`, "error");
    dom.transport.hidden = true;
    dom.hud.hidden = true;
  }
}

function renderChips(meta) {
  const chips = [
    ["gender", meta.gender],
    ["fps", Math.round(meta.fps)],
    ["frames", meta.sampled_frame_count],
    ["span", formatSeconds(meta.duration_s)],
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

function applyFrame(frame) {
  const meta = state.meta;
  if (!meta) return;
  const total = meta.sampled_frame_count;
  state.frame = Math.max(0, Math.min(total - 1, Math.round(frame)));
  state.phase = state.frame;
  view.setFrame(state.frame);
  dom.scrub.value = String(state.frame);
  dom.clock.textContent = `${state.frame + 1} / ${total}`;
  dom.hudFrame.textContent = String(state.frame);
  dom.hudTime.textContent = formatSeconds(state.frame / meta.fps);
  const root = rootPosition(state.frame);
  dom.hudX.textContent = `${root[0].toFixed(2)} m`;
  dom.hudY.textContent = `${root[1].toFixed(2)} m`;
  dom.hudZ.textContent = `${root[2].toFixed(2)} m`;
}

function rootPosition(frame) {
  const offset = frame * state.meta.joints.count * 3;
  const points = state.points;
  return [points[offset], points[offset + 1], points[offset + 2]];
}

function setPlaying(playing) {
  state.playing = playing && Boolean(state.meta);
  syncPlayButton();
}

function syncPlayButton() {
  dom.play.innerHTML = state.playing ? PAUSE_ICON : PLAY_ICON;
  dom.play.title = state.playing ? "一時停止" : "再生";
}

// ------------------------------------------------------------------- frame

function loop(now) {
  const delta = state.lastTime ? (now - state.lastTime) / 1000 : 0;
  state.lastTime = now;
  if (state.playing && state.meta) {
    const total = state.meta.sampled_frame_count;
    state.phase += delta * state.meta.fps * state.speed;
    if (state.phase >= total) {
      state.phase -= total;
    }
    const target = Math.floor(state.phase);
    if (target !== state.frame) {
      state.frame = target;
      applyFrame(target);
    }
  }
  view.updateFollow();
  view.draw();
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
  dom.dataRoot.textContent = state.catalog.root;
  dom.dataRoot.title = state.catalog.root;
  renderTree();
  const first = state.catalog.subjects[0]?.motions[0];
  if (first) {
    await selectMotion(state.catalog.subjects[0].id, first.id);
  }
}

dom.query.addEventListener("input", () => {
  state.query = dom.query.value.trim();
  dom.clear.hidden = state.query.length === 0;
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
  const button = event.target.closest(".motion");
  if (!button) return;
  selectMotion(button.dataset.subject, button.dataset.motion);
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
  applyFrame(state.meta ? state.meta.sampled_frame_count - 1 : 0);
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
dom.follow.addEventListener("click", () => {
  view.setFollow(!view.follow);
  dom.follow.setAttribute("aria-pressed", String(view.follow));
});
dom.trail.addEventListener("click", () => {
  view.showTrail = !view.showTrail;
  dom.trail.setAttribute("aria-pressed", String(view.showTrail));
  view.dirty = true;
});
dom.reset.addEventListener("click", () => {
  view.resetView();
  view.draw();
});
view.onFollowChange = (on) => dom.follow.setAttribute("aria-pressed", String(on));

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
    view.draw();
  }
});

syncPlayButton();
requestAnimationFrame(loop);
init();
