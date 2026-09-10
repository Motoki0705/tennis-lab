import { COURT_CORNERS, resizeFromCorner } from "/static/resize.mjs";

const $ = (id) => document.getElementById(id);
const ns = "http://www.w3.org/2000/svg";
const colors = [
  "#b5f2bf",
  "#72d4ff",
  "#ffd38a",
  "#deaeff",
  "#ffa4b4",
  "#67e8d3",
];
let state,
  layout,
  selected,
  view,
  home,
  drag,
  dirty = false,
  history = [],
  future = [],
  busy = false,
  previewVersion = 0;
const clone = (value) => structuredClone(value);
const node = (tag, attrs = {}) => {
  const el = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  return el;
};
const error = (message) => {
  $("error-text").textContent = message;
  $("error").hidden = false;
};
const toast = (message) => {
  $("toast").textContent = message;
  $("toast").hidden = false;
  setTimeout(() => {
    $("toast").hidden = true;
  }, 4000);
};
const court = () => layout.courts.find((c) => c.court_id === selected);
async function api(path, body) {
  const response = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Editor-Token": state.token,
          },
          body: JSON.stringify(body),
        },
  );
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail),
    );
  return data;
}
function checkpoint() {
  history.push(clone(layout));
  if (history.length > 100) history.shift();
  future = [];
}
function changed() {
  dirty = true;
  $("save-state").textContent = "未保存の変更";
  render();
}
function mutate(fn) {
  if (busy) return;
  checkpoint();
  fn();
  changed();
  updatePreview();
}
function setView() {
  $("canvas").setAttribute(
    "viewBox",
    `${view.x} ${view.y} ${view.w} ${view.h}`,
  );
  $("zoom-level").textContent = `${Math.round((home.w / view.w) * 100)}%`;
  drawCourts();
}
function fit() {
  const [u0, u1, v0, v1] = state.bounds;
  let minU = u0,
    maxU = u1,
    minV = v0,
    maxV = v1;
  for (const c of layout.courts) {
    const r = 14 * layout.scale;
    minU = Math.min(minU, c.u - r);
    maxU = Math.max(maxU, c.u + r);
    minV = Math.min(minV, c.v - r);
    maxV = Math.max(maxV, c.v + r);
  }
  home = { x: minU - 2, y: -maxV - 2, w: maxU - minU + 4, h: maxV - minV + 4 };
  view = { ...home };
  setView();
}
function world(event) {
  const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(
    $("canvas").getScreenCTM().inverse(),
  );
  return { u: p.x, v: -p.y };
}
function zoom(factor, point) {
  const x = point?.u ?? view.x + view.w / 2,
    y = point ? -point.v : view.y + view.h / 2;
  const next = view.w * factor;
  if (next < 0.2 || next > 1e5) return;
  view = {
    x: x + (view.x - x) * factor,
    y: y + (view.y - y) * factor,
    w: view.w * factor,
    h: view.h * factor,
  };
  setView();
}
function drawCourts() {
  const layer = $("court-layer");
  layer.replaceChildren();
  if (!$("show-lines").checked) return;
  const px = $("canvas").getScreenCTM()?.a || 10;
  const handleSize = 6 / px;
  // Selected handles must stay above every other court, even in overlapping layouts.
  const ordered = layout.courts
    .map((c, i) => ({ c, i }))
    .sort(
      (a, b) =>
        Number(a.c.court_id === selected) - Number(b.c.court_id === selected),
    );
  ordered.forEach(({ c, i }) => {
    const color = colors[i % colors.length],
      active = c.court_id === selected;
    const g = node("g", {
      transform: `translate(${c.u},${-c.v}) rotate(${-c.angle_degrees}) scale(${layout.scale})`,
      "data-court": c.court_id,
    });
    g.append(
      node("rect", {
        x: -5.485,
        y: -11.885,
        width: 10.97,
        height: 23.77,
        class: "court-hit",
        "data-action": "move",
      }),
    );
    const d = state.segments
      .map(([a, b]) => `M ${a[0]} ${-a[1]} L ${b[0]} ${-b[1]}`)
      .join(" ");
    const lines = node("g", {
      class: "court-lines",
      opacity: $("court-opacity").value,
    });
    g.append(lines);
    lines.append(
      node("path", {
        d,
        fill: "none",
        stroke: color,
        "stroke-width": active ? 2 : 1.5,
        "vector-effect": "non-scaling-stroke",
        "data-action": "move",
        opacity: active ? 1 : 0.85,
      }),
    );
    // A dashed net and direction marker distinguish the otherwise symmetric ends.
    lines.append(
      node("path", {
        d: "M -5.485 0 H 5.485",
        fill: "none",
        stroke: color,
        "stroke-width": 1,
        "stroke-dasharray": "4 4",
        "vector-effect": "non-scaling-stroke",
        "pointer-events": "none",
      }),
    );
    if (active) {
      g.append(
        node("rect", {
          x: -5.9,
          y: -12.3,
          width: 11.8,
          height: 24.6,
          fill: "none",
          stroke: color,
          "stroke-width": 1,
          "stroke-dasharray": "4 4",
          "vector-effect": "non-scaling-stroke",
          "pointer-events": "none",
        }),
      );
      const r = handleSize / layout.scale;
      g.append(
        node("line", {
          x1: 0,
          y1: -11.885,
          x2: 0,
          y2: -15.5,
          stroke: color,
          "stroke-width": 1,
          "vector-effect": "non-scaling-stroke",
        }),
      );
      g.append(
        node("circle", {
          cx: 0,
          cy: -15.5,
          r,
          fill: color,
          stroke: "#13221a",
          "stroke-width": 1.5,
          "vector-effect": "non-scaling-stroke",
          class: "handle",
          "data-action": "rotate",
        }),
      );
      COURT_CORNERS.forEach(([x, y], index) => {
        const handle = node("rect", {
          x: x - r,
          y: -y - r,
          width: r * 2,
          height: r * 2,
          fill: color,
          stroke: "#13221a",
          "stroke-width": 1.5,
          "vector-effect": "non-scaling-stroke",
          class: "handle",
          "data-action": "scale",
          "data-corner": index,
        });
        const title = node("title");
        title.textContent = "対角の角を固定して拡縮";
        handle.append(title);
        g.append(handle);
      });
      g.append(
        node("circle", {
          cx: 0,
          cy: 0,
          r: r * 0.55,
          fill: color,
          "pointer-events": "none",
        }),
      );
    }
    layer.append(g);
    const label = node("text", {
      x: c.u + handleSize,
      y: -c.v - handleSize,
      fill: color,
      "font-size": 11 / px,
      "font-family": "system-ui",
      "pointer-events": "none",
      "paint-order": "stroke",
      stroke: "#10151b",
      "stroke-width": 3 / px,
    });
    label.textContent = c.court_id;
    layer.append(label);
  });
}
function render() {
  $("court-count").textContent = `${layout.courts.length} 面`;
  $("court-list").replaceChildren();
  layout.courts.forEach((c, i) => {
    const b = document.createElement("button");
    b.className = `court-item ${c.court_id === selected ? "selected" : ""}`;
    b.dataset.courtId = c.court_id;
    const dot = document.createElement("span");
    dot.className = "swatch";
    dot.style.background = colors[i % colors.length];
    b.append(dot, document.createTextNode(c.court_id));
    const desc = document.createElement("small");
    desc.textContent =
      c.court_id === layout.primary_court_id
        ? "基準コート"
        : `${c.angle_degrees.toFixed(1)}°`;
    b.append(desc);
    b.onclick = () => {
      selected = c.court_id;
      render();
    };
    $("court-list").append(b);
  });
  const c = court();
  $("selected-id").textContent = c?.court_id ?? "—";
  for (const [id, key] of [
    ["u", "u"],
    ["v", "v"],
    ["angle", "angle_degrees"],
  ]) {
    $(id).value = c ? Number(c[key].toFixed(4)) : "";
    $(id).disabled = !c || busy;
  }
  $("primary-court").checked = !!c && c.court_id === layout.primary_court_id;
  $("primary-court").disabled = !c || busy;
  $("scale").value = Number(layout.scale.toFixed(6));
  $("duplicate").disabled = !c || busy;
  $("delete").disabled = !c || busy;
  $("rotate90").disabled = !c || busy;
  $("undo").disabled = !history.length || busy;
  $("redo").disabled = !future.length || busy;
  $("apply").disabled = !layout.courts.length || busy;
  for (const id of ["save", "add", "reset", "scale", "scale-down", "scale-up"])
    $(id).disabled = busy;
  $("selection-hint").textContent = c
    ? `${c.court_id} · U ${c.u.toFixed(2)} / V ${c.v.toFixed(2)} · ${c.angle_degrees.toFixed(1)}°`
    : "「＋ 追加」でコートを配置できます";
  drawCourts();
}
function newId() {
  let n = 1;
  while (
    layout.courts.some(
      (c) => c.court_id === `court-${String(n).padStart(3, "0")}`,
    )
  )
    n++;
  return `court-${String(n).padStart(3, "0")}`;
}
function restore(isRedo = false) {
  if (busy) return;
  const from = isRedo ? future : history,
    to = isRedo ? history : future;
  if (!from.length) return;
  to.push(clone(layout));
  layout = from.pop();
  selected = layout.courts.some((c) => c.court_id === selected)
    ? selected
    : layout.courts[0]?.court_id;
  changed();
  updatePreview();
}
$("undo").onclick = () => restore();
$("redo").onclick = () => restore(true);
$("add").onclick = () =>
  mutate(() => {
    const angle = court()?.angle_degrees ?? 0;
    selected = newId();
    layout.courts.push({
      court_id: selected,
      u: view.x + view.w / 2,
      v: -(view.y + view.h / 2),
      angle_degrees: angle,
    });
    if (layout.courts.length === 1) layout.primary_court_id = selected;
  });
$("duplicate").onclick = () =>
  mutate(() => {
    const c = clone(court());
    c.court_id = newId();
    c.u += 12 * layout.scale;
    layout.courts.push(c);
    selected = c.court_id;
  });
$("delete").onclick = () => {
  if (!court()) return;
  mutate(() => {
    layout.courts = layout.courts.filter((c) => c.court_id !== selected);
    if (layout.primary_court_id === selected) layout.primary_court_id = null;
    selected = layout.courts[0]?.court_id;
  });
};
for (const [id, key] of [
  ["u", "u"],
  ["v", "v"],
  ["angle", "angle_degrees"],
])
  $(id).onchange = () => {
    const value = Number($(id).value);
    if (!$(id).value || !Number.isFinite(value)) {
      render();
      return;
    }
    mutate(() => (court()[key] = value));
  };
$("rotate90").onclick = () => mutate(() => (court().angle_degrees += 90));
$("primary-court").onchange = () => {
  const checked = $("primary-court").checked;
  mutate(() => (layout.primary_court_id = checked ? selected : null));
};
$("scale").onchange = () => {
  const value = Number($("scale").value);
  if (!Number.isFinite(value) || value <= 0) {
    error("共通スケールには正の数を指定してください。");
    render();
    return;
  }
  mutate(() => (layout.scale = value));
};
$("scale-down").onclick = () => mutate(() => (layout.scale /= 1.02));
$("scale-up").onclick = () => mutate(() => (layout.scale *= 1.02));
$("fit").onclick = fit;
$("zoom-in").onclick = () => zoom(0.8);
$("zoom-out").onclick = () => zoom(1.25);
$("show-lines").onchange = drawCourts;
$("opacity").oninput = () => {
  $("heatmap").style.opacity = $("opacity").value;
};
function updateCourtOpacity() {
  const opacity = $("court-opacity").value;
  $("court-opacity-value").textContent =
    `${Math.round(Number(opacity) * 100)}%`;
  document
    .querySelectorAll(".court-lines")
    .forEach((lines) => lines.setAttribute("opacity", opacity));
  $("preview-lines").style.opacity = opacity;
}
$("court-opacity").oninput = updateCourtOpacity;
$("reset").onclick = () =>
  mutate(() => {
    layout = clone(state.initial_layout);
    selected = layout.courts[0]?.court_id;
    fit();
  });
$("dismiss-error").onclick = () => {
  $("error").hidden = true;
};
$("canvas").addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    zoom(Math.exp(e.deltaY * 0.001), world(e));
  },
  { passive: false },
);
$("canvas").addEventListener("pointerdown", (e) => {
  if (busy || e.button !== 0) return;
  const target = e.target.closest("[data-court]"),
    point = world(e);
  $("canvas").focus();
  if (target && !e.shiftKey) {
    selected = target.dataset.court;
    checkpoint();
    drag = {
      action: e.target.dataset.action ?? "move",
      start: point,
      court: clone(court()),
      scale: layout.scale,
      corner: COURT_CORNERS[Number(e.target.dataset.corner)],
    };
    render();
  } else
    drag = {
      action: "pan",
      clientX: e.clientX,
      clientY: e.clientY,
      view: { ...view },
      matrix: $("canvas").getScreenCTM().inverse(),
    };
  $("canvas").setPointerCapture(e.pointerId);
  e.preventDefault();
});
$("canvas").addEventListener("pointermove", (e) => {
  if (!drag) return;
  if (drag.action === "pan") {
    const m = drag.matrix;
    view = {
      ...drag.view,
      x: drag.view.x - (e.clientX - drag.clientX) * m.a,
      y: drag.view.y - (e.clientY - drag.clientY) * m.d,
    };
    setView();
    return;
  }
  const p = world(e),
    c = court();
  if (drag.action === "move") {
    c.u = drag.court.u + p.u - drag.start.u;
    c.v = drag.court.v + p.v - drag.start.v;
  }
  if (drag.action === "rotate") {
    c.angle_degrees = (Math.atan2(p.v - c.v, p.u - c.u) * 180) / Math.PI - 90;
    if (e.shiftKey) c.angle_degrees = Math.round(c.angle_degrees / 5) * 5;
  }
  if (drag.action === "scale") {
    const resized = resizeFromCorner(drag.court, drag.scale, drag.corner, {
      u: p.u - drag.start.u,
      v: p.v - drag.start.v,
    });
    layout.scale = resized.scale;
    c.u = resized.u;
    c.v = resized.v;
  }
  changed();
});
function endDrag() {
  if (!drag) return;
  const edit = drag.action !== "pan";
  drag = null;
  if (edit) updatePreview();
}
$("canvas").addEventListener("pointerup", endDrag);
$("canvas").addEventListener("pointercancel", endDrag);
document.addEventListener("keydown", (e) => {
  if (
    e.target.matches("input,select,textarea") ||
    $("confirm-dialog").open ||
    busy
  )
    return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
    e.preventDefault();
    restore(e.shiftKey);
    return;
  }
  if (e.key === "Delete" || e.key === "Backspace") {
    e.preventDefault();
    $("delete").click();
    return;
  }
  if (
    court() &&
    ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)
  ) {
    e.preventDefault();
    const step = (e.shiftKey ? 1 : 0.05) * layout.scale;
    mutate(() => {
      const c = court();
      if (e.key === "ArrowUp") c.v += step;
      if (e.key === "ArrowDown") c.v -= step;
      if (e.key === "ArrowLeft") c.u -= step;
      if (e.key === "ArrowRight") c.u += step;
    });
  }
});
window.addEventListener("beforeunload", (e) => {
  if (dirty) {
    e.preventDefault();
    e.returnValue = "";
  }
});
$("save").onclick = async () => {
  busy = true;
  render();
  try {
    await api("/api/draft", { revision: state.revision, layout });
    dirty = false;
    $("save-state").textContent = "下書き保存済み";
    toast("下書きを保存しました");
  } catch (e) {
    error(e.message);
  } finally {
    busy = false;
    render();
  }
};
$("apply").onclick = () => {
  $("confirm-description").textContent =
    `${state.scene_id} · ${layout.courts.length} 面 · 共通スケール ${layout.scale.toFixed(4)}`;
  $("confirm-dialog").showModal();
};
$("cancel-apply").onclick = () => {
  $("confirm-dialog").close();
};
$("confirm-apply").onclick = async () => {
  $("confirm-dialog").close();
  busy = true;
  render();
  $("save-state").textContent = "検証・適用中…";
  try {
    const result = await api("/api/apply", {
      revision: state.revision,
      layout,
      human_confirmed: true,
    });
    state.revision = result.revision;
    dirty = false;
    $("save-state").textContent = "手動確定・適用済み";
    toast(result.message);
  } catch (e) {
    error(e.message);
    $("save-state").textContent = "適用できませんでした";
  } finally {
    busy = false;
    render();
  }
};
async function updatePreview() {
  const version = ++previewVersion;
  if (!layout.courts.length) {
    $("preview-lines").replaceChildren();
    $("metrics").replaceChildren();
    $("preview-status").textContent = "コートを追加すると投影を表示します。";
    return;
  }
  $("preview-status").textContent = "投影を確認中…";
  $("preview-lines").replaceChildren();
  try {
    const index = Number($("camera").value),
      c = state.cameras[index];
    $("preview").setAttribute("viewBox", `0 0 ${c.width} ${c.height}`);
    $("camera-image").setAttribute("href", `/api/cameras/${index}/image`);
    const data = await api("/api/preview", { layout, camera_index: index });
    if (version !== previewVersion) return;
    $("preview-lines").replaceChildren();
    data.courts.forEach((court, i) => {
      $("preview-lines").append(
        node("path", {
          d: court.segments
            .map(([a, b]) => `M ${a[0]} ${a[1]} L ${b[0]} ${b[1]}`)
            .join(" "),
          stroke: colors[i % colors.length],
          fill: "none",
          "stroke-width": 1.5,
          "vector-effect": "non-scaling-stroke",
        }),
      );
    });
    $("preview-status").textContent = "元画像への3D投影 · 評価値は参考情報です";
    $("metrics").replaceChildren();
    for (const d of data.diagnostics) {
      const el = document.createElement("span");
      el.className = "metric";
      el.textContent = `${d.court_id} · 線までの距離 q95 ${d.fit.metrics.q95_error_m.toFixed(2)} m`;
      $("metrics").append(el);
    }
  } catch (e) {
    if (version === previewVersion) {
      $("preview-status").textContent = "投影に失敗しました";
      error(e.message);
    }
  }
}
$("camera").onchange = updatePreview;
$("refresh-preview").onclick = updatePreview;
async function init() {
  state = await api("/api/state");
  layout = clone(state.layout);
  selected = layout.courts[0]?.court_id;
  $("scene").textContent = state.scene_id;
  document.title = `${state.scene_id} · Court Alignment Studio`;
  $("save-state").textContent = state.has_draft
    ? "保存した下書きを復元"
    : state.applied_manually
      ? "手動確定・適用済み"
      : "自動配置から編集";
  if (state.stale_draft)
    error(
      "別の配置に対する古い下書きが残っています。現在の配置を読み込みました。",
    );
  const [u0, , v0] = state.bounds,
    [height, width] = state.raster_shape,
    spacing = state.grid_spacing;
  for (const [key, value] of Object.entries({
    x: u0 - spacing / 2,
    y: -(v0 + (height - 0.5) * spacing),
    width: width * spacing,
    height: height * spacing,
  }))
    $("heatmap").setAttribute(key, value);
  for (const [key, value] of Object.entries({
    x: -1e6,
    y: -1e6,
    width: 2e6,
    height: 2e6,
  }))
    $("grid-bg").setAttribute(key, value);
  for (const c of state.cameras) {
    const option = document.createElement("option");
    option.value = c.index;
    option.textContent = `${c.camera_id}`;
    $("camera").append(option);
  }
  fit();
  render();
  updateCourtOpacity();
  await updatePreview();
  new ResizeObserver(drawCourts).observe($("canvas-wrap"));
}
init().catch((e) => {
  error(e.message);
  $("save-state").textContent = "読み込み失敗";
});
