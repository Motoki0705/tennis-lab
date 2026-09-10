import { SceneView, COLORS } from "./scene.mjs";
const $ = (id) => document.getElementById(id),
  gallery = $("gallery"),
  dialog = $("lightbox");
let scenes = [],
  data = null,
  group = null,
  selectedImage = 0,
  request = null,
  observer = null,
  sceneSequence = 0;
const view = new SceneView($("view"), (id) => selectGroup(id, true));
function status(message) {
  $("status").textContent = message;
  $("status").hidden = !message;
}
function imageURL(sample, width) {
  return `/api/scenes/${encodeURIComponent(data.id)}/images/${encodeURIComponent(sample.id)}?revision=${encodeURIComponent(data.revision)}&width=${width}`;
}
function resizeTiles() {
  gallery.style.setProperty(
    "--tile-height",
    `${Math.max(30, (gallery.clientHeight + 8) / 5.5 - 8)}px`,
  );
}
new ResizeObserver(resizeTiles).observe(gallery);
async function loadScene() {
  if (dialog.open) dialog.close();
  const scene = scenes.find((s) => s.id === $("scene").value);
  const sequence = ++sceneSequence;
  request?.abort();
  request = new AbortController();
  observer?.disconnect();
  gallery.replaceChildren();
  $("trajectories").replaceChildren();
  $("stat-cards").replaceChildren();
  for (const id of ["splits", "shapes", "coverage"]) $(id).replaceChildren();
  $("schema").textContent = "";
  $("image-count").textContent = "";
  $("trajectory-count").textContent = "";
  data = null;
  group = null;
  view.groups = [];
  view.courts = [];
  view.selected = null;
  view.schedule();
  if (!scene) return;
  if (scene.error) {
    status(scene.error);
    return;
  }
  status(`${scene.id} を読み込んでいます…`);
  $("scene-title").textContent = scene.id;
  try {
    const response = await fetch(
      `/api/scenes/${encodeURIComponent(scene.id)}?revision=${scene.revision}`,
      { signal: request.signal, cache: "no-store" },
    );
    const result = await response.json();
    if (!response.ok) throw Error(result.detail || "読み込みに失敗しました");
    if (sequence !== sceneSequence) return;
    data = result;
    view.setScene(data);
    renderList();
    renderStats();
    status("");
    if (data.groups.length) selectGroup(data.groups[0].id);
    else status("表示できる軌道がありません。");
  } catch (error) {
    if (error.name !== "AbortError" && sequence === sceneSequence)
      status(error.message);
  }
}
function renderList() {
  $("trajectory-count").textContent = data.groups.length;
  for (const g of data.groups) {
    const button = document.createElement("button");
    button.className = "trajectory";
    button.dataset.group = g.id;
    button.style.setProperty("--split", COLORS[g.split] || "#fff");
    button.setAttribute("aria-current", "false");
    const name = document.createElement("span");
    name.className = "name";
    const label = document.createElement("span");
    label.textContent = g.trajectory.trajectory_id;
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.textContent = "●";
    dot.title = g.split;
    name.append(label, dot);
    const description = document.createElement("span");
    description.className = "description";
    description.textContent = `${g.trajectory.shape} · ${g.samples.length} views`;
    button.append(name, description);
    button.onclick = () => selectGroup(g.id);
    $("trajectories").append(button);
  }
}
function selectGroup(id, scroll = false) {
  if (!data) return;
  group = data.groups.find((g) => g.id === id);
  if (!group) return;
  view.select(id);
  for (const button of $("trajectories").children) {
    button.setAttribute("aria-current", String(button.dataset.group === id));
    if (scroll && button.dataset.group === id)
      button.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  $("selected-title").textContent = group.trajectory.trajectory_id;
  $("image-count").textContent =
    `${group.samples.length} images · ${group.split}`;
  observer?.disconnect();
  gallery.replaceChildren();
  gallery.scrollTop = 0;
  observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          const img = entry.target;
          img.src = img.dataset.src;
          observer.unobserve(img);
        }
      }
    },
    { root: gallery, rootMargin: "250px" },
  );
  group.samples.forEach((sample, index) => {
    const button = document.createElement("button");
    button.className = "thumbnail";
    button.dataset.sample = sample.id;
    button.setAttribute("aria-label", `${sample.id} を全画面表示`);
    const img = document.createElement("img");
    img.alt = `${sample.id}：画像と教師ラベル`;
    img.decoding = "async";
    img.dataset.src = imageURL(sample, 480);
    img.onerror = () => {
      img.remove();
      const message = document.createElement("span");
      message.className = "failure";
      message.textContent = "画像を読み込めません。クリックして詳細を確認";
      button.append(message);
    };
    button.append(img);
    button.onclick = () => openImage(index);
    button.onmouseenter = () => view.select(group.id, index);
    button.onmouseleave = () =>
      view.select(group.id, dialog.open ? selectedImage : null);
    gallery.append(button);
    observer.observe(img);
  });
  resizeTiles();
}
function openImage(index) {
  if (!group) return;
  selectedImage = index;
  $("large-image").removeAttribute("src");
  $("image-error").hidden = true;
  const sample = group.samples[index];
  $("large-image").alt = `${sample.id}：画像と教師ラベル`;
  $("large-image").src = imageURL(sample, 0);
  $("position").textContent =
    `${index + 1} / ${group.samples.length} · ${sample.id}`;
  $("previous").disabled = index === 0;
  $("next").disabled = index === group.samples.length - 1;
  view.select(group.id, index);
  for (const [i, button] of [...gallery.children].entries())
    button.classList.toggle("active", i === index);
  if (!dialog.open) dialog.showModal();
}
$("large-image").onerror = async () => {
  const src = $("large-image").src;
  let message = "画像を読み込めません。";
  try {
    const response = await fetch(src);
    if (!response.ok) message = (await response.json()).detail || message;
  } catch {}
  if (src === $("large-image").src) {
    $("image-error").textContent = message;
    $("image-error").hidden = false;
  }
};
$("previous").onclick = () => {
  if (selectedImage > 0) openImage(selectedImage - 1);
};
$("next").onclick = () => {
  if (selectedImage < group.samples.length - 1) openImage(selectedImage + 1);
};
$("close").onclick = () => dialog.close();
dialog.addEventListener("close", () => {
  if (group) view.select(group.id);
});
dialog.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    $("previous").click();
  }
  if (e.key === "ArrowRight") {
    e.preventDefault();
    $("next").click();
  }
});
function bars(id, values, colors = {}) {
  const root = $(id);
  root.replaceChildren();
  const max = Math.max(...Object.values(values), 1);
  for (const [name, value] of Object.entries(values)) {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("div");
    label.className = "bar-label";
    const text = document.createElement("span"),
      count = document.createElement("span");
    text.textContent = name;
    count.textContent = Number(value).toLocaleString();
    label.append(text, count);
    const bar = document.createElement("div"),
      fill = document.createElement("i");
    bar.className = "bar";
    fill.style.width = `${(value / max) * 100}%`;
    if (colors[name]) fill.style.background = colors[name];
    bar.append(fill);
    row.append(label, bar);
    root.append(row);
  }
}
function renderStats() {
  const m = data.metrics;
  $("schema").textContent = data.schema;
  const cards = [
    [m.accepted_frame_count.toLocaleString(), "採用画像"],
    [data.groups.length, "カメラ軌道"],
    [data.courts.length, "コート"],
    [`${(m.accepted_fraction * 100).toFixed(1)}%`, "採用率"],
  ];
  $("stat-cards").replaceChildren();
  for (const [value, label] of cards) {
    const card = document.createElement("div");
    card.className = "stat-card";
    const strong = document.createElement("strong"),
      caption = document.createElement("span");
    strong.textContent = value;
    caption.textContent = label;
    card.append(strong, caption);
    $("stat-cards").append(card);
  }
  bars("splits", m.split_frame_counts, COLORS);
  bars("shapes", data.shapes);
  bars("coverage", m.coverage_counts);
}
$("scene").onchange = loadScene;
$("reset").onclick = () => view.reset();
try {
  const response = await fetch("/api/scenes", { cache: "no-store" });
  if (!response.ok) throw Error("シーン一覧の読み込みに失敗しました");
  scenes = await response.json();
  for (const scene of scenes) {
    const option = document.createElement("option");
    option.value = scene.id;
    option.textContent = scene.id;
    $("scene").append(option);
  }
  if (scenes.length) await loadScene();
  else status("公開済みのCourtデータセットがありません。");
} catch (error) {
  status(error.message);
}
