// Annotation visibility and play-selection evidence intentionally have different masks.
const KIND = {observed: "実測", interpolated: "補間", occlusion_estimated: "遮蔽位置推定",
  unresolved: "位置未確定", out_of_frame: "画面外"};
const REASON = {reference_only: "参照用フレーム", unreviewed: "未レビュー",
  no_ball: "球の注釈なし", multiple_balls: "複数球", out_of_frame: "画面外"};
function mask(frames, ranges) {
  const result = new Uint8Array(frames);
  for (const [a, b] of ranges) result.fill(1, a, b);
  return result;
}
function runs(values) {
  const result = [];
  for (let a = 0; a < values.length;) {
    let b = a + 1;
    while (b < values.length && values[b] === values[a]) b++;
    result.push([a, b, values[a]]);
    a = b;
  }
  return result;
}
export class PlayTimeline {
  constructor(root, seek) {
    this.root = root;
    this.seek = seek;
    this.clear();
  }
  clear() {
    this.data = null;
    this.frame = 0;
    this.detailStart = 0;
    this.detailCount = 32;
    this.root.hidden = true;
    this.root.replaceChildren();
  }
  message(text) {
    this.clear();
    this.root.hidden = false;
    this.root.textContent = text;
  }
  render(data) {
    this.clear();
    if (!Array.isArray(data.annotations) || data.annotations.length !== data.frames)
      throw new Error("位置注釈のフレーム情報がありません。サーバーを更新してください。");
    this.data = data;
    this.root.hidden = false;
    const title = document.createElement("strong");
    title.textContent = "プレイ区間の候補";
    const scope = document.createElement("p");
    scope.className = "muted";
    scope.textContent = data.pose_approved
      ? "Pose承認済みclip / 対応する固定snapshot"
      : "選択storeの注釈による候補 / pose承認条件は未適用";
    const summary = document.createElement("p");
    summary.className = "muted";
    const c = data.config, n = data.counts;
    summary.textContent = `プレイ候補 ${n.play} / 除外 ${n.excluded} frame · 教師窓被覆 ${n.training} frame (${n.windows}窓)`;
    const conditions = document.createElement("details");
    const heading = document.createElement("summary");
    heading.textContent = "推定条件";
    conditions.append(heading, document.createTextNode(`${c.window_length}frame窓・stride ${c.window_stride}・欠損 ≤${c.max_gap_seconds}秒・存在証拠率 ≥${c.min_presence_fraction}・observed ≥${c.min_observed_frames}`));
    conditions.className = "muted";
    this.root.append(title, scope, summary, conditions);
    this.bridged = mask(data.frames, data.bridged);
    this.rows = [
      {key: "play", label: "プレイ / 除外", values: mask(data.frames, data.play), colors: ["excluded", "play"]},
      {key: "training", label: "教師窓の被覆", values: mask(data.frames, data.training), colors: ["", "training"]},
      {key: "annotations", label: "位置注釈", values: data.annotations.map(a => a.located_count ? (a.kinds.includes("observed") ? 1 : 2) : 0), colors: ["", "located", "estimated"]},
      {key: "evidence", label: "選択用の証拠", values: mask(data.frames, data.presence), colors: ["", "evidence"]},
    ];
    for (const row of this.rows) {
      const container = document.createElement("div");
      container.className = "play-row";
      const name = document.createElement("span");
      name.textContent = row.label;
      const track = document.createElement("div");
      track.className = "play-track";
      track.dataset.row = row.key;
      track.tabIndex = 0;
      track.setAttribute("role", "slider");
      track.setAttribute("aria-label", `${row.label} 全体：クリックまたは矢印キーで移動`);
      track.setAttribute("aria-valuemin", "0");
      track.setAttribute("aria-valuemax", String(data.frames - 1));
      for (const [a, b, value] of runs(row.values)) {
        if (!row.colors[value]) continue;
        const span = document.createElement("span");
        span.className = `play-segment ${row.colors[value]}`;
        span.style.left = `${100 * a / data.frames}%`;
        span.style.width = `${100 * (b - a) / data.frames}%`;
        span.title = `${row.label}: frame ${a}–${b - 1}`;
        track.append(span);
      }
      const cursor = document.createElement("i");
      cursor.className = "play-cursor";
      track.append(cursor);
      track.onclick = event => {
        const rect = track.getBoundingClientRect();
        this.seek(Math.max(0, Math.min(data.frames - 1,
          Math.floor((event.clientX - rect.left) / rect.width * data.frames))));
      };
      track.onkeydown = event => this.navigateKey(event);
      container.append(name, track);
      this.root.append(container);
    }
    const controls = document.createElement("div");
    controls.className = "play-controls";
    this.buttons = [];
    for (const [key, label] of [["play", "区間境界"], ["annotations", "注釈変化"]]) {
      for (const direction of [-1, 1]) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = `${direction < 0 ? "前" : "次"}の${label}`;
        button.onclick = () => this.seek(this.boundary(direction, key));
        this.buttons.push({button, direction, key});
        controls.append(button);
      }
    }
    this.label = document.createElement("output");
    this.label.id = "play-frame-state";
    this.diagnostic = document.createElement("p");
    this.diagnostic.id = "play-annotation-state";
    this.diagnostic.className = "play-annotation-state";
    this.root.append(controls, this.label, this.diagnostic);
    const detailControls = document.createElement("div");
    detailControls.className = "play-controls";
    const zoomLabel = document.createElement("label");
    zoomLabel.textContent = "1frame単位で拡大 ";
    const zoom = document.createElement("select");
    zoom.id = "play-detail-count";
    zoom.setAttribute("aria-label", "拡大表示のフレーム数");
    zoom.append(...[16, 32, 64].map(n => new Option(`${n}frame`, String(n))));
    zoom.value = String(this.detailCount);
    zoom.onchange = () => {
      this.detailCount = Number(zoom.value);
      this.centerDetail();
      this.renderDetail();
      this.setFrame(this.frame);
    };
    zoomLabel.append(zoom);
    this.detailLabel = document.createElement("span");
    this.detailLabel.className = "muted";
    this.detailLabel.id = "play-detail-range";
    detailControls.append(zoomLabel, this.detailLabel);
    this.detailScroll = document.createElement("div");
    this.detailScroll.className = "play-detail-scroll";
    this.detailScroll.setAttribute("aria-label", "フレーム単位の区間詳細");
    this.root.append(detailControls, this.detailScroll);
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = "青＝実測位置、斜線＝補間・推定位置、緑青＝区間選択の証拠。位置未確定でも証拠には含まれます。灰色のマスは位置／証拠なし。短い欠損は拡大欄で確認できます（1frame ≥16px、横スクロール可）。除外は非プレイの確定ラベルではありません。";
    this.root.append(note);
    this.renderDetail();
    this.setFrame(0);
  }
  navigateKey(event) {
    const target = {ArrowLeft: this.frame - 1, ArrowRight: this.frame + 1,
      Home: 0, End: this.data.frames - 1}[event.key];
    if (target !== undefined) {
      event.preventDefault();
      this.seek(Math.max(0, Math.min(this.data.frames - 1, target)));
    }
  }
  boundary(direction, key) {
    const row = this.rows.find(r => r.key === key);
    const edges = runs(row.values).map(([a]) => a)
      .filter(x => direction < 0 ? x < this.frame : x > this.frame);
    return direction < 0 ? edges.at(-1) : edges[0];
  }
  describe(frame) {
    const a = this.data.annotations[frame];
    const kinds = [...new Set(a.kinds)].map(k => KIND[k]).join("・") || "球の注釈なし";
    const evidence = a.evidence ? "あり" : `なし：${a.exclusion_reasons.map(r => REASON[r]).join("・")}`;
    return `位置注釈 ${a.located_count}個（${kinds}） / 選択用の証拠 ${evidence}`;
  }
  centerDetail() {
    this.detailStart = Math.max(0, Math.min(this.data.frames - this.detailCount,
      this.frame - Math.floor(this.detailCount / 2)));
  }
  renderDetail() {
    const start = this.detailStart, stop = Math.min(this.data.frames, start + this.detailCount);
    const grid = document.createElement("div");
    grid.className = "play-detail-grid";
    grid.style.setProperty("--detail-frames", String(stop - start));
    this.detailLabel.textContent = `frame ${start}–${stop - 1}`;
    for (const row of [{key: "axis", label: "frame"}, ...this.rows]) {
      const line = document.createElement("div");
      line.className = `play-detail-row ${row.key === "axis" ? "play-detail-axis" : ""}`;
      line.dataset.row = row.key;
      const label = document.createElement("span");
      label.className = "play-detail-label";
      label.textContent = row.label;
      line.append(label);
      for (let frame = start; frame < stop; frame++) {
        const cell = document.createElement(row.key === "axis" ? "span" : "button");
        if (row.key === "axis") {
          cell.textContent = (frame - start) % 4 === 0 ? String(frame) : "";
        } else {
          const value = row.values[frame];
          cell.type = "button";
          cell.className = `play-cell ${row.colors[value] || "missing"}`;
          cell.dataset.frame = String(frame);
          cell.title = `frame ${frame} (${this.data.timestamps[frame].toFixed(3)}秒) / ${this.describe(frame)}`;
          cell.setAttribute("aria-label", `${row.label} frame ${frame} / ${this.describe(frame)}`);
          cell.onclick = () => this.seek(frame);
          cell.onkeydown = event => this.navigateKey(event);
        }
        line.append(cell);
      }
      grid.append(line);
    }
    this.detailScroll.replaceChildren(grid);
  }
  setFrame(frame) {
    this.frame = frame;
    const d = this.data;
    if (!d) return;
    for (const track of this.root.querySelectorAll(".play-track")) {
      track.querySelector(".play-cursor").style.left = `${100 * (frame + 0.5) / d.frames}%`;
      track.setAttribute("aria-valuenow", String(frame));
    }
    for (const {button, direction, key} of this.buttons)
      button.disabled = this.boundary(direction, key) === undefined;
    this.label.textContent = `frame ${frame} (${d.timestamps[frame].toFixed(2)}秒) · ${this.rows[0].values[frame] ? "プレイ候補" : "除外候補"} · ${this.rows[1].values[frame] ? "教師窓内" : "教師窓外"}${this.bridged[frame] ? " · 証拠欠損を連結" : ""}`;
    this.diagnostic.textContent = this.describe(frame);
    if (frame < this.detailStart || frame >= this.detailStart + this.detailCount) {
      this.centerDetail();
      this.renderDetail();
    }
    for (const cell of this.detailScroll.querySelectorAll(".play-cell")) {
      const current = Number(cell.dataset.frame) === frame;
      cell.classList.toggle("current", current);
      cell.setAttribute("aria-current", String(current));
    }
    const current = this.detailScroll.querySelector(".play-cell.current");
    if (current) {
      const box = current.getBoundingClientRect(), viewport = this.detailScroll.getBoundingClientRect();
      const labelWidth = 104;
      if (box.left < viewport.left + labelWidth)
        this.detailScroll.scrollLeft -= viewport.left + labelWidth - box.left;
      else if (box.right > viewport.right)
        this.detailScroll.scrollLeft += box.right - viewport.right;
    }
  }
}
