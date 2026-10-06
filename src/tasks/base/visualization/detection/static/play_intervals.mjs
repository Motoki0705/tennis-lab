// Read-only proposal timeline. Frame indices match the image/GT endpoints.
const contains = (ranges, frame) => ranges.some(([a, b]) => a <= frame && frame < b);
export class PlayTimeline {
  constructor(root, seek) {
    this.root = root;
    this.seek = seek;
    this.clear();
  }
  clear() {
    this.data = null;
    this.frame = 0;
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
    summary.textContent = `プレイ候補 ${n.play} / 除外 ${n.excluded} frame · 教師窓被覆 ${n.training} frame (${n.windows}窓) · ${c.window_length}frame窓・stride ${c.window_stride}・欠損 ≤${c.max_gap_seconds}秒・存在率 ≥${c.min_presence_fraction}・observed ≥${c.min_observed_frames}`;
    this.root.append(title, scope, summary);
    const definitions = [
      ["プレイ / 除外", [[data.play, "play"], [data.excluded, "excluded"]]],
      ["教師窓の被覆", [[data.training, "training"]]],
      ["ボール存在", [[data.presence, "presence"]]],
    ];
    for (const [label, layers] of definitions) {
      const row = document.createElement("div");
      row.className = "play-row";
      const name = document.createElement("span");
      name.textContent = label;
      const track = document.createElement("div");
      track.className = "play-track";
      track.tabIndex = 0;
      track.setAttribute("role", "slider");
      track.setAttribute("aria-label", `${label}：クリックまたは矢印キーで移動`);
      track.setAttribute("aria-valuemin", "0");
      track.setAttribute("aria-valuemax", String(data.frames - 1));
      for (const [ranges, kind] of layers) {
        for (const [a, b] of ranges) {
          const span = document.createElement("span");
          span.className = `play-segment ${kind}`;
          span.style.left = `${100 * a / data.frames}%`;
          span.style.width = `${100 * (b - a) / data.frames}%`;
          span.title = `${label}: frame ${a}–${b - 1}`;
          track.append(span);
        }
      }
      const cursor = document.createElement("i");
      cursor.className = "play-cursor";
      track.append(cursor);
      track.onclick = (event) => {
        const rect = track.getBoundingClientRect();
        this.seek(Math.max(0, Math.min(data.frames - 1,
          Math.floor((event.clientX - rect.left) / rect.width * data.frames))));
      };
      track.onkeydown = (event) => {
        const target = { ArrowLeft: this.frame - 1, ArrowRight: this.frame + 1,
          Home: 0, End: data.frames - 1 }[event.key];
        if (target !== undefined) {
          event.preventDefault();
          this.seek(Math.max(0, Math.min(data.frames - 1, target)));
        }
      };
      row.append(name, track);
      this.root.append(row);
    }
    const controls = document.createElement("div");
    controls.className = "play-controls";
    this.label = document.createElement("output");
    this.label.id = "play-frame-state";
    this.buttons = [-1, 1].map((direction) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = direction < 0 ? "前の区間境界" : "次の区間境界";
      button.onclick = () => this.seek(this.boundary(direction));
      controls.append(button);
      return button;
    });
    controls.append(this.label);
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = "緑＝プレイ候補、橙＝除外候補、紫＝教師窓被覆、青＝ボール存在の証拠。除外は非プレイの確定ラベルではありません。区間の編集・承認は行いません。";
    this.root.append(controls, note);
    this.setFrame(0);
  }
  boundary(direction) {
    const edges = [...new Set([...this.data.play, ...this.data.excluded].flat())]
      .filter((x) => x < this.data.frames && (direction < 0 ? x < this.frame : x > this.frame))
      .sort((a, b) => a - b);
    return direction < 0 ? edges.at(-1) : edges[0];
  }
  setFrame(frame) {
    this.frame = frame;
    const d = this.data;
    if (!d) return;
    for (const track of this.root.querySelectorAll(".play-track")) {
      track.querySelector(".play-cursor").style.left = `${100 * (frame + 0.5) / d.frames}%`;
      track.setAttribute("aria-valuenow", String(frame));
    }
    this.buttons[0].disabled = this.boundary(-1) === undefined;
    this.buttons[1].disabled = this.boundary(1) === undefined;
    this.label.textContent = `frame ${frame} (${d.timestamps[frame].toFixed(2)}秒) · ${contains(d.play, frame) ? "プレイ候補" : "除外候補"} · ${contains(d.training, frame) ? "教師窓内" : "教師窓外"}${contains(d.bridged, frame) ? " · 欠損を連結" : ""}`;
  }
}
