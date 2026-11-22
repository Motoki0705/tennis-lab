"use strict";

// Read-only parent → child worker graph. Execution state is supplied by
// worker_status; Herdr joins are decided by the server. This file only orders,
// folds and renders the observation, and polls while the tab is visible.
const RUNNING = "running";
const ATTENTION = new Set(["turn_completed", "stopped", "failed"]);
const SETTING_BASIS = {
  effective: "",
  selected: "（指定値）",
};
const HERDR_STATUS = {
  matched: "Herdrで照合済み",
  absent: "Herdrに担当セッションなし",
  ambiguous:
    "照合不可（同じセッションが複数の端末で開かれているか、照合元が一致しません）",
  not_observed: "Herdr未照合（担当の登録・世代がHerdr取得後に変わりました）",
  not_requested: "Herdr未取得",
};
const BRAND_ICONS = new Map([
  ["claude", "/static/icon-claude.svg"],
  ["agy", "/static/icon-gemini.svg"],
  ["gemini", "/static/icon-gemini.svg"],
  ["codex", "/static/icon-codex.svg"],
]);
const DRAWER_MIN_WIDTH = 320;
const DRAWER_MAX_WIDTH = 960;
const DRAWER_GRAPH_WIDTH = 240;

function addBrandIcon(node, engine) {
  const src = BRAND_ICONS.get(engine?.toLowerCase());
  if (!src) return;
  const image = element("img", "", "brand-icon");
  image.src = src;
  image.alt = "";
  node.textContent = "";
  node.classList.add("has-brand-icon");
  node.append(image);
}

class ExecutionView {
  constructor() {
    this.active = false;
    this.busy = false;
    this.timer = null;
    this.data = null;
    this.error = null;
    this.herdrError = null;
    this.herdrRequested = false;
    this.refreshError = null;
    this.refreshBusy = false;
    this.refreshState = "idle";
    this.refreshRevision = -1;
    this.refreshEpoch = null;
    this.needsLocal = false;
    // UI state survives every 10s re-render: selected node, opened folds and drawer width.
    this.selected = null;
    this.expanded = new Set();
    this.drawerSignature = null;
    byId("refresh-execution").addEventListener("click", () => this.load(true));
    byId("close-execution-drawer").addEventListener("click", () =>
      this.select(null),
    );
    byId("execution-drawer").addEventListener("keydown", (event) => {
      if (event.key === "Escape") this.select(null);
    });
    this.setupResizer();
  }

  setupResizer() {
    const drawer = byId("execution-drawer");
    const resizer = byId("execution-drawer-resizer");
    let drag = null;
    resizer.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const delta = event.key === "ArrowLeft" ? 24 : -24;
      this.setDrawerWidth(drawer.getBoundingClientRect().width + delta);
    });
    resizer.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || drag) return;
      event.preventDefault();
      resizer.focus();
      resizer.setPointerCapture(event.pointerId);
      drag = {
        id: event.pointerId,
        x: event.clientX,
        width: drawer.getBoundingClientRect().width,
      };
    });
    resizer.addEventListener("pointermove", (event) => {
      if (drag?.id === event.pointerId)
        this.setDrawerWidth(drag.width + drag.x - event.clientX);
    });
    resizer.addEventListener("lostpointercapture", () => { drag = null; });
  }

  setDrawerWidth(width) {
    const body = byId("execution-body");
    const max = Math.min(DRAWER_MAX_WIDTH, body.clientWidth - DRAWER_GRAPH_WIDTH);
    const clamped = Math.max(
      DRAWER_MIN_WIDTH,
      Math.min(max, Math.round(width)),
    );
    body.style.setProperty("--drawer-width", `${clamped}px`);
    byId("execution-drawer-resizer").setAttribute("aria-valuenow", String(clamped));
  }

  setActive(active) {
    this.active = active;
    clearTimeout(this.timer);
    if (active) this.load();
  }

  async load(refresh = false) {
    if (refresh) return this.refreshGithub();
    clearTimeout(this.timer);
    if (!this.active || document.visibilityState !== "visible" || this.busy)
      return;
    this.busy = true;
    this.needsLocal = false;
    // Herdr is read on the first visible display only; later reads require
    // the explicit refresh button. A failed first read is not retried here.
    const herdr = !this.herdrRequested;
    this.herdrRequested = true;
    try {
      const data = await getJSON(
        herdr ? "/api/execution?herdr=true" : "/api/execution",
      );
      this.data = data;
      this.error = null;
      if (herdr) this.herdrError = null;
      this.observeRefresh(data.github_refresh);
      this.render();
    } catch (error) {
      if (herdr) {
        this.herdrError = error;
        this.needsLocal = true;
      } else this.error = error;
      this.renderError();
    } finally {
      this.busy = false;
      if (this.active && document.visibilityState === "visible")
        this.timer = setTimeout(() => this.load(), this.needsLocal ? 0 : 10000);
    }
  }

  async refreshGithub() {
    if (
      !this.active ||
      document.visibilityState !== "visible" ||
      this.refreshBusy ||
      this.refreshState === "running"
    )
      return;
    this.refreshBusy = true;
    this.herdrRequested = true;
    this.renderRefresh();
    try {
      // GitHub can take minutes. Local requests keep their own timeout/timer.
      const data = await getJSON("/api/execution?refresh=true", null);
      this.observeRefresh(data.github_refresh);
    } catch (error) {
      this.refreshError = error;
      // A network failure does not prove the server operation has finished.
      // The next local observation resolves that state, including HTTP 202.
      this.refreshState = "running";
    } finally {
      this.refreshBusy = false;
      this.renderRefresh();
      this.renderError();
      this.needsLocal = true;
      // Never render the old local snapshot returned by the long request.
      this.load();
    }
  }

  observeRefresh(meta) {
    if (meta.epoch !== this.refreshEpoch) {
      this.refreshEpoch = meta.epoch;
      this.refreshRevision = -1;
    }
    if (meta.revision < this.refreshRevision) return;
    const advanced = meta.revision > this.refreshRevision;
    this.refreshRevision = meta.revision;
    this.refreshState = meta.status;
    if (advanced && meta.status === "idle") {
      if (meta.result === "succeeded") {
        this.refreshError = null;
        this.herdrError = null;
      }
      if (meta.result === "failed" && !this.refreshError)
        this.refreshError = new Error(
          "GitHub・Herdrの更新を完了できませんでした。",
        );
    }
    this.renderRefresh();
  }

  renderRefresh() {
    const running = this.refreshBusy || this.refreshState === "running";
    const node = byId("execution-refresh");
    node.hidden = !running;
    node.textContent = running
      ? "GitHub・Herdr取得中 · ローカル記録の10秒更新は継続しています。"
      : "";
    const button = byId("refresh-execution");
    button.disabled = running;
    button.textContent = running ? "取得中…" : "更新（GitHub・Herdr）";
  }

  renderError() {
    const parts = [];
    if (this.refreshError)
      parts.push(
        `GitHub・Herdrの更新に失敗しました。${this.refreshError.message}`,
      );
    if (this.herdrError)
      parts.push(
        `Herdrの取得に失敗しました。${this.herdrError.message} 「更新」で再取得できます。`,
      );
    if (this.error)
      parts.push(`ローカル記録の取得に失敗しました。${this.error.message}`);
    const node = byId("execution-error");
    node.hidden = !parts.length;
    node.textContent = parts.length
      ? `${parts.join(" ")} 前回の表示を保持しています。最終取得: ${this.data ? formatTime(this.data.observed_at) : "未取得"} / GitHub: ${this.data?.github.fetched_at ? formatTime(this.data.github.fetched_at) : "未取得"} / Herdr: ${this.data?.herdr.observed_at ? formatTime(this.data.herdr.observed_at) : "未取得"}`
      : "";
  }

  time(value) {
    return value ? formatTime(value) : "未取得";
  }

  text(value) {
    if (!value || value.status === "unavailable") return "未取得";
    if (value.status === "hidden") return "非表示（秘匿検査）";
    return `${value.text}${value.truncated ? "…（一部省略）" : ""}`;
  }

  link(repo, number, kind = "issues", label = `#${number}`) {
    const [owner, name] = repo.split("/");
    const link = element("a", label);
    link.href = `https://github.com/${encodeURIComponent(owner)}/${encodeURIComponent(name)}/${kind}/${encodeURIComponent(number)}`;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    return link;
  }

  // ---- Graph model: ordering and folding of the server observation ----

  done(child) {
    // Completion requires an explicit GitHub observation; never inferred.
    const github = child.github;
    return (
      github.status === "observed" &&
      (github.issue.state === "closed" ||
        github.pull_requests.some((pr) => pr.merged))
    );
  }

  fold(child) {
    if (child.status === RUNNING) return "children";
    if (this.done(child)) return "done";
    if (child.status === "not_delegated") return "undelegated";
    return "children";
  }

  childTitle(child) {
    const title = child.title;
    if (title.status === "visible")
      return `${title.text}${title.truncated ? "…" : ""}`;
    if (title.status === "hidden") return "（題名非表示）";
    return "（題名未取得）";
  }

  setting(value) {
    return `${value.value}${SETTING_BASIS[value.basis] ?? ""}`;
  }

  workerLine(child) {
    const worker = child.worker;
    if (worker.model.basis === "not_started") return "未起動";
    if (worker.model.basis === "unavailable") return "モデル・エフォート未取得";
    return `${this.setting(worker.model)} · ${this.setting(worker.effort)}`;
  }

  visibleParents() {
    const { parents, herdr } = this.data;
    if (herdr.status !== "observed") return parents;
    return parents.filter((parent) => !["absent", "ambiguous"].includes(parent.herdr.status));
  }

  sessionRank(state) {
    if (state === "working") return 0;
    if (state === "blocked") return 1;
    if (state === "idle") return 2;
    return 3;
  }

  parentRank(parent, children) {
    const stateRank = this.sessionRank(parent.herdr.session?.state);
    if (children.some(({ child }) => child.status === RUNNING)) return 0;
    const workerRank = children.some(({ child }) => ATTENTION.has(child.status))
      ? 1
      : children.length ? 2 : 3;
    return Math.min(stateRank, workerRank);
  }

  graph(parents) {
    const items = parents.map((parent) => {
      const key = `parent:${parent.repo}#${parent.parent}`;
      const folded = { children: [], done: [], undelegated: [] };
      for (const child of parent.children) {
        const node = {
          key: `child:${parent.repo}#${parent.parent}/${child.child}`,
          parent,
          child,
        };
        const kind = this.fold(child);
        folded[kind].push(node);
      }
      return {
        key,
        kind: "parent",
        parent,
        folded,
        group: this.parentRank(parent, folded.children),
      };
    });
    for (const session of this.data.sessions) {
      items.push({
        key: `session:${session.key}`,
        kind: "session",
        session,
        group: this.sessionRank(session.state),
      });
    }
    return items.sort((a, b) => a.group - b.group);
  }

  find(items, key) {
    for (const item of items) {
      if (item.key === key) return item;
      for (const node of Object.values(item.folded || {}).flat())
        if (node.key === key) return { ...node, kind: "child" };
    }
    return null;
  }

  // ---- Rendering ----

  parentName(parent) {
    const name = parent.roster.name || "未割当";
    return `${name} · #${parent.parent}`;
  }

  parentIdentity(parent) {
    return [this.parentName(parent), parent.roster.identifier].filter(Boolean).join(" · ");
  }

  sessionTitle(session) {
    if (session.title_status === "redacted")
      return "タイトル非表示（ローカル情報を含むため）";
    return session.title || "タイトルなし";
  }

  icon(text, className, state, engine = null) {
    const node = element("span", text, `graph-icon ${className}`);
    node.setAttribute("aria-hidden", "true");
    addBrandIcon(node, engine);
    if (state !== undefined)
      node.append(element("span", "", `graph-dot state-${state}`));
    return node;
  }

  nodeButton(key, className, label) {
    const node = button("", `graph-node ${className}`, () => this.select(key));
    node.dataset.node = key;
    node.setAttribute("aria-pressed", String(this.selected === key));
    node.setAttribute("aria-label", label);
    return node;
  }

  parentNode(item) {
    if (item.kind === "session") {
      const session = item.session;
      const title = this.sessionTitle(session);
      const node = this.nodeButton(
        item.key,
        "graph-parent-node is-session",
        `${title} · ${session.label} · ${labels[session.state] || session.state} · 未登録セッション`,
      );
      const text = element("span", "", "graph-text");
      text.append(
        element("strong", title, "graph-title"),
        element("span", `${session.label} · 未登録セッション`, "graph-sub"),
      );
      node.append(
        this.icon(
          session.engine.slice(0, 2).toUpperCase(),
          "engine",
          session.state,
          session.engine,
        ),
        text,
      );
      return node;
    }
    const parent = item.parent;
    const herdr = parent.herdr;
    const registered = this.parentIdentity(parent);
    const session = herdr.session;
    const state = session ? session.state : "absent";
    const engine = session?.engine || parent.engine || null;
    // Joined: the terminal title leads, followed by the board name. Without a
    // usable title (or without Herdr) the board name and identifier lead.
    const titled = session?.title_status === "available" && session.title;
    const primary = titled ? session.title : registered;
    const secondary = !session
      ? HERDR_STATUS[herdr.status] || herdr.status
      : titled
        ? `${registered} · ${session.label}`
        : `${session.label} · ${this.sessionTitle(session)}`;
    const node = this.nodeButton(
      item.key,
      `graph-parent-node herdr-${herdr.status}`,
      `${primary} · ${secondary} · ${labels[state] || "Herdr不在"}`,
    );
    const text = element("span", "", "graph-text");
    text.append(
      element("strong", primary, "graph-title"),
      element("span", secondary, "graph-sub"),
    );
    node.append(
      this.icon(
        (parent.roster.name || "?")[0].toUpperCase(),
        "parent",
        state,
        engine,
      ),
      text,
    );
    return node;
  }

  childNode(entry, folded) {
    const { child } = entry;
    const title = this.childTitle(child);
    const workerLine = this.workerLine(child);
    const node = this.nodeButton(
      entry.key,
      `graph-child-node child-${child.status}${folded ? " is-folded" : ""}`,
      `#${child.child} ${title} · ${workerLine} · ${child.label}`,
    );
    const agent = child.worker.agent;
    const text = element("span", "", "graph-text");
    const heading = element("strong", "", "graph-title");
    heading.append(
      element("span", `#${child.child}`, "graph-number"),
      document.createTextNode(` ${title}`),
    );
    const meta = element("span", "", "graph-sub");
    meta.append(
      element("span", workerLine, "graph-worker"),
      element("span", child.label, `graph-status status-${child.status}`),
    );
    text.append(heading, meta);
    node.append(
      this.icon(
        agent ? agent.slice(0, 2).toUpperCase() : "—",
        `agent agent-${agent || "none"}`,
        undefined,
        agent,
      ),
      text,
    );
    if (child.updates_stalled)
      node.append(element("span", "更新途絶", "graph-flag"));
    if (folded && this.done(child))
      node.append(element("span", "完了（GitHub）", "graph-flag is-done"));
    return node;
  }

  foldButton(item, kind, label) {
    const nodes = item.folded[kind];
    const key = `${item.key}|${kind}`;
    const open = this.expanded.has(key);
    const runningCount =
      kind === "children"
        ? nodes.filter((n) => n.child.status === RUNNING).length
        : 0;
    const suffix = runningCount ? `（実行中 ${runningCount}）` : "";
    const node = button(
      `${open ? "−" : "+"}${nodes.length} ${label}${suffix}`,
      `graph-fold fold-${kind}${runningCount ? " has-running" : ""}`,
      () => {
        if (this.expanded.has(key)) this.expanded.delete(key);
        else this.expanded.add(key);
        this.render();
      },
    );
    node.dataset.fold = key;
    node.setAttribute("aria-expanded", String(open));
    node.title = open ? `${label}の子を折りたたむ` : `${label}の子を表示`;
    return node;
  }

  row(item) {
    const row = element("section", "", `graph-row group-${item.group}`);
    row.dataset.row = item.key;
    row.append(this.parentNode(item));
    if (item.kind === "session") return row;
    const folds = element("div", "", "graph-folds");
    if (item.folded.children.length)
      folds.append(this.foldButton(item, "children", "子worker"));
    if (item.folded.done.length)
      folds.append(this.foldButton(item, "done", "完了"));
    if (item.folded.undelegated.length)
      folds.append(this.foldButton(item, "undelegated", "未委任"));
    if (folds.children.length) row.append(folds);
    const entries = [
      ...(this.expanded.has(`${item.key}|children`)
        ? item.folded.children.map((entry) => [entry, false])
        : []),
      ...["done", "undelegated"].flatMap((kind) =>
        this.expanded.has(`${item.key}|${kind}`)
          ? item.folded[kind].map((entry) => [entry, true])
          : [],
      ),
    ];
    if (entries.length) {
      const list = element("ul", "", "graph-children");
      for (const [entry, folded] of entries) {
        const leaf = element(
          "li",
          "",
          `graph-leaf leaf-${entry.child.status}${folded ? " is-folded" : ""}`,
        );
        leaf.append(this.childNode(entry, folded));
        list.append(leaf);
      }
      row.append(list);
    }
    return row;
  }

  render() {
    const data = this.data;
    byId("execution-observed").textContent =
      `ローカル ${this.time(data.observed_at)} · GitHub ${this.time(data.github.fetched_at)} · Herdr ${this.time(data.herdr.observed_at)}`;
    byId("execution-dispatcher").textContent = data.dispatcher.label;
    const visibleParents = this.visibleParents();
    const items = this.graph(visibleParents);
    this.items = items;
    const children = visibleParents.flatMap((parent) => parent.children);
    const running = children.filter((child) => child.status === RUNNING).length;
    byId("execution-summary").textContent =
      `担当親 ${visibleParents.length} · 未登録セッション ${data.sessions.length} · 実行中の子 ${running}` +
      (data.herdr.status === "observed" ? "" : " · Herdr未取得");
    const focused = document.activeElement?.dataset?.node;
    const focusedFold = document.activeElement?.dataset?.fold;
    byId("execution-graph").replaceChildren(
      ...(items.length
        ? items.map((item) => this.row(item))
        : [
            element(
              "p",
              data.herdr.status === "observed"
                ? "Herdrの稼働セッションはありません。"
                : "Herdrの稼働セッションは未取得です。",
              "empty",
            ),
          ]),
    );
    if (focused || focusedFold)
      [
        ...(document
          .getElementById("execution-graph")
          .querySelectorAll?.("[data-node], [data-fold]") || []),
      ]
        .find(
          (node) =>
            (focused &&
              node.dataset.node === focused &&
              node.tagName === "BUTTON") ||
            (focusedFold && node.dataset.fold === focusedFold),
        )
        ?.focus();
    this.renderDrawer();
    this.renderError();
  }

  select(key) {
    const changed = this.selected !== key;
    this.selected = key;
    if (changed) this.drawerSignature = null;
    if (this.data) this.render();
    else this.renderDrawer();
    if (key && changed) byId("close-execution-drawer").focus?.();
  }

  renderDrawer() {
    const drawer = byId("execution-drawer");
    const open = Boolean(this.selected);
    drawer.hidden = !open;
    byId("execution-body").classList.toggle("drawer-open", open);
    if (!open) {
      this.drawerSignature = null;
      return;
    }
    const item = this.items ? this.find(this.items, this.selected) : null;
    // Every local poll stamps a new observed_at; the toolbar shows it. The
    // drawer is rebuilt only when the selected node itself changed, so reading
    // position and focus survive the 10s refresh.
    const signature = JSON.stringify(
      [
        this.selected,
        item?.child ?? item?.session ?? (item?.parent && {
          roster: item.parent.roster,
          engine: item.parent.engine,
          herdr: item.parent.herdr,
          summary: item.parent.summary,
        }),
        this.data?.stale_after_seconds ?? null,
      ],
      (key, value) => (key === "observed_at" ? undefined : value),
    );
    if (signature === this.drawerSignature) return;
    this.drawerSignature = signature;
    const content = byId("execution-drawer-content");
    const scroll = content.scrollTop;
    const focusable = () => [...content.querySelectorAll("a")];
    const focused = focusable().indexOf(document.activeElement);
    let nodes;
    if (!item)
      nodes = [
        element(
          "p",
          "選択したノードは現在の観測にありません。終了・担当変更・Herdrからの離脱の可能性があります。",
          "execution-note",
        ),
      ];
    else if (item.kind === "child") nodes = this.childDetail(item);
    else if (item.kind === "parent") nodes = this.parentDetail(item);
    else nodes = this.sessionDetail(item.session);
    byId("execution-drawer-heading").textContent = !item
      ? "詳細"
      : item.kind === "child"
        ? `#${item.child.child} ${this.childTitle(item.child)}`
        : item.kind === "parent"
          ? item.parent.herdr.session?.title || this.parentIdentity(item.parent)
          : this.sessionTitle(item.session);
    content.replaceChildren(...nodes);
    content.scrollTop = scroll;
    if (focused >= 0) focusable()[focused]?.focus();
  }

  detailHeader(workerText, badges = []) {
    const header = element("div", "", "execution-detail-header");
    header.append(element("p", `ワーカー ${workerText}`, "execution-detail-worker"));
    if (badges.length) {
      const heading = element("div", "", "execution-detail-links");
      heading.append(...badges);
      header.append(heading);
    }
    return header;
  }

  actionSummaries(summary) {
    const box = element("div", "", "execution-summary");
    box.append(element("h3", "行動説明・思考サマリー"));
    const messages = summary?.messages || [];
    if (!messages.length) {
      box.append(element("p", "未取得", "execution-note"));
    } else {
      const list = element("ol", "", "execution-steps");
      for (const msg of messages) {
        const step = element("li", "", "execution-step");
        const when = msg.received_at
          ? element("time", this.time(msg.received_at), "execution-step-time")
          : element("span", "時刻未取得", "execution-step-time");
        if (msg.received_at) when.dateTime = msg.received_at;
        step.append(when);
        step.append(element("p", this.text(msg), "execution-step-text"));
        list.append(step);
      }
      box.append(list);
    }
    if (summary?.tail_truncated)
      box.append(
        element(
          "p",
          "読取上限により過去の要約を省略しています。",
          "execution-note",
        ),
      );
    return box;
  }

  sessionDetail(session) {
    return [
      this.detailHeader(session.label),
      this.actionSummaries(session.summary),
    ];
  }

  parentDetail(item) {
    const { parent } = item;
    return [
      this.detailHeader(parent.herdr.session?.label || parent.engine || "未取得", [
        this.link(parent.repo, parent.parent),
      ]),
      this.actionSummaries(parent.summary),
    ];
  }

  stopRecord(record, label) {
    const node = element("div", "", "execution-stop");
    const fields = {
      reason: "理由",
      facts: "確認事実",
      impact: "影響",
      resume_condition: "再開条件",
      next_action: "次の対応",
    };
    if (record && Object.keys(record).length) {
      node.append(element("strong", label));
      for (const [key, value] of Object.entries(record))
        node.append(element("p", `${fields[key] || key}: ${this.text(value)}`));
    }
    return node;
  }

  childDetail({ parent, child }) {
    const github = child.github;
    const issueState = github.status === "observed" ? github.issue?.state : null;
    const issueLabel = issueState === "open"
      ? "Issue 未完了（OPEN）"
      : issueState === "closed" ? "Issue 完了（CLOSED）" : "Issue 未取得";
    const badges = [
      this.link(parent.repo, child.child),
      element(
        "span",
        issueLabel,
        "execution-badge issue-state",
      ),
    ];
    if (github.status === "observed") {
      for (const pr of github.pull_requests) {
        const prBadge = this.link(parent.repo, pr.number, "pull", `PR #${pr.number}`);
        prBadge.className = "execution-badge";
        badges.push(prBadge);
      }
    }
    const worker = child.worker;
    const workerText = worker.agent
      ? `${worker.agent} · ${this.workerLine(child)}`
      : this.workerLine(child);
    const nodes = [this.detailHeader(workerText, badges)];
    if (child.updates_stalled)
      nodes.push(
        element(
          "p",
          `更新途絶 · ${this.data.stale_after_seconds}秒を超えてイベントの受信がありません。停止とは限りません。`,
          "execution-warning",
        ),
      );
    if (github.status === "observed" && github.stopped)
      nodes.push(this.stopRecord(github.stop_record, "GitHubの停止記録"));
    nodes.push(
      this.stopRecord(child.stop_record, "ローカルの停止記録"),
      this.actionSummaries(child.summary),
    );
    return nodes;
  }
}
