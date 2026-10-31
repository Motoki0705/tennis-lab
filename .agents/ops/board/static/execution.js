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
const MATCH_METHOD = {
  owner: "担当表示の確認済み対応",
  native: "Herdrのセッション参照",
  owner_native: "担当表示の確認済み対応・Herdrのセッション参照",
};

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
    // UI state survives every 10s re-render: selected node and opened folds.
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
    if (child.status === RUNNING) return null;
    if (this.done(child)) return "done";
    if (child.status === "not_delegated") return "undelegated";
    return null;
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

  graph() {
    const data = this.data;
    const items = data.parents.map((parent, index) => {
      const key = `parent:${parent.repo}#${parent.parent}`;
      const visible = [];
      const folded = { done: [], undelegated: [] };
      for (const child of parent.children) {
        const node = {
          key: `child:${parent.repo}#${parent.parent}/${child.child}`,
          parent,
          child,
        };
        const kind = this.fold(child);
        if (kind) folded[kind].push(node);
        else visible.push(node);
      }
      const statuses = visible.map((node) => node.child.status);
      const group = statuses.includes(RUNNING)
        ? 0
        : statuses.some((status) => ATTENTION.has(status))
          ? 1
          : visible.length
            ? 2
            : 3;
      return { key, kind: "parent", parent, visible, folded, group, index };
    });
    data.sessions.forEach((session, index) =>
      items.push({
        key: `session:${session.key}`,
        kind: "session",
        session,
        visible: [],
        folded: { done: [], undelegated: [] },
        group: 3,
        index: data.parents.length + index,
      }),
    );
    return items.sort((a, b) => a.group - b.group || a.index - b.index);
  }

  find(items, key) {
    for (const item of items) {
      if (item.key === key) return item;
      for (const node of [
        ...item.visible,
        ...item.folded.done,
        ...item.folded.undelegated,
      ])
        if (node.key === key) return { ...node, kind: "child" };
    }
    return null;
  }

  // ---- Rendering ----

  parentName(parent) {
    const name = parent.roster.name || "未割当";
    return `${name} · #${parent.parent}`;
  }

  sessionTitle(session) {
    if (session.title_status === "redacted")
      return "タイトル非表示（ローカル情報を含むため）";
    return session.title || "タイトルなし";
  }

  icon(text, className, state) {
    const node = element("span", text, `graph-icon ${className}`);
    node.setAttribute("aria-hidden", "true");
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
        ),
        text,
      );
      return node;
    }
    const parent = item.parent;
    const herdr = parent.herdr;
    const name = this.parentName(parent);
    const registered = [name, parent.roster.identifier]
      .filter(Boolean)
      .join(" · ");
    const session = herdr.session;
    const state = session ? session.state : "absent";
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
      this.icon((parent.roster.name || "?")[0].toUpperCase(), "parent", state),
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
    const node = button(
      `${open ? "−" : "+"}${nodes.length} ${label}`,
      `graph-fold fold-${kind}`,
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
    const folds = element("div", "", "graph-folds");
    if (item.folded.done.length)
      folds.append(this.foldButton(item, "done", "完了"));
    if (item.folded.undelegated.length)
      folds.append(this.foldButton(item, "undelegated", "未委任"));
    if (item.kind === "parent" && !item.parent.children.length)
      folds.append(element("span", "子の登録なし", "graph-empty"));
    if (folds.children.length) row.append(folds);
    const entries = [
      ...item.visible.map((entry) => [entry, false]),
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
    const items = this.graph();
    this.items = items;
    const children = data.parents.flatMap((parent) => parent.children);
    const running = children.filter((child) => child.status === RUNNING).length;
    byId("execution-summary").textContent =
      `担当親 ${data.parents.length} · 未登録セッション ${data.sessions.length} · 実行中の子 ${running}` +
      (data.herdr.status === "observed" ? "" : " · Herdr未取得");
    const focused = document.activeElement?.dataset?.node;
    const focusedFold = document.activeElement?.dataset?.fold;
    byId("execution-graph").replaceChildren(
      ...(items.length
        ? items.map((item) => this.row(item))
        : [
            element(
              "p",
              "担当中・引継ぎ待ちの親とHerdrの稼働セッションはありません。",
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
    // position, opened evidence and focus survive the 10s refresh.
    const signature = JSON.stringify(
      [
        this.selected,
        item?.parent ?? null,
        item?.session ?? null,
        item?.child ?? null,
        this.data?.github.fetched_at ?? null,
        this.data?.stale_after_seconds ?? null,
      ],
      (key, value) => (key === "observed_at" ? undefined : value),
    );
    if (signature === this.drawerSignature) return;
    this.drawerSignature = signature;
    const content = byId("execution-drawer-content");
    const scroll = content.scrollTop;
    const opened = new Set(
      [...(content.querySelectorAll?.("details[open]") || [])].map(
        (node) => node.dataset.key,
      ),
    );
    const focusable = () => [
      ...(content.querySelectorAll?.("a, button, summary") || []),
    ];
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
          ? this.parentName(item.parent)
          : this.sessionTitle(item.session);
    content.replaceChildren(...nodes);
    for (const node of content.querySelectorAll?.("details") || [])
      if (opened.has(node.dataset.key)) node.open = true;
    content.scrollTop = scroll;
    if (focused >= 0) focusable()[focused]?.focus();
  }

  field(label, value) {
    const row = element("div", "", "detail-field");
    row.append(element("dt", label));
    const data = element("dd");
    if (typeof value === "string") data.textContent = value;
    else data.append(value);
    row.append(data);
    return row;
  }

  sessionFields(session) {
    return [
      this.field("端末タイトル", this.sessionTitle(session)),
      this.field(
        "Herdr",
        `${session.label} · ${labels[session.state] || session.state}`,
      ),
    ];
  }

  sessionDetail(session) {
    const list = element("dl", "", "detail-fields");
    list.append(
      ...this.sessionFields(session),
      this.field(
        "掲示板",
        "担当親として未登録、またはHerdrと担当を照合できない会話です。",
      ),
    );
    return [list];
  }

  parentDetail(item) {
    const { parent } = item;
    const herdr = parent.herdr;
    const list = element("dl", "", "detail-fields");
    const issue = element("span");
    issue.append(
      this.link(parent.repo, parent.parent),
      document.createTextNode(` · ${parent.repo}`),
    );
    list.append(
      this.field("親issue", issue),
      this.field("名前", parent.roster.name || "未割当"),
      this.field("識別名", parent.roster.identifier || "なし"),
      this.field(
        "担当",
        `${parent.assignment === "active" ? "担当中" : "引継ぎ待ち"} · 世代 ${parent.generation}`,
      ),
      this.field(
        "Herdr",
        herdr.status === "matched"
          ? `${HERDR_STATUS.matched}（${MATCH_METHOD[herdr.method] || herdr.method}）`
          : HERDR_STATUS[herdr.status] || herdr.status,
      ),
    );
    if (herdr.session) list.append(...this.sessionFields(herdr.session));
    list.append(
      this.field("取得元", `${parent.source} · 名前 ${parent.roster.source}`),
    );
    const nodes = [list];
    const counts = {};
    for (const child of parent.children)
      counts[child.label] = (counts[child.label] || 0) + 1;
    nodes.push(
      element(
        "p",
        parent.children.length
          ? `子 ${parent.children.length}件 · ${Object.entries(counts)
              .map(([label, count]) => `${label} ${count}`)
              .join(
                " · ",
              )} · 完了 ${item.folded.done.length}（GitHub取得時の判定）`
          : "子の登録なし",
        "execution-meta",
      ),
    );
    if (parent.children.length) {
      const children = element("ul", "", "detail-children");
      for (const child of parent.children) {
        const row = element("li");
        const key = `child:${parent.repo}#${parent.parent}/${child.child}`;
        const open = button(
          `#${child.child} ${this.childTitle(child)}`,
          "detail-child-link",
          () => this.select(key),
        );
        row.append(
          open,
          element(
            "span",
            child.label,
            `execution-badge status-${child.status}`,
          ),
        );
        children.append(row);
      }
      nodes.push(children);
    }
    return nodes;
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
    const nodes = [];
    const heading = element("div", "", "execution-child-heading");
    heading.append(
      this.link(parent.repo, child.child),
      element("span", child.label, `execution-badge status-${child.status}`),
    );
    const github = child.github;
    heading.append(
      element(
        "span",
        github.status === "observed"
          ? `Issue ${github.issue.state === "open" ? "未完了（OPEN）" : github.issue.state.toUpperCase()}`
          : "Issue 未取得",
        "execution-badge issue-state",
      ),
    );
    nodes.push(heading);
    const worker = child.worker;
    const list = element("dl", "", "detail-fields");
    list.append(
      this.field("親", `${this.parentName(parent)}`),
      this.field(
        "題名",
        `${this.childTitle(child)}${child.title.source === "github.issue" ? "（GitHub取得値）" : ""}`,
      ),
      this.field(
        "worker",
        worker.agent
          ? `${worker.agent} · ${this.workerLine(child)}`
          : this.workerLine(child),
      ),
    );
    if (worker.model.basis === "selected" || worker.effort.basis === "selected")
      list.append(
        this.field(
          "設定の根拠",
          "（指定値）は実行記録の選択値です。CLIが報告した実効値を取得できていません。",
        ),
      );
    list.append(
      this.field(
        "委任・run",
        [child.delegation, child.run?.alias].filter(Boolean).join(" · ") ||
          "なし",
      ),
      this.field("最終活動", this.time(child.last_activity_at)),
    );
    if (child.run)
      list.append(
        this.field(
          "ターン",
          `開始 ${this.time(child.run.started_at)} · 終了 ${this.time(child.run.finished_at)} · 終了理由 ${child.run.turn_end_reason || "未取得"}`,
        ),
      );
    nodes.push(list);
    if (child.status === "turn_completed")
      nodes.push(
        element(
          "p",
          "ターン終了はIssueの受入完了を意味しません。",
          "execution-note",
        ),
      );
    if (child.updates_stalled)
      nodes.push(
        element(
          "p",
          `更新途絶 · ${this.data.stale_after_seconds}秒を超えてイベントの受信がありません。停止とは限りません。`,
          "execution-warning",
        ),
      );
    if (child.cached_local_observation)
      nodes.push(
        element(
          "p",
          "ローカル委任記録なし · 前回のGitHub更新時の観測を表示しています。",
          "execution-note",
        ),
      );
    for (const dependency of child.dependencies) {
      const row = element("p", "依存先 ", "execution-meta");
      row.append(
        this.link(parent.repo, dependency.child),
        document.createTextNode(
          ` · ${dependency.status === "stopped" ? "停止記録あり" : "状態未取得"}`,
        ),
      );
      nodes.push(row);
    }
    const message = child.summary.message;
    const summary = element("div", "", "execution-summary");
    summary.append(
      element("h4", `最後の発言 · ${this.time(message.received_at)}`),
    );
    summary.append(element("p", this.text(message)));
    summary.append(element("h4", "直近のツール"));
    if (!child.summary.tools.length) summary.append(element("p", "未取得"));
    for (const tool of child.summary.tools) {
      const row = element("p", `${tool.kind} · ${this.time(tool.received_at)}`);
      const text =
        tool.kind === "command_execution"
          ? this.text(tool.command)
          : tool.kind === "mcp_tool_call"
            ? `${this.text(tool.server)} · ${this.text(tool.tool)}`
            : `変更 ${tool.change_count}件`;
      row.append(element("code", text));
      summary.append(row);
    }
    if (child.summary.tail_truncated)
      summary.append(
        element(
          "p",
          "読取上限により過去の要約を省略しています。",
          "execution-note",
        ),
      );
    nodes.push(summary);
    const remote = element("div", "", "execution-github");
    remote.append(
      element(
        "p",
        `GitHub取得 ${this.time(github.fetched_at)}${github.issue?.state_reason ? ` · ${github.issue.state_reason}` : ""}`,
        "execution-meta",
      ),
    );
    if (github.status !== "observed")
      remote.append(
        element("p", "更新ボタンでIssue・PRの状態を取得できます。"),
      );
    else {
      if (!github.pull_requests.length)
        remote.append(element("p", "対応PRなし（取得時点）"));
      for (const pr of github.pull_requests) {
        const row = element("p");
        row.append(
          this.link(parent.repo, pr.number, "pull", `PR #${pr.number}`),
          document.createTextNode(
            ` · ${pr.merged ? "MERGED" : pr.state.toUpperCase()}`,
          ),
        );
        remote.append(row);
      }
      for (const reason of github.launch.reasons)
        remote.append(
          element("p", `${reason.code}: ${this.text(reason.message)}`),
        );
      if (github.stopped)
        remote.append(this.stopRecord(github.stop_record, "GitHubの停止記録"));
    }
    nodes.push(
      remote,
      this.stopRecord(child.stop_record, "ローカルの停止記録"),
    );
    const evidence = element("details", "", "execution-evidence");
    evidence.dataset.key = "evidence";
    evidence.append(element("summary", "取得元・判定根拠"));
    for (const row of child.evidence)
      evidence.append(
        element("p", `${row.source} · ${row.code} · ${row.fields.join(", ")}`),
      );
    evidence.append(
      element(
        "p",
        `worker設定 · ${worker.source || "なし"} · model ${worker.model.basis} · effort ${worker.effort.basis}`,
      ),
      element("p", `題名 · ${child.title.source || "未取得"}`),
    );
    nodes.push(evidence);
    return nodes;
  }
}
