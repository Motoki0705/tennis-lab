"use strict";

// Read-only rendering and polling. Execution state is supplied by worker_status.
class ExecutionView {
  constructor() {
    this.active = false;
    this.busy = false;
    this.timer = null;
    this.data = null;
    this.error = null;
    this.refreshError = null;
    byId("refresh-execution").addEventListener("click", () => this.load(true));
  }

  setActive(active) {
    this.active = active;
    clearTimeout(this.timer);
    if (active) this.load();
  }

  async load(refresh = false) {
    clearTimeout(this.timer);
    if (!this.active || document.visibilityState !== "visible" || this.busy)
      return;
    this.busy = true;
    byId("refresh-execution").disabled = true;
    try {
      const data = await getJSON(
        refresh ? "/api/execution?refresh=true" : "/api/execution",
        refresh ? 120000 : 15000,
      );
      this.data = data;
      this.error = null;
      if (refresh) this.refreshError = null;
      this.render();
    } catch (error) {
      if (refresh) this.refreshError = error;
      else this.error = error;
      this.renderError();
    } finally {
      this.busy = false;
      byId("refresh-execution").disabled = false;
      if (this.active && document.visibilityState === "visible")
        this.timer = setTimeout(() => this.load(), 10000);
    }
  }

  renderError() {
    const error = this.refreshError || this.error;
    const node = byId("execution-error");
    node.hidden = !error;
    node.textContent = error
      ? `${this.refreshError ? "GitHub・名簿の更新" : "ローカル記録の取得"}に失敗しました。${error.message} 前回の表示を保持しています。最終取得: ${this.data ? formatTime(this.data.observed_at) : "未取得"} / GitHub: ${this.data?.github.fetched_at ? formatTime(this.data.github.fetched_at) : "未取得"} / 名簿: ${this.data ? formatTime(this.data.roster_observed_at) : "未取得"}`
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

  childCard(parent, child) {
    const card = element("article", "", "execution-child");
    card.dataset.child = child.child;
    const heading = element("div", "", "execution-child-heading");
    const title = element("h3");
    title.append(this.link(parent.repo, child.child));
    heading.append(
      title,
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
    card.append(heading);
    card.append(
      element(
        "p",
        [child.delegation, child.run?.alias].filter(Boolean).join(" · "),
        "execution-meta",
      ),
    );
    card.append(
      element(
        "p",
        `最終活動 ${this.time(child.last_activity_at)} · 観測 ${this.time(child.observed_at)}`,
        "execution-meta",
      ),
    );
    if (child.run)
      card.append(
        element(
          "p",
          `開始 ${this.time(child.run.started_at)} · 終了 ${this.time(child.run.finished_at)} · 終了理由 ${child.run.turn_end_reason || "未取得"}`,
          "execution-meta",
        ),
      );
    if (child.status === "turn_completed")
      card.append(
        element(
          "p",
          "ターン終了はIssueの受入完了を意味しません。",
          "execution-note",
        ),
      );
    if (child.updates_stalled)
      card.append(
        element(
          "p",
          `更新途絶 · ${this.data.stale_after_seconds}秒を超えてイベントの受信がありません。停止とは限りません。`,
          "execution-warning",
        ),
      );
    if (child.cached_local_observation)
      card.append(
        element(
          "p",
          "ローカル委任記録なし · 前回のGitHub更新時の観測を表示しています。",
          "execution-note",
        ),
      );
    const evidence = element("details", "", "execution-evidence");
    evidence.append(element("summary", "取得元・判定根拠"));
    for (const row of child.evidence)
      evidence.append(
        element("p", `${row.source} · ${row.code} · ${row.fields.join(", ")}`),
      );
    card.append(evidence);
    for (const dependency of child.dependencies) {
      const row = element("p", "依存先 ", "execution-meta");
      row.append(
        this.link(parent.repo, dependency.child),
        document.createTextNode(
          ` · ${dependency.status === "stopped" ? "停止記録あり" : "状態未取得"}`,
        ),
      );
      card.append(row);
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
    card.append(summary);
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
    card.append(
      remote,
      this.stopRecord(child.stop_record, "ローカルの停止記録"),
    );
    return card;
  }

  render() {
    const data = this.data;
    byId("execution-observed").textContent =
      `ローカル観測 ${this.time(data.observed_at)} · GitHub取得 ${this.time(data.github.fetched_at)} · 名簿取得 ${this.time(data.roster_observed_at)}`;
    byId("execution-dispatcher").textContent = data.dispatcher.label;
    const nodes = data.parents.map((parent) => {
      const group = element("section", "", "execution-parent");
      const heading = element("header", "", "execution-parent-heading");
      const title = element("h2");
      title.append(
        party({ ...parent.roster, repo: parent.repo, parent: parent.parent }),
      );
      heading.append(
        title,
        element(
          "span",
          `親 · Herdr ${labels[parent.roster.state] || parent.roster.state}`,
          `execution-badge parent-${parent.roster.state}`,
        ),
      );
      group.append(heading);
      group.append(
        element(
          "p",
          `識別名 ${parent.roster.identifier || "なし"} · ${parent.assignment === "active" ? "担当中" : "引継ぎ待ち"} · 世代 ${parent.generation}`,
          "execution-meta",
        ),
      );
      group.append(
        element(
          "p",
          `取得元 ${parent.source} · 名簿 ${parent.roster.source}`,
          "execution-meta",
        ),
      );
      if (!parent.children.length)
        group.append(element("p", "子の登録なし", "execution-note"));
      for (const child of parent.children)
        group.append(this.childCard(parent, child));
      return group;
    });
    byId("execution-parents").replaceChildren(
      ...(nodes.length
        ? nodes
        : [element("p", "担当中・引継ぎ待ちの親はありません。", "empty")]),
    );
    this.renderError();
  }
}
