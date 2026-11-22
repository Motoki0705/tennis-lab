# PR #2376 正式レビュー Major 指摘（4件）の修正指示

正式レビュー（run `parent-2366-0c16916c7544499bbbbf66111e223aea`）で以下の 4 件の `major` 指摘が報告されました。すべて修正し、回帰テストを追加・更新してコミット・push し、`PARENT_RESULT` で CI 待機を宣言して終了してください。

---

## Major 1: 子workerの要約にHerdrの私的な識別子が残る

### 指摘内容
> 実行検証。キャッシュ済みHerdr観測に含まれるpane IDをCodex子workerの発言に入れると、`summary.messages` に `visible` のまま残りました。`web_data.py` の `execution_names()` はHerdrの私的値を親・未登録セッションの検査に渡しますが、子の検査にはその値が渡らず、応答全体の再検査にも含まれません。ac-6 とREADMEの秘匿契約を満たすよう、子の原文にも切り詰め前に同じ検査を適用し、既存HTTPテストで非表示を確認してください。

### 修正方針
1. `.agents/ops/worker_status/collector.py`（および `__init__.py` の `collect_status` シグネチャ）に任意のキーワード引数 `secrets: Mapping[str, Any] | None = None` を追加し、`Guard({**store.parents, **(secrets or {})})` で初期化する（再取得時の `guard = Guard(...)` も同様に `secrets` を含める）。これにより、子 worker の `events.jsonl` の原文を 300 文字に切り詰める前に Herdr の私的値（`pane_id`, `terminal_id`, `cwd`, `foreground_cwd`, `agent_session`）で秘匿検査できる。
2. `.agents/ops/board/execution_data.py`:
   - `_local()` では先に `with self.lock:` で `remote, herdr` と `state` を取得し、`collect_status` に `**({"secrets": herdr["private"]} if herdr else {})` を渡す。
   - `read(refresh=True)` では先に `observed = runtime.sessions(self.board.env)` を取得してから `collect_status(..., github=True, secrets=observed["private"])` を呼ぶ（これにより初回の `refresh=True` でも常に最新の `observed["private"]` が子の原文切り詰め前に適用される）。
3. `.agents/ops/board/web_data.py`:
   - `execution_names()` 末尾の `public(observation, parents, self.env)` を `public(observation, {**parents, **private}, self.env)` に変更し、応答全体の再検査にも Herdr の私的値を含める。
4. `tests/e2e/agents_ops/test_board_web.py`:
   - 実 collector を通す HTTP テスト（例: `test_execution_reads_real_collector_records_without_changing_any_file` または `test_execution_session_summaries_follow_cached_herdr_identity`）で、キャッシュ済み Herdr 観測の `pane_id`（例: 300 文字超の位置に `"w1:p1"` を含む発言）を子 worker の `events.jsonl` に入れた際、`child["summary"]["messages"]` と `child["summary"]["message"]` が `status: "hidden"` になり応答テキストに `"w1:p1"` が含まれないことを検証する。

---

## Major 2: Codex子workerの思考サマリーが表示されない

### 指摘内容
> 実行検証。実際の `codex exec --json` と同じ形式の `item.completed` を発言、`reasoning`、発言の順で `summarize()` に渡すと、両端の発言だけが `summary.messages` に残りました。子workerの抽出処理が `agent_message` だけを扱うため、作業中に思考サマリーだけが出た区間はドロワーに反映されず、ac-6 を満たしません。既存の子workerテストに `reasoning` を加え、秘匿検査後の表示と順序を確認してください。

### 修正方針
1. `.agents/ops/worker_status/records.py` の `summarize()`（352行目付近）:
   - `event["type"] == "item.completed"` のとき、`kind in {"agent_message", "reasoning"}` を対象にする。
   - `value = item.get("text")` を検査（`if value is not None and not isinstance(value, str): raise ValueError("invalid free text")`）し、`if kind == "agent_message": message = (value, when)` としたうえで、`agent_message` と `reasoning` の両方で `if value is not None: append(value, when)` を呼ぶ（従来の `summary["message"]` は最後の `agent_message` を維持し、`summary["messages"]` には `agent_message` と `reasoning` の両方が時系列順に入る）。
2. `.agents/ops/worker_status/README.md`:
   - `summary.messages` が Codex の `agent_message`・`reasoning` と Claude の `text`・`thinking`・`result` を時系列で保持し、`summary.message` は最後のユーザー向け発言を保持する旨に記述を更新する。
3. `tests/e2e/agents_ops/test_worker_status.py`:
   - `test_summary_last_message_recent_tools_and_legacy_times` で、発言（`agent_message`）→ 思考サマリー（`reasoning`、可視テキストおよび秘匿対象を含むテキスト）→ 発言（`agent_message`）の順に渡したとき、`summary["messages"]` に秘匿検査後の `reasoning` が順序どおり含まれ、`summary["message"]` は最後の `agent_message` のままであることを検証する。

---

## Major 3: Herdr取得成功後も稼働未確認の親をグラフに表示する

### 指摘内容
> 実行検証。全体のHerdr状態が `observed` の応答で、`matched` の親と `not_observed` の親が両方表示対象になりました。観測後に新規登録された親や担当世代が変わった親は、稼働を確認できていなくても次の10秒更新で一覧に加わります。これは初回未取得や取得失敗時に前回表示を保持する例外とは別の経路で、ac-2 の絞り込みを破ります。成功済みの観測では `matched` の親だけをグラフに出し、未照合の端末は既存の未登録セッション表示で扱ってください。

### 修正方針
1. `.agents/ops/board/static/execution.js` の `visibleParents()`:
   - `if (herdr.status !== "observed") return parents;`
   - `return parents.filter((parent) => parent.herdr.status === "matched");`
   に変更し、Herdr 観測成功後は `matched` の親だけをグラフに表示する（`not_observed` / `absent` / `ambiguous` をすべて除外する）。
2. `.agents/ops/board/web_data.py` の `execution_names()`:
   - `parent["summary"]` に渡す `session_id` も `session_id if parent["herdr"]["status"] in {"matched", "not_requested"} else None` に絞る。
3. `tests/e2e/agents_ops/board_execution.cjs`:
   - `herdr.status === "observed"` のとき、`herdr.status === "not_observed"` の親も `visibleParents()` およびグラフ描画から除外されることを検証する。

---

## Major 4: 前景プロセスが複数ある端末を担当親として照合できてしまう

### 指摘内容
> 実行検証。Herdrの既存検査が同じエンジンの前景プロセスを一意に特定できず拒否する条件で、保存済みPIDが前景に残っていると、新しい照合経路は `working` を返しました。`_saved_foreground()` は保存済みPIDだけを確認するため、Herdrの会話参照がない端末でも担当親に結び付け、登録側の会話ログをその端末の詳細として表示できます。初回・明示更新で既存の一意性検査を保ち、10秒ポーリングでは観測キャッシュを使ってください。

### 修正方針
1. `.agents/ops/board/service.py` と `.agents/ops/board/runtime.py`:
   - `service.py` の `_runtime_state` から `process_matches` 引数を削除し、常に既存の `process_stamp(client, agent) == stamps`（`herdr_labels/coordination.py` の一意性検査: `herdr pane process-info` で同じエンジンの前景プロセスがただ 1 つであることを検証）を使うように戻す。
   - `runtime.py` から `_saved_foreground` を削除する。`runtime.sessions(env)` は初回（`herdr=true`）と明示更新（`refresh=true`）でしか呼ばれず、10秒ポーリング（`[local path]`）では `ExecutionReader` のキャッシュ済み `self.herdr`（`session_ids` 等）が使われるため、`runtime.sessions(env)` では既存の一意性検査をそのまま実行する。
   - 未登録の `agy` 端末の会話 ID 解決も、`process_stamp(server.client, agent)` で一意な前景 `agy` プロセスを取得できた場合（または `_agy_panes` でも同一 pane に複数の `agy` プロセスがない一意な場合）に `AgyTui(stamp["pid"], stamp["start"]).conversation()` で解決するように整理する。
2. `tests/e2e/agents_ops/test_board_web.py`:
   - `test_execution_session_summaries_follow_cached_herdr_identity` で、`FAKE_HERDR` の `pane process-info`（既存の一意性検査）を通るように `[local path]` を用意し、前景プロセスが一意な場合は `matched` になり 10 秒ポーリングでは `herdr` を再実行せずにログ追記が反映されること、および同じエンジンの前景プロセスが 2 つある端末（`foreground_processes` に同エンジンが 2 件ある場合）は `absent` として拒否され未登録セッションに回ることを検証する。
