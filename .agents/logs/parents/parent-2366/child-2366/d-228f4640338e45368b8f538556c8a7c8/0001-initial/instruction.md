# Issue #2366 実装指示: 掲示板 Web UI「実行状況」タブの稼働セッション中心への刷新と詳細表示の改善

担当 Issue: `#2366`（親自身への単独実装ワーカー `primary`）

## 1. 人間との合意事項と受入条件（AC）

以下の要件・受入条件（`agreement`, `ac-1`〜`ac-7`）が人間と正式合意済みです。すべて満たす実装・テスト更新・README更新を行い、検証・commit・push して `Closes #2366` を含む日本語 PR を作成し、CI 待機を `PARENT_RESULT` で宣言して終了してください。

### 受入条件一覧
- **ac-1（タブ順序）**:
  - `.agents/ops/board/static/index.html` の上部タブ（`.tabs`）の並び順を **`会話` (`timeline`) → `実行状況` (`execution`) → `応答待ち` (`waiting`) → `配送の問題` (`issues`)** に変更する。
- **ac-2（Herdr 稼働セッション絞り込み・実行中上位ソート）**:
  - 実行状況タブに表示される登録済み親ノードを、**Herdr 上に存在する稼働セッション（`working` / `idle` / `blocked` / `done` 等）に絞り込み、`absent`（未起動・終了済み）の親をグラフ表示から除外**する。
  - ただし、Herdr が未取得（`not_requested` / `not_observed`）または取得失敗のときは除外せず従来どおりエラーバナーと前回キャッシュを保持し、「稼働セッションがゼロ」と誤認させない。
  - グラフの並び順（`parentRank` / `sessionRank`）で、**実行中のエージェント（実行中の子を持つ親、`herdr.session.state === "working"` の親、`state === "working"` の未登録セッション）を最上位（グループ 0）に表示**する。
- **ac-3（サブエージェントのデフォルト折りたたみとクリック展開・収納）**:
  - 親ノード配下のサブエージェント（アクティブな子ノード群: 実行中・ターン終了・停止中・失敗・依存待ち・委任予約済み・状態不明）を**デフォルトで折りたたみ**（例: `+N 子worker`、実行中の子がある場合は `+N 子worker（実行中 M）` バッジ）、クリックで展開・収納を切り替えられるようにする。
  - 既存の `+N 完了`（`|done`）および `+M 未委任`（`|undelegated`）の折りたたみと独立して動作し、10 秒ポーリングや明示更新後も展開状態を保持する。
- **ac-4（ブランドアイコンのローカル静的配信と表示）**:
  - Web から取得・同梱した Anthropic (Claude) / Google Gemini (agy / gemini) / OpenAI (Codex) の SVG アイコンを `.agents/ops/board/static/` にローカル静的アセット（`icon-claude.svg`, `icon-gemini.svg`, `icon-codex.svg`）として配置し、`.agents/ops/board/web.py` の `ASSETS` に登録する（CSP `img-src 'self'` 準拠、外部 URL 参照は禁止）。
  - 実行状況タブの**登録済み親ノード・未登録セッションノード・子ノード**、および画面上部の**「現在のエージェント」（`#runtime-agents`）カード**にブランドアイコンを表示する。
  - 親ノードは Herdr 未照合時でも登録セッションから `engine`（`claude` / `codex` / `agy`）が分かるよう、`.agents/ops/board/web_data.py` の `execution_names` で `parent["engine"]` を付与する。
- **ac-5（右詳細パネル上部の簡素化と `<dl>` 廃止）**:
  - 右詳細パネル（`#execution-drawer`）上部は**「題名」と「ワーカー」**を中心とし、従来の 7 項目 `<dl class="detail-fields">`（親・題名・worker・設定の根拠・委任・run・最終活動・ターン）および `<details class="execution-evidence">` を廃止する。
  - ただし、運用判断に必要な **Issue / PR リンク（`Issue 未取得` / `Issue 未完了（OPEN）` / `Issue 完了（CLOSED）`、`PR #番号`）および異常系ステータス（`更新途絶: ... 停止とは限りません`、停止記録 `stop_record` / `github.stop_record`）のみ最小限のバッジ・ノートとして残す**。
- **ac-6（逐次行動説明・思考サマリーの時系列表示）**:
  - 右詳細パネル下部で、従来の「最後の発言」1 件＋「直近のツール」に代わり、**親ノード・未登録セッション（Claude / Codex / agy）および子ノード（Codex / Claude）が処理途中で逐次的に出力する行動説明・思考サマリーを時系列リスト（`.execution-steps`）で表示**する。
  - すべての抽出テキストは既存の秘匿検査（`worker_status/safety.py` の `Guard.text` および `board/safety.py` の `public`）を通し、秘匿対象を含む項目は `status: "hidden"`（「非表示（秘匿検査）」）にする。
  - **子ノード（`.agents/ops/worker_status/records.py`）**:
    - `summarize()` で従来の `summary["message"]`・`summary["tools"]`・`summary["tail_truncated"]` を CLI 互換のため維持しつつ、時系列の `summary["messages"]`（上限 `message_limit=20` 件、各要素は `guard.text(..., limit=300)` + `received_at` + `time_status`）を追加する。
    - Codex の `item.completed` (`item.type == "agent_message"`) に加え、Claude (`stream-json`) の `type == "assistant"` イベント内の `message.content` 配列にある `type in {"thinking", "text"}` ブロックおよび `type == "result"` イベントの `result` 文字列からも抽出する（同一テキストの連続重複は除外）。
  - **親ノード・未登録セッション（`.agents/ops/board/runtime.py` & `web_data.py`）**:
    - `runtime.py` に `session_summary(engine, session_id, guard, env, *, tail_bytes=262144, limit=20)` を実装し、ローカルの会話ログ末尾から行動説明・思考サマリーを抽出して `row["summary"]`（および `web_data.py` の `parent["summary"]`）に付与する。
    - **agy**: `[local path]` の `CORTEX_STEP_TYPE_PLANNER_RESPONSE` から `plannerResponse.thinking` と `plannerResponse.response` を抽出。
    - **claude**: `[local path]`（`CLAUDE_CONFIG_DIR` 対応）の `type == "assistant"` から `message.content` 内の `thinking` / `text` を抽出。
    - **codex**: `[local path]`（`CODEX_HOME` 対応）の `response_item` (`payload.type == "message"` / `role == "assistant"` の `output_text`、または `payload.type == "reasoning"` の `summary` 配列内 `summary_text`) および `event_msg` (`payload.type == "item_completed"` の `agent_message`) を item ID で重複排除して抽出。
    - **重要な制約 1**: `runtime.sessions(env)` の中で **`herdr` CLI の追加サブプロセス呼び出しを増やさないこと**（`test_execution_herdr_joins_parents_only_by_unique_identity` が Herdr CLI 呼び出しが `session list --json` と `--session default agent list` のみであることを厳密に検証しているため）。セッション ID の特定は `agent.get("agent_session")` と `herdr-labels/{panes,owners}[local path]` のローカル JSON ファイルおよび `AgyTui`（`[local path]`）のみで行うこと。
    - **重要な制約 2**: 会話ログ内に Herdr の `pane_id`（例: `wC:p6P`）や `terminal_id` が含まれる場合があるため、`runtime.sessions(env)` で `Guard` を初期化する際は `secrets = {**parents, **_private(rows)}` を渡すこと。また `worker_status/safety.py` の `Guard.__init__` に `env` 引数を追加し、`observe_parents` で `row.get("tasks", {})` を使うことで v2 `registry.json` や `_private(rows)` でも `KeyError: 'tasks'` が起きないようにすること。
- **ac-7（右詳細パネルの横幅リサイズとテスト・README 更新）**:
  - 右詳細パネル（`#execution-drawer`）の左端にドラッグおよびキーボード（`ArrowLeft` / `ArrowRight`）で横幅を拡大・縮小できるリサイズハンドル（`#execution-drawer-resizer`）を設ける。
  - CSS 変数 `--drawer-width`（例: 320px〜960px）で幅を保持し、10 秒ポーリング更新やノード切替でも幅がリセットされないようにする（1150px 以下のモバイル重ね表示ではリサイズハンドルを非表示にする）。
  - 関連テスト（`tests/e2e/agents_ops/board_execution.cjs`, `tests/e2e/agents_ops/board_browser.cjs`, `tests/e2e/agents_ops/test_board_web.py`, `tests/e2e/agents_ops/test_worker_status.py`）およびドキュメント（`.agents/ops/board/README.md`, `.agents/ops/worker_status/README.md`）を更新する。

## 2. 参考パッチ

親で事前検証した差分が `[local path]` と SVG アイコン `[local path]` に保存されています。必要に応じて参照・適用し、テストと README を仕上げてください。

## 3. 検証と完了手順

1. 以下のテストを実行してすべてパスすることを確認する:
   ```bash
   .venv/bin/python -m pytest tests/e2e/agents_ops/test_worker_status.py tests/e2e/agents_ops/test_board_web.py tests/unit/scripts/ci_metrics/test_lanes.py -n 4 -q
   ```
2. `ruff check`, `ruff format --check`, `mypy` を実行してすべてパスすることを確認する。
3. 変更をコミット・push し、`Closes #2366` を含む日本語の PR を作成する。
4. 最終メッセージ末尾に `PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":<PR番号>,"head":"<完全なHEAD SHA>","resume_on_success":false}}` を出力してターンを終了する。
