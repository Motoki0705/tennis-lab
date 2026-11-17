# #2175 作成workerを修正ターンで再開する

担当Issue #2175 の本文（目的・変更内容・対象外・Acceptance checklist）を正本として実装してください。親は #2157 です。#2157 の 2026-11-16 のコメントで、修正ターンは `effort=high`（同じモデル・同じsession）、上限はPRごとに3回（回数管理は #2176 の担当）と決まっています。

設計調査で確認済みの事実:
- 再開は `parent/manual_workers.py` の `prepare_run` / `run_worker`（445行付近）と、共通ランナーの `session.lock`・`latest_run` 確認（`lib/agent_runtime.py` 424行付近）を再利用できる。
- effort=high は現状3段で拒否される: `lib/agent_settings.resolve` の実装worker=max固定、`agent_runtime.run` の再開時settings一致、`manual_workers.model_settings` と `worker_logs/schema.py`（178行付近）のmax固定。
- 既存sessionへ途中から `--json-schema` を追加すると共通ランナーの保存contract一致検査に抵触するため、最終文は呼出側（pipeline）でschema検証する方式がよい。
- dispatcherの `start_child`（`dispatcher/supervisor.py` 1058行付近）は「実行中の親」を要求する。親が登録した自動進行に限った起動経路の土台を用意する（環境の偽装はしない）。本格的な接続は #2176。

要件:
- 修正ターンだけを例外にし、通常の実装・調査の max 固定と同一session内の設定一致は維持する。
- 同じ修正試行IDの再実行で二重起動しないこと。
- AGENTS.md と parent-session skill の該当記述を、この例外だけ追記する形で更新する（二重管理しない）。
- 静かなフォールバックを作らない。
- テストは tests/README.md の方針で設計してから書く。
- 完了したら PR を作成する（本文は日本語、`Closes #2175`）。CIの結果は待たずに報告してよい。
