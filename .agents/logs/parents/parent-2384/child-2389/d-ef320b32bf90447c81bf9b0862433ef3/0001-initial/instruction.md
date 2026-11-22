親Issue #2384 配下の子Issue #2389 の実装担当として、専用worktreeで実装・検証・PR作成まで行ってください。

## 対象Issueと目的
- 担当子Issue: #2389（親 #2384）
- 目的: 採用した手動worker起動・待機系の改善提案2件を実装し、親AIが安全・確実に手動Codexワーカーを起動・復旧・待機できるようにする。

## 対象2項目と事前調査箇所（最新コードを再確認すること）
1. **`I-2157-unconfirmed-launch-recovery` 残存部（元提案 #2219）**:
   - 箇所: `.agents/ops/parent/manual_workers.py`（`prepare_run`, `run_worker` L800-L1070付近）・`improvement_runtime.py`・`cli_impl.py`・`cli_help.py`。
   - 現状確認: `admission.spawned == False`（Popen 前の拒否）の自動ロールバック＋`worker_logs.reject_unstarted` は既存コード（L977-L1054）に実装済み。しかし以下 2 点が未実装のまま：
     a. **起動前環境検査**: `agent_exec.sh` は `$lib_dir/../../../.venv/bin/python` を `exec` するが、worktree 等に `.venv/bin/python` が存在しない（または実行不可・ランナー不在）状態で `run-worker` を呼ぶと Popen 後に exit 127 で即死し、run dir 未作成の `runtime_unconfirmed` 予約が残る。Popen 前にランナー実行環境（`RUNNER` および対応する `.venv/bin/python` 等の実行可能性）を検査し、不備があれば理由付きで即拒否して予約や捕捉ログの残骸を残さないこと。
     b. **Popen 後の未起動確認済み予約の明示取消**: Popen 後でも run dir が作られておらず、かつ launcher / worker プロセスが存在しない（終了・消滅を確認できた）起動について、親が明示操作（例: 専用コマンドまたはオプション）で取り消し、同じ委任で再起動でき、対応する worker ログ捕捉も `worker_logs.reject_unstarted` 等で未起動として終結し公開待ちに残らないようにする。
   - 重要制約: 起動済みか不明な場合（プロセス生存不明・run dir 存在時）の取消や別 worker での埋め合わせは従来どおり厳格に禁止すること。
2. **`I-2157-wait-worker-new-run`（元提案 #2219）**:
   - 箇所: `.agents/ops/parent/worker_wait.py`（`target`, `wait_worker`）・`cli_impl.py`・`cli_help.py`。
   - 現象: `wait-worker` は呼び出し時点で `runs[-1]` を即 pin するため、`run-worker` を背景起動した直後に `wait-worker` を呼ぶと、新 run が記録される前は前回 run の終了結果または `no recorded run` が返る。
   - 要件: `wait-worker` で待つ対象の run（または前回 run より後の新 run、例: `--after-run <前回run ID/ディレクトリ>` / `--run-id` 等、既存設計と最も整合する明確なオプション）を指定でき、新しい run がまだ記録されていない状態で呼んでも前回 run の結果を返さず、新 run の記録出現から終了・回収（手動委任の followup 含む）まで待てるようにする。タイムアウト（終了コード 124）も適切に機能すること。

## 受入条件（#2389 の Acceptance checklist をすべて満たすこと）
1. worktree に `.venv` 等が無い状態の `run-worker` が Popen 前に理由付きで拒否され、予約やログ残骸を残さないテスト。
2. run dir 不在・launcher 消滅の起動を親の明示操作で取り消し、同じ委任で再起動でき、worker ログ捕捉が未起動として終結して公開待ちに残らない E2E テスト（および run dir あり・生存不明では拒否されるテスト）。
3. `run-worker` 起動直後の `wait-worker` が前回 run の結果を返さず、新 run 記録前から呼んでも記録後の終了・回収まで待つ E2E テスト（初回 run 記録前・2回目以降の `--resume` 双方および timeout を確認）。
4. 必要最小限の CLI help・運用文書（`.agents/skills/parent-session/references/workers.md` / `recovery.md`）を更新し、既存テスト・新規テスト・ruff・mypy がすべて通る。

## 検証・PR作成・終了ルール
- 関連テスト（`tests/unit/agents_ops/`, `tests/e2e/agents_ops/` の manual_workers / worker_wait / worker_logs 関連）＋新規テストを全通過させる。
- pre-commit（ruff / mypy）を通してコミット・push し、PR 本文に `Closes #2389`、変更概要、受入条件ごとの検証証拠、テスト結果を日本語で記載する。
- push・PR作成が終わったら、このターン内で CI 待ちループを行わず、最終メッセージ末尾に以下の形式で 1 行だけ出力して終了する（`<PR番号>` と `<40桁HEAD SHA>` は実際の値に置換）：
`PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":<PR番号>,"head":"<40桁HEAD SHA>","resume_on_success":false}}`
