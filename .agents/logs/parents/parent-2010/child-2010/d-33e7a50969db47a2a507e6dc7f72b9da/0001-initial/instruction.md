# 今回のターン: 調査と報告のみ（コード変更・PR作成はしない）

担当Issue #2010 の「全体の完了条件」1項目目にあたる全数調査を行い、修正計画を報告してください。実装は親が報告を読んで指示してから行います。

## 調査してほしいこと

1. `.agents/ops/` 全体で `Store.locked()`（registry.lock、LOCK_EX）を保持している区間をすべて列挙し、その区間内で次が起き得るかを呼び出しを辿って判定する。
   - GitHub通信（`GitHub.get/write/graphql`、`gh` の subprocess 直接実行、`gh pr view` 等）
   - 共通ランナー・git・pytest など外部コマンドの実行、ファイル全走査、sleep/待機
   - 各箇所について: ファイル:行、どのコマンド/入口から到達するか、`GitHub(..., store=store)` が渡っていて `unlocked_read()` で通信中にロックが外れるか、外れない場合の最悪保持時間の見積もり（`issues?state=all` 全件ページング等）。
2. Issue本文に挙がっている `parent_review.py` 6箇所の `GitHub(request["repo"])` について、`store=store` を渡した場合の副作用を検証する。特に:
   - `unlocked_read()` 復帰時の競合検出（取得前後で担当・generation・tasks・reviewsが変われば終了値2）が、レビューCLIの各コマンドで正しく働くか。レビュー自身が保存する記録（reviews等）で自己競合にならないか。
   - `store.focus()` との整合、例外パスで古いregistryを保存できない契約が守られるか。
   - `collect()` 内などで `GitHub` クラスを経由しない外部通信があれば、それもロック外へ出せるか。
3. `dispatcher/`（cli/worker/supervisor/judgement 等）、`ops-report/opsreport/triage.py`、`review/merge_gate.py`、その他 `Store.locked()` を使う全呼び出し元の同種問題。
4. `status` など読み取り専用コマンドが LOCK_EX を取る現状のボトルネック評価。`unlocked_read()` 徹底だけで足りるか、共有ロック・snapshot読み取りへの変更が必要かについて、根拠付きの意見（この変更は人間判断が必要なので、採否の材料を出す）。
5. `issues?state=all` 全件ページング（`EvidenceIssues.find()` → `evidence_rows()`）を毎回踏む経路と、`history_read_scope` / `_cache` / ローカルキャッシュで回数を減らせるか。
6. 可能なら、fake `gh` で通信を遅延させ、別プロセスが registry.lock を取れるかを確かめる小さな再現（現状の parent_review.py で取れないことの確認）。再現スクリプトはworktree内の一時ファイルに置き、コミットしない。

## 報告の形式

- 発見した箇所の表（ファイル:行 / 入口 / 保持中に走る処理 / 現状ロックが外れるか / 深刻度 / 修正案）。
- 確定バグと、設計判断が必要な点（人間判断が要るもの）を分けて書く。
- 修正計画: 変更するファイル、方針（既存の `Store.unlocked_read()` / `Store.operation()` と整合、静かなフォールバックや競合無視は入れない）、追加する回帰テスト（fake gh・並列ロック競合テストなど、`tests/e2e/agents_ops/` の既存fixtureを再利用できるか）、1PRで収まるか分割すべきかの見立て。
- 時間の目安: 調査は最大3時間。全数調査の表がそろい、修正計画が書けた時点で終了してよい。途中で止める場合は、未調査の範囲を明記する。

## 親が観測した追加事象（調査5の材料）

2026-11-14 08:47 UTC、親の `notify` が `repos/Motoki0705/tennis-lab/issues?state=all&per_page=100&page=10` の取得中に HTTP 403（API rate limit exceeded, core 0/5000）で失敗した。複数の親・dispatcherが各コマンドで証拠用Issueの探索のために全Issueを10ページ以上ページングしていることが、ロック保持時間だけでなくレート制限の枯渇にも効いている可能性がある。1コマンドあたりのGitHub要求数（`PARENT_METRICS_FILE` で計測できる）も見積もりに含め、全件ページングを避ける手段（証拠用Issue番号のローカルキャッシュとGitHub応答での照合など、既存の「タイトルや番号だけで信用しない」契約と両立するもの）を修正案に含めること。調査中のGitHub API使用は最小限にし、fake gh による再現を優先する。

## 親が観測した追加不具合（調査対象に含める）

初回の run-worker は、`improvement_runtime.execute_manual` の admission（`store.locked()` 内）で `ParentSession(...)` が `gh api GET user` を呼んだところ、レート制限の HTTP 403 で例外になった。この例外（`opsproc.CommandError` 系で `ValueError`[local path]`OSError` ではない）は `execute_manual` の except も `run_worker` の起動前拒否の巻き戻しも通らず、run dir の無い run 記録だけが worker_runs に残った。そのため再実行は「worker already has a run」で拒否され、親は record-worker-ended と再委任で手動復旧した。

- GitHub通信の失敗（レート制限・ネットワーク断）が、ロック境界の内側や起動前の admission で起きたときに、状態を中途半端に残す経路を洗い出す。
- この経路を修正対象に含めるかどうかも、修正計画に書く。
