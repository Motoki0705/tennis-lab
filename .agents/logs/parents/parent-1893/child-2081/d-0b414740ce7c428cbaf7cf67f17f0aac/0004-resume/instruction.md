PR #2085（HEAD 54e9c5ce）は、その後 main に入った変更（PR #2116 merge gate の読み取り削減、PR #2054 gh の HTTP 化、PR #2056 dispatcher・triage を registry ロックの外へ、PR #2052 worktree-cleanup ほか）と次の7ファイルで衝突している: `.agents/ops/dispatcher/cli.py`、`.agents/ops/dispatcher/worker.py`、`.agents/ops/lib/agent_runtime.py`、`.agents/ops/parent/cli.py`、`.agents/ops/review/merge_gate.py`、`tests/e2e/agents_ops/test_parent_development.py`、`tests/e2e/agents_ops/test_worktree_checkout_job.py`。

親の判断: 再レビューの前に、origin/main を取り込んで衝突を解消してほしい（merge commit でよい、force push しない）。

- main 側で入口ファイル（merge_gate.py・agent_runtime.py・parent/cli.py・dispatcher など）の本体に入った変更は、入口へ戻さず、本体を移した `*_impl.py` などへ移植する。全16入口が薄いbootstrapのまま（lock取得前に実処理を読み込まない）であることを、既存の機械的検査テストで確かめる。
- main 側の lock 取得の変更（#2056 の registry ロック外への移動など）と、このPRの checkout → registry の取得順の統一が矛盾しないか確認し、循環待機の競合テスト（test_checkout_registry_order.py）を通す。
- #2116 の merge gate の読み取り削減（同一観測内での読み直しの抑止）が、本体移動後も失われていないことを、#2116 が追加したテストで確かめる。
- 関連テストと pre-commit を通し、通常の push、PR本文の更新（取り込んだ main と衝突解消の要点）、対象HEADのCI確認まで進める。pytest の並列は `-n 4` 以下。GitHub API は共有上限（5000回/時）が逼迫しているので、CI の確認は数分おきの必要最小限にする。systemd unit 操作・git-sync のON・メインcheckoutの変更はしない。作業時間の目安は最大4時間。
