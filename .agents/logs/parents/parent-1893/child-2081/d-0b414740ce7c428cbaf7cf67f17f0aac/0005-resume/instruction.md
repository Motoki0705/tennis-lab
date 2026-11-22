PR #2085（HEAD 2ced17f6）は、その後の main（2736a24a3 時点）と次の5ファイルで再び衝突している: `.agents/ops/lib/agent_runtime.py`、`.agents/ops/parent/cli.py`、`.agents/ops/review/merge_gate.py`、`.agents/ops/review/parent_review.py`、`.agents/ops/worktree_cleanup/registry.py`。main には PR #2149（#2142: resolve の確認範囲の変更）などが入っている。

前回と同じ方針で origin/main を取り込んで衝突を解消してほしい（merge commit でよい、force push しない）。
- main 側で入口ファイルの本体に入った変更は入口へ戻さず `*_impl.py` などへ移植し、全16入口が薄いbootstrapのままであることを機械的検査テストで確かめる。PR #2149 の resolve/gate の変更（review_workspace・parent_review_policy・review_inspection・review_evidence など）は parent_review_impl.py / merge_gate_impl.py 側へ漏れなく移す。
- lock の取得順（checkout → registry）と、worktree_cleanup/registry.py の main 側変更が矛盾しないことを確認する。
- 関連テスト（checkout_registry_order・interactive_checkout・git_sync・checkout_job・worktree_checkout_job・review_resolution_scope・review_inheritance・merge_gate・parent_review）と pre-commit を通し、通常の push、PR本文の更新、対象HEADのCI確認まで進める。pytest の並列は `-n 4` 以下。GitHub API は共有上限が逼迫しているので CI の確認は必要最小限にする。systemd unit 操作・git-sync のON・メインcheckoutの変更はしない。
- main は頻繁に進むので、取り込み後はすぐ push して CI を確認し、終わったら速やかに報告する。作業時間の目安は最大3時間。
