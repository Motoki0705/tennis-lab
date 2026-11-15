2026-11-15 09:20 JST 頃に WSL が shutdown され、前回のturn（親判断「関連未観測の確認不能プロセスは診断に留める」を受けた新方針の実装）が途中で止まった。プロセスはすべて消滅し、親が終了を記録済み。

worktree `.claude/worktrees/worktree-cleanup-uncertain-scope` には、前回のturnの未commit差分（references.py・test_worktree_reference_gate.py・test_worktree_references.py の新規と、processes.py・lifecycle.py・cleanup.py・README・既存テストの変更）がそのまま残っている。branch HEAD は 44865cd80 のまま。

差分を確認してから、前回の指示（新方針の実装、修正を外すと失敗することの確認、test_worktree_checkout_job.py の安定性確認、pre-commit、通常のpush、PR本文の書き直し、対象HEADのCI確認、6件の読み取りだけの診断）を最後まで進めてほしい。制約は前回と同じ（pytest -n 4 以下、本番の削除・unit操作・状態変更・シグナル送信をしない、force pushしない）。
