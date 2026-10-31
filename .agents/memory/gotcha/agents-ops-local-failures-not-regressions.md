---
title: "Claude Code配下でローカル実行した tests/e2e/agents_ops は、変更と無関係な失敗を含む"
applies_to: claude
source: "PR #1621 (2026-10-31)"
created: 2026-10-31
last_verified: 2026-10-31
evidence: "origin/main を一時worktreeで取り出し、同じテストIDを `-n 0` で実行して比較する"
---

Claude Code のセッションから `tests/e2e/agents_ops` を実行すると、CIでは合格するテストが変更と無関係に失敗する。

- プロセス祖先・hook系（`test_board_doorbell.py` の headless root hook、`test_herdr_labels.py` の lifecycle hook）は、実行プロセスの祖先に本物のエージェントがいるため決定的に失敗する。`test_agy_headless.py` の permission smoke、`test_dispatcher_supervisor.py`・`test_execution_basis.py` の一部も origin/main で同様に失敗した。
- 全体を `-n 8` で回すと30分近くかかり、`wait_until` 系（`test_merge_gate.py`・`test_parent_development.py`）が待機タイムアウトで落ちる。並列数を下げて再実行すると合格する。

**使い方:** 失敗を変更の退行と判断する前に、同じテストIDを origin/main の一時worktree（`.claude/worktrees/` 配下、終わったら削除）でも実行して比べる。全体実行は `timeout` を1時間程度にし、`-v` で失敗IDを逐次記録する。最終判断はPRのCIで行う。
