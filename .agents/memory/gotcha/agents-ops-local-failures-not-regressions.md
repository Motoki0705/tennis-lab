---
title: "親CLIのローカルE2Eは実セッションのAGY識別子を引き継ぐと登録に失敗する"
summary: "親CLIのE2Eが識別子の重複で落ちたら、pytestの実行環境から実セッションIDを外し、同じテストを比べる。"
applies_to: all
source: "#2324 の棚卸しでの再現（2026-11-20）、以前のローカル失敗調査は PR #1621"
created: 2026-10-31
last_verified: 2026-11-20
evidence: "test_execution_basis.py の test_valid_autonomous_basis_allows_split_delegation_and_result を同じ環境で origin/main と比較する。ANTIGRAVITY_CONVERSATION_ID の有無も比較する。"
citations:
  - path: ".agents/ops/lib/identity.py"
    symbol: "def session_identity"
  - path: "tests/e2e/agents_ops/test_parent_session.py"
    symbol: "def parent_env"
  - path: "tests/e2e/agents_ops/test_execution_basis.py"
    symbol: "test_valid_autonomous_basis_allows_split_delegation_and_result"
---

親CLIはセッション識別子が一つに定まるときに登録する。親のE2E fixtureはテスト用の `CLAUDE_CODE_SESSION_ID` を入れるため、実行元の `ANTIGRAVITY_CONVERSATION_ID` も引き継ぐと、`parent register` が識別子の曖昧さを報告する。2026-11-20に作業treeと同じ版の `origin/main` で再現した。

**使い方:** このエラーが出たら、`env -u ANTIGRAVITY_CONVERSATION_ID -u CODEX_SESSION_ID -u CLAUDE_CODE_SESSION_ID .venv/bin/python -m pytest <node>` で、テストの呼び出しだけから実セッションIDを外して比較する。
