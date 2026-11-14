# PR #2038 を最新 main へ rebase する

子B（#2030）の PR #2035 が main へmergeされました（merge commit 6c6e4fb8291ab6d9c488f0c21e572fc190ff6dbc）。#2035 は次を変更しています。

- `parent/github.py` の `evidence_rows()`: 一覧を `issues?state=all&labels=parent-evidence` に限定。history scope 内では `_fresh` で一覧を捨て、取り直しは `fresh_reads()` で行う。
- `parent/evidence.py` の `find()` の docstring、`parent/cli.py` の親CLI transaction に `gh.history_read_scope()` を追加（worker系コマンドは除外）。
- `review/parent_review.py` の6入口に `history_read_scope()` を追加。
- README の機械証拠の節（ラベルの無いmarker付きIssueは探索対象外という保証範囲の変更）。

## 依頼

1. `codex/registry-txn-client` を `origin/main` へ rebase し、衝突を解消する。両方の契約を保つこと: #2035 のラベル限定探索と保証範囲、#2038 の transaction用/ロック外client の分離、GET cache の寿命、「探索索引は明示した1観測内だけ再利用」。#2035 が親CLI・review入口に掛けた `history_read_scope()` が、#2038 の観測境界（独立した観測は cache を捨てる、書込前照合は fresh）と矛盾する場合は、#2038 の観測単位に合わせて整理し、どちらの退行テストも通ること。
2. #2035 で追加された `tests/e2e/agents_ops/test_parent_evidence_discovery.py`（要求数の比較を含む）と、#2038 の新規テスト・関連e2eを rebase 後のHEADで実行する。
3. force-push し、対象HEADの CI（Python）の成功を確認する。PR本文の検証欄を rebase 後のHEADに合わせて更新する。

時間の上限は3時間。止めてよいのは、二つの契約が両立しないと判明し親の判断が要る場合だけ。その場合は衝突箇所と選択肢を報告する。
