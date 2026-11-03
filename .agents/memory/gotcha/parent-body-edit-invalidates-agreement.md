---
title: "親issue本文の進捗4節以外を書き換えると、人間の合意と出済みのレビュー要求が無効になる"
applies_to: all
source: "#1554 の PR #1668 の明示レビュー（2026-11-03）"
created: 2026-11-03
last_verified: 2026-11-03
evidence: "`.agents/ops/parent/execution.py` の `PROGRESS_SECTIONS` と、`requirements_sha256` を本文digestと比べる箇所（`human agreement requirements changed`）"
---

合意（`agree`）は、親issue本文から進捗の4節（子タスクのチェックリスト・進捗概要・成果物・関連PR・次の改善候補）を除いた部分のdigestを記録する。親が「前提」「受け入れ条件」などの他の節に経過や補足を書き足すと、digestが変わる。その後の `review-request`・レビュー実行・委任は `human agreement requirements changed` で止まり、checking → agree の再合意を求められる。再合意すると、実行中の委任も使えなくなる（[[reagreement-invalidates-delegation]]）。

**使い方:** 作業中の経過や補足は `note` か進捗4節に書き、他の節は人間と要件を変えるときだけ書き換える。うっかり書き換えたら、再合意する前に本文を元に戻す。digestが合意時点に戻れば、止まった要求はそのまま通る。
