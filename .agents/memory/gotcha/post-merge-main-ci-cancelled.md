---
title: merge commit の main CI が開始前に取り消されるとマージ後確認が進まない
summary: 後続pushでmerge commitのmain CIがジョブ開始前にcancelledになると、Pythonのcheck runが無くci_python_missingのまま待ち続ける。gh run rerunで同じSHAのCIを走らせる。
applies_to: all
source: '2026-11-26 #1722 の PR #2657 のマージ後確認（親セッションの観察）'
created: &id001 2026-11-27
last_verified: *id001
evidence: merge-status の main_ci.reasons が ci_python_missing のまま、merge commit の push CI run が jobs 0件の cancelled になっていることを gh run view で確かめる。
citations:
- path: .agents/ops/review/merge_gate_policy.py
  symbol: def python_ci_errors
- path: .agents/ops/review/progression.py
  symbol: def post_merge
---

- マージ後確認は、merge commit のSHAでの push CI の `Python` check run の成功だけを見る。子孫commitのCIが成功していても代わりにならない。
- main の CI は同じ concurrency group なので、直後に別のPRがマージされると、merge commit の run がジョブ開始前に `cancelled` になることがある。このとき check run が1件も無く、理由は `ci_python_missing`（pending 扱い）のまま CI のタイムアウトまで進まない。
- `gh run view RUN --json jobs` でジョブが0件なのを確かめ、`gh run rerun RUN` で同じSHAの CI を走らせる。成功すれば controller がそのまま `completed` にする。
- 逆に、merge commit の CI が実際に失敗した（`main_ci_failed`）場合は、後続の修正PRで main が戻っても、そのrunは `completed` にならない。再実行しても同じ結果になるので、原因と回復のPR・commitを `note` に残す。
