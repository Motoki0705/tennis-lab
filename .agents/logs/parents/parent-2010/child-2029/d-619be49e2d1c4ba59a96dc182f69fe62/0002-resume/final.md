[PR #2038](https://github.com/Motoki0705/tennis-lab/pull/2038) をmain `6c6e4fb82`へrebaseし、force-pushしました。

branch: `codex/registry-txn-client`  
HEAD: `e4ebe915c`

両方の契約とworker session置換を保持しました。要求数の期待値も維持しています（証拠一覧はstatus 1・resolve 3ページ、全Issue一覧0ページ）。

- 関連e2e **1236件成功**、unit **1259件成功**
- ruff・pre-commit（ruff/mypy）成功
- 対象HEADの [Python CI成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43603667016/job/132477899517)

PR本文・担当issueを更新済みです。worktreeはclean、PRはmergeableです。残作業は親の統合レビュー・merge・main CI確認です。validatorは未指定、試行0／完了0です。