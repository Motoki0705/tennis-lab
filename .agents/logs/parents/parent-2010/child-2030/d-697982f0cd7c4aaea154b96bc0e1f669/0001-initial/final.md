[PR #2035](https://github.com/Motoki0705/tennis-lab/pull/2035)を作成し、対象HEADの[Python CI成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43589760151/job/132435849454)まで確認しました。

- branch: `codex/evidence-label-scope`
- commit: `ec459b7d213009334cb5e5a786677007366d4b7e`
- fake ghの要求数: status **24→14**、resolve **146→50**。全Issueページ取得は両方 **0**。
- 関連218件、追加確認9件（重複あり）、ruff・pre-commitのruff/mypyが成功。
- binding検証・並列作成・曖昧POST契約を維持し、ラベルなしmarkerの保証変更をREADMEに明記しました。

ops直接指定のmypyにはbase共通の既存15件があり、PRに記録済みです。validatorは指定なし、試行／完了 **0／0回**。

残作業は親による統合レビュー・merge・merge後mainのCI確認です。