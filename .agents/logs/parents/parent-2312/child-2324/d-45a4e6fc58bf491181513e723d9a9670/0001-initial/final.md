共有memoryを4層化し、#2324を閉じる非ドラフトPRを作成しました。 [PR #2333](https://github.com/Motoki0705/tennis-lab/pull/2333)

- ブランチ `codex/parent-2312-issue-2324-primary`
- HEAD `97c0e43a7354a00372cdbf55a985d7b3a8681692`
- 旧 `review/` の3件を `decision/` へ移し、重複・陳腐化した8件を削除しました。残した26件に要約と計44件の構造化参照を整備しました。WSL2とClaude Codeの観察3件は元の検証日を維持しました。
- `validate`、`sync-index --check`、`audit` が通過しました。期限超過・参照切れ・上限超過は0件で、SUMMARYは41/80行、4,480/8,192バイトです。
- 指定のpytestは395件すべて成功し、Ruff・mypy・pre-commitも通過しました。`.agents/ops/review/` と `.agents/memory/ops-report/` の差分は0件です。

PRの `Python` CI結果の確認を親へ引き継ぎます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2333,"head":"97c0e43a7354a00372cdbf55a985d7b3a8681692","resume_on_success":false}}