#2432 の実装を push し、`Closes #2432` の紐付けと CI の起動を確認しました。 [PR #2478](https://github.com/Motoki0705/tennis-lab/pull/2478)

手動クローズと PR マージを模擬し、紐付けを確認できた証拠用 Issue だけが閉じること、閉じた後も同じコメントを permalink と親 CLI の履歴から読めることを確認しました。関連テスト147件と pre-commit が通過しました。

実 GitHub の dry-run では、対象24件と親が open の4件を確認しました。再実行の結果は一致し、対象番号と照合根拠を PR に記載しています。実 GitHub での両経路の確認と、既存分への `--apply` の判断はマージ後に残っています。

ブランチは `codex/parent-2474-issue-2432-primary`、HEAD は `bdc429fd04a4ff7fabc89f86da1a7a9031f30ed9` です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2478,"head":"bdc429fd04a4ff7fabc89f86da1a7a9031f30ed9","resume_on_success":false}}