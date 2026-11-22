[PR #2371](https://github.com/Motoki0705/tennis-lab/pull/2371) の CI 失敗を修正しました。手動 worker の E2E テストに E: 相当の出力配置を用意し、2つの worker が別々の出力先を作ることと、重複起動が拒否されることを確認しました。

失敗ケースは修正前に再現し、修正後は関連テスト125件と pre-commit の ruff・mypy が通過しました。`codex/parent-2361-issue-2368-primary` に push 済みで、新しい HEAD の Python CI 結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2371,"head":"b8119e7a658ee4d0e011d51ea1f7deb7cfa3b316","resume_on_success":false}}