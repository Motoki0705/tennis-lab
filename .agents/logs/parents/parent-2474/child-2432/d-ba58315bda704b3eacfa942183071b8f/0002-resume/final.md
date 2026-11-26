正式レビューの major 指摘を修正し、同じブランチ `codex/parent-2474-issue-2432-primary` に push しました。現 HEAD の CI は実行中です。 [PR #2478](https://github.com/Motoki0705/tennis-lab/pull/2478)

ラベル付き一覧と全 Issue 一覧の両方で一意な紐付けを確認してから閉じます。候補なしや一覧の不一致は終了コード `3` で報告し、ワークフローが同じ親を最大2回再照合します。

回帰テスト3件を追加し、関連150件、pre-commit、actionlint が通過しました。実 GitHub の dry-run では対象25件と親が open の3件を確認し、根拠を PR に更新しました。実 GitHub での自動クローズの確認と一括適用はマージ後に残っています。

HEAD は `c4c9143bf1af163bb9b23179a55a9815ea17c603` です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2478,"head":"c4c9143bf1af163bb9b23179a55a9815ea17c603","resume_on_success":false}}