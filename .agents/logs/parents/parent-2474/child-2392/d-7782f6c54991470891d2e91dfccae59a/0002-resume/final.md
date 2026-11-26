レビュー指摘を修正し、同じブランチに push しました。 [PR #2477](https://github.com/Motoki0705/tennis-lab/pull/2477)

Herdr の応答を使用前に検証し、`tokens: null`、前景プロセス情報の欠落、不正な JSON 構造などを `HerdrError` として扱います。追加した7ケースすべてで、hook が終了値1と固定診断1行だけを返すことを確認しました。README の追記位置を移し、PR #2478 とのマージ試算でも競合が解消しました。

関連 E2E は496件通過、Playwright が必要な1件はスキップされました。pre-commit も通過しました。異常応答はテスト用の Herdr で再現し、新しい HEAD の Python CI 確認を親CLIに引き渡します。

ブランチ `codex/parent-2474-issue-2392-primary`  
HEAD `cd5537969f7ae332e186030ac5e9b32d037d0f2f`

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2477,"head":"cd5537969f7ae332e186030ac5e9b32d037d0f2f","resume_on_success":false}}