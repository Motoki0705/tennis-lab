単一ユーザー専用グループと確認できた上位ディレクトリの `0775` を許可する修正を、`codex/parent-2384-issue-2019-primary` の `958a940be` として push しました。アカウント情報を照合できない場合は `shared_write` で拒否します。`Closes #2019` を維持し、PR本文に修正と検証結果を追記しました。[PR #2399](https://github.com/Motoki0705/tennis-lab/pull/2399)

実機の `0750 → 0775 → 0775 → 0755 → 0700` の経路で、`parent-sessions` と `dispatcher` の状態ディレクトリを開けました。関連テスト567件が通過し、追加・更新した9件は `umask 0002` でも通過しました。pre-commit の Ruff・mypy も通過しています。

新HEADのCIは親CLIに引き渡します。実ホストの systemd journal は未検証です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2399,"head":"958a940be66b0c9031ea36a50e7a100aa8458f40","resume_on_success":false}}