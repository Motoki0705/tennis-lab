#2449 の最終版を含む `origin/main` を作業ブランチへマージし、pushしました。PRのbaseは `main`、GitHubの競合判定は `MERGEABLE` で、レビュー可能な状態です。本文の先頭は `Closes #2459` です。 [PR #2461](https://github.com/Motoki0705/tennis-lab/pull/2461)

mainの実ソースと照合し、完全一致時は16条件すべてでキャッシュidentityが一致しました。1ファイルずつ変更した128ケースでは、すべて変更がidentityに反映されました。関連テスト54件、pre-commit、`--base-ref origin/main` を指定したknowledge検査が通りました。`tennis_scene/.sequence` は `100` です。

ブランチ `codex/parent-1739-issue-2459-primary`  
新HEAD `997d9ad094804cadc5d0e316296095062fdf2814`  
[CI実行中](https://github.com/Motoki0705/tennis-lab/actions/runs/45796477978)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2461,"head":"997d9ad094804cadc5d0e316296095062fdf2814","resume_on_success":false}}