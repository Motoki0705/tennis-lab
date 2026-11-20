[PR #2340](https://github.com/Motoki0705/tennis-lab/pull/2340) の既存ブランチに修正を push しました。初回は `paths-ignore` の完全一致テストへの反映が漏れ、CI の16件が失敗していました。

除外の検査を既存の設定テストに統合し、共有memory単独・`ops-report`・コード混在の起動条件をイベントごとに確認しました。設定テスト33件、関連テスト470件、Ruff、mypy、pre-commit が通過しました。

新しい HEAD は `4d1e5d580c2f10994690d44d77d8b2d1b309b873` です。この HEAD の `Python` CI は親へ引き継ぎます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2340,"head":"4d1e5d580c2f10994690d44d77d8b2d1b309b873","resume_on_success":false}}