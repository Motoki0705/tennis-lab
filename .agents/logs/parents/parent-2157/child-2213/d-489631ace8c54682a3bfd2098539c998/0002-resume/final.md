複数行の受入条件への対応を修正し、同じbranch `codex/completion-candidate-groups` へpushしました。[PR #2214](https://github.com/Motoki0705/tennis-lab/pull/2214)

- 新HEADは `fc21f448b7fdb351661e0911e5889f3add00d685`。
- 同じグループの項目後にある字下げ行を連結し、項目前の自由文・別見出し・結合後の重複を拒否します。
- completion関連83件が成功。修正除去で拡張E2Eの失敗を確認し、復元後も成功しました。ruff・mypy・pre-commitも成功しています。
- CIはpendingです。親による `merge-continue` とmerge後確認が残っています。[対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43993453127)