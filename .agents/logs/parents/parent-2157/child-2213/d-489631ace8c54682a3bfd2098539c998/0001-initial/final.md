[PR #2214](https://github.com/Motoki0705/tennis-lab/pull/2214) を作成しました。

- branchは `codex/completion-candidate-groups`、commitは `198b541f6`。作業treeはcleanです。
- 候補見出しの解析・不正形式の拒否・snapshot変更検出を実装。IDに `ac-I-<元親>-<key>-<候補内連番>`、表示に `[候補ID] 条件文` を採用しました。
- 関連178件成功。故障注入でE2Eの失敗を確認し、復元後50件成功。ruff・mypy・pre-commitも成功しました。
- 実際の #2157 で `available=true`、合意1件＋受入12項目を確認しました。
- 対象HEADのCIは実行中です。親による `review-request` 1回とmerge後確認が残作業です。[CI実行](https://github.com/Motoki0705/tennis-lab/actions/runs/43990893689)