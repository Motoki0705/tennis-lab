PR #2640 のレビュー指摘を修正し、同じブランチに `ec474812f` を push しました。README に個別パスの `git add -f` と `git diff --cached --name-only` による確認手順をまとめ、制作スキルからリンクしました。[PR #2640](https://github.com/Motoki0705/tennis-lab/pull/2640)

`--check-tracked` は未登録の図版・表・参考文献を列挙して終了値1になり、個別にステージすると終了値0になりました。根拠資料がステージされ、`build/` の生成物が含まれていないことも確認しました。新HEADの通常ビルドはA4・2ページ、警告0件で、文書テスト40件と pre-commit も通過しました。

新HEADのCIは実行中です。完了確認を親CLIに引き渡します。[CI 実行](https://github.com/Motoki0705/tennis-lab/actions/runs/45817673161)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2640,"head":"ec474812f4168721cabfb3c85047e30329bc569a","resume_on_success":false}}