[PR #1786](https://github.com/Motoki0705/tennis-lab/pull/1786) をmainへ載せ替え、baseも`main`へ変更しました。

- branch: `codex/explain-html-visuals`
- HEAD: `0a7220d5e52351b72e6b98a3c85d70faa142393e`
- 差分: 本PRの4 commit・15ファイルのみ。旧3 commitの内容不変を確認済み。
- 検証: **51 passed / 6.17秒**、テスト増分0。Ruff・mypy・pre-commit成功。
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/42563851978): **Python・Lint・CPU全10本成功**。

初版のみ画像生成し、閲覧中新規画像は理由付き未完了・現行版保持とする確定方針を、README 3言語・adapterの理由文・PRのAcceptance表へ反映しました。追加の人間判断はありません。

worktreeはcleanです。統合レビュー・merge・merge後main CI確認は親へ引き継ぎます。