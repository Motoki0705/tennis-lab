全4層の共有memoryを定期統合する AutoDream と、日次の `memory-autodream` ターゲットを実装しました。[PR #2346](https://github.com/Motoki0705/tennis-lab/pull/2346)

- ブランチ `codex/parent-2312-issue-2326-primary`、HEAD `f4f2231722ad5e391f358e64771c8d3b8420175e`
- pytest 603件が通過し、単発起動ケース追加後の `ops_control` も62件通過しました。共有memory検査、Ruff、mypy、pre-commit も通過しています。
- 初回は経過時間の条件を満たした扱いとし、`main` の first-parent 全履歴に件数条件を適用します。既存9ファイルの整形は別コミットに分けました。

CI は起動済みです。`Python` の結果確認を親へ引き継ぎます。実機の timer 有効化はマージ後に人間が行います。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2346,"head":"f4f2231722ad5e391f358e64771c8d3b8420175e","resume_on_success":false}}