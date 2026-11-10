[PR #1742](https://github.com/Motoki0705/tennis-lab/pull/1742)を作成しました。v3契約、reader/writer、利用判定・集計API、v2移行対応表を実装済みです。

- branch: `codex/ball-annotation-v3`
- commit: `8e6325b9b`（worktree clean）
- 検証: 49件成功、pre-commit成功、対象HEADの[CI・Python成功](https://github.com/Motoki0705/tennis-lab/actions/runs/42485851452)
- 画像外座標は保存時に拒否する設計です。

scene・dataset・worker接続と実データ移行は後続対象です。未解決の判断事項はありません。統合レビュー・merge・merge後mainのCI確認を親へ引き継ぎます。