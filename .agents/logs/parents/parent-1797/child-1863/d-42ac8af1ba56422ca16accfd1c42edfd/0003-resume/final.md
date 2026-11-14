準備実装を [PR #1907](https://github.com/Motoki0705/tennis-lab/pull/1907) にまとめました。

- branch: `codex/review-eval-acceptance`
- HEAD: `7f258e95b`
- 固定7ケース、正解major 9件、隔離harness、集計、判断ログ接続を実装。
- ローカル62件とpre-commit成功。[HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/42975569043)も13ジョブ成功。失敗ジョブの再実行は1回です。
- #1790の5回はすべて `not_reviewed` で、現行復旧による省略可能件数は0件でした。

拡張mypyの既存5診断は据置で、PRに記録しています。

実LLM比較と正式レビューの実機受入は未実施です。次の実行前に親によるケース別モデル選択の固定が必要です。既定はAstra fixedを維持しています。