実装し、[PR #1888](https://github.com/Motoki0705/tennis-lab/pull/1888) を作成しました。未mergeです。

- branch: `codex/review-model-selection`
- commit: `8bd2f159f`
- 選択をrequest v3へ固定し、実効値・復旧・公開・gateの照合を実装。既定fixedを維持。
- 新規45ケース、既存の対象テスト、pre-commitが成功。[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42940735588)も13ジョブすべて成功、再実行なし。

実モデル比較とselect切替は #1863 に残しています。統合レビューとmergeを親へ引き継ぎます。