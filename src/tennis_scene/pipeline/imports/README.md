# 確認済みデータのimport（暫定ユニット）

モデル実装が無い、または実動画で未合格のcomponentの出力を、人手で確認したデータから同じ出力schemaで公開する。
import方針の正本はこのREADMEで、利用者は[実clip qualification](../../../../tests/benchmarks/component_pipeline.py)だけ。
置き換えるモデルが入った時点で、該当ファイル・本READMEの行・benchmarkの呼び出しを削除する。
[可視化](../../scripts/visualize_component_store.py)はartifactの`provenance.origin`を表示するだけで、import経路を特別扱いしない。

| ファイル | 公開先node | 入力 | 置き換える予定 |
|---|---|---|---|
| `ball_annotations.py` | `ball_detection/<camera>` | 外注の`video_ball_annotation.v2`（`<clip>/outsource/<camera>_annotations.json`） | ball検出・2D refiner（#934/#935） |

## 規則

- 公開は`publish.py`の`bind_import`→`publish_import`だけを通す。対象nodeは`execution.<node>=load`で宣言済みであること。
  依存はnodeのbindingsから、storeの採用版を解決して記録する。上流が差し替わるとrunnerがloadを停止する。
- 公開artifactの`provenance.origin`は`import`、`model_inference`は`false`。identityには入力ファイルのSHA-256と判定閾値を含める。
- ballは`observed`点だけを観測にする。補間・遮蔽推定の座標は`point_kind`付きで保持し、confidenceは受理の0/1で確率ではない。
