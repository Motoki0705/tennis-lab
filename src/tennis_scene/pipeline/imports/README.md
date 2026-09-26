# 確認済みデータのimport（暫定ユニット）

モデル実装が無い、または実動画で未合格のcomponentの出力を、人手で確認したデータから同じ出力schemaで公開する。
import方針の正本はこのREADMEで、利用者は[実clip qualification](../../../../tests/benchmarks/component_pipeline.py)だけ。
置き換えるモデルが入った時点で、該当ファイル・本READMEの行・benchmarkの呼び出しを削除する。
[可視化](../../../../scripts/visualize_component_store.py)はartifactの`provenance.origin`を表示するだけで、import経路を特別扱いしない。

| ファイル | 公開先node | 入力 | 置き換える予定 |
|---|---|---|---|
| `ball_annotations.py` | `ball_detection/<camera>` | 外注の`video_ball_annotation.v2`（`<clip>/outsource/<camera>_annotations.json`） | ball検出・2D refiner（#934/#935） |
| `court_side.py` | `court_side` | 同nodeへbindされた校正とball artifact | side推定（#932） |
| `person_association.py` | `player_association` | 旧手動対応（`annotations/player_association_result.json`）と旧GVHMRのbbox軌跡（`annotations/gvhmr_result_<camera>.json`） | 人物対応（#933） |

## 規則

- 公開は`publish.py`の`bind_import`→`publish_import`だけを通す。対象nodeは`execution.<node>=load`で宣言済みであること。
  依存はnodeのbindingsから、storeの採用版を解決して記録する。上流が差し替わるとrunnerがloadを停止する。
- 公開artifactの`provenance.origin`は`import`、`model_inference`は`false`。identityには入力ファイルのSHA-256と判定閾値を含める。
- ballは`observed`点だけを観測にする。補間・遮蔽推定の座標は`point_kind`付きで保持し、confidenceは受理の0/1で確率ではない。
- sideは、基準cameraを固定した全half-turn仮説を、bindされたballだけで`camera_alignment`と同じ幾何検定にかける。
  一意に支持されなければ停止する。モデル精度の評価ではない。
- 人物対応は、旧GVHMRのplayer軸ごとに、観測30frame以上の現pose carrierのうちbbox中心距離（旧box sizeで正規化）の
  中央値が0.25以下、かつ次点と0.5以上離れた1本を照合する。候補が無い・一意でない場合は停止する。
  手動対応が選手に割り当てた軸のcarrierだけがplayer IDを持ち、他は`-1`として除外理由（未割当の旧軸、対象外、観測不足、未観測）を記録する。
  carrier順は`gather_people(...).select_views(校正済みcamera)`。
