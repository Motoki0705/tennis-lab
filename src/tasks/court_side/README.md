# court_side

各cameraのcamera-localコートが、reference cameraのコートに対して180度 half-turn しているか（side）を、
**多視点のball観測だけ**を証拠とする幾何的な仮説検定で決める。学習モデルは使わない。

出力先の規約は[タスク出力規約](../OUTPUTS.md)を参照。

## なぜballだけか

- CourtKP14とコートラインは `Rz(π)` に対して完全に対称で、camera-local校正は常にcameraを `-Y` 側に置く。
  1台のcameraだけではsideは原理的に決まらない。
- ballは1個しかないので、camera間の対応付けが要らない。人物を証拠にすると人物対応（#933）が必要になり、
  その人物対応がsideを使うため循環する。
- コートの対称軸（ネット中央の真上、`x=y=0`）上のballはどの仮説でも同じに見えるので、判定の情報を持たない。

## 判定（`hypothesis.py`）

reference cameraを固定し、他のcameraごとに「そのまま／half-turn」の `2^(V-1)` 通りの仮説を列挙する。
各仮説について、ballが2 view以上に写ったframeごとに全観測viewのDLTで三角測量し、
`src/utils/geometry/multiview_consistency.py` で次を計算する。

| 量 | 定義 |
|---|---|
| cost | frameごとの `min((再投影誤差/閾値)^2, 1)` の観測view平均（物理的に不可能な点は1）をframe平均したもの |
| support | 物理的に妥当で、全観測viewが閾値以内のframeの割合 |
| margin | 次点の仮説のcost − 最良の仮説のcost |

物理的な妥当性は、DLTが非退化、全観測cameraの前方、1度以上離れた光線の組がある、`|x|,|y|≤40 m`、高さ `-0.2〜20 m`。
DLTは外れviewを除外しない。誤った姿勢のcameraを多数決で無視させないためで、再構成用の
`triangulation.triangulate_points`（inlier選択あり）とは目的が違う。

次の場合は `CourtSideUndecided` で停止し、理由・全仮説のscore・camera対ごとの共通frame数を持つ。別の判定手段へは切り替えない。

| 理由 | 条件 |
|---|---|
| `insufficient_frames` | 2 view以上に写ったframeが `min_frames` 未満 |
| `disconnected_views` | 共通frameが `min_frames` 以上のcamera対のグラフで、referenceにつながらないcameraがある |
| `no_consistent_hypothesis` | 最良の仮説が `cost > max_cost` または `support < min_support` |
| `ambiguous_margin` | `margin < min_margin` |

閾値の正本はpipelineの設定 `court_side:`（[pipeline.yaml](../../tennis_scene/configs/pipeline.yaml)）。
pipeline component は[tennis_scene pipeline](../../tennis_scene/pipeline/README.md)を参照。
