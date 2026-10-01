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
まず、直前に残したframeと観測viewの組が同じで、どのviewも `min_motion_px` 未満しか動いていないframeを除く。
静止した誤検出（ボールボーイが持つball、ballに似た模様）が何十frameも同じ証拠として数えられ、誤った仮説を支持するのを防ぐ（合成ベンチマークで観測した失敗）。
残ったframeのうち、ballが2 view以上に写ったframeごとに全観測viewのDLTで三角測量し、
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
| `insufficient_frames` | 2 view以上に写った（重複を除いた）frameが `min_frames` 未満 |
| `disconnected_views` | 共通frameが `min_frames` 以上のcamera対のグラフで、referenceにつながらないcameraがある |
| `no_consistent_hypothesis` | 最良の仮説が `cost > max_cost` または `support < min_support` |
| `ambiguous_margin` | `margin < min_margin` |

閾値の正本はpipelineの設定 `court_side:`（[pipeline.yaml](../../tennis_scene/configs/pipeline.yaml)）。
pipeline component は[tennis_scene pipeline](../../tennis_scene/pipeline/README.md)を参照。

## 合成ベンチマークと閾値の選定（`benchmark.py`）

BLCSの合成rally（`blcs/single_object_camera_view_v2` のtest split、物理cameraが既知、30 fps）で、
cameraとreferenceをランダムに選び、`camera_view_v2` のcamera-local校正（ネットの向こうのcameraはlocalにhalf-turn）を作る。
校正の摂動（回転・焦点距離・位置のGauss雑音）と、観測の摂動を条件ごとに与え、閾値に依存しない証拠を保存する。

| 条件 | 内容 |
|---|---|
| 欠落 | view・frameごとに独立に観測を落とす |
| 誤検出 | 各cameraのwindowの一定割合を、静止した偽のballの区間で置き換える（画像内の固定点、またはコート外で人が持つballの投影） |
| 共有された誤検出 | 全cameraが同じframeで同じ3D点（ボールボーイのball）を検出する |
| 同期ずれ | reference以外の1台を数frame遅らせる |
| その他 | pixel雑音、校正雑音の倍率、windowの長さ（frame数）、camera 4台 |

名目の校正雑音は、Meiji clip_000の確認済みballで測った正解仮説のcost（0.10）に中央値が近くなるように決めた。
閾値は、本番と同じ `judge_side_evidence` で格子上の全点を判定して選ぶ。自身と、各閾値を1段緩めた近傍のすべてが
全条件で誤判定0の点（1段の余裕を持ち、格子の緩い端には乗らない）のうち、条件平均の停止率が最小の点を採る。
選定に使っていないsceneと乱数seedで、選んだ閾値を `--fixed-thresholds` で再評価して確認する。
各runは同じ摂動の観測を、以前の方式（Meiji 1clipで決めた閾値、重複除去なし）でも判定して比較する。

```bash
.venv/bin/python -m src.tasks.court_side.scripts.benchmark_synthetic \
    --data-root /absolute/data --output-root /absolute/outputs --experiment synthetic_blcs_v2 --run-id <run-id> \
    [--scenes 600 --scene-offset 0 --seed 0] [--fixed-thresholds <selection run>/report.json]
```

出力は `court_side/evaluate/<experiment>/<run-id>/` の `conditions.json`、`evidence.jsonl`、`thresholds.json`、`report.json`。
結果と採用した閾値の根拠は `knowledge/` のrun記録にある。
