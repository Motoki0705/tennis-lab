# court_side

各cameraのcamera-localコートが、reference cameraのコートに対して180度 half-turn しているか（side）を、
多視点の観測に基づく幾何で決める。`court_side.evidence` は `ball`（既定）と `player` を選べる。

出力先の規約は[タスク出力規約](../OUTPUTS.md)を参照。

保存済みの同期観測・校正・side診断は[dataset review画面](review/README.md)で確認できます。
専用の学習datasetはなく、既存component storeがレビュー対象です。

## 観測と座標

- CourtKP14とコートラインは `Rz(π)` に対して完全に対称で、camera-local校正は常にcameraを `-Y` 側に置く。
  1台のcameraだけではsideは原理的に決まらない。
- ballは1個しかないので、camera間の対応付けが要らない。プレーヤーを使う場合は、同時刻に写った足元の点集合を比べる。
- コートの対称軸（ネット中央の真上、`x=y=0`）上のballはどの仮説でも同じに見えるので、判定の情報を持たない。

## ボールから判定する（`hypothesis.py`）

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
既定はclipごとの判定。固定カメラで同じ動画の証拠をまとめる設定・診断・build/loadは[動画共有のcourt side](../../tennis_scene/pipeline/video_court_side/README.ja.md)、下流の接続は[tennis_scene pipeline](../../tennis_scene/pipeline/README.md)を参照。

## プレーヤーから判定する

`court_side.evidence=player` を指定すると、camera-local校正と `player_selection/<camera>` の観測だけから判定する。
`player_association` はこの判定を受け取る後段。ボールの処理範囲は次の設定で選ぶ。

| 構成 | `ball_detection.enabled` | `ball_reconstruction.enabled` |
|---|---|---|
| 選手の証拠でsideを決め、ボールも推論・再構成する | `true` | `true` |
| 選手だけを再構成し、ボール関連の3段を省く | `false` | `false` |

ボールの3段を省く構成では、`execution.ball_detection=load`、`execution.ball_points=load`、`execution.ball_triangulation=load` は設定エラーになる。
`cache.source=load` は計画内の段を読み出す。

各cameraの観測bboxの下端中央を、camera-local校正で地面の `z=0` へ投影する。
各仮説で、同一フレームのcamera対ごとに、足元の幾何距離だけを使うHungarian法で点を一対一に対応させる。
人数が違うcamera対では、観測が少ない方の点数に合わせる。
中心反転した自分自身との全点対が `symmetry_distance_m` 以内なら、そのcameraの観測は判定から除く。
複数cameraで点集合の変化が `min_motion_m` 未満の繰り返しも除く。

各点対のcostは `min((距離 / match_distance_m)^2, 1)`。camera対内、frame内のcamera対、全frameの順に平均する。
supportは、同一frameの全camera対で全対応点が `match_distance_m` 以内のframeの割合。
全仮説のcost・support・frame数を出し、上記の判定規則と同じ理由で `CourtSideUndecided` を返す。
閾値は `court_side.player` に置く。既定の距離3.0 mは、meiji_4camの保存済み3clipで観測した足元のずれと別仮説のsupportを比較して選んだ。[実測と選定根拠](../../../knowledge/nodes/court_side/000008-run-i2269-player-court-side-20261119.md)
中心対称の許容距離0.5 mと移動量0.15 mで情報の少ない観測を除き、8 frame・support 0.2・cost 0.7・margin 0.05の各閾値で判定する。

## 合成ベンチマークと閾値の選定（`benchmark.py`）

以下は旧 `single_object_camera_view_v2` による保存実験の説明です。
BLCSの現行datasetは `single_object` のみで、旧形式は廃止済みです。
このbenchmark入口は現行形式へ移行していないため、以下のコマンドは過去実験の契約として残しています。
dataset reviewでは保存済みevidenceを読み、新しいbenchmarkや推論を実行しません。

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

#935の固定confidence filterを加えた回帰試験は [knowledge court_side/000004](../../../knowledge/nodes/court_side/000004-run-i935-filtered-side-safety-r29-20261001.md) を参照。追加欠測で誤判定が発生し、全pipeline qualificationは保留した。閾値を緩めて通過させていない。
