# Top-Kボール対応付け

`associate_topk(..., candidate_uv_px=...)` は保持camera IDをkeyに、明示的に再復号した
候補画素を受け取れる。shape・画像範囲・無効slotの0を検証する。元cell・score・mask・
heatmapの出自は変えず、引数を省略すれば従来の計算を保つ。

[English](README.md) · [简体中文](README.zh-CN.md)

標準pipelineで `ball_reconstruction.method=topk_multihypothesis` を選ぶ。
既定は `triangulation`。設定は `ball_reconstruction.topk`、検証は `TopKConfig`。
保存済みgate前候補だけを使い、RGB、再推論、dense map、GT、学習済みパラメータを使わない。
画素閾値は1920×1080に対する対角長比、運動はm/s・m/s²、欠測上限はsource frame単位。
既定値はすべてHydra compositionが所有する。評価側も
`PipelineRuntimeConfig.from_config(composed, bind_inputs=False)` の結果から
`TopKMultiHypothesis(runtime.ball_topk, min_frames=runtime.ball_min_frames)` を作る。
設定型とadapterは欠落した設定値を補完しない。

validなK≤8候補をscoreで切らず、全camera-pair候補組を等重みDLTで三角測量する。
0 scoreはlog評価だけ1e-8へ丸める。各pairの幾何・scoreに最も整合する他view候補を
追加した仮説も作り、pair-onlyの欠測view仮説も残す。選択cameraの正のdepth、
seed pairのray角≥1°、再投影閾値、高さ[-0.2,20]m、|X/Y|≤40mを要求する。
平均二乗再投影/sigma²、平均負log score、未使用viewコストで順位を付け、
重複candidate-index組をまとめて `max_hypotheses` まで残す。
全viewの直積を網羅する探索ではない。

時間方向は速度・加速度・欠測を持つbeam動的計画法。速度上限でリンクを禁止し、
加速度は隣接速度差/平均時間間隔で計算する。二乗加速度コストを `turn_cost_cap`
で打ち切り、実イベント注釈なしでbounce/hitの急変を有限コストにする。
(直前,その前)仮説組ごとの最小コスト履歴を保持し、`beam_width` に剪定する。
厳密な全探索Viterbiではなく、最適経路が剪定される可能性がある。同点は安定した候補・コスト順。

nullは `missing_frame_cost` を払い、`max_missing_frames` まで実frame間隔で履歴を保持する。
超えると履歴を捨て `restart_cost` を払い再捕捉する。欠測・端は補間せず欠測のまま。
有効frameが `ball_reconstruction.min_frames` 未満なら全軌道を無効にする。

`AssociationResult` は選択index（未使用-1）、raw画素、再投影、仮説数、emission cost、
exp(-cost)と再捕捉maskを保持する。scoreは未較正の相対重みで、存在確率ではない。
無効3D・画素・再投影は0。有効3Dは必ず2view以上を使う。選択後の2D LOESSは行わない。
optional componentが既存ball payloadとともにtyped保存し、評価adapterも同じ関数を使う。

候補配列だけを小さくコピーし、heatmap/patchを複製しない。計算量はpair提案、仮説上限と
beam幅で制限する。[基盤描画](../evaluation/README.ja.md)は動画をstreamし、optional表示で
灰色の候補順位・色付き選択squareを描く。ringは3D再投影。既定描画は変更しない。

合成契約は `tests/unit/tennis_scene/ball_association/test_topk.py`、設定・component/codec・
LOVO隔離・main配列回帰は `tests/integration/tennis_scene/test_ball3d_topk.py`。
実clipのコマンド・記録は `knowledge/runs/run-ball3d-topk-multihypothesis-20261103/`。
