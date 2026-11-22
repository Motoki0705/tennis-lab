---
id: run-court-video-shared-stage1-20261117
type: run
task: court_detection
sequence: 76
recorded_at: '2026-11-17'
title: モデル初期値と複数frame白線による共通Hの独立対応検証
issue: 2191
provider: codex
date: '2026-11-17'
status: done
summary_impact: none
config:
  mode: video_shared
  neural_inputs: run75_frozen_region_homographies
  seed_clips_per_camera: 64
  seed_frame_index: 0
  fit_clips_per_camera: 69
  fit_frames_per_clip: 3
  reference_h_used: false
  reference_frames_used: false
  manual_points_or_net_mask_used: false
  heldout_rgb_used: false
  gpu_used: false
  cpu_processes: 4
metrics:
  finite_seed_count: 1577
  clustered_initializers: 568
  bootstrap_frames: 12
  fit_frames: 828
  all_fit_measurement_jobs: 1863
  R1_seconds: 269.5157848710078
  R2_seconds: 418.90406113502104
  R3_seconds: 451.1667902400077
  R4_seconds: 7.0546481370111
  R4_selection_admissible_cameras: 4
  R4_raw_merge_admissible_cameras: 3
  geometry_and_lock_tests_passed: 30
  geometry_and_lock_tests_seconds: 10.12
  pipeline_regression_tests_passed: 164
  pipeline_regression_tests_seconds: 34.01
repro:
  commit: a9fd35ff4
  command: See knowledge/runs/run-court-video-shared-stage1-20261117/reproduce.md
artifacts:
  run_dir: knowledge/runs/run-court-video-shared-stage1-20261117
  output_dir: .worktree-outputs/court-video-shared/tennis_scene/evaluate/court-video-shared-stage1-20261117
parents:
- run-court-legacy-default-20261117
- run-court-white-line-reference-20261116
relations: []
papers: []
tags:
- video-shared
- CPU
- semantic-correspondence
---

モデルの凍結Hから白線を測り、共通Hの候補を4cameraとも得た。ただし、このrunはfit集合上の設計確認である。
承認済み参照との精度、heldout RGBでのdrift、全clip成立、人間の目視はまだ評価していない。
summary_impactはnoneとする。現行clip既定の採用判断を変えず、#2191の動画共有を検証するための中間証拠である。
製品の準備APIを通した評価はrun 77で分けて記録する。

## 入力と独立性

run 75の各regionの有限Hをcrop座標のまま保存し、readerで一度だけ原画像へ平行移動した。
64clip×4camera、有限候補1577（cam0/1/2/3: 389/436/356/396）を使用した。
旧採否に落ちた有限Hも初期値として許可し、採否・clip・frame・region・元比較ファイルhashを保持する。
全14点の最大距離12px以内の初期値をまとめ、代表568（187/96/82/203）を比較した。候補数を投票数にはしない。

run 71の参照作成に使った15frame/cameraを、初期値・fit・評価から除く親承認subsetに従う。
初期値はclip 000/017/034/051/068を除いた64clipのframe 0、fitは全69clipの
`floor(k*(N-1)/5), k=1,2,3`。bootstrapは001/033/067の最初のfit frameである。
全276動画のframe 0 RGBが凍結raw取得時のhashと一致し、fit828 RGBと動画内容も固定した。
各frameのsource frame番号はplanningのsplitに残す。参照H、手入力点、手指定ネット帯は推定器に渡していない。

## 負の結果を含む設計確認

|段階|変更と根拠|共通解の判定|
|---|---|---|
|R1 (`3e8ecf5b9`)|白線中心・projective grid・複数線の整合。固定ネットも静止し、線の位置だけでは意味を決められない|cam0/2は別解、cam1/3は支持不足。0/4|
|R2 (`dfff8d751`)|template端の内外を測り、探索窓の端と実際の塗線の終端を区別。端の先まで続く線を除いて再fit|cam0/1は別解、cam3は3枚目で端を確認できず。cam2のみ|
|R3 (`757f0fc22`)|bootstrapからのRGB初期値4/2/2/1を全828 fit frameへ適用。raw点のmergeも比較|cam0はnear/far serviceの取り違え（far baseline約311px差）、cam2/3は共通Hへの6px残差条件で失敗。cam1のみ|
|R4 (`a9fd35ff4`)|横線が2本だけなら、baseline/service端に加えて有限なcentre service線を必須化。共通fitの時間残差と局所fitの6px精度条件を分離|有効な個別解2/2/2/1。全14点3px以内の同等解から選ぶ方式なら4/4|

R4のraw点mergeはcam2の2frame（clip026/frame756、clip040/frame249）で、同じleft singles線の
双方向p95が3.06〜4.35pxになり不成立だった。点群mergeをcamera別のfallbackにはしない。
製品方式は**全cameraで同じ解選択**に固定した。同等な複数frame解から残差で選び、異なる意味のHが
3pxを超えて残る場合は曖昧として失敗する。H係数を平均して中間のコートを作らない。
全候補・raw測定点・不採用理由はoutputsに残し、[集計](../../runs/run-court-video-shared-stage1-20261117/summary.json)にhashを保持する。

時間変化があるのに「全frameが共通Hに6px以内」を要求すると、親合意のwarning/失敗基準を
別の閾値で上書きする。そのためR4で6px条件は局所Hの信頼性へ限定した。
共通Hの大残差groupは削除せず、`large_common_residual_groups`へ保存する。
各診断frameの局所Hが成立し、線単位の不確かさで説明できない3px超ならwarning、
平均10px超・最大25px超またはdrift不明なら失敗とする。時間区間は分割しない。

## テスト設計と固定値

独立の人工描画はliteralのcourt座標と既知の非affine Hを使う。静止したネット、影、動く人物・遮蔽物を含み、
推定側のline extractorでGTを作らない。端の外側へ続くネット、横線2本の意味曖昧、支持のないframe、
複数初期値の重複、異なる静止stripe、局所外れ線、有限だが不採用の候補とcrop平行移動を検証する。
driftは0/5/11pxの既知移動と、平均10px未満でも1点が25px超となるprojective移動を含む。
単一の誤線を整合したcamera移動と呼ばないことも確かめる。

開発集合で、支持6線（横2・縦3以上）、各線3clip以上、局所line p95 6px以下、
normalized condition 10000以下、可視KPの最大projection std 5px以下を固定した。
warningは平均変位の2標準偏差下限が3px超。mean10/max25の受入閾値は親合意のまま。
局所Hは診断だけに使い、共通Hの代わりには渡さない。

最終幾何29ケース＋store同時build1ケースは30 passed/10.12秒。
pipeline・config・旧reader・import・gallery・共通計画・新入口の関連164ケースは34.01秒で成功した。
新入口の実Hydra subprocess E2Eは12.81/11.64秒で、5秒目安を超える。
これは入口の設定読込からreceipt/NPZまでを守る2ケースで、unitへ重複させない。
新integration最長は3.63秒。既存galleryは5.35秒だった。
pre-commit ruff/mypyは`37499432b`で成功した。

CPUのみで実行し、CUDA・再学習・モデルforwardは使っていない。TensorBoard曲線の対象外。
人工画像は実映像のすべてのネットや照明を代表しない。参照と白線抽出器を共有する独立性の限界は
実データ評価で明記し、4/4の数値解が得られたことを精度合格とは扱わない。
