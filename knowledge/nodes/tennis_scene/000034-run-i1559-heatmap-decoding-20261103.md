---
id: run-i1559-heatmap-decoding-20261103
type: run
task: tennis_scene
sequence: 34
recorded_at: '2026-11-03'
title: clip_000のnative heatmap復号と座標補正のCPU比較
issue: 1559
provider: codex
date: '2026-11-03'
status: done
config:
  clip_id: meiji_3cam/video_000/clip_000
  cameras:
  - cam0
  - cam1
  - cam2
  num_frames: 1010
  tuning_frames:
  - 0
  - 505
  evaluation_frames:
  - 505
  - 1010
  loess:
  - true
  - false
  prediction_conditions: 32
  scoring_reference_conditions: 1
  gpu_used: false
  inference_rerun: false
  affine_fit: shared XY scale/offset; front-half peak matches <=20px; soft_l1 5px
  selection: front-half LOVO median, p90, >20px; coverage >= baseline-0.03
  foundation_head: 92e199ba936a9f44443e385d1f97aa6ed3fde1e2
  base_winner: quadratic_logit_3
  selected: selected_affine
  selected_config:
    coordinate_correction: affine
    method: quadratic_logit
    model_stride: 4
    offset_xy:
    - -0.0013464178797410453
    - -0.0022516805393959153
    patch_size: 3
    scale_xy:
    - 1.0031469720321813
    - 1.0018819231001146
    temperature: 1.0
  reference_observed: 2589
  reference_estimated: 370
  reference_excluded: 28
  reference_missing: 43
metrics:
  baseline_evaluation_lovo_median_px: 9.340188442451916
  baseline_evaluation_lovo_p90_px: 27.078044438775773
  baseline_evaluation_lovo_above20_rate: 0.16363636363636364
  selected_evaluation_lovo_median_px: 9.490560844043122
  selected_evaluation_lovo_p90_px: 29.632887870763696
  selected_evaluation_lovo_above20_rate: 0.1859903381642512
  selected_evaluation_lovo_coverage: 0.5663474692202463
  evaluation_lovo_median_delta_px: 0.15037240159120557
  evaluation_lovo_p90_delta_px: 2.5548434319879227
  cpu_pytest_cases: 130
  cpu_pytest_elapsed_seconds: 25.33
  script_elapsed_seconds: 3110.6536996299983
  gnu_time_wall_seconds: 2932.1
  max_rss_kib: 1277500
artifacts:
  run_dir: knowledge/runs/run-i1559-heatmap-decoding-20261103
  output_dir: outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/heatmap_decoding_v4
  manifest_sha256: c0a523ff8e1692fef8c714463ba3277dc976b353680eedb4af6221afef83c4cb
  verification: knowledge/runs/run-i1559-heatmap-decoding-20261103/artifact-verification.json
parents:
- run-ball3d-evaluation-foundation-20261103
relations:
- to: run-ball3d-foundation-review-20261103
  rel: uses
papers: []
tags: []
repro:
  commit: 5587c9a4caf7bc8587aac846d0b05faa1dac5d3f
  command: bash knowledge/runs/run-i1559-heatmap-decoding-20261103/repro.sh "$BALL3D_INPUT/fixed_input_bundle_v1"
    "$BALL3D_ROOT/data/samples" "$BALL3D_INPUT/human_annotations/ground_truth_v1/ground_truth.json"
    "$BALL3D_INPUT/heatmap_decoding_replay_v4"
---

## 結果と解釈

前半だけで選んだ方式は3×3 logit二次曲面＋全camera共通affine補正（`selected_affine`）。
後半の主指標はbaselineより悪化した。前半の改善は後半へ一般化せず、既定変更や採用の根拠とはしない。
全view再投影の中央値は4.625→4.074px、全区間LOVO中央値は10.208→9.829pxへ下がるが、
主区間LOVOの中央値・p90・20px超率はいずれも悪化する。

| 区間 | 方式 | LOVO median / p90 px | >20px | 採点数/対象数 |
|---|---|---:|---:|---:|
| 主 [505,1010) | pipeline_baseline | 9.340 / 27.078 | 16.36% | 825/1462 |
| 主 [505,1010) | selected_affine | 9.491 / 29.633 | 18.60% | 828/1462 |
| 調整 [0,505) | pipeline_baseline | 10.716 / 28.483 | 16.44% | 821/1497 |
| 調整 [0,505) | selected_affine | 10.128 / 27.425 | 14.96% | 822/1497 |
| 参考 全区間 | pipeline_baseline | 10.208 / 28.196 | 16.40% | 1646/2959 |
| 参考 全区間 | selected_affine | 9.829 / 28.259 | 16.79% | 1650/2959 |

baselineと選定方式だけの共通採点823点でも、中央値9.356→9.500px、p90 27.092→29.538px、
20px超率16.40→18.59%へ悪化した。採点集合の差だけでは説明できない。
cam1を除くfoldでは20px超率が22.94→31.03%となり、cam2を除くfoldもp90が40.72→43.29pxへ増えた。
全33条件の主表、前半/全区間、代表条件の各LOVO・各区間と共通点の表は
[metrics.md](../../runs/run-i1559-heatmap-decoding-20261103/metrics.md)。
全view・3 LOVOの全指標、観測/推定の内訳、イベント診断は成果物の `scores.json` が正本。

補正なし3×3 logit曲面は後半中央値9.230px、20px超15.92%だが、p90は28.232pxへ増える。
半画素stride補正の後半中央値は10.323pxへ悪化した。
ここから後半最良の方式へ選定し直してはいない。

原値でpeak=1のframeは3 cameraとも0件。Gaussianの204/3030点、5×5 logit曲面の291/3030点は
許容1セルの範囲外として格子点を保持した。選定3×3は非凹3点・範囲外7点を明示した。
したがって、このclipの差を「失われた飽和情報を回復した」とは解釈しない。
改善しない原因が復号、時刻/校正、誤候補、正解の推定区間のどれに由来するかは本比較だけでは分離できない。
次は同じ親の候補・校正・同期の比較と合わせ、特にcam1と外れframeの寄与を解釈する。


## 比較の定義

1 frame・1 viewの単一点復号だけを変え、scoreは格子peak値を保持する。
従来の軸別log、軸別logit、交差項付き3×3/5×5のlog/logit二次曲面、
背景を引いた3×3/5×5の重心、局所soft-argmax（温度0.5/2、温度1は重心と等価）、
5×5の異方性Gaussian＋定数背景を比較する。理論上のstride4半画素中心補正と、
前半だけから推定する全camera共通の軸別affine補正も別条件とする。

固定bundle・runner・採点・保存形式・可視化は #1556 の基盤を使う。
閾値・trajectory gate・consensus・LOESS・三角測量はproduction部品を使う。
LOESSなしは同じconsensus maskでraw座標を三角測量する比較であり、採用maskの変更ではない。
GTの配列は方式へ渡さず、係数推定と選定だけを [0,505) に限定する。
後半 [505,1010) の指標は方式を固定してから確認する。

## テスト設計と証拠

実装前の設計は [test-plan.md](../../runs/run-i1559-heatmap-decoding-20261103/test-plan.md)。
非対称な既知中心の合成mapを使い、交差項・logit域・背景・切り詰めた端・飽和を検証した。
main由来の固定配列との厳密一致と、既定pipelineがpredictorの配列を保持する回帰も置いた。
実runner・平滑化・三角測量を結ぶadapterはintegrationへ置き、
LOVOのcamera順、除外viewのcache破棄、設定変更時のcache破棄、同じcamera IDを持つ別timelineを検証した。

最新基盤headを含む130件が成功した。`--collect-only` は130件/5.59秒、
`--durations=0 --durations-min=0` は130 passed/25.33秒、`-n 0`。
全nodeと各caseのsetup+call+teardownはrun bundleの `test-results-latest.json`、
全phaseの実測は `test-durations-latest.txt` に保存した。
交差項を一時的にゼロにする故障注入では、既知中心から最大0.2785セルずれてassertが失敗し、
復元後に同じnodeが成功した。原記録は `fault-injection.txt`。

## 仮説と限界

交差項が斜めの山の偏りを減らす可能性はあるが、実heatmapが単一Gaussianとは限らない。
float32 sigmoidで失われた飽和情報はlogit変換でも復元できず、端の重心には切り詰めバイアスがある。
半画素stride補正は学習モデルの位置依存バイアスの仮説であり、正しい格子規約と断定しない。
1clip、近似校正、承認済み推定を含む2D参照であり、独立3D正解はない。
採用の判断は人間の統合比較後に行う。実装・選定・採点はその判断を代行しない。

## 成果物の検証と資源

132個の型付き結果（32予測条件×4fold＋採点専用oracle×4fold）を共通loaderで再読込した。
全1314ファイルのサイズとchecksum、自己完結HTMLの参照切れ0を照合した。
baselineの3D位置・有効mask・使用view maskは `foundation_v2` と完全一致した。
元storeとの位置差は最大9.5367e-7mで、2µm許容内だがbitwise一致とは記録しない。
既定のmain復号との厳密一致はunit fixtureで別に確認している。

比較MP4は1200×1150、59.940060fps、1010frameを全decode済み。
frame 0/252/505/757/1009、指標表・軌道図、誤差上位765/764/581/580/980/981の3camera cropを目視確認した。
0/505の欠測は補完せず、1009のcam0は除外外挿、cam1はunknownとして正解markerを出さない。
白い菱形の承認推定と緑十字の直接観測、methodのringを区別した。
近い方式のringは重なり、動画は選定条件を含む6方式の表示である。全条件は主表と保存結果に残す。

script timerは3112.6秒、GNU time wallは2932.10秒、最大RSSは1,277,500KiB（約1.22GiB）、swap 0。
時計の原値は一致しないため統合せず、それぞれの記録を残した。GPU・検出器再推論は0回。
1frameずつ動画を処理し、native mapは64枚ずつコピーした。
shared baselineが平滑化を再実行し、adapterがview別にcacheするため、時間は最適化した推論速度の比較ではない。

v1は比較scriptの出力directory契約違反で停止、v2/v3は基盤API・cache修正前に中断した。
入力・過去の成果物を上書きせず残し、採用した実行はv4だけとする。
履歴と理由は `execution-history.json`。TensorBoardとcheckpointは、学習を行わない比較なので該当しない。
人間の採否判断・mergeは未実施であり、このnodeのdoneは実装・比較成果物の完成を表す。
