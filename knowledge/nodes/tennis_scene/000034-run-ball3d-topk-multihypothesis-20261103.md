---
id: run-ball3d-topk-multihypothesis-20261103
type: run
task: tennis_scene
sequence: 34
recorded_at: '2026-11-03'
title: clip_000でtop-K多仮説対応付けをCPU replayしLOVO悪化とcoverage低下を確認
issue: 1560
provider: codex
date: '2026-11-03'
status: done
config:
  clip_id: meiji_3cam/video_000/clip_000
  num_frames: 1010
  tuning_frames: [0, 505]
  primary_evaluation_frames: [505, 1010]
  parameter_selection: predeclared_defaults_without_label_tuning
  top_k: 8
  max_hypotheses: 24
  beam_width: 96
  max_missing_frames: 3
  gpu_used: false
  inference_rerun: false
metrics:
  baseline_primary_lovo_median_px: 9.340188442451916
  method_primary_lovo_median_px: 10.7804879790132
  baseline_primary_lovo_p90_px: 27.078044438775773
  method_primary_lovo_p90_px: 103.59736745483048
  baseline_primary_lovo_count: 825
  method_primary_lovo_count: 533
  primary_lovo_eligible: 1462
  baseline_primary_lovo_coverage: 0.5642954856361149
  method_primary_lovo_coverage: 0.3645690834473324
  paired_two_method_primary_count: 353
  paired_baseline_primary_median_px: 8.77846832956076
  paired_method_primary_median_px: 9.27354366119481
  tests_passed: 106
artifacts:
  run_dir: knowledge/runs/run-ball3d-topk-multihypothesis-20261103
  output_dir: outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103/topk_multihypothesis_v2
  reproduction: knowledge/runs/run-ball3d-topk-multihypothesis-20261103/repro.sh
  metrics: knowledge/runs/run-ball3d-topk-multihypothesis-20261103/metrics.json
  folds: knowledge/runs/run-ball3d-topk-multihypothesis-20261103/fold-metrics.csv
  tests: knowledge/runs/run-ball3d-topk-multihypothesis-20261103/test-summary.json
  inventory_validation: knowledge/runs/run-ball3d-topk-multihypothesis-20261103/validation.json
parents:
- run-ball3d-foundation-review-20261103
relations: []
papers: []
tags: []
repro:
  commit: 49cc2b31f9e3dee3de28dc4d032d17bfed7199de
  command: bash knowledge/runs/run-ball3d-topk-multihypothesis-20261103/repro.sh BUNDLE_DIR VIDEOS_DIR GROUND_TRUTH_JSON PRIOR_BASELINE_DIR NEW_OUTPUT_DIR
---

## 比較の条件と結果

親 #1554 revision 6、子 #1560。公開clip_000の3camera・1010frame、固定bundleだけをCPU replayした。
基盤のloader・runner・保存/再読込・採点・動画/図/crop/HTMLを使い、独自loaderやscorerは作っていない。
GTは全予測の保存後に採点側だけで読み込む。直接観測2589点・承認推定370点を採点し、
cam0末尾外挿28点と座標なし43点を補完しない。観測/推定・イベントの区別は共通scoresに残る。
方式設定はラベルを読む前にcommitし、`frozen-config.json`へ保存した。前半を含めラベルでの調整は行っていない。
主評価は[505,1010)、前半[0,505)と全区間は併記する。後半を見て方式設定を変えていない。

| 区間 | 方式 | LOVO 中央値/p90 px | >20px率 | 採点数/対象数・coverage | 全view有効3D率 |
|---|---|---:|---:|---:|---:|
| **主評価 [505,1010)** | baseline | 9.340 / 27.078 | 16.36% | 825/1462・56.43% | 83.17% |
| 同 | top-K多仮説 | 10.780 / 103.597 | 28.52% | 533/1462・36.46% | 41.39% |
| 前半 [0,505) | baseline | 10.716 / 28.483 | 16.44% | 821/1497・54.84% | 86.34% |
| 同 | top-K多仮説 | 11.903 / 84.848 | 28.86% | 589/1497・39.35% | 43.37% |
| 参考 [0,1010) | baseline | 10.208 / 28.196 | 16.40% | 1646/2959・55.63% | 84.75% |
| 同 | top-K多仮説 | 11.361 / 98.961 | 28.70% | 1122/2959・37.92% | 42.38% |

主評価では中央値+1.440px、p90+76.519px、>20px率+12.15pp、採点数-292点・coverage-19.97ppで悪化した。
各camera除外・全view・区間別の値は [fold-metrics.csv](../../runs/run-ball3d-topk-multihypothesis-20261103/fold-metrics.csv)。
主評価のcam0/1/2除外はtop-Kでそれぞれ8.949/33.509px（166/473）、14.315/72.753px（151/484）、
10.792/279.974px（216/505）。baselineは7.920/14.235px（271/473）、11.990/25.147px（231/484）、
9.262/40.718px（323/505）であり、特にcam2を除いた2viewの裾が大きい。

## 共通点と欠測を併せた判断

2方式の共通LOVO353点でもbaseline 8.778/27.825px・>20px率15.01%に対し、方式は9.274/34.073px・17.85%で悪化。
oracleを含む3方式共通348点でもbaseline 8.669/25.190px、方式9.101/29.883px、oracle 7.135/18.918px。
共通集合も可用性を条件とする23.8%の支持に限られ、欠測を回復した比較ではない。
LOVO集約は個々の残差のpoolで、camera別中央値の平均ではない。

主評価全viewでは505frameすべてに幾何仮説があったが、動的計画法の出力は209frameで296frameを欠測にした。
選択した候補517件のうちrank1が475件、rank2以降は42件。8回の履歴リセット/再捕捉があった。
[candidate-diagnostics.json](../../runs/run-ball3d-topk-multihypothesis-20261103/candidate-diagnostics.json) はラベルを使わない診断。
運動の外れが少ないことを精度改善とみなさない。欠測を増やしながらLOVO裾も悪化しているので、今回方式の採用は勧めない。
採否は人間の統合比較後の判断に残し、既定pipelineは維持する。

## 方式と先行知見

方式・単位・出力契約の正本は [ball_association README](../../../src/tennis_scene/ball_association/README.ja.md)。
gate前のvalidなtop-8をscoreで切らず、全pair候補のDLTと他viewへの幾何/score拡張から仮説を作る。
pair-onlyも含み、幾何棄却後の上位24仮説を96状態の二次運動beam DPで選ぶ。
速度上限65m/s、加速度コストは上限で打ち切り、bounce/hitを有限コストにする。実イベント/GTは入力にしない。
3frameを超える欠測で履歴をリセットし、欠測自体は補間しない。厳密な全組合せViterbiではない。

[#1558の診断（PR #1670）](https://github.com/Motoki0705/tennis-lab/pull/1670) とそのknowledgeを読んだ。
前半argmax失敗はcam0が102点（top-8で63点回収可能）、cam2が125点（39点）、cam1が44点（29点）。
mask前で候補を扱う動機になる一方、cameraごとに回収可能性が違い、top-8に正解がない点も多い。
同診断の「既存採用maskを保つcandidate oracleの改善は小さい」「LOESS除去は共通点の精度を悪化」も踏まえた。
今回は候補＋時間の選択だけを検証し、選択後LOESSは追加していない。
候補剪定・未較正score・nullと加速度コストの釣合い・2D LOESS欠如が限界の候補だが、
独立ablationを行っておらず、各要因へ悪化量を割り当てることはできない。

## 検証・再現と制限

実装前に [test-plan.md](../../runs/run-ball3d-topk-multihypothesis-20261103/test-plan.md) を作成した。
非対称camera/frame、途中の順位逆転と長い張り付き、欠測/再捕捉、急な方向転換、
既知の2m/0.3秒のgap速度、低scoreの保持、異なるK、設定拒否を軽いunitで検証する。
公開config→definition→assembler→方法→実codec、LOVO隔離とmainの固定期待配列をintegrationで確認する。
既定componentとmainの位置・mask・inliers・pixel・rejection・再投影配列を厳密比較した。
全106nodeのcollect-onlyと各phaseの実測はtest-collection/test-results/test-summary、正確なコマンドはtest.sh。
`-n 0 --collect-only` は106件/6.69秒、`--durations=0 --durations-min=0` は106 passed/29.13秒。
全19追加caseを含む各node/stageの厳密な時間はJSONに保持し、pytest全体の時間とcase合計を区別する。
加速度コストをゼロにする故障注入で数値assertが失敗し、元bytesへ復元後は成功した。既存nodeの削除・統合はない。

基盤#1669のmain mergeを確認し、origin/mainを取り込んでテストを再実行した。実行コードからsrc差分は0。
v1は初回run、v2は負の整数重み/bool設定の拒否を補強した後の同設定再実行で、
全8予測結果・全診断配列と比較MP4はbitwise一致。baselineの4foldはfoundation_v2ともbitwise一致した。
元store照合も有効856frameとmask一致、3D最大差は2µm許容内。GTや入力bundleは変更していない。

v2は約259.5秒、最大RSS1,133,200KiB。検出器再推論・GPU・学習・追加入力取得はない。
比較MP4は1200×1128・1010frameを全decode。代表0/252/505/757/1009、指標表、6パネルの軌道/分布図、
極端419/417/415/411/407/406の3camera cropを目視した。frame252で選択squareと候補順位の重なりを確認し、
frame757のunknown・frame1009の末尾外挿除外が正解markerとして描かれないことも確認した。
HTML参照切れ0、全129ファイルのサイズ/SHAと未宣言ファイルなしを照合した。記録はvalidation/verificationに保存。
大きな配列・動画・HTMLはoutputsに残し、knowledgeには小さな指標・図・実行証拠だけを保存する。

単一clip・固定の近似校正で、独立3D正解はない。oracleは数学的下限ではなく幾何参照。
beamとpair拡張は近似探索であり、完全なjoint候補列を保証しない。
次は同一条件の他方式との統合比較を行う。改良する場合も、未使用clip又は前半だけで
null/運動コスト、候補保持、選択後LOESSを独立に検証し、今回後半の再調整はしない。
学習ではないためTensorBoard・seed間学習曲線は適用しない。PRは人間の採否までmergeしない。
