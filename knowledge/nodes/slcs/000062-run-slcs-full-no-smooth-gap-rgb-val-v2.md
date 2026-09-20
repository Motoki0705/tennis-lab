---
task: slcs
sequence: 62
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-full-no-smooth-gap-rgb-val-v2
type: run
title: 'SLCS全体版5条件評価: 欠損区間でもRGBは寄与、broadcastの裾誤差は残存'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: SLCSFusionModel validation-selected epoch56
  loss: inference only
  data: slcs/real_rgb_v1
  split: val
  gap_no_rgb: true
metrics:
  exit_code: 0
  val_windows: 343
  detector_gap_ball_position_error_m: 3.0972740650177
  detector_gap_no_rgb_ball_position_error_m: 3.2916150093078613
  detector_gap_player_position_error_m: 1.463090181350708
  detector_gap_no_rgb_player_position_error_m: 1.6152840852737427
  gap_region_ball_position_error_m: 3.7014848732052794
  gap_region_no_rgb_ball_position_error_m: 3.8975350501989046
  broadcast_gap_region_ball_position_error_m: 5.7260225497320025
  broadcast_gap_region_no_rgb_ball_position_error_m: 6.727564711949069
  broadcast_gap_region_ball_error_p95_m: 16.0131806373596
  broadcast_gap_region_no_rgb_ball_error_p95_m: 13.558939313888542
repro:
  commit: 1758eda44debb618395fc7fe86e9ecb30f32a4fa
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -m scripts.analysis.evaluate_slcs_run --output-root
    /home/kamimura/projects/tennis-lab/outputs --training-run slcs/train/real_rgb_no_ball_smooth/s42-takeover-003
    --output slcs/evaluate/real_rgb_no_ball_smooth/s42-gap-diagnostic-002 --domain-prefix
    video_=meiji --default-domain broadcast --device cuda --batch-size 4 --ball-train-mean
    --gap-no-rgb
artifacts:
  run_dir: knowledge/runs/run-slcs-full-no-smooth-gap-rgb-val-v2
  output_dir: outputs/slcs/evaluate/real_rgb_no_ball_smooth/s42-gap-diagnostic-002
  log: knowledge/runs/run-slcs-full-no-smooth-gap-rgb-val-v2/queue.log
parents: [run-slcs-full-real-rgb-no-ball-smooth-val-v3]
relations:
- {to: run-slcs-full-no-smooth-gap-rgb-crc-v1, rel: supersedes}
tags: [slcs, real-rgb, validation, input-conditions, detector-gap, rgb]
---

## 考察 / Findings

### 要約

CRC停止後、入力を変更せず別runで公開CLIの5条件評価を完走した。検出欠損条件のball誤差は
RGBあり3.0973m、なし3.2916m。人工欠損の中央40frameだけでもRGBありの平均誤差が小さい。
ただしbroadcastの欠損区間p95はRGBありの方が悪く、平均値の改善を頑健性達成とは扱わない。

### アーキテクチャ詳細

親runと同じvalidation選定epoch56・checkpoint SHA
`870407e89ca0ff31d07a33395acf3a3538f49146274f3566bd36ff969b6733a2`を固定。
`--gap-no-rgb`は同じ中央1/3のball・全player検出欠損にRGB除去を重ね、courtは維持する。
全5条件で教師・mask・weight・window対応・checkpointを照合し、testは選定にも評価にも使っていない。

### メトリクスの解釈

通常headlineは欠損区間外も含む120frame全体の有効occurrencesの非加重平均である。
全343窓がpaddingなし120frameであることを確認し、production条件関数が消す`[40:80)`だけを別集計した。
欠損区間のball有効occurrencesは全体11791、うちbroadcast334。位置誤差配列に同じ教師maskを適用した。

| ball評価範囲 | RGBありm | RGBなしm |
|---|---:|---:|
| 全体・120frame | 3.0973 | 3.2916 |
| 全体・欠損40frame | 3.7015 | 3.8975 |
| Meiji・欠損40frame | 3.6425 | 3.8150 |
| broadcast・欠損40frame | 5.7260 | 6.7276 |

欠損40frameのplayerは全体1.5697/1.7733m、broadcast2.8174/2.8487m。
ball p95は全体9.4771/9.9146mだが、broadcastは16.0132/13.5589mで逆転した。
全配列・設定・motion・比較・train-only定数をevaluation/へ保存した。学習曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

欠損区間内に限定しても位置平均が改善するため、RGBが単に欠損区間外だけへ寄与しているという説明では不十分。
一方、単一学習済みmodelへの入力除去であり、RGBなし再学習との因果比較ではない。
broadcastは1clip・10窓で、教師weightはball0.15。大きな裾誤差と擬似教師の不確かさが残る。

### 既存実験との比較

親の4条件との照合でfull/no_rgb/rgb_onlyの全保存配列は完全一致した。
detector_gapの教師・mask・weight・metadataは一致し、予測には微小差があった（ball位置誤差の最大差5.722e-6m）。
丸めたheadlineは変わらないが、4条件すべてがbit-exactに再現したとは記載しない。
前のCRC失敗runはfailedのまま保持し、この完走を過去の停止原因解決の証拠にはしない。

### 次に有効な実験

gap48の60epoch再開学習を完了し、同じ5条件・欠損区間・domain・裾・motionで比較する。
検出visibility境界の大きな速度誤差について、教師との一次差分整合を加える単一loss比較も準備する。
