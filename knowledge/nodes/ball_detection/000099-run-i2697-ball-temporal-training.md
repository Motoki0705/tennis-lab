---
id: run-i2697-ball-temporal-training
type: run
task: ball_detection
sequence: 99
recorded_at: '2026-11-28'
title: 単フレームと時系列ボール検出の統制学習
issue: 2697
provider: codex
date: '2026-11-28'
status: done
summary_impact: updated
config:
  dataset: ball-mix1800-720p-all-frame-train-val-v1
  architecture: X3D/CSN + DPT heatmap
  image_wh: [512, 288]
  seed: 42
  frames: [1, 8, 32]
  input_channels: [3, 5, 5]
  updates_per_condition: 4700
  frames_per_update: 128
  train_packs: 30478
  evaluation_scored_frames: 117358
  evaluation_anchors: 144
  plan_sha256: 2aca1c997c43ac4bee01b0c7f26f7f16cabe4d2da81092addede24a49723adc2
metrics:
  input_frames_per_condition: 601600
  validation_20px_hits: {T1: 1417, T8: 34959, T32: 37803}
  validation_20px_support: 85473
  validation_f1_balanced: {T1: 0.740719, T8: 0.820174, T32: 0.826289}
  anchors_20px_hits: {T1: 2, T8: 40, T32: 45}
  anchors_20px_support: 88
  training_and_evaluation_seconds: 17411.642
  probe_seconds: 171.897
artifacts:
  run_dir: knowledge/runs/run-i2697-ball-temporal-training
  metrics: knowledge/runs/run-i2697-ball-temporal-training/metrics.json
  comparison: knowledge/runs/run-i2697-ball-temporal-training/comparison.csv
  figure: knowledge/runs/run-i2697-ball-temporal-training/temporal_training_ablation.pdf
  output_dir: /mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/parent-2627-issue-2697-primary/ball-temporal-training/s42
  log: .training_queue/logs/0-8204205277883712570_2026-11-27T15-52-02-116287429Z_1813138_i2697-ball-temporal-training-s42.log
repro:
  commit: bd0d76a14cdf22cc5b7f928d19831efc89c49b5a
  branch: codex/parent-2627-issue-2697-primary
  command: >-
    bash knowledge/runs/run-i2697-ball-temporal-training/reproduce.sh train --dataset-root
    /mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/ball-query-desktop/ball_detection/train/query_controls/desktop-resume-20261112
    --output outputs/ball-temporal-training/s42 --plan knowledge/runs/run-i2697-ball-temporal-training/plan.json
parents:
- run-ball-all-frame-protocol-20261022
relations:
- {to: run-i2637-ball-temporal-context, rel: compares, note: 同一の144アンカー・元画像pxの採点で学習比較と推論時介入を併記}
- {to: run-ball-sequential-length-20261025, rel: related, note: 全validationと採点規約を共有する逐次学習}
papers: []
tags: []
---

## 考察 / Findings

### 共通条件と実行記録

同一X3D/CSN＋DPTとseed 42の初期重みを使った。T=1はRGB 3chで、5chで引いた初期stemからRGB部だけを転写した。
T=8・T=32はRGB+MDD 5chで、各子窓の先頭からMDDを計算した。全条件で外部重みを使わずに学習を開始した。
固定30,478 packから同じ順で復元抽出し、各4,700更新に128枚ずつ入力した。各条件の総601,600枚には、observed 484,743、
reviewed_empty 60,083、out_of_frame 14,160、masked 42,614件の教師・マスク出現が含まれる。
各更新のLR、pack ID・教師数、終端チェックポイント、初期重みの共通部分を照合した。
[固定plan](../../runs/run-i2697-ball-temporal-training/plan.json) ·
[学習の照合結果](../../runs/run-i2697-ball-temporal-training/training-verification.json)

720pのT=1がVRAM上限12.72 GiBに達したため、全条件を512×288で実行した。計測した本学習と評価は
RTX 5060 Tiで4時間50分12秒、プローブ込みで4時間53分04秒。1,537 JPEG shard（156,770,864,484 bytes）の
SHA-256を照合し、18分19秒を本学習と評価の実測に含めた。queueは両方とも `resource=all`、終端 `done`。
[プローブ](../../runs/run-i2697-ball-temporal-training/probe.json) ·
[プローブqueue](../../runs/run-i2697-ball-temporal-training/probe-queue.json) ·
[本学習queue](../../runs/run-i2697-ball-temporal-training/training-queue.json) ·
[事前に固定した設計](https://github.com/Motoki0705/tennis-lab/issues/2697#issuecomment-7222572419)

| 入力 | 学習（保存を含む） | 全validation | アンカー | 学習ピーク予約 GiB |
| --- | ---: | ---: | ---: | ---: |
| T=1 RGB | 3,815.3秒 | 348.7秒 | 5.0秒 | 3.82 |
| T=8 RGB+MDD | 4,628.8秒 | 851.6秒 | 7.8秒 | 3.84 |
| T=32 RGB+MDD | 5,410.3秒 | 1,209.4秒 | 21.6秒 | 3.84 |

### 固定validationの結果

128,668 ownerフレームのうち、observed 85,473、reviewed_empty 29,173、out_of_frame 2,712の計117,358件を採点した。
位置指標は存在閾値適用前の全observedを元画像pxで集計した。FPRの分母は確認済み不在31,885件、FNRはobserved 85,473件。
存在F1は座標誤差を含めない。下表の存在判定は各条件で `(FPR + FNR) / 2` を最小化する全validation閾値を使う。
[指標JSON](../../runs/run-i2697-ball-temporal-training/metrics.json) ·
[比較CSV](../../runs/run-i2697-ball-temporal-training/comparison.csv) ·
[論文用図 PDF](../../runs/run-i2697-ball-temporal-training/temporal_training_ablation.pdf)

| 入力 | 平均 / 中央値 / P95（px） | 20 px以内 | 閾値 | FPR | FNR | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T=1 RGB | 226.75 / 177.02 / 597.20 | 1,417 / 85,473（1.66%） | 0.1480 | 21.19% | 36.53% | 74.07% |
| T=8 RGB+MDD | 134.06 / 42.48 / 561.26 | 34,959 / 85,473（40.90%） | 0.3166 | 36.22% | 21.09% | 82.02% |
| T=32 RGB+MDD | 125.74 / 31.68 / 557.85 | 37,803 / 85,473（44.23%） | 0.3277 | 31.33% | 21.37% | 82.63% |

固定0.5でのFPR/FNR/F1はT=1で0.00% / 100.00% / 0.00%、T=8で4.58% / 75.09% / 39.35%、
T=32で6.23% / 68.38% / 47.21%だった。T=1は採点対象全件を不在と判定した。
最適バランス閾値でもT=1のFPRは両時系列条件より低く、FNRは高かった。位置精度と、選んだ閾値での存在判定は別々に読む。

20 px以内率はChatで1.29% → 45.54% → 49.20%、Meiji 3camで2.66% → 27.36% → 29.77%、
TrackNet game9で1.37% → 65.18% → 69.10%。Meiji 3camは確認済み不在が0件でFPRは未定義。
最適バランス閾値でのT=1のMeiji 3camのF1は1.00で、同sourceの位置精度2.66%とは別の結果である。

### #2637の固定アンカー

observed 88、reviewed_empty 32、out_of_frame 24の同じアンカーで、20 px以内率はT=1が2/88（2.27%）、
T=8が40/88（45.45%）、T=32が45/88（51.14%）。平均は203.45、114.98、114.08 px、中央値は152.04、24.63、18.96 px。
P95は542.59、506.23、530.83 pxで、T=8が最小だった。全validationで選んだ閾値をそのまま適用したF1は
61.82%、79.14%、78.95%。固定0.5では0.00%、46.15%、58.46%だった。

#2637の通常入力の位置20 px以内率はX3D T32で76/88（86.36%）、i935 T8で73/88（82.95%）。
同一フレーム反復では2/88と0/88、形状基準では7/88だった。比較CSVは入力介入の元の閾値と、今回の統制学習の2つの
閾値を収録する。#2637とのスコア差には学習履歴と入力解像度も含まれる。
[#2637の実験条件と結果](000098-run-i2637-ball-temporal-context.md)

### 第7章に使える範囲

同じデータ・予算で個別に学習しても、時系列条件の位置精度は全3 sourceで大きく上がった。この結果は、
この学習設定での時系列入力の有用性と単フレーム入力の限界を支持する。T=8からT=32への延長による20 px以内率の改善は
全validationで3.33ポイント、アンカーで5.68ポイントだった。存在判定と誤差の裾では、閾値や集合によってモデルの順位が異なる。
T=1からの差にはMDDの有無も含まれ、時間長のみの寄与はT=8とT=32の範囲で比較できる。対象はseed 42の1回、512×288、
4,700更新、固定validationとその部分集合である。最適バランス閾値は同じvalidationに合わせており、独立testは未計測。

再集計を2回実行し、JSON・CSV・PNG・PDFのバイト一致を確認した。公開するのは集計と実行記録で、重みと全フレーム予測は
実行出力先に保存した。学習はJSONLログを使ったためTensorBoardファイルは生成していない。
[再現手順](../../runs/run-i2697-ball-temporal-training/README.md)
