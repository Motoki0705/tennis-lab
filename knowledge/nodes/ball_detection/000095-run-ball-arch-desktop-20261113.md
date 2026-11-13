---
id: run-ball-arch-desktop-20261113
type: run
task: ball_detection
sequence: 95
recorded_at: '2026-11-13'
title: RTX 5060 Tiへの本学習移管とactivation checkpointing検証
issue: 1859
provider: codex
date: '2026-11-13'
status: running
summary_impact: none
config:
  seed: 42
  gpu: NVIDIA GeForce RTX 5060 Ti
  torch: 2.14.1+cu130
  python: 3.13.16
  allocator_fraction: 0.8
  msda_levels: [16, 32, 64]
  msda_chunk_size: 4
  frames: 64
  compile_mode: default
metrics:
  official_msda_gradcheck_widths_passed: 7
  full_shape_query_parameters_compared: 242
  full_shape_cnn_parameters_compared: 133
  maximum_query_absolute_gradient_error: 1.1920928955078125e-07
  maximum_cnn_absolute_gradient_error: 0.0
artifacts:
  run_dir: knowledge/runs/run-ball-arch-desktop-20261113
  output_dir: outputs/ball_detection/train/arch_efficiency/desktop-resume-20261113
  drive_outputs: outputs/ball_detection/train/arch_efficiency/s42-20261111/desktop-attempts/20261113-a1
repro:
  commit: 768f4cad27741e3a827f8a411ccc655d3e9be742
  branch: codex/ball-arch-desktop-20261113
parents: []
relations: []
papers: []
tags: [checkpointing, desktop-resume, msda]
---

# 移管の範囲と観測

#1859のT1後半5,500更新（query累積11,500）から、T1残り→T2→C1→C2→C3を再開する。元のL4実験・Drive世代は読み取り専用とし、デスクトップの保存先を分けた。T1/T2だけでactivation checkpointingを使い、同じ範囲を本学習前に固定する。T・解像度・batch・microbatch・dtype・MSDA level/chunk・compile方針とC1〜C3のrecipeは保持する。別GPUなのでbitwise継続を保証せず、このGPUの時間・VRAMを旧L4未学習ベンチマークと比較しない。

元の固定snapshot2,925 payload・157,028,755,547 bytesを同じstoreからhardlinkし、distributionと全件SHAを照合した。元のDrive実験1,917ファイル・19,243,387,824 bytesはE保管領域へ期待manifest付きで取得し、独立readbackと更新停止を確認した。Eは学習データ・tensorの入出力には使わない。

専用Python3.13.16 / Torch2.14.1+cu130と元の167 runtime package版を一致させた。元環境自身の依存metadata conflictは記録して明示的に同じ版を復元し、共有.venvを変更していない。sm_120向け公式MSDAの7 CUDA/C++ソースは既存の公式照合済みmanifestと同一。無制限register版はdouble backwardの1,024-thread blockでregister上限を超えたが、ソースを変えずcompilerのmaxrregcount=64で公式全7幅gradcheckが通った。

BF16実形状T64/720pのquery・CNNをchain-rule VJPで比較し、全375パラメータの勾配を照合した。query最大絶対差1.192e-7、CNN差0。全体非checkpoint参照が16GBに収まらないため、queryのfeature VJPを各非checkpoint CNN stageへ適用した診断であり、本番にはdetachを入れていない。native CUDA5テストも成功した。FP32のcompiled replayは出力・全勾配一致を確認したが、BF16 compiled graph同士はfusionが異なり要素ごとの勾配一致を主張しない。出力の近似一致・全勾配有限と、別のeager BF16等価性を記録している。

# 判断と残る評価

このノードは移管・数値・メモリ資格確認の記録で、精度改善の根拠にはしない。モデル採用、独立test未評価、#1859の5条件完走後に精度を比較するというsummaryの判断は変わらない（summary_impact: none）。実データprobe・起動確認の結果は同run bundleへ保存する。学習曲線とper-frame予測はDriveに保持し、Gitには小さな受入要約・再現手順を残す。T1の12k評価、T2の500保存と600解除、残条件の完了を親が監視する。workerの終了を本学習完了とは扱わない。
