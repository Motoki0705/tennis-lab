---
id: run-i1899-convnext-1query-t64-t64-s42-12k
type: run
task: ball_detection
sequence: 93
recorded_at: '2026-11-13'
title: 前半からT64で学習した旧query構成の12k終端結果
issue: 1899
provider: codex
date: '2026-11-13'
status: done
config:
  seed: 42
  cnn: convnext_v2
  queries: 1
  mhc: false
  dim: 256
  layers: 4
  heads: 8
  ffn_dim: 704
  stages:
  - T64 6000
  - T64 6000
  input_frames_per_update: 128
  precision: bf16
  freeze_updates_per_stage: 600
  encoder_lr_ratio: 0.1
  gpus:
  - NVIDIA L4 (first5500 query updates)
  - RTX5060Ti (remaining6500 query updates)
metrics:
  mean_error_px: 28.526567810816783
  median_error_px: 7.652594566345215
  p95_error_px: 120.72167205810501
  within_distance_rate: 0.7616908263428217
  observed_frames: 85473
  query_updates: 12000
  stage1_mean_error_px: 33.92097911269842
artifacts:
  run_dir: knowledge/runs/run-i1899-convnext-1query-t64-t64-s42-12k
  output_dir: .claude/worktrees/ball-query-desktop/outputs/ball_detection/train/query_controls/desktop-resume-20261112
  drive_outputs: outputs/ball_detection/train/query_controls/s42-20261111-main/desktop-attempts/20261112-a1
  validation: knowledge/runs/run-i1899-convnext-1query-t64-t64-s42-12k/stage2-validation.json
  comparison: knowledge/runs/run-i1899-convnext-1query-t64-t64-s42-12k/comparison.json
parents:
- run-ball-sequential-length-20261025
relations:
- to: run-ball-sequential-length-20261025
  rel: compares
  note: 同じ共通DPT・query初期化・128枚pack・教師と評価identityで、前半T32をT64へ変更。GPU切替は限界として記録。
papers: []
tags:
- ablation
- validation
- single-seed
- desktop-resume
summary_impact: updated
repro:
  commit: 929ad65645c1c7522e8921f1b439619d636a0128
  adapter_commit: eda5ee08073da498347d411e075905f4ac94e25e
  command: env TRAINING_QUEUE_DIR="$QUEUE" TORCHINDUCTOR_COMPILE_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "$PY" "$ADAPTER/desktop.py" production --root "$OUTPUT" --source "$SOURCE" --rclone-config
    "$RCLONE_CONFIG"
---

## 結果

#1899の③、ConvNeXt V2＋1query・mHCなしを前半からT64で6k、続いてT64で6k学習した。query累積12kを完了した。前半5,500更新のColab full checkpointからdesktopへ継続し、最後の重みだけでなくAdamW・全RNG・LR位置・sample位置も復元した。

終端のmeanは28.53 pxで、旧T32→T64基準28.20 pxより+0.33 px。medianは7.65 px（−0.01）、P95は120.72 px（+3.83）、20px以内率は76.17%（−0.12ポイント）。前半T64の6kではmean33.92 pxで、同じT64をさらに6k学んだ終端で低下した。前半からT64にする優位は、このseed42・validationの観測では確認できなかった。[4条件の実測値と差分](../../runs/run-i1899-convnext-1query-t64-t64-s42-12k/comparison.md)を参照。

## 条件と照合

モデルはdim256・4層・8heads・FFN704、RGB＋MDDの5ch・720p、128frames/update・microbatch1、seed42、BF16・nvJPEG・default compile。各段の最初600更新はCNNを凍結し、解除後のCNN LRはheadの0.1倍。offsetは段順0→6,000とし、後半は同構造の全重み・named AdamWを継承した。元の共通128枚pack・抽出順・拡張・teacher数を両段全12k行で照合した。[実行設定](../../runs/run-i1899-convnext-1query-t64-t64-s42-12k/configuration.json)に固定source・adapter・checkpoint来歴を記録した。

ローカル既存720p storeから同じ配布版を復元し、2,925 payloadの内容と完全なファイル集合を照合した。移管では参照rootと派生manifest digestのみをコピー上で変更し、学習stateを再帰比較した。Python3.11.17とPyTorch2.14.1+cu130の専用環境を使った。詳細な移管・試走・復旧手順は[固定したadapterの記録](https://github.com/Motoki0705/tennis-lab/blob/f06f1bda6610838313e4b306a92a4db5e42ed579/knowledge/runs/run-ball-query-desktop-20261112/README.md)。

前半・後半6kの完成snapshotをDriveから独立して戻し、marker・全payloadのbyte・model/AdamW/RNG/schedule/sampleとlogを親が再確認した。両段6,000更新、sample位置16,000→22,000。AdamWはCNN116件／query85件で、終端stepは20,800／12,000。周期版・完成版15世代すべての独立readback receiptがverifiedだった。

固定validationの全128,668 frames、位置教師85,473 frames、元画像px・距離20px条件を維持し、旧基準および①②とscored/target/teacherのidentityが一致した。保存済みの全observed誤差からmean・median・線形P95・20px以内率を独立再集計して終端JSONと一致した。[親の完了照合](../../runs/run-i1899-convnext-1query-t64-t64-s42-12k/completion-verification.json)と[再実行スクリプト](../../runs/run-i1899-convnext-1query-t64-t64-s42-12k/verify_completed.py)を保持した。これは推論を再実行した評価ではない。

## 解釈と限界

旧基準との差は小さく、中央値以外の3指標は悪い向きだった。統計的な同等性や系列長の一般的な効果は判定しない。seed42の1回・同じvalidationだけで、seed間分散・独立testは未評価である。さらに③は最初5,500 query更新までL4、その後6,500更新をRTX5060Tiで実行した。データ・recipe・初期重みと抽出順を合わせても、GPUをまたぐ数値的な差をこの小差から除けない。

既存の①・②の終端と合わせて、指定した3変更の範囲では旧基準の主指標を改善する結果は得られなかった。summaryの「③が停止中」を完了した比較へ置き換えるためsummary_impact=updatedとする。採用checkpointの変更や新しい学習は行わない。checkpoint・frame予測・全学習ログはDriveと実行出力へ保持し、Gitには小さな設定・指標・receipt・検証だけを置いた。
