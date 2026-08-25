---
id: run-i786-normv2-large-cuda-ablation-a-b8-a4-eb32-e100-gpu0
type: run
title: BLCS normalization v2 large CUDA ablation A（per-attention FFN / early mHC、EB32）
issue: 786
provider: codex
session: 01a02e2e-b46e-7fc2-9a24-831f51144889
date: '2026-08-24'
status: done
config:
  model: track_query_ablation_a
  ffn_mode: per_attention
  mhc_writeback: after_object_temporal
  hidden_dim: 512
  num_stages: 12
  effective_batch_size: 32
metrics:
  loss: 0.179519
  loss_position: 0.107192
  loss_position_x: 0.090376
  loss_position_y: 0.199099
  loss_position_z: 0.032101
  loss_presence: 0.072327
  loss_smoothness: 0.0
  loss_gravity: 0.0
  position_error: 0.296367
  presence_precision: 0.956233
  presence_recall: 0.973329
  presence_f1: 0.964586
  lifecycle_presence_f1: 0.964586
  birth_frame_error: 7.099174
  death_frame_error: 8.530895
  query_reuse_count: 1.12
  illegal_overlap_count: 0.0
  segment_id_switches: 20.16
  id_switches: 20.16
  duplicate_active_tracks: 9.32
  missed_gt_frames: 23.360001
  inactive_query_false_positives: 1.12
  position_error_m: 3.522323
  position_mae_x_m: 1.441131
  position_mae_y_m: 2.782214
  position_mae_z_m: 0.666136
repro:
  commit: 3667b024cf53145e79d2a9cd32249e772284404a
  branch: exp/issue-786-track-query-ablation-v2-cuda
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -c ''import atexit,runpy,torch;
    atexit.register(lambda: print("TRAIN_PEAK_CUDA_ALLOCATED_BYTES="+str(torch.cuda.max_memory_allocated())+"
    TRAIN_PEAK_CUDA_RESERVED_BYTES="+str(torch.cuda.max_memory_reserved()))); runpy.run_module("src.tasks.blcs.scripts.train",run_name="__main__")''
    --config-name train_tracking court_coordinate_normalization=v2 model=track_query_ablation_a
    paths.data_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-786-ablation-v2-cuda/data
    data.scene_dir=blcs/multi_object_norm_v2 data.seq_len_range=[128,128] data.num_views_range=[4,4]
    data.batch_size=8 data.num_workers=16 training.compile.enabled=false training.trainer.accumulate_grad_batches=4
    training.trainer.max_epochs=100 training.trainer.check_val_every_n_epoch=5 training.trainer.enable_progress_bar=false
    training.trainer.enable_model_summary=false training.early_stopping.enabled=false
    run.fast_dev_run=false run.test_after_fit=true run.output_dir=blcs/norm-v2/issue-786/t128-v4/ablation-a-large-cuda-eb32'
artifacts:
  run_dir: knowledge/runs/run-i786-normv2-large-cuda-ablation-a-b8-a4-eb32-e100-gpu0
  predictions: knowledge/runs/run-i786-normv2-large-cuda-ablation-a-b8-a4-eb32-e100-gpu0/pred_test.npz
  log: .training_queue/logs/1787506200648474581_1359687_i786_normv2_large_cuda_ablation_a_b8_a4_eb32_e100_gpu0.log
  output_dir: outputs/blcs/norm-v2/issue-786/t128-v4/ablation-a-large-cuda-eb32/logs/version_0
  curves: knowledge/runs/run-i786-normv2-large-cuda-ablation-a-b8-a4-eb32-e100-gpu0/curves.png
  tb_logdir: outputs/blcs/norm-v2/issue-786/t128-v4/ablation-a-large-cuda-eb32/logs/version_0
parents: []
relations: []
tags:
- blcs
- tracking
- normalization-v2
- cuda
- ablation
- effective-batch-32
- per-attention-ffn
- early-mhc
---

## 考察 / Findings

### 要約

normalization v2とlarge CUDA構成を組み合わせ、micro-batch 8 × gradient accumulation 4（effective batch 32）で100 epochを完走したA。4 variant中で最小の`position_error_m=3.5223m`を得た。`presence_f1=0.9646`、`id_switches=20.16`、birth/death frame errorは`7.10/8.53`だった。

### アーキテクチャ詳細

hidden 512、8 heads、12 stages、FFN 1408、4 queries、RoPE 64を使用した。Aは各attention branchに独立FFNを持つ`per_attention`と、object temporal直後にmHCをwritebackする`after_object_temporal`の組合せである。CSWAはCUDA backend、compression ratio 4、window radius 4。パラメータ数は109,418,048、学習時peak CUDA allocated memoryは16,001,261,568 bytes（14.90 GiB）だった。T=128、V=4、seed 42、bf16 mixed、eager実行、early stoppingなしで、共有training queueからphysical GPU 0へ投入した。

### メトリクスの解釈

軸別MAEはX `1.4411m`、Y `2.7822m`、Z `0.6661m`で、Y誤差が支配的だった。positionを主目的にするとAが最良だが、presence/lifecycleではC/D、ID switchではBが良く、単一の総合勝者ではない。全variantで`illegal_overlap_count=0`だった。

### アーキテクチャ⇄メトリクスの因果考察

観測上、独立FFNとearly mHCの組合せは座標回帰に最も有利だった。A→BでFFN共有だけを変えるとposition errorが`0.2349m`増え、A→Cでwriteback位置だけを変えると`0.4072m`増えたため、Aの両要素はこのseedでposition精度へ寄与する方向で整合する。一方、FFN共有はパラメータ数も変えるため、A/B差を共有方式そのものだけへ帰属させることはできない。

### 既存実験との比較

B比でposition errorは6.3%低く、C比で10.4%、D比で16.4%低い。反対にpresence F1はCより`0.0067`、Dより`0.0075`低く、位置精度とlifecycle検出のtrade-offが見える。以前の`multiview_axial_base` v2 baselineとはmodel・tracking task・batch契約が異なるため、絶対値の因果比較には使わない。

### 次に有効な実験

A/CとB/Dを3 seeds以上で再実行し、同一パラメータ数内でearly mHCのposition優位とlayer-end mHCのlifecycle優位が再現するか確認する。その後、Aを基準にpresence loss weightまたはlifecycle thresholdを小規模sweepし、position優位を保ったままC/DのF1へ近づける。
