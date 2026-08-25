---
id: run-i786-normv2-large-cuda-ablation-b-b8-a4-eb32-e100-gpu0
type: run
title: BLCS normalization v2 large CUDA ablation B（shared FFN / early mHC、EB32）
issue: 786
provider: codex
session: 01a02e2e-b46e-7fc2-9a24-831f51144889
date: '2026-08-24'
status: done
config:
  model: track_query_ablation_b
  ffn_mode: shared
  mhc_writeback: after_object_temporal
  hidden_dim: 512
  num_stages: 12
  effective_batch_size: 32
metrics:
  loss: 0.186157
  loss_position: 0.114546
  loss_position_x: 0.094657
  loss_position_y: 0.216466
  loss_position_z: 0.032517
  loss_presence: 0.071611
  loss_smoothness: 0.0
  loss_gravity: 0.0
  position_error: 0.316127
  presence_precision: 0.956876
  presence_recall: 0.9713
  presence_f1: 0.963868
  lifecycle_presence_f1: 0.963868
  birth_frame_error: 8.774966
  death_frame_error: 10.517749
  query_reuse_count: 1.04
  illegal_overlap_count: 0.0
  segment_id_switches: 17.040001
  id_switches: 17.040001
  duplicate_active_tracks: 15.16
  missed_gt_frames: 24.200001
  inactive_query_false_positives: 2.48
  position_error_m: 3.757173
  position_mae_x_m: 1.480358
  position_mae_y_m: 2.995347
  position_mae_z_m: 0.664296
repro:
  commit: 3667b024cf53145e79d2a9cd32249e772284404a
  branch: exp/issue-786-track-query-ablation-v2-cuda
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -c ''import atexit,runpy,torch;
    atexit.register(lambda: print("TRAIN_PEAK_CUDA_ALLOCATED_BYTES="+str(torch.cuda.max_memory_allocated())+"
    TRAIN_PEAK_CUDA_RESERVED_BYTES="+str(torch.cuda.max_memory_reserved()))); runpy.run_module("src.tasks.blcs.scripts.train",run_name="__main__")''
    --config-name train_tracking court_coordinate_normalization=v2 model=track_query_ablation_b
    paths.data_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-786-ablation-v2-cuda/data
    data.scene_dir=blcs/multi_object_norm_v2 data.seq_len_range=[128,128] data.num_views_range=[4,4]
    data.batch_size=8 data.num_workers=16 training.compile.enabled=false training.trainer.accumulate_grad_batches=4
    training.trainer.max_epochs=100 training.trainer.check_val_every_n_epoch=5 training.trainer.enable_progress_bar=false
    training.trainer.enable_model_summary=false training.early_stopping.enabled=false
    run.fast_dev_run=false run.test_after_fit=true run.output_dir=blcs/norm-v2/issue-786/t128-v4/ablation-b-large-cuda-eb32'
artifacts:
  run_dir: knowledge/runs/run-i786-normv2-large-cuda-ablation-b-b8-a4-eb32-e100-gpu0
  predictions: knowledge/runs/run-i786-normv2-large-cuda-ablation-b-b8-a4-eb32-e100-gpu0/pred_test.npz
  log: .training_queue/logs/1787506200719864547_1359734_i786_normv2_large_cuda_ablation_b_b8_a4_eb32_e100_gpu0.log
  output_dir: outputs/blcs/norm-v2/issue-786/t128-v4/ablation-b-large-cuda-eb32/logs/version_0
  curves: knowledge/runs/run-i786-normv2-large-cuda-ablation-b-b8-a4-eb32-e100-gpu0/curves.png
  tb_logdir: outputs/blcs/norm-v2/issue-786/t128-v4/ablation-b-large-cuda-eb32/logs/version_0
parents:
- run-i786-normv2-large-cuda-ablation-a-b8-a4-eb32-e100-gpu0
relations:
- to: run-i786-normv2-large-cuda-ablation-a-b8-a4-eb32-e100-gpu0
  rel: compares
tags:
- blcs
- tracking
- normalization-v2
- cuda
- ablation
- effective-batch-32
- shared-ffn
- early-mhc
---

## 考察 / Findings

### 要約

AからFFNだけをsharedへ変更し、effective batch 32で100 epochを完走したB。`position_error_m=3.7572m`、`presence_f1=0.9639`で、4 variant中最少の`id_switches=17.04`を得た。

### アーキテクチャ詳細

hidden 512、8 heads、12 stages、FFN 1408、4 queries、RoPE 64、CUDA CSWA（ratio 4 / radius 4）は共通である。Bは`shared` FFNと`after_object_temporal` mHC writebackを使用する。パラメータ数は57,501,248でA/Cより47.4%少なく、peak CUDA allocated memoryは13,434,806,272 bytes（12.51 GiB）だった。T=128、V=4、seed 42、bf16 mixed、eager、micro-batch 8 × accumulation 4、physical GPU 0という学習条件もAと一致する。

### メトリクスの解釈

軸別MAEはX `1.4804m`、Y `2.9953m`、Z `0.6643m`。A比でpositionは`0.2349m`悪化し、presence F1はほぼ同等（`-0.0007`）だった一方、ID switchesは`20.16→17.04`へ減った。birth/death errorは`8.77/10.52`で4 variant中最大だった。

### アーキテクチャ⇄メトリクスの因果考察

A/BはmHC timingを固定しているため、差はFFN共有とそれに伴う容量削減に対応する。共有FFNがbranch間の表現を揃える正則化として働きID対応を安定させた可能性があるが、同時に約半分へ減ったパラメータ容量が座標精度とlifecycle timingを制限した可能性もある。単一seedのため両者は分離できない。

### 既存実験との比較

Aに対してpeak allocated memoryを16.0%削減しつつ、ID switchesを15.5%削減したが、position errorは6.7%増えた。同じshared FFNのDと比べるとpositionとID switchesはBが良く、presence F1とbirth/death timingはDが良い。

### 次に有効な実験

A/Bを複数seedで再実行したうえで、shared FFNのhidden/FFN幅を増やしAとパラメータ数を揃える対照を追加する。これにより、ID switch改善がweight sharing由来か、容量削減による正則化由来かを分離する。
