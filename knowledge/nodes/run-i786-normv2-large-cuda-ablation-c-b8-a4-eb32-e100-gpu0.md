---
id: run-i786-normv2-large-cuda-ablation-c-b8-a4-eb32-e100-gpu0
type: run
title: BLCS normalization v2 large CUDA ablation C（per-attention FFN / layer-end mHC、EB32）
issue: 786
provider: codex
session: 01a02e2e-b46e-7fc2-9a24-831f51144889
date: '2026-08-24'
status: done
config:
  model: track_query_ablation_c
  ffn_mode: per_attention
  mhc_writeback: layer_end
  hidden_dim: 512
  num_stages: 12
  effective_batch_size: 32
metrics:
  loss: 0.177992
  loss_position: 0.121324
  loss_position_x: 0.122893
  loss_position_y: 0.209889
  loss_position_z: 0.031189
  loss_presence: 0.056668
  loss_smoothness: 0.0
  loss_gravity: 0.0
  position_error: 0.330631
  presence_precision: 0.965399
  presence_recall: 0.977639
  presence_f1: 0.971312
  lifecycle_presence_f1: 0.971312
  birth_frame_error: 5.976288
  death_frame_error: 6.222175
  query_reuse_count: 1.36
  illegal_overlap_count: 0.0
  segment_id_switches: 19.92
  id_switches: 19.92
  duplicate_active_tracks: 10.68
  missed_gt_frames: 19.52
  inactive_query_false_positives: 0.32
  position_error_m: 3.929544
  position_mae_x_m: 1.852081
  position_mae_y_m: 2.916369
  position_mae_z_m: 0.65812
repro:
  commit: 3667b024cf53145e79d2a9cd32249e772284404a
  branch: exp/issue-786-track-query-ablation-v2-cuda
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -c ''import atexit,runpy,torch;
    atexit.register(lambda: print("TRAIN_PEAK_CUDA_ALLOCATED_BYTES="+str(torch.cuda.max_memory_allocated())+"
    TRAIN_PEAK_CUDA_RESERVED_BYTES="+str(torch.cuda.max_memory_reserved()))); runpy.run_module("src.tasks.blcs.scripts.train",run_name="__main__")''
    --config-name train_tracking court_coordinate_normalization=v2 model=track_query_ablation_c
    paths.data_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-786-ablation-v2-cuda/data
    data.scene_dir=blcs/multi_object_norm_v2 data.seq_len_range=[128,128] data.num_views_range=[4,4]
    data.batch_size=8 data.num_workers=16 training.compile.enabled=false training.trainer.accumulate_grad_batches=4
    training.trainer.max_epochs=100 training.trainer.check_val_every_n_epoch=5 training.trainer.enable_progress_bar=false
    training.trainer.enable_model_summary=false training.early_stopping.enabled=false
    run.fast_dev_run=false run.test_after_fit=true run.output_dir=blcs/norm-v2/issue-786/t128-v4/ablation-c-large-cuda-eb32'
artifacts:
  run_dir: knowledge/runs/run-i786-normv2-large-cuda-ablation-c-b8-a4-eb32-e100-gpu0
  predictions: knowledge/runs/run-i786-normv2-large-cuda-ablation-c-b8-a4-eb32-e100-gpu0/pred_test.npz
  log: .training_queue/logs/1787506200788061418_1359772_i786_normv2_large_cuda_ablation_c_b8_a4_eb32_e100_gpu0.log
  output_dir: outputs/blcs/norm-v2/issue-786/t128-v4/ablation-c-large-cuda-eb32/logs/version_0
  curves: knowledge/runs/run-i786-normv2-large-cuda-ablation-c-b8-a4-eb32-e100-gpu0/curves.png
  tb_logdir: outputs/blcs/norm-v2/issue-786/t128-v4/ablation-c-large-cuda-eb32/logs/version_0
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
- per-attention-ffn
- layer-end-mhc
---

## 考察 / Findings

### 要約

AからmHC writeback位置だけをlayer endへ変更し、effective batch 32で100 epochを完走したC。`position_error_m=3.9295m`、`presence_f1=0.9713`、birth/death frame error `5.98/6.22`を得た。位置はAより悪いが、presenceとlifecycle timingは改善した。

### アーキテクチャ詳細

hidden 512、8 heads、12 stages、FFN 1408、4 queries、RoPE 64、CUDA CSWA（ratio 4 / radius 4）を使用する。Cは`per_attention` FFNを維持し、mHC writebackを`layer_end`へ遅らせる。パラメータ数はAと同じ109,418,048、peak CUDA allocated memoryは11,121,110,016 bytes（10.36 GiB）だった。T=128、V=4、seed 42、bf16 mixed、eager、micro-batch 8 × accumulation 4、physical GPU 0で学習した。

### メトリクスの解釈

軸別MAEはX `1.8521m`、Y `2.9164m`、Z `0.6581m`。A比でposition errorは`0.4072m`増えたが、presence F1は`0.0067`上昇し、birth/death errorはそれぞれ`1.12/2.31` frames低下した。inactive query false positivesは4 variant最少の`0.32`だった。

### アーキテクチャ⇄メトリクスの因果考察

A/Cはパラメータ数とFFN modeが同一なので、このseedではwriteback位置の影響を直接比較できる。layer-end writebackはpeak memoryを抑え、presence/lifecycleを改善する一方、早い段階でquery stateへ混合を反映しないため座標精度が下がった可能性がある。ただし、内部表現の直接計測をしていないため機序は仮説である。

### 既存実験との比較

Aに対してpeak allocated memoryを30.5%削減し、presence F1とbirth/death timingを改善したが、position errorは11.6%増えた。同じlayer-endのDに対してはposition errorが`0.2813m`低く、ID switchesも`2.72`少ないが、presence F1とbirth/death timingは僅かに劣る。

### 次に有効な実験

A/Cのmulti-seed比較を優先し、query混合係数・presence logit・position residualのstage別統計を保存する。layer-endのmemory/lifecycle優位を維持しつつpositionを回復できるか、最後の数stageだけearly writebackにするhybridを検証する。
