---
id: run-i786-normv2-large-cuda-ablation-d-b8-a4-eb32-e100-gpu0
type: run
title: BLCS normalization v2 large CUDA ablation D（shared FFN / layer-end mHC、EB32）
issue: 786
provider: codex
session: 01a02e2e-b46e-7fc2-9a24-831f51144889
date: '2026-08-24'
status: done
config:
  model: track_query_ablation_d
  ffn_mode: shared
  mhc_writeback: layer_end
  hidden_dim: 512
  num_stages: 12
  effective_batch_size: 32
metrics:
  loss: 0.181107
  loss_position: 0.132236
  loss_position_x: 0.133199
  loss_position_y: 0.229764
  loss_position_z: 0.033746
  loss_presence: 0.04887
  loss_smoothness: 0.0
  loss_gravity: 0.0
  position_error: 0.354301
  presence_precision: 0.961018
  presence_recall: 0.983999
  presence_f1: 0.972115
  lifecycle_presence_f1: 0.972115
  birth_frame_error: 4.428693
  death_frame_error: 5.36457
  query_reuse_count: 1.28
  illegal_overlap_count: 0.0
  segment_id_switches: 22.639999
  id_switches: 22.639999
  duplicate_active_tracks: 14.56
  missed_gt_frames: 14.64
  inactive_query_false_positives: 2.4
  position_error_m: 4.210864
  position_mae_x_m: 1.988215
  position_mae_y_m: 3.167047
  position_mae_z_m: 0.696831
repro:
  commit: 3667b024cf53145e79d2a9cd32249e772284404a
  branch: exp/issue-786-track-query-ablation-v2-cuda
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -c ''import atexit,runpy,torch;
    atexit.register(lambda: print("TRAIN_PEAK_CUDA_ALLOCATED_BYTES="+str(torch.cuda.max_memory_allocated())+"
    TRAIN_PEAK_CUDA_RESERVED_BYTES="+str(torch.cuda.max_memory_reserved()))); runpy.run_module("src.tasks.blcs.scripts.train",run_name="__main__")''
    --config-name train_tracking court_coordinate_normalization=v2 model=track_query_ablation_d
    paths.data_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-786-ablation-v2-cuda/data
    data.scene_dir=blcs/multi_object_norm_v2 data.seq_len_range=[128,128] data.num_views_range=[4,4]
    data.batch_size=8 data.num_workers=16 training.compile.enabled=false training.trainer.accumulate_grad_batches=4
    training.trainer.max_epochs=100 training.trainer.check_val_every_n_epoch=5 training.trainer.enable_progress_bar=false
    training.trainer.enable_model_summary=false training.early_stopping.enabled=false
    run.fast_dev_run=false run.test_after_fit=true run.output_dir=blcs/norm-v2/issue-786/t128-v4/ablation-d-large-cuda-eb32'
artifacts:
  run_dir: knowledge/runs/run-i786-normv2-large-cuda-ablation-d-b8-a4-eb32-e100-gpu0
  predictions: knowledge/runs/run-i786-normv2-large-cuda-ablation-d-b8-a4-eb32-e100-gpu0/pred_test.npz
  log: .training_queue/logs/1787506200856834056_1359794_i786_normv2_large_cuda_ablation_d_b8_a4_eb32_e100_gpu0.log
  output_dir: outputs/blcs/norm-v2/issue-786/t128-v4/ablation-d-large-cuda-eb32/logs/version_0
  curves: knowledge/runs/run-i786-normv2-large-cuda-ablation-d-b8-a4-eb32-e100-gpu0/curves.png
  tb_logdir: outputs/blcs/norm-v2/issue-786/t128-v4/ablation-d-large-cuda-eb32/logs/version_0
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
- layer-end-mhc
---

## 考察 / Findings

### 要約

shared FFNとlayer-end mHCを組み合わせ、effective batch 32で100 epochを完走したD。4 variant中最高の`presence_f1=0.9721`、最小のbirth/death frame error `4.43/5.36`、最少のmissed GT frames `14.64`を得た。一方、`position_error_m=4.2109m`と`id_switches=22.64`は4 variant中最大だった。

### アーキテクチャ詳細

hidden 512、8 heads、12 stages、FFN 1408、4 queries、RoPE 64、CUDA CSWA（ratio 4 / radius 4）は共通である。Dは`shared` FFNと`layer_end` mHC writebackを使用する。パラメータ数は57,501,248、peak CUDA allocated memoryは8,550,460,416 bytes（7.96 GiB）で、いずれも同じ実験群の最小クラスだった。T=128、V=4、seed 42、bf16 mixed、eager、micro-batch 8 × accumulation 4、physical GPU 0で学習した。

### メトリクスの解釈

軸別MAEはX `1.9882m`、Y `3.1670m`、Z `0.6968m`で全軸が4 variant中最大だった。presence recallは`0.9840`と高く、lifecycle区間を拾う能力は最良だが、追跡対象の座標精度とidentity continuityにはつながっていない。

### アーキテクチャ⇄メトリクスの因果考察

B/Dはshared FFNとパラメータ数を固定している。layer-endへ変えたDはBよりpresence/lifecycleが改善しpeak memoryも減ったが、positionとID switchesは悪化した。この比較は、遅いwritebackがqueryのactive/inactive判定には有利でも、object temporal処理中のidentity-conditioned座標更新を弱める可能性を示す。機序は単一seedの観測からの仮説である。

### 既存実験との比較

B比でpresence F1は`0.0082`上昇し、birth/death errorは`4.35/5.15` frames低下、peak allocated memoryは36.4%減った。一方、position errorは12.1%増え、ID switchesは`5.60`増えた。A比ではmemoryを46.6%削減したがposition errorは19.5%増えた。

### 次に有効な実験

B/Dのmulti-seed比較でlifecycle優位を確認する。Dを採用候補にする場合は、position head容量またはidentity consistency lossだけを増やす小規模ablationを行い、低memoryと高presenceを保ちながらposition/IDを回復できるか評価する。
