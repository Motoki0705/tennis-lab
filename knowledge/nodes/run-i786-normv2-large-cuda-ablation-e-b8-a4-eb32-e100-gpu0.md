---
id: run-i786-normv2-large-cuda-ablation-e-b8-a4-eb32-e100-gpu0
type: run
title: BLCS normalization v2 large CUDA ablation E（query-only FFN / layer-end mHC、EB32）
issue: 786
provider: codex
session: 01a02e2e-b46e-7fc2-9a24-831f51144889
date: '2026-08-24'
status: done
config:
  model: track_query_ablation_e
  ffn_mode: shared
  mhc_writeback: layer_end
  query_ffn_after_spatial: true
  hidden_dim: 512
  num_stages: 12
  effective_batch_size: 32
metrics:
  loss: 0.184039
  loss_position: 0.134181
  loss_position_x: 0.115731
  loss_position_y: 0.254438
  loss_position_z: 0.032373
  loss_presence: 0.049858
  loss_smoothness: 0.0
  loss_gravity: 0.0
  position_error: 0.362328
  presence_precision: 0.960738
  presence_recall: 0.982253
  presence_f1: 0.97119
  lifecycle_presence_f1: 0.97119
  birth_frame_error: 4.513837
  death_frame_error: 5.700521
  query_reuse_count: 1.36
  illegal_overlap_count: 0.0
  segment_id_switches: 21.120001
  id_switches: 21.120001
  duplicate_active_tracks: 6.84
  missed_gt_frames: 16.32
  inactive_query_false_positives: 2.8
  position_error_m: 4.306263
  position_mae_x_m: 1.760334
  position_mae_y_m: 3.456646
  position_mae_z_m: 0.679378
repro:
  commit: 5169be60cee3a10c67f77380d296e2677097ad65
  branch: exp/issue-786-track-query-ablation-v2-cuda
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -c ''import atexit,runpy,torch;
    atexit.register(lambda: print("TRAIN_PEAK_CUDA_ALLOCATED_BYTES="+str(torch.cuda.max_memory_allocated())+"
    TRAIN_PEAK_CUDA_RESERVED_BYTES="+str(torch.cuda.max_memory_reserved()))); runpy.run_module("src.tasks.blcs.scripts.train",run_name="__main__")''
    --config-name train_tracking court_coordinate_normalization=v2 model=track_query_ablation_e
    paths.data_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-786-ablation-v2-cuda/data
    data.scene_dir=blcs/multi_object_norm_v2 data.seq_len_range=[128,128] data.num_views_range=[4,4]
    data.batch_size=8 data.num_workers=16 training.compile.enabled=false training.trainer.accumulate_grad_batches=4
    training.trainer.max_epochs=100 training.trainer.check_val_every_n_epoch=5 training.trainer.enable_progress_bar=false
    training.trainer.enable_model_summary=false training.early_stopping.enabled=false
    run.fast_dev_run=false run.test_after_fit=true run.output_dir=blcs/norm-v2/issue-786/t128-v4/ablation-e-large-cuda-eb32'
artifacts:
  run_dir: knowledge/runs/run-i786-normv2-large-cuda-ablation-e-b8-a4-eb32-e100-gpu0
  predictions: knowledge/runs/run-i786-normv2-large-cuda-ablation-e-b8-a4-eb32-e100-gpu0/pred_test.npz
  log: .training_queue/logs/1787544230191286318_1839923_i786_normv2_large_cuda_ablation_e_b8_a4_eb32_e100_gpu0.log
  output_dir: outputs/blcs/norm-v2/issue-786/t128-v4/ablation-e-large-cuda-eb32/logs/version_0
  curves: knowledge/runs/run-i786-normv2-large-cuda-ablation-e-b8-a4-eb32-e100-gpu0/curves.png
  tb_logdir: outputs/blcs/norm-v2/issue-786/t128-v4/ablation-e-large-cuda-eb32/logs/version_0
parents:
- run-i786-normv2-large-cuda-ablation-d-b8-a4-eb32-e100-gpu0
relations:
- to: run-i786-normv2-large-cuda-ablation-d-b8-a4-eb32-e100-gpu0
  rel: compares
tags:
- blcs
- tracking
- normalization-v2
- cuda
- ablation
- effective-batch-32
- shared-ffn
- query-only-ffn
- layer-end-mhc
---

## 考察 / Findings

### 要約

Dのspatial attentionとquery temporal attentionの間へquery-only pre-norm SwiGLU residualを追加し、effective batch 32で100 epochを完走したE。`position_error_m=4.3063m`、`presence_f1=0.9712`、`id_switches=21.12`だった。D比でID switchesは`1.52`減ったが、position、presence、birth/death timingはいずれも改善しなかった。

### アーキテクチャ詳細

hidden 512、8 heads、12 stages、FFN 1408、4 queries、RoPE 64、CUDA CSWA（ratio 4 / radius 4）はA〜Dと共通である。EはDと同じ`shared` FFN、`layer_end` mHC writebackを維持し、各stageを`spatial attention → query-only pre-norm SwiGLU residual → query temporal attention → query/object共有FFN → mHC.post`の順に処理する。追加FFNの入力はquery tensor `(B,T,Q,D)`だけで、object tokenは通らない。パラメータ数は83,459,648、peak CUDA allocated memoryは9,873,533,952 bytes（9.20 GiB）だった。T=128、V=4、seed 42、bf16 mixed、eager、micro-batch 8 × accumulation 4、physical GPU 0で学習した。

### メトリクスの解釈

軸別MAEはX `1.7603m`、Y `3.4566m`、Z `0.6794m`である。Dに対してX/Zは改善した一方、支配的なY誤差が`0.2896m`増えたため、統合position errorは`0.0954m`（2.3%）悪化した。presence F1はDから`0.0009`低下し、birth/death errorも`0.09/0.34` frames増えた。ID switchesは6.7%減ったものの、Bの`17.04`には届いていない。

### アーキテクチャ⇄メトリクスの因果考察

D/Eは追加query-only FFN以外の構成を固定している。このseedでは、spatial query表現の容量追加がX/Z回帰とidentity continuityには部分的に寄与した可能性がある一方、Y回帰とlifecycle判定を含む総合性能には寄与しなかった。パラメータを45.1%、peak memoryを15.5%増やしても主要metricが改善しないため、Dの弱点をspatial後のquery容量不足だけへ帰属させる観測証拠は得られない。内部表現を直接測定しておらず、因果機序は単一seedからの仮説である。

### 既存実験との比較

D比でposition errorは2.3%増、presence F1は0.09 percentage point低下、birth/death errorは`0.09/0.34` frames増加した。ID switchesだけは`1.52`減った。A〜E全体では、positionはA、identity continuityはB、presence/lifecycle timingはDが最良で、Eはいずれの主要metricでも新しい最良値を作らなかった。

### 次に有効な実験

query-only FFN自体を検証し続ける場合はD/Eを3 seeds以上で比較し、ID switch低下が再現するか確認する。ただし現時点では追加容量に対する総合的な利得がなく、positionを主目的とするAまたは低容量・高lifecycle性能のDを優先する。Dのposition改善にはquery-only FFN追加より、Y軸誤差へ直接作用するloss/head設計を小規模に比較する方が有効である。
