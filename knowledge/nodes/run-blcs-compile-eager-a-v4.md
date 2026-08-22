---
id: run-blcs-compile-eager-a-v4
type: run
title: shared_compile_blcs_eager_a_3ep_v4
provider: codex
session: 01a023be-2b89-7a33-9856-7b46d34326d5
date: '2026-08-21'
status: done
config:
  model: track_query_small
  data: compile_benchmark_schema_v1
  attention_backend: hybrid_cswa_cuda
  compile_enabled: false
  precision: 32-true
  epochs: 3
  sequence_length: 1024
  num_views: 3
  batch_size: 1
  accumulate_grad_batches: 8
metrics:
  wall_time_s: 138.27
  steady_train_epoch_s: 38.0
  steady_train_batch_ms: 475.0
  peak_cuda_allocated_bytes: 6299367424
  train_loss_epoch_3: 0.100989655
  test_loss: 0.09877719
  test_position_error: 0.33964476
repro:
  commit: 2f92bcb473e7ba653a6998db1049551bd7ac7938
  branch: feat/shared-torch-compile
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: '/usr/bin/time -f "BENCHMARK_WALL_SECONDS=%e BENCHMARK_MAX_RSS_KB=%M" env
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -c "import atexit,runpy,torch; atexit.register(lambda: print(\"BENCHMARK_PEAK_CUDA_BYTES=\"+str(torch.cuda.max_memory_allocated())));
    runpy.run_module(\"src.tasks.blcs.scripts.train\",run_name=\"__main__\")" --config-name
    train_tracking paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs
    paths.checkpoint_root=/home/kamimura/projects/tennis-lab/ckpt data.scene_dir=blcs/compile_benchmark_schema_v1
    data.seq_len_range=[1024,1024] data.num_views_range=[3,3] data.batch_size=1 model=track_query_small
    model.dropout=0.0 model.cswa.backend=cuda training.compile.enabled=false training.trainer.precision=32-true
    training.trainer.max_epochs=3 training.trainer.log_every_n_steps=1 training.trainer.enable_model_summary=false
    training.warmup_steps=0 training.checkpoint.enabled=false training.early_stopping.enabled=false
    training.lr_monitor.enabled=false training.qualitative_logging.enabled=false run.output_dir=blcs/compile_training_abba_v4/eager_a'
artifacts:
  run_dir: knowledge/runs/run-blcs-compile-eager-a-v4
  output_dir: outputs/blcs/compile_training_abba_v4/eager_a/logs/version_0
parents: []
relations: []
tags: [blcs, torch-compile, hybrid-cswa, cuda, training-benchmark, eager]
---

## 考察 / Findings

### 要約

ABBA順序の先頭に置いたeager基準run。3 epochとtestを138.27秒で完走し、epoch 2–3の学習は平均38.0秒（475.0 ms/batch）だった。

### アーキテクチャ詳細

`track_query_small`（hidden dim 256、4 heads、8 stages）を使用し、時系列attentionはHybrid CSWAのCUDA backendとした。T=1024、3 cameras、batch size 1、gradient accumulation 8、FP32、attention dropout 0で固定した。datasetは正式なBLCS disk schemaを満たす速度比較専用100 scene datasetであり、物理精度の主張には使用しない。

### メトリクスの解釈

wall timeはcold-startを含むtraining、sanity check、testの全体時間である。peak CUDA allocatedは6,299,367,424 bytes。最終train loss 0.100989655、test loss 0.09877719をcompiled群の数値同等性基準とする。

### アーキテクチャ⇄メトリクスの因果考察

eagerではcompiler初期化がなく、3 epochという短いrunのwall timeがsteady-state batch時間をほぼ直接反映する。一方、同じ固定shapeはcompiled側のgraph再利用には有利な条件である。

### 既存実験との比較

比較対象は同groupのcompiled A/Bと順序末尾のeager B。単独runの差を採用せず、ABBA平均で判断する。

### 次に有効な実験

同じshapeで18 epoch以上走らせ、cold-startを償却したtotal wall timeの交差点を確認する。run別prediction bundleを残す場合はqueue実行時にartifact rootをrepro directoryへ明示する。
