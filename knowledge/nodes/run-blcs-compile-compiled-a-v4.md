---
id: run-blcs-compile-compiled-a-v4
type: run
title: shared_compile_blcs_compiled_a_3ep_v4
provider: codex
session: 01a023be-2b89-7a33-9856-7b46d34326d5
date: '2026-08-21'
status: done
config:
  model: track_query_small
  data: compile_benchmark_schema_v1
  attention_backend: hybrid_cswa_cuda
  compile_enabled: true
  compile_mode: default
  precision: 32-true
  epochs: 3
  sequence_length: 1024
  num_views: 3
  batch_size: 1
  accumulate_grad_batches: 8
metrics:
  wall_time_s: 463.46
  steady_train_epoch_s: 18.5
  steady_train_batch_ms: 231.25
  peak_cuda_allocated_bytes: 5066085888
  train_loss_epoch_3: 0.100996658
  test_loss: 0.098784387
  test_position_error: 0.339659393
repro:
  commit: 2f92bcb473e7ba653a6998db1049551bd7ac7938
  branch: feat/shared-torch-compile
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: '/usr/bin/time -f "BENCHMARK_WALL_SECONDS=%e BENCHMARK_MAX_RSS_KB=%M" env
    TORCHINDUCTOR_CACHE_DIR=/home/kamimura/projects/tennis-lab/.cache/torchinductor/shared_compile_abba_v4_compiled_a
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -c "import atexit,runpy,torch; atexit.register(lambda: print(\"BENCHMARK_PEAK_CUDA_BYTES=\"+str(torch.cuda.max_memory_allocated())));
    runpy.run_module(\"src.tasks.blcs.scripts.train\",run_name=\"__main__\")" --config-name
    train_tracking paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs
    paths.checkpoint_root=/home/kamimura/projects/tennis-lab/ckpt data.scene_dir=blcs/compile_benchmark_schema_v1
    data.seq_len_range=[1024,1024] data.num_views_range=[3,3] data.batch_size=1 model=track_query_small
    model.dropout=0.0 model.cswa.backend=cuda training.trainer.precision=32-true training.trainer.max_epochs=3
    training.trainer.log_every_n_steps=1 training.trainer.enable_model_summary=false
    training.warmup_steps=0 training.checkpoint.enabled=false training.early_stopping.enabled=false
    training.lr_monitor.enabled=false training.qualitative_logging.enabled=false run.output_dir=blcs/compile_training_abba_v4/compiled_a'
artifacts:
  run_dir: knowledge/runs/run-blcs-compile-compiled-a-v4
  output_dir: outputs/blcs/compile_training_abba_v4/compiled_a/logs/version_0
parents: [run-blcs-compile-eager-a-v4]
relations:
- {to: run-blcs-compile-eager-a-v4, rel: compares}
tags: [blcs, torch-compile, hybrid-cswa, cuda, training-benchmark, compiled]
---

## 考察 / Findings

### 要約

独立Inductor cacheを使ったcompiled A。3 epochとtestを完走し、steady-stateは18.5秒/epoch（231.25 ms/batch）でeager Aの2.05倍だったが、cold-start込みwall timeは463.46秒だった。

### アーキテクチャ詳細

eager Aと同じ`track_query_small`、Hybrid CSWA CUDA、T=1024、3 cameras、batch size 1、gradient accumulation 8、FP32、dropout 0を使用した。差分は共有training config既定の`torch.compile(backend="inductor", mode="default", fullgraph=False, dynamic=False)`のみ。

### メトリクスの解釈

peak CUDA allocatedは5,066,085,888 bytesでeagerから19.6%低下した。最終train lossのeager Aとの差は7.0e-6、test loss差は7.2e-6で、短期学習の数値挙動は同等だった。wall timeはeager Aの3.35倍であり、3 epochではcompile costを償却できない。

### アーキテクチャ⇄メトリクスの因果考察

固定shapeのgraph再利用によりepoch 2–3は高速化した。一方、sanity/evalとtrainingのgraph生成がcold-startを支配する。CUDA Graphsを有効化する`reduce-overhead`はmodel内部graph breakのtensor lifetimeと衝突したため、全task既定には非CUDA-Graphsの`default`を採用した。

### 既存実験との比較

compiled Bも独立cacheで同方向のsteady-state高速化とmemory削減を示している。compile時間のrun間分散が大きいため、cold wall timeはA/B平均で評価する。

### 次に有効な実験

長期runでcache生成後のepochを増やし、rough break-evenである約18 epochを実測する。可変T/Vによるrecompile回数も別途測定する。
