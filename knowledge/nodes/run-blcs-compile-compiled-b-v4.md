---
id: run-blcs-compile-compiled-b-v4
type: run
title: shared_compile_blcs_compiled_b_3ep_v4
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
  wall_time_s: 372.58
  steady_train_epoch_s: 23.5
  steady_train_batch_ms: 293.75
  peak_cuda_allocated_bytes: 5066085888
  train_loss_epoch_3: 0.10099487
  test_loss: 0.098785713
  test_position_error: 0.339648724
repro:
  commit: 2f92bcb473e7ba653a6998db1049551bd7ac7938
  branch: feat/shared-torch-compile
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: '/usr/bin/time -f "BENCHMARK_WALL_SECONDS=%e BENCHMARK_MAX_RSS_KB=%M" env
    TORCHINDUCTOR_CACHE_DIR=/home/kamimura/projects/tennis-lab/.cache/torchinductor/shared_compile_abba_v4_compiled_b
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -c "import atexit,runpy,torch; atexit.register(lambda: print(\"BENCHMARK_PEAK_CUDA_BYTES=\"+str(torch.cuda.max_memory_allocated())));
    runpy.run_module(\"src.tasks.blcs.scripts.train\",run_name=\"__main__\")" --config-name
    train_tracking paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs
    paths.checkpoint_root=/home/kamimura/projects/tennis-lab/ckpt data.scene_dir=blcs/compile_benchmark_schema_v1
    data.seq_len_range=[1024,1024] data.num_views_range=[3,3] data.batch_size=1 model=track_query_small
    model.dropout=0.0 model.cswa.backend=cuda training.trainer.precision=32-true training.trainer.max_epochs=3
    training.trainer.log_every_n_steps=1 training.trainer.enable_model_summary=false
    training.warmup_steps=0 training.checkpoint.enabled=false training.early_stopping.enabled=false
    training.lr_monitor.enabled=false training.qualitative_logging.enabled=false run.output_dir=blcs/compile_training_abba_v4/compiled_b'
artifacts:
  run_dir: knowledge/runs/run-blcs-compile-compiled-b-v4
  output_dir: outputs/blcs/compile_training_abba_v4/compiled_b/logs/version_0
parents: [run-blcs-compile-eager-a-v4]
relations:
- {to: run-blcs-compile-eager-b-v4, rel: compares}
tags: [blcs, torch-compile, hybrid-cswa, cuda, training-benchmark, compiled]
---

## 考察 / Findings

### 要約

compiled Aとは別の空cacheを使った再現run。3 epochとtestを372.58秒で完走し、steady-stateは23.5秒/epoch（293.75 ms/batch）だった。

### アーキテクチャ詳細

条件はcompiled Aと同一で、Inductor cache directoryだけを分離した。datasetは速度比較専用schema datasetであり、物理精度の評価用ではない。

### メトリクスの解釈

steady-stateは対応するeager Bの1.79倍。peak CUDA allocatedは5,066,085,888 bytesでeager Bより19.6%低い。最終train lossは0.10099487、test lossは0.098785713でeager Bとの差はいずれも1e-5未満だった。cold wall timeはeager Bの2.63倍。

### アーキテクチャ⇄メトリクスの因果考察

compiled Aと同じmemory値および近いlossを再現したため、memory削減と数値同等性はcache偶然ではない。cold compile時間とsteady throughputにはA/B差があり、1 runだけの倍率は代表値にしない。

### 既存実験との比較

compiled A/Bのsteady epoch平均は21.0秒、eager A/Bは40.0秒で、ABBA集約のsteady-state高速化は1.90倍。cold wall平均は418.02秒対140.08秒でcompiledが2.98倍遅い。

### 次に有効な実験

実運用の長期epoch数とshape分布でcompile cache hit率を計測し、固定shape結果を一般化できるか検証する。
