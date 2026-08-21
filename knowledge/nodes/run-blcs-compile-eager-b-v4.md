---
id: run-blcs-compile-eager-b-v4
type: run
title: shared_compile_blcs_eager_b_3ep_v4
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
  wall_time_s: 141.89
  steady_train_epoch_s: 42.0
  steady_train_batch_ms: 525.0
  peak_cuda_allocated_bytes: 6299367424
  train_loss_epoch_3: 0.100999817
  test_loss: 0.098792292
  test_position_error: 0.339671254
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
    training.lr_monitor.enabled=false training.qualitative_logging.enabled=false run.output_dir=blcs/compile_training_abba_v4/eager_b'
artifacts:
  run_dir: knowledge/runs/run-blcs-compile-eager-b-v4
  output_dir: outputs/blcs/compile_training_abba_v4/eager_b/logs/version_0
parents: [run-blcs-compile-eager-a-v4]
relations:
- {to: run-blcs-compile-eager-a-v4, rel: confirms}
tags: [blcs, torch-compile, hybrid-cswa, cuda, training-benchmark, eager]
---

## 考察 / Findings

### 要約

ABBA順序の末尾に置いたeager再現run。3 epochとtestを141.89秒で完走し、steady-stateは42.0秒/epoch（525.0 ms/batch）だった。

### アーキテクチャ詳細

条件はeager Aと同一。2本のcompiled cold runの後に実行し、温度・順序など時間方向のドリフトを確認する役割を持つ。

### メトリクスの解釈

eager Aとのwall time差は3.62秒（2.6%）、peak CUDA allocatedは同じ6,299,367,424 bytesだった。最終train/test lossも約1e-5以内で一致し、比較期間中の大きな環境ドリフトは観測されない。

### アーキテクチャ⇄メトリクスの因果考察

eager A/Bの近いwall timeと同一peak memoryにより、compiled A/Bとの差は実行順だけでは説明しにくい。steady epochはAより約10.5%遅く、ABBA平均を用いる必要性は残る。

### 既存実験との比較

compiled Bはsteady-stateで1.79倍高速、peak memoryは19.6%低いが、cold wall timeは2.63倍長い。これはforward-only benchmarkの高速化をtraining steady-stateでも支持する一方、短期run全体の高速化は否定する。

### 次に有効な実験

18 epoch以上のAB比較と、実datasetでの可変長bucket別compile挙動を測る。
