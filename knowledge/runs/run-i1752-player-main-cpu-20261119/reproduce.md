# CPU確認の再現

元の観測と原reportは#2270の `run-i2269-player-court-side-20261119` を使用する。
この確認は一括CLIのGPU実行もqueue登録もしない。結果は毎回、新しい出力先へ保存する。

```bash
task_cpu_output=outputs/tennis_scene/evaluate/video-batch-player-main-reproduce-1
mkdir -p "$task_cpu_output"
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B knowledge/runs/run-i1752-player-main-cpu-20261119/replay_saved_side.py \
  --output "$task_cpu_output/replay"
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B knowledge/runs/run-i1752-shared-fp16-20261118/check_full69_command.py \
  --report "$task_cpu_output/full69.json"
```

replayは既存evaluate.pyを呼び、NN構築・load・native loader・InferenceRuntime開始・CUDA初期化を禁止する。
69clipのcheckerは保存した本番コマンドを文字列として読み、実Hydra設定と入力検証から1計画を作る。
モデル実行とPlanExecutor.runは禁止し、仮のstoreはTemporaryDirectoryへ置く。

全体テストは `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/unit/tennis_scene tests/integration/tennis_scene tests/unit/submodules tests/unit/tasks/court_side tests/e2e/tennis_scene/test_video_batch_cli.py -n 4 --tb=short --durations=0 --durations-min=0 -q`。
収集数は同じ範囲へ `--collect-only -q -o addopts='--strict-markers --import-mode=importlib'` を指定して測定する。
