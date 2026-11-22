`codex/video-batch-benchmark` の HEAD `8c91bf074816b3d809ac2548f8d8050fbcda3ac9` に実装し、Draft PR #2273を更新しました。 [Draft PR #2273](https://github.com/Motoki0705/tennis-lab/pull/2273)

4条件とも、probe直前の `nvidia-smi` 1回で環境を記録し、その後はheartbeat切れ180秒と外部時間上限（probe 480秒、本計測10,800秒）で監視します。peak VRAMは各repeatのPyTorch allocatorの `allocated_peak_bytes` と `reserved_peak_bytes` です。probeと本計測は同じqueue jobで実行します。 [計測定義と再投入手順](https://github.com/Motoki0705/tennis-lab/blob/8c91bf074816b3d809ac2548f8d8050fbcda3ac9/knowledge/runs/run-i1757-video-batch-meiji/reproduce.md)

CPUテスト60件（新規9件）、pre-commit、対象HEADのPython CI・Lint・Knowledge libraryが成功しました。前回CIの修正 #2280 もmainから取り込んでいます。実GPU計測とqueue投入は行っていません。 [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44476836354) [Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/44476836347)

人間の許可後、指定worktreeから逐次条件を再投入するコマンドです。

```bash
(
  set -e
  runs=outputs/tennis_scene/evaluate/video-batch-benchmark-20261119
  driver=knowledge/runs/run-i1757-video-batch-meiji/benchmark.py
  run_dir="$runs/serial-heartbeat-1"
  head=8c91bf074816b3d809ac2548f8d8050fbcda3ac9
  native=$(jq -er .native_dino.path "$runs/serial-retry-1/preflight.json")
  common_git_dir=$(git rev-parse --path-format=absolute --git-common-dir)
  export TRAINING_QUEUE_DIR="${common_git_dir%[local path]}[local path]"
  queue=.agents/skills/training-queue/scripts/training_queue.sh
  runner=(env "TENNIS_LAB_DINO_EXTENSION=$native" .venv/bin/python -B -u "$driver")
  args=(--root "$run_dir" --condition serial --expect-commit "$head")

  CUDA_VISIBLE_DEVICES='' "${runner[@]}" freeze --root "$run_dir"
  CUDA_VISIBLE_DEVICES='' "${runner[@]}" preflight "${args[@]}"
  printf -v probe '%q ' "${runner[@]}" probe "${args[@]}"
  printf -v measure '%q ' "${runner[@]}" run "${args[@]}"
  bash "$queue" add "$probe && $measure" --resource all --provider codex \
    --session "${CODEX_THREAD_ID:?}" --issue 1757 --name i1757-serial-heartbeat-1 \
    --idempotency-key "i1757/serial-heartbeat-1/$head"
)
```