# DINO fp16のGPU停止の切り分け

[#1757の逐次probe](../run-i1757-video-batch-meiji/README.md)の `video_000/clip_027` の `cam0` を使う。両条件で標準の人物検出コンポーネントを使い、全235フレームを先頭から順に、DINO fp16の単一フレーム推論で処理する。動画・設定・重み・DINOソース・ネイティブ拡張は逐次probeの固定記録と照合する。

各条件は新規プロセスで起動する。Aでは `nvidia-smi` の照会を行わない。Aの完走時だけBを実行する。Bでは親プロセスから既存のhealth guardと同じ `nvidia-smi` を1秒間隔、10秒のタイムアウトで呼ぶ。実行開始前に1分load averageとpytestプロセス数を記録し、load averageが2未満かつpytestが0件、または待機30分が経過した時点で開始する。

停止の判定はheartbeatの更新停止60秒、条件全体の上限480秒、GPU照会エラーまたはworkerの異常終了。停止を記録したらworkerを終了させ、その条件の再試行と後続条件の実行を行わない。heartbeatに0始まりのフレーム位置、完了済みフレームとDINO forwardの数、そのプロセスのPyTorch割当メモリとピークを記録する。

## 準備とqueue投入

`repro.commit` のworktreeで、`previous` に前回逐次probeの固定済み出力を指定する。`root` は未作成のディレクトリを指定する。準備はCPUで行う。

```bash
driver=knowledge/runs/run-i1757-dino-gpu-health/run.py
previous=outputs/tennis_scene/evaluate/video-batch-benchmark-20261119/serial-retry-1
root="$PWD/outputs/tennis_scene/evaluate/dino-gpu-health-20261119"
head=$(git rev-parse HEAD)
native=$(jq -er .native_dino.path "$previous/preflight.json")

CUDA_VISIBLE_DEVICES='' TENNIS_LAB_DINO_EXTENSION="$native" \
  .venv/bin/python -B "$driver" prepare --root "$root" --previous "$previous" --expect-commit "$head"

common_git_dir=$(git rev-parse --path-format=absolute --git-common-dir)
export TRAINING_QUEUE_DIR="${common_git_dir%/.git}/.training_queue"
queue=.agents/skills/training-queue/scripts/training_queue.sh
printf -v job '%q ' env CUDA_VISIBLE_DEVICES=0 "TENNIS_LAB_DINO_EXTENSION=$native" \
  .venv/bin/python -B -u "$driver" run --root "$root" --expect-commit "$head"
bash "$queue" add "$job" --resource all --name i1757-dino-gpu-health \
  --provider codex --session "$CODEX_THREAD_ID" --issue 1757 \
  --idempotency-key "i1757/dino-gpu-health/$head/attempt-1"
bash "$queue" start
```

## 記録

- 共有queueの `logs/<job ID>.log` と `root/results.json` で条件の終了と実行順を確認する。
- `root/A` と `root/B` の `started.json`・`cpu-load.jsonl`・`heartbeat.json`・`terminal.json` に開始条件と到達位置が残る。`completed.json` は全フレーム完了時に作成する。
- `worker.log` と `recording.json` にモデル実行の記録、`gpu-queries.jsonl` にGPU照会の結果を残す。条件Aの照会記録は空になる。
- `stop.json` に監視が検知した停止時刻を記録する。条件ごとの `kernel-dxg.log`、`windows-nvlddmkm.log` と `driver-log-status.json` にWSL・Windowsの関連イベントと取得状態を残す。
