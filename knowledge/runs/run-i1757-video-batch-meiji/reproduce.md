# meiji_4cam 3クリップの動画一括ベンチマーク

[conditions.json](conditions.json) が対象・順序・設定・資源枠の正本です。meiji_4cam の video_000 から clip_000・clip_027・clip_037 を選びます。4カメラで合計3,532 camera-frameです。共有courtの校正・drift bundleをpinし、各clipのcourt sideは #2269 のplayer証拠で判定します。基準方向はcam0〜cam3の順に `[false, false, true, true]` です。X3D/CSN＋DPT T32によるball_detection、ball_points、ball_triangulationを含みます。ダブルス各side 2人、DINOとViTPoseはfp16です。

| 条件 | 実行方法 |
| --- | --- |
| `serial` | 固定した実行commitの `TennisSceneOrchestrator.run` をクリップ順に呼びます。各クリップは独立したモデルruntimeを持ち、共通のdataset publisherから公開します。 |
| `reuse` | 同じ実行commitの `run_video_batch` CLIで3クリップを1計画に入れます。クリップ・カメラ横断のbatch上限は1です。 |
| `batched` | #1755取り込み後、採用したadapter・モデル別上限を定義して同CLIで計測します。 |
| `embedding_reuse` | #1756取り込み後、人物対応用CLIP embedding再利用の条件を追加します。 |

`serial` と `reuse` はどちらも各モデルの既存のclip内設定を使います。DINOの画像とBallの窓はB=1、ViTPoseとHMR2のcrop上限は16、CLIP-ReIDは64です。ViTPoseの充填率はflip前のcrop数で計算します。GVHMRは1人物segmentずつ実行します。#1748のrecorderと逐次の計測手順を使い、両入口でplayer経路と入力・重み・精度・保存条件をそろえます。

## 投入前の確認

過去の同じ3クリップの一括runは約32分、device VRAM peakは6,198 MiB、RSS peakは約8.5 GiB、保存先は約1.9 GiBでした。これらを時間・容量の見積りに使います。逐次の追加ロードと新しい中間保存に余裕を取り、条件ごとに最長180分、GPU all枠、PyTorch allocator 10 GiBで実行します。開始時に空きVRAM 12 GiB相当、利用可能RAM 12 GiB、空きdisk 12 GiBを確認します。probeは同じ入力・設定のclip_027/cam0でDINO・ViTPose・CLIP-ReID・Ballの段を8分以内に実行し、録画・資源記録を各条件の直前に残します。身体段の資源見積りには過去の全段runも使います。 [先行run](../../nodes/tennis_scene/000064-run-i1752-shared-fp16-20261118.md)

各条件で初回1回、warm 2回を予約します。warmの2回で範囲・標本標準偏差を出せる最小回数です。毎回空のcomponent storeから再計算し、初回は新規workerと空のInductor/Triton cache、warmは同じworkerのimport・CUDA/compile状態を使います。モデルは各入口の通常の寿命でロード・解放します。OS page cacheは通常運用の状態を保ちます。

## 実行

worktreeの `.venv`、共有data/ckpt、gitlinkのDINO source、[共有court/ドリフト成果物](../../../src/tennis_scene/pipeline/video_court/README.ja.md)を用意します。既存DINO binaryの選択は[導入案内](../../../docs/installation/README.ja.md#dino-cuda-extension)に従います。GPUは[共有training queue](../../../.agents/skills/training-queue/SKILL.md)のall枠で実行します。

```bash
driver=knowledge/runs/run-i1757-video-batch-meiji/benchmark.py
condition=serial
run_dir="outputs/tennis_scene/evaluate/video-batch-benchmark-20261119/$condition"
commit=$(git rev-parse HEAD)
common_git_dir=$(git rev-parse --path-format=absolute --git-common-dir)
export TRAINING_QUEUE_DIR="${common_git_dir%/.git}/.training_queue"
queue=.agents/skills/training-queue/scripts/training_queue.sh

# TENNIS_LAB_DINO_EXTENSION は使用するバイナリの絶対パス
export TENNIS_LAB_DINO_EXTENSION
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B "$driver" freeze --root "$run_dir"
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B "$driver" preflight \
  --root "$run_dir" --condition "$condition" --expect-commit "$commit"

printf -v job '%q %q %q %q %q %q %q %q %q' \
  env "TENNIS_LAB_DINO_EXTENSION=$TENNIS_LAB_DINO_EXTENSION" \
  .venv/bin/python -B -u "$driver" probe --root "$run_dir"
printf -v suffix ' %q %q %q %q' --condition "$condition" --expect-commit "$commit"
printf -v measure '%q %q %q %q %q %q %q %q %q' \
  env "TENNIS_LAB_DINO_EXTENSION=$TENNIS_LAB_DINO_EXTENSION" \
  .venv/bin/python -B -u "$driver" run --root "$run_dir"
bash "$queue" add "$job$suffix && $measure$suffix" --resource all --provider codex \
  --session "$CODEX_THREAD_ID" --issue 1757 --name "i1757-$condition" \
  --idempotency-key "i1757/$condition/$commit/attempt-1"
```

queueが停止中なら `bash "$queue" start` で起動します。job名は `bash "$queue" list` で照合でき、stdout/stderrは `.training_queue/logs/<job-id>.log` にあります。probeは `probe/terminal.json` と `probe/recording.json`、本計測は `measurement/worker.log`、各repeatの `measurement.json` とC1 recorder JSONを保存します。各repeatの `first-inference.json` は、実モデルの初回forwardが完了した証拠です。正規終了では `measurement/terminal.json` の成否をqueueのterminalと突き合わせます。CUDAエラー・ハング・時間上限では診断を保存して停止します。

中断・cache混在の検証は冷えた保存先での速度計測と別のqueue jobで行います。既存の `run_video_batch` の設定を変えず、使い捨てのdataset copyでstage保存後にqueueから中断し、同じcopyで再開します。進捗の `cached`・`executed`・`skipped`、モデルイベントと最終markerを照合し、再開したsceneを冷えた保存先からのsceneと比較します。CUDAエラー・ハング時はそこで停止します。

## 確認と集計

```bash
report=knowledge/runs/run-i1757-video-batch-meiji/report.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B "$report" summarize \
  --root "$run_dir" --output "$run_dir/summary.json"
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B "$report" compare \
  --reference <serialのattempts/1-first> --candidate <比較条件のattempts/1-first> \
  --output <差分レポート.json>
```

集計はwall time、完了clips/hourとcamera-frame/s、モデルload/unload、実batch分布と充填率、RAM・CUDA allocator・device VRAM peak、process I/O byte数、中間配列保存とscene export・dataset marker公開を出します。wallには初回のworker importを含み、prepareとGPU probeは計測外です。速度比較には3クリップを生成し、recorderが正常終了したrepeatを使います。

比較は公開markerからsceneを読み直し、componentの採用配列・payloadとsceneの配列を比較します。[許容誤差](tolerances.json) は測定前に固定します。離散値、失敗理由、選択、ID、欠測・有効性maskは完全一致を要求します。連続値はdtypeと単位ごとに判定し、変更の最上流候補と範囲を記録します。出力差の原因と下流への影響は、差分レポートの該当node・clipから追跡します。[component overlayの操作](../../../src/tennis_scene/visualization/review/README.md) で両条件の同じclip・camera・stageを動画化し、差分フレームを添えて提示します。大きなrecorder JSON、出力配列や動画はworktreeの `outputs/` に保持します。
