# 動画一括のプレーヤー3D再構築をTensorRT FP16で計測する

[#2363](https://github.com/Motoki0705/tennis-lab/issues/2363)の評価条件は [condition.json](condition.json) に固定する。meiji_4camの `video_000` 全69クリップで横断バッチを有効にし、人物対応は各side 2人、コートの向きは各clipのplayer証拠、ボール検出と再構成は無効とする。コートには[承認済みの共通Hと線offsetによる診断](../../nodes/court_detection/000078-run-court-video-drift-lines-20261117.md)を用いる。

## バッチサイズの根拠

| モデル | B | 単体throughput | Peak VRAM増分 | 単体精度 |
| --- | ---: | ---: | ---: | --- |
| ViTPose-H | 8 crop | 142.43 crop/s | 1,699 MiB | xy差P95 0.024687 px、最大0.302900 px、mask差0 |
| HMR2 | 16 | 290.65 crop/s | 1,597 MiB | cosine P05 0.999994、相対L2 P95 0.003534 |
| DINO GridSample | 1 | 13.33画像/s | 1,627 MiB | IoU P05 0.929859、score差P95 0.017161。単体基準未達 |
| CLIP-ReID | 16 | 2,358 crop/s | 435.1 MiB | cosine P05 0.999987、最近傍ID 236/236一致 |
| GVHMR | 4 | 1,783.6系列/s (L=145)、1,607.8 (L=177) | 287.1 MiB | カメラ座標COCO-17 P95 8.78 mm (L=145)、8.99 mm (L=177) |

画像系のthroughputは計測した完全バッチの値。ViTPoseのcrop数はflip前で入力Bは16になる。ViTPose、HMR2、CLIP-ReIDは各FP16掃引のthroughput最大の候補。DINOはB=2のthroughput差が小さく、B=4では欠落が増えたのでB=1を使う。GVHMRの単体値はB=1〜4・L=145〜177のprofile。全クリップ用profileは、最大のsampled frame数に合わせてL=2〜1055とし、長さの等しい系列をまとめる。VRAM値は、各先行runでのNVMLのidleからの増分で、全体のPeak VRAMは別に測る。 [画像3モデル #2336](../../nodes/tennis_scene/000080-run-i2336-vision-batch-sweep.md)・[CLIP-ReID #2335](../../nodes/tennis_scene/000079-run-i2335-trt-batch-sweep.md)・[GVHMR #2355](../../nodes/tennis_scene/000082-run-i2355-gvhmr-pred-cam-clamp.md)

## 再現

共有 `.venv` にTensorRTのoptional依存がある環境を使う。DINOとDINOv3のsubmoduleを用意する。指定worktreeのrootから、元datasetと先行成果物の実パスを渡して共有training queueへ登録する。

```bash
bundle=knowledge/runs/run-i2363-trt-player-video
queue=.agents/skills/training-queue/scripts/training_queue.sh
printf -v command '%q ' env "PYTHONPATH=$PWD" .venv/bin/python -B "$bundle/run.py" \
    i2363-qualification-v1 --dataset "$MEIJI_DATASET" representative \
    --phase2 "$PHASE2_TENSORRT" --court "$COURT_EVALUATION_ROOT"
bash "$queue" add "$command" --name i2363-trt-qualification-v1 --issue 2363 \
    --resource all --provider codex --session "$CODEX_THREAD_ID"
```

`PHASE2_TENSORRT` は #2362 の `outputs/tensorrt`、`COURT_EVALUATION_ROOT` はrun 77・78の `outputs/tennis_scene/evaluate`。DINOはそのままのplan、ViTPose・HMR2・CLIP-ReIDは保存済みONNXと本番buildコマンドを使う。GVHMRは本番export・build・qualificationコマンドを使う。エンジンの識別と各コマンドの終了値を、worktreeの `outputs/i2363/<tag>/` に保存する。

代表clipは #2362 と同じ `video_000/clip_037`。同一の映像・pass・profileを用い、ホスト計時、同期付き計時の順に実行する。各条件は別プロセスでcomponent cacheが空の状態から開始する。OSのファイルcacheを共有する。資源記録は両条件で0.5秒間隔とし、OMP・MKL・OpenBLASのスレッド上限を2に固定する。全件評価はホスト計時とし、`--progress` で実際に通ったGVHMRの全shapeをORTで照合する。元datasetのmanifestとmediaを評価用ディレクトリへ複製し、評価結果を同ディレクトリへ公開する。

```bash
printf -v command '%q ' env "PYTHONPATH=$PWD" .venv/bin/python -B "$bundle/run.py" \
    i2363-all-v1 --dataset "$MEIJI_DATASET" all \
    --prepared outputs/i2363/i2363-qualification-v1
bash "$queue" add "$command" --name i2363-trt-all-v1 --issue 2363 \
    --resource all --provider codex --session "$CODEX_THREAD_ID"
```
