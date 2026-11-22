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

初回は5モデルのエンジン準備が完了した。GVHMRは24 shapeでONNXの全出力がビット一致し、B=1/4 × L=2/177/1055の6 shapeでTensorRT出力の有限性とFP16層精度を確認した。代表clipはHydraが絶対パスの `progress_directory` を拒否し、推論前に停止した。コマンド記録がGVHMRの計測JSONを上書きしたため、終了値・生の層情報・ログから初回の証拠を保存した。 [初回の終了状態とエンジン識別](qualification-v1.json)

再試行ではHydraへの出力子パスを相対にし、コマンド記録を `*.command.json` に保存する。初回のエンジンをmanifestとSHA-256で再照合し、そのplanでGVHMR単体検証と代表clipを実行する。

```bash
printf -v command '%q ' env "PYTHONPATH=$PWD" .venv/bin/python -B "$bundle/run.py" \
    i2363-qualification-v2 --dataset "$MEIJI_DATASET" representative \
    --prepared outputs/i2363/i2363-qualification-v1 --court "$COURT_EVALUATION_ROOT"
bash "$queue" add "$command" --name i2363-trt-qualification-v2 --issue 2363 \
    --resource all --provider codex --session "$CODEX_THREAD_ID"
```

代表clipは #2362 と同じ `video_000/clip_037`。同一の映像・pass・profileを用い、ホスト計時、同期付き計時の順に実行する。各条件は別プロセスでcomponent cacheが空の状態から開始する。OSのファイルcacheを共有する。資源記録は両条件で0.5秒間隔とし、OMP・MKL・OpenBLASのスレッド上限を2に固定する。全件評価はホスト計時とし、`--progress` で実際に通ったGVHMRの全shapeをORTで照合する。元datasetのmanifestとmediaを評価用ディレクトリへ複製し、評価結果を同ディレクトリへ公開する。

## 代表クリップの結果

`clip_037` は両条件で `generated`。38 nodeを実行し、各選手で3D関節の有効frameは290/292、root・SMPLは289/292だった。38件のcomponent参照と318個のcomponent/scene exportファイルを照合し、artifact ID、execution key、ファイルのSHA-256が一致した。

| 計時用CUDA同期 | CLI wall (秒) | 記録区間 (秒) | CUDA同期span (秒) | Peak VRAM (GiB) | Peak RSS (GiB) | 資源採取中央値 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| なし | 739.79 | 722.96 | 0.176 | 1.76 | 3.67 | 5.79 |
| あり | 899.28 | 878.64 | 141.009 | 1.76 | 3.63 | 8.87 |

同期ありのCLIは159.49秒（21.56%）長かった。同一clipで順に1回ずつ測った値で、OSのファイルcacheや他プロセスの負荷は固定していない。Peak VRAMはNVMLのdevice使用量、Peak RSSはプロセスの最大RSS。WSL全体の空きメモリは計測中に最小22.74 GiBだった。両条件で5モデルのforwardを確認した。この条件で全件評価を行う。

GVHMRは24 shapeでONNX書換え前後の出力がビット一致した。B=1/4 × L=2/177/1055の6 shapeで出力が有限であり、各shapeで110演算、加算mask付きfused MHA 12層、K=2のFP16積和を照合した。代表クリップで実際に通ったshapeはB=4、L=145。 [代表検証の指標と層精度](qualification-v2.json)

ホスト計時の代表clipだけで86,924 spanが記録された。全クリップ実行では、時間窓ごとに完了spanをディスクへ退避し、最終 `recording.json` に全spanと集計を書き出す。 [記録形式](../../../src/utils/inference_recording.md)

## 全クリップ実行

```bash
printf -v command '%q ' env "PYTHONPATH=$PWD" .venv/bin/python -B "$bundle/run.py" \
    i2363-all-v1 --dataset "$MEIJI_DATASET" all \
    --prepared outputs/i2363/i2363-qualification-v2
bash "$queue" add "$command" --name i2363-trt-all-v1 --issue 2363 \
    --resource all --provider codex --session "$CODEX_THREAD_ID"
```
