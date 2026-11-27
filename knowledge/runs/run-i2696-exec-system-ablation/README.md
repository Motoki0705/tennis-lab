# モデル共有・横断バッチ・TensorRT FP16 の統制比較

[#2696](https://github.com/Motoki0705/tennis-lab/issues/2696) で、固定3クリップのプレーヤー3D再構成を4条件で計測する。入力は `meiji_4cam/video_000` の `clip_000`、`clip_037`、`clip_027` の順で、各カメラに356、292、235元フレームがある。12映像とdataset・clip manifestのSHA-256、および共有コートとdriftの内容digestを [condition.json](condition.json) に固定した。これらは [#2363 の全件計測](../run-i2363-trt-player-video/README.md) に照合する。

| 条件 | CLIプロセス | clip間モデル共有 | 横断バッチ | 5モデルの推論 |
| --- | ---: | --- | --- | --- |
| C1 `serial_pt16` | 3 | 無効 | 無効 | PyTorch FP16 |
| C2 `shared_pt16` | 1 | 有効 | 無効 | PyTorch FP16 |
| C3 `batched_pt16` | 1 | 有効 | 有効 | PyTorch FP16 |
| C4 `batched_trt16` | 1 | 有効 | 有効 | TensorRT FP16 |

全条件で標準recipeの DINO Swin-L、ViTPose-H、CLIP-ReID ViT-B/16、HMR2、GVHMR を使い、ダブルス各side 2人、clip単位のplayer証拠、ボール検出・再構成offとする。イベント設定は現在のscene契約に従って `rule` を明示する。ViTPoseは8 crop（flip前）、HMR2とCLIP-ReIDは16 crop、DINOは1画像。GVHMR上限4は全条件共通で、同一clipの等長系列に適用する。

C1〜C3のDINOとViTPoseは本体のPyTorch FP16経路を使う。CLIP-ReID・HMR2・GVHMRは [pipeline_cli.py](pipeline_cli.py) がネットワークのforwardにCUDA autocast FP16を適用する。FP32パラメータ、DINOのCUDA deformable attention、GVHMRの復号・補正・IKの精度は対応する標準処理に従う。実験用のモデル宣言にautocast方式と実装SHA-256を記録する。CLI・モデル所有・公開は本体のvideo batch経路を通る。

C4は [prepare_engines.py](prepare_engines.py) で #2363 のplanを照合する。現在のモデル実装と資産を束縛し、元のmanifest・export・planとTensorRT/GPU環境を確認する。互換性を満たすplanは複製して使う。再ビルド時は #2363 と同じprofileを指定し、秒数を推論計時から除く。

## 再現

指定worktreeのrootで、DINOとDINOv3のsubmodule、CUDAとTensorRT対応の共有 `.venv`、DINO CUDA拡張を用意する。`MEIJI_DATASET` は `data/tennis_multiview/processed/meiji_4cam/dataset`、`I2363_OUTPUTS` は #2363 の保存元で `i2363/` と `tensorrt/` を含む。`DINO_EXTENSION` には導入済みの拡張の絶対パスを指定する。

```bash
bundle=knowledge/runs/run-i2696-exec-system-ablation
queue=.agents/skills/training-queue/scripts/training_queue.sh
tag=ablation-v1

CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD" .venv/bin/python -B "$bundle/run.py" "$tag" \
    --check-inputs --dataset "$MEIJI_DATASET" --prior "$I2363_OUTPUTS" --dino-extension "$DINO_EXTENSION"

printf -v command '%q ' env CUDA_VISIBLE_DEVICES=0 "PYTHONPATH=$PWD" \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    .venv/bin/python -B -u "$bundle/run.py" "$tag" --expect-head "$(git rev-parse HEAD)" \
    --dataset "$MEIJI_DATASET" --prior "$I2363_OUTPUTS" --dino-extension "$DINO_EXTENSION"
TRAINING_QUEUE_DIR=/home/kamimura/projects/tennis-lab/.training_queue \
    bash "$queue" add "$command" --name "i2696-$tag" --issue 2696 --resource all \
    --provider codex --session "$CODEX_THREAD_ID"
```

各プロセスの入力ディレクトリはmanifestとmediaから作る。`outputs/i2696/<tag>/` に条件ごとの独立store、`progress.json`、`recording.json`、ログ、コマンドと終了値の `outcome.json` を保存する。同一HEADで順に一度ずつ実行し、OSのファイルcacheは計測環境に従う。recorderのCUDA計時同期はoff、NVMLとRSSの採取間隔は0.5秒、OMP・MKL・OpenBLASは2スレッドとする。
