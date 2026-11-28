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

## 計測結果

RTX 5060 Ti の共有キュー all 枠で、`2dbd8c355faa2d530b8b0e17164f691e60a57d07` を計測した。6プロセスの終了値はすべて0で、各条件の3クリップが生成された。各条件で117 nodeの実行を確認した。C4は TensorRT 10.14.1.48.post1 上で5 planを再利用した。重み、manifest、export、planのSHA-256を現在のモデル宣言と突き合わせ、ロード時のplanハッシュも一致した。推論前の照合と複製は188.75秒かかった。 [キュー実行記録](queue-receipt.json)・[資産と計測の明細](metrics.json)

| 条件 | CLI wall [秒] | Recorder wall [秒] | C1 比 | 直前からの短縮 [秒] | 直前比 | Peak VRAM [GiB] | Peak RSS [GiB] | 管理モデル load / unload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 2,186.31 | 2,151.17 | 1.00× | — | — | 4.494 | 7.606 | 18 / 18 |
| C2 | 1,686.15 | 1,672.52 | 1.30× | 500.17 | 1.30× | 4.494 | 7.764 | 6 / 6 |
| C3 | 1,706.81 | 1,686.10 | 1.28× | −20.66 | 0.99× | 4.490 | 8.158 | 6 / 6 |
| C4 | 1,319.31 | 1,299.46 | 1.66× | 387.50 | 1.29× | 1.762 | 3.774 | 6 / 6 |

C1のCLI時間は指定順に858.13、719.69、608.50秒。C1→C2ではランタイム管理の6モデルがそれぞれ3回から1回のロード・解放になった。モデルのロード・解放に帰属した時間は247.03→68.21秒、資産照合を含む計測処理は271.58→87.76秒だった。CPUのAFLinkは各条件で12回ロードされた。 [プロセス別計時・モデル寿命](metrics.json)

C2→C3では、HMR2の実forwardの充填率が91.46%から99.77%に上がり、120回から110回に減った。C3ではViTPose 3,232回、CLIP-ReID 1,616回、HMR2 102回のdispatchがclipをまたいだ。各adapterの充填率は99.98%、99.92%、99.77%だった。C4でも各3,228・1,614・102回を記録した。DINOは上限1、GVHMRはclip内の等長系列なので、いずれもclipをまたがなかった。今回のC3はC2より1.23%長かった。 [バッチ・経過時間の基表](comparison.csv)

C3→C4では、CLI wallが22.70%、Peak VRAMが60.76%、Peak RSSが53.73%減った。video pass内のGPU稼働区間率と時間平均使用率はC1で61.21%・47.50%、C2で74.66%・58.52%、C3で74.91%・58.18%、C4で57.66%・38.72%だった。 [資源指標と段階・モデル別時間](metrics.json)・[論文用PDF](comparison.pdf)・[PNG](comparison.png)

段階別の時間は recorder の最深spanに帰属する時間配分。モデル別は各モデルに属するホストspanの区間和で、モデル間は重なり得る。CLIP-ReIDの区間和には `clip` と `appearance` の両ラベルを含める。GPU利用率はvideo pass中の観測区間で時間加重し、Peak VRAMは0.5秒間隔のNVMLデバイス使用量の最大、Peak RSSはプロセス寿命内の最大。C1の資源ピークには3プロセス中の最大値を使う。計測は固定順で各1回。プロセス起動、重み・入力の照合、OSのファイルcache状態もCLI時間に影響する。 [recorderの計時定義](../../../src/utils/inference_recording.md)

## 出力整合

12シーンの状態はすべて `ok`。表のフレーム被覆は4選手×883元フレームの選手フレーム数で、3D関節は17関節中1点以上の有効フレームを数える。全条件でC1の12選手IDに対応し、数値IDも一致した。対応は同じ視点・元フレームで信頼度0.30以上の2D関節を使い、共通観測のある元フレームが30以上、座標距離の中央値が画像対角長の1%以内の組から1対1で決めた。L2差は対応済みの共通有効関節をコート座標で比較した。 [clip別結果と対応・マスクの内訳](parity.json)・[比較CSV](parity.csv)

| 条件 | 有効3D関節フレーム | 有効SMPLフレーム | 3D関節点マスク一致率 | SMPLマスク一致率 | C1比の共通有効関節 | 3D L2 中央値 / P95 / 最大 [m] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 3,508 | 3,496 | 100% | 100% | 59,540 | 0 / 0 / 0 |
| C2 | 3,508 | 3,496 | 100% | 100% | 59,540 | 0 / 0 / 0 |
| C3 | 3,508 | 3,496 | 100% | 100% | 59,540 | 0 / 0 / 0.0000793 |
| C4 | 3,508 | 3,495 | 99.970% | 99.915% | 59,533 | 0.00181 / 0.00906 / 0.69784 |

C4の3D関節フレームマスクはC1と完全一致した。関節点マスクは60,044点中18点で変わり、C1のみ有効7点、C4のみ有効11点。SMPLは3,532選手フレームのうちC1のみ有効2、C4のみ有効1だった。最大3D差は `clip_000` で0.698 m、残る2クリップの最大差も0.436–0.442 mだった。差の基準はC1の推定出力で、評価対象はこの3クリップの関節座標と被覆。 [対象別の比較](parity.json)

## 再現

指定worktreeのrootで、DINOとDINOv3のsubmodule、CUDAとTensorRT対応の共有 `.venv`、DINO CUDA拡張を用意する。`MEIJI_DATASET` は `data/tennis_multiview/processed/meiji_4cam/dataset`、`I2363_OUTPUTS` は #2363 の保存元で `i2363/` と `tensorrt/` を含む。`DINO_EXTENSION` には導入済みの拡張の絶対パスを指定する。

```bash
bundle=knowledge/runs/run-i2696-exec-system-ablation
queue=.agents/skills/training-queue/scripts/training_queue.sh
tag=ablation-replay

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

保存済みの `ablation-v1` からCPUで集計と図を再生成できる。図の環境は Matplotlib 3.11.1、Noto Sans CJK JP、DejaVu Sans。再集計は保存sceneのハッシュと系譜を検証する。

```bash
bundle=knowledge/runs/run-i2696-exec-system-ablation
destination=outputs/i2696/verification-b
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -B "$bundle/report.py" outputs/i2696/ablation-v1 "$destination"
.venv/bin/python -B "$bundle/make_figures.py" "$destination/metrics.json" "$destination"
for file in metrics.json comparison.csv parity.json parity.csv comparison.png comparison.pdf; do
    cmp "$bundle/$file" "$destination/$file"
done
```

再集計ジョブ `20261127T160158Z_1b9729a49bee_i2696-report-repro` で、JSON・CSV・PNG・PDFの6ファイルすべてがバイト一致した。再集計の実測は71.68秒。

```bash
PYTHONPATH="$PWD/$bundle:$PWD" .venv/bin/python -B -m pytest -n 0 "$bundle"
.venv/bin/ruff check "$bundle"
```
