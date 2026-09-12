# Meiji 3カメラ・ConvNeXtUNet 学習レシピ

`train_meiji_3cam` は `MultiviewBallDataModule` を追加して動画と
`video_ball_annotation.v2` を取り込む。共通の `data/dataset.py` と
`models/conv_next_unet.py` は変更しない。

この学習で得るのは、**1カメラの画像系列から2Dボール位置を推定するモデル**。
3カメラ分の映像を学習データとして使用し、1つの窓には同じカメラの連続フレームを入れる。
同期した3カメラを1サンプルへ結合したり、3D座標を教師にしたりする構成ではない。

実装の準備・CPUでの入出力検証に加え、Colabでの前処理・L4のバッチ較正・20 epochsの本学習完了を確認した。
今回の測定値、validation結果、checkpoint SHA-256は[実行記録](../../../knowledge/nodes/run-meiji-convnext-l4-20260911.md)を参照する。
このレシピは画像準備、L4バッチ較正、本学習の順で処理を実行する。

## 入力ファイルを確認する

Drive上の入力は次の1ファイル。

```text
マイドライブ/tennis_lab/data/meiji_3cam.tar
```

Colabからは `/content/drive/MyDrive/tennis_lab/data/meiji_3cam.tar` として読む。
tar内部には、`dataset.json` と `clips/` が並ぶデータセットを1つ含める。
`dataset/`、`meiji_3cam/dataset/`、元の長いディレクトリ階層に対応し、
その上位ディレクトリ名を手作業で揃える必要はない。

```text
dataset/
  dataset.json
  clips/
    meiji_3cam/
      clip_000/
        clip.json
        media/
          cam0.mp4
          cam1.mp4
          cam2.mp4
        outsource/
          cam0_annotations.json
          cam1_annotations.json
          cam2_annotations.json
      ...
    meiji_3cam_2/
      ...
```

`media/` の元動画とJSONを使用する。アノテーション描画済み動画は入力にしない。
学習前に画像へ変換してDriveへ再アップロードする作業は不要。
スクリプトがVMローカルへ展開し、動画をCPUでデコードして画像キャッシュを作る。

ローカルで確認した元データは約1.7 GiB。全62,967フレームを288×512のBMPへ展開すると約26 GiBになる。
VMでは依存ライブラリに加え、入力tar・その一時コピー・展開済み動画・画像キャッシュの領域を使う。
BMP容量は1フレーム442,422 bytesからの計算値。Drive上のtar容量とは別に確認する。
入力動画と展開済みBMPはVMローカルディスクから読む。Driveへは学習成果物を保存する。

## 実行方法A：ローカル端末からColabへ送る

### 1. 実装を含むworktreeへ移動する

```bash
cd /home/kamimura/projects/tennis-lab-worktrees/meiji-ball-colab
```

今回の実装はこのworktreeにある。メインworktreeからの実行や、GitHubのmainをcloneするだけでは
未コミットの追加ファイルは反映されない。ここでは `--source snapshot` を明示して転送する。

Colab CLIの初回インストールは[共通READMEの前提条件](../README.md#前提条件)に従う。
Colab認証とDrive認証は[Drive認証方式](../README.md#drive認証方式)に従う。
この学習jobはDriveへの直接保存を行うため、`--drive-mode mount` を使用する。

### 2. 起動せずに設定を確認する

```bash
bash scripts/colab/run.sh run ball_meiji_3cam_l4 \
  --gpu L4 --drive-mode mount --source snapshot --dry-run \
  > /tmp/meiji-colab-plan.json
```

このdry-runではColabセッションを作らず、GPUも使用しない。
出力JSONの `request.runtime.gpu` が `L4`、入力sourceが `data/meiji_3cam.tar`、
jobが `ball_meiji_3cam_l4` であることを確認する。
Driveの実在ファイルやL4の割り当て可否は、本実行時に確認される。

Hydra設定の合成結果だけを読む場合は次を実行する。こちらも学習は開始しない。

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.train_meiji_l4 \
  --cfg job --resolve > /tmp/meiji-training-config.yaml
```

この時点の `data.batch_size: 1` は較正前の初期値。
実際に採用した値は本実行後の `calibration/batch_size.json` と `config.yaml` で確認する。

### 3. 標準レシピを実行する

```bash
bash scripts/colab/run.sh run ball_meiji_3cam_l4 \
  --gpu L4 --drive-mode mount --source snapshot \
  --download-to ./colab-artifacts
```

Colab・Drive認証が表示された場合は端末の案内に従う。
実行時に表示されるrun IDを控える。以後の進捗確認と保存先の特定に使う。
このコマンドは処理終了まで待つので、監視には別の端末を使用できる。
成功時は成果物を取得してセッションを停止する。

失敗したVMを調査のため残したい場合は、本実行のコマンドへ `--keep-on-failure` を追加する。
既定のjob timeoutは86,400秒。これはworkflow側の待ち時間上限であり、所要時間の予測ではない。

### 4. 初回は標準設定を基準にする

比較実験でepoch数と学習率を明示変更する場合の構文は次のとおり。

```bash
bash scripts/colab/run.sh run ball_meiji_3cam_l4 \
  --gpu L4 --drive-mode mount --source snapshot \
  --download-to ./colab-artifacts -- \
  training.trainer.max_epochs=30 \
  training.learning_rate=5.0e-5
```

これは20 epochs・lr 1e-4の標準レシピとは別の実験になる。
この専用jobでは `model`・`data` 以下、データや保存先のroot、GPU数をworkflowが管理する。
`data.batch_size=16` のような手動指定は使用しない。
L4較正launcherは `run.dry_run=true`、`run.fast_dev_run=true`、`run.resume`、
複数GPU、勾配蓄積数の変更を受け付けない。

## 実行方法B：Colabノートブックから実行する

方法Aを使う場合、この節のコード転送・Driveマウントはworkflowが行うため不要。
ブラウザ上のノートブックだけで操作する場合は、次の手順で今回の実装を含むソースを用意する。

### 1. ローカルでソースのsnapshotを作る

ローカルの実装worktreeで次を実行する。リポジトリ既存のsnapshot処理を使用するため、
未コミットの変更と追加ファイルも含まれる。data・venv・画像キャッシュは同梱しない。

```bash
cd /home/kamimura/projects/tennis-lab-worktrees/meiji-ball-colab
.venv/bin/python - <<'PY'
from pathlib import Path
from scripts.colab.workflow.snapshot import create_snapshot

root = Path.cwd()
snapshot = create_snapshot(
    root, root / 'outputs/colab_source/meiji_ball_source.tar.gz'
)
print(snapshot.archive_path)
print('sha256:', snapshot.archive_sha256)
PY
```

生成された `outputs/colab_source/meiji_ball_source.tar.gz` を
`マイドライブ/tennis_lab/code/meiji_ball_source.tar.gz` にアップロードする。
データ用の `data/meiji_3cam.tar` はそのまま維持する。

### 2. L4ランタイムとDriveを用意する

ColabでGPUをL4に指定し、Driveを別セルでマウントする。

```python
from google.colab import drive
drive.mount('/content/drive')
```

次の確認セルでGPU名が `NVIDIA L4` であり、2つのtarが見えることを確認する。

```bash
!nvidia-smi --query-gpu=name --format=csv,noheader
!ls -lh /content/drive/MyDrive/tennis_lab/data/meiji_3cam.tar
!ls -lh /content/drive/MyDrive/tennis_lab/code/meiji_ball_source.tar.gz
```

### 3. 初回だけソースを展開する

新しいランタイムで実行する。既存の `/content/tennis-lab` がある場合、このセルは停止する。
既に今回のコードを展開済みなら、次の学習セルへ進む。

```bash
%%bash
set -euo pipefail
test ! -e /content/tennis-lab
mkdir -p /content/tennis-lab
tar -xzf /content/drive/MyDrive/tennis_lab/code/meiji_ball_source.tar.gz \
  --no-same-owner -C /content/tennis-lab
```

### 4. 学習シェルを実行する

```bash
!bash /content/tennis-lab/scripts/colab/train/ball_meiji_3cam_l4.sh
```

依存環境の `uv sync --locked`、データtarの展開、画像生成、バッチ較正、本学習をシェルが実行する。
Colab標準Pythonへ個別にtorchやLightningを追加する必要はない。
モデルとDataModuleの実行にはリポジトリの `.venv/bin/python` を使用する。

ノートブックから同じシェルを再実行すると、新しいUTC時刻の保存先で較正・本学習を開始する。
前回と同一のアーカイブ・動画に対応する完全な画像キャッシュは再利用する。
checkpointを自動的に探して途中再開する動作はない。

## 教師と分割

| アノテーション状態 | 使用方法 |
|---|---|
| `observed` | 実画像で確認した座標を正例にする |
| `interpolated` | 補間された座標を正例にする |
| `occlusion_estimated` | 遮蔽中の推定座標を正例にする |
| `unresolved` | このフレームを含む8フレーム窓を除外する |

座標がある3種類は同じ重みで使用する。推定ラベルに対する個別の重み係数は設けていない。
共通Datasetに渡すvisibilityは「教師座標がある」ことを意味し、物理的な可視性とは区別する。
`unresolved` は位置不明であり、「ボールが存在しない」として負例にはしない。
`break_before` が窓の2フレーム目以降にある場合も、その窓を除外する。
欠測フレームを削って前後をつなぐ処理は行わない。

例えばフレーム0～15のうち9だけが `unresolved` の場合、stride 4の候補は0～7、4～11、8～15。
このうち0～7だけを採用する。窓は1カメラ・1クリップの範囲内で構築する。
座標は元の1920×1080画素座標を使用する。cam0のletterboxも元動画に含まれるため追加変換しない。

分割の正本は [train.txt](../../../src/tasks/ball_detection/configs/data/splits/meiji_3cam/train.txt)、
[val.txt](../../../src/tasks/ball_detection/configs/data/splits/meiji_3cam/val.txt)、
[test.txt](../../../src/tasks/ball_detection/configs/data/splits/meiji_3cam/test.txt) の3ファイル。
各撮影内のソート済みclip IDをPython `random.Random(42)` でシャッフルし、
撮影ごとに `round(clip数 × 0.1)` 個ずつvalidation/testへ、残りをtrainへ割り当てた。
撮影の処理順は `meiji_3cam`、`meiji_3cam_2`。同じクリップの3カメラをまとめる。
起動時に重複と欠落を検査し、データセットの全clip IDとの完全な一致を要求する。

| 分割 | クリップ | カメラ動画 | 8フレーム窓（stride 4） |
|---|---:|---:|---:|
| train | 27 | 81 | 10,101 |
| validation | 3 | 9 | 1,270 |
| test | 3 | 9 | 788 |

これは約82/9/9%のクリップ分割。同じ撮影の別クリップが複数の分割に含まれるため、
未知の撮影環境への汎化性能を測る分割ではない。件数は現在のローカルデータで確認した値で、
実行時の `dataset_summary.json` にDriveアーカイブの実際の件数とclip IDを保存する。
学習中はvalidationで評価し、testは保留する（`run.test_after_fit=false`）。
validationを使ってcheckpointと学習条件を決めた後、固定した条件でtestを評価する。
通常の学習中のmetricsは窓に含まれるフレームを集計するので、重複窓に含まれる同じフレームは
複数回評価される。動画全体をフレームごとに1回だけ評価した数値とは区別する。

## データがモデルへ届くまで

処理の責務は次のように分けている。

1. `prepare_meiji_archive.py` がtarを検査し、データセットを所定のディレクトリへ展開する。
2. `MultiviewBallDataModule` がmanifest、split、99本のアノテーションを読み、欠落・重複・座標・フレーム番号を検査する。
3. 初回の画像生成では動画のSHA-256をアノテーションと照合し、解像度・フレーム数をデコード結果でも確認する。
4. **学習前に全フレームを288×512へリサイズしてBMPへ展開する**。8本の動画を並列処理し、カメラごとに生成が完了してから公開する。
5. DataModuleが教師の揃った窓を作り、既存の `BallDetectionDataset` へ渡す。
6. DatasetがRGB画像・Gaussian heatmap・座標・visibilityを返し、model I/O adapterがRGBをMDD特徴へ変換する。
7. ConvNeXtUNetの出力を教師heatmapの解像度へ合わせ、損失とmetricsを計算する。

| 境界 | テンソル形状（Bはバッチ数） |
|---|---|
| Datasetからの画像 | `(B, 8, 3, 288, 512)`、RGB float32、値域 `[0,1]` |
| モデル入力 | `(B, 2, 8, 288, 512)`、MDD |
| 損失計算時の予測・教師heatmap | `(B, 8, 288, 512)` |
| 教師座標 | `(B, 8, 1, 2)`、元画像のx/y画素座標 |
| 教師visibility | `(B, 8, 1)` |

キャッシュは `.cache/ball_detection/meiji_3cam/288x512_bmp/<recording>__<clip>/<camera>/`。
新しいランタイムでは再生成される。同じランタイムで再利用するときはキャッシュの動画情報・
生成設定・ファイル一覧を照合し、不完全なキャッシュを黙って使用しない。

### BMPへ事前展開する理由と読み込みの高速化

以前のJPEGキャッシュも学習前に全動画を展開していたが、学習時に窓を読むたびにJPEGの解凍が必要だった。
今回はストレージに余裕がある条件に合わせ、**リサイズ済みの非圧縮BMP**を全フレーム分保存する。
元動画の1920×1080のまま保存すると約365 GiB必要だが、学習解像度なら約26 GiBで済む。
BMPはデコード・リサイズ後の画素をそのまま保存するため、JPEG再圧縮による画素の変化も加わらない。
`BallDetectionDataset` が元々対応する `cv2.imread` で読めるので、Datasetの変更は必要ない。

ローカルCPUで実動画の連続128フレームを使い、7回の `imread → resize → RGB変換` の中央値を比較した。

| 形式 | 128フレームの読み込み | 1フレームの容量 |
|---|---:|---:|
| JPEG品質95 | 0.1194秒 | 約63.7 kB |
| BMP | 0.0154秒 | 約442.4 kB |

この測定ではBMPが約7.7倍速い。ただしOSのファイルキャッシュが効いたローカルCPUでの比較であり、
Colab上のディスク速度やモデル計算を含む学習全体が7.7倍速くなるという意味ではない。
初回の展開に追加の書き込み時間と容量を使い、その後20 epochsの繰り返し読み込みを軽くする選択である。

- `prepare_workers=8`：8動画を並列に展開し、各FFmpegデコーダは1スレッドに制限する。
- `num_workers=8`：8プロセスが画像、augmentation、heatmapを準備する。
- `persistent_workers=true`：epochをまたいでワーカーを維持する。CPUテストでnum_workers=0にした場合のみ無効になる。
- `prefetch_factor=1`：各ワーカーの先読みを1batchにして、8ワーカーによるホストメモリ使用を抑える。
- `pin_memory=true`：GPUへの転送に使うページ固定メモリを有効にする。
- 各読み込みワーカー内ではOpenCVを1スレッドにし、CPUスレッド数の過剰な増加を防ぐ。

前処理はモデル生成・バッチ較正より先に完了する。空き容量が必要量＋1 GiBを下回る場合は停止する。
前処理のフレーム数、保存形式、並列数、所要時間は `preprocessing.json` に保存する。
旧JPEGキャッシュと新BMPキャッシュはパスを分け、意図せず混在させない。
全フレームを展開するため `unresolved` の画像も保存されるが、そのフレームを含む窓は学習には使わない。

実装の根拠：
[PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html)、
[OpenCV画像読み込み・保存](https://docs.opencv.org/4.13.0/d4/da8/group__imgcodecs.html)。

## モデルと学習条件

設定の正本は [train_meiji_3cam.yaml](../../../src/tasks/ball_detection/configs/train_meiji_3cam.yaml)
とそのdefaults。以下は現在の標準設定の説明で、最終的な実行条件は保存された `config.yaml` を確認する。

| 項目 | 値 |
|---|---|
| モデル | ConvNeXtUNet、dims `[64,128,256,512]`、depth 2 |
| 初期化 | ランダム初期化（checkpoint不要） |
| 入力 | 8連続フレーム、288×512、既存MDD入力（2ch） |
| MDD設定 | `mdd_a=0.2`、`mdd_b=0.15` |
| 教師heatmap | 288×512、sigma_ratio 0.012 |
| サンプリング | stride 4、trainはshuffle、末尾の不完全batchはdrop |
| DataLoader | num_workers 8、pin_memory true、persistent_workers true、prefetch_factor 1 |
| 前処理 | prepare_workers 8、全動画を学習前に288×512 BMPへ展開 |
| GPU / precision | NVIDIA L4 ×1 / bf16-mixed |
| physical batch | L4で較正した最大の2の累乗 |
| 勾配蓄積 | 4（通常のeffective batch = physical batch × 4） |
| optimizer | AdamW、lr 1e-4、weight_decay 0.1 |
| AdamW betas / gradient clip | `(0.9, 0.999)` / 1.0 |
| schedule | warmup 200 updates、開始lrは基本lrの0.01倍、その後cosineでmin_lr 1e-6へ |
| epochs / seed | 20 / 42 |
| loss | Focal BCE、gamma 2.0、GAN無効 |
| compile | 有効、inductor / default / dynamic=false |
| 数値設定 | TF32許可、matmul_precision high、deterministic warn、benchmark false |
| validation / early stopping | 毎epoch / 無効 |
| ログ / 可視化 | 50 stepsごと / epoch index 0、5、10、15にvalidationの1batchから先頭サンプルを描画 |
| checkpoint | val/loss最小の上位2個とlast |

バッチサイズに応じた学習率の自動スケーリングは行わない。
バッチ数が大きくなると、同じ20 epochsでもoptimizerの更新回数は少なくなる。
`deterministic: warn` とseed固定は再現性を高める設定であり、全環境のbit単位一致を保証する設定ではない。

### Augmentation

[light.yaml](../../../src/tasks/ball_detection/configs/data/augmentation/light.yaml) を使用する。
Meiji用data configでImageNet正規化だけを無効化し、model I/OのRGB `[0,1]` 契約を維持する。

| 変換 | 設定 |
|---|---|
| 左右反転 | 確率0.5 |
| affine | 確率0.2、回転±5°、平行移動±5%、x shear±1°、y shear±0.5° |
| 拡大・crop | 確率0.2、scale 1.0～1.5 |
| 明るさ / contrast / gamma jitter | 0.05 / 0.05 / 0.04 |
| Gaussian noise | std 0.006 |
| camera rotation / blur / ball-area zero mask | 無効 |

幾何変換では座標も同時に更新する。validation/testへランダムなaugmentationは適用しない。

### 損失と評価指標

損失はheatmapの各画素に対するFocal BCE。実装は
`mean((1 - p_t)^2 × BCEWithLogits(logits, target))`、
`p_t = sigmoid(logits) × target + (1 - sigmoid(logits)) × (1 - target)`。
Gaussian heatmapのsoft targetを使用する。

| 指標 | 読み方 |
|---|---|
| `train/loss` | 学習データへの適合の進み方 |
| `val/loss` | checkpointを選ぶ指標。小さいepochを採用する |
| `val/precision` | 検出した点のうち正しく対応付いた割合 |
| `val/recall` | 教師ボールのうち検出できた割合 |
| `val/f1` | precisionとrecallの調和平均 |
| `val/mean_distance_px` | 正しく対応付いた検出だけの平均距離 |

予測heatmapのpeak thresholdは0.5、NMS kernelは9、1フレーム最大8候補、subpixel refinementは有効。
対応付けは元画像座標で行い、**距離4.0 px未満**を正解にする。288×512のheatmap上の4 pxではない。
`mean_distance_px` は未検出を含む全フレーム平均ではなく、対応が0件のときは実装上0になる。
距離だけで良否を決めず、recall/F1と可視化を併せて確認する。

## L4の最大バッチサイズを測定する

バッチサイズは1、2、4…を独立プロセスで試し、最初のCUDA OOMの直前の値を採用する。
各試行で本番と同じモデル・データ・精度・compile・勾配蓄積4の設定により、
**12回の学習batch＝3回のoptimizer更新**（逆伝播とAdamWの状態確保を含む）、
2回の検証batchと可視化を実行する。蓄積途中で既存の勾配と次の計算が重なる状態も測定する。
前の試行のGPUメモリや重みは引き継がず、本学習は同じseedで初期化し直す。
親プロセスではCUDAコンテキストを作らず、較正のための余分なGPU予約を避ける。

`calibration/batch_size.json` に採用値・次の2倍がOOMになった結果・各試行のpeak memoryを保存する。
`config.yaml` にも採用バッチサイズが記録される。CPU上で最大値を推定して固定しない。
バッチ1が入らない場合、CUDA OOM以外の失敗、較正後の本学習の失敗はそのままエラーとする。
較正は初期の実batchでの実測であり、その後のGPU利用状況の変化まで保証するものではない。

1サンプルは1カメラの8フレーム窓なので、batch 16なら同時に扱うのは16窓・128フレーム。
勾配蓄積は4で固定する。batch 16なら通常のoptimizer更新1回で64窓・512フレームを使う。
例えばbatch 16が成功し32がCUDA OOMなら、物理batchは16、実効batchは64を採用する。
これは説明用の例であり、今回のL4で16が最大と測定済みという意味ではない。

較正中の目印は以下のログ。

```text
[L4 calibration] trial batch_size=1; log=.../calibration/batch_1.log
[L4 calibration] trial batch_size=2; log=.../calibration/batch_2.log
...
[L4 calibration] selected batch_size=...
```

各試行は独立したログへ書き込むため、メインの表示がしばらく変わらないことがある。
`compile` を含む初回試行の所要時間は未測定。現在の `batch_<B>.log` を確認する。
較正は本番の総更新数に基づくschedulerを保ったまま、実行するbatch数だけを制限する。
較正で学習した重みやAdamW状態は本学習へ持ち込まない。

`batch_size.json` には採用バッチ数 `batch_size`、その2倍の `next_power_failed`、
`accumulate_grad_batches=4`、`effective_batch_size=B×4`、各試行の `fits` / `cuda_oom` を記録する。成功試行のpeak memoryはPyTorch allocatorの
allocated/reserved bytesであり、GPU全体の全プロセス使用量ではない。

### 更新回数と時間の見積もり

学習窓数をN、採用physical batchをBとすると、1 epochのtrain batch数は `floor(N/B)`、
optimizer更新数は `ceil(floor(N/B)/4)`、20 epochsの総更新回数はその20倍。
epoch末に蓄積が4回に満たなくても、Lightningが残りの勾配で更新するため、その最終更新だけは
B×4窓より少なくなる。validationは勾配蓄積せず、末尾の不完全batchも評価する。
現在のN=10,101で計算した例は次のとおり。

| 仮のB（実測値ではない） | train更新/epoch | 20 epochsの総更新 | validation batches/epoch |
|---:|---:|---:|---:|
| 8 | 316 | 6,320 | 159 |
| 16 | 158 | 3,160 | 80 |
| 32 | 79 | 1,580 | 40 |
| 64 | 40 | 800 | 20 |

学習率warmupの200 stepsはoptimizer更新200回を意味する。
所要時間は「環境構築＋データ展開＋較正＋本学習＋成果物保存」に分けて考える。
本学習の2 epoch目以降の経過時間を見て残り時間を見積もり、初回compileを含むepochだけで
20 epochsの時間を外挿しない。

## 進捗を確認する

方法Aでは、ローカルの別端末で実装worktreeへ移動し、開始時に表示されたrun IDを指定する。
下の `実際のrun-id` は置き換える。

```bash
cd /home/kamimura/projects/tennis-lab-worktrees/meiji-ball-colab
MEIJI_RUN_ID='実際のrun-id'
bash scripts/colab/run.sh progress "${MEIJI_RUN_ID}" --watch
```

直近ログとworkflowの状態を確認する場合：

```bash
bash scripts/colab/run.sh logs "${MEIJI_RUN_ID}" --tail 80
bash scripts/colab/run.sh status "${MEIJI_RUN_ID}"
```

これらのコマンドのライフサイクル・停止後の参照方法は
[共通READMEの学習出力と進捗確認](../README.md#学習出力と進捗確認)を参照する。
方法Bでは学習セルの出力と、次節のDrive上のログを読む。

| 表示・ファイル | 現在の段階 |
|---|---|
| `uv sync` の出力 | Python依存環境の構築 |
| datasetへの展開先表示 | 入力tarの展開完了 |
| splitごとのJSON summary | アノテーション検査と窓数の集計完了 |
| `[multiview] preprocess ...` / `prepared N/99 ...` | 全動画の照合・BMP事前展開。完了まで学習を開始しない |
| `[L4 calibration] trial batch_size=...` | バッチ較正中。詳細はその試行のlog |
| `[L4 calibration] selected batch_size=...` | 最大バッチを決定し、本学習へ移行 |
| 本学習のepoch / global_step | 較正後に初期化し直したモデルの学習進捗 |

`calibration/batch_*/` のTensorBoardログと、本学習の `logs/` は別に読む。
較正の3更新や1 epoch終了の表示は、本学習の完了を意味しない。
workflowの本学習用進捗JSONは、較正中のoptimizer更新までは通知しない。

## 保存先とcheckpointの選択

CLI経由では既存workflowのDrive保存先を使う。

```text
MyDrive/tennis_lab/colab-live/<run-id>/outputs/colab/ball_meiji_3cam_l4/
  dataset_summary.json
  preprocessing.json
  calibration/batch_size.json
  calibration/batch_*.json
  calibration/batch_*.log
  calibration/probe_config.yaml
  config.yaml
  logs/version_*/
    events.out.tfevents.*
    checkpoints/
      meiji-convnext-epoch=XX.ckpt
      last.ckpt
    qualitative/epoch_XXXX/ball_batch00.gif
```

ノートブックから直接シェルを実行した場合は
`MyDrive/tennis_lab/outputs/ball_detection/meiji_3cam_<UTC時刻>/` に保存する。
この2つの保存方式を混同せず、自分が選んだ実行方法の出力を確認する。

学習が終了したら次の順に確認する。

1. `dataset_summary.json` のclip ID・窓数・status件数が想定したデータに対応することを確認する。
2. `calibration/batch_size.json` で採用batchと次の2倍のOOM結果を確認する。
3. 本学習の `config.yaml` に採用batchと実行したhyperparameterが保存されていることを確認する。
4. TensorBoardで `val/loss`、precision、recall、F1の推移を確認する。
5. 可視化GIFの元画像・MDD・予測点・heatmapで、ボール位置と時間的な追従を確認する。
6. validation lossで保存されたcheckpointから、最も小さいlossのものを最終候補とする。

`last.ckpt` は最新の学習状態を保存するファイルであり、最良のvalidation lossを保証しない。
epoch番号の大きさだけで最良checkpointを選ばない。`val/loss` のログか、checkpointに含まれる
ModelCheckpoint callbackの `best_model_path` / `best_model_score` を確認する。
別環境へファイルを移した場合、callbackに記録されたパスは学習時のパスなので、
移動先の実ファイルへ対応付ける。

現在の学習GIFには教師座標の重ね合わせを渡していない。またGIFで描くのは各フレームの
代表peakで、metricsは最大8候補の対応付けを使う。GIFで追従の様子を確認し、
教師座標との定量的な一致は保存されたmetricsで判断する。

標準レシピはtestを実行しないため、終了時にtest指標がなくても異常ではない。
このデータの学習・validationでは位置不明窓を除いているので、位置不明区間を含む長い動画全体での
検出率や、未知の撮影環境での性能まで、このvalidation指標だけで判断しない。

## 中断・エラーが発生した場合

| 症状 | 確認と対応 |
|---|---|
| GPU名がL4以外 | L4が割り当てられたランタイムへ切り替える。scriptのGPU検査を無効にしない |
| Driveのtarが見つからない | マウントしたアカウントと `tennis_lab/data/meiji_3cam.tar` を確認する |
| `Expected exactly one dataset.json` | tar内のデータセットが0件または複数。入力のディレクトリ構成を確認する |
| `splits must partition ...` | 現在の33クリップとsplitファイルが一致しているか確認する。別versionのtarを使っていないか確認する |
| `Video SHA-256 differs ...` | 動画とアノテーションの組合せが一致していない。ファイルの出所を確認する |
| `Incomplete frame cache` / `Stale frame cache` | エラーに表示されたVMローカルのカメラキャッシュを確認し、そのキャッシュだけを除去して再生成する |
| 較正中、次の2倍のbatchがCUDA OOM | 最大値を決めるための想定した結果。直前まで成功していればその値で本学習へ進む |
| `Batch size 1 does not fit` | この設定で学習可能なbatchがない。GPU割り当て・他プロセス・設定変更の有無を確認する |
| `Calibration failed ... not an accepted CUDA OOM` | 指定された `batch_<B>.log` を読む。データ・依存・compile・CPUメモリの問題を、GPUメモリ不足として扱わない |
| 較正後の本学習がOOM | GPU使用状況と実行条件を確認する。自動的にbatchを縮める動作はない |
| `uv sync --locked` が失敗 | 依存構築のログを確認する。torchだけを別versionに入れ替えて同じレシピと扱わない |

### 再試行とcheckpoint再開を区別する

このレシピのL4 launcherは、新規の較正と新規学習を行うための入口。
ノートブックの学習セル再実行やworkflowの `resume` による処理再試行は、checkpointからの学習再開ではない。
同じrun IDで再試行する場合、較正JSONなどが再書き込みされるため、比較したいログは先に保管する。

checkpointから続けるには、保存済みconfig・採用batch・同一データを維持し、
通常の `src.tasks.ball_detection.scripts.train` に `run.resume` を明示する必要がある。
`resume_ball_meiji_3cam_l4.sh` はその条件を固定して再開するためのシェルで、L4、batch 8、
num_workers 8、勾配蓄積4、max epochs 20、test保留を再確認してからLightningを起動する。
新しいVMではDriveをマウントし、初回レシピと同じソースsnapshot、依存環境、展開済みdatasetを用意する。
画像キャッシュは学習開始時にVMローカルへ再生成される。

まず、再開元は**最も新しい完全な`last.ckpt`**を明示的に選ぶ。validation最良checkpointは
最終モデル選択用であり、学習状態を先へ進める用途ではない。checkpointをCPUへ読み、少なくとも
`epoch`、`global_step`、`state_dict`、`optimizer_states`、`lr_schedulers`が存在することを確認する。

```python
from pathlib import Path
import torch

checkpoint = Path(
    "/content/drive/MyDrive/tennis_lab/colab-live/<run-id>/outputs/colab/"
    "ball_meiji_3cam_l4/logs/version_N/checkpoints/last.ckpt"
)
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
print("epoch:", state["epoch"], "global_step:", state["global_step"])
print("optimizer states:", len(state["optimizer_states"]))
print("lr schedulers:", len(state["lr_schedulers"]))
```

完全状態を確認したcheckpointから次のように再開する。

```bash
%%bash
set -euo pipefail
cd /content/tennis-lab

RUN_ID='<run-id>'
OUTPUT_ROOT="/content/drive/MyDrive/tennis_lab/colab-live/${RUN_ID}/outputs/colab"
CHECKPOINT="${OUTPUT_ROOT}/ball_meiji_3cam_l4/logs/version_N/checkpoints/last.ckpt"

bash scripts/colab/train/resume_ball_meiji_3cam_l4.sh \
  "${CHECKPOINT}" \
  "paths.output_root=${OUTPUT_ROOT}" \
  "run.output_dir=ball_meiji_3cam_l4"
```

再開時は新しい`logs/version_N`が作られる。LightningのModelCheckpoint callbackは保存先が変わると
以前の`best_model_score`一覧を引き継がないため、学習完了後は各versionのイベントとcheckpointを
横断し、全epoch中の最小`val/loss`を選ぶ。Drive上のcheckpointを自動検索して暗黙に再開する処理はない。
workflowの再試行・成果物取得の具体的な操作は[共通README](../README.md#実行とlifecycle)に従う。

## 今回確認済みの範囲

| 項目 | 確認結果 |
|---|---|
| 実データの取り込み | 99動画、62,967フレームのアノテーションと動画ハッシュを確認、CPUで画像展開済み |
| BMP前処理 | 実データ8動画・8,791フレームを8並列で展開・再検証。ローカルCPUで約16.8秒 |
| 共通Datasetとの接続 | train/validation/testの画像・heatmap・座標テンソルを生成 |
| 本番入力サイズでのモデル実行 | CPUで `(1,8,3,288,512)` の入力から予測と有限の損失を確認 |
| optimizer更新 | CPUで空間解像度のみ32×64へ縮小し、完全なモデルの逆伝播とAdamW更新を確認 |
| 較正loop | CPU用の統合テストで3更新・validation・可視化とwarmup設定の接続を確認 |
| 自動テスト | 関連テスト（データ・設定・較正・Colab転送・再試行）とruff・mypyで確認 |
| Colab接続 | job schemaとdry-runでL4指定・入力・出力・snapshot転送経路を確認 |
| L4上の実測 | physical batch 8を採用し、勾配蓄積4で20 epochs・global step 6,320まで完了。測定値は上記の実行記録を参照 |
| 学習後のモデル精度 | 全epochのvalidationと最良checkpointを実行記録へ保存。testは意図どおり保留 |
