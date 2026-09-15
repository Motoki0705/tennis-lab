# BLCS Web UI 利用ガイド

ボールの3D軌道を、コートとカメラ位置とともに確認します。
[共通の実行前確認](../../base/visualization/README.md#実行前確認)を済ませ、コードのあるリポジトリまたはworktreeの直下で実行してください。各ブロックは単独でコピーできます。

## データセット閲覧

```bash
ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
./scripts/run_in_repo_venv.sh python -m src.tasks.blcs.scripts.review_dataset \
  --data-root "$ROOT/data" \
  --port 8773
```

[閲覧UIを開く](http://127.0.0.1:8773)。左の形式・シーンを選ぶと、checkpoint・GPUなしでGTを確認できます。single/multi × 通常/broadcast/camera_view_v2の6形式に対応します。

## 推論

```bash
ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
./scripts/run_in_repo_venv.sh python -m src.tasks.blcs.scripts.inference_ui \
  --data-root "$ROOT/data" \
  --outputs-root "$ROOT/outputs/blcs" \
  --checkpoints-root "$ROOT/ckpt/blcs" \
  --device cuda --port 8770
```

[推論UIを開く](http://127.0.0.1:8770)。閲覧と同時に使う場合は別ターミナルで起動します。

1. 左でcheckpointを選び、互換な形式・splitからシーンを選択します。
2. 右でdevice、使用カメラ、reference camera、開始フレーム・窓長を指定します。
3. 推論を実行し、中央でGT・予測の軌道を重ねて再生します。
4. single-objectでは平均位置誤差・終端誤差・0.3m以内のフレーム率を確認します。

GPU要求は[共有キュー](../../base/visualization/README.md#web-uiのgpu実行)で実行します。CPUで試す場合は起動引数を`--device cpu`にするか、画面でCPUを選択します。

## 表示操作

ドラッグで回転、Shift＋ドラッグで移動、ホイールでズームします。視点プリセット・カメラ位置・カメラ視点・追従・軌跡を切り替え、下部で再生・フレーム送り・シーク・速度を調整します。

## パスと注意点

- 閲覧・推論とも`--data-root`は**`data`**です。`data/blcs`ではありません。
- checkpointは`outputs/blcs`と`ckpt/blcs`から再帰探索します。
- 別の保存先を使う場合は各root引数を実際の絶対パスに置き換えます。
- ポートが使用中なら`--port`を空き番号へ変更し、その番号のURLを開きます。終了はCtrl+Cです。

[モデル対応・推論契約](inference/README.md) / [閲覧API・座標系](../../base/visualization/review/README.md)
