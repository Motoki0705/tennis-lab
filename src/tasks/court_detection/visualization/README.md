# Court Detection Web UI 利用ガイド

原画像にコートのGT・予測KP、seg、line、semantic_lineを重ねて確認します。
[共通の実行前確認](../../base/visualization/README.md#実行前確認)を済ませ、コードのあるリポジトリまたはworktreeの直下で実行してください。各ブロックは単独でコピーできます。

## データセット閲覧

```bash
ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
./scripts/run_in_repo_venv.sh python -m src.tasks.court_detection.scripts.review_dataset \
  --project-root "$ROOT" --data-root "$ROOT/data" \
  --port 8774
```

[閲覧UIを開く](http://127.0.0.1:8774)。左でTennisCourtDetectorまたは合成シーンのsplitを選び、画像を選択します。checkpoint・GPUは不要です。

## 推論

```bash
ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
./scripts/run_in_repo_venv.sh python -m src.tasks.court_detection.scripts.inference_ui \
  --project-root "$ROOT" --data-root "$ROOT/data" \
  --outputs-root "$ROOT/outputs/court_detection" \
  --checkpoints-root "$ROOT/ckpt/court_detection" \
  --port 8775
```

[推論UIを開く](http://127.0.0.1:8775)。閲覧と同時に使う場合は別ターミナルで起動します。

1. 左でcheckpointを検索・選択し、対応するdataset・画像を選びます。
2. 右のDeviceをCUDAにします。CPUで試す場合はCPUを明示選択します。
3. 開始フレーム0・フレーム数1で推論を実行します。GPU要求は[共有キュー](../../base/visualization/README.md#web-uiのgpu実行)で実行されます。
4. 右でGT・Prediction、ラスターレイヤー、不透明度を切り替え、画像上の重なりと各headの指標を確認します。

## 表示操作

ドラッグで画像を移動、ホイールでズーム、フィットで全体表示へ戻します。ダウンロードボタンで表示PNGを保存できます。Courtは単画像なのでフレーム再生はありません。

## パスと注意点

- `--data-root`は**`data`**です。TennisCourtDetectorは`data/court`、合成データは`data/synthetic_data_generation/scenes`から解決します。
- 必須configや`target_bundle_state`がない旧checkpointは非対応理由を表示します。
- GTマスクが未生成・古い場合は、そのレイヤーだけ警告になります。更新後は画面右上でカタログを再読み込みします。
- 別の保存先を使う場合は各root引数を実際の絶対パスに置き換えます。
- ポートが使用中なら`--port`を空き番号へ変更し、その番号のURLを開きます。終了はCtrl+Cです。

[データ・checkpointの契約](../README.md#dataset-review--inference-ui) / [共通画面・HTTP API](../../base/visualization/detection/README.md)
