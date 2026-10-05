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

「体系・内訳を見る」で、現物の保存件数・現行readerのsplit別件数・保存schema・
source設定の除外を確認できます。実写のtestは未提供です。合成のvalidationはUIのvalに
対応し、trajectory group単位で分割されます。対応する形式と教師契約は
[学習データ](../README.md#学習データ)が正本です。

1. 左の「注釈の確認候補」で、画面外KP・画像内のrenderer不可視KP・重複座標があるsampleを絞ります。source/splitはdataset選択、sample IDは検索で指定します。
2. 右の「保存教師の検品」で14点の名前・元画像pixel座標・KP教師採用状態を確認します。実写の「画像内・遮蔽不明」は保存visibilityがないという意味です。合成は保存in_front / in_frame / renderer_visibleと、KP教師への採用を区別します。
3. 点を選ぶとchannel番号・physical IDを表示し、画像内の点へ拡大します。画面外の座標は保持し、画像内へ移動させません。「全体表示」で戻ります。ラベルを有効にすると画像上は短いchannel番号、右は完全な点名になります。
4. 画像上の「画像だけ」「保存KP」「SEG」「LINE」「semantic LINE」を切り替えて元注釈と派生教師を比較します。「不可視点を参考表示」はKP教師対象外の位置を黄色の破線で示します。可視教師へ変更する操作ではありません。
5. 「raw注釈」で、選択sampleの保存record・source/provenanceをJSONとして開きます。合成compact storeでは実際のstored_recordとreaderが復元するlogical labelsを分けます。「schema / source / sample」「派生targetのschemaと読み方」で保存先と派生schemaを確認できます。

SEG/LINE/semantic LINEは両sourceとも保存マスクではなく、同じKP14からhomographyを
推定して生成します。画面ではaugmentationなしの原画像解像度、学習では変換後の解像度と
paddingに対して同じgeneratorを使用します。遮蔽物を除くマスクではなく、元KPの誤りも
引き継ぐため、派生ラスタの一致だけで注釈の正しさを判断しないでください。

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

ドラッグで画像を移動、ホイールでズーム、フィットで全体表示へ戻します。ダウンロードボタンで表示PNGを保存できます。Courtは単画像（frame 0）なので再生操作は表示しません。

## パスと注意点

- `--data-root`は**`data`**です。TennisCourtDetectorは`data/court_detection/tennis_court_detector-v1`、合成データは`data/synthetic_data_generation/scenes`から解決します。
- 必須configや`target_bundle_state`がない旧checkpointは非対応理由を表示します。
- GTマスクはKPからオンザフライ生成します。datasetを更新した場合はカタログを再読み込みします。
- 別の保存先を使う場合は各root引数を実際の絶対パスに置き換えます。
- ポートが使用中なら`--port`を空き番号へ変更し、その番号のURLを開きます。終了はCtrl+Cです。

[データ・checkpointの契約](../README.md#dataset-review--inference-ui) / [共通画面・HTTP API](../../base/visualization/detection/README.md)
