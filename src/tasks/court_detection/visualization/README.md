# Court Detection 可視化ガイド

原画像にコートのGT・予測KP、seg、line、semantic_lineを重ねて確認します。
[共通の実行前確認](../../base/visualization/README.md#実行前確認)を済ませ、コードのあるリポジトリまたはworktreeの直下で実行してください。各ブロックは単独でコピーできます。

## 発表用のモデル・学習ビジュアル

```bash
.venv/bin/python -m src.tasks.court_detection.scripts.visualize_learning_stages
```

CPUで、左にアーキテクチャ・混合バッチとpose教師の流れ、右に実測の検証4指標・学習loss・
学習率を描きます。凍結状態、LoRAバッジ、バッチ数、射影段数、後段の容量は保存configから決めます。
既定は45秒・1920×1080・30fpsのMP4、幅960・15fpsのGIFと、
`stills/{intro,train,select,handoff}.png`です。解像度・fps・各区間の秒数・tag対応は
[`configs/visualize_learning_stages.yaml`](../configs/visualize_learning_stages.yaml)で明示します。
出力は`paths.output_root`配下の`court_detection/visualize/learning_stages/<run-id>/presentation/`で、
同じrunの`hydra/`にCLIログを置きます。既存の映像出力先は拒否します。
worktreeから共有成果物へ出す場合は`paths.output_root=<元repoの絶対path>/outputs`を指定します。

入力はrepo内の`knowledge/runs/run-i983-court-dinov3-b-s42`です。
train/valの0始まりevent stepを+1し、更新回数0〜33,160で表示します。
検証は6点、学習lossは50更新ごとの663点で、lossだけを過去20点の移動平均・対数軸で表示します。
LRはepoch開始時の実測20点を区間内で保持して描きます。
run保存基準に従ってraw eventを保持せず、`scalar_summary`で指定した
[発表用assets](../../../../assets/court_detection/learning_stages/)の`scalars.json`を読みます。
元runの内容は変更しません。同assetsの`export_scalars.py`がhash検証済み原本から
必要な13 tagのstep・値・wall_timeを無加工で抽出し、`derivation.json`に由来を記録します。
再抽出時は`--bundle`に元run、`--raw-file`に原本、`--tags-config`に上記の可視化config、
`--output-dir`に新しい出力先を指定します。
必要tag・件数・step・保存config・checkpoint選択が食い違う場合は停止します。

図と粒子は模式表現、曲線は**#983のB runの実測値**です。pipeline既定のcheckpointとは別runです。
数値をスライドに記す場合はB runの値であると注記してください。
`manifest.json`には入力SHA-256、選択checkpointの記録済みSHA-256、実装commit、解決済みconfig、
全scalar（画面に出さない線種mIoUを含む）、`measured` / `schematic`の区分を保存します。
checkpoint本体は読みません。最後のKP14・LINEの色は
[`presentation/style.py`](../../base/visualization/presentation/style.py)が正本です。

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

## 発表用: KP＋LINEからのホモグラフィー

`scripts.visualize_hybrid_fit` は、KPだけのホモグラフィーが誤り、KP＋LINEの
同時当てはめで改善する実画像を探し、1枚の画像で推論の各段階を描きます。
GPUを使う以下のコマンドは、コードのあるworktreeをcwdにして
[training queue](../../../../.agents/skills/training-queue/SKILL.md)へhalf枠で登録します。
workerは親セッションが管理します。`ROOT` は上記と同じ元リポジトリです。

```bash
# 候補探索（TennisCourtDetector valの先頭300件。条件を自動で緩めません）
.venv/bin/python -m src.tasks.court_detection.scripts.visualize_hybrid_fit \
  mode=search paths.data_root="$ROOT/data" paths.checkpoint_root="$ROOT/ckpt" \
  paths.external_asset_root="$ROOT/third_party" paths.output_root="$ROOT/outputs"

# ranking.jsonから選んだsampleを14秒・30fpsで描画
.venv/bin/python -m src.tasks.court_detection.scripts.visualize_hybrid_fit \
  mode=render sample_id=CHOSEN_SAMPLE \
  paths.data_root="$ROOT/data" paths.checkpoint_root="$ROOT/ckpt" \
  paths.external_asset_root="$ROOT/third_party" paths.output_root="$ROOT/outputs"
```

checkpointはpipelineと同じ `court_detection/multiscale_depth3/b863df1f01f0.ckpt`。
`search.min_kp_only_error_px=10` 以上かつ `search.max_hybrid_error_px=3` 以下の
成功例を、改善量の降順で最大 `search.top_k=20` 件残します。両誤差は同じ可視GT14点での
原画像pixelの平均距離です。`search.max_samples` が探索予算です。
0件なら空のrankingとcontact sheetを保存したうえでエラー終了します。
sampleの最終採用はユーザーが行います。

出力先は `<OUTPUT>/court_detection/visualize/hybrid_fit/<run-id>/presentation/`。
Hydraログは同じrunの `hydra/` に置き、既存のpresentationは上書きしません。
searchは `ranking.json`（候補・全探索対象の誤差・画像hash）と `contact_sheet.png`、
renderは `hybrid_fit.mp4`・`hybrid_fit.gif` と
`stills/{01_input,02_kp,03_kp_only,04_line,05_seeds,06_hybrid}.png` を生成します。
MP4は既定1920×1080・30fps、GIFは幅960・15fpsです。
比較用には `output.width=960 output.height=540` を指定できます。画像の縦横比は保ちます。

描画は原画像 → KP heatmap/peak → KPだけの赤線 → 水色のLINE → 黄色のseed →
緑の最終fit → 原画像と緑線の静止、の順です。PROSACのfit inlierは塗りつぶし、
その他は中空、最終fitに選ばれなかったKPは赤い×です。画面内に文字は出しません。
seedは推定器が実際に選んだものだけを使い、native gridのKP/LINEは表示用にだけresizeします。

`mode=render input.kind=image input.path=<DATA相対PNG> sample_id=null` では
別の単一画像も描画できます。この場合GT誤差は算出しません。
`manifest.json` には解決済みconfig・commit・入力とcheckpointのSHA-256・段階名を記録します。
画像・予測・seed・fitは `measured`、表示の時間と色は `schematic` です。
JPEG storeではdescriptor/indexのfile hashと、読取時に検証したJPEG byte列のhash・
RGB hash・注釈recordのhashを記録し、個別画像ファイルがあるとは仮定しません。
