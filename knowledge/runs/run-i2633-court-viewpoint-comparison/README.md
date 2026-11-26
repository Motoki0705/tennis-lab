# コート視点分布比較の再現

実験の解釈と主要結果は比較groupにある。`metrics.csv`は実画像val/testの全層・各seedの指標、`comparison.json`はseed間の対応を保った効果量と区間を含む。`test-effects.png`は実画像testの図。4本の個別runには実行コマンド、解決済み設定、checkpointと採点データのhashを保存した。 [比較group](../../nodes/court_detection/000084-group-i2633-court-viewpoint-diversity.md) · [指標CSV](metrics.csv) · [比較JSON](comparison.json) · [testの図](test-effects.png)

## 入力の準備

`SOURCE_POOL`に既存の`court_scene_diversity_6000_v1`のversionファイル、`SCENES_ROOT`に撮影カメラを含むB00〜B03の親ディレクトリ、`TCD_ROOT`にTennisCourtDetector-v1のJPEG storeを指定する。`CAMPAIGN`は同じファイルシステム上の新しい出力パスとする。準備はsourceを検査し、同じJPEGをhard linkした作業用storeにsubsetを追加する。 [描画プール](../../nodes/synthetic_data_generation/000028-run-court-scene-diversity-6000-v1.md) · [入力と分布の記録](preparation.json)

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent.local_jobs start --name court-viewpoint-inputs -- \
  env PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B knowledge/runs/run-i2633-court-viewpoint-comparison/prepare.py \
  --source-pool "$SOURCE_POOL" --scenes-root "$SCENES_ROOT" --tcd "$TCD_ROOT" --output "$CAMPAIGN"
```

準備結果の`inputs/pool/`にできるtrain 2ファイルと共通のval、`registry.json`のバイト列を保存版と比較できる。`strata.json`と`selected-views.json`は作業領域に置き、そのSHA-256を保存した。実効区画数は1軸ずつ各シーンで区画のShannon entropyを計算し、その指数を4シーンで等重みに平均した。 [狭分布](narrow.train.subset.json) · [広分布](broad.train.subset.json) · [合成val](common.val.subset.json) · [実画像registry](registry.json)

## 学習と採点

実行時のソースは `38eb91cde9d5667de2bd648b27c5cb593331c1b9`。`CHECKPOINT_ROOT`は凍結DINOv3-S/16の事前学習重みを含むディレクトリ、`EXTERNAL_ASSET_ROOT`は対応する`dinov3/` checkoutの親。実行時の環境はRTX 5060 Ti、PyTorch 2.13.0+cu130、OpenCV 5.0.0だった。学習とGPU採点は共有training queueに`--resource all`で登録する。 [4本のqueue照合](queue-evidence.json) · [queue運用](../../../.agents/skills/training-queue/SKILL.md)

以下のコマンド文字列を、`CONDITION`が`narrow`／`broad`、`SEED`が`42`／`43`の組合せごとに登録する。各runは専用の新しい出力ディレクトリを使う。

```bash
timeout --foreground --signal=TERM --kill-after=30s 170m \
  env PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -u knowledge/runs/run-i2633-court-viewpoint-comparison/train_and_evaluate.py \
  --campaign "$CAMPAIGN" --checkpoint-root "$CHECKPOINT_ROOT" \
  --external-asset-root "$EXTERNAL_ASSET_ROOT" --condition "$CONDITION" --seed "$SEED"
```

採点では予測の各channelから閾値なしで1点を選び、サブピクセル補正をした原画像座標を使う。KP誤差には画像内GT点を使い、元の長辺から512へ線形換算する。Hには信頼度0.05以上の点、画像対角長0.5%のinlier距離、既存のconfidence PROSACを使う。H@1%は全画像を分母にし、成立したHの可視GT点への再投影誤差平均が画像対角長1%以内の画像を数える。GTによる層別の角度は投影から求める指標。 [採点実装](protocol.py)

## 保存済み結果の再集計

完了した`CAMPAIGN`には4本のcheckpoint、`complete.json`、`val.json`／`test.json`と準備記録が残る。実行時の実体はworktreeに対応するE:側の`court_detection/campaign/viewpoint-diversity-v1`。各runの出力の絶対パスは、そのrunのqueue再現記録から確認できる。再集計はCPUで動作し、checkpoint・設定・採点行を照合する。`REPORT_DIR`に新規の出力先を指定する。 [キューに登録したコマンドと出力先の例](../run-i2633-viewpoint-narrow-s42/run.json)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -B \
  knowledge/runs/run-i2633-court-viewpoint-comparison/report.py \
  --campaign "$CAMPAIGN" --output "$REPORT_DIR"
```

効果量の符号は広分布−狭分布。seedごとの誤差と率を求めてから2 seedを同じ重みで平均する。区間は動画ごとの全対象画像を対で2,000回再抽出したパーセンタイル95%区間。seedは固定し、動画標本の変動を示す。空の視点層は数値を欠測として保持する。
