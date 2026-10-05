# Court Side dataset review

component storeが採用するボール観測・camera-local校正・side判定を、同じsource frameの全カメラ画像で確認します。
画面の採用・停止は**その保存実験の判定**です。現在の本番選択、現行モデルの成績、独立GTの保証を表しません。
dataset体系と実データの調査・画像は [Dataset Review Issue #1006](https://github.com/Motoki0705/tennis-lab/issues/1006) に集約しています。

## 起動

リポジトリのPython環境で、絶対パスを明示します。worktreeでは `scripts/run_in_repo_venv.sh` も使用できます。

```bash
.venv/bin/python -m src.tasks.court_side.scripts.review_dataset \
  --store /absolute/component/store \
  --diagnostics /absolute/saved/production/diagnostics \
  --comparison-store /absolute/another/store \
  --port 8899
```

`--store` は `scene.json` を持つdirectoryです。`--diagnostics` と `--comparison-store` は省略できます。
比較storeは独立した観測セットとしてselectorに表示し、frame別costを他のセットへ流用しません。
HTTPは127.0.0.1にbindし、dataset/注釈/storeへの書込APIは持ちません。
GPU、モデルload/推論、side scoring、benchmark、人物処理を実行しません。

## 見るもの

- frame入力・slider・timeline・再生で全cameraの画像と保存点を同時に移動します。128 px範囲の拡大を併記し、全画像のクリックで拡大中心を変えられます。
- 観測timelineは保存maskの分布です。採点frame IDが未保存なら、観測frameを採点・支持frameと呼びません。
- 仮説表は保存cost/supportと、このframeの保存CSVを比較します。支持数は保存support率×frame数の換算です。欠測frameを跨いでcostを結びません。
- 校正図は保存camera位置・向きにside仮説の半回転を適用した表示です。三角測量・再投影scoreを再計算しません。
- 注釈のobserved / interpolated / occlusion_estimated / unresolved、未保存、未採点、画像外、RGBファイル欠落を区別します。

## 読取契約

`tennis_scene_index_v1` の公開referenceをchecksum・lineage検証して読みます。
`local_court_calibration v1`、`court_side v3`、現行 `ball_points v2` を扱います。
明示的なhistorical readerとして `ball_detections v1/v2`、`court_side v2` も扱い、schemaとprovenanceを出典panelへ出します。
旧side v2のcost/support/frame数は未保存とし、現在の閾値を代入しません。
現行 `ball_points v2` は全frameの最大weight成分平均点を表示し、presenceで除外しません。
sideがあるstoreでは表示点がそのsideのinput lineageに一致することも要求します。
unsupported schema、checksum不一致、古いdependencyは停止します。起動後のscene index変更も再起動を要求します。

optional診断は、既存の `production.json`、`production-observations.npz`、`production-frames.csv` を読む契約です。
reportのconfig/execute入力receipt、対象source/indexと実行時artifact lineage、全cameraの観測配列、sampling/maskとCSV frame/camera軸、CSVと集計scoreの一致を検証します。
診断がなければframe別cost/support、採点frame ID、pair支持数は未保存です。
この既存診断形式は全source cameraが校正された保存production停止runに対応します。

## 検証

```bash
.venv/bin/python -m pytest -n 2 tests/unit/tasks/court_side/review
NODE_PATH=/path/to/node_modules node tests/e2e/tasks/court_side/dataset_review_browser.cjs \
  http://127.0.0.1:8899 /absolute/capture/output
```

browser検証は、上記の実データproduction/比較storeで起動したサーバーを使います。
通常unit testのfixtureはcontract確認専用で、dataset画像の代用には使用しません。
