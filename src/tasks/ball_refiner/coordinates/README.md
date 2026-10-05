# 座標時系列の2D / 3D Refiner

[#991](https://github.com/Motoki0705/tennis-lab/issues/991)・[#1014](https://github.com/Motoki0705/tennis-lab/issues/1014)の実装。
モデルは**座標時系列と欠損maskだけ**を受け、観測区間も含む全frameの座標を1本返す。
offlineの双方向Transformerを共通にし、2Dは直接回帰、3Dは直接回帰／x0予測のconditional flow matchingを比較する。
回帰は全frameの復元損失に、任意のconditional LSGAN補助を加える。
GANの有効性は比較で判断し、平均化の回避を保証しない。

## 共有データ

`data/ball_refiner/single_object/` が2D・3Dで共通のdataset。
`rallies/<id>.npz` に3D真値を一度だけ保存し、投影2D座標・camera行列・イベント・時刻を同じファイルに持つ。
`manifest.json` がhash、rally単位のtrain/val/test、生成条件を固定する。
別camera・重複windowを別splitへ入れない。モデル別のdatasetコピーを作らない。

BLCSの `RallySimulator` を物理解像度で呼び出し、返球より後の仮想bounceを除外した後で
出力frameへイベントを最近傍対応させる。fence退出時に終端を切り、短すぎる／camera後方に出る
ラリーは有限回数で再提案する。各recordに試行数と元のshot metadataを残す。
cameraは両baseline後方のfence面でX/Z、注視方向、画角を変え、各rally中は固定する。
cleanな軌道全体が画面内に収まるcameraを有限回数で選ぶ。近接cameraの画面外投影が
数万pxになって比較を支配することを防ぎ、この実験では欠損を人工遮蔽・離散欠損として制御する。
`reprojection.reproject_dataset()` は保存済み3D・イベント・splitを変えず、cameraだけを変更できる。
生成値・FPS・件数の正本は [generate_coordinates.yaml](../configs/generate_coordinates.yaml)。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m src.tasks.ball_refiner.scripts.generate_coordinates
```

既存datasetへの上書き・暗黙の再開は拒否する。不完全な生成は `state.json` で区別する。

## 劣化と公平な比較

- bounce／shotをイベント単位で選択し、左右の幅を異ならせた連続欠損を作る。重複区間は和集合、端はclip境界で切る。
- 選択イベントの欠損はcamera間で同期させ、3Dにも実際の連続欠損を作る。残りの観測frameの離散欠損はcameraごとに独立。
- 小さな定位jitterと大きな誤検出の混合分布を使い、1280×720での非欠損frame全体の距離誤差P95を規定する。画面端でnoiseをclipしない。
- 3D入力は同じ劣化済み2Dを既存のmulti-view DLT＋再投影最小化へ渡して作る。2視点未満・退化・負depthは欠損とし、3D真値から入力を作らない。
- 学習の劣化は一定更新間隔で再生成する。評価は学習側のイベント選択率に依存しない固定seed／率を使い、モデル間で入力hashを照合できる。

具体値は [train_coordinates.yaml](../configs/train_coordinates.yaml) が正本。
2Dは画像幅・高さで、3Dは固定の世界座標scaleで正規化する。court位置・検出confidence・camera・eventをモデル特徴へ渡さない。
欠損値はforward前にゼロ化し、missing tokenも双方向attentionのqueryとして全frameを復元する。

## 学習とアブレーション

GPU学習は必ず[共有training queue](../../../../.agents/skills/training-queue/SKILL.md)を使う。
単一条件のCLIは次のとおり（queueのcommand文字列として登録する）。

```bash
.venv/bin/python -m src.tasks.ball_refiner.scripts.train_coordinates \
  model.dimensions=3 model.architecture=flow training.gan_weight=0
```

事前宣言したイベント選択率3条件×〔2D回帰/GAN、3D回帰/GAN/Flow〕の15条件は
[比較用queue登録script](../../../../tests/benchmarks/ball_refiner_coordinates.sh)で同一seed・更新予算を登録する。
同時学習時の推論時間はGPU競合の影響を受けるため、性能報告では単独GPU計測を別に行う。
compileは設定で明示する。この比較の既定はeager実行。

validationの全frame RMSEでcheckpointを選び、その後だけtestを評価する。
`outputs/ball_refiner/train/<条件>/<run>/` に設定・劣化監査・TensorBoard・best/last checkpointと
`predictions/{pred_test.npz,metrics.json,diagnostic_metrics.json,examples.png}` を保存する。
RMSEは軸平均でなくユークリッド距離の二乗平均平方根。全体・欠損・観測・イベント近傍を分け、
線形補間対照と実際のframe欠損率も残す。学習曲線と実験結論は `knowledge/` に登録する。

## 推論

`inference.load_checkpoint()` と `refine_coordinates()` は2方式で共通。
入力は `(B,T,D)` とboolの `(B,T)`、**trueが欠損**。Bは独立した系列で、camera間attentionはない。
2Dは1280×720基準の画素、3Dはm。出力も同じ単位で、観測座標を固定せず全frameを予測する。
短い系列は実frameだけで処理し、長い系列は重複windowの中心に近い予測を採用する。
Flowは明示seedから1本だけ生成し、複数sampleの平均や正解による選択をしない。

`detection.heatmaps_to_coordinates()` はheatmapの最大点を座標化し、scoreは捨てる。
検出器由来の欠損maskは呼び出し元から明示する。
既存のGMM checkpointとは別schemaで、読み替え・自動fallbackをしない。
旧scene pipelineの配布モデル切替は、この合成比較の自動的な結果とはしない。

NPZ入力は `coordinates`, `missing`, `fps`、2Dの場合は追加で `image_size_wh` を持つ。
CLIはsource画素をreference gridへ変換して推論し、source画素へ戻す。
FPS不一致は明示的なresamplingを要求する。

```bash
.venv/bin/python -m src.tasks.ball_refiner.scripts.predict_coordinates \
  --checkpoint /absolute/run/logs/version_0/checkpoints/best.ckpt \
  --input /absolute/input.npz --output /absolute/prediction.npz --device cpu
```

主要な検証は [unit](../../../../tests/unit/tasks/ball_refiner/test_coordinates.py) と
[CPU end-to-end](../../../../tests/integration/tasks/ball_refiner/test_coordinates_training.py)。
