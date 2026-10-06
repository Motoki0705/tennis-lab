# 座標時系列の2D / 3D Refiner

[#991](https://github.com/Motoki0705/tennis-lab/issues/991)・[#1014](https://github.com/Motoki0705/tennis-lab/issues/1014)の実装。
モデルは**座標時系列と欠損maskだけ**を受け、観測区間も含む全frameの座標を1本返す。
offlineの双方向RoPE Transformerを共通にし、2Dは直接回帰、3Dは直接回帰／x0予測のconditional flow matchingを比較する。
回帰は全frameのSmooth L1復元損失に、任意の軌道のみを判別するLSGAN補助を加える。
GANの有効性は比較で判断し、平均化の回避を保証しない。

## モデル構成

| 場所 | 責務 |
|---|---|
| `models/generators/transformer.py` | 共通componentsのTransformerBlock・RoPE・RMSNorm・SwiGLUによる全frame生成 |
| `models/discriminators/` | BLCS/PLCS共通のTransformerSequenceDiscriminatorを構築。座標だけを入力しCLSから系列ごとに1スコア |
| `models/legacy.py` | v1のsinusoidalモデルを保存済みcheckpointから明示的に復元 |
| [`configs/model/coordinate_transformer.yaml`](../configs/model/coordinate_transformer.yaml) | Generatorの幅・深さ・FFN・RoPEの正本 |
| [`configs/training/coordinate_gan.yaml`](../configs/training/coordinate_gan.yaml) | 学習条件・GANの開始/線形増加・Discriminatorの正本 |

Generatorは座標と欠損mask、Discriminatorは生成/GT座標 `(B,T,D)` だけを受け取る。
Discriminatorには観測座標、欠損mask、速度・加速度特徴を渡さず、復元した全時刻を有効tokenとして扱う。
Flowは同じGenerator backboneを使用するがGANと併用せず、独立した比較方式として残す。

共通データ、学習済み重み、拡張前後をブラウザで比較する場合は
[2D / 3D Dataset Review](review/README.md)を参照。

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
- 既定はイベント連続欠損のみで、jitter・外れ値・離散欠損は0。ノイズを使う比較ではjitter/外れ値混合の非欠損frame全体の距離誤差P95を規定する。ノイズ無効時はP95・jitter・外れ値率を全て0にし、混在指定は拒否する。
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
  model.dimensions=3 model.architecture=flow training.gan.enabled=false
```

2D/3DのGAN各1本は [queue登録script](../../../../tests/benchmarks/ball_refiner_rope_gan.sh) を使う。
イベント選択率3条件×〔2D回帰/GAN、3D回帰/GAN/Flow〕の追加比較は
[比較用queue登録script](../../../../tests/benchmarks/ball_refiner_coordinates.sh)で現在の設定に対して登録する。
過去の15条件は当時のknowledge/repro bundleの設定を使い、現在の既定で再現したとは扱わない。
同時学習時の推論時間はGPU競合の影響を受けるため、性能報告では単独GPU計測を別に行う。
compileは設定で明示する。この比較の既定はeager実行。

`training.gan.transition.start_step` 回の更新を座標lossのみで行った後、
`warmup_steps` 回でGAN係数を `target_weight` まで線形に増やす。
BLCS/PLCSのepochスケジュールと同じ共通関数を更新回数に適用する。
G/DともAdamWを使い、GAN有効期間はD1回/G1回。Gの学習率だけ全学習期間でcosine減衰する。
TensorBoardとJSONLへ復元loss、生GAN loss、重み付きGAN loss、合計、実際の係数を記録する。

位置lossを後半で0へ減衰させる実験は [`training=coordinate_gan_only`](../configs/training/coordinate_gan_only.yaml) を選ぶ。
GANの開始・増加期間を保ち、最大係数を1にする。復元係数は最初の2,000更新で1、続く1,000更新で0へ線形減衰し、最後の1,000更新はGANだけでGeneratorを更新する。
復元誤差は係数0でも診断用に記録するが、更新の計算グラフから外す。
2D/3Dの登録は [GAN-only queue script](../../../../tests/benchmarks/ball_refiner_gan_only.sh) を使う。
通常の `coordinate_gan` は復元係数1を維持し、Flow比較にも使用できる。

validationの全frame RMSEでcheckpointを選び、その後だけtestを評価する。
`outputs/ball_refiner/train/<条件>/<run>/` に設定・劣化監査・TensorBoard・best/last checkpointと
`predictions/{pred_test.npz,metrics.json,diagnostic_metrics.json,examples.png}`（best）を保存する。
最終更新のlastも同じtest入力で評価し、同形式の `predictions_last/` に保存する。
各診断JSONはcheckpointの更新回数・hash・loss係数と評価入力hashを持つ。
GAN-onlyの評価ではbestの選択時点が移行前の可能性があるため、lastの結果を必ず区別する。
RMSEは軸平均でなくユークリッド距離の二乗平均平方根。全体・欠損・観測・イベント近傍を分け、
線形補間対照と実際のframe欠損率も残す。学習曲線と実験結論は `knowledge/` に登録する。

2026-10-05の15条件の学習・共通評価・単独GPU計測は
[比較記録](../../../../knowledge/nodes/ball_refiner/000040-group-i991-i1014-coordinate-refiners-s42.md)を参照。
各runのnodeから重みの場所・設定・全frame予測・学習曲線を確認できる。

## 推論

`inference.load_checkpoint()` と `refine_coordinates()` は2方式で共通。
入力は `(B,T,D)` とboolの `(B,T)`、**trueが欠損**。Bは独立した系列で、camera間attentionはない。
2Dは1280×720基準の画素、3Dはm。出力も同じ単位で、観測座標を固定せず全frameを予測する。
短い系列は実frameだけで処理し、長い系列は重複windowの中心に近い予測を採用する。
Flowは明示seedから1本だけ生成し、複数sampleの平均や正解による選択をしない。

`detection.heatmaps_to_coordinates()` はheatmapの最大点を座標化し、scoreは捨てる。
検出器由来の欠損maskは呼び出し元から明示する。
RoPEモデルはFFN種別を明示するv3 schemaで保存する。SwiGLU固定のv2重みはその形式を厳密に検査し、同じSwiGLU構成で復元する。v1座標モデルは専用legacy classから復元し、設定や重みを新構造へ読み替えない。
既存のGMM checkpointとは別schemaで、自動fallbackをしない。
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
