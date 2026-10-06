# Data

`schema.py` がrally・拡張結果・モデル入力の型を定義し、`dataset.py` が
`data/ball_refiner/single_object/` のmanifestとNPZを照合する。
BLCSの3D軌道と物理記録（[`ball_physics.v1`](../../../utils/README.md#physics)）を一度だけ保存し、多視点投影・camera・event・splitを共有する。
イベントは物理記録から導出し、shot・bounce・net・fenceの各bitを、イベントのsim step以後で最初の出力frameに置く。
train/val/test分割はrally単位で固定する。

`augmentation/occlusion.py` はイベントの前後を異なる幅で隠し、選択区間をcamera間で同期する。
`augmentation/noise.py` は観測座標へノイズを適用する。
`preprocessing.py` は拡張後の観測だけを共通の多視点三角測量へ渡す。
2view未満・退化・負depthは3Dでも欠損とし、GTを入力へコピーしない。
2D座標はこの生成過程にのみ必要で、2D Refinerは存在しない。
設定の正本は [augmentation/event_only.yaml](../configs/augmentation/event_only.yaml)。

`sampling.py` は既定で128 frameの窓を作り、確率 `long_probability` でそれより長い窓
（ラリー全長まで、上限 `max_length`）を一様に選ぶ。batch内は最長の窓へpaddingし、paddingは欠損かつ教師なしとする。
物理教師（`physics/targets.py`）は全frameの区間ラベルと状態、ラリーの場とsurfaceで、窓ごとに区間を0から振り直す。

`targets/events.py` はイベント種別を区別せず、各イベントframeで1となるGaussian教師を作る。
重なる教師はmaxで合成する。ラリー全体で生成してから `sampling.py` が窓を切り出すので、窓外イベントの裾も保持する。
モデルへ渡す入力は座標・欠損maskだけ。GT座標とイベント教師はloss専用。

`datamodule.py` はLightningへ接続する。学習の拡張を `evaluate_every` 更新ごとに作り直し、
固定した評価拡張・seedでvalidation/testを用意する。
窓の抽選は独立した状態付き乱数生成器を持ち、checkpointへ保存する。
厳密な再開のためDataLoaderは `num_workers=0`、1batch=1generator更新とする。
