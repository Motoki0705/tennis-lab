# Training

`runner.py` は `BaseTrainingRunner` のfit・callback・artifact lifecycleを使い、
`lightning_module.py` は `BaseLightningModule` のoptimizer・scheduler・compileを使う。
独自の手書き学習ループは持たない。

## 設定

- optimizer・batch・更新回数: [training/_base.yaml](../configs/training/_base.yaml)
- 位置／イベントlossと位置係数schedule: [loss/default.yaml](../configs/loss/default.yaml)
- optional GAN・discriminator・GAN係数schedule: [training/_gan.yaml](../configs/training/_gan.yaml)
- 実行先・resume・初期重み: [run/train.yaml](../configs/run/train.yaml)

回帰の既定は全frameの正規化座標L1とイベントsoft cross-entropy。
教師は `[1-p,p]`。位置:イベントの係数は1:1、GAN・loss係数scheduleはoff。
座標はXYZを10/20/5mで割り、L1を軸・frame・batchで平均する。
係数の比率はloss実測値や勾配量の比率とは異なる。
Flowでは位置側をFlow matchingの速度MSEに置き換える。

GANは共通 `ManualGANTrainingStrategy` / LSGANを使う。
`training.gan.enabled=true` で有効化し、段階増加は
`training.gan.schedule_enabled=true` を明示する。FlowとGANの併用は拒否する。

GANのD更新を含むLightning global_stepとは別に、generator更新数を数える。
loss係数schedule・学習予算・拡張の更新周期はこの数に従う。
`evaluate_every` 更新を1epochとし、最後の端数epochも処理する。

## 保存と再開

出力は `outputs/ball_refiner_3d/train/<architecture>/<run>/`。
`logs/version_0/checkpoints/best.ckpt` は固定validation全frameの位置RMSEで選び、
`last.ckpt` は最終状態を保存する。bestとlastを同じtest入力で評価し、
`predictions/` と `predictions_last/` に座標・イベント確率・対応frame・指標を保存する。
位置RMSEはm単位のユークリッド距離。イベントはBrier scoreとsoft CEを記録する。

新しい `events.v2` のresumeは、元の出力先と同じ設定・seed・dataset・CPU/CUDA種別で、
validationを完了した拡張block境界から行う。optimizer・scheduler・窓抽選・Flow・dropoutの乱数状態も復元する。
`run.resume` には共通path契約のcheckpoint相対パス、または
`{role: artifact, path: <artifact_rootからの相対パス>}` を指定する。

旧 `events.v1` はoptimizer全状態のresumeに対応しない。
`run.init_weights` なら既存の回帰／Flow重みを新しい実験の初期値に使える。
モデル構成とGaussian幅が一致する場合だけ読み込む。
