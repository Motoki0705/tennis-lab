# Ball detection training

公開入口は `src.tasks.ball_detection.scripts.train`。GPU実行は元repo共通のtraining queueを使う。

| stage | 初期化 | 実装と教師 |
|---|---|---|
| `convnext_heatmap` | 保存configで明示 | `convnext_heatmap/`の既存Lightning/GAN。既存heatmap契約 |
| `dpt_pretrain` | ランダム初期化、または同run再開 | `dpt_pretrain/`。CNN＋DPT、observed正例と確認済み不在のheatmap教師 |
| `coordinates` | `--initialize scratch` | `coordinates/scratch.py`。pose32条件/query-only4条件、observed座標、constant LR |
| `coordinates` | `--initialize encoder` | `coordinates/finetune.py`。完了DPTのCNN転送、query-only、拡張、freeze→joint |
| `coordinates` | `--initialize full_model` | `coordinates/full_model.py`。採用query checkpointの全重みを転送し、optimizer/scheduleを初期化。累積trainと固定validationを個別指定 |

encoder/full_modelのrunは`coordinates/joint.py`、座標系のepoch更新は `coordinates/loop.py` と `runtime/execution.py` を共有する。
JPEG復号・GPU転送・precision/compileは `runtime/`、LRは `schedules.py`、epoch内FPS混合は `sampling.py`。
段階別runnerが他のrunnerから共通関数をimportしない。評価は `evaluation/`、保存形式は `checkpoints/` が担当する。
窓datasetは同じsampleに`position_valid`と`heatmap_valid`を供給し、各objectiveが使う教師を明示する。

## 実行例

以下のartifact/outputは絶対pathで指定する。元/1/2/1/4 FPSはmanifestで固定され、各窓32枚を選んでからMDDを生成する。

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.train --stage dpt_pretrain \
  --manifest /absolute/mdd_only_windows.json \
  --model-config /absolute/configs/model/mdd_dpt_convnext_v2.yaml \
  --output /absolute/new-pretrain-run --epochs 10 --windows-per-epoch 6000 \
  --learning-rate 0.0002 --warmup-updates 500 --seed 42 \
  --device cuda --precision bf16 --batch-size 1 --jpeg-decoder nvjpeg --pin-memory

.venv/bin/python -m src.tasks.ball_detection.scripts.train --stage coordinates --initialize encoder \
  --manifest /absolute/mdd_only_windows.json --pretraining-run /absolute/completed-pretrain-run \
  --augmentation-config /absolute/configs/augmentation/mdd_posttraining.yaml \
  --output /absolute/new-posttrain-run --epochs 10 --freeze-epochs 1 \
  --windows-per-epoch 6000 --learning-rate 0.0001 --encoder-lr-ratio 0.1 \
  --warmup-updates 500 --seed 42 --device cuda --precision bf16 --batch-size 1 \
  --jpeg-decoder nvjpeg --pin-memory --stress-evaluation
```

CNN候補は`configs/model/mdd_dpt_{pretrain,convnext_v2,fasternet}.yaml`で選ぶ。
pose/圧縮比較は`prepare_data`が作るmodel設定を`--stage coordinates --initialize scratch --model-config ...`へ渡す。
CPU smokeでは`--device cpu --precision fp32 --jpeg-decoder opencv --compile-mode off --num-workers 0`を明示する。CUDAの事前/事後学習はBF16固定。

画像先読みは`--image-prefetch`の明示指定で有効化する。compile、decoder、prefetch、入力整合性検証の設定はrun/checkpointに保存する。
CPUはJPEG byte列まで、nvJPEGでRGB uint8へ復号し、モデル内FP32で輝度・MDDを計算する。
OpenCVとnvJPEGには画素差があるため、評価時は保存decoder/API/versionを照合し、変更は明示overrideで記録する。

## 学習済み重みと再開

追加学習は`--stage coordinates --initialize full_model`を使用し、`--manifest`に累積train、
`--validation-manifest`に固定validationのwindow manifest、`--initial-checkpoint`に採用済みの
query posttraining v1、`--initial-checkpoint-sha256`にそのSHA-256を明示する。
`--augmentation-config`・新しい`--output`と学習予算/runtime引数はencoder初期化と共通。
親と新しいtrainのmanifestが異なっていてもよい。validationのclip/source groupがtrainに混入した場合、
またはCommon評価subsetが無い場合はモデル初期化前に停止する。

全パラメータ・bufferの名前/dtype/shape/値を照合し、入力前処理とJPEG decoder契約を継承する。
親checkpointのoptimizer・epoch・step・bestは使わず、AdamWとscheduleをstep 0から開始する。
親ファイルhash・全重みhash・train/validation各manifest hashをrecipe/checkpointに保存し、
両manifestをrunへ複製する。これらの入力はrunの途中では変更できない。

`--resume`は同じrunの最新完了epoch、同じmanifest/recipe/source hashを要求する。
full_modelでは元の初期checkpoint/hashを指定したまま、当該run内のcheckpointを`--resume`に渡す。
新しい親モデル・累積データを使う次サイクルと、同一サイクルの中断再開を区別する。
coordinate v4のconstant LRでは総epochを延長できる。DPT/事後学習のcosineでは元の総予算を維持する。
`--stop-after-epoch`によるDPTの診断停止も元の総予算を変えない。自動で次stageを起動しない。

保存形式はcoordinate v4、DPT v1/v2、query posttraining v1、既存Lightningを明示的に区別する。
ファイルの移動前に開始したrunを厳密に再開する場合は、記録された旧commitを使用する。
現行コードから旧checkpointのモデル読込・評価・encoder転送は可能だが、source hashを偽装して旧runへ追記しない。
現行構成で保存→同構成で再開した結果が連続学習と一致することをintegrationで検証する。

## 評価と可視化

位置が既知のobserved GTだけを座標loss/誤差に使う。不明位置には教師を作らない。
validationは同じclip/frame/FPSを窓中心に近い予測で一度採点し、同距離なら早いstartを使う。
Commonはpose側とGT/split等のidentityが一致するsubset、Fullは全validation。
FPSごとの平均source-pixel誤差を等重み平均し、事後学習のbestはclean Commonで選ぶ。testは使わない。

`evaluation/fixed_validation.py`は採用MDDと候補の学習manifestが異なる場合も、同じ固定validation loader、
decoder・precision・compile設定で比較する。採点するFPS/clip/frame集合を教師maskから事前計算し、
両モデルの実採点集合hashと支持数を照合する。全scope/FPSに教師が必要で、欠測を別scopeで代用しない。
比較artifactには両checkpoint/state hash・各学習manifest hash・固定validation hash・runtime・
Common/Full/source別指標と候補−baseline差を保存する。この比較API自体は採用閾値を持たない。

人工遮蔽でも既知GTは維持する。camera拡張はRGBとGTへ同じ2D affineを適用し、画面外GTを除く。
設定の正本は`configs/augmentation/mdd_posttraining.yaml`。epoch/窓/FPS/seedで変換を固定する。
最終bestにはcamera/occlusion/combinedのstressを別集計し、人工遮蔽点だけの指標を分ける。
これは自然遮蔽・視差・新視点・scene cutの再現ではなく、単一seedの値で因果的優越を主張しない。

DPT/事後学習は`--preview-clips`で各epochのGIFを保存する。描画実装は`visualization/`へ置く。
特定runの予算・GPU復旧判断・監視timer・bestの自動queue投入はこのpackageに持たない。
