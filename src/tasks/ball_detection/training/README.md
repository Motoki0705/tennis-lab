# Ball detection training

公開入口は `src.tasks.ball_detection.scripts.train`。GPU実行は元repo共通のtraining queueを使う。

| stage | 初期化 | 実装と教師 |
|---|---|---|
| `convnext_heatmap` | 保存configで明示 | `convnext_heatmap/`の既存Lightning/GAN。既存heatmap契約 |
| `dpt_pretrain` | ランダム初期化、または同run再開 | `dpt_pretrain/`。CNN＋DPT、observed正例と確認済み不在のheatmap教師 |
| `coordinates` | `--initialize scratch` | `coordinates/scratch.py`。pose32条件/query-only4条件、observed座標、constant LR |
| `coordinates` | `--initialize encoder` | `coordinates/finetune.py`。完了DPTのCNN転送、query-only、拡張、freeze→joint |
| `coordinates` | `--initialize full_model` | `coordinates/full_model.py`。採用query checkpointの全重みを転送し、optimizer/scheduleを初期化。累積trainと固定validationを個別指定 |
| `hybrid` | Hydra `initialization.mode=query` | `hybrid/runner.py`。2段目のencoder/queryを転送し、DPTと両headを新規初期化 |
| `hybrid` | Hydra `initialization.mode=continual` | 採用hybridの全重みを、新しい累積snapshotへweights-only転送 |

encoder/full_modelのrunは`coordinates/joint.py`、座標系のepoch更新は `coordinates/loop.py` と `runtime/execution.py` を共有する。
注釈公開後の共通Learningへの接続は[annotation/](annotation/README.md)。固定intent、工程receipt、比較と採用条件を検証する。
JPEG復号・GPU転送・precision/compileは `runtime/`、LRは `schedules.py`、epoch内FPS混合は `sampling.py`。
段階別runnerが他のrunnerから共通関数をimportしない。評価は `evaluation/`、保存形式は `checkpoints/` が担当する。
窓datasetは同じsampleに`position_valid`と`heatmap_valid`を供給し、各objectiveが使う教師を明示する。

## 共通heatmap objective

`training/heatmap_objective.py` の `heatmap_objective(logits, batch, sigma_ratio=.012, gamma=2.)`
をDPTとheatmap専用hybridで共用する。既存 `dpt_pretrain.objective` は同じ関数の再exportで、
従来DPTの教師・演算順・数値を維持する。存在head付きhybridもこのheatmap項を共用し、
従来どおり存在BCEを別に加える。

logitsはnative /4格子の `(B,T,H,W)`、batchはnormalized `uv: (B,T,2)` と
boolean `position_valid` / `heatmap_valid: (B,T)` を持つ。
position_validはheatmap_validの部分集合で、位置教師は有限の[0,1]座標を要求する。
位置教師には格子対角比sigmaのGaussian、確認済み不在（heatmap_validのみtrue）には全面0を作る。
未知・座標不明・教師領域外はdatasetがheatmap_validをfalseにし、目的関数で除外する。
目的関数は領域・不在ラベルを推測せず、渡されたmaskだけを使う。
空間平均のFocal BCEを求め、heatmap_validなframeだけで平均する。教師・損失はFP32。
返り値はscalar lossと `(B,T)` のper-frame lossで、後者の集計にも必ずmaskを使う。
全不在のbatchは有効、教師0件のbatchは拒否する。

任意Tのモデルと重み専用artifactは [MDDモデル](../models/mdd/README.md#任意長のheatmap専用モデル) を参照。
このモデルAPIは、既存3段学習のrunnerやscene既定モデルを切り替えない。

## 実行例

窓/FPSの仕様と`prepare_data`による凍結手順は[data/windows](../data/windows/README.md)を参照。
snapshotを共有するrunの開始前に、同READMEの[snapshot運用制約](../data/windows/README.md#snapshotを共有するrunの運用)を確認する。
DPT/座標系CLIのartifact/outputは絶対pathで指定する。
DPT/座標系のCLIでは`--precision`・`--jpeg-decoder`・`--compile-mode`・`--input-verification`を必ず指定する。
事前/事後学習のseed・warmup、事後学習のepoch数・窓数・学習率も必須で、指定漏れは学習前に拒否する。

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.train --stage dpt_pretrain \
  --manifest /absolute/mdd_only_windows.json \
  --model-config /absolute/configs/model/mdd_dpt_rgb_mdd.yaml \
  --output /absolute/new-pretrain-run --epochs 10 --windows-per-epoch 6000 \
  --learning-rate 0.0002 --warmup-updates 500 --seed 42 \
  --device cuda --precision bf16 --batch-size 1 --jpeg-decoder nvjpeg --pin-memory \
  --compile-mode default --input-verification upfront

.venv/bin/python -m src.tasks.ball_detection.scripts.train --stage coordinates --initialize encoder \
  --manifest /absolute/mdd_only_windows.json --pretraining-run /absolute/completed-pretrain-run \
  --augmentation-config /absolute/configs/augmentation/mdd_posttraining.yaml \
  --output /absolute/new-posttrain-run --epochs 10 --freeze-epochs 1 \
  --windows-per-epoch 6000 --learning-rate 0.0001 --encoder-lr-ratio 0.1 \
  --warmup-updates 500 --seed 42 --device cuda --precision bf16 --batch-size 1 \
  --jpeg-decoder nvjpeg --pin-memory --compile-mode default --input-verification upfront --stress-evaluation
```

CNN候補は`configs/model/mdd_dpt_{pretrain,convnext_v2,fasternet}.yaml`で選ぶ。
5chの事前学習には `configs/model/mdd_dpt_rgb_mdd.yaml` を指定する。
事後学習は保存された入力契約とencoder構成を引き継ぐ。
入力仕様と旧2chとの互換範囲は [models/mdd](../models/mdd/README.md) を参照。
pose/圧縮比較は`prepare_data`が作るmodel設定を`--stage coordinates --initialize scratch --model-config ...`へ渡す。
CPU smokeでは`--device cpu --precision fp32 --jpeg-decoder opencv --compile-mode off --input-verification lazy --num-workers 0`を明示する。CUDAの事前/事後学習はBF16固定。

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

5chの3段目（hybrid）の構造と初期化APIは [models/mdd](../models/mdd/README.md#3段目への重み転送) にある。
1段目は `mdd_dpt_rgb_mdd.yaml`、2段目はその保存構成を引き継ぐ。
3段目のHydra設定の正本は [`train_mdd_hybrid.yaml`](../configs/train_mdd_hybrid.yaml)。
`???` は要決定値で、省略するとデータ読込・学習artifact作成前に拒否する。
値の候補は同YAMLのコメントだけに置き、コードの既定値にはしない。

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.train --stage hybrid \
  paths.artifact_root=/absolute/snapshots manifest=mdd_only_windows.json \
  paths.output_root=/absolute/training-runs paths.checkpoint_root=/absolute/query-run \
  initialization.mode=query initialization.checkpoint=epoch-NNN.pt \
  epochs=10 windows_per_epoch=6000 learning_rate=0.0001 warmup_updates=500 seed=42 \
  device=cuda runtime.precision=bf16 runtime.jpeg_decoder=nvjpeg \
  runtime.compile_mode=default runtime.input_verification=upfront runtime.pin_memory=true \
  presence_threshold=<要決定> loss.heatmap_weight=<要決定> loss.presence_weight=<要決定> \
  freeze.encoder_epochs=<要決定> freeze.query_epochs=<要決定> \
  'absence.negative_kinds=[<要決定>]' \
  absence.window_weights.play=<要決定> absence.window_weights.absent=<要決定> \
  absence.window_weights.nonplay=<要決定>
```

モデル幅を変えたCPU試験等では `model.*` も指定し、転送元構成と一致させる。
hybridのパスは共通の`paths`契約で解決する。manifestはartifact root、初期化checkpointは
checkpoint root、`resume`はoutput rootからの相対パスで、絶対パスやroot外への逸脱を拒否する。
出力先は`paths.output_root`配下の`run.output_dir`、Hydraログはその`hydra/`配下になる。
既定のrun名は共通の`ball_detection/train/<model>/<run-id>`形式で、再開時は元の`run.output_dir`を指定する。
新規runではHydraログ以外の既存artifactを拒否する。
freezeはepoch単位で独立指定する。0は最初から更新、総epoch数は最後まで固定、
Nはepoch N（0始まり）から解除。固定中はdropoutも停止し、optimizerの対象群のLRを0にする。
query固定中にも、queryを通るencoderの勾配は保持する。DPT・heatmap head・presence headは常に更新する。
損失はnative /4格子の空間平均focal BCEと存在logitのBCEを、同じ教師frameで平均して重み付けする。
不在frameのheatmap教師は全面0。未レビュー・推定・未確定・複数球・reference frameは負例にしない。
3段目では画像増強を行わず、cropや人工遮蔽から新しい不在教師も生成しない。
2段目の既存増強と、3段目の不在供給を区別して比較する。

### 不在教師と窓の混合

3段目は元の凍結snapshotを変更せず、追加窓をrun内の`data_manifest.json`へ凍結する。
`absence.negative_kinds`で`reviewed_empty`（レビュー済み0 instance）と`out_of_frame`
（明示的な画面外1 instance）を選ぶ。選ばないkindは既存play窓内でも不在損失の対象外になる。
位置の分からない不可視球（`unresolved`等）は不在へ読み替えない。

| window source | 供給する窓 |
|---|---|
| `play` | 従来の位置教師数・存在率を満たす窓。選択した不在frameも同じ窓内で教師に使う |
| `absent` | 選ばれた32枚すべてが確認済み不在。元FPSでの途中もreference/PTS gapを跨がない |
| `nonplay` | play候補の外で、既知教師を1枚以上含む残りの窓。非プレー判定だけでは不在ラベルを作らない |

同じ窓は1区分にだけ所属する。各FPSへほぼ等数の予算を配り、その中で
`window_weights`（合計1、0も明記）を最大剰余法で整数窓数にする。
比率は**frame数でなく窓数**。有効な供給源/FPSに窓がない場合や、割当が0になる予算は拒否する。
`train.jsonl`に実割当・教師frame数・各損失・処理速度を保存する。
validationは比率サンプリングせず、生成manifest内の全供給区分を採点する。
`nonplay`を混ぜなくても、その確認済みラベルで誤検出・見逃しを検証できる。

snapshotを新しく作る責任は既存のデータ準備/継続学習workflow側にある。
`notes`や未移植の`confirm_absence`操作を学習側で推測解釈しない。
教師は統一storeの`annotated`・`is_target`・`point_kind`の契約に従う。
ただしChatの空ラベルは、partial注釈のnotesがstoreに残らないため、これだけでは不在の証拠にならない。
manifest生成時にmetadata内の元注釈path/SHA・clip/frame対応を照合し、notes非空の空frameを除外する。
元注釈がない・hashが変わった場合は停止する。検証済みframe maskと注釈hashを生成manifestへ凍結するので、
hybridの読込・評価・同run再開では元注釈を再度開かない。1段目DPTも読込時に同じ検証・除外を行う。

`--resume`は同じrunの最新完了epoch、同じmanifest/recipe/source hashを要求する。
保存済みbestの指標は、学習時と同じrecipeの`selection_scope`で再計算して照合する。
`balanced_existing_new`では既存・新規validationの等重み平均を使い、Commonで代用しない。
最終epoch保存後に完了記録を残す前に停止した場合は、更新を追加せず残りのstress評価と完了記録を再開する。
full_modelでは元の初期checkpoint/hashを指定したまま、当該run内のcheckpointを`--resume`に渡す。
新しい親モデル・累積データを使う次サイクルと、同一サイクルの中断再開を区別する。
coordinate v4のconstant LRでは総epochを延長できる。DPT/事後学習のcosineでは元の総予算を維持する。
`--stop-after-epoch`によるDPTの診断停止も元の総予算を変えない。自動で次stageを起動しない。

保存形式はcoordinate v4、DPT v1/v2、query posttraining v1、既存Lightningを明示的に区別する。
ファイルの移動前に開始したrunを厳密に再開する場合は、記録された旧commitを使用する。
現行コードから旧checkpointのモデル読込・評価・encoder転送は可能だが、source hashを偽装して旧runへ追記しない。
現行構成で保存→同構成で再開した結果が連続学習と一致することをintegrationで検証する。

hybridは`mdd_hybrid_training.v1`、`model_kind=mdd_hybrid`で保存する。
各段階にsnapshotのindex/metadata hash、全clipのsplitとそのhash、manifest、解決済みrecipe、
seed・初期化方針、コードhash、転送元checkpoint hashと祖先receiptを記録する。
3段目には追加窓manifestのhashと`transfer.json`も残す。
queryからの転送は同じ元manifestを要求し、1段目のDPTファイルは開かない。
3段目の保存モデル読込・再開は祖先ファイルを必要としない。

`run.output_dir=<元runの相対パス> resume=<元runの相対パス>/epoch-NNN.pt`は
元run内の最新完了epoch・同じ総予算・同じrecipe/閾値/
freeze/供給源/snapshot/コードを要求し、optimizerとRNGを復元する。
`stop_after_epoch=N`は予算を変えない診断停止。2段目にも`--stop-after-epoch N`がある。
継続学習は新しい`run.output_dir`で`initialization.mode=continual`と採用hybrid checkpointを指定する。
全重みを引き継ぐが、optimizer・step・freeze方針は新規runのものを使う。
snapshotのindex/metadataは変わっている必要があり、既存clipとそのsplitを保持する。
旧clipの注釈更新は新snapshotのhashとして記録される。同snapshotでmodeだけ変えて再開することは拒否する。

hybridの接続APIは`training.hybrid.runner.train_hybrid(DictConfig)`で、累積snapshotのmanifestと
採用checkpointを受け取る。推論は`model_io.mdd_hybrid.load_mdd_hybrid_detector(path)`、
独立した固定holdoutの採点は`evaluation.hybrid.evaluate_hybrid(...)`を使える。
fixtureで新snapshotへの全重み転送・optimizer/stepリセット・追加学習・評価を確認する。
現行の注釈・継続学習adapterはqueryモデル専用で、hybridは未接続。
接続前に必要な設計判断は[annotation/](annotation/README.md#hybridの接続境界)を参照する。
公開snapshot作成・採否・deployはadapter側の責任で、このAPIは自動実行しない。

## 評価と可視化

位置が既知のobserved GTだけを座標loss/誤差に使う。不明位置には教師を作らない。
validationは同じclip/frame/FPSを窓中心に近い予測で一度採点し、同距離なら早いstartを使う。
Commonはpose側とGT/split等のidentityが一致するsubset、Fullは全validation。
full_modelの単独CLI評価は`--mode coordinates --split val --manifest <固定validation manifest>`を使い、保存済み`validation_manifest_sha256`と照合する。累積trainのmanifestでは代用できない。
FPSごとの平均source-pixel誤差を等重み平均する。事後学習のbestは既定ではclean Common、
新旧cohortを持つ固定manifestでは明示した`balanced_existing_new`で選ぶ。選択にtestは使わない。
注釈cycleのreplay比率・新旧評価・最終採否は[annotation/](annotation/README.md)を参照する。

`evaluation/fixed_validation.py`は採用MDDと候補の学習manifestが異なる場合も、同じ固定validation loader、
decoder・precision・compile設定で比較する。採点するFPS/clip/frame集合を教師maskから事前計算し、
両モデルの実採点集合hashと支持数を照合する。全scope/FPSに教師が必要で、欠測を別scopeで代用しない。
比較artifactには両checkpoint/state hash・各学習manifest hash・固定validation hash・runtime・
Common/Full/source別指標と候補−baseline差を保存する。この比較API自体は採用閾値を持たない。

hybridも同じ窓中心優先、FPS別・source別・Full/Common集計を使い、平均・P95と支持数を出す。
位置誤差は存在判定で選別する前のheatmap座標を、全observed GTに対して採点する。
存在判定は保存された閾値を使い、確認済み不在のFP数/FPRとobservedのFN数/FNRを別に報告する。
分母0は`null`。全不在評価では位置指標を`null`として存在指標を返す。
bestは引き続き全observedのmacro平均位置誤差で選ぶため、全不在のvalidationだけでは学習のbestを選べない。
存在率を含めた採否や閾値較正は、別途固定評価とユーザー判断で行う。

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.evaluate --mode hybrid \
  --checkpoint /absolute/hybrid-run/epoch-NNN.pt \
  --manifest /absolute/hybrid-run/data_manifest.json \
  --output /absolute/new-evaluation.json --split val --device cuda
```

CLIは保存されたmanifest・decoder/version・precision・閾値を照合し、
保存されたcompile modeとrecompile limitをモデルのdevice配置後に適用する。
testは明示した単独評価でのみ利用し、学習・checkpoint選択に使わない。

2段目の人工遮蔽でも既知GTは維持する。camera拡張はRGBとGTへ同じ2D affineを適用し、画面外GTを除く。
設定の正本は`configs/augmentation/mdd_posttraining.yaml`。epoch/窓/FPS/seedで変換を固定する。
最終bestにはcamera/occlusion/combinedのstressを別集計し、人工遮蔽点だけの指標を分ける。
これは自然遮蔽・視差・新視点・scene cutの再現ではなく、単一seedの値で因果的優越を主張しない。

DPT/事後学習は`--preview-clips`で各epochのGIFを保存する。描画実装は`visualization/`へ置く。
特定runの予算・GPU復旧判断・監視timer・bestの自動queue投入はこのpackageに持たない。
