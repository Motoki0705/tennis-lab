# BLCS

2D観測とコートkeypointから、コート座標系のボール軌道を推定します。
対応するデータは `data/blcs/single_object`、推定モデルは
`models/blcs_multiview_axial_model.py` の `BLCSMultiViewAxialModel` です。
1対象を複数カメラから観測し、3D位置と任意の速度を出力します。

座標・CourtKP metadataは[共有契約](../base/generate_dataset/README.md)、
正規化は[src/utils/README.md](../../utils/README.md)、配置は[タスク出力規約](../OUTPUTS.md)を参照してください。
このタスクの生成・学習・推論は `physical_v1` のみを受け付けます。

## 構成

| 場所 | 責務 |
|---|---|
| `configuration.py` / `configs/` | axial、single_object、学習・生成の厳密な設定検証 |
| `generate_dataset/` | 物理シミュレーションによる1ボールのラリー、カメラ投影、保存、サンプルGIF |
| `data/` | multiview Dataset、augmentation、固定データとchunk生成のDataModule |
| `models/` | axial本体、必要なhead等の部品、discriminator |
| `model_io/` | バッチ・出力契約、モデルとadapterの構築、checkpoint検証 |
| `training/` | 通常学習・GAN・損失・指標・runner |
| `physics/` | 記録clock・場・イベントの教師契約と共有積分器の短区間呼出し |
| `ablation/` | 固定予算の段階的損失比較、全長保存評価、幾何・GT摂動対照 |
| `inference/` | axial checkpointの厳密な復元と推論 |
| `visualization/` | single_objectの閲覧とGT・推論比較 |
| `scripts/` | 生成・学習・解析・可視化の公開入口 |

multi_object、tracking、axial reference、残差補正モデル、broadcast専用データ、
camera_view_v2専用データは廃止しました。旧設定や別モデルのcheckpointは拒否します。
同じaxial実装のsmall/base/large/xlarge設定と、GAN用discriminatorは維持します。

## データ生成

```bash
.venv/bin/python -m src.tasks.blcs.scripts.generate_dataset
.venv/bin/python -m src.tasks.blcs.scripts.generate_dataset_samples
```

既定出力は `data/blcs/single_object` です。`scenes/`、train/val/testのsplit、
設定とmetadata、閲覧用`samples/`を保持します。生成はCPUの並列workerを使います。
重力は固定で、sceneごとにk_drag・k_magnus・風と、`physics.surface_choices`から均等にcourt surfaceを1つ選びます。
バウンド係数はsurface名から共有の表を引くだけで、設定では直接指定しません。
各sceneは`physics_*.npy`として[物理記録](../../utils/README.md#physics)を持ち、生成時に記録状態の積分が軌道をbit単位で再現することを検証します。
dataset schemaは`blcs_generated_dataset_v3`で、物理記録のない旧datasetは読み込みを拒否します。
サンプル生成もsingle_object・physical_v1に限定します。

## 学習

ローカルGPUでの学習・実験は必ず[training queue](../../../.agents/skills/training-queue/SKILL.md)
を経由します。queueへ渡す学習コマンドは次のとおりです。

```bash
.venv/bin/python -m src.tasks.blcs.scripts.train
.venv/bin/python -m src.tasks.blcs.scripts.train --config-name train_chunked
.venv/bin/python -m src.tasks.blcs.scripts.train --config-name train_chunked_gan
```

固定データとchunk生成の両方が同じaxial・single_object契約を使います。
chunk生成はtrainだけを更新し、val/testは固定splitから読みます。
`training=gan_small` / `gan_base` / `gan_large`でdiscriminatorを選択できます。
checkpointの座標契約・設定・state dictの不一致はエラーになります。

標準lossは従来どおり、通常学習とGANは位置教師、chunked学習は位置教師＋clean再投影です。
新しい速度教師と同じ飛行区間内の位置／速度rollout整合は `loss=physics` で明示的に有効化します。
打球・バウンド等をまたぐstencilとpaddingは除外し、全view欠損frameは教師があれば対象にします。
物理teacherはモデル入力へ渡さず、記録されたdt・substeps・場・spinをloss側だけで使います。
尺度・robust遷移点・重み・warmupは `configs/loss/default.yaml` が正本です。
旧smoothness/gravity prior、終端／閾値指標のキーは廃止しました。

checkpointは `val/trajectory_rmse_m` 最小で選択します。新schemaは
`blcs.physics_metrics.v1`、距離は3軸Euclidean誤差のframe母集団で集約します。
速度headと位置差分、イベント安全な積分残差、clean再投影と支持数を分けて記録し、
支持数0の保存値はnullです。学習中は200更新ごとに各lossの出力勾配L2を記録します。
`model.physics_head=true` は最小drag/風/有効Magnus/event headの比較用で、標準出力はxyz/vです。
通常推論はモデル設定と重みを厳密に復元し、学習時のloss／metrics設定には依存しません。
`physics_head` の記録がない旧checkpointは明示的なmetadata移行が必要です。既存の本番重みは
重みを変えずに移行した `*-inference-v1.ckpt` を使用します（[検証記録](../../../knowledge/nodes/blcs/000069-run-blcs-inference-checkpoint-migration.md)）。
未移行ファイルはUIに拒否理由を表示し、参考評価するときだけ
`src.tasks.ball_detection.scripts.evaluate_blcs_ablation --historical` を明示します。

比較条件・全長評価・保存物の再計算は[実験プロトコル](ablation/README.md)を参照してください。

## 閲覧・推論

起動方法は[Web UIガイド](visualization/README.md)を参照してください。
checkpointからaxialを復元し、single_objectシーンのGTと予測を比較します。

力・バウンド・積分は[共有物理](../../utils/README.md#physics)を使い、ネット／フェンス衝突、ラリー連鎖、着地点サンプリングは`generate_dataset/simulation/`が所有します。
物理proposalの再試行には`generation.maximum_physics_attempts_per_scene`の有限budgetを使用します。
`generate_dataset/api_server/`と`webui/`は物理シミュレータの操作・確認用です。
