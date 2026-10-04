# PLCS

2D観測とコートkeypointから、コート座標系のプレーヤーを推定します。
対応するデータは `data/plcs/single_object`、推定モデルは
`models/plcs_multiview_axial_model.py` の `PLCSMultiViewAxialModel` です。
1対象を複数カメラから観測し、3D位置・yawと任意のcanonical 3D poseを出力します。

座標・CourtKP metadataは[共有契約](../base/generate_dataset/README.md)、
正規化は[src/utils/README.md](../../utils/README.md)、配置は[タスク出力規約](../OUTPUTS.md)を参照してください。
このタスクの生成・学習・推論は `physical_v1` のみを受け付けます。

## 構成

| 場所 | 責務 |
|---|---|
| `configuration.py` / `configs/` | axial、single_object、学習・生成の厳密な設定検証 |
| `generate_dataset/` | ACCAD/AMASSのSMPL-Hモーションによる1人の動作、カメラ投影、保存、サンプルGIF |
| `data/` | multiview Dataset、augmentation、固定データとchunk生成のDataModule |
| `models/` | axial本体、必要なhead等の部品、discriminator |
| `model_io/` | バッチ・出力契約、モデルとadapterの構築、checkpoint検証 |
| `training/` | 通常学習・GAN・損失・指標・runner |
| `inference/` | axial checkpointの厳密な復元と推論 |
| `visualization/` | single_objectの閲覧とGT・推論比較 |
| `scripts/` | 生成・学習・解析・可視化の公開入口 |

multi_object、tracking、axial reference、残差補正モデル、broadcast専用データ、
camera_view_v2専用データは廃止しました。旧設定や別モデルのcheckpointは拒否します。
同じaxial実装のsmall/base/large/xlarge設定と、GAN用discriminatorは維持します。

## 外部資産

SMPL-Hは `paths.external_asset_root=ckpt` 配下の `body_models/smplh` を読みます。
学習のresume/init_weightsを解決する `paths.checkpoint_root=outputs` とは独立しています。
chunked生成のCOCO17 regressorはvendoredの静的資産を参照します。

## データセットの体系

現行の学習データ系列は **`single_object` / `physical_v1` のみ**です。
ACCADの動作を物理コートへ配置し、保存カメラへ投影した合成データです。
RGB映像、テニス選手の実測3D教師、人手の2Dアノテーションは含みません。

| 配置 | 役割 | 入力・教師と単位 |
|---|---|---|
| `data/ACCAD/**/*.npz` | 元動作（AMASS/SMPL-H）。学習シーンの生成素材 | `poses`、`trans`、`betas`、gender、fps。生成時にCOCO17へ変換しコートへ配置 |
| `data/plcs/single_object/scenes/<scene>/` | 学習・検品する生成シーン | カメラ別COCO17/CourtKP20の正規化UV・vis、正規化root位置、yawのcos/sin、世界COCO17[m] |
| `data/plcs/single_object/{train,val,test}.txt` | scene IDの分割 | 保存splitを読み、未割当・重複・同一元動作のsplit共有を確認する |
| `data/plcs/single_object/samples/` | シーンに付属する閲覧用GIFと選定manifest | [共有サンプル仕様](../base/generate_dataset/README.md#human-readable-dataset-samples)に従う補助成果物 |

`train_chunked` / `train_chunked_gan` は同じsingle_object契約のtrainシーンを
逐次生成する供給方式です。独立したデータ系列ではなく、val/testは固定splitを使います。
設定・元動作は`configs/data/`、`configs/motion_sources/accad.yaml`を参照してください。

保存データと実際の学習入力は区別します。現行の固定・chunked data profileは
`num_court_kp=14`で、保存CourtKP20の先頭14点を切り出します。
COCO17・CourtKPともvis=0のUVを0化し、選択したcamera・時間窓へcropして
augmentationを適用します。レビューUIはこの前段の保存観測を表示します。

scene内の`position.npy`は共有のコート正規化契約に従います。レビューUIのroot位置は
`denormalize_court_position()`でメートルへ復元し、COCO17のhip中心とは区別して表示します。
`human_kp_3d.npy`は物理コート座標のCOCO17です。学習のcanonical教師は
この世界COCO17とroot/yawから構成します。保存された`canonical_pose_3d.npy`の関節数は
検品画面にそのまま表示し、旧シーンに残るSMPL-H由来の表現をCOCO17と取り違えません。

splitの独立性を元動作単位で確保したい場合、生成時に`run.split_group=motion_source`を指定します。
保存済みのscene単位splitをUIが書き換えることはありません。
データセット検品の起動・見方は[Web UIガイド](visualization/README.md#データセット閲覧)へまとめています。

## データ生成

```bash
.venv/bin/python -m src.tasks.plcs.scripts.generate_dataset
.venv/bin/python -m src.tasks.plcs.scripts.generate_dataset_samples
```

既定出力は `data/plcs/single_object` です。`scenes/`、train/val/testのsplit、
設定とmetadata、閲覧用`samples/`を保持します。生成はCPUの並列workerを使います。
サンプル生成もsingle_object・physical_v1に限定します。

## 学習

ローカルGPUでの学習・実験は必ず[training queue](../../../.agents/skills/training-queue/SKILL.md)
を経由します。queueへ渡す学習コマンドは次のとおりです。

```bash
.venv/bin/python -m src.tasks.plcs.scripts.train
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_chunked
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_chunked_gan
```

固定データとchunk生成の両方が同じaxial・single_object契約を使います。
chunk生成はtrainだけを更新し、val/testは固定splitから読みます。
`training=gan_small` / `gan_base` / `gan_large`でdiscriminatorを選択できます。
checkpointの座標契約・設定・state dictの不一致はエラーになります。

## 閲覧・推論

起動方法は[Web UIガイド](visualization/README.md)を参照してください。
データセット閲覧はcheckpointなしで保存入力と合成教師を確認します。
推論UIはcheckpointからaxialを復元し、single_objectシーンの教師と予測を比較します。

`generate_dataset/sampling/`はACCADを読み、`motion/`はCOCO-17変換とコートへの配置を所有します。
SMPL-HとCOCO-17 regressorは明示した外部assetを使用します。GVHMRモーション抽出・混合は廃止しました。
ACCAD閲覧UIのSMPL-H読込は`motion/smplh_model.py`に配置しています。
