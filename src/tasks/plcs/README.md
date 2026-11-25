# PLCS

2D観測とコートkeypointまたは校正済みカメラパラメータから、コート座標系のプレーヤーを推定します。
対応するデータは `data/plcs/single_object`、推定モデルは多視点の
`models/plcs_multiview_axial_model.py`（`PLCSMultiViewAxialModel`）および
単眼 GVHMR型 Early Fusion + HTD-Refine の
`models/plcs_monocular_gvhmr_model.py`（`PLCSMonocularGVHMRModel`）です。
多視点モデルは複数カメラの2D観測から3D位置・yaw・canonical 3D poseを推定し、
単眼モデルは校正済みカメラ内外パラメータ・2D COCO17・合成 `bbx_xys`（HMR2特徴量なし）から
物理コート座標の3D COCO17骨格および PVA-Heads（速度・加速度・静的接地）を推定します。

座標・CourtKP metadataは[共有契約](../base/generate_dataset/README.md)、
正規化は[src/utils/README.md](../../utils/README.md)、配置は[タスク出力規約](../OUTPUTS.md)を参照してください。
このタスクの生成・学習・推論は `physical_v1` のみを受け付けます。

## 構成

| 場所 | 責務 |
|---|---|
| `configuration.py` / `configs/` | axial・単眼GVHMR、single_object、学習・生成の厳密な設定検証 |
| `generate_dataset/` | ACCAD/AMASSのSMPL-Hモーションによる1人の動作、fixed/broadcast/mixedカメラ投影、保存、サンプルGIF |
| `geometry/` | 単眼合成正方形bbox（`bbx_xys`）生成・摂動、CLIFF特徴、カメラ内外パラメータ特徴化、コート3D COCO17変換 |
| `data/` | multiview / monocular Dataset、augmentation（校正誤差摂動を含む）、固定データとchunk生成のDataModule |
| `models/` | axial本体、単眼GVHMR型Early Fusion + PVA-Heads本体、必要なhead等の部品、discriminator |
| `model_io/` | バッチ・出力契約、モデルとadapterの構築、checkpoint検証 |
| `training/` | 通常学習・GAN・損失（単眼/PVA高次時間動態含む）・指標・runner |
| `inference/` | axial・単眼GVHMR checkpointの厳密な復元・推論、HTD-Refine Stage B 軌道最適化（`PLCSHTDRefiner`） |
| `visualization/` | single_objectの閲覧とGT・推論比較 |
| `scripts/` | 生成・学習・解析・可視化の公開入口 |

multi_object、tracking、axial reference、残差補正モデル、broadcast専用データ、
camera_view_v2専用データは廃止しました。旧設定や別モデルのcheckpointは拒否します。
同じaxial実装のsmall/base/large/xlarge設定と、GAN用discriminatorは維持します。

## 外部資産

SMPL-Hは `paths.external_asset_root=ckpt` 配下の `body_models/smplh` を読みます。
学習のresume/init_weightsを解決する `paths.checkpoint_root=outputs` とは独立しています。
生成・chunked学習のCOCO17 regressorも同じ外部asset root配下の
`body_models/gvhmr/smpl_coco17_J_regressor.pt` を参照します。
配置方法は[SMPL補助資産の取得](../../submodules/README.md#smpl補助資産の取得)を参照してください。

## データセットの体系

現行の学習データ系列は **`single_object` / `physical_v1` のみ**です。
ACCADの動作を物理コートへ配置し、保存カメラへ投影した合成データです。
RGB映像、テニス選手の実測3D教師、人手の2Dアノテーションは含みません。

| 配置 | 役割 | 入力・教師と単位 |
|---|---|---|
| `data/ACCAD/**/*.npz` | 元動作（AMASS/SMPL-H）。学習シーンの生成素材 | `poses`、`trans`、`betas`、gender、fps。生成時にCOCO17へ変換しコートへ配置 |
| `data/plcs/single_object/scenes/<scene>/` | 学習・検品する生成シーン | カメラ別COCO17/CourtKP20の正規化UV・vis、正規化root位置、yawのcos/sin、世界COCO17[m] |
| `data/plcs/single_object/{train,val,test}.txt` | scene IDの分割 | 保存splitを読み、未割当・重複・同一元動作のsplit共有を確認する |
| `data/plcs/single_object/samples/` | シーンに付属する閲覧用GIFと選定manifest | 冒頭の共有契約にあるサンプル仕様に従う補助成果物 |

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
# 多視点 axial（既定）
.venv/bin/python -m src.tasks.plcs.scripts.train
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_chunked
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_chunked_gan

# 単眼 GVHMR型 Early Fusion（Baseline / HTD-Refine Stage A PVA-Heads）
.venv/bin/python -m src.tasks.plcs.scripts.train \
  model=monocular_gvhmr_base data=monocular_sequence loss=monocular_base
.venv/bin/python -m src.tasks.plcs.scripts.train \
  model=monocular_gvhmr_htd data=monocular_sequence loss=monocular_htd
```

固定データとchunk生成の両方が `single_object` / `physical_v1` 契約を使います。
単眼モデルではモデル構成と損失の組合わせを学習開始前に厳密検証し、`monocular_gvhmr_htd` に対して
PVA損失が有効でない設定（例: 既定の `loss=canonical`）はエラーとして拒否します。
chunk生成はtrainだけを更新し、val/testは固定splitから読みます。
`training=gan_small` / `gan_base` / `gan_large`でdiscriminatorを選択できます。
checkpointの座標契約・設定・state dictの不一致はエラーになります。

## 骨格損失と退化頻度

幾何の有効domain・結果型・しきい値は
[共有骨格契約](../../utils/README.md#skeleton-geometry-contract)を参照してください。
static 4 termと角速度3 termは、padding frameを幾何計算前に除外します。
targetの退化は欠測として特徴量ごとに除外し、predの退化はtargetの有無にかかわらず例外です。
各frame内の有効特徴の平均を取り、観測のあるframe間で平均します。
角速度では隣接する両端の特徴maskをANDし、paddingや欠測を跨いで差分を取りません。
有効な特徴・遷移がterm全体に1つもなければ例外です。角速度には2 frame以上が必要です。
全有効入力の損失数値と既存の重み・角速度の特徴別重みのscaleは維持します。

weight=0のtermは計算せず、`PLCSLoss`の結果にも含めません。
`analyze_loss_dominance`も有効termのみを報告します。raw lossを調べるtermは、
設定・解析overrideで明示的に有効にしてください。
`analyze_angle_velocity`は特徴ごとの有効遷移を分母に使い、joint/torsionの
`sample_count`を特徴順の配列で保存します。観測0のchannelは統計・重みをJSONのnullにし、
family全体が観測0なら例外にします。nullを含む重みは学習設定へそのまま転記できません。

退化頻度はCPU CLIで教師・学習済み予測を別集計できます。全pathは絶対pathで指定します。

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m src.tasks.plcs.scripts.analysis.analyze_skeleton_validity \
  --target-scenes /absolute/data/plcs/single_object/scenes \
  --prediction /absolute/exports/trained_predictions.npz \
  --length-thresholds-m 0 1e-6 1e-4 --plane-thresholds 0 1e-6 1e-3 \
  --output /absolute/reports/skeleton_validity.json
```

`--target-scenes`は保存された`human_kp_3d.npy`を再帰走査します。保存sceneには
collatorのpaddingがないため全frameを観測とします。元ACCAD/AMASSのSMPL-H配列を直接渡さず、
既存生成経路でCOCO17[m]へ変換した教師を使ってください。
補間・augmentation後の実際の学習教師は`--target`でNPZを指定します。
`--target` / `--prediction`は複数回指定でき、各NPZは次の2配列だけを持ちます。

- `pose_m`: float32/64の`(T,17,3)`または`(B,T,17,3)`。教師はworld/canonicalのCOCO17[m]、
  予測は学習済みcheckpointから得た`outputs.canonical_pose`。court正規化座標は不可。
- `padding_mask`: bool、True=padding。poseのframe軸と同shape。
  batchの`(B,V,T)`も受け付け、全viewがpaddingのframeを除外します。

推論の各batchで、例えば
`np.savez(path, pose_m=outputs.canonical_pose.detach().cpu().numpy(), padding_mask=batch["padding_mask"].cpu().numpy())`
としてshardを保存します。教師は`pose_m=batch["human_kp_3d"].cpu().numpy()`です。
checkpoint・dataset split・seed・augmentation条件をexportと一緒に記録し、CPU/GPUの
推論環境と入力資産を準備してください。このCLI自体は推論・学習を起動しません。

出力はsource別のpadding件数、非padding件数、骨/面/torso軸ごとの候補しきい値以下の
件数・割合、少なくとも1特徴が該当するframeの件数・割合、特徴順、入力fileと処理秒数です。
面は正規化crossで判定し、zero骨/軸から作れない面の診断値は0とします。
したがって短い骨と無効面の理由別件数は重なり得ます。しきい値0でexact zeroも数えます。
非paddingの非有限値や全padding入力は例外です。
保存sceneはmmapと4096 frame単位で走査し、NPZは1 fileずつ展開するため、
大きい予測はbatch単位のshardに分けてください。

## 閲覧・推論

起動方法は[Web UIガイド](visualization/README.md)を参照してください。
データセット閲覧はcheckpointなしで保存入力と合成教師を確認します。
推論UI・`PLCSPredictor.predict_scene()` は保存シーンの `camera.params` と checkpoint から
axial および単眼 GVHMR モデルを復元して推論でき、単眼系列推論の公開入口として
`PLCSPredictor.predict_monocular_sequence(..., htd_refiner=PLCSHTDRefiner(...))` を提供します。

`generate_dataset/sampling/`はACCADを読み、`motion/`はCOCO-17変換とコートへの配置を所有します。
SMPL-HとCOCO-17 regressorは明示した外部assetを使用します。GVHMRモーション抽出・混合は廃止しました。
ACCAD閲覧UIのSMPL-H読込は`motion/smplh_model.py`に配置しています。
