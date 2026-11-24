# FastHMR

`recipe=fasthmr` は、同じ `body_view_selection` を入力とする CameraHMR ベースのFastHMRを選ぶ。上流から選手ID・選択カメラ・時刻・cropを受け取り、`body_parameters` v3を既存の配置アルゴリズムへ渡す。既定 `recipe=standard` は `hmr2_features → gvhmr` とv2を使う。

## 公式構成と実行条件

[公式コードの固定版](https://github.com/TaatiTeam/FastHMR/tree/dc5c8daf0beea56a419c631f494ef7bd7ce35d3c)・[公開重みの固定版](https://huggingface.co/SoroushMehraban/FastHMR/tree/898332e1059fc0579f986e03cab11471d4e27b65)を参照する。チェックポイントとマスク用モデルのSHA-256は [assets.py](../../submodules/models/fasthmr/assets.py) にある。

| 構成 | PT32 | PT16 | TRT16 |
| --- | --- | --- | --- |
| HMR2特徴＋GVHMR | #2496で計測 | #2496で計測 | #2496で計測 |
| FastHMR CameraHMR | 公式補助資産の取得後に検証 | 公式補助資産の取得後に検証 | 同等な変換とparityの確認待ち |
| FastHMR HMR 2.0 | 実行不能（公開版のMask-ToMe非対応） | 同左 | 同左 |

公式重みには両backboneのlayer-mergedチェックポイント、専用diffusion decoder、共通VAEがある。公開デモのMask-ToMeパッチはCameraHMR側のBlockとAttentionにだけ適用される。HMR 2.0で同じ公開デモを呼ぶとlayer mergingとdiffusionは動く構成だが、Mask-ToMeは適用されない。TensorRT用には、person maskの有無で分岐するtoken mergingや `scatter_reduce` を含む処理の同等性確認が要る。[デモ](https://github.com/TaatiTeam/FastHMR/blob/dc5c8daf0beea56a419c631f494ef7bd7ce35d3c/demo.py)・[Mask-ToMe](https://github.com/TaatiTeam/FastHMR/blob/dc5c8daf0beea56a419c631f494ef7bd7ce35d3c/model/hmr2_tools/patch_vit.py)・[論文](https://arxiv.org/abs/2510.10868)

## 資産と設定

`fasthmr.repository` は外部資産root相対の公式コードcheckout。`fasthmr.assets` はcheckpoint root相対で、次のファイルを参照する。モデルのロードには公式実装が参照するPython依存も必要になる。

```text
feature_extractors/camerahmr.pth.tr
feature_extractors/cam_model_cleaned.ckpt
feature_extractors/yolo11x-seg.pt
fasthmr_ckpts/camerahmr_diffdecoder.pth.tr
fasthmr_ckpts/vae.pth.tr
body_models/smpl_mean_params.npz
body_models/SMPL_NEUTRAL.pkl
body_models/J_regressor_h36m.npy
```

CameraHMRの `cam_model_cleaned.ckpt`、`smpl_mean_params.npz`、`SMPL_NEUTRAL.pkl` は、同意済みの利用者がCameraHMRのDownloadsから取得する。CameraHMRの資産と該当コードは非商用目的に制限され、再配布には許諾が要る。[CameraHMRの登録](https://camerahmr.is.tue.mpg.de/register.php)・[ライセンス](https://camerahmr.is.tue.mpg.de/license.html)

| 公開資産 | 配布元と利用条件 |
| --- | --- |
| `camerahmr.pth.tr`、`camerahmr_diffdecoder.pth.tr`、`vae.pth.tr` | FastHMRの固定revision。モデルカードはApache-2.0を表示する。固定版コードにはリポジトリ全体のLICENSEが見つからないため、コードは外部checkoutとして参照する。[FastHMRモデルカード](https://huggingface.co/SoroushMehraban/FastHMR/blob/898332e1059fc0579f986e03cab11471d4e27b65/README.md) |
| `yolo11x-seg.pt` | Ultralyticsの固定revision、AGPL-3.0。ローカルの評価用。提供形態を決める本番統合時にはAGPL-3.0の条件またはUltralyticsの商用ライセンスを別に確認する。[配布重み](https://huggingface.co/Ultralytics/YOLO11/blob/8b8ac7d1fae7468f85dbf89670dd66f41f485aab/yolo11x-seg.pt)・[モデルカード](https://huggingface.co/Ultralytics/YOLO11) |
| `J_regressor_h36m.npy` | FastHMRの取得スクリプトが指定するOpenMMLabの版付き公開オブジェクト。配布元にこのファイル単体のライセンス表記は確認できていない。比較用に取得し、Gitには同梱しない。[FastHMRの取得スクリプト](https://github.com/TaatiTeam/FastHMR/blob/dc5c8daf0beea56a419c631f494ef7bd7ce35d3c/download_hmr_dependencies.sh#L19)・[MMHuman3Dの資産説明](https://mmhuman3d.readthedocs.io/en/stable/getting_started.html#body-model-preparation) |

COCO17にはシーン側で設定済みの6890頂点用回帰行列を使う。

公式推論モジュールを動かす追加のPython依存はworktreeの `outputs/fasthmr-setup/python` に置き、実行時に `PYTHONPATH=outputs/fasthmr-setup/python` を渡す。次は検証した追加パッケージとchumpyのビルド手順。既存環境からPyTorch、Ultralytics、NumPyなどを読み込む。

```sh
uv pip install --python .venv/bin/python --target outputs/fasthmr-setup/python --no-deps \
  diffusers==0.37.1 easydict==1.13 yacs==0.1.8 scikit-image==0.25.2 tifffile==2025.6.11 lazy-loader==0.4
uv venv --python .venv/bin/python outputs/fasthmr-setup/chumpy-builder
uv pip install --python outputs/fasthmr-setup/chumpy-builder/bin/python pip==25.3 setuptools==81.0.0 wheel==0.45.1
outputs/fasthmr-setup/chumpy-builder/bin/python -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir outputs/fasthmr-setup/wheels \
  'chumpy @ git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da'
uv pip install --python .venv/bin/python --target outputs/fasthmr-setup/python --no-deps \
  outputs/fasthmr-setup/wheels/chumpy-0.71-py3-none-any.whl
```

取得済み資産を持つ環境で、既存のpipeline CLIまたはdataset用のpipeline overridesに次を追加する。既存の身体復元フラグ `gvhmr.enabled=true` は身体復元段の有効化を表す。

```sh
recipe=fasthmr '+fasthmr={repository:FastHMR,assets:fasthmr,backend:pt32,feature_batch_size:16,seed:0}'
```

PT16は `backend:pt16` で選ぶ。CUDA autocastとYOLOのhalf推論を使い、カメラ変換とHybrIK、SMPL geometryはFP32を使う。PT32/PT16とも実重みとGPUでの検証が残っている。処理済みのFastHMR成果を選択的に読む場合は `+execution.fasthmr=load` を使う。

## 固定入力と出力

- 選手ごとの動画・source frame・選択済み正方形cropをそのまま使う。公式YOLO追跡に相当する入力を固定して、YOLO11xセグメンテーション、HumanFoV、両向きのMask-ToMe特徴、diffusion/VAE、HybrIKを実行する。固定の校正カメラは配置に用い、HumanFoVの入力条件と配置前の対角長による焦点距離換算は公式デモの規約を使う。
- 既存の選手別・カメラ別のリクエスト区間ごとに、最大243サンプルの重複なしの窓へ分ける。末尾は実長で処理する。入力に含まれる補間cropも指定frameで推論され、観測サンプル数とsource frameは元の選択に対応する。区間外はこのステージの出力に含まれない。配置は区間内の推論サンプル間で補間する。
- 保存する姿勢はカメラ座標系（右・下・前方）のrootと23局所関節のaxis-angle。betasは人物内で平均し、translはメートル。配置はnative SMPLの6890頂点、COCO17と骨盤回帰を使ってコート座標へ変換する。SceneResultでは共通21関節・配置済み頂点・有効性を公開し、FastHMRのv3成果が完全なSMPL姿勢を保存する。
- 固定ベンチマークの上流は #2495 の保存成果と照合する。配置前後のメッシュは目視比較に用い、三角測量骨格の残差は幾何診断として扱う。
