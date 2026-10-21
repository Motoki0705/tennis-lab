# 標準マルチカメラpipelineの導入 / Installation

同期済み固定カメラ3〜5台の動画からSceneResult v2を生成します。ここが環境構築・資産配置・
公開サンプル実行の正本です。入力の制約と保存形式は[tennis_scene](../../src/tennis_scene/README.md)、
利用条件・帰属表示は[LICENSES.md](../../LICENSES.md)を参照してください。

This is the canonical setup guide for the standard multiview pipeline. It uses uv,
local NVIDIA CUDA, user-acquired third-party models and the published three-camera sample.
Obtain each asset under its upstream terms before running the commands below.

## 動作環境 / Requirements

- Linux x86-64、またはNVIDIA GPUが利用できるWSL2。Python 3.11は`.python-version`で固定し、uvで同期します。
- NVIDIA GPU。確認済みは **RTX 5060 Ti / VRAM 16 GB**。独立clone・新venvでの公開サンプル実行中、
  `nvidia-smi`による約5秒間隔の測定最大は**GPU全体で4,928 MiB**でした。
  開始前の使用量1,215 MiB（他プロセス）を含み、冒頭約22秒とサンプル間の瞬間的なピークは測れていません。
  より小さいVRAMでの動作下限は未検証です。ホストRAM上限24 GBの既存WSL2ホストを使用し、RAMピークは未測定です。
  GPUは他の処理と共有せず実行してください。
- **CUDA toolkit 13系の`nvcc`**と対応ドライバ。lockは`torch 2.14.1+cu130`です。
  検証構成はtoolkit **13.0.88**、driver **610.88**。CUDA 13.xのdriver最低系列は580、
  CUDA 13.0 GAのLinux driverは580.65.06以降です（[NVIDIAの固定版release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-driver)）。
  `nvidia-smi`の「CUDA Version」はtoolkitのインストール確認にはなりません。`nvcc --version`も確認します。
  torchとtoolkitのCUDAメジャー版が違うと拡張ビルドが停止します。13.0を使うとlockのcu130と揃います。
- Git、[uv](https://docs.astral.sh/uv/getting-started/installation/)、C++ compiler、CUDA対応のhost compiler、
  `ffmpeg`（MP4保存）、Ninja。Ubuntuでは`build-essential`・`ninja-build`・`ffmpeg`をOS側に用意します。
- ディスクは、独立clone・venv・分離したuv/HF cache・重み・サンプル・CUDA build・component cache・
  scene・全編可視化を含むworkspaceで**約20.87 GB（19.43 GiB）**を使用しました（準備完了時は約19.94 GB）。
  CUDA toolkit・OSのパッケージ・第三者資産を別途ダウンロードする作業領域は含みません。
  必要容量はcacheや実行回数で変わるため、それらと追加の出力を見込んで空きを確保してください。
  測定範囲は[導入検証記録](../../knowledge/nodes/tennis_scene/000030-run-clean-standard-install-20261021.md)を参照してください。

CPU実行はこの手順の対象外です。CPUで行うのは資産検査・拡張のimport/dispatch検査・可視化です。
The tested GPU has 16 GB VRAM. Device-wide samples reached 4,928 MiB, including other processes;
sampling misses brief peaks and the first 22 seconds, so this does not establish minimum VRAM.
Use a CUDA 13 toolkit matching the locked cu130 PyTorch build.

## cloneと依存関係 / Clone and sync

```bash
git clone https://github.com/Motoki0705/tennis-lab.git
cd tennis-lab
uv sync --locked
git submodule update --init third_party/DINO third_party/dinov3
```

`uv sync --locked`はdev・gpu groupを含む新しい`.venv`を作ります。
`third_party/GVHMR`と`third_party/nht`は標準pipelineには不要です。
submoduleの固定版はこのcheckoutのgitlinkを使い、`--remote`を付けません。
以後のコマンドはrepo rootから実行します。

<a id="dino-cuda-extension"></a>
## DINO CUDA拡張 / DINO CUDA extension

GPUのcompute capabilityを[NVIDIA一覧](https://developer.nvidia.com/cuda-gpus)で確認し、
`TORCH_CUDA_ARCH_LIST`を明示してください。次の**12.0はRTX 5060 Tiの例**です。
別GPUではそのGPUとCUDA 13が対応する値へ置き換えます。

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=2
"$CUDA_HOME/bin/nvcc" --version
nvidia-smi
bash scripts/build_dino_extension.sh "$PWD" "$PWD/outputs/dino_extension"
export PYTHONPATH="$PWD/outputs/dino_extension/lib"
CUDA_VISIBLE_DEVICES='' .venv/bin/python -c \
  'from src.submodules.models import validate_dino_extension; print(validate_dino_extension())'
```

環境によって`CUDA_HOME=/usr/local/cuda`などに変更します。`PYTHONPATH`は同じshellでの
preflightとpipeline実行にも必要です。別shellでは再設定してください。

スクリプトは`TENNIS_LAB_DINO_OPS_BUILD_CONFIG`を正規のpath契約で作り、ルート`setup.py`から
DINOと共通CUDA拡張をビルドします。`outputs/dino_extension/{lib,temp,cache}`と`build.json`へ
生成し、submodule・共有venvのソースを変更しません。互換修正は生成したDINOソースだけに適用します。
ビルド中は`CUDA_VISIBLE_DEVICES=''`とし、GPUを使いません。`validate_dino_extension()`も
CPU tensorでforward/backwardのdispatchを検査するもので、CUDA kernelの数値検証ではありません。
異なるtorchへ同期し直した場合は、新しい出力先へ再ビルドし、その`lib`を明示してください。

The build script requires explicit toolkit and GPU architecture settings. Keep the generated
`lib` directory on `PYTHONPATH`. CPU validation checks binary compatibility; GPU inference is a later step.

## 必要資産 / Required assets

配置はすべてrepo内の**実ファイル**にします。checkpoint root外へのsymlinkはpath契約で拒否されます。
表のSHA-256は公開標準経路の検証対象であり、再圧縮・変換・別リリースへの自動切替はしません。
ライセンスの詳しい適用範囲は[第三者重み](../../LICENSES.md#第三者の重みbody-model資産)と
[公開物](../../LICENSES.md#公開物自作の重みサンプルdatasetのライセンス)を正とします。

<!-- checkpoint-assets:start -->
| 資産 | 配置先（`ckpt/`相対） | SHA-256 | 入手先 | ライセンス・条件 |
| --- | --- | --- | --- | --- |
| court | `court_detection/multiscale_depth3/b863df1f01f0.ckpt` | `bca48cdbdf0b26847b3f650cc4249397b2f7c0a963ba4801b7cd111c6e81522c` | [配布元](https://huggingface.co/Motoki098/tennis-lab-models) | DINOv3 License |
| ball | `ball_detection/i935-mixed-ft-s42-epoch09.ckpt` | `489bb90fb6bb4deab81e507b17e698fbe40ee945533196e78d6c69e4ee3f5828` | [配布元](https://huggingface.co/Motoki098/tennis-lab-models) | CC BY-NC 4.0 |
| detector | `dino/checkpoint0029_4scale_swin.pth` | `e61688afe3af91b25955e9f9601d04b42068327a2e076ee871828089aa4ffed5` | [配布元](https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_) | DINO / Apache-2.0¹ |
| vitpose | `vitpose/vitpose-h-multi-coco.pth` | `50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc` | [配布元](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) | ViTPose / Apache-2.0¹ |
| hmr2 | `hmr2/epoch=10-step=25000.ckpt` | `2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e` | [配布元](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) | HMR2 / MIT¹; SMPL条件は別 |
| gvhmr | `gvhmr/gvhmr_siga24_release.ckpt` | `4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91` | [配布元](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) | GVHMR非商用条件 |
| body_model | `body_models/smplx/SMPLX_NEUTRAL.npz` | `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992` | [配布元](https://smpl-x.is.tue.mpg.de/) | SMPL-X Model License / 登録・同意 |
| aflink | `person_tracking/AFLink_epoch20.pth` | `b35cbeddd3acc48fece820bd640640e6bfb1f5fbf570aa79af26c6a38958daa4` | [配布元](https://drive.google.com/file/d/1DFMUkL-dc-j8-fibcJIq-46Xoq_bFoO9/view) | AFLink重みの条件は未確認 |
| tracking_encoder | `player_association/person_vit_clip_reid.pth` | `bedc8fcc37f296045df19d901b162dd032bfa85d9b4f2406eff8313536d444dc` | [配布元](https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view) | CLIP-ReID / MIT¹ |
| smpl_faces | `body_models/smplh/neutral/model.npz` | `a60a7e29d33f09ef1a6352907e2f485e161f471330ca40592764374063e751df` | [配布元](https://mano.is.tue.mpg.de/) | MANO / SMPL+H License / 登録・同意 |
| dinov3 | `dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c` | [配布元](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) | DINOv3 License / 申請 |
| smpl_hmr2_mean_params | `body_models/gvhmr/smpl_mean_params.npz` | `6fd6dd687800da946d0a0492383f973b92ec20f166a0b829775882868c35fcdd` | [配布元](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/network/hmr2/configs/smpl_mean_params.npz) | GVHMR・SMPL・SMPL-X条件 |
| smpl_smplx_to_smpl | `body_models/gvhmr/smplx2smpl_sparse.pt` | `0fc821a9e79ec3e76d6a9796b96d5bef8cd67055e18497bbe370d8aed9e07e06` | [配布元](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smplx2smpl_sparse.pt) | GVHMR・SMPL・SMPL-X条件 |
| smpl_smpl_coco17_regressor | `body_models/gvhmr/smpl_coco17_J_regressor.pt` | `bacdaf756629493994cc869f4c27d179f5e4a5d06b8797ee3dcb94571522079f` | [配布元](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt) | GVHMR・SMPL・SMPL-X条件 |
| smpl_smplx_verts437 | `body_models/gvhmr/smplx_verts437.pt` | `ef0ea64c470a1fea80adb5e4b866c78a792a04f5f98744c9367b15e57bfb4a4d` | [配布元](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smplx_verts437.pt) | GVHMR・SMPL・SMPL-X条件 |
| smpl_smpl_neutral_joint_regressor | `body_models/gvhmr/smpl_neutral_J_regressor.pt` | `70e3213bd30fe8d8ce37b54675282745e406f915a51511a003aeff99b6da04cf` | [配布元](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt) | GVHMR・SMPL・SMPL-X条件 |
<!-- checkpoint-assets:end -->

¹ repo側のコードのライセンス表示です。コードの許諾が、学習元データ・body modelや別配布重みの
全権利を一括して許諾するとは扱いません。AFLinkは利用条件未確認のまま標準経路で継続使用する判断です。
根拠と制約は[StrongSORT NOTICE](../../src/tasks/person_tracking/strongsort_NOTICE.md)にあります。

### 自作重みと公開サンプル

[重み取得CLI](../../src/utils/README.md#published-detector-weights)と
[サンプル取得CLI](../../src/utils/README.md#published-multiview-sample)を使います。
既存ファイルは検査し、不一致を上書きしません。

```bash
.venv/bin/python -m src.utils.model_weights --checkpoint-root "$PWD/ckpt"
.venv/bin/python -m src.utils.sample_dataset --data-root "$PWD/data"
```

サンプルは`Motoki098/tennis-lab-multiview-sample`の`video_000/clip_000`、
1920×1080・59.94006 fps・1,010フレーム・3カメラです。cam0の拡大・letterboxを含みます。
同期・出自は配布先のdataset cardを参照してください。利用条件は**CC BY-NC-ND 4.0**です。
動画とmanifest・LICENSEは以下へ配置され、同CLIとpreflightがSHA-256を検査します。

<!-- sample-assets:start -->
| 配置先 | SHA-256 | 入手先・条件 |
| --- | --- | --- |
| `data/samples/cam0.mp4` | `f99f0024067da49805654d65eae848407c47ee710411d91892f0921fe8331f39` | [公開Dataset](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/cam1.mp4` | `ac333c334b81141ee72098be6c95f5a6f6e1ccc5faf6e058f6fd3491cc1ac96e` | [公開Dataset](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/cam2.mp4` | `fd82f0b573e9f8788cc56567dabdaf0c0c3775fd5dbede4835707277ef70c461` | [公開Dataset](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/clip.json` | `410d77571f553123ffd57ebb457f45f4798f57d7ab6a0828e70b0a47ec42eaf2` | [公開Dataset](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/LICENSE` | `38762e3777f4ec00a6f769062a7c3f704fb78ce08303ecff88558da4c49cf9ea` | [公開Dataset](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
<!-- sample-assets:end -->

### 第三者の重み・body model

- **DINO**: [公式model zoo](https://github.com/IDEA-Research/DINO/tree/d84a491d41898b3befd8294d1cf2614661fc0953#model-zoo)
  の4-scale Swin-LのDriveから`checkpoint0029_4scale_swin.pth`を取得します。学習用Swin初期重みは不要です。
- **ViTPose-H・HMR2・GVHMR**: [GVHMR公式INSTALL](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/docs/INSTALL.md)
  が案内する[公式Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD)から表の3ファイルを取得します。
  GVHMR用のconda環境やYOLO・DPVOは不要です。
- **CLIP-ReID**: [公式README](https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/README.md#trained-models-and-test-logs)
  の**ViT-CLIP-ReID / Market / model**を選びます。ファイル名は`Market1501_clipreid_ViT-B-16_60.pth`です。
  内容を変えず`ckpt/player_association/person_vit_clip_reid.pth`へ配置します。
  公式配布と従来使用版の同一性は[検証記録](../../knowledge/nodes/tennis_scene/000030-run-clean-standard-install-20261021.md)にあります。
- **AFLink**: [StrongSORT公式README](https://github.com/dyhBUPT/StrongSORT#datamodel-preparation)が案内するDriveの
  `AFLink_epoch20.pth`を取得します。条件の未確認は上記NOTICEのとおりです。
- **DINOv3 ViT-B/16**: [Meta公式](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)に申請し、
  `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`を取得します。
  公開コートcheckpointはbackboneを含みますが、**現行のモデル構築にはこの単体ファイルも必要**です。
- **SMPL-X**: [公式サイト](https://smpl-x.is.tue.mpg.de/)で登録・同意し、NPZ形式のbody modelから
  `SMPLX_NEUTRAL.npz`を配置します。SMPL-Xのmale/female、SMPL本体、SMPL-HのPKLはこの標準実行には不要です。
- **SMPL-H**: [MANO / SMPL+H](https://mano.is.tue.mpg.de/)で登録・同意し、AMASS向けのExtended SMPL+Hモデルを取得します。
  展開した`neutral/model.npz`を表の場所へ置きます。可視化でmeshのfacesを読みます。PKLをNPZへ変換しません。
- **SMPL補助資産5点**: 取得・検査・配置は[src/submodulesの正本](../../src/submodules/README.md#smpl補助資産の取得)に従います。
  GVHMR公式の固定commitから取得し、`install_smpl_assets`で表の場所へ配置します。この導入ガイドでは手順を複製しません。

Google Driveのfolderは中の該当ファイルだけをダウンロードして表の名前へ配置してください。
登録・申請が必要なモデルをこのrepoから再配布したり、別のミラーで補ったりしません。

## 一括確認 / Preflight

```bash
export PYTHONPATH="$PWD/outputs/dino_extension/lib"
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m src.tennis_scene.scripts.preflight
```

`status: ok`と終了値0が必要です。欠損・SHA不一致・読取失敗・拡張の不適合を**全件まとめて**
JSONへ出力し、1件でもあれば終了値1になります。入手したファイルを明示的に直し、再実行してください。
`enabled_model_assets()`の全資産に加え、DINOとDINOv3 source、DINOv3単体重み、可視化のSMPL-H・regressor、
入力動画（公開サンプルならmanifest・LICENSEも）を検査します。モデルの推論・GPU使用は行いません。
ソースは必要ファイルの存在、既知の重みとサンプルは固定SHAで検証します。
別モデルの重みを評価する汎用コマンドではなく、上表の公開標準経路を導入するための検査です。

With assets missing, preflight lists all failures and exits nonzero. It does not download,
substitute models, or run inference. Resolve every failure before starting the pipeline.

## サンプルで3D再構成 / Run the sample

```bash
PYTHONPATH="$PWD/outputs/dino_extension/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m src.tennis_scene.scripts.run_pipeline \
  output_directory=tennis_scene/generate/clean_install output_name=scene \
  cache.directory=tennis_scene/cache/clean_install
```

このrepoの共有GPU上では[training queue](../../.agents/skills/training-queue/SKILL.md)から実行します。
RTX 5060 Tiでのクリーン導入検証では、pipeline開始からscene保存まで**30分27秒**でした
（[実測記録](../../knowledge/nodes/tennis_scene/000030-run-clean-standard-install-20261021.md)）。
公開サンプル1clipの参考値で、queue待ち・プロセス起動・資産取得・ビルド・可視化は含みません。

`outputs/tennis_scene/generate/clean_install/scene.npz`と`scene.metadata.json`が保存され、
metadataの`scene_schema_version=2`・`status=ok`であることを確認します。
正規readerでのmask・理由コード・座標系の検査も必要です。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
from pathlib import Path
from src.tennis_scene.archive import load_scene_result
scene = load_scene_result(Path('outputs/tennis_scene/generate/clean_install/scene.npz'))
assert scene.metadata['scene_schema_version'] == 2 and scene.metadata['status'] == 'ok'
assert scene.num_frames == 1010
print(scene.num_frames, scene.fps, scene.metadata['camera_ids'])
PY
```

出力cacheはこの入力・設定・コードに対するものです。新しい導入検証では新規cloneと空の出力先を使います。
完走は処理・保存契約の成立を示し、真値に対する3D精度の保証にはしません。
同じ入力・推論コードの先行実行とは、有効なball 3Dや選手root/SMPLのフレーム数に差がありました。
原因は未確定であり、同一byteの出力を保証する検証ではありません。詳細は上記の実測記録を参照してください。

## 可視化 / Visualize

`ffmpeg -version`が成功することを確かめ、同じcheckpoint rootのSMPL-H・regressorを使って描画します。

```bash
CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg .venv/bin/python -m src.tennis_scene.scripts.visualization \
  input=tennis_scene/generate/clean_install/scene.npz \
  output=tennis_scene/visualize/clean_install/scene.mp4 \
  style.player_representation=smpl
```

MP4を再生して、コート・両選手・ボールと欠測区間を確認します。
クリーン導入検証のCPU描画は**6分3秒**（親セッションの計時、起動時間込み）で、
1200×800・1,010フレーム・約16.85秒のMP4を生成しました。親セッションが9秒時点のフレームで
コート・両選手のSMPL・ボール・ミニマップの描画を確認しています。
componentごとの入力との照合は[review UI](../../src/tennis_scene/visualization/review/README.md)を参照してください。

## クリーン導入の再検証 / Rehearse a clean install

[scripts/prepare_clean_install.py](../../scripts/prepare_clean_install.py)は、この順序を独立したclone・venvで実行し、
stepごとの時間・ログ・コピー元とSHA・ディスク使用量を`preparation.json`へ記録します。
登録・申請やDriveから利用者が取得した第三者資産は、**このローカル検証に限り**明示したcheckpoint rootから
SHA検査後にコピーし、コピー後にも再検査します。自作重みとサンプルはHubから取得します。
既存venv・data・ckptへのリンクや、重みの無検査コピーは行いません。
利用者自身の導入では、上記公式配布元から取得してください。

```bash
python3 scripts/prepare_clean_install.py \
  --workspace /absolute/path/to/new-clean-install \
  --repository https://github.com/Motoki0705/tennis-lab.git --revision <commit> \
  --local-checkpoint-root /absolute/path/to/user-acquired/ckpt \
  --cuda-home /usr/local/cuda-13.0 --cuda-arch 12.0
```

未公開branchを試す場合は`--repository /absolute/path/to/worktree`とそのcommitを指定できます。
`git clone --no-local`を使い、git objectも元checkoutと共有しません。既存workspaceは拒否します。
CPU準備が成功すると`ready_for_gpu`になり、**GPU実行前で停止**します。
生成された`run_pipeline.sh`を親がtraining queueへ投入し、完了後に`visualize.sh`を実行します。
GPUの所要時間・ピークVRAM・scene検査・動画確認は、その後の実行記録へ追記してください。

2026-10-21の[検証記録](../../knowledge/nodes/tennis_scene/000030-run-clean-standard-install-20261021.md)では、
CPU準備**12分41秒**の後、queue経由のGPU完走・正規readerによるscene検査・新sceneのCPU可視化を確認しました。
この検証のclone元はGitHubではなく上記worktreeで、`git clone --no-local`と固定commitを使用しました。
登録・申請が必要な第三者資産は利用者取得済みファイルの検査・コピーであり、新規登録・手動取得や
CUDA toolkit・OSの導入時間は含みません。

The rehearsal used a local worktree clone at a pinned commit, independent asset files and a fresh venv.
CPU preparation took about 12m41s, pipeline execution 30m27s and CPU visualization 6m03s on the tested host.
GitHub transfer, toolkit/OS installation and manual third-party acquisition were not timed.
