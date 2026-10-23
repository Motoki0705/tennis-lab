# 标准多视角处理流程安装指南

[English](README.md) / [日本語](README.ja.md) / [简体中文](README.zh-CN.md)

本流程根据 3–5 台固定摄像机拍摄的同步视频生成 SceneResult v2。
本文是环境配置、资产存放和运行公开示例的权威指南。
输入限制和存储格式见 [tennis_scene](../../src/tennis_scene/README.md)；
使用条款和署名要求见 [LICENSES.zh-CN.md](../../LICENSES.zh-CN.md#code-license-scope)。
请先按照各上游项目的条款获取所需资产，再运行以下命令。

<a id="requirements"></a>

## 环境要求

- Linux x86-64，或可使用 NVIDIA GPU 的 WSL2。Python 3.11 版本由 `.python-version` 固定，并通过 uv 同步。
- NVIDIA GPU。已验证的配置为 **RTX 5060 Ti / 16 GB 显存**。在独立克隆和新建虚拟环境中运行公开示例时，
  约每 5 秒通过 `nvidia-smi` 测量一次，记录到的**整卡显存使用量最高为 4,928 MiB**。
  该数值包括运行前其他进程已占用的 1,215 MiB；开始后约 22 秒内的使用量及采样间隔内的瞬时峰值未被测量。
  尚未验证显存更小的 GPU 所需的最低容量。验证使用现有的 WSL2 主机，其主机内存上限为 24 GB，内存峰值未测量。
  运行时请勿与其他任务共用 GPU。
- **CUDA toolkit 13.x 的 `nvcc`** 和兼容的驱动程序。锁定的依赖为 `torch 2.14.1+cu130`。
  已验证的配置使用 toolkit **13.0.88** 和驱动程序 **610.88**。CUDA 13.x 要求驱动程序至少为 580 系列；
  CUDA 13.0 GA 要求 Linux 驱动程序为 580.65.06 或更高版本
  （[NVIDIA 对应版本的发行说明](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-driver)）。
  `nvidia-smi` 显示的“CUDA Version”不能证明 toolkit 已安装，还需检查 `nvcc --version`。
  如果 torch 与 toolkit 的 CUDA 主版本不同，扩展构建会停止。使用 13.0 可与锁定的 cu130 构建版本保持一致。
- Git、[uv](https://docs.astral.sh/uv/getting-started/installation/)、C++ 编译器、兼容 CUDA 的主机编译器、
  `ffmpeg`（用于输出 MP4）和 Ninja。在 Ubuntu 上，请在操作系统中安装 `build-essential`、`ninja-build` 和 `ffmpeg`。
- 工作目录占用了**约 20.87 GB（19.43 GiB）**磁盘空间，包含独立克隆、虚拟环境、隔离的 uv/HF 缓存、权重、
  示例、CUDA 构建文件、组件缓存、场景和完整时长的可视化视频（准备完成时约为 19.94 GB）。
  此数值不包含 CUDA toolkit、操作系统软件包和单独下载第三方资产的工作目录。
  所需空间会随缓存和运行次数变化，请为这些内容及额外输出预留空间。
  测量范围见[安装验证记录](../../knowledge/nodes/tennis_scene/000030-run-clean-standard-install-20261021.md)。

本文不涵盖在 CPU 上运行处理流程。资产检查、扩展导入及调用分派检查、可视化均在 CPU 上执行。

<a id="clone-and-sync"></a>

## 克隆仓库并同步依赖

```bash
git clone https://github.com/Motoki0705/tennis-lab.git
cd tennis-lab
uv sync --locked
git submodule update --init third_party/DINO third_party/dinov3
```

`uv sync --locked` 会创建包含 dev 和 gpu 依赖组的新 `.venv`。
标准流程不需要 `third_party/GVHMR` 和 `third_party/nht`。
请使用当前检出版本的 gitlink 固定的子模块版本，不要添加 `--remote`。
后续所有命令均从仓库根目录执行。

<a id="dino-cuda-extension"></a>

## DINO CUDA 扩展

请在 [NVIDIA 列表](https://developer.nvidia.com/cuda-gpus)中查看 GPU 的计算能力，
并显式设置 `TORCH_CUDA_ARCH_LIST`。**下面的 12.0 仅为 RTX 5060 Ti 的示例**。
使用其他 GPU 时，请替换为该 GPU 与 CUDA 13 均支持的值。

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

请根据环境调整 `CUDA_HOME`，例如设为 `/usr/local/cuda`。在同一 shell 中执行预检和运行处理流程时，
也需要设置 `PYTHONPATH`。换用其他 shell 时请重新设置。

脚本按照统一的路径约定生成 `TENNIS_LAB_DINO_OPS_BUILD_CONFIG`，并通过仓库根目录的 `setup.py`
构建 DINO 和共用的 CUDA 扩展。输出写入 `outputs/dino_extension/{lib,temp,cache}` 和 `build.json`，
不会修改子模块或共用虚拟环境中的源代码。兼容性修复仅应用于生成的 DINO 源代码。
构建时设置 `CUDA_VISIBLE_DEVICES=''`，不使用 GPU。`validate_dino_extension()` 也仅使用 CPU 张量
检查前向和反向运算的调用分派，不对 CUDA 内核进行数值验证。
如果同步后 torch 版本发生变化，请在新的输出目录中重新构建，并显式使用其 `lib`。

<a id="required-assets"></a>

## 所需资产

所有资产均须以**实际文件**存放在仓库内。路径约定会拒绝指向检查点根目录以外的符号链接。
预检会将已知权重和示例与固定的 SHA-256 值进行比对，不会重新压缩、转换文件或自动切换到其他发行版本。
许可证适用范围的权威说明见[第三方权重](../../LICENSES.zh-CN.md#third-party-weights)和
[公开发布内容](../../LICENSES.zh-CN.md#published-artifacts)。

<!-- checkpoint-assets:start -->
| 资产 | 保存位置（相对于 `ckpt/`） | 下载来源 | 许可证 / 条款 |
| --- | --- | --- | --- |
| court | `court_detection/multiscale_depth3/b863df1f01f0.ckpt` | [下载来源](https://huggingface.co/Motoki098/tennis-lab-models) | DINOv3 License |
| ball | `ball_detection/i935-mixed-ft-s42-epoch09.ckpt` | [下载来源](https://huggingface.co/Motoki098/tennis-lab-models) | CC BY-NC 4.0 |
| detector | `dino/checkpoint0029_4scale_swin.pth` | [下载来源](https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_) | DINO / Apache-2.0¹ |
| vitpose | `vitpose/vitpose-h-multi-coco.pth` | [下载来源](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) | ViTPose / Apache-2.0¹ |
| hmr2 | `hmr2/epoch=10-step=25000.ckpt` | [下载来源](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) | HMR2 / MIT¹；SMPL 条款另行适用 |
| gvhmr | `gvhmr/gvhmr_siga24_release.ckpt` | [下载来源](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) | GVHMR 非商业用途条款 |
| body_model | `body_models/smplx/SMPLX_NEUTRAL.npz` | [下载来源](https://smpl-x.is.tue.mpg.de/) | SMPL-X Model License / 注册并同意条款 |
| aflink | `person_tracking/AFLink_epoch20.pth` | [下载来源](https://drive.google.com/file/d/1DFMUkL-dc-j8-fibcJIq-46Xoq_bFoO9/view) | AFLink 权重条款尚未核实 |
| tracking_encoder | `player_association/person_vit_clip_reid.pth` | [下载来源](https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view) | CLIP-ReID / MIT¹ |
| smpl_faces | `body_models/smplh/neutral/model.npz` | [下载来源](https://mano.is.tue.mpg.de/) | MANO / SMPL+H License / 注册并同意条款 |
| dinov3 | `dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | [下载来源](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) | DINOv3 License / 需提交申请 |
| smpl_hmr2_mean_params | `body_models/gvhmr/smpl_mean_params.npz` | [下载来源](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/network/hmr2/configs/smpl_mean_params.npz) | GVHMR / SMPL / SMPL-X 条款 |
| smpl_smplx_to_smpl | `body_models/gvhmr/smplx2smpl_sparse.pt` | [下载来源](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smplx2smpl_sparse.pt) | GVHMR / SMPL / SMPL-X 条款 |
| smpl_smpl_coco17_regressor | `body_models/gvhmr/smpl_coco17_J_regressor.pt` | [下载来源](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt) | GVHMR / SMPL / SMPL-X 条款 |
| smpl_smplx_verts437 | `body_models/gvhmr/smplx_verts437.pt` | [下载来源](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smplx_verts437.pt) | GVHMR / SMPL / SMPL-X 条款 |
| smpl_smpl_neutral_joint_regressor | `body_models/gvhmr/smpl_neutral_J_regressor.pt` | [下载来源](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt) | GVHMR / SMPL / SMPL-X 条款 |
<!-- checkpoint-assets:end -->

¹ 这些是上游仓库为其代码声明的许可证。代码许可不应被视为对训练数据、人体模型或单独发布的权重
所涉及全部权利的一揽子授权。按照当前决定，AFLink 在使用条款尚未核实的情况下继续用于标准流程。
依据和限制见 [StrongSORT NOTICE](../../src/tasks/person_tracking/strongsort_NOTICE.md)。

### 项目权重和公开示例

请使用[权重下载 CLI](../../src/utils/README.md#published-detector-weights)和
[示例下载 CLI](../../src/utils/README.md#published-multiview-sample)。
下载工具会检查已有文件，不会覆盖内容不匹配的文件。

```bash
.venv/bin/python -m src.utils.model_weights --checkpoint-root "$PWD/ckpt"
.venv/bin/python -m src.utils.sample_dataset --data-root "$PWD/data"
```

示例为 `Motoki098/tennis-lab-multiview-sample` 中的 `video_000/clip_000`：
1920×1080、59.94006 fps、1,010 帧、3 台摄像机，其中 cam0 经过放大和加黑边处理。
同步方式和数据来源见公开数据集说明页。使用条款为 **CC BY-NC-ND 4.0**。
视频、清单文件和 LICENSE 的存放位置如下。

<!-- sample-assets:start -->
| 保存位置 | 下载来源 / 条款 |
| --- | --- |
| `data/samples/cam0.mp4` | [公开数据集](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/cam1.mp4` | [公开数据集](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/cam2.mp4` | [公开数据集](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/clip.json` | [公开数据集](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/LICENSE` | [公开数据集](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
<!-- sample-assets:end -->

### 第三方权重和人体模型

- **DINO**：从[官方模型库](https://github.com/IDEA-Research/DINO/tree/d84a491d41898b3befd8294d1cf2614661fc0953#model-zoo)
  中 4-scale Swin-L 对应的 Drive 链接下载 `checkpoint0029_4scale_swin.pth`。不需要训练用的 Swin 初始权重。
- **ViTPose-H、HMR2 和 GVHMR**：从 [GVHMR 官方安装说明](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/docs/INSTALL.md)
  提供的[官方 Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) 下载表中的三个文件。
  不需要 GVHMR 的 conda 环境、YOLO 或 DPVO。
- **CLIP-ReID**：在[官方 README](https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/README.md#trained-models-and-test-logs)
  中选择 **ViT-CLIP-ReID / Market / model**。文件名为 `Market1501_clipreid_ViT-B-16_60.pth`。
  请保持内容不变，将其存放到 `ckpt/player_association/person_vit_clip_reid.pth`。
  官方发布文件与此前使用的版本内容相同，详见[验证记录](../../knowledge/nodes/tennis_scene/000030-run-clean-standard-install-20261021.md)。
- **AFLink**：从 [StrongSORT 官方 README](https://github.com/dyhBUPT/StrongSORT#datamodel-preparation)
  提供的 Drive 链接下载 `AFLink_epoch20.pth`。其条款尚未核实，详见上述 NOTICE。
- **DINOv3 ViT-B/16**：通过 [Meta 官方网站](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
  提交申请，获取 `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`。
  公开的球场检查点已包含骨干网络，但**当前的模型构建仍需要这个独立文件**。
- **SMPL-X**：在[官方网站](https://smpl-x.is.tue.mpg.de/)注册并同意条款后，从 NPZ 格式的人体模型中取出
  `SMPLX_NEUTRAL.npz` 并存放到指定位置。本次标准运行不需要男性或女性 SMPL-X 模型、SMPL 本体或 SMPL-H 的 PKL 文件。
- **SMPL-H**：在 [MANO / SMPL+H](https://mano.is.tue.mpg.de/) 注册并同意条款后，获取用于 AMASS 的 Extended SMPL+H 模型。
  将解压后的 `neutral/model.npz` 存放到表中位置。可视化会读取其中的网格面信息。请勿将 PKL 转换为 NPZ。
- **五项 SMPL 辅助文件**：获取、校验和存放步骤以 [src/submodules 指南](../../src/submodules/README.md#smpl補助資産の取得)为准。
  请从官方 GVHMR 的固定提交获取文件，并通过 `install_smpl_assets` 存放到表中位置。本文不重复这些步骤。

对于 Google Drive 文件夹，请仅下载其中所需的文件，并按表中的名称存放。
需要注册或申请才能获取的模型不会由本仓库重新分发，也不会通过其他镜像提供。

<a id="preflight"></a>

## 运行前预检

```bash
export PYTHONPATH="$PWD/outputs/dino_extension/lib"
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m src.tennis_scene.scripts.preflight
```

预检必须返回 `status: ok` 和退出码 0。文件缺失、内容不匹配、读取失败和扩展不兼容等问题会**汇总在一起**
以 JSON 输出；只要存在任何问题，退出码就是 1。请明确修复已获取的文件，然后重新运行。
检查范围包括 `enabled_model_assets()` 中的全部资产、DINO 和 DINOv3 源代码、独立的 DINOv3 权重、
可视化所需的 SMPL-H 和回归器，以及输入视频（公开示例还包括清单文件和 LICENSE）。
预检不会下载文件、替换模型、执行推理或使用 GPU。源代码检查仅验证必需文件是否存在。
这项检查用于安装上表所列的公开标准流程，并非评估其他模型权重的通用命令。
请在解决所有问题后再启动处理流程。

<a id="run-the-sample"></a>

## 运行示例

```bash
PYTHONPATH="$PWD/outputs/dino_extension/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m src.tennis_scene.scripts.run_pipeline \
  output_directory=tennis_scene/generate/clean_install output_name=scene \
  cache.directory=tennis_scene/cache/clean_install
```

在本仓库的共用 GPU 上，请通过[训练队列](../../.agents/skills/training-queue/SKILL.md)运行。
在 RTX 5060 Ti 上进行的全新安装验证中，从处理流程启动到场景保存耗时 **30 分 27 秒**
（[测量记录](../../knowledge/nodes/tennis_scene/000030-run-clean-standard-install-20261021.md)）。
这是公开示例中一个片段的参考值，不包含排队等待、进程启动、资产获取、构建和可视化时间。

运行后会保存 `outputs/tennis_scene/generate/clean_install/scene.npz` 和 `scene.metadata.json`。
请检查元数据中的 `scene_schema_version=2` 和 `status=ok`。
还需通过标准读取器检查掩码、原因代码和坐标系。

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

输出缓存仅适用于此次输入、配置和代码。进行新的安装验证时，请使用新的克隆和空输出目录。
流程完整运行仅表明处理和存储约定得到满足，不保证相对于真实值的 3D 精度。
与此前使用相同输入和推理代码的运行相比，有效的球体 3D 帧数以及球员根节点/SMPL 帧数存在差异。
原因尚未确定，本次验证不保证输出逐字节相同。详情见上述测量记录。

<a id="visualize"></a>

## 可视化

请先确认 `ffmpeg -version` 执行成功，再使用同一检查点根目录中的 SMPL-H 和回归器进行渲染。

```bash
CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg .venv/bin/python -m src.tennis_scene.scripts.visualization \
  input=tennis_scene/generate/clean_install/scene.npz \
  output=tennis_scene/visualize/clean_install/scene.mp4 \
  style.player_representation=smpl
```

播放 MP4，检查球场、两位球员、网球和数据缺失的时间段。
全新安装验证中的 CPU 渲染耗时 **6 分 3 秒**（由父会话计时，包含启动时间），
生成了分辨率为 1200×800、共 1,010 帧、时长约 16.85 秒的 MP4。
父会话检查了 9 秒处的画面，确认球场、两位球员的 SMPL 网格、网球和小地图均已渲染。
与各组件输入进行对照的方法见[审查界面](../../src/tennis_scene/visualization/review/README.md)。

<a id="rehearse-clean-install"></a>

## 复现全新安装

[scripts/prepare_clean_install.py](../../scripts/prepare_clean_install.py) 会在独立克隆和虚拟环境中执行上述步骤，
将每一步的耗时、日志、复制来源、校验结果和磁盘使用量记录到 `preparation.json`。
对于用户通过注册、申请或 Drive 获取的第三方资产，**仅在此次本地验证中**，脚本会先从明确指定的检查点根目录
校验并复制文件，再在复制后重新校验。项目权重和示例从 Hub 下载。
脚本不会链接已有的虚拟环境、data 或 ckpt，也不会在未校验的情况下复制权重。
自行安装时，请从上述官方来源获取资产。

```bash
python3 scripts/prepare_clean_install.py \
  --workspace /absolute/path/to/new-clean-install \
  --repository https://github.com/Motoki0705/tennis-lab.git --revision <commit> \
  --local-checkpoint-root /absolute/path/to/user-acquired/ckpt \
  --cuda-home /usr/local/cuda-13.0 --cuda-arch 12.0
```

测试尚未发布的分支时，请指定 `--repository /absolute/path/to/worktree` 及其提交。
脚本使用 `git clone --no-local`，因此连 Git 对象也不会与源检出目录共用。
如果工作目录已存在，脚本会拒绝使用。CPU 准备成功后，状态变为 `ready_for_gpu`，脚本会**在使用 GPU 执行前停止**。
父会话将生成的 `run_pipeline.sh` 提交到训练队列，完成后再运行 `visualize.sh`。
请将 GPU 耗时、显存峰值、场景检查和视频审查结果补充到后续执行记录中。

[2026-10-21 的验证记录](../../knowledge/nodes/tennis_scene/000030-run-clean-standard-install-20261021.md)表明，
CPU 准备耗时 **12 分 41 秒**，随后通过队列完成 GPU 运行、使用标准读取器检查场景，并在 CPU 上对新场景进行可视化。
本次验证的克隆来源为上述本地 worktree，而非 GitHub，使用了 `git clone --no-local` 和固定提交。
需要注册或申请的第三方资产来自用户已获取文件的校验副本。
计时不包含重新注册、手动获取文件、安装 CUDA toolkit 或配置操作系统的时间。
