<div align="center">
  <img src="assets/branding/tennis-lab-header.png" width="420" alt="Tennis-Lab">
</div>

<hr>
<div align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/Language-English-blue" alt="English"></a>
  <a href="README.ja.md"><img src="https://img.shields.io/badge/Language-%E6%97%A5%E6%9C%AC%E8%AA%9E-blue" alt="日本語"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue" alt="简体中文"></a>
  <br>
  <a href="https://huggingface.co/Motoki098/tennis-lab-models"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107" alt="Hugging Face 模型"></a>
  <a href="https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-ffc107" alt="Hugging Face 数据集"></a>
  <br>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code-Apache--2.0-blue" alt="代码：Apache-2.0"></a>
  <a href="LICENSES.zh-CN.md#published-artifacts"><img src="https://img.shields.io/badge/Weights-Terms-orange" alt="权重使用条款"></a>
  <a href="LICENSES.zh-CN.md#sample-dataset-terms"><img src="https://img.shields.io/badge/Sample-CC%20BY--NC--ND%204.0-blue" alt="示例：CC BY-NC-ND 4.0"></a>
</div>

## 目录

1. [概述](#overview)
2. [模型下载](#model-downloads)
3. [在本地运行](#run-locally)
4. [许可证](#license)
5. [引用](#citation)

<a id="overview"></a>

## 1. 概述

本流程用于研究，可根据 **3–5 台固定摄像机拍摄的同步网球视频**，在球场坐标系中三维重建球员动作和网球轨迹。流程将各视角的球场、球员姿态和网球的 2D 检测结果，与相机标定、跨视角匹配、三角测量和人体重建相结合。

![三个同步输入视角与重建的 3D 场景](assets/tennis_scene/multiview_reconstruction.gif)

© 2026 Motoki0705 · CC BY-NC-ND 4.0 · [完整署名与适用范围](LICENSES.zh-CN.md#sample-dataset-terms)。

左侧是公开示例 `video_000/clip_000` 的三个视角，右侧是同一帧的 3D 重建结果。

请预先同步输入视频，并保持 FPS、分辨率和帧数一致。输入限制及包含观测缺失和质量信息的输出格式，见[处理流程 README](src/tennis_scene/README.md)。

<a id="model-downloads"></a>

## 2. 模型下载

### 检查点与辅助资产

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

¹ 这些是上游代码的许可证标识，并不代表一并授予训练数据、人体模型或单独发布权重所涉及的全部权利。详情见[第三方权重与人体模型资产](LICENSES.zh-CN.md#third-party-weights)。

### 多视角示例

<!-- sample-assets:start -->
| 保存位置 | 下载来源 / 条款 |
| --- | --- |
| `data/samples/cam0.mp4` | [公开数据集](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/cam1.mp4` | [公开数据集](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/cam2.mp4` | [公开数据集](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/clip.json` | [公开数据集](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/LICENSE` | [公开数据集](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
<!-- sample-assets:end -->

部分资产需要在上游注册、同意条款或提交申请。请按照[资产获取说明](docs/installation/README.zh-CN.md#required-assets)获取。下载 CLI（`src.utils.model_weights` 和 `src.utils.sample_dataset`）会验证下载的文件；[运行前预检](docs/installation/README.zh-CN.md#preflight)会在执行前检查所需资产和 CUDA 扩展。

<a id="run-locally"></a>

## 3. 在本地运行

环境要求、环境配置和示例运行步骤见 **[安装指南](docs/installation/README.zh-CN.md)**。

开发环境诊断、代码风格检查、类型检查和测试见 [spin README](.spin/README.md)。

<a id="license"></a>

## 4. 许可证

除另有说明外，原创代码和文档采用 [Apache License 2.0](LICENSE)。另见 [NOTICE](NOTICE)。
权重、示例数据集、GIF 和第三方资产分别适用各自的条款。许可证适用范围、来源和署名要求见 [LICENSES.zh-CN.md](LICENSES.zh-CN.md)。

<a id="citation"></a>

## 5. 引用

以下是建议的仓库引用格式；为便于复现，请同时注明所使用的提交。上游项目的引用和署名要求，请按照 [LICENSES.zh-CN.md](LICENSES.zh-CN.md) 中的来源链接查阅。

```bibtex
@software{tennis_lab,
  author = {{Tennis-Lab contributors}},
  title = {Tennis-Lab},
  url = {https://github.com/Motoki0705/tennis-lab}
}
```
