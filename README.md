<div align="center">
  <img src="assets/branding/tennis-lab-header.png" width="420" alt="Tennis-Lab">
</div>

<hr>
<div align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/Language-English-blue" alt="English"></a>
  <a href="README.ja.md"><img src="https://img.shields.io/badge/Language-%E6%97%A5%E6%9C%AC%E8%AA%9E-blue" alt="日本語"></a>
  <br>
  <a href="https://huggingface.co/Motoki098/tennis-lab-models"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107" alt="Hugging Face Model"></a>
  <a href="https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-ffc107" alt="Hugging Face Dataset"></a>
  <br>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code-Apache--2.0-blue" alt="Code: Apache-2.0"></a>
  <a href="LICENSES.md#published-artifacts"><img src="https://img.shields.io/badge/Weights-Terms-orange" alt="Weights: terms"></a>
  <a href="LICENSES.md#sample-dataset-terms"><img src="https://img.shields.io/badge/Sample-CC%20BY--NC--ND%204.0-blue" alt="Sample: CC BY-NC-ND 4.0"></a>
</div>

## Table of Contents

1. [Overview](#overview)
2. [Model Downloads](#model-downloads)
3. [How to Run Locally](#run-locally)
4. [License](#license)
5. [Citation](#citation)

<a id="overview"></a>

## 1. Overview

A research pipeline that reconstructs player motion and ball trajectories in court coordinates from **3–5 synchronized fixed cameras** filming tennis. It combines per-view 2D detection of the court, player poses and ball with camera calibration, cross-view association, triangulation and body recovery.

![Three synchronized input views and the reconstructed 3D scene](assets/tennis_scene/multiview_reconstruction.gif)

© 2026 Motoki0705 · CC BY-NC-ND 4.0 · [Full attribution and scope](LICENSES.md#sample-dataset-terms).

The three views of the public sample `video_000/clip_000` (left) and its 3D reconstruction at the same frame (right).

Synchronize the videos beforehand and use matching FPS, resolution and frame counts. See the [pipeline README](src/tennis_scene/README.md) for input constraints and the output format, including missing observations and quality information.

<a id="model-downloads"></a>

## 2. Model Downloads

### Checkpoints and supporting assets

<!-- checkpoint-assets:start -->
| Asset | Destination (relative to `ckpt/`) | Source | License / terms |
| --- | --- | --- | --- |
| court | `court_detection/multiscale_depth3/b863df1f01f0.ckpt` | [Source](https://huggingface.co/Motoki098/tennis-lab-models) | DINOv3 License |
| ball | `ball_detection/i935-mixed-ft-s42-epoch09.ckpt` | [Source](https://huggingface.co/Motoki098/tennis-lab-models) | CC BY-NC 4.0 |
| detector | `dino/checkpoint0029_4scale_swin.pth` | [Source](https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_) | DINO / Apache-2.0¹ |
| vitpose | `vitpose/vitpose-h-multi-coco.pth` | [Source](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) | ViTPose / Apache-2.0¹ |
| hmr2 | `hmr2/epoch=10-step=25000.ckpt` | [Source](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) | HMR2 / MIT¹; separate SMPL terms apply |
| gvhmr | `gvhmr/gvhmr_siga24_release.ckpt` | [Source](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) | GVHMR non-commercial terms |
| body_model | `body_models/smplx/SMPLX_NEUTRAL.npz` | [Source](https://smpl-x.is.tue.mpg.de/) | SMPL-X Model License / registration and acceptance |
| aflink | `person_tracking/AFLink_epoch20.pth` | [Source](https://drive.google.com/file/d/1DFMUkL-dc-j8-fibcJIq-46Xoq_bFoO9/view) | AFLink weight terms unconfirmed |
| tracking_encoder | `player_association/person_vit_clip_reid.pth` | [Source](https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view) | CLIP-ReID / MIT¹ |
| smpl_faces | `body_models/smplh/neutral/model.npz` | [Source](https://mano.is.tue.mpg.de/) | MANO / SMPL+H License / registration and acceptance |
| dinov3 | `dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | [Source](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) | DINOv3 License / application required |
| smpl_hmr2_mean_params | `body_models/gvhmr/smpl_mean_params.npz` | [Source](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/network/hmr2/configs/smpl_mean_params.npz) | GVHMR / SMPL / SMPL-X terms |
| smpl_smplx_to_smpl | `body_models/gvhmr/smplx2smpl_sparse.pt` | [Source](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smplx2smpl_sparse.pt) | GVHMR / SMPL / SMPL-X terms |
| smpl_smpl_coco17_regressor | `body_models/gvhmr/smpl_coco17_J_regressor.pt` | [Source](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt) | GVHMR / SMPL / SMPL-X terms |
| smpl_smplx_verts437 | `body_models/gvhmr/smplx_verts437.pt` | [Source](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smplx_verts437.pt) | GVHMR / SMPL / SMPL-X terms |
| smpl_smpl_neutral_joint_regressor | `body_models/gvhmr/smpl_neutral_J_regressor.pt` | [Source](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt) | GVHMR / SMPL / SMPL-X terms |
<!-- checkpoint-assets:end -->

¹ These labels identify upstream code licenses; they do not grant all rights to training data, body models or separately distributed weights. See [third-party weights and body-model assets](LICENSES.md#third-party-weights) for details.

### Multiview sample

<!-- sample-assets:start -->
| Destination | Source / terms |
| --- | --- |
| `data/samples/cam0.mp4` | [Published dataset](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/cam1.mp4` | [Published dataset](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/cam2.mp4` | [Published dataset](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/clip.json` | [Published dataset](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/LICENSE` | [Published dataset](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
<!-- sample-assets:end -->

Some assets require registration, acceptance of terms or an application to the upstream provider. Follow the [asset acquisition guide](docs/installation/README.md#required-assets). The acquisition CLIs (`src.utils.model_weights` and `src.utils.sample_dataset`) verify downloaded files; [preflight](docs/installation/README.md#preflight) checks the required assets and CUDA extension before execution.

<a id="run-locally"></a>

## 3. How to Run Locally

Follow the **[installation guide](docs/installation/README.md)** for requirements, setup and sample execution.

For development environment checks, linting, type checking and tests, see the [spin README](.spin/README.md).

<a id="license"></a>

## 4. License

Original code and documentation are under [Apache License 2.0](LICENSE) unless stated otherwise; see also [NOTICE](NOTICE).
Weights, the sample dataset, the GIF and third-party assets retain their respective terms. See [LICENSES.md](LICENSES.md) for license scope, sources and attribution.

<a id="citation"></a>

## 5. Citation

The suggested repository citation is below; include the commit used for reproducibility. Follow the sources linked in [LICENSES.md](LICENSES.md) for upstream citation and attribution requirements.

```bibtex
@software{tennis_lab,
  author = {{Tennis-Lab contributors}},
  title = {Tennis-Lab},
  url = {https://github.com/Motoki0705/tennis-lab}
}
```
