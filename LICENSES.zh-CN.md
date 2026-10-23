# 许可证与来源说明

[English](LICENSES.md) | [日本語](LICENSES.ja.md) | [简体中文](LICENSES.zh-CN.md)

<a id="code-license-scope"></a>

## Apache-2.0 对代码的适用范围

除非另有说明，Tennis-Lab 的原创代码和文档均遵循仓库根目录的
[Apache License 2.0](LICENSE)。版权声明和适用范围说明见 [NOTICE](NOTICE)。
这不会取代第三方代码的版权或许可证。`third_party/`、移植代码、权重、人体模型资源、
视频、标注、动作数据和论文分别适用下文所列条款，不能将整个仓库或流水线视为均采用 Apache-2.0。
对于采用 MIT 的第三方代码，也须保留原有版权声明和许可声明。

sdist 和 wheel 的 `License-Expression` 在 [`pyproject.toml`](pyproject.toml) 中
使用 `AND` 连接原创代码与各发行包实际包含的第三方代码的许可条款。
各项条款的适用范围及全文见本节和下方的第三方代码清单。
GVHMR 的自定义条款以及 KPR 采用的完整 HL3-LAW-MEDIA-MIL-SOC-SV 许可证
在 [SPDX License List](https://spdx.org/licenses/) 中没有对应标识符，因此使用以下 `LicenseRef-` 标识符。
SPDX 标识符 `Hippocratic-3.0-core` 不能代表包含 KPR 所采用附加条款在内的完整条款。
此对应表及链接中的许可证全文通过 `license-files` 包含在上述两种发行包中。

| 元数据标识符 | 适用范围 | 随包提供的许可证全文 |
| --- | --- | --- |
| `LicenseRef-GVHMR` | `src/submodules/vendor/gvhmr/` 中源自 GVHMR 的部分。其包含的第三方部分另有条款，见下表。 | [`src/submodules/vendor/gvhmr/LICENSE`](src/submodules/vendor/gvhmr/LICENSE) |
| `LicenseRef-HL3-LAW-MEDIA-MIL-SOC-SV` | 移植到 `src/tasks/player_association/appearance/kpr_vendor/` 的 KPR 代码。 | [`src/tasks/player_association/appearance/kpr_vendor/LICENSE`](src/tasks/player_association/appearance/kpr_vendor/LICENSE) |

原创部分改用 Apache-2.0，自 [#1313](https://github.com/Motoki0705/tennis-lab/issues/1313)
对应 PR 的合并提交起生效。此前以 MIT 分发的版本所授予的许可不会撤销，历史标签和提交记录也不会更改。

以下为截至 2026-10-19 的来源清单。对于已记录移植来源版本的代码，列出相应提交；
子模块则列出其 gitlink 固定的提交。标为“用于比较”或“用于核对”的版本是编制本清单时参考的版本，
并非历史移植来源或模型权重的版本。对于在 Git 之外分发的资源，Git 提交不适用，改为列出分发名称或版本。
**将许可尚未核实的项目列入清单，并不表示允许再次分发。**

<a id="third-party-code"></a>

## 第三方代码

<a id="vendor-directories"></a>

### vendor 目录

本节列出所有名为 `vendor/` 或 `*_vendor/` 的目录，以及直接位于 `vendor/` 下的代码包，
也包括测试中的同名目录。

| 路径 | 来源 URL 与移植来源提交 | 许可证与声明 |
| --- | --- | --- |
| `src/submodules/vendor` | [Tennis-Lab](https://github.com/Motoki0705/tennis-lab/tree/37c115e15cf0ba63668d3bc75d468c75a7e8d08c/src/submodules/vendor)（用于核对：`37c115e15cf0ba63668d3bc75d468c75a7e8d08c`） | 容器目录的 `__init__.py` 是采用 Apache-2.0 的原创代码。其下的 GVHMR 适用下一行的条款。 |
| `src/submodules/vendor/gvhmr` | [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) 的 [Motoki0705/GVHMR 分支](https://github.com/Motoki0705/GVHMR/tree/900e8c4471e80bb6db5cb14c404c1a0766aba5f4)，移植来源为 `900e8c4471e80bb6db5cb14c404c1a0766aba5f4` | GVHMR 自定义的非商业用途条款。[LICENSE 全文](src/submodules/vendor/gvhmr/LICENSE)、[移植说明](src/submodules/vendor/gvhmr/README.md)。所包含的第三方部分另见下表。 |
| `src/tasks/person_tracking/deep_ocsort_vendor` | [GerardMaggiolino/Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT/tree/6bb51d027b137233f5c520b6fcc4f2ae387a6ba9)，`6bb51d027b137233f5c520b6fcc4f2ae387a6ba9` | MIT。也保留了 Roger R. Labbe Jr. 的卡尔曼滤波实现的 MIT 声明。[LICENSE](src/tasks/person_tracking/deep_ocsort_vendor/LICENSE)、[NOTICE](src/tasks/person_tracking/deep_ocsort_vendor/NOTICE.md)。 |
| `src/tasks/player_association/appearance/solider_vendor` | [tinyvision/SOLIDER-REID](https://github.com/tinyvision/SOLIDER-REID/tree/8c08e1c3255e8e1e51e006bf189e52cc57b009ed)，`8c08e1c3255e8e1e51e006bf189e52cc57b009ed` | MIT。[LICENSE](src/tasks/player_association/appearance/solider_vendor/LICENSE)、[NOTICE](src/tasks/player_association/appearance/solider_vendor/NOTICE.md)。 |
| `src/tasks/player_association/appearance/kpr_vendor` | [VlSomers/keypoint_promptable_reidentification](https://github.com/VlSomers/keypoint_promptable_reidentification/tree/e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa)，`e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa` | Hippocratic License 3.0 **HL3-LAW-MEDIA-MIL-SOC-SV**。包含伦理条件和衍生作品的条件，不属于仓库根目录 Apache-2.0 的适用范围。[LICENSE](src/tasks/player_association/appearance/kpr_vendor/LICENSE)、[NOTICE](src/tasks/player_association/appearance/kpr_vendor/NOTICE.md)。 |
| `tests/unit/submodules/vendor` | [Tennis-Lab](https://github.com/Motoki0705/tennis-lab/tree/37c115e15cf0ba63668d3bc75d468c75a7e8d08c/tests/unit/submodules/vendor)（用于核对：`37c115e15cf0ba63668d3bc75d468c75a7e8d08c`） | Apache-2.0。用于验证移植代码的原创测试，并非上游代码的分发目录。 |
| `tests/unit/submodules/vendor/gvhmr` | [Tennis-Lab](https://github.com/Motoki0705/tennis-lab/tree/37c115e15cf0ba63668d3bc75d468c75a7e8d08c/tests/unit/submodules/vendor/gvhmr)（用于核对：`37c115e15cf0ba63668d3bc75d468c75a7e8d08c`） | Apache-2.0。上述测试中用于 GVHMR 的包。 |

GVHMR 的 LICENSE 仅允许出于教育、研究和非营利目的使用、复制、修改和分发，
要求公开修改部分的源代码，禁止商业用途，并要求保留版权、署名等声明及相关信息。
随包提供的文本与 [zju3dv/GVHMR `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1` 的原文](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/LICENSE)
逐字节一致，也与移植来源分支 `900e8c4471e80bb6db5cb14c404c1a0766aba5f4` 的 LICENSE 一致。

<a id="gvhmr-bundled-code"></a>

### GVHMR 中包含的第三方部分

以下路径均相对于 `src/submodules/vendor/gvhmr/`。
除 ViTPose 的预测头和模型构建器外，直接来源均为上述 GVHMR 分支的
`900e8c4471e80bb6db5cb14c404c1a0766aba5f4`。
下表链接到用于比较的提交和文件。GVHMR 的统一声明不会消除原权利人的条款或中间移植版本的条款。

| 路径 | 来源 URL 与比较版本提交 | 许可证与核查结果 |
| --- | --- | --- |
| `utils/quaternion.py` | [GVHMR 的直接来源文件](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/geo/quaternion.py)。与 [QuaterNet `9d8485b732b0a44b99b6cf4b12d3915703507ddc`](https://github.com/facebookresearch/QuaterNet/blob/9d8485b732b0a44b99b6cf4b12d3915703507ddc/common/quaternion.py) 及 [VideoPose3D `1afb1ca0f1237776518469876342fc8669d3f6a9`](https://github.com/facebookresearch/VideoPose3D/blob/1afb1ca0f1237776518469876342fc8669d3f6a9/common/quaternion.py) 进行了比较。 | Facebook, Inc.。QuaterNet / VideoPose3D 采用 **CC BY-NC 4.0**。随包提供 [QuaterNet 许可证全文](src/submodules/vendor/gvhmr/LICENSE.QuaterNet)。当前的 7 个函数均与 GVHMR 版本的 AST 一致。`qmul` 和 `qrot` 相比 QuaterNet 版本有 reshape/contiguous 方面的修改。该文件与 VideoPose3D 并不完全一致。GVHMR 的新增和修改部分也适用其条款。 |
| `hmr2/*`（尤其是 `vit.py`） | [4D-Humans/HMR2 `efe18deff163b29dff87ddbd575fa29b716a356c`](https://github.com/shubham-goel/4D-Humans/tree/efe18deff163b29dff87ddbd575fa29b716a356c/hmr2)。[ViTPose 在 `c050ed29112da7704797cc1a65af0234b525010d` 的 ViT](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/models/backbones/vit.py)。 | HMR2 采用 [MIT](src/submodules/vendor/gvhmr/LICENSE.HMR2)；包含的 OpenMMLab/ViTPose 部分采用 [Apache-2.0](src/submodules/vendor/gvhmr/LICENSE.ViTPose)。GVHMR 的 HMR2 ViT 与 4D-Humans 版本逐字节一致；当前版本相较于该版本新增 6 行、删除 20 行。未将 GPL 版本的 ViT 认定为当前版本的直接来源。SMPL 平均参数见下文。 |
| `vitpose/kp2d_utils.py` | [GVHMR 来源文件](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/kpts/kp2d_utils.py)。ViTPose 在 `c050ed29112da7704797cc1a65af0234b525010d` 的 [top_down_eval.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/core/evaluation/top_down_eval.py) 和 [post_transforms.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/core/post_processing/post_transforms.py)。 | OpenMMLab，Apache-2.0（全文见上文）。6 个函数的 AST 均与对应的 ViTPose 版本一致。也与 MMPose v0.29.0（`0d9a00c1d64b68999e4b2a4ee5ed0d5ac2c3e296`）进行了比较，但由于 `keypoints_from_heatmaps` 存在差异，未将其认定为完全一致的来源。 |
| `vitpose/heatmap_head.py` | ViTPose 官方在 `c050ed29112da7704797cc1a65af0234b525010d` 的 [topdown_heatmap_simple_head.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/models/heads/topdown_heatmap_simple_head.py) 和 [topdown_heatmap_base_head.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/models/heads/topdown_heatmap_base_head.py)。 | OpenMMLab，[Apache-2.0 全文](src/submodules/vendor/gvhmr/LICENSE.ViTPose)。直接基于官方版本重新构建。以 torch 层替换 MMCV 工厂函数；仅实现已验证配置下单个特征图的推理。在文件开头保留了版权声明、来源及修改说明。 |
| `vitpose/__init__.py` | ViTPose 官方在 `c050ed29112da7704797cc1a65af0234b525010d` 的 [top_down.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/models/detectors/top_down.py) 和 [ViTPose_huge_coco_256x192.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py)。 | OpenMMLab，Apache-2.0（全文见上文）。将官方 TopDown 的 backbone→head 路径重新构建为带名称的 torch 序列，明确组装官方 ViT-H 配置。严格加载检查点，不转换键名。在文件开头保留了版权声明、来源及修改说明。 |
| `vitpose/flip_utils.py` | [GVHMR flip_utils.py，`900e8c4471e80bb6db5cb14c404c1a0766aba5f4`](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/geo/flip_utils.py) | GVHMR 条款。保留的 `flip_heatmap_coco17` 与来源文件的 AST 一致。 |

ViTPose 的预测头和模型构建器已替换：此前经由 GPL-3.0 ViTPose-Pytorch 移植的实现，
现已改为直接基于上述 Apache-2.0 官方版本的实现。发行包不包含旧实现代码；
长期保留的测试使用由合成输入和随机权重生成的数值测试数据。
来源比较及使用实际检查点进行 CPU 兼容性验证的记录，见 [#1295](https://github.com/Motoki0705/tennis-lab/issues/1295) 对应的 PR。

<a id="third-party-submodules"></a>

### 第三方子模块（含递归依赖）

提交由本仓库或父级子模块的 gitlink 固定。
请保留各检出版本中随附的 LICENSE 及各文件中的声明。

| 路径 | 来源 URL 与固定提交 | 许可证 |
| --- | --- | --- |
| `third_party/GVHMR` | [Motoki0705/GVHMR](https://github.com/Motoki0705/GVHMR/tree/8a956ca7f4d7071d56f8cc0a987270f302015990)，是 [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) 的分支，`8a956ca7f4d7071d56f8cc0a987270f302015990` | [GVHMR 非商业用途条款](https://github.com/Motoki0705/GVHMR/blob/8a956ca7f4d7071d56f8cc0a987270f302015990/LICENSE)。所包含的代码另有各自条款。此提交与移植到 src 的来源提交不同。 |
| `third_party/dinov3` | [Motoki0705/dinov3](https://github.com/Motoki0705/dinov3/tree/4add255125e6cbf446c6a43c5f3028aa41b63648)，是 [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) 的分支，`4add255125e6cbf446c6a43c5f3028aa41b63648` | [DINOv3 License](https://github.com/Motoki0705/dinov3/blob/4add255125e6cbf446c6a43c5f3028aa41b63648/LICENSE.md)。不采用 MIT 或 Apache-2.0；包含分发时须附上协议等要求。 |
| `third_party/DINO` | [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO/tree/d84a491d41898b3befd8294d1cf2614661fc0953)，`d84a491d41898b3befd8294d1cf2614661fc0953` | [Apache-2.0](https://github.com/IDEA-Research/DINO/blob/d84a491d41898b3befd8294d1cf2614661fc0953/LICENSE)。 |
| `third_party/nht` | [Motoki0705/neural-harmonic-textures](https://github.com/Motoki0705/neural-harmonic-textures/tree/9f35cf66e5ff8e6df2b02d68faad667c1eca0ba6)，是 [nv-tlabs/neural-harmonic-textures](https://github.com/nv-tlabs/neural-harmonic-textures) 的分支，`9f35cf66e5ff8e6df2b02d68faad667c1eca0ba6` | [Apache-2.0](https://github.com/Motoki0705/neural-harmonic-textures/blob/9f35cf66e5ff8e6df2b02d68faad667c1eca0ba6/LICENSE)。 |
| `third_party/nht/gsplat` | [Motoki0705/gsplat](https://github.com/Motoki0705/gsplat/tree/c8d5acd22314c1c1144a30e635e08d9c738a8b39)，`c8d5acd22314c1c1144a30e635e08d9c738a8b39`（[nht .gitmodules](https://github.com/Motoki0705/neural-harmonic-textures/blob/9f35cf66e5ff8e6df2b02d68faad667c1eca0ba6/.gitmodules)） | [Apache-2.0](https://github.com/Motoki0705/gsplat/blob/c8d5acd22314c1c1144a30e635e08d9c738a8b39/LICENSE)，适用于此分支的固定版本，不适用于所有其他版本的 gsplat。 |
| `third_party/nht/gsplat/gsplat/cuda/csrc/third_party/glm` | [g-truc/glm](https://github.com/g-truc/glm/tree/e7970a8b26732f1b0df9690f7180546f8c30e48e)，`e7970a8b26732f1b0df9690f7180546f8c30e48e`（[gsplat .gitmodules](https://github.com/Motoki0705/gsplat/blob/c8d5acd22314c1c1144a30e635e08d9c738a8b39/.gitmodules)） | [Happy Bunny License 或 MIT](https://github.com/g-truc/glm/blob/e7970a8b26732f1b0df9690f7180546f8c30e48e/copying.txt)。请保留所选条款和版权声明。 |

<a id="other-adapted-code-and-assets"></a>

### 其他移植或参考实现及展示资源

| 项目 | 来源 URL 与提交 / 版本 | 许可证与处理方式 |
| --- | --- | --- |
| `src/utils/geometry/rotation_conversions.py` | [PyTorch3D rotation_conversions.py](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/pytorch3d/transforms/rotation_conversions.py)。未记录移植时的提交；链接指向用于核对的 `88e182f989c80836f4bd744e0d9cb1852762ce01`。 | BSD-3-Clause，Meta Platforms, Inc. and affiliates。文件头部注明了移植情况。[许可原文见下文](#pytorch3d-bsd-3-clause)。 |
| `src/tasks/person_tracking/strongsort*.py` | [StrongSORT 论文，arXiv:2202.13514v2](https://arxiv.org/abs/2202.13514v2)。常量参考自 [dyhBUPT/StrongSORT `ee995076da5083e28d0da1f885297df62705ebd7`](https://github.com/dyhBUPT/StrongSORT/tree/ee995076da5083e28d0da1f885297df62705ebd7)。 | 论文采用 CC BY 4.0。本仓库根据公式重新实现的代码采用 Apache-2.0，未复制 GPL-3.0 的参考代码。[记录来源、差异和验证方法的 NOTICE](src/tasks/person_tracking/strongsort_NOTICE.md)。权重单独处理。 |
| `src/tasks/base/visualization/shared/{three.core.js,three.module.js,OrbitControls.js}` | [mrdoob/three.js r180](https://github.com/mrdoob/three.js/tree/0af9729d0c143a86a1d725d6e2c3ad83301f3f34)，`0af9729d0c143a86a1d725d6e2c3ad83301f3f34`。core/module 逐字节一致；OrbitControls 仅修改了导入目标。 | MIT，three.js authors。[全文](src/tasks/base/visualization/shared/LICENSE-three.txt)。 |
| `src/tasks/base/visualization/detection/static/icons.mjs` | [Lucide](https://github.com/lucide-icons/lucide)。未记录获取时的提交。核对时使用了 [`1ef60af944de207e0d5176905a0759ba12e6ee8d` 的 LICENSE](https://github.com/lucide-icons/lucide/blob/1ef60af944de207e0d5176905a0759ba12e6ee8d/LICENSE)。 | ISC；源自 Feather 的部分采用 MIT。[两者的全文及版权声明](src/tasks/base/visualization/shared/LICENSE-lucide.txt)。 |
| `src/utils/models/kimi_delta_attention.py` | [Kimi Linear 论文，arXiv:2510.26692](https://arxiv.org/abs/2510.26692)、[FLA naive recurrence](https://github.com/fla-org/flash-linear-attention/blob/9e97506c338769e8031966fb22fe31780addd8fa/fla/ops/kda/naive.py)（用于核对：`9e97506c338769e8031966fb22fe31780addd8fa`；移植来源提交不适用）。 | 本仓库基于论文和递推式重新实现的代码采用 Apache-2.0。FLA 参考代码也采用 [MIT](https://github.com/fla-org/flash-linear-attention/blob/9e97506c338769e8031966fb22fe31780addd8fa/LICENSE)。 |
| `.agents/skills/research-loop` 中源自 ARIS 的部分 | [wanshuiyin/Auto-claude-code-research-in-sleep](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/tree/3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d)，`3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d`。 | MIT。涉及的文件及版权声明见 [LICENSE.ARIS.txt](.agents/skills/research-loop/LICENSE.ARIS.txt)。 |

<a id="pytorch3d-bsd-3-clause"></a>

### PyTorch3D 的 BSD-3-Clause 声明

针对上述旋转转换代码的移植，保留以下
[PyTorch3D 许可原文](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/LICENSE)。

```text
BSD License

For PyTorch3D software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Meta nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

<a id="third-party-weights"></a>

## 第三方权重与人体模型资源

<a id="separately-obtained-weights"></a>

### 需要单独获取的权重

本仓库不再次分发这些检查点或人体模型。下列 URL 指向官方说明。
请区分代码许可证与权重、训练数据各自的条款。
下表的 Git 提交标识核查条款和说明时参考的版本，不能代替检查点的校验和。

| 资源 | 官方来源与来源提交 / 版本 | 许可证与条款 |
| --- | --- | --- |
| DINO 4-scale Swin-L（`checkpoint0029_4scale_swin.pth`） | [官方模型库 README](https://github.com/IDEA-Research/DINO/blob/d84a491d41898b3befd8294d1cf2614661fc0953/README.md#model-zoo)、[官方 Google Drive](https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_)。参考提交：`d84a491d41898b3befd8294d1cf2614661fc0953`。 | 官方仓库采用 [Apache-2.0](https://github.com/IDEA-Research/DINO/blob/d84a491d41898b3befd8294d1cf2614661fc0953/LICENSE)。在上述说明中未发现专门针对权重的单独许可。代码许可证不授予原始数据的相关权利。 |
| ViTPose-H（COCO 256×192） | [ViTPose 官方模型库](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/README.md)，参考提交：`c050ed29112da7704797cc1a65af0234b525010d`。供 GVHMR 使用的分发文件名为 `vitpose-h-multi-coco.pth`。 | ViTPose 官方仓库采用 [Apache-2.0](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/LICENSE)。替换代码不会改变 COCO 等训练数据或权重的使用条款。 |
| HMR2.0（`epoch=10-step=25000.ckpt`） | [4D-Humans 官方 README](https://github.com/shubham-goel/4D-Humans/blob/efe18deff163b29dff87ddbd575fa29b716a356c/README.md)，参考提交：`efe18deff163b29dff87ddbd575fa29b716a356c`。 | 仓库采用 [MIT](https://github.com/shubham-goel/4D-Humans/blob/efe18deff163b29dff87ddbd575fa29b716a356c/LICENSE.md)。官方 README 要求另行注册以获取 SMPL。MIT 不会对 SMPL 模型重新授予许可。 |
| GVHMR（`gvhmr_siga24_release.ckpt`） | [官方安装指南](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/docs/INSTALL.md)、[官方 Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD)。参考提交：`ee960bb6e2ea2d381aa97f08e9b71ef320b624b1`。 | [GVHMR 非商业用途条款](src/submodules/vendor/gvhmr/LICENSE)。分发文件夹中也包含 HMR2 和 ViTPose，但须分别同意各自的条款，不能视为统一授予 Apache-2.0 许可。 |
| CLIP-ReID ViT-B/16 Market-1501 | [官方 README 中的已训练模型](https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/README.md#trained-models-and-test-logs)、[官方 Google Drive 上的 Market 模型](https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view)。参考提交：`eb1898b72c882875f478bebfc6d41644eece0a5d`。 | 官方仓库采用 [MIT](https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/LICENSE)。请另行核查 Market-1501 等原始数据的条款。根据 2026-10-18 达成的一致意见，指定获取来源为此官方 Drive。 |
| AFLink（`AFLink_epoch20.pth`） | [StrongSORT 官方 README](https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/README.md)、[官方 Google Drive](https://drive.google.com/drive/folders/1Zk6TaSJPbpnqbz1w4kfhkKFCEzQbjfp_)。参考提交：`ee995076da5083e28d0da1f885297df62705ebd7`。 | **使用条款尚未从一手资料中得到核实。根据用户于 2026-10-19 作出的决定，标准流水线继续使用该权重。** 参考代码采用 GPL-3.0，但尚未核实专门针对权重的条款。[现有 NOTICE](src/tasks/person_tracking/strongsort_NOTICE.md) 中的临时使用决定不构成再次分发的许可。 |
| SMPL-X（`SMPLX_NEUTRAL.npz`） | [SMPL-X 官方网站](https://smpl-x.is.tue.mpg.de/)；Git 提交不适用。请核查模型分发版本及注册时同意的协议。 | [SMPL-X Model License](https://smpl-x.is.tue.mpg.de/modellicense.html)。限于非商业性的科学研究、教育和艺术目的。权利授予个人且不可转让，不允许向第三方再次分发全部或任何部分。每位用户均须自行注册、同意条款并获取模型。 |
| SMPL-H（用于动作生成） | [MANO / SMPL+H 官方网站](https://mano.is.tue.mpg.de/)；Git 提交不适用。 | [MANO / SMPL+H license](https://mano.is.tue.mpg.de/license.html)。限于非商业性的科学研究、教育和艺术目的，且有再次分发限制。每位用户均须自行注册、同意条款并获取模型。 |
| SMPL（用于 HMR2 和模型转换） | [SMPL 官方网站](https://smpl.is.tue.mpg.de/)；Git 提交不适用。 | [SMPL Model License](https://smpl.is.tue.mpg.de/modellicense.html)。限于非商业性的科学研究、教育和艺术目的；不允许向第三方再次分发全部或任何部分。 |
| DINOv3 ViT-S/16 / ViT-B/16 | [Meta 官方申请页面](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)、[官方仓库分支的 README](https://github.com/Motoki0705/dinov3/blob/4add255125e6cbf446c6a43c5f3028aa41b63648/README.md)。参考提交：`4add255125e6cbf446c6a43c5f3028aa41b63648`；分发文件名：`dinov3_vits16_pretrain_lvd1689m-08c60483.pth` / `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`。 | [DINOv3 License](https://github.com/Motoki0705/dinov3/blob/4add255125e6cbf446c6a43c5f3028aa41b63648/LICENSE.md)。适用于代码和权重等 DINO Materials 的条款。对于基于 DINOv3 衍生的本项目权重，见[权重许可依据](#weight-license-basis)。 |
| SOLIDER-REID / KPR（用于比较） | [SOLIDER 官方 README](https://github.com/tinyvision/SOLIDER-REID/blob/8c08e1c3255e8e1e51e006bf189e52cc57b009ed/README.md)（`8c08e1c3255e8e1e51e006bf189e52cc57b009ed`）、[KPR 官方 README](https://github.com/VlSomers/keypoint_promptable_reidentification/blob/e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa/README.md#download-the-pre-trained-models)（`e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa`）。 | 两个仓库分别采用 MIT / HL3-LAW-MEDIA-MIL-SOC-SV。本仓库不再次分发这些权重；请核查官方说明及训练数据的条款。 |
| OSNet（MSMT17，用于比较） | [deep-person-reid 官方模型库](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)、[官方仓库](https://github.com/KaiyangZhou/deep-person-reid/tree/f8cd150fdf77e8d9e1ed143b7f308c2c609ded50)（用于核对：`f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`）。 | 仓库采用 [MIT](https://github.com/KaiyangZhou/deep-person-reid/blob/f8cd150fdf77e8d9e1ed143b7f308c2c609ded50/LICENSE)。权重及 MSMT17 等数据须遵循原分发方的条款。 |

<a id="smpl-auxiliary-assets"></a>

### 用户自行获取的 SMPL 辅助资源

本仓库不分发以下 5 个文件。用户须从 [GVHMR 官方仓库](https://github.com/zju3dv/GVHMR/tree/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1)
的固定提交 `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1` 获取，并以原文件名
放入 CHECKPOINT 根目录下的 `body_models/gvhmr/`。
请按照[文件放置与 SHA-256 验证步骤](src/submodules/README.md#smpl補助資産の取得)操作。
2026-10-19 已核实：官方提供的这 5 个文件与此前随包提供的版本均逐字节一致。
获取 URL 和预期 SHA-256 值以[获取清单](src/submodules/smpl_assets.py)为准。

| 文件名（目标位置的基本文件名） | 官方来源（上述提交）与内容 |
| --- | --- |
| `smpl_coco17_J_regressor.pt` | [GVHMR 官方文件](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt)。从 SMPL 顶点到 COCO-17 关节的回归矩阵。 |
| `smpl_neutral_J_regressor.pt` | [GVHMR 官方文件](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt)。SMPL 的 24 关节回归矩阵。 |
| `smplx2smpl_sparse.pt` | [GVHMR 官方文件](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smplx2smpl_sparse.pt)。从 SMPL-X 到 SMPL 的稀疏顶点变换。 |
| `smplx_verts437.pt` | [GVHMR 官方文件](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smplx_verts437.pt)。用于选取 437 个 SMPL-X 顶点的索引。 |
| `smpl_mean_params.npz` | [GVHMR 官方文件](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/network/hmr2/configs/smpl_mean_params.npz)。HMR2 的平均姿态、体型及相机初始值。 |

[GVHMR 官方安装指南](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/docs/INSTALL.md)
要求克隆官方仓库，并注册获取 SMPL 和 SMPL-X。
GVHMR 适用[教育、研究和非营利目的条款](src/submodules/vendor/gvhmr/LICENSE)。
对于 SMPL/SMPL-X 模型及衍生数据，还须核查
[SMPL Model License 中的 License Grant / No Distribution](https://smpl.is.tue.mpg.de/modellicense.html)
以及 [SMPL-X Model License 的对应条款](https://smpl-x.is.tue.mpg.de/modellicense.html)。
两份模型许可证均要求：向第三方提供时须事先取得书面许可，并限制全部或任何部分的复制、共享和分发等行为。
已核实官方 GVHMR 公开了这些文件，但尚未核实 SMPL 权利人是否书面允许向第三方再次分发，
也未核实每项辅助资源的生成历史和单独许可。
用户从官方来源获取文件，与 Tennis-Lab 是否获准再次分发，是两回事。
本说明也未断定所有索引或平均统计量均适用模型本身的条款。
请勿将获取或生成的文件加入本仓库的 Git 历史或发行包。

我们也核查了其他获取途径的公开信息。
[SMPL 官方网站](https://smpl.is.tue.mpg.de/) 提供注册后获取中性模型的说明；
[SMPL-X 官方模型转换说明](https://github.com/vchoutas/smplx/blob/main/transfer_model/README.md#data)
提及注册后可获取的“Model correspondences”和 `smplx2smpl_deftrafo_setup.pkl`。
但尚未查看注册后才能获取的压缩包内容，因此获取步骤不假定从中生成所需 `.pt` 文件后会得到相同文件，
也不假定其中包含全部 5 个文件。
[SPIN 官方 README](https://github.com/nkolot/SPIN#fetch-data) 和[获取脚本](https://github.com/nkolot/SPIN/blob/master/fetch_data.sh)
提供了辅助数据的获取途径，但本项目使用的版本固定为上述 GVHMR 官方文件，不替换为其他版本。
条款如何适用于输出，见[流水线输出](#pipeline-outputs)。

<a id="third-party-data"></a>

## 第三方数据与论文

训练视频、图像、标注和动作数据不属于原创代码的 Apache-2.0 许可范围。
本仓库不再次分发下列外部数据集。公开样例的条款在后文单独说明。

| 项目 | 来源 URL 与提交 / 版本 | 许可证与处理方式 |
| --- | --- | --- |
| TennisCourtDetector 图像与球场标注 | [yastrebksv/TennisCourtDetector README](https://github.com/yastrebksv/TennisCourtDetector/blob/e5cd4f1ce26b15361700d3d89e068cbf0e82749e/README.md)。用于核对的提交：`e5cd4f1ce26b15361700d3d89e068cbf0e82749e`；分发内容包含 8841 张图像。数据本身的 Git 提交不适用。 | 无 LICENSE。官方说明称图像收集自 YouTube 上的赛事集锦。尚未核实是否获准再次分发转播画面、视频或标注。 |
| TrackNet 视频帧与球位置标注 | [yastrebksv/TrackNet README](https://github.com/yastrebksv/TrackNet/blob/730ea17a324321c553c480179caa8ff13e4cf553/README.md)。用于核对的提交：`730ea17a324321c553c480179caa8ff13e4cf553`；分发内容包含 10 段转播视频和 19835 帧图像。数据本身的 Git 提交不适用。 | 无 LICENSE。仓库公开并不代表允许自由复用或再次分发其中的数据。 |
| 来自 YouTube 的视频与原创标注 | [YouTube 服务条款](https://www.youtube.com/static?template=terms)；数据的 Git 提交不适用。各数据集的来源记录按视频管理来源和版本。 | 须遵守各视频的相关权利、声明要求及服务条款。添加原创标注不会消除视频的相关权利或合同条款。关于《著作权法》第 30 条之 4 对训练用途的判断范围及合同问题，见[训练数据与公开决定](#training-data-publication)。 |
| ACCAD / AMASS 动作数据 | [ACCAD 官方网站](https://accad.osu.edu/research/motion-lab/mocap-system-and-data)、[AMASS 官方网站](https://amass.is.tue.mpg.de/)；Git 提交不适用。使用 AMASS 分发的 ACCAD 数据时，须遵循该版本及协议。 | ACCAD Open Motion Project 的官方声明采用 [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)。AMASS 版本另有[限于非商业性科学研究的 Dataset License](https://amass.is.tue.mpg.de/license.html) 及再次分发限制。请区分原始数据与 AMASS 转换为 SMPL 格式后的数据。 |
| Meiji 多相机视频与标注 | [本项目的公开准备及已达成的一致意见，#1267](https://github.com/Motoki0705/tennis-lab/issues/1267)。非公开录制数据；Git 提交不适用。数据集清单标识各视频和片段。 | 已取得被拍摄者的同意（据项目记录）。不属于 Apache-2.0 的适用范围。公开样例的范围及许可证见[样例数据集使用条款](#sample-dataset-terms)。 |
| `knowledge/Papers/paper-2024-gvhmr/paper.pdf` | [Shen et al., GVHMR, arXiv:2409.06662v1](https://arxiv.org/abs/2409.06662v1)；Git 提交不适用；论文版本为 v1。 | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。保留作者的原始 PDF。[随附的来源说明](knowledge/Papers/paper-2024-gvhmr/paper.md)。仓库根目录的 Apache-2.0 不适用于论文本身。 |

<a id="published-artifacts"></a>

## 公开文件的许可证：本项目权重与样例数据集

以下公开条款基于[用户于 2026-10-19 作出的决定](https://github.com/Motoki0705/tennis-lab/issues/1278)。
预期用途为研究和非商业演示；此用途说明不为各许可证增加额外的用途限制。
符合标准 CC 条款的其他非商业用途也同样获得许可，并不限于研究和演示。

<a id="published-artifacts-hosting"></a>

### 发布位置与权重标识

权重发布于 Hugging Face 模型仓库
[`Motoki098/tennis-lab-models`](https://huggingface.co/Motoki098/tennis-lab-models)。
以下路径既相对于该模型仓库，也相对于 Tennis-Lab 的 `ckpt/`。

| 权重文件 | 许可证 |
| --- | --- |
| `ball_detection/i935-mixed-ft-s42-epoch09.ckpt` | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) |
| `court_detection/multiscale_depth3/b863df1f01f0.ckpt` | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) |

公开文件是 Lightning 格式的推理检查点，保留全部模型权重（包括球场检测模型的骨干网络）、
`hyper_parameters` 和 Lightning 版本信息，移除了优化器、学习率调度器、循环、回调及训练进度状态。
文件名与训练版本相同，但内容是**公开的推理版本**。
获取和 SHA 校验见[权重获取 CLI](src/utils/README.md#published-detector-weights)，
复现步骤见[导出脚本](scripts/export_detector_weights.py)。

<a id="weight-license-basis"></a>

### 权重许可依据

**球检测：**[训练配置](src/tasks/ball_detection/configs/train_meiji_mixed.yaml)
以 `ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt` 初始化，
其前身是旧 Colab 环境中的 `ckpt/ball_detection/last.ckpt`。
**第一代模型从零开始训练（用户于 2026-10-19 的答复）**，未使用外部预训练权重。
第一代模型初始化来源的依据是上述答复，并非对第一代原始检查点的重新核验。
这些本项目权重以 CC BY-NC 4.0 发布。遵守署名等条件时，允许出于非商业用途复制、共享及共享改编作品。
详细要求见[该许可证第 2、3 节](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)。

**球场检测：**检查点是包含 DINOv3 ViT-B/16 参数的衍生作品。
[调查记录](https://github.com/Motoki0705/tennis-lab/issues/1278#issuecomment-6257877131)
证实骨干网络的全部 188 个张量均与初始权重一致。
依据 [DINOv3 License 第 1.b.i 节](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)，
整个检查点按该协议的条款分发，并附上协议全文。
使用时也须遵守该协议，包括在研究论文中注明使用了 DINOv3 等要求。
仅球场权重属于本项目“权重限非商业用途”方针的例外，不额外附加本项目自定义的非商业用途条款或 CC BY-NC 4.0。

<a id="training-data-publication"></a>

### 训练数据与公开决定

球检测使用 TrackNet、Meiji 和来自 YouTube 视频的额外球位置标注进行训练
（见[生成配置](src/tasks/ball_detection/configs/generate_dataset.yaml)及上述训练配置）。
球场检测混合使用来自 TennisCourtDetector 的真实图像和 Synthetic Court B00–B03 的合成图像
（见[保存的训练运行配置](paper/court_robustness/evidence/training_run_config.yaml)）。
第三方数据的来源及分发条款核查结果见[第三方数据与论文](#third-party-data)。

训练用于信息分析，目的是估计球的位置和球场几何形状。
公开球检测权重，基于本项目认为日本《著作权法》第 30 条之 4 适用于训练阶段使用行为的判断。
该条允许在必要范围内进行不以欣赏或享受作品为目的的使用，
但排除不合理损害著作权人利益的情形。是否适用不能仅凭非商业用途来判断。
法条见 [e-Gov 上的《著作权法》第 30 条之 4](https://laws.e-gov.go.jp/law/345AC0000000048#Mp-At_30_4)，
解释上的问题见[文化厅关于 AI 与著作权的讨论文件](https://www.bunka.go.jp/seisaku/bunkashingikai/chosakuken/pdf/94037901_01.pdf)。

此判断针对训练阶段涉及著作权的使用行为，不代表已经解决与分发方之间的合同条款问题，
也不代表获得了再次分发第三方材料的许可。当时的获取途径、适用条款及是否取得许可，仍有部分尚未核实
（见[调查记录](https://github.com/Motoki0705/tennis-lab/issues/1278#issuecomment-6257877131)）。
本次公开内容不包括第三方训练视频、图像或数据集本身。

<a id="sample-dataset-terms"></a>

### 样例数据集使用条款

样例发布于 Hugging Face 数据集仓库
[`Motoki098/tennis-lab-multiview-sample`](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample)。
范围为 Meiji 已导出的 `video_000/clip_000` 中全部 3 个相机的视频及公开版 `clip.json`，
采用 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en)。
被拍摄者同意的记录见[第三方数据与论文](#third-party-data)中的 Meiji 行。
公开文件名、SHA-256 信息、拍摄条件、同步信息和 cam0 的 letterbox 填充方式见分发页面的数据集卡片；
获取和验证见[样例获取 CLI](src/utils/README.md#published-multiview-sample)。
公开清单已移除本地绝对路径，不包含原始视频、标注或推理结果。

根据该许可证第 2、3 节，在满足署名等条件时，允许出于非商业用途共享。
允许出于非商业用途创作和复制改编作品，但不授予共享这些改编作品的许可。
如果裁剪、叠加标注、制作 GIF 等行为构成改编，共享这些内容须另行取得许可。
第 2(a)(4) 节规定的必要技术格式转换，以及依适用法律无须取得许可的使用行为，均按许可证原文处理。
CC 不统一授予肖像权、隐私权等权利的许可。

署名示例：Motoki0705, “Tennis-Lab Multiview Sample”, 2026,
[分发页面](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample),
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)。

随附的 README 素材 [`assets/tennis_scene/multiview_reconstruction.gif`](assets/tennis_scene/multiview_reconstruction.gif)
是本项目作为许可方另行提供的 GIF：对上述样例视频缩小尺寸、抽帧，再将三个视角与 3D 重建结果合成。
© 2026 Motoki0705。本项目对该 GIF 拥有的权利按
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en) 提供，不属于 Apache-2.0 的适用范围。
研究和演示查看是预期用途，并不增加额外的用途限制。
提供此 GIF 不代表允许第三方共享其从原视频制作的任意改编作品。
画面中的人物由 GVHMR 恢复的 SMPL 结果渲染而成。SMPL-Body 的来源为
Max Planck Institute for Intelligent Systems（[Body License / CC BY 4.0](https://smpl.is.tue.mpg.de/bodylicense.html)）。
模型本身与人体输出的条款见[流水线输出](#pipeline-outputs)。

<a id="hf-license-metadata"></a>

### Hugging Face 的 license 元数据

在模型仓库 `Motoki098/tennis-lab-models` 的模型卡片中设置以下内容：

```yaml
---
license: other
license_name: tennis-lab-per-file-licenses
license_link: LICENSE
---
```

模型仓库根目录的 `LICENSE` 包含各文件的适用范围和两份许可证的全文。
此处的 `LICENSE` 是 Hugging Face 上的文件，与本仓库用于代码的 Apache License 2.0 不同。
`license_name` 是条款清单的标识名称，不会创建额外的自定义条件。
由于仓库包含适用不同条款的权重，请勿将整个模型仓库标记为 `cc-by-nc-4.0`。
元数据格式见 [HF 官方规范](https://huggingface.co/docs/hub/model-cards#specifying-a-license)。

在数据集仓库 `Motoki098/tennis-lab-multiview-sample` 的数据集卡片中设置以下内容：

```yaml
---
license: cc-by-nc-nd-4.0
---
```

<a id="pipeline-outputs"></a>

### 流水线输出

上述公开条款不会对另行使用的第三方资源或流水线输出统一重新授予许可。
GVHMR、SMPL、SMPL-X 等资源的获取和使用条款、继续使用 AFLink 的决定及 SMPL 辅助资源获取步骤，
见[第三方权重与人体模型资源](#third-party-weights)。

在 GVHMR 的 [LICENSE](src/submodules/vendor/gvhmr/LICENSE) 中，未发现要求所有估计数值均采用相同许可证的明确条款。
使用 SMPL / SMPL-X 生成输出时，须遵守各自 Model License 的条款，包括对生成商业用途成果的限制。
如果导出的人体网格等输出符合 [SMPL Body](https://smpl.is.tue.mpg.de/bodylicense.html)
或 [SMPL-X Body](https://smpl-x.is.tue.mpg.de/bodylicense.html) 的定义，
则另有 CC BY 4.0 许可，允许在遵守署名、提供许可证链接、注明修改，以及论文和媒体来源说明等要求的前提下共享。
Body 不包括体型混合形状（shape blendshapes）或使用这些混合形状生成人体的工具；
Body License 不会取代模型本身的条款。
仅共享估计参数时适用的条款，尚无法根据已核查的一手资料确定。
请区分模型、辅助矩阵等资源与输出的参数、人体网格。
