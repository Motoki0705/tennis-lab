# Licenses and attribution

[English](LICENSES.md) | [日本語](LICENSES.ja.md)

<a id="code-license-scope"></a>

## Apache-2.0 scope for code

Original Tennis-Lab code and documentation are subject to the root
[Apache License 2.0](LICENSE) unless stated otherwise. See [NOTICE](NOTICE) for
copyright notices and notes on scope. This does not replace the copyrights or
licenses of third-party code. `third_party/`, adapted code, weights, body-model
assets, videos, annotations, motion data and papers have the terms listed below;
the repository or pipeline as a whole cannot be treated as Apache-2.0.
Retain the original copyright and permission notices for MIT-licensed third-party
code as well.

The sdist and wheel `License-Expression` in [`pyproject.toml`](pyproject.toml)
combines the terms for original code and the third-party code actually bundled
in each distribution with `AND`. The scope and full text of each set of terms
are identified in this section and the third-party code inventory below.
The following `LicenseRef-` identifiers are used because the custom GVHMR terms
and the full KPR HL3-LAW-MEDIA-MIL-SOC-SV license have no identifiers in the
[SPDX License List](https://spdx.org/licenses/).
The SPDX identifier `Hippocratic-3.0-core` does not represent the full terms,
including the additional clauses adopted by KPR. This mapping and the linked
full license texts are included in both distributions through `license-files`.

| Metadata identifier | Scope | Bundled full license text |
| --- | --- | --- |
| `LicenseRef-GVHMR` | GVHMR-derived portions of `src/submodules/vendor/gvhmr/`. See the table below for the separate terms of bundled third-party portions. | [`src/submodules/vendor/gvhmr/LICENSE`](src/submodules/vendor/gvhmr/LICENSE) |
| `LicenseRef-HL3-LAW-MEDIA-MIL-SOC-SV` | KPR code adapted into `src/tasks/player_association/appearance/kpr_vendor/`. | [`src/tasks/player_association/appearance/kpr_vendor/LICENSE`](src/tasks/player_association/appearance/kpr_vendor/LICENSE) |

The migration of original portions to Apache-2.0 applies from the merge commit
of the PR for [#1313](https://github.com/Motoki0705/tennis-lab/issues/1313).
Permissions for earlier versions distributed under MIT are not revoked, and
past tags and history are not changed.

The following is an inventory of sources as of 2026-10-19. Where the source
revision of imported code is recorded, that commit is shown; submodules use
their gitlink commits. Revisions marked “for comparison” or “for verification”
are those consulted for this inventory, not the historical import revisions or
model-weight versions. Assets distributed outside Git have no applicable commit;
their distribution names or versions are given instead.
**Listing an item whose permissions are unverified does not mean it may be redistributed.**

<a id="third-party-code"></a>

## Third-party code

<a id="vendor-directories"></a>

### Vendor directories

This lists every directory named `vendor/` or `*_vendor/`, as well as code
packages directly under `vendor/`. It also includes directories with those
names in the tests.

| Path | Source URL and import commit | License and notices |
| --- | --- | --- |
| `src/submodules/vendor` | [Tennis-Lab](https://github.com/Motoki0705/tennis-lab/tree/37c115e15cf0ba63668d3bc75d468c75a7e8d08c/src/submodules/vendor) (for verification: `37c115e15cf0ba63668d3bc75d468c75a7e8d08c`) | The container's `__init__.py` is original code under Apache-2.0. GVHMR below it has the terms in the next row. |
| `src/submodules/vendor/gvhmr` | The [Motoki0705/GVHMR fork](https://github.com/Motoki0705/GVHMR/tree/900e8c4471e80bb6db5cb14c404c1a0766aba5f4) of [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR), imported from `900e8c4471e80bb6db5cb14c404c1a0766aba5f4` | Custom GVHMR noncommercial terms. [Full LICENSE](src/submodules/vendor/gvhmr/LICENSE), [adaptation details](src/submodules/vendor/gvhmr/README.md). See also the table below for bundled third-party portions. |
| `src/tasks/person_tracking/deep_ocsort_vendor` | [GerardMaggiolino/Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT/tree/6bb51d027b137233f5c520b6fcc4f2ae387a6ba9), `6bb51d027b137233f5c520b6fcc4f2ae387a6ba9` | MIT. The MIT notice for Roger R. Labbe Jr.'s Kalman implementation is also retained. [LICENSE](src/tasks/person_tracking/deep_ocsort_vendor/LICENSE), [NOTICE](src/tasks/person_tracking/deep_ocsort_vendor/NOTICE.md). |
| `src/tasks/player_association/appearance/solider_vendor` | [tinyvision/SOLIDER-REID](https://github.com/tinyvision/SOLIDER-REID/tree/8c08e1c3255e8e1e51e006bf189e52cc57b009ed), `8c08e1c3255e8e1e51e006bf189e52cc57b009ed` | MIT. [LICENSE](src/tasks/player_association/appearance/solider_vendor/LICENSE), [NOTICE](src/tasks/player_association/appearance/solider_vendor/NOTICE.md). |
| `src/tasks/player_association/appearance/kpr_vendor` | [VlSomers/keypoint_promptable_reidentification](https://github.com/VlSomers/keypoint_promptable_reidentification/tree/e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa), `e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa` | Hippocratic License 3.0 **HL3-LAW-MEDIA-MIL-SOC-SV**. Includes ethical conditions and conditions for derivative works; outside the root Apache-2.0 license. [LICENSE](src/tasks/player_association/appearance/kpr_vendor/LICENSE), [NOTICE](src/tasks/player_association/appearance/kpr_vendor/NOTICE.md). |
| `tests/unit/submodules/vendor` | [Tennis-Lab](https://github.com/Motoki0705/tennis-lab/tree/37c115e15cf0ba63668d3bc75d468c75a7e8d08c/tests/unit/submodules/vendor) (for verification: `37c115e15cf0ba63668d3bc75d468c75a7e8d08c`) | Apache-2.0. Original tests for the adapted code; not a distribution directory for upstream code. |
| `tests/unit/submodules/vendor/gvhmr` | [Tennis-Lab](https://github.com/Motoki0705/tennis-lab/tree/37c115e15cf0ba63668d3bc75d468c75a7e8d08c/tests/unit/submodules/vendor/gvhmr) (for verification: `37c115e15cf0ba63668d3bc75d468c75a7e8d08c`) | Apache-2.0. The GVHMR package for the tests above. |

The GVHMR LICENSE permits use, copying, modification and distribution only for
educational, research and nonprofit purposes. It requires disclosure of the
source of modifications, prohibits commercial use, and requires retention of
copyright and attribution notices and related information. The bundled text is
byte-for-byte identical to the [original at zju3dv/GVHMR `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1`](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/LICENSE).
It is also identical to the LICENSE in the source fork at
`900e8c4471e80bb6db5cb14c404c1a0766aba5f4`.

<a id="gvhmr-bundled-code"></a>

### Third-party portions bundled in GVHMR

The paths below are relative to `src/submodules/vendor/gvhmr/`.
Except for the ViTPose head and model builder, the direct source is the GVHMR
fork above at `900e8c4471e80bb6db5cb14c404c1a0766aba5f4`.
The comparison commits and files are linked below. A blanket GVHMR notice does
not remove the original rights holders' terms or the terms of intermediate adaptations.

| Path | Source URL and comparison commit | License and findings |
| --- | --- | --- |
| `utils/quaternion.py` | [Direct GVHMR source file](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/geo/quaternion.py). Compared with [QuaterNet `9d8485b732b0a44b99b6cf4b12d3915703507ddc`](https://github.com/facebookresearch/QuaterNet/blob/9d8485b732b0a44b99b6cf4b12d3915703507ddc/common/quaternion.py) and [VideoPose3D `1afb1ca0f1237776518469876342fc8669d3f6a9`](https://github.com/facebookresearch/VideoPose3D/blob/1afb1ca0f1237776518469876342fc8669d3f6a9/common/quaternion.py). | Facebook, Inc. QuaterNet / VideoPose3D use **CC BY-NC 4.0**. The [full QuaterNet license](src/submodules/vendor/gvhmr/LICENSE.QuaterNet) is bundled. All 7 current functions match the GVHMR version's AST. `qmul` and `qrot` have reshape/contiguous changes from QuaterNet. The file does not fully match VideoPose3D. GVHMR additions and modifications also have their terms. |
| `hmr2/*` (especially `vit.py`) | [4D-Humans/HMR2 `efe18deff163b29dff87ddbd575fa29b716a356c`](https://github.com/shubham-goel/4D-Humans/tree/efe18deff163b29dff87ddbd575fa29b716a356c/hmr2). [ViTPose ViT at `c050ed29112da7704797cc1a65af0234b525010d`](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/models/backbones/vit.py). | HMR2 uses [MIT](src/submodules/vendor/gvhmr/LICENSE.HMR2); bundled OpenMMLab/ViTPose portions use [Apache-2.0](src/submodules/vendor/gvhmr/LICENSE.ViTPose). The GVHMR HMR2 ViT is byte-identical to 4D-Humans; the current version has 6 added and 20 removed lines relative to it. The GPL ViT is not identified as the direct source of the current version. SMPL mean parameters are discussed below. |
| `vitpose/kp2d_utils.py` | [GVHMR source file](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/kpts/kp2d_utils.py). ViTPose at `c050ed29112da7704797cc1a65af0234b525010d`: [top_down_eval.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/core/evaluation/top_down_eval.py) and [post_transforms.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/core/post_processing/post_transforms.py). | OpenMMLab, Apache-2.0 (full text above). The ASTs of all 6 functions match the corresponding ViTPose versions. MMPose v0.29.0 (`0d9a00c1d64b68999e4b2a4ee5ed0d5ac2c3e296`) was also compared, but is not identified as an exact source because `keypoints_from_heatmaps` differs. |
| `vitpose/heatmap_head.py` | Official ViTPose at `c050ed29112da7704797cc1a65af0234b525010d`: [topdown_heatmap_simple_head.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/models/heads/topdown_heatmap_simple_head.py) and [topdown_heatmap_base_head.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/models/heads/topdown_heatmap_base_head.py). | OpenMMLab, [full Apache-2.0 text](src/submodules/vendor/gvhmr/LICENSE.ViTPose). Rebuilt directly from the official version. MMCV factories are replaced with torch layers; only inference with a single feature map is implemented for the validated configuration. Copyright, source and modification notices are retained at the top of the file. |
| `vitpose/__init__.py` | Official ViTPose at `c050ed29112da7704797cc1a65af0234b525010d`: [top_down.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/models/detectors/top_down.py) and [ViTPose_huge_coco_256x192.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py). | OpenMMLab, Apache-2.0 (full text above). The official TopDown backbone→head path is rebuilt as a named torch sequence, explicitly assembling the official ViT-H configuration. Checkpoints are loaded strictly without key conversion. Copyright, source and modification notices are retained at the top of the file. |
| `vitpose/flip_utils.py` | [GVHMR flip_utils.py at `900e8c4471e80bb6db5cb14c404c1a0766aba5f4`](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/geo/flip_utils.py) | GVHMR terms. The retained `flip_heatmap_coco17` matches the source file's AST. |

The ViTPose head and model builder were replaced: the implementations previously
adapted through GPL-3.0 ViTPose-Pytorch now use the official Apache-2.0 version
above as their direct source. The old implementation code is not bundled;
numerical fixtures produced with synthetic inputs and random weights are used
in permanent tests. See the PR for
[#1295](https://github.com/Motoki0705/tennis-lab/issues/1295) for source comparisons
and records of CPU compatibility checks with the actual checkpoint.

<a id="third-party-submodules"></a>

### Third-party submodules, including recursive dependencies

The commits are pinned by gitlinks in this repository or the parent submodule.
Retain the LICENSE bundled in each checkout and the notices in individual files.

| Path | Source URL and pinned commit | License |
| --- | --- | --- |
| `third_party/GVHMR` | [Motoki0705/GVHMR](https://github.com/Motoki0705/GVHMR/tree/8a956ca7f4d7071d56f8cc0a987270f302015990), a fork of [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR), `8a956ca7f4d7071d56f8cc0a987270f302015990` | [GVHMR noncommercial terms](https://github.com/Motoki0705/GVHMR/blob/8a956ca7f4d7071d56f8cc0a987270f302015990/LICENSE). Bundled code has separate terms. This differs from the commit imported into src. |
| `third_party/dinov3` | [Motoki0705/dinov3](https://github.com/Motoki0705/dinov3/tree/4add255125e6cbf446c6a43c5f3028aa41b63648), a fork of [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3), `4add255125e6cbf446c6a43c5f3028aa41b63648` | [DINOv3 License](https://github.com/Motoki0705/dinov3/blob/4add255125e6cbf446c6a43c5f3028aa41b63648/LICENSE.md). Neither MIT nor Apache-2.0. Includes requirements such as providing the agreement on distribution. |
| `third_party/DINO` | [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO/tree/d84a491d41898b3befd8294d1cf2614661fc0953), `d84a491d41898b3befd8294d1cf2614661fc0953` | [Apache-2.0](https://github.com/IDEA-Research/DINO/blob/d84a491d41898b3befd8294d1cf2614661fc0953/LICENSE). |
| `third_party/nht` | [Motoki0705/neural-harmonic-textures](https://github.com/Motoki0705/neural-harmonic-textures/tree/9f35cf66e5ff8e6df2b02d68faad667c1eca0ba6), a fork of [nv-tlabs/neural-harmonic-textures](https://github.com/nv-tlabs/neural-harmonic-textures), `9f35cf66e5ff8e6df2b02d68faad667c1eca0ba6` | [Apache-2.0](https://github.com/Motoki0705/neural-harmonic-textures/blob/9f35cf66e5ff8e6df2b02d68faad667c1eca0ba6/LICENSE). |
| `third_party/nht/gsplat` | [Motoki0705/gsplat](https://github.com/Motoki0705/gsplat/tree/c8d5acd22314c1c1144a30e635e08d9c738a8b39), `c8d5acd22314c1c1144a30e635e08d9c738a8b39` ([nht .gitmodules](https://github.com/Motoki0705/neural-harmonic-textures/blob/9f35cf66e5ff8e6df2b02d68faad667c1eca0ba6/.gitmodules)) | [Apache-2.0 of the pinned fork](https://github.com/Motoki0705/gsplat/blob/c8d5acd22314c1c1144a30e635e08d9c738a8b39/LICENSE). This does not apply to every other gsplat version. |
| `third_party/nht/gsplat/gsplat/cuda/csrc/third_party/glm` | [g-truc/glm](https://github.com/g-truc/glm/tree/e7970a8b26732f1b0df9690f7180546f8c30e48e), `e7970a8b26732f1b0df9690f7180546f8c30e48e` ([gsplat .gitmodules](https://github.com/Motoki0705/gsplat/blob/c8d5acd22314c1c1144a30e635e08d9c738a8b39/.gitmodules)) | [Happy Bunny License or MIT](https://github.com/g-truc/glm/blob/e7970a8b26732f1b0df9690f7180546f8c30e48e/copying.txt). Retain the selected terms and copyright notices. |

<a id="other-adapted-code-and-assets"></a>

### Other adapted or referenced implementations and display assets

| Item | Source URL and commit / version | License and handling |
| --- | --- | --- |
| `src/utils/geometry/rotation_conversions.py` | [PyTorch3D rotation_conversions.py](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/pytorch3d/transforms/rotation_conversions.py). The import commit was not recorded; the link is for verification at `88e182f989c80836f4bd744e0d9cb1852762ce01`. | BSD-3-Clause, Meta Platforms, Inc. and affiliates. The file header identifies the adaptation. [Permission text below](#pytorch3d-bsd-3-clause). |
| `src/tasks/person_tracking/strongsort*.py` | [StrongSORT paper, arXiv:2202.13514v2](https://arxiv.org/abs/2202.13514v2). Constants were referenced from [dyhBUPT/StrongSORT `ee995076da5083e28d0da1f885297df62705ebd7`](https://github.com/dyhBUPT/StrongSORT/tree/ee995076da5083e28d0da1f885297df62705ebd7). | The paper uses CC BY 4.0. This repository's reimplementation from the equations uses Apache-2.0; the GPL-3.0 reference code was not copied. [NOTICE with sources, differences and verification methods](src/tasks/person_tracking/strongsort_NOTICE.md). Weights are treated separately. |
| `src/tasks/base/visualization/shared/{three.core.js,three.module.js,OrbitControls.js}` | [mrdoob/three.js r180](https://github.com/mrdoob/three.js/tree/0af9729d0c143a86a1d725d6e2c3ad83301f3f34), `0af9729d0c143a86a1d725d6e2c3ad83301f3f34`. core/module are byte-identical; only the import target in OrbitControls was changed. | MIT, three.js authors. [Full text](src/tasks/base/visualization/shared/LICENSE-three.txt). |
| `src/tasks/base/visualization/detection/static/icons.mjs` | [Lucide](https://github.com/lucide-icons/lucide). The acquisition commit was not recorded. [LICENSE at `1ef60af944de207e0d5176905a0759ba12e6ee8d`](https://github.com/lucide-icons/lucide/blob/1ef60af944de207e0d5176905a0759ba12e6ee8d/LICENSE) was used for verification. | ISC; Feather-derived portions use MIT. [Both full texts and copyright notices](src/tasks/base/visualization/shared/LICENSE-lucide.txt). |
| `src/utils/models/kimi_delta_attention.py` | [Kimi Linear paper, arXiv:2510.26692](https://arxiv.org/abs/2510.26692), [FLA naive recurrence](https://github.com/fla-org/flash-linear-attention/blob/9e97506c338769e8031966fb22fe31780addd8fa/fla/ops/kda/naive.py) (for verification: `9e97506c338769e8031966fb22fe31780addd8fa`; no applicable import commit). | This repository's reimplementation based on the paper and recurrence uses Apache-2.0. The FLA reference code also uses [MIT](https://github.com/fla-org/flash-linear-attention/blob/9e97506c338769e8031966fb22fe31780addd8fa/LICENSE). |
| ARIS-derived portions of `.agents/skills/research-loop` | [wanshuiyin/Auto-claude-code-research-in-sleep](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/tree/3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d), `3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d`. | MIT. Covered files and copyright notices are listed in [LICENSE.ARIS.txt](.agents/skills/research-loop/LICENSE.ARIS.txt). |

<a id="pytorch3d-bsd-3-clause"></a>

### PyTorch3D BSD-3-Clause notice

The [PyTorch3D permission text](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/LICENSE)
is retained for the rotation-conversion adaptation above.

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

## Third-party weights and body-model assets

<a id="separately-obtained-weights"></a>

### Weights obtained separately

These checkpoints and body models are not redistributed from this repository.
The URLs point to official guidance. Distinguish code licenses from separate
terms for weights and their training data. The Git commits below identify the
terms and guidance consulted; they do not replace checkpoint checksums.

| Asset | Official source and source commit / version | License and terms |
| --- | --- | --- |
| DINO 4-scale Swin-L (`checkpoint0029_4scale_swin.pth`) | [Official model-zoo README](https://github.com/IDEA-Research/DINO/blob/d84a491d41898b3befd8294d1cf2614661fc0953/README.md#model-zoo), [official Google Drive](https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_). Reference commit: `d84a491d41898b3befd8294d1cf2614661fc0953`. | The official repository uses [Apache-2.0](https://github.com/IDEA-Research/DINO/blob/d84a491d41898b3befd8294d1cf2614661fc0953/LICENSE). No separate license specifically for the weights was found in that guidance. The code license does not grant rights to the original data. |
| ViTPose-H (COCO 256×192) | [Official ViTPose model zoo](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/README.md), reference commit: `c050ed29112da7704797cc1a65af0234b525010d`. Distributed for GVHMR as `vitpose-h-multi-coco.pth`. | The official ViTPose repository uses [Apache-2.0](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/LICENSE). Replacing the code does not change the usage terms for training data such as COCO or for the weights. |
| HMR2.0 (`epoch=10-step=25000.ckpt`) | [Official 4D-Humans README](https://github.com/shubham-goel/4D-Humans/blob/efe18deff163b29dff87ddbd575fa29b716a356c/README.md), reference commit: `efe18deff163b29dff87ddbd575fa29b716a356c`. | The repository uses [MIT](https://github.com/shubham-goel/4D-Humans/blob/efe18deff163b29dff87ddbd575fa29b716a356c/LICENSE.md). The official README requires separate registration to obtain SMPL. MIT does not relicense the SMPL model. |
| GVHMR (`gvhmr_siga24_release.ckpt`) | [Official installation guide](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/docs/INSTALL.md), [official Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD). Reference commit: `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1`. | [GVHMR noncommercial terms](src/submodules/vendor/gvhmr/LICENSE). The distribution folder also contains HMR2 and ViTPose, but their respective terms require agreement; this is not a blanket Apache-2.0 permission. |
| CLIP-ReID ViT-B/16 Market-1501 | [Trained models in the official README](https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/README.md#trained-models-and-test-logs), [Market model on the official Google Drive](https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view). Reference commit: `eb1898b72c882875f478bebfc6d41644eece0a5d`. | The official repository uses [MIT](https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/LICENSE). Check the original data terms, including Market-1501, separately. Under the 2026-10-18 agreement, this official Drive is the designated source. |
| AFLink (`AFLink_epoch20.pth`) | [Official StrongSORT README](https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/README.md), [official Google Drive](https://drive.google.com/drive/folders/1Zk6TaSJPbpnqbz1w4kfhkKFCEzQbjfp_). Reference commit: `ee995076da5083e28d0da1f885297df62705ebd7`. | **Usage terms remain unverified in primary sources. The standard pipeline continues to use the weights under the user's decision of 2026-10-19.** The reference code uses GPL-3.0, but terms specifically for the weights have not been confirmed. The provisional usage decision in the [existing NOTICE](src/tasks/person_tracking/strongsort_NOTICE.md) is not redistribution permission. |
| SMPL-X (`SMPLX_NEUTRAL.npz`) | [Official SMPL-X site](https://smpl-x.is.tue.mpg.de/); no applicable Git commit. Check the model distribution version and the agreement accepted at registration. | [SMPL-X Model License](https://smpl-x.is.tue.mpg.de/modellicense.html). Noncommercial scientific research, educational and artistic purposes. The rights are personal and nontransferable; redistribution of all or any part to third parties is not permitted. Each user must register, agree to the terms and obtain the model. |
| SMPL-H (for motion generation) | [Official MANO / SMPL+H site](https://mano.is.tue.mpg.de/); no applicable Git commit. | [MANO / SMPL+H license](https://mano.is.tue.mpg.de/license.html). Noncommercial scientific research, educational and artistic purposes, with redistribution restrictions. Each user must register, agree to the terms and obtain the model. |
| SMPL (for HMR2 and model conversion) | [Official SMPL site](https://smpl.is.tue.mpg.de/); no applicable Git commit. | [SMPL Model License](https://smpl.is.tue.mpg.de/modellicense.html). Noncommercial scientific research, educational and artistic purposes; redistribution of all or any part to third parties is not permitted. |
| DINOv3 ViT-S/16 / ViT-B/16 | [Meta's official application page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/), [official fork README](https://github.com/Motoki0705/dinov3/blob/4add255125e6cbf446c6a43c5f3028aa41b63648/README.md). Reference commit: `4add255125e6cbf446c6a43c5f3028aa41b63648`; distribution names: `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` / `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`. | [DINOv3 License](https://github.com/Motoki0705/dinov3/blob/4add255125e6cbf446c6a43c5f3028aa41b63648/LICENSE.md). Terms for DINO Materials, including code and weights. See the [basis for weight licenses](#weight-license-basis) for their application to project weights derived from DINOv3. |
| SOLIDER-REID / KPR (for comparison) | [Official SOLIDER README](https://github.com/tinyvision/SOLIDER-REID/blob/8c08e1c3255e8e1e51e006bf189e52cc57b009ed/README.md) (`8c08e1c3255e8e1e51e006bf189e52cc57b009ed`), [official KPR README](https://github.com/VlSomers/keypoint_promptable_reidentification/blob/e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa/README.md#download-the-pre-trained-models) (`e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa`). | The repositories use MIT / HL3-LAW-MEDIA-MIL-SOC-SV respectively. Weights are not redistributed from this repository; check official guidance and training-data terms. |
| OSNet (MSMT17, for comparison) | [Official deep-person-reid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO), [official repository](https://github.com/KaiyangZhou/deep-person-reid/tree/f8cd150fdf77e8d9e1ed143b7f308c2c609ded50) (for verification: `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`). | The repository uses [MIT](https://github.com/KaiyangZhou/deep-person-reid/blob/f8cd150fdf77e8d9e1ed143b7f308c2c609ded50/LICENSE). Follow the original distribution's terms for the weights and data such as MSMT17. |

<a id="smpl-auxiliary-assets"></a>

### User-acquired SMPL auxiliary assets

The following five files are not distributed from this repository. Users obtain
them from the [official GVHMR repository](https://github.com/zju3dv/GVHMR/tree/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1)
at pinned commit `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1` and place them, with
the same filenames, in `body_models/gvhmr/` under the CHECKPOINT root.
Use the [placement and SHA-256 verification instructions](src/submodules/README.md#smpl補助資産の取得).
On 2026-10-19, all five official files were confirmed to be byte-identical to the
previously bundled versions. The [acquisition manifest](src/submodules/smpl_assets.py)
is the canonical source for acquisition URLs and expected SHA-256 values.

| Filename (basename at destination) | Official source at the commit above and contents |
| --- | --- |
| `smpl_coco17_J_regressor.pt` | [Official GVHMR file](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt). Regression matrix from SMPL vertices to COCO-17 joints. |
| `smpl_neutral_J_regressor.pt` | [Official GVHMR file](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt). SMPL's 24-joint regression matrix. |
| `smplx2smpl_sparse.pt` | [Official GVHMR file](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smplx2smpl_sparse.pt). Sparse vertex transformation from SMPL-X to SMPL. |
| `smplx_verts437.pt` | [Official GVHMR file](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smplx_verts437.pt). Indices selecting 437 SMPL-X vertices. |
| `smpl_mean_params.npz` | [Official GVHMR file](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/network/hmr2/configs/smpl_mean_params.npz). HMR2 mean pose, shape and camera initial values. |

The [official GVHMR installation guide](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/docs/INSTALL.md)
instructs users to clone the official repository and register to obtain SMPL and
SMPL-X. GVHMR has [terms for educational, research and nonprofit purposes](src/submodules/vendor/gvhmr/LICENSE).
For SMPL/SMPL-X models and derived data, also check
[License Grant / No Distribution in the SMPL Model License](https://smpl.is.tue.mpg.de/modellicense.html)
and the [same sections of the SMPL-X Model License](https://smpl-x.is.tue.mpg.de/modellicense.html).
Both model licenses require prior written permission for provision to third
parties and restrict copying, sharing and distribution of all or any part.
Publication by official GVHMR has been confirmed, but written permission from
the SMPL rights holder for redistribution to third parties, and the generation
history and individual permissions for each auxiliary asset, have not been
confirmed. User acquisition from the official source and permission for
Tennis-Lab to redistribute are separate matters. No conclusion is made that all
indices or mean statistics have the same terms as the models themselves.
Do not add acquired or generated files to this repository's Git history or distributions.

Public information on alternative sources was also checked.
The [official SMPL site](https://smpl.is.tue.mpg.de/) provides guidance on the
neutral model after registration, and the [official SMPL-X model-conversion instructions](https://github.com/vchoutas/smplx/blob/main/transfer_model/README.md#data)
refer to “Model correspondences” and `smplx2smpl_deftrafo_setup.pkl` after
registration. However, the contents of the archives available after registration
have not been inspected. The acquisition procedure therefore does not assume
that generating the required `.pt` files from them would produce identical
files, or that all five files are included there.
The [official SPIN README](https://github.com/nkolot/SPIN#fetch-data) and
[acquisition script](https://github.com/nkolot/SPIN/blob/master/fetch_data.sh)
provide a route for auxiliary data, but this project pins the official GVHMR
files above and does not substitute other versions.
See [pipeline outputs](#pipeline-outputs) for how terms apply to outputs.

<a id="third-party-data"></a>

## Third-party data and papers

Training videos, images, annotations and motion data are not covered by the
Apache-2.0 license for original code. The external datasets below are not
redistributed from this repository. The public sample is covered in a separate
section near the end.

| Item | Source URL and commit / version | License and handling |
| --- | --- | --- |
| TennisCourtDetector images and court annotations | [yastrebksv/TennisCourtDetector README](https://github.com/yastrebksv/TennisCourtDetector/blob/e5cd4f1ce26b15361700d3d89e068cbf0e82749e/README.md). Verification commit: `e5cd4f1ce26b15361700d3d89e068cbf0e82749e`; distribution of 8841 images. No applicable Git commit for the data itself. | No LICENSE. The official description says the images were collected from tournament highlights on YouTube. Permission to redistribute broadcasts, videos or annotations has not been confirmed. |
| TrackNet frames and ball annotations | [yastrebksv/TrackNet README](https://github.com/yastrebksv/TrackNet/blob/730ea17a324321c553c480179caa8ff13e4cf553/README.md). Verification commit: `730ea17a324321c553c480179caa8ff13e4cf553`; distribution of 10 broadcast videos and 19835 frames. No applicable Git commit for the data itself. | No LICENSE. A public repository is not treated as permission to freely reuse or redistribute the data. |
| YouTube-derived videos and original annotations | [YouTube Terms of Service](https://www.youtube.com/static?template=terms); no applicable Git commit for the data. Each dataset's provenance tracks sources and versions per video. | Each video's rights, notice requirements and the Terms of Service apply. Adding original annotations does not remove video rights or contractual terms. See [training data and publication decisions](#training-data-publication) for the scope of the judgment on Article 30-4 of the Copyright Act and contractual issues. |
| ACCAD / AMASS motion | [Official ACCAD site](https://accad.osu.edu/research/motion-lab/mocap-system-and-data), [official AMASS site](https://amass.is.tue.mpg.de/); no applicable Git commit. When using the ACCAD distribution in AMASS, follow its version and agreement. | The official ACCAD Open Motion Project notice specifies [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The AMASS version has a [Dataset License for noncommercial scientific research](https://amass.is.tue.mpg.de/license.html) and redistribution restrictions. Distinguish original data from AMASS conversions into SMPL format. |
| Meiji multicamera videos and annotations | [This project's publication preparation and agreement, #1267](https://github.com/Motoki0705/tennis-lab/issues/1267). Private recordings; no applicable Git commit. The dataset manifest identifies videos and clips. | Consent from the people filmed has been obtained (project records). Outside Apache-2.0. See [sample dataset terms](#sample-dataset-terms) for the public sample's scope and license. |
| `knowledge/Papers/paper-2024-gvhmr/paper.pdf` | [Shen et al., GVHMR, arXiv:2409.06662v1](https://arxiv.org/abs/2409.06662v1); no applicable Git commit; paper version v1. | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The authors' original PDF is retained. [Bundled source attribution](knowledge/Papers/paper-2024-gvhmr/paper.md). The root Apache-2.0 license does not apply to the paper itself. |

<a id="published-artifacts"></a>

## Licenses for published artifacts: project weights and sample dataset

The following publication terms are based on the
[user's decision of 2026-10-19](https://github.com/Motoki0705/tennis-lab/issues/1278).
Research and noncommercial demos are intended uses; this description adds no
usage restrictions to the respective licenses. Other noncommercial uses beyond
research and demos are also permitted when they meet the standard CC terms.

<a id="published-artifacts-hosting"></a>

### Hosting and weight identification

Weights are published in the Hugging Face Model repository
[`Motoki098/tennis-lab-models`](https://huggingface.co/Motoki098/tennis-lab-models).
The following paths are relative to both the Model repository and Tennis-Lab's `ckpt/`.

| Weight file | License |
| --- | --- |
| `ball_detection/i935-mixed-ft-s42-epoch09.ckpt` | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) |
| `court_detection/multiscale_depth3/b863df1f01f0.ckpt` | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) |

The published files are inference checkpoints in Lightning format. They retain
all model weights (including the court backbone), `hyper_parameters` and
Lightning version information, while removing optimizer, scheduler, loop,
callback and training-progress state. Filenames are the same as the training
versions, but the contents are the **published inference versions**.
See the [weight acquisition CLI](src/utils/README.md#published-detector-weights)
for acquisition and SHA checks, and the
[export script](scripts/export_detector_weights.py) for reproduction.

<a id="weight-license-basis"></a>

### Basis for weight licenses

**Ball detection:** The [training configuration](src/tasks/ball_detection/configs/train_meiji_mixed.yaml)
initializes from `ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt`, whose
predecessor is `ckpt/ball_detection/last.ckpt` from the old Colab environment.
**The first model was trained from scratch (user's answer of 2026-10-19)**,
without external pretrained weights. The basis for the first model's
initialization is that answer, not a reinspection of the original first checkpoint.
These project weights are published under CC BY-NC 4.0. Noncommercial copying,
sharing and sharing of adaptations are permitted subject to attribution and the
other terms. Follow [Sections 2 and 3 of the license](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)
for details.

**Court detection:** The checkpoint is a derivative containing DINOv3 ViT-B/16
parameters. The [investigation record](https://github.com/Motoki0705/tennis-lab/issues/1278#issuecomment-6257877131)
confirms that all 188 backbone tensors match the initial weights.
Under [Section 1.b.i of the DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md),
the entire checkpoint is distributed under that agreement with its full text
attached. Its terms also apply to use, including acknowledgment of DINOv3 use in
research publications. The court weights alone are an exception to the project's
“weights are noncommercial” policy; no separate project-specific noncommercial
terms or CC BY-NC 4.0 are added.

<a id="training-data-publication"></a>

### Training data and publication decisions

Ball detection was trained using TrackNet, Meiji and additional ball annotations
from YouTube-derived videos (the [generation configuration](src/tasks/ball_detection/configs/generate_dataset.yaml)
and the training configuration above). Court detection mixes real images from
TennisCourtDetector with synthetic images from Synthetic Court B00–B03 (the
[saved training-run configuration](paper/court_robustness/evidence/training_run_config.yaml)).
See [third-party data and papers](#third-party-data) for findings on the sources
and distribution terms of third-party data.

Training was conducted for information analysis to estimate ball positions and
court geometry. Publication of the ball weights is based on the project's
judgment that Article 30-4 of Japan's Copyright Act applies to the use during
training. That provision allows use without the purpose of enjoying the work,
within the necessary extent, but excludes cases that unreasonably prejudice the
copyright holder's interests. Noncommercial status alone does not determine its
application. See [Article 30-4 of the Copyright Act on e-Gov](https://laws.e-gov.go.jp/law/345AC0000000048#Mp-At_30_4)
for the provision and the [Agency for Cultural Affairs' discussion of AI and copyright](https://www.bunka.go.jp/seisaku/bunkashingikai/chosakuken/pdf/94037901_01.pdf)
for interpretive issues.

This judgment concerns copyright use during training. It does not mean that
contractual terms with distributors have been resolved or that redistribution
of third-party materials is permitted. The acquisition routes, applicable terms
and existence of permissions at the time remain partly unverified (the
[investigation record](https://github.com/Motoki0705/tennis-lab/issues/1278#issuecomment-6257877131)).
The third-party training videos, images and datasets themselves are not included
in this publication.

<a id="sample-dataset-terms"></a>

### Sample dataset terms

The sample is published in the Hugging Face Dataset repository
[`Motoki098/tennis-lab-multiview-sample`](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample).
It covers all three camera videos of Meiji's exported `video_000/clip_000` and
the public `clip.json`, under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en).
See the Meiji row in [third-party data and papers](#third-party-data) for the
record of consent from the people filmed. The distribution's dataset card lists
published filenames, SHA-256 information, capture conditions, synchronization
information and cam0 letterboxing; see the
[sample acquisition CLI](src/utils/README.md#published-multiview-sample) for
acquisition and verification. The public manifest has local absolute paths
removed. Raw videos, annotations and inference results are not included.

Under Sections 2 and 3 of the license, noncommercial sharing is permitted subject
to attribution and the other terms. Creating and reproducing adaptations for
noncommercial purposes is permitted, but sharing them is not licensed.
If crops, annotation overlays, GIFs or similar changes constitute adaptations,
sharing them requires separate permission. Follow the license text for necessary
technical format changes under Section 2(a)(4) and uses that do not require
permission under applicable law. CC does not grant blanket permissions for
portrait, privacy or similar rights.

Suggested attribution: Motoki0705, “Tennis-Lab Multiview Sample”, 2026,
[distribution page](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample),
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

The bundled README asset [`assets/tennis_scene/multiview_reconstruction.gif`](assets/tennis_scene/multiview_reconstruction.gif)
is a GIF separately provided by this project, as licensor, by resizing and
subsampling the sample footage above and combining three views with a 3D
reconstruction. © 2026 Motoki0705. The project's rights in the GIF are provided
under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en),
outside Apache-2.0. Research and demo inspection are intended uses, not added
usage restrictions. Providing this GIF does not authorize third parties to
share arbitrary adaptations they create from the original videos.
The people shown are rendered from SMPL recoveries by GVHMR. SMPL-Body is
attributed to the Max Planck Institute for Intelligent Systems
([Body License / CC BY 4.0](https://smpl.is.tue.mpg.de/bodylicense.html)).
See [pipeline outputs](#pipeline-outputs) for terms on the models and body outputs.

<a id="hf-license-metadata"></a>

### Hugging Face license metadata

Set the following in the model card for the Model repository `Motoki098/tennis-lab-models`:

```yaml
---
license: other
license_name: tennis-lab-per-file-licenses
license_link: LICENSE
---
```

The Model repository's root `LICENSE` contains the scope for each file and the
full text of both licenses. Here, `LICENSE` is the file on Hugging Face, separate
from this repository's Apache License 2.0 for code. `license_name` identifies the
list of terms; it does not create additional custom conditions. Because the
repository contains weights with different terms, do not label the entire Model
repository `cc-by-nc-4.0`. See the [official HF specification](https://huggingface.co/docs/hub/model-cards#specifying-a-license)
for the metadata format.

Set the following in the dataset card for the Dataset repository `Motoki098/tennis-lab-multiview-sample`:

```yaml
---
license: cc-by-nc-nd-4.0
---
```

<a id="pipeline-outputs"></a>

### Pipeline outputs

The publication terms above do not collectively relicense separately used
third-party assets or pipeline outputs. See [third-party weights and body-model assets](#third-party-weights)
for acquisition and usage terms for GVHMR, SMPL, SMPL-X and similar assets, the
decision to continue using AFLink, and SMPL auxiliary-asset acquisition instructions.

No explicit provision was found in the GVHMR [LICENSE](src/submodules/vendor/gvhmr/LICENSE)
requiring all estimated numerical values to carry the same license.
Generation of outputs using SMPL / SMPL-X is subject to the respective Model
Licenses, including restrictions on generating outputs for commercial purposes.
When exported body meshes or similar outputs meet the definition of
[SMPL Body](https://smpl.is.tue.mpg.de/bodylicense.html) or
[SMPL-X Body](https://smpl-x.is.tue.mpg.de/bodylicense.html), a separate CC BY 4.0
permission allows sharing subject to attribution, license links, indication of
changes, and source acknowledgment for papers and media. Body excludes shape
blendshapes and tools that use them to generate bodies; the Body License does
not replace the model's terms. Terms for sharing estimated parameters alone
could not be determined from the primary sources checked.
Distinguish assets such as models and auxiliary matrices from output parameters
and body meshes.
