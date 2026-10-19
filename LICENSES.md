# ライセンスと出典 / Licenses and attribution

## コードのMIT適用範囲

Tennis-Labの独自コード・文書は、個別の表示があるものを除き、ルートの
[MIT License](LICENSE) に従います。第三者のコードの著作権・ライセンスを置き換えるものではありません。
`third_party/`、移植コード、重み、body model資産、映像・注釈・motion data、論文には
下記の条件があり、リポジトリ全体やパイプライン全体をMITとして扱うことはできません。
MITの第三者コードも、元の著作権表示と許諾文を保持してください。

Original Tennis-Lab code and documentation are MIT licensed unless stated otherwise.
Third-party materials retain their own terms, including restrictions on research use,
commercial use, and redistribution. The root MIT license does not license those materials.

以下は2026-10-19時点の出典の棚卸しです。取込元が記録されているコードはそのcommitを、
submoduleはgitlinkのcommitを示します。「比較用」「確認用」は今回参照した版であり、
過去の取込元やモデル重みの版を示すものではありません。
Git以外で配布される資産のcommitは「該当なし」とし、配布物名や版を記載します。
**許諾未確認の項目を、再配布可能という意味で掲載していません。**

## 第三者のコード

### vendorディレクトリ

`vendor/`・`*_vendor/`という名前の全ディレクトリと、`vendor/`直下のコードパッケージを列挙しています。
テスト側の同名ディレクトリも含めています。

| パス | 出典URL・取込元commit | ライセンス・表示 |
| --- | --- | --- |
| `src/submodules/vendor` | [Tennis-Lab](https://github.com/Motoki0705/tennis-lab/tree/beba5f9a94d598efe52cf7606a9cbdebae6e2f5d/src/submodules/vendor)（確認用 `beba5f9a94d598efe52cf7606a9cbdebae6e2f5d`） | コンテナの `__init__.py` は独自コード、MIT。配下のGVHMRは次行の条件。 |
| `src/submodules/vendor/gvhmr` | [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) の [Motoki0705/GVHMR fork](https://github.com/Motoki0705/GVHMR/tree/900e8c4471e80bb6db5cb14c404c1a0766aba5f4)、取込元 `900e8c4471e80bb6db5cb14c404c1a0766aba5f4` | GVHMR独自の非商用条件。[LICENSE全文](src/submodules/vendor/gvhmr/LICENSE)、[移植内容](src/submodules/vendor/gvhmr/README.md)。内包する第三者部分は下表も参照。 |
| `src/tasks/person_tracking/deep_ocsort_vendor` | [GerardMaggiolino/Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT/tree/6bb51d027b137233f5c520b6fcc4f2ae387a6ba9)、`6bb51d027b137233f5c520b6fcc4f2ae387a6ba9` | MIT。Roger R. Labbe Jr.のKalman実装のMIT表示も保持。[LICENSE](src/tasks/person_tracking/deep_ocsort_vendor/LICENSE)・[NOTICE](src/tasks/person_tracking/deep_ocsort_vendor/NOTICE.md)。 |
| `src/tasks/player_association/appearance/solider_vendor` | [tinyvision/SOLIDER-REID](https://github.com/tinyvision/SOLIDER-REID/tree/8c08e1c3255e8e1e51e006bf189e52cc57b009ed)、`8c08e1c3255e8e1e51e006bf189e52cc57b009ed` | MIT。[LICENSE](src/tasks/player_association/appearance/solider_vendor/LICENSE)・[NOTICE](src/tasks/player_association/appearance/solider_vendor/NOTICE.md)。 |
| `src/tasks/player_association/appearance/kpr_vendor` | [VlSomers/keypoint_promptable_reidentification](https://github.com/VlSomers/keypoint_promptable_reidentification/tree/e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa)、`e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa` | Hippocratic License 3.0 **HL3-LAW-MEDIA-MIL-SOC-SV**。倫理上の条件と派生物の条件を含み、MITの対象外。[LICENSE](src/tasks/player_association/appearance/kpr_vendor/LICENSE)・[NOTICE](src/tasks/player_association/appearance/kpr_vendor/NOTICE.md)。 |
| `tests/unit/submodules/vendor` | [Tennis-Lab](https://github.com/Motoki0705/tennis-lab/tree/beba5f9a94d598efe52cf7606a9cbdebae6e2f5d/tests/unit/submodules/vendor)（確認用 `beba5f9a94d598efe52cf7606a9cbdebae6e2f5d`） | MIT。移植コードを検証する独自テスト。上流コードの配布ディレクトリではありません。 |
| `tests/unit/submodules/vendor/gvhmr` | [Tennis-Lab](https://github.com/Motoki0705/tennis-lab/tree/beba5f9a94d598efe52cf7606a9cbdebae6e2f5d/tests/unit/submodules/vendor/gvhmr)（確認用 `beba5f9a94d598efe52cf7606a9cbdebae6e2f5d`） | MIT。上記テストのGVHMR用パッケージ。 |

GVHMRのLICENSEは、教育・研究・非営利目的のみの使用・複製・改変・配布を許し、
改変物のソース公開と商用利用禁止、著作権・帰属表示等の保持を求めます。
同梱文は [zju3dv/GVHMR `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1` の原文](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/LICENSE)
とbyte単位で同一です。取込元forkの `900e8c4471e80bb6db5cb14c404c1a0766aba5f4` のLICENSEとも同一です。

### GVHMRに内包される第三者部分

以下のパスは `src/submodules/vendor/gvhmr/` からの相対パスです。
直接の取込元は上記GVHMR forkの `900e8c4471e80bb6db5cb14c404c1a0766aba5f4`。
比較先のcommitとファイルをリンクしています。GVHMRの一括した表示だけで、
元の権利者の条件や中間移植の条件が消えるわけではありません。

| パス | 出典URL・比較先commit | ライセンス・確認結果 |
| --- | --- | --- |
| `utils/quaternion.py` | [GVHMRの直接の元ファイル](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/geo/quaternion.py)。[QuaterNet `9d8485b732b0a44b99b6cf4b12d3915703507ddc`](https://github.com/facebookresearch/QuaterNet/blob/9d8485b732b0a44b99b6cf4b12d3915703507ddc/common/quaternion.py)、[VideoPose3D `1afb1ca0f1237776518469876342fc8669d3f6a9`](https://github.com/facebookresearch/VideoPose3D/blob/1afb1ca0f1237776518469876342fc8669d3f6a9/common/quaternion.py) と比較。 | Facebook, Inc.。QuaterNet / VideoPose3Dは **CC BY-NC 4.0**。[QuaterNetの全文](src/submodules/vendor/gvhmr/LICENSE.QuaterNet)を同梱。現版の7関数はGVHMR版とAST一致。`qmul`・`qrot`はQuaterNet版にreshape/contiguousの変更がある。VideoPose3D版とは完全一致しない。GVHMRでの追加・改変にもその条件がある。 |
| `hmr2/*`（特に `vit.py`） | [4D-Humans/HMR2 `efe18deff163b29dff87ddbd575fa29b716a356c`](https://github.com/shubham-goel/4D-Humans/tree/efe18deff163b29dff87ddbd575fa29b716a356c/hmr2)。[ViTPoseのViT `c050ed29112da7704797cc1a65af0234b525010d`](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/models/backbones/vit.py)。 | HMR2は [MIT](src/submodules/vendor/gvhmr/LICENSE.HMR2)、内包するOpenMMLab/ViTPose部分は [Apache-2.0](src/submodules/vendor/gvhmr/LICENSE.ViTPose)。GVHMRのHMR2 ViTは4D-Humans版とbyte一致し、現版との差分は6行追加・20行削除。GPL版のViTを現版の直接の出典とはしない。SMPL mean paramsは後述。 |
| `vitpose/kp2d_utils.py` | [GVHMR元ファイル](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/kpts/kp2d_utils.py)。[ViTPose `c050ed29112da7704797cc1a65af0234b525010d` のtop_down_eval.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/core/evaluation/top_down_eval.py)・[post_transforms.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/core/post_processing/post_transforms.py)。 | OpenMMLab、Apache-2.0（上記全文）。6関数のASTが対応するViTPose版とすべて一致。MMPose v0.29.0（`0d9a00c1d64b68999e4b2a4ee5ed0d5ac2c3e296`）とも比較したが、`keypoints_from_heatmaps`に差があるため、完全一致の出典にはしない。 |
| `vitpose/heatmap_head.py` | [GVHMR元ファイル](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/preproc/vitpose_pytorch/src/vitpose_infer/builder/heads/topdown_heatmap_simple_head.py) は [ViTPose-Pytorch `a9f8025ea676b529a1f498bc2c048c6286ca171b` の同名ファイル](https://github.com/gpastal24/ViTPose-Pytorch/blob/a9f8025ea676b529a1f498bc2c048c6286ca171b/src/vitpose_infer/builder/heads/topdown_heatmap_simple_head.py) とbyte一致。 | 元のOpenMMLab部分はApache-2.0だが、中間配布元は [GPL-3.0](https://github.com/gpastal24/ViTPose-Pytorch/blob/a9f8025ea676b529a1f498bc2c048c6286ca171b/LICENSE)。現版はそのファイルを縮約・改変したもので、GPL-3.0配布元を経由している。Apache-2.0のViTPose公式実装への置き換えを [#1295](https://github.com/Motoki0705/tennis-lab/issues/1295) で行う。 |
| `vitpose/__init__.py` | [GVHMR model_builder.py](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/preproc/vitpose_pytorch/src/vitpose_infer/model_builder.py) と [ViTPose-Pytorch `a9f8025ea676b529a1f498bc2c048c6286ca171b` のmodel_builder.py](https://github.com/gpastal24/ViTPose-Pytorch/blob/a9f8025ea676b529a1f498bc2c048c6286ca171b/src/vitpose_infer/model_builder.py)。 | 同じモデル構成と `VitPoseModel` の移植。GVHMR条件に加え、GPL-3.0配布元を経由した部分がある。Apache-2.0のViTPose公式実装への置き換えを [#1295](https://github.com/Motoki0705/tennis-lab/issues/1295) で行う。 |
| `vitpose/flip_utils.py` | [GVHMR flip_utils.py `900e8c4471e80bb6db5cb14c404c1a0766aba5f4`](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/geo/flip_utils.py) | GVHMR条件。保持した `flip_heatmap_coco17` は元ファイルとAST一致。 |

**ViTPoseの対応方針（ユーザー判断、2026-10-19）:** 上記2ファイルは、
Apache-2.0の [ViTPose公式実装](https://github.com/ViTAE-Transformer/ViTPose/tree/c050ed29112da7704797cc1a65af0234b525010d)
を出典に置き換え、既存checkpointとの互換性を #1295 で検証します。
置き換えは未実施です。ApacheヘッダやGPL全文の追加だけで、GPLとGVHMRの非商用条件の
併存を解消したとは扱いません。

### third_partyのsubmodule（再帰的な依存を含む）

commitはこのリポジトリまたは親submoduleのgitlinkが固定する値です。
各checkoutに同梱されるLICENSEと、ファイル個別の表示を保持してください。

| パス | 出典URL・固定commit | ライセンス |
| --- | --- | --- |
| `third_party/GVHMR` | [Motoki0705/GVHMR](https://github.com/Motoki0705/GVHMR/tree/8a956ca7f4d7071d56f8cc0a987270f302015990)（[zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) のfork）、`8a956ca7f4d7071d56f8cc0a987270f302015990` | [GVHMR非商用条件](https://github.com/Motoki0705/GVHMR/blob/8a956ca7f4d7071d56f8cc0a987270f302015990/LICENSE)。内包するコードには個別条件がある。srcへの取込元commitとは異なる。 |
| `third_party/dinov3` | [Motoki0705/dinov3](https://github.com/Motoki0705/dinov3/tree/4add255125e6cbf446c6a43c5f3028aa41b63648)（[facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) のfork）、`4add255125e6cbf446c6a43c5f3028aa41b63648` | [DINOv3 License](https://github.com/Motoki0705/dinov3/blob/4add255125e6cbf446c6a43c5f3028aa41b63648/LICENSE.md)。MIT・Apache-2.0ではない。配布時の契約添付等の条件がある。 |
| `third_party/DINO` | [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO/tree/d84a491d41898b3befd8294d1cf2614661fc0953)、`d84a491d41898b3befd8294d1cf2614661fc0953` | [Apache-2.0](https://github.com/IDEA-Research/DINO/blob/d84a491d41898b3befd8294d1cf2614661fc0953/LICENSE)。 |
| `third_party/nht` | [Motoki0705/neural-harmonic-textures](https://github.com/Motoki0705/neural-harmonic-textures/tree/9f35cf66e5ff8e6df2b02d68faad667c1eca0ba6)（[nv-tlabs/neural-harmonic-textures](https://github.com/nv-tlabs/neural-harmonic-textures) のfork）、`9f35cf66e5ff8e6df2b02d68faad667c1eca0ba6` | [Apache-2.0](https://github.com/Motoki0705/neural-harmonic-textures/blob/9f35cf66e5ff8e6df2b02d68faad667c1eca0ba6/LICENSE)。 |
| `third_party/nht/gsplat` | [Motoki0705/gsplat](https://github.com/Motoki0705/gsplat/tree/c8d5acd22314c1c1144a30e635e08d9c738a8b39)、`c8d5acd22314c1c1144a30e635e08d9c738a8b39`（[nhtの.gitmodules](https://github.com/Motoki0705/neural-harmonic-textures/blob/9f35cf66e5ff8e6df2b02d68faad667c1eca0ba6/.gitmodules)） | 固定forkの [Apache-2.0](https://github.com/Motoki0705/gsplat/blob/c8d5acd22314c1c1144a30e635e08d9c738a8b39/LICENSE)。他版のgsplatに一括して適用するものではない。 |
| `third_party/nht/gsplat/gsplat/cuda/csrc/third_party/glm` | [g-truc/glm](https://github.com/g-truc/glm/tree/e7970a8b26732f1b0df9690f7180546f8c30e48e)、`e7970a8b26732f1b0df9690f7180546f8c30e48e`（[gsplatの.gitmodules](https://github.com/Motoki0705/gsplat/blob/c8d5acd22314c1c1144a30e635e08d9c738a8b39/.gitmodules)） | [Happy Bunny License または MIT](https://github.com/g-truc/glm/blob/e7970a8b26732f1b0df9690f7180546f8c30e48e/copying.txt)。選択した条件と著作権表示を保持。 |

### その他の移植・参照実装・表示資産

| 対象 | 出典URL・commit / 版 | ライセンス・取扱い |
| --- | --- | --- |
| `src/utils/geometry/rotation_conversions.py` | [PyTorch3D rotation_conversions.py](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/pytorch3d/transforms/rotation_conversions.py)。取込時commitは未記録、リンクは確認用 `88e182f989c80836f4bd744e0d9cb1852762ce01`。 | BSD-3-Clause、Meta Platforms, Inc. and affiliates。移植である旨はファイル先頭に表示。[許諾文は下記](#pytorch3dのbsd-3-clause表示)。 |
| `src/tasks/person_tracking/strongsort*.py` | [StrongSORT論文 arXiv:2202.13514v2](https://arxiv.org/abs/2202.13514v2)。定数の参照先は [dyhBUPT/StrongSORT `ee995076da5083e28d0da1f885297df62705ebd7`](https://github.com/dyhBUPT/StrongSORT/tree/ee995076da5083e28d0da1f885297df62705ebd7)。 | 論文はCC BY 4.0。本repoの数式からの再実装はMIT。GPL-3.0の参照コードはコピーしていない。[出典・相違・検証方法のNOTICE](src/tasks/person_tracking/strongsort_NOTICE.md)。重みは別扱い。 |
| `src/tasks/base/visualization/shared/{three.core.js,three.module.js,OrbitControls.js}` | [mrdoob/three.js r180](https://github.com/mrdoob/three.js/tree/0af9729d0c143a86a1d725d6e2c3ad83301f3f34)、`0af9729d0c143a86a1d725d6e2c3ad83301f3f34`。core/moduleはbyte一致、OrbitControlsはimport先のみ変更。 | MIT、three.js authors。[全文](src/tasks/base/visualization/shared/LICENSE-three.txt)。 |
| `src/tasks/base/visualization/detection/static/icons.mjs` | [Lucide](https://github.com/lucide-icons/lucide)。取得時commitは未記録。確認用 [LICENSE `1ef60af944de207e0d5176905a0759ba12e6ee8d`](https://github.com/lucide-icons/lucide/blob/1ef60af944de207e0d5176905a0759ba12e6ee8d/LICENSE)。 | ISC、Feather由来部分はMIT。[両方の全文・著作権表示](src/tasks/base/visualization/shared/LICENSE-lucide.txt)。 |
| `src/utils/models/kimi_delta_attention.py` | [Kimi Linear論文 arXiv:2510.26692](https://arxiv.org/abs/2510.26692)、[FLA naive recurrence](https://github.com/fla-org/flash-linear-attention/blob/9e97506c338769e8031966fb22fe31780addd8fa/fla/ops/kda/naive.py)（確認用 `9e97506c338769e8031966fb22fe31780addd8fa`、取込commitは該当なし）。 | 論文・recurrenceを参照した本repoの再実装はMIT。参照コードFLAも [MIT](https://github.com/fla-org/flash-linear-attention/blob/9e97506c338769e8031966fb22fe31780addd8fa/LICENSE)。 |
| `.agents/skills/research-loop` のARIS由来部分 | [wanshuiyin/Auto-claude-code-research-in-sleep](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/tree/3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d)、`3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d`。 | MIT。対象ファイルと著作権表示は [LICENSE.ARIS.txt](.agents/skills/research-loop/LICENSE.ARIS.txt)。 |

### PyTorch3DのBSD-3-Clause表示

上記のrotation conversionsの移植に対する
[PyTorch3Dの許諾文](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/LICENSE) を保持します。

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

## 第三者の重み・body model資産

### 別途入手する重み

これらのcheckpointやbody model本体を、本repoから再配布することはありません。
URLは公式の案内先です。コードのLICENSEと、重み・学習元データに関する個別条件は区別してください。
下表のGit commitは条件・案内の確認先であり、checkpointのchecksumに代わるものではありません。

| 資産 | 公式の入手先・出典commit / 版 | ライセンス・条件 |
| --- | --- | --- |
| DINO 4-scale Swin-L（`checkpoint0029_4scale_swin.pth`） | [公式model zooのREADME](https://github.com/IDEA-Research/DINO/blob/d84a491d41898b3befd8294d1cf2614661fc0953/README.md#model-zoo)、[公式Google Drive](https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_)。参照commit `d84a491d41898b3befd8294d1cf2614661fc0953`。 | 公式repoは [Apache-2.0](https://github.com/IDEA-Research/DINO/blob/d84a491d41898b3befd8294d1cf2614661fc0953/LICENSE)。重み専用の別許諾は上記案内に見当たらない。元データの権利までMITで付与するものではない。 |
| ViTPose-H（COCO 256×192） | [ViTPose公式model zoo](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/README.md)、参照commit `c050ed29112da7704797cc1a65af0234b525010d`。GVHMR用の配布名は `vitpose-h-multi-coco.pth`。 | ViTPose公式repoは [Apache-2.0](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/LICENSE)。COCO等の学習元データと、中間移植コードのGPL問題はそれぞれ別の確認対象。 |
| HMR2.0（`epoch=10-step=25000.ckpt`） | [4D-Humans公式README](https://github.com/shubham-goel/4D-Humans/blob/efe18deff163b29dff87ddbd575fa29b716a356c/README.md)、参照commit `efe18deff163b29dff87ddbd575fa29b716a356c`。 | repoは [MIT](https://github.com/shubham-goel/4D-Humans/blob/efe18deff163b29dff87ddbd575fa29b716a356c/LICENSE.md)。公式READMEはSMPL本体を別途登録して取得するよう指定。MITがSMPLモデルを再許諾するものではない。 |
| GVHMR（`gvhmr_siga24_release.ckpt`） | [公式installation](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/docs/INSTALL.md)、[公式Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD)。参照commit `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1`。 | [GVHMR非商用条件](src/submodules/vendor/gvhmr/LICENSE)。配布フォルダにはHMR2・ViTPoseも含まれるが、それぞれの条件への同意が必要で、一括したMIT許諾ではない。 |
| CLIP-ReID ViT-B/16 Market-1501 | [公式READMEのtrained models](https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/README.md#trained-models-and-test-logs)、[公式Google DriveのMarketモデル](https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view)。参照commit `eb1898b72c882875f478bebfc6d41644eece0a5d`。 | 公式repoは [MIT](https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/LICENSE)。Market-1501等の元データ条件は別途確認する。2026-10-18合意により正式な案内先はこの公式Drive。 |
| AFLink（`AFLink_epoch20.pth`） | [StrongSORT公式README](https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/README.md)、[公式Google Drive](https://drive.google.com/drive/folders/1Zk6TaSJPbpnqbz1w4kfhkKFCEzQbjfp_)。参照commit `ee995076da5083e28d0da1f885297df62705ebd7`。 | **利用条件は一次情報で未確認。ユーザー判断（2026-10-19）により標準経路で使い続ける。** 参照コードはGPL-3.0だが、重み専用の条件は確認できていない。[既存NOTICE](src/tasks/person_tracking/strongsort_NOTICE.md) の暫定利用判断は、再配布許諾ではない。 |
| SMPL-X（`SMPLX_NEUTRAL.npz`） | [SMPL-X公式](https://smpl-x.is.tue.mpg.de/)、Git commitは該当なし。登録時のモデル配布版・同意した契約を確認する。 | [SMPL-X Model License](https://smpl-x.is.tue.mpg.de/modellicense.html)。非商用の科学研究・教育・芸術目的。個人に付与される非移転の権利で、第三者への全部・一部の再配布を許可しない。各利用者が登録・同意して取得する。 |
| SMPL-H（motion生成用） | [MANO / SMPL+H公式](https://mano.is.tue.mpg.de/)、Git commitは該当なし。 | [MANO / SMPL+HのLicense](https://mano.is.tue.mpg.de/license.html)。非商用の科学研究・教育・芸術目的、再配布制限。各利用者が登録・同意して取得する。 |
| SMPL（HMR2・モデル変換用） | [SMPL公式](https://smpl.is.tue.mpg.de/)、Git commitは該当なし。 | [SMPL Model License](https://smpl.is.tue.mpg.de/modellicense.html)。非商用の科学研究・教育・芸術目的、第三者への全部・一部の再配布を許可しない。 |
| DINOv3 ViT-S/16・ViT-B/16 | [Meta公式の申請先](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)、[公式fork README](https://github.com/Motoki0705/dinov3/blob/4add255125e6cbf446c6a43c5f3028aa41b63648/README.md)。参照commit `4add255125e6cbf446c6a43c5f3028aa41b63648`、配布名 `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` / `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`。 | [DINOv3 License](https://github.com/Motoki0705/dinov3/blob/4add255125e6cbf446c6a43c5f3028aa41b63648/LICENSE.md)。コードと重みを含むDINO Materialsの条件。派生した自作重みへの適用は #1278 が確定する。 |
| 比較用SOLIDER-REID・KPR | [SOLIDER公式](https://github.com/tinyvision/SOLIDER-REID/blob/8c08e1c3255e8e1e51e006bf189e52cc57b009ed/README.md)（`8c08e1c3255e8e1e51e006bf189e52cc57b009ed`）、[KPR公式](https://github.com/VlSomers/keypoint_promptable_reidentification/blob/e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa/README.md#download-the-pre-trained-models)（`e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa`）。 | repoはそれぞれMIT / HL3-LAW-MEDIA-MIL-SOC-SV。重みを本repoから再配布せず、公式案内・学習元データの条件を確認する。 |
| 比較用OSNet（MSMT17） | [deep-person-reid公式model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)、[公式repo](https://github.com/KaiyangZhou/deep-person-reid/tree/f8cd150fdf77e8d9e1ed143b7f308c2c609ded50)（確認用 `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`）。 | repoは [MIT](https://github.com/KaiyangZhou/deep-person-reid/blob/f8cd150fdf77e8d9e1ed143b7f308c2c609ded50/LICENSE)。重み・MSMT17等のデータ条件は元配布に従う。 |

### 現在同梱されているSMPL派生資産

次の5ファイルはGVHMR forkの `900e8c4471e80bb6db5cb14c404c1a0766aba5f4` からの取得物と
byte単位で同一です。パスは `src/submodules/vendor/gvhmr/` からの相対パスです。
**GVHMR経由で公開されている事実と、SMPL権利者による再配布許諾があることは区別します。**

| パス | 出典URL（上記commit）・内容 | ライセンス・再配布条件の確認状況 |
| --- | --- | --- |
| `body_model/data/smpl_coco17_J_regressor.pt` | [GVHMR元ファイル](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt)。SMPL頂点からCOCO-17関節への回帰行列。 | GVHMRは非商用条件で配布。SMPL派生データとしての個別の再配布許諾・生成履歴は未確認。[SMPL条件](https://smpl.is.tue.mpg.de/modellicense.html) に留意。 |
| `body_model/data/smpl_neutral_J_regressor.pt` | [GVHMR元ファイル](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt)。SMPL関節回帰行列。 | GVHMRの非商用表示だけでは [SMPL条件](https://smpl.is.tue.mpg.de/modellicense.html) にある第三者への一部配布の許可を確認できない。個別許諾未確認。 |
| `body_model/data/smplx2smpl_sparse.pt` | [GVHMR元ファイル](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/body_model/smplx2smpl_sparse.pt)。SMPL-X頂点からSMPL頂点への疎な変換行列。 | GVHMRの非商用表示。[SMPL](https://smpl.is.tue.mpg.de/modellicense.html)・[SMPL-X条件](https://smpl-x.is.tue.mpg.de/modellicense.html) に照らした、この行列の個別再配布許諾未確認。 |
| `body_model/data/smplx_verts437.pt` | [GVHMR元ファイル](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/body_model/smplx_verts437.pt)。SMPL-Xの頂点を選ぶindex。 | GVHMRの非商用表示。indexにどの範囲まで [SMPL-X条件](https://smpl-x.is.tue.mpg.de/modellicense.html) が及ぶか、独立した許諾があるかは未確認。モデル本体と同じ内容とはみなさない。 |
| `hmr2/smpl_mean_params.npz` | [GVHMR元ファイル](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/network/hmr2/configs/smpl_mean_params.npz)。平均pose・shape・cameraの初期値。[ViTPose `c050ed29112da7704797cc1a65af0234b525010d` のtest資産](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/tests/data/smpl/smpl_mean_params.npz) ともbyte一致。 | GVHMR・ViTPoseに同じ配布物があるが、元の統計量の生成データと権利者の個別許諾までは確認できていない。HMR2のMIT、ViTPoseのApache-2.0だけで [SMPL条件](https://smpl.is.tue.mpg.de/modellicense.html) の適用を除外したとは扱わない。 |

[SMPL Model LicenseのLicense Grant / No Distribution](https://smpl.is.tue.mpg.de/modellicense.html) と
[SMPL-X Model Licenseの同項](https://smpl-x.is.tue.mpg.de/modellicense.html) は、
第三者への提供について事前の書面許諾を求め、全部・一部の複製・共有・配布等を制限しています。
これらの資産について、GVHMRのLICENSE以外にその許諾を裏づける一次情報は確認できていません。
**対応方針（ユーザー判断、2026-10-19）:** 上記5点は同梱をやめ、利用者が公式の配布元から
取得する経路へ [#1296](https://github.com/Motoki0705/tennis-lab/issues/1296) で切り替えます。
現時点では同梱されており、削除と取得・生成手順の整備は #1296 で行います。
パイプライン出力への条件の及び方は #1278 が扱います。

## 第三者のデータ・論文

学習元の映像・画像・注釈・motion dataは、独自コードのMIT許諾に含まれません。
下記の外部データ本体は本repoから再配布しません。公開サンプルは末尾の別節で扱います。

| 対象 | 出典URL・commit / 版 | ライセンス・取扱い |
| --- | --- | --- |
| TennisCourtDetectorの画像・コート注釈 | [yastrebksv/TennisCourtDetector README](https://github.com/yastrebksv/TennisCourtDetector/blob/e5cd4f1ce26b15361700d3d89e068cbf0e82749e/README.md)。確認用commit `e5cd4f1ce26b15361700d3d89e068cbf0e82749e`、8841画像の配布。データ自体のGit commitは該当なし。 | LICENSEなし。公式説明ではYouTubeの大会highlightから収集。放送・動画・注釈の再配布許諾を確認できていない。 |
| TrackNetのフレーム・ボール注釈 | [yastrebksv/TrackNet README](https://github.com/yastrebksv/TrackNet/blob/730ea17a324321c553c480179caa8ff13e4cf553/README.md)。確認用commit `730ea17a324321c553c480179caa8ff13e4cf553`、放送動画10本・19835フレームの配布。データ自体のGit commitは該当なし。 | LICENSEなし。リポジトリが公開されていることをデータの自由な再利用・再配布の許諾とは扱わない。 |
| YouTube由来の映像・独自注釈 | [YouTube利用規約](https://www.youtube.com/static?template=terms)、データのGit commitは該当なし。動画ごとの出典・版は各datasetのprovenanceで管理する。 | 動画ごとの権利・表示条件と利用規約がある。独自注釈を付けても映像の権利や契約条件は消えない。学習利用に関する著作権法第30条の4の判断範囲と契約の論点は #1278 が記録する。 |
| ACCAD / AMASS motion | [ACCAD公式](https://accad.osu.edu/research/motion-lab/mocap-system-and-data)、[AMASS公式](https://amass.is.tue.mpg.de/)、Git commitは該当なし。AMASSのACCAD配布を使う場合はその版・契約に従う。 | ACCAD Open Motion Projectの公式表示は [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)。一方、AMASS版には [非商用科学研究用のDataset License](https://amass.is.tue.mpg.de/license.html) と再配布制限がある。元データとAMASSのSMPL形式への変換物を混同しない。 |
| Meijiのマルチカメラ映像・注釈 | [本プロジェクトの公開準備・合意 #1267](https://github.com/Motoki0705/tennis-lab/issues/1267)。非公開収録データ、Git commitは該当なし。動画・clip単位の識別はdatasetのmanifestによる。 | 被撮影者の同意を取得済み（プロジェクト側の記録）。MITの対象外。公開サンプルの範囲・ライセンスは #1273 / #1278 で確定する。 |
| `knowledge/Papers/paper-2024-gvhmr/paper.pdf` | [Shen et al., GVHMR, arXiv:2409.06662v1](https://arxiv.org/abs/2409.06662v1)、Git commitは該当なし、論文版はv1。 | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。著者の原PDFを保持。[同梱の出典表示](knowledge/Papers/paper-2024-gvhmr/paper.md)。論文本体にMITは適用しない。 |

## 公開物（自作の重み・サンプルdataset）のライセンス

#1278 で確定する。
