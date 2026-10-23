# ライセンスと出典

[English](LICENSES.md) | [日本語](LICENSES.ja.md) | [简体中文](LICENSES.zh-CN.md)

<a id="code-license-scope"></a>

## コードのApache-2.0適用範囲

Tennis-Labの独自コード・文書は、個別の表示があるものを除き、ルートの
[Apache License 2.0](LICENSE) に従います。著作権表示と適用範囲の注記は [NOTICE](NOTICE) を参照してください。
第三者のコードの著作権・ライセンスを置き換えるものではありません。
`third_party/`、移植コード、重み、body model資産、映像・注釈・motion data、論文には
下記の条件があり、リポジトリ全体やパイプライン全体をApache-2.0として扱うことはできません。
MITの第三者コードも、元の著作権表示と許諾文を保持してください。

sdist・wheelの `License-Expression` は、[`pyproject.toml`](pyproject.toml) で独自コードと
実際に同梱する第三者コードの条件を `AND` で結合しています。各条件の適用範囲と全文は
本節と以下の第三者コードの一覧に従います。GVHMRの独自条件とKPRの
HL3-LAW-MEDIA-MIL-SOC-SV全体に対応する識別子は
[SPDX License List](https://spdx.org/licenses/) にないため、次の `LicenseRef-` を使います。
SPDXの `Hippocratic-3.0-core` は、KPRが採用する追加条項を含む条件全体を表しません。
この対応表とリンク先のライセンス全文は、`license-files` によって両配布物に同梱します。

| メタデータの識別子 | 適用範囲 | 同梱するライセンス全文 |
| --- | --- | --- |
| `LicenseRef-GVHMR` | `src/submodules/vendor/gvhmr/` のGVHMR由来部分。内包する第三者部分の個別条件は下表を参照。 | [`src/submodules/vendor/gvhmr/LICENSE`](src/submodules/vendor/gvhmr/LICENSE) |
| `LicenseRef-HL3-LAW-MEDIA-MIL-SOC-SV` | `src/tasks/player_association/appearance/kpr_vendor/` のKPR移植コード。 | [`src/tasks/player_association/appearance/kpr_vendor/LICENSE`](src/tasks/player_association/appearance/kpr_vendor/LICENSE) |

独自部分のApache-2.0への移行は、[#1313](https://github.com/Motoki0705/tennis-lab/issues/1313) の対応PRのmerge commitから適用します。
それ以前にMITで配布した版の許諾は取り消さず、過去のtag・履歴は変更しません。

以下は2026-10-19時点の出典の棚卸しです。取込元が記録されているコードはそのcommitを、
submoduleはgitlinkのcommitを示します。「比較用」「確認用」は今回参照した版であり、
過去の取込元やモデル重みの版を示すものではありません。
Git以外で配布される資産のcommitは「該当なし」とし、配布物名や版を記載します。
**許諾未確認の項目を、再配布可能という意味で掲載していません。**

<a id="third-party-code"></a>

## 第三者のコード

<a id="vendor-directories"></a>

### vendorディレクトリ

`vendor/`・`*_vendor/`という名前の全ディレクトリと、`vendor/`直下のコードパッケージを列挙しています。
テスト側の同名ディレクトリも含めています。

| パス | 出典URL・取込元commit | ライセンス・表示 |
| --- | --- | --- |
| `src/submodules/vendor` | [Tennis-Lab](https://github.com/Motoki0705/tennis-lab/tree/37c115e15cf0ba63668d3bc75d468c75a7e8d08c/src/submodules/vendor)（確認用 `37c115e15cf0ba63668d3bc75d468c75a7e8d08c`） | コンテナの `__init__.py` は独自コード、Apache-2.0。配下のGVHMRは次行の条件。 |
| `src/submodules/vendor/gvhmr` | [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) の [Motoki0705/GVHMR fork](https://github.com/Motoki0705/GVHMR/tree/900e8c4471e80bb6db5cb14c404c1a0766aba5f4)、取込元 `900e8c4471e80bb6db5cb14c404c1a0766aba5f4` | GVHMR独自の非商用条件。[LICENSE全文](src/submodules/vendor/gvhmr/LICENSE)、[移植内容](src/submodules/vendor/gvhmr/README.md)。内包する第三者部分は下表も参照。 |
| `src/tasks/person_tracking/deep_ocsort_vendor` | [GerardMaggiolino/Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT/tree/6bb51d027b137233f5c520b6fcc4f2ae387a6ba9)、`6bb51d027b137233f5c520b6fcc4f2ae387a6ba9` | MIT。Roger R. Labbe Jr.のKalman実装のMIT表示も保持。[LICENSE](src/tasks/person_tracking/deep_ocsort_vendor/LICENSE)・[NOTICE](src/tasks/person_tracking/deep_ocsort_vendor/NOTICE.md)。 |
| `src/tasks/player_association/appearance/solider_vendor` | [tinyvision/SOLIDER-REID](https://github.com/tinyvision/SOLIDER-REID/tree/8c08e1c3255e8e1e51e006bf189e52cc57b009ed)、`8c08e1c3255e8e1e51e006bf189e52cc57b009ed` | MIT。[LICENSE](src/tasks/player_association/appearance/solider_vendor/LICENSE)・[NOTICE](src/tasks/player_association/appearance/solider_vendor/NOTICE.md)。 |
| `src/tasks/player_association/appearance/kpr_vendor` | [VlSomers/keypoint_promptable_reidentification](https://github.com/VlSomers/keypoint_promptable_reidentification/tree/e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa)、`e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa` | Hippocratic License 3.0 **HL3-LAW-MEDIA-MIL-SOC-SV**。倫理上の条件と派生物の条件を含み、ルートのApache-2.0の対象外。[LICENSE](src/tasks/player_association/appearance/kpr_vendor/LICENSE)・[NOTICE](src/tasks/player_association/appearance/kpr_vendor/NOTICE.md)。 |
| `tests/unit/submodules/vendor` | [Tennis-Lab](https://github.com/Motoki0705/tennis-lab/tree/37c115e15cf0ba63668d3bc75d468c75a7e8d08c/tests/unit/submodules/vendor)（確認用 `37c115e15cf0ba63668d3bc75d468c75a7e8d08c`） | Apache-2.0。移植コードを検証する独自テスト。上流コードの配布ディレクトリではありません。 |
| `tests/unit/submodules/vendor/gvhmr` | [Tennis-Lab](https://github.com/Motoki0705/tennis-lab/tree/37c115e15cf0ba63668d3bc75d468c75a7e8d08c/tests/unit/submodules/vendor/gvhmr)（確認用 `37c115e15cf0ba63668d3bc75d468c75a7e8d08c`） | Apache-2.0。上記テストのGVHMR用パッケージ。 |

GVHMRのLICENSEは、教育・研究・非営利目的のみの使用・複製・改変・配布を許し、
改変物のソース公開と商用利用禁止、著作権・帰属表示等の保持を求めます。
同梱文は [zju3dv/GVHMR `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1` の原文](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/LICENSE)
とbyte単位で同一です。取込元forkの `900e8c4471e80bb6db5cb14c404c1a0766aba5f4` のLICENSEとも同一です。

<a id="gvhmr-bundled-code"></a>

### GVHMRに内包される第三者部分

以下のパスは `src/submodules/vendor/gvhmr/` からの相対パスです。
ViTPoseのhead・model builderを除き、直接の取込元は上記GVHMR forkの
`900e8c4471e80bb6db5cb14c404c1a0766aba5f4`。
比較先のcommitとファイルをリンクしています。GVHMRの一括した表示だけで、
元の権利者の条件や中間移植の条件が消えるわけではありません。

| パス | 出典URL・比較先commit | ライセンス・確認結果 |
| --- | --- | --- |
| `utils/quaternion.py` | [GVHMRの直接の元ファイル](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/geo/quaternion.py)。[QuaterNet `9d8485b732b0a44b99b6cf4b12d3915703507ddc`](https://github.com/facebookresearch/QuaterNet/blob/9d8485b732b0a44b99b6cf4b12d3915703507ddc/common/quaternion.py)、[VideoPose3D `1afb1ca0f1237776518469876342fc8669d3f6a9`](https://github.com/facebookresearch/VideoPose3D/blob/1afb1ca0f1237776518469876342fc8669d3f6a9/common/quaternion.py) と比較。 | Facebook, Inc.。QuaterNet / VideoPose3Dは **CC BY-NC 4.0**。[QuaterNetの全文](src/submodules/vendor/gvhmr/LICENSE.QuaterNet)を同梱。現版の7関数はGVHMR版とAST一致。`qmul`・`qrot`はQuaterNet版にreshape/contiguousの変更がある。VideoPose3D版とは完全一致しない。GVHMRでの追加・改変にもその条件がある。 |
| `hmr2/*`（特に `vit.py`） | [4D-Humans/HMR2 `efe18deff163b29dff87ddbd575fa29b716a356c`](https://github.com/shubham-goel/4D-Humans/tree/efe18deff163b29dff87ddbd575fa29b716a356c/hmr2)。[ViTPoseのViT `c050ed29112da7704797cc1a65af0234b525010d`](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/models/backbones/vit.py)。 | HMR2は [MIT](src/submodules/vendor/gvhmr/LICENSE.HMR2)、内包するOpenMMLab/ViTPose部分は [Apache-2.0](src/submodules/vendor/gvhmr/LICENSE.ViTPose)。GVHMRのHMR2 ViTは4D-Humans版とbyte一致し、現版との差分は6行追加・20行削除。GPL版のViTを現版の直接の出典とはしない。SMPL mean paramsは後述。 |
| `vitpose/kp2d_utils.py` | [GVHMR元ファイル](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/kpts/kp2d_utils.py)。[ViTPose `c050ed29112da7704797cc1a65af0234b525010d` のtop_down_eval.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/core/evaluation/top_down_eval.py)・[post_transforms.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/core/post_processing/post_transforms.py)。 | OpenMMLab、Apache-2.0（上記全文）。6関数のASTが対応するViTPose版とすべて一致。MMPose v0.29.0（`0d9a00c1d64b68999e4b2a4ee5ed0d5ac2c3e296`）とも比較したが、`keypoints_from_heatmaps`に差があるため、完全一致の出典にはしない。 |
| `vitpose/heatmap_head.py` | [ViTPose公式 `c050ed29112da7704797cc1a65af0234b525010d` のtopdown_heatmap_simple_head.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/models/heads/topdown_heatmap_simple_head.py)・[topdown_heatmap_base_head.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/models/heads/topdown_heatmap_base_head.py)。 | OpenMMLab、[Apache-2.0全文](src/submodules/vendor/gvhmr/LICENSE.ViTPose)。公式版を直接の出典に再構成。MMCV factoryをtorch層へ置換し、検証済み設定で単一feature mapの推論だけを実装。著作権表示・出典・変更の旨をファイル冒頭に保持。 |
| `vitpose/__init__.py` | [ViTPose公式 `c050ed29112da7704797cc1a65af0234b525010d` のtop_down.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/models/detectors/top_down.py)・[ViTPose_huge_coco_256x192.py](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py)。 | OpenMMLab、Apache-2.0（上記全文）。公式TopDownのbackbone→head経路を名前付きtorch sequenceに再構成し、公式ViT-H設定を明示的に組み立てる。checkpointはキー変換なしでstrictに読み込む。著作権表示・出典・変更の旨をファイル冒頭に保持。 |
| `vitpose/flip_utils.py` | [GVHMR flip_utils.py `900e8c4471e80bb6db5cb14c404c1a0766aba5f4`](https://github.com/Motoki0705/GVHMR/blob/900e8c4471e80bb6db5cb14c404c1a0766aba5f4/hmr4d/utils/geo/flip_utils.py) | GVHMR条件。保持した `flip_heatmap_coco17` は元ファイルとAST一致。 |

ViTPoseのhead・model builderは、GPL-3.0のViTPose-Pytorch経由の実装から
上記Apache-2.0公式版を直接の出典とする実装へ置き換えました。
旧実装のコードは同梱せず、合成入力・乱数重みで得た数値fixtureを常設テストに使います。
出典比較と実checkpointによるCPU互換性の検証記録は
[#1295](https://github.com/Motoki0705/tennis-lab/issues/1295) の対応PRを参照してください。

<a id="third-party-submodules"></a>

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

<a id="other-adapted-code-and-assets"></a>

### その他の移植・参照実装・表示資産

| 対象 | 出典URL・commit / 版 | ライセンス・取扱い |
| --- | --- | --- |
| `src/utils/geometry/rotation_conversions.py` | [PyTorch3D rotation_conversions.py](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/pytorch3d/transforms/rotation_conversions.py)。取込時commitは未記録、リンクは確認用 `88e182f989c80836f4bd744e0d9cb1852762ce01`。 | BSD-3-Clause、Meta Platforms, Inc. and affiliates。移植である旨はファイル先頭に表示。[許諾文は下記](#pytorch3d-bsd-3-clause)。 |
| `src/tasks/person_tracking/strongsort*.py` | [StrongSORT論文 arXiv:2202.13514v2](https://arxiv.org/abs/2202.13514v2)。定数の参照先は [dyhBUPT/StrongSORT `ee995076da5083e28d0da1f885297df62705ebd7`](https://github.com/dyhBUPT/StrongSORT/tree/ee995076da5083e28d0da1f885297df62705ebd7)。 | 論文はCC BY 4.0。本repoの数式からの再実装はApache-2.0。GPL-3.0の参照コードはコピーしていない。[出典・相違・検証方法のNOTICE](src/tasks/person_tracking/strongsort_NOTICE.md)。重みは別扱い。 |
| `src/tasks/base/visualization/shared/{three.core.js,three.module.js,OrbitControls.js}` | [mrdoob/three.js r180](https://github.com/mrdoob/three.js/tree/0af9729d0c143a86a1d725d6e2c3ad83301f3f34)、`0af9729d0c143a86a1d725d6e2c3ad83301f3f34`。core/moduleはbyte一致、OrbitControlsはimport先のみ変更。 | MIT、three.js authors。[全文](src/tasks/base/visualization/shared/LICENSE-three.txt)。 |
| `src/tasks/base/visualization/detection/static/icons.mjs` | [Lucide](https://github.com/lucide-icons/lucide)。取得時commitは未記録。確認用 [LICENSE `1ef60af944de207e0d5176905a0759ba12e6ee8d`](https://github.com/lucide-icons/lucide/blob/1ef60af944de207e0d5176905a0759ba12e6ee8d/LICENSE)。 | ISC、Feather由来部分はMIT。[両方の全文・著作権表示](src/tasks/base/visualization/shared/LICENSE-lucide.txt)。 |
| `src/utils/models/kimi_delta_attention.py` | [Kimi Linear論文 arXiv:2510.26692](https://arxiv.org/abs/2510.26692)、[FLA naive recurrence](https://github.com/fla-org/flash-linear-attention/blob/9e97506c338769e8031966fb22fe31780addd8fa/fla/ops/kda/naive.py)（確認用 `9e97506c338769e8031966fb22fe31780addd8fa`、取込commitは該当なし）。 | 論文・recurrenceを参照した本repoの再実装はApache-2.0。参照コードFLAも [MIT](https://github.com/fla-org/flash-linear-attention/blob/9e97506c338769e8031966fb22fe31780addd8fa/LICENSE)。 |
| `.agents/skills/research-loop` のARIS由来部分 | [wanshuiyin/Auto-claude-code-research-in-sleep](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/tree/3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d)、`3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d`。 | MIT。対象ファイルと著作権表示は [LICENSE.ARIS.txt](.agents/skills/research-loop/LICENSE.ARIS.txt)。 |

<a id="pytorch3d-bsd-3-clause"></a>

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

<a id="third-party-weights"></a>

## 第三者の重み・body model資産

<a id="separately-obtained-weights"></a>

### 別途入手する重み

これらのcheckpointやbody model本体を、本repoから再配布することはありません。
URLは公式の案内先です。コードのLICENSEと、重み・学習元データに関する個別条件は区別してください。
下表のGit commitは条件・案内の確認先であり、checkpointのchecksumに代わるものではありません。

| 資産 | 公式の入手先・出典commit / 版 | ライセンス・条件 |
| --- | --- | --- |
| DINO 4-scale Swin-L（`checkpoint0029_4scale_swin.pth`） | [公式model zooのREADME](https://github.com/IDEA-Research/DINO/blob/d84a491d41898b3befd8294d1cf2614661fc0953/README.md#model-zoo)、[公式Google Drive](https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_)。参照commit `d84a491d41898b3befd8294d1cf2614661fc0953`。 | 公式repoは [Apache-2.0](https://github.com/IDEA-Research/DINO/blob/d84a491d41898b3befd8294d1cf2614661fc0953/LICENSE)。重み専用の別許諾は上記案内に見当たらない。コードのライセンスが元データの権利を付与するものではない。 |
| ViTPose-H（COCO 256×192） | [ViTPose公式model zoo](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/README.md)、参照commit `c050ed29112da7704797cc1a65af0234b525010d`。GVHMR用の配布名は `vitpose-h-multi-coco.pth`。 | ViTPose公式repoは [Apache-2.0](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/LICENSE)。コードの置き換えは、COCO等の学習元データや重みの利用条件を変更するものではない。 |
| HMR2.0（`epoch=10-step=25000.ckpt`） | [4D-Humans公式README](https://github.com/shubham-goel/4D-Humans/blob/efe18deff163b29dff87ddbd575fa29b716a356c/README.md)、参照commit `efe18deff163b29dff87ddbd575fa29b716a356c`。 | repoは [MIT](https://github.com/shubham-goel/4D-Humans/blob/efe18deff163b29dff87ddbd575fa29b716a356c/LICENSE.md)。公式READMEはSMPL本体を別途登録して取得するよう指定。MITがSMPLモデルを再許諾するものではない。 |
| GVHMR（`gvhmr_siga24_release.ckpt`） | [公式installation](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/docs/INSTALL.md)、[公式Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD)。参照commit `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1`。 | [GVHMR非商用条件](src/submodules/vendor/gvhmr/LICENSE)。配布フォルダにはHMR2・ViTPoseも含まれるが、それぞれの条件への同意が必要で、一括したApache-2.0許諾ではない。 |
| CLIP-ReID ViT-B/16 Market-1501 | [公式READMEのtrained models](https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/README.md#trained-models-and-test-logs)、[公式Google DriveのMarketモデル](https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view)。参照commit `eb1898b72c882875f478bebfc6d41644eece0a5d`。 | 公式repoは [MIT](https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/LICENSE)。Market-1501等の元データ条件は別途確認する。2026-10-18合意により正式な案内先はこの公式Drive。 |
| AFLink（`AFLink_epoch20.pth`） | [StrongSORT公式README](https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/README.md)、[公式Google Drive](https://drive.google.com/drive/folders/1Zk6TaSJPbpnqbz1w4kfhkKFCEzQbjfp_)。参照commit `ee995076da5083e28d0da1f885297df62705ebd7`。 | **利用条件は一次情報で未確認。ユーザー判断（2026-10-19）により標準経路で使い続ける。** 参照コードはGPL-3.0だが、重み専用の条件は確認できていない。[既存NOTICE](src/tasks/person_tracking/strongsort_NOTICE.md) の暫定利用判断は、再配布許諾ではない。 |
| SMPL-X（`SMPLX_NEUTRAL.npz`） | [SMPL-X公式](https://smpl-x.is.tue.mpg.de/)、Git commitは該当なし。登録時のモデル配布版・同意した契約を確認する。 | [SMPL-X Model License](https://smpl-x.is.tue.mpg.de/modellicense.html)。非商用の科学研究・教育・芸術目的。個人に付与される非移転の権利で、第三者への全部・一部の再配布を許可しない。各利用者が登録・同意して取得する。 |
| SMPL-H（motion生成用） | [MANO / SMPL+H公式](https://mano.is.tue.mpg.de/)、Git commitは該当なし。 | [MANO / SMPL+HのLicense](https://mano.is.tue.mpg.de/license.html)。非商用の科学研究・教育・芸術目的、再配布制限。各利用者が登録・同意して取得する。 |
| SMPL（HMR2・モデル変換用） | [SMPL公式](https://smpl.is.tue.mpg.de/)、Git commitは該当なし。 | [SMPL Model License](https://smpl.is.tue.mpg.de/modellicense.html)。非商用の科学研究・教育・芸術目的、第三者への全部・一部の再配布を許可しない。 |
| DINOv3 ViT-S/16・ViT-B/16 | [Meta公式の申請先](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)、[公式fork README](https://github.com/Motoki0705/dinov3/blob/4add255125e6cbf446c6a43c5f3028aa41b63648/README.md)。参照commit `4add255125e6cbf446c6a43c5f3028aa41b63648`、配布名 `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` / `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`。 | [DINOv3 License](https://github.com/Motoki0705/dinov3/blob/4add255125e6cbf446c6a43c5f3028aa41b63648/LICENSE.md)。コードと重みを含むDINO Materialsの条件。派生した自作重みへの適用は[重みのライセンスの根拠](#weight-license-basis)を参照。 |
| 比較用SOLIDER-REID・KPR | [SOLIDER公式](https://github.com/tinyvision/SOLIDER-REID/blob/8c08e1c3255e8e1e51e006bf189e52cc57b009ed/README.md)（`8c08e1c3255e8e1e51e006bf189e52cc57b009ed`）、[KPR公式](https://github.com/VlSomers/keypoint_promptable_reidentification/blob/e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa/README.md#download-the-pre-trained-models)（`e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa`）。 | repoはそれぞれMIT / HL3-LAW-MEDIA-MIL-SOC-SV。重みを本repoから再配布せず、公式案内・学習元データの条件を確認する。 |
| 比較用OSNet（MSMT17） | [deep-person-reid公式model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)、[公式repo](https://github.com/KaiyangZhou/deep-person-reid/tree/f8cd150fdf77e8d9e1ed143b7f308c2c609ded50)（確認用 `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`）。 | repoは [MIT](https://github.com/KaiyangZhou/deep-person-reid/blob/f8cd150fdf77e8d9e1ed143b7f308c2c609ded50/LICENSE)。重み・MSMT17等のデータ条件は元配布に従う。 |

<a id="smpl-auxiliary-assets"></a>

### 利用者が取得するSMPL補助資産

以下の5点は本repoから配布しません。利用者が
[GVHMR公式リポジトリ](https://github.com/zju3dv/GVHMR/tree/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1)
の固定commit `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1` から取得し、
CHECKPOINT rootの `body_models/gvhmr/` に同じファイル名で置きます。
[配置・SHA-256検証手順](src/submodules/README.md#smpl補助資産の取得)を使ってください。
2026-10-19に、この公式配布の5点と旧同梱版がすべてbyte単位で一致することを確認しました。
取得元・期待するSHA-256の正本は [取得用manifest](src/submodules/smpl_assets.py) です。

| ファイル名（配置先のbasename） | 公式配布元（上記commit）・内容 |
| --- | --- |
| `smpl_coco17_J_regressor.pt` | [GVHMR公式ファイル](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt)。SMPL頂点からCOCO-17関節への回帰行列。 |
| `smpl_neutral_J_regressor.pt` | [GVHMR公式ファイル](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt)。SMPLの24関節回帰行列。 |
| `smplx2smpl_sparse.pt` | [GVHMR公式ファイル](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smplx2smpl_sparse.pt)。SMPL-XからSMPLへの疎な頂点変換。 |
| `smplx_verts437.pt` | [GVHMR公式ファイル](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smplx_verts437.pt)。SMPL-Xの437頂点を選ぶindex。 |
| `smpl_mean_params.npz` | [GVHMR公式ファイル](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/network/hmr2/configs/smpl_mean_params.npz)。HMR2の平均pose・shape・camera初期値。 |

[GVHMR公式の導入手順](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/docs/INSTALL.md)
は公式repoのcloneとSMPL/SMPL-Xの登録取得を案内しています。
GVHMRには [教育・研究・非営利目的の条件](src/submodules/vendor/gvhmr/LICENSE) があり、
SMPL/SMPL-Xモデルと派生データについては
[SMPL Model LicenseのLicense Grant / No Distribution](https://smpl.is.tue.mpg.de/modellicense.html) と
[SMPL-X Model Licenseの同項](https://smpl-x.is.tue.mpg.de/modellicense.html) も確認してください。
両モデルの条件は、第三者への提供について事前の書面許諾を求め、
全部・一部の複製・共有・配布等を制限しています。
公式GVHMRでの公開は確認しましたが、SMPL権利者による第三者への再配布の書面許諾、
各小資産の生成履歴・個別許諾は確認できていません。
公式配布元から利用者が取得することと、Tennis-Labが再配布できることは区別します。
indexや平均統計量すべてにモデル本体と同じ条件が及ぶとの断定もしていません。
取得・生成したファイルを本repoのGitや配布物へ追加しないでください。

代替経路の公開情報も確認しました。
[SMPL公式](https://smpl.is.tue.mpg.de/) は登録後のneutralモデルを案内し、
[SMPL-X公式のモデル変換手順](https://github.com/vchoutas/smplx/blob/main/transfer_model/README.md#data)
は登録後の「Model correspondences」と `smplx2smpl_deftrafo_setup.pkl` を案内しています。
ただし登録後のアーカイブ内容は閲覧できておらず、これらから今回の `.pt` を生成すれば一致する、
あるいは5点すべてがその中に含まれる、という推測は取得手順に採用しません。
[SPIN公式README](https://github.com/nkolot/SPIN#fetch-data)・
[取得スクリプト](https://github.com/nkolot/SPIN/blob/master/fetch_data.sh) には補助データの取得経路が
ありますが、今回使う版は上のGVHMR公式ファイルに固定し、別版への置き換えは行いません。
パイプライン出力への条件の及び方は[パイプライン出力](#pipeline-outputs)を参照してください。

<a id="third-party-data"></a>

## 第三者のデータ・論文

学習元の映像・画像・注釈・motion dataは、独自コードのApache-2.0許諾に含まれません。
下記の外部データ本体は本repoから再配布しません。公開サンプルは末尾の別節で扱います。

| 対象 | 出典URL・commit / 版 | ライセンス・取扱い |
| --- | --- | --- |
| TennisCourtDetectorの画像・コート注釈 | [yastrebksv/TennisCourtDetector README](https://github.com/yastrebksv/TennisCourtDetector/blob/e5cd4f1ce26b15361700d3d89e068cbf0e82749e/README.md)。確認用commit `e5cd4f1ce26b15361700d3d89e068cbf0e82749e`、8841画像の配布。データ自体のGit commitは該当なし。 | LICENSEなし。公式説明ではYouTubeの大会highlightから収集。放送・動画・注釈の再配布許諾を確認できていない。 |
| TrackNetのフレーム・ボール注釈 | [yastrebksv/TrackNet README](https://github.com/yastrebksv/TrackNet/blob/730ea17a324321c553c480179caa8ff13e4cf553/README.md)。確認用commit `730ea17a324321c553c480179caa8ff13e4cf553`、放送動画10本・19835フレームの配布。データ自体のGit commitは該当なし。 | LICENSEなし。リポジトリが公開されていることをデータの自由な再利用・再配布の許諾とは扱わない。 |
| YouTube由来の映像・独自注釈 | [YouTube利用規約](https://www.youtube.com/static?template=terms)、データのGit commitは該当なし。動画ごとの出典・版は各datasetのprovenanceで管理する。 | 動画ごとの権利・表示条件と利用規約がある。独自注釈を付けても映像の権利や契約条件は消えない。学習利用に関する著作権法第30条の4の判断範囲と契約の論点は[学習データと公開判断](#training-data-publication)を参照。 |
| ACCAD / AMASS motion | [ACCAD公式](https://accad.osu.edu/research/motion-lab/mocap-system-and-data)、[AMASS公式](https://amass.is.tue.mpg.de/)、Git commitは該当なし。AMASSのACCAD配布を使う場合はその版・契約に従う。 | ACCAD Open Motion Projectの公式表示は [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)。一方、AMASS版には [非商用科学研究用のDataset License](https://amass.is.tue.mpg.de/license.html) と再配布制限がある。元データとAMASSのSMPL形式への変換物を混同しない。 |
| Meijiのマルチカメラ映像・注釈 | [本プロジェクトの公開準備・合意 #1267](https://github.com/Motoki0705/tennis-lab/issues/1267)。非公開収録データ、Git commitは該当なし。動画・clip単位の識別はdatasetのmanifestによる。 | 被撮影者の同意を取得済み（プロジェクト側の記録）。Apache-2.0の対象外。公開サンプルの範囲・ライセンスは[サンプルdatasetの利用条件](#sample-dataset-terms)を参照。 |
| `knowledge/Papers/paper-2024-gvhmr/paper.pdf` | [Shen et al., GVHMR, arXiv:2409.06662v1](https://arxiv.org/abs/2409.06662v1)、Git commitは該当なし、論文版はv1。 | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。著者の原PDFを保持。[同梱の出典表示](knowledge/Papers/paper-2024-gvhmr/paper.md)。論文本体にルートのApache-2.0は適用しない。 |

<a id="published-artifacts"></a>

## 公開物（自作の重み・サンプルdataset）のライセンス

以下は[2026-10-19のユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/1278)に基づく公開条件です。
研究と非商用デモを想定していますが、この用途説明は各ライセンスに追加の用途制限を設けません。
標準CCの条件を満たす研究・デモ以外の非商用利用も認められます。

<a id="published-artifacts-hosting"></a>

### 公開先と重みの識別情報

重みの公開先はHugging Face Model repo
[`Motoki098/tennis-lab-models`](https://huggingface.co/Motoki098/tennis-lab-models)です。
次のパスはModel repo内、およびTennis-Labの `ckpt/` からの相対パスです。

| 重みのファイル | ライセンス |
| --- | --- |
| `ball_detection/i935-mixed-ft-s42-epoch09.ckpt` | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) |
| `court_detection/multiscale_depth3/b863df1f01f0.ckpt` | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) |

公開版はLightning形式の推論用checkpointで、モデル全重み（コートのbackboneを含む）・
`hyper_parameters`・Lightningのバージョン情報を保持し、optimizer・scheduler・loop・callback・
学習進捗を除去しています。ファイル名は学習版と同じですが、内容は**公開する推論版**です。
取得とSHA検査は[重み取得CLI](src/utils/README.md#published-detector-weights)、
再現は[exportスクリプト](scripts/export_detector_weights.py)を参照してください。

<a id="weight-license-basis"></a>

### 重みのライセンスの根拠

**ボール検出:** [学習設定](src/tasks/ball_detection/configs/train_meiji_mixed.yaml)の初期値は
`ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt`で、その前身は旧Colabの
`ckpt/ball_detection/last.ckpt`です。
**初代学習はスクラッチ（2026-10-19のユーザー回答）**で、外部の事前学習重みは使っていません。
初代の初期化元の根拠は上記の回答であり、初代checkpoint原本の再検証ではありません。
この自作重みをCC BY-NC 4.0で公開します。帰属表示等の条件に従った非商用の複製・共有・改変物の共有が認められます。
詳細は[同ライセンス第2・3条](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)に従ってください。

**コート検出:** checkpointはDINOv3 ViT-B/16のパラメータを含む派生物です。
[調査記録](https://github.com/Motoki0705/tennis-lab/issues/1278#issuecomment-6257877131)では、
backboneの188 tensorすべてが初期重みと一致することを確認しています。
[DINOv3 License第1条b.i](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)に従い、
checkpoint全体を同契約の条件で配布し、契約全文を添付します。
研究発表でのDINOv3使用の明示など、利用時にも同契約の条件が適用されます。
コート重み単体は「重みは非商用」というプロジェクト方針の例外であり、独自の非商用条件やCC BY-NC 4.0を追加しません。

<a id="training-data-publication"></a>

### 学習データと公開判断

ボール検出はTrackNet・Meiji・YouTube由来の追加ボール注釈を使って学習しています
（[生成設定](src/tasks/ball_detection/configs/generate_dataset.yaml)、上記の学習設定）。
コート検出はTennisCourtDetector由来の実画像とSynthetic Court B00–B03の合成画像を混合しています
（[学習runの保存設定](paper/court_robustness/evidence/training_run_config.yaml)）。
第三者データの出典・配布条件の確認結果は[第三者のデータ・論文](#third-party-data)を参照してください。

ボール位置やコート形状を推定する情報解析を目的として学習しました。
ボール重みの公開は、日本の著作権法第30条の4が学習時の利用に適用されるというプロジェクトの判断に基づきます。
同条は非享受目的の利用を必要な範囲で認める一方、著作権者の利益を不当に害する場合を除外しています。
非商用であることだけで適用が決まるものではありません。
根拠条文は[e-Govの著作権法第30条の4](https://laws.e-gov.go.jp/law/345AC0000000048#Mp-At_30_4)、
解釈上の論点は[文化庁「AIと著作権に関する考え方について」](https://www.bunka.go.jp/seisaku/bunkashingikai/chosakuken/pdf/94037901_01.pdf)を参照してください。

この判断は学習時の著作権上の利用に関するもので、配布元との契約条件を解決したことや、第三者素材の再配布許諾を意味しません。
当時の取得経路・適用規約・許可の有無には未確認事項が残っています
（[調査記録](https://github.com/Motoki0705/tennis-lab/issues/1278#issuecomment-6257877131)）。
第三者の学習映像・画像・データセット自体は、今回の公開物に含めません。

<a id="sample-dataset-terms"></a>

### サンプルdatasetの利用条件

公開先はHugging Face Dataset repo
[`Motoki098/tennis-lab-multiview-sample`](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample)です。
対象はMeijiの書き出し済み `video_000/clip_000` の3カメラ動画全体と公開用 `clip.json` で、
ライセンスは[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en)です。
被撮影者の同意に関する記録は[第三者のデータ・論文](#third-party-data)のMeiji行を参照してください。
公開ファイル名・SHA-256・撮影条件・同期情報・cam0のletterboxは配布先のdataset card、
取得と検証は[サンプル取得CLI](src/utils/README.md#published-multiview-sample)を参照してください。
公開用manifestはローカル絶対パスを除いたものです。raw動画・注釈・推論結果は含めません。

同ライセンス第2・3条に従い、帰属表示等の条件を満たす非商用の共有が認められます。
改変物の非商用での作成・複製は認められますが、その共有は許諾されません。
切り抜き・注釈重畳・GIF等が翻案に当たる場合、その共有には別途許諾が必要です。
第2条(a)(4)の必要な技術的形式変換や、法令上許諾を必要としない利用は、ライセンス本文の扱いに従います。
CCは肖像・プライバシー等の権利を一律に許諾するものではありません。

帰属表示例: Motoki0705, “Tennis-Lab Multiview Sample”, 2026,
[配布ページ](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample),
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)。

README用の同梱素材 [`assets/tennis_scene/multiview_reconstruction.gif`](assets/tennis_scene/multiview_reconstruction.gif)
は、ライセンス元である本プロジェクトが上記サンプル映像を縮小・間引きし、3視点と3D再構成を
合成して別途提供するGIFです。© 2026 Motoki0705。GIFに関してプロジェクトが持つ権利は
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en)で提供し、Apache-2.0の対象外です。
研究・デモ確認は想定用途であり、ライセンスに追加の用途制限を設けるものではありません。
このGIFの提供は、第三者が元動画から作った任意の改変物の共有を許諾するものではありません。
人物はGVHMRによるSMPL復元を描画しています。SMPL-Bodyの出典は
Max Planck Institute for Intelligent Systems（[Body License / CC BY 4.0](https://smpl.is.tue.mpg.de/bodylicense.html)）です。
モデル本体・身体出力の条件は[パイプライン出力](#pipeline-outputs)を参照してください。

<a id="hf-license-metadata"></a>

### Hugging Faceのlicenseメタデータ

Model repo `Motoki098/tennis-lab-models` のmodel cardには次を設定します。

```yaml
---
license: other
license_name: tennis-lab-per-file-licenses
license_link: LICENSE
---
```

Model repoのルート `LICENSE` にファイルごとの適用範囲と両ライセンスの全文を収録します。
ここでの `LICENSE` はHugging Face側のファイルであり、コード用の本repoのApache License 2.0とは別です。
`license_name` は条件一覧の識別名で、独自の追加条件を作るものではありません。
異なる条件の重みを含むため、Model repo全体を `cc-by-nc-4.0` と表示しません。
メタデータの書式は[HF公式仕様](https://huggingface.co/docs/hub/model-cards#specifying-a-license)を参照してください。

Dataset repo `Motoki098/tennis-lab-multiview-sample` のdataset cardには次を設定します。

```yaml
---
license: cc-by-nc-nd-4.0
---
```

<a id="pipeline-outputs"></a>

### パイプライン出力

上記の公開条件は、別途使用する第三者資産やパイプライン出力を一括して再許諾するものではありません。
GVHMR・SMPL・SMPL-X等の取得・利用条件、AFLinkの継続利用判断、SMPL補助資産の取得手順は、
[第三者の重み・body model資産](#third-party-weights)を参照してください。

GVHMRの[LICENSE](src/submodules/vendor/gvhmr/LICENSE)には、推定した数値すべてに同じライセンスを付すという明示規定は確認できません。
SMPL／SMPL-Xを使用した出力の生成には、各Model Licenseの条件（商用目的の成果物生成の制限を含む）が適用されます。
書き出した身体mesh等が[SMPL Body](https://smpl.is.tue.mpg.de/bodylicense.html)または
[SMPL-X Body](https://smpl-x.is.tue.mpg.de/bodylicense.html)の定義を満たす場合には、
別途CC BY 4.0の許諾があり、帰属表示・ライセンスリンク・変更表示と、論文・メディア向けの出典表示に従って共有できます。
Bodyはshape blendshapeやそれを使って身体を生成するツールを含まず、Body Licenseがモデル本体の条件を置き換えるものではありません。
推定パラメータのみの共有条件は、確認した一次情報からは確定できていません。
モデル本体・補助行列等の資産と、出力したパラメータ・身体meshを区別してください。
