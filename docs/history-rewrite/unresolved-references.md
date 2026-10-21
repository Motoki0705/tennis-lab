# 未解決参照と保持する同内容blob

参照更新候補のうち、map外は188箇所 / 40種類。外部repoの由来・内容hash・テスト用短縮値も含むため、すべてが欠落commitという意味ではない。曖昧な短縮形・消滅commit・公開refから到達不能な対応先は、repo本体の更新対象では0件。

| map外候補 | 箇所数 | 最初の位置 |
|---|---:|---|
| `0123456` | 2 | `tests/e2e/skills/test_knowledge_commit_remap.py:97` |
| `0d9a00c1d64b68999e4b2a4ee5ed0d5ac2c3e296` | 1 | `LICENSES.md:55` |
| `1e68f39f` | 2 | `knowledge/nodes/plcs/000003-run-i518-exp10.md:72` |
| `2955b969` | 1 | `knowledge/nodes/ball_refiner/000027-run-i935-context-budget-r28-20261001.md:34` |
| `2955b969b00b506f4d231a2f8cbe634032b1f306` | 1 | `knowledge/nodes/ball_refiner/000027-run-i935-context-budget-r28-20261001.md:21` |
| `3667b024cf53145e79d2a9cd32249e772284404a` | 20 | `knowledge/nodes/blcs/000020-run-i786-normv2-large-cuda-ablation-a-b8-a4-eb32-e100-gpu0.md:50` |
| `3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d` | 2 | `.agents/skills/research-loop/LICENSE.ARIS.txt:3` |
| `4a416b2a07f67518448e669ba5e985668373cdf4` | 6 | `knowledge/nodes/plcs/000079-run-mono3d-plcs-bcast.md:38` |
| `4fdc4d42dd32486344ae5438f9e0517a05f173ec` | 6 | `knowledge/nodes/tennis_scene/000022-run-i931-default-meiji-clip000-20260927.md:37` |
| `5169be60cee3a10c67f77380d296e2677097ad65` | 5 | `knowledge/nodes/blcs/000024-run-i786-normv2-large-cuda-ablation-e-b8-a4-eb32-e100-gpu0.md:51` |
| `51992ecd77eb65e20f7e332695f16508e0e9ac92` | 1 | `knowledge/runs/run-i988-pose-after-review-20261005/first-tracking-run.json:10` |
| `5d56499a04af35863ba1` | 5 | `assets/dataset_review/20261004/synthetic_court/capture-metadata.json:128` |
| `640766639f3b16ae5048f31602e29a2f03b41cc5` | 6 | `knowledge/nodes/ball_detection/000010-run-i618-convnext-v2-scratch.md:23` |
| `6479c948d6329a4ec6258c93585042c80386869c` | 6 | `knowledge/nodes/blcs/000002-run-mono3d-blcs-bcast-v2-yw4.md:31` |
| `689a8acd` | 1 | `knowledge/nodes/ball_refiner/000027-run-i935-context-budget-r28-20261001.md:41` |
| `689a8acd9b7514092c45a265707ee954e5a41e76` | 1 | `knowledge/runs/run-i935-context-budget-r28-20261001/verification.json:6` |
| `6bb51d027b137233f5c520b6fcc4f2ae387a6ba9` | 1 | `src/tasks/person_tracking/deep_ocsort_vendor/NOTICE.md:4` |
| `6d24b4d` | 5 | `knowledge/nodes/plcs/000043-run-i535-asym-deep16-rerun.md:75` |
| `6d24b4dccad36a73bd56a526602757e16bac0275` | 18 | `knowledge/nodes/plcs/000027-run-i525-asym.md:38` |
| `70096b47167ec5af9ff5f74f7d305e1d0f2e784e` | 1 | `knowledge/runs/run-court-hybrid-downstream-migration/verification.json:115` |
| `71426cb84519d4fa716ac9b221d90b00d26b4e63` | 12 | `knowledge/nodes/plcs/000084-run-i801-a2-plcs-d-reference.md:70` |
| `771649c98814914f4a9164cf4ca6df44aabc9041` | 14 | `knowledge/nodes/blcs/000041-run-blcs-residual-v2-asinh-legacy-s42.md:41` |
| `8c08e1c3255e8e1e51e006bf189e52cc57b009ed` | 1 | `knowledge/runs/run-i964-default-solider-cpu-r7-20260930/solider_cpu_parity.json:3` |
| `900e8c4471e80bb6db5cb14c404c1a0766aba5f4` | 4 | `LICENSES.md:41` |
| `9876abc` | 2 | `tests/e2e/skills/test_knowledge_commit_remap.py:187` |
| `9c3764404b11216681a79b58114cd597c01fe549` | 1 | `knowledge/runs/run-court-hybrid-downstream-migration/verification.json:138` |
| `b392bbcbab877172b74c190af32b4dcc12366853` | 11 | `knowledge/nodes/blcs/000025-group-i801-reference-selector-ablation.md:39` |
| `c050ed29112da7704797cc1a65af0234b525010d` | 2 | `src/submodules/vendor/gvhmr/vitpose/__init__.py:3` |
| `caa677e3995f4bafadf5f354813b1944c83bef1c` | 6 | `knowledge/nodes/plcs/000028-run-i525-parameff.md:38` |
| `d407e54cdb903d7082aa4011b2a6f8cb0426c7cc` | 18 | `knowledge/nodes/plcs/000045-run-i539-deep16-chunked.md:38` |
| `d50e513e101f7465d15508dc4c797c328200c78e` | 2 | `knowledge/runs/run-slcs-vitpose-redownload-v1/source.json:3` |
| `d84a491d41898b3befd8294d1cf2614661fc0953` | 8 | `knowledge/runs/run-ball-checkpoint-normalization-meiji-20260923/environment.json:20` |
| `e002a6a8` | 1 | `knowledge/nodes/court_detection/000031-run-court-hybrid-downstream-migration.md:79` |
| `e002a6a801b2cffd654e54a0cd790e5ae973adeb` | 2 | `knowledge/nodes/court_detection/000031-run-court-hybrid-downstream-migration.md:25` |
| `e266a85b2` | 1 | `knowledge/nodes/player_pose_annotation/000003-run-i988-pose-after-review-20261005.md:51` |
| `e266a85b2103e15e7ce66b4abe5221285382b2c8` | 1 | `knowledge/runs/run-i988-pose-after-review-20261005/launch.json:13` |
| `e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa` | 2 | `knowledge/runs/run-i964-kpr-cpu-parity-r8-20260930/parity.json:1` |
| `e5cd4f1ce26b15361700d3d89e068cbf0e82749e` | 3 | `paper/court_robustness/README.md:70` |
| `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1` | 4 | `LICENSES.md:156` |
| `ee995076da5083e28d0da1f885297df62705ebd7` | 2 | `knowledge/runs/run-i964-tracker-linking-r9-20260930/protocol-addendum.md:40` |

ノードのrepro.commitにはmap外の25ノード / 15種類を保持した。別実装へ推測で置換しない。

| 元の参照 | ノード数 |
|---|---:|
| `2955b969b00b506f4d231a2f8cbe634032b1f306` | 1 |
| `3667b024cf53145e79d2a9cd32249e772284404a` | 4 |
| `4a416b2a07f67518448e669ba5e985668373cdf4` | 1 |
| `4fdc4d42dd32486344ae5438f9e0517a05f173ec` | 1 |
| `5169be60cee3a10c67f77380d296e2677097ad65` | 1 |
| `640766639f3b16ae5048f31602e29a2f03b41cc5` | 1 |
| `6479c948d6329a4ec6258c93585042c80386869c` | 1 |
| `6d24b4dccad36a73bd56a526602757e16bac0275` | 3 |
| `71426cb84519d4fa716ac9b221d90b00d26b4e63` | 2 |
| `771649c98814914f4a9164cf4ca6df44aabc9041` | 2 |
| `b392bbcbab877172b74c190af32b4dcc12366853` | 2 |
| `caa677e3995f4bafadf5f354813b1944c83bef1c` | 1 |
| `d407e54cdb903d7082aa4011b2a6f8cb0426c7cc` | 3 |
| `e002a6a801b2cffd654e54a0cd790e5ae973adeb` | 1 |
| `e266a85b2` | 1 |

再現scriptのunverifiable 17件は従来のまま。missingは0件。issue本文のオフライン案にはmap外108箇所とPR refだけの1箇所（#731）を未変更で保持した。

knowledge除去対象と同内容のまま、範囲外のpaper/に残るblobは以下。現存mainの内容を維持するため、この範囲をblob単位の全面削除へ拡張していない。

| blob | 残るパス | 現main使用 |
|---|---|---|
| `21d7033f821520ac7f5a45fd7f22d348846ab6a1` | `paper/court_robustness/images/local02.jpg` | あり |
| `5bad304e034d8a7f4e1bbd11cb6a06e9600b9d00` | `paper/src/court_detection/figures/generated/classical/synthetic_degenerate_fit.png` | なし |
| `674cfb24ba46be64f093d08ea71bd208b01bce6c` | `paper/src/court_detection/figures/generated/classical/real_outlier_overlay.png` | なし |
| `6abd41902eff386b1efaa10a4c719b08c6edc997` | `paper/src/court_detection/figures/generated/classical/synthetic_wrong_fit.png` | なし |
| `79ba80fba5c09e4c370667018def7c4592048c8f` | `paper/src/court_detection/figures/generated/montage_synthetic_test.png` | なし |
| `83b3d348ee5595a3ed8fd37542843cde519cceed` | `paper/court_robustness/evidence/scene_sources/B00_000720.png` | あり |
| `a1c6ed187c9a8bf5a969444ce2f21f52f6867640` | `paper/src/court_detection/figures/generated/classical/synthetic_extreme_overlay.png` | なし |
| `a54a294a8641ba4ebc720579d17b7722c7a234e4` | `paper/court_robustness/images/local03.jpg` | あり |
| `d05e94276fc1331286ee885da69fc531e2f12c2e` | `paper/court_robustness/images/local04.jpg` | あり |
| `dfaeeba01ebc785c4762dde0a56b812465a66dec` | `paper/src/court_detection/figures/generated/montage_real_validation.png` | なし |
| `e80a89b8ae4bcccd733c2ae0f4ca91a585de0ac5` | `paper/court_robustness/images/local01.jpg` | あり |
