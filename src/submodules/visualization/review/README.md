# 保存GVHMR / Poseのレビュー

`scene.json`で採用されたGVHMR入力・SMPL parameters・配置vertexと、2D pose / 観測mask / 三角測量COCO17を同じsource frameで確認する読取専用のWeb UI。
モデル推論・学習・annotation更新は行わない。保存vertexのcourt座標へのCPU変換だけを実施する。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/kamimura/projects/tennis-lab/.venv/bin/python \
  -m src.submodules.visualization.review \
  --store /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/i931-default-meiji-clip000-20260927/store \
  --topology /home/kamimura/projects/tennis-lab/ckpt/body_models/smplh/neutral/model.npz \
  --port 8902
# http://127.0.0.1:8902/?person=0&camera=cam1&frame=148
```

`--store`・任意の`--topology`は明示的な絶対パス。topologyはSMPL-H npzの`f`配列のみを読む。未指定なら保存vertexを点群表示する。body model重みはdatasetの件数に含めない。

## 何を見るか

- source frame・人物・cameraを選び、全画面RGBと小さいpose点、人物crop、各jointのconfidenceを確認する。入力viewへのボタンで保存GVHMR requestのviewへ移動できる。
- 2D実観測・GVHMR sample・root / mesh maskのtimelineをクリックして同じframeへ移動する。「次のmesh拒否」でmask=falseのframeへ移動できる。
- 保存SMPL surfaceと有効三角測量jointをcourt m / Z-upで表示する。drag / wheelで見回せる。mesh無効時はmeshを描画せず、有効3D jointだけを表示する。
- GVHMR request / parametersは**完全一致するsource frameだけ**表示する。未sampleを近傍sampleで代用しない。denseな保存配置は別の時間格子として区別する。
- 無効なroot・yaw・reprojectionは数値を表示しない。観測なしを「人物不在」「画面外」の確定ラベルに変換しない。表示thresholdは描画だけに作用する。
- 同じframeの全cameraの実観測（confidence ≥ 0.3のjoint数）と、保存3D jointの拒否codeを確認する。単一viewで2D poseが見えても、多視点の支持不足で3Dが欠測する状態を区別できる。

## データ体系とreader

`src/submodules`は推論専用の移植で、GVHMRの学習dataset / loss / augmentationは移植していない（[vendor README](../../vendor/gvhmr/README.md)）。この画面は学習datasetの検品画面ではなく、実動画から得た**モデル推定の保存artifact**の診断画面であり、独立3D GTは持たない。

readerは以下のschemaを明示的に受け付ける。未知version・checksum不一致・上流artifactの差替え・frame軸不一致は停止する。保存index・配列・元動画は書き換えない。読込後のsourceファイル変更も検出する。

| component | schema / version | 解釈 |
|---|---|---|
| pose_estimation | person_poses v1 | pixel pose、joint confidence、実観測mask |
| player_association | person_identities v2 / v3 | v2は**履歴の固定track ID**、v3はframeごとのID |
| body_view_selection | body_view_selection v1 | GVHMRが受けたpixel pose / box / source frame |
| gvhmr | body_parameters v1 | camera-local translation m、rotation vector rad、betas |
| player_triangulation | player_skeletons v1 | 推定COCO17 m、joint validity |
| body_placement | placed_bodies v1 | root / heading / mesh mask、配置拒否code、保存local vertex |

未保存のoptional component・RGB欠損は明示する。配置rowと人物IDを結びつけるidentityが無い場合は推測せず停止する。
coordinate decoderは`motion_alignment/mesh_placement.py`の保存変換に従い、center / scale適用済みvertexへscaleやhip offsetを二重適用しない。

2026-10-04の現物調査では、`data/tennis_multivew`のclip manifestはファイル76件（Meiji 57件、放送の`dataset` / `curated_ball_v1`各9件、tennis 1件）。公開体系別の実ファイル数であり、独立source数・GVHMR学習量ではない。旧annotation marker 1件はあり、採用component storeの`scene.json`は無かった。`outputs/tennis_scene`にはindex 5件、そのうちGVHMR / placement保存3件（2026-09-27の**同じ**Meiji実clip: 3 camera・1010 frame・2人）。これらは履歴snapshotであり、現在のproduction pipeline実行済みとは扱わない。
`data/ACCAD`の252 npzは現行PLCS single_objectのmotion sourceであり、GVHMR学習datasetと同一視しない。

## 検証

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/kamimura/projects/tennis-lab/.venv/bin/python -m pytest -n 2 \
  tests/unit/submodules/visualization/review \
  tests/unit/utils/configuration/test_inventory.py \
  tests/e2e/development/test_configuration_audit.py
```

単体テスト用fixtureを実データの画像証拠として使わない。実データの撮影・Issueの証拠は[Issue #1012](https://github.com/Motoki0705/tennis-lab/issues/1012)にまとめる。
