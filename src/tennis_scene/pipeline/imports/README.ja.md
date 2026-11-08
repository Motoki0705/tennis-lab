# 保存済みscene入力

[English](README.md) · [简体中文](README.zh-CN.md)

`BallAnnotationImporter`はtaskの保存v3注釈を読み、`FixedGeometryImporter`は
保存したコート検出・校正・向き・カメラ配置を固定外部入力として登録する。
いずれも`bind_import` / `publish_import`で新しい不変scene成果物を公開する。
GPT、モデル、校正、学習データ公開は実行せず、workflow/CLIのcompositionは呼出元が担う。

## 契約

| 入力 | scene成果物schema | 意味 |
|---|---|---|
| cameraごとの保存v3 | `ball_annotations` v1 | taskのreader・利用maskを使用 |
| 注釈点の受渡し | `ball_annotation_points` v1 | 座標を維持し、位置利用可・直接観測・推定を分離 |
| 注釈の三角測量 | `ball_annotation_trajectory` v1 | 新しい3D・寄与view・欠測理由・注釈の由来 |
| 固定bundle | 既存の幾何出力schema | 旧依存を出自に残した外部入力 |

[taskのv3契約・reader・usage API](../../../tasks/ball_detection/annotation/video/README.md)
を正本とし、sceneに状態表を複製しない。動画登録時に固定した`VideoAnnotationInput`を、
注釈自身から取り直さずに渡す。元動画bytesのidentity、camera、画像サイズ、FPS、clipの全frame軸と
実PTSを照合する。行は`[0, source.num_frames)`を覆い、元動画自体はそれより長くてもよい。
未処理は拒否し、処理済み`unknown`は欠測として受け入れる。
実bytesと復号PTSの検証は動画登録側が担い、importでは動画を復号しない。

`BallAnnotationOutput`は元のtask文書・path・digest、元PTS、native pixel、taskの点種文字列
（座標なしは空文字）、独立した`position_usable`・`observed`・`estimated`を保存する。
confidenceは位置採用の0/1で、モデルの確率ではない。support、遮蔽、連続性の切れ目、
notes、プレー球の対象定義はtask文書に残る。保存した派生fieldもtask APIと照合する。

## 幾何・点の方針

観測・推定はともに重み**1**とし、注釈座標は変更しない。利用可能な全viewをDLTと
pixel再投影の最小二乗で三角測量する。**平滑化、補間秒数上限、再投影誤差によるview選別、
速度・高さの制限、軌道長によるgateは設けない**。モデル/v2経路の観測限定LOESS・
consensus・再構成方針とschemaは維持する。

位置利用可が2view未満なら、座標0・欠測mask、`missing_reasons="insufficient_views"`と
rejection code 1を保存する。退化rayや非正の深度を含む幾何solverの不成立は
`invalid_multiview_geometry`と共通solver失敗code 7にする。欠測点の寄与viewはfalse。
`position_usable`は1view時刻も含む利用可能入力、`used_views` / `trajectory.inliers`は
有効3Dへ寄与したviewを表す。有効点では利用可能な全viewが寄与する。入力点を黙って除外せず、
再投影誤差は診断だけに使う。3D成果物にも各viewのmask・点種・元注釈文書/参照・方針を残す。

## 固定bundleと利用例

`bundle.json`と`scene.json`を持つ固定した`tennis_lab_ball3d_fixed_inputs.v1`を使う。
形式の起点は[evaluationのbundle reader](../../evaluation/ball3d/bundle.py)。
期待するmanifest digestは明示的に渡す。ファイル/配列checksum、入力identity、clip、camera順序、
校正viewのsource index、コートframe、参照camera、向き/配置の一致を公開前に検証する。
マシン間で元動画pathを移せるが、bytes・camera・時間軸は変えられない。
対象clipはmanifestのclip IDと照合し、元sceneのclip ID・pathは歴史的metadataとして保持する。

`bind_nodes`は幾何nodeにload指定を要求し、依存なしの外部入力にする。
旧side/alignmentを現在のballへ付け替えない。新descriptorにbundle参照、元artifact参照、
元identity・provenance・依存・実装の出自を記録する。元bundleは読取専用で、出力storeは別にする。
旧ball_points/旧3Dをimportせず、出力先に旧成果物があっても新しい注釈schemaで再生成する。

既定の`court_detection="stored_artifact"`は元コート配列を必須とする。
指定の比較bundleはその配列を省略し、descriptorと校正側の静的broadcastを保存している。
`court_detection="verified_calibration_frame0"`を明示すると、frame 0から取り出したNPYの
shape・dtype・bytesが元配列checksumと一致する場合だけ復元する。不一致は失敗とし、
自動fallback・コート再推論は行わない。

```python
from src.tennis_scene.pipeline.imports import BallAnnotationImporter, FixedGeometryImporter
from src.tennis_scene.pipeline.runner import ComponentRunner

# source/store/nodesは呼出元のclip recipeで構築する。
# 幾何とball_detectionのnodeは事前にload指定する。
fixed = FixedGeometryImporter(bundle_json, pinned_bundle_digest,
                             court_detection="verified_calibration_frame0")
balls = BallAnnotationImporter(annotation_paths, registered_inputs)
nodes = fixed.bind_nodes(nodes)
nodes = balls.bind_nodes(nodes)
fixed.publish(nodes, store, source)
balls.publish(nodes, store, source)
runner = ComponentRunner(nodes, store)
runner.run(targets=["ball_triangulation"])
result = runner.output("ball_triangulation")
```

bind時に期待するimport identityも固定する。bundleや注釈revisionの指定を変えた後、
旧採用artifactをloadすると失敗するため、指定したrevisionを先にimportする。
最終scene assemblyは共通のball出力base契約を受け、binderが注釈trajectoryのportを宣言する。

## 旧v2 import

`ball_annotations.py::import_ball_annotations`は[component qualification](../../../../tests/benchmarks/component_pipeline.py)
向けの暫定v2 importとして残す。`<directory>/<camera>_annotations.json`を読み、数値点種を保持し、
observedだけを実測にする。v3の利用規則を付け替えず、旧注釈を変換しない。
importの`origin=import`を記録し、nodeに宣言した上流artifactの依存を引き続き必須とする。

## 検証

```bash
.venv/bin/python -m pytest -n 0 \
  tests/unit/tennis_scene/pipeline/test_imports.py \
  tests/unit/tennis_scene/pipeline/test_video_ball_imports.py \
  tests/unit/tennis_scene/pipeline/components/test_ball_points.py \
  tests/integration/tennis_scene/test_ball_annotation_imports.py
```

小さい非対称カメラと独立に投影した既知3Dにより、保存v3、推定の寄与、欠測、保存復元、
再実行なしの固定幾何、入力/配列改変の拒否、誤差を含むviewの等重みfitを確認する。
実動画の認識精度・受入合格を示すものではない。
