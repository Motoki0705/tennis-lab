5モデルのTensorRT FP16統合を `codex/parent-2278-issue-2362-primary` の `fb73f59738096d9ad70e0d5c863ce87c013a36c5` に記録しました。共有 `.venv` で5モデルすべてのエンジンを作成でき、4カメラ・292フレームのsmokeでは全モデルがTensorRTで実行され、4選手それぞれで289フレームの有効なSMPL出力を得ました。再検証ジョブ全体は、単体照合の失敗により終了コード1でした。 [#2362の実行証拠](https://github.com/Motoki0705/tennis-lab/issues/2362#issuecomment-7077053548)

| モデル | 指標 | Phase 1 | 今回 | 判定 |
|---|---|---:|---:|---|
| CLIP-ReID | cosine P05、ID一致 | 0.999987、236/236 | 0.999987、236/236 | 範囲内 |
| ViTPose-H | 座標差 P95 | 0.02471 px | 0.02475 px | 範囲内 |
| HMR2 | 相対L2差 P95 | 0.00401 | 0.00370 | 範囲内 |
| GVHMR | カメラ座標の関節差 P95、系列長145 / 177 | 8.61 / 8.72 mm | 8.35 / 8.81 mm | 範囲内 |
| DINO GridSample | IoU P05、score差 P95 | 0.93252、0.01726 | 0.93056、0.01724 | 両計測で基準未達 |

GVHMRの初回判定は、重力整列座標にも10 mmの閾値を適用していました。判定コードを修正し、保存済みの実測値をPhase 1と同じカメラ座標の基準で再判定しました。DINOの基準はIoU P05 ≥ 0.98、score差 P95 ≤ 0.01です。 [実測記録](https://github.com/Motoki0705/tennis-lab/blob/fb73f59738096d9ad70e0d5c863ce87c013a36c5/knowledge/nodes/tennis_scene/000081-run-i2362-tensorrt-pipeline.md)

各モデルは既定でPyTorch FP32を使い、`backend`・`precision`・`engines`・`batch_size`でTensorRTを明示選択できます。エンジンは専用コマンドで作成し、`outputs/tensorrt/<model>[local path]<shape>[local path]<build-tag>[local path]` に配置します。実行時にONNXとplanのdigest、TensorRT版、GPU、入出力仕様、profileを照合します。TensorRTではこれらの識別情報と設定を `ModelSpec`、component、publicationに反映し、FP32では導入前とのidentity一致を確認しました。後続のバッチ調整は作成済みprofile内なら設定値で行えます。GVHMRのTensorRTはB=1です。

共有 `.venv` に任意依存のTensorRT、ONNX、ONNX Runtime GPUとその依存を追加し、既存185パッケージすべての版を維持しました。CPU回帰は1,840件成功（導入前比29件増、181.09秒）で、pre-commitも通過しました。

**親への判断依頼**。DINOの単体基準未達を明記して現行GridSampleを後続の全クリップE2Eの比較候補に含めるか、#2362でGridSample版の演算精度配分を追加検証して単体基準の達成を求めるか、判断をお願いします。PRはこの判断を受けて作成します。

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}