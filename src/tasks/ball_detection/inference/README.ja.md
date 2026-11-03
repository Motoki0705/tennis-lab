# Ball推論とnative heatmap復号

[English](README.md) · [简体中文](README.zh-CN.md)

`predictor.py` はcheckpoint推論、`trajectory_gate.py` は単一点系列の受理を担当する。
`heatmap_decoding.py` は選択済みnative sigmoid mapをCPUで再復号する。モデルや正解は使用しない。

`refine_peak_cells` は明示したnative peak cellごとに同じ代替局所fitを適用し、
正規化座標とfit状態を返す。座標補正後も無効slotは0で、元evidenceとscoreは変更しない。

設定は `ball_detection.decoding`。既定値の正本は
[pipeline.yaml](../../../tennis_scene/configs/pipeline.yaml)で、Python設定型の全fieldは必須入力。
比較scriptも同じYAMLをcomposeし、試行で明示した差分だけを上書きする。
既定経路はpredictorの配列をそのまま保持する。
他の方式には `ball_detection.subpixel_refine=true` が必要。

| method | 局所推定 |
|---|---|
| `legacy` | 従来の軸別3点log-parabola |
| `axis_logit` | logit域の軸別3点二次fit |
| `quadratic_log` / `quadratic_logit` | 3×3/5×5の交差項を含む二次曲面の最小二乗fit |
| `centroid` | patch最小値を差し引いた重心 |
| `softargmax` | 背景を引いた確率のlogを温度で割った空間softmax。T=1はcentroidと同じ |
| `gaussian` | 背景＋異方性Gaussianの5×5非線形fit。Choleskyでprecisionを正定値にする |

```bash
# GPU推論は通常pipelineと同じく共有training queueへ登録する。
.venv/bin/python -m src.tennis_scene.scripts.run_pipeline \
  'video_paths=[cam0.mp4,cam1.mp4,cam2.mp4]' \
  'camera_ids=[cam0,cam1,cam2]' \
  ball_detection.decoding.method=quadratic_log \
  ball_detection.decoding.patch_size=5
```

座標はx/(W−1), y/(H−1)。格子peak score・dense map・top-Kの意味は維持する。
top-Kは元predictorの復号を保持し、単一観測だけを選択方式に変える。
閾値・gate・consensus・LOESS・三角測量は同じ部品を使い、全設定はcomponent identityへ保存する。

端のpatchは実在するセルだけを使う。二次fitはrank 6・負定値Hessian・格子peakから1セル以内を要求する。
Gaussianも中心を1セル以内に制限し、最大100回評価する。識別不能・非凹・局所範囲外・最適化失敗・
patch不足は格子点を保持し、`FitStatus` に理由を返す。軸別fitは端/退化軸を保持し、部分補正を明示する。
sceneは理由別件数をlogし、replay結果はcamera/frame別のコードを保持する。

`coordinate_correction=stride` は半画素中心の仮説 `((u*(N−1)+0.5)*s−0.5)/(N*s−1)`。
学習済みmodelのバイアスを実証した補正ではない。`affine` は明示した `u*scale_xy+offset_xy` を適用し、
結果を[0,1]へ制限する。正解による係数推定は復号器の外の前半評価だけで行い、正解配列を方式へ渡さない。

float32で飽和したsigmoidの情報は復元できない。logitは[1e−7,1−1e−7]へclipする。
端の重心は切り詰めによる偏りを持ち、Gaussianは単一の山＋定数背景を仮定する。
CPU replayのコピーは64mapずつで、RGB全frameは保持しない。

共通adapterは `src.tennis_scene.evaluation.ball3d.heatmap_decoding.HeatmapDecodingMethod`。
[固定入力runner・採点](../../../tennis_scene/evaluation/README.ja.md)とproduction部品を使う。
clip固有の比較は `knowledge/runs/run-i1559-heatmap-decoding-20261103/` に保存する。
