# Dense heatmap尤度

[English](README.md) · [简体中文](README.zh-CN.md)

保存済みモデルheatmapからコートm単位の3D ballをCPUで直接推定する選択式の方式。
既定の `triangulation` は維持し、`ball_reconstruction.method=dense_likelihood` で選ぶ。
設定は [`pipeline.yaml`](../configs/pipeline.yaml) の `ball_reconstruction.dense_likelihood`、
検証済みの既定値は [`DenseLikelihoodConfig`](config.py) が持つ。

新componentはgate前の `ball_detections` v2を読み、side/alignmentは従来の `ball_points` を使う。
3D推定には2D LOESS・検出の採用mask・正解・正解イベントを渡さない。
評価adapterも同じ `reconstruct_dense` を呼び、[固定入力のrunnerと採点](../evaluation/README.md)を使う。

## 尤度と補間

sigmoid値をFP32 epsilonの半分（約5.96e-8）で対称clipし、有限logitへ戻す。
ちょうど1の画素数を診断へ保存する。飽和で失った曲率を復元する処理ではない。
`quadratic_log` はlog確率を補間するため、温度と支持dropもその領域の単位になる。

`bicubic_logit` はlogitのテンソル積cubic spline。
`quadratic_logit` / `quadratic_log` は最近傍cellの実3×3点へxy交差項を含む二次面をfitする。
端では実在する片側の点だけを使う。近傍が切り替わる境界で二次fitは不連続になり得る。
line searchが失敗したら欠測にし、初期値へ黙って戻さない。双線形補間は使わない。

補間値f、温度T、native画素数Nに対し、viewの目的は
`log(epsilon + (1-epsilon) * exp(f/T - logsumexp(grid/T) + log(N)))`。
信号と一様背景の混合密度を、一様密度に対する比の対数で表し、全viewで和を最大化する。
画面外・camera背後は `log(epsilon)` と勾配0。平坦／低peak mapは支持に数えない。
誤った1viewの罰則が有限になる。未較正のsigmoid値を存在確率とは呼ばない。

`view_temperatures` / `view_backgrounds` は `[camera_id, 値]` の列。
LOVOのslotに依存せず、未指定cameraにはglobal値を使う。
最低2つの情報を持つ画面内viewがnative peakから `support_logit_drop * temperature` 以内で、
ray角が非退化な場合だけ有効にする。

## 探索と時間

`initialization=candidates` はvalidな全候補pairを三角測量し、dense目的で順位付けした
離れた初期値から連続最適化する。最終位置を候補座標へ制限しない。
`initialization=coarse` は候補座標を読まず、コートvolumeの上位格子の周辺を細分化して最適化する。
大域最適の保証はない。範囲と予算を明示し、20万cellを超える粗格子は拒否する。

`temporal_weight>0` は実測支持のあるframe窓を、m/s²の3D加速度Cauchy priorと同時最適化する。
入力3D加速度から急変を検出して窓を切る。欠測も境界になり、補完しない。
未検出の急変の影響もrobust priorで制限するが、雑音で過分割する可能性がある。
窓は非重複で `window_size` 以下。物理軌道・同期fitではない。
収束失敗は理由コード付きの欠測にし、独立推定へ自動fallbackしない。

対数変換は1frame／短い窓だけに行い、全動画のheatmapを複製しない。
volume・勾配・補間作業領域の予算を設定で制限する。

## 結果と検証

`DenseTrajectory` は全source時刻のfloat32位置・有効mask・使用view・対数尤度・急変境界・
収束・飽和数・理由コードを保持する。無効位置と使用viewは0/false。
標準componentは既存ball軌道と共に診断を保存する。2D位置は支持map上の3D modeの投影で、元検出peakとは区別する。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -n 0 \
  tests/unit/tennis_scene/ball_likelihood \
  tests/integration/tennis_scene/test_ball3d_dense_likelihood.py
```

clip固有の調整・コマンド・図・考察は
[`knowledge/runs/run-ball3d-dense-likelihood-20261103`](../../../knowledge/runs/run-ball3d-dense-likelihood-20261103/)に置く。
既定設定へ調整結果を反映せず、採否は人間の統合比較を待つ。
