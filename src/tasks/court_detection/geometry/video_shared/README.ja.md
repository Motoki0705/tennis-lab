# コート共有の幾何部品

[English](README.md) · [简体中文](README.zh-CN.md)

固定cameraの観測を処理するCPU部品。明示したモデル初期値と元RGBの測定を受け取り、
検出器・参照H・手指定点・camera固有maskを読み込まない。dataset選択、scene設定、
成果物公開、GPU実行の管理は呼出し側の責務。

- `candidates.read_frozen_candidates`は固定manifestの内容とscopeを検査し、モデルの
  採否を保持してcropの平行移動を1回だけ適用する。不採用の有限Hは初期値にできるが、null Hは補わない。
- `observations.measure_frame`は白色stripeの両edgeからgridを提案し、残る線で検査する。
  `endpoints.measure_endings`は有限線分の内外を測る。未観測の端を「ここで終わる」根拠にはしない。
- `fit.fit_common_court`は線・clip・frameを均等に重み付けし、baseline/serviceの端の根拠を要求する。
  失敗時の`SharedFit.matrix`はNone。別の`fitted_matrix`は診断用で、収束だけでは採用しない。
- `consensus.merge_measurements`は実際のstripeを双方向に照合し、元frame/線の群の中で測定を統合する。
  H行列の係数を平均せず、初期値の重複を新しいframeとして数えない。
- `drift.inspect_drift`は線の相関を考慮した不確かさと診断用局所Hから、warning/失敗/検査不能を返す。
  共通Hは置き換えない。3/10/25pxは1920×1080の対角長を基準に拡縮し、
  `SharedPaintConfig`と`PaintConfig`の白線測定の許容値は元画像pixelで指定する。

線の順序とcourt metresは`geometry.homography`に従い、pixelは元画像の非正規化座標。
線の意味、空間支持、時間支持、driftは別々に検査する。幾何の成功は独立した精度検証ではない。
見えない線は外挿であり、静止した白色stripeもネットである可能性がある。

CPU契約テスト:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/unit/tasks/court_detection/geometry/test_video_shared.py
```
