# Positional Encoding 仕様

実装: `src/models/scene_model/positional.py:1`

## AbsTimePE（絶対時間）
- 入力: `t_sec[B]`（秒）
- 出力: `[B, 2K]`（cos/sin を結合）
- 周波数: `w ∈ [2π/Thorizon, 2π·0.45·fps]` を対数分割（`K` 本）
- 学習可能パラメータ: `scale[K]`, `phase[K]`
- 射影: `Linear(2K -> D)` を他モジュールで適用
- 例外: `K <= 0`, `Thorizon <= 0` や周波数非正 → `ValueError`

## RelTimePE（相対時間）
- 入力: `delta_sec[B,Q,F]`（参照時刻との差、秒）
- 出力: `[B,Q,F, 2Qrel]`
- 周波数: `w ∈ [rel_wmin, rel_wmax]` を対数分割（`Qrel` 本）
- 例外: `Qrel <= 0` や周波数非正 → `ValueError`

## fixed_2d_sincos（空間グリッド）
- 入力: `H', W', D`
- 出力: `[1, H'*W', D]`
- 生成: Y と X 方向の 1D sin/cos を分割結合して 2D 格子を作成。
- 例外: 非正引数 → `ValueError`

