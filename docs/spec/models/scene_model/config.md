# SceneConfig 仕様

`src/models/scene_model/config.py:1` に定義。`SceneModel` の全サブモジュールで共有されるハイパー群。

## フィールド一覧
- 画像/ViT
  - `img_size: int` — 入力画像の一辺ピクセル数（正方）。
  - `patch_size: int` — ViT のパッチサイズ（`img_size % patch_size == 0` 必須）。
  - `vit_depth: int` — ViT エンコーダ層数。
  - `vit_heads: int` — ViT マルチヘッド数。
  - `D_model: int` — 埋め込み次元。全体で共有。

- 時間エンコーダ（RoPE）
  - `temporal_depth: int` — 時間ブロックの層数。
  - `temporal_heads: int` — 時間注意のヘッド数（`D_model % heads == 0` かつ `head_dim` 偶数）。
  - `max_T_hint: int` — 推定最大フレーム長ヒント（キャッシュ/検証用途）。

- 時間位置埋め込み
  - `fps: float` — 映像のフレームレート。
  - `absK: int` — 絶対時間 PE の周波数本数（出力は `2*absK` 次元）。
  - `abs_Thorizon: float` — 最低周波数の周期（秒）。
  - `relQ: int` — 相対時間 PE の周波数本数（出力は `2*relQ` 次元）。
  - `rel_wmin: float` — 相対時間最小角周波数。
  - `rel_wmax: float` — 相対時間最大角周波数。

- デコーダ
  - `num_queries: int` — クエリ数 `Q`。
  - `decoder_layers: int` — デコーダ層数 `L`。
  - `num_points: int` — 1 τ あたりのサンプル点数 `M`。
  - `window_k: int` — 時間窓半径 `k`（窓長は `2*k+1`）。
  - `offset_mode: str` — 変位条件の取り方（`per_tau` または `pooled`）。
  - `tbptt_detach: bool` — 時刻間の再帰状態を TBPTT で detach するか。

- 予測ヘッド
  - `smpl_param_dim: int` — SMPL パラメータ出力次元。0 の場合は空テンソルを返す。

## 妥当性制約と例外
- `img_size % patch_size == 0` でない → `ValueError`（backbone）。
- `D_model % temporal_heads != 0` または `head_dim` 奇数 → `ValueError`（temporal）。
- `absK > 0`, `abs_Thorizon > 0`, `relQ > 0`, `rel_wmin > 0`, `rel_wmax > 0`（positional）。

