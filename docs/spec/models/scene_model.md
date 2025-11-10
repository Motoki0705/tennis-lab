# シーンモデル 仕様書

## 1. 目的
単眼動画からボール/プレーヤーの役割・存在確率・3D位置・SMPLパラメータを時系列で推定するエンドツーエンド推論器の仕様を定義する。docs/design/models/scene_model.md で確定したアーキテクチャに対し、実装時のインターフェース/データ契約/挙動を明文化し、将来の改修やテストの基準とする。

## 2. 前提・依存
- フレームは `img_size × img_size` (正方) に前処理済み、かつ `patch_size` で割り切れる。
- PyTorch 2.9 互換。`uv run` 経由で実行する。
- SceneConfig で設定されるハイパーパラメータ群が唯一の外部API。`D_model` は ViT/Temporal/Decoder で共有。
- RoPE 適用のため `D_model / temporal_heads` は偶数。
- `AbsTimePE` の周波数帯は `[2π/Thorizon, 2π·0.45·fps]` を対数で分割。`RelTimePE` の周波数帯は `[rel_wmin, rel_wmax]`。
- 学習/推論ともオフライン処理（全フレームを一括）。

## 3. I/O・データ契約
### 入力
- `frames`: Tensor `[B, T, 3, H, W]` (float32)。H=W=`img_size`。バッチごと独立。

### 出力 (dict)
- `role_logits`: `[B, T, Q, 2]` （プレーヤー役割分類ログits）。
- `exist_conf`: `[B, T, Q, 1]` （存在確率、sigmoid後）。
- `ball_xyz`, `player_xyz`: `[B, T, Q, 3]` 座標 (単位: コート座標系メートル換算想定)。
- `smpl`: `[B, T, Q, smpl_param_dim]`。`smpl_param_dim=0` の場合は空テンソルを返す。

## 4. 処理概要（擬似コード）
```python
frames -> ViTBackbone -> patch_tokens[B,T,Np,D], cls_tokens[B,T,D]
cls_tokens -> TemporalEncoderRoPE -> z[B,T,D]
for t in range(T):
    q_prev = detach(q_prev) if tbptt_detach
    abs_t = AbsTimePE(t/fps) -> Linear -> abs_vec[D]
    q0 = QueryMergeLayer(q_prev, init_queries, ctx=Linear(z[:,t]), abs_vec)
    kv = build_kv(window=[t-k, t+k])
        # patch + fixed_2d_sincos + AbsTimePE(tau)
        # returns kv_feats[B,F,H,W,D], kv_mask[B,F,H,W], delta_sec[B,Q,F]
    for layer in decoder_layers:
        q = TemporalDeformableLayer(q, kv, delta_sec)
            rel = RelTimePE(delta_sec) -> Linear
            offsets/logits -> M sample points per τ
            sample_trilinear(kv_feats, kv_mask, coords, delta_t)
            weights = softmax(logits - λ|Δ|)
            q += Linear(sum(weights * samples))
            q = q + FFN(LN(q))
    store q
decoded = concat_t q -> PredictionHead -> 出力dict
```

## 5. エッジケース / 失敗時挙動
- `patch_tokens` 数が `grid_h × grid_w` と一致しない場合 `ValueError("Number of patches mismatch")`。
- 入力テンソル次元が `[B,T,3,H,W]` でない場合 `ValueError("frames must be [B,T,3,H,W]")`。
- Window 外フレームはゼロ充填＋マスク。`delta_sec` では実時間差 (秒) を返し、欠損スロットは ±k/fps に対応。
- RoPE 用 head_dim が奇数の設定は即時例外 (`ValueError("head_dim must be even for RoPE")`)。
- `smpl_param_dim=0` では `_ZeroProjection` が空テンソルを返し、後段で shape mismatch を起こさない。

## 6. テスト観点
1. **前向き形状**: `SceneModel(cfg)` に乱数動画を通し、各キーの shape と `exist_conf∈[0,1]` を確認。(`tests/unit/models/test_scene_model.py::test_scene_model_forward_shapes`)
2. **KV構築パディング**: `build_kv` が範囲外フレームをマスクし、`delta_sec` に負の秒差を含めること。(`tests/unit/models/test_scene_model.py::test_build_kv_handles_padding`)
3. **型検査**: `uv run mypy src` を必須とし、公称 API の戻り値型を保証。
4. **Lint整合**: `uv run ruff check` を通過し、docstring / import 規約を担保。

## 7. 制約・リスク・代替案
- ViT/Temporal/Decoder の次元が固定であるため、`D_model` 変更は全サブモジュール再初期化が必要。
- デコーダは完全オフラインで逐次処理するため、長尺動画でのメモリ消費が大きい。将来的にオンライン化する場合は状態保存インターフェース追加が必要。
- `sample_trilinear` は `torch.grid_sample` を多用するため GPU 依存。CPU 実行は大幅に遅い。
- `RelTimePE` 周波数帯の設定が窓長と不一致だと学習が不安定になる可能性あり。検証時は `rel_wmin, rel_wmax` をログ計測すること。

## 8. 変更履歴 / 参照
- 2025-11-10: シーンモデル実装に合わせて初版作成。
- 参照: docs/design/models/scene_model.md, src/models/scene_model/*.py
