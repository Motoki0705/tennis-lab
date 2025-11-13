# SceneModel 仕様

実装: `src/models/scene_model/model.py:1`

## 目的
- 全サブモジュールを結合したエンドツーエンド推論器。`forward(frames)` で辞書出力を返す。

## 構成要素
- `ViTBackbone(img_size, patch_size, D, vit_depth, vit_heads)`
- `TemporalEncoderRoPE(D, temporal_depth, temporal_heads)`
- `AbsTimePE(absK, fps, abs_Thorizon)` + `Linear(2*absK -> D)`
- `RelTimePE(relQ, rel_wmin, rel_wmax)` + `Linear(2*relQ -> D)`
- `RecurrentTemporalDeformableDecoder(..., abs_pe, abs_proj, rel_pe, rel_proj, fps, img_size, patch_size)`
- `PredictionHead(D, smpl_param_dim)`

## 入出力
- 入力: `frames[B,T,3,H,W]`
- 出力: 予測辞書（`head.md` 参照）

## 擬似コード
```python
patch_tokens, cls_tokens = backbone(frames)          # [B,T,Np,D], [B,T,D]
temporal_tokens = temporal(cls_tokens)               # [B,T,D]
decoded = decoder(patch_tokens, temporal_tokens)     # [B,T,Q,D]
outputs = head(decoded)                              # dict[str, Tensor]
return outputs
```

## 注意
- バックボーンのパッチグリッドは `img_size/patch_size` 由来で、デコーダの空間整形（`H'×W'`）と一致する必要がある。

