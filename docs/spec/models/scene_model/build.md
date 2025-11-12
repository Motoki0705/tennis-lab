# Build ユーティリティ 仕様

実装: `src/models/scene_model/build.py:1`

## build_scene_model
```python
def build_scene_model(model_cfg, debug_cfg=None) -> SceneModel
```
- `model_cfg`: OmegaConf/Mapping を許容。`{"scene": {...}, "backbone": {...}, "head": {...}}` 想定。
- `debug_cfg.minimal=True` の場合、学習/検証用にミニ構成で上書き（画像 64px, D=64 等）。
- 戻り値: `SceneModel` インスタンス。必要に応じてバックボーン/ヘッドを差し替え。

## バックボーン差し替え
- `_build_backbone(cfg, default, debug)`
  - `debug.minimal` の場合は `default` をそのまま使用。
  - `type` が `dinov3*` の場合、`_load_dinov3_backbone` で torch.hub から読込。
  - 失敗（例: 重み未発見, API 不一致）は警告を出し `default` にフォールバック。

## DINOv3 アダプタ
- `Dinov3BackboneAdapter(backbone)`
  - `get_intermediate_layers(..., n=1, return_class_token=True)` を要求。
  - `[BT, Np, D], [BT, D]` を `[B,T,...]` に整形して返す。
  - `embed_dim/num_features` が検出できない場合は `ValueError`。

## ヘッド差し替え
- `_build_head(cfg, default_head, model_dim)`
  - `type == "bbox"` の場合、`BBoxHead(model_dim, num_classes)` を返す。
  - それ以外はデフォルトの `PredictionHead` を利用。

## 付帯
- `_to_dict` は `DictConfig | Mapping | dict | None` を平易な `dict` に変換。

## 例外/注意
- `third_party/dinov3` を `torch.hub.load(..., source="local")` で参照。重みパス未存在時は警告。

