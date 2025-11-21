# load_dinov3 ユーティリティ仕様

実装: `src/models/utils/load_dinov3.py`

## 目的
- DINOv3 系の ViT モデルをローカル `third_party/dinov3` リポジトリから読み込み、SceneModel 用バックボーンへ差し替えられる状態で返す。
- 事前学習済み重み（`.pth`）の存在チェックと、必須インターフェースの検証を一括で行う。

## API
```python
load_dinov3(
    arch: str = "dinov3_vits16",
    weights_path: str | None = "third_party/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    **hub_kwargs: Any,
) -> nn.Module
```

### 引数
- `arch`: 読み込むアーキテクチャ名（`torch.hub.load` の第 2 引数に相当）。
- `weights_path`: 事前学習済み重みファイルへのパス。`None` の場合は torch.hub のデフォルトに委譲。
- `**hub_kwargs`: `torch.hub.load` に渡す追加パラメータ。`source="local"`, `type=arch` を内部で補完する。

### 戻り値
- `torch.nn.Module`: `third_party/dinov3` からロードした DINOv3 モデル。`get_intermediate_layers` を実装していることを前提とする。

## 実装ノート
- `weights_path` が指定されている場合、`Path(weights_path).expanduser()` で解決し、存在しなければ `FileNotFoundError`
  を送出する。
- `torch.hub.load("third_party/dinov3", arch, ...)` を使用してローカルリポジトリからロードする。`hub_kwargs` に `weights`
  を追加して重みを明示的に指定する。
- 取得したモデルが `get_intermediate_layers` を持たない場合、`AttributeError` を送出する。
- ロード処理で発生した例外は `RuntimeError("Failed to load ...")` にラップして再送出する。呼び出し側で適切に捕捉すること。

## SceneModel との関係
- `src/models/scene_model/build.py` 内の `_load_dinov3_backbone` から利用され、`Dinov3BackboneAdapter` を通じて SceneModel のバックボーンに適合させる。
- Adapter ではモデルの `embed_dim` と SceneModel の `D_model` をチェックし、不一致の場合は `ValueError` を投げる。

## 想定ユースケース
```python
from src.models.utils.load_dinov3 import load_dinov3

backbone = load_dinov3(
    arch="dinov3_vits16",
    weights_path="third_party/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
)
```

- SceneModel の config で `model.backbone.type: dinov3_vits16` のように指定すると、ビルダーがこの関数を呼び出してバックボーンを差し替える。
