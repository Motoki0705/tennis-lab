# dinov3_patch_tracker ユーティリティ仕様

実装: `src/tools/dinov3_patch_tracker.py`
CLI: `src/cli/tools/dinov3_patch_tracker.py`
シェル: `scripts/tools/run_dinov3_patch_tracker.sh`

## 目的
- `third_party/dinov3` リポジトリからロードした DINOv3 ViT を用いて、
  単一オブジェクトの **トラッキング + セグメンテーション** を同時に行う。
- ユーザーが最初のフレームで指定したバウンディングボックスから
  **テンプレートパッチ埋め込み** を構築し、以降のフレームでは
  パッチトークンとの **コサイン類似度マップ** に基づいて対象領域を推定する。

## コア API

```python
from src.tools.dinov3_patch_tracker import Dinov3PatchTracker, TrackerConfig

cfg = TrackerConfig(
    arch="dinov3_vits16",
    weights_path="third_party/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    img_size=224,
    threshold=0.6,
    template_update_alpha=0.0,
    device="cuda",
)
tracker = Dinov3PatchTracker(cfg)

# 1. テンプレート初期化
tracker.set_template(frame_bgr, bbox_xywh=(x, y, w, h))

# 2. 各フレームでトラッキング + セグメンテーション
mask_uint8, bbox_tracked, sim_map = tracker.track(frame_bgr)
```

### TrackerConfig

- `arch`: `load_dinov3` に渡す DINOv3 アーキテクチャ名。
- `weights_path`: 事前学習済み重みファイルへのパス。`None` にすると `torch.hub` のデフォルトに委譲。
- `img_size`: DINOv3 入力解像度（通常 224）。フレームは内部で正方形にリサイズされる。
- `threshold`: コサイン類似度の閾値。パッチ類似度が `>= threshold` の領域を前景とみなす。
- `template_update_alpha`: テンプレート更新の EMA 係数。`0.0` で更新なし、`0.1` などで徐々に追従。
- `device`: `"cuda"` or `"cpu"`。

### Dinov3PatchTracker.set_template

```python
set_template(frame_bgr: np.ndarray, bbox_xywh: tuple[float, float, float, float]) -> None
```

- `frame_bgr`: OpenCV 形式の `H x W x 3` uint8 BGR 画像。
- `bbox_xywh`: 元画像座標系での `(x, y, w, h)`。

処理内容:
- 画像を `img_size x img_size` にリサイズし、ImageNet の平均・分散で正規化。
- DINOv3 の `get_intermediate_layers(..., n=1, reshape=True)` から最終層の
  パッチ特徴マップ `[1, C, H_p, W_p]` を取得。
- BBox をパッチグリッドに写像し、該当パッチの平均ベクトルをテンプレートとする。
- テンプレートベクトルは L2 正規化され、以後のフレームで再利用される。

### Dinov3PatchTracker.track

```python
track(frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None, np.ndarray]
```

戻り値:
- `mask_uint8`: 元解像度 `H x W` の 0/255 バイナリマスク (uint8)。
- `bbox_tracked`: マスクの最大連結成分に対する `(x, y, w, h)`。前景がなければ `None`。
- `sim_map`: パッチ空間 `H_p x W_p` の類似度マップ (float32, [-1, 1])。

処理フロー:
1. フレームを前処理 (`img_size` リサイズ & 正規化) して DINOv3 に入力。
2. 最終層パッチ特徴 `[1, C, H_p, W_p]` を取得し、空間次元を平坦化。
3. 各パッチ特徴を L2 正規化し、テンプレートベクトルとの内積で
   コサイン類似度を計算 → `H_p x W_p` の `sim_map`。
4. `sim_map >= threshold` を閾値処理してパッチマスクを得る。
5. パッチマスクを `img_size x img_size`、さらに元解像度 `H x W` へ
   最近傍補間でアップサンプリングし、2値マスクに変換。
6. `cv2.findContours` を使って最大連結成分を求め、その外接矩形を
   `bbox_tracked` として返す。
7. `template_update_alpha > 0` の場合、現在フレームの BBox で抽出した
   パッチトークンからテンプレートを EMA 更新する。

## CLI

実装: `src/cli/tools/dinov3_patch_tracker.py`

```bash
uv run python src/cli/tools/dinov3_patch_tracker.py --video-path path/to/video.mp4 \
    --threshold 0.6 \
    --template-update-alpha 0.0 \
    --device cuda
```

- 起動すると最初のフレームが表示され、OpenCV の `selectROI` で
  対象オブジェクトの BBox を指定する。
- ENTER キーで確定するとテンプレートが初期化され、その後全フレームに対して
  トラッキング + セグメンテーションを実行する。
- 結果は **元フレームにマスクと BBox をオーバーレイした動画** として
  書き出される。

主な引数:
- `--video-path`: 入力動画パス（必須）。
- `--output-path`: 出力動画パス。省略時は `input_stem_dinov3_track.mp4`。
- `--arch`: DINOv3 アーキテクチャ名（`dinov3_vits16` など）。
- `--weights-path`: ローカル `.pth` 重みパス。
- `--threshold`: 類似度閾値。
- `--template-update-alpha`: テンプレート EMA 係数。
- `--device`: `cuda` / `cpu`。

## 実行用シェル

実装: `scripts/tools/run_dinov3_patch_tracker.sh`

```bash
./scripts/tools/run_dinov3_patch_tracker.sh --video-path path/to/video.mp4
```

内部で以下を実行する:

```bash
uv run python src/cli/tools/dinov3_patch_tracker.py "$@"
```

## 想定ユースケースと拡張

- テニス動画から特定プレイヤー、ラケット、ボールなど任意の領域を
  パッチベースでトラッキングしたいとき。
- SceneModel とは独立した、軽量な DINOv3 ベース解析ツールとして利用可能。

将来的な拡張案:
- 複数オブジェクト（複数テンプレート）の同時トラッキング。
- CRF / GrabCut などを組み合わせたマスクの高品質化。
- Web UI (Gradio / Streamlit) による操作性向上。
