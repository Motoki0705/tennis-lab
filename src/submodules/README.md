# `src/submodules` — self-contained model submodules

`third_party/GVHMR` が提供していた前処理モデル群（person tracking / 2D pose /
画像特徴）と GVHMR 本体を、メインの `.venv` で完結して動くように移植した
パッケージです。`src/tennis_scene` の GVHMR コンポーネントはここに依存し、
`third_party/GVHMR` のコードには依存しません（学習済み重みのみ `ckpt/` の
symlink 経由で参照します）。

## 構成

```
src/submodules/
├── models/            # 型付きの推論モデル（下流はここだけ見ればよい）
│   ├── _base/         # BaseInferenceModel: load/unload/predict の共通契約
│   ├── dino/          # DinoPersonDetector: frame -> person boxes
│   ├── tracker/       # YOLO or DINO detections + BoT-SORT -> bbox tracks
│   ├── vitpose/       # ViTPosePose2D: video + boxes -> COCO-17 keypoints
│   ├── hmr2/          # Hmr2FeatureExtractor: video + boxes -> (F, 1024) features
│   └── gvhmr/         # GvhmrMeshRecovery: keypoints+boxes+features -> SMPL-X params
│                      # SmplVertexReconstructor: params -> SMPL vertices (F, 6890, 3)
├── vendor/{gvhmr,dino}/ # 隔離した上流研究コード（型チェック緩和）
├── scripts/           # demo_gvhmr.py: 動画 -> カメラ視点 SMPL レンダリング
└── configs/           # scripts 用 Hydra config
```

## 共通インターフェース（`models/_base`）

すべてのモデルは `BaseInferenceModel[RequestT, ResultT]` を実装します。

- 構築は軽量（重みは触らない）。`load()` は冪等、`unload()` で解放。
- `predict(request) -> result` は自動 load + `torch.no_grad` 込み。
- request / result はモデルごとの frozen dataclass。

```python
from src.submodules.models import DinoPersonTracker, TrackRequest

tracker = DinoPersonTracker(device="auto")
tracks = tracker.predict(TrackRequest(video_path="video.mp4", num_tracks=2))
bbx_xys = tracks.bbx_xys(tracks.track_ids[0])  # 下流モデルの入力形式
```

## デモ

```bash
python -m src.submodules.scripts.demo_gvhmr \
    video_path=data/samples/tennis_clip.mp4 num_tracks=2 max_frames=120
# -> outputs/gvhmr_demo/<stem>_incam.mp4 （SMPL メッシュのカメラ視点オーバーレイ）
```

レンダリングは `src/utils/rendering/mesh_renderer.py`（cv2 painter's algorithm、
pytorch3d 不要）を使用します。

## 必要な資産

| 資産 | 場所 | 備考 |
|---|---|---|
| YOLO / ViTPose / HMR2 / GVHMR 重み | `ckpt/{yolo,vitpose,hmr2,gvhmr}/` | `third_party/GVHMR/inputs/checkpoints` への symlink |
| DINO 4-scale Swin-L | `ckpt/dino/checkpoint0029_4scale_swin.pth` | COCO person (class id 1) のみ使用 |
| SMPL-X 本体 (`SMPLX_NEUTRAL.npz`) | `ckpt/body_models/smplx/` | 要ライセンス登録: https://smpl-x.is.tue.mpg.de/ |
| regressor 等の小物 `.pt` | `vendor/gvhmr/body_model/data/` | リポジトリに同梱 |
| SMPL faces（レンダリング用） | `data/smplh/neutral/model.npz` など | SMPL と SMPL-H は同一トポロジー |

## vendor/ の扱い

`vendor/gvhmr/` と `vendor/dino/` は上流からの移植コードで、mypy strict と ruff の
style ルールを緩和しています（`pyproject.toml` 参照）。**新規コードを vendor に
追加しないこと**。型付きのロジックは `models/` 側に置き、vendor は「上流に近い形の
推論コード」を保つ方針です。詳細な変更点は各vendorのREADMEに記載します。

DINOはcustom CUDA opを必要とします。初回のみ次を実行してください。

```bash
uv pip install -v --no-build-isolation ./src/submodules/vendor/dino/models/dino/ops
```

`DinoPersonTracker` は検出だけをDINOへ変更し、時系列対応付けには既存YOLO
trackingと同じUltralytics BoT-SORT設定を使います。DINO自体からtrack IDが出る
わけではありません。
