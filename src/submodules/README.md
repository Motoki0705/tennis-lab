# `src/submodules` — typed model submodules

`third_party/GVHMR` が提供していた前処理モデル群（person tracking / 2D pose /
画像特徴）と GVHMR 本体を、メインの `.venv` で完結して動くように移植した
パッケージです。`src/tennis_scene` の GVHMR コンポーネントはここに依存し、
`third_party/GVHMR` のコードには依存しません（学習済み重みのみ `ckpt/` の
symlink 経由で参照します）。

DINO person検出は `third_party/DINO` の公式git submoduleを型付きwrapperから
利用します。上流コードは `src/submodules` へコピーしません。

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
├── vendor/gvhmr/      # 環境競合を避けるため移植したGVHMR研究コード
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
| DINO source | `third_party/DINO/` | IDEA-Research/DINOをgit submoduleとして固定 |
| SMPL-X 本体 (`SMPLX_NEUTRAL.npz`) | `ckpt/body_models/smplx/` | 要ライセンス登録: https://smpl-x.is.tue.mpg.de/ |
| regressor 等の小物 `.pt` | `vendor/gvhmr/body_model/data/` | リポジトリに同梱 |
| SMPL faces（レンダリング用） | `data/smplh/neutral/model.npz` など | SMPL と SMPL-H は同一トポロジー |

## 上流コードの扱い

GVHMRはメイン環境との競合を避けるため `vendor/gvhmr/` に移植しています。一方、
DINOは上流ソースを変更せず `third_party/DINO/` のgit submoduleから読み込み、型付きの
推論契約だけを `models/dino/` に置きます。

DINO利用時はsubmoduleを初期化し、custom CUDA opをルート `setup.py` からビルドします。
PyTorch互換修正は `build/` 内の生成ソースだけに適用し、submodule自体は変更しません。

```bash
git submodule update --init third_party/DINO
TENNIS_LAB_BUILD_CUDA_OPS=1 .venv/bin/python setup.py build_ext --inplace
```

`DinoPersonTracker` は検出だけをDINOへ変更し、時系列対応付けには既存YOLO
trackingと同じUltralytics BoT-SORT設定を使います。DINO自体からtrack IDが出る
わけではありません。
