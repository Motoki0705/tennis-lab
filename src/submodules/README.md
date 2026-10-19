# `src/submodules` — typed model submodules

`third_party/GVHMR` が提供していた前処理モデル群（person tracking / 2D pose /
画像特徴）と GVHMR 本体を、メインの `.venv` で完結して動くように移植した
パッケージです。`src/tennis_scene` の GVHMR コンポーネントはここに依存し、
`third_party/GVHMR` のコードには依存しません。学習済み重みとbody modelは
`ckpt/` に実体を配置し、checkpoint rootから参照します。

DINO person検出は `third_party/DINO` の公式git submoduleを型付きwrapperから
利用します。上流コードは `src/submodules` へコピーしません。

## 構成

```
src/submodules/
├── models/            # 型付きの推論モデル（下流はここだけ見ればよい）
│   ├── _base/         # BaseInferenceModel: load/unload/predict の共通契約
│   ├── dino/          # DinoPersonDetector: frame -> person boxes
│   ├── tracker/       # 共通のtrack契約・bbox補間・足元polygonフィルタ
│   ├── vitpose/       # ViTPosePose2D: video + boxes -> COCO-17 keypoints
│   ├── hmr2/          # Hmr2FeatureExtractor: video + boxes -> (F, 1024) features
│   └── gvhmr/         # GvhmrMeshRecovery: keypoints+boxes+features -> SMPL-X params
│                      # SmplVertexReconstructor: params -> SMPL vertices (F, 6890, 3)
└── vendor/gvhmr/      # 環境競合を避けるため移植したGVHMR研究コード
```

## 共通インターフェース（`models`）

下流のcanonical importは`src.submodules.models`だけです。`models/_base`や各
model family packageは内部実装であり、同じsymbolを再exportしません。すべての
モデルは `BaseInferenceModel[RequestT, ResultT]` を実装します。

- 構築は軽量（重みは触らない）。`load()` は冪等、`unload()` で解放。
- `predict(request) -> result` は自動 load + `torch.no_grad` 込み。
- deviceは`"auto"`のみavailabilityに応じて選択し、明示したCUDAが利用できない場合はmodel構築前にエラーにする。CPUへの暗黙fallbackは持たない。
- `GvhmrRequest`は`(F,17,3)` keypoints、`(F,3)` boxes、`(F,1024)` features、frame/device/dtypeと正の画像・bbox寸法を構築時とvendor entry直前に検証する。
- request / result はモデルごとの frozen dataclass。
- `TrackRequest.num_tracks=None, interactive=False`は全trackを返し、検出0件も空結果にする。`observed_mask`はbbox補間と実検出を区別する。
- `filter_detections_by_footpoint`は評価で共有する足元polygonフィルタです。標準sceneのDINO検出・StrongSORT++追跡とIO契約は[scene pipeline README](../tennis_scene/pipeline/README.md)を参照してください。
- `Pose2DRequest` / `ImageFeatureRequest`の`frame_indices`を指定すると、明示したsource frameを逐次decodeし、全動画をRAMへ展開せずcrop batchを作る。
- `Pose2DFrameSequenceRequest`は明示したBGR画像列とframe ID・boxを受け、動画と同じcrop/正規化/復号でViTPoseを実行する。JPEG storeやRGB増強を動画へ再圧縮せず接続できる。同一frameの複数人物を受理し、要求frameの欠落・順序不整合・画像サイズ変更は停止する。最後の要求frameまでだけを読み、全clipの入力hash/PTS検証は呼び出し側が担当する。

```python
import cv2

from src.submodules.models import DinoPersonDetector, PersonDetectionRequest

detector = DinoPersonDetector(
    checkpoint="/absolute/path/to/dino.pth",
    repository="/absolute/path/to/third_party/DINO",
    device="cuda",
    confidence=0.30,
    short_side=800,
    max_long_side=1333,
)
frame = cv2.imread("frame.jpg")
assert frame is not None
try:
    detections = detector.predict(PersonDetectionRequest(frame))
finally:
    detector.unload()
```

## デモ

保存済みの入力pose・観測mask・3D身体と配置拒否を同じsource frameで確認する
読取専用UIは[保存GVHMR / Pose review](visualization/review/README.md)を参照してください。

## 必要な資産

| 資産 | 場所 | 備考 |
|---|---|---|
| ViTPose / HMR2 / GVHMR 重み | `ckpt/{vitpose,hmr2,gvhmr}/` | モデル重みの実体。外部sourceの場所と分離 |
| DINO 4-scale Swin-L | `ckpt/dino/checkpoint0029_4scale_swin.pth` | COCO person (class id 1) のみ使用 |
| DINO source | `third_party/DINO/` | IDEA-Research/DINOをgit submoduleとして固定 |
| SMPL-X 本体 (`SMPLX_NEUTRAL.npz`) | `ckpt/body_models/smplx/` | 要ライセンス登録: https://smpl-x.is.tue.mpg.de/ |
| regressor 等の小物 `.pt` | `vendor/gvhmr/body_model/data/` | リポジトリに同梱 |
| SMPL-H（facesをレンダリングに使用） | `ckpt/body_models/smplh/{neutral,male,female}/model.npz` | SMPL と SMPL-H は同一トポロジー |
| SMPL-H（ACCAD adapter用） | `ckpt/body_models/smplh/SMPLH_{MALE,FEMALE}.pkl` | 既存PKLを変換せず配置 |

scene pipelineは重みの配置に `paths.checkpoint_root` を使います。
`paths.external_asset_root` はDINO等の外部source専用です。旧配置から移す際は原本を保持し、
コピー元/先のSHA-256一致を確認してからckptへ配置します。旧パスを探索するfallbackはありません。
SMPL-X/SMPL-H本体と外部重みはgitへ追加しません。

## 上流コードの扱い

GVHMRはメイン環境との競合を避けるため `vendor/gvhmr/` に移植しています。一方、
DINOは上流ソースを変更せず `third_party/DINO/` のgit submoduleから読み込み、型付きの
推論契約だけを `models/dino/` に置きます。

DINO利用時はsubmoduleを初期化し、custom CUDA opをルート `setup.py` からビルドします。
PyTorch互換修正は `build/` 内の生成ソースだけに適用し、submodule自体は変更しません。

`validate_dino_extension()`は明示したimport pathの拡張にCPUテンソルを渡し、
forward/backwardが上流のCPU非対応エラーへ到達することを確認します。import可能でも
古いPyTorch APIで停止するバイナリは拒否します。CUDA kernelの動作・精度は別途queueで検証します。
run専用のビルドには[build_dino_extension.sh](../../tests/benchmarks/build_dino_extension.sh)を使い、
出力の`lib/`を`PYTHONPATH`へ明示してください。共有の古いビルドへの自動切替はしません。

```bash
git submodule update --init third_party/DINO
TENNIS_LAB_BUILD_CUDA_OPS=1 .venv/bin/python setup.py build_ext --inplace
```
