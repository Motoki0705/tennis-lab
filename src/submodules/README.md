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
| SMPL補助資産5点（regressor・変換・頂点index・平均パラメータ） | `ckpt/body_models/gvhmr/` | 利用者がGVHMR公式から取得。下記参照 |
| SMPL-H（facesをレンダリングに使用） | `ckpt/body_models/smplh/{neutral,male,female}/model.npz` | SMPL と SMPL-H は同一トポロジー |
| SMPL-H（ACCAD adapter用） | `ckpt/body_models/smplh/SMPLH_{MALE,FEMALE}.pkl` | 既存PKLを変換せず配置 |

scene pipelineは重みの配置に `paths.checkpoint_root` を使います。
`paths.external_asset_root` はDINO等の外部source専用です。旧配置から移す際は原本を保持し、
コピー元/先のSHA-256一致を確認してからckptへ配置します。旧パスを探索するfallbackはありません。
SMPL-X/SMPL-H本体、SMPL補助資産、外部重みはgitへ追加しません。

### SMPL補助資産の取得

5点は **[GVHMR公式リポジトリ](https://github.com/zju3dv/GVHMR/tree/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1)**
の固定commit `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1` から利用者が取得します。
各ファイルの公式配布パス・用途・条件は [LICENSES.ja.md](../../LICENSES.ja.md#smpl-auxiliary-assets)
を参照してください。本repoにはバイナリを同梱せず、自動ダウンロードもしません。

先に [GVHMRの条件](vendor/gvhmr/LICENSE) と
[SMPL](https://smpl.is.tue.mpg.de/modellicense.html)・
[SMPL-X](https://smpl-x.is.tue.mpg.de/modellicense.html) の条件を確認してください。
[GVHMR公式導入手順](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/docs/INSTALL.md)
が案内するとおり、body model本体はSMPL/SMPL-Xの公式サイトで登録・同意して取得します。
以下は、その条件に従って利用する補助資産の配置手順です。GVHMR用の別環境の構築は不要です。

```bash
git clone --filter=blob:none --no-checkout https://github.com/zju3dv/GVHMR downloads/gvhmr-assets
git -C downloads/gvhmr-assets checkout ee960bb6e2ea2d381aa97f08e9b71ef320b624b1 -- \
  LICENSE hmr4d/utils/body_model hmr4d/network/hmr2/configs/smpl_mean_params.npz
.venv/bin/python -m src.submodules.scripts.install_smpl_assets \
  --gvhmr-root "$PWD/downloads/gvhmr-assets" --checkpoint-root "$PWD/ckpt"
```

両rootは絶対パスで明示し、`--checkpoint-root` には実行設定の `paths.checkpoint_root` を指定します。
5点すべてのSHA-256を検査してから、次の場所へコピーします。
欠損・異なる版・配置先の競合はエラーにし、別の資産へ切り替えません。

```text
ckpt/body_models/gvhmr/
├── smpl_coco17_J_regressor.pt
├── smpl_neutral_J_regressor.pt
├── smplx2smpl_sparse.pt
├── smplx_verts437.pt
└── smpl_mean_params.npz
```

取得元URL・配置先・期待するSHA-256は以下で確認できます。

```bash
.venv/bin/python -m src.submodules.scripts.install_smpl_assets --list --checkpoint-root "$PWD/ckpt"
```

2026-10-19に、公式配布の5点が旧同梱版とすべてbyte単位で一致することを確認しました。
生成による置き換えは行っていません。設定の `people_models.smpl_assets` はCHECKPOINT root相対です。
旧 `bundled_assets` は受け付けません。可視化・Blenderも同じCHECKPOINT rootを参照します。
PLCSの生成・chunked学習では、SMPL-Hと同じ `paths.external_asset_root`（既定 `ckpt`）
から `body_models/gvhmr/smpl_coco17_J_regressor.pt` を解決します。

The five SMPL auxiliary files are user-acquired from the pinned official GVHMR
checkout. Review the upstream terms and register for SMPL/SMPL-X body models
before use. The command above verifies all five hashes and installs them into
your checkpoint root; it neither downloads nor regenerates model data.

## 上流コードの扱い

GVHMRはメイン環境との競合を避けるため `vendor/gvhmr/` に移植しています。一方、
DINOは上流ソースを変更せず `third_party/DINO/` のgit submoduleから読み込み、型付きの
推論契約だけを `models/dino/` に置きます。

DINO利用時のsubmodule初期化・CUDA toolkit要件・拡張ビルドは
[導入手順](../../docs/installation/README.ja.md#dino-cuda-extension)を正本とします。
PyTorch互換修正は出力ディレクトリ内の生成ソースだけに適用し、submodule自体は変更しません。

`validate_dino_extension()`は明示したimport pathの拡張にCPUテンソルを渡し、
forward/backwardが上流のCPU非対応エラーへ到達することを確認します。import可能でも
古いPyTorch APIで停止するバイナリは拒否します。CUDA kernelの動作・精度は別途queueで検証します。
出力の`lib/`を`PYTHONPATH`へ明示し、共有の古いビルドへの自動切替はしません。
