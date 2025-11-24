# Court Pose (third_party/court_pose) 概要

## 1. 目的と位置づけ

- **用途**: 2D テニスコートのキーポイント（ライン交点など）を、単一フレーム RGB 画像から推定するモデル群。
- **配置**: リポジトリ内では `third_party/court_pose/` に配置されており、実装は "third_party ライブラリ" として扱う。
- **将来像**: 本ドキュメントは、将来的に Court Pose を **推論 API（例: Web サービス / 内部 Python API）として提供する際の設計ベース** として使うことを目的とする。

学習コードや実験スクリプトではなく、**推論専用のモデルロード・前処理・後処理フロー**をどうラップするかにフォーカスする。

---

## 2. ディレクトリ構成（コードレベル）

`third_party/court_pose/` 直下の主な構成は次の通り。

- `README.md`
  - Court Pose 推論の概要と、`dino_fpn` / `dino_fpn_v2` のクイックスタートが記載されている。
- `dino_fpn/`
  - `dino_fpn_loader.py`
    - v1 モデル（`DinoVitHeatmap`）用のローダ。
    - `CoatLoadConfig` dataclass と、`load_coat_with_ckpt` 関数を提供する。
  - `model/model.py`
    - `DinoVitHeatmap` 本体の定義。
  - `model/dino_backbone.py`
    - DINOv3 ベースの ViT バックボーン `DINOBackbone` を定義（`torch.hub` で DINOv3 を local repo からロード）。
  - `dino_fpn_utils.py`
    - Lightning の checkpoint を「素の nn.Module」に読み込むためのユーティリティ。
- `dino_fpn_v2/`
  - `dino_fpn_v2_loader.py`
    - v2 モデル（DINOv3 + FPN 構成）用のローダ。
    - `DinoFpnV2LoadConfig` dataclass と、`load_dino_fpn_v2_with_ckpt` 関数を提供する。
  - `model/architecture.py`
    - `DinoFpnHeatmapNet`（DINOv3 + FPN + Heatmap デコーダ）を定義。
  - `model/backbone.py`
    - `DinoBackboneConfig`, `DinoFpnBackbone`（DINOv3 ViT を FPN で包むバックボーン）を定義。
  - `model/decoder.py`
    - `HeatmapDecoderConfig`, `FpnHeatmapDecoder`（FPN 特徴からアップサンプリングしてキーポイント heatmap を生成）を定義。
  - `model/factory.py`
    - `create_model(backbone=..., decoder=...)` で `DinoFpnHeatmapNet` を組み立てるファクトリ。
  - `model/__init__.py`
    - `DinoFpnHeatmapNet`, `CourtPoseLitModule`, `create_model` などを re-export（Lightning 側からの利用を想定）。

API 化を検討する際に直接関わるのは主に **`dino_fpn_loader.py` / `dino_fpn_v2_loader.py` と、その背後のモデル定義**。

---

## 3. v1: DinoVitHeatmap + CoatLoadConfig

### 3.1 モデル構造（`DinoVitHeatmap`）

- **クラス**: `third_party/court_pose/dino_fpn/model/model.py` の `DinoVitHeatmap`。
- **バックボーン**: `DINOBackbone`（`dino_backbone.py`）
  - `torch.hub.load` を用いて DINOv3 ViT をローカルリポジトリ（例: `third_party/dinov3`）からロード。
  - `embed_dim` と `patch_size` を持つ ViT から、約 `(H/16, W/16)` 解像度の特徴マップを出力。
- **デコーダ**:
  - Conv + ReLU + 2x Upsample のブロックを複数段積んで、入力解像度までアップサンプリング。
  - 末尾の `nn.Conv2d` により `num_keypoints` チャンネルの heatmap を出力。
- **入出力テンソル**:
  - 入力: `x: torch.Tensor` 形状 `[B, 3, H, W]`。
  - 出力: `y: torch.Tensor` 形状 `[B, K, H, W]`（K は `num_keypoints`）。
  - `y.shape[-2:]` と `x.shape[-2:]` が異なる場合、`F.interpolate` で最終的に入力解像度にリサイズしてからヘッドへ。

### 3.2 ローダと前処理（`CoatLoadConfig`, `load_coat_with_ckpt`）

- **設定クラス**: `CoatLoadConfig`（`dino_fpn_loader.py`）
  - `checkpoint_path`: Lightning checkpoint のパス（必須）。
  - `heatmap_channels`: 出力キーポイント数（デフォルト 15）。
  - `decoder_channels`: デコーダ中間チャネルのリスト（省略時は `[256, 128, 64, 32]`）。
  - `backbone_name`: `torch.hub.load` に渡す DINOv3 エントリ名（例: `"dinov3_vits16"`）。
  - `weights_path`: DINOv3 事前学習 weight のパス（`None` やパス文字列）。
  - `mean`, `std`: 画像正規化の統計量（ImageNet 相当）。
  - `pad_to_multiple`: 高さ・幅を指定倍数の整数倍にするためのパディング（例: 16）。
  - `device`: `"cuda" | "cpu" | "mps"` から選択。
  - `strict`, `remove_prefix`, `allow_partial`: checkpoint ロード時の挙動制御。
  - `resize_long_side`: 長辺リサイズ（None の場合、元画像サイズを維持）。
- **前処理**: `_build_transform(cfg)` が返す `transform` を通す。
  - オプションの `Resize(long_side)` → `ToTensor()` → `Normalize(mean, std)`。
  - `pad_to_multiple` が設定されている場合、`_PadToMultiple` で右・下方向に 0 パディング。
- **ロード関数**: `load_coat_with_ckpt(cfg)`
  - checkpoint 存在確認 → デバイス選択。
  - `DinoVitHeatmap` を `cfg` に従って構築。
  - transform を構築。
  - Lightning checkpoint から `state_dict` を取得 → `strip_prefix` でプレフィックスを削除 → `align_and_load` で key を突き合わせてロード。
  - `model.eval()` して `(model, transform, device)` を返す。

### 3.3 典型的な推論フロー

- config 読み込み → checkpoint パス設定 → `load_coat_with_ckpt` で (model, transform, device) を取得。
- PIL.Image を `transform` に通して `[C, H, W]` テンソルに変換し、`unsqueeze(0)` でバッチ化後、`device` に移す。
- `with torch.inference_mode():` の中で `model(tensor)` を実行し、`[1, K, H, W]` の heatmap を得る。
- README.md にあるように、ベースラインとしては各チャンネルに対する argmax で (x, y) 位置を取得できる。

API 化の観点では、この「config → (model, transform, device) → heatmap → (x, y) 座標」のパイプラインを 1 つの関数 / クラスにラップするのが自然。

---

## 4. v2: DinoFpnHeatmapNet + DinoFpnV2LoadConfig

### 4.1 モデル構造（`DinoFpnHeatmapNet`）

- **クラス**: `third_party/court_pose/dino_fpn_v2/model/architecture.py` の `DinoFpnHeatmapNet`。
- **構成要素**:
  - `DinoFpnBackbone`（`backbone.py`）
    - DINOv3 ViT をラップし、`get_intermediate_layers` から得たトークンマップを FPN に渡せる特徴マップに変換。
    - 中間層（`vit_layers`）を複数指定可能で、その最後の特徴マップを使用。
    - Conv とアップサンプリング / ストライド Conv で 3 スケールの特徴マップ（C3/C4/C5 相当）を構成し、`FeaturePyramidNetwork` で `P3`, `P4`, `P5` を生成。
  - `FpnHeatmapDecoder`（`decoder.py`）
    - `P3` を基準に `P4`, `P5` をアップサンプリング＆加算して特徴を融合。
    - Conv + BatchNorm + GELU + Upsample のブロックで解像度を上げ、最終的に `num_keypoints` チャンネルの heatmap を出力。
    - オプションで `sigmoid` / `softmax` の最終活性化を適用可能。
- **入出力テンソル**:
  - 入力: `[B, 3, H, W]`。
  - 出力: `[B, K, H, W]` の heatmap（必要に応じて `F.interpolate` で入力解像度にリサイズ）。

### 4.2 ローダと前処理（`DinoFpnV2LoadConfig`, `load_dino_fpn_v2_with_ckpt`）

- **設定クラス**: `DinoFpnV2LoadConfig`（`dino_fpn_v2_loader.py`）
  - `checkpoint_path`: Lightning checkpoint（必須）。
  - `repo_dir`: DINOv3 リポジトリディレクトリ（デフォルト `"third_party/dinov3"`）。
  - `entry`: `torch.hub.load` に渡す DINOv3 のエントリ名（例: `"dinov3_vits16"`）。
  - `weights`: バックボーン用の weights パス（省略時は DINO 側のデフォルト動作に従う）。
  - `freeze`: バックボーンを `requires_grad=False` にするかどうか。
  - `vit_layers`: `get_intermediate_layers` から取得する層インデックス（列挙）。
  - `fpn_channels`: FPN のチャネル数。
  - `num_keypoints`, `decoder_base_channels`, `decoder_channels`, `decoder_upsample_mode`, `decoder_final_activation` など、デコーダ周りの設定。
  - `mean`, `std`, `pad_to_multiple`, `resize_long_side`, `device`, `strict`, `remove_prefix`, `allow_partial` など、v1 と同様の前処理・ロード制御パラメータ。
  - `from_yaml(path)`, `to_yaml(path)` で YAML との相互変換が可能。
- **前処理**: `_build_transform(cfg)`
  - v1 と同様に、Resize → ToTensor → Normalize → optional PadToMultiple。
- **ロード関数**: `load_dino_fpn_v2_with_ckpt(cfg)`
  - checkpoint / repo_dir / weights の存在チェック。
  - `create_model(backbone=..., decoder=...)` を通じて `DinoFpnHeatmapNet` を構築。
  - transform を構築。
  - `load_lightning_state_dict` → `strip_prefix` → `align_and_load` で checkpoint を nn.Module にロード。
  - `model.eval()` 後に `(model, transform, device)` を返す。

### 4.3 README における v2 クイックスタート

`third_party/court_pose/README.md` では、次のような v2 用サンプルが示されている（概要のみ）。

- `DinoFpnV2LoadConfig.from_yaml("trained_models/court_pose/dino_fpn_v2/config.yaml")` で設定読込。
- `cfg.checkpoint_path` に ckpt パスを設定。
- `load_dino_fpn_v2_with_ckpt(cfg)` で `(model, transform, device)` を取得。

API 設計時には、このパターンを **そのまま内製 API の初期化ロジックに取り込む** と理解しておけばよい。

---

## 5. Court Pose を API として提供する際に押さえておきたいポイント

将来的に `court_pose` を HTTP / gRPC / 内部 Python API として公開する際、最低限押さえておきたいインターフェース設計上のポイントを整理する。

### 5.1 モデル初期化 API

- **入力**: config YAML パス、もしくは事前に構築済みの `CoatLoadConfig` / `DinoFpnV2LoadConfig`。
- **処理**:
  - YAML → dataclass 変換（`from_yaml`）。
  - `checkpoint_path` を上書きできるようにする（デプロイ環境ごとにパスが変わる想定）。
  - `load_*_with_ckpt` を呼び出して `(model, transform, device)` を構築。
- **出力**: 「推論ハンドラ」オブジェクト or 関数
  - 例: `CourtPosePredictor` のようなクラスで `__call__(image: PIL.Image | np.ndarray) -> Heatmap/Keypoints` を提供するイメージ。

### 5.2 推論 API（コア）

- **入力形式の候補**:
  - Python API: `PIL.Image`, `numpy.ndarray(H, W, 3)`, あるいは raw bytes から変換。
  - HTTP API: 画像ファイル（multipart/form-data）や base64 など。
- **内部処理**:
  1. 画像を RGB `PIL.Image` に変換。
  2. `transform` に通して `[C, H, W]` → `unsqueeze(0)` → `device` に転送。
  3. `with torch.inference_mode():` 下で `model(tensor)` を実行し、`[1, K, H, W]` の heatmap を得る。
  4. heatmap を座標列に変換（argmax, soft-argmax, Gaussian peak fitting など）。
- **出力形式の候補**:
  - Heatmap 生値: `float32` テンソル or NumPy 配列。
  - キーポイント: `[(x, y, score), ...]` 形式のリスト。

### 5.3 後処理ポリシー

- README では **シンプルな argmax** 実装が紹介されている。
- 実運用では、より安定した出力のために:
  - soft-argmax による連続座標推定。
  - 周辺の Gaussian fitting による peak refinement。
- などを行う余地があるが、現時点では **後処理そのものは third_party 側には含まれていない** ため、
  API 層での実装ポリシーを別途決める必要がある。

### 5.4 デバイス管理とスレッドセーフティ

- `load_*_with_ckpt` は単一の `torch.device` にモデルを構築する前提。
- マルチリクエスト / マルチスレッド環境では:
  - モデルはプロセスあたり 1 インスタンスを共有し、推論時はロックなし / もしくは最小限のロックで扱う。
  - GPU メモリ管理（複数モデルを同時に載せるか、SceneModel など他タスクとの GPU 共有をどうするか）を考慮する。

### 5.5 設定・バージョン管理

- `DinoFpnV2LoadConfig` / `CoatLoadConfig` は YAML 化・復元が容易。
- API サービス側では:
  - 「モデル ID → (config.yaml, checkpoint.ckpt)」のマッピングを持つ。
  - 起動時にどの組み合わせをロードするかを環境変数 / サービス設定で切り替える。

---

## 6. tennis-lab コードベースから見た依存関係のメモ

- Court Pose は `third_party/` 配下の独立モジュールとして実装されており、
  `DINOv3` のリポジトリ（例: `third_party/dinov3`）に依存する。
- tennis-lab 側のトレーニング / 推論パイプラインとは疎結合になっているため、
  API 化の際は **tennis-lab 内の別コンポーネント（例: テニスシーン解析）から Court Pose API を呼び出す** 形を想定しやすい。

---

## 7. 今後の具体的な API 設計の出発点

Court Pose を tennis-lab 内外から再利用しやすくするため、今後検討すべき具体的タスクの例を挙げておく。

- Court Pose 向けの **薄い Python API ラッパ** を追加する案
  - 例: `src/models/court_pose/api.py`（仮）に、`load_court_pose_model(kind="v1"|"v2")` や `predict_keypoints(image)` を定義。
- HTTP / gRPC などの **外部サービス化** を行う場合の I/O 仕様整理
  - 入力画像形式（解像度制約、ファイル形式）
  - 出力 JSON schema（キーポイントインデックス / 意味、座標系の定義など）。
- tennis-lab 内の他コンポーネント（例: テニス multi-cam 3D pose 系）のどこで Court Pose を呼び出すか、
  **データフロー上のインテグレーションポイント** を設計。

本ドキュメントは、上記のような API 設計を行う際の「コードベースの読み解きメモ」として参照することを想定している。
