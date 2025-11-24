# SAM3（third_party/sam3）概要と API 化メモ

## 1. 目的と位置づけ

- **用途**: テキスト / ボックス / ポイントなどのプロンプトから、画像・動画に対してオブジェクトのセグメンテーションとトラッキングを行う foundation model。
- **配置**: 本リポジトリでは、Meta 公開の `facebookresearch/sam3` を `third_party/sam3/` としてサブモジュール化している。
- **このドキュメントの目的**:
  - tennis-lab から SAM3 を **API として呼び出すときに押さえておくべき構造・入出力・依存関係**をまとめる。
  - 具体的な HTTP / gRPC 仕様を決める前の「読み方メモ」として使う。

---

## 2. コード構成（tennis-lab から見て重要な部分）

`third_party/sam3/` 直下:

- `README.md`
  - インストール方法、HuggingFace からの checkpoint ダウンロード、Image/Video のクイックスタート。
- `pyproject.toml`
  - 依存関係定義。
    - コア依存: `timm>=1.0.17`, `numpy==1.26`, `ftfy==6.1.1`, `iopath>=0.1.10`, `huggingface_hub` など。
    - オプション: `dev`, `train`, `notebooks` など。
- `sam3/`
  - **本体ライブラリ**。API 化で主に見るのは次のファイル群:
    - `model_builder.py`
      - Image / Video 用 SAM3 モデルを組み立てるファクトリ関数を提供。
      - `build_sam3_image_model`, `build_sam3_video_model`, `build_sam3_video_predictor` など。
    - `model/sam3_image_processor.py`
      - 画像向けの高レベル推論ユーティリティ `Sam3Processor` を定義。
    - `model/sam3_video_inference.py`
      - 動画向けの推論ロジック本体 `Sam3VideoInference*` を定義。
    - `model/sam3_video_predictor.py`
      - セッション管理付きの Video Predictor（`Sam3VideoPredictor`, `Sam3VideoPredictorMultiGPU`）。

このあたりをラップすることで、tennis-lab 側からの「1枚画像 segmentation API」や「動画 tracking API」を実装できる。

---

## 3. Image 推論パス（build_sam3_image_model + Sam3Processor）

### 3.1 モデル構築: `build_sam3_image_model`

- 実装: `sam3/model_builder.py`
- シグネチャ（簡略）:

  ```python
  def build_sam3_image_model(
      bpe_path=None,
      device="cuda" if torch.cuda.is_available() else "cpu",
      eval_mode=True,
      checkpoint_path=None,
      load_from_HF=True,
      enable_segmentation=True,
      enable_inst_interactivity=False,
      compile=False,
  ) -> Sam3Image:
      ...
  ```

- 役割（内部で行っていること）:
  - ViT ベースの vision encoder + text encoder + VL 結合部 + Transformer encoder/decoder + segmentation head を組み立てる。
  - `enable_inst_interactivity=True` の場合、SAM1 タスク相当のインタラクティブセグメンテーション用 predictor を組み込む。
  - checkpoint ロード:
    - `load_from_HF=True` かつ `checkpoint_path=None` のとき、`facebook/sam3` から `sam3.pt` を自動ダウンロードしてロード。
  - `device` / `eval_mode` に応じて `.to(device)` / `.eval()` を設定。

### 3.2 画像前処理とプロンプト処理: `Sam3Processor`

- 実装: `sam3/model/sam3_image_processor.py`
- コンストラクタ:

  ```python
  class Sam3Processor:
      def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
          ...
  ```

- 内部前処理:
  - `ToDtype(uint8)` → `Resize((resolution, resolution))` → `ToDtype(float32)` → `Normalize(mean=[0.5]*3, std=[0.5]*3)`。

#### 3.2.1 画像セット: `set_image`

```python
state = processor.set_image(image)
```

- `image`: `PIL.Image.Image` または `torch.Tensor` / `np.ndarray`。
- 処理:
  - 元の高さ・幅を `state["original_height"]`, `state["original_width"]` に保存。
  - 正規化済み画像を `model.backbone.forward_image` に通し、特徴を `state["backbone_out"]` にキャッシュ。
- 戻り値 `state` はその後の text / box プロンプト推論でも再利用する。

#### 3.2.2 テキストプロンプト: `set_text_prompt`

```python
state = processor.set_text_prompt(prompt="a tennis player", state=state)
```

- `model.backbone.forward_text([prompt])` でテキスト埋め込みを計算し、`state["backbone_out"]` に追加。
- 幾何プロンプトが未設定ならダミープロンプトをセットし、`_forward_grounding` を実行。
- `_forward_grounding` の結果として `state` に次が入る:
  - `state["masks"]`: バイナリマスク（画像サイズにリサイズ済み）。
  - `state["boxes"]`: `[x0, y0, x1, y1]` 形式のバウンディングボックス（元画像のピクセル座標）。
  - `state["scores"]`: 信頼度スコア（`confidence_threshold` でフィルタ済み）。

#### 3.2.3 ボックスプロンプト: `add_geometric_prompt`

```python
state = processor.add_geometric_prompt(box=[cx, cy, w, h], label=True, state=state)
```

- `box`: 画像全体に対する正規化座標 `[center_x, center_y, width, height]`（0〜1）。
- 正負ラベルをもとに、テキストプロンプトと組み合わせたセグメンテーション結果を再計算し、同様に `masks` / `boxes` / `scores` を更新する。

### 3.3 tennis-lab から見た Image API のラップ方針（例）

- 内部 Python API のイメージ:
  - `load_sam3_image_model(device) -> (model, processor)`
  - `run_sam3_on_image(image, text_prompt, box_prompt=None) -> List[MaskResult]`
- HTTP API として出す場合:
  - 入力: 画像 (multipart/form-data) + テキスト文字列 + オプションでボックス座標リスト。
  - 出力: マスク（例: RLE もしくは PNG path）、ボックス座標、スコア、ラベル。

---

## 4. Video 推論パス（Sam3VideoPredictor[MultiGPU]）

### 4.1 モデル構築: `build_sam3_video_model` / `build_sam3_video_predictor`

- 実装: `sam3/model_builder.py`
- `build_sam3_video_model(...)`:
  - Detector（image + text + box から mask を出す部分）と Tracker を組み合わせた `Sam3VideoInferenceWithInstanceInteractivity` を構築。
  - checkpoint が指定されていなければ、`facebook/sam3` から `sam3.pt` をダウンロードしてロード。
- `build_sam3_video_predictor(*args, gpus_to_use=None, **kwargs)`:
  - 上記 video モデルを内部に持つ `Sam3VideoPredictorMultiGPU` を返す。
  - 単一 GPU の場合も MultiGPU 実装を通して扱われる。

### 4.2 セッション管理付き API: `Sam3VideoPredictor`

- 実装: `sam3/model/sam3_video_predictor.py`
- 典型的なフロー（README のサンプルと同様）:

  ```python
  from sam3.model_builder import build_sam3_video_predictor

  predictor = build_sam3_video_predictor()

  # 1) セッション開始
  resp = predictor.handle_request({
      "type": "start_session",
      "resource_path": video_path,  # MP4 もしくは JPEG 連番ディレクトリ
  })
  session_id = resp["session_id"]

  # 2) あるフレームにテキストプロンプトを追加
  resp = predictor.handle_request({
      "type": "add_prompt",
      "session_id": session_id,
      "frame_index": 0,
      "text": "a tennis player",
  })
  outputs = resp["outputs"]
  ```

- `handle_request` の主な `type`（tennis-lab からラップするときに意識すべきもの）:
  - `"start_session"`: 画像 / 動画パスから推論セッションを開始し、`session_id` を返す。
  - `"add_prompt"`: 指定フレームにテキスト / ポイント / ボックスプロンプトを追加し、そのフレームの出力を返す。
  - `"reset_session"`, `"close_session"`, `"remove_object"`: セッション状態のリセット / 終了や特定オブジェクトの削除。
