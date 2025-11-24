# ByteTrack（third_party）概要と API 化メモ

このドキュメントは、`third_party/ByteTrack` のコードベースをテニスプロジェクトから利用する際の **読み方メモ / 将来の API 設計の足場** をまとめたもの。

- 元リポジトリ: https://github.com/ifzhang/ByteTrack
- 論文: *ByteTrack: Multi-Object Tracking by Associating Every Detection Box* (ECCV 2022)
- 本リポジトリでは `third_party/` 以下に「ほぼ素のまま」配置し、主に **トラッキング部分（`BYTETracker`）** を利用する想定。

---

## 1. 全体像と役割

- ByteTrack は **Multi-Object Tracking (MOT)** の手法で、
  - **YOLOX 系の物体検出器**からの検出結果（バウンディングボックス）を入力に
  - 各フレーム間で **ID を一貫させるトラッカー** を提供する。
- 構成要素（高レベル）
  - **検出器（Detector）**: `yolox` パッケージ内のモデル（例: YOLOX-X）
  - **トラッカー（Tracker）**: `yolox/tracker/byte_tracker.py` 内の `BYTETracker`
  - **推論／評価スクリプト**: `tools/demo_track.py`, `tools/track.py`, `yolox/evaluators/mot_evaluator.py` など
  - **デプロイ用コード**: `deploy/ONNXRuntime`, `deploy/TensorRT`, `deploy/ncnn`, `deploy/DeepStream` など

テニスプロジェクト視点では、**「検出結果 → トラッキング結果」への変換器** として `BYTETracker` を API 化するのが主眼となる。

---

## 2. ディレクトリ構成（利用上重要な部分）

`third_party/ByteTrack/` のうち、今後参照頻度が高そうなものだけをピックアップ:

- **`tools/`**
  - `demo_track.py`: 画像 / 動画 / WebCam 向けのデモスクリプト。
    - 検出器 + `BYTETracker` を組み合わせた、**オンライン MOT 推論パイプライン**のリファレンス実装。
  - `track.py`: MOT 形式の評価用トラッキングスクリプト。
  - `train.py`: YOLOX + ByteTrack の学習スクリプト。
- **`exps/`**
  - `exps/example/mot/*.py`: YOLOX / ByteTrack の設定（モデル構成、入力サイズ、データセットなど）の **Experiment 定義**。
  - `get_exp()` でロードされ、デモやトレーニングから使用される。
- **`yolox/tracker/`**
  - `byte_tracker.py`: 中心となる `BYTETracker` / `STrack` 実装。
  - `kalman_filter.py`: 状態推定用の KalmanFilter。
  - `matching.py`: IoU ベースのマッチング、スコア融合など。
  - `basetrack.py`: `BaseTrack` / `TrackState` の定義。
- **`yolox/evaluators/mot_evaluator.py`**
  - MOT 用の評価ロジック。`BYTETracker` をどのようにループに組み込んでいるかを見るのに有用。
- **`deploy/`**
  - ONNXRuntime / TensorRT / ncnn / DeepStream 向けのデプロイコード。
  - **推論エンジンを差し替えたいときの参考実装**として利用可能。

---

## 3. 推論パイプラインの流れ（demo_track.py ベース）

`tools/demo_track.py` の `image_demo` / `imageflow_demo` が、標準的なオンライン推論の流れを示している。

1. **モデル・設定のロード**
   - `exp = get_exp(args.exp_file, args.name)`
   - `model = exp.get_model()` で YOLOX 系モデルを構築。
   - チェックポイント（`ckpt`）をロードし、`model.eval()`。
2. **Predictor の準備** (`Predictor` クラス)
   - `preproc()` により、画像を `exp.test_size` にリサイズ + 正規化。
   - `model(img)` 実行後、`postprocess()` で
     - NMS
     - スコア閾値フィルタリング
     - 最終的な `output`（検出ボックス）の生成
3. **BYTETracker の準備**
   - `tracker = BYTETracker(args, frame_rate=args.fps)`
   - `args.track_thresh`, `args.match_thresh`, `args.track_buffer` などのトラッキング設定を受け取る。
4. **フレームごとのループ**
   - `outputs, img_info = predictor.inference(frame, timer)`
   - `online_targets = tracker.update(outputs[0], [img_info["height"], img_info["width"]], exp.test_size)`
   - `online_targets` から `tlwh`・`track_id`・`score` を取り出し、可視化やテキスト出力を行う。

この「**検出結果 → BYTETracker.update → online_targets**」の部分が、将来 API 化するときにそのまま中核インターフェースになる。

---

## 4. `BYTETracker` のインターフェース

実装: `third_party/ByteTrack/yolox/tracker/byte_tracker.py`

### 4.1 コンストラクタ

```python
tracker = BYTETracker(args, frame_rate=30)
```

- **`args`** に期待されている主なフィールド
  - `track_thresh: float`
    - トラッキングに使う高スコア検出の閾値。
    - 内部では `self.det_thresh = args.track_thresh + 0.1` にも利用される。
  - `track_buffer: int`
    - ロスト状態のトラックを保持するフレーム数（`max_time_lost`）。
  - `match_thresh: float`
    - IOU マッチング時の距離閾値。
  - `mot20: bool`
    - MOT20 のときの特別扱い（`matching.fuse_score` を使わない等）。
- **`frame_rate`**
  - デフォルト 30。`track_buffer` と組み合わされて `buffer_size` を決定。

### 4.2 `update` メソッド

```python
online_targets = tracker.update(output_results, img_info, img_size)
```

- **`output_results`**
  - 形式は 2 パターンを許容：
    - `shape == (N, 5)` の場合
      - `[:, :4]` が `x1, y1, x2, y2`
      - `[:, 4]` が `score`
      - 想定: 外部検出器からの **CPU 上の numpy array** を直接渡すケース。
    - それ以外（YOLOX の生出力）
      - `torch.Tensor` を `.cpu().numpy()` に変換後、
      - `scores = output[:, 4] * output[:, 5]`（objectness × class confidence）
      - `bboxes = output[:, :4]`（`x1, y1, x2, y2`）
- **`img_info`**
  - 実装では `img_h, img_w = img_info[0], img_info[1]` として使用。
  - デモでは `[height, width]` を渡している。
  - **`img_size` と組み合わせてスケールを計算**するために用いられ、
    - `scale = min(img_size[0]/img_h, img_size[1]/img_w)`
    - `bboxes /= scale` で **元画像座標系に戻す**。
- **`img_size`**
  - YOLOX の入力サイズ（例: `(640, 640)`）。
- **戻り値: `online_targets`**
  - `STrack` のリスト。
  - 各 `STrack` には少なくとも以下のプロパティがある:
    - `track_id: int` — トラック固有 ID
    - `tlwh: np.ndarray` — `(x, y, w, h)`（左上 + 幅高さ）
    - `tlbr: np.ndarray` — `(x1, y1, x2, y2)`
    - `score: float` — 最終スコア
    - `is_activated: bool` — 有効かどうか

将来 API 化する場合、この `update` を **1 フレーム 1 回叩くストリーミング API** としてラップするのが自然。

---

## 5. 検出器の差し替えと外部からの利用

README の「Combining BYTE with other detectors」では、以下のような利用パターンが紹介されている。

```python
from yolox.tracker.byte_tracker import BYTETracker
tracker = BYTETracker(args)
for image in images:
    dets = detector(image)  # dets: (N, 5) = x1, y1, x2, y2, score
    online_targets = tracker.update(dets, info_imgs, img_size)
```

テニスプロジェクトでの利用を考えると、次のような前提を置くと扱いやすい:

- **入力**
  - 1 フレームごとに
    - 元画像サイズ `(orig_h, orig_w)`
    - 検出結果 `dets: np.ndarray[N, 5]` (`x1, y1, x2, y2, score`)
  - 座標系は **推論に使ったリサイズ済み画像の座標系** だが、
    - ByteTrack 内部で `img_info` / `img_size` からスケールを計算し、
    - **元画像座標系に戻してからトラッキング**する設計になっている。
- **出力**
  - `Track`（`STrack`）のリスト
    - 各トラックに `track_id`, `tlwh`, `score` など。

後段のコード（例: テニス用の SceneModel 入力生成など）からすると、

- 「**あるフレームに現れている人物の ID 付きバウンディングボックス群**」

が取れればよいので、`BYTETracker.update` のラッパーで

- `List[Track]` あるいは
- `List[dict(x1=..., y1=..., x2=..., y2=..., track_id=..., score=...)]`

のように整形するのが現実的な API となる。

---

## 6. tennis-lab 内で API として提供するときの論点

将来的にテニスプロジェクトから ByteTrack を「モジュール / サービス」として使いたい場合、少なくとも以下の観点を整理しておくとよい。

### 6.1 どこまでを ByteTrack 側に持たせるか

- **候補 A: 検出 + トラッキング一体型 API**
  - 入力: 画像（`np.ndarray` / パス）
  - 内部: YOLOX で検出 → `BYTETracker.update` → `Track` を返す
  - 利点: 呼び出し側は非常にシンプル。
  - 欠点: 既存の検出器（他モデル）との組み合わせがしづらい。
- **候補 B: 検出は外部、ByteTrack はトラッキング専用 API**
  - 入力: 検出結果 `dets (N,5)` + 画像メタ（`orig_h, orig_w`）
  - 内部: `BYTETracker.update(dets, img_info, img_size)`
  - 利点: 検出器を自由に差し替えられる（テニス専用 Detector と組み合わせなど）。

テニスの文脈では、既に別の Detector を持っている可能性が高いため、
**最初は「候補 B（トラッキング専用）」を優先して設計**しておくのが良さそう。

### 6.2 API 境界と状態管理

- ByteTrack は **ステートフルなトラッカー**（内部にトラック集合と KalmanFilter 状態を持つ）。
- よって API としては:
  - **セッション / シーケンス単位**でインスタンスを分ける必要がある。
  - 候補インターフェース
    - `tracker = create_bytetrack_tracker(config)`
    - `tracks = tracker.update(dets, frame_info)` をフレームごとに呼ぶ。
  - REST や gRPC で外出しする場合は
    - `session_id` を明示的に渡し、サーバ側で `BYTETracker` インスタンスを管理する設計が必要。

### 6.3 パラメータチューニング

- MOT ベンチマーク向けコードでは、`MOTEvaluator` 内で
  - シーケンス名（例: `MOT17-01-FRCNN`）ごとに
  - `track_buffer`, `track_thresh` を個別に調整している。
- テニス用途では
  - カメラ設置条件や人物数、動きの激しさなどに応じて
  - `track_thresh`, `match_thresh`, `track_buffer`, `min_box_area` 等を**実験的に決める必要**がある。

API レベルでは、少なくとも

- `track_thresh`
- `track_buffer`
- `match_thresh`
- `min_box_area`

あたりを **設定ファイル / コンストラクタ引数**として露出しておくとよい。

### 6.4 依存・ビルド観点

- Python 依存関係
  - `third_party/ByteTrack/requirements.txt` を見ると、`torch`, `opencv_python`, `loguru`, `motmetrics` などが必要。
  - 本プロジェクトの `pyproject.toml` / `uv` 管理に合わせて、必要な範囲だけを取り込む方針を検討する。
- デプロイ
  - `deploy/ONNXRuntime`, `deploy/TensorRT`, `deploy/ncnn`, `deploy/DeepStream` にサンプルコードがある。
  - 推論エンジンを PyTorch のままにするか、ONNX/TensorRT に変換するかは
    - 運用環境（GPU 有無、レイテンシ要件）に応じて判断する。

---

## 7. 今後の作業候補

- **ByteTrack ラッパーモジュールの試作**
  - 例: `src/third_party/bytetrack_api.py`（名称は未定）
  - `create_tracker(config) -> Tracker` と `Tracker.update(dets, frame_info) -> List[Track]` を定義。
- **テニス用検出器との接続実験**
  - 既存の人物検出結果を ByteTrack に渡し、
  - テニスシーンでの ID 一貫性・ロバスト性を確認する。
- **パラメータ探索**
  - `track_thresh`, `match_thresh`, `track_buffer` をテニス動画でスイープし、
  - 代表的な設定プリセットを docs に追記する。

このドキュメントは「ByteTrack コードを深掘りするときの入り口」として維持し、
具体的な API 実装が固まったら、別途 `src/` 側の設計ドキュメントからも参照することを想定している。
