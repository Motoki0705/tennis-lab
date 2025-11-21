# SceneModel 用 Dataset 仕様

本書は、SceneModel タスクで利用する **DanceTrack 系 Dataset とバッチ表現** の仕様をまとめる。

- コード:
  - Dataset / サンプル表現: `src/datasets/scene_model/dancetrack.py`
  - collate 関数 / バッチ表現: `src/datasets/scene_model/collate_tracking.py`
  - DataModule: `src/training/scene_model/datamodule.py`
- 関連ドキュメント:
  - モデル: `docs/spec/models/scene_model.md`
  - トレーニングパイプライン: `docs/spec/training/scene_model.md`

ここでは **データの在り方と形状** にフォーカスし、実行方法やトレーニングフローは既存ドキュメントに委ねる。

---

## 1. 全体像

SceneModel 用の Dataset は、おおまかに次のレイヤーで構成される。

1. **生データ (DanceTrack)**
   - 公式 DanceTrack データセットのディレクトリ構造とアノテーションを利用する。
   - 画像フレームと、各フレームに対するトラック ID 付きバウンディングボックス列が前提。
2. **DancetrackDataset** (`src/datasets/scene_model/dancetrack.py`)
   - 生データからウィンドウ単位のシーケンスをサンプリングし、`TrackingSample` を返す。
   - 窓長・ストライド・シーケンス分割などのロジックをひとまとめにする。
3. **collate_tracking / SceneBatch** (`src/datasets/scene_model/collate_tracking.py`)
   - 長さの異なる `TrackingSample` をパディングしてバッチ化し、`SceneBatch` として LightningModule へ渡す。
4. **DancetrackDataModule** (`src/training/scene_model/datamodule.py`)
   - Dataset / DataLoader の構築と、学習・検証用データローダのエントリポイントを提供する。

---

## 2. 生データとルートディレクトリ

- ルートディレクトリ:
  - 既定値: `third_party/DanceTrack/dancetrack`
  - YAML 設定の `dataset.dataset.root` などで上書き可能。
- 内部構造:
  - 公式 DanceTrack 配布フォーマットに準拠したシーケンス単位のディレクトリを前提とする。
  - 例 (概念的な構造):

    ```text
    <root>/
      train/
        sequence_0001/
          img1.jpg
          img2.jpg
          ...
          gt/gt.txt        # 追跡アノテーション
          seqinfo.ini      # 画像サイズやフレームレート等のメタ情報
      val/
        sequence_0002/
        ...
    ```

- `DancetrackDataset` 内部では、次の情報をまとめた `_SequenceMeta` を構築する:
  - `name`: シーケンス名
  - `frame_paths`: 各フレーム画像の相対パス列
  - `annotations`: フレームごとの `_Annotation` リスト
  - `frame_rate`: フレームレート (float)
  - `image_size`: `(width, height)`

`_SequenceMeta` は JSON キャッシュを通じて保存・再利用できるようになっており、大規模データセットでも繰り返しの起動を高速化する設計になっている。

---

## 3. DancetrackDataset の仕様

### 3.1 コンストラクタ

- 実装: `src/datasets/scene_model/dancetrack.py:DancetrackDataset`
- シグネチャ (簡略):

  ```python
  class DancetrackDataset(Dataset[TrackingSample]):
      def __init__(
          self,
          cfg: Mapping[str, Any] | DictConfig,
          split: str,
          window_sampler: WindowSampler | None = None,
          debug: Mapping[str, Any] | None = None,
      ) -> None:
          ...
  ```

- 主な引数:
  - `cfg`:
    - Dataset 設定 (`DictConfig` または `dict`)。
    - 少なくとも次のキー群を想定:
      - `root`: DanceTrack ルートディレクトリパス
      - `split`: train/val/test ごとのサブディレクトリ名マッピング
      - `window.size`: ウィンドウ長 (フレーム数)
      - `window.stride`: ウィンドウストライド (フレーム数)
      - `cache`: メタ情報キャッシュの有効/無効やパス
  - `split`:
    - `"train" | "val" | "test"` 等、利用する分割名。
  - `window_sampler`:
    - ウィンドウ列挙ロジックを上書きするためのコールバック。
    - 省略時は内部の `_default_sampler` を使用する。
  - `debug`:
    - デバッグ用途の追加設定 (シーケンス数の制限、乱数シード等)。

### 3.2 ウィンドウサンプリング

- 既定の `_default_sampler(length, window, stride)` は次のような振る舞いをする:
  - シーケンス長 `length` に対し、`[start, end)` のウィンドウを列挙。
  - `length <= 0` の場合: 空リスト。
  - `length <= window` の場合: `(0, length)` の 1 ウィンドウのみ。
  - それ以外:
    - `start = 0, stride, 2*stride, ...` で `start + window <= length` の範囲を列挙。
    - 最後のウィンドウの終端が `length` に満たない場合、末尾合わせの `(length-window, length)` を追加。

- Dataset 内部では、各シーケンスに対してこのサンプラーを適用し、

  ```python
  windows: list[tuple[int, int, int]]  # (sequence_index, start, end)
  ```

  を `__getitem__` に渡すためのインデックスとして保持する。

### 3.3 __len__ / __getitem__

- `__len__`:
  - すべてのシーケンスに対してサンプリングされたウィンドウ総数を返す。
- `__getitem__(index)`:
  - `(seq_idx, start, end)` を引き当て、該当シーケンスから `[start, end)` のフレームをロード。
  - 画像とアノテーションを読み込み、変換・正規化したうえで `TrackingSample` を返す。

---

## 4. TrackingSample / TargetFrame の仕様

### 4.1 TargetFrame

- 実装: `src/datasets/scene_model/dancetrack.py:TargetFrame`
- 1 フレーム分のターゲット (検出結果) を表すデータクラス。

  ```python
  @dataclass(slots=True)
  class TargetFrame:
      center: Tensor        # [N, 2], 正規化済みバウンディングボックス中心 (x, y)
      size: Tensor          # [N, 2], 正規化済み幅・高さ (w, h)
      track_ids: Tensor     # [N], トラック ID (int64)
      confidence: Tensor    # [N], 信頼度スコア (float32)
  ```

- 座標系:
  - `center` / `size` は、`_boxes_to_target` 内で画像サイズ `(W, H)` で割った上で `0.0〜1.0` にクリップされる。
  - したがって **モデルは常に正規化済み座標を受け取る** ことを前提にできる。
- 補助メソッド:
  - `TargetFrame.empty(device=None)`:
    - 0 行テンソルを持つ空ターゲットを生成する。
    - パディングフレームやアノテーションのないフレームで利用される。

### 4.2 TrackingSample

- 実装: `src/datasets/scene_model/dancetrack.py:TrackingSample`
- `__getitem__` が返す 1 ウィンドウ分のデータを表す。

  ```python
  @dataclass(slots=True)
  class TrackingSample:
      frames: Tensor              # [T, C, H, W]
      targets: list[TargetFrame]  # 長さ T の TargetFrame リスト
      sequence_id: str            # シーケンス名
      frame_indices: list[int]    # 元シーケンス中のフレームインデックス列
  ```

- `frames`:
  - 画像は `torchvision.io.read_image` で読み込まれ、必要に応じて
    - Data Augmentation
    - 正規化 / 型変換
    を通した後、`[C, H, W]` 形状として積み上げられる。
  - ウィンドウ次元 `T` はサンプラーで決まる長さ。
- `targets`:
  - 各フレームに 1 つの `TargetFrame` が対応し、バウンディングボックス列を保持する。
  - アノテーションが存在しない場合でも、`TargetFrame.empty()` が使われるためリスト長は常に `T` となる。

---

## 5. collate_tracking と SceneBatch

### 5.1 collate_tracking

- 実装: `src/datasets/scene_model/collate_tracking.py:collate_tracking`
- 役割:
  - 長さの異なる `TrackingSample` 列を受け取り、最大長に合わせてパディングした `SceneBatch` を構築する。

- シグネチャ:

  ```python
  def collate_tracking(samples: Sequence[TrackingSample]) -> SceneBatch:
      ...
  ```

- 振る舞い (概要):
  - `samples` が空の場合、`ValueError` を送出。
  - バッチ内の最大シーケンス長 `max_len` を求める。
  - `frames` をゼロパディングし、`[B, max_len, C, H, W]` のテンソルを構築。
  - `padding_mask` を `[B, max_len]` 形状の bool テンソルとして構築:
    - 実データが存在するタイムステップ: `False`
    - パディングステップ: `True`
  - `targets` は
    - 実データ部分: もとの `TargetFrame` をそのまま利用
    - パディング部分: `TargetFrame.empty(device=...)` を追加
    することで、各サンプルについて長さ `max_len` のリストに揃える。

### 5.2 SceneBatch

- 実装: `src/datasets/scene_model/collate_tracking.py:SceneBatch`
- LightningModule が受け取る最終的なバッチ表現。

  ```python
  @dataclass(slots=True)
  class SceneBatch:
      frames: Tensor                    # [B, T_max, C, H, W]
      targets: list[list[TargetFrame]]  # [B][T_max] 構造
      padding_mask: Tensor              # [B, T_max], bool
      sequence_ids: list[str]          # 長さ B
  ```

- 特徴:
  - `frames`:
    - 画素値 (型は Dataset 内の transform 設定に依存) を持つ 5 次元テンソル。
  - `targets`:
    - Python のネストしたリスト構造だが、`TargetFrame` 自体はテンソルを内部に持つため GPU 上でも扱いやすい。
  - `padding_mask`:
    - 時間方向の可変長をマスクとして表現する。
    - Transformer 系モジュールで `key_padding_mask` としてそのまま利用しやすいレイアウト。
  - `sequence_ids`:
    - 各バッチ要素がどのシーケンスに由来するかの ID。

---

## 6. DancetrackDataModule による利用方法

- 実装: `src/training/scene_model/datamodule.py:DancetrackDataModule`
- 役割:
  - YAML 設定から `DancetrackDataset` を構築し、`collate_tracking` を使う DataLoader を返す。

### 6.1 コンストラクタと setup

- シグネチャ (簡略):

  ```python
  class DancetrackDataModule(LightningDataModule):
      def __init__(
          self,
          dataset_cfg: DictConfig | Mapping[str, Any] | None,
          debug_cfg: DictConfig | Mapping[str, Any] | None,
      ) -> None:
          ...
  ```

- 主な責務:
  - `setup(stage)`:
    - `stage in (None, "fit")` のとき:
      - `train_dataset = DancetrackDataset(dataset_cfg, split="train", debug=debug_cfg)`
      - `val_dataset = DancetrackDataset(dataset_cfg, split="val", debug=debug_cfg)`
    - `stage in ("validate", "test")` のとき:
      - `val_dataset` が未構築なら `split="val"` で構築。

### 6.2 DataLoader

- `train_dataloader` / `val_dataloader`:
  - `dataset_cfg.loader.train` / `dataset_cfg.loader.val` から次のパラメータを読む:
    - `batch_size`
    - `num_workers`
    - `pin_memory`
    - `drop_last`
    - `persistent_workers`
  - `shuffle` の既定値:
    - train: `True`
    - val: `False`
  - `collate_fn=collate_tracking` を固定で使用し、戻り値の型は `DataLoader[SceneBatch]` となる。

- 乱数シード:
  - `debug_cfg.seed` または `dataset_cfg.seed` が指定されている場合、
    - `torch.Generator` による再現性のあるシャッフルを行う。

---

## 7. 関連ドキュメント

- モデル / アーキテクチャ:
  - `docs/spec/models/scene_model.md`
- トレーニングパイプライン:
  - `docs/spec/training/scene_model.md`
- Tennis Multi-Cam 3D Pose Dataset (別タスクの Dataset 仕様):
  - `docs/spec/datasets/tennis_multi_cam_3d_pose_dataset.md`
