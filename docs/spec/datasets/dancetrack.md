# DanceTrack データセット仕様（Spec）

本仕様は `src/datasets/dancetrack.py` の振る舞いを、ワークフローとテンソル形状に焦点を当てて記述します。

---

## 目的

- DanceTrack の各シーケンスから、固定長（ただしデータ不足時は短いこともある）ウィンドウをサンプリングし、学習で扱いやすい `TrackingSample` を提供する。
- 画像変換（リサイズ、正規化）と必要に応じた左右反転を適用し、画像と対応する 2D バウンディングボックスを整合的に変換する。`tv_tensors.BoundingBoxes` のキャンバスサイズを用い、変換後に 0〜1 に正規化された中心・サイズを生成する。

---

## 主要データ構造

- `TargetFrame`
  - 1 フレーム内の全ターゲット注釈。
  - `center: Tensor[N, 2]`（中心座標 `cx, cy`、**リサイズ後画像に対する 0〜1 正規化値**）
  - `size: Tensor[N, 2]`（幅高 `w, h`、**リサイズ後画像に対する 0〜1 正規化値**）
  - `track_ids: Tensor[N]`（トラック ID）
  - `confidence: Tensor[N]`（スコア）
  - `empty(device)`: 長さ 0 の空コンテナを返す（パディング整合用）。

- `TrackingSample`
  - ウィンドウ 1 本分のサンプル。
  - `frames: Tensor[T, C, H, W]`
  - `targets: list[TargetFrame]`（長さ `T`）
  - `sequence_id: str`
  - `frame_indices: list[int]`（長さ `T`）

---

## 入力引数と設定スキーマ

`DancetrackDataset(cfg, split, window_sampler=None, debug=None)`

- `cfg: Mapping | DictConfig`（例）
  - `root: str`（データルート。既定: `third_party/DanceTrack/dancetrack`）
  - `split: {str: str}`（論理 split → 実ディレクトリ名の写像。なければ `split` をそのまま使う）
  - `window`
    - `size: int`（既定: 8。`> 0` 必須）
    - `stride: int`（既定: `size`）
  - `cache`
    - `enabled: bool`（既定: `False`）
    - `path: str | null`（相対なら `root` 基準。未指定時は `root/.dancetrack_meta.json`）
  - `image`
    - `resize: int`（正方リサイズ。既定: 224）
    - `normalize.mean: list[float]`（既定: `[0.485, 0.456, 0.406]`）
    - `normalize.std: list[float]`（既定: `[0.229, 0.224, 0.225]`）
    - `horizontal_flip_prob: float`（既定: 0.0 → 無効）

- `split: str`（例: `train`, `val`, `test`）
- `window_sampler: Callable | None`（`(length, window, stride) -> Iterable[(start, end)]`）
- `debug: Mapping | None`
  - `minimal: bool`（ウィンドウを最大 4 本までに制限）

---

## データレイアウト（DanceTrack）

- 各シーケンスディレクトリに `seqinfo.ini` があり、フレームディレクトリ `imDir` と拡張子 `imExt`、画像サイズ/フレームレートが記載される。
- アノテーションは `gt/gt.txt`（CSV）にあり、各行は `frame_id,track_id,x,y,w,h,confidence,...`。

---

## キャッシュ仕様

- 目的: `seqinfo.ini` と `gt.txt` のパース結果（フレームパス・画像サイズ・注釈）を JSON で保存し、再読込を高速化。
- 形式: ルートに JSON（既定 `root/.dancetrack_meta.json`）。キーは split 名。値はシーケンス配列。
- `cache.enabled=false` またはファイル未存在の場合は再構築。

---

## 画像変換と拡張

- 変換（`_build_transform`）
  - `Resize(size, size)` → `ConvertImageDtype(float32)` → `Normalize(mean, std)`
- 拡張（`_build_augment`）
  - `RandomHorizontalFlip(p)`（`p <= 0` なら無効）
- ボックスは `torchvision.tv_tensors` の `BoundingBoxes(XYXY)` として画像と一緒に変換される。

---

## ウィンドウサンプリング仕様

- 既定サンプラ `_default_sampler(length, window, stride)` の擬似コード:

```
if length <= 0: return []
if length <= window: return [(0, length)]
windows = []
idx = 0
while idx + window <= length:
  windows.append((idx, idx + window))
  idx += max(stride, 1)
if windows[-1][1] < length:
  windows.append((length - window, length))
return windows
```

- データセットは各シーケンスに対して上記サンプリングを行い、`windows: list[(seq_idx, start, end)]` を列挙・結合する。

---

## `__len__`, `__getitem__` の仕様

- `__len__()` は `windows` の総数を返す。
- `__getitem__(i)` の擬似コード:

```
seq_idx, start, end = windows[i]
seq = sequences[seq_idx]
frame_ids = range(start, end)
frames = []
targets = []
for t in frame_ids:
  image = read_or_black(seq.frame_paths[t], size=seq.image_size)
  anns = seq.annotations.get(t+1, [])
  boxes_xyxy = xywh_to_xyxy(anns)
  img_tensor, bbox_tensor = maybe_augment(image, boxes_xyxy)
  img_tensor, bbox_tensor = transform(img_tensor, bbox_tensor)
  target = bbox/ID を `TargetFrame`（center/size/track_id/conf）へ整形。bbox は torchvision の変換後キャンバスサイズで正規化。
  frames.append(img_tensor)
  targets.append(target)
stacked = stack(frames, dim=0)  # [T, C, H, W]
return TrackingSample(
  frames=stacked,
  targets=targets,
  sequence_id=seq.name,
  frame_indices=list(frame_ids),
)
```

注意:
- 画像ファイルが欠損している場合は、シーケンスの `image_size` に合わせたゼロ埋め `uint8` 画像で代替する。
- アノテーションが空のときは `TargetFrame.empty()` を返す。

---

## 返却テンソルと形状

- 画像: `frames` は `float32`、形状は `T x C x H x W`。
- ターゲット: 各 `TargetFrame` は同一フレーム内の N 個のインスタンスを持ち、`center(size)= [N,2]`, `size= [N,2]`, `track_ids= [N]`, `confidence= [N]`。

---

## エラー・前提

- `window.size <= 0` は `ValueError`。
- `cfg` は `Mapping`/`DictConfig`/`dict` を受け、内部で辞書化して解釈される。

