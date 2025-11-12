# Collate（可変長時系列のバッチ化）仕様（Spec）

本仕様は `src/datasets/collate_tracking.py` の `collate_tracking` と `SceneBatch` を記述します。

---

## 目的

- 可変長の `TrackingSample` を、学習ループで扱いやすい密なバッチ表現 `SceneBatch` に変換する。
- 可変長をパディングで吸収し、時系列長の位置を `padding_mask` で明示する。

---

## 入出力

- 入力: `samples: Sequence[TrackingSample]`（バッチ）
  - 各 `TrackingSample.frames`: `Tensor[T_i, C, H, W]`
  - 各 `TrackingSample.targets`: `list[TargetFrame]`（長さ `T_i`）
  - 各 `TrackingSample.sequence_id: str`

- 出力: `SceneBatch`
  - `frames: Tensor[B, T_max, C, H, W]`
  - `targets: list[list[TargetFrame]]`（サイズ `B x T_max`）
  - `padding_mask: BoolTensor[B, T_max]`（`False=実データ`, `True=パディング`）
  - `sequence_ids: list[str]`（サイズ `B`）

前提・制約:
- `samples` が空の場合は `ValueError`。
- 画像テンソルは `samples[0].frames` と同じ `device`/dtype に揃える。

---

## アルゴリズム（擬似コード）

```
if len(samples) == 0: raise ValueError
T_max = max(sample.frames.shape[0] for sample in samples)
B = len(samples)
frames = zeros([B, T_max, C, H, W], device=samples[0].frames.device)
padding_mask = ones([B, T_max], dtype=bool)  # 既定 True=パディング
batched_targets = []
sequence_ids = []
for b, sample in enumerate(samples):
  T = sample.frames.shape[0]
  frames[b, :T] = sample.frames
  padding_mask[b, :T] = False
  padded_targets = list(sample.targets)
  if T_max > T:
    padded_targets.extend(TargetFrame.empty(device) for _ in range(T_max - T))
  batched_targets.append(padded_targets)
  sequence_ids.append(sample.sequence_id)
return SceneBatch(frames, batched_targets, padding_mask, sequence_ids)
```

---

## 形状と意味づけ

- `frames[B, T_max, C, H, W]`: パディング領域は 0 埋め。
- `padding_mask[B, T_max]`:
  - `False`: 実データが入っている時刻
  - `True` : パディング（損失計算や可視化で無視するために使用）
- `targets[B][t]`: 各時刻の `TargetFrame`。パディング領域には `TargetFrame.empty()` が入る。

---

## 注意点

- `targets` はテンソルではなく `TargetFrame` のリストを保持し、検出数 `N` が可変であることに対応する。
- パディングは画像テンソルのみ 0 埋め。ターゲットは空の `TargetFrame` で表現する。
- `SceneBatch` は学習モジュール（LightningModule など）でそのまま消費できる構造を意図する。

