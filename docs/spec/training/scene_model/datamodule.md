# DancetrackDataModule 仕様

`src/training/scene_model/datamodule.py`: DanceTrack を LightningDataModule として提供する。

## 目的
- `train_dataloader` / `val_dataloader` を構築し、`SceneBatch` で `[B, T, ...]` へ統一。
- 乱数シードやワーカ設定を YAML から反映。

## 入出力と形状
- 入力設定: `cfg.dataset` と `cfg.debug`
- `__getitem__`（データセット側）: `TrackingSample`
  - `frames`: `[T, 3, H, W]`
  - `targets`: `list[TargetFrame]`（長さ T）
  - `sequence_id`: `str`
- `collate_tracking(samples)` → `SceneBatch`
  - `frames`: `[B, T, 3, H, W]`
  - `targets`: `list[list[TargetFrame]]`（各サンプルに長さ T）
  - `padding_mask`: `Bool[B, T]`（True=pad）
  - `sequence_ids`: `list[str]`

## 擬似コード

```
class DancetrackDataModule(LightningDataModule):
  def __init__(dataset_cfg, debug_cfg):
    self.dataset_cfg = to_dict(dataset_cfg)
    self.debug_cfg   = to_dict(debug_cfg)
    seed = debug.seed or dataset.seed
    self._generator = torch.Generator().manual_seed(seed) if seed else None

  def setup(stage):
    if stage in (None, 'fit'):
      self.train_dataset = DancetrackDataset(self.dataset_cfg, split='train', debug=self.debug_cfg)
      self.val_dataset   = DancetrackDataset(self.dataset_cfg, split='val',   debug=self.debug_cfg)
    elif stage in ('validate', 'test') and self.val_dataset is None:
      self.val_dataset = DancetrackDataset(..., split='val', ...)

  def train_dataloader():
    cfg = dataset.loader.train
    return DataLoader(train_dataset,
                      batch_size=cfg.batch_size,
                      shuffle=cfg.shuffle(default=True),
                      num_workers=cfg.num_workers,
                      pin_memory=cfg.pin_memory,
                      drop_last=cfg.drop_last,
                      persistent_workers=cfg.persistent_workers and num_workers>0,
                      generator=self._generator if shuffle else None,
                      collate_fn=collate_tracking)

  def val_dataloader():
    cfg = dataset.loader.val
    return DataLoader(val_dataset, ..., shuffle=cfg.shuffle(default=False), collate_fn=collate_tracking)
```

## 主な設定キー（例）
- `dataset.root`: DanceTrack ルート（既定: `third_party/DanceTrack/dancetrack`）
- `dataset.split.{train,val}`: サブディレクトリ名
- `dataset.window.size/stride`: ウィンドウ化パラメータ（>0 必須）
- `dataset.image.resize/normalize.mean/normalize.std/horizontal_flip_prob`
- `dataset.loader.{train,val}.batch_size/num_workers/shuffle/pin_memory/drop_last/persistent_workers`
- `dataset.seed` または `debug.seed`

