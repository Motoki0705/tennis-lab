# チケット: LightningDataModule 実装（DanceTrack）

## 背景/目的
Lightning 標準の DataModule を実装し、Dataset と Collate の結線、メタ情報のキャッシュで IO を最適化する。

## 対象
- `src/training/scene_model/datamodule.py`

## タスク
- [ ] `LightningDataModule` を実装（`setup(stage)` で train/val の `DancetrackDataset` を構築）。
- [ ] `DataLoader` の `collate_fn` に `collate_tracking` を設定、`num_workers/pin_memory/persistent_workers` を config から反映。
- [ ] シーケンスメタ（フレーム一覧・アノテーション索引）を JSON でキャッシュし初回以降のロード短縮。
- [ ] 乱数シードの適用（Lightning の `seed_everything` or DataLoader generator）。
- [ ] 最小のスモークテストが通るようにモック/小サンプル設定を想定。

## 受け入れ基準（DoD）
- [ ] `uv run pytest tests/training/test_dancetrack_datamodule.py -k smoke` が通る（2バッチ生成）。
- [ ] `uv run python -m src.cli.train_scene_model --config configs/scene_model.yaml --help` で DataModule パラが表示。
