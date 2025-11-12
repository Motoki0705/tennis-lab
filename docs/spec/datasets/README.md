# datasets モジュール仕様（Spec）

本ディレクトリは `src/datasets/` 配下のデータセット関連コードの仕様をまとめます。実装を直接読まずに、役割・入出力・ワークフロー・テンソル形状を把握できることを目的とします。

- 対象モジュール
  - `src/datasets/dancetrack.py`: DanceTrack データセット読み込みとウィンドウサンプリング
  - `src/datasets/collate_tracking.py`: 可変長サンプルのバッチ化（collate）

---

## 全体ワークフロー

1) `DancetrackDataset(cfg, split, ...)` が、シーケンス単位のメタ情報を構築し、時系列ウィンドウのリストを作成する。
2) `__getitem__(i)` で、対象ウィンドウのフレーム画像とアノテーションを読み込み、以下の `TrackingSample` を返す。
   - `frames`: `Tensor[T, C, H, W]`
   - `targets`: `list[TargetFrame]`（長さ `T`）
   - `sequence_id`: `str`
   - `frame_indices`: `list[int]`（長さ `T`）
3) DataLoader の `collate_fn=collate_tracking` が、可変長サンプル列をパディングし、以下の `SceneBatch` を返す。
   - `frames`: `Tensor[B, T_max, C, H, W]`
   - `targets`: `list[list[TargetFrame]]`（サイズ `B x T_max`）
   - `padding_mask`: `BoolTensor[B, T_max]`（`False=実データ`, `True=パディング`）
   - `sequence_ids`: `list[str]`（サイズ `B`）

形状や意味は各ファイルの仕様を参照。

---

## 参照

- 詳細仕様: `docs/spec/datasets/dancetrack.md`
- Collate 仕様: `docs/spec/datasets/collate_tracking.md`

