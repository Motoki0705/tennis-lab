# チケット: ロギング/コールバック構成（TensorBoard/Checkpoint/LR Monitor）

## 背景/目的
学習過程の可観測性と再開性のため、TensorBoard ログ、ModelCheckpoint、LearningRateMonitor を設定ファイルから構築する。

## 対象
- `src/training/scene_model/callbacks.py`
- `configs/logging/default.yaml`

## タスク
- [ ] TensorBoardLogger（`save_dir=runs`, `name=dancetrack_scene_model`）のFactoryを実装。
- [ ] ModelCheckpoint（監視キー/保存間隔/上位N保存）を設定に沿って生成。
- [ ] LearningRateMonitor を有効化。
- [ ] CLI から渡す `experiment_name` でログディレクトリを切り替え可能に。
- [ ] Lightning Trainer への組み込みを確認。

## 受け入れ基準（DoD）
- [ ] `runs/dancetrack_scene_model` 配下にイベントファイルが生成される。
- [ ] エポック終了時に checkpoint が保存される（設定に応じて）。
