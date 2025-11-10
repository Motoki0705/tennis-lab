# チケット: 学習CLIの追加（uv実行対応）

## 背景/目的
設定ファイルを読み込み、Trainer/Module/DataModule を初期化して `trainer.fit()` を実行する CLI を実装する。

## 対象
- `src/cli/train_scene_model.py`

## タスク
- [ ] `--config`（必須）と `--set key=value ...`（任意）を受け取り、`training/utils/config.py` を経由して cfg を構築。
- [ ] Trainer へ TensorBoardLogger/Callbacks を設定（`configs/logging/default.yaml`）。
- [ ] 実行サンプル: `uv run python -m src.cli.train_scene_model --config configs/scene_model.yaml --set dataset.root=...` を README or doc に追記。
- [ ] 終了コード/例外ハンドリング（設定不足/ファイル欠損時に明瞭なメッセージ）。

## 受け入れ基準（DoD）
- [ ] `--help` で主要オプションが表示される。
- [ ] 最小構成で起動し、初期化が通る（実学習までは不要）。
