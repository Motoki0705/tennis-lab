# チケット: 設定ローダとYAML分割の実装（Lightning統合）

## 背景/目的
`configs/scene_model.yaml` を入口に datasets/training/models/logging の各YAMLを合成し、CLIのドット表記上書きを可能にする設定基盤を整備する。

## 対象
- `training/utils/config.py`
- `configs/scene_model.yaml`
- `configs/datasets/dancetrack.yaml`
- `configs/training/dino.yaml`
- `configs/models/scene_model.yaml`
- `configs/logging/default.yaml`

## タスク
- [ ] `training/utils/config.py` に `load_cfg(path: str, overrides: list[str] | None)` を実装（OmegaConf でマージ）。
- [ ] `ConfigLoader` あるいは等価のFactory群に `build_trainer()`, `build_lit_module()`, `build_datamodule()` を実装。
- [ ] `--config` と `--set key=value` のCLI統合（argparse + OmegaConf.merge）。
- [ ] `configs/datasets/dancetrack.yaml` を追加（root, img_size, window, transforms定義）。
- [ ] `configs/training/dino.yaml` を追加（optimizer=AdamW, lr, weight_decay, cosine scheduler, trainer設定: max_epochs, precision, gradient_clip等）。
- [ ] `configs/models/scene_model.yaml` を追加（backbone=dinov3_vits16, head=bbox などの切替、freezeパラ、pos_embed補間の有無）。
- [ ] `configs/logging/default.yaml` を追加（TensorBoardLogger: save_dir=`runs`, name=`dancetrack_scene_model`、checkpoint/early stopping/LR monitor）。
- [ ] `configs/scene_model.yaml` を作成し、上記4ファイルを参照・合成。
- [ ] ruff/mypy が通る最小実装（型ヒント・ドキュストリング含む）。

## 受け入れ基準（DoD）
- [ ] `uv run python -m src.cli.train_scene_model --config configs/scene_model.yaml --help` が実行できる（設定項目が表示される）。
- [ ] `load_cfg()` に対する最小の単体テストを追加/通過（設定合成・上書き動作）。
- [ ] `uv run ruff .` / `uv run mypy .` がグリーン。
