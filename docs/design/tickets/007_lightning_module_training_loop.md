# チケット: LightningModule 実装（最適化/ロギング/学習ループ）

## 背景/目的
SceneModel 構築・最適化・Denoiser/HeadAdapter の結線・TensorBoard ログまでを担う LightningModule を実装する。

## 対象
- `src/training/scene_model/lightning.py`

## タスク
- [ ] `__init__` で `build_scene_model(cfg)`・`HeadAdapter`・`Denoiser` を初期化。
- [ ] `training_step` にて `outputs = model(frames)` → `dn_state = denoiser.make_noise(targets)` → `loss_dict = head_adapter.compute_loss(...)` → `self.log_dict(loss_dict)`。
- [ ] `validation_step` で同様にロスをログ（必要に応じて指標の下準備）。
- [ ] `configure_optimizers` で `AdamW + cosine`（warmupオプション）を返却。
- [ ] AMP と grad clip は Lightning 設定から反映。
- [ ] TensorBoardLogger のスカラー/画像（例: サンプルフレーム可視化）出力を実装（任意）。

## 受け入れ基準（DoD）
- [ ] 1 step の forward/backward が通るユニットテストがグリーン。
- [ ] `uv run python -m src.cli.train_scene_model --config configs/scene_model.yaml` が最小設定で起動し、`runs/` にログが生成される。
